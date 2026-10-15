import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from step5b_bootstrap import FAMILIES, THRESHOLDS, Prepared, weighted_quantile, independent_counts, run_bootstrap


def fixture():
    rng = np.random.default_rng(52)
    n = 9
    d = {'composition_id': ['a']*2 + ['b']*3 + ['c']*4,
         'family': ['BASELINE']*2 + ['ZR']*7}
    for f in FAMILIES:
        d[f'{f}_presence_true'] = rng.integers(0, 2, n)
        d[f'{f}_presence_prob'] = rng.random(n)
    for task, tag in [('fraction', None), ('cr_partition','CR_P'), ('nb_partition','NB_P'), ('x_partition','X_P')]:
        t, p = rng.dirichlet(np.ones(5),n), rng.dirichlet(np.ones(5),n)
        if task == 'x_partition': t[:2] = np.nan
        for j,f in enumerate(FAMILIES):
            d[f'{f}_volfrac_true' if tag is None else f'{tag}_{f}_true'] = t[:,j]
            d[f'{f}_frac_proj' if tag is None else f'{f}_{tag}_proj'] = p[:,j]
    return pd.DataFrame(d)


class BootstrapTests(unittest.TestCase):
    def test_quantile_equals_expansion(self):
        x = np.array([4., 1., 1., 9., 6.])
        w = np.array([2, 0, 3, 1, 7])
        for q in [0,.025,.5,.95,.975,1]:
            self.assertAlmostEqual(weighted_quantile(x,w,q), np.quantile(np.repeat(x,w),q))

    def test_whole_cluster_matches_explicit_rows(self):
        df = fixture()
        prep = Prepared(df)
        counts = np.array([2,0,1])
        got = prep.metrics(counts)
        rows = np.repeat(np.arange(len(df)),counts[prep.inverse])
        repeated = df.iloc[rows]
        expected = Prepared(repeated).metrics(np.ones(2,dtype=int))
        for key in got: np.testing.assert_allclose(got[key], expected[key],equal_nan=True,err_msg=str(key))
        t = repeated[[f'{f}_presence_true' for f in FAMILIES]].to_numpy()
        p = repeated[[f'{f}_presence_prob' for f in FAMILIES]].to_numpy()
        self.assertAlmostEqual(got[('presence','macro_f1')],f1_score(t,p>=THRESHOLDS,average='macro',zero_division=0))
        self.assertAlmostEqual(got[('presence','liquid_pr_auc')],average_precision_score(t[:,4],p[:,4]))
        y = repeated[[f'{f}_volfrac_true' for f in FAMILIES]].to_numpy()
        z = repeated[[f'{f}_frac_proj' for f in FAMILIES]].to_numpy()
        self.assertAlmostEqual(got[('fraction','macro_rmse')],np.sqrt(np.mean((y-z)**2)))

    def test_independent_repeatable_streams(self):
        a,b = independent_counts(30,30,20,20260905)
        aa,bb = independent_counts(30,30,20,20260905)
        np.testing.assert_array_equal(a,aa); np.testing.assert_array_equal(b,bb)
        self.assertFalse(np.array_equal(a,b))
        np.testing.assert_array_equal(a.sum(axis=1),30)

    def test_outputs_and_paired_skill(self):
        df = fixture()
        trivial = {'presence': np.full((len(df),5),.5),'x_partition':np.full(5,.2)}
        with tempfile.TemporaryDirectory() as tmp:
            out = run_bootstrap(df,df,tmp,n_reps=20,trivial_vectors=trivial)
            self.assertEqual(len(out['ood']),26)
            self.assertTrue((Path(tmp)/'BOOTSTRAP_REPLICATES.npz').exists())
            self.assertEqual(len(out['skill']),2)
            self.assertTrue((out['comparison']['delta_estimate']==0).all())
            x = out['ood'].query("task == 'x_partition'")
            self.assertTrue((x.n_compositions==2).all())
            self.assertTrue((x.n_sampled_compositions==3).all())
            xcmp = out['comparison'].query("task == 'x_partition'")
            for domain in ['iid','ood']:
                self.assertTrue((xcmp[f'{domain}_n_compositions']==2).all())
                self.assertTrue((xcmp[f'{domain}_n_sampled_compositions']==3).all())
            xs = out['skill'].query("task == 'x_partition'")
            self.assertTrue((xs.n_compositions==2).all())
            self.assertTrue((xs.n_sampled_compositions==3).all())

    def test_x_excludes_baseline_even_if_finite_and_never_fills(self):
        df = fixture()
        keys = [f'X_P_{f}_true' for f in FAMILIES]
        before = Prepared(df).metrics(np.ones(3,dtype=int))
        df.loc[df.family=='BASELINE',keys] = 1000.
        after = Prepared(df).metrics(np.ones(3,dtype=int))
        for key in before:
            np.testing.assert_allclose(before[key],after[key],equal_nan=True)
        only_baseline = Prepared(df).metrics(np.array([3,0,0]))
        self.assertTrue(np.isnan(only_baseline[('x_partition','mean_tvd')]))
        self.assertTrue(np.isnan(only_baseline[('x_partition','p95_tvd')]))

    def test_paired_identical_regression_is_exactly_zero(self):
        df = fixture()
        p = df[[f'{f}_frac_proj' for f in FAMILIES]].to_numpy()
        prep = Prepared(df,{'fraction':p})
        for c in [np.array([2,0,1]),np.array([0,0,3]),np.ones(3,dtype=int)]:
            self.assertEqual(prep.skill_metrics(c,prep.metrics(c))[('fraction','mean_tvd_improvement')],0.)

    def test_liquid_single_class_matches_frozen_policy(self):
        df = fixture()
        df['LIQUID_presence_true'] = 0
        got = Prepared(df).metrics(np.ones(3,dtype=int))
        self.assertTrue(np.isnan(got[('presence','liquid_pr_auc')]))
        self.assertEqual(got[('presence','liquid_f1')],0.)
        df['LIQUID_presence_true'] = 1
        got = Prepared(df).metrics(np.ones(3,dtype=int))
        expected = f1_score(df['LIQUID_presence_true'],df['LIQUID_presence_prob']>=THRESHOLDS[4],zero_division=0)
        self.assertAlmostEqual(got[('presence','liquid_f1')],expected)
        self.assertTrue(np.isnan(got[('presence','liquid_pr_auc')]))

    def test_invalid_nonbaseline_x_targets_raise(self):
        for invalid in [np.nan,np.inf]:
            df = fixture()
            df.loc[2,'X_P_FCC_true'] = invalid
            with self.assertRaisesRegex(ValueError,'Nonfinite required targets for x_partition'):
                Prepared(df)


if __name__ == '__main__': unittest.main()
