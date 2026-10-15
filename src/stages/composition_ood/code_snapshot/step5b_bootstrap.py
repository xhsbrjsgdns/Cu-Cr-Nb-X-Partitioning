"""Frozen composition bootstrap; inputs are already-opened prediction dataframes.

No file input, model training, prediction, or target materialization occurs here.
Intervals use percentile quantiles with NumPy's default linear interpolation.
Cluster multiplicities preserve every available temperature row. IID and OOD
draws use independent SeedSequence children; paired skill uses the OOD draws.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

FAMILIES = ('FCC','BCC','C15','OTHER_SOLID','LIQUID')
THRESHOLDS = np.array([.5,.37,.5,.32,.42])
TASK_TAGS = {'fraction':None,'cr_partition':'CR_P','nb_partition':'NB_P','x_partition':'X_P'}


def weighted_quantile(values, weights, q, presorted=False):
    """Linear quantile exactly matching np.quantile(np.repeat(x, integer_w))."""
    x, w = np.asarray(values), np.asarray(weights)
    if np.any(w < 0) or np.any(w != np.floor(w)):
        raise ValueError('Quantile weights must be nonnegative integer multiplicities')
    if not presorted:
        order = np.argsort(x,kind='stable'); x,w = x[order],w[order]
    total = w.sum()
    if total == 0: return float('nan')
    h = (total-1)*q
    cumulative = np.cumsum(w)
    lo = x[np.searchsorted(cumulative,np.floor(h),side='right')]
    hi = x[np.searchsorted(cumulative,np.ceil(h),side='right')]
    return float(lo+(hi-lo)*(h-np.floor(h)))


def independent_counts(n_iid,n_ood,n_reps,seed):
    children = np.random.SeedSequence(seed).spawn(2)
    result = []
    for n,child in zip((n_iid,n_ood),children):
        if n < 1: raise ValueError('Both datasets require compositions')
        rng = np.random.default_rng(child)
        result.append(np.stack([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(n_reps)]))
    return tuple(result)


class Prepared:
    def __init__(self,df,trivial_vectors=None):
        if df.empty or df['composition_id'].isna().any(): raise ValueError('Missing composition IDs or empty dataset')
        self.compositions,self.inverse = np.unique(df['composition_id'].to_numpy(),return_inverse=True)
        self.n = len(self.compositions)
        self.trivial = trivial_vectors or {}
        self.truth = df[[f'{f}_presence_true' for f in FAMILIES]].to_numpy(dtype=float)
        self.prob = df[[f'{f}_presence_prob' for f in FAMILIES]].to_numpy(dtype=float)
        if not np.isin(self.truth,[0,1]).all() or not np.isfinite(self.prob).all():
            raise ValueError('Presence targets must be binary and predictions finite')
        self.presence = self._presence(self.prob,THRESHOLDS)
        self.ap_order = np.argsort(-self.prob[:,4],kind='stable')
        scores = self.prob[self.ap_order,4]
        self.ap_ends = np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1]
        self.reg = {}
        self.skill = {}
        self.eligible_n = {'presence':self.n}
        for task,tag in TASK_TAGS.items():
            true_cols = [f'{f}_volfrac_true' if tag is None else f'{tag}_{f}_true' for f in FAMILIES]
            pred_cols = [f'{f}_frac_proj' if tag is None else f'{f}_{tag}_proj' for f in FAMILIES]
            t,p = df[true_cols].to_numpy(float),df[pred_cols].to_numpy(float)
            mask = df['family'].to_numpy() != 'BASELINE' if task == 'x_partition' else np.ones(len(df),dtype=bool)
            if not np.isfinite(t[mask]).all(): raise ValueError(f'Nonfinite required targets for {task}')
            self.eligible_n[task] = len(np.unique(self.inverse[mask]))
            if not np.isfinite(p[mask]).all(): raise ValueError(f'Nonfinite predictions for {task}')
            self.reg[task] = self._regression(t,p,mask)
            if task in self.trivial:
                base = self._vector(self.trivial[task],len(df))
                self.skill[task] = self._regression(t,base,mask)
        if 'presence' in self.trivial:
            self.skill['presence'] = self._presence(self._vector(self.trivial['presence'],len(df)),np.full(5,.5))

    @staticmethod
    def _vector(x,n):
        x = np.asarray(x,dtype=float)
        if x.shape == (5,): x = np.broadcast_to(x,(n,5))
        if x.shape != (n,5) or not np.isfinite(x).all(): raise ValueError('Trivial vectors must be finite (5,) or (n_rows,5)')
        return x

    def _aggregate(self,a):
        out = np.zeros((self.n,a.shape[1]))
        np.add.at(out,self.inverse,a)
        return out

    def _presence(self,prob,thresholds):
        t,b = self.truth,prob>=thresholds
        return self._aggregate(np.column_stack([np.ones(len(t)),(t==b).all(axis=1),(t!=b).mean(axis=1),t*b,(1-t)*b,t*(1-b)]))

    def _regression(self,t,p,mask):
        diff = p[mask]-t[mask]
        tvd = .5*np.abs(diff).sum(axis=1)
        rows = np.flatnonzero(mask)
        order = np.argsort(tvd,kind='stable')
        a = np.zeros((len(t),4))
        a[mask] = np.column_stack([np.ones(len(diff)),tvd,np.abs(diff).mean(axis=1),(diff**2).mean(axis=1)])
        return self._aggregate(a),tvd[order],self.inverse[rows[order]]

    @staticmethod
    def _pmetrics(s):
        n,exact,hamming = s[:3]
        tp,fp,fn = s[3:8],s[8:13],s[13:18]
        den = 2*tp+fp+fn
        f1 = np.divide(2*tp,den,out=np.zeros(5),where=den!=0)
        return {'exact_match':exact/n,'hamming_loss':hamming/n,'micro_f1':2*tp.sum()/den.sum() if den.sum() else 0.,'macro_f1':f1.mean(),'liquid_f1':f1[4]}

    def metrics(self,counts):
        p = self._pmetrics(counts@self.presence)
        # F1 remains defined (zero_division=0); AP requires both classes.
        w = counts[self.inverse][self.ap_order]
        truth = self.truth[self.ap_order,4]
        positives = np.dot(w,truth); total = w.sum()
        if positives == 0 or positives == total:
            p['liquid_pr_auc'] = np.nan
        else:
            tp = np.cumsum(w*truth)[self.ap_ends]
            seen = np.cumsum(w)[self.ap_ends]
            precision = np.divide(tp,seen,out=np.zeros_like(tp),where=seen!=0)
            p['liquid_pr_auc'] = np.sum(np.diff(np.r_[0.,tp])*precision)/positives
        out = {('presence',k):float(v) for k,v in p.items()}
        for task,(a,tvd,inv) in self.reg.items():
            sums = counts@a
            n = sums[0]
            vals = {'mean_tvd':sums[1]/n if n else np.nan,'macro_mae':sums[2]/n if n else np.nan,'macro_rmse':np.sqrt(sums[3]/n) if n else np.nan,
                    'median_tvd':weighted_quantile(tvd,counts[inv],.5,True),'p95_tvd':weighted_quantile(tvd,counts[inv],.95,True)}
            out.update({(task,k):float(v) for k,v in vals.items()})
        return out

    def skill_metrics(self,counts,model):
        out = {}
        for task,a in self.skill.items():
            if task == 'presence':
                out[(task,'macro_f1_improvement')] = model[(task,'macro_f1')]-self._pmetrics(counts@a)['macro_f1']
            else:
                s = counts@a[0]
                out[(task,'mean_tvd_improvement')] = s[1]/s[0]-model[(task,'mean_tvd')] if s[0] else np.nan
        return out


def _interval(arr):
    valid = np.asarray(arr)[np.isfinite(arr)]
    return (*np.quantile(valid,[.025,.975]),len(valid)) if len(valid) else (np.nan,np.nan,0)


def run_bootstrap(ood_df,iid_df,out_dir,seed=20260905,n_reps=1000,trivial_vectors=None):
    """Return {'ood','comparison','skill'} tables and write CSV/NPZ/meta audit.

    trivial_vectors maps presence/fraction/cr_partition/nb_partition/x_partition
    to a constant five-vector or an array aligned to ood_df row order. Presence
    is probability, thresholded at .5. Regression inputs must already be the
    frozen trivial predictions; this function never estimates baseline vectors.
    """
    if not isinstance(n_reps,int) or n_reps<1: raise ValueError('n_reps must be positive integer')
    iid,ood = Prepared(iid_df),Prepared(ood_df,trivial_vectors)
    ic,oc = independent_counts(iid.n,ood.n,n_reps,seed)
    ip,op = iid.metrics(np.ones(iid.n,dtype=int)),ood.metrics(np.ones(ood.n,dtype=int))
    keys = list(op)
    ir,orr = np.empty((n_reps,len(keys))),np.empty((n_reps,len(keys)))
    sp = ood.skill_metrics(np.ones(ood.n,dtype=int),op)
    skeys = list(sp); sr = np.empty((n_reps,len(skeys)))
    for b in range(n_reps):
        im,om = iid.metrics(ic[b]),ood.metrics(oc[b])
        ir[b],orr[b] = [im[k] for k in keys],[om[k] for k in keys]
        sm = ood.skill_metrics(oc[b],om)
        sr[b] = [sm[k] for k in skeys]
    rows,comparison,skill = [],[],[]
    for j,(task,metric) in enumerate(keys):
        il,iu,ni = _interval(ir[:,j]); ol,ou,no = _interval(orr[:,j]); dl,du,nd = _interval(orr[:,j]-ir[:,j])
        key = (task,metric)
        rows.append(dict(task=task,metric=metric,estimate=op[key],ci_lower=ol,ci_upper=ou,n_replicates=no,n_compositions=ood.eligible_n[task],n_sampled_compositions=ood.n))
        comparison.append(dict(task=task,metric=metric,iid_estimate=ip[key],iid_ci_lower=il,iid_ci_upper=iu,ood_estimate=op[key],ood_ci_lower=ol,ood_ci_upper=ou,delta_estimate=op[key]-ip[key],delta_ci_lower=dl,delta_ci_upper=du,n_replicates=nd,iid_n_replicates=ni,ood_n_replicates=no,iid_n_compositions=iid.eligible_n[task],ood_n_compositions=ood.eligible_n[task],iid_n_sampled_compositions=iid.n,ood_n_sampled_compositions=ood.n))
    for j,(task,metric) in enumerate(skeys):
        lo,hi,n = _interval(sr[:,j])
        skill.append(dict(task=task,metric=metric,estimate=sp[(task,metric)],ci_lower=lo,ci_upper=hi,n_replicates=n,n_compositions=ood.eligible_n[task],n_sampled_compositions=ood.n))
    result = {'ood':pd.DataFrame(rows),'comparison':pd.DataFrame(comparison),'skill':pd.DataFrame(skill,columns=['task','metric','estimate','ci_lower','ci_upper','n_replicates','n_compositions','n_sampled_compositions'])}
    out = Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    for key,name in [('ood','DOMAIN_B_BOOTSTRAP_CI.csv'),('comparison','IID_VS_OOD_WITH_CI.csv'),('skill','DOMAIN_B_SKILL_BOOTSTRAP_CI.csv')]:
        result[key].to_csv(out/name,index=False)
    np.savez_compressed(out/'BOOTSTRAP_REPLICATES.npz',iid=ir,ood=orr,delta=orr-ir,skill=sr,metric_keys=np.asarray(keys),skill_keys=np.asarray(skeys),iid_counts=ic,ood_counts=oc,iid_compositions=np.asarray(iid.compositions,dtype=str),ood_compositions=np.asarray(ood.compositions,dtype=str))
    (out/'BOOTSTRAP_META.json').write_text(json.dumps(dict(seed=seed,n_replicates_requested=n_reps,scheme='independent composition bootstrap; all rows carried; paired skill uses same OOD draws',composition_counts='n_compositions counts original task-eligible compositions; n_sampled_compositions counts full sampling universe. X draws from full universe then excludes BASELINE rows.',rng='numpy default_rng; SeedSequence.spawn(2), child 0 IID, child 1 OOD',ci='95% percentile; finite replicates only; linear quantile',liquid_single_class='F1 defined with zero_division=0; AP undefined',x_policy='exclude BASELINE only; reject nonfinite nonBASELINE targets; no zero fill',macro_rmse='sqrt(mean(component squared errors))'),indent=2),encoding='utf-8')
    return result
