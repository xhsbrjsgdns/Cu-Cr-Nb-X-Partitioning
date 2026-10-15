# -*- coding: utf-8 -*-
"""
115_i_bootstrap.py
STEP5I Stage 5: composition-level paired bootstrap (spec 43).

Labelled POST-BLIND EXPLORATORY CI, and that label is not decoration: these intervals
are supplementary. They do not alter B_skill, B_recovery25, the primary estimator, the
role of maximin, or any model comparison. Nothing in this stage may be used to promote
a budget, a replicate or a method.

Design:
  paired          the same bootstrap resample of compositions is applied to both
                  methods in each contrast, so the interval is on the DIFFERENCE and
                  the shared composition noise cancels
  composition-level  resampling is by composition_id, never by state: the 11
                  temperatures of one composition are one cluster and are not
                  independent observations
  1000 replicates, seed 20260913, over the 1400 recovery compositions

Contrasts (spec 43): M_REFIT - M_ZERO, M_REFIT - R0, M_REFIT - R2. A negative mean
difference means M_REFIT has the lower TVD.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i_common as C

CONTRASTS = [("M_REFIT", "M_ZERO"), ("M_REFIT", "R0"), ("M_REFIT", "R2")]


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5I  Stage 5: composition-level paired bootstrap "
          "(POST-BLIND EXPLORATORY CI)")
    print("=" * 78)

    base = pd.read_parquet(os.path.join(C.D_METRICS,
                                        "STEP5I_RECOVERY_BASELINE_PER_STATE.parquet"))
    ctrl = pd.read_parquet(os.path.join(C.D_CONTROL,
                                        "STEP5I_CONTROL_PER_STATE.parquet"))
    refit = pd.read_parquet(os.path.join(C.D_METRICS, "STEP5I_REFIT_PER_STATE.parquet"))
    man = pd.read_csv(os.path.join(C.D_MANIFEST,
                                   "STEP5I_RECOVERY_CHECKPOINT_MANIFEST.csv"))

    comps = np.array(sorted(set(base.composition_id)))
    if len(comps) != C.N_RECOVERY_EVAL:
        raise SystemExit("FATAL: %d recovery compositions, expected %d."
                         % (len(comps), C.N_RECOVERY_EVAL))
    cidx = {c: i for i, c in enumerate(comps)}
    state_comp = base.set_index("state_id").composition_id

    # one shared resample matrix: identical composition draws for every contrast and
    # every cell, which is what makes the differences comparable across budgets
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    draws = rng.integers(0, len(comps), size=(C.BOOTSTRAP_REPS, len(comps)))

    def comp_means(state_ids, values):
        """Per-composition mean TVD, in the fixed `comps` order."""
        s = pd.Series(values, index=pd.Index(state_ids).map(state_comp))
        m = s.groupby(level=0).mean()
        return m.reindex(comps).to_numpy(float)

    zero_by_comp = comp_means(base.state_id, base.tvd_M_ZERO.to_numpy(float))

    rows = []
    for _, m in man.iterrows():
        tag = m["tag"]
        rf = refit[refit.tag == tag]
        ct = ctrl[ctrl.tag == tag]
        by = {"M_REFIT": comp_means(rf.state_id, rf.tvd_M_REFIT.to_numpy(float)),
              "M_ZERO": zero_by_comp,
              "R0": comp_means(ct.state_id, ct.tvd_R0.to_numpy(float)),
              "R2": comp_means(ct.state_id, ct.tvd_R2.to_numpy(float))}
        for a, b in CONTRASTS:
            d = by[a] - by[b]
            boot = d[draws].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({
                "label": "POST-BLIND EXPLORATORY CI",
                "kind": m["kind"], "replicate": int(m["replicate"]),
                "budget": int(m["budget"]), "tag": tag,
                "contrast": "%s - %s" % (a, b),
                "observed_difference": float(d.mean()),
                "boot_mean": float(boot.mean()), "boot_std": float(boot.std(ddof=1)),
                "ci95_low": float(lo), "ci95_high": float(hi),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "direction": ("M_REFIT lower TVD" if d.mean() < 0
                              else "M_REFIT higher TVD"),
                "n_compositions": int(len(comps)),
                "replicates": C.BOOTSTRAP_REPS, "seed": C.BOOTSTRAP_SEED,
                "resample_unit": "composition_id",
                "used_to_alter_primary_definitions": False})
    out = pd.DataFrame(rows).sort_values(["kind", "replicate", "budget", "contrast"])
    out.to_csv(os.path.join(C.D_BOOT, "STEP5I_PAIRED_BOOTSTRAP.csv"), index=False,
               encoding="utf-8")

    C.write_json(os.path.join(C.D_BOOT, "STEP5I_BOOTSTRAP_SCOPE.json"), {
        "generated_utc": C.utcnow(),
        "label": "POST-BLIND EXPLORATORY CI (spec 43)",
        "method": "composition-level paired bootstrap; the same composition resample "
                  "is applied to both methods in a contrast",
        "replicates": C.BOOTSTRAP_REPS, "seed": C.BOOTSTRAP_SEED,
        "n_compositions": int(len(comps)),
        "contrasts": ["%s - %s" % c for c in CONTRASTS],
        "sign_convention": "negative = M_REFIT has the lower TVD",
        "may_alter_primary_definitions": False,
        "may_alter_model_roles": False})

    n_sig = int(out.ci_excludes_zero.sum())
    print("  %d contrasts computed | %d have a 95%% interval excluding zero"
          % (len(out), n_sig))
    print("STAGE 5: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
