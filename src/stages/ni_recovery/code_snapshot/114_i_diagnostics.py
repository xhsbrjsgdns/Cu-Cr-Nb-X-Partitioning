# -*- coding: utf-8 -*-
"""
114_i_diagnostics.py
STEP5I Stage 4: topology / NBNI3 / temperature stratification and anchor coverage
(spec 37-42).

Everything in this stage is POST-BLIND DESCRIPTIVE ANALYSIS. It may explain the
recovery curve; it cannot and does not alter anchor selection, the model, the budgets
or any metric definition. The frozen sequences were fixed before the reveal, and the
NBNI3 phase was unknown when they were fixed - which is exactly what makes the
coverage-versus-recovery comparison in spec 41 interesting rather than circular.

Explicitly NOT done here (spec 42, 46): no NBNI3-targeted or OTHER_SOLID-targeted
anchor set is designed, and no Shargh-style targeted augmentation is attempted. Those
would be post-hoc target-informed selection, and they belong to a later stage that has
not been authorized.

Strata:
  true topology     FCC vs OTHER_SOLID (the only two the frozen Ni targets contain)
  NBNI3             present vs absent, read from the frozen raw phase names
  temperature       LOW_T 300-600 (the NBNI3 window) vs HIGH_T 700-1000
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i_common as C

METHOD_COLS = {"M_ZERO": "tvd_M_ZERO", "M_TRIVIAL": "tvd_M_TRIVIAL",
               "M_REFIT": "tvd_M_REFIT", "R0": "tvd_R0", "R1": "tvd_R1", "R2": "tvd_R2"}
TOPO_COLS = {k: v.replace("tvd_", "topo_correct_") for k, v in METHOD_COLS.items()}


def stratify(joined, by, label):
    out = []
    for (tag, kind, rep, budget, stratum), g in joined.groupby(
            ["tag", "kind", "replicate", "budget", by], observed=True):
        for method, col in METHOD_COLS.items():
            if col not in g.columns:
                continue
            out.append({"stratification": label, "stratum": str(stratum),
                        "kind": kind, "replicate": int(rep), "budget": int(budget),
                        "tag": tag, "method": method, "n_states": int(len(g)),
                        "mean_tvd": float(g[col].mean()),
                        "median_tvd": float(g[col].median()),
                        "topology_accuracy": float(g[TOPO_COLS[method]].mean())})
    return pd.DataFrame(out)


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5I  Stage 4: topology / NBNI3 / temperature diagnostics (descriptive)")
    print("=" * 78)

    base = pd.read_parquet(os.path.join(C.D_METRICS,
                                        "STEP5I_RECOVERY_BASELINE_PER_STATE.parquet"))
    ctrl = pd.read_parquet(os.path.join(C.D_CONTROL,
                                        "STEP5I_CONTROL_PER_STATE.parquet"))
    refit = pd.read_parquet(os.path.join(C.D_METRICS, "STEP5I_REFIT_PER_STATE.parquet"))
    man = pd.read_csv(os.path.join(C.D_MANIFEST,
                                   "STEP5I_RECOVERY_CHECKPOINT_MANIFEST.csv"))
    cov = pd.read_csv(os.path.join(C.D_TOPO, "STEP5I_ANCHOR_TOPOLOGY_COVERAGE.csv"))
    agg = pd.read_csv(os.path.join(C.D_METRICS, "STEP5I_RANDOM_FEWSHOT_METRICS.csv"))

    cells = man[["tag", "kind", "replicate", "budget"]].copy()
    joined = (ctrl.merge(refit, on=["tag", "state_id"], validate="one_to_one")
                  .merge(base, on="state_id", validate="many_to_one")
                  .merge(cells, on="tag", validate="many_to_one"))
    if len(joined) != len(man) * C.N_RECOVERY_STATES:
        raise SystemExit("FATAL: stratification frame has %d rows, expected %d."
                         % (len(joined), len(man) * C.N_RECOVERY_STATES))

    # ------------------------------------------------ spec 37: topology ---- #
    topo = stratify(joined, "true_topology", "TRUE_TOPOLOGY")
    topo.to_csv(os.path.join(C.D_TOPO, "STEP5I_TRUE_TOPOLOGY_STRATIFIED.csv"),
                index=False, encoding="utf-8")

    # ------------------------------------------------- spec 38: NBNI3 ------ #
    joined["nbni3_status"] = np.where(joined.nbni3_present == 1, "NBNI3_PRESENT",
                                      "NBNI3_ABSENT")
    nb = stratify(joined, "nbni3_status", "NBNI3_STATUS")
    nb.to_csv(os.path.join(C.D_NBNI3, "STEP5I_NBNI3_STRATIFIED.csv"),
              index=False, encoding="utf-8")

    # ------------------------------------ spec 39: temperature regimes ----- #
    tt = stratify(joined, "t_regime", "TEMPERATURE_REGIME")
    tt.to_csv(os.path.join(C.D_TOPO, "STEP5I_TEMPERATURE_REGIME_METRICS.csv"),
              index=False, encoding="utf-8")

    # -------------------------- spec 41: coverage vs recovery, descriptive -- #
    refit_cell = (joined.groupby(["tag", "kind", "replicate", "budget"], observed=True)
                  .tvd_M_REFIT.mean().reset_index(name="M_REFIT_mean_tvd"))
    r0_cell = (joined.groupby("tag", observed=True).tvd_R0.mean()
               .reset_index(name="R0_mean_tvd"))
    mech = (cov.merge(refit_cell, on=["tag", "kind", "replicate", "budget"])
               .merge(r0_cell, on="tag"))
    zero_tvd = json.load(open(os.path.join(C.D_METRICS,
                                           "STEP5I_RECOVERY_BASELINES.json"),
                              encoding="utf-8"))["M_ZERO"]["mean_tvd"]
    mech["recovery_vs_zero_shot"] = zero_tvd - mech["M_REFIT_mean_tvd"]
    mech["R0_recovery_vs_zero_shot"] = zero_tvd - mech["R0_mean_tvd"]

    from scipy.stats import spearmanr

    def rho(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            # a constant column has no rank correlation; reporting 0 here would
            # invent a "no relationship" finding that was never measured
            return None, None, "undefined (constant input)"
        r = spearmanr(x, y)
        return float(r.statistic), float(r.pvalue), "ok"

    mech_rows = []
    rnd = mech[mech.kind == "RANDOM"]
    for scope, g in [("ALL_RANDOM_CELLS", rnd)] + \
            [("BUDGET_%d" % b, rnd[rnd.budget == b]) for b in C.BUDGETS]:
        for cov_col in ("fraction_anchor_states_NBNI3",
                        "n_anchor_comps_with_any_NBNI3",
                        "fraction_anchor_states_true_OTHER_SOLID"):
            for target, tlabel in (("recovery_vs_zero_shot", "M_REFIT"),
                                   ("R0_recovery_vs_zero_shot", "R0")):
                r, p, note = rho(g[cov_col], g[target])
                mech_rows.append({"scope": scope, "n_cells": int(len(g)),
                                  "coverage_metric": cov_col,
                                  "recovery_metric": target, "method": tlabel,
                                  "spearman_rho": r, "p_value": p, "note": note})
    mech_df = pd.DataFrame(mech_rows)
    mech.to_csv(os.path.join(C.D_NBNI3, "STEP5I_ANCHOR_COVERAGE_VS_RECOVERY.csv"),
                index=False, encoding="utf-8")
    mech_df.to_csv(os.path.join(C.D_NBNI3, "STEP5I_COVERAGE_MECHANISM_CORRELATIONS.csv"),
                   index=False, encoding="utf-8")

    C.write_json(os.path.join(C.D_NBNI3, "STEP5I_DIAGNOSTIC_SCOPE.json"), {
        "generated_utc": C.utcnow(),
        "label": "POST-BLIND DESCRIPTIVE ANALYSIS (spec 37-41)",
        "may_explain_recovery": True,
        "may_alter_anchor_selection": False,
        "targeted_anchor_design_performed": False,
        "shargh_style_targeted_augmentation_performed": False,
        "low_order_binary_ni_study_performed": False,
        "dft_used": False,
        "nbni3_mapping": "NBNI3 -> OTHER_SOLID, frozen at STEP5H-B and unchanged here",
        "strata": {"TRUE_TOPOLOGY": sorted(set(joined.true_topology)),
                   "NBNI3_STATUS": ["NBNI3_PRESENT", "NBNI3_ABSENT"],
                   "TEMPERATURE_REGIME": {"LOW_T": C.LOW_T, "HIGH_T": C.HIGH_T}},
        "recovery_subset_composition": {
            "n_states": int(len(base)),
            "n_NBNI3_present": int(base.nbni3_present.sum()),
            "n_NBNI3_absent": int((base.nbni3_present == 0).sum()),
            "true_topology_counts": base.true_topology.value_counts().to_dict(),
            "n_LOW_T": int((base.t_regime == "LOW_T").sum()),
            "n_HIGH_T": int((base.t_regime == "HIGH_T").sum())}})

    print("  recovery subset: %d states | NBNI3 present %d / absent %d | true topology %s"
          % (len(base), int(base.nbni3_present.sum()),
             int((base.nbni3_present == 0).sum()),
             base.true_topology.value_counts().to_dict()))
    print("  anchor NBNI3 coverage across the 18 random cells: %d-%d of the selected "
          "compositions carry an NBNI3 state"
          % (rnd.n_anchor_comps_with_any_NBNI3.min(),
             rnd.n_anchor_comps_with_any_NBNI3.max()))
    print("STAGE 4: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
