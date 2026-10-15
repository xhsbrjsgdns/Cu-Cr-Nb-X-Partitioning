# -*- coding: utf-8 -*-
"""
113_i_metrics.py
STEP5I Stage 3: metrics, learning curves, B_skill and B_recovery25
(spec 30-36).

Everything is scored on the identical 15,400 RECOVERY_EVALUATION states, so no
comparison here is between different row sets (spec 31).

Two rules that shape the output:
  - Every individual replicate is preserved (spec 35). The mean across the three
    random replicates is reported alongside them, never instead of them.
  - Every frozen budget is reported (spec 36). No budget, replicate or anchor sequence
    is selected after the fact as "the final model", and the tables carry all of them
    whatever they show.

B_recovery25 is measured against M_ZERO on the SAME recovery subset (spec 34), which is
computed in stage 2 - not against the 22,000-state blind value.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i_common as C


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5I  Stage 3: metrics, learning curves and budget thresholds")
    print("=" * 78)

    base_json = json.load(open(os.path.join(C.D_METRICS,
                                            "STEP5I_RECOVERY_BASELINES.json"),
                               encoding="utf-8"))
    m_zero_tvd = base_json["M_ZERO"]["mean_tvd"]
    m_triv_tvd = base_json["M_TRIVIAL"]["mean_tvd"]
    base = pd.read_parquet(os.path.join(C.D_METRICS,
                                        "STEP5I_RECOVERY_BASELINE_PER_STATE.parquet"))
    ctrl = pd.read_csv(os.path.join(C.D_CONTROL, "STEP5I_CONTROL_METRICS.csv"))
    ctrl_state = pd.read_parquet(os.path.join(C.D_CONTROL,
                                              "STEP5I_CONTROL_PER_STATE.parquet"))
    man = pd.read_csv(os.path.join(C.D_MANIFEST,
                                   "STEP5I_RECOVERY_CHECKPOINT_MANIFEST.csv"))

    ni, _ = C.load_frozen_ni()
    ev = ni[ni.fewshot_role == "RECOVERY_EVALUATION"].reset_index(drop=True)
    y_ev = ev[C.TRUE_COLS].to_numpy(float)
    fallback = C.frozen_fallback_row()
    order = {s: i for i, s in enumerate(ev.state_id)}

    # -------------------------------------------------- M_REFIT metrics ---- #
    rows, refit_state = [], []
    for _, m in man.iterrows():
        p = pd.read_parquet(m["predictions"])
        p = p.set_index("state_id").loc[ev.state_id].reset_index()
        yp = p[["refit_pred_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)
        blk = C.metrics_block(y_ev, yp, fallback, zero_shot_tvd=m_zero_tvd)
        rows.append(dict(kind=m["kind"], replicate=int(m["replicate"]),
                         budget=int(m["budget"]), tag=m["tag"], method="M_REFIT",
                         estimator="PRIMARY_FORMULATION_REFIT",
                         n_anchor_compositions=int(m["n_anchor_compositions"]),
                         n_anchor_states=int(m["n_anchor_states"]),
                         anchor_fraction_of_training_rows=float(
                             m["anchor_fraction_of_training_rows"]),
                         best_epoch=int(m["best_epoch"]), **blk))
        d = pd.DataFrame({"tag": m["tag"], "state_id": p.state_id,
                          "tvd_M_REFIT": C.per_state_tvd(y_ev, yp),
                          "topo_correct_M_REFIT": (y_ev.argmax(1)
                                                   == yp.argmax(1)).astype(int)})
        refit_state.append(d)
    refit = pd.DataFrame(rows)
    refit_state = pd.concat(refit_state, ignore_index=True)
    refit_state.to_parquet(os.path.join(C.D_METRICS,
                                        "STEP5I_REFIT_PER_STATE.parquet"), index=False)

    allm = pd.concat([refit, ctrl], ignore_index=True)
    allm = allm.sort_values(["kind", "replicate", "budget", "method"])

    rand = allm[allm.kind == "RANDOM"].copy()
    maxi = allm[allm.kind == "MAXIMIN"].copy()
    rand.to_csv(os.path.join(C.D_METRICS,
                             "STEP5I_RANDOM_FEWSHOT_PER_REPLICATE.csv"),
                index=False, encoding="utf-8")
    maxi.to_csv(os.path.join(C.D_METRICS, "STEP5I_MAXIMIN_FEWSHOT_METRICS.csv"),
                index=False, encoding="utf-8")

    # ------------------------------------------- across-replicate summary --- #
    agg = []
    for (method, budget), g in rand.groupby(["method", "budget"]):
        if len(g) != len(C.REPLICATES):
            raise SystemExit("FATAL: %s budget %d has %d replicates, expected %d."
                             % (method, budget, len(g), len(C.REPLICATES)))
        row = {"method": method, "budget": int(budget),
               "n_replicates": len(g),
               "replicate_tvds": ";".join("%.6f" % v for v in g.mean_tvd),
               "zero_shot_tvd_same_subset": m_zero_tvd,
               "trivial_tvd_same_subset": m_triv_tvd}
        for col in ("mean_tvd", "median_tvd", "p95_tvd", "topology_accuracy",
                    "topology_macro_f1", "skill_vs_trivial",
                    "relative_tvd_reduction_vs_zero_shot"):
            row["%s_mean" % col] = float(g[col].mean())
            row["%s_std" % col] = float(g[col].std(ddof=1))
            row["%s_min" % col] = float(g[col].min())
            row["%s_max" % col] = float(g[col].max())
        agg.append(row)
    agg = pd.DataFrame(agg).sort_values(["method", "budget"])
    agg.to_csv(os.path.join(C.D_METRICS, "STEP5I_RANDOM_FEWSHOT_METRICS.csv"),
               index=False, encoding="utf-8")

    # -------------------------------------------------- budget thresholds -- #
    refit_agg = agg[agg.method == "M_REFIT"].sort_values("budget")
    b_skill = next((int(b) for b, v in zip(refit_agg.budget, refit_agg.mean_tvd_mean)
                    if v < m_triv_tvd), None)
    target25 = 0.75 * m_zero_tvd
    b_rec25 = next((int(b) for b, v in zip(refit_agg.budget, refit_agg.mean_tvd_mean)
                    if v <= target25), None)
    thresholds = {
        "generated_utc": C.utcnow(),
        "definitions": {
            "B_skill": "smallest budget whose across-replicate mean M_REFIT TVD is "
                       "below the frozen PRIMARY trivial TVD on the SAME recovery "
                       "subset (spec 32)",
            "B_recovery25": "smallest budget whose across-replicate mean M_REFIT TVD "
                            "is <= 0.75 x the frozen PRIMARY zero-shot TVD on the "
                            "SAME recovery subset (spec 33, 34)"},
        "zero_shot_tvd_recovery_subset": m_zero_tvd,
        "trivial_tvd_recovery_subset": m_triv_tvd,
        "recovery25_target_tvd": target25,
        "full_blind_tvd_not_used_here": C.ZERO_SHOT_FULL_MEAN_TVD,
        "per_budget_mean_tvd": {int(b): float(v) for b, v in
                                zip(refit_agg.budget, refit_agg.mean_tvd_mean)},
        "B_skill": b_skill if b_skill is not None else ">32",
        "B_recovery25": b_rec25 if b_rec25 is not None else ">32",
        "budgets_evaluated": C.BUDGETS,
        "all_budgets_retained": True,
        "post_hoc_selection_performed": False}
    C.write_json(os.path.join(C.D_METRICS, "STEP5I_BUDGET_THRESHOLDS.json"), thresholds)

    # ------------------------------------- model vs control, per cell ------- #
    cmp_rows = []
    piv = allm.pivot_table(index=["kind", "replicate", "budget", "tag"],
                           columns="method", values="mean_tvd").reset_index()
    topo = allm.pivot_table(index=["kind", "replicate", "budget", "tag"],
                            columns="method", values="topology_accuracy").reset_index()
    for (_, r), (_, t) in zip(piv.iterrows(), topo.iterrows()):
        cmp_rows.append({
            "kind": r["kind"], "replicate": int(r["replicate"]),
            "budget": int(r["budget"]), "tag": r["tag"],
            "M_ZERO_tvd": m_zero_tvd, "M_TRIVIAL_tvd": m_triv_tvd,
            "M_REFIT_tvd": float(r["M_REFIT"]), "R0_tvd": float(r["R0"]),
            "R1_tvd": float(r["R1"]), "R2_tvd": float(r["R2"]),
            "M_REFIT_minus_M_ZERO": float(r["M_REFIT"]) - m_zero_tvd,
            "M_REFIT_minus_R0": float(r["M_REFIT"]) - float(r["R0"]),
            "M_REFIT_minus_R1": float(r["M_REFIT"]) - float(r["R1"]),
            "M_REFIT_minus_R2": float(r["M_REFIT"]) - float(r["R2"]),
            "M_REFIT_beats_M_ZERO": bool(float(r["M_REFIT"]) < m_zero_tvd),
            "M_REFIT_beats_M_TRIVIAL": bool(float(r["M_REFIT"]) < m_triv_tvd),
            "M_REFIT_beats_R0": bool(float(r["M_REFIT"]) < float(r["R0"])),
            "M_REFIT_beats_R2": bool(float(r["M_REFIT"]) < float(r["R2"])),
            "M_REFIT_topology_accuracy": float(t["M_REFIT"]),
            "R0_topology_accuracy": float(t["R0"]),
            "R2_topology_accuracy": float(t["R2"]),
            "comparison_note": "descriptive; identical %d states for every method"
                               % C.N_RECOVERY_STATES})
    cmp = pd.DataFrame(cmp_rows).sort_values(["kind", "replicate", "budget"])
    cmp.to_csv(os.path.join(C.D_METRICS,
                            "STEP5I_MODEL_VS_CONTROL_COMPARISON.csv"),
               index=False, encoding="utf-8")

    # ------------------------------------------------------------ report ---- #
    print("\n  budget |   M_REFIT (mean of 3)  |      R0      |      R2      | "
          "M_ZERO %.6f  trivial %.6f" % (m_zero_tvd, m_triv_tvd))
    for b in C.BUDGETS:
        g = agg[agg.budget == b].set_index("method")
        print("    %4d | %.6f +/- %.6f | %.6f | %.6f"
              % (b, g.loc["M_REFIT", "mean_tvd_mean"], g.loc["M_REFIT", "mean_tvd_std"],
                 g.loc["R0", "mean_tvd_mean"], g.loc["R2", "mean_tvd_mean"]))
    print("\n  B_skill = %s | B_recovery25 = %s (target <= %.6f)"
          % (thresholds["B_skill"], thresholds["B_recovery25"], target25))
    r_beats = int(cmp[cmp.kind == "RANDOM"].M_REFIT_beats_R0.sum())
    print("  M_REFIT beats R0 in %d of %d random cells; beats R2 in %d"
          % (r_beats, len(cmp[cmp.kind == "RANDOM"]),
             int(cmp[cmp.kind == "RANDOM"].M_REFIT_beats_R2.sum())))
    print("\nSTAGE 3: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
