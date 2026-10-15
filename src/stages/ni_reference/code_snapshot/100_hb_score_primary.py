# -*- coding: utf-8 -*-
"""
100_hb_score_primary.py
STEP5H-B Stage 3: PRIMARY BLIND SCORING - and nothing else (spec 25-35).

Binding constraints:
  - runs only after the target freeze exists (spec 25)
  - loads ONLY the frozen PRIMARY prediction file and the frozen target table;
    the SECONDARY score directory is not touched here (spec 25)
  - exact state_id alignment verified before any metric (spec 25)
  - scoring population: all CONVERGED Ni states; failed rows stay reported but
    are excluded from numerical error metrics (spec 26)
  - metrics are EXACTLY the frozen STEP5G metric-preregistration definitions
    (spec 27-29); trivial baseline is the frozen PRIMARY train-mean simplex
    (spec 31); skill = trivial mean TVD - PRIMARY mean TVD, never redefined
    (spec 32)
  - composition-level cluster bootstrap, 1000 replicates, seed 20260911 (spec 30)
  - no threshold is applied to the result and none is created (spec 47-49)
  - the score freeze is written IMMEDIATELY after the metrics (spec 34); after
    that nothing about the PRIMARY result may change (spec 35)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hb_common as C


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5H-B  Stage 3: PRIMARY blind scoring (SECONDARY not loaded here)")
    print("=" * 78)

    tf_path = os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGET_FREEZE.json")
    if not os.path.exists(tf_path):
        raise SystemExit("FATAL: target freeze missing - run 99 first (spec 25).")
    tf = json.load(open(tf_path, encoding="utf-8"))

    # hard gate again, at scoring time (spec 2 applies to the whole stage)
    C.verify_hash(C.G_NI_COMP_CSV, "ni_compositions_csv", "Ni composition table")
    C.verify_hash(C.HA_PRIMARY_PRED, "primary_predictions", "PRIMARY predictions")
    C.verify_hash(C.G_FREEZE_JSON, "step5g_pipeline_freeze", "STEP5G pipeline freeze")

    targets = pd.read_parquet(os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGETS.parquet"))
    if C.sha256_file(os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGETS.parquet")) \
            != tf["sha256"]:
        raise SystemExit("FATAL: target table changed after freeze (spec 23).")
    pred = pd.read_parquet(C.HA_PRIMARY_PRED)

    # exact state_id alignment (spec 25)
    if set(pred.state_id) != set(targets.state_id) or len(pred) != len(targets):
        raise SystemExit("FATAL: prediction/target state_id misalignment")
    df = targets.merge(pred, on=["state_id", "composition_id", "Temperature_C",
                                 "fewshot_role"], suffixes=("", "_p"),
                       validate="1:1")
    assert len(df) == C.N_STATES

    scored = df[df.calphad_status == "CONVERGED"].copy()
    y = scored[C.TRUE_COLS].to_numpy(float)
    p = scored[["pred_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)
    assert np.isfinite(y).all() and np.isfinite(p).all()

    m = C.metric_block(y, p, C.PRIMARY_TRIVIAL)
    v = C.tvd(p, y)
    boot = C.cluster_bootstrap_mean_tvd(v, scored.composition_id.to_numpy())

    per_t = (scored.assign(tvd=v)
             .groupby("Temperature_C")
             .apply(lambda g: pd.Series({
                 "n": len(g), "mean_tvd": g.tvd.mean(),
                 "median_tvd": g.tvd.median(),
                 "topology_accuracy": (C.topology_of(g[C.TRUE_COLS].to_numpy(float))
                                       == C.topology_of(
                     g[["pred_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float))).mean()}),
                 include_groups=False)
             .reset_index())
    per_t.to_csv(os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_PER_TEMPERATURE.csv"),
                 index=False, encoding="utf-8")

    fam_rows = [{"family": f,
                 "mae": float(np.abs(p[:, j] - y[:, j]).mean()),
                 "true_mean": float(y[:, j].mean()),
                 "pred_mean": float(p[:, j].mean())}
                for j, f in enumerate(C.FAMILIES_ORDER)]
    pd.DataFrame(fam_rows).to_csv(os.path.join(C.D_PSCORE,
                                               "STEP5H_B_PRIMARY_PER_FAMILY.csv"),
                                  index=False, encoding="utf-8")

    err = scored[["state_id", "composition_id", "Temperature_C", "fewshot_role",
                  "true_topology", "pred_topology"]].copy()
    err["tvd"] = v
    err["topology_correct"] = (err.true_topology == err.pred_topology).astype(int)
    err.to_parquet(os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_BLIND_ERRORS.parquet"),
                   index=False)

    metrics = {
        "role": "PRIMARY", "model": "ORIGINAL6_DIRECT_MLP",
        "endpoint": "mean TVD over all converged Ni blind states (frozen STEP5G "
                    "metric preregistration)",
        "n_formal": C.N_STATES,
        "n_converged": int(len(scored)),
        "n_failed": int((df.calphad_status == "FAILED").sum()),
        "n_not_in_store": int((df.calphad_status == "NOT_IN_STORE").sum()),
        "n_scored": m["n_rows"],
        "mean_tvd": m["mean_tvd"], "median_tvd": m["median_tvd"],
        "p95_tvd": m["p95_tvd"], "macro_mae": m["macro_mae"],
        "per_family_mae": m["per_family_mae"],
        "topology_accuracy": m["topology_accuracy"],
        "topology_macro_f1": m["topology_macro_f1"],
        "trivial_baseline": C.PRIMARY_TRIVIAL,
        "trivial_mean_tvd": m["trivial_mean_tvd"],
        "skill_vs_trivial": m["skill_vs_trivial"],
        "relative_skill": m["relative_skill"],
        "simplex_fallback_count": int(scored.simplex_fallback.sum()),
        "bootstrap_mean_tvd": boot,
        "true_topology_distribution":
            {k: float(x) for k, x in
             scored.true_topology.value_counts(normalize=True).items()},
        "pred_topology_distribution":
            {k: float(x) for k, x in
             scored.pred_topology.value_counts(normalize=True).items()},
        "interpretation_guides_descriptive_only": {
            "historical_original_six_retrospective_loeo_mean_tvd":
                C.HISTORICAL_LOEO_ORIGINAL6,
            "note": ("per STEP5G_CLARIFICATION_01 these historical values are NOT "
                     "scores of this exact production checkpoint; they are "
                     "model-family context only (spec 47, 48)"),
            "below_0.752644_means": "better than the average historical original-six "
                                    "retrospective LOEO (descriptive only)",
            "below_0.50_means": "relatively strong prospective transfer "
                                "(descriptive only)"},
        "success_threshold": None,
        "no_journal_claim": True,
    }
    C.write_json(os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_BLIND_METRICS.json"),
                 metrics)
    pd.DataFrame(boot, index=[0]).to_csv(
        os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_BOOTSTRAP.csv"), index=False,
        encoding="utf-8")

    # ---- PRIMARY SCORE FREEZE, immediately (spec 34) -------------------------- #
    arts = {f: os.path.join(C.D_PSCORE, f) for f in
            ("STEP5H_B_PRIMARY_BLIND_METRICS.json",
             "STEP5H_B_PRIMARY_BLIND_ERRORS.parquet",
             "STEP5H_B_PRIMARY_BOOTSTRAP.csv",
             "STEP5H_B_PRIMARY_PER_TEMPERATURE.csv",
             "STEP5H_B_PRIMARY_PER_FAMILY.csv")}
    C.write_json(os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_SCORE_FREEZE.json"), {
        "freeze_id": "STEP5H_B_PRIMARY_SCORE_FREEZE_V1",
        "generated_utc": C.utcnow(),
        "primary_prediction_sha256": C.FROZEN["primary_predictions"],
        "target_sha256": tf["sha256"],
        "metric_prereg_sha256": C.sha256_file(C.G_METRIC_JSON),
        "mean_tvd": m["mean_tvd"],
        "ci95": [boot["ci95_low"], boot["ci95_high"]],
        "all_secondary_metrics": {k: metrics[k] for k in
                                  ("median_tvd", "p95_tvd", "macro_mae",
                                   "per_family_mae", "topology_accuracy",
                                   "topology_macro_f1", "trivial_mean_tvd",
                                   "skill_vs_trivial", "relative_skill",
                                   "simplex_fallback_count", "n_scored")},
        "trivial_baseline_tvd": m["trivial_mean_tvd"],
        "skill": m["skill_vs_trivial"],
        "row_count": m["n_rows"],
        "artifact_hashes": {os.path.basename(p): C.sha256_file(p)
                            for p in arts.values()},
        "immutable": True,
        "scored_before_secondary": True})
    print("  PRIMARY mean TVD %.6f | 95%% CI [%.6f, %.6f] | topology acc %.4f | "
          "trivial %.6f | skill %+.6f" % (m["mean_tvd"], boot["ci95_low"],
                                          boot["ci95_high"], m["topology_accuracy"],
                                          m["trivial_mean_tvd"], m["skill_vs_trivial"]))
    print("  scored %d converged of %d formal (failed %d excluded from error metrics "
          "but reported)" % (m["n_rows"], C.N_STATES, metrics["n_failed"]))
    print("STAGE 3: PASS - PRIMARY score frozen; SECONDARY may now be loaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
