# -*- coding: utf-8 -*-
"""
101_hb_score_secondary.py
STEP5H-B Stage 4: SECONDARY PROSPECTIVE SENSITIVITY SCORING (spec 36-40).

Runs ONLY after STEP5H_B_PRIMARY_SCORE_FREEZE.json exists (checked first).
Same frozen target table, same converged states, same metric definitions.
The frozen SECONDARY train-mean simplex is used for its OWN descriptive skill
sensitivity (spec 39); the PRIMARY scientific skill claim remains based on the
PRIMARY baseline. SECONDARY stays PROSPECTIVE SENSITIVITY ONLY: regardless of
the numbers, it cannot replace PRIMARY (spec 37), and no model selection is
performed anywhere in this stage (spec 41).
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
    print("STEP5H-B  Stage 4: SECONDARY prospective-sensitivity scoring")
    print("=" * 78)

    pf = os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_SCORE_FREEZE.json")
    if not os.path.exists(pf):
        raise SystemExit("FATAL: PRIMARY score freeze does not exist yet - SECONDARY "
                         "must not be scored first (spec 36).")
    pfz = json.load(open(pf, encoding="utf-8"))
    tf = json.load(open(os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGET_FREEZE.json"),
                        encoding="utf-8"))
    C.verify_hash(C.HA_SECONDARY_PRED, "secondary_predictions",
                  "SECONDARY predictions")
    C.verify_hash(C.G_NI_COMP_CSV, "ni_compositions_csv", "Ni composition table")

    targets = pd.read_parquet(os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGETS.parquet"))
    if C.sha256_file(os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGETS.parquet")) \
            != tf["sha256"]:
        raise SystemExit("FATAL: target table changed after freeze (spec 23).")
    pred = pd.read_parquet(C.HA_SECONDARY_PRED)
    if set(pred.state_id) != set(targets.state_id):
        raise SystemExit("FATAL: state_id misalignment")
    df = targets.merge(pred, on=["state_id", "composition_id", "Temperature_C",
                                 "fewshot_role"], suffixes=("", "_p"), validate="1:1")
    scored = df[df.calphad_status == "CONVERGED"].copy()
    y = scored[C.TRUE_COLS].to_numpy(float)
    p = scored[["pred_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)

    m = C.metric_block(y, p, C.SECONDARY_TRIVIAL)
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
    per_t.to_csv(os.path.join(C.D_SSCORE, "STEP5H_B_SECONDARY_PER_TEMPERATURE.csv"),
                 index=False, encoding="utf-8")
    fam_rows = [{"family": f, "mae": float(np.abs(p[:, j] - y[:, j]).mean()),
                 "true_mean": float(y[:, j].mean()), "pred_mean": float(p[:, j].mean())}
                for j, f in enumerate(C.FAMILIES_ORDER)]
    pd.DataFrame(fam_rows).to_csv(os.path.join(C.D_SSCORE,
                                               "STEP5H_B_SECONDARY_PER_FAMILY.csv"),
                                  index=False, encoding="utf-8")
    err = scored[["state_id", "composition_id", "Temperature_C", "fewshot_role",
                  "true_topology", "pred_topology"]].copy()
    err["tvd"] = v
    err["topology_correct"] = (err.true_topology == err.pred_topology).astype(int)
    err.to_parquet(os.path.join(C.D_SSCORE, "STEP5H_B_SECONDARY_BLIND_ERRORS.parquet"),
                   index=False)

    metrics = {
        "role": "SECONDARY - PROSPECTIVE SENSITIVITY ONLY",
        "model": "FULL11_DIRECT_MLP",
        "cannot_replace_primary_regardless_of_result": True,
        "n_scored": m["n_rows"],
        "mean_tvd": m["mean_tvd"], "median_tvd": m["median_tvd"],
        "p95_tvd": m["p95_tvd"], "macro_mae": m["macro_mae"],
        "per_family_mae": m["per_family_mae"],
        "topology_accuracy": m["topology_accuracy"],
        "topology_macro_f1": m["topology_macro_f1"],
        "own_trivial_baseline": C.SECONDARY_TRIVIAL,
        "own_trivial_mean_tvd": m["trivial_mean_tvd"],
        "own_descriptive_skill": m["skill_vs_trivial"],
        "bootstrap_mean_tvd": boot,
        "true_topology_distribution":
            {k: float(x) for k, x in
             scored.true_topology.value_counts(normalize=True).items()},
        "pred_topology_distribution":
            {k: float(x) for k, x in
             scored.pred_topology.value_counts(normalize=True).items()},
        "primary_skill_claim_basis": "PRIMARY trivial baseline (spec 39) - unchanged",
        "success_threshold": None,
    }
    C.write_json(os.path.join(C.D_SSCORE, "STEP5H_B_SECONDARY_BLIND_METRICS.json"),
                 metrics)
    arts = {f: os.path.join(C.D_SSCORE, f) for f in
            ("STEP5H_B_SECONDARY_BLIND_METRICS.json",
             "STEP5H_B_SECONDARY_BLIND_ERRORS.parquet",
             "STEP5H_B_SECONDARY_PER_TEMPERATURE.csv",
             "STEP5H_B_SECONDARY_PER_FAMILY.csv")}
    C.write_json(os.path.join(C.D_SSCORE, "STEP5H_B_SECONDARY_SCORE_FREEZE.json"), {
        "freeze_id": "STEP5H_B_SECONDARY_SCORE_FREEZE_V1",
        "generated_utc": C.utcnow(),
        "secondary_prediction_sha256": C.FROZEN["secondary_predictions"],
        "target_sha256": tf["sha256"],
        "metric_prereg_sha256": C.sha256_file(C.G_METRIC_JSON),
        "primary_score_freeze_sha256": C.sha256_file(pf),
        "primary_score_frozen_at": pfz["generated_utc"],
        "mean_tvd": m["mean_tvd"],
        "ci95": [boot["ci95_low"], boot["ci95_high"]],
        "topology_accuracy": m["topology_accuracy"],
        "own_descriptive_skill": m["skill_vs_trivial"],
        "row_count": m["n_rows"],
        "artifact_hashes": {os.path.basename(p): C.sha256_file(p)
                            for p in arts.values()},
        "immutable": True,
        "scored_after_primary_freeze": True})
    print("  SECONDARY mean TVD %.6f | topology acc %.4f | own descriptive skill %+.6f"
          % (m["mean_tvd"], m["topology_accuracy"], m["skill_vs_trivial"]))
    print("STAGE 4: PASS (sensitivity only; no model selection performed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
