# -*- coding: utf-8 -*-
"""60_step5e_audit.py — Stage A: hash STEP5C/STEP5D frozen artifacts and re-assert
the inherited STEP5D findings. READ-ONLY."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5e_common as C

EXPECTED_ETA2 = {"fraction": 0.077461, "cr_partition": 0.204647,
                 "nb_partition": 0.041723, "x_partition": 0.885455}
EXPECTED_MEANS = {"Tree": 0.850498, "MLP": 0.752644, "Ridge": 0.842478,
                  "Nearest-element analog": 0.865868}
TOPOLOGY_WRONG_SHARE = 0.96696


def main():
    checks, failures = [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL",
                       "detail": str(detail)[:200]})
        if not ok:
            failures.append(name)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    hashes = {}
    rel = [(C.STEP5C, ["02_fold_manifests/LOEO_MASTER_FOLD_MANIFEST.csv",
                       "11_manifest/STEP5C_SCORING_GATE.json",
                       "06_metrics/LOEO_ALL_FOLDS_METRICS.csv",
                       "06_metrics/LOEO_PER_TARGET_METRICS.csv"]
            + [f"11_manifest/FOLD_{e}_FREEZE.json" for e in C.HELDOUT_ELEMENTS]
            + [f"05_predictions/FOLD_{e}_HELDOUT_PREDICTIONS.parquet" for e in C.HELDOUT_ELEMENTS]),
           (C.STEP5D, ["03_variance_decomposition/TASK_ELEMENT_VARIANCE_DECOMPOSITION.csv",
                       "05_topology_analysis/X_PARTITION_ERROR_BY_TOPOLOGY_CORRECTNESS.csv",
                       "05_topology_analysis/X_PARTITION_TOPOLOGY_COLLAPSE.csv",
                       "07_model_bias_diagnostic/X_DIAGNOSTIC_MODEL_COMPARISON.csv",
                       "STEP5D_FAILURE_DIAGNOSIS_SUMMARY.json"]
            + [f"07_model_bias_diagnostic/FOLD_{e}/DIAGNOSTIC_HELDOUT_PREDICTIONS.parquet"
               for e in C.HELDOUT_ELEMENTS])]
    for root, files in rel:
        for f in files:
            p = os.path.join(root, f)
            hashes[f"{os.path.basename(root)}/{f}"] = C.sha256_file(p)
            check(f"exists {f}", os.path.exists(p))
    # immutability vs STEP5C gate record
    gate = C.load_json(os.path.join(C.STEP5C, "11_manifest", "STEP5C_SCORING_GATE.json"))
    check("STEP5C gate still AUTHORIZED", gate["scoring_authorized"] is True)
    fr = C.load_json(C.C_FREEZE("ZR"))
    art = os.path.join(C.STEP5C, fr["tasks"]["x_partition"]["model_artifact"])
    check("STEP5C fold freeze hash intact (ZR sample)",
          C.sha256_file(art) == fr["tasks"]["x_partition"]["model_artifact_sha256"])

    # inherited STEP5D numbers
    v = pd.read_csv(C.D_ETA2)
    wv = v[v.target_component == "WHOLE_VECTOR"].set_index("task")["eta2_element"]
    for task, val in EXPECTED_ETA2.items():
        check(f"eta2_element WHOLE_VECTOR {task} == {val}",
              abs(float(wv[task]) - val) < 1e-5, f"actual {float(wv[task]):.6f}")

    te = pd.read_csv(C.D_TOPOERR)
    tot = te["sum_tvd"].sum()
    wrong = te.loc[te.topology_status == "TOPOLOGY_WRONG", "sum_tvd"].sum()
    share = float(wrong / tot)
    check("topology-wrong share of summed X-partition TVD == 96.696%",
          abs(share - TOPOLOGY_WRONG_SHARE) < 5e-4, f"actual {share:.6f}")

    mc = pd.read_csv(C.D_MODELCOMP)
    mean_tvd = mc.groupby("method")["mean_tvd"].mean()
    for method, val in EXPECTED_MEANS.items():
        check(f"{method} 6-fold mean TVD == {val}",
              abs(float(mean_tvd[method]) - val) < 1e-5, f"actual {float(mean_tvd[method]):.6f}")
    ag = mc[(mc.heldout == "AG") & (mc.method == "Ridge")].iloc[0]
    check("Ridge rescues AG (TVD 0.428753, topo acc 0.980017)",
          abs(float(ag.mean_tvd) - 0.428753) < 1e-5
          and abs(float(ag.topology_accuracy) - 0.980017) < 1e-5,
          f"actual {float(ag.mean_tvd):.6f} / {float(ag.topology_accuracy):.6f}")

    # 5 unique training X identities per fold (spec: 18,375 X-bearing rows but 5 identities)
    man = C.load_fold_manifest()
    for e in C.HELDOUT_ELEMENTS:
        tr = man[(man.fold == f"FOLD_{e}") & (man.loeo_role == "FOLD_TRAIN")]
        xb = tr[tr.family != "BASELINE"]
        check(f"FOLD_{e}: 5 unique training X identities, {len(xb)} X-bearing compositions",
              xb.family.nunique() == 5 and len(xb) == 18375,
              f"{sorted(xb.family.unique())}")

    audit = {"stage": "STEP5E Stage A", "generated_utc": C.utcnow(),
             "status": "PASS" if not failures else "FAIL",
             "n_checks": len(checks), "failures": failures,
             "artifact_hashes": hashes, "checks": checks}
    C.write_json(os.path.join(C.D_QA, "STEP5E_STAGE_A_AUDIT.json"), audit)
    print(f"\nSTAGE A: {audit['status']} ({len(checks)} checks, {len(failures)} failures)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
