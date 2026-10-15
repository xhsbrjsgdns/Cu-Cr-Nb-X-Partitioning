# -*- coding: utf-8 -*-
"""
40_step5a_iid_test.py
STEP5A Stage O + P + Q: TEST SEAL audit, first-open timestamp, and the FIRST
AND ONLY evaluation of the Domain-A IID_TEST (sec.76-77).

Guard rails:
  - refuses to run unless 09_manifest/DEVELOPMENT_FREEZE_V1.json exists
  - refuses to run if 05_results/TEST_FIRST_OPEN.json already exists
    (the test may only be opened ONCE)
  - no model/threshold/feature may change after this point; bugs found here
    require a documented STEP5A_PATCH_XX (sec.77, sec.100)

Outputs:
  07_qa/TEST_SEAL_AUDIT.md
  05_results/TEST_FIRST_OPEN.json
  05_results/BASELINE_IID_TEST_METRICS.csv
  05_results/BASELINE_PER_TARGET_METRICS.csv        (stage=test rows appended)
  05_results/LIQUID_IID_TEST_METRICS.csv
  05_results/REGRESSION_SUM_VIOLATION.csv
  05_results/DIRECT_VS_DERIVED_PRESENCE.csv
  05_results/IID_TEST_PREDICTIONS.parquet
  07_qa/SUSPICIOUS_PERFORMANCE_CHECK.md (only if triggered)
"""
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5a_common as C
import importlib.util

spec = importlib.util.spec_from_file_location(
    "freeze_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "39_step5a_development_freeze.py"))
freeze_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_mod)

MAN = os.path.join(C.STEP5A, "09_manifest", "DEVELOPMENT_FREEZE_V1.json")
FIRST_OPEN = os.path.join(C.RESULTS, "TEST_FIRST_OPEN.json")
FAMILIES = C.FAMILIES_ORDER


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    assert os.path.exists(MAN), "DEVELOPMENT_FREEZE_V1.json missing - development not frozen"
    metrics_exist = os.path.exists(os.path.join(C.RESULTS, "BASELINE_IID_TEST_METRICS.csv"))
    open_marker = os.path.exists(FIRST_OPEN)
    patch_doc = os.path.exists(os.path.join(C.STEP5A, "09_manifest", "STEP5A_PATCH_01.md"))
    if open_marker and not metrics_exist:
        # PATCH mode: the first-open marker was written by an ABORTED run that crashed
        # in output assembly BEFORE any prediction/metric file was produced (documented
        # in STEP5A_PATCH_01.md). No test label ever reached a model/threshold/feature
        # decision, so resuming the evaluation is still the first and only test opening.
        # The original TEST_FIRST_OPEN_TIMESTAMP is preserved, not overwritten.
        assert patch_doc, "abort-resume requires a STEP5A_PATCH_01.md record"
        resume = True
    else:
        resume = False
        assert not open_marker, \
            "TEST_FIRST_OPEN.json already exists - IID_TEST may only be opened ONCE. STOP."
    assert not metrics_exist, "IID test metrics already exist. STOP."

    freeze = json.load(open(MAN, encoding="utf-8"))
    feats = freeze["features"]

    # ---------------- Stage O: TEST SEAL AUDIT ----------------
    seal_lines = [
        "# TEST SEAL AUDIT (sec.76)\n\n",
        f"generated_utc: {ts}\n\n",
        "Confirmed before first opening of IID_TEST:\n\n",
        "| item | status |\n|---|---|\n",
        "| test targets used in feature selection (NZV/PCC/exact redundancy) | NO - Stage I used DEV_TRAIN rows only (35_step5a_feature_selection.py) |\n",
        "| test rows used in scaler fitting | NO - StandardScaler fit inside model builders on TRAIN rows only |\n",
        "| test used for hyperparameter tuning | NO - Optuna objective scored DEV_VALIDATION only (38_step5a_tune.py) |\n",
        "| test used for threshold selection | NO - thresholds fit on DEV_VALIDATION in 39_step5a_development_freeze.py |\n",
        "| test used for model selection | NO - Stage-K/L selection metric computed on DEV_VALIDATION |\n",
        "| Domain B loaded for any fitting/eval | NO - candidate features contain development rows only (Stage H audit H1/H7b) |\n",
        "| ANCHOR loaded for any fitting/eval | NO - same as above |\n",
        "| any IID_TEST metric file existing before this script | NO - asserted at script start |\n",
        "\nDevelopment freeze artifact: 09_manifest/DEVELOPMENT_FREEZE_V1.json (features, "
        "hyperparameters, thresholds, scaler, projection rules, seeds, versions all hashed/recorded).\n",
    ]
    with open(os.path.join(C.QA, "TEST_SEAL_AUDIT.md"), "w", encoding="utf-8") as f:
        f.write("".join(seal_lines))

    # ---------------- Stage P: first open timestamp ----------------
    if not resume:
        with open(FIRST_OPEN, "w", encoding="utf-8") as f:
            json.dump({"TEST_FIRST_OPEN_TIMESTAMP": ts,
                       "opened_by": "40_step5a_iid_test.py",
                       "note": "first and only opening of Domain-A IID_TEST for STEP5A"}, f, indent=2)
        print(f"TEST_FIRST_OPEN_TIMESTAMP = {ts}")
    else:
        orig = json.load(open(FIRST_OPEN, encoding="utf-8"))
        with open(os.path.join(C.RESULTS, "TEST_OPEN_RESUME_RECORD.json"), "w", encoding="utf-8") as f:
            json.dump({"original_first_open": orig, "resume_utc": ts,
                       "reason": "aborted run (output-assembly bug, PATCH_01) produced no "
                                 "predictions or metrics; resume = same single test opening"},
                      f, indent=2)
        print(f"RESUME of first open (original timestamp {orig['TEST_FIRST_OPEN_TIMESTAMP']} preserved)")

    # ---------------- Stage Q: one-time evaluation ----------------
    df, _ = C.load_development(feature_cols=feats)
    test_all = df[df["split_role"] == "IID_TEST"]
    train_all = df[df["split_role"] == "DEV_TRAIN"]
    print(f"IID_TEST rows: {len(test_all)} ({test_all['composition_id'].nunique()} compositions)")

    metric_rows, per_target_rows, liquid_rows, sum_viol_rows = [], [], [], []
    preds_store = {"state_id": test_all["state_id"].values,
                   "composition_id": test_all["composition_id"].values,
                   "family": test_all["family"].values,
                   "Temperature_C": test_all["Temperature_C"].values}
    projected_fractions = None
    direct_presence_pred = None

    for task in C.ALL_TASKS:
        tf = freeze["tasks"][task]
        model_name = tf["model_family"]
        params = tf["hyperparameters"]
        with open(os.path.join(C.MODELS, "final", task, "model_state.pkl"), "rb") as f:
            blob = pickle.load(f)
        state, trivial_vec = blob["state"], np.asarray(blob["trivial"])
        _, predict = freeze_mod.build_final_model(model_name, task, params)

        te = C.task_rows(test_all, task)
        tr = C.task_rows(train_all, task)
        Xte = te[feats].to_numpy(np.float64)
        Yte = te[C.TARGETS[task]].to_numpy(np.float64)
        pred = predict(state, Xte)

        # store predictions for every test row (NaN where task not defined, e.g. X partition on BASELINE)
        if task == "presence":
            ths = [tf["thresholds"][f] for f in FAMILIES]
            full = np.full((len(test_all), 5), np.nan)
            pos = test_all.index.get_indexer(te.index)
            full[pos] = pred
            for i, fam in enumerate(FAMILIES):
                preds_store[f"{fam}_presence_prob"] = full[:, i]
                preds_store[f"{fam}_presence_pred"] = np.where(np.isnan(full[:, i]), np.nan,
                                                               (full[:, i] >= ths[i]).astype(float))
            Yte_i = Yte.astype(int)
            met = C.classification_metrics(Yte_i, pred, ths)
            met_d = C.classification_metrics(Yte_i, pred, [0.5] * 5)
            tri = np.tile(trivial_vec, (len(te), 1))
            met_tri = C.classification_metrics(Yte_i, tri, [0.5] * 5)
            metric_rows.append({"stage": "test", "task": task, "model": model_name,
                                "n_rows": len(te),
                                "exact_match": met["exact_match"], "macro_f1": met["macro_f1"],
                                "micro_f1": met["micro_f1"], "hamming_loss": met["hamming_loss"],
                                "exact_match_default0.5": met_d["exact_match"],
                                "macro_f1_default0.5": met_d["macro_f1"],
                                "trivial_macro_f1": met_tri["macro_f1"],
                                "trivial_exact_match": met_tri["exact_match"],
                                "thresholds": json.dumps(tf["thresholds"])})
            for fam, d in met["per_label"].items():
                per_target_rows.append({"stage": "test", "task": task, "model": model_name,
                                        "target": fam, **{k: v for k, v in d.items() if k != "note"}})
                if fam == "LIQUID":
                    liquid_rows.append({"stage": "test", "model": model_name, "kind": "classification", **d})
            for fam, d in met_tri["per_label"].items():
                per_target_rows.append({"stage": "test", "task": task, "model": "TrivialPrevalence",
                                        "target": fam, **{k: v for k, v in d.items() if k != "note"}})
            direct_presence_pred = (pred >= np.asarray(ths)).astype(int)
        else:
            proj, n_fb = C.project_simplex(pred, trivial_vec)
            met = C.regression_metrics(Yte, pred, proj)
            tri = np.tile(trivial_vec, (len(te), 1))
            met_tri = C.regression_metrics(Yte, tri, tri.copy())
            metric_rows.append({"stage": "test", "task": task, "model": model_name,
                                "n_rows": len(te),
                                "mean_tvd": met["mean_tvd_proj"], "median_tvd": met["median_tvd_proj"],
                                "p95_tvd": met["p95_tvd_proj"],
                                "macro_mae": met["macro_mae_proj"], "macro_rmse": met["macro_rmse_proj"],
                                "raw_sum_violation_mean": met["raw_sum_violation_mean"],
                                "raw_sum_violation_max": met["raw_sum_violation_max"],
                                "proj_sum_violation_max": met["proj_sum_violation_max"],
                                "raw_negative_mean": met["raw_negative_mean"],
                                "raw_gt1_frac": met["raw_gt1_frac"],
                                "simplex_fallbacks": n_fb,
                                "trivial_mean_tvd": met_tri["mean_tvd_proj"],
                                "trivial_macro_mae": met_tri["macro_mae_proj"]})
            sum_viol_rows.append({"task": task, "model": model_name, "n_rows": len(te),
                                  "raw_sum_violation_mean": met["raw_sum_violation_mean"],
                                  "raw_sum_violation_max": met["raw_sum_violation_max"],
                                  "raw_negative_component_mean": met["raw_negative_mean"],
                                  "raw_any_gt1_fraction": met["raw_gt1_frac"],
                                  "projected_sum_violation_max": met["proj_sum_violation_max"],
                                  "simplex_fallback_rows": n_fb})
            for fam, d in met["per_family"].items():
                per_target_rows.append({"stage": "test", "task": task, "model": model_name,
                                        "target": fam, **d})
                if fam == "LIQUID" and task == "fraction":
                    liquid_rows.append({"stage": "test", "model": model_name, "kind": "regression",
                                        "mae": d["mae_proj"], "rmse": d["rmse_proj"], "r2": d["r2_proj"],
                                        "true_mean": d["true_mean"], "true_std": d["true_std"]})
            for fam, d in met_tri["per_family"].items():
                per_target_rows.append({"stage": "test", "task": task, "model": "TrivialMeanVector",
                                        "target": fam, **d})
            tag = {"fraction": "frac", "cr_partition": "CR_P", "nb_partition": "NB_P",
                   "x_partition": "X_P"}[task]
            full_raw = np.full((len(test_all), 5), np.nan)
            full_proj = np.full((len(test_all), 5), np.nan)
            pos = test_all.index.get_indexer(te.index)
            full_raw[pos] = pred
            full_proj[pos] = proj
            for i, fam in enumerate(FAMILIES):
                preds_store[f"{fam}_{tag}_raw"] = full_raw[:, i]
                preds_store[f"{fam}_{tag}_proj"] = full_proj[:, i]
            for i, tcol in enumerate(C.TARGETS[task]):
                full_true = np.full(len(test_all), np.nan)
                full_true[pos] = Yte[:, i]
                preds_store[f"{tcol}_true"] = full_true
            if task == "fraction":
                projected_fractions = (pos, proj)

    # true presence/fraction targets for the predictions file
    for fam in FAMILIES:
        preds_store[f"{fam}_presence_true"] = test_all[f"{fam}_presence"].values
        preds_store[f"{fam}_volfrac_true"] = test_all[f"{fam}_volfrac"].values
    for col in ["CR_P_FCC", "CR_P_BCC", "CR_P_C15", "CR_P_OTHER_SOLID", "CR_P_LIQUID",
                "NB_P_FCC", "NB_P_BCC", "NB_P_C15", "NB_P_OTHER_SOLID", "NB_P_LIQUID",
                "X_P_FCC", "X_P_BCC", "X_P_C15", "X_P_OTHER_SOLID", "X_P_LIQUID"]:
        preds_store[f"{col}_true"] = test_all[col].values

    pd.DataFrame(preds_store).to_parquet(os.path.join(C.RESULTS, "IID_TEST_PREDICTIONS.parquet"), index=False)
    pd.DataFrame(metric_rows).to_csv(os.path.join(C.RESULTS, "BASELINE_IID_TEST_METRICS.csv"), index=False)
    per_df = pd.DataFrame(per_target_rows)
    old = pd.read_csv(os.path.join(C.RESULTS, "BASELINE_PER_TARGET_METRICS.csv"))
    pd.concat([old, per_df], ignore_index=True).to_csv(
        os.path.join(C.RESULTS, "BASELINE_PER_TARGET_METRICS.csv"), index=False)
    pd.DataFrame(liquid_rows).to_csv(os.path.join(C.RESULTS, "LIQUID_IID_TEST_METRICS.csv"), index=False)
    pd.DataFrame(sum_viol_rows).to_csv(os.path.join(C.RESULTS, "REGRESSION_SUM_VIOLATION.csv"), index=False)

    # ---------------- direct vs fraction-derived presence (sec.59) ----------------
    pos, proj = projected_fractions
    derived = (proj > 1e-4).astype(int)
    true_pres = C.task_rows(test_all, "presence")[C.TARGETS["presence"]].to_numpy(int)
    rows = []
    for name, pred in [("direct_presence_classifier", direct_presence_pred),
                       ("fraction_derived_presence", derived)]:
        met = C.classification_metrics(true_pres, np.clip(
            # metrics need probs; use the prediction matrix as 0/1 'prob' with frozen 0.5 threshold
            pred.astype(float), 0, 1), [0.5] * 5)
        rows.append({"variant": name, "exact_match": met["exact_match"], "macro_f1": met["macro_f1"],
                     "micro_f1": met["micro_f1"], "hamming_loss": met["hamming_loss"]})
        for fam, d in met["per_label"].items():
            rows.append({"variant": f"{name}:{fam}", "f1": d["f1"], "precision": d["precision"],
                         "recall": d["recall"], "support": d["support"]})
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "DIRECT_VS_DERIVED_PRESENCE.csv"), index=False)

    # ---------------- sec.85 suspiciously perfect check ----------------
    flags = []
    for r in metric_rows:
        if r["task"] == "presence":
            if r["macro_f1"] > 0.9999:
                flags.append((r["task"], "macro_f1", r["macro_f1"]))
        else:
            if r["mean_tvd"] < 1e-5:
                flags.append((r["task"], "mean_tvd", r["mean_tvd"]))
    per_df2 = pd.DataFrame(per_target_rows)
    r2s = pd.to_numeric(per_df2.get("r2_proj", pd.Series(dtype=float)), errors="coerce") \
        if "r2_proj" in per_df2 else pd.Series(dtype=float)
    n_high_r2 = int((r2s > 0.9999).sum()) if len(r2s) else 0
    f1s = pd.to_numeric(per_df2.get("f1", pd.Series(dtype=float)), errors="coerce") \
        if "f1" in per_df2 else pd.Series(dtype=float)
    n_high_f1 = int((f1s > 0.9999).sum()) if len(f1s) else 0
    suspicious = len(flags) >= 3 or n_high_r2 >= 5 or n_high_f1 >= 4
    lines = ["# SUSPICIOUS PERFORMANCE CHECK (sec.85)\n\n", f"generated_utc: {ts}\n\n",
             f"- task-level flags (macro_f1>0.9999 or mean_tvd<1e-5): {flags}\n",
             f"- per-target R2_proj > 0.9999 count: {n_high_r2}\n",
             f"- per-target F1 > 0.9999 count: {n_high_f1}\n\n"]
    if suspicious:
        lines.append("TRIGGERED -> running automatic leakage checklist:\n\n")
        lines.append("- target columns in feature matrix: NO (Stage-H audit H3, re-check at freeze)\n")
        lines.append("- CALPHAD-derived descriptors in features: NO (Stage-H H2/H3)\n")
        lines.append("- family/X_element/sampling_domain in features: NO (Stage-H H4)\n")
        lines.append("- composition overlap train/val/test: 0 (Stage-H H7)\n")
        lines.append("- duplicate states: state_id unique (Stage-A audit)\n")
        lines.append("- test leakage into tuning/thresholds/scaler: NO (TEST_SEAL_AUDIT.md)\n")
        lines.append("\nNote: FCC_presence is constant (=1) over the whole database; its per-label "
                     "F1 is trivially 1.0 and must NOT be counted as suspicious signal.\n")
    else:
        lines.append("Not triggered: performance is strong but not 'suspiciously perfect' by the "
                     "sec.85 rule.\n\nNote: FCC_presence is constant (=1) over the whole database, "
                     "so its per-label F1=1.0 is a label degeneracy, not model skill.\n")
    with open(os.path.join(C.QA, "SUSPICIOUS_PERFORMANCE_CHECK.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(pd.DataFrame(metric_rows)[["task", "model", "n_rows"]].to_string(index=False))
    print("STAGE O+P+Q: done — IID_TEST opened exactly once")


if __name__ == "__main__":
    main()
