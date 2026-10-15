# -*- coding: utf-8 -*-
"""
36_step5a_feature_ablation.py
STEP5A Stage J: F1 -> F1+F2 -> F1+F2+F3 -> F1+F2+F3+F4 ablation (sec.50-51).

Compute policy (sec.51): ONE stable tree baseline (XGBoost, GPU hist), a fixed
family-stratified 20%-composition subset of DEV_TRAIN, and the full
DEV_VALIDATION. No huge grid. IID_TEST is NOT touched.

Ablation thresholds are the default 0.5 for comparability; per-label
validation-tuned thresholds are a Stage-L/K concern.

Outputs:
  05_results/FEATURE_SET_ABLATION.csv
  05_results/FEATURE_ABLATION_Q1_Q4.md
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5a_common as C

FEATURE_SETS = {
    "F1": "SELECTED_FEATURES_F1.txt",
    "F1_F2": "SELECTED_FEATURES_F1_F2.txt",
    "F1_F2_F3": "SELECTED_FEATURES_F1_F2_F3.txt",
    "F1_F2_F3_F4": "SELECTED_FEATURES_F1_F2_F3_F4.txt",
}
XGB_PARAMS = dict(n_estimators=300, max_depth=8, learning_rate=0.1,
                  subsample=0.9, colsample_bytree=0.9, random_state=C.MASTER_SEED,
                  n_jobs=-1, tree_method="hist", device="cuda")


def fit_predict_classifier(Xtr, ytr, Xva):
    """One XGBClassifier per label; constant labels handled explicitly."""
    from xgboost import XGBClassifier
    probs = np.zeros((len(Xva), ytr.shape[1]))
    for i in range(ytr.shape[1]):
        col = ytr[:, i]
        if len(np.unique(col)) < 2:
            probs[:, i] = float(col.mean())
            continue
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(Xtr, col)
        probs[:, i] = m.predict_proba(Xva)[:, 1]
    return probs


def fit_predict_regressor(Xtr, Ytr, Xva):
    from xgboost import XGBRegressor
    P = np.zeros((len(Xva), Ytr.shape[1]))
    for i in range(Ytr.shape[1]):
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(Xtr, Ytr[:, i])
        P[:, i] = m.predict(Xva)
    return P


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    all_feats = C.read_feature_list("SELECTED_FEATURES_F1_F2_F3_F4.txt")
    df, _ = C.load_development(feature_cols=all_feats)
    train_all = df[df["split_role"] == "DEV_TRAIN"]
    val_all = df[df["split_role"] == "DEV_VALIDATION"]
    train_sub = C.subset_compositions(train_all)
    print(f"ablation train subset: {train_sub['composition_id'].nunique()} compositions / {len(train_sub)} rows; "
          f"validation: {val_all['composition_id'].nunique()} / {len(val_all)}")

    rows = []
    for fs_name, fs_file in FEATURE_SETS.items():
        feats = C.read_feature_list(fs_file)
        for task in C.ALL_TASKS:
            tr = C.task_rows(train_sub, task)
            va = C.task_rows(val_all, task)
            Xtr = tr[feats].to_numpy(np.float64)
            Xva = va[feats].to_numpy(np.float64)
            t0 = time.time()
            if task in C.CLASSIFICATION_TASKS:
                ytr = tr[C.TARGETS[task]].to_numpy(int)
                yva = va[C.TARGETS[task]].to_numpy(int)
                prob = fit_predict_classifier(Xtr, ytr, Xva)
                met = C.classification_metrics(yva, prob, [0.5] * 5)
                # trivial baseline: train prevalence
                tri = C.trivial_classification_probs(ytr)
                tri_prob = np.tile(tri, (len(yva), 1))
                met_tri = C.classification_metrics(yva, tri_prob, [0.5] * 5)
                rows.append({"feature_set": fs_name, "task": task, "n_features": len(feats),
                             "model": "XGBClassifier(ablation)",
                             "macro_f1": met["macro_f1"], "exact_match": met["exact_match"],
                             "micro_f1": met["micro_f1"], "hamming_loss": met["hamming_loss"],
                             "trivial_macro_f1": met_tri["macro_f1"],
                             "trivial_exact_match": met_tri["exact_match"],
                             "seconds": round(time.time() - t0, 1)})
                for fam, d in met["per_label"].items():
                    rows.append({"feature_set": fs_name, "task": f"{task}:{fam}", "n_features": len(feats),
                                 "model": "XGBClassifier(ablation)", "f1": d["f1"],
                                 "precision": d["precision"], "recall": d["recall"],
                                 "support": d["support"], "pr_auc": d["pr_auc"],
                                 "trivial_f1": met_tri["per_label"][fam]["f1"],
                                 "seconds": None})
            else:
                Ytr = tr[C.TARGETS[task]].to_numpy(np.float64)
                Yva = va[C.TARGETS[task]].to_numpy(np.float64)
                raw = fit_predict_regressor(Xtr, Ytr, Xva)
                fb = C.trivial_regression_pred(Ytr)
                fb = fb / fb.sum()
                proj, n_fb = C.project_simplex(raw, fb)
                met = C.regression_metrics(Yva, raw, proj)
                tri_raw = np.tile(fb, (len(Yva), 1))
                met_tri = C.regression_metrics(Yva, tri_raw, tri_raw.copy())
                rows.append({"feature_set": fs_name, "task": task, "n_features": len(feats),
                             "model": "XGBRegressor(ablation)",
                             "mean_tvd": met["mean_tvd_proj"], "macro_mae": met["macro_mae_proj"],
                             "raw_sum_violation_mean": met["raw_sum_violation_mean"],
                             "simplex_fallbacks": n_fb,
                             "trivial_mean_tvd": met_tri["mean_tvd_proj"],
                             "trivial_macro_mae": met_tri["macro_mae_proj"],
                             "seconds": round(time.time() - t0, 1)})
                for fam, d in met["per_family"].items():
                    rows.append({"feature_set": fs_name, "task": f"{task}:{fam}", "n_features": len(feats),
                                 "model": "XGBRegressor(ablation)", "mae": d["mae_proj"],
                                 "rmse": d["rmse_proj"], "r2": d["r2_proj"],
                                 "true_mean": d["true_mean"],
                                 "trivial_mae": met_tri["per_family"][fam]["mae_proj"],
                                 "seconds": None})
        print(f"  {fs_name} done")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.RESULTS, "FEATURE_SET_ABLATION.csv"), index=False)

    # Q1-Q4 summary
    def get(fs, task, col):
        r = out[(out["feature_set"] == fs) & (out["task"] == task)]
        return float(r[col].iloc[0]) if col in r.columns and len(r) else np.nan

    lines = ["# FEATURE ABLATION Q1-Q4 (Stage J)\n",
             f"generated_utc: {ts}\n",
             f"model: XGBoost tree baseline; train = fixed family-stratified 20% composition "
             f"subset of DEV_TRAIN ({train_sub['composition_id'].nunique()} compositions / {len(train_sub)} rows); "
             f"validation = full DEV_VALIDATION ({len(val_all)} rows). IID_TEST untouched.\n",
             "\n## Headline validation numbers by feature set\n",
             "\n| feature set | n_feat | presence macroF1 | fraction TVD | Cr TVD | Nb TVD | X TVD |\n|---|---|---|---|---|---|---|\n"]
    for fs in FEATURE_SETS:
        nf = len(C.read_feature_list(FEATURE_SETS[fs]))
        lines.append(f"| {fs} | {nf} | {get(fs,'presence','macro_f1'):.4f} | "
                     f"{get(fs,'fraction','mean_tvd'):.4f} | {get(fs,'cr_partition','mean_tvd'):.4f} | "
                     f"{get(fs,'nb_partition','mean_tvd'):.4f} | {get(fs,'x_partition','mean_tvd'):.4f} |\n")
    lines.append("\n## Answers\n")
    lines.append("Q1 (raw composition+T learns basic trends?): compare F1 row against trivial baselines "
                 "in FEATURE_SET_ABLATION.csv.\n")
    lines.append("Q2 (general physical descriptors gain?): F1_F2 vs F1.\n")
    lines.append("Q3 (X-specific descriptors gain?): F1_F2_F3 vs F1_F2.\n")
    lines.append("Q4 (wider Magpie set: gain or redundancy?): F1_F2_F3_F4 vs F1_F2_F3.\n")
    lines.append("\nFull per-label / per-family numbers: FEATURE_SET_ABLATION.csv\n")
    with open(os.path.join(C.RESULTS, "FEATURE_ABLATION_Q1_Q4.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print("STAGE J: done")


if __name__ == "__main__":
    main()
