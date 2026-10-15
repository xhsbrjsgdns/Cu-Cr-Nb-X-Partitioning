# -*- coding: utf-8 -*-
"""
step5a_common.py
Shared definitions for STEP5A modeling stages (J/K/L/M-O/Q/R/S).

Hard rules encoded here:
  - development universe = sampling_domain A only (DEV_TRAIN / DEV_VALIDATION / IID_TEST)
  - Domain B, ANCHOR rows are NEVER loaded for training/eval in STEP5A
  - all data-dependent operations (scaler, class weights, thresholds, PCC,
    feature selection) are fit on DEV_TRAIN only
  - IID_TEST stays sealed until DEVELOPMENT_FREEZE_V1.json exists
  - trivial baselines: prevalence/majority (classification), DEV_TRAIN mean
    vector (regression)
  - simplex projection: clip at 0, renormalize; sum==0 fallback = DEV_TRAIN
    target mean simplex (fallback count logged); never uses val/test labels
"""
import json
import os

import numpy as np
import pandas as pd

FULL_V1 = r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1"
STEP5A = r"E:\Cu_CALPHAD_ML\04_ML\STEP5A_JMST_ML_FOUNDATION_V2"
FEAT = os.path.join(STEP5A, "02_features")
SPLITS = os.path.join(STEP5A, "03_splits")
MODELS = os.path.join(STEP5A, "04_models")
RESULTS = os.path.join(STEP5A, "05_results")
QA = os.path.join(STEP5A, "07_qa")

MASTER_SEED = 20260904
SUBSET_FRACTION = 0.20
FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]

META_COLS = ["state_id", "composition_id", "family", "X_element", "sampling_domain", "split_role"]

TARGETS = {
    "presence": [f"{f}_presence" for f in FAMILIES_ORDER],
    "fraction": [f"{f}_volfrac" for f in FAMILIES_ORDER],
    "cr_partition": [f"CR_P_{f}" for f in FAMILIES_ORDER],
    "nb_partition": [f"NB_P_{f}" for f in FAMILIES_ORDER],
    "x_partition": [f"X_P_{f}" for f in FAMILIES_ORDER],
}
REGRESSION_TASKS = ["fraction", "cr_partition", "nb_partition", "x_partition"]
CLASSIFICATION_TASKS = ["presence"]
ALL_TASKS = CLASSIFICATION_TASKS + REGRESSION_TASKS

FORBIDDEN_FEATURE_TOKENS = [
    "state_id", "composition_id", "family", "X_element", "sampling_domain",
    "state_converged", "solver_level_used", "bridge_used", "phase_count",
    "has_multiple_FCC_instances", "has_multiple_BCC_instances", "has_liquid",
    "_presence", "_volfrac", "_molefrac", "_AUX", "CR_P_", "NB_P_", "X_P_",
    "X_Psolid", "dominant_",
]


def read_feature_list(name):
    with open(os.path.join(FEAT, name), encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


PRIMARY_FEATURES = None  # lazy


def primary_features():
    global PRIMARY_FEATURES
    if PRIMARY_FEATURES is None:
        PRIMARY_FEATURES = read_feature_list("PRIMARY_FEATURE_LIST.txt")
    return PRIMARY_FEATURES


def load_development(feature_cols=None):
    """Load dev feature rows joined with targets. Domain A development rows only."""
    if feature_cols is None:
        feature_cols = primary_features()
    # PATCH_01 fix: dedupe the column spec - Temperature_C is both an explicit
    # metadata request and a member of the feature list; a duplicated spec made
    # read_parquet return the column TWICE (feature matrix had 44 columns).
    cand_cols = list(dict.fromkeys(META_COLS + ["Temperature_C"] + list(feature_cols)))
    cand = pd.read_parquet(os.path.join(FEAT, "FULL_V1_CANDIDATE_FEATURES.parquet"),
                           columns=cand_cols)
    assert not cand.columns.duplicated().any(), "duplicate columns after read"
    tgt_cols = sorted({c for t in ALL_TASKS for c in TARGETS[t]})
    mlr = pd.read_parquet(
        os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet"),
        columns=["state_id"] + tgt_cols)
    df = cand.merge(mlr, on="state_id", how="left", validate="1:1")
    assert df[tgt_cols[0]].notna().all(), "target join produced NaN on FCC_volfrac"
    # stratification metadata for post-hoc error analysis (NOT features)
    diag = pd.read_parquet(
        os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet"),
        columns=["state_id", "has_multiple_FCC_instances", "has_multiple_BCC_instances"])
    df = df.merge(diag, on="state_id", how="left", validate="1:1")
    return df, list(feature_cols)


def subset_compositions(df, fraction=SUBSET_FRACTION, seed=MASTER_SEED):
    """Family-stratified composition-level subset (never state-random)."""
    rng = np.random.default_rng(seed)
    comp = df[["composition_id", "family"]].drop_duplicates()
    keep = []
    for fam in ["BASELINE", "ZR", "TI", "CO", "SI", "AG", "MO"]:
        ids = np.sort(comp.loc[comp["family"] == fam, "composition_id"].values)
        n = int(round(len(ids) * fraction))
        keep.append(rng.choice(ids, size=n, replace=False))
    keep = np.concatenate(keep)
    return df[df["composition_id"].isin(keep)].copy()


def task_rows(df, task):
    """Rows usable for a task (X partition excludes BASELINE, never fills NaN with 0)."""
    if task == "x_partition":
        sub = df[df["family"] != "BASELINE"]
    else:
        sub = df
    cols = TARGETS[task]
    assert sub[cols].notna().all().all(), f"NaN targets in {task}"
    return sub


# ---------------- metrics ----------------

def tvd(pred, true):
    return 0.5 * np.abs(pred - true).sum(axis=1)


def project_simplex(Y, fallback_row):
    """clip negatives, renormalize; sum==0 -> fallback_row (DEV_TRAIN mean simplex)."""
    P = np.clip(Y, 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    n_fallback = int((s[:, 0] == 0).sum())
    out = np.where(s > 0, P / np.where(s > 0, s, 1.0), fallback_row)
    return out, n_fallback


def classification_metrics(y_true, y_prob, thresholds):
    """y_true (n,k) 0/1, y_prob (n,k). Returns dict of aggregate + per-label metrics."""
    from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                                 average_precision_score, hamming_loss)
    y_pred = (y_prob >= np.asarray(thresholds)).astype(int)
    out = {
        "exact_match": float((y_pred == y_true).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    per_label = {}
    for i, fam in enumerate(FAMILIES_ORDER):
        yt, yp, ypr = y_true[:, i], y_pred[:, i], y_prob[:, i]
        d = {"support": int(yt.sum()),
             "precision": float(precision_score(yt, yp, zero_division=0)),
             "recall": float(recall_score(yt, yp, zero_division=0)),
             "f1": float(f1_score(yt, yp, zero_division=0))}
        if len(np.unique(yt)) > 1:
            d["roc_auc"] = float(roc_auc_score(yt, ypr))
            d["pr_auc"] = float(average_precision_score(yt, ypr))
        else:
            d["roc_auc"] = None
            d["pr_auc"] = None
            d["note"] = "constant label on this split"
        per_label[fam] = d
    out["per_label"] = per_label
    return out


def regression_metrics(y_true, y_pred_raw, y_pred_proj):
    """Per-family MAE/RMSE/R2 on projected preds + TVD + sum violations."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    out = {"per_family": {}, "macro_mae_proj": float(np.abs(y_pred_proj - y_true).mean()),
           "macro_rmse_proj": float(np.sqrt(((y_pred_proj - y_true) ** 2).mean()))}
    tvd_proj = tvd(y_pred_proj, y_true)
    tvd_raw = tvd(y_pred_raw, y_true)
    out["mean_tvd_proj"] = float(tvd_proj.mean())
    out["median_tvd_proj"] = float(np.median(tvd_proj))
    out["p95_tvd_proj"] = float(np.percentile(tvd_proj, 95))
    out["mean_tvd_raw"] = float(tvd_raw.mean())
    out["raw_sum_violation_mean"] = float(np.abs(y_pred_raw.sum(axis=1) - 1).mean())
    out["raw_sum_violation_max"] = float(np.abs(y_pred_raw.sum(axis=1) - 1).max())
    out["raw_negative_mean"] = float(np.clip(y_pred_raw, None, 0).sum(axis=1).mean() * -1)
    out["raw_gt1_frac"] = float((y_pred_raw > 1).any(axis=1).mean())
    out["proj_sum_violation_max"] = float(np.abs(y_pred_proj.sum(axis=1) - 1).max())
    for i, fam in enumerate(FAMILIES_ORDER):
        yt = y_true[:, i]
        out["per_family"][fam] = {
            "mae_raw": float(mean_absolute_error(yt, y_pred_raw[:, i])),
            "mae_proj": float(mean_absolute_error(yt, y_pred_proj[:, i])),
            "rmse_raw": float(np.sqrt(mean_squared_error(yt, y_pred_raw[:, i]))),
            "rmse_proj": float(np.sqrt(mean_squared_error(yt, y_pred_proj[:, i]))),
            "r2_raw": float(r2_score(yt, y_pred_raw[:, i])) if yt.std() > 0 else None,
            "r2_proj": float(r2_score(yt, y_pred_proj[:, i])) if yt.std() > 0 else None,
            "true_mean": float(yt.mean()), "true_std": float(yt.std()),
        }
    return out


def fit_thresholds_f1(y_true, y_prob):
    """Per-label F1-maximizing threshold on validation; tie-break closest to 0.5."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 3)
    ths = []
    for i in range(y_true.shape[1]):
        from sklearn.metrics import f1_score
        best_f1, best_t = -1.0, 0.5
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            ths.append(0.5)
            continue
        for t in grid:
            f1 = f1_score(yt, (y_prob[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
                best_f1, best_t = f1, float(t)
        ths.append(best_t)
    return ths


def class_weights_from_train(y):
    """sklearn 'balanced' style weights per label from DEV_TRAIN only."""
    w = []
    for i in range(y.shape[1]):
        col = y[:, i]
        n = len(col)
        p1 = max(col.mean(), 1e-12)
        # weight for positive class; negative weight normalized to 1
        w.append(float((1.0 / p1) / 2.0) if p1 < 1.0 else 1.0)
    return w


def trivial_regression_pred(y_train):
    return y_train.mean(axis=0)


def trivial_classification_probs(y_train):
    return y_train.mean(axis=0)
