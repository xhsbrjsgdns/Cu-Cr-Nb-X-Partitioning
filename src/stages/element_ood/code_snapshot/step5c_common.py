# -*- coding: utf-8 -*-
"""
step5c_common.py
Shared definitions for STEP 5C - STRICT 6x LEAVE-ONE-ELEMENT-OUT
retrospective unseen-element OOD.

TERMINOLOGY (spec 0, inherited from PROTOCOL_AMENDMENT_V1):
    This stage is a RETROSPECTIVE leave-one-element-out / unseen-element OOD
    experiment. ZR/TI/CO/SI/AG/MO all exist historically in FULL_V1 and took part in
    the STEP5A six-element models. The strictness here comes from
        "the held-out element's targets are excluded from THIS fold's model
         development, preprocessing, feature selection, hyperparameter optimization,
         threshold selection and model selection"
    and NOT from "these data were never accessed". The words
    "never read" / "never accessed" / "fully untouched" / "prospective blind" are
    forbidden anywhere in STEP5C output. The only prospective blind test is the
    final Ni stage.

Hard rules encoded here:
  - primary LOEO universe = sampling_domain A ONLY (spec 6). Domain B never enters.
  - for fold X: every Domain-A composition with family == X is HELDOUT_ELEMENT_TEST,
    whatever its STEP5A split_role was (spec 8).
  - FOLD_TRAIN      = STEP5A DEV_TRAIN      minus family X   (spec 9)
    FOLD_VALIDATION = STEP5A DEV_VALIDATION minus family X
    the other elements' IID_TEST compositions are NEVER used to enlarge training.
  - every data-dependent operation (NZV, exact redundancy, PCC, scaler, class
    weights, tuning, thresholds, model selection, trivial baseline, simplex
    fallback) uses FOLD_TRAIN or FOLD_VALIDATION only - never the held-out element.
  - the STEP5A global 43-feature list is NOT the LOEO primary list (spec 20):
    every fold re-runs selection from the frozen 193-feature candidate schema.
  - ANCHOR, Domain B and Ni are sealed and never loaded.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths -- #
FULL_V1 = r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1"
STEP5A = r"E:\Cu_CALPHAD_ML\04_ML\STEP5A_JMST_ML_FOUNDATION_V2"
STEP5B = r"E:\Cu_CALPHAD_ML\05_OOD\STEP5B_DOMAIN_B_OOD_V1"
STEP5C = r"E:\Cu_CALPHAD_ML\05_OOD\STEP5C_6X_LOEO_V1"

D_CONFIG = os.path.join(STEP5C, "00_config")
D_PREREG = os.path.join(STEP5C, "01_preregistration")
D_FOLDS = os.path.join(STEP5C, "02_fold_manifests")
D_FEAT = os.path.join(STEP5C, "03_features")
D_MODELS = os.path.join(STEP5C, "04_models")
D_PRED = os.path.join(STEP5C, "05_predictions")
D_METRICS = os.path.join(STEP5C, "06_metrics")
D_DIST = os.path.join(STEP5C, "07_distance_analysis")
D_FIG = os.path.join(STEP5C, "08_figures")
D_QA = os.path.join(STEP5C, "09_qa")
D_LOGS = os.path.join(STEP5C, "10_logs")
D_MANIFEST = os.path.join(STEP5C, "11_manifest")
D_CODE = os.path.join(STEP5C, "code_snapshot")
ALL_DIRS = [D_CONFIG, D_PREREG, D_FOLDS, D_FEAT, D_MODELS, D_PRED, D_METRICS,
            D_DIST, D_FIG, D_QA, D_LOGS, D_MANIFEST, D_CODE]

A_FEAT = os.path.join(STEP5A, "02_features")
CANDIDATE_PARQUET = os.path.join(A_FEAT, "FULL_V1_CANDIDATE_FEATURES.parquet")
FEATURE_BUILD_SUMMARY = os.path.join(A_FEAT, "FEATURE_BUILD_SUMMARY.json")
SPLIT_MASTER = os.path.join(STEP5A, "03_splits", "COMPOSITION_SPLIT_MASTER_V1.csv")
ELEMENT_DESCRIPTORS = os.path.join(STEP5A, "01_feature_sources",
                                   "ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.csv")
ML_READY = os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet")
ML_ALL = os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_TARGET_ALL.parquet")

# --------------------------------------------------------------- protocol -- #
MASTER_SEED = 20260906                      # spec 62
HELDOUT_ELEMENTS = ["ZR", "TI", "CO", "SI", "AG", "MO"]   # frozen fold order (spec 5)
FOLD_ORDER = [f"FOLD_{e}" for e in HELDOUT_ELEMENTS]
FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
ALL_FAMILIES = ["BASELINE"] + HELDOUT_ELEMENTS

MODEL_FAMILIES = ["XGBoost", "CatBoost"]    # spec 25, frozen for all six folds
N_TRIALS = 15                                # spec 30
SUBSET_FRACTION = 0.20                       # spec 31, composition-level
PCC_THRESHOLD = 0.90                         # spec 22
EXACT_TOL = 1e-10
NZV_MINORITY_FRAC = 1e-4
BOOTSTRAP_REPS = 1000                        # spec 62
PRIMARY_DOMAIN = "A"                         # spec 6
N_DOMAIN_A_COMPOSITIONS = 35000              # spec 7, verified against the split master
N_HELDOUT_COMPOSITIONS = 5250                # spec 7/8: 5,250 per held-out family
N_TEMPERATURES = 11                          # frozen FULL_V1 grid

# spec 32 tie rule, fixed in advance
TIE_BAND = {"presence": 1e-3, "regression": 1e-4}
TIE_PREFERENCE = "smaller estimated fit cost (trees x depth); if still tied, XGBoost"

META_COLS = ["state_id", "composition_id", "family", "X_element", "sampling_domain",
             "split_role"]
HELPER_COLS = ["Temperature_K"]

TARGETS = {
    "presence": [f"{f}_presence" for f in FAMILIES_ORDER],
    "fraction": [f"{f}_volfrac" for f in FAMILIES_ORDER],
    "cr_partition": [f"CR_P_{f}" for f in FAMILIES_ORDER],
    "nb_partition": [f"NB_P_{f}" for f in FAMILIES_ORDER],
    "x_partition": [f"X_P_{f}" for f in FAMILIES_ORDER],
}
CLASSIFICATION_TASKS = ["presence"]
REGRESSION_TASKS = ["fraction", "cr_partition", "nb_partition", "x_partition"]
ALL_TASKS = CLASSIFICATION_TASKS + REGRESSION_TASKS

LOEO_ROLES = ["FOLD_TRAIN", "FOLD_VALIDATION", "HELDOUT_ELEMENT_TEST",
              "EXCLUDED_FROM_FOLD"]

# STEP5A keep priority, inherited verbatim (spec 23)
KEEP_PRIORITY = (
    ["Temperature_C", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct", "S_CrNb_wt_pct",
     "R_CrNb_atomic_actual", "Cu_wt_pct", "has_X",
     "T_over_Tm_mix", "Delta_Smix", "delta_r", "delta_chi", "r_bar", "chi_bar",
     "VEC_mix", "Tm_mix", "Zbar", "M_bar", "CovalentR_bar", "density_bar",
     "Ecoh_bar", "IE1_bar", "EA_bar", "Mendeleev_bar", "AtomicVol_bar"]
    + [f"X_{p}" for p in ["AtomicNumber", "AtomicWeight", "MetallicRadius",
                          "Electronegativity", "VEC", "MeltingT", "Density",
                          "FirstIonizationEnergy", "CohesiveEnergy"]]
    + [f"Delta_{p}_X_{b}" for p in ["AtomicNumber", "AtomicWeight", "MetallicRadius",
                                    "Electronegativity", "VEC", "MeltingT", "Density",
                                    "FirstIonizationEnergy", "CohesiveEnergy"]
       for b in ("Cu", "Cr", "Nb")]
)
STAT_ORDER = {"mean": 0, "avg_dev": 1, "minimum": 2, "maximum": 3, "range": 4, "mode": 5}

# element-level descriptor vector for the distance analysis (spec 53):
# PURE ELEMENT external properties only - no CALPHAD, no phase, no partition, no target
ELEMENT_DISTANCE_PROPERTIES = [
    "AtomicNumber", "AtomicWeight", "CovalentRadius", "Electronegativity", "VEC",
    "MeltingT", "FirstIonizationEnergy", "ElectronAffinity", "AtomicVolume_GS",
    "MetallicRadius", "Density", "MendeleevNumber", "CohesiveEnergy",
]

FORBIDDEN_FEATURE_TOKENS = [
    "state_id", "composition_id", "family", "X_element", "sampling_domain",
    "split_role", "loeo_role", "state_converged", "solver_level_used", "bridge_used",
    "phase_count", "has_multiple_FCC_instances", "has_multiple_BCC_instances",
    "has_liquid", "_presence", "_volfrac", "_molefrac", "_AUX", "CR_P_", "NB_P_",
    "X_P_", "X_Psolid", "dominant_", "solid_volume_fraction",
]
NI_TOKEN = re.compile(r"(?<![A-Za-z])NI(?![A-Za-z])")

FORBIDDEN_PHRASES = [
    "never read", "never accessed", "never been read", "fully untouched",
    "prospective blind", "first open", "first_open", "never historically",
]


# ------------------------------------------------------------------ utils -- #
def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fold_seed(heldout):
    """Deterministic child seed per fold from the master seed (spec 62)."""
    idx = HELDOUT_ELEMENTS.index(heldout)
    ss = np.random.SeedSequence(MASTER_SEED)
    child = ss.spawn(len(HELDOUT_ELEMENTS))[idx]
    return int(child.generate_state(1, dtype=np.uint32)[0])


def ni_scan(texts_or_frames):
    """Element-token Ni scan. 'DOMINANT'/'NIOBIUM' cannot match (token-bounded)."""
    hits = []
    for label, obj in texts_or_frames:
        if isinstance(obj, pd.DataFrame):
            for c in obj.columns:
                if NI_TOKEN.search(str(c).upper()):
                    hits.append(f"{label}::column::{c}")
            for c in obj.columns:
                s = obj[c]
                # scan every NON-NUMERIC column. Testing for object/"string" dtype misses
                # pandas >= 3 str columns, which is exactly where an element symbol lives.
                if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                    vals = s.dropna().astype(str)
                    if not len(vals):
                        continue
                    for v in pd.unique(vals):
                        if NI_TOKEN.search(v.upper()):
                            hits.append(f"{label}::{c}::{v[:60]}")
                            if len(hits) > 20:
                                return hits
        else:
            for m in NI_TOKEN.finditer(str(obj).upper()):
                hits.append(f"{label}::text::offset{m.start()}")
                if len(hits) > 20:
                    return hits
    return hits


DISCLAIMER_OPEN = "<!--TERMINOLOGY_DISCLAIMER-->"
DISCLAIMER_CLOSE = "<!--/TERMINOLOGY_DISCLAIMER-->"


def forbidden_phrase_scan(text):
    """Spec 0/41 forbids ASSERTING 'never read' / 'prospective blind' / etc.

    A report must still be able to DISCLAIM them - saying "this is NOT a prospective
    blind validation" is exactly the honesty the rule exists to enforce. So the scan
    skips the single explicitly delimited terminology-disclaimer block and checks
    everything else. The block must exist; a report with no disclaimer at all is itself
    a violation.
    """
    text = str(text)
    if DISCLAIMER_OPEN not in text or DISCLAIMER_CLOSE not in text:
        return ["MISSING terminology disclaimer block"]
    head, rest = text.split(DISCLAIMER_OPEN, 1)
    _, tail = rest.split(DISCLAIMER_CLOSE, 1)
    body = (head + tail).lower()
    return [p for p in FORBIDDEN_PHRASES if p in body]


# ------------------------------------------------------------------- data -- #
def candidate_feature_columns():
    """The frozen 193-feature candidate schema (spec 16), read from STEP5A."""
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(CANDIDATE_PARQUET).schema.names
    return [c for c in cols if c not in META_COLS + HELPER_COLS]


def load_domain_a(feature_cols=None):
    """Domain-A candidate features joined to converged FULL_V1 targets.

    Only sampling_domain == 'A' rows are ever materialized here: Domain B and ANCHOR
    are not read (spec 6, 77).
    """
    if feature_cols is None:
        feature_cols = candidate_feature_columns()
    cols = list(dict.fromkeys(META_COLS + ["Temperature_C"] + list(feature_cols)))
    cand = pd.read_parquet(CANDIDATE_PARQUET, columns=cols)
    cand = cand[cand["sampling_domain"] == PRIMARY_DOMAIN].reset_index(drop=True)
    assert not cand.columns.duplicated().any(), "duplicate columns after read"
    tgt_cols = sorted({c for t in ALL_TASKS for c in TARGETS[t]})
    mlr = pd.read_parquet(ML_READY, columns=["state_id"] + tgt_cols)
    df = cand.merge(mlr, on="state_id", how="left", validate="1:1")
    assert df["FCC_volfrac"].notna().all(), "target join produced NaN"
    diag = pd.read_parquet(ML_READY, columns=["state_id", "has_multiple_FCC_instances",
                                              "has_multiple_BCC_instances"])
    df = df.merge(diag, on="state_id", how="left", validate="1:1")
    return df, list(feature_cols)


def loeo_role(split_role, family, heldout):
    """Fold role of one composition (spec 8, 9)."""
    if family == heldout:
        return "HELDOUT_ELEMENT_TEST"
    if split_role == "DEV_TRAIN":
        return "FOLD_TRAIN"
    if split_role == "DEV_VALIDATION":
        return "FOLD_VALIDATION"
    return "EXCLUDED_FROM_FOLD"      # other elements' IID_TEST: never used (spec 9)


def task_rows(df, task):
    """Rows usable for a task. X partition has no target for BASELINE - those rows are
    dropped, never filled with 0 (spec 77)."""
    sub = df[df["family"] != "BASELINE"] if task == "x_partition" else df
    cols = TARGETS[task]
    bad = sub[cols].isna().any(axis=1)
    if bad.any():
        sub = sub[~bad]
    return sub


def subset_compositions(df, fraction=SUBSET_FRACTION, seed=MASTER_SEED):
    """Family-stratified COMPOSITION-level subsample (never state-row random, spec 31)."""
    rng = np.random.default_rng(seed)
    comp = df[["composition_id", "family"]].drop_duplicates()
    keep = []
    for fam in sorted(comp["family"].unique()):
        ids = np.sort(comp.loc[comp["family"] == fam, "composition_id"].values)
        n = int(round(len(ids) * fraction))
        if n > 0:
            keep.append(rng.choice(ids, size=n, replace=False))
    keep = np.concatenate(keep) if keep else np.array([], dtype=object)
    return df[df["composition_id"].isin(keep)].copy()


# ---------------------------------------------------------------- metrics -- #
def tvd(pred, true):
    return 0.5 * np.abs(pred - true).sum(axis=1)


def project_simplex(Y, fallback_row):
    """clip negatives, renormalize; sum==0 -> fallback (FOLD_TRAIN mean simplex, spec 38).
    The fallback is NEVER computed from the held-out element."""
    P = np.clip(np.asarray(Y, dtype=np.float64), 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    n_fallback = int((s[:, 0] == 0).sum())
    out = np.where(s > 0, P / np.where(s > 0, s, 1.0), np.asarray(fallback_row))
    return out, n_fallback


def fit_thresholds_f1(y_true, y_prob):
    """Per-label F1-max threshold; tie -> closest to 0.5 (spec 33). Validation only."""
    from sklearn.metrics import f1_score
    grid = np.round(np.arange(0.05, 0.951, 0.01), 3)
    ths = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            ths.append(0.5)
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(yt, (y_prob[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12
                                        and abs(t - 0.5) < abs(best_t - 0.5)):
                best_f1, best_t = f1, float(t)
        ths.append(best_t)
    return ths


def classification_metrics(y_true, y_prob, thresholds):
    """spec 34 + spec 35/36: adds Macro_F1_non_degenerate and DEGENERATE_TEST_LABEL."""
    from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                                 average_precision_score, hamming_loss)
    y_pred = (y_prob >= np.asarray(thresholds)).astype(int)
    out = {
        "exact_match": float((y_pred == y_true).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    per_label, nd_f1, degenerate = {}, [], []
    for i, fam in enumerate(FAMILIES_ORDER):
        yt, yp, ypr = y_true[:, i], y_pred[:, i], y_prob[:, i]
        is_degenerate = len(np.unique(yt)) < 2
        d = {"support": int(yt.sum()), "n": int(len(yt)),
             "precision": float(precision_score(yt, yp, zero_division=0)),
             "recall": float(recall_score(yt, yp, zero_division=0)),
             "f1": float(f1_score(yt, yp, zero_division=0)),
             "degenerate_test_label": bool(is_degenerate)}
        if is_degenerate:
            d["roc_auc"] = None
            d["pr_auc"] = None
            d["note"] = ("DEGENERATE_TEST_LABEL: only one class present in the held-out "
                         "test; F1 here is not evidence of generalization (spec 36)")
            degenerate.append(fam)
        else:
            d["roc_auc"] = float(roc_auc_score(yt, ypr))
            d["pr_auc"] = float(average_precision_score(yt, ypr))
            nd_f1.append(d["f1"])
        per_label[fam] = d
    out["per_label"] = per_label
    out["degenerate_test_labels"] = degenerate
    out["n_non_degenerate_labels"] = len(nd_f1)
    out["macro_f1_non_degenerate"] = float(np.mean(nd_f1)) if nd_f1 else None
    return out


def regression_metrics(y_true, y_pred_raw, y_pred_proj, n_fallback=0):
    """spec 37."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    out = {"macro_mae": float(np.abs(y_pred_proj - y_true).mean()),
           "macro_rmse": float(np.sqrt(((y_pred_proj - y_true) ** 2).mean())),
           "simplex_fallbacks": int(n_fallback), "per_family": {}}
    t_proj, t_raw = tvd(y_pred_proj, y_true), tvd(y_pred_raw, y_true)
    out["mean_tvd"] = float(t_proj.mean())
    out["median_tvd"] = float(np.median(t_proj))
    out["p95_tvd"] = float(np.percentile(t_proj, 95))
    out["mean_tvd_raw"] = float(t_raw.mean())
    out["raw_sum_violation_mean"] = float(np.abs(y_pred_raw.sum(axis=1) - 1).mean())
    out["raw_sum_violation_max"] = float(np.abs(y_pred_raw.sum(axis=1) - 1).max())
    out["projected_sum_violation_max"] = float(np.abs(y_pred_proj.sum(axis=1) - 1).max())
    for i, fam in enumerate(FAMILIES_ORDER):
        yt = y_true[:, i]
        out["per_family"][fam] = {
            "mae": float(mean_absolute_error(yt, y_pred_proj[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(yt, y_pred_proj[:, i]))),
            "r2": float(r2_score(yt, y_pred_proj[:, i])) if yt.std() > 0 else None,
            "true_mean": float(yt.mean()), "true_std": float(yt.std()),
        }
    return out


def topology_metrics(y_true, y_pred_proj):
    """spec 47/48: dominant X-partition family, argmax of true vs projected prediction.
    The '#n' instance suffix plays no part here; families are the frozen core five."""
    from sklearn.metrics import f1_score, confusion_matrix
    t = np.asarray(y_true).argmax(axis=1)
    p = np.asarray(y_pred_proj).argmax(axis=1)
    labels = list(range(len(FAMILIES_ORDER)))
    cm = confusion_matrix(t, p, labels=labels)
    return {
        "accuracy": float((t == p).mean()),
        "macro_f1": float(f1_score(t, p, average="macro", labels=labels, zero_division=0)),
        "support": int(len(t)),
        "true_distribution": {FAMILIES_ORDER[i]: int((t == i).sum()) for i in labels},
        "predicted_distribution": {FAMILIES_ORDER[i]: int((p == i).sum()) for i in labels},
        "confusion_matrix": cm.tolist(),
        "confusion_labels": FAMILIES_ORDER,
    }


def trivial_regression_simplex(y_train):
    """FOLD_TRAIN mean vector, renormalized to the simplex (spec 39)."""
    m = np.asarray(y_train, dtype=np.float64).mean(axis=0)
    m = np.clip(m, 0.0, None)
    s = m.sum()
    return m / s if s > 0 else np.full(len(m), 1.0 / len(m))


def trivial_classification_probs(y_train):
    """FOLD_TRAIN prevalence (spec 39)."""
    return np.asarray(y_train, dtype=np.float64).mean(axis=0)


def class_weight_positive(y_col):
    p1 = max(float(np.asarray(y_col).mean()), 1e-12)
    return float((1.0 / p1) / 2.0) if p1 < 1.0 else 1.0
