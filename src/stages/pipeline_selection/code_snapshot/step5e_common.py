# -*- coding: utf-8 -*-
"""
step5e_common.py
STEP 5E - PREREGISTERED X-PARTITION REMEDY STUDY (retrospective).

TERMINOLOGY: retrospective remedy / retrospective few-shot recovery experiment.
ZR/TI/CO/SI/AG/MO targets historically exist and were evaluated in STEP5C.
Forbidden phrasing: "prospective", "blind", "never seen" (except inside the
explicit terminology disclaimer block). Ni stays sealed for the final
prospective blind test; ANCHOR stays sealed; Domain B is not used.

Inherited frozen facts (STEP5D, read-only): whole-vector eta2_element
(X partition) = 0.885455; 96.696% of summed X-partition TVD comes from states
with wrong dominant topology; per-fold Tree/MLP/Ridge/analog diagnostics.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = r"E:\Cu_CALPHAD_ML"
FULL_V1 = os.path.join(ROOT, "03_full_CALPHAD", "FULL_V1")
STEP5A = os.path.join(ROOT, "04_ML", "STEP5A_JMST_ML_FOUNDATION_V2")
STEP5B = os.path.join(ROOT, "05_OOD", "STEP5B_DOMAIN_B_OOD_V1")
STEP5C = os.path.join(ROOT, "05_OOD", "STEP5C_6X_LOEO_V1")
STEP5D = os.path.join(ROOT, "05_OOD", "STEP5D_FAILURE_DIAGNOSIS_V1")
STEP5E = os.path.join(ROOT, "05_OOD", "STEP5E_XPART_REMEDY_V1")

D_CONFIG = os.path.join(STEP5E, "00_config")
D_PREREG = os.path.join(STEP5E, "01_preregistration")
D_FROZEN = os.path.join(STEP5E, "02_frozen_baselines")
D_AFF = os.path.join(STEP5E, "03_affinity_descriptors")
D_ZS = os.path.join(STEP5E, "04_zero_shot")
D_TOPO = os.path.join(STEP5E, "05_topology_first")
D_FEWSHOT = os.path.join(STEP5E, "06_fewshot")
D_ABL = os.path.join(STEP5E, "07_ablation")
D_FIG = os.path.join(STEP5E, "08_figures")
D_QA = os.path.join(STEP5E, "09_qa")
D_LOGS = os.path.join(STEP5E, "10_logs")
D_MAN = os.path.join(STEP5E, "11_manifest")
D_CODE = os.path.join(STEP5E, "code_snapshot")
ALL_DIRS = [D_CONFIG, D_PREREG, D_FROZEN, D_AFF, D_ZS, D_TOPO, D_FEWSHOT,
            D_ABL, D_FIG, D_QA, D_LOGS, D_MAN, D_CODE]

# ---- frozen STEP5C references (read-only) ----
C_MANIFEST = os.path.join(STEP5C, "02_fold_manifests", "LOEO_MASTER_FOLD_MANIFEST.csv")
C_FEAT = os.path.join(STEP5A, "02_features", "FULL_V1_CANDIDATE_FEATURES.parquet")
C_ML_READY = os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet")
C_FREEZE = lambda x: os.path.join(STEP5C, "11_manifest", f"FOLD_{x}_FREEZE.json")
C_PRED = lambda x: os.path.join(STEP5C, "05_predictions", f"FOLD_{x}_HELDOUT_PREDICTIONS.parquet")
C_FEATLIST = lambda x: os.path.join(STEP5C, "03_features", f"FOLD_{x}_SELECTED_FEATURES.txt")
D_Z1PRED = lambda x: os.path.join(STEP5D, "07_model_bias_diagnostic", f"FOLD_{x}",
                                  "DIAGNOSTIC_HELDOUT_PREDICTIONS.parquet")
D_ETA2 = os.path.join(STEP5D, "03_variance_decomposition", "TASK_ELEMENT_VARIANCE_DECOMPOSITION.csv")
D_TOPOERR = os.path.join(STEP5D, "05_topology_analysis", "X_PARTITION_ERROR_BY_TOPOLOGY_CORRECTNESS.csv")
D_MODELCOMP = os.path.join(STEP5D, "07_model_bias_diagnostic", "X_DIAGNOSTIC_MODEL_COMPARISON.csv")

HELDOUT_ELEMENTS = ["ZR", "TI", "CO", "SI", "AG", "MO"]
FOLDS = [f"FOLD_{e}" for e in HELDOUT_ELEMENTS]
FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
X_TARGETS = [f"X_P_{f}" for f in FAMILIES_ORDER]
META_COLS = ["state_id", "composition_id", "family", "X_element", "sampling_domain", "split_role"]
HELPER_COLS = ["Temperature_K"]

MASTER_SEED = 20260908            # spec 46/50
FEWSHOT_BUDGETS = [1, 2, 4, 8, 16, 32]   # spec 48, nested
N_RANDOM_REPLICATES = 3           # spec 50
CALIBRATION_FRACTION = 0.30       # spec 46
MIN_EXPERT_COMPS = 100            # spec 36
MIN_EXPERT_STATES = 500           # spec 36
MLP_LRS = [1e-4, 3e-4, 1e-3]      # spec 29 frozen candidates
MLP_EPOCHS = 200
MLP_PATIENCE = 15

# topology classifier preregistered small grid (spec 33): validation-only selection
TOPO_CLF_GRID = [dict(n_estimators=n, max_depth=d, learning_rate=lr)
                 for n in (300, 600) for d in (6, 8) for lr in (0.05, 0.1)]

# A1-A9 formulas (spec 24) on top of ONE uniform source column
AFFINITY_COLUMN = "Miedema_deltaH_inter"   # frozen in preregistration (spec 18-21)

ZERO_SHOT_BRANCHES = ["Z0_TREE_DIRECT", "Z1_MLP_DIRECT", "Z2_AFFINITY_XGB",
                      "Z3_TOPOLOGY_FIRST", "Z4_AFFINITY_TOPOLOGY_FIRST", "Z5_AFFINITY_MLP"]

# preregistered zero-shot selection rule (spec 42) and success criteria (spec 79)
SELECTION_GUARDRAIL_MIN_FOLDS_BEAT_TRIVIAL = 3
SUCCESS_PRACTICAL_TVD_GAIN = 0.10
SUCCESS_STRONG_TVD_GAIN = 0.25
SUCCESS_MIN_FOLDS_BEAT_TRIVIAL = 3
SUCCESS_STRONG_MIN_FOLDS = 4

NI_TOKEN = re.compile(r"(?<![A-Za-z])NI(?![A-Za-z])")
FORBIDDEN_PHRASES = ["never read", "never accessed", "fully untouched",
                     "prospective blind", "never seen", "never historically"]
DISCLAIMER_OPEN = "<!--TERMINOLOGY_DISCLAIMER-->"
DISCLAIMER_CLOSE = "<!--/TERMINOLOGY_DISCLAIMER-->"


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jd)
    return path


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_json(path):
    return json.loads(open(path, encoding="utf-8").read())


def ni_scan(frames_texts):
    hits = []
    for label, obj in frames_texts:
        if isinstance(obj, pd.DataFrame):
            for c in obj.columns:
                if NI_TOKEN.search(str(c).upper()):
                    hits.append(f"{label}::column::{c}")
            for c in obj.columns:
                s = obj[c]
                if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                    for v in pd.unique(s.dropna().astype(str)):
                        if NI_TOKEN.search(str(v).upper()):
                            hits.append(f"{label}::{c}::{str(v)[:60]}")
        else:
            for m in NI_TOKEN.finditer(str(obj).upper()):
                hits.append(f"{label}::text::offset{m.start()}")
    return hits


def forbidden_phrase_scan(text):
    text = str(text)
    if DISCLAIMER_OPEN not in text or DISCLAIMER_CLOSE not in text:
        return ["MISSING terminology disclaimer block"]
    head, rest = text.split(DISCLAIMER_OPEN, 1)
    _, tail = rest.split(DISCLAIMER_CLOSE, 1)
    body = (head + tail).lower()
    return [p for p in FORBIDDEN_PHRASES if p in body]


# ------------------------------------------------------------------ data -- #
def load_fold_manifest():
    m = pd.read_csv(C_MANIFEST)
    assert set(m["loeo_role"].unique()) <= {"FOLD_TRAIN", "FOLD_VALIDATION",
                                            "HELDOUT_ELEMENT_TEST", "EXCLUDED_FROM_FOLD"}
    return m


def load_fold_features_and_targets(fold, selected_features):
    """Domain-A candidate features for the fold's roles + X-partition targets.

    Reads the frozen STEP5A candidate parquet (Domain A only) and the converged
    ML_READY targets; fold membership comes from the frozen STEP5C manifest.
    Held-out rows ARE materialized in this table (their targets historically
    exist) but are excluded from every fit/selection/tuning path; they are only
    scored in the designated held-out evaluation stage.
    """
    feats = list(dict.fromkeys(list(selected_features)))
    cols = list(dict.fromkeys(META_COLS + ["Temperature_C"] + feats))
    cand = pd.read_parquet(C_FEAT, columns=cols)
    cand = cand[cand["sampling_domain"] == "A"].reset_index(drop=True)
    assert not cand.columns.duplicated().any()
    tgt = pd.read_parquet(C_ML_READY, columns=["state_id"] + X_TARGETS)
    df = cand.merge(tgt, on="state_id", how="left", validate="1:1")
    man = load_fold_manifest()
    fm = man[man["fold"] == fold][["composition_id", "loeo_role"]]
    df = df.merge(fm, on="composition_id", how="left", validate="m:1")
    assert df["loeo_role"].notna().all(), f"{fold}: composition missing from manifest"
    # BASELINE rows have NaN X-partition targets BY DESIGN (no X element); they are
    # excluded from every x_partition fit/evaluation downstream (never zero-filled).
    xb = df[df["family"] != "BASELINE"]
    assert len(xb) and xb[X_TARGETS].notna().all().all(), "NaN target on an X-bearing row"
    base = df[df["family"] == "BASELINE"]
    if len(base):
        assert base[X_TARGETS].isna().all().all(), "BASELINE X targets must be NaN"
    return df, feats


AFF_COLS = ["A1_H_XCu", "A2_H_XCr", "A3_H_XNb", "A4_H_XCr_minus_H_XCu",
            "A5_H_XNb_minus_H_XCu", "A6_H_XNb_minus_H_XCr", "A7_min",
            "A8_max", "A9_range"]


def attach_affinity(df):
    """Merge the frozen A1-A9 table onto family (X identity). BASELINE rows get
    NaN affinity and are never used by any x_partition fit."""
    aff = pd.read_csv(os.path.join(D_AFF, "STEP5E_AFFINITY_DESCRIPTORS.csv"))
    df = df.merge(aff[["element_code"] + AFF_COLS], left_on="family",
                  right_on="element_code", how="left", validate="m:1")
    assert not df.columns.duplicated().any()
    xb = df[df["family"] != "BASELINE"]
    assert xb[AFF_COLS].notna().all().all(), "missing affinity value on an X-bearing row"
    return df


def fold_rows(df, role):
    sub = df[df["loeo_role"] == role]
    assert sub["family"].nunique() >= 1
    return sub


def load_frozen_z0(fold):
    """Frozen STEP5C Tree X-partition projected predictions (Z0). No refit."""
    p = pd.read_parquet(C_PRED(fold.replace("FOLD_", "")))
    cols = [f"pred_proj_x_partition_{f}" for f in FAMILIES_ORDER]
    z0 = p[["state_id", "composition_id"] + cols].rename(
        columns={f"pred_proj_x_partition_{f}": f"Z0_{f}" for f in FAMILIES_ORDER})
    return z0


def load_frozen_z1(fold):
    """Frozen STEP5D MLP diagnostic predictions (Z1). No refit."""
    p = pd.read_parquet(D_Z1PRED(fold.replace("FOLD_", "")))
    cols = [f"MLP_{f}" for f in FAMILIES_ORDER]
    z1 = p[["state_id", "composition_id"] + cols].rename(
        columns={f"MLP_{f}": f"Z1_{f}" for f in FAMILIES_ORDER})
    return z1


def fold_xgb_params(fold):
    """STEP5C fold X-partition XGBoost frozen hyperparameters (used by Z2/Z3/Z4
    experts and F0/F1; if the STEP5C winner was CatBoost the preregistered rule
    is to use the XGBoost branch's tuned params recorded in the freeze)."""
    fr = load_json(C_FREEZE(fold))
    hp = fr["tasks"]["x_partition"]["hyperparameters"]
    if fr["tasks"]["x_partition"]["model_family"] != "XGBoost":
        hp = fr["tasks"]["x_partition"]["xgboost_branch_hyperparameters"]
    assert hp, f"{fold}: no XGBoost x-partition hyperparameters in freeze"
    return dict(hp)


# ---------------------------------------------------------------- metrics -- #
def tvd(pred, true):
    return 0.5 * np.abs(np.asarray(pred) - np.asarray(true)).sum(axis=1)


def project_simplex(Y, fallback_row):
    P = np.clip(np.asarray(Y, dtype=np.float64), 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    n_fb = int((s[:, 0] == 0).sum())
    out = np.where(s > 0, P / np.where(s > 0, s, 1.0), np.asarray(fallback_row))
    return out, n_fb


def dominant_topology(Y):
    return np.asarray(Y).argmax(axis=1)


def topology_accuracy(y_true_proj, y_pred_proj):
    return float((dominant_topology(y_true_proj) == dominant_topology(y_pred_proj)).mean())


def eval_summary(y_true, pred_raw, pred_proj, trivial_row):
    t = tvd(pred_proj, y_true)
    return {
        "mean_tvd": float(t.mean()), "median_tvd": float(np.median(t)),
        "p95_tvd": float(np.percentile(t, 95)),
        "macro_mae": float(np.abs(pred_proj - y_true).mean()),
        "topology_accuracy": topology_accuracy(y_true, pred_proj),
        "topology_macro_f1": _topo_macro_f1(y_true, pred_proj),
        "trivial_tvd": float(tvd(np.tile(trivial_row, (len(y_true), 1)), y_true).mean()),
        "n_rows": int(len(y_true)),
    }


def _topo_macro_f1(y_true_proj, y_pred_proj):
    from sklearn.metrics import f1_score
    t, p = dominant_topology(y_true_proj), dominant_topology(y_pred_proj)
    labels = list(range(len(FAMILIES_ORDER)))
    return float(f1_score(t, p, average="macro", labels=labels, zero_division=0))


def trivial_simplex(y_train):
    m = np.clip(np.asarray(y_train, dtype=np.float64).mean(axis=0), 0.0, None)
    s = m.sum()
    return m / s if s > 0 else np.full(len(m), 1.0 / len(m))


# ------------------------------------------------------- few-shot splits -- #
def split_hash(element, composition_id):
    """Deterministic ordering hash for the retrospective recovery split (spec 46)."""
    payload = f"{element}|{composition_id}|{MASTER_SEED}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def frozen_recovery_split(element, comp_ids):
    """30% lowest hash -> CALIBRATION_POOL; rest -> RETROSPECTIVE_RECOVERY_EVAL."""
    h = pd.DataFrame({"composition_id": list(comp_ids)})
    h["order_hash"] = [split_hash(element, c) for c in h["composition_id"]]
    h = h.sort_values("order_hash", kind="mergesort").reset_index(drop=True)
    n_cal = int(round(len(h) * CALIBRATION_FRACTION))
    return (h["composition_id"][:n_cal].tolist(),
            h["composition_id"][n_cal:].tolist())


def maximin_sequence(element, pool_ids, train_ids, feature_matrix_pool,
                     feature_matrix_train):
    """Target-free greedy maximin over the calibration pool (spec 51-53).

    Distance space: composition-level vectors (Temperature_C and every
    fold-constant column removed), StandardScaler fit on FOLD_TRAIN only.
    First pick = largest 1-NN distance to FOLD_TRAIN; then iteratively the
    candidate maximizing the minimum distance to FOLD_TRAIN + already selected.
    """
    from scipy.spatial import cKDTree
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(feature_matrix_train)
    zt = scaler.transform(feature_matrix_train)
    zp = scaler.transform(feature_matrix_pool)
    tree_train = cKDTree(zt)
    d0, _ = tree_train.query(zp, k=1, workers=-1)
    chosen = [int(np.argmax(d0))]
    min_d = d0.copy()
    while len(chosen) < FEWSHOT_BUDGETS[-1]:
        active = np.ones(len(zp), dtype=bool)
        active[chosen] = False
        cur = np.array(chosen)
        tree = cKDTree(zp[cur])
        d, _ = tree.query(zp[active], k=1, workers=-1)
        upd = np.full(len(zp), np.inf)
        upd[np.flatnonzero(active)] = d
        min_d = np.minimum(min_d, upd)
        min_d[chosen] = -np.inf
        chosen.append(int(np.argmax(min_d)))
    return [pool_ids[i] for i in chosen]


def random_sequence(replicate, pool_ids):
    """Nested random permutation prefix (spec 50): deterministic child seed."""
    ss = np.random.SeedSequence([MASTER_SEED, replicate])
    rng = np.random.default_rng(ss)
    idx = np.array(pool_ids)[rng.permutation(len(pool_ids))]
    return idx.tolist()
