# -*- coding: utf-8 -*-
"""
f5f_common.py
STEP5F  -  FULL NON-Ni CHEMICAL-IDENTITY AUGMENTATION + AUGMENTED STRICT LOEO TEST.

Purpose (FINAL TASK): generate a controlled five-element non-Ni CALPHAD
chemical-identity augmentation and test whether increasing the training
chemistry from five to ten visible X identities improves strict zero-shot
X-partition transfer on the exact frozen original STEP5C held-out elements,
while keeping model formulation, feature selection, and Ni completely fixed.

Hard rules encoded here:
  - the five new identities are FROZEN from the STEP5F-PILOT selection:
    BE SN MN LA P (spec 4). No sixth element, no removal, no post-hoc swap.
  - Ni is absolutely sealed (spec 1): no descriptor, no property, no binary
    affinity, no composition, no CALPHAD, no prediction, no target, no
    similarity. Nothing here reads a Ni row of anything.
  - ANCHOR stays sealed (spec 2); Domain B is not used for sampling/training/
    tuning/selection (spec 3).
  - the primary model formulation is the frozen STEP5D/STEP5E Z1_MLP_DIRECT
    protocol (spec 25, 27, 30): 128/64/5 MLP, raw-component MSE, Adam,
    batch 1024, <= 200 epochs, patience 15 on raw validation MSE, per-fold
    seed, LR grid selected on projected validation TVD, StandardScaler on
    FOLD_TRAIN, simplex projection with the frozen fold-train trivial row as
    fallback. NO new Optuna, NO new architecture/LR search.
  - feature lists are the frozen per-fold STEP5C selections (spec 28, 29).
    No feature reselection because of the augmentation.
  - fold membership / heldout tests are the frozen STEP5C manifest (spec 31,
    33, 34). Validation = original FOLD_VALIDATION only; new-element targets
    never enter validation or model selection.
  - partition definition is the frozen MOLAR one (spec 19); volume-fraction
    partitioning is FORBIDDEN.
  - a CALPHAD failure is retained, never interpolated/fabricated/dropped
    (spec 21).
  - thresholds 0.20 / 0.35 / 4-of-6 / 0.50 / 5-of-6 are preregistered here
    and must not be changed after results are seen (spec 39-43).
"""
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths -- #
ROOT = r"E:\Cu_CALPHAD_ML\06_AUGMENTATION\STEP5F_FULL_V1"
PILOT_ROOT = r"E:\Cu_CALPHAD_ML\06_AUGMENTATION\STEP5F_PILOT_V1"
MLROOT = r"E:\Cu_CALPHAD_ML"
FULL_V1 = os.path.join(MLROOT, "03_full_CALPHAD", "FULL_V1")
FULL_V1_CODE = os.path.join(FULL_V1, "code_snapshot")
STEP5A = os.path.join(MLROOT, "04_ML", "STEP5A_JMST_ML_FOUNDATION_V2")
STEP5C = os.path.join(MLROOT, "05_OOD", "STEP5C_6X_LOEO_V1")
STEP5D = os.path.join(MLROOT, "05_OOD", "STEP5D_FAILURE_DIAGNOSIS_V1")
STEP5E = os.path.join(MLROOT, "05_OOD", "STEP5E_XPART_REMEDY_V1")

D_CONFIG = os.path.join(ROOT, "00_config")
D_COMP = os.path.join(ROOT, "01_compositions")
D_RAW = os.path.join(ROOT, "02_raw_CALPHAD")
D_PROC = os.path.join(ROOT, "03_processed")
D_CALQA = os.path.join(ROOT, "04_calphad_qa")
D_ML = os.path.join(ROOT, "05_augmented_ML")
D_ABL = os.path.join(ROOT, "06_ablation")
D_FIG = os.path.join(ROOT, "07_figures")
D_QA = os.path.join(ROOT, "08_qa")
D_MAN = os.path.join(ROOT, "09_manifest")
D_LOGS = os.path.join(ROOT, "10_logs")
ALL_DIRS = [D_CONFIG, D_COMP, D_RAW, D_PROC, D_CALQA, D_ML, D_ABL, D_FIG, D_QA,
            D_MAN, D_LOGS, os.path.join(ROOT, "code_snapshot")]

# frozen STEP5C / STEP5D read-only references
C_MANIFEST = os.path.join(STEP5C, "02_fold_manifests", "LOEO_MASTER_FOLD_MANIFEST.csv")
C_FREEZE = lambda x: os.path.join(STEP5C, "11_manifest", "FOLD_%s_FREEZE.json" % x)
C_FEATLIST = lambda x: os.path.join(STEP5C, "03_features", "FOLD_%s_SELECTED_FEATURES.txt" % x)
C_HELDOUT_PRED = lambda x: os.path.join(STEP5C, "05_predictions",
                                        "FOLD_%s_HELDOUT_PREDICTIONS.parquet" % x)
D_Z1PRED = lambda x: os.path.join(STEP5D, "07_model_bias_diagnostic", "FOLD_%s" % x,
                                  "DIAGNOSTIC_HELDOUT_PREDICTIONS.parquet")
D_DIAG_DATA = os.path.join(STEP5D, "02_frozen_inputs", "DOMAIN_A_DEV6_DIAGNOSTIC_DATA.parquet")
D_MODEL_COMP = os.path.join(STEP5D, "07_model_bias_diagnostic", "X_DIAGNOSTIC_MODEL_COMPARISON.csv")
STEP5A_FEAT = os.path.join(STEP5A, "02_features", "FULL_V1_CANDIDATE_FEATURES.parquet")
STEP5A_DESC = os.path.join(STEP5A, "01_feature_sources", "ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.parquet")
PILOT_COMPS = os.path.join(PILOT_ROOT, "01_compositions", "STEP5F_PILOT_COMPOSITIONS.csv")
PILOT_STATE_RESULTS = os.path.join(PILOT_ROOT, "03_processed",
                                   "STEP5F_PILOT_STATE_RESULTS.parquet")

# --------------------------------------------------------------- protocol -- #
SELECTED_ELEMENTS = ["BE", "SN", "MN", "LA", "P"]          # spec 4, frozen
FORBIDDEN_ELEMENTS = ["NI", "AU", "CE"]
HELDOUT_ELEMENTS = ["ZR", "TI", "CO", "SI", "AG", "MO"]    # frozen STEP5C six
FOLDS = ["FOLD_%s" % e for e in HELDOUT_ELEMENTS]
FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
X_TARGETS = ["X_P_%s" % f for f in FAMILIES_ORDER]
TRUE_COLS = ["true_x_partition_%s" % f for f in FAMILIES_ORDER]

N_PILOT_PER_ELEMENT = 50                       # spec 6 (reused, never recomputed)
N_ADD_PER_ELEMENT = 1950                       # spec 6
N_TOTAL_PER_ELEMENT = 2000                     # spec 5
N_NEW_COMPOSITIONS = N_ADD_PER_ELEMENT * len(SELECTED_ELEMENTS)       # 9750
N_TOTAL_COMPOSITIONS = N_TOTAL_PER_ELEMENT * len(SELECTED_ELEMENTS)   # 10000
TEMPERATURES_C = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]  # spec 7
N_FORMAL_STATES = N_TOTAL_COMPOSITIONS * len(TEMPERATURES_C)          # 110000
N_NEW_STATES = N_NEW_COMPOSITIONS * len(TEMPERATURES_C)               # 107250

MASTER_SEED = 20260910                         # spec 12
S_CRNB_RANGE = (0.6, 2.0)                      # spec 10, Domain-A-like
R_CRNB_RANGE = (2.0, 6.5)
X_WT_RANGE = (0.02, 0.20)                      # identical for all five (spec 10, 11)
M_CR, M_NB = 51.9961, 92.9064
WT_DECIMALS = 4

# preregistered thresholds (spec 22, 39-43) - DO NOT CHANGE AFTER RESULTS
GATE_FAIL_BELOW = 0.95                         # spec 22: <95% -> FAIL
GATE_WARN_BELOW = 0.98                         # spec 22: 95-98% -> WARN
Z1_BASELINE_MEAN_TVD = 0.752644                # spec 39 (verified from STEP5D table)
PRIMARY_SUCCESS_TVD = 0.602115                 # spec 39: Z1 * 0.8
TOPOLOGY_ACC_CRITERION = 0.35                  # spec 40
FOLDS_BEAT_TRIVIAL_MIN = 4                     # spec 41
STRONG_TVD = 0.50                              # spec 42
STRONG_FOLDS_BEAT = 5                          # spec 42

NI_TOKEN = re.compile(r"(?<![A-Za-z])NI(?![A-Za-z])")
X_SYMBOL = {"ZR": "Zr", "TI": "Ti", "CO": "Co", "SI": "Si", "AG": "Ag", "MO": "Mo",
            "BE": "Be", "SN": "Sn", "MN": "Mn", "LA": "La", "P": "P"}


# ------------------------------------------------------------------ utils -- #
def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if (isinstance(o, float) and math.isnan(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def ni_scan(items):
    """Element-token Ni scan over frames and text ('DOMINANT'/'NIOBIUM' cannot match)."""
    hits = []
    for label, obj in items:
        if isinstance(obj, pd.DataFrame):
            for c in obj.columns:
                if NI_TOKEN.search(str(c).upper()):
                    hits.append("%s::column::%s" % (label, c))
            for c in obj.columns:
                s = obj[c]
                if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                    continue
                vals = s.dropna().astype(str)
                if not len(vals):
                    continue
                for v in pd.unique(vals):
                    if NI_TOKEN.search(v.upper()):
                        hits.append("%s::%s::%s" % (label, c, v[:60]))
                        if len(hits) > 20:
                            return hits
        else:
            if NI_TOKEN.search(str(obj).upper()):
                hits.append("%s::text" % label)
    return hits


def invert_s_r(s_crnb, r_atomic):
    """FULL_V1 inversion, unchanged: wNb = S / (1 + R*(M_Cr/M_Nb)); wCr = S - wNb."""
    w_nb = s_crnb / (1.0 + r_atomic * (M_CR / M_NB))
    w_cr = s_crnb - w_nb
    return w_cr, w_nb


def actual_ratio(w_cr, w_nb):
    return (w_cr / M_CR) / (w_nb / M_NB)


def element_child_seed(element, elements=None):
    """Independent deterministic child stream per element from the master seed."""
    elements = elements or SELECTED_ELEMENTS
    idx = elements.index(element)
    ss = np.random.SeedSequence(MASTER_SEED)
    return ss.spawn(len(elements))[idx]


# ---------------------------------------------------------------- metrics -- #
def tvd(pred, true):
    return 0.5 * np.abs(np.asarray(pred) - np.asarray(true)).sum(axis=1)


def project_simplex(Y, fallback_row):
    """Frozen STEP5D projection: clip negatives, renormalize, zero-sum -> fallback."""
    P = np.clip(np.asarray(Y, dtype=np.float64), 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    out = np.where(s > 0, P / np.where(s > 0, s, 1.0), np.asarray(fallback_row))
    return out


def dominant_topology(Y):
    return np.asarray(Y).argmax(axis=1)


def topology_metrics(y_true, y_pred):
    from sklearn.metrics import f1_score
    t, p = dominant_topology(y_true), dominant_topology(y_pred)
    return (float((t == p).mean()),
            float(f1_score(t, p, average="macro", labels=list(range(5)), zero_division=0)))


def metrics_block(y_true, y_pred, trivial_row):
    """Frozen STEP5D metric block (identical definition to d_common.metrics)."""
    v = tvd(y_pred, y_true)
    b = float(tvd(np.tile(np.asarray(trivial_row, dtype=float), (len(y_true), 1)), y_true).mean())
    acc, f1 = topology_metrics(y_true, y_pred)
    return {"n_rows": int(len(y_true)),
            "mean_tvd": float(v.mean()), "median_tvd": float(np.median(v)),
            "p95_tvd": float(np.percentile(v, 95)),
            "topology_accuracy": acc, "topology_macro_f1": f1,
            "trivial_tvd": b,
            "skill_vs_trivial": float(b - v.mean()),
            "relative_skill": float(1.0 - v.mean() / b) if b > 0 else float("nan")}


def trivial_simplex(y_train):
    m = np.clip(np.asarray(y_train, dtype=np.float64).mean(axis=0), 0.0, None)
    s = m.sum()
    return m / s if s > 0 else np.full(len(m), 1.0 / len(m))


def beats_trivial(m):
    """A fold 'beats the frozen trivial baseline' iff mean TVD < trivial TVD."""
    return m["mean_tvd"] < m["trivial_tvd"]
