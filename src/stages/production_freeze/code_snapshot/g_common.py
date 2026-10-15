# -*- coding: utf-8 -*-
"""
g_common.py
STEP5G  -  FINAL PRODUCTION PIPELINE FREEZE
         +  PROSPECTIVE Ni BLIND PREREGISTRATION
         +  POST-BLIND Ni FEW-SHOT PREREGISTRATION

One-sentence scope (spec, final): freeze a preregistered original-six direct-MLP PRIMARY
and FULL11 direct-MLP SECONDARY pipeline, define a target-independent 2,000-composition
prospective Ni zero-shot blind test, and preregister a post-blind random-first Ni few-shot
recovery study, WITHOUT accessing any Ni elemental descriptor, prediction or CALPHAD
target during STEP5G.

Ni policy in this stage (spec 6, 41):
    Ni appears ONLY as a symbolic X identity in a composition table.
    Ni elemental descriptor access   = 0
    Ni elemental property access     = 0
    Ni Miedema / similarity          = 0
    Ni prediction                    = 0
    Ni CALPHAD / TC-Python           = 0
    Ni phase / partition / target    = 0
The Ni composition table is built from Cr and Nb atomic masses only (spec 38); the Ni
weight percent is a composition variable, not a property of nickel.

Why PRIMARY stays ORIGINAL6 (spec 11): every preregistered FULL5 switching gate failed
(TVD 0.710332 vs required <= 0.602115; topology 0.3323 vs required >= 0.35; 3/6 folds
beat trivial vs required >= 4/6). Keeping ORIGINAL6 as PRIMARY is protocol-following,
not a preference.
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
MLROOT = r"E:\Cu_CALPHAD_ML"
FULL_V1 = os.path.join(MLROOT, "03_full_CALPHAD", "FULL_V1")
STEP5A = os.path.join(MLROOT, "04_ML", "STEP5A_JMST_ML_FOUNDATION_V2")
STEP5C = os.path.join(MLROOT, "05_OOD", "STEP5C_6X_LOEO_V1")
STEP5D = os.path.join(MLROOT, "05_OOD", "STEP5D_FAILURE_DIAGNOSIS_V1")
STEP5E = os.path.join(MLROOT, "05_OOD", "STEP5E_XPART_REMEDY_V1")
STEP5F = os.path.join(MLROOT, "06_AUGMENTATION", "STEP5F_FULL_V1")
ROOT = os.path.join(MLROOT, "07_FINAL_BLIND", "STEP5G_PIPELINE_FREEZE_V1")

D_CONFIG = os.path.join(ROOT, "00_config")
D_FREEZE = os.path.join(ROOT, "01_pipeline_freeze")
D_PRIMARY = os.path.join(ROOT, "02_primary_model")
D_SECOND = os.path.join(ROOT, "03_secondary_model")
D_NIPRE = os.path.join(ROOT, "04_ni_composition_prereg")
D_METRIC = os.path.join(ROOT, "05_metric_prereg")
D_FEWSHOT = os.path.join(ROOT, "06_postblind_fewshot_prereg")
D_QA = os.path.join(ROOT, "07_qa")
D_MANIFEST = os.path.join(ROOT, "08_manifest")
D_LOGS = os.path.join(ROOT, "09_logs")
D_CODE = os.path.join(ROOT, "code_snapshot")
ALL_DIRS = [D_CONFIG, D_FREEZE, D_PRIMARY, D_SECOND, D_NIPRE, D_METRIC, D_FEWSHOT,
            D_QA, D_MANIFEST, D_LOGS, D_CODE]

# read-only inputs
A_FEATLIST = os.path.join(STEP5A, "02_features", "SELECTED_FEATURES_F1_F2_F3_F4.txt")
A_SPLIT = os.path.join(STEP5A, "03_splits", "COMPOSITION_SPLIT_MASTER_V1.csv")
A_DEVFREEZE = os.path.join(STEP5A, "09_manifest", "DEVELOPMENT_FREEZE_V1.json")
D_DIAG_DATA = os.path.join(STEP5D, "02_frozen_inputs",
                           "DOMAIN_A_DEV6_DIAGNOSTIC_DATA.parquet")
F_NEWFEAT = os.path.join(STEP5F, "03_processed",
                         "STEP5F_NEW_ELEMENT_CANDIDATE_FEATURES.parquet")
F_NEWDATA = os.path.join(STEP5F, "03_processed", "STEP5F_FULL_NEW_ELEMENT_DATA.parquet")
F_REPORT = os.path.join(STEP5F, "STEP5F_FULL_REPORT.md")
A_PRIMARY_LIST = os.path.join(STEP5A, "02_features", "PRIMARY_FEATURE_LIST.txt")
C_FEATDIR = os.path.join(STEP5C, "03_features")

def c_featlist(e):
    return os.path.join(C_FEATDIR, "FOLD_%s_SELECTED_FEATURES.txt" % e)

# Verified 2026-09-06 against DEVELOPMENT_FREEZE_V1.json: the 43-feature list file
# hashes to this value, so the production schema is provable, not merely asserted.
A_FEATLIST_SHA256 = "f4ddf4a2587e33a7d441656110c589208d9d860095198dd4b276df64f8a2fb0c"

# STEP5A frozen x_partition row counts (DEVELOPMENT_FREEZE_V1.tasks.x_partition).
# These are the X-bearing Domain-A rows after BASELINE exclusion; the preflight
# reproduces them from the data instead of trusting them.
A_XPART_TRAIN_ROWS = 241748
A_XPART_TRAIN_COMPOSITIONS = 22050
A_XPART_VAL_ROWS = 51802
# frozen STEP5A x_partition simplex fallback (DEV_TRAIN mean simplex) - carried as a
# cross-check only; the production fallback is recomputed from production train rows.
A_XPART_FALLBACK_ROW = [0.3924063060849874, 0.17997911835521582, 0.2687588732102482,
                        0.1582142342482923, 0.0006414681012562028]
STEP5F_NEW_ROWS = 110000

# STEP5F report patch (spec 7): the only wrong number in an otherwise correct report.
PATCH_FIND = "the 4,950 pilot states were reused, not recomputed"
PATCH_REPLACE = "the 2,750 pilot states were reused, not recomputed"

# --------------------------------------------------------------- protocol -- #
FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
X_TARGETS = ["X_P_%s" % f for f in FAMILIES_ORDER]
TRUE_COLS = ["true_x_partition_%s" % f for f in FAMILIES_ORDER]

ORIGINAL_SIX = ["ZR", "TI", "CO", "SI", "AG", "MO"]
FULL5_ADDED = ["BE", "SN", "MN", "LA", "P"]
FULL11 = ORIGINAL_SIX + FULL5_ADDED

PRIMARY_ROLE = "ORIGINAL6_DIRECT_MLP"          # spec 8
SECONDARY_ROLE = "FULL11_DIRECT_MLP"           # spec 9
FORBIDDEN_PRODUCTION_MODELS = ["BASE_PLUS_MN", "FULL5_MINUS_BE", "FULL5_MINUS_LA",
                               "FULL5_MINUS_SN", "FULL5_MINUS_P"]   # spec 10

N_FEATURES = 43                                 # spec 12
MLP_LRS = [1e-4, 3e-4, 1e-3]                    # spec 24
MLP_EPOCHS = 200                                # spec 23
MLP_PATIENCE = 15
MLP_BATCH = 1024
PRODUCTION_MASTER_SEED = 20260911               # spec 25

# Ni blind design
NI_MASTER_SEED = 20260911                       # spec 35, 44
NI_N_COMPOSITIONS = 2000                        # spec 34
TEMPERATURES_C = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]   # spec 39
NI_FORMAL_STATES = NI_N_COMPOSITIONS * len(TEMPERATURES_C)                  # 22000
S_CRNB_RANGE = (0.6, 2.0)                       # spec 32
R_CRNB_RANGE = (2.0, 6.5)
NI_WT_RANGE = (0.02, 0.20)
M_CR, M_NB = 51.9961, 92.9064                   # spec 38: Cr and Nb only
WT_DECIMALS = 4
BOOTSTRAP_REPS = 1000                           # spec 44

# post-blind few-shot
FEWSHOT_SPLIT_SALT = "20260912"                 # spec 61
FEWSHOT_MASTER_SEED = 20260912                  # spec 65
CALIBRATION_FRACTION = 0.30                     # spec 61
N_CALIBRATION = int(NI_N_COMPOSITIONS * CALIBRATION_FRACTION)     # 600
N_RECOVERY_EVAL = NI_N_COMPOSITIONS - N_CALIBRATION               # 1400
FEWSHOT_BUDGETS = [1, 2, 4, 8, 16, 32]          # spec 63
FEWSHOT_REPLICATES = 3                          # spec 65

# frozen historical reference numbers (spec 1) - carried, never recomputed here
Z1_HISTORICAL_LOEO_MEAN_TVD = 0.752644
FULL5_MEAN_TVD = 0.710332
FULL5_RELATIVE_REDUCTION = 0.0562
FULL5_MEAN_TOPOLOGY_ACC = 0.3323
FULL5_FOLDS_BEATING_TRIVIAL = 3
FULL5_GATES = {
    "primary_tvd": {"required_max": 0.602115, "observed": 0.710332, "result": "FAIL"},
    "topology": {"required_min": 0.35, "observed": 0.3323, "result": "FAIL"},
    "cross_element_skill": {"required_min_folds": 4, "observed_folds": 3,
                            "of": 6, "result": "FAIL"},
    "strong": {"required": "mean TVD <= 0.50 AND >= 5/6 folds beat trivial",
               "result": "FAIL"},
}

NI_TOKEN = re.compile(r"(?<![A-Za-z])NI(?![A-Za-z])")
# the ONLY places a Ni token may legitimately appear in STEP5G outputs
NI_ALLOWED_CONTEXT = ("symbolic composition identity", "seal statement", "preregistration")


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


def sha256_text(t):
    return hashlib.sha256(str(t).encode("utf-8")).hexdigest()


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jd)
    return path


def feature_schema():
    """The frozen STEP5A 43-feature schema, read from STEP5A - never re-selected."""
    with open(A_FEATLIST, encoding="utf-8") as f:
        feats = [ln.strip() for ln in f if ln.strip()]
    if len(feats) != N_FEATURES:
        raise SystemExit("FATAL: STEP5A feature list has %d entries, expected %d. The "
                         "production schema must be the frozen 43-feature list."
                         % (len(feats), N_FEATURES))
    return feats


def ni_scan_frame(df, label):
    """Element-token Ni scan over a frame. Scans every non-numeric column, because
    pandas >= 3 stores strings as 'str' dtype where an object-only check misses them."""
    hits = []
    for c in df.columns:
        if NI_TOKEN.search(str(c).upper()):
            hits.append("%s::column::%s" % (label, c))
    for c in df.columns:
        s = df[c]
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
    return hits


# ------------------------------------------------------------ composition -- #
def invert_s_r(s_crnb, r_atomic):
    """FULL_V1 inversion, unchanged. Uses Cr and Nb atomic masses ONLY (spec 38);
    no property of the X element enters the composition construction."""
    w_nb = s_crnb / (1.0 + r_atomic * (M_CR / M_NB))
    return s_crnb - w_nb, w_nb


def actual_ratio(w_cr, w_nb):
    return (w_cr / M_CR) / (w_nb / M_NB)


def child_seed(master, label):
    """Deterministic, recorded derivation (spec 25)."""
    h = hashlib.sha256(("%d|%s" % (master, label)).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


# ---------------------------------------------------------------- metrics -- #
def tvd(pred, true):
    return 0.5 * np.abs(np.asarray(pred) - np.asarray(true)).sum(axis=1)


def project_simplex_counted(Y, fallback_row):
    """Frozen projection (spec 27): clip negatives, renormalize, zero-sum -> the
    corresponding production training-set mean X simplex. Returns (projected,
    n_fallback_rows) so the fallback count can be reported, never hidden."""
    P = np.clip(np.asarray(Y, dtype=np.float64), 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    n_fb = int((s[:, 0] == 0).sum())
    out = np.where(s > 0, P / np.where(s > 0, s, 1.0), np.asarray(fallback_row, float))
    return out, n_fb


def project_simplex(Y, fallback_row):
    return project_simplex_counted(Y, fallback_row)[0]


def train_mean_simplex(Y):
    m = np.asarray(Y, dtype=np.float64).mean(axis=0)
    m = np.clip(m, 0.0, None)
    s = m.sum()
    return m / s if s > 0 else np.full(len(m), 1.0 / len(m))


def topology_metrics(y_true, y_pred):
    from sklearn.metrics import f1_score
    t = np.asarray(y_true).argmax(axis=1)
    p = np.asarray(y_pred).argmax(axis=1)
    return (float((t == p).mean()),
            float(f1_score(t, p, average="macro", labels=list(range(5)), zero_division=0)))


# ------------------------------------------------- post-blind recovery ----- #
# Frozen, fully specified, GRADIENT-FREE recovery estimators (spec 30, 60).
# None of them updates a single production weight: each is a corrector applied on
# top of the frozen PRIMARY prediction, so model immutability after Ni descriptor
# access is preserved by construction. They are preregistered here, before any Ni
# state exists, so that nothing can be chosen after seeing the blind result.
RECOVERY_ESTIMATORS = {
    "R0_ANCHOR_TRIVIAL_PER_T": (
        "Ignore the model. For each temperature T, predict the mean true X-partition "
        "simplex of the k anchor compositions at that same T. This is the control: if "
        "R2 does not beat R0, no improvement beyond the anchors is demonstrated by R2 under this protocol."),
    "R1_GLOBAL_MEAN_SHIFT": (
        "delta = mean over all k*11 anchor states of (true - raw model output); "
        "corrected = project_simplex(raw + delta). One 5-vector for the whole set."),
    "R2_PER_TEMPERATURE_MEAN_SHIFT": (
        "delta_T = mean over the k anchor states at temperature T of (true - raw model "
        "output); corrected = project_simplex(raw + delta_T). Eleven 5-vectors; exactly "
        "k anchor states feed each, so it is defined for every budget including k=1."),
}
RECOVERY_PRIMARY_ESTIMATOR = "PRIMARY_FORMULATION_REFIT"


def recovery_apply(name, raw, temps, anchor_raw, anchor_true, anchor_temps, fallback):
    """Apply a frozen recovery estimator. raw/anchor_raw are UNPROJECTED model
    outputs; anchor_true are true X-partition simplices."""
    raw = np.asarray(raw, dtype=np.float64)
    temps = np.asarray(temps)
    anchor_raw = np.asarray(anchor_raw, dtype=np.float64)
    anchor_true = np.asarray(anchor_true, dtype=np.float64)
    anchor_temps = np.asarray(anchor_temps)
    if name == "R0_ANCHOR_TRIVIAL_PER_T":
        out = np.zeros_like(raw)
        for t in np.unique(temps):
            m = anchor_temps == t
            if not m.any():
                out[temps == t] = fallback
                continue
            out[temps == t] = anchor_true[m].mean(axis=0)
        return project_simplex(out, fallback)
    if name == "R1_GLOBAL_MEAN_SHIFT":
        if not len(anchor_true):
            return project_simplex(raw, fallback)
        return project_simplex(raw + (anchor_true - anchor_raw).mean(axis=0), fallback)
    if name == "R2_PER_TEMPERATURE_MEAN_SHIFT":
        out = raw.copy()
        for t in np.unique(temps):
            m = anchor_temps == t
            if not m.any():
                continue
            out[temps == t] = raw[temps == t] + (anchor_true[m] - anchor_raw[m]).mean(axis=0)
        return project_simplex(out, fallback)
    raise ValueError("unknown recovery estimator %r" % name)


def fewshot_split_role(composition_id):
    """Deterministic, salt-based, composition-level calibration/evaluation split.

    Exact spec-61 rule: SHA256("NI" | composition_id | "20260912"), ascending;
    lowest 30% -> CALIBRATION_POOL, rest -> RECOVERY_EVAL. Depends only on the
    composition id and the frozen salt, so it is reproducible and cannot be
    redrawn after the blind result is known.
    """
    h = hashlib.sha256(("NI|%s|%s" % (composition_id, FEWSHOT_SPLIT_SALT))
                       .encode("utf-8"))
    return int.from_bytes(h.digest(), "big")


FEWSHOT_SPLIT_RULE = ('SHA256("NI" | composition_id | "%s"), ascending; lowest %d%% '
                      '(= %d compositions) -> FEWSHOT_CALIBRATION, remaining %d -> '
                      'RECOVERY_EVALUATION' % (FEWSHOT_SPLIT_SALT,
                                               int(CALIBRATION_FRACTION * 100),
                                               N_CALIBRATION, N_RECOVERY_EVAL))

# spec 36: Ni blind composition ids
NI_ID_FORMAT = "NI_BLIND_%06d"          # NI_BLIND_000001 .. NI_BLIND_002000
NI_ID_REGEX = r"NI_BLIND_\d{6}"
