# -*- coding: utf-8 -*-
"""
h5h_common.py
STEP5H-A  -  Ni INPUT DESCRIPTOR EXTRACTION + FEATURE CONSTRUCTION
           + PRIMARY/SECONDARY ZERO-SHOT PREDICTION FREEZE
           + TARGET-FREE OOD DIAGNOSTICS + POST-BLIND MAXIMIN-ID FREEZE

One-sentence scope (spec, final): access only the Ni elemental inputs required by
the already-frozen 43-feature schema, generate and cryptographically freeze
PRIMARY and SECONDARY zero-shot predictions for the 22,000 preregistered Ni
states, freeze target-free OOD diagnostics and the secondary maximin few-shot
sequence, and STOP before any Ni CALPHAD target is generated or accessed.

Absolute prohibitions encoded here (spec 5, 37, 38, 45):
  - NO Thermo-Calc launch, NO tc_python import anywhere in STEP5H-A
  - NO Ni phase labels / fractions / partition vectors / CALPHAD files / targets
  - NO Ni error calculation, NO scoring of PRIMARY or SECONDARY
  - NO few-shot execution, NO B_skill / B_recovery25 computation
  - NO model retraining, NO scaler refit, NO feature reselection
  - NO clipping / winsorizing / capping / recentering of standardized OOD values
The target-column guard (spec 39) is a hard check run before any prediction
artifact is written.

Source policy (spec 2, 3): ONLY the frozen STEP5A approved sources -
matminer MagpieData (magpie.data.json), mendeleev built-in elemental database,
matminer CohesiveEnergyData - with the same fields, units, formulas and
missing-value policy as STEP5A Stage F. No web search, no manual entry, no
Wikipedia, no Materials Project, no DFT, no alternate databases.
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
STEP5A = os.path.join(MLROOT, "04_ML", "STEP5A_JMST_ML_FOUNDATION_V2")
STEP5G = os.path.join(MLROOT, "07_FINAL_BLIND", "STEP5G_PIPELINE_FREEZE_V1")
ROOT = os.path.join(MLROOT, "07_FINAL_BLIND", "STEP5H_A_PREDICTION_FREEZE_V1")

D_CONFIG = os.path.join(ROOT, "00_config")
D_ACCESS = os.path.join(ROOT, "01_ni_descriptor_access")
D_FEAT = os.path.join(ROOT, "02_ni_features")
D_OOD = os.path.join(ROOT, "03_ood_diagnostics")
D_PRIMPRED = os.path.join(ROOT, "04_primary_predictions")
D_SECPRED = os.path.join(ROOT, "05_secondary_predictions")
D_DIS = os.path.join(ROOT, "06_disagreement")
D_MAXIMIN = os.path.join(ROOT, "07_maximin_freeze")
D_QA = os.path.join(ROOT, "08_qa")
D_MAN = os.path.join(ROOT, "09_manifest")
D_LOGS = os.path.join(ROOT, "10_logs")
D_CODE = os.path.join(ROOT, "code_snapshot")
ALL_DIRS = [D_CONFIG, D_ACCESS, D_FEAT, D_OOD, D_PRIMPRED, D_SECPRED, D_DIS,
            D_MAXIMIN, D_QA, D_MAN, D_LOGS, D_CODE]

# frozen STEP5G artifacts (read-only)
G_FREEZE_JSON = os.path.join(STEP5G, "01_pipeline_freeze", "STEP5G_FINAL_PIPELINE_FREEZE.json")
G_SCHEMA_CSV = os.path.join(STEP5G, "01_pipeline_freeze", "STEP5G_FEATURE_SCHEMA.csv")
G_NI_COMP_CSV = os.path.join(STEP5G, "04_ni_composition_prereg",
                             "STEP5G_NI_BLIND_COMPOSITIONS.csv")
G_NI_PREREG_JSON = os.path.join(STEP5G, "04_ni_composition_prereg",
                                "STEP5G_NI_BLIND_PREREGISTRATION.json")
G_METRIC_JSON = os.path.join(STEP5G, "05_metric_prereg", "STEP5G_METRIC_PREREGISTRATION.json")
G_FEWSHOT_JSON = os.path.join(STEP5G, "06_postblind_fewshot_prereg",
                              "STEP5G_POSTBLIND_FEWSHOT_PREREGISTRATION.json")
G_SEQ_CSV = os.path.join(STEP5G, "06_postblind_fewshot_prereg",
                         "STEP5G_NESTED_RANDOM_SEQUENCES.csv")
G_PRIMARY_CKPT = os.path.join(STEP5G, "02_primary_model",
                              "STEP5G_PRIMARY_MODEL_CHECKPOINT.pt")
G_PRIMARY_SCALER = os.path.join(STEP5G, "02_primary_model", "STEP5G_PRIMARY_SCALER.joblib")
G_SECONDARY_CKPT = os.path.join(STEP5G, "03_secondary_model",
                                "STEP5G_SECONDARY_MODEL_CHECKPOINT.pt")
G_SECONDARY_SCALER = os.path.join(STEP5G, "03_secondary_model",
                                  "STEP5G_SECONDARY_SCALER.joblib")
G_TRAIN_IDS = os.path.join(STEP5G, "02_primary_model", "STEP5G_PRIMARY_TRAIN_IDS.csv")

# historical feature sources (read-only)
A_CANDIDATE_FEATURES = os.path.join(STEP5A, "02_features", "FULL_V1_CANDIDATE_FEATURES.parquet")
A_DESC_TABLE = os.path.join(STEP5A, "01_feature_sources",
                            "ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.parquet")
F5F_DESC_TABLE = os.path.join(MLROOT, "06_AUGMENTATION", "STEP5F_FULL_V1", "00_config",
                              "STEP5F_ELEMENT_DESCRIPTOR_TABLE_14X_V1.parquet")
# frozen STEP5D diagnostic table: Domain-A six-identity rows with all candidate
# features + true targets; STEP5G trained the PRIMARY model on its DEV_TRAIN rows
D_DIAG_DATA = os.path.join(MLROOT, "05_OOD", "STEP5D_FAILURE_DIAGNOSIS_V1",
                           "02_frozen_inputs", "DOMAIN_A_DEV6_DIAGNOSTIC_DATA.parquet")

# --------------------------------------------------- frozen hash contract -- #
# spec 6, 13, 14, 15: verified EXACTLY at runtime; mismatch -> STOP.
FROZEN_HASHES = {
    "primary_checkpoint": "f89e4ddc2f367a572a16a29062faa3c3d580fc8aae73a8e6bfaa28a8bcd86d0f",
    "primary_scaler": "c2619e4aed54304a9ed895b98164b0205421310f613f50346c249de41d7e3475",
    "secondary_checkpoint": "a685b2138f114f3274f3b789753b28971484b797ca7d375a9354021242d55719",
    "secondary_scaler": "b247fdae442c475d58a052427a4744a62a88114d266b865e7e0f4116245d43d4",
    "feature_schema_csv": "faebd4de9423b6087e5ae2bb7242dade2538299eb294b3318538d62c3ca86126",
    "ni_compositions_csv": "7680b5a1ba5a2409f2b23900d770dd4939df0cf76bd2f1d9c1a2c2c5cce0030b",
    "feature_list_txt": "f4ddf4a2587e33a7d441656110c589208d9d860095198dd4b276df64f8a2fb0c",
    "fewshot_sequences_csv": "9c9389a100ee064bc8a52ab4a4d8cedb7558c9b95462cca0fa6736da92d6507e",
}

# --------------------------------------------------------------- protocol -- #
TEMPERATURES_C = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]  # spec 8
N_COMPOSITIONS = 2000
N_STATES = N_COMPOSITIONS * len(TEMPERATURES_C)                              # 22000
N_FEATURES = 43
FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
ORIGINAL_SIX = ["ZR", "TI", "CO", "SI", "AG", "MO"]
R_GAS = 8.314462618          # frozen STEP5A constant (J/mol K)
M_CR, M_NB = 51.9961, 92.9064
CONSISTENCY_TEMPS = [300, 700, 1000]     # spec 12
CONSISTENCY_TOL = 1e-10                  # spec 12

# Ni property minimization (spec 4): the ONLY Ni elemental quantities extracted
# directly, each mapped to the frozen 43-feature columns that require it.
NI_REQUIRED_PROPERTIES = [
    # (property, source_key, field, unit, required_by_features)
    ("AtomicNumber", "magpie", "Number", "-",
     "Zbar; X_AtomicNumber; Delta_AtomicNumber_X_Cu"),
    ("AtomicWeight", "magpie", "AtomicWeight", "amu",
     "wt%->atomic-fraction conversion for EVERY composition-level feature; "
     "magpie_minimum_AtomicWeight; magpie_range_AtomicWeight"),
    ("CovalentRadius", "magpie", "CovalentRadius", "pm",
     "magpie_range_CovalentRadius (via frozen matminer preset)"),
    ("Electronegativity", "magpie", "Electronegativity", "Pauling",
     "delta_chi; Delta_Electronegativity_X_Cu; magpie_minimum/maximum/range_"
     "Electronegativity"),
    ("VEC", "magpie", "NValence", "count",
     "X_VEC; Delta_VEC_X_Cu; Delta_VEC_X_Cr; magpie_minimum_NValence"),
    ("MeltingT", "magpie", "MeltingT", "K",
     "X_MeltingT; Delta_MeltingT_X_Cu; Delta_MeltingT_X_Cr; magpie_minimum/"
     "maximum/range_MeltingT"),
    ("FirstIonizationEnergy", "magpie", "FirstIonizationEnergy", "eV",
     "Delta_FirstIonizationEnergy_X_Cu; Delta_FirstIonizationEnergy_X_Cr"),
    ("AtomicVolume_GS", "magpie", "GSvolume_pa", "A^3/atom",
     "AtomicVol_bar; magpie_minimum/maximum_GSvolume_pa"),
    ("MetallicRadius", "mendeleev", "atomic_radius", "pm",
     "r_bar; delta_r"),
    ("Density", "mendeleev", "density", "g/cm^3",
     "density_bar; Delta_Density_X_Cu"),
]
# Explicitly NOT extracted (spec 4 minimization): ElectronAffinity,
# MendeleevNumber, CohesiveEnergy - no frozen 43-feature column requires them.
NI_NOT_EXTRACTED = ["ElectronAffinity", "MendeleevNumber", "CohesiveEnergy"]

PMG_SYMBOL = {"CU": "Cu", "CR": "Cr", "NB": "Nb", "NI": "Ni",
              "ZR": "Zr", "TI": "Ti", "CO": "Co", "SI": "Si", "AG": "Ag", "MO": "Mo",
              "BE": "Be", "SN": "Sn", "MN": "Mn", "LA": "La", "P": "P"}

# target-column guard (spec 39)
TARGET_COLUMN_PATTERNS = [r"true", r"target", r"phase_fraction", r"X_P_", r"Ni_P_",
                          r"CALPHAD", r"ground_truth"]

NI_TOKEN = re.compile(r"(?<![A-Za-z])NI(?![A-Za-z])")


# ------------------------------------------------------------------ utils -- #
def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def verify_hash(path, expected, label):
    """Spec 6/13/14/15: exact-hash gate. Any mismatch is a hard STOP."""
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit("FATAL: %s hash mismatch.\n  expected %s\n  actual   %s\n"
                         "STOP (spec 6): a frozen artifact changed; nothing may "
                         "proceed until this is resolved." % (label, expected, actual))
    return actual


def feature_schema():
    """The frozen 43-feature production schema, in its exact frozen order."""
    sch = pd.read_csv(G_SCHEMA_CSV)
    feats = list(sch.feature)
    if len(feats) != N_FEATURES:
        raise SystemExit("FATAL: feature schema has %d rows, expected %d"
                         % (len(feats), N_FEATURES))
    return feats


def assert_no_tc_python():
    """Spec 38: STEP5H-A must not import tc_python or any Thermo-Calc API."""
    import sys
    banned = [m for m in sys.modules if m.startswith("tc_python")]
    if banned:
        raise SystemExit("FATAL: tc_python is loaded in this process (%s). STEP5H-A "
                         "must not import Thermo-Calc (spec 38)." % banned)


def target_column_guard(df, label):
    """Spec 39: no target-like column may be present in any prediction artifact
    or in any frame about to be used as Ni labels."""
    hits = []
    for c in df.columns:
        cu = str(c)
        for pat in TARGET_COLUMN_PATTERNS:
            if re.search(pat, cu, re.IGNORECASE):
                hits.append("%s::%s (pattern %s)" % (label, cu, pat))
    return hits


def ni_scan_frame(df, label, allowed_cols=()):
    """Element-token Ni scan. In STEP5H-A the Ni identity columns are the POINT of
    the stage, so they are declared; everything else must be clean of stray Ni
    references except those declared."""
    hits = []
    for c in df.columns:
        if c in allowed_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        for v in pd.unique(df[c].dropna().astype(str)):
            if NI_TOKEN.search(v.upper()):
                hits.append("%s::%s::%s" % (label, c, v[:60]))
                if len(hits) > 20:
                    return hits
    return hits


def project_simplex(Y, fallback_row):
    """Frozen projection (STEP5G spec 27): clip negatives, renormalize, zero-sum
    rows -> the frozen production training-mean simplex."""
    P = np.clip(np.asarray(Y, dtype=np.float64), 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    n_fb = int((s[:, 0] == 0).sum())
    out = np.where(s > 0, P / np.where(s > 0, s, 1.0), np.asarray(fallback_row, float))
    return out, n_fb


def load_frozen_checkpoint(which):
    """Load a frozen STEP5G checkpoint dict and verify its recorded hash."""
    import torch
    ck_path = G_PRIMARY_CKPT if which == "PRIMARY" else G_SECONDARY_CKPT
    sc_path = G_PRIMARY_SCALER if which == "PRIMARY" else G_SECONDARY_SCALER
    key = "primary" if which == "PRIMARY" else "secondary"
    verify_hash(ck_path, FROZEN_HASHES[key + "_checkpoint"], "%s checkpoint" % which)
    verify_hash(sc_path, FROZEN_HASHES[key + "_scaler"], "%s scaler" % which)
    ck = torch.load(ck_path, weights_only=False, map_location="cpu")
    return ck, ck_path, sc_path
