# -*- coding: utf-8 -*-
"""
k_common.py
STEP5K  -  ORIGINAL-PLAN GAP CLOSURE ONLY:
  PART A  external materials-literature validation (thermodynamic plausibility)
  PART B  virtual thermodynamic screening (in-domain, non-Ni, frozen surrogates)
  PART C  original V1.0 plan completion audit

ABSOLUTE RULE (brief): STEP5K must not alter any frozen Ni blind artifact. No Ni
CALPHAD, no new Ni composition, no Ni model prediction, no Ni target/mapping/
metric/few-shot change. Every calculation here is POST-FREEZE and NON-Ni ONLY.

Part B claim boundary (spec 16, 36): outputs are THERMODYNAMICALLY PROMISING /
CALPHAD-INFORMED / EQUILIBRIUM SCREENING candidates only - never "optimal",
"best", "highest-strength" or "industrial" alloys.
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
FULL_V1_CODE = os.path.join(FULL_V1, "code_snapshot")
STEP5A = os.path.join(MLROOT, "04_ML", "STEP5A_JMST_ML_FOUNDATION_V2")
STEP5A_CODE = os.path.join(STEP5A, "code_snapshot")
FINAL = os.path.join(MLROOT, "07_FINAL_BLIND")
STEP5G = os.path.join(FINAL, "STEP5G_PIPELINE_FREEZE_V1")
STEP5HA = os.path.join(FINAL, "STEP5H_A_PREDICTION_FREEZE_V1")
STEP5HB = os.path.join(FINAL, "STEP5H_B_NI_BLIND_REVEAL_V1")
STEP5I = os.path.join(FINAL, "STEP5I_POSTBLIND_FEWSHOT_RECOVERY_V1")
STEP5J = os.path.join(MLROOT, "08_FINAL_MANUSCRIPT", "STEP5J_FINAL_EVIDENCE_FREEZE_V1")
ROOT = os.path.join(MLROOT, "08_FINAL_MANUSCRIPT", "STEP5K_ORIGINAL_PLAN_GAP_CLOSURE_V1")

D_CONFIG = os.path.join(ROOT, "00_config")
D_LIT = os.path.join(ROOT, "01_literature_search")
D_EXTVAL = os.path.join(ROOT, "02_external_validation")
D_EXTCAL = os.path.join(ROOT, "03_external_validation_CALPHAD")
D_SCREEN = os.path.join(ROOT, "04_virtual_screening")
D_CONFIRM = os.path.join(ROOT, "05_screening_confirmation")
D_FIG = os.path.join(ROOT, "06_figures")
D_AUDIT = os.path.join(ROOT, "07_original_plan_audit")
D_INSERT = os.path.join(ROOT, "08_manuscript_insertions")
D_QA = os.path.join(ROOT, "09_qa")
D_MAN = os.path.join(ROOT, "10_manifest")
D_LOGS = os.path.join(ROOT, "11_logs")
D_CODE = os.path.join(ROOT, "code_snapshot")
ALL_DIRS = [D_CONFIG, D_LIT, D_EXTVAL, D_EXTCAL, D_SCREEN, D_CONFIRM, D_FIG,
            D_AUDIT, D_INSERT, D_QA, D_MAN, D_LOGS, D_CODE]

# frozen artifact hashes that STEP5K must leave untouched (brief ABSOLUTE RULE, audit 1-6)
PROTECTED_HASHES = {
    "step5g_primary_ckpt": (os.path.join(STEP5G, "02_primary_model",
                                         "STEP5G_PRIMARY_MODEL_CHECKPOINT.pt"),
                            "f89e4ddc2f367a572a16a29062faa3c3d580fc8aae73a8e6bfaa28a8bcd86d0f"),
    "step5g_secondary_ckpt": (os.path.join(STEP5G, "03_secondary_model",
                                           "STEP5G_SECONDARY_MODEL_CHECKPOINT.pt"),
                              "a685b2138f114f3274f3b789753b28971484b797ca7d375a9354021242d55719"),
    "step5ha_primary_pred": (os.path.join(STEP5HA, "04_primary_predictions",
                                          "STEP5H_A_PRIMARY_NI_PREDICTIONS.parquet"),
                             "100b40ed67c396cb296d53db80e7fef7cb94c0c8223873f5a13c01af28f0f488"),
    "step5ha_secondary_pred": (os.path.join(STEP5HA, "05_secondary_predictions",
                                            "STEP5H_A_SECONDARY_NI_PREDICTIONS.parquet"),
                               "a6c0fc4a98a05a4c53704d85b5caf91a5477577789a69d8dc24f6976f98a72ad"),
}
PROTECTED_FILES = [
    os.path.join(STEP5HB, "03_processed_targets", "STEP5H_B_NI_TARGETS.parquet"),
    os.path.join(STEP5HB, "06_primary_score", "STEP5H_B_PRIMARY_SCORE_FREEZE.json"),
    os.path.join(STEP5HB, "07_secondary_score", "STEP5H_B_SECONDARY_SCORE_FREEZE.json"),
    os.path.join(STEP5I, "STEP5I_SUMMARY.json"),
    os.path.join(STEP5I, "11_manifest", "STEP5I_FINAL_SEAL_HASHES.csv"),
    os.path.join(STEP5J, "00_freeze", "FINAL_NUMERICAL_RESULTS.json"),
]

# --------------------------------------------------------------- protocol -- #
SIX = ["ZR", "TI", "CO", "SI", "AG", "MO"]            # spec 17: screening chemistry
FORBIDDEN_X = ["NI", "AU", "CE", "BE", "SN", "MN", "LA", "P"]  # non-Ni absolute; others not developed-identity surrogates for screening
FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
SCREEN_TEMPS = [450, 600, 950, 1000]                  # spec 21
N_PER_FAMILY = 20000                                  # spec 20
SCREEN_MASTER_SEED = 20260913                         # new deterministic seed, recorded
S_CRNB_RANGE = (0.6, 2.0)                             # spec 18: Domain A bounds
R_CRNB_RANGE = (2.0, 6.5)
X_WT_RANGE = (0.02, 0.20)
M_CR, M_NB = 51.9961, 92.9064
WT_DECIMALS = 4
MAX_FINALISTS_TOTAL = 30                              # spec 28
FINALISTS_PER_FAMILY = (3, 5)
CONFIRM_TEMPS = [450, 600, 950, 1000]                 # spec 30
C15_BAND_QUANTILES = (0.25, 0.75)                     # spec 25 example approach, frozen here
OOD_KNN = 5
OOD_MAX_KNN_DIST = 1.5        # frozen BEFORE any screening result is seen (spec 25/26)
LIQUID_TOLERANCE = 0.01       # Objective E: LIQUID family fraction must stay <= 1%
                              # at every screening temperature (in-domain: Domain-A
                              # liquid fraction is 0; this is pure QA slack)
N_FEATURES = 43
A_FEATLIST = os.path.join(STEP5A, "02_features", "SELECTED_FEATURES_F1_F2_F3_F4.txt")
A_FEATLIST_SHA256 = "f4ddf4a2587e33a7d441656110c589208d9d860095198dd4b276df64f8a2fb0c"

NI_TOKEN = re.compile(r"(?<![A-Za-z])NI(?![A-Za-z])")
FORBIDDEN_CANDIDATE_WORDS = ["optimal alloy", "best alloy", "highest-strength",
                             "highest strength", "industrial alloy"]
ALLOWED_CANDIDATE_WORDS = ["thermodynamically promising", "CALPHAD-informed candidate",
                           "equilibrium screening candidate"]


# ------------------------------------------------------------------ utils -- #
def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def invert_s_r(s_crnb, r_atomic):
    """Frozen FULL_V1 inversion (spec 19): wNb = S/(1+R*M_Cr/M_Nb); wCr = S - wNb."""
    w_nb = s_crnb / (1.0 + r_atomic * (M_CR / M_NB))
    return s_crnb - w_nb, w_nb


def actual_ratio(w_cr, w_nb):
    return (w_cr / M_CR) / (w_nb / M_NB)


def protect_baseline():
    """Record sha256 of every protected frozen artifact at STEP5K start."""
    rows = {}
    for k, (p, expected) in PROTECTED_HASHES.items():
        actual = sha256_file(p)
        if actual != expected:
            raise SystemExit("FATAL: protected artifact %s CHANGED (expected %s, "
                             "actual %s). STEP5K must not alter frozen Ni blind "
                             "artifacts - STOP." % (k, expected[:16], actual[:16]))
        rows[k] = actual
    for p in PROTECTED_FILES:
        rows[os.path.basename(p)] = sha256_file(p)
    return rows


def verify_protected(baseline):
    """Re-verify at the end: nothing protected moved during STEP5K."""
    bad = []
    for k, (p, _) in PROTECTED_HASHES.items():
        if sha256_file(p) != baseline[k]:
            bad.append(k)
    for p in PROTECTED_FILES:
        if sha256_file(p) != baseline[os.path.basename(p)]:
            bad.append(os.path.basename(p))
    return bad


def ni_guard(paths):
    """Scan STEP5K text outputs for stray Ni computation evidence (spec: no Ni
    composition/state may be generated here). The Ni blind artifacts themselves
    live outside STEP5K and are only REFERENCED by hash."""
    hits = []
    for p in paths:
        if not os.path.exists(p):
            continue
        if p.endswith((".csv", ".md", ".json", ".txt")):
            txt = open(p, encoding="utf-8", errors="ignore").read()
            for m in NI_TOKEN.finditer(txt):
                ctx = txt[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
                hits.append("%s::%s" % (os.path.basename(p), ctx))
                if len(hits) > 10:
                    return hits
    return hits
