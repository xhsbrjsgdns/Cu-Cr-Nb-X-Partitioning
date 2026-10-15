# -*- coding: utf-8 -*-
"""
f5p_common.py
STEP5F-PILOT  -  NON-Ni CHEMICAL-IDENTITY CALPHAD FEASIBILITY PILOT

Purpose (spec 38): run a small, controlled CALPHAD feasibility pilot for nine non-Ni
candidate chemical identities, then select four to six technically stable and
thermodynamically complementary elements for the full chemical-identity augmentation,
WITHOUT accessing Ni and WITHOUT retraining any model.

Hard rules encoded here:
  - the nine pilot elements are fixed: Be Sn Mn Al Mg Zn La Ge P.
    Ni, Au and Ce are NOT in this pilot (spec 1).
  - Ni is absolutely sealed: no descriptor, no property, no composition, no prediction,
    no CALPHAD, no target (spec 3). Nothing here reads a Ni row of anything.
  - NO model training, NO Optuna, NO feature selection, NO LOEO, NO few-shot (spec 4).
  - thermodynamic settings are identical to FULL_V1 (spec 7) and are imported from the
    frozen FULL_V1 code snapshot rather than restated.
  - the 5-family mapping and the MOLAR partition definition are the frozen FULL_V1 ones
    (spec 15, 17). phase_volume_fraction x element_mole_fraction is forbidden.
  - a CALPHAD failure is retained, never interpolated, fabricated or silently dropped
    (spec 19).
  - this pilot shortlist is NOT the final augmentation set, and the STEP5F-0 automated
    PRIMARY is a target-free descriptor audit result, not a CALPHAD verdict (spec 2).
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
FULL_V1 = r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1"
FULL_V1_CODE = os.path.join(FULL_V1, "code_snapshot")
STEP5F0 = r"E:\Cu_CALPHAD_ML\05_OOD\STEP5F0_NONNI_CANDIDATE_AUDIT_V1"
ROOT = r"E:\Cu_CALPHAD_ML\06_AUGMENTATION\STEP5F_PILOT_V1"

D_CONFIG = os.path.join(ROOT, "00_config")
D_COMP = os.path.join(ROOT, "01_compositions")
D_RAW = os.path.join(ROOT, "02_raw_CALPHAD")
D_PROC = os.path.join(ROOT, "03_processed")
D_VOCAB = os.path.join(ROOT, "04_phase_vocab")
D_TOPO = os.path.join(ROOT, "05_topology")
D_QA = os.path.join(ROOT, "06_qa")
D_FIG = os.path.join(ROOT, "07_figures")
D_MANIFEST = os.path.join(ROOT, "08_manifest")
D_LOGS = os.path.join(ROOT, "09_logs")
ALL_DIRS = [D_CONFIG, D_COMP, D_RAW, D_PROC, D_VOCAB, D_TOPO, D_QA, D_FIG,
            D_MANIFEST, D_LOGS]

ML_READY = os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet")
F5P_DESCRIPTORS = os.path.join(STEP5F0, "02_descriptors",
                               "STEP5F0_CANDIDATE_DESCRIPTOR_MATRIX.csv")

# --------------------------------------------------------------- protocol -- #
PILOT_ELEMENTS = ["BE", "SN", "MN", "AL", "MG", "ZN", "LA", "GE", "P"]     # spec 1
FORBIDDEN_ELEMENTS = ["NI", "AU", "CE"]                                    # spec 1, 3
ROBUST_CORE = ["BE", "SN", "MN"]                                           # spec 31
EXISTING_SIX = ["ZR", "TI", "CO", "SI", "AG", "MO"]

N_PER_ELEMENT = 50                                                          # spec 5
N_COMPOSITIONS = N_PER_ELEMENT * len(PILOT_ELEMENTS)                        # 450
TEMPERATURES_C = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]   # spec 6
N_FORMAL_STATES = N_COMPOSITIONS * len(TEMPERATURES_C)                      # 4950

SAMPLING_SEED = 20260909                                                    # spec 11
S_CRNB_RANGE = (0.6, 2.0)                                                   # spec 9, Domain A
R_CRNB_RANGE = (2.0, 6.5)                                                   # spec 9, Domain A
X_WT_RANGE = (0.02, 0.20)                     # spec 10: identical for all nine
M_CR, M_NB = 51.9961, 92.9064                 # FULL_V1 molar masses, unchanged
WT_DECIMALS = 4                               # FULL_V1 input precision, unchanged

FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
TOPOLOGY_CLASSES = FAMILIES_ORDER

# thresholds, all preregistered before any result is seen
NEAR_DEGENERATE_FRACTION = 0.99      # spec 25
OTHER_DOMINATED_FRACTION = 0.90      # spec 26
UNSTABLE_CONVERGENCE = 0.95          # spec 32
ANOMALOUS_STATE_FRACTION = 0.20      # spec 32
CORE_RETENTION_CONVERGENCE = 0.98    # spec 31
SELECT_MIN, SELECT_MAX, SELECT_TARGET = 4, 6, 5    # spec 29

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


def sha256_obj(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


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
    """Element-token Ni scan over frames and text. 'DOMINANT' / 'NIOBIUM' cannot match.

    Scans every NON-NUMERIC column, not just object dtype: pandas >= 3 stores strings as
    'str' dtype, and an element symbol is exactly the sort of value that would hide there.
    """
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


def read_table(stem):
    """Read whichever form io_store.write_table produced (.parquet / .csv.gz / .csv).

    io_store falls back to gzipped CSV when pyarrow is unavailable, so a reader that
    only checks .parquet and .csv silently misses a real table.
    """
    for suf, fn in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv),
                    (".csv", pd.read_csv)):
        p = stem + suf
        if os.path.exists(p):
            return fn(p)
    raise FileNotFoundError(stem + ".{parquet|csv.gz|csv}")


# ----------------------------------------------------------- composition -- #
def invert_s_r(s_crnb, r_atomic):
    """FULL_V1 inversion, unchanged: wNb = S / (1 + R*(M_Cr/M_Nb)); wCr = S - wNb.

    Cr and Nb are NEVER sampled independently (frozen FULL_V1 sampling policy).
    """
    w_nb = s_crnb / (1.0 + r_atomic * (M_CR / M_NB))
    w_cr = s_crnb - w_nb
    return w_cr, w_nb


def actual_ratio(w_cr, w_nb):
    return (w_cr / M_CR) / (w_nb / M_NB)


def element_child_seed(element):
    """Independent child sequence per element from the one pilot seed (spec 11)."""
    idx = PILOT_ELEMENTS.index(element)
    ss = np.random.SeedSequence(SAMPLING_SEED)
    return ss.spawn(len(PILOT_ELEMENTS))[idx]


# -------------------------------------------------------------- topology -- #
def dominant_topology(partition_matrix):
    """argmax of the X partition vector over the frozen five families."""
    P = np.asarray(partition_matrix, dtype=np.float64)
    idx = np.nanargmax(P, axis=1)
    return np.array([FAMILIES_ORDER[i] for i in idx], dtype=object)


def topology_distribution(labels):
    lab = pd.Series(list(labels))
    counts = lab.value_counts()
    n = max(int(counts.sum()), 1)
    return {c: float(counts.get(c, 0)) / n for c in TOPOLOGY_CLASSES}


def shannon_entropy(dist, base=None):
    """Entropy of a dominant-topology distribution (spec 23).

    A high value is NOT automatically 'better'. It is only an input to the
    chemical-complementarity discussion.
    """
    p = np.array([dist.get(c, 0.0) for c in TOPOLOGY_CLASSES], dtype=np.float64)
    p = p[p > 0]
    if not len(p):
        return 0.0
    h = float(-(p * np.log(p)).sum())
    if base == "classes":
        h /= math.log(len(TOPOLOGY_CLASSES))
    return h


def jensen_shannon(p, q):
    """JS distance between two topology distributions; 0 = identical, 1 = disjoint."""
    a = np.array([p.get(c, 0.0) for c in TOPOLOGY_CLASSES], dtype=np.float64)
    b = np.array([q.get(c, 0.0) for c in TOPOLOGY_CLASSES], dtype=np.float64)
    a = a / a.sum() if a.sum() > 0 else a
    b = b / b.sum() if b.sum() > 0 else b
    m = 0.5 * (a + b)

    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log(x[mask] / np.where(y[mask] > 0, y[mask], 1e-300))).sum())

    js = 0.5 * kl(a, m) + 0.5 * kl(b, m)
    return float(math.sqrt(max(js, 0.0) / math.log(2)))
