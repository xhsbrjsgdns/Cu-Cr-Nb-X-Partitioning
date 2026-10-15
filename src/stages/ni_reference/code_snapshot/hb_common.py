# -*- coding: utf-8 -*-
"""
hb_common.py
STEP5H-B  -  PROSPECTIVE Ni CALPHAD BLIND REVEAL
             + PRIMARY BLIND SCORING
             + SECONDARY PROSPECTIVE SENSITIVITY SCORING

One-sentence scope (spec, final): run the exact preregistered 22,000-state Ni
CALPHAD blind reveal ONLY AFTER verifying the frozen prediction hashes, freeze
the Ni thermodynamic targets, score and hash the PRIMARY prospective zero-shot
endpoint FIRST, score the SECONDARY augmentation sensitivity SECOND, and STOP
before any post-blind few-shot recovery (spec 50, 51, 56).

Hard rules encoded here:
  - predictions are IMMUTABLE: never regenerated, never re-featurized (spec 3)
  - no prediction-guided CALPHAD: every one of the 22,000 states is treated
    equally; no prediction, disagreement, OOD distance, predicted topology or
    few-shot role influences what is calculated, at which temperature, with
    which tolerance, or in which retry order (spec 9)
  - the frozen FULL_V1 runner is reused byte-identically (spec 6); its sha256
    is verified against the value recorded in STEP5F/STEP5G before any call
  - failures are retained with verbatim exceptions; targets NaN; never
    interpolated, imputed, temperature-copied, fabricated or deleted (spec 18)
  - the target table is frozen BEFORE any scoring (spec 22, 23)
  - PRIMARY is scored and frozen BEFORE SECONDARY is even loaded (spec 25, 36)
  - skill uses the FROZEN train-mean simplices (spec 31, 32, 39); never
    redefined after reveal
  - no few-shot of any kind executes in this stage (spec 50)
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
FROZEN_RUNNER = os.path.join(FULL_V1_CODE, "22_fullv1_run.py")
STEP5G = os.path.join(MLROOT, "07_FINAL_BLIND", "STEP5G_PIPELINE_FREEZE_V1")
STEP5HA = os.path.join(MLROOT, "07_FINAL_BLIND", "STEP5H_A_PREDICTION_FREEZE_V1")
ROOT = os.path.join(MLROOT, "07_FINAL_BLIND", "STEP5H_B_NI_BLIND_REVEAL_V1")

D_CONFIG = os.path.join(ROOT, "00_config")
D_ACCESS = os.path.join(ROOT, "01_first_target_access")
D_RAW = os.path.join(ROOT, "02_raw_CALPHAD")
D_TARGETS = os.path.join(ROOT, "03_processed_targets")
D_VOCAB = os.path.join(ROOT, "04_phase_vocab")
D_CALQA = os.path.join(ROOT, "05_calphad_qa")
D_PSCORE = os.path.join(ROOT, "06_primary_score")
D_SSCORE = os.path.join(ROOT, "07_secondary_score")
D_POSTREV = os.path.join(ROOT, "08_postreveal_diagnostics")
D_FIG = os.path.join(ROOT, "09_figures")
D_QA = os.path.join(ROOT, "10_qa")
D_MAN = os.path.join(ROOT, "11_manifest")
D_LOGS = os.path.join(ROOT, "12_logs")
D_CODE = os.path.join(ROOT, "code_snapshot")
ALL_DIRS = [D_CONFIG, D_ACCESS, D_RAW, D_TARGETS, D_VOCAB, D_CALQA, D_PSCORE,
            D_SSCORE, D_POSTREV, D_FIG, D_QA, D_MAN, D_LOGS, D_CODE]

# frozen inputs (read-only)
G_NI_COMP_CSV = os.path.join(STEP5G, "04_ni_composition_prereg",
                             "STEP5G_NI_BLIND_COMPOSITIONS.csv")
G_FREEZE_JSON = os.path.join(STEP5G, "01_pipeline_freeze", "STEP5G_FINAL_PIPELINE_FREEZE.json")
G_METRIC_JSON = os.path.join(STEP5G, "05_metric_prereg", "STEP5G_METRIC_PREREGISTRATION.json")
HA_PRIMARY_PRED = os.path.join(STEP5HA, "04_primary_predictions",
                               "STEP5H_A_PRIMARY_NI_PREDICTIONS.parquet")
HA_SECONDARY_PRED = os.path.join(STEP5HA, "05_secondary_predictions",
                                 "STEP5H_A_SECONDARY_NI_PREDICTIONS.parquet")
HA_PRIMARY_FREEZE = os.path.join(STEP5HA, "04_primary_predictions",
                                 "STEP5H_A_PRIMARY_PREDICTION_FREEZE.json")
HA_SECONDARY_FREEZE = os.path.join(STEP5HA, "05_secondary_predictions",
                                   "STEP5H_A_SECONDARY_PREDICTION_FREEZE.json")
HA_OOD_STATE = os.path.join(STEP5HA, "03_ood_diagnostics", "STEP5H_A_NI_OOD_DIAGNOSTICS.csv")
HA_DISAGREE = os.path.join(STEP5HA, "06_disagreement",
                           "STEP5H_A_PRIMARY_SECONDARY_DISAGREEMENT.csv")
HA_MANIFEST = os.path.join(STEP5HA, "09_manifest", "STEP5H_A_FILE_HASHES.csv")

# ------------------------------------------------- frozen hash contract ---- #
FROZEN = {
    "ni_compositions_csv": "7680b5a1ba5a2409f2b23900d770dd4939df0cf76bd2f1d9c1a2c2c5cce0030b",
    "primary_predictions": "100b40ed67c396cb296d53db80e7fef7cb94c0c8223873f5a13c01af28f0f488",
    "secondary_predictions": "a6c0fc4a98a05a4c53704d85b5caf91a5477577789a69d8dc24f6976f98a72ad",
    "step5g_pipeline_freeze": "15efe35e2ad4f5372323067feccb989079d9fd5666153298a8c2d9f45d3b1788",
    "frozen_runner": "4325f256d28bd2f3d9c21d087720c325f372adefeb155d645a812e758f2ab5f7",
    "ha_manifest_recorded_self_hash":
        "712521fff857c166f9611fac1b5e88235b1148fcfdfe36f5ff82d4698811b25a",
    "ha_manifest_actual_final":
        "9db63f711baf58b55094839d22be96eb19d7ff2d1595bd7ad9fe25ce461aec7b",
}

# --------------------------------------------------------------- protocol -- #
FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]     # spec 14, 29
TRUE_COLS = ["true_%s" % f for f in FAMILIES_ORDER]
TEMPERATURES_C = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]
N_COMPOSITIONS = 2000
N_STATES = 22000

# frozen trivial baselines (spec 31, 39) - never recomputed, never redefined
PRIMARY_TRIVIAL = [0.3924063060849874, 0.17997911835521582, 0.2687588732102482,
                   0.1582142342482923, 0.0006414681012562028]
SECONDARY_TRIVIAL = [0.45812328163558386, 0.1240658371930271, 0.18471172806626412,
                     0.21560165117755808, 0.01749750192756682]

BOOTSTRAP_REPS = 1000            # spec 30
BOOTSTRAP_SEED = 20260911
HISTORICAL_LOEO_ORIGINAL6 = 0.752644      # spec 47 - context ONLY
HISTORICAL_LOEO_FULL5 = 0.710332
INTERPRETATION_GUIDES = {                 # spec 48 - descriptive, never a threshold
    "better_than_historical_loeo_mean_if_below": HISTORICAL_LOEO_ORIGINAL6,
    "relatively_strong_prospective_transfer_if_below": 0.50}


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


def verify_hash(path, key, label=None):
    """Hard pre-run gate (spec 2): exact match or STOP."""
    expected = FROZEN[key]
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit("FATAL: %s hash mismatch.\n  expected %s\n  actual   %s\n"
                         "STOP (spec 2)." % (label or key, expected, actual))
    return actual


def hard_gate():
    """Spec 2: all four frozen hashes verified before ANY Thermo-Calc call."""
    verify_hash(G_NI_COMP_CSV, "ni_compositions_csv", "Ni composition table")
    verify_hash(HA_PRIMARY_PRED, "primary_predictions", "PRIMARY predictions")
    verify_hash(HA_SECONDARY_PRED, "secondary_predictions", "SECONDARY predictions")
    verify_hash(G_FREEZE_JSON, "step5g_pipeline_freeze", "STEP5G pipeline freeze")
    verify_hash(FROZEN_RUNNER, "frozen_runner", "frozen FULL_V1 CALPHAD runner")
    return True


# ---------------------------------------------------------------- metrics -- #
def tvd(pred, true):
    return 0.5 * np.abs(np.asarray(pred, float) - np.asarray(true, float)).sum(axis=1)


def topology_of(Y):
    """Spec 29: argmax over the frozen family order (ties -> first in order,
    which np.argmax gives by construction)."""
    return np.asarray(Y, float).argmax(axis=1)


def metric_block(y_true, y_pred, trivial_row):
    """The frozen STEP5G metric-preregistration block, exactly."""
    from sklearn.metrics import f1_score
    v = tvd(y_pred, y_true)
    triv = np.tile(np.asarray(trivial_row, float), (len(y_true), 1))
    tv = tvd(triv, y_true)
    t, p = topology_of(y_true), topology_of(y_pred)
    mae_per_family = np.abs(y_pred - y_true).mean(axis=0)
    return {
        "n_rows": int(len(y_true)),
        "mean_tvd": float(v.mean()), "median_tvd": float(np.median(v)),
        "p95_tvd": float(np.percentile(v, 95)),
        "macro_mae": float(np.abs(y_pred - y_true).mean()),
        "per_family_mae": {f: float(mae_per_family[j])
                           for j, f in enumerate(FAMILIES_ORDER)},
        "topology_accuracy": float((t == p).mean()),
        "topology_macro_f1": float(f1_score(t, p, average="macro",
                                            labels=list(range(5)), zero_division=0)),
        "trivial_mean_tvd": float(tv.mean()),
        "skill_vs_trivial": float(tv.mean() - v.mean()),
        "relative_skill": float(1.0 - v.mean() / tv.mean()) if tv.mean() > 0 else None,
    }


def cluster_bootstrap_mean_tvd(v, comp_ids, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Spec 30: composition-level cluster bootstrap; all 11 temperatures of one
    composition move together; 95% percentile CI."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(comp_ids)
    idx_by_comp = {c: np.flatnonzero(comp_ids == c) for c in uniq}
    n = len(uniq)
    means = np.empty(reps)
    for b in range(reps):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([idx_by_comp[uniq[k]] for k in pick])
        means[b] = v[idx].mean()
    return {"replicates": reps, "seed": seed, "unit": "composition_id",
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)),
            "boot_mean": float(means.mean()),
            "boot_std": float(means.std(ddof=1))}


def write_hashes_no_selfref(out_csv, root, extra_sidecar=None):
    """Spec 54 item 31: hash every file under ROOT EXCEPT the manifest itself;
    the manifest's own hash goes only into an external sidecar."""
    rows = []
    for base, _, files in os.walk(root):
        if "__pycache__" in base:
            continue
        for fn in sorted(files):
            if fn.endswith(".pyc"):
                continue
            p = os.path.join(base, fn)
            rel = os.path.relpath(p, root).replace("\\", "/")
            if rel == os.path.relpath(out_csv, root).replace("\\", "/"):
                continue                      # never self-referential
            rows.append({"relative_path": rel, "bytes": os.path.getsize(p),
                         "sha256": sha256_file(p)})
    pd.DataFrame(sorted(rows, key=lambda r: r["relative_path"])).to_csv(
        out_csv, index=False, encoding="utf-8")
    if extra_sidecar:
        with open(extra_sidecar, "w", encoding="utf-8") as f:
            f.write("%s  %s\n" % (sha256_file(out_csv),
                                  os.path.basename(out_csv)))
    return len(rows)
