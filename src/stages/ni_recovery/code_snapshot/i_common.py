# -*- coding: utf-8 -*-
"""
i_common.py
STEP5I  -  POST-BLIND Ni FEW-SHOT RECOVERY
        +  ANCHOR-ONLY R0/R1/R2 CONTROLS
        +  TOPOLOGY / NBNI3 MECHANISM DIAGNOSIS

What this stage is (spec 1): POST-BLIND RETROSPECTIVE RECOVERY.
What it is not: zero-shot, blind, or prospective zero-shot. Those three words are
reserved for the STEP5H-A/STEP5H-B result and are never applied to anything computed
here.

The zero-shot result is already frozen and is not touched:
    PRIMARY   ORIGINAL6_DIRECT_MLP   mean TVD 0.8643856187  topology 0.0785909
    SECONDARY FULL11_DIRECT_MLP      mean TVD 0.7234409099  topology 0.2406818
Both are worse than their own trivial baselines. STEP5I asks a different question:
how many DIRECT Ni thermodynamic anchors does it take to make the frozen formulation
useful, and does refitting the network actually beat simply knowing a handful of Ni
partition vectors?

Two design facts that must be stated rather than discovered later in the numbers:

1. Natural row weighting (spec 18) means the anchors are a vanishing fraction of the
   refit training set: 11 rows in 241,759 at budget 1, 352 in 242,100 at budget 32 -
   between 0.005% and 0.145%. No oversampling, reweighting or balancing is allowed,
   and none is applied. If the refit barely moves, that is a result about this
   preregistered protocol, not a bug to be corrected afterwards.
2. Early stopping uses the original-six DEV_VALIDATION rows only (spec 14), so the
   stopping rule carries no Ni signal at all.

Both were frozen before the reveal. They are reported, not repaired.
"""
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths -- #
MLROOT = r"E:\Cu_CALPHAD_ML"
FINAL = os.path.join(MLROOT, "07_FINAL_BLIND")
STEP5G = os.path.join(FINAL, "STEP5G_PIPELINE_FREEZE_V1")
STEP5HA = os.path.join(FINAL, "STEP5H_A_PREDICTION_FREEZE_V1")
STEP5HB = os.path.join(FINAL, "STEP5H_B_NI_BLIND_REVEAL_V1")
STEP5D = os.path.join(MLROOT, "05_OOD", "STEP5D_FAILURE_DIAGNOSIS_V1")
ROOT = os.path.join(FINAL, "STEP5I_POSTBLIND_FEWSHOT_RECOVERY_V1")

D_CONFIG = os.path.join(ROOT, "00_config")
D_FROZEN = os.path.join(ROOT, "01_frozen_inputs")
D_RANDOM = os.path.join(ROOT, "02_random_refit")
D_MAXIMIN = os.path.join(ROOT, "03_maximin_refit")
D_CONTROL = os.path.join(ROOT, "04_R0_R1_R2_controls")
D_METRICS = os.path.join(ROOT, "05_metrics")
D_TOPO = os.path.join(ROOT, "06_topology_diagnostics")
D_NBNI3 = os.path.join(ROOT, "07_NBNI3_diagnostics")
D_BOOT = os.path.join(ROOT, "08_bootstrap")
D_FIG = os.path.join(ROOT, "09_figures")
D_QA = os.path.join(ROOT, "10_qa")
D_MANIFEST = os.path.join(ROOT, "11_manifest")
D_LOGS = os.path.join(ROOT, "12_logs")
D_CODE = os.path.join(ROOT, "code_snapshot")
ALL_DIRS = [D_CONFIG, D_FROZEN, D_RANDOM, D_MAXIMIN, D_CONTROL, D_METRICS, D_TOPO,
            D_NBNI3, D_BOOT, D_FIG, D_QA, D_MANIFEST, D_LOGS, D_CODE]

# ------------------------------------------------- frozen, read-only inputs - #
G_COMMON = os.path.join(STEP5G, "code_snapshot", "g_common.py")
G_FREEZE_JSON = os.path.join(STEP5G, "01_pipeline_freeze",
                             "STEP5G_FINAL_PIPELINE_FREEZE.json")
G_SCHEMA_CSV = os.path.join(STEP5G, "01_pipeline_freeze", "STEP5G_FEATURE_SCHEMA.csv")
G_FEWSHOT_JSON = os.path.join(STEP5G, "06_postblind_fewshot_prereg",
                              "STEP5G_POSTBLIND_FEWSHOT_PREREGISTRATION.json")
G_SEQ_CSV = os.path.join(STEP5G, "06_postblind_fewshot_prereg",
                         "STEP5G_NESTED_RANDOM_SEQUENCES.csv")
G_PRIMARY_CKPT = os.path.join(STEP5G, "02_primary_model",
                              "STEP5G_PRIMARY_MODEL_CHECKPOINT.pt")
G_PRIMARY_SCALER = os.path.join(STEP5G, "02_primary_model", "STEP5G_PRIMARY_SCALER.joblib")
G_PRIMARY_SCALER_JSON = os.path.join(STEP5G, "02_primary_model",
                                     "STEP5G_PRIMARY_SCALER_PARAMETERS.json")
G_PRIMARY_TRAIN_IDS = os.path.join(STEP5G, "02_primary_model",
                                   "STEP5G_PRIMARY_TRAIN_IDS.csv")
G_PRIMARY_VAL_IDS = os.path.join(STEP5G, "02_primary_model",
                                 "STEP5G_PRIMARY_VALIDATION_IDS.csv")
G_SECONDARY_CKPT = os.path.join(STEP5G, "03_secondary_model",
                                "STEP5G_SECONDARY_MODEL_CHECKPOINT.pt")

HA_PRED = os.path.join(STEP5HA, "04_primary_predictions",
                       "STEP5H_A_PRIMARY_NI_PREDICTIONS.parquet")
HA_FEATURES = os.path.join(STEP5HA, "02_ni_features", "STEP5H_A_NI_FEATURES.parquet")
HA_MAXIMIN_CSV = os.path.join(STEP5HA, "07_maximin_freeze", "STEP5H_A_MAXIMIN_SEQUENCE.csv")
HA_MAXIMIN_JSON = os.path.join(STEP5HA, "07_maximin_freeze", "STEP5H_A_MAXIMIN_FREEZE.json")

HB_TARGETS = os.path.join(STEP5HB, "03_processed_targets", "STEP5H_B_NI_TARGETS.parquet")
HB_TARGET_FREEZE = os.path.join(STEP5HB, "03_processed_targets",
                                "STEP5H_B_NI_TARGET_FREEZE.json")
HB_PRIMARY_SCORE = os.path.join(STEP5HB, "06_primary_score",
                                "STEP5H_B_PRIMARY_SCORE_FREEZE.json")
HB_PRIMARY_METRICS = os.path.join(STEP5HB, "06_primary_score",
                                  "STEP5H_B_PRIMARY_BLIND_METRICS.json")
HB_SECONDARY_SCORE = os.path.join(STEP5HB, "07_secondary_score",
                                  "STEP5H_B_SECONDARY_SCORE_FREEZE.json")
HB_PHASE_VOCAB = os.path.join(STEP5HB, "04_phase_vocab", "STEP5H_B_PHASE_VOCABULARY.csv")

D_DIAG_DATA = os.path.join(STEP5D, "02_frozen_inputs",
                           "DOMAIN_A_DEV6_DIAGNOSTIC_DATA.parquet")

# Hashes given in the STEP5I brief (sections 3, 4, 26) plus those recorded by the
# earlier freezes. Every one is checked before anything is trained; a mismatch stops
# the stage rather than being worked around.
FROZEN_HASHES = {
    "HB_TARGETS": "3e1f4ba2cf6e47120edc4e5150059541ac3d7713e850ac867f411c58126858e3",
    "HA_PRIMARY_PREDICTIONS": "100b40ed67c396cb296d53db80e7fef7cb94c0c8223873f5a13c01af28f0f488",
    "HA_MAXIMIN_SEQUENCE": "6704c7e9d71f64dc8395c55b0beed239f29f99f6e7e9bfa510ab8f4d64221fda",
    "G_RANDOM_SEQUENCES": "9c9389a100ee064bc8a52ab4a4d8cedb7558c9b95462cca0fa6736da92d6507e",
    "G_PRIMARY_CHECKPOINT": "f89e4ddc2f367a572a16a29062faa3c3d580fc8aae73a8e6bfaa28a8bcd86d0f",
    "G_PRIMARY_SCALER": "c2619e4aed54304a9ed895b98164b0205421310f613f50346c249de41d7e3475",
    "G_SECONDARY_CHECKPOINT": "a685b2138f114f3274f3b789753b28971484b797ca7d375a9354021242d55719",
}

# --------------------------------------------------------------- protocol -- #
FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
TRUE_COLS = ["true_%s" % f for f in FAMILIES_ORDER]
RAW_COLS = ["raw_%s" % f for f in FAMILIES_ORDER]
PRED_COLS = ["pred_%s" % f for f in FAMILIES_ORDER]
DIAG_TRUE_COLS = ["true_x_partition_%s" % f for f in FAMILIES_ORDER]

TEMPERATURES_C = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]
LOW_T = [300, 350, 400, 450, 500, 600]      # spec 39: the NBNI3 window
HIGH_T = [700, 800, 900, 950, 1000]

N_FEATURES = 43
N_CALIBRATION = 600
N_RECOVERY_EVAL = 1400
N_RECOVERY_STATES = N_RECOVERY_EVAL * len(TEMPERATURES_C)          # 15400
BUDGETS = [1, 2, 4, 8, 16, 32]
REPLICATES = [1, 2, 3]

REFIT_LR = 0.0003                    # spec 13, fixed, never retuned
REFIT_BATCH = 1024                   # spec 14
REFIT_EPOCHS = 200
REFIT_PATIENCE = 15
REFIT_SEED = 2266358626              # frozen PRIMARY production child seed (spec 16)

# frozen PRIMARY row counts (STEP5G_FINAL_PIPELINE_FREEZE.json, x_partition task)
PRIMARY_TRAIN_ROWS = 241748
PRIMARY_VAL_ROWS = 51802

BOOTSTRAP_REPS = 1000                # spec 43
BOOTSTRAP_SEED = 20260913

# frozen zero-shot numbers, carried for cross-checking only - never recomputed and
# never substituted for a recovery-subset value (spec 34)
ZERO_SHOT_FULL_MEAN_TVD = 0.8643856186628641
ZERO_SHOT_FULL_TOPOLOGY_ACC = 0.07859090909090909
ZERO_SHOT_FULL_TRIVIAL_TVD = 0.5957669021263107
SECONDARY_FULL_MEAN_TVD = 0.7234409099
NBNI3_RAW_NAME = "NBNI3"
NBNI3_N_STATES_FROZEN = 8579
NBNI3_TEMPERATURE_RANGE = (300.0, 600.0)

METHODS = ["M_REFIT", "R0", "R1", "R2"]
CONTROL_IDS = {"R0": "R0_ANCHOR_TRIVIAL_PER_T",
               "R1": "R1_GLOBAL_MEAN_SHIFT",
               "R2": "R2_PER_TEMPERATURE_MEAN_SHIFT"}


# ---------------------------------------------------------------- helpers -- #
def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if v != v else v
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jd)
    return path


def load_g_common():
    """Import the FROZEN STEP5G g_common by path.

    The R0/R1/R2 estimators and the simplex projection are used exactly as they were
    preregistered, by executing the preregistered code - not by re-typing it here.
    Re-typing is how a control quietly stops being the control it claims to be."""
    spec = importlib.util.spec_from_file_location("frozen_g_common", G_COMMON)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["frozen_g_common"] = mod
    spec.loader.exec_module(mod)
    for name in ("recovery_apply", "project_simplex", "tvd", "topology_metrics",
                 "fewshot_split_role"):
        if not hasattr(mod, name):
            raise SystemExit("FATAL: frozen g_common.py has no %s; the STEP5G code "
                             "snapshot is not the one STEP5I was written against."
                             % name)
    return mod


def feature_schema():
    """The frozen 43-feature production schema, in order, read from the STEP5G freeze.

    Deriving it by excluding metadata columns would be wrong: Temperature_C is BOTH a
    bookkeeping column and one of the 43 inputs, so an exclusion list silently drops a
    real feature and trains on 42."""
    df = pd.read_csv(G_SCHEMA_CSV)
    feats = list(df.sort_values("order").feature) if "order" in df.columns \
        else list(df.feature)
    if len(feats) != N_FEATURES or len(set(feats)) != N_FEATURES:
        raise SystemExit("FATAL: frozen feature schema has %d entries (%d unique), "
                         "expected %d." % (len(feats), len(set(feats)), N_FEATURES))
    return feats


def fewshot_prereg():
    return json.load(open(G_FEWSHOT_JSON, encoding="utf-8"))


def assert_protocol_matches_prereg(pre):
    """The STEP5I brief and the frozen STEP5G preregistration must agree. If they
    ever diverge, the preregistration wins and this stage stops - a brief cannot
    silently retune a frozen protocol."""
    r = pre["refit"]
    bad = []
    for label, got, want in (("learning_rate", r["learning_rate"], REFIT_LR),
                             ("batch_size", r["batch_size"], REFIT_BATCH),
                             ("max_epochs", r["max_epochs"], REFIT_EPOCHS),
                             ("patience", r["patience"], REFIT_PATIENCE),
                             ("seed", r["seed"], REFIT_SEED),
                             ("budgets", pre["budgets_compositions"], BUDGETS),
                             ("replicates", pre["replicates"], len(REPLICATES)),
                             ("nested", pre["budgets_nested"], True),
                             ("n_calibration", pre["split"]["n_calibration"],
                              N_CALIBRATION),
                             ("n_recovery_evaluation",
                              pre["split"]["n_recovery_evaluation"], N_RECOVERY_EVAL),
                             ("primary_estimator", pre["primary_estimator"],
                              "PRIMARY_FORMULATION_REFIT")):
        if got != want:
            bad.append("%s: preregistration says %r, STEP5I expects %r"
                       % (label, got, want))
    if r.get("retuning") is not False:
        bad.append("preregistration does not assert retuning=False")
    if bad:
        raise SystemExit("FATAL: STEP5I protocol disagrees with the frozen "
                         "preregistration:\n  " + "\n  ".join(bad))
    return True


def random_sequences():
    """The three frozen nested 32-composition sequences, read from the frozen CSV and
    cross-checked against the preregistration JSON."""
    df = pd.read_csv(G_SEQ_CSV)
    pre = fewshot_prereg()
    out = {}
    for r in REPLICATES:
        sub = df[df.replicate == r].sort_values("rank")
        seq = list(sub.composition_id)
        if list(sub["rank"]) != list(range(1, 33)) or len(set(seq)) != 32:
            raise SystemExit("FATAL: replicate %d sequence is not 32 unique ranked "
                             "IDs." % r)
        if seq != list(pre["sequences"][str(r)]):
            raise SystemExit("FATAL: replicate %d sequence in the CSV differs from the "
                             "preregistration JSON." % r)
        out[r] = seq
    return out


def maximin_sequence():
    df = pd.read_csv(HA_MAXIMIN_CSV).sort_values("rank")
    seq = list(df.composition_id)
    jj = json.load(open(HA_MAXIMIN_JSON, encoding="utf-8"))
    if seq != list(jj["sequence"]) or len(set(seq)) != 32:
        raise SystemExit("FATAL: maximin sequence CSV and freeze JSON disagree.")
    return seq


def anchors(seq, budget):
    """Nested budget rule (spec 8): the first b IDs of that sequence. Nothing is
    re-drawn, re-ordered or re-selected."""
    if budget > len(seq):
        raise SystemExit("FATAL: budget %d exceeds the frozen 32-composition sequence."
                         % budget)
    return list(seq[:budget])


def load_frozen_ni():
    """Targets, frozen PRIMARY zero-shot predictions and Ni features, joined on
    state_id. Read-only; nothing here is regenerated (spec 4, 20)."""
    t = pd.read_parquet(HB_TARGETS)
    p = pd.read_parquet(HA_PRED)
    f = pd.read_parquet(HA_FEATURES)
    for name, d, n in (("targets", t, 22000), ("predictions", p, 22000),
                       ("features", f, 22000)):
        if len(d) != n:
            raise SystemExit("FATAL: frozen %s has %d rows, expected %d."
                             % (name, len(d), n))
        if not d.state_id.is_unique:
            raise SystemExit("FATAL: frozen %s has duplicate state_id." % name)
    keep_p = ["state_id"] + RAW_COLS + PRED_COLS + ["pred_topology", "simplex_fallback"]
    d = t.merge(p[keep_p], on="state_id", validate="one_to_one")
    feat_cols = feature_schema()
    missing = [c for c in feat_cols if c not in f.columns]
    if missing:
        raise SystemExit("FATAL: Ni feature table is missing frozen production "
                         "features: %s" % missing)
    # Temperature_C is a feature AND already a column of the target frame; take the
    # feature copy from the feature table and keep exactly one of it
    dup = [c for c in feat_cols if c in d.columns]
    d = d.drop(columns=dup).merge(f[["state_id"] + feat_cols], on="state_id",
                                  validate="one_to_one")
    d["nbni3_present"] = d.raw_phase_names.astype(str).str.upper().str.contains(
        NBNI3_RAW_NAME).astype(int)
    d["t_regime"] = np.where(d.Temperature_C.isin(LOW_T), "LOW_T", "HIGH_T")
    return d, feat_cols


def frozen_fallback_row():
    """The frozen PRIMARY training-mean simplex, taken from the pipeline freeze."""
    fz = json.load(open(G_FREEZE_JSON, encoding="utf-8"))
    for m in fz["models"].values():
        if m.get("tag") == "PRIMARY" or m.get("model_id", "").endswith("PRIMARY"):
            return np.asarray(m["fallback_row"], dtype=float)
    raise SystemExit("FATAL: no PRIMARY fallback row in the pipeline freeze.")


def metrics_block(y_true, y_pred, trivial_row, zero_shot_tvd=None, gc=None):
    """The preregistered metric set (spec 30). trivial_row is the FROZEN PRIMARY
    training-mean simplex; relative reduction is always against a zero-shot value
    measured on the SAME subset (spec 31, 34)."""
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    v = 0.5 * np.abs(y_pred - y_true).sum(axis=1)
    triv = np.repeat(np.asarray(trivial_row, float)[None, :], len(y_true), axis=0)
    tv = 0.5 * np.abs(triv - y_true).sum(axis=1)
    t_arg, p_arg = y_true.argmax(axis=1), y_pred.argmax(axis=1)
    out = {"n_states": int(len(y_true)),
           "mean_tvd": float(v.mean()), "median_tvd": float(np.median(v)),
           "p95_tvd": float(np.quantile(v, 0.95)),
           "topology_accuracy": float((t_arg == p_arg).mean()),
           "topology_macro_f1": float(f1_score(t_arg, p_arg, labels=list(range(5)),
                                               average="macro", zero_division=0)),
           "trivial_tvd_same_subset": float(tv.mean()),
           "skill_vs_trivial": float(tv.mean() - v.mean())}
    if zero_shot_tvd is not None and zero_shot_tvd > 0:
        out["zero_shot_tvd_same_subset"] = float(zero_shot_tvd)
        out["relative_tvd_reduction_vs_zero_shot"] = float(
            1.0 - out["mean_tvd"] / zero_shot_tvd)
    return out


def per_state_tvd(y_true, y_pred):
    return 0.5 * np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float)).sum(axis=1)


def cell_tag(kind, replicate, budget):
    return ("RANDOM_R%d_B%d" % (replicate, budget) if kind == "RANDOM"
            else "MAXIMIN_B%d" % budget)
