# -*- coding: utf-8 -*-
"""
j_common.py
STEP5J  -  FINAL EVIDENCE FREEZE + MANUSCRIPT-GRADE CONSOLIDATION
        +  CLAIM AUDIT + FIGURE/TABLE FREEZE + REPRODUCIBILITY PACKAGE

STEP5J runs no experiment. It reads frozen files, recomputes every manuscript-facing
number from them, and refuses to invent anything that is not already on disk. Where a
value cannot be derived from a frozen file it is written as NOT_AVAILABLE or
REQUIRES_NEW_EXPERIMENT rather than quietly computed (spec 35).

Evidence levels (spec 4) are kept apart everywhere - table, figure, claim and draft:

  LEVEL 1  PROSPECTIVE PRIMARY        STEP5H-B PRIMARY Ni zero-shot, and nothing else
  LEVEL 2  PROSPECTIVE SECONDARY      STEP5H-B FULL11 SECONDARY sensitivity
  LEVEL 3  POST-BLIND PREREGISTERED   STEP5I random few-shot, maximin, R0/R1/R2
  LEVEL 4  POST-BLIND DESCRIPTIVE     NBNI3, topology, temperature, coverage analyses

The retrospective LOEO numbers (0.752644 / 0.710332) are model-family context and are
never presented as scores of the frozen STEP5G production checkpoints (spec 18).
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths -- #
MLROOT = r"E:\Cu_CALPHAD_ML"
FULL_V1 = os.path.join(MLROOT, "03_full_CALPHAD", "FULL_V1")
STEP5A = os.path.join(MLROOT, "04_ML", "STEP5A_JMST_ML_FOUNDATION_V2")
STEP5B = os.path.join(MLROOT, "05_OOD", "STEP5B_DOMAIN_B_OOD_V1")
STEP5C = os.path.join(MLROOT, "05_OOD", "STEP5C_6X_LOEO_V1")
STEP5D = os.path.join(MLROOT, "05_OOD", "STEP5D_FAILURE_DIAGNOSIS_V1")
STEP5E = os.path.join(MLROOT, "05_OOD", "STEP5E_XPART_REMEDY_V1")
STEP5F = os.path.join(MLROOT, "06_AUGMENTATION", "STEP5F_FULL_V1")
FINAL = os.path.join(MLROOT, "07_FINAL_BLIND")
STEP5G = os.path.join(FINAL, "STEP5G_PIPELINE_FREEZE_V1")
STEP5HA = os.path.join(FINAL, "STEP5H_A_PREDICTION_FREEZE_V1")
STEP5HB = os.path.join(FINAL, "STEP5H_B_NI_BLIND_REVEAL_V1")
STEP5I = os.path.join(FINAL, "STEP5I_POSTBLIND_FEWSHOT_RECOVERY_V1")
ROOT = os.path.join(MLROOT, "08_FINAL_MANUSCRIPT", "STEP5J_FINAL_EVIDENCE_FREEZE_V1")

D_FREEZE = os.path.join(ROOT, "00_freeze")
D_TABLES = os.path.join(ROOT, "01_final_tables")
D_FIGS = os.path.join(ROOT, "02_final_figures")
D_RESULTS = os.path.join(ROOT, "03_results_draft")
D_DISC = os.path.join(ROOT, "04_discussion_draft")
D_LIT = os.path.join(ROOT, "05_literature_positioning")
D_CLAIM = os.path.join(ROOT, "06_claim_audit")
D_PROV = os.path.join(ROOT, "07_provenance")
D_REPRO = os.path.join(ROOT, "08_reproducibility")
D_QA = os.path.join(ROOT, "09_qa")
D_MANIFEST = os.path.join(ROOT, "10_manifest")
D_CODE = os.path.join(ROOT, "code_snapshot")
ALL_DIRS = [D_FREEZE, D_TABLES, D_FIGS, D_RESULTS, D_DISC, D_LIT, D_CLAIM, D_PROV,
            D_REPRO, D_QA, D_MANIFEST, D_CODE]

NOT_AVAILABLE = "NOT_AVAILABLE"
REQUIRES_NEW_EXPERIMENT = "REQUIRES_NEW_EXPERIMENT"


def P(*a):
    return os.path.join(*a)


# ------------------------------------------ frozen source registry (spec 3) - #
# key -> (path, stage, role, expected_sha256 or None)
# A None hash means "record it, do not assert it": the hash is published by STEP5J
# rather than checked against a value the earlier stage never recorded. Asserting an
# invented expectation would be worse than recording the truth.
SOURCES = {
    "G_PRIMARY_CKPT": (P(STEP5G, "02_primary_model", "STEP5G_PRIMARY_MODEL_CHECKPOINT.pt"),
                       "STEP5G", "frozen PRIMARY production checkpoint",
                       "f89e4ddc2f367a572a16a29062faa3c3d580fc8aae73a8e6bfaa28a8bcd86d0f"),
    "G_PRIMARY_SCALER": (P(STEP5G, "02_primary_model", "STEP5G_PRIMARY_SCALER.joblib"),
                         "STEP5G", "frozen PRIMARY scaler",
                         "c2619e4aed54304a9ed895b98164b0205421310f613f50346c249de41d7e3475"),
    "G_SECONDARY_CKPT": (P(STEP5G, "03_secondary_model",
                           "STEP5G_SECONDARY_MODEL_CHECKPOINT.pt"),
                         "STEP5G", "frozen SECONDARY production checkpoint",
                         "a685b2138f114f3274f3b789753b28971484b797ca7d375a9354021242d55719"),
    "G_FREEZE_JSON": (P(STEP5G, "01_pipeline_freeze", "STEP5G_FINAL_PIPELINE_FREEZE.json"),
                      "STEP5G", "production pipeline freeze", None),
    "G_SCHEMA_CSV": (P(STEP5G, "01_pipeline_freeze", "STEP5G_FEATURE_SCHEMA.csv"),
                     "STEP5G", "frozen 43-feature schema", None),
    "G_SEQ_CSV": (P(STEP5G, "06_postblind_fewshot_prereg",
                    "STEP5G_NESTED_RANDOM_SEQUENCES.csv"),
                  "STEP5G", "frozen nested random anchor sequences",
                  "9c9389a100ee064bc8a52ab4a4d8cedb7558c9b95462cca0fa6736da92d6507e"),
    "G_FEWSHOT_JSON": (P(STEP5G, "06_postblind_fewshot_prereg",
                         "STEP5G_POSTBLIND_FEWSHOT_PREREGISTRATION.json"),
                       "STEP5G", "post-blind few-shot preregistration", None),
    "G_NI_COMPS": (P(STEP5G, "04_ni_composition_prereg", "STEP5G_NI_BLIND_COMPOSITIONS.csv"),
                   "STEP5G", "2000 symbolic Ni blind compositions",
                   "7680b5a1ba5a2409f2b23900d770dd4939df0cf76bd2f1d9c1a2c2c5cce0030b"),
    "HA_PRIMARY_PRED": (P(STEP5HA, "04_primary_predictions",
                          "STEP5H_A_PRIMARY_NI_PREDICTIONS.parquet"),
                        "STEP5H-A", "frozen PRIMARY Ni zero-shot predictions",
                        "100b40ed67c396cb296d53db80e7fef7cb94c0c8223873f5a13c01af28f0f488"),
    "HA_SECONDARY_PRED": (P(STEP5HA, "05_secondary_predictions",
                            "STEP5H_A_SECONDARY_NI_PREDICTIONS.parquet"),
                          "STEP5H-A", "frozen SECONDARY Ni zero-shot predictions",
                          "a6c0fc4a98a05a4c53704d85b5caf91a5477577789a69d8dc24f6976f98a72ad"),
    "HA_RANGE_DIAG": (P(STEP5HA, "03_ood_diagnostics",
                        "STEP5H_A_NI_FEATURE_RANGE_DIAGNOSTICS.csv"),
                      "STEP5H-A", "pre-reveal Ni feature-range diagnostics",
                      "7ccaca20596cbf25d2acf01f24444ffacceb444e8dc5f036ea069aab77b483a3"),
    "HA_OOD_FREEZE": (P(STEP5HA, "03_ood_diagnostics",
                        "STEP5H_A_OOD_DIAGNOSTICS_FREEZE.json"),
                      "STEP5H-A", "pre-reveal OOD diagnostic freeze", None),
    "HA_SUMMARY": (P(STEP5HA, "STEP5H_A_SUMMARY.json"), "STEP5H-A", "stage summary", None),
    "HA_MAXIMIN": (P(STEP5HA, "07_maximin_freeze", "STEP5H_A_MAXIMIN_SEQUENCE.csv"),
                   "STEP5H-A", "frozen maximin anchor sequence",
                   "6704c7e9d71f64dc8395c55b0beed239f29f99f6e7e9bfa510ab8f4d64221fda"),
    "HB_TARGETS": (P(STEP5HB, "03_processed_targets", "STEP5H_B_NI_TARGETS.parquet"),
                   "STEP5H-B", "frozen Ni CALPHAD targets",
                   "3e1f4ba2cf6e47120edc4e5150059541ac3d7713e850ac867f411c58126858e3"),
    "HB_TARGET_FREEZE": (P(STEP5HB, "03_processed_targets",
                           "STEP5H_B_NI_TARGET_FREEZE.json"),
                         "STEP5H-B", "target freeze record", None),
    "HB_PRIMARY_SCORE": (P(STEP5HB, "06_primary_score", "STEP5H_B_PRIMARY_SCORE_FREEZE.json"),
                         "STEP5H-B", "frozen PRIMARY blind score", None),
    "HB_PRIMARY_METRICS": (P(STEP5HB, "06_primary_score",
                             "STEP5H_B_PRIMARY_BLIND_METRICS.json"),
                           "STEP5H-B", "PRIMARY blind metric block", None),
    "HB_SECONDARY_SCORE": (P(STEP5HB, "07_secondary_score",
                             "STEP5H_B_SECONDARY_SCORE_FREEZE.json"),
                           "STEP5H-B", "frozen SECONDARY blind score", None),
    "HB_SECONDARY_METRICS": (P(STEP5HB, "07_secondary_score",
                               "STEP5H_B_SECONDARY_BLIND_METRICS.json"),
                             "STEP5H-B", "SECONDARY blind metric block", None),
    "HB_PHASE_VOCAB": (P(STEP5HB, "04_phase_vocab", "STEP5H_B_PHASE_VOCABULARY.csv"),
                       "STEP5H-B", "Ni phase vocabulary incl. NBNI3", None),
    "HB_PER_TEMP": (P(STEP5HB, "06_primary_score", "STEP5H_B_PRIMARY_PER_TEMPERATURE.csv"),
                    "STEP5H-B", "PRIMARY blind error by temperature", None),
    "HB_SUMMARY": (P(STEP5HB, "STEP5H_B_SUMMARY.json"), "STEP5H-B", "stage summary", None),
    "I_BASELINES": (P(STEP5I, "05_metrics", "STEP5I_RECOVERY_BASELINES.json"),
                    "STEP5I", "M_ZERO / M_TRIVIAL on the recovery subset", None),
    "I_THRESHOLDS": (P(STEP5I, "05_metrics", "STEP5I_BUDGET_THRESHOLDS.json"),
                     "STEP5I", "B_skill / B_recovery25", None),
    "I_RANDOM_AGG": (P(STEP5I, "05_metrics", "STEP5I_RANDOM_FEWSHOT_METRICS.csv"),
                     "STEP5I", "across-replicate random few-shot metrics", None),
    "I_RANDOM_REP": (P(STEP5I, "05_metrics", "STEP5I_RANDOM_FEWSHOT_PER_REPLICATE.csv"),
                     "STEP5I", "per-replicate random few-shot metrics", None),
    "I_MAXIMIN": (P(STEP5I, "05_metrics", "STEP5I_MAXIMIN_FEWSHOT_METRICS.csv"),
                  "STEP5I", "maximin few-shot metrics (secondary)", None),
    "I_CONTROLS": (P(STEP5I, "04_R0_R1_R2_controls", "STEP5I_CONTROL_METRICS.csv"),
                   "STEP5I", "R0 / R1 / R2 anchor-only controls", None),
    "I_CMP": (P(STEP5I, "05_metrics", "STEP5I_MODEL_VS_CONTROL_COMPARISON.csv"),
              "STEP5I", "model versus control, per cell", None),
    "I_TOPO": (P(STEP5I, "06_topology_diagnostics", "STEP5I_TRUE_TOPOLOGY_STRATIFIED.csv"),
               "STEP5I", "recovery by true topology", None),
    "I_NBNI3": (P(STEP5I, "07_NBNI3_diagnostics", "STEP5I_NBNI3_STRATIFIED.csv"),
                "STEP5I", "recovery by NBNI3 status", None),
    "I_TEMP": (P(STEP5I, "06_topology_diagnostics", "STEP5I_TEMPERATURE_REGIME_METRICS.csv"),
               "STEP5I", "recovery by temperature regime", None),
    "I_COVERAGE": (P(STEP5I, "06_topology_diagnostics", "STEP5I_ANCHOR_TOPOLOGY_COVERAGE.csv"),
                   "STEP5I", "what the frozen anchor sequences contained", None),
    "I_MECH": (P(STEP5I, "07_NBNI3_diagnostics",
                 "STEP5I_COVERAGE_MECHANISM_CORRELATIONS.csv"),
               "STEP5I", "coverage-recovery rank correlations", None),
    "I_BOOT": (P(STEP5I, "08_bootstrap", "STEP5I_PAIRED_BOOTSTRAP.csv"),
               "STEP5I", "post-blind exploratory paired bootstrap", None),
    "I_CKPT_MANIFEST": (P(STEP5I, "11_manifest", "STEP5I_RECOVERY_CHECKPOINT_MANIFEST.csv"),
                        "STEP5I", "24 recovery checkpoints with hashes", None),
    "I_SEAL": (P(STEP5I, "11_manifest", "STEP5I_FINAL_SEAL_HASHES.csv"),
               "STEP5I", "authoritative post-run file index", None),
    "I_SUMMARY": (P(STEP5I, "STEP5I_SUMMARY.json"), "STEP5I", "stage summary", None),
    "I_ADDENDUM": (P(STEP5I, "STEP5I_CODEX_REVIEW_ADDENDUM.md"), "STEP5I",
                   "independent review addendum and its four caveats", None),
    "B_IID_VS_OOD": (P(STEP5B, "04_metrics", "IID_VS_OOD_WITH_CI.csv"),
                     "STEP5B", "IID vs composition-domain OOD with CIs", None),
    "B_OOD_METRICS": (P(STEP5B, "04_metrics", "DOMAIN_B_OOD_METRICS.csv"),
                      "STEP5B", "Domain-B OOD metrics", None),
    "B_COUNTS": (P(STEP5B, "07_qa", "DOMAIN_B_COUNTS.json"), "STEP5B",
                 "Domain-B state counts", None),
    "C_LADDER": (P(STEP5C, "06_metrics", "IID_VS_COMPOSITION_OOD_VS_LOEO.csv"),
                 "STEP5C", "IID / composition-OOD / LOEO ladder", None),
    "C_PER_ELEMENT": (P(STEP5C, "06_metrics", "LOEO_PER_ELEMENT_HEADLINE.csv"),
                      "STEP5C", "per-held-out-element LOEO headline", None),
    "C_TOPOLOGY": (P(STEP5C, "06_metrics", "LOEO_TOPOLOGY_METRICS.csv"),
                   "STEP5C", "LOEO topology metrics", None),
    "E_PIPELINE": (P(STEP5E, "11_manifest", "STEP5E_RECOMMENDED_ZERO_SHOT_PIPELINE.json"),
                   "STEP5E", "zero-shot pipeline selection among Z0-Z5", None),
    "F_REPORT": (P(STEP5F, "STEP5F_FULL_REPORT.md"), "STEP5F",
                 "chemical-identity augmentation report", None),
    "F_AUG_METRICS": (P(STEP5F, "05_augmented_ML", "STEP5F_AUGMENTED_LOEO_METRICS.csv"),
                      "STEP5F", "augmented LOEO per fold", None),
    "F_AUG_SUMMARY": (P(STEP5F, "05_augmented_ML", "STEP5F_AUGMENTED_LOEO_SUMMARY.json"),
                      "STEP5F", "augmented LOEO summary", None),
    "V1_TOPOLOGY": (P(FULL_V1, "04_qa", "FULL_V1_TOPOLOGY_STATS.csv"), "FULL_V1",
                    "dominant X-partition topology per identity and temperature", None),
    "V1_DOMAIN": (P(FULL_V1, "04_qa", "FULL_V1_DOMAIN_SUMMARY.csv"), "FULL_V1",
                  "per-domain CALPHAD summary", None),
    "V1_PHASES": (P(FULL_V1, "04_qa", "FULL_V1_RAW_PHASE_INVENTORY.csv"), "FULL_V1",
                  "raw phase inventory", None),
    "A_DEVFREEZE": (P(STEP5A, "09_manifest", "DEVELOPMENT_FREEZE_V1.json"), "STEP5A",
                    "model / hyperparameter / metric freeze", None),
    "A_FEATLIST": (P(STEP5A, "02_features", "SELECTED_FEATURES_F1_F2_F3_F4.txt"),
                   "STEP5A", "frozen 43-feature list",
                   "f4ddf4a2587e33a7d441656110c589208d9d860095198dd4b276df64f8a2fb0c"),
}

# ------------------------------------------------------------- constants --- #
FAMILIES_ORDER = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
ORIGINAL_SIX = ["ZR", "TI", "CO", "SI", "AG", "MO"]
FULL5_ADDED = ["BE", "SN", "MN", "LA", "P"]
BUDGETS = [1, 2, 4, 8, 16, 32]
REPLICATES = [1, 2, 3]
TEMPERATURES_C = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]
LOW_T = [300, 350, 400, 450, 500, 600]
HIGH_T = [700, 800, 900, 950, 1000]
N_RECOVERY_EVAL = 1400
N_RECOVERY_STATES = 15400
N_BLIND_STATES = 22000

EVIDENCE_LEVELS = {
    1: "PROSPECTIVE PRIMARY EVIDENCE (STEP5H-B PRIMARY zero-shot only)",
    2: "PROSPECTIVE SECONDARY SENSITIVITY (STEP5H-B FULL11 SECONDARY only)",
    3: "POST-BLIND PREREGISTERED RECOVERY (STEP5I random few-shot, maximin, R0/R1/R2)",
    4: "POST-BLIND DESCRIPTIVE MECHANISM (NBNI3, topology, temperature, coverage)",
}

# spec 13 - phrases that must never appear outside an explicit disclaimer block
FORBIDDEN_CLAIMS = [
    "accurate zero-shot unseen-element model",
    "universal ood generalization",
    "universal ood generalisation",
    "full11 solves chemical ood",
    "distance predicts failure",
    "nbni3 causes all error",
    "solves chemical ood",
    "successful zero-shot",
    "zero-shot success",
    "predicted strongest alloy",
    "strongest alloy",
    "kinetic precipitation driving force",
    "jmst success threshold",
    "prospective blind validation",
]
# "one point is sufficient" is allowed ONLY with the state-count clarification
ONE_POINT_PATTERN = re.compile(r"one\s+(point|composition)\s+is\s+sufficient", re.I)
ONE_POINT_CLARIFIER = "11 CALPHAD"

DISCLAIMER_OPEN = "<!-- DISCLAIMER_OPEN -->"
DISCLAIMER_CLOSE = "<!-- DISCLAIMER_CLOSE -->"

# retrospective context that must never be called a checkpoint score (spec 18)
LOEO_CONTEXT = {
    "original_six_Z1_MLP_DIRECT_mean_tvd": 0.752644,
    "full5_augmented_mean_tvd": 0.710332,
    "caveat": ("model-family retrospective leave-one-element-out context; NOT a score "
               "of the frozen STEP5G production checkpoints, which were trained on all "
               "six identities and never scored by LOEO"),
}


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


def load_json(key_or_path):
    p = SOURCES[key_or_path][0] if key_or_path in SOURCES else key_or_path
    return json.load(open(p, encoding="utf-8"))


def load_csv(key_or_path, **kw):
    p = SOURCES[key_or_path][0] if key_or_path in SOURCES else key_or_path
    return pd.read_csv(p, **kw)


def src(key):
    return SOURCES[key][0]


def cite(key):
    """A citation string a table or draft can carry: stage + file + short hash."""
    path, stage, role, _ = SOURCES[key]
    h = sha256_file(path) if os.path.exists(path) else "MISSING"
    return "%s :: %s :: %s" % (stage, os.path.basename(path), h[:16])


def scan_forbidden(path):
    """Text inside a marked disclaimer block is exempt: a manuscript must be able to
    say which claims it is NOT making. Everything else is scanned."""
    txt = open(path, encoding="utf-8").read()
    body = re.sub(re.escape(DISCLAIMER_OPEN) + r".*?" + re.escape(DISCLAIMER_CLOSE),
                  " ", txt, flags=re.S)
    # collapse whitespace so a phrase broken across a line wrap is still detected
    body = re.sub(r"\s+", " ", body)
    low = body.lower()
    hits = [p for p in FORBIDDEN_CLAIMS if p in low]
    for m in ONE_POINT_PATTERN.finditer(body):
        window = body[max(0, m.start() - 400):m.end() + 400]
        if ONE_POINT_CLARIFIER not in window:
            hits.append("'one point/composition is sufficient' without the "
                        "'11 CALPHAD temperature states' clarification")
    return hits


def to_md(df):
    """Render a DataFrame as a GitHub-flavoured markdown table without depending on
    `tabulate`, which is not guaranteed to be installed on the analysis machine."""
    cols = [str(c) for c in df.columns]
    def cell(v):
        if v is None or (isinstance(v, float) and v != v):
            return ""
        if isinstance(v, float):
            return ("%.6f" % v) if abs(v) >= 1e-4 or v == 0 else ("%.3e" % v)
        return str(v).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(cell(r[c]) for c in df.columns) + " |")
    return "\n".join(out)


def flat(path):
    """File text with runs of whitespace collapsed, so a phrase search is not defeated
    by a line wrap."""
    return re.sub(r"\s+", " ", open(path, encoding="utf-8").read())


def value(number, source_key, definition, level, note=None):
    """Every manuscript-facing number is carried with where it came from, what it
    means, and which evidence level it belongs to. A bare float is not admissible."""
    d = {"value": number, "source_file": src(source_key),
         "source_stage": SOURCES[source_key][1],
         "source_sha256": sha256_file(src(source_key))
         if os.path.exists(src(source_key)) else "MISSING",
         "metric_definition": definition, "evidence_level": level}
    if note:
        d["note"] = note
    return d


def unavailable(reason, requires_experiment=False):
    return {"value": REQUIRES_NEW_EXPERIMENT if requires_experiment else NOT_AVAILABLE,
            "reason": reason, "evidence_level": None}


def close(a, b, tol=5e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


# values quoted in the STEP5J brief; every one is RECOMPUTED from frozen files and the
# recomputed number is what ships. These are kept only so a drift can be reported
# rather than silently absorbed (spec 15).
BRIEF_QUOTED = {
    "primary_blind_mean_tvd": 0.8643856187,
    "primary_blind_topology_accuracy": 0.0785909,
    "primary_blind_trivial_tvd": 0.5957669021,
    "primary_blind_skill": -0.2686187165,
    "secondary_blind_mean_tvd": 0.7234409099,
    "secondary_blind_topology_accuracy": 0.2406818,
    "recovery_subset_zero_shot_tvd": 0.8632602865,
    "recovery_subset_trivial_tvd": 0.5960189307,
    "refit_b1": 0.2481623701, "refit_b2": 0.1656707301, "refit_b4": 0.1235011339,
    "refit_b8": 0.0507380675, "refit_b16": 0.0358137552, "refit_b32": 0.0271879848,
    "r0_b1": 0.136716, "r0_b2": 0.111797, "r0_b4": 0.097504,
    "r0_b8": 0.090807, "r0_b16": 0.076495, "r0_b32": 0.079531,
    "r2_b1": 0.191315, "r2_b2": 0.152236, "r2_b4": 0.144340,
    "r2_b8": 0.139418, "r2_b16": 0.130683, "r2_b32": 0.130985,
    "topo_b1": 0.722446, "topo_b2": 0.915844, "topo_b4": 0.943312,
    "topo_b8": 0.974978, "topo_b16": 0.978247, "topo_b32": 0.986277,
    "fcc_zero": 0.839291, "fcc_b32": 0.025630,
    "other_zero": 0.923644, "other_b32": 0.031113,
    "nbni3_present_zero": 0.922072, "nbni3_present_b32": 0.036934,
    "nbni3_absent_zero": 0.825526, "nbni3_absent_b32": 0.020935,
    "lowT_zero": 0.924387, "lowT_b32": 0.036026,
    "highT_zero": 0.789909, "highT_b32": 0.016582,
    "coverage_rho": 0.7597,
}
