# -*- coding: utf-8 -*-
"""
r3_common.py
STEP5K-R3 shared paths, constants and guards.

R3 is a POST-FREEZE, NON-Ni, source-grounded experimental-consistency stage. Nothing
in it trains, refits, reselects or reinterprets anything. The 471 protected artifacts
recorded in 00_config/protected_before.json are re-hashed by every stage that writes
output, and any stage aborts if one of them has moved.
"""
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MLROOT = r"E:\Cu_CALPHAD_ML"
ROOT = os.path.join(MLROOT, "08_FINAL_MANUSCRIPT",
                    "STEP5K_R3_TARGETED_EXTERNAL_VALIDATION")

D_CONFIG = os.path.join(ROOT, "00_config")
D_PDF = os.path.join(ROOT, "01_source_pdfs")
D_EXTRACT = os.path.join(ROOT, "02_source_extraction")
D_FREEZE = os.path.join(ROOT, "03_inclusion_freeze")
D_CONV = os.path.join(ROOT, "04_composition_conversion")
D_CROSS = os.path.join(ROOT, "05_phase_crosswalk")
D_REQ = os.path.join(ROOT, "06_calphad_requests")
D_RAW = os.path.join(ROOT, "07_calphad_raw")
D_PROC = os.path.join(ROOT, "08_calphad_processed")
D_CMP = os.path.join(ROOT, "09_condition_comparison")
D_PHCOMP = os.path.join(ROOT, "10_phase_composition")
D_EVID = os.path.join(ROOT, "11_system_evidence")
D_FIG = os.path.join(ROOT, "12_figures")
D_INSERT = os.path.join(ROOT, "13_manuscript_insertions")
D_QA = os.path.join(ROOT, "14_qa")
D_MAN = os.path.join(ROOT, "15_manifest")
D_LOGS = os.path.join(ROOT, "16_logs")
D_CODE = os.path.join(ROOT, "code_snapshot")
ALL_DIRS = [D_CONFIG, D_PDF, D_EXTRACT, D_FREEZE, D_CONV, D_CROSS, D_REQ, D_RAW,
            D_PROC, D_CMP, D_PHCOMP, D_EVID, D_FIG, D_INSERT, D_QA, D_MAN, D_LOGS,
            D_CODE]

FAMILIES = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
PRESENCE = 1e-4                      # frozen FULL_V1 phase-presence threshold
ALLOWED_ELEMENTS = {"CU", "CR", "NB", "AG", "MO", "CO", "SI", "TI"}
NI_TOKEN = re.compile(r"(?<![A-Za-z])NI(?![A-Za-z])")

SYSTEMS = ["Cu-Cr-Nb", "Cu-Cr-Ag", "Cu-Cr-Mo", "Cu-Cr-Co", "Cu-Cr-Si", "Cu-Cr-Ti"]
SYSTEM_ROWS = SYSTEMS + ["GRCop (qualitative context)"]

VERDICTS = ["STRONGLY_CONSISTENT", "CONSISTENT", "PARTIALLY_CONSISTENT",
            "DISCREPANT", "NOT_COMPARABLE", "CALPHAD_FAILED"]
EVIDENCE_STRENGTH = ["STRONG", "MODERATE", "LIMITED", "NONE"]

# ----------------------------------------------------------------------------- #
# Comparison rules. These are constants in the shipped source, authored before any
# R3 equilibrium existed (the engine had produced nothing when they were written),
# and stage 30 seals them to STEP5K_R3_COMPARISON_RULES_FREEZE.json together with
# this file's sha256 so the seal cannot drift from the code that applied it.
# ----------------------------------------------------------------------------- #
# Two matched phases are chemically compatible when no element differs by more than
# this many at.%. A single generous, conventional tolerance for EPMA/EDS against a
# calculated equilibrium - deliberately not per-system, and never tuned to results.
PHASE_CHEM_TOL_AT_PCT = 5.0
# A condition's phase multiplicity is judged only where the paper itself reports more
# than one instance of a family (two FCC in Cu-Cr-Ag, two BCC in Cu-Cr-Mo).
MULTIPLICITY_FAMILIES = ["FCC", "BCC"]


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
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jd, ensure_ascii=False)
    return path


def load_json(path):
    return json.load(open(path, encoding="utf-8"))


# ------------------------------------------------------------------ guards --- #
def protected_baseline():
    return load_json(os.path.join(D_CONFIG, "protected_before.json"))


def verify_protected(strict=True):
    """Re-hash every protected artifact. Returns (n_checked, [changed], [missing])."""
    changed, missing = [], []
    rows = protected_baseline()
    for r in rows:
        p = r["path"]
        if not os.path.exists(p):
            missing.append(p)
            continue
        if sha256_file(p) != r["sha256"]:
            changed.append(p)
    if strict and (changed or missing):
        raise SystemExit(
            "FATAL: the frozen baseline moved during R3. changed=%s missing=%s\n"
            "R3 must not alter any frozen Ni, model, OOD, recovery or screening "
            "artifact. Stopping before anything else is written."
            % (changed[:5], missing[:5]))
    return len(rows), changed, missing


def assert_no_ni(paths, allow_substrings=()):
    """R3 prose legitimately explains that the Cu-Cr-Ni half of source 1 was ignored.
    Those sentences are allowed by explicit substring, recorded in the audit; every
    other occurrence of the token NI in an R3 output is a finding."""
    hits = []
    for p in paths:
        if not os.path.exists(p) or not p.lower().endswith((".csv", ".md", ".json",
                                                            ".txt")):
            continue
        txt = open(p, encoding="utf-8", errors="ignore").read()
        for m in NI_TOKEN.finditer(txt):
            ctx = re.sub(r"\s+", " ", txt[max(0, m.start() - 90):m.end() + 90])
            if any(a.lower() in ctx.lower() for a in allow_substrings):
                continue
            hits.append((os.path.basename(p), ctx))
    return hits


def walk_outputs(dirs, exts=(".csv", ".md", ".json", ".txt"), exclude_dirs=()):
    ex = tuple(os.path.abspath(d) + os.sep for d in exclude_dirs)
    out = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            ap = os.path.abspath(root) + os.sep
            if any(ap.startswith(e) for e in ex):
                continue
            for f in files:
                if f.lower().endswith(exts):
                    out.add(os.path.abspath(os.path.join(root, f)))
    return sorted(out)


# ------------------------------------------------------------------- inputs -- #
def base_phase(raw):
    """FCC_L12#2 -> FCC_L12. Instance numbers are counted separately, never merged
    away before they have been counted (spec 15, 32)."""
    return re.sub(r"#\d+$", "", str(raw))


def read_inputs():
    """Every frozen R3 input, loaded once, with the seal verified correctly.

    The parent seal's key `condition_sha256` covers
    `02_source_extraction/ALL_CONDITIONS.json`, not the condition master CSV - the
    generator hashes that file, and the key name is simply misleading. The master CSV
    is a deterministic rendering of the sealed JSON and was never itself sealed.
    Stage 05 establishes the canonical rendering and writes a superseding seal that
    says so in as many words.

    Three artifacts are verified hard: the request table and the inclusion rules,
    which decide what is calculated and how it is scored, and the sealed condition
    JSON, which is the evidence record. The master CSV is then checked against the
    canonical rendering when stage 05 has produced one.

    Eight cross-file consistency checks run on every call, not only when something
    looks wrong, and their result is written out each time.
    """
    seal = load_json(os.path.join(D_FREEZE, "INPUT_FREEZE.json"))
    req_p = os.path.join(D_REQ, "STEP5K_R3_CALPHAD_REQUESTS.csv")
    rules_p = os.path.join(D_FREEZE, "STEP5K_R3_CONDITION_INCLUSION_FREEZE.md")
    cond_p = os.path.join(D_EXTRACT, "STEP5K_R3_EXPERIMENTAL_CONDITIONS_MASTER.csv")
    allcond_p = os.path.join(D_EXTRACT, "ALL_CONDITIONS.json")
    canon_p = os.path.join(D_EXTRACT, "STEP5K_R3_CANONICAL_MASTER.csv")

    for path, key, what in (
            (req_p, "request_sha256", "the request table decides what is calculated"),
            (rules_p, "rules_sha256", "the inclusion rules decide how it is scored"),
            (allcond_p, "condition_sha256",
             "the sealed condition record is the evidence base")):
        got = sha256_file(path)
        if got != seal[key]:
            raise SystemExit(
                "FATAL: %s does not match the frozen seal (%s != %s). %s; it was "
                "frozen before any calculation and must not change now."
                % (os.path.basename(path), got[:16], seal[key][:16], what))

    out = {
        "seal": seal,
        "conditions": pd.read_csv(cond_p),
        "selection": pd.read_csv(os.path.join(D_FREEZE, "CONDITION_SELECTION.csv")),
        "requests": pd.read_csv(req_p),
        "conversions": pd.read_csv(os.path.join(
            D_CONV, "STEP5K_R3_COMPOSITION_CONVERSIONS.csv")),
        "domain": pd.read_csv(os.path.join(D_CONV,
                                           "STEP5K_R3_DOMAIN_RELEVANCE.csv")),
        "crosswalk": pd.read_csv(os.path.join(D_CROSS,
                                              "STEP5K_R3_PHASE_CROSSWALK.csv")),
        "expcomp": pd.read_csv(os.path.join(D_PHCOMP,
                                            "EXPERIMENTAL_PHASE_COMPOSITIONS.csv")),
        "manifest": pd.read_csv(os.path.join(D_PDF,
                                             "STEP5K_R3_SOURCE_MANIFEST.csv")),
    }
    out["all_conditions_sha256"] = sha256_file(allcond_p)
    out["master_csv_sha256"] = sha256_file(cond_p)
    if os.path.exists(canon_p):
        a = pd.read_csv(canon_p, encoding="utf-8-sig", dtype=str,
                        keep_default_na=False)
        b = pd.read_csv(cond_p, encoding="utf-8-sig", dtype=str,
                        keep_default_na=False)
        same = (list(a.columns) == list(b.columns) and len(a) == len(b)
                and not [c for c in a.columns if not a[c].equals(b[c])])
        out["master_matches_canonical"] = bool(same)
        if not same:
            raise SystemExit(
                "FATAL: the condition master no longer matches the canonical "
                "rendering produced by stage 05. Re-run stage 05 and read "
                "STEP5K_R3_CANONICAL_MASTER_PROVENANCE.json before going further.")
    else:
        out["master_matches_canonical"] = None
    _verify_consistency(out)
    return out


def _verify_consistency(I):
    """Eight cross-file checks, run every time and written out every time. They bound
    what any single artifact can be wrong about, independently of any hash."""
    m, req, selc = I["conditions"], I["requests"], I["selection"]
    conv, dom, seal = I["conversions"], I["domain"], I["seal"]
    ids = set(m.condition_id)
    checks = {
        "condition ids are unique": bool(m.condition_id.is_unique),
        "every sealed request exists in the master": set(req.condition_id) <= ids,
        "selection covers exactly the master": set(selc.condition_id) == ids,
        "conversions cover exactly the master": set(conv.condition_id) == ids,
        "domain table covers exactly the master": set(dom.condition_id) == ids,
        "selected set equals the sealed request set":
            set(selc.loc[selc.selected.astype(bool), "condition_id"])
            == set(req.condition_id),
        "eligible count agrees between master and selection":
            int(m.direct_eligible.fillna(0).astype(float).sum())
            == int(selc.direct_eligible.fillna(0).astype(float).sum()),
    }
    cc = conv.set_index("condition_id")
    mismatched = 0
    for _, r in req.iterrows():
        try:
            a = {k.upper(): round(float(v), 9)
                 for k, v in json.loads(r.composition_wt_pct).items()}
            b = {k.upper(): round(float(v), 9) for k, v in json.loads(
                cc.loc[r.condition_id, "converted_composition_wt_pct"]).items()}
            mismatched += int(a != b)
        except Exception:                                         # noqa: BLE001
            mismatched += 1
    checks["every request composition equals its conversion-table entry"] = \
        mismatched == 0
    os.makedirs(D_QA, exist_ok=True)
    write_json(os.path.join(D_QA, "STEP5K_R3_INPUT_CONSISTENCY.json"), {
        "generated_utc": utcnow(),
        "parent_seal_frozen_utc": seal.get("frozen_utc"),
        "all_conditions_json_sha256": I["all_conditions_sha256"],
        "all_conditions_json_matches_parent_seal": True,
        "master_csv_sha256": I["master_csv_sha256"],
        "master_csv_was_never_covered_by_the_parent_seal": True,
        "master_matches_canonical_rendering": I["master_matches_canonical"],
        "checks": checks,
        "n_requests_whose_composition_disagrees_with_the_conversion_table":
            mismatched,
        "verdict": "PASS" if all(checks.values()) else "FAIL"})
    if not all(checks.values()):
        raise SystemExit("FATAL: cross-file consistency failed: %s"
                         % [k for k, v in checks.items() if not v])


def experimental_phases(row):
    """The phases a paper actually identified for one condition, in printed order."""
    out = []
    for c in ("phase_1", "phase_2", "phase_3", "phase_4"):
        v = row.get(c)
        if isinstance(v, str) and v.strip() and v.strip().upper() != "UNKNOWN":
            out.append(v.strip())
    return out


def raw_states():
    """The runner's checkpointed output, or None when no calculation has been made.
    A missing file means PENDING. It never means phase absence and never becomes
    CALPHAD_FAILED (spec 28, 37)."""
    p = os.path.join(D_RAW, "STATES.csv")
    if not os.path.exists(p):
        return None, None, None
    states = pd.read_csv(p)
    ph = os.path.join(D_RAW, "PHASES.csv")
    pc = os.path.join(D_RAW, "PHASE_COMPOSITIONS.csv")
    phases = pd.read_csv(ph) if os.path.exists(ph) else pd.DataFrame()
    comps = pd.read_csv(pc) if os.path.exists(pc) else pd.DataFrame()
    return states, phases, comps
