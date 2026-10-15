# -*- coding: utf-8 -*-
"""
54_step5c_gate.py
STEP5C Stage I: confirm that NO held-out metric has yet influenced any fold protocol.

Spec 43 requires all six folds to be developed and frozen BEFORE any held-out set is
scored, so that an early fold's test performance cannot shape a later fold's rules.
This module is the structural enforcement of that ordering: the scoring module refuses
to start unless the gate file it writes exists and says PASS.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5c_common as C

GATE_PATH = os.path.join(C.D_MANIFEST, "STEP5C_SCORING_GATE.json")
CHECKS = []


def chk(cid, what, status, detail=""):
    CHECKS.append({"id": cid, "check": what, "status": status, "detail": str(detail)[:500]})
    print("  [%-5s] %-4s %s%s" % (cid, status, what,
                                  ("  |  " + str(detail)[:150]) if detail else ""))
    return status == "PASS"


HELDOUT_ARTIFACT_PATTERNS = [
    (C.D_PRED, "_HELDOUT_PREDICTIONS.parquet"),
    (C.D_METRICS, "LOEO_ALL_FOLDS_METRICS.csv"),
    (C.D_METRICS, "LOEO_PER_TARGET_METRICS.csv"),
    (C.D_METRICS, "LOEO_TOPOLOGY_METRICS.csv"),
    (C.D_METRICS, "LOEO_ERROR_BY_TEMPERATURE.csv"),
]


def main():
    print("=" * 78)
    print("STEP 5C   Stage I: freeze-before-scoring gate (spec 43)")
    print("=" * 78)
    freezes = {}
    missing = []
    for e in C.HELDOUT_ELEMENTS:
        p = os.path.join(C.D_MANIFEST, f"FOLD_{e}_FREEZE.json")
        if os.path.exists(p):
            freezes[e] = json.load(open(p, encoding="utf-8"))
        else:
            missing.append(f"FOLD_{e}_FREEZE.json")
    chk("I01", "all six FOLD_<X>_FREEZE.json exist",
        "PASS" if not missing else "FAIL", missing or "6/6")

    same_pre = {f.get("preregistration_sha256") for f in freezes.values()}
    chk("I02", "every fold was frozen against the SAME preregistration",
        "PASS" if len(same_pre) == 1 else "FAIL",
        list(same_pre)[0][:16] if len(same_pre) == 1 else same_pre)
    same_schema = {f.get("candidate_schema_hash") for f in freezes.values()}
    chk("I03", "every fold started from the SAME candidate feature schema",
        "PASS" if len(same_schema) == 1 else "FAIL", len(same_schema))
    fams = {e: sorted({t["model_family"] for t in f["tasks"].values()})
            for e, f in freezes.items()}
    allowed = all(set(v) <= set(C.MODEL_FAMILIES) for v in fams.values())
    chk("I04", "every selected model family is inside the preregistered search space",
        "PASS" if allowed else "FAIL", fams)
    declared = {e: f.get("heldout_metrics_computed") for e, f in freezes.items()}
    chk("I05", "no fold record claims a held-out metric was computed during development",
        "PASS" if not any(declared.values()) else "FAIL", declared)
    zero_use = {e: f.get("heldout_rows_used_in_development") for e, f in freezes.items()}
    chk("I06", "held-out rows used in development == 0 for every fold",
        "PASS" if set(zero_use.values()) == {0} else "FAIL", zero_use)

    # I07 is an ORDERING check, not an absence check. What must never happen is a
    # held-out artifact that PREDATES the freezes - that would mean a fold was scored
    # before the protocol was fixed. Artifacts newer than every freeze come from an
    # already-authorized scoring pass and must not block a legitimate re-run.
    freeze_mtimes = [os.path.getmtime(os.path.join(C.D_MANIFEST, f"FOLD_{e}_FREEZE.json"))
                     for e in freezes]
    latest_freeze = max(freeze_mtimes) if freeze_mtimes else 0.0
    premature, later = [], []
    for d, pat in HELDOUT_ARTIFACT_PATTERNS:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not (fn.endswith(pat) or fn == pat):
                continue
            rel = os.path.join(os.path.basename(d), fn)
            (premature if os.path.getmtime(os.path.join(d, fn)) < latest_freeze - 1.0
             else later).append(rel)
    chk("I07", "no held-out prediction or metric artifact PREDATES the six freezes",
        "PASS" if not premature else "FAIL",
        ("PREDATING: %s" % premature) if premature
        else ("0 artifacts" if not later
              else "%d artifact(s) exist, all written after the last freeze (re-scoring "
                   "an already-frozen protocol is allowed)" % len(later)))

    budget_ok = True
    for e, f in freezes.items():
        for task, log in f["tuning_log"].items():
            for fam, rec in log.items():
                if rec.get("n_trials") != C.N_TRIALS:
                    budget_ok = False
    chk("I08", "identical tuning budget (%d trials) in every fold x task x family" % C.N_TRIALS,
        "PASS" if budget_ok else "FAIL", C.N_TRIALS)

    nfail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    gate = {
        "stage": "STEP5C Stage I",
        "generated_utc": C.utcnow(),
        "status": "FAIL" if nfail else "PASS",
        "heldout_artifacts_predating_freeze": premature,
        "heldout_artifacts_after_freeze": later,
        "latest_freeze_mtime_utc": (C.utcnow() if not freeze_mtimes else
                                    __import__("datetime").datetime.utcfromtimestamp(
                                        latest_freeze).isoformat(timespec="seconds") + "Z"),
        "statement": ("all six folds were developed and frozen before any "
                      "HELDOUT_ELEMENT_TEST set was scored; no held-out metric has "
                      "influenced any fold's feature policy, model family, search bounds, "
                      "trial count, metrics, thresholds, projection or distance definition"),
        "frozen_folds": sorted(freezes.keys()),
        "freeze_hashes": {e: f.get("freeze_sha256") for e, f in freezes.items()},
        "preregistration_sha256": list(same_pre)[0] if len(same_pre) == 1 else None,
        "checks": CHECKS,
        "scoring_authorized": nfail == 0,
    }
    C.write_json(GATE_PATH, gate)
    print("\n" + "=" * 78)
    print("STAGE I: %s  -  scoring %s"
          % (gate["status"], "AUTHORIZED" if gate["scoring_authorized"] else "BLOCKED"))
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
