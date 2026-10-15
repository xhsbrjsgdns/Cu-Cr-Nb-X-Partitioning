# -*- coding: utf-8 -*-
"""
120_j_freeze_verify.py
STEP5J Stage 0: verify every freeze hash before a single number is consolidated
(spec 3). If any mismatch: STOP.

Three layers of verification, because the earlier stages recorded their hashes in
three different ways:

  A  CONSTANT-CHECKED  artifacts whose sha256 was published by an earlier freeze
     (STEP5G checkpoints and scaler, STEP5H-A predictions, STEP5H-B targets, the
     anchor sequences, the 43-feature list). These are asserted.
  B  MANIFEST-CHECKED  the 24 STEP5I recovery checkpoints and their predictions,
     verified against STEP5I_RECOVERY_CHECKPOINT_MANIFEST.csv, and the whole STEP5I
     tree verified against STEP5I_FINAL_SEAL_HASHES.csv.
  C  RECORDED-ONLY     files no earlier stage hashed. STEP5J publishes their hashes
     rather than asserting an expectation it invented. That distinction is kept
     visible in the output instead of being blurred into a single "verified" column.

STEP5I_FINAL_SEAL_HASHES.csv is used as the authoritative STEP5I index, not
STEP5I_FILE_HASHES.csv: the independent review of STEP5I recorded that the earlier
index was written mid-run, so a log file and the completion record were hashed before
they were finished. Using the superseded index would manufacture a false mismatch.

Nothing is written outside the STEP5J root, and nothing frozen is touched.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j_common as C

CHECKS = []


def rec(cid, name, ok, detail):
    CHECKS.append({"id": cid, "check": name, "result": "PASS" if ok else "FAIL",
                   "detail": str(detail)[:600]})
    print("  [%s] %-5s %s" % ("PASS" if ok else "FAIL", cid, name))
    if not ok:
        print("          -> %s" % detail)
    return ok


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5J  Stage 0: freeze verification (read-only)")
    print("=" * 78)

    missing = [(k, v[0]) for k, v in C.SOURCES.items() if not os.path.exists(v[0])]
    if missing:
        print("NOT READY - frozen source(s) absent:")
        for k, p in missing:
            print("   %-22s %s" % (k, p))
        C.write_json(C.P(C.D_QA, "STEP5J_MISSING_SOURCES.json"),
                     {"generated_utc": C.utcnow(),
                      "missing": [{"key": k, "path": p} for k, p in missing]})
        return 2

    ok = True
    rows = []
    for key, (path, stage, role, expected) in sorted(C.SOURCES.items()):
        live = C.sha256_file(path)
        mode = "CONSTANT_CHECKED" if expected else "RECORDED_ONLY"
        match = None if expected is None else (live == expected)
        rows.append({"key": key, "stage": stage, "role": role, "path": path,
                     "bytes": os.path.getsize(path), "sha256": live,
                     "expected_sha256": expected or "",
                     "verification_mode": mode,
                     "matches_expected": "" if match is None else str(match)})
        if match is False:
            ok = False
    inv = pd.DataFrame(rows)
    inv.to_csv(C.P(C.D_FREEZE, "STEP5J_FROZEN_SOURCE_INVENTORY.csv"), index=False,
               encoding="utf-8")
    n_const = int((inv.verification_mode == "CONSTANT_CHECKED").sum())
    bad = inv[inv.matches_expected == "False"]
    ok &= rec("F01", "all %d constant-checked frozen artifacts hash as published"
              % n_const, bad.empty,
              "mismatched: %s" % list(bad.key) if not bad.empty else
              "STEP5G x3, STEP5H-A x3, STEP5H-B x1, sequences x2, feature list x1")

    # ---- STEP5H-B score freeze binds the exact prediction and target hashes -- #
    ps = C.load_json("HB_PRIMARY_SCORE")
    ok &= rec("F02", "STEP5H-B PRIMARY score is bound to the frozen prediction and "
                     "target hashes",
              ps["primary_prediction_sha256"] == C.sha256_file(C.src("HA_PRIMARY_PRED"))
              and ps["target_sha256"] == C.sha256_file(C.src("HB_TARGETS")),
              "prediction %s | target %s" % (ps["primary_prediction_sha256"][:16],
                                             ps["target_sha256"][:16]))
    ss = C.load_json("HB_SECONDARY_SCORE")
    sec_tvd = ss.get("mean_tvd", ss.get("all_secondary_metrics", {}).get("mean_tvd"))
    ok &= rec("F03", "STEP5H-B PRIMARY and SECONDARY blind scores are present and "
                     "immutable",
              ps.get("immutable") is True and sec_tvd is not None,
              "PRIMARY %.10f | SECONDARY %s" % (ps["mean_tvd"], sec_tvd))

    # ---- STEP5I recovery checkpoints against their own manifest -------------- #
    man = C.load_csv("I_CKPT_MANIFEST")
    ck_bad, pred_bad = [], []
    for _, r in man.iterrows():
        for col_path, col_hash, bucket in (("checkpoint", "checkpoint_sha256", ck_bad),
                                           ("predictions", "predictions_sha256",
                                            pred_bad)):
            p = r[col_path]
            if not os.path.exists(p) or C.sha256_file(p) != r[col_hash]:
                bucket.append(r["tag"])
    ok &= rec("F04", "all %d STEP5I recovery checkpoints hash as recorded" % len(man),
              not ck_bad, ck_bad or "24 checkpoints verified")
    ok &= rec("F05", "all %d STEP5I recovery prediction files hash as recorded"
              % len(man), not pred_bad, pred_bad or "24 prediction files verified")

    # ---- the whole STEP5I tree against its authoritative seal index ---------- #
    seal = C.load_csv("I_SEAL")
    seal_bad, seal_missing = [], []
    for _, r in seal.iterrows():
        p = C.P(C.STEP5I, *str(r["relative_path"]).replace("\\", "/").split("/"))
        if not os.path.exists(p):
            seal_missing.append(r["relative_path"])
        elif C.sha256_file(p) != r["sha256"]:
            seal_bad.append(r["relative_path"])
    ok &= rec("F06", "STEP5I final seal index verifies (%d files)" % len(seal),
              not seal_bad and not seal_missing,
              {"mismatched": seal_bad[:10], "missing": seal_missing[:10]}
              if (seal_bad or seal_missing) else "%d files unchanged" % len(seal))

    # ---- the metric files STEP5J will actually read -------------------------- #
    metric_keys = ["I_BASELINES", "I_THRESHOLDS", "I_RANDOM_AGG", "I_RANDOM_REP",
                   "I_MAXIMIN", "I_CONTROLS", "I_CMP", "I_TOPO", "I_NBNI3", "I_TEMP",
                   "I_COVERAGE", "I_MECH", "I_BOOT"]
    seal_map = {str(r["relative_path"]).replace("\\", "/"): r["sha256"]
                for _, r in seal.iterrows()}
    unsealed = []
    for k in metric_keys:
        rel = os.path.relpath(C.src(k), C.STEP5I).replace("\\", "/")
        if seal_map.get(rel) != C.sha256_file(C.src(k)):
            unsealed.append(k)
    ok &= rec("F07", "every STEP5I metric file STEP5J reads is in the seal index and "
                     "unchanged", not unsealed, unsealed or "%d metric files"
              % len(metric_keys))

    # ---- no new experiment has been started under the STEP5J root ------------ #
    # match by extension, not by filename text: a table legitimately called
    # "Table_01_Dataset_and_CALPHAD_design.csv" is not a solver artifact, and a
    # substring check on the word turns every re-run into a false alarm
    SOLVER_EXT = (".pt", ".pth", ".ckpt", ".joblib", ".sqlite", ".npz")
    stray = []
    if os.path.exists(C.ROOT):
        for base, _, files in os.walk(C.ROOT):
            for fn in files:
                if fn.lower().endswith(SOLVER_EXT):
                    stray.append(os.path.relpath(C.P(base, fn), C.ROOT))
    ok &= rec("F08", "no model checkpoint or CALPHAD artifact exists under the STEP5J "
                     "root", not stray, stray or "STEP5J writes no model and no "
              "thermodynamic state")

    pd.DataFrame(CHECKS).to_csv(C.P(C.D_QA, "STEP5J_FREEZE_CHECKS.csv"), index=False,
                                encoding="utf-8")
    C.write_json(C.P(C.D_FREEZE, "STEP5J_FREEZE_VERIFICATION.json"), {
        "generated_utc": C.utcnow(),
        "verdict": "VERIFIED" if ok else "MISMATCH",
        "n_sources": len(C.SOURCES),
        "n_constant_checked": n_const,
        "n_recorded_only": len(C.SOURCES) - n_const,
        "n_step5i_checkpoints": int(len(man)),
        "n_step5i_sealed_files": int(len(seal)),
        "authoritative_step5i_index": "STEP5I_FINAL_SEAL_HASHES.csv",
        "superseded_step5i_index": "STEP5I_FILE_HASHES.csv (written mid-run; a log and "
                                   "the completion record were hashed before they were "
                                   "finished - see the STEP5I review addendum)",
        "checks": CHECKS})

    print("-" * 78)
    print("FREEZE VERDICT: %s" % ("VERIFIED" if ok else "MISMATCH"))
    if not ok:
        print("STEP5J stops here. No consolidation is performed against a moved freeze.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
