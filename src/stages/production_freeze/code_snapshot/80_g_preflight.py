# -*- coding: utf-8 -*-
"""
80_g_preflight.py
STEP5G Stage 0: fast, read-only pre-flight.

Purpose: every way STEP5G can fail cheaply, it must fail HERE - in seconds - and
not ten minutes into a GPU training run or, worse, after a freeze has already been
written. Nothing is trained, nothing is written outside 00_config / 07_qa / 09_logs.

Checks (spec 12-21, 41):
  P01  the frozen STEP5A 43-feature schema is present and hashes to the value
       recorded in DEVELOPMENT_FREEZE_V1.json (no re-selection anywhere, spec 14)
  P02  every one of the 43 features exists in the ORIGINAL-SIX table
  P03  every one of the 43 features exists in the NEW-ELEMENT table
       (P02/P03 together are the real risk: STEP5C selected per-fold subsets, so a
        43-feature production schema being available in both tables is a fact to be
        proven, not assumed)
  P04  production row availability for PRIMARY  (DEV_TRAIN / DEV_VALIDATION,
       X-bearing, Domain A, X-target present) and reconciliation against the frozen
       STEP5A x_partition counts
  P05  production row availability for SECONDARY (PRIMARY rows + STEP5F FULL rows)
  P06  no IID_TEST / Domain B / ANCHOR / BASELINE row can enter production training
  P07  X-partition targets are finite simplices on the rows that will be used
  P08  element-token Ni scan over every input this stage touches -> must be 0
  P09  the STEP5F report patch target sentence exists exactly once
  P10  the environment carries the libraries the later stages need

Exit code 0 = GO, 2 = NO-GO with the reason printed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_common as C
import g_contract as G

CHECKS = []


def rec(cid, name, ok, detail):
    CHECKS.append({"id": cid, "check": name, "result": "PASS" if ok else "FAIL",
                   "detail": str(detail)[:600]})
    print("  [%s] %-4s %s" % ("PASS" if ok else "FAIL", cid, name))
    if not ok:
        print("         -> %s" % detail)
    return ok


def parquet_columns(path):
    """Read the schema only - never materialize a 44 MB table to ask for its names."""
    import pyarrow.parquet as pq
    return list(pq.ParquetFile(path).schema_arrow.names)


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5G  Stage 0: pre-flight (read-only)")
    print("=" * 78)

    missing = [p for p in (C.A_FEATLIST, C.A_DEVFREEZE, C.A_SPLIT, C.D_DIAG_DATA,
                           C.F_NEWFEAT, C.F_NEWDATA, C.F_REPORT) if not os.path.exists(p)]
    if missing:
        print("NO-GO: required read-only input(s) absent:")
        for m in missing:
            print("   %s" % m)
        return 2

    ok = True

    # ---- P01 frozen feature schema ------------------------------------- #
    feats = C.feature_schema()
    sha = C.sha256_file(C.A_FEATLIST)
    dev = json.load(open(C.A_DEVFREEZE, encoding="utf-8"))
    same_as_freeze = (sha == dev.get("feature_list_sha256")
                      and list(dev.get("features", [])) == feats
                      and int(dev.get("n_features", -1)) == C.N_FEATURES)
    ok &= rec("P01", "STEP5A 43-feature schema present and hash-identical to "
                     "DEVELOPMENT_FREEZE_V1", same_as_freeze and sha == C.A_FEATLIST_SHA256,
              "file sha256 %s | freeze sha256 %s | expected %s | n=%d"
              % (sha[:16], str(dev.get("feature_list_sha256"))[:16],
                 C.A_FEATLIST_SHA256[:16], len(feats)))

    # ---- P02 / P03 schema availability in BOTH tables ------------------- #
    cols_orig = parquet_columns(C.D_DIAG_DATA)
    cols_new = parquet_columns(C.F_NEWFEAT)
    miss_o = [f for f in feats if f not in cols_orig]
    miss_n = [f for f in feats if f not in cols_new]
    ok &= rec("P02", "all 43 production features exist in the ORIGINAL-SIX table",
              not miss_o, "missing from %s: %s" % (os.path.basename(C.D_DIAG_DATA),
                                                   miss_o or "none"))
    ok &= rec("P03", "all 43 production features exist in the NEW-ELEMENT table",
              not miss_n, "missing from %s: %s" % (os.path.basename(C.F_NEWFEAT),
                                                   miss_n or "none"))
    if miss_o or miss_n:
        # Without this the later stages cannot run at all; say exactly what is absent
        # rather than letting 82_g_train.py die on a KeyError ten minutes in.
        print("\n  A missing production feature cannot be repaired inside STEP5G: "
              "re-selecting or substituting a feature is forbidden (spec 14).")

    for t in C.TRUE_COLS:
        if t not in cols_orig:
            ok &= rec("P02t", "X-partition target %s present in ORIGINAL-SIX table" % t,
                      False, "absent")

    # ---- P04 PRIMARY rows ----------------------------------------------- #
    meta = ["state_id", "composition_id", "family", "sampling_domain", "split_role"]
    o = pd.read_parquet(C.D_DIAG_DATA, columns=meta + feats + C.TRUE_COLS)
    o_valid = o[o[C.TRUE_COLS].notna().all(axis=1)]
    tr = o_valid[o_valid.split_role == "DEV_TRAIN"]
    va = o_valid[o_valid.split_role == "DEV_VALIDATION"]
    G.verify_master(tr)
    G.verify_master(va)
    counts_match = (len(tr) == C.A_XPART_TRAIN_ROWS and len(va) == C.A_XPART_VAL_ROWS
                    and tr.composition_id.nunique() == C.A_XPART_TRAIN_COMPOSITIONS)
    ok &= rec("P04", "PRIMARY rows reconcile with the frozen STEP5A x_partition counts",
              counts_match,
              "train %d (frozen %d) over %d compositions (frozen %d) | validation %d "
              "(frozen %d)" % (len(tr), C.A_XPART_TRAIN_ROWS,
                               tr.composition_id.nunique(), C.A_XPART_TRAIN_COMPOSITIONS,
                               len(va), C.A_XPART_VAL_ROWS))

    # ---- P05 SECONDARY rows --------------------------------------------- #
    nf = pd.read_parquet(C.F_NEWFEAT, columns=["state_id", "composition_id", "family",
                                               "sampling_domain", "split_role",
                                               "Temperature_C"] + feats)
    nt = pd.read_parquet(C.F_NEWDATA, columns=["state_id", "element", "state_converged"]
                         + C.X_TARGETS)
    nt = nt[nt.state_converged == 1].rename(columns=dict(zip(C.X_TARGETS, C.TRUE_COLS)))
    n_join = nf.merge(nt[["state_id"] + C.TRUE_COLS], on="state_id", how="inner",
                      validate="one_to_one")
    n_valid = n_join[n_join[C.TRUE_COLS].notna().all(axis=1)]
    fams_new = sorted(set(n_valid.family))
    ok &= rec("P05", "SECONDARY augmentation rows available for exactly BE SN MN LA P",
              fams_new == sorted(C.FULL5_ADDED) and len(n_valid) == C.STEP5F_NEW_ROWS,
              "identities %s | usable rows %d (expected %d) | SECONDARY train would be "
              "%d rows" % (fams_new, len(n_valid), C.STEP5F_NEW_ROWS,
                           len(tr) + len(n_valid)))

    # ---- P06 nothing sealed can leak into production training ------------ #
    prod = pd.concat([tr[meta], va[meta], n_valid[meta]], ignore_index=True)
    leak = {
        "IID_TEST rows": int((prod.split_role == "IID_TEST").sum()),
        "Domain B rows": int((prod.sampling_domain.astype(str) != "A").sum()),
        "ANCHOR rows": int(prod.composition_id.astype(str)
                           .str.contains("ANCHOR", case=False).sum()
                           + prod.split_role.astype(str)
                           .str.contains("ANCHOR", case=False).sum()),
        "BASELINE rows": int((prod.family.astype(str) == "BASELINE").sum()),
    }
    ok &= rec("P06", "no IID_TEST / Domain B / ANCHOR / BASELINE row reaches production",
              all(v == 0 for v in leak.values()), leak)

    # ---- P07 targets are finite simplices -------------------------------- #
    bad = []
    for label, frame in (("PRIMARY_train", tr), ("PRIMARY_validation", va),
                         ("SECONDARY_added", n_valid)):
        y = frame[C.TRUE_COLS].to_numpy(float)
        if not np.isfinite(y).all():
            bad.append("%s: non-finite target" % label)
        elif len(y) and float(np.abs(y.sum(axis=1) - 1.0).max()) > 1e-6:
            bad.append("%s: max |sum-1| = %.3e"
                       % (label, float(np.abs(y.sum(axis=1) - 1.0).max())))
        elif len(y) and float(y.min()) < -1e-12:
            bad.append("%s: negative component %.3e" % (label, float(y.min())))
    ok &= rec("P07", "X-partition targets are finite, non-negative, sum to 1",
              not bad, bad or "all three production frames clean")

    # ---- P08 Ni scan ------------------------------------------------------ #
    # Column NAMES of both tables in full - not only the columns production selects.
    # A Ni-named descriptor sitting unused beside the 43 is still a Ni descriptor in a
    # file this stage opened, and the seal is about access, not about use.
    hits = (["ORIGINAL_SIX::column::%s" % c for c in cols_orig
             if C.NI_TOKEN.search(str(c).upper())]
            + ["NEW_ELEMENT::column::%s" % c for c in cols_new
               if C.NI_TOKEN.search(str(c).upper())]
            + ["FEATURE_SCHEMA::%s" % f for f in feats if C.NI_TOKEN.search(f.upper())]
            + C.ni_scan_frame(o[meta], "ORIGINAL_SIX_meta")
            + C.ni_scan_frame(n_join[["state_id", "composition_id", "family",
                                      "sampling_domain", "split_role"]], "NEW_ELEMENT_meta"))
    split = pd.read_csv(C.A_SPLIT)
    hits += C.ni_scan_frame(split, "COMPOSITION_SPLIT_MASTER")
    ok &= rec("P08", "element-token Ni scan over every STEP5G input = 0 hits",
              not hits, hits or "0 hits across 4 inputs")

    # ---- P09 STEP5F patch target ------------------------------------------ #
    rep = open(C.F_REPORT, encoding="utf-8").read()
    n_find = rep.count(C.PATCH_FIND)
    already = rep.count(C.PATCH_REPLACE)
    ok &= rec("P09", "STEP5F report patch target present exactly once (or already applied)",
              (n_find == 1 and already == 0) or (n_find == 0 and already == 1),
              "target occurrences %d | corrected occurrences %d" % (n_find, already))

    # ---- P10 environment --------------------------------------------------- #
    need = ["torch", "sklearn", "pyarrow", "joblib", "scipy"]
    absent = []
    vers = {}
    for m in need:
        try:
            mod = __import__(m)
            vers[m] = getattr(mod, "__version__", "?")
        except Exception as exc:
            absent.append("%s (%s)" % (m, exc))
    try:
        import torch
        vers["cuda"] = str(torch.cuda.is_available())
    except Exception:
        pass
    ok &= rec("P10", "libraries required by stages 82-85 are importable", not absent,
              absent or vers)

    # ---- record ------------------------------------------------------------ #
    df = pd.DataFrame(CHECKS)
    df.to_csv(os.path.join(C.D_QA, "STEP5G_PREFLIGHT_CHECKS.csv"), index=False,
              encoding="utf-8")
    C.write_json(os.path.join(C.D_CONFIG, "STEP5G_PREFLIGHT.json"), {
        "generated_utc": C.utcnow(),
        "verdict": "GO" if ok else "NO-GO",
        "feature_schema_sha256": sha,
        "n_features": len(feats),
        "features": feats,
        "primary_train_rows": int(len(tr)),
        "primary_train_compositions": int(tr.composition_id.nunique()),
        "primary_validation_rows": int(len(va)),
        "primary_validation_compositions": int(va.composition_id.nunique()),
        "secondary_added_rows": int(len(n_valid)),
        "secondary_added_identities": fams_new,
        "secondary_train_rows": int(len(tr) + len(n_valid)),
        "sealed_leakage": leak,
        "ni_scan_hits": hits,
        "library_versions": vers,
        "inputs": {os.path.basename(p): C.sha256_file(p) for p in
                   (C.A_FEATLIST, C.A_DEVFREEZE)},
        "checks": CHECKS})

    print("-" * 78)
    print("PRE-FLIGHT VERDICT: %s" % ("GO" if ok else "NO-GO"))
    snapshot_path = os.path.join(C.D_CONFIG, "STEP5F_SCIENCE_BEFORE.json")
    if ok and not os.path.exists(snapshot_path):
        scientific = {}
        for base, dirs, files in os.walk(C.STEP5F):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "09_logs", "code_snapshot")]
            for name in files:
                p = os.path.join(base, name)
                if p != C.F_REPORT and not name.endswith((".log", ".pyc")):
                    scientific[p] = C.sha256_file(p)
        C.write_json(snapshot_path, scientific)
    if not ok:
        print("STEP5G stops here. Nothing was trained and no freeze was written.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
