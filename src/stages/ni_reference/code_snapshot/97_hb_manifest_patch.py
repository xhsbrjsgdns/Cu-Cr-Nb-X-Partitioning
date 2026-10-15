# -*- coding: utf-8 -*-
"""
97_hb_manifest_patch.py
STEP5H-B Stage 0 (spec 0, 1): DOCUMENTATION-ONLY manifest patch, executed before
the first Ni CALPHAD call.

The original STEP5H_A_FILE_HASHES.csv contains a self-referential row: it tried
to store the sha256 of the manifest itself. That is cryptographically impossible
to satisfy - writing the row changes the file being hashed - so the recorded
value is the hash of the incomplete file at write time, not of the finished
manifest. This module documents the divergence, verifies EVERY other recorded
hash against the file on disk, and writes the manifest's true final hash to an
external sidecar instead.

NOTHING frozen is edited: not the predictions, not the checkpoints, not the
scalers, not the Ni compositions, not any STEP5G preregistration, and not the
original manifest CSV itself.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hb_common as C

PATCH_MD = """# STEP5H_A_MANIFEST_PATCH_01.md

**Written (UTC):** {ts}
**Scope:** DOCUMENTATION ONLY. No frozen artifact was edited: not the PRIMARY or
SECONDARY predictions, not the model checkpoints, not the scalers, not the Ni
composition table, not any STEP5G preregistration, and not the original
`STEP5H_A_FILE_HASHES.csv` itself.

## The issue

The original STEP5H-A manifest `09_manifest/STEP5H_A_FILE_HASHES.csv` contains a
self-referential row that attempts to store the SHA256 of the manifest file
itself:

| | value |
|---|---|
| relative_path of the row | `09_manifest/STEP5H_A_FILE_HASHES.csv` |
| recorded self-hash | `{rec}` |
| actual final SHA256 of the completed manifest | `{act}` |

## Why this happens, and why it is unavoidable in that design

A cryptographic manifest cannot reliably contain its own final hash without an
external checksum scheme: the hash row is written INTO the file, and writing it
changes the bytes of the file being hashed. The recorded value is therefore the
hash of the partially written file captured while the row was computed - by
construction it can never equal the hash of the finished file. This is a
self-reference artifact of the writer, not corruption, tampering or a hash
collision.

## The fix (documentation + external sidecar, no edits)

1. This document records the divergence and both hash values.
2. `STEP5H_A_FILE_HASHES_VERIFIED.csv` (written beside this file) re-verifies
   EVERY row of the original manifest against the file on disk, EXCLUDING the
   self-referential row: `relative_path, recorded_sha256, actual_sha256, match,
   bytes`.
3. The true final hash of the original manifest is stored ONLY here and in the
   external sidecar `STEP5H_A_FILE_HASHES.csv.sha256`, written NEXT TO the
   original manifest. The sidecar is a new file; the manifest is untouched.
4. All STEP5H-B manifests are written without any self-referential row (the
   manifest's own hash goes only into a sidecar), so this cannot recur.

## Impact assessment

| domain | impact |
|---|---|
| scientific | **NONE** |
| predictions (PRIMARY / SECONDARY) | **NONE** - files unedited, hashes re-verified |
| models / scalers | **NONE** |
| Ni compositions | **NONE** |
| any preregistration | **NONE** |

## Verification summary (this run)

- rows in the original manifest: {nrows}
- rows re-verified against disk (excluding the self row): {nver}
- mismatches found: {nmis}
- the single divergent row: the self-referential manifest row, exactly as
  documented above
"""


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5H-B  Stage 0: H-A manifest documentation patch (no frozen file edited)")
    print("=" * 78)

    man = pd.read_csv(C.HA_MANIFEST)
    self_rel = "09_manifest/STEP5H_A_FILE_HASHES.csv"
    self_rows = man[man.relative_path == self_rel]
    if len(self_rows) != 1:
        print("  note: %d self-referential rows found (expected 1); documenting all"
              % len(self_rows))
    recorded_self = self_rows.sha256.iloc[0] if len(self_rows) else ""
    actual_self = C.sha256_file(C.HA_MANIFEST)
    print("  recorded self-hash: %s" % recorded_self)
    print("  actual final hash : %s" % actual_self)
    if recorded_self != C.FROZEN["ha_manifest_recorded_self_hash"]:
        raise SystemExit("FATAL: recorded self-hash differs from the value stated in "
                         "the stage brief - the manifest changed unexpectedly; STOP.")
    if actual_self != C.FROZEN["ha_manifest_actual_final"]:
        raise SystemExit("FATAL: actual manifest hash differs from the value stated in "
                         "the stage brief - the manifest changed unexpectedly; STOP.")

    # verify every NON-self row against disk
    base = C.STEP5HA
    rows, nmis = [], 0
    for _, r in man.iterrows():
        if r.relative_path == self_rel:
            continue
        p = os.path.join(base, r.relative_path.replace("/", os.sep))
        act = C.sha256_file(p) if os.path.exists(p) else "FILE_MISSING"
        match = int(act == r.sha256)
        nmis += (1 - match)
        rows.append({"relative_path": r.relative_path, "recorded_sha256": r.sha256,
                     "actual_sha256": act, "match": match,
                     "bytes": os.path.getsize(p) if os.path.exists(p) else -1})
    ver = pd.DataFrame(rows)
    ver_path = os.path.join(C.D_MAN, "STEP5H_A_FILE_HASHES_VERIFIED.csv")
    ver.to_csv(ver_path, index=False, encoding="utf-8")
    print("  re-verified %d rows against disk | mismatches: %d" % (len(ver), nmis))
    if nmis:
        print(ver[ver.match == 0].to_string(index=False))
        print("FATAL: a frozen STEP5H-A artifact diverges from its recorded hash. "
              "STOP before any CALPHAD call.")
        return 2

    # external sidecar next to the ORIGINAL manifest (new file; original untouched)
    sidecar = C.HA_MANIFEST + ".sha256"
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write("%s  STEP5H_A_FILE_HASHES.csv\n" % actual_self)
    print("  sidecar written: %s" % sidecar)

    doc = PATCH_MD.format(ts=C.utcnow(), rec=recorded_self, act=actual_self,
                          nrows=len(man), nver=len(ver), nmis=nmis)
    patch_path = os.path.join(C.D_MAN, "STEP5H_A_MANIFEST_PATCH_01.md")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(doc)
    C.write_json(os.path.join(C.D_QA, "STEP5H_A_MANIFEST_PATCH_01.json"), {
        "generated_utc": C.utcnow(),
        "patch_id": "STEP5H_A_MANIFEST_PATCH_01",
        "scope": "documentation + external sidecar only; NO frozen file edited",
        "original_manifest": C.HA_MANIFEST,
        "recorded_self_hash": recorded_self,
        "actual_final_self_hash": actual_self,
        "sidecar": sidecar,
        "rows_reverified": int(len(ver)), "mismatches": int(nmis),
        "verified_csv": ver_path,
        "scientific_impact": "NONE", "prediction_impact": "NONE",
        "model_impact": "NONE", "composition_impact": "NONE"})
    print("STAGE 0: PASS (documentation only; every other recorded hash verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
