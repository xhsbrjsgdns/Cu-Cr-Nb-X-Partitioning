# -*- coding: utf-8 -*-
"""
39b_step5a_code_snapshot.py
STEP5A sec.99: freeze the code snapshot BEFORE the final IID baseline run
(40_step5a_iid_test.py). Records filename / bytes / mtime / SHA256 of every
script in code_snapshot/.

Output: 09_manifest/CODE_SNAPSHOT_MANIFEST_STEP5A.json
"""
import hashlib
import json
import os
from datetime import datetime, timezone

STEP5A = r"E:\Cu_CALPHAD_ML\04_ML\STEP5A_JMST_ML_FOUNDATION_V2"
SNAP = os.path.join(STEP5A, "code_snapshot")
MAN = os.path.join(STEP5A, "09_manifest")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries = []
    for fn in sorted(os.listdir(SNAP)):
        p = os.path.join(SNAP, fn)
        if os.path.isfile(p):
            entries.append({"filename": fn, "bytes": os.path.getsize(p),
                            "mtime_utc": datetime.fromtimestamp(os.path.getmtime(p),
                                                                tz=timezone.utc).isoformat(timespec="seconds"),
                            "sha256": sha256_file(p)})
    out = {"manifest": "CODE_SNAPSHOT_MANIFEST_STEP5A", "generated_utc": ts,
           "note": "frozen before the final IID baseline run (sec.99); any later code change "
                   "requires a STEP5A_PATCH_XX record (sec.100)",
           "n_files": len(entries), "files": entries}
    os.makedirs(MAN, exist_ok=True)
    with open(os.path.join(MAN, "CODE_SNAPSHOT_MANIFEST_STEP5A.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"code snapshot manifest: {len(entries)} files")
    for e in entries:
        print(f"  {e['filename']:45s} {e['bytes']:>8d} B  {e['sha256'][:16]}")


if __name__ == "__main__":
    main()
