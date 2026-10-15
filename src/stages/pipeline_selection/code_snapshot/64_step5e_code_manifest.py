# -*- coding: utf-8 -*-
"""64_step5e_code_manifest.py — freeze the STEP5E code snapshot BEFORE the first
held-out evaluation of the zero-shot remedy branches."""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5e_common as C


def sha(p):
    return C.sha256_file(p)


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = []
    for fn in sorted(os.listdir(C.D_CODE)):
        p = os.path.join(C.D_CODE, fn)
        if os.path.isfile(p) and fn.endswith(".py"):
            files.append({"filename": fn, "path": p, "bytes": os.path.getsize(p),
                          "mtime_utc": datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc).isoformat(timespec="seconds"),
                          "sha256": sha(p)})
    out = {"manifest": "CODE_SNAPSHOT_MANIFEST_STEP5E",
           "generated_utc": ts,
           "note": "frozen before the first held-out evaluation of the zero-shot remedy "
                   "branches; later changes require STEP5E_PATCH_XX (spec 83)",
           "n_files": len(files), "files": files}
    C.write_json(os.path.join(C.D_MAN, "CODE_SNAPSHOT_MANIFEST_STEP5E.json"), out)
    print(f"code snapshot manifest: {len(files)} files")


if __name__ == "__main__":
    main()
