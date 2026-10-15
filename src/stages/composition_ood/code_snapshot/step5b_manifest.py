"""Write CODE_SNAPSHOT_MANIFEST_STEP5B.json (runtime code freeze before first open)."""
import json
from pathlib import Path
from datetime import datetime, timezone
from step5b_core import O, now, sha

SELF = 'step5b_manifest.py'

def main():
    files = []
    for p in sorted((O / 'code_snapshot').glob('*.py')):
        files.append(dict(filename=p.name, path=str(p), bytes=p.stat().st_size,
                          mtime_utc=datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(timespec='seconds'),
                          sha256=sha(p)))
    out = dict(manifest='CODE_SNAPSHOT_MANIFEST_STEP5B', generated_utc=now(),
               note='frozen before the first prospective Domain-B opening; any later code '
                    'change requires a STEP5B_PATCH record (spec 54)',
               n_files=len(files), files=files)
    (O / '09_manifest' / 'CODE_SNAPSHOT_MANIFEST_STEP5B.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'code snapshot manifest: {len(files)} files')


if __name__ == '__main__':
    main()
