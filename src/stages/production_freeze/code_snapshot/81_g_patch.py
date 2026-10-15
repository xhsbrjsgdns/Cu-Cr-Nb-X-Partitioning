# -*- coding: utf-8 -*-
"""
81_g_patch.py
STEP5G Stage 1: the STEP5F report-only correction (spec 7).

STEP5F_FULL_REPORT.md answer 4 reads:

    **4. New solver states?** 107250 (the 4,950 pilot states were reused, not recomputed).

107,250 is right. "4,950" is not. 4,950 is the size of the WHOLE nine-element
STEP5F-PILOT (9 x 50 compositions x 11 T). Only the five selected identities'
pilot states were carried into STEP5F FULL: 5 x 50 x 11 = 2,750, which is exactly
what the same report already states at answer 2 ("250 pilot reused") and at
integrity item 6 ("250 pilot compositions / 2750 states"), and exactly what closes
the arithmetic 110,000 - 2,750 = 107,250.

So this is an isolated wrong number in prose, not a data error. Accordingly:
  - ONLY the report text is touched. No STEP5F data file, model, metric, hash-bearing
    artifact or conclusion is altered (spec 7).
  - The original file is copied to STEP5G before editing, and both the pre- and
    post-patch sha256 are recorded, so the change is auditable rather than silent.
  - The patch is idempotent: re-running after it has been applied is a no-op.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_common as C

PATCH_DOC = """# STEP5F_REPORT_PATCH_01.md

**Patch id:** STEP5F_REPORT_PATCH_01
**Applied (UTC):** {ts}
**Scope:** report prose only. No STEP5F data file, model artifact, metric, figure or
conclusion is modified. STEP5F's own file-hash manifest is left untouched; this
document is the record of the divergence it will now show for one file.

## What was wrong

`STEP5F_FULL_REPORT.md`, answer 4:

> **4. New solver states?** 107250 (the {old} pilot states were reused, not recomputed).

## What is right

> **4. New solver states?** 107250 (the {new} pilot states were reused, not recomputed).

## Why

- STEP5F FULL computed 10,000 compositions x 11 temperatures = **110,000** states.
- Of those, the states already computed by STEP5F-PILOT for the five SELECTED
  identities were reused: 5 identities x 50 pilot compositions x 11 T = **2,750**.
- New solver states = 110,000 - 2,750 = **107,250**, which is the number the report
  already gives and which the CALPHAD stage actually ran.
- **4,950** is the size of the entire nine-element pilot (9 x 50 x 11). Four of those
  nine identities (AL, MG, ZN, GE) were not selected, so their pilot states were never
  carried into STEP5F FULL and must not be counted as reused.

The same report was already internally correct elsewhere - answer 2 ("250 pilot
reused + 9,750 new") and integrity item 6 ("250 pilot compositions / 2750 states") -
so only this one sentence disagreed with the rest of the document and with the data.

## Verification

| | value |
|---|---|
| file | `{path}` |
| sha256 before | `{sha_before}` |
| sha256 after | `{sha_after}` |
| occurrences replaced | {n} |
| bytes before / after | {b_before} / {b_after} |
| pre-patch copy retained at | `{backup}` |

No other line of the report differs; the diff is exactly one substring on one line.
"""


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5G  Stage 1: STEP5F report-only patch (spec 7)")
    print("=" * 78)

    if not os.path.exists(C.F_REPORT):
        print("FATAL: %s not found." % C.F_REPORT)
        return 2

    text = open(C.F_REPORT, encoding="utf-8", newline="").read()
    sha_before = C.sha256_file(C.F_REPORT)
    b_before = len(text.encode("utf-8"))
    n = text.count(C.PATCH_FIND)
    n_done = text.count(C.PATCH_REPLACE)

    if n == 0 and n_done >= 1:
        print("  patch already applied (%d corrected occurrence(s)); nothing to do."
              % n_done)
        already = os.path.join(C.D_FREEZE, "STEP5F_REPORT_PATCH_01.md")
        if os.path.exists(already):
            print("  existing patch record: %s" % already)
            return 0
    if n != 1:
        print("FATAL: expected exactly 1 occurrence of the patch target, found %d." % n)
        print("       target: %r" % C.PATCH_FIND)
        print("       The report may have been edited by hand. STEP5G will not guess "
              "which sentence to change; fix the target or update the patch record.")
        return 2

    backup = os.path.join(C.D_FREEZE, "STEP5F_FULL_REPORT.pre_patch01.md")
    shutil.copy2(C.F_REPORT, backup)

    new_text = text.replace(C.PATCH_FIND, C.PATCH_REPLACE, 1)
    with open(C.F_REPORT, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    sha_after = C.sha256_file(C.F_REPORT)

    # prove the diff is exactly one line and exactly the intended substring
    a, b = text.splitlines(), new_text.splitlines()
    diff = [i for i, (x, y) in enumerate(zip(a, b), start=1) if x != y]
    assert len(a) == len(b) and len(diff) == 1, "patch changed more than one line"
    print("  line %d changed:" % diff[0])
    print("    -  %s" % a[diff[0] - 1].strip())
    print("    +  %s" % b[diff[0] - 1].strip())

    doc = PATCH_DOC.format(
        ts=C.utcnow(), old="4,950", new="2,750", path=C.F_REPORT,
        sha_before=sha_before, sha_after=sha_after, n=1,
        b_before=b_before, b_after=len(new_text.encode("utf-8")), backup=backup)
    out = os.path.join(C.D_FREEZE, "STEP5F_REPORT_PATCH_01.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    C.write_json(os.path.join(C.D_QA, "STEP5F_REPORT_PATCH_01.json"), {
        "generated_utc": C.utcnow(), "patch_id": "STEP5F_REPORT_PATCH_01",
        "scope": "report prose only; no data, model, metric or conclusion changed",
        "file": C.F_REPORT, "sha256_before": sha_before, "sha256_after": sha_after,
        "line_changed": diff[0], "old_text": C.PATCH_FIND, "new_text": C.PATCH_REPLACE,
        "pre_patch_copy": backup})
    print("  wrote %s" % out)
    print("STAGE 1: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
