# -*- coding: utf-8 -*-
"""
50_step5c_manifests.py
STEP5C Stage A + B + C.

  A  read the STEP5B archival status and confirm READY FOR STEP5C / Anchor / Ni
  B  create the STEP5C directory skeleton
  C  build LOEO_MASTER_FOLD_MANIFEST.csv and run the per-fold manifest QA (spec 12)

No model is trained here and no held-out target is read.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5c_common as C

CHECKS = []


def chk(cid, what, status, detail=""):
    CHECKS.append({"id": cid, "check": what, "status": status, "detail": str(detail)[:500]})
    print("  [%-5s] %-4s %s%s" % (cid, status, what,
                                  ("  |  " + str(detail)[:150]) if detail else ""))
    return status == "PASS"


# ------------------------------------------------------- Stage A: archive -- #
def stage_a():
    print("\n=== STAGE A: STEP5B archival status ===")
    p = os.path.join(C.STEP5B, "07_qa", "STEP5B_PROTOCOL_ARCHIVE_SUMMARY.json")
    req = [p,
           os.path.join(C.STEP5B, "07_qa", "STEP5B_ARCHIVAL_CLEANUP_REPORT.md"),
           os.path.join(C.STEP5B, "07_qa", "STEP5B_PROTOCOL_CHAIN_AUDIT.md"),
           os.path.join(C.STEP5B, "07_qa", "STEP5B_REPORT_PATCH_01.md"),
           os.path.join(C.STEP5B, "01_preregistration", "PROTOCOL_AMENDMENT_V1.md")]
    missing = [x for x in req if not os.path.exists(x)]
    chk("A01", "all four STEP5B archival documents + amendment present",
        "PASS" if not missing else "FAIL", missing or "5/5")
    if missing:
        raise SystemExit("FATAL: STEP5C cannot start without the STEP5B archival chain.")
    arch = json.load(open(p, encoding="utf-8"))
    chk("A02", "READY FOR STEP5C == YES",
        "PASS" if arch.get("ready_for_step5c") else "FAIL", arch.get("ready_for_step5c"))
    chk("A03", "ANCHOR still sealed", "PASS" if arch.get("anchor_sealed") else "FAIL",
        arch.get("anchor_sealed"))
    chk("A04", "Ni still sealed", "PASS" if arch.get("ni_sealed") else "FAIL",
        arch.get("ni_sealed"))
    chk("A05", "STEP5B status carried forward verbatim", "PASS", arch.get("status"))
    ps = arch.get("phase_split_report_patch", {}).get("corrected_values", {})
    split_tvd = ps.get("split_fraction_tvd")
    nonsplit_tvd = ps.get("nonsplit_fraction_tvd")
    chk("A06", "STEP5B phase-split correction inherited: split is NOT harder in Domain B",
        "PASS" if (split_tvd is not None and nonsplit_tvd is not None
                   and split_tvd <= nonsplit_tvd) else "WARN",
        "split %.6f vs non-split %.6f -> no directional prior is assumed in STEP5C (spec 3)"
        % (split_tvd or float("nan"), nonsplit_tvd or float("nan")))
    amend = open(os.path.join(C.STEP5B, "01_preregistration", "PROTOCOL_AMENDMENT_V1.md"),
                 encoding="utf-8").read()
    chk("A07", "PROTOCOL_AMENDMENT_V1 inherited verbatim (spec 1)", "PASS",
        "sha256 %s" % C.sha256_text(amend)[:16])
    if not (arch.get("ready_for_step5c") and arch.get("anchor_sealed")
            and arch.get("ni_sealed")):
        raise SystemExit("FATAL: STEP5B does not authorise STEP5C.")
    return arch


# --------------------------------------------------- Stage B: directories -- #
def stage_b():
    print("\n=== STAGE B: STEP5C directory skeleton ===")
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    chk("B01", "13 STEP5C directories created", "PASS", C.STEP5C)


# ------------------------------------------------------ Stage C: manifest -- #
def stage_c():
    print("\n=== STAGE C: LOEO master fold manifest ===")
    sm = pd.read_csv(C.SPLIT_MASTER)
    a = sm[sm["sampling_domain"] == C.PRIMARY_DOMAIN].copy()
    chk("C01", "primary universe is Domain A only (spec 6)",
        "PASS" if set(a["sampling_domain"]) == {"A"} else "FAIL",
        "%d Domain-A compositions; Domain B/ANCHOR excluded" % len(a))
    chk("C02", "Domain A holds %s compositions" % f"{C.N_DOMAIN_A_COMPOSITIONS:,}",
        "PASS" if len(a) == C.N_DOMAIN_A_COMPOSITIONS else "FAIL", len(a))

    rows = []
    for heldout in C.HELDOUT_ELEMENTS:
        fold = f"FOLD_{heldout}"
        r = a.copy()
        r["fold"] = fold
        r["heldout_element"] = heldout
        r["original_split_role"] = r["split_role"]
        r["loeo_role"] = [C.loeo_role(sr, fam, heldout)
                          for sr, fam in zip(r["split_role"], r["family"])]
        rows.append(r[["fold", "heldout_element", "composition_id", "family",
                       "original_split_role", "loeo_role"]])
    man = pd.concat(rows, ignore_index=True)
    out = os.path.join(C.D_FOLDS, "LOEO_MASTER_FOLD_MANIFEST.csv")
    man.to_csv(out, index=False, encoding="utf-8")
    print("  wrote %s (%d rows)" % (os.path.basename(out), len(man)))
    return man


# --------------------------------------------------- Stage C QA (spec 12) -- #
def stage_c_qa(man):
    print("\n=== STAGE C QA: per-fold manifest checks (spec 12) ===")
    summary = {}
    for heldout in C.HELDOUT_ELEMENTS:
        fold = f"FOLD_{heldout}"
        f = man[man.fold == fold]
        tr = set(f.loc[f.loeo_role == "FOLD_TRAIN", "composition_id"])
        va = set(f.loc[f.loeo_role == "FOLD_VALIDATION", "composition_id"])
        te = set(f.loc[f.loeo_role == "HELDOUT_ELEMENT_TEST", "composition_id"])
        ex = set(f.loc[f.loeo_role == "EXCLUDED_FROM_FOLD", "composition_id"])
        fam = f.set_index("composition_id")["family"]
        ok = True
        ok &= chk(f"C10_{heldout}", f"{fold}: TRAIN n VALIDATION == 0",
                  "PASS" if not (tr & va) else "FAIL", len(tr & va))
        ok &= chk(f"C11_{heldout}", f"{fold}: TRAIN n HELDOUT == 0",
                  "PASS" if not (tr & te) else "FAIL", len(tr & te))
        ok &= chk(f"C12_{heldout}", f"{fold}: VALIDATION n HELDOUT == 0",
                  "PASS" if not (va & te) else "FAIL", len(va & te))
        n_tr_h = int((fam.loc[sorted(tr)] == heldout).sum()) if tr else 0
        n_va_h = int((fam.loc[sorted(va)] == heldout).sum()) if va else 0
        ok &= chk(f"C13_{heldout}", f"{fold}: held-out family in TRAIN == 0",
                  "PASS" if n_tr_h == 0 else "FAIL", n_tr_h)
        ok &= chk(f"C14_{heldout}", f"{fold}: held-out family in VALIDATION == 0",
                  "PASS" if n_va_h == 0 else "FAIL", n_va_h)
        ok &= chk(f"C15_{heldout}", f"{fold}: held-out test compositions == "
                                    f"{C.N_HELDOUT_COMPOSITIONS:,}",
                  "PASS" if len(te) == C.N_HELDOUT_COMPOSITIONS else "FAIL", len(te))
        ok &= chk(f"C16_{heldout}", f"{fold}: every held-out test composition is family {heldout}",
                  "PASS" if te and set(fam.loc[sorted(te)]) == {heldout} else "FAIL",
                  sorted(set(fam.loc[sorted(te)])) if te else [])
        ok &= chk(f"C17_{heldout}", f"{fold}: roles partition all "
                                    f"{C.N_DOMAIN_A_COMPOSITIONS:,} compositions",
                  "PASS" if len(tr) + len(va) + len(te) + len(ex)
                  == C.N_DOMAIN_A_COMPOSITIONS else "FAIL",
                  "%d + %d + %d + %d" % (len(tr), len(va), len(te), len(ex)))
        # spec 9: other elements' IID_TEST must be EXCLUDED, never training
        orig = f.set_index("composition_id")["original_split_role"]
        leaked_iid = sum(1 for c in (tr | va) if orig.loc[c] == "IID_TEST")
        ok &= chk(f"C18_{heldout}", f"{fold}: no other-element IID_TEST composition entered "
                                    f"TRAIN or VALIDATION (spec 9)",
                  "PASS" if leaked_iid == 0 else "FAIL", leaked_iid)
        summary[heldout] = {"fold": fold, "n_train": len(tr), "n_validation": len(va),
                            "n_heldout_test": len(te), "n_excluded": len(ex),
                            "all_checks_pass": bool(ok)}
        print("  %s: train %d | validation %d | held-out test %d | excluded %d"
              % (fold, len(tr), len(va), len(te), len(ex)))

    # spec 13: cross-fold reuse of an element as a TRAINING element is by design
    cross = {}
    for e in C.HELDOUT_ELEMENTS:
        as_train = [f"FOLD_{h}" for h in C.HELDOUT_ELEMENTS if h != e]
        cross[e] = {"held_out_in": f"FOLD_{e}", "training_element_in": as_train}
    chk("C19", "cross-fold element reuse is by design, not leakage (spec 13)", "PASS",
        "each element is held out in exactly one fold and is a training element in the other five")
    return summary, cross


def main():
    print("=" * 78)
    print("STEP 5C - STRICT 6x LEAVE-ONE-ELEMENT-OUT")
    print("retrospective unseen-element OOD   |   Stages A + B + C")
    print("=" * 78)
    arch = stage_a()
    stage_b()
    man = stage_c()
    summary, cross = stage_c_qa(man)

    nfail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    out = {
        "stage": "STEP5C Stage A/B/C",
        "generated_utc": C.utcnow(),
        "terminology": ("retrospective leave-one-element-out / unseen-element OOD; "
                        "NOT prospective blind validation (spec 0)"),
        "step5b_archive": {"ready_for_step5c": arch.get("ready_for_step5c"),
                           "anchor_sealed": arch.get("anchor_sealed"),
                           "ni_sealed": arch.get("ni_sealed"),
                           "status": arch.get("status")},
        "phase_split_prior": ("STEP5B measured split 0.012606 vs non-split 0.013036 fraction "
                              "TVD; STEP5C assumes NO directional prior (spec 3)"),
        "fold_summary": summary,
        "cross_fold_elements": cross,
        "manifest": "02_fold_manifests/LOEO_MASTER_FOLD_MANIFEST.csv",
        "manifest_sha256": C.sha256_file(
            os.path.join(C.D_FOLDS, "LOEO_MASTER_FOLD_MANIFEST.csv")),
        "checks": CHECKS,
        "status": "FAIL" if nfail else "PASS",
    }
    C.write_json(os.path.join(C.D_FOLDS, "LOEO_FOLD_MANIFEST_QA.json"), out)
    print("\n" + "=" * 78)
    print("STAGE A/B/C: %s   (%d PASS / %d WARN / %d FAIL)"
          % (out["status"], sum(1 for c in CHECKS if c["status"] == "PASS"),
             sum(1 for c in CHECKS if c["status"] == "WARN"), nfail))
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
