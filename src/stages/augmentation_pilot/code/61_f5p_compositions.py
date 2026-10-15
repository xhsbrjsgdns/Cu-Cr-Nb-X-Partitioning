# -*- coding: utf-8 -*-
"""
61_f5p_compositions.py
STEP5F-PILOT Stage 1: pilot compositions + pre-run QA (spec 5, 8-14).

50 compositions per element, 9 elements, 450 total.

Sampling (spec 11): Sobol d=3, one pilot seed 20260909, an INDEPENDENT child sequence
per element via numpy SeedSequence, and the same sampling rule for every element.

Composition domain (spec 8-10):
    S_CrNb        0.6 - 2.0  wt.%      (Domain A, unchanged)
    R_CrNb_atomic 2.0 - 6.5            (Domain A, unchanged)
    X_wt_pct      0.02 - 0.20 wt.%     (IDENTICAL for all nine elements)

The uniform X range is deliberate: this pilot compares CHEMICAL IDENTITY, so an X-range
difference must not become a confounder. It is not copied from any single earlier X.

Cr and Nb are never sampled independently - S and R are sampled and inverted, exactly as
in FULL_V1. No BASELINE compositions are generated: the Cu-Cr-Nb baseline already exists
in FULL_V1 (spec 12).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5p_common as C

CHECKS = []


def chk(cid, what, status, detail=""):
    CHECKS.append({"id": cid, "check": what, "status": status, "detail": str(detail)[:400]})
    print("  [%-5s] %-4s %s%s" % (cid, status, what,
                                  ("  |  " + str(detail)[:140]) if detail else ""))
    return status == "PASS"


def generate_for_element(element):
    child = C.element_child_seed(element)
    seed = int(child.generate_state(1, dtype=np.uint32)[0])
    sob = qmc.Sobol(d=3, scramble=True, seed=seed)
    pts, rows, tries = [], [], 0
    seen = set()
    while len(rows) < C.N_PER_ELEMENT and tries < 20000:
        p = sob.random(1)[0]
        tries += 1
        s = C.S_CRNB_RANGE[0] + p[0] * (C.S_CRNB_RANGE[1] - C.S_CRNB_RANGE[0])
        r = C.R_CRNB_RANGE[0] + p[1] * (C.R_CRNB_RANGE[1] - C.R_CRNB_RANGE[0])
        x = C.X_WT_RANGE[0] + p[2] * (C.X_WT_RANGE[1] - C.X_WT_RANGE[0])
        w_cr_e, w_nb_e = C.invert_s_r(s, r)
        w_cr = round(w_cr_e, C.WT_DECIMALS)
        w_nb = round(w_nb_e, C.WT_DECIMALS)
        w_x = round(x, C.WT_DECIMALS)
        key = (w_cr, w_nb, w_x)
        if key in seen or w_cr <= 0 or w_nb <= 0 or w_x <= 0:
            continue                       # take the next Sobol point; never overwrite
        seen.add(key)
        cu = round(100.0 - w_cr - w_nb - w_x, 6)
        rows.append({
            "composition_id": "F5P_%s_%04d" % (element, len(rows) + 1),
            "family": element, "X_element": element,
            "sampling_domain": "F5P_PILOT",
            "sampling_method": "Sobol(d=3, scramble=True) child sequence",
            "pilot_seed": C.SAMPLING_SEED, "element_child_seed": seed,
            "sobol_index": tries - 1,
            "Cu_wt_pct": cu, "Cr_wt_pct": w_cr, "Nb_wt_pct": w_nb, "X_wt_pct": w_x,
            "S_CrNb_wt_pct": round(w_cr + w_nb, 6),
            "R_CrNb_atomic_target": round(r, 6),
            "R_CrNb_atomic_actual": round(C.actual_ratio(w_cr, w_nb), 6),
            "Cr_wt_pct_exact": w_cr_e, "Nb_wt_pct_exact": w_nb_e, "X_wt_pct_exact": x,
            "S_CrNb_target": round(s, 6),
            "composition_sum_wt_pct": round(cu + w_cr + w_nb + w_x, 9),
        })
        pts.append(p)
    if len(rows) < C.N_PER_ELEMENT:
        raise SystemExit("FATAL: only %d unique compositions for %s after %d Sobol points"
                         % (len(rows), element, tries))
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("STEP5F-PILOT  Stage 1: composition generation + pre-run QA")
    print("=" * 78)
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)

    probe_path = os.path.join(C.D_QA, "STEP5F_PILOT_ELEMENT_PROBE.json")
    if not os.path.exists(probe_path):
        raise SystemExit("FATAL: run 60_f5p_probe.py first - generating 450 compositions "
                         "before knowing which elements TCCU6 actually has is wasted work.")
    probe = json.load(open(probe_path, encoding="utf-8"))
    available = [e for e in C.PILOT_ELEMENTS if e in probe["available"]]
    blocked = [e for e in C.PILOT_ELEMENTS if e not in probe["available"]]
    if not available:
        raise SystemExit(
            "FATAL: the TCCU6 probe found NO usable candidate (%s). Generating 450 "
            "compositions for a run that cannot happen would produce a misleading "
            "record. Investigate the probe result in "
            "06_qa/STEP5F_PILOT_ELEMENT_PROBE.json first." % ", ".join(blocked))
    if blocked:
        print("  NOTE: %s are not available in TCCU6 and are excluded from the run."
              % " ".join(blocked))
        print("        Their compositions are still generated and marked, so the record")
        print("        shows exactly what would have been run (spec 19 spirit).")

    frames = [generate_for_element(e) for e in C.PILOT_ELEMENTS]
    comps = pd.concat(frames, ignore_index=True)
    comps["tccu6_available"] = comps["X_element"].isin(available).astype(int)
    comps["run_planned"] = comps["tccu6_available"]
    out = os.path.join(C.D_COMP, "STEP5F_PILOT_COMPOSITIONS.csv")
    comps.to_csv(out, index=False, encoding="utf-8")
    print("\n  wrote %s (%d rows)" % (os.path.basename(out), len(comps)))

    # ------------------------------------------------ pre-run QA (spec 14) -- #
    print("\n=== pre-run composition QA (spec 14) ===")
    per = comps.groupby("X_element").size()
    chk("Q01", "50 unique compositions per element",
        "PASS" if set(per.values) == {C.N_PER_ELEMENT} else "FAIL", per.to_dict())
    chk("Q02", "450 compositions in total",
        "PASS" if len(comps) == C.N_COMPOSITIONS else "FAIL", len(comps))
    chk("Q03", "composition_id unique and correctly formatted",
        "PASS" if (comps.composition_id.nunique() == len(comps)
                   and comps.composition_id.str.match(r"^F5P_[A-Z]{1,2}_\d{4}$").all())
        else "FAIL", comps.composition_id.iloc[0])
    sig = comps.apply(lambda r: (r.X_element, r.Cr_wt_pct, r.Nb_wt_pct, r.X_wt_pct), axis=1)
    chk("Q04", "no duplicate composition within an element",
        "PASS" if sig.nunique() == len(comps) else "FAIL",
        "%d unique / %d" % (sig.nunique(), len(comps)))
    chk("Q05", "Cu > 0 for every composition",
        "PASS" if (comps.Cu_wt_pct > 0).all() else "FAIL",
        "min Cu %.4f wt%%" % comps.Cu_wt_pct.min())
    chk("Q06", "wt.% sums to 100 within 1e-6",
        "PASS" if (comps.composition_sum_wt_pct - 100.0).abs().max() < 1e-6 else "FAIL",
        "max dev %.2e" % (comps.composition_sum_wt_pct - 100.0).abs().max())
    ok_s = comps.S_CrNb_wt_pct.between(C.S_CRNB_RANGE[0] - 1e-6, C.S_CRNB_RANGE[1] + 1e-6)
    chk("Q07", "S_CrNb inside the Domain-A range 0.6 - 2.0 wt.%",
        "PASS" if ok_s.all() else "FAIL",
        "[%.4f, %.4f]" % (comps.S_CrNb_wt_pct.min(), comps.S_CrNb_wt_pct.max()))
    ok_r = comps.R_CrNb_atomic_actual.between(C.R_CRNB_RANGE[0] - 1e-2,
                                              C.R_CRNB_RANGE[1] + 1e-2)
    chk("Q08", "ACTUAL Cr/Nb atomic ratio inside 2.0 - 6.5 (recomputed, not assumed)",
        "PASS" if ok_r.all() else "FAIL",
        "[%.4f, %.4f]" % (comps.R_CrNb_atomic_actual.min(),
                          comps.R_CrNb_atomic_actual.max()))
    ok_x = comps.X_wt_pct.between(C.X_WT_RANGE[0] - 1e-9, C.X_WT_RANGE[1] + 1e-9)
    chk("Q09", "X in 0.02 - 0.20 wt.% and IDENTICAL for all nine elements",
        "PASS" if ok_x.all() else "FAIL",
        "[%.4f, %.4f] across %d elements" % (comps.X_wt_pct.min(), comps.X_wt_pct.max(),
                                             comps.X_element.nunique()))
    spans = comps.groupby("X_element").X_wt_pct.agg(["min", "max"])
    chk("Q10", "no element has a narrower or wider X range than the others",
        "PASS" if (spans["min"] >= C.X_WT_RANGE[0] - 1e-9).all()
        and (spans["max"] <= C.X_WT_RANGE[1] + 1e-9).all() else "FAIL",
        "identical bounds enforced by construction")
    chk("Q11", "no BASELINE composition generated (spec 12)",
        "PASS" if "BASELINE" not in set(comps.family) else "FAIL", sorted(set(comps.family)))
    forb = sorted(set(comps.X_element) & set(C.FORBIDDEN_ELEMENTS))
    chk("Q12", "no Ni / Au / Ce composition (spec 1, 3)",
        "PASS" if not forb else "FAIL", forb or "none")
    hits = C.ni_scan([("compositions", comps)])
    chk("Q13", "Ni token count == 0", "PASS" if not hits else "FAIL",
        hits[:4] or "0 hits (element-token regex)")
    seeds = comps.groupby("X_element").element_child_seed.nunique()
    chk("Q14", "each element used its own independent child sequence",
        "PASS" if (seeds == 1).all()
        and comps.element_child_seed.nunique() == len(C.PILOT_ELEMENTS) else "FAIL",
        comps.groupby("X_element").element_child_seed.first().to_dict())

    nfail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    C.write_json(os.path.join(C.D_QA, "STEP5F_PILOT_COMPOSITION_QA.json"), {
        "stage": "STEP5F-PILOT Stage 1", "generated_utc": C.utcnow(),
        "n_compositions": int(len(comps)), "n_per_element": int(C.N_PER_ELEMENT),
        "elements": C.PILOT_ELEMENTS, "available_in_tccu6": available,
        "excluded_not_in_tccu6": blocked,
        "formal_states_if_all_run": int(C.N_FORMAL_STATES),
        "formal_states_planned": int(len(comps[comps.run_planned == 1])
                                     * len(C.TEMPERATURES_C)),
        "sampling": {"method": "Sobol(d=3, scramble=True)", "pilot_seed": C.SAMPLING_SEED,
                     "per_element_child_sequence": True,
                     "S_CrNb_range": list(C.S_CRNB_RANGE),
                     "R_CrNb_atomic_range": list(C.R_CRNB_RANGE),
                     "X_wt_pct_range": list(C.X_WT_RANGE),
                     "x_range_identical_across_elements": True,
                     "cr_nb_sampled_independently": False},
        "compositions_sha256": C.sha256_file(out),
        "checks": CHECKS, "status": "FAIL" if nfail else "PASS"})
    print("\n" + "=" * 78)
    print("STAGE 1: %s  (%d PASS / %d FAIL)"
          % ("FAIL" if nfail else "PASS",
             sum(1 for c in CHECKS if c["status"] == "PASS"), nfail))
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
