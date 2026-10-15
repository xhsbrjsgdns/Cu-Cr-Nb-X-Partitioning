# -*- coding: utf-8 -*-
"""
70_f5f_compositions.py
STEP5F Stage A: 1,950 additional compositions per selected element + pre-CALPHAD QA
(spec 5, 6, 10-15).

Sampling (spec 12): Sobol d=3 scramble, master seed 20260910, one independent
deterministic child stream per element, identical sampling rule for all five.

Composition domain (spec 10): Domain-A-like only - S_CrNb 0.6-2.0 wt.%,
R_CrNb_atomic 2.0-6.5, X 0.02-0.20 wt.% IDENTICAL for all five (spec 11: the
X-range difference must not become a confounder).

Pilot reuse (spec 6, 13): the 50 STEP5F-PILOT compositions per element are NOT
recomputed; new draws that exactly duplicate a pilot composition (4-decimal
Cr/Nb/X keys) are skipped and the Sobol sequence simply continues.

IDs (spec 14): new compositions F5F_<X>_<0001-1950>; pilot rows keep their
F5P_<X>_<...> ids untouched.

No BASELINE compositions (the Cu-Cr-Nb baseline exists in FULL_V1).
Cr and Nb are never sampled independently: S and R are sampled then inverted.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5f_common as C

CHECKS = []


def chk(cid, what, status, detail=""):
    CHECKS.append({"id": cid, "check": what, "status": status, "detail": str(detail)[:400]})
    print("  [%-5s] %-4s %s%s" % (cid, status, what,
                                  ("  |  " + str(detail)[:150]) if detail else ""))
    return status == "PASS"


def pilot_keys(element):
    p = pd.read_csv(C.PILOT_COMPS)
    p = p[p.X_element == element]
    return set(zip(p.Cr_wt_pct.round(4), p.Nb_wt_pct.round(4), p.X_wt_pct.round(4)))


def generate_for_element(element, pilot_dup_keys):
    """1,950 additional unique compositions; duplicates of the pilot 50 are skipped
    and the Sobol sequence continues (spec 13)."""
    child = C.element_child_seed(element)
    seed = int(child.generate_state(1, dtype=np.uint32)[0])
    sob = qmc.Sobol(d=3, scramble=True, seed=seed)
    rows, seen = [], set()
    tries, dups_pilot = 0, 0
    while len(rows) < C.N_ADD_PER_ELEMENT and tries < 2_000_000:
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
        if key in pilot_dup_keys:
            dups_pilot += 1
            continue
        if key in seen or w_cr <= 0 or w_nb <= 0 or w_x <= 0:
            continue
        seen.add(key)
        cu = round(100.0 - w_cr - w_nb - w_x, 6)
        rows.append({
            "composition_id": "F5F_%s_%04d" % (element, len(rows) + 1),
            "family": element, "X_element": element,
            "sampling_domain": "A",
            "split_role": "AUGMENTATION_TRAIN",
            "composition_source": "STEP5F_FULL_augmentation",
            "sampling_method": "Sobol(d=3, scramble=True) child stream",
            "master_seed": C.MASTER_SEED, "element_child_seed": seed,
            "sobol_index": tries - 1,
            "Cu_wt_pct": cu, "Cr_wt_pct": w_cr, "Nb_wt_pct": w_nb, "X_wt_pct": w_x,
            "S_CrNb_wt_pct": round(w_cr + w_nb, 6),
            "R_CrNb_atomic_target": round(r, 6),
            "R_CrNb_atomic_actual": round(C.actual_ratio(w_cr, w_nb), 6),
            "Cr_wt_pct_exact": w_cr_e, "Nb_wt_pct_exact": w_nb_e, "X_wt_pct_exact": x,
            "S_CrNb_target": round(s, 6),
            "composition_sum_wt_pct": round(cu + w_cr + w_nb + w_x, 9),
        })
    if len(rows) < C.N_ADD_PER_ELEMENT:
        raise SystemExit("FATAL: only %d unique compositions for %s after %d Sobol points"
                         % (len(rows), element, tries))
    print("  %-3s %d Sobol points -> 1950 unique (%d pilot-duplicate keys skipped)"
          % (element, tries, dups_pilot))
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("STEP5F FULL  Stage A: composition generation + pre-CALPHAD QA")
    print("selected identities (frozen): %s" % " ".join(C.SELECTED_ELEMENTS))
    print("=" * 78)
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)

    pilot = pd.read_csv(C.PILOT_COMPS)
    frames = []
    for el in C.SELECTED_ELEMENTS:
        frames.append(generate_for_element(el, pilot_keys(el)))
    new = pd.concat(frames, ignore_index=True)
    new["run_planned"] = 1
    new.to_csv(os.path.join(C.D_COMP, "STEP5F_FULL_NEW_COMPOSITIONS.csv"),
               index=False, encoding="utf-8")

    pilot_kept = pilot[pilot.X_element.isin(C.SELECTED_ELEMENTS)].copy()
    pilot_kept["run_planned"] = 0
    pilot_kept["composition_source"] = "STEP5F_PILOT_reused_not_recomputed"
    shared = ["composition_id", "family", "X_element", "sampling_domain", "split_role",
              "composition_source", "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
              "S_CrNb_wt_pct", "R_CrNb_atomic_actual", "composition_sum_wt_pct", "run_planned"]
    pilot_kept["sampling_domain"] = "A"
    pilot_kept["split_role"] = "AUGMENTATION_TRAIN"
    both = pd.concat([new[shared], pilot_kept[shared]], ignore_index=True)
    both.to_csv(os.path.join(C.D_COMP, "STEP5F_FULL_COMPOSITIONS.csv"),
                index=False, encoding="utf-8")
    print("  wrote STEP5F_FULL_NEW_COMPOSITIONS.csv (%d rows) + STEP5F_FULL_COMPOSITIONS.csv (%d rows)"
          % (len(new), len(both)))

    # ------------------------------------------------- pre-CALPHAD QA (15) -- #
    print("\n=== pre-CALPHAD composition QA (spec 15) ===")
    per = new.groupby("X_element").size()
    chk("A01", "1950 additional compositions per element",
        "PASS" if set(per.values) == {C.N_ADD_PER_ELEMENT} else "FAIL", per.to_dict())
    chk("A02", "9750 additional compositions in total",
        "PASS" if len(new) == C.N_NEW_COMPOSITIONS else "FAIL", len(new))
    chk("A03", "composition_id unique and correctly formatted F5F_<X>_<0001-1950>",
        "PASS" if (new.composition_id.nunique() == len(new)
                   and new.composition_id.str.match(r"^F5F_[A-Z]{1,2}_\d{4}$").all()
                   and new.composition_id.str[-4:].astype(int).between(1, 1950).all())
        else "FAIL", new.composition_id.iloc[0])
    sig = new.apply(lambda r: (r.X_element, r.Cr_wt_pct, r.Nb_wt_pct, r.X_wt_pct), axis=1)
    chk("A04", "no duplicate composition within an element (new set)",
        "PASS" if sig.nunique() == len(new) else "FAIL",
        "%d unique / %d" % (sig.nunique(), len(new)))
    pil = pd.concat([pilot[pilot.X_element == el].assign(e=el) for el in C.SELECTED_ELEMENTS])
    both_sig = pd.concat([sig, pil.apply(lambda r: (r.e, r.Cr_wt_pct, r.Nb_wt_pct, r.X_wt_pct),
                                         axis=1)]).nunique()
    chk("A05", "pilot 50 + 1950 = 2000 UNIQUE per element (spec 6, 13)",
        "PASS" if both_sig == len(new) + len(pil) else "FAIL",
        "%d unique of %d total" % (both_sig, len(new) + len(pil)))
    chk("A06", "pilot composition_ids NOT renamed (spec 14)",
        "PASS" if set(pil.composition_id.str[:4]) == {"F5P_"}
        and not both.composition_id.str.startswith("F5P_").fillna(False).eq(False).all()
        else "FAIL", "%d F5P rows preserved" % len(pil))
    chk("A07", "Cu > 0 for every composition",
        "PASS" if (new.Cu_wt_pct > 0).all() else "FAIL",
        "min Cu %.4f wt%%" % new.Cu_wt_pct.min())
    chk("A08", "wt.%% sums to 100 within 1e-6",
        "PASS" if (new.composition_sum_wt_pct - 100.0).abs().max() < 1e-6 else "FAIL",
        "max dev %.2e" % (new.composition_sum_wt_pct - 100.0).abs().max())
    ok_s = new.S_CrNb_wt_pct.between(C.S_CRNB_RANGE[0] - 1e-6, C.S_CRNB_RANGE[1] + 1e-6)
    chk("A09", "S_CrNb inside Domain-A range 0.6-2.0 wt.%",
        "PASS" if ok_s.all() else "FAIL",
        "[%.4f, %.4f]" % (new.S_CrNb_wt_pct.min(), new.S_CrNb_wt_pct.max()))
    ok_r = new.R_CrNb_atomic_actual.between(C.R_CRNB_RANGE[0] - 1e-2, C.R_CRNB_RANGE[1] + 1e-2)
    chk("A10", "ACTUAL Cr/Nb atomic ratio inside 2.0-6.5 (recomputed)",
        "PASS" if ok_r.all() else "FAIL",
        "[%.4f, %.4f]" % (new.R_CrNb_atomic_actual.min(), new.R_CrNb_atomic_actual.max()))
    ok_x = new.X_wt_pct.between(C.X_WT_RANGE[0] - 1e-9, C.X_WT_RANGE[1] + 1e-9)
    spans = new.groupby("X_element").X_wt_pct.agg(["min", "max"])
    chk("A11", "X in 0.02-0.20 wt.%% IDENTICAL for all five (spec 10, 11)",
        "PASS" if ok_x.all()
        and (spans["min"] >= C.X_WT_RANGE[0] - 1e-9).all()
        and (spans["max"] <= C.X_WT_RANGE[1] + 1e-9).all() else "FAIL",
        "[%.4f, %.4f] across %d elements" % (new.X_wt_pct.min(), new.X_wt_pct.max(),
                                             new.X_element.nunique()))
    chk("A12", "exactly the five frozen identities, no sixth element (spec 4)",
        "PASS" if sorted(set(new.X_element)) == sorted(C.SELECTED_ELEMENTS) else "FAIL",
        sorted(set(new.X_element)))
    forb = sorted(set(both.X_element) & set(C.FORBIDDEN_ELEMENTS))
    chk("A13", "no Ni / Au / Ce composition (spec 1)",
        "PASS" if not forb else "FAIL", forb or "none")
    hits = C.ni_scan([("new_compositions", new), ("full_compositions", both)])
    chk("A14", "Ni token count == 0", "PASS" if not hits else "FAIL",
        hits[:4] or "0 hits (element-token regex)")
    seeds = new.groupby("X_element").element_child_seed.nunique()
    chk("A15", "each element used its own independent child stream (spec 12)",
        "PASS" if (seeds == 1).all()
        and new.element_child_seed.nunique() == len(C.SELECTED_ELEMENTS) else "FAIL",
        new.groupby("X_element").element_child_seed.first().to_dict())
    chk("A16", "master seed 20260910 recorded everywhere", "PASS"
        if (new.master_seed == C.MASTER_SEED).all() else "FAIL", C.MASTER_SEED)

    nfail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    C.write_json(os.path.join(C.D_CALQA, "STEP5F_FULL_COMPOSITION_QA.json"), {
        "stage": "STEP5F FULL Stage A", "generated_utc": C.utcnow(),
        "n_new_compositions": int(len(new)), "n_pilot_reused": int(len(pil)),
        "n_total": int(len(both)), "per_element_total": C.N_TOTAL_PER_ELEMENT,
        "elements": C.SELECTED_ELEMENTS,
        "formal_states_total": C.N_FORMAL_STATES,
        "new_solver_states": C.N_NEW_STATES,
        "sampling": {"method": "Sobol(d=3, scramble=True)", "master_seed": C.MASTER_SEED,
                     "per_element_child_stream": True,
                     "S_CrNb_range": list(C.S_CRNB_RANGE),
                     "R_CrNb_atomic_range": list(C.R_CRNB_RANGE),
                     "X_wt_pct_range": list(C.X_WT_RANGE),
                     "x_range_identical_across_elements": True},
        "new_compositions_sha256": C.sha256_file(
            os.path.join(C.D_COMP, "STEP5F_FULL_NEW_COMPOSITIONS.csv")),
        "full_compositions_sha256": C.sha256_file(
            os.path.join(C.D_COMP, "STEP5F_FULL_COMPOSITIONS.csv")),
        "checks": CHECKS, "status": "FAIL" if nfail else "PASS"})
    print("\n" + "=" * 78)
    print("STAGE A: %s  (%d PASS / %d FAIL)"
          % ("FAIL" if nfail else "PASS",
             sum(1 for c in CHECKS if c["status"] == "PASS"), nfail))
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
