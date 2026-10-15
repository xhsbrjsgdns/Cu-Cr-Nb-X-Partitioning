# -*- coding: utf-8 -*-
"""
130_k_screen_design.py
STEP5K Part B Stage 1: virtual candidate space + PRE-FROZEN screening rules
(spec 17-28). Everything that could be tuned after seeing results - the
objective set, the "moderate C15" admissible band, the OOD applicability
filter, the Pareto protocol and the diversity-selection rule - is derived here
from FROZEN historical data (FULL_V1 / STEP5A) and frozen into
STEP5K_SCREENING_RULES_FREEZE.json BEFORE any surrogate prediction exists.

Candidate space (spec 17-20): the six originally developed identities only
(ZR TI CO SI AG MO); NO Ni and no other element; Domain-A bounds; the exact
frozen S/R inversion; deterministic Sobol, 20,000 unique compositions per
family, seed recorded; duplicates skipped by continuing the sequence.

No post-hoc weighted score is defined anywhere (spec 24): primary selection is
non-dominated sorting on Objectives A-D with E as a hard constraint and F as a
hard applicability filter.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C


def generate_candidates():
    frames = []
    log = {}
    ss = np.random.SeedSequence(C.SCREEN_MASTER_SEED)
    for idx, el in enumerate(C.SIX):
        child = ss.spawn(len(C.SIX))[idx]
        seed = int(child.generate_state(1, dtype=np.uint32)[0])
        sob = qmc.Sobol(d=3, scramble=True, seed=seed)
        rows, seen, tries, dups = [], set(), 0, 0
        while len(rows) < C.N_PER_FAMILY and tries < 2_000_000:
            p = sob.random(1)[0]
            tries += 1
            s = C.S_CRNB_RANGE[0] + p[0] * (C.S_CRNB_RANGE[1] - C.S_CRNB_RANGE[0])
            r = C.R_CRNB_RANGE[0] + p[1] * (C.R_CRNB_RANGE[1] - C.R_CRNB_RANGE[0])
            x = C.X_WT_RANGE[0] + p[2] * (C.X_WT_RANGE[1] - C.X_WT_RANGE[0])
            w_cr_e, w_nb_e = C.invert_s_r(s, r)
            w_cr, w_nb, w_x = (round(w_cr_e, C.WT_DECIMALS), round(w_nb_e, C.WT_DECIMALS),
                               round(x, C.WT_DECIMALS))
            key = (w_cr, w_nb, w_x)
            if key in seen or w_cr <= 0 or w_nb <= 0 or w_x <= 0:
                dups += 1
                continue
            seen.add(key)
            cu = round(100.0 - w_cr - w_nb - w_x, 6)
            rows.append({
                "candidate_id": "K5_%s_%05d" % (el, len(rows) + 1),
                "X_identity": el, "family": el, "X_element": el,
                "sampling_domain": "A_VIRTUAL_SCREEN",
                "Cu_wt_pct": cu, "Cr_wt_pct": w_cr, "Nb_wt_pct": w_nb, "X_wt_pct": w_x,
                "S_CrNb": round(w_cr + w_nb, 6),
                "R_CrNb_atomic": round(C.actual_ratio(w_cr, w_nb), 6),
                "S_target": round(s, 6), "R_target": round(r, 6), "X_target": round(x, 6),
                "master_seed": C.SCREEN_MASTER_SEED, "family_seed": seed,
                "sobol_index": tries - 1,
                "composition_sum_wt_pct": round(cu + w_cr + w_nb + w_x, 9)})
        frames.append(pd.DataFrame(rows))
        log[el] = {"family_seed": seed, "sobol_points_drawn": tries,
                   "duplicates_or_invalid_skipped": dups, "n_unique": len(rows)}
    df = pd.concat(frames, ignore_index=True)
    return df, log


def derive_c15_band():
    """Spec 25: 'moderate C15' = the middle-quantile band [q25, q75] of NON-ZERO
    C15 family volume fractions in the FROZEN FULL_V1 ML-ready table, pooled over
    Domain A at 450 and 600 C. Derived BEFORE any screening prediction."""
    ml = pd.read_parquet(os.path.join(C.FULL_V1, "03_processed",
                                      "FULL_V1_ML_READY_CONVERGED.parquet"),
                         columns=["state_id", "sampling_domain", "Temperature_C",
                                  "C15_volfrac"])
    ml = ml[(ml.sampling_domain == "A") & (ml.Temperature_C.isin([450.0, 600.0]))]
    nz = ml.loc[ml.C15_volfrac > 1e-4, "C15_volfrac"]
    lo, hi = np.quantile(nz, C.C15_BAND_QUANTILES)
    return {"rule": "middle quantile band (q%.2f, q%.2f) of NON-ZERO C15 family "
                    "volume fraction, FULL_V1 Domain A, 450+600 C pooled"
            % C.C15_BAND_QUANTILES,
            "source": "FULL_V1_ML_READY_CONVERGED.parquet (frozen)",
            "n_nonzero_states": int(len(nz)),
            "band_low": float(lo), "band_high": float(hi)}


def derive_ood_reference():
    """Spec 26: trusted-region filter geometry, fixed BEFORE any screening result.

    Convention (identical in spirit to the frozen STEP5E/STEP5G maximin geometry):
    reference = STEP5A DEV_TRAIN rows at 600 C (one row per training composition,
    22,050 points); StandardScaler fitted on those reference rows ONLY (geometry
    use; not a model object); candidates are embedded at 600 C; metric = mean
    Euclidean distance to the k nearest reference points. The tree surrogates
    themselves consume RAW features exactly as in the frozen STEP5A IID test."""
    return {"rule": "mean %d-NN Euclidean distance (600 C embedding) to STEP5A "
                    "DEV_TRAIN composition rows <= %.2f"
            % (C.OOD_KNN, C.OOD_MAX_KNN_DIST),
            "reference": "STEP5A DEV_TRAIN rows at 600 C from "
                         "FULL_V1_CANDIDATE_FEATURES (frozen)",
            "scaler": "StandardScaler fitted on the reference rows only (geometry "
                      "convention from frozen STEP5E maximin; not a model scaler)",
            "threshold": C.OOD_MAX_KNN_DIST, "k": C.OOD_KNN,
            "note": "threshold fixed pre-ranking and documented, not tuned"}


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K Part B Stage 1: candidate space + PRE-FROZEN screening rules")
    print("=" * 78)
    baseline = C.protect_baseline()
    print("  protected frozen artifacts verified unchanged: %d" % len(baseline))
    C.write_json(os.path.join(C.D_CONFIG, "STEP5K_PROTECTED_BASELINE.json"), {
        "generated_utc": C.utcnow(), "hashes": baseline})

    # feature schema gate
    feats = [ln.strip() for ln in open(C.A_FEATLIST, encoding="utf-8") if ln.strip()]
    assert len(feats) == C.N_FEATURES
    assert C.sha256_file(C.A_FEATLIST) == C.A_FEATLIST_SHA256
    print("  frozen 43-feature schema verified (sha %s...)" % C.A_FEATLIST_SHA256[:12])

    comps, log = generate_candidates()
    out = os.path.join(C.D_SCREEN, "STEP5K_VIRTUAL_CANDIDATE_COMPOSITIONS.csv")
    comps.to_csv(out, index=False, encoding="utf-8")
    assert len(comps) == 6 * C.N_PER_FAMILY
    assert comps.duplicated(["X_identity", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct"]).sum() == 0
    assert comps.X_identity.isin(C.SIX).all()
    print("  candidate space: %d compositions (%d per family x 6) -> %s"
          % (len(comps), C.N_PER_FAMILY, os.path.basename(out)))

    band = derive_c15_band()
    print("  moderate-C15 band (pre-frozen): [%.6f, %.6f] from %d non-zero FULL_V1 states"
          % (band["band_low"], band["band_high"], band["n_nonzero_states"]))
    ood = derive_ood_reference()

    rules = {
        "freeze_id": "STEP5K_SCREENING_RULES_FREEZE_V1",
        "generated_utc": C.utcnow(),
        "frozen_before_any_screening_prediction": True,
        "candidate_space": {
            "identities": C.SIX, "n_per_family": C.N_PER_FAMILY,
            "total": int(len(comps)), "master_seed": C.SCREEN_MASTER_SEED,
            "sampler": "Sobol(d=3, scramble=True), one child stream per family",
            "bounds": {"S_CrNb_wt_pct": list(C.S_CRNB_RANGE),
                       "R_CrNb_atomic": list(C.R_CRNB_RANGE),
                       "X_wt_pct": list(C.X_WT_RANGE)},
            "inversion": "w_Nb = S/(1+R*M_Cr/M_Nb); w_Cr = S - w_Nb (frozen FULL_V1)",
            "per_family_log": log,
            "compositions_file": out,
            "compositions_sha256": C.sha256_file(out),
            "no_ni_no_unseen_x": True},
        "objectives": {
            "A_maximize": "pred Cr partition to FCC at 950 C (Cr_FCC_950)",
            "B_maximize": "pred useful Cr release: Delta_Cr_FCC = Cr_FCC(950) - Cr_FCC(450)",
            "C_constraint": "pred C15 family volume fraction within the moderate band "
                            "at BOTH 450 C and 600 C (band below); NOT maximized",
            "D_minimize": "pred Cr partition to OTHER_SOLID at 450 C (Cr_OTHER_450)",
            "E_constraint": "pred LIQUID family fraction <= %.2f at every screening "
                            "temperature (hard QA; in-domain liquid fraction is 0)"
                            % C.LIQUID_TOLERANCE,
            "F_filter": "OOD applicability (below) - hard filter, not an objective"},
        "moderate_c15_band": band,
        "ood_filter": ood,
        "pareto_protocol": {
            "method": "non-dominated sorting (maximize A, B; minimize D) over the "
                      "APPLICABLE set (C, E, F satisfied)",
            "no_weighted_score": True,
            "global_front": True, "family_specific_fronts": True,
            "family_fairness": "both global and per-family fronts are reported; no "
                               "family may silently fill the finalist list (spec 27)"},
        "diversity_selection": {
            "method": "deterministic maximin diversity within the global Pareto set, "
                      "in (S_CrNb, R_CrNb_atomic, X_wt_pct) scaled space",
            "per_family_target": list(C.FINALISTS_PER_FAMILY),
            "max_total": C.MAX_FINALISTS_TOTAL,
            "no_target_performance_used": True,
            "rule": "iterate families round-robin; within a family pick the Pareto "
                    "member maximizing the minimum scaled distance to already-picked "
                    "finalists; ties -> lexicographically smallest candidate_id; stop "
                    "at 5 per family or 30 total; if a family's applicable Pareto set "
                    "has fewer than 3 members, take all of them and report the shortfall"},
        "confirmation": {
            "engine": "Thermo-Calc 2025b / TC-Python 2025.2.30 / TCCU6 v6.2, 1 bar, "
                      "1 mol, wt.%, Cu balance, equilibrium only (frozen FULL_V1 runner)",
            "temperatures": C.CONFIRM_TEMPS,
            "max_states": C.MAX_FINALISTS_TOTAL * len(C.CONFIRM_TEMPS),
            "label": "POST-FREEZE NON-Ni screening confirmation; NOT a blind test; "
                     "never used to retrain or retune anything (spec 13, 34)"},
        "claim_language": {
            "allowed": C.ALLOWED_CANDIDATE_WORDS,
            "forbidden": C.FORBIDDEN_CANDIDATE_WORDS},
    }
    C.write_json(os.path.join(C.D_SCREEN, "STEP5K_SCREENING_RULES_FREEZE.json"), rules)
    print("  rules frozen -> STEP5K_SCREENING_RULES_FREEZE.json (before any prediction)")
    print("STAGE B1: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
