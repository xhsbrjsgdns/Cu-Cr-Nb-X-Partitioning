# -*- coding: utf-8 -*-
"""
134_k_screen_compare.py
STEP5K Part B Stage 5: descriptive ML-vs-CALPHAD comparison on the confirmed
finalists (spec 32-36).

WHAT THIS IS. A post-freeze consistency check: for the <=30 compositions the frozen
surrogate itself selected, how close is its prediction to the equilibrium the same
frozen CALPHAD runner returns?

WHAT THIS IS NOT, and the report says so in these words. The finalists were chosen
BY the model, from the extreme of its own predicted objective space, so this is a
SELECTED, NON-BLIND subset. Agreement here is not a generalization estimate, is not
comparable to the prospective Ni blind numbers, and cannot be quoted as a test-set
error. It is reported only as: on the compositions this screening puts forward, does
the surrogate's equilibrium picture survive an actual equilibrium calculation.

Nothing computed here is fed back: no retraining, no re-ranking, no reselection, no
threshold change (spec 13, 34). The candidate list is not re-sorted by the CALPHAD
result - CALPHAD_confirmed is a flag added to the frozen ordering, nothing more.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C

FAM = C.FAMILIES_ORDER
PRESENCE = 1e-4


def tvd(a, b):
    """Total variation distance between two 5-family vectors (0.5 * L1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    return 0.5 * float(np.abs(a - b).sum())


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K Part B Stage 5: descriptive ML vs CALPHAD on the finalists")
    print("=" * 78)

    cand = pd.read_csv(os.path.join(C.D_SCREEN,
                                    "STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv"))
    conf = pd.read_csv(os.path.join(C.D_CONFIRM, "STEP5K_CONFIRM_PROCESSED.csv"))
    space = pd.read_parquet(os.path.join(C.D_SCREEN,
                                         "STEP5K_VIRTUAL_SCREENING_SPACE.parquet"))
    space = space[space.candidate_id.isin(cand.candidate_id)].copy()
    if space.candidate_id.nunique() != len(cand):
        raise SystemExit("FATAL: the prediction space does not cover every finalist.")

    pred_cols = (["frac_%s" % f for f in FAM] + ["Cr_P_%s" % f for f in FAM]
                 + ["X_P_%s" % f for f in FAM])
    m = conf.merge(
        space[["candidate_id", "Temperature_C"] + pred_cols].rename(
            columns={c: "pred_%s" % c for c in pred_cols}),
        on=["candidate_id", "Temperature_C"], how="left", validate="one_to_one")
    missing = m[[c for c in m.columns if c.startswith("pred_")]].isna().all(axis=1)
    if missing.any():
        raise SystemExit("FATAL: %d confirmation states have no matching prediction."
                         % int(missing.sum()))

    # per-state descriptive distances (converged states only; failures are KEPT as
    # rows with NaN distances and are counted separately - never dropped)
    for tag, prefix in (("family", "frac"), ("Cr_partition", "Cr_P"),
                        ("X_partition", "X_P")):
        m["TVD_%s" % tag] = [
            tvd([r["pred_%s_%s" % (prefix, f)] for f in FAM],
                [r["calphad_%s_%s" % (prefix, f)] for f in FAM])
            for _, r in m.iterrows()]
    for f in FAM:
        m["abs_err_frac_%s" % f] = (m["pred_frac_%s" % f]
                                    - m["calphad_frac_%s" % f]).abs()
        m["abs_err_Cr_P_%s" % f] = (m["pred_Cr_P_%s" % f]
                                    - m["calphad_Cr_P_%s" % f]).abs()

    def dom(row, prefix):
        v = {f: row["%s_%s_%s" % (prefix, "frac", f)] for f in FAM}
        if not np.isfinite(list(v.values())).all():
            return ""
        return max(v, key=v.get)

    m["pred_dominant_family"] = [dom(r, "pred") for _, r in m.iterrows()]
    m["calphad_dominant_family"] = [dom(r, "calphad") for _, r in m.iterrows()]
    m["dominant_family_match"] = np.where(
        m.converged == 1,
        (m.pred_dominant_family == m.calphad_dominant_family).astype(int), -1)

    def pres(row, prefix):
        return tuple(int(row["%s_frac_%s" % (prefix, f)] > PRESENCE) for f in FAM) \
            if np.isfinite([row["%s_frac_%s" % (prefix, f)] for f in FAM]).all() else None

    pp = [pres(r, "pred") for _, r in m.iterrows()]
    cp = [pres(r, "calphad") for _, r in m.iterrows()]
    m["presence_pattern_pred"] = ["".join(map(str, p)) if p else "" for p in pp]
    m["presence_pattern_calphad"] = ["".join(map(str, p)) if p else "" for p in cp]
    m["presence_pattern_match"] = np.where(
        m.converged == 1, [int(a == b and a is not None) for a, b in zip(pp, cp)], -1)

    m.to_csv(os.path.join(C.D_CONFIRM,
                          "STEP5K_SCREENING_CALPHAD_CONFIRMATION.csv"),
             index=False, encoding="utf-8")

    ok = m[m.converged == 1]
    rules = json.load(open(os.path.join(C.D_SCREEN,
                                        "STEP5K_SCREENING_RULES_FREEZE.json"),
                           encoding="utf-8"))
    band = rules["moderate_c15_band"]
    lowT = ok[ok.Temperature_C.isin([450, 600])]
    in_band = lowT.groupby("candidate_id")["calphad_frac_C15"].apply(
        lambda s: bool(len(s) == 2 and s.between(band["band_low"],
                                                 band["band_high"]).all()))

    def rho(a, b):
        d = pd.DataFrame({"a": a, "b": b}).dropna()
        return float(d.a.corr(d.b, method="spearman")) if len(d) > 2 else None

    a950 = ok[ok.Temperature_C == 950]
    summary = {
        "generated_utc": C.utcnow(),
        "WHAT_THIS_IS": "descriptive post-freeze equilibrium check on the finalists",
        "WHAT_THIS_IS_NOT": ("not a blind test, not a generalization estimate, not "
                             "comparable to the prospective Ni blind metric: these "
                             "compositions were selected BY the frozen surrogate from "
                             "the extreme of its own predicted objective space, so the "
                             "subset is selected and non-blind"),
        "used_to_retrain_reselect_or_rerank": False,
        "n_finalists": int(cand.candidate_id.nunique()),
        "n_states": int(len(m)), "n_converged": int(len(ok)),
        "n_failed_kept_not_deleted": int((m.converged == 0).sum()),
        "TVD_family_mean": float(ok.TVD_family.mean()),
        "TVD_family_median": float(ok.TVD_family.median()),
        "TVD_family_max": float(ok.TVD_family.max()),
        "TVD_Cr_partition_mean": float(ok.TVD_Cr_partition.mean()),
        "TVD_X_partition_mean": float(ok.TVD_X_partition.mean()),
        "MAE_frac_per_family": {f: float(ok["abs_err_frac_%s" % f].mean())
                                for f in FAM},
        "MAE_Cr_partition_per_family": {f: float(ok["abs_err_Cr_P_%s" % f].mean())
                                        for f in FAM},
        "dominant_family_agreement": float((ok.dominant_family_match == 1).mean()),
        "presence_pattern_agreement": float((ok.presence_pattern_match == 1).mean()),
        "moderate_C15_band": {"low": band["band_low"], "high": band["band_high"],
                              "n_finalists_with_both_450_600_in_band":
                                  int(in_band.sum()),
                              "n_finalists_evaluated": int(len(in_band))},
        "spearman_pred_vs_calphad_Cr_P_FCC_950": rho(a950.pred_Cr_P_FCC,
                                                     a950.calphad_Cr_P_FCC),
        "per_temperature_TVD_family_mean": {
            str(int(t)): float(g.TVD_family.mean())
            for t, g in ok.groupby("Temperature_C")},
        "per_identity_TVD_family_mean": {
            str(k): float(g.TVD_family.mean())
            for k, g in ok.groupby("X_identity")}}
    C.write_json(os.path.join(C.D_CONFIRM, "STEP5K_CONFIRMATION_SUMMARY.json"),
                 summary)

    # add the confirmation flag to the frozen candidate ordering WITHOUT re-sorting it
    flag = (ok.groupby("candidate_id").size() == len(C.CONFIRM_TEMPS))
    cand["CALPHAD_confirmed"] = cand.candidate_id.map(flag).fillna(False).astype(int)
    cand["CALPHAD_C15_in_moderate_band"] = cand.candidate_id.map(
        in_band).fillna(False).astype(int)
    cand.to_csv(os.path.join(C.D_SCREEN,
                             "STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv"),
                index=False, encoding="utf-8")

    print("  states %d (converged %d) | mean family TVD %.6f | dominant-family "
          "agreement %.4f" % (len(m), len(ok), summary["TVD_family_mean"],
                              summary["dominant_family_agreement"]))
    print("  finalists with CALPHAD C15 inside the pre-frozen band at BOTH 450 and "
          "600 C: %d of %d" % (int(in_band.sum()), int(len(in_band))))
    print("STAGE B5: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
