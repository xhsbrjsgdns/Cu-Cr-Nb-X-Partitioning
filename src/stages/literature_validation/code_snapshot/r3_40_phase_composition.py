# -*- coding: utf-8 -*-
"""
r3_40_phase_composition.py
STEP5K-R3 stage 40: measured EPMA/EDS phase compositions against the calculated ones
(spec 33), and the chemistry gate on the set-level verdict.

PHASE MATCHING IS BY STRUCTURE THEN CHEMISTRY, NEVER BY ERROR. An experimental phase
is matched to a calculated phase only through the resolved crosswalk. Where the
calculation returns several instances of the same phase model - the Cu-rich and
Ag-rich FCC pair, the Cr- and Mo-rich BCC pair - the instance is chosen as the one
richest in the element that dominates the EXPERIMENTAL phase. Choosing whichever
instance minimised the error would be fitting the comparison to the answer, and is
not done anywhere in this file. Where the phase identity is ambiguous, or the
calculation offers no instance of an admissible phase, no match is forced and the row
is recorded as unmatched.

THE CHEMISTRY GATE runs one way. A condition whose phase sets agree exactly but whose
matched phase compositions disagree by more than the pre-declared tolerance cannot be
STRONGLY_CONSISTENT; it becomes CONSISTENT. A condition whose every matched phase
disagrees by more than twice that tolerance falls to PARTIALLY_CONSISTENT. Nothing is
ever upgraded, and the pre-gate verdict is kept beside the final one.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C

OUT = os.path.join(C.D_PHCOMP, "STEP5K_R3_PHASE_COMPOSITION_COMPARISON.csv")
CMP = os.path.join(C.D_CMP, "STEP5K_R3_CONDITION_COMPARISON.csv")
MAPPED_OK = ("CONFIRMED_IN_DATABASE", "CONFIRMED_BUT_CONSTITUENTS_DIFFER",
             "RESOLVED_BY_DATABASE_LOOKUP", "AMBIGUOUS_POLYMORPH_ADMITS_EITHER")


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K-R3 stage 40: measured vs calculated phase compositions")
    print("=" * 78)
    n_prot, _, _ = C.verify_protected()
    print("  %d protected artifacts re-hashed, all unchanged" % n_prot)

    I = C.read_inputs()
    cw = I["crosswalk"]
    resolved = "resolution_status" in cw.columns
    admissible = {}
    for _, r in cw.iterrows():
        if not resolved or str(r.resolution_status) not in MAPPED_OK:
            continue
        lit = str(r.literature_phase).strip()
        admissible.setdefault(lit, set()).update(
            {x.strip() for x in str(r.resolved_TCCU_raw_phase).split("|") if x.strip()})

    calc = pd.read_parquet(os.path.join(C.D_PROC,
                                        "STEP5K_R3_EXTERNAL_CALPHAD.parquet"))
    conv = calc[calc.execution_status == "CONVERGED"].set_index("condition_id")
    pc_path = os.path.join(C.D_RAW, "PHASE_COMPOSITIONS.csv")
    pcomp = pd.read_csv(pc_path) if os.path.exists(pc_path) else pd.DataFrame()
    exp = I["expcomp"]

    rows = []
    for _, er in exp.iterrows():
        cid = er.condition_id
        lit = str(er.literature_phase).strip()
        try:
            eat = {k.upper(): float(v)
                   for k, v in json.loads(er.comparison_at_pct).items()}
        except Exception:                                         # noqa: BLE001
            eat = {}
        d = {"condition_id": cid, "source_id": er.source_id,
             "phase_index": er.phase_index, "literature_phase": lit,
             "experimental_at_pct": json.dumps(eat),
             "page": er.page, "table": er.table,
             "admissible_calculated_phases": ";".join(sorted(
                 admissible.get(lit, set())))}
        if cid not in conv.index:
            if cid not in set(calc.condition_id):
                ms = "NOT_SELECTED"          # outside the frozen 120-state budget
            else:
                st = calc.loc[calc.condition_id == cid, "execution_status"].iloc[0]
                ms = "PENDING" if st == "PENDING" else "NO_CONVERGED_CALCULATION"
            d.update(match_status=ms,
                     matched_calculated_phase="", calculated_at_pct="",
                     max_abs_error_at_pct=np.nan, MAE_at_pct=np.nan,
                     experimental_dominant_element="", calculated_dominant_element="",
                     dominant_enrichment_agrees="", chemistry_compatible="")
            rows.append(d)
            continue
        if not resolved or not admissible.get(lit):
            d.update(match_status="UNMAPPABLE_PHASE", matched_calculated_phase="",
                     calculated_at_pct="", max_abs_error_at_pct=np.nan,
                     MAE_at_pct=np.nan, experimental_dominant_element="",
                     calculated_dominant_element="", dominant_enrichment_agrees="",
                     chemistry_compatible="")
            rows.append(d)
            continue

        s = conv.loc[cid]
        names = {x for x in str(s.calculated_raw_instances).split(";") if x}
        cands = [n for n in names if C.base_phase(n) in admissible[lit]]
        if not cands:
            d.update(match_status="PHASE_NOT_CALCULATED",
                     matched_calculated_phase="", calculated_at_pct="",
                     max_abs_error_at_pct=np.nan, MAE_at_pct=np.nan,
                     experimental_dominant_element=(max(eat, key=eat.get)
                                                    if eat else ""),
                     calculated_dominant_element="", dominant_enrichment_agrees="",
                     chemistry_compatible="")
            rows.append(d)
            continue

        g = pcomp[pcomp.state_id == s.state_id] if len(pcomp) else pd.DataFrame()
        cat = {}
        for ph, gg in g.groupby("raw_phase_name") if len(g) else []:
            if ph in cands:
                cat[ph] = {str(x.element).upper():
                           100.0 * float(x.element_mole_fraction)
                           for _, x in gg.iterrows()
                           if np.isfinite(x.element_mole_fraction)}
        if not cat:
            d.update(match_status="NO_CALCULATED_COMPOSITION",
                     matched_calculated_phase=";".join(sorted(cands)),
                     calculated_at_pct="", max_abs_error_at_pct=np.nan,
                     MAE_at_pct=np.nan,
                     experimental_dominant_element=(max(eat, key=eat.get)
                                                    if eat else ""),
                     calculated_dominant_element="", dominant_enrichment_agrees="",
                     chemistry_compatible="")
            rows.append(d)
            continue

        edom = max(eat, key=eat.get) if eat else None
        if len(cat) == 1 or edom is None:
            pick = sorted(cat)[0]
            basis = "only one calculated instance of an admissible phase"
        else:
            pick = max(cat, key=lambda p: cat[p].get(edom, -1.0))
            basis = ("several instances of the same phase model; chosen as the one "
                     "richest in %s, the element that dominates the experimental "
                     "phase" % edom)
        c = cat[pick]
        els = sorted(set(eat) | set(c))
        errs = {e: abs(eat.get(e, 0.0) - c.get(e, 0.0)) for e in els}
        cdom = max(c, key=c.get) if c else ""
        d.update(match_status="MATCHED", matched_calculated_phase=pick,
                 match_basis=basis,
                 calculated_at_pct=json.dumps({k: round(v, 4)
                                               for k, v in sorted(c.items())}),
                 element_abs_errors_at_pct=json.dumps({k: round(v, 4)
                                                       for k, v in errs.items()}),
                 max_abs_error_at_pct=float(max(errs.values())) if errs else np.nan,
                 MAE_at_pct=float(np.mean(list(errs.values()))) if errs else np.nan,
                 experimental_dominant_element=edom,
                 calculated_dominant_element=cdom,
                 dominant_enrichment_agrees=int(edom == cdom),
                 chemistry_compatible=int(max(errs.values())
                                          <= C.PHASE_CHEM_TOL_AT_PCT)
                 if errs else "")
        rows.append(d)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")

    # ---- chemistry gate on the set-level verdict --------------------------- #
    cmp_ = pd.read_csv(CMP)
    # stage 30 writes chemistry_gate empty, so pandas reads it back as all-NaN
    # float64 and refuses a string assignment. Coerce before writing into it.
    for col in ("chemistry_gate", "verdict", "verdict_set_level"):
        if col in cmp_.columns:
            cmp_[col] = cmp_[col].astype(object).where(cmp_[col].notna(), "")
    matched = out[out.match_status == "MATCHED"]
    gate, downgraded = {}, 0
    for cid, g in matched.groupby("condition_id"):
        worst = g.max_abs_error_at_pct.max()
        allbad = bool((g.max_abs_error_at_pct
                       > 2 * C.PHASE_CHEM_TOL_AT_PCT).all())
        gate[cid] = ("ALL_MATCHED_PHASES_EXCEED_2X_TOLERANCE" if allbad
                     else "AT_LEAST_ONE_MATCHED_PHASE_EXCEEDS_TOLERANCE"
                     if worst > C.PHASE_CHEM_TOL_AT_PCT else "WITHIN_TOLERANCE")
    for i, r in cmp_.iterrows():
        g = gate.get(r.condition_id)
        cmp_.at[i, "chemistry_gate"] = g or ""
        if not g or r.verdict_set_level not in ("STRONGLY_CONSISTENT", "CONSISTENT"):
            continue
        if g == "ALL_MATCHED_PHASES_EXCEED_2X_TOLERANCE":
            cmp_.at[i, "verdict"] = "PARTIALLY_CONSISTENT"
            downgraded += 1
        elif (g == "AT_LEAST_ONE_MATCHED_PHASE_EXCEEDS_TOLERANCE"
              and r.verdict_set_level == "STRONGLY_CONSISTENT"):
            cmp_.at[i, "verdict"] = "CONSISTENT"
            downgraded += 1
    worst_by_cond = matched.groupby("condition_id").max_abs_error_at_pct.max()
    mae_by_cond = matched.groupby("condition_id").MAE_at_pct.mean()
    cmp_["phase_composition_max_abs_error_at_pct"] = cmp_.condition_id.map(
        worst_by_cond)
    cmp_["phase_composition_MAE_at_pct"] = cmp_.condition_id.map(mae_by_cond)
    cmp_["n_matched_phase_compositions"] = cmp_.condition_id.map(
        matched.groupby("condition_id").size()).fillna(0).astype(int)
    bad = sorted(set(cmp_.verdict.dropna().astype(str)) - set(C.VERDICTS) - {""})
    if bad:
        raise SystemExit("FATAL: verdict outside the frozen vocabulary: %s" % bad)
    cmp_.to_csv(CMP, index=False, encoding="utf-8")

    C.write_json(os.path.join(C.D_PHCOMP,
                              "STEP5K_R3_PHASE_COMPOSITION_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "n_experimental_phase_compositions": int(len(out)),
        "n_matched": int(len(matched)),
        "match_status_counts": out.match_status.value_counts().to_dict(),
        "tolerance_at_pct": C.PHASE_CHEM_TOL_AT_PCT,
        "tolerance_declared_before_any_result": True,
        "matching_rule": "structure via the resolved crosswalk, then the instance "
                         "richest in the experimental phase's dominant element; "
                         "never the instance that minimises the error",
        "MAE_at_pct_overall": (float(matched.MAE_at_pct.mean())
                               if len(matched) else None),
        "MAE_at_pct_by_source": (matched.groupby("source_id").MAE_at_pct.mean()
                                 .round(4).to_dict() if len(matched) else {}),
        "max_abs_error_at_pct_by_source": (
            matched.groupby("source_id").max_abs_error_at_pct.max().round(4).to_dict()
            if len(matched) else {}),
        "dominant_enrichment_agreement_by_source": (
            matched.assign(dominant_enrichment_agrees=pd.to_numeric(
                matched.dominant_enrichment_agrees, errors="raise"))
            .groupby("source_id").dominant_enrichment_agrees.mean()
            .round(4).to_dict() if len(matched) else {}),
        "n_conditions_downgraded_by_chemistry_gate": int(downgraded),
        "gate_can_only_downgrade": True})

    print("  %d experimental phase compositions | matched %d | %s"
          % (len(out), len(matched), out.match_status.value_counts().to_dict()))
    print("  chemistry gate downgraded %d condition verdicts" % downgraded)
    print("STAGE 40: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
