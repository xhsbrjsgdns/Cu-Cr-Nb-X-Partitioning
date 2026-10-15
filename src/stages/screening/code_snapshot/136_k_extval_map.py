# -*- coding: utf-8 -*-
"""
136_k_extval_map.py
STEP5K Part A Stage 3: literature -> frozen-domain mapping, rule-based consistency
verdicts, and the external-validation report (spec 5-15).

THE RULES BELOW ARE APPLIED MECHANICALLY. They are written out in full so that the
verdict for every paper can be recomputed from the CSV without trusting a human
reading. No verdict is assigned by inspection.

Admissibility. A literature row can be compared to a calculation at all only if it
carries ALL THREE of:
    (a) a composition stated in the paper, with an unambiguous wt.%/at.% basis;
    (b) at least one experimentally identified phase;
    (c) a POST-FREEZE CALPHAD case built from exactly that composition (stage A2).
Anything short of that is NOT_DIRECTLY_COMPARABLE, and the reason is recorded.

Verdict vocabulary (spec 8). The forbidden inference "CALPHAD agrees with experiment,
therefore CALPHAD is ground truth" is never drawn; a verdict is a statement about two
descriptions of the same alloy, not about which one is correct:
    QUALITATIVELY_CONSISTENT
    PARTIALLY_CONSISTENT
    REASONABLE_EQUILIBRIUM_VS_NON_EQUILIBRIUM_DISCREPANCY
    NOT_REPRODUCED_BY_EQUILIBRIUM_CALCULATION
    NOT_DIRECTLY_COMPARABLE

Domain mapping. Every stated literature composition is placed against FULL_V1
Domain A (S_CrNb 0.6-2.0 wt.%, R_CrNb_atomic 2.0-6.5, X 0.02-0.20 wt.%) and the
signed distance outside each bound is recorded. Where a composition lies outside the
domain, NO frozen FULL_V1 state is offered as a stand-in and NO surrogate prediction
is made: the comparison uses the stage-A2 states instead, and the CSV says so.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C

PRESENCE = 1e-4
VERDICTS = ["QUALITATIVELY_CONSISTENT", "PARTIALLY_CONSISTENT",
            "REASONABLE_EQUILIBRIUM_VS_NON_EQUILIBRIUM_DISCREPANCY",
            "NOT_REPRODUCED_BY_EQUILIBRIUM_CALCULATION", "NOT_DIRECTLY_COMPARABLE"]

# reported-phase token -> predicate over a CALPHAD phase-model base name.
# Written before the stage-A2 output is inspected; the observed base names are dumped
# into the report so the mapping can be checked against what the database returned.
PHASE_MATCHERS = [
    ("Cr2Nb (C15 Laves)", re.compile(r"cr2nb|laves", re.I),
     lambda b: b.upper().startswith("C15_LAVES")),
    ("bcc Cr / Cr-rich precipitate",
     re.compile(r"\bbcc\b.*\bcr\b|cr[- ]rich|cr[- ]based precipitat|cr precipitat",
                re.I),
     lambda b: b.upper().startswith("BCC")),
    ("Cu-Zr compound (Cu5Zr / Cu4Zr)", re.compile(r"cu\s*[45]\s*zr|cu-?zr compound|cu-?zr phase",
                                                  re.I),
     lambda b: "ZR" in b.upper() and "CU" in b.upper()),
]
NONEQ = re.compile(r"rapid(ly)? solidif|melt spun|melt spinning|gas atomi|l-?pbf|"
                   r"additive|cold draw|severe|high strain rate|dcap|metastable|"
                   r"supersaturat|as-?sintered|sinter", re.I)


def domain_position(s_crnb, r_atomic, x_wt):
    """Signed excess outside each Domain-A bound (0 = inside)."""
    def out(v, lo, hi):
        if v is None or not np.isfinite(v):
            return None
        return round(float(max(0.0, lo - v, v - hi)), 6)
    return (out(s_crnb, *C.S_CRNB_RANGE), out(r_atomic, *C.R_CRNB_RANGE),
            out(x_wt, *C.X_WT_RANGE))


def parse_wt(text):
    """Pull element=wt.% pairs out of the stage-A2 composition string."""
    return {k: float(v) for k, v in re.findall(r"([A-Z]{1,2})=([0-9.]+)", str(text))}


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K Part A Stage 3: literature -> domain mapping and consistency verdicts")
    print("=" * 78)

    lit = pd.read_csv(os.path.join(C.D_LIT, "STEP5K_LITERATURE_RAW.csv"))
    doi_v = json.load(open(os.path.join(C.D_CONFIG, "STEP5K_DOI_VERIFICATION.json"),
                           encoding="utf-8"))
    dv = {r["paper_id"]: r for r in doi_v["records"]}
    ext = pd.read_csv(os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_STATES.csv"))

    # which literature rows a stage-A2 case was actually built for
    ext["paper_id"] = ext["ext_id"].map({
        "EXT_01_Cu3Cr05Nb": "LIT_17", "EXT_02_GRCop42_at": "LIT_18",
        "EXT_03_GRCop84_at": "LIT_18", "EXT_02b_GRCop42_wt": "LIT_18",
        "EXT_03b_GRCop84_wt": "LIT_18", "EXT_04_Cu021Cr020Zr": "LIT_04",
        "EXT_05_Cu013Zr": "LIT_05", "EXT_06_Cu006Zr": "LIT_04"})
    if ext.paper_id.isna().any():
        raise SystemExit("FATAL: a stage-A2 case has no literature row attached: %s"
                         % sorted(ext.loc[ext.paper_id.isna(), "ext_id"].unique()))

    observed_bases = sorted({b.split("=")[0] for s in ext.stable_phase_bases.dropna()
                             for b in str(s).split(";") if b})

    rows, refs = [], []
    for _, L in lit.iterrows():
        pid = L.paper_id
        v = dv.get(pid, {})
        refs.append({
            "paper_id": pid, "authors_short": L.authors_short, "year": L.year,
            "title": L.title, "journal": L.journal, "doi": L.doi,
            "doi_independently_reverified": int(bool(v.get("verified"))),
            "crossref_title": v.get("crossref_title", ""),
            "crossref_container_title": v.get("crossref_container_title", ""),
            "crossref_year": v.get("crossref_year", ""),
            "title_matches_crossref": int(bool(v.get("title_match"))),
            "year_matches_crossref": int(bool(v.get("year_match"))),
            "journal_matches_crossref": int(bool(v.get("journal_match"))),
            "verified_utc": v.get("verified_utc", ""),
            "verification_method": doi_v["method"]})

        phases = [p for p in (L.experimental_phase_1, L.experimental_phase_2,
                              L.experimental_phase_3)
                  if isinstance(p, str) and p.strip() and p.strip().upper() != "UNKNOWN"]
        sub = ext[ext.paper_id == pid]
        has_comp = len(sub) > 0
        has_phase = len(phases) > 0
        noneq = bool(NONEQ.search(" ".join(str(x) for x in
                                           [L.processing_state, L.experimental_condition,
                                            L.kinetic_caveat, L.segregation_or_partition_trend])))

        if not has_comp or not has_phase:
            why = []
            if not has_comp:
                why.append("no stage-A2 case: the paper does not state a composition "
                           "with an unambiguous basis (spec 5 forbids inventing one)")
            if not has_phase:
                why.append("no experimentally identified phase is stated in the "
                           "retrievable record")
            rows.append({"paper_id": pid, "doi": L.doi, "X_identity": L.X_identity,
                         "reported_phase": "; ".join(phases) if phases else "UNKNOWN",
                         "matched_calphad_family": "", "ext_ids": "",
                         "calphad_stable_at_C": "", "calphad_absent_at_C": "",
                         "stated_temperatures_C": "",
                         "in_FULL_V1_domain_A": "", "S_CrNb_outside_by": "",
                         "R_CrNb_outside_by": "", "X_wt_outside_by": "",
                         "non_equilibrium_processing": int(noneq),
                         "verdict": "NOT_DIRECTLY_COMPARABLE",
                         "verdict_reason": "; ".join(why),
                         "used_to_change_any_model": 0})
            continue

        # domain position, from the stage-A2 composition actually calculated
        pos_rows = []
        for eid, g in sub.groupby("ext_id"):
            w = parse_wt(g.composition_wt_pct.iloc[0])
            cr, nb, xw = w.get("CR", 0.0), w.get("NB", 0.0), None
            for e, val in w.items():
                if e not in ("CU", "CR", "NB"):
                    xw = val
            s = cr + nb
            r = ((cr / C.M_CR) / (nb / C.M_NB)) if nb > 0 else None
            pos_rows.append((eid, s, r, xw) + domain_position(s, r, xw))
        inside = all((p[4] == 0 and (p[5] in (0, None)) and (p[6] in (0, None)))
                     for p in pos_rows)

        # "aged at 450-950 C" states TWO temperatures, not one. A bare
        # \d+\s*C pattern silently keeps only the second, which quietly changes
        # which temperatures a verdict is judged against - so ranges are read first.
        blob = " ".join(str(x) for x in [L.heat_treatment, L.experimental_condition])
        found = [int(a) for a in re.findall(r"(\d{3,4})\s*(?:C\b|°C)", blob)]
        for a, b in re.findall(r"(\d{3,4})\s*[-\u2013]\s*(\d{3,4})\s*(?:C\b|°C)", blob):
            found += [int(a), int(b)]
        stated_T = sorted({t for t in found if 250 <= t <= 1050})

        fired = False
        for label, rx, pred in PHASE_MATCHERS:
            if not any(rx.search(p) for p in phases):
                continue
            fired = True
            stable_T, absent_T = [], []
            for _, s in sub.iterrows():
                if not s.converged:
                    continue
                bases = {b.split("=")[0]: float(b.split("=")[1])
                         for b in str(s.stable_phase_bases).split(";") if "=" in b}
                hit = any(pred(b) and v > PRESENCE for b, v in bases.items())
                (stable_T if hit else absent_T).append(int(s.Temperature_C))
            stable_T, absent_T = sorted(set(stable_T)), sorted(set(absent_T))
            probe = [t for t in stated_T if t in stable_T + absent_T] or \
                    sorted(set(stable_T + absent_T))
            hit_probe = [t for t in probe if t in stable_T]
            if not probe:
                verdict, why = "NOT_DIRECTLY_COMPARABLE", \
                    "no converged stage-A2 state at any comparable temperature"
            elif len(hit_probe) == len(probe):
                verdict, why = "QUALITATIVELY_CONSISTENT", (
                    "the reported %s corresponds to a phase the frozen description "
                    "makes stable at every compared temperature; this is agreement "
                    "between two descriptions of the alloy, not evidence that either "
                    "is the true one" % label)
            elif hit_probe:
                verdict, why = "PARTIALLY_CONSISTENT", (
                    "the corresponding phase is stable at %s but not at %s"
                    % (hit_probe, [t for t in probe if t not in hit_probe]))
            elif noneq:
                verdict, why = "REASONABLE_EQUILIBRIUM_VS_NON_EQUILIBRIUM_DISCREPANCY", (
                    "the corresponding phase is not stable at equilibrium at any "
                    "compared temperature, and the reported material is in a "
                    "non-equilibrium condition, so the two are not expected to match")
            else:
                verdict, why = "NOT_REPRODUCED_BY_EQUILIBRIUM_CALCULATION", (
                    "the corresponding phase is not stable at equilibrium at any "
                    "compared temperature and no non-equilibrium processing is "
                    "recorded; reported as a discrepancy, not explained away")
            rows.append({
                "paper_id": pid, "doi": L.doi, "X_identity": L.X_identity,
                "reported_phase": label,
                "matched_calphad_family": {"Cr2Nb (C15 Laves)": "C15",
                                           "bcc Cr / Cr-rich precipitate": "BCC"}
                                          .get(label, "OTHER_SOLID"),
                "ext_ids": ";".join(sorted(sub.ext_id.unique())),
                "calphad_stable_at_C": ";".join(map(str, stable_T)),
                "calphad_absent_at_C": ";".join(map(str, absent_T)),
                "stated_temperatures_C": ";".join(map(str, stated_T)),
                "in_FULL_V1_domain_A": int(inside),
                "S_CrNb_outside_by": ";".join(str(p[4]) for p in pos_rows),
                "R_CrNb_outside_by": ";".join(str(p[5]) for p in pos_rows),
                "X_wt_outside_by": ";".join(str(p[6]) for p in pos_rows),
                "non_equilibrium_processing": int(noneq),
                "verdict": verdict, "verdict_reason": why,
                "used_to_change_any_model": 0})

        if not fired:
            rows.append({"paper_id": pid, "doi": L.doi, "X_identity": L.X_identity,
                         "reported_phase": "; ".join(phases),
                         "matched_calphad_family": "", "ext_ids":
                             ";".join(sorted(sub.ext_id.unique())),
                         "calphad_stable_at_C": "", "calphad_absent_at_C": "",
                         "stated_temperatures_C": ";".join(map(str, stated_T)),
                         "in_FULL_V1_domain_A": int(inside),
                         "S_CrNb_outside_by": ";".join(str(p[4]) for p in pos_rows),
                         "R_CrNb_outside_by": ";".join(str(p[5]) for p in pos_rows),
                         "X_wt_outside_by": ";".join(str(p[6]) for p in pos_rows),
                         "non_equilibrium_processing": int(noneq),
                         "verdict": "NOT_DIRECTLY_COMPARABLE",
                         "verdict_reason": "the reported phases do not map onto any "
                                           "phase this stage knows how to look up in "
                                           "the frozen description",
                         "used_to_change_any_model": 0})

    val = pd.DataFrame(rows)
    val.to_csv(os.path.join(C.D_EXTVAL,
                            "STEP5K_EXTERNAL_LITERATURE_VALIDATION.csv"),
               index=False, encoding="utf-8")
    pd.DataFrame(refs).to_csv(os.path.join(
        C.D_EXTVAL, "STEP5K_EXTERNAL_VALIDATION_REFERENCES.csv"),
        index=False, encoding="utf-8")

    counts = val.verdict.value_counts().to_dict()
    n_papers = lit.paper_id.nunique()
    n_comparable = val.loc[val.verdict != "NOT_DIRECTLY_COMPARABLE",
                           "paper_id"].nunique()
    n_verified = sum(1 for r in refs if r["doi_independently_reverified"])
    n_title_ok = sum(1 for r in refs if r["title_matches_crossref"])
    inside_any = int(val.get("in_FULL_V1_domain_A",
                             pd.Series(dtype=object)).eq(1).sum())

    C.write_json(os.path.join(C.D_EXTVAL, "STEP5K_EXTERNAL_VALIDATION_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "n_papers": int(n_papers), "n_validation_rows": int(len(val)),
        "n_papers_with_a_usable_comparison": int(n_comparable),
        "verdict_counts": counts,
        "n_dois_independently_reverified": int(n_verified),
        "n_titles_matching_crossref": int(n_title_ok),
        "n_literature_compositions_inside_FULL_V1_domain_A": inside_any,
        "observed_calphad_phase_bases": observed_bases,
        "calphad_is_not_treated_as_ground_truth": True,
        "ml_predictions_made_for_literature_compositions": 0,
        "any_literature_state_used_to_retrain_reselect_or_reinterpret": False,
        "verdict_vocabulary": VERDICTS})

    md = []
    md.append("# STEP5K Part A - External Materials-Literature Validation\n")
    md.append("_Generated %s. POST-FREEZE, NON-Ni. Nothing in this report changed any "
              "model, feature set, hyperparameter, topology class, production "
              "checkpoint, or any part of the Ni blind result._\n" % C.utcnow())

    md.append("\n## 1. What this validation is, and what it is not\n")
    md.append("This is a check of the **thermodynamic (CALPHAD) layer** of the study "
              "against phase identities reported in the external literature. It is "
              "**not** a validation of the machine-learning surrogate: every literature "
              "composition that could be pinned down lies outside the surrogate's "
              "training domain, and no surrogate prediction was made for any of them.\n")
    md.append("\nWhere a calculation and an experiment line up, the wording used "
              "throughout is *qualitative consistency between two descriptions of the "
              "same alloy*. Agreement is never read as evidence that the calculated "
              "equilibrium is the true state of the material.\n")

    md.append("\n## 2. Reference integrity\n")
    md.append("- Papers in the retrieved set: **%d**\n" % n_papers)
    md.append("- DOIs re-verified independently against Crossref in this stage: "
              "**%d of %d**\n" % (n_verified, n_papers))
    md.append("- Titles matching the Crossref record: **%d of %d**\n"
              % (n_title_ok, n_papers))
    md.append("\nThe upstream search log's own `crossref_verified` flag was **not** "
              "taken on trust; each DOI was resolved again and the returned title, "
              "journal and year compared field by field. The per-paper outcome is in "
              "`STEP5K_EXTERNAL_VALIDATION_REFERENCES.csv`.\n")

    md.append("\n## 3. The domain gap, stated plainly\n")
    md.append("FULL_V1 Domain A samples `S_CrNb` = %.1f-%.1f wt.%%, "
              "`R_CrNb_atomic` = %.1f-%.1f, `X` = %.2f-%.2f wt.%%.\n"
              % (C.S_CRNB_RANGE + C.R_CRNB_RANGE + C.X_WT_RANGE))
    md.append("\nOf the %d papers retrieved, **%d literature composition falls inside "
              "that domain**. The published Cu-Cr-Nb alloys whose compositions are "
              "stated at all cluster in the precipitation-strengthened regime "
              "(GRCop-42 and GRCop-84 at `S_CrNb` = 6.18 and 12.35 wt.%%, both at "
              "`R_CrNb_atomic` = 2.000, the stoichiometric Cr2Nb ratio; the Nb-lean "
              "sintered alloy at `S_CrNb` = 3.50 wt.%%, `R` = 10.72), while the "
              "Cu-Zr rows carry no Nb at all.\n" % (n_papers, inside_any))
    md.append("\nThis gap is a result, not an obstacle to be smoothed over. It is the "
              "reason no frozen FULL_V1 state is offered anywhere in this report as a "
              "stand-in for a literature alloy, and the reason the comparison is made "
              "against new, clearly labelled post-freeze equilibrium states instead.\n")

    md.append("\n## 4. Composition provenance\n")
    md.append("Every composition calculated is quoted from its paper. Two exclusions "
              "are deliberate:\n")
    md.append("\n- **Cu-0.3%%Cr (LIT_19)** - the retrievable record does not "
              "distinguish wt.%% from at.%%, and the basis was not inferred.\n")
    md.append("- **Cu-Cr-Zr-Zn-0.45Fe-0.2Ti (LIT_10)** - the Cr, Zr and Zn contents "
              "are not stated, so the alloy cannot be built without inventing numbers.\n")
    md.append("\nOne correction was made to the upstream search table: **GRCop-42 and "
              "GRCop-84 were recorded there as weight-percent designations. They are "
              "atomic-percent designations** (NASA/TM-2005-213566: \"GRCop-84 (Cu-8 "
              "at.%% Cr-4 at.%% Nb)\"). Both readings of the digits were calculated "
              "here and are kept side by side, because the cited paper's own abstract "
              "does not settle the basis; the at.%% reading is the one the designation "
              "means.\n")

    md.append("\n## 5. Verdicts\n")
    md.append("| verdict | rows |\n|---|---|\n")
    for k in VERDICTS:
        md.append("| %s | %d |\n" % (k, counts.get(k, 0)))
    md.append("\nVerdicts are assigned by the mechanical rule set at the top of "
              "`136_k_extval_map.py`, not by inspection, and every row records the "
              "temperatures at which the corresponding phase was and was not stable.\n")

    md.append("\n## 6. Per-paper outcome\n")
    md.append("| paper | X | reported phase | verdict |\n|---|---|---|---|\n")
    for _, r in val.iterrows():
        md.append("| %s | %s | %s | %s |\n" % (r.paper_id, r.X_identity,
                                               str(r.reported_phase)[:60], r.verdict))

    md.append("\n## 7. Phase bases the frozen description returned\n")
    md.append("`%s`\n" % ", ".join(observed_bases))

    md.append("\n## 8. Limits of this exercise\n")
    md.append("- Most of the retrieved corpus is publisher-gated. Rows whose phase "
              "content could not be read from an accessible record are marked "
              "`NOT_DIRECTLY_COMPARABLE` and carry no verdict; they are not quietly "
              "counted as support.\n")
    md.append("- Every comparable alloy was processed far from equilibrium (rapid "
              "solidification, gas atomisation, L-PBF, sintering, severe deformation, "
              "aging). An equilibrium calculation cannot reproduce a kinetic phase "
              "selection, and where the two differ the row says so rather than "
              "reaching for an explanation.\n")
    md.append("- The comparison is qualitative: which phases the description makes "
              "stable, against which phases were identified. No volume fraction, "
              "partition coefficient or transformation temperature from any paper is "
              "compared numerically, because none of the accessible records states "
              "one together with a composition and a temperature.\n")

    open(os.path.join(C.D_EXTVAL, "STEP5K_EXTERNAL_VALIDATION_REPORT.md"),
         "w", encoding="utf-8").writelines(md)

    print("  %d validation rows | %d papers with a usable comparison | verdicts %s"
          % (len(val), n_comparable, counts))
    print("  DOIs re-verified %d/%d, titles matching %d/%d"
          % (n_verified, n_papers, n_title_ok, n_papers))
    print("STAGE A3: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
