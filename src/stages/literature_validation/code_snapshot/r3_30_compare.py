# -*- coding: utf-8 -*-
"""
r3_30_compare.py
STEP5K-R3 stage 30: condition-level comparison at BOTH the raw-phase-instance level
and the frozen five-family level (spec 14-16, 30-32, 34-37).

The rules below are constants in this file. They were written before any R3
equilibrium existed - the engine had produced nothing - and stage 30 seals them, with
this file's own sha256, to STEP5K_R3_COMPARISON_RULES_FREEZE.json, so the recorded
rule set cannot drift from the code that applied it.

MAPPING. An experimental phase enters the metric only through the resolved crosswalk.
  * a phase resolved to one database phase contributes that phase;
  * Cr2Nb contributes {C14_LAVES, C15_LAVES} and counts as reproduced if EITHER is
    calculated, because the sources identify the compound and not the polymorph, and
    the frozen map sends C14 to OTHER_SOLID but C15 to C15;
  * a phase absent from the database contributes nothing and is counted as MISSING -
    a real result about the description's scope;
  * a phase still unresolved contributes nothing and is counted as UNMAPPABLE, which
    is excluded from the metric denominator and reported separately. It is never
    force-matched to whichever calculated phase happens to be there.

VERDICT (set level), applied in this order:
  not selected by the frozen inclusion/budget rules  -> NOT_COMPARABLE
  no calculation submitted                           -> (no verdict; PENDING)
  calculation failed                                 -> CALPHAD_FAILED
  no mappable experimental phase                     -> NOT_COMPARABLE
  nothing missing, nothing extra, multiplicity ok    -> STRONGLY_CONSISTENT
  nothing missing, multiplicity ok, extra phases     -> CONSISTENT
  at least one experimental phase reproduced         -> PARTIALLY_CONSISTENT
  none reproduced                                    -> DISCREPANT

MULTIPLICITY is judged only where the paper itself reports two or more instances of
one family - the Cu-rich/Ag-rich FCC pair, the Cr/Mo-rich BCC pair. Where a paper
reports one, multiplicity is not applicable and extra calculated instances are
already carried by the "extra phases" term.

Stage 40 may downgrade a set-level verdict on measured phase chemistry. It can never
upgrade one, and both columns are kept.
"""
import json
import os
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C

OUT = os.path.join(C.D_CMP, "STEP5K_R3_CONDITION_COMPARISON.csv")
RULES = os.path.join(C.D_CMP, "STEP5K_R3_COMPARISON_RULES_FREEZE.json")
MAPPED_OK = ("CONFIRMED_IN_DATABASE", "CONFIRMED_BUT_CONSTITUENTS_DIFFER",
             "RESOLVED_BY_DATABASE_LOOKUP", "AMBIGUOUS_POLYMORPH_ADMITS_EITHER")


def build_map(cw):
    """literature phase label -> {'admissible': {db phase names}, 'families': {...},
    'status': str}. Labels appearing in more than one source row are unioned, and a
    genuine conflict is recorded rather than silently resolved."""
    m, conflicts = {}, []
    resolved = "resolution_status" in cw.columns
    for _, r in cw.iterrows():
        lit = str(r.literature_phase).strip()
        if resolved:
            status = str(r.resolution_status)
            names = {x.strip() for x in str(r.resolved_TCCU_raw_phase).split("|")
                     if x.strip()}
            fams = {x.strip() for x in str(r.admissible_frozen_family).split("|")
                    if x.strip()}
        else:
            status = "PENDING_ENGINE_CHECK"
            names, fams = set(), set()
        e = m.setdefault(lit, {"admissible": set(), "families": set(),
                               "status": status})
        if e["status"] != status:
            conflicts.append((lit, e["status"], status))
            e["status"] = "CONFLICTING_CROSSWALK_ROWS"
        e["admissible"] |= names
        e["families"] |= fams
    return m, conflicts


def jaccard(matched, missing, extra):
    d = len(matched) + len(missing) + len(extra)
    return float(len(matched)) / d if d else np.nan


def topology_from(compdict, labelmap):
    """Which family each element is most enriched in, from measured or calculated
    phase compositions. Returns {'CR': 'BCC', ...} plus any family that carries two
    or more coexisting phases (composition separation)."""
    best, sep = {}, set()
    fams = {}
    for label, at in compdict.items():
        fam = labelmap.get(label, set())
        fam = sorted(fam)[0] if len(fam) == 1 else ("|".join(sorted(fam))
                                                    if fam else "")
        fams.setdefault(fam, []).append(label)
        for el, v in at.items():
            if not np.isfinite(v):
                continue
            if el not in best or v > best[el][1]:
                best[el] = (fam, v)
    for fam, labels in fams.items():
        if fam and "|" not in fam and len(labels) >= 2:
            sep.add(fam)
    return {el: fam for el, (fam, _) in best.items() if fam}, sep


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K-R3 stage 30: condition-level comparison")
    print("=" * 78)
    n_prot, _, _ = C.verify_protected()
    print("  %d protected artifacts re-hashed, all unchanged" % n_prot)

    I = C.read_inputs()
    cond = I["conditions"]
    sel = I["selection"].set_index("condition_id")
    cw = I["crosswalk"]
    dom = I["domain"].set_index("condition_id")
    pmap, conflicts = build_map(cw)
    crosswalk_resolved = "resolution_status" in cw.columns
    if conflicts:
        print("  WARNING: %d literature labels carry conflicting crosswalk rows: %s"
              % (len(conflicts), conflicts[:3]))

    calc = pd.read_parquet(os.path.join(C.D_PROC,
                                        "STEP5K_R3_EXTERNAL_CALPHAD.parquet"))
    calc_by_cond = calc.set_index("condition_id")
    pc_path = os.path.join(C.D_RAW, "PHASE_COMPOSITIONS.csv")
    pcomp = pd.read_csv(pc_path) if os.path.exists(pc_path) else pd.DataFrame()
    exp = I["expcomp"]

    rows = []
    for _, r in cond.iterrows():
        cid = r.condition_id
        selected = int(sel.loc[cid, "selected"]) if cid in sel.index else 0
        eligible = int(r.direct_eligible) if not pd.isna(r.direct_eligible) else 0
        lit_phases = C.experimental_phases(r)
        d = dict(
            condition_id=cid, source_id=r.source_id, system=r.system,
            X_identity=r.X_identity, temperature_C=r.temperature_C,
            temperature_K=r.temperature_K,
            direct_eligible=eligible, selected=selected,
            system_relation_class=r.system_relation_class,
            TARGET_DOMAIN_RELEVANCE=(dom.loc[cid, "TARGET_DOMAIN_RELEVANCE"]
                                     if cid in dom.index else ""),
            experimental_raw_phase_set=";".join(lit_phases),
            page=r.page, table=r.table)

        if cid not in calc_by_cond.index:
            d.update(execution_status="NOT_SELECTED", verdict="NOT_COMPARABLE",
                     verdict_set_level="NOT_COMPARABLE",
                     notes=(str(sel.loc[cid, "selection_reason"])
                            if cid in sel.index else
                            "not part of the frozen request set"))
            rows.append(d)
            continue

        s = calc_by_cond.loc[cid]
        status = s.execution_status
        d["execution_status"] = status
        d["solver_level"] = s.solver_level
        d["calphad_error"] = s.error_message
        calc_names = {x for x in str(s.calculated_raw_instances).split(";") if x}
        calc_bases = {C.base_phase(x) for x in calc_names}
        calc_fams = {x for x in str(s.calculated_family_set).split(";") if x}
        for f in C.FAMILIES:
            d["n_%s_raw_instances" % f] = s.get("n_%s_raw_instances" % f, 0)
        d["calculated_raw_phase_set"] = ";".join(sorted(calc_bases))
        d["calculated_raw_instances"] = s.calculated_raw_instances
        d["calculated_family_set"] = ";".join(sorted(calc_fams))

        matched, missing, unmappable, absent = set(), [], [], []
        exp_fams, fam_matched, fam_missing = set(), set(), []
        exp_fam_counts = {}
        for lit in lit_phases:
            e = pmap.get(lit)
            if e is None or e["status"] == "PENDING_ENGINE_CHECK":
                unmappable.append(lit)
                continue
            if e["status"] == "ABSENT_FROM_DATABASE":
                absent.append(lit)
                missing.append(lit)
                continue
            if e["status"] not in MAPPED_OK or not e["admissible"]:
                unmappable.append(lit)
                continue
            hit = e["admissible"] & calc_bases
            if hit:
                matched |= hit
            else:
                missing.append(lit)
            if len(e["families"]) == 1:
                f = next(iter(e["families"]))
                exp_fams.add(f)
                exp_fam_counts[f] = exp_fam_counts.get(f, 0) + 1
                (fam_matched.add(f) if f in calc_fams else fam_missing.append(f))
            elif e["families"]:
                hitf = e["families"] & calc_fams
                if hitf:
                    fam_matched |= hitf
                    exp_fams |= hitf
                else:
                    fam_missing.append("|".join(sorted(e["families"])))
                    exp_fams |= e["families"]

        extra = calc_bases - matched
        extra_fams = calc_fams - fam_matched
        d["mapped_experimental_phases"] = ";".join(sorted(matched))
        d["missing_experimental_phases"] = ";".join(missing)
        d["unmappable_experimental_phases"] = ";".join(unmappable)
        d["experimental_phases_absent_from_database"] = ";".join(absent)
        d["extra_calculated_phases"] = ";".join(sorted(extra))
        d["experimental_family_set"] = ";".join(sorted(exp_fams))
        d["extra_calculated_families"] = ";".join(sorted(extra_fams))
        n_comparable = len(matched) + len(missing)
        d["n_comparable_experimental_phases"] = n_comparable
        d["exact_raw_phase_set_match"] = int(bool(n_comparable)
                                             and not missing and not extra)
        d["raw_phase_Jaccard"] = jaccard(matched, missing, extra)
        d["family_exact_set_match"] = int(bool(fam_matched) and not fam_missing
                                          and not extra_fams)
        d["family_Jaccard"] = jaccard(fam_matched, fam_missing, extra_fams)

        mult_checks, mult_ok = [], None
        for f in C.MULTIPLICITY_FAMILIES:
            ne = exp_fam_counts.get(f, 0)
            nc = int(d.get("n_%s_raw_instances" % f, 0) or 0)
            if ne >= 2:
                mult_checks.append("%s exp %d calc %d" % (f, ne, nc))
                mult_ok = (nc >= ne) if mult_ok is None else (mult_ok and nc >= ne)
        d["multiplicity_checks"] = "; ".join(mult_checks)
        d["multiplicity_match"] = ("" if mult_ok is None else int(mult_ok))

        # ---- topology support (spec 34) ------------------------------------- #
        e_rows = exp[exp.condition_id == cid]
        e_comp = {}
        for _, er in e_rows.iterrows():
            try:
                e_comp[str(er.literature_phase).strip()] = {
                    k.upper(): float(v)
                    for k, v in json.loads(er.comparison_at_pct).items()}
            except Exception:                                     # noqa: BLE001
                pass
        labelfam = {lit: pmap.get(lit, {}).get("families", set())
                    for lit in e_comp}
        e_topo, e_sep = topology_from(e_comp, labelfam)
        c_comp = {}
        if len(pcomp) and status == "CONVERGED":
            g = pcomp[pcomp.state_id == s.state_id]
            for ph, gg in g.groupby("raw_phase_name"):
                if ph not in calc_names:
                    continue
                c_comp[ph] = {str(x.element).upper():
                              100.0 * float(x.element_mole_fraction)
                              for _, x in gg.iterrows()
                              if np.isfinite(x.element_mole_fraction)}
        calcfam = {ph: {("FCC" if C.base_phase(ph) == "FCC_L12" else
                         "BCC" if C.base_phase(ph) == "BCC_B2" else
                         "C15" if C.base_phase(ph) == "C15_LAVES" else
                         "LIQUID" if C.base_phase(ph) == "LIQUID" else
                         "OTHER_SOLID")} for ph in c_comp}
        c_topo, c_sep = topology_from(c_comp, calcfam)
        d["experiment_topology_support"] = "; ".join(
            "%s->%s" % (k, v) for k, v in sorted(e_topo.items())) or \
            "no topology conclusion"
        d["experiment_phase_separation"] = ";".join(sorted(e_sep))
        d["calculated_topology"] = "; ".join(
            "%s->%s" % (k, v) for k, v in sorted(c_topo.items()))
        d["calculated_phase_separation"] = ";".join(sorted(c_sep))
        shared = set(e_topo) & set(c_topo)
        d["topology_agreement"] = ("" if not shared else
                                   int(all(e_topo[k] == c_topo[k] for k in shared)))
        d["topology_elements_compared"] = ";".join(sorted(shared))

        # ---- verdict --------------------------------------------------------- #
        if not selected:
            v = "NOT_COMPARABLE"
        elif status == "PENDING":
            v = ""
        elif status == "FAILED":
            v = "CALPHAD_FAILED"
        elif n_comparable == 0:
            v = "NOT_COMPARABLE"
        elif not missing and not extra and mult_ok is not False:
            v = "STRONGLY_CONSISTENT"
        elif not missing and mult_ok is not False:
            v = "CONSISTENT"
        elif matched:
            v = "PARTIALLY_CONSISTENT"
        else:
            v = "DISCREPANT"
        d["verdict_set_level"] = v
        d["verdict"] = v
        d["chemistry_gate"] = ""
        d["notes"] = ("crosswalk not yet resolved against the database; mappings "
                      "pending" if not crosswalk_resolved else "")
        rows.append(d)

    out = pd.DataFrame(rows)
    bad = sorted(set(out.verdict.dropna().astype(str)) - set(C.VERDICTS) - {""})
    if bad:
        raise SystemExit("FATAL: verdict outside the frozen vocabulary: %s" % bad)
    out.to_csv(OUT, index=False, encoding="utf-8")

    C.write_json(RULES, {
        "generated_utc": C.utcnow(),
        "authored_before_any_r3_equilibrium_existed": True,
        "rules_source_file": os.path.basename(__file__),
        "rules_source_sha256": C.sha256_file(os.path.abspath(__file__)),
        "presence_threshold_mole_fraction": C.PRESENCE,
        "verdict_vocabulary": C.VERDICTS,
        "verdict_order": [
            "not selected -> NOT_COMPARABLE",
            "no calculation submitted -> no verdict (PENDING)",
            "calculation failed -> CALPHAD_FAILED",
            "no mappable experimental phase -> NOT_COMPARABLE",
            "nothing missing, nothing extra, multiplicity ok -> STRONGLY_CONSISTENT",
            "nothing missing, multiplicity ok, extra phases -> CONSISTENT",
            "at least one experimental phase reproduced -> PARTIALLY_CONSISTENT",
            "none reproduced -> DISCREPANT"],
        "multiplicity_families": C.MULTIPLICITY_FAMILIES,
        "multiplicity_rule": "judged only where the paper reports two or more "
                             "instances of one family; satisfied when the "
                             "calculation returns at least as many",
        "Cr2Nb_rule": "admitted against either Laves polymorph; the polymorph is not "
                      "assumed and the calculated one is reported as a result",
        "unmappable_rule": "excluded from the metric denominator and reported, never "
                           "force-matched",
        "absent_from_database_rule": "counted as a missing experimental phase, "
                                     "because the description cannot produce it",
        "chemistry_tolerance_at_pct": C.PHASE_CHEM_TOL_AT_PCT,
        "chemistry_gate": "stage 40 may downgrade a set-level verdict on measured "
                          "phase chemistry; it can never upgrade one",
        "crosswalk_resolved_against_database": bool(crosswalk_resolved),
        "crosswalk_conflicts": conflicts,
        "no_single_global_accuracy_is_reported": True})

    vc = out.verdict.replace("", np.nan).value_counts().to_dict()
    print("  %d condition rows | verdicts %s" % (len(out), vc))
    print("  pending %d | not selected %d"
          % (int((out.execution_status == "PENDING").sum()),
             int((out.execution_status == "NOT_SELECTED").sum())))
    print("STAGE 30: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
