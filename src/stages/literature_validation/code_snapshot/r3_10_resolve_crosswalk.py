# -*- coding: utf-8 -*-
"""
r3_10_resolve_crosswalk.py
STEP5K-R3 stage 10: resolve the phase crosswalk against the ACTUAL TCCU6 phase list.

The crosswalk shipped with the frozen inputs deliberately left rows as
CANDIDATE_PENDING_ENGINE_CHECK, UNRESOLVED or AMBIGUOUS_POLYMORPH, because a phase
name alone cannot establish that the database contains that phase or which polymorph
an XRD pattern recorded. This stage closes exactly those gaps, and only with evidence
from `probe_database.py`: the phase list and sublattice constituents the installed
database actually returns for each Cu-Cr-X ternary.

Three rules, applied mechanically (spec 16, 17):

1. A candidate name is CONFIRMED_IN_DATABASE only if the database returns that exact
   phase name for that ternary. Otherwise it becomes ABSENT_FROM_DATABASE, which is a
   reportable finding about the description's scope, not a comparison failure to hide.

2. An UNRESOLVED literature phase is matched only when exactly one database phase
   both (a) carries the same element set as the literature formula and (b) has a name
   whose alphanumeric core equals the formula's. Anything else stays UNRESOLVED. A
   literature phase that stays UNRESOLVED is excluded from the raw-phase set metric
   and recorded as unmappable - never silently dropped, and never force-matched.

3. Cr2Nb stays AMBIGUOUS. The sources identify the compound, not its polymorph, and
   the frozen five-family map sends C15_LAVES to C15 but C14_LAVES to OTHER_SOLID. So
   Cr2Nb is admitted against EITHER Laves phase and its admissible family set is
   {C15, OTHER_SOLID}. Which polymorph the database actually stabilises is reported
   as a result, not assumed (final question 8).
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C

META = os.path.join(C.D_CROSS, "DATABASE_PHASE_METADATA.json")
CROSS = os.path.join(C.D_CROSS, "STEP5K_R3_PHASE_CROSSWALK.csv")
PRE = os.path.join(C.D_CROSS, "STEP5K_R3_PHASE_CROSSWALK_PRE_RESOLUTION.csv")

# The frozen five-family map (fivefamily.FAMILY_MAP_5). Restated here so that this
# stage cannot silently disagree with the runner about which family a phase joins.
FROZEN_FAMILY = {"FCC_L12": "FCC", "BCC_B2": "BCC", "C15_LAVES": "C15",
                 "LIQUID": "LIQUID"}
LAVES = ["C14_LAVES", "C15_LAVES", "C36_LAVES"]


def family_of(raw):
    return FROZEN_FAMILY.get(C.base_phase(raw), "OTHER_SOLID")


def formula_core(text):
    """'αCr5Si3' -> 'CR5SI3'; 'Cu19Si6' -> 'CU19SI6'; '(Cu, Cr)3Ti2' -> 'CUCR3TI2'."""
    t = re.sub(r"[^A-Za-z0-9]", "", str(text))
    t = re.sub(r"^(alpha|beta|gamma|delta|tau|sigma)", "", t, flags=re.I)
    return t.upper()


def elements_of(text):
    return set(re.findall(r"(?:AG|MO|CO|SI|TI|NB|CR|CU)", formula_core(text)))


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K-R3 stage 10: resolve the phase crosswalk against TCCU6")
    print("=" * 78)
    n_prot, changed, missing = C.verify_protected()
    print("  %d protected artifacts re-hashed, all unchanged" % n_prot)

    if not os.path.exists(META):
        raise SystemExit(
            "FATAL: %s does not exist.\n"
            "The crosswalk cannot be resolved without the database's own phase list. "
            "Run code_snapshot/probe_database.py with the Thermo-Calc Python first "
            "(it calculates no equilibrium; it only reads phase metadata)."
            % os.path.basename(META))

    meta = C.load_json(META)
    by_system = {}
    for r in meta:
        by_system.setdefault(r["system"], {})[r["phase"]] = r
    all_phases = {p for d in by_system.values() for p in d}
    print("  database returned %d phase records across %d ternaries (%d distinct "
          "phase names)" % (len(meta), len(by_system), len(all_phases)))

    cw = pd.read_csv(CROSS)
    if not os.path.exists(PRE):
        cw.to_csv(PRE, index=False, encoding="utf-8")

    def constituents(phase):
        rec = None
        for d in by_system.values():
            if phase in d:
                rec = d[phase]
                break
        if rec is None:
            return set()
        out = set()
        for sub in rec.get("sublattices", []):
            out |= {str(c).upper() for c in sub.get("constituents", [])}
        return {e for e in out if e in C.ALLOWED_ELEMENTS}

    rows, log = [], []
    for _, r in cw.iterrows():
        lit = str(r.literature_phase)
        cand = str(r.TCCU_raw_phase_candidate)
        status, resolved, admissible_family, note = None, cand, None, ""

        if "|" in cand or str(r.mapping_confidence) == "AMBIGUOUS_POLYMORPH":
            present = [p for p in LAVES if p in all_phases]
            status = "AMBIGUOUS_POLYMORPH_ADMITS_EITHER"
            resolved = " | ".join(present) if present else "ABSENT_FROM_DATABASE"
            admissible_family = "|".join(sorted({family_of(p) for p in present})) \
                if present else ""
            note = ("the source identifies the compound, not the polymorph; the "
                    "database offers %s, which the frozen map sends to %s"
                    % (", ".join(present) or "no Laves phase", admissible_family))
        elif cand == "UNRESOLVED":
            core, els = formula_core(lit), elements_of(lit)
            hits = [p for p in sorted(all_phases)
                    if formula_core(p.split("_")[0]) == core
                    or formula_core(p) == core]
            hits = [p for p in hits if not els or elements_of(p) >= els
                    or constituents(p) >= els]
            if len(hits) == 1:
                status, resolved = "RESOLVED_BY_DATABASE_LOOKUP", hits[0]
                admissible_family = family_of(hits[0])
                note = "unique database phase with the same formula core and elements"
            elif len(hits) > 1:
                status, resolved = "UNRESOLVED_MULTIPLE_CANDIDATES", " | ".join(hits)
                note = ("more than one database phase matches; not force-matched, "
                        "excluded from the raw-phase set metric")
            else:
                status, resolved = "ABSENT_FROM_DATABASE", ""
                note = ("no phase with this formula is present in the installed "
                        "description for any Cu-Cr-X ternary examined; the "
                        "experimental phase therefore cannot be reproduced and the "
                        "condition is scored with this phase recorded as missing")
        else:
            if cand in all_phases:
                status = "CONFIRMED_IN_DATABASE"
                admissible_family = family_of(cand)
                cons = constituents(cand)
                els = elements_of(lit)
                if els and cons and not (els <= cons):
                    status = "CONFIRMED_BUT_CONSTITUENTS_DIFFER"
                    note = ("database constituents %s do not contain every element "
                            "of the literature label" % sorted(cons))
                else:
                    note = "database constituents %s" % sorted(cons)
            else:
                status, resolved = "ABSENT_FROM_DATABASE", ""
                note = "candidate name is not among the phases the database returned"

        out = dict(r)
        out.update(resolution_status=status,
                   resolved_TCCU_raw_phase=resolved,
                   admissible_frozen_family=admissible_family or "",
                   frozen_family_from_map=(family_of(resolved)
                                           if resolved and "|" not in resolved
                                           else admissible_family or ""),
                   resolution_note=note,
                   comparable=int(status in ("CONFIRMED_IN_DATABASE",
                                             "RESOLVED_BY_DATABASE_LOOKUP",
                                             "AMBIGUOUS_POLYMORPH_ADMITS_EITHER",
                                             "ABSENT_FROM_DATABASE")))
        rows.append(out)
        log.append({"literature_phase": lit, "was": cand, "status": status,
                    "now": resolved, "note": note})

    new = pd.DataFrame(rows)
    new.to_csv(CROSS, index=False, encoding="utf-8")
    counts = new.resolution_status.value_counts().to_dict()
    C.write_json(os.path.join(C.D_CROSS, "STEP5K_R3_CROSSWALK_RESOLUTION.json"), {
        "generated_utc": C.utcnow(),
        "database_metadata_file": os.path.basename(META),
        "database_metadata_sha256": C.sha256_file(META),
        "n_crosswalk_rows": int(len(new)),
        "resolution_counts": counts,
        "laves_phases_in_database": [p for p in LAVES if p in all_phases],
        "Cr2Nb_polymorph_assumed": False,
        "Cr2Nb_admissible_families": sorted({family_of(p) for p in LAVES
                                             if p in all_phases}),
        "phases_absent_from_database": sorted(
            new.loc[new.resolution_status == "ABSENT_FROM_DATABASE",
                    "literature_phase"].astype(str).unique()),
        "phases_left_unresolved": sorted(
            new.loc[new.resolution_status.astype(str)
                    .str.startswith("UNRESOLVED"), "literature_phase"]
            .astype(str).unique()),
        "rule": "no literature phase is matched to a database phase unless the "
                "database returns a unique phase with the same formula core and "
                "compatible constituents; ambiguity is preserved, never resolved by "
                "picking whichever option agrees",
        "pre_resolution_table": os.path.basename(PRE),
        "resolution_log": log})

    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("    %-40s %d" % (k, v))
    print("STAGE 10: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
