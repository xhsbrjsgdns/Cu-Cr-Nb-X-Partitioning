# -*- coding: utf-8 -*-
"""
r3_70_insertions.py
STEP5K-R3 stage 70: the four JMST insertion documents (spec 43-46).

These replace the provisional R2 external-validation language. Nothing else in the
manuscript is rewritten.

Claim level is fixed by spec 43 and enforced by the audit: this work provides
source-grounded experimental CONSISTENCY support for CALPHAD-predicted phase-affinity
and partition-topology trends. It does not validate the machine-learning model, and it
does not independently validate TCCU6. The three evidence layers of spec 44 -
experimental phase equilibria, CALPHAD reference behaviour, machine-learning and OOD
behaviour - are named separately and never merged, and the central manuscript story of
spec 45 is restated unchanged.

Every number is read from the frozen R3 outputs. When the calculations are still
pending the documents say so plainly rather than describing results that do not exist.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C

FORBIDDEN = ["experimentally validated", "independently validated",
             "independent validation", "validates the model",
             "experimental validation of the machine", "proves", "proven"]
D_OPEN = "<!-- DISCLAIMER_BLOCK_START -->"
D_CLOSE = "<!-- DISCLAIMER_BLOCK_END -->"


def w(name, text):
    open(os.path.join(C.D_INSERT, name), "w", encoding="utf-8").write(text)
    print("    %s" % name)


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K-R3 stage 70: JMST insertions")
    print("=" * 78)
    C.verify_protected()

    I = C.read_inputs()
    man = I["manifest"]
    cmp_ = pd.read_csv(os.path.join(C.D_CMP, "STEP5K_R3_CONDITION_COMPARISON.csv"))
    ev = pd.read_csv(os.path.join(C.D_EVID, "STEP5K_R3_SYSTEM_EVIDENCE.csv"))
    pcmp = pd.read_csv(os.path.join(C.D_PHCOMP,
                                    "STEP5K_R3_PHASE_COMPOSITION_COMPARISON.csv"))
    calc = pd.read_parquet(os.path.join(C.D_PROC,
                                        "STEP5K_R3_EXTERNAL_CALPHAD.parquet"))
    xres = os.path.join(C.D_CROSS, "STEP5K_R3_CROSSWALK_RESOLUTION.json")
    xr = C.load_json(xres) if os.path.exists(xres) else {}
    seal = I["seal"]
    conv = cmp_[cmp_.execution_status == "CONVERGED"]
    matched = pcmp[pcmp.match_status == "MATCHED"] if len(pcmp) else pcmp
    n_pending = int((calc.execution_status == "PENDING").sum())
    done = len(conv) > 0
    vc = conv.verdict.value_counts().to_dict() if done else {}

    pending_note = (
        "\n> **Status note (remove before submission).** %d of the %d frozen "
        "requests have not yet been calculated: the Thermo-Calc engine on the "
        "analysis machine fails during initialisation, before any database is "
        "selected. The conditions, compositions, inclusion rules and comparison "
        "rules below are frozen and final; the numbers they produce are not yet "
        "available. No pending condition is reported as a failed calculation or as "
        "evidence that a phase is absent.\n" % (n_pending, len(calc))
    ) if n_pending else ""

    # ------------------------------------------------------------- METHODS --- #
    src_tbl = "\n".join(
        "| %s | %s | %s (%s) | %s | %s |"
        % (r.source_id, r.experimental_system, r.journal, r.year, r.doi,
           r.source_role.replace("_", " ").lower())
        for _, r in man.iterrows())
    w("STEP5K_R3_JMST_METHODS_INSERTIONS.md", """\
## Methods - Source-grounded experimental consistency assessment

Six primary full texts were used as the sole external evidence base; no further
literature search contributed to this assessment. Each PDF was verified from the
document itself rather than from its filename - title, authors, journal, year, DOI and
experimental system - and hashed.

| source | system | journal | DOI | role |
|---|---|---|---|---|
%s

Only experimentally measured information was extracted. The authors' own calculated
phase diagrams were never treated as data: a condition enters this assessment through
its nominal composition, composition basis, annealing temperature and time, quenching
condition, and the phases identified by XRD, EPMA, SEM-EDS or diffusion equilibrium,
each recorded with the page and table or figure it came from. %d experimental
alloy-temperature conditions were transcribed this way.

Inclusion was frozen before any calculation. A condition is admitted to direct
comparison only if it carries a primary experimental observation, an exact nominal
composition, a known composition basis, a known temperature, an equilibration
treatment suitable for a phase-equilibrium study, and phases identified by one of the
methods above. %d conditions met all six criteria. Rapid-solidification, gas-atomised
and laser-powder-bed-fused states are excluded from direct comparison by rule and are
retained only as qualitative context. Compositions reported in atomic per cent were
converted to weight per cent with explicit atomic weights, at full precision, with the
printed original preserved unchanged; the %d admitted conditions all sum to 100 wt.%%
within floating-point tolerance.

A state budget of 120 equilibrium calculations was fixed in advance. Where eligible
conditions exceeded it, a stratified rule frozen before any calculation selected
across source, temperature and experimental phase region; agreement was never used to
select. %d requests were sealed, together with the sha256 of the request table, the
inclusion rules and the condition master.

Each request was calculated at its exact literature composition and exact experimental
temperature - never at a nearest sampled composition - using Thermo-Calc 2025b,
TC-Python 2025.2.30 and the TCCU6 Cu-alloys description at 1 bar, 1 mol, weight per
cent with Cu as balance, equilibrium only, with the same phase-detection logic as the
main dataset and a deterministic four-level convergence escalation. Failed
calculations are reported with their engine error; none is removed, interpolated, or
replaced by a neighbouring temperature, and no phase is ever suppressed to obtain
agreement.

Comparison is made at two levels. Raw Thermo-Calc phase instances are preserved before
any aggregation, because the five-family aggregation merges exactly the multiplicity
this assessment is designed to test - the Cu-rich and Ag-rich face-centred-cubic pair
in Cu-Cr-Ag, the Cr- and Mo-rich body-centred-cubic pair in Cu-Cr-Mo. A phase
crosswalk relates each reported phase to a database phase, and refuses to relate the
ones it cannot: Cr2Nb is admitted against either Laves polymorph because the sources
identify the compound and not its structure, and the ternary phase tau1 is left
unmapped because no supported database equivalent exists. Unmapped phases are excluded
from the set metric and reported separately; they are never matched to whichever
calculated phase happens to be present. Where measured phase compositions are
available, the matched phase is selected by structure and then by the element that
dominates the experimental phase, never by the choice that minimises the error.

Conditions are scored with a fixed vocabulary - strongly consistent, consistent,
partially consistent, discrepant, not comparable, CALPHAD failed - by a rule set
sealed before any result was inspected.

%s
These are source-grounded experimental consistency checks. They test the frozen
thermodynamic description, not the machine-learning surrogate, and they are not
independent validation of that description: four of the five phase-equilibrium sources
present their own thermodynamic assessment of the same ternary, so agreement partly
measures internal consistency. No single global experimental-accuracy figure is
reported, because the six studies differ in method, temperature range and phase
identification, and one number over them would be uninterpretable.
%s
%s""" % (src_tbl, seal["condition_count"], seal["eligible_count"],
         seal["eligible_count"], seal["selected_count"], D_OPEN, D_CLOSE,
         pending_note))

    # ------------------------------------------------------------- RESULTS --- #
    if done:
        per_sys = "\n".join(
            "| %s | %d | %d | %d | %d | %d | %s |"
            % (r.system, r.n_calphad_converged, r.n_strongly_consistent,
               r.n_consistent, r.n_partially_consistent, r.n_discrepant,
               r.evidence_strength)
            for _, r in ev.iterrows())
        chem = ("Across %d matched phases the mean absolute element error is %.2f "
                "at.%% and the largest single deviation is %.2f at.%%; the direction "
                "of dominant enrichment agrees in %.0f%% of matched phases."
                % (len(matched), matched.MAE_at_pct.mean(),
                   matched.max_abs_error_at_pct.max(),
                   100 * matched.dominant_enrichment_agrees.astype(float).mean())
                if len(matched) else
                "No experimental phase composition could be matched to a calculated "
                "phase.")
        body = """\
Of the %d frozen requests, %d converged and %d failed. Across the converged
conditions: %d strongly consistent, %d consistent, %d partially consistent, %d
discrepant, %d not comparable. No aggregate accuracy is quoted over the six studies;
the per-system breakdown is the result.

| system | converged | strongly consistent | consistent | partially consistent | discrepant | evidence |
|---|---|---|---|---|---|---|
%s

%s

Discrepancies are reported in full and are not removed. Every condition, including
every failed calculation and every phase that could not be mapped, remains in
`STEP5K_R3_CONDITION_COMPARISON.csv`.
""" % (len(calc), len(conv), int((calc.execution_status == "FAILED").sum()),
       vc.get("STRONGLY_CONSISTENT", 0), vc.get("CONSISTENT", 0),
       vc.get("PARTIALLY_CONSISTENT", 0), vc.get("DISCREPANT", 0),
       vc.get("NOT_COMPARABLE", 0), per_sys, chem)
    else:
        body = """\
The comparison set is complete and frozen: %d experimental conditions transcribed from
six primary full texts, %d admitted by the frozen inclusion rules, and %d exact
composition-temperature requests sealed together with the rules that will score them.
**No equilibrium calculation has yet been submitted**, because the Thermo-Calc engine
on the analysis machine fails during initialisation before any database is selected.

No result is therefore reported here. In particular, no condition is described as
discrepant, no phase is described as absent, and no evidence-strength claim is made:
a pending calculation is not a failed one.
""" % (seal["condition_count"], seal["eligible_count"], seal["selected_count"])

    w("STEP5K_R3_JMST_RESULTS_INSERTIONS.md",
      "## Results - Source-grounded experimental consistency\n\n" + body
      + pending_note)

    # ---------------------------------------------------------- DISCUSSION --- #
    absent = ", ".join(xr.get("phases_absent_from_database", [])[:8]) or "none recorded"
    w("STEP5K_R3_JMST_DISCUSSION_INSERTIONS.md", """\
## Discussion - What this external evidence supports, and what it does not

Three layers of evidence run through this work and must not be merged.

**Layer A, experimental phase equilibria.** Six primary studies report which phases
form in Cu-Cr-Nb and in the adjacent Cu-Cr-X ternaries, at stated compositions and
temperatures, with measured phase compositions.

**Layer B, CALPHAD reference behaviour.** The frozen thermodynamic description
reproduces - or fails to reproduce - those assemblages at the same conditions. This is
what the present comparison tests.

**Layer C, machine-learning and out-of-distribution behaviour.** The surrogate's
accuracy within its sampled domain, its failure on an unseen chemical identity, and
its recovery from a small same-element anchor. **Nothing in Layer A or B speaks to
Layer C.** Every literature condition lies outside the sampled composition domain, and
no surrogate prediction was made for any of them.

Two limits bound what Layer B can establish. The first is circularity: four of the
five phase-equilibrium sources present their own thermodynamic assessment of the same
ternary they measured, and a commercial Cu-alloys description is built from exactly
this literature. Where such an assessment was incorporated, agreement partly measures
internal consistency. Independence is not established and is not claimed. The second
is scope: several reported phases have no counterpart in the installed description
(%s), and the ternary phase tau1 has no supported equivalent, so those observations
cannot be reproduced at all and are recorded as missing rather than explained away.

The adjacent ternaries need a further distinction. Cu-Cr-Ag, Cu-Cr-Mo, Cu-Cr-Co,
Cu-Cr-Si and Cu-Cr-Ti support the *chemistry* of each element alongside Cu and Cr.
They do not support the Cu-Cr-Nb-X quaternary behaviour the surrogate predicts. In
particular, the Cu-Cr-Co study validates Co chemistry adjacent to Cu and Cr; it says
nothing about Co-Nb Laves topology, which is the behaviour the surrogate actually
assigns to Co-bearing alloys, and it must not be cited as if it did.

The central story of this paper is unchanged by any of this: high in-domain accuracy,
severe chemical-identity out-of-distribution failure, prospective blind evidence on an
unseen element, a topology-selection mismatch, generic chemistry augmentation that
does not repair it, and a small same-element thermodynamic anchor that restores
transfer quickly. The external evidence here supports the physical plausibility of the
thermodynamic layer those conclusions rest on. It does not replace the prospective
blind result, and no discrepancy found here bears on it.
""" % absent)

    # ------------------------------------------------------- SUPPLEMENTARY --- #
    w("STEP5K_R3_JMST_SUPPLEMENTARY_INSERTIONS.md", """\
## Supplementary - Source-grounded experimental consistency assessment

The complete evidence chain is provided so that every condition can be traced from the
printed page to the calculation that was compared with it.

- `STEP5K_R3_SOURCE_MANIFEST.csv` - the six full texts, verified from the documents
  themselves, with sha256 and role.
- `STEP5K_R3_EXPERIMENTAL_CONDITIONS_MASTER.csv` - all %d transcribed conditions with
  nominal composition, basis, temperature, annealing time, cooling condition,
  characterisation method, identified phases, and the page and table each came from.
- `STEP5K_R3_CONDITION_INCLUSION_FREEZE.md` - the six inclusion criteria and the
  exclusion rules, frozen before any calculation.
- `STEP5K_R3_COMPOSITION_CONVERSIONS.csv` - atomic-to-weight conversions with the
  atomic weights used, the formula, and the closure check.
- `STEP5K_R3_DOMAIN_RELEVANCE.csv` - each condition's position relative to the sampled
  composition domain, and its system relation class.
- `STEP5K_R3_PHASE_CROSSWALK.csv` - reported phase, structure, database candidate,
  frozen family, and the resolution status against the database's own phase list,
  including the phases it does not contain.
- `STEP5K_R3_CALPHAD_REQUESTS.csv` and `STEP5K_R3_EXTERNAL_CALPHAD.parquet` - the
  exact conditions submitted and the state-by-state outcome.
- `STEP5K_R3_EXTERNAL_CALPHAD_RAW_PHASES.csv` - every raw phase instance, preserved
  before family aggregation.
- `STEP5K_R3_CONDITION_COMPARISON.csv` - raw-phase and family set comparison, phase
  multiplicity, verdict, and the topology support for each condition.
- `STEP5K_R3_PHASE_COMPOSITION_COMPARISON.csv` - %d measured phase compositions
  against the matched calculated phases, with element-wise errors.
- `STEP5K_R3_SYSTEM_EVIDENCE.csv` and `STEP5K_R3_DATABASE_INDEPENDENCE_NOTE.md` - the
  per-system evidence strength and the independence caveat.

Figures Fig. R3-1 to R3-4 each carry a source CSV. A panel that could not be drawn
from the available data was not drawn, and `FIGURE_STATUS.md` says which and why.
""" % (seal["condition_count"], len(pcmp)))

    hits = []
    for f in sorted(os.listdir(C.D_INSERT)):
        if not f.endswith(".md"):
            continue
        import re as _re
        raw = open(os.path.join(C.D_INSERT, f), encoding="utf-8").read()
        txt = _re.sub(r"\s+", " ",
                      _re.sub(_re.escape(D_OPEN) + r".*?" + _re.escape(D_CLOSE), " ",
                              raw, flags=_re.S)).lower()
        for ph in FORBIDDEN:
            if ph in txt:
                hits.append((f, ph))
    if hits:
        raise SystemExit("FATAL: an insertion document makes a forbidden claim: %s"
                         % hits[:3])
    print("  claim scan clean outside disclaimer blocks")
    print("STAGE 70: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
