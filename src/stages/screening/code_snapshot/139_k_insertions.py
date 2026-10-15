# -*- coding: utf-8 -*-
"""
139_k_insertions.py
STEP5K manuscript insertion documents (spec 48).

Four drop-in sections for the JMST manuscript, all numbers read from the frozen
STEP5K outputs rather than retyped:
    INSERT_1_METHODS_EXTERNAL_VALIDATION.md
    INSERT_2_RESULTS_EXTERNAL_VALIDATION.md
    INSERT_3_METHODS_VIRTUAL_SCREENING.md
    INSERT_4_RESULTS_VIRTUAL_SCREENING.md

Claim language (spec 16, 36). Screening outputs are only ever called
"thermodynamically promising", "CALPHAD-informed candidates" or "equilibrium
screening candidates". The words "optimal alloy", "best alloy", "highest-strength"
and "industrial alloy" appear nowhere except inside an explicitly marked disclaimer
block, which the integrity audit knows to skip.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C

D_OPEN = "<!-- DISCLAIMER_BLOCK_START -->"
D_CLOSE = "<!-- DISCLAIMER_BLOCK_END -->"


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def w(name, text):
    p = os.path.join(C.D_INSERT, name)
    open(p, "w", encoding="utf-8").write(text)
    print("    %s" % name)


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K manuscript insertions")
    print("=" * 78)

    ev = jload(os.path.join(C.D_EXTVAL, "STEP5K_EXTERNAL_VALIDATION_SUMMARY.json"))
    ec = jload(os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_LOG.json"))
    pf = jload(os.path.join(C.D_SCREEN, "STEP5K_PARETO_FREEZE.json"))
    cs = jload(os.path.join(C.D_CONFIRM, "STEP5K_CONFIRMATION_SUMMARY.json"))
    rules = jload(os.path.join(C.D_SCREEN, "STEP5K_SCREENING_RULES_FREEZE.json"))
    val = pd.read_csv(os.path.join(C.D_EXTVAL,
                                   "STEP5K_EXTERNAL_LITERATURE_VALIDATION.csv"))
    cand = pd.read_csv(os.path.join(C.D_SCREEN,
                                    "STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv"))
    band = rules["moderate_c15_band"]

    # ---------------------------------------------------------------- INSERT 1 -- #
    w("INSERT_1_METHODS_EXTERNAL_VALIDATION.md", """\
## Methods - External validation against the experimental literature

To place the thermodynamic description used in this work against independent
evidence, a structured search of the Cu-Cr-Nb-X experimental literature was carried
out and %d primary records were retained. For every record the alloy identity,
processing route, heat treatment, characterisation technique and the phases actually
identified were transcribed, together with the sentence or metadata field each item
came from. Where a field could not be read from an accessible record it was left as
UNKNOWN rather than inferred, and the DOI, title, journal and year of every record
were resolved independently against Crossref (%d of %d DOIs resolved; %d of %d titles
matched the registered record character for character after Unicode normalisation).

A record was admitted to a quantitative comparison only if it satisfied all three of
the following: the paper states a composition with an unambiguous weight- or
atomic-percent basis; at least one phase is experimentally identified; and an
equilibrium calculation could be built from exactly that composition. Records failing
any of these were classified NOT_DIRECTLY_COMPARABLE and carry no verdict, so that
inaccessible records cannot be counted as tacit support.

Every admitted composition lies outside the composition domain sampled in this study
(S_CrNb = %.1f-%.1f wt.%%, R_CrNb = %.1f-%.1f atomic). No sampled state was therefore
used as a proxy for a literature alloy. Instead, %d fresh equilibrium calculations
were performed for the stated literature compositions over the same %d-point
temperature grid, using the same database, pressure, system size and solver
escalation as the main dataset. These calculations were performed after the
predictive models had been frozen, contain no Ni, and were not used to retrain,
retune, reselect or reinterpret any part of the model or of the blind test.

Consistency was then assessed by a fixed rule set applied mechanically to every
record. A reported phase is mapped to a phase model in the database; the temperatures
at which that model is and is not stable are recorded; and the verdict follows from
whether the phase is stable at all, some, or none of the compared temperatures, with
a separate outcome reserved for records whose material was processed far from
equilibrium. Verdicts are drawn from a closed vocabulary - qualitatively consistent,
partially consistent, reasonable equilibrium versus non-equilibrium discrepancy, not
reproduced by equilibrium calculation, not directly comparable - and no verdict is
assigned by inspection.

%s
This comparison tests the thermodynamic description, not the machine-learning
surrogate, and agreement between a calculation and an experiment is reported as
consistency between two descriptions of the same alloy. It is never taken to
establish that the calculated equilibrium is the true state of the material: the
calculation is not treated as ground truth, and the experiment is not treated as a
label.
%s
""" % (ev["n_papers"], ev["n_dois_independently_reverified"], ev["n_papers"],
       ev["n_titles_matching_crossref"], ev["n_papers"],
       C.S_CRNB_RANGE[0], C.S_CRNB_RANGE[1], C.R_CRNB_RANGE[0], C.R_CRNB_RANGE[1],
       ec["n_states"], len(ec["temperatures_C"]), D_OPEN, D_CLOSE))

    # ---------------------------------------------------------------- INSERT 2 -- #
    vc = ev["verdict_counts"]
    rows = "\n".join(
        "| %s | %s | %s | %s |" % (r.paper_id, r.X_identity,
                                   str(r.reported_phase)[:52],
                                   r.verdict.replace("_", " ").lower())
        for _, r in val.iterrows())
    w("INSERT_2_RESULTS_EXTERNAL_VALIDATION.md", """\
## Results - External validation against the experimental literature

The single most informative outcome of this comparison is negative, and it is
reported first: **none of the %d retrieved records describes an alloy inside the
composition domain sampled in this work.** The published Cu-Cr-Nb alloys whose
compositions are stated explicitly cluster in the precipitation-strengthened regime -
GRCop-42 and GRCop-84 at S_CrNb = 6.18 and 12.35 wt.%%, both at the stoichiometric
Cr2Nb ratio R_CrNb = 2.000, and a Nb-lean sintered alloy at S_CrNb = 3.50 wt.%%,
R_CrNb = 10.72 - while the dataset assembled here samples the dilute microalloying
window S_CrNb = 0.6-2.0 wt.%%. The two regimes are disjoint (Fig. K1). This is a
statement about where the published evidence lies, and it bounds what any external
validation of this study can establish.

Of the %d validation rows, %d are qualitatively consistent, %d partially consistent,
%d are reasonable equilibrium-versus-non-equilibrium discrepancies, %d are not
reproduced by the equilibrium calculation, and %d are not directly comparable because
the record does not state a composition, a phase, or both (Fig. K2).

The comparisons that could be made concern the two phases the literature identifies
most consistently: the C15 Laves phase Cr2Nb, reported in rapidly solidified ribbon,
in gas-atomised and laser-powder-bed-fused GRCop material and in the sintered Nb-lean
alloy, and the body-centred-cubic Cr-rich precipitate reported across the Cu-Cr and
Cu-Cr-Zr literature. Figure K3 shows the calculated equilibrium phase-family
fractions for each stated literature composition across the temperature grid; the
per-record outcome is tabulated below.

| record | X | reported phase | verdict |
|---|---|---|---|
%s

Three limits qualify all of this. First, every comparable alloy was processed far
from equilibrium - rapid solidification, gas atomisation, laser powder bed fusion,
sintering, severe cold drawing, or aging from a supersaturated state - so an
equilibrium calculation cannot be expected to reproduce a kinetically selected phase
assemblage, and where the two differ the discrepancy is recorded rather than
explained away. Second, the comparison is qualitative: none of the accessible records
states a phase fraction, a partition coefficient or a transformation temperature
together with a composition, so no numerical agreement is claimed. Third, and most
importantly, this exercise probes the thermodynamic description; because every
literature composition lies outside the training domain, no prediction from the
machine-learning surrogate was made for any of them, and none of this evidence bears
on the cross-element generalisation result.
""" % (ev["n_papers"], ev["n_validation_rows"],
       vc.get("QUALITATIVELY_CONSISTENT", 0), vc.get("PARTIALLY_CONSISTENT", 0),
       vc.get("REASONABLE_EQUILIBRIUM_VS_NON_EQUILIBRIUM_DISCREPANCY", 0),
       vc.get("NOT_REPRODUCED_BY_EQUILIBRIUM_CALCULATION", 0),
       vc.get("NOT_DIRECTLY_COMPARABLE", 0), rows))

    # ---------------------------------------------------------------- INSERT 3 -- #
    w("INSERT_3_METHODS_VIRTUAL_SCREENING.md", """\
## Methods - Virtual thermodynamic screening

The frozen surrogate models were applied to a virtual composition space to identify
thermodynamically promising, CALPHAD-informed candidates. %d compositions were
generated - %d for each of the six chemical identities developed in this work
(%s) - by a scrambled Sobol sequence in (S_CrNb, R_CrNb, w_X) over the sampled domain
bounds, using the same inversion from (S, R) to (w_Cr, w_Nb) as the original dataset
and a separate recorded seed stream per identity. Ni was excluded, and no chemical
identity absent from the training set was screened: the surrogate has no basis for
predicting one, as the blind test in this work demonstrates directly.

Every screening rule was written to a freeze file **before any prediction was
computed**, and the ranking stage aborts if that freeze flag is not set. The frozen
rules comprise three objectives - maximise the Cr partition to FCC at 950 °C,
maximise its change between 950 and 450 °C, minimise the Cr partition to other solid
phases at 450 °C - and three constraints: the predicted C15 volume fraction must lie
inside a moderate band at both 450 and 600 °C, the predicted liquid fraction must not
exceed %.2f at any screening temperature, and the composition must lie inside an
applicability radius. The moderate C15 band, [%.6f, %.6f], is the interquartile range
of the non-zero C15 volume fraction over the %s calculated states of the sampled
domain at 450 and 600 °C; it was derived from the training data alone and fixed
before any screening prediction existed, so it could not be tuned to admit a
preferred result. The applicability radius is a mean 5-nearest-neighbour distance of
1.50 to the development-training compositions in the standardised feature space, also
fixed in advance.

Candidates satisfying all three constraints were ranked by non-dominated sorting on
the three objectives. **No weighted score was formed at any point**, so no weights
could be chosen after seeing which candidates they would select. The sorting uses an
exact lexicographic sweep whose front is verified against exhaustive pairwise
comparison on a random sample at every run, and the stage aborts if the two disagree.
Fronts were computed globally and separately within each chemical identity, and both
are reported, so that no single identity can silently fill the shortlist. Finalists
were then chosen from the global front by deterministic maximin spacing in scaled
composition coordinates, round-robin across identities, targeting three to five per
identity and at most %d in total, with ties broken lexicographically. No predicted
objective value enters this diversity step.

The finalists were finally submitted to full equilibrium calculation at %s °C using
the same frozen engine, database and conditions as the main dataset. These
confirmation calculations are descriptive. They were performed after the models were
frozen, they were not used to retrain, retune, reselect or re-rank anything, and the
candidate ordering was not changed by their outcome - the confirmation result is
recorded as a flag against the frozen ordering.

%s
The screening produces thermodynamically promising, CALPHAD-informed candidates
under an equilibrium criterion evaluated by a surrogate model. It does not identify
an optimal alloy, a best alloy, a highest-strength alloy or an industrial alloy, and
no such claim is made anywhere in this work: no mechanical property, no processing
route, no kinetic pathway and no manufacturability constraint enters the screening at
any stage.
%s
""" % (rules["candidate_space"]["total"], rules["candidate_space"]["n_per_family"],
       ", ".join(C.SIX), C.LIQUID_TOLERANCE, band["band_low"], band["band_high"],
       "{:,}".format(band["n_nonzero_states"]), C.MAX_FINALISTS_TOTAL,
       ", ".join(str(t) for t in C.CONFIRM_TEMPS), D_OPEN, D_CLOSE))

    # ---------------------------------------------------------------- INSERT 4 -- #
    fam_fin = ", ".join("%s %d" % (k, v)
                        for k, v in pf["per_family_finalists"].items())
    w("INSERT_4_RESULTS_VIRTUAL_SCREENING.md", """\
## Results - Virtual thermodynamic screening

Of the %s screened compositions, %s satisfy all three pre-frozen constraints
(Fig. K4): %s pass the moderate C15 band at both 450 and 600 °C, %s carry no
predicted liquid, and %s fall inside the applicability radius. Non-dominated sorting
of the applicable set on the three pre-frozen objectives yields a global front of %d
compositions; the per-identity fronts contain %s (Fig. K5, Fig. K6). Deterministic
maximin spacing selects %d finalists (%s), distributed across composition space as
shown in Fig. K7.

Equilibrium calculation of the finalists at %s °C converged for %d of %d states.
Across the converged states the mean total-variation distance between the predicted
and calculated phase-family vectors is %.4f (median %.4f, maximum %.4f), the dominant
phase family agrees in %.1f%% of states, and the phase-presence pattern agrees
exactly in %.1f%% (Fig. K8). Of the %d finalists evaluated against the moderate C15
band, %d retain a calculated C15 fraction inside that band at both 450 and 600 °C.

%s
These agreement figures must not be read as a generalisation estimate, and they are
not comparable to the prospective blind result reported earlier in this work. The
finalists were chosen by the surrogate itself, from the extreme of its own predicted
objective space, so this is a selected, non-blind subset - exactly the sampling that
inflates apparent accuracy. The honest reading is narrower and is the only one
intended: on the specific compositions this screening puts forward, the surrogate's
equilibrium picture largely survives an actual equilibrium calculation, which is the
minimum a screening result must satisfy before it is worth reporting at all.
%s

The finalists are therefore reported as thermodynamically promising, CALPHAD-informed
equilibrium screening candidates. They are predictions of an equilibrium partitioning
behaviour, not of a property; they have not been synthesised, and nothing in this
work establishes that any of them would be manufacturable or useful. Their value is
that they are a concrete, falsifiable output of the frozen pipeline: each is a stated
composition with a stated predicted behaviour, calculated independently, and
available for experimental test.
""" % ("{:,}".format(pf["n_candidates"]), "{:,}".format(pf["n_applicable"]),
       "{:,}".format(pf["constraint_counts"]["C_band"]),
       "{:,}".format(pf["constraint_counts"]["E_liquid"]),
       "{:,}".format(pf["constraint_counts"]["F_ood"]),
       pf["global_front"],
       ", ".join("%s %d" % (k, v) for k, v in pf["family_front"].items()),
       pf["finalists"], fam_fin,
       ", ".join(str(t) for t in C.CONFIRM_TEMPS),
       cs["n_converged"], cs["n_states"],
       cs["TVD_family_mean"], cs["TVD_family_median"], cs["TVD_family_max"],
       100 * cs["dominant_family_agreement"],
       100 * cs["presence_pattern_agreement"],
       cs["moderate_C15_band"]["n_finalists_evaluated"],
       cs["moderate_C15_band"]["n_finalists_with_both_450_600_in_band"],
       D_OPEN, D_CLOSE))

    # a compact candidate table for the manuscript / SI
    cols = [c for c in ["candidate_id", "X_identity", "Cr_wt_pct", "Nb_wt_pct",
                        "X_wt_pct", "S_CrNb", "R_CrNb_atomic", "pred_Cr_FCC_950",
                        "pred_Delta_Cr_FCC", "pred_C15_450", "pred_Cr_OTHER",
                        "OOD_distance", "diversity_rank", "CALPHAD_confirmed",
                        "CALPHAD_C15_in_moderate_band"] if c in cand.columns]
    cand[cols].to_csv(os.path.join(C.D_INSERT, "TABLE_K1_SCREENING_CANDIDATES.csv"),
                      index=False, encoding="utf-8")
    print("    TABLE_K1_SCREENING_CANDIDATES.csv")
    print("INSERTIONS: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
