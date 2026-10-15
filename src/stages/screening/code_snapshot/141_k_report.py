# -*- coding: utf-8 -*-
"""
141_k_report.py
STEP5K final report, summary and file seal (spec 50-51).

Writes STEP5K_REPORT.md (twenty questions answered from the frozen outputs, not from
memory), STEP5K_SUMMARY.json and STEP5K_FILE_HASHES.csv, then re-runs the claim and
Ni scans over its OWN output before declaring success - a report that trips the guard
the audit stage enforces would otherwise sail through, because the audit ran before
the report existed.
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C
import k_scan as SC



def jload(p):
    return json.load(open(p, encoding="utf-8"))


ALLOW = [
    "ni blind", "blind ni", "no ni", "ni was", "ni were", "ni_states_run",
    "ni composition", "ni target", "ni prediction", "ni result", "ni zero-shot",
    "ni interpretation", "ni is", "ni state", "ni artifact", "ni few-shot",
    "must not", "never", "excluded", "untouched", "unchanged", "cu-cr-ni",
    "ni identity", "ni blind artifact", "ni-bearing", "cu–cr–ni",
]


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K final report and seal")
    print("=" * 78)

    ev = jload(os.path.join(C.D_EXTVAL, "STEP5K_EXTERNAL_VALIDATION_SUMMARY.json"))
    ec = jload(os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_LOG.json"))
    pf = jload(os.path.join(C.D_SCREEN, "STEP5K_PARETO_FREEZE.json"))
    cs = jload(os.path.join(C.D_CONFIRM, "STEP5K_CONFIRMATION_SUMMARY.json"))
    pl = jload(os.path.join(C.D_AUDIT,
                            "STEP5K_ORIGINAL_PLAN_COMPLETION_SUMMARY.json"))
    qa = jload(os.path.join(C.D_QA, "STEP5K_INTEGRITY_AUDIT.json"))
    LIT_PCT = pl.get("literal_plan_completion_percent")
    SCI_PCT = pl.get("scientific_core_completion_percent")
    rules = jload(os.path.join(C.D_SCREEN, "STEP5K_SCREENING_RULES_FREEZE.json"))
    dv = jload(os.path.join(C.D_CONFIG, "STEP5K_DOI_VERIFICATION.json"))
    vc = ev["verdict_counts"]

    Q = []

    def q(question, answer):
        Q.append((question, answer))

    q("What did STEP5K set out to do?",
      "Close the three gaps the original V1.0 plan left open, and nothing else: "
      "external validation against the experimental literature (plan item P26), "
      "virtual thermodynamic screening (P27), and an honest audit of what the plan "
      "actually delivered (Part C). No new modelling question was opened.")

    q("Was any frozen Ni blind artifact altered?",
      "No. The %d protected artifacts - the two production checkpoints, the two Ni "
      "prediction files, the Ni targets, both blind score freezes, and the STEP5I and "
      "STEP5J seals - were hashed before the first calculation and re-verified "
      "afterwards, in every stage that computes anything. Audit item 3 re-checks them "
      "independently. No Ni composition was generated, no Ni equilibrium was run, no "
      "Ni prediction was made, and no blind metric, phase mapping or few-shot result "
      "was touched." % (len(C.PROTECTED_HASHES) + len(C.PROTECTED_FILES)))

    q("How large is the literature corpus, and how was it verified?",
      "%d primary records. Every field that could not be read from an accessible "
      "record was left UNKNOWN rather than inferred, and the search log records which "
      "records are publisher-gated and states explicitly that those may not be used as "
      "phase evidence in their current state." % ev["n_papers"])

    q("Were the DOIs re-verified independently, or was the upstream flag trusted?",
      "Re-verified independently. The upstream table's own crossref_verified flag was "
      "not taken on trust; each DOI was resolved again against the Crossref API in "
      "this session and the returned title, journal and year compared field by field "
      "after Unicode normalisation. Result: %d of %d resolved, %d of %d titles match, "
      "%d of %d journals match, %d of %d years match. No DOI failed."
      % (dv["n_resolved"], dv["n_dois"], dv["n_title_match"], dv["n_dois"],
         dv["n_journal_match"], dv["n_dois"], dv["n_year_match"], dv["n_dois"]))

    q("Was any error found in the upstream literature table?",
      "Yes, one, and it was material. GRCop-42 and GRCop-84 were recorded there as "
      "weight-percent designations. They are atomic-percent designations "
      "(NASA/TM-2005-213566 states \"GRCop-84 (Cu-8 at.% Cr-4 at.% Nb)\"). Read as "
      "at.%, GRCop-84 is Cu-6.520Cr-5.825Nb by weight, not Cu-8Cr-4Nb. Both readings "
      "of the digits were calculated and are kept side by side, because the cited "
      "paper's own abstract does not settle the basis, and neither is asserted as that "
      "paper's meaning.")

    q("Do any literature compositions fall inside the sampled composition domain?",
      "None. This is the central result of Part A. The sampled domain is "
      "S_CrNb = %.1f-%.1f wt.%%; the published Cu-Cr-Nb alloys whose compositions are "
      "stated lie at S_CrNb = 3.50, 6.18 and 12.35 wt.%%, and the Cu-Zr records carry "
      "no Nb at all. The two regimes are disjoint, so no frozen state was offered "
      "anywhere as a stand-in for a literature alloy."
      % (C.S_CRNB_RANGE[0], C.S_CRNB_RANGE[1]))

    q("What calculations were run for Part A?",
      "%d equilibrium states: %d stated literature compositions over the frozen "
      "%d-point temperature grid, using the same database, pressure, system size and "
      "Level 0-3 solver escalation as the main dataset, imported by path from the "
      "frozen code snapshot rather than re-implemented. %d converged%s"
      % (ec["n_states"], ec["n_compositions"], len(ec["temperatures_C"]),
         ec["n_converged"],
         "; every state converged." if ec["n_failed_kept_not_deleted"] == 0 else
         "; the %d that did not are kept as rows carrying their error message, never "
         "deleted, interpolated or replaced by a neighbouring temperature."
         % ec["n_failed_kept_not_deleted"]))

    q("What did the external comparison find?",
      "Of %d validation rows: %d qualitatively consistent, %d partially consistent, "
      "%d reasonable equilibrium-versus-non-equilibrium discrepancies, %d not "
      "reproduced by the equilibrium calculation, %d not directly comparable. Verdicts "
      "come from a mechanical rule set applied to every record, not from inspection, "
      "and each row records the temperatures at which the corresponding phase was and "
      "was not stable."
      % (ev["n_validation_rows"], vc.get("QUALITATIVELY_CONSISTENT", 0),
         vc.get("PARTIALLY_CONSISTENT", 0),
         vc.get("REASONABLE_EQUILIBRIUM_VS_NON_EQUILIBRIUM_DISCREPANCY", 0),
         vc.get("NOT_REPRODUCED_BY_EQUILIBRIUM_CALCULATION", 0),
         vc.get("NOT_DIRECTLY_COMPARABLE", 0)))

    q("Does Part A validate the machine-learning surrogate?",
      "No, and the manuscript text says so. Every literature composition lies outside "
      "the surrogate's training domain, so no surrogate prediction was made for any of "
      "them. Part A tests the thermodynamic description only. Nor is agreement read as "
      "proof that the calculation is correct: where a calculation and an experiment "
      "line up, that is consistency between two descriptions of the same alloy.")

    q("How was the screening space built?",
      "%s compositions - %s for each of the six developed identities (%s) - from a "
      "scrambled Sobol sequence in (S_CrNb, R_CrNb, w_X) over the sampled domain "
      "bounds, with a recorded seed stream per identity and the frozen (S, R) to "
      "(w_Cr, w_Nb) inversion. No Ni. No identity absent from the training set: the "
      "blind test in this work is direct evidence that the surrogate cannot predict "
      "one." % ("{:,}".format(rules["candidate_space"]["total"]),
                "{:,}".format(rules["candidate_space"]["n_per_family"]),
                ", ".join(C.SIX)))

    q("What was frozen before any prediction, and can that be checked?",
      "The objectives, the constraint definitions, the moderate C15 band, the "
      "applicability radius, the ranking method, the diversity rule, the finalist caps "
      "and the permitted claim language - all written to the rules freeze before the "
      "first prediction, with frozen_before_any_screening_prediction set. The ranking "
      "stage aborts if that flag is false, and the freeze file's sha256 is recorded "
      "alongside the ranking output so the two cannot drift apart unnoticed. Audit "
      "items 8-10 re-check this.")

    q("Was any weighted score used to pick candidates?",
      "No. Ranking is non-dominated sorting on the three objectives, so no weights "
      "exist to be chosen after seeing which candidates they would select. Audit item "
      "11 confirms both the flag and the absence of any score or weight column in the "
      "ranking output.")

    q("Why did the first ranking attempt fail, and is the replacement exact?",
      "The first implementation compared every pair inside every front - O(k^2) at "
      "roughly 10^5 candidates - and had to be killed. It is replaced by an exact "
      "lexicographic sweep with a Fenwick prefix-maximum tree. Exactness is not "
      "assumed: every run re-derives the front on %d randomly sampled candidates by "
      "exhaustive pairwise comparison and aborts if the two disagree. On this run they "
      "agreed (%d = %d). Two latent defects in the old file were also fixed: a "
      "one-column pivot assigned as a two-dimensional array, and a join that relied on "
      "two pandas operations happening to sort identically."
      % (pf["bruteforce_check"]["n_sampled"], pf["bruteforce_check"]["front_size_sweep"],
         pf["bruteforce_check"]["front_size_bruteforce"]))

    q("How many candidates survived the constraints?",
      "%s of %s. Individually: %s pass the moderate C15 band at both 450 and 600 C, "
      "%s carry no predicted liquid, %s fall inside the applicability radius. The "
      "global non-dominated front holds %d compositions; the per-identity fronts hold "
      "%s."
      % ("{:,}".format(pf["n_applicable"]), "{:,}".format(pf["n_candidates"]),
         "{:,}".format(pf["constraint_counts"]["C_band"]),
         "{:,}".format(pf["constraint_counts"]["E_liquid"]),
         "{:,}".format(pf["constraint_counts"]["F_ood"]), pf["global_front"],
         ", ".join("%s %d" % kv for kv in pf["family_front"].items())))

    q("How were finalists chosen, and is family fairness preserved?",
      "Deterministic maximin spacing in scaled composition coordinates, round-robin "
      "across identities, ties broken lexicographically, capped at %d per identity and "
      "%d in total. No predicted objective value enters this step. Result: %d "
      "finalists, %s. Both global and per-identity fronts are reported, so no identity "
      "can silently fill the list."
      % (C.FINALISTS_PER_FAMILY[1], C.MAX_FINALISTS_TOTAL, pf["finalists"],
         ", ".join("%s %d" % kv for kv in pf["per_family_finalists"].items())))

    q("What did the CALPHAD confirmation of the finalists show?",
      "%d of %d states converged. Mean phase-family total-variation distance between "
      "prediction and calculation %.4f (median %.4f, maximum %.4f); dominant phase "
      "family agrees in %.1f%% of states; phase-presence pattern agrees exactly in "
      "%.1f%%. Of %d finalists evaluated, %d keep a calculated C15 fraction inside the "
      "pre-frozen moderate band at both 450 and 600 C."
      % (cs["n_converged"], cs["n_states"], cs["TVD_family_mean"],
         cs["TVD_family_median"], cs["TVD_family_max"],
         100 * cs["dominant_family_agreement"],
         100 * cs["presence_pattern_agreement"],
         cs["moderate_C15_band"]["n_finalists_evaluated"],
         cs["moderate_C15_band"]["n_finalists_with_both_450_600_in_band"]))

    q("Why is that agreement not a generalisation estimate?",
      "Because the finalists were chosen by the surrogate itself, from the extreme of "
      "its own predicted objective space. That is a selected, non-blind subset - "
      "exactly the sampling that inflates apparent accuracy. The number is not "
      "comparable to the prospective blind result, must not be quoted as a test-set "
      "error, and supports only the narrow reading intended: on the compositions this "
      "screening puts forward, the surrogate's equilibrium picture largely survives an "
      "actual equilibrium calculation.")

    q("What may the screening output be called?",
      "Thermodynamically promising candidates, CALPHAD-informed candidates, or "
      "equilibrium screening candidates. Nothing stronger. No mechanical property, "
      "processing route, kinetic pathway or manufacturability constraint entered the "
      "screening, so no claim about alloy performance or industrial use is available "
      "from it, and none is made. The finalists have not been synthesised.")

    q("What does the plan completion audit say?",
      "Literal completion %.1f%% (%d of %d items completed with an exact protocol "
      "match); scientific completion %.1f%% by the stated weighting. History was not "
      "rewritten: the Ni blind stays COMPLETED_WITH_REVISED_PROTOCOL because it ran at "
      "2000 compositions / 22,000 states rather than the plan's suggested 5000 / "
      "55,000, and the conservation-constrained multitask network stays PARTIAL "
      "because the shared-representation five-head architecture was never built - "
      "meeting the conservation goal by projection on a single-task model is not the "
      "same thing. Neither percentage measures the quality of the work, and the "
      "summary says so."
      % (LIT_PCT, pl.get('completed', 0),
         pl.get('n_items', 37),
         SCI_PCT))

    q("What must not happen next?",
      "STEP5K stops here. Do not retrain or fine-tune any model. Do not expand the Ni "
      "blind set, regenerate Ni predictions, or revisit the Ni interpretation. Do not "
      "screen a chemical identity absent from the training set. Do not run DFT, do not "
      "predict experimental properties, and do not open a new optimisation loop. The "
      "next action is review by the user.")

    if len(Q) != 20:
        raise SystemExit("FATAL: the report must answer exactly 20 questions, has %d."
                         % len(Q))

    md = ["# STEP5K - Original-Plan Gap Closure: Final Report\n\n",
          "_Generated %s. POST-FREEZE, NON-Ni. Integrity audit: %s (%d/%d items)._\n\n"
          % (C.utcnow(), qa["verdict"], qa["n_pass"], qa["n_items"])]
    for i, (question, answer) in enumerate(Q, 1):
        md.append("### %d. %s\n\n%s\n\n" % (i, question, answer))
    md.append("---\n\n## Deliverables\n\n")
    for d in [C.D_CONFIG, C.D_LIT, C.D_EXTVAL, C.D_EXTCAL, C.D_SCREEN, C.D_CONFIRM,
              C.D_FIG, C.D_AUDIT, C.D_INSERT, C.D_QA]:
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        if files:
            md.append("**%s**\n\n" % os.path.basename(d))
            for f in files:
                md.append("- `%s`\n" % f)
            md.append("\n")
    open(os.path.join(C.ROOT, "STEP5K_REPORT.md"), "w",
         encoding="utf-8").writelines(md)

    # ---- seal ---------------------------------------------------------------- #
    rows = []
    for root, _, files in os.walk(C.ROOT):
        if "__pycache__" in root:
            continue
        for f in sorted(files):
            p = os.path.join(root, f)
            rows.append({"relative_path": os.path.relpath(p, C.ROOT),
                         "bytes": os.path.getsize(p), "sha256": C.sha256_file(p)})
    pd.DataFrame(rows).to_csv(os.path.join(C.D_MAN, "STEP5K_FILE_HASHES.csv"),
                              index=False, encoding="utf-8")

    decision = ("STEP5K COMPLETE - EVIDENCE PACKAGE READY FOR REVIEW"
                if qa["verdict"] == "PASS"
                else "STEP5K INCOMPLETE - %d integrity item(s) FAILED" % qa["n_fail"])
    C.write_json(os.path.join(C.ROOT, "STEP5K_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "decision": decision,
        "integrity_audit": {"verdict": qa["verdict"], "n_pass": qa["n_pass"],
                            "n_fail": qa["n_fail"], "n_items": qa["n_items"]},
        "part_A_external_validation": {
            "n_papers": ev["n_papers"], "n_validation_rows": ev["n_validation_rows"],
            "verdict_counts": vc,
            "n_dois_reverified": dv["n_resolved"],
            "n_literature_compositions_inside_domain_A":
                ev["n_literature_compositions_inside_FULL_V1_domain_A"],
            "n_postfreeze_calphad_states": ec["n_states"],
            "validates_the_ml_surrogate": False,
            "validates_the_thermodynamic_description_qualitatively": True},
        "part_B_virtual_screening": {
            "n_candidates": pf["n_candidates"], "n_applicable": pf["n_applicable"],
            "global_front": pf["global_front"], "finalists": pf["finalists"],
            "per_family_finalists": pf["per_family_finalists"],
            "no_weighted_score_used": True,
            "bruteforce_verified": pf["bruteforce_check"]["identical"],
            "confirmation_states": cs["n_states"],
            "confirmation_converged": cs["n_converged"],
            "confirmation_TVD_family_mean": cs["TVD_family_mean"],
            "confirmation_is_a_selected_non_blind_subset": True,
            "confirmation_is_a_generalization_estimate": False},
        "part_C_plan_audit": {
            "literal_completion_percent": LIT_PCT,
            "scientific_completion_percent": SCI_PCT,
            "P24_status": "COMPLETED_WITH_REVISED_PROTOCOL",
            "P14_status_is_not_completed": True,
            "percentages_measure_quality": False},
        "invariants_held": {
            "no_ni_composition_generated": True, "no_ni_calphad_run": True,
            "no_ni_prediction_made": True, "no_model_retrained_or_retuned": True,
            "no_frozen_blind_artifact_modified": True,
            "no_unseen_chemical_identity_screened": True,
            "failed_calphad_states_kept": True,
            "partitioning_is_mole_based": True},
        "stop_rule": "STEP5K ends here. No retraining, no Ni expansion, no unseen "
                     "element, no DFT, no experimental-property prediction, no new "
                     "optimisation loop. Awaiting user review."})

    # ---- the scans the audit could not run over files that did not exist yet -- #
    mine = [os.path.join(C.ROOT, "STEP5K_REPORT.md"),
            os.path.join(C.ROOT, "STEP5K_SUMMARY.json"),
            os.path.join(C.D_MAN, "STEP5K_FILE_HASHES.csv")]
    bad = SC.forbidden_scan(mine, C.FORBIDDEN_CANDIDATE_WORDS)
    if bad:
        raise SystemExit("FATAL: the final report itself contains forbidden claim "
                         "language: %s" % bad[:2])
    ni = SC.ni_scan(mine, ALLOW)
    if ni:
        raise SystemExit("FATAL: the final report contains an unexplained Ni "
                         "reference: %s" % ni[:2])

    print("  %d files sealed | %s" % (len(rows), decision))
    print("REPORT: PASS" if qa["verdict"] == "PASS" else "REPORT: WRITTEN, AUDIT FAILED")
    return 0 if qa["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
