# -*- coding: utf-8 -*-
"""
137_k_plan_audit.py
STEP5K Part C: finalise the original V1.0 plan completion audit (spec 37-44).

The 37-row audit table is NOT rewritten here. Two rows were left as
IN_PROGRESS_IN_STEP5K because their deliverables did not exist yet; this stage
resolves exactly those two, and only against files it can see on disk. Every other
row is carried through byte-for-byte, and the pre-edit table is kept beside the new
one so the change is auditable.

DO NOT RETROACTIVELY REWRITE HISTORY (spec 40). A revised protocol stays
COMPLETED_WITH_REVISED_PROTOCOL and never becomes COMPLETED:
  * the Ni blind ran at 2000 compositions / 22,000 states, not the 5000 / 55,000 the
    plan suggested;
  * the shared-representation multitask network of plan item P14 was never built -
    the conservation goal was met by projection on a single-task model - so P14 stays
    PARTIAL.
This stage asserts those two invariants and aborts if either has been softened.

Two completion numbers are reported because they answer different questions, and both
formulas are printed with them:
  LITERAL      = (# rows COMPLETED with literal_match == YES) / 37
  SCIENTIFIC   = sum(weight) / 37, weights COMPLETED 1.0,
                 COMPLETED_WITH_REVISED_PROTOCOL 1.0, SUPERSEDED_BY_STRONGER_DESIGN 1.0,
                 PARTIAL 0.5, IN_PROGRESS 0.5, NOT_DONE 0.0
Neither number is a measure of the work's quality (spec 44); the summary says so in
those words, because a completion percentage is the kind of figure that gets quoted
out of context.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C

AUDIT = os.path.join(C.D_AUDIT, "STEP5K_ORIGINAL_PLAN_COMPLETION_AUDIT.csv")
PRE = os.path.join(C.D_AUDIT, "STEP5K_ORIGINAL_PLAN_COMPLETION_AUDIT_PRE_STEP5K.csv")
WEIGHTS = {"COMPLETED": 1.0, "COMPLETED_WITH_REVISED_PROTOCOL": 1.0,
           "SUPERSEDED_BY_STRONGER_DESIGN": 1.0, "PARTIAL": 0.5,
           "IN_PROGRESS_IN_STEP5K": 0.5, "NOT_DONE": 0.0}

# plan item -> (the files that must exist for it to count as delivered, what it is)
DELIVERABLES = {
    "P26": ([os.path.join(C.D_EXTVAL, "STEP5K_EXTERNAL_LITERATURE_VALIDATION.csv"),
             os.path.join(C.D_EXTVAL, "STEP5K_EXTERNAL_VALIDATION_REPORT.md"),
             os.path.join(C.D_EXTVAL, "STEP5K_EXTERNAL_VALIDATION_REFERENCES.csv"),
             os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_STATES.csv")],
            "external literature validation"),
    "P27": ([os.path.join(C.D_SCREEN, "STEP5K_VIRTUAL_SCREENING_PARETO.csv"),
             os.path.join(C.D_SCREEN, "STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv"),
             os.path.join(C.D_SCREEN, "STEP5K_PARETO_FREEZE.json"),
             os.path.join(C.D_CONFIRM,
                          "STEP5K_SCREENING_CALPHAD_CONFIRMATION.csv")],
            "virtual thermodynamic screening"),
}


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K Part C: original V1.0 plan completion audit - finalisation")
    print("=" * 78)

    a = pd.read_csv(AUDIT)
    if len(a) != 37:
        raise SystemExit("FATAL: the audit table has %d rows, expected 37." % len(a))
    if not os.path.exists(PRE):
        a.to_csv(PRE, index=False, encoding="utf-8")

    # ---- history guards (spec 40) ------------------------------------------- #
    p24 = a.loc[a.plan_item_id == "P24"].iloc[0]
    if p24.status != "COMPLETED_WITH_REVISED_PROTOCOL":
        raise SystemExit("FATAL: P24 (Final Blind Ni) is marked %r. The blind ran at "
                         "2000 compositions / 22,000 states, not the 5000 / 55,000 the "
                         "plan suggested, so it must stay "
                         "COMPLETED_WITH_REVISED_PROTOCOL." % p24.status)
    p14 = a.loc[a.plan_item_id == "P14"].iloc[0]
    if p14.status == "COMPLETED":
        raise SystemExit("FATAL: P14 is marked COMPLETED. The shared-representation "
                         "multitask network was never built; meeting the conservation "
                         "goal by projection on a single-task model is not the same "
                         "thing and must not be recorded as literal completion.")

    # ---- resolve only the two STEP5K rows, and only against files on disk --- #
    changes = []
    for pid, (files, what) in DELIVERABLES.items():
        i = a.index[a.plan_item_id == pid]
        if not len(i):
            raise SystemExit("FATAL: plan item %s is missing from the audit." % pid)
        i = i[0]
        if a.at[i, "status"] != "IN_PROGRESS_IN_STEP5K":
            print("  %s already resolved as %s - left untouched"
                  % (pid, a.at[i, "status"]))
            continue
        present = [f for f in files if os.path.exists(f)]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            print("  %s stays IN_PROGRESS_IN_STEP5K - missing %s"
                  % (pid, [os.path.basename(m) for m in missing]))
            a.at[i, "difference"] = ("%s; as of this audit the following STEP5K "
                                     "deliverables do not exist: %s"
                                     % (a.at[i, "difference"],
                                        ", ".join(os.path.basename(m) for m in missing)))
            continue
        old = a.at[i, "status"]
        a.at[i, "status"] = "COMPLETED_WITH_REVISED_PROTOCOL"
        a.at[i, "literal_match"] = "APPROX"
        a.at[i, "actual_stage"] = "STEP5K_ORIGINAL_PLAN_GAP_CLOSURE_V1"
        a.at[i, "actual_execution"] = ("delivered in STEP5K: %s"
                                       % ", ".join(os.path.basename(f) for f in files))
        changes.append({"plan_item_id": pid, "what": what, "from": old,
                        "to": "COMPLETED_WITH_REVISED_PROTOCOL",
                        "evidence": [os.path.basename(f) for f in files]})
        print("  %s -> COMPLETED_WITH_REVISED_PROTOCOL (%s)" % (pid, what))

    a.to_csv(AUDIT, index=False, encoding="utf-8")

    # ---- the two numbers ---------------------------------------------------- #
    n = len(a)
    unknown = sorted(set(a.status) - set(WEIGHTS))
    if unknown:
        raise SystemExit("FATAL: unweighted status values %s" % unknown)
    literal_n = int(((a.status == "COMPLETED")
                     & (a.literal_match.astype(str).str.upper() == "YES")).sum())
    sci = float(a.status.map(WEIGHTS).sum())
    counts = a.status.value_counts().to_dict()

    summary = {
        "generated_utc": C.utcnow(),
        "n_plan_items": n,
        "status_counts": counts,
        "literal_completion": {
            "formula": "(# rows with status == COMPLETED AND literal_match == YES) / 37",
            "numerator": literal_n, "denominator": n,
            "percent": round(100.0 * literal_n / n, 2)},
        "scientific_completion": {
            "formula": "sum(weight(status)) / 37",
            "weights": WEIGHTS,
            "numerator": round(sci, 4), "denominator": n,
            "percent": round(100.0 * sci / n, 2)},
        "items_not_done": a.loc[a.status == "NOT_DONE",
                                ["plan_item_id", "original_requirement"]]
                           .to_dict("records"),
        "items_partial": a.loc[a.status == "PARTIAL",
                               ["plan_item_id", "original_requirement"]]
                          .to_dict("records"),
        "items_superseded": a.loc[a.status == "SUPERSEDED_BY_STRONGER_DESIGN",
                                  ["plan_item_id", "original_requirement"]]
                             .to_dict("records"),
        "step5k_resolved_items": changes,
        "history_guards": {
            "P24_ni_blind_stays_revised_protocol": True,
            "P24_note": "2000 compositions / 22,000 states, not the plan's suggested "
                        "5000 / 55,000; recorded as revised, never as literal",
            "P14_shared_multitask_network_not_claimed_complete": True,
            "P14_note": "the shared-representation multitask network was never built; "
                        "the conservation goal was met by projection on a single-task "
                        "model, which is not the same architecture"},
        "WHAT_THESE_NUMBERS_ARE_NOT": (
            "A completion percentage against the original plan is a bookkeeping "
            "figure. It does not measure the quality, novelty or correctness of the "
            "work, and it must not be quoted as if it did. Several items scored below "
            "1.0 because the plan was deliberately departed from in favour of a "
            "stronger design, and at least one item scored 1.0 for a protocol that "
            "was reduced in scale."),
        "pre_step5k_table": os.path.basename(PRE)}
    C.write_json(os.path.join(C.D_AUDIT,
                              "STEP5K_ORIGINAL_PLAN_COMPLETION_SUMMARY.json"), summary)

    md = ["# STEP5K Part C - Original V1.0 Plan Completion Audit\n\n",
          "_Generated %s._\n\n" % C.utcnow(),
          "## Two numbers, two questions\n\n",
          "| measure | formula | value |\n|---|---|---|\n",
          "| Literal completion | (# COMPLETED with literal_match = YES) / %d | "
          "**%d / %d = %.1f%%** |\n" % (n, literal_n, n, 100.0 * literal_n / n),
          "| Scientific completion | sum of status weights / %d | **%.1f / %d = "
          "%.1f%%** |\n" % (n, sci, n, 100.0 * sci / n),
          "\nWeights: COMPLETED 1.0, COMPLETED_WITH_REVISED_PROTOCOL 1.0, "
          "SUPERSEDED_BY_STRONGER_DESIGN 1.0, PARTIAL 0.5, IN_PROGRESS 0.5, "
          "NOT_DONE 0.0.\n\n",
          "**Neither number measures the quality of the work.** It is bookkeeping "
          "against a plan written before any data existed. Some items score below 1.0 "
          "precisely because the plan was departed from in favour of a stronger "
          "design, and one item scores 1.0 for a protocol that was executed at "
          "reduced scale.\n\n",
          "## Status distribution\n\n| status | items |\n|---|---|\n"]
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        md.append("| %s | %d |\n" % (k, v))
    md.append("\n## What was deliberately not done, and what is still short\n\n")
    for _, r in a[a.status.isin(["NOT_DONE", "PARTIAL",
                                 "SUPERSEDED_BY_STRONGER_DESIGN"])].iterrows():
        md.append("- **%s (%s)** - %s\n  - %s\n"
                  % (r.plan_item_id, r.status, str(r.original_requirement)[:200],
                     str(r.difference)[:400]))
    md.append("\n## History that was not rewritten\n\n")
    md.append("- **P24 Final Blind Ni** stays `COMPLETED_WITH_REVISED_PROTOCOL`. The "
              "blind ran at 2000 compositions / 22,000 states, not the 5000 / 55,000 "
              "the plan suggested. The reduction was preregistered in STEP5G, and it "
              "is recorded as a revised protocol rather than as literal completion.\n")
    md.append("- **P14 conservation-constrained multitask network** stays `PARTIAL`. "
              "The conservation goal was met by projection on a single-task model. "
              "The shared-representation, five-head network the plan specified was "
              "never built, so the item is not marked complete.\n")
    if changes:
        md.append("\n## Resolved by STEP5K\n\n")
        for c in changes:
            md.append("- **%s** (%s): `%s` -> `%s`, on the evidence of %s\n"
                      % (c["plan_item_id"], c["what"], c["from"], c["to"],
                         ", ".join("`%s`" % e for e in c["evidence"])))
    open(os.path.join(C.D_AUDIT, "STEP5K_ORIGINAL_PLAN_COMPLETION_SUMMARY.md"),
         "w", encoding="utf-8").writelines(md)

    print("  literal %d/%d = %.1f%% | scientific %.1f/%d = %.1f%%"
          % (literal_n, n, 100.0 * literal_n / n, sci, n, 100.0 * sci / n))
    print("STAGE C: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
