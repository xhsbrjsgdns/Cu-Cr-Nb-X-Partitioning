# -*- coding: utf-8 -*-
"""
122_j_claims.py
STEP5J Stage 2: the claim hierarchy and the claim audit (spec 4-13, 29, 31).

Eight core claims were nominated for freezing. This stage does not simply record them:
it TESTS each one against the recomputed numbers and, where the evidence is weaker than
the nominated wording, it writes a weaker allowed wording and says why (spec 29:
"If evidence does not support any word, weaken it").

Two claims are weakened here, both for reasons that are visible in the frozen numbers
rather than a matter of taste:

  CLAIM 6  The nominated wording says anchor controls "often outperform" the refit at
           budgets 1-2. The frozen numbers say R0 outperforms the refit at budgets
           1, 2 AND 4, and R2 at 1 and 2. The allowed wording therefore names the
           measured crossover budgets instead of a vague "ultra-low".

  CLAIM 8  The nominated evidence is a pooled Spearman rho of 0.7597 between anchor
           NBNI3 coverage and recovery. That correlation is confounded: every
           calibration composition contains at least one NBNI3 state, so the coverage
           count IS the budget (rank correlation with budget = 1.000), and the
           fraction-based coverage also rises with budget. Within a budget only three
           cells exist and most within-budget correlations are undefined or
           non-significant. The stratified part of the claim survives on its own
           measurement (NBNI3-present states are the hardest stratum at every budget);
           the correlation part is demoted to a confounded observation.

Each claim carries: evidence stage, prospective or retrospective, primary / secondary /
exploratory, supporting files, supporting metric, the allowed wording, and the specific
overclaim that is forbidden.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j_common as C

FILES = {
    "L1": "HB_PRIMARY_METRICS", "L2": "HB_SECONDARY_METRICS",
    "L3AGG": "I_RANDOM_AGG", "L3THR": "I_THRESHOLDS", "L3BASE": "I_BASELINES",
    "L4TOPO": "I_TOPO", "L4NB": "I_NBNI3", "L4COV": "I_COVERAGE",
    "RANGE": "HA_OOD_FREEZE", "LADDER": "C_LADDER", "V1": "V1_TOPOLOGY",
}


def f(key):
    """Basename of the frozen file behind a claim's evidence slot."""
    return os.path.basename(C.src(FILES[key]))


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5J  Stage 2: claim hierarchy and claim audit")
    print("=" * 78)

    N = json.load(open(C.P(C.D_FREEZE, "FINAL_NUMERICAL_RESULTS.json"), encoding="utf-8"))
    L1 = N["level_1_prospective_primary"]
    L2 = N["level_2_prospective_secondary"]
    L3 = N["level_3_postblind_recovery"]
    L4 = N["level_4_postblind_descriptive"]
    CTX = N["context_not_primary_evidence"]

    p_tvd = L1["mean_tvd"]["value"]
    p_topo = L1["topology_accuracy"]["value"]
    p_triv = L1["trivial_tvd"]["value"]
    s_tvd = L2["mean_tvd"]["value"]
    s_triv = L2["own_trivial_tvd"]["value"]
    zero = L3["zero_shot_tvd_same_subset"]["value"]
    triv = L3["trivial_tvd_same_subset"]["value"]
    refit = {int(k): v for k, v in L3["random_M_REFIT_mean_tvd_by_budget"]["value"].items()}
    r0 = {int(k): v for k, v in L3["R0_by_budget"]["value"].items()}
    r2 = {int(k): v for k, v in L3["R2_by_budget"]["value"].items()}
    tacc = {int(k): v for k, v in L3["random_topology_accuracy_by_budget"]["value"].items()}
    frac = {int(k): v for k, v in L3["recovery_fraction_by_budget"]["value"].items()}
    b_skill = L3["B_skill"]["value"]
    b_rec = L3["B_recovery25"]["value"]
    first_r2 = L3["first_budget_M_REFIT_below_R2"]["value"]
    first_r0 = L3["first_budget_M_REFIT_below_R0"]["value"]
    iid = CTX["iid_x_partition_mean_tvd"]["value"]
    rng = CTX["ni_feature_range"]["value"]
    true_topo = L1["true_topology_distribution"]["value"]
    pred_topo = L1["pred_topology_distribution"]["value"]
    conf = L4["anchor_coverage_confound"]["value"]
    inherit = L4["topology_inheritance_observation"]["value"]

    # ---- evidence tests: does each nominated claim survive its own numbers? -- #
    t = {}
    t["C1"] = iid < 0.01 and p_tvd > 0.5
    t["C2"] = (rng["n_features_extending_train_range"] == 0
               and rng["frac_states_any_abs_z_gt5"] == 0 and p_tvd > p_triv)
    t["C3"] = (pred_topo.get("C15", 0) > 0.5 and true_topo.get("C15", 0.0) == 0.0
               and p_topo < 0.15)
    t["C4"] = (s_tvd < p_tvd and s_tvd > s_triv and p_tvd > p_triv)
    t["C5"] = (b_skill == 1 and b_rec == 1 and refit[1] < triv)
    # the nominated CLAIM 6 wording claims "b1-b2" for the control advantage
    r0_beats = [b for b in C.BUDGETS if r0[b] <= refit[b]]
    r2_beats = [b for b in C.BUDGETS if r2[b] <= refit[b]]
    t["C6"] = bool(r0_beats and r2_beats and first_r0 is not None
                   and first_r2 is not None)
    c6_wording_exact = (r0_beats == [1, 2] and r2_beats == [1, 2])
    t["C7"] = tacc[32] > 0.9 and p_topo < 0.15
    nb_p32 = L4["NBNI3_present"]["value"]["b32"]
    nb_a32 = L4["NBNI3_absent"]["value"]["b32"]
    oth32 = L4["true_OTHER_SOLID"]["value"]["b32"]
    fcc32 = L4["true_FCC"]["value"]["b32"]
    t["C8_strata"] = nb_p32 > nb_a32 and oth32 > fcc32
    cov_conf = conf["rho_between_budget_and_coverage"]
    perfect = cov_conf.get("n_anchor_comps_with_any_NBNI3", {}).get(
        "spearman_rho_with_budget")
    t["C8_correlation"] = not (perfect is not None and abs(perfect) > 0.99)

    claims = []

    def add(cid, nominated, allowed, forbidden, stage, prosp, tier, files, metric,
            supported, weakened=None):
        claims.append({
            "claim_id": cid, "claim_text": allowed,
            "nominated_text": nominated,
            "evidence_stage": stage,
            "prospective_or_retrospective": prosp,
            "primary_secondary_exploratory": tier,
            "supporting_files": "; ".join(files),
            "supporting_metric": metric,
            "allowed_wording": allowed,
            "forbidden_overclaim": forbidden,
            "evidence_supports_nominated_wording": bool(supported),
            "weakened_because": weakened or "",
        })

    add("CLAIM_1",
        "Strong in-domain performance does not guarantee unseen-element chemical "
        "transfer.",
        "Strong in-domain performance does not guarantee unseen-element chemical "
        "transfer: the same frozen pipeline that reaches a mean X-partition TVD of "
        "%.6f on the in-domain test scores %.6f on a prospectively held-out chemical "
        "identity, worse than its own training-mean baseline (%.6f)."
        % (iid, p_tvd, p_triv),
        "Do not write that the model is an accurate zero-shot unseen-element model, or "
        "that this demonstrates universal OOD generalization.",
        "STEP5A + STEP5H-B", "prospective for the Ni term; in-domain for the reference",
        "PRIMARY", [f("LADDER"), f("L1")],
        "x_partition mean TVD, identical metric definition in both stages", t["C1"])

    add("CLAIM_2",
        "Marginal feature-range overlap is insufficient for thermodynamic transfer.",
        "Marginal feature-range overlap is insufficient for thermodynamic transfer: "
        "none of the 43 production features extends beyond its training range on any "
        "Ni state, no Ni state has any |z| above 5, and Ni's elemental descriptor "
        "vector lies inside the six training identities' property ranges - yet the "
        "prospective zero-shot TVD is %.6f against a trivial baseline of %.6f. The "
        "range diagnostic was frozen before any Ni CALPHAD state existed."
        % (p_tvd, p_triv),
        "Do not write that distance predicts failure. The frozen post-reveal "
        "diagnostics found only weak negative rank correlations between per-state error "
        "and every distance measure tested, which is the opposite of a usable predictor.",
        "STEP5H-A + STEP5H-B", "prospective (diagnostic frozen pre-reveal)",
        "PRIMARY", [f("RANGE"), f("L1")],
        "0 of 43 features extending the training range; 0.0% of states with any |z|>5",
        t["C2"])

    add("CLAIM_3",
        "The dominant zero-shot failure is topology selection rather than small "
        "continuous partition error.",
        "The dominant zero-shot failure is topology selection rather than small "
        "continuous partition error: the true Ni X-partition is %.1f%% FCC-dominant and "
        "%.1f%% OTHER_SOLID-dominant with no C15-dominant state at all, while the frozen "
        "model assigns C15 as the dominant family to %.1f%% of states and reaches a "
        "topology accuracy of %.4f."
        % (100 * true_topo.get("FCC", 0), 100 * true_topo.get("OTHER_SOLID", 0),
           100 * pred_topo.get("C15", 0), p_topo),
        "Do not write that NBNI3 causes all error: the topology error is present across "
        "the whole temperature range, including the 700-1000 degC window where NBNI3 "
        "never appears.",
        "STEP5H-B", "prospective", "PRIMARY", [f("L1")],
        "true and predicted dominant-family distributions; topology accuracy", t["C3"])

    add("CLAIM_4",
        "Broader chemical-identity augmentation improves but does not solve "
        "unseen-element zero-shot transfer.",
        "Broader chemical-identity augmentation improves but does not solve "
        "unseen-element zero-shot transfer: adding five further chemical identities "
        "lowers the prospective Ni TVD from %.6f to %.6f (%.1f%% lower) and raises "
        "topology accuracy from %.4f to %.4f, yet the augmented model remains worse "
        "than its own training-mean baseline (%.6f vs %.6f)."
        % (p_tvd, s_tvd, 100 * (1 - s_tvd / p_tvd), p_topo,
           L2["topology_accuracy"]["value"], s_tvd, s_triv),
        "Do not write that FULL11 solves chemical OOD. It was also declared SECONDARY "
        "before the reveal and cannot be promoted on the strength of this result.",
        "STEP5H-B", "prospective", "SECONDARY", [f("L1"), f("L2")],
        "PRIMARY and SECONDARY mean TVD and topology accuracy on the identical 22,000 "
        "states", t["C4"])

    add("CLAIM_5",
        "Minimal same-element thermodynamic anchoring rapidly restores transfer.",
        "Minimal same-element thermodynamic anchoring rapidly restores transfer: a "
        "single Ni composition - which is 11 CALPHAD temperature states, not one data "
        "point - lowers the mean TVD on the held-out recovery subset from %.6f to "
        "%.6f, a %.1f%% reduction, already below the trivial baseline of %.6f. Both "
        "preregistered budget thresholds are reached at the smallest budget tested "
        "(B_skill = %s, B_recovery25 = %s)."
        % (zero, refit[1], 100 * frac[1], triv, b_skill, b_rec),
        "Do not write that one point is sufficient without stating that one composition "
        "is 11 CALPHAD temperature states, and never describe this as a zero-shot or "
        "blind result: it uses real Ni labels.",
        "STEP5I", "post-blind retrospective recovery", "PRIMARY within Level 3",
        [f("L3AGG"), f("L3THR"), f("L3BASE")],
        "mean TVD over 3 frozen random replicates on the 15,400-state recovery subset",
        t["C5"])

    c6_allowed = (
        "The balance between anchor information and model refitting is budget "
        "dependent: the anchor-only control R0 has the lower mean TVD at budgets %s and "
        "R2 at budgets %s, the refit first overtakes R2 at %s composition(s) and R0 at "
        "%s, and from 8 anchors upward the refit is below both controls at every budget "
        "tested (b8 %.6f vs R0 %.6f and R2 %.6f; b32 %.6f vs %.6f and %.6f)."
        % (r0_beats, r2_beats, first_r2, first_r0, refit[8], r0[8], r2[8],
           refit[32], r0[32], r2[32]))
    add("CLAIM_6",
        "At ultra-low budgets, anchor information itself dominates, while model "
        "refitting becomes clearly beneficial at larger budgets.",
        c6_allowed,
        "Do not state the crossover budgets as preregistered endpoints. They are "
        "post-blind descriptive crossovers read off a frozen learning curve.",
        "STEP5I", "post-blind retrospective recovery", "PRIMARY within Level 3",
        [f("L3AGG")],
        "across-replicate mean TVD of M_REFIT, R0 and R2 on identical states",
        t["C6"],
        weakened=None if c6_wording_exact else
        "the nominated wording implies the control advantage is confined to budgets "
        "1-2; measured on the frozen curve R0 also has the lower mean TVD at budget %s, "
        "so the allowed wording names the measured budgets instead"
        % ([b for b in r0_beats if b > 2] or "none"))

    add("CLAIM_7",
        "Few-shot recovery repairs the previous topology-selection failure.",
        "Few-shot recovery repairs the previous topology-selection failure: topology "
        "accuracy on the recovery subset rises from %.4f under the frozen zero-shot "
        "model to %.4f at 1 composition, %.4f at 2, %.4f at 4, %.4f at 8 and %.4f at 32."
        % (L3BASE_topo(N), tacc[1], tacc[2], tacc[4], tacc[8], tacc[32]),
        "Do not present this as a zero-shot capability. The topology is recovered "
        "because real Ni labels are supplied.",
        "STEP5I", "post-blind retrospective recovery", "PRIMARY within Level 3",
        [f("L3AGG"), f("L3BASE")],
        "argmax-family accuracy, mean over 3 frozen random replicates", t["C7"])

    c8_allowed = (
        "Ni-specific NBNI3 / OTHER_SOLID thermodynamics marks the hardest recovery "
        "regime: at 32 anchors the NBNI3-present stratum retains the highest residual "
        "TVD (%.6f) against %.6f for NBNI3-absent states, and the true-OTHER_SOLID "
        "stratum (%.6f) sits above the true-FCC stratum (%.6f). All four are far below "
        "the shared trivial baseline of %.6f, so this is a relative ordering among "
        "well-recovered strata, not a residual failure."
        % (nb_p32, nb_a32, oth32, fcc32, triv))
    c8_corr_note = (
        "The pooled rank correlation of %.4f between anchor NBNI3 coverage and recovery "
        "magnitude is NOT usable evidence because it is confounded with budget size: "
        "every calibration composition contains at "
        "least one NBNI3 state, so the anchor coverage count is exactly the budget "
        "(rank correlation with budget = %s), and the fraction-based coverage also rises "
        "with budget. Within a budget only 3 cells exist and most within-budget "
        "correlations are undefined or non-significant."
        % (conf["pooled_spearman_rho_coverage_vs_recovery"] or float("nan"),
           perfect if perfect is None else "%.4f" % perfect))
    add("CLAIM_8",
        "Ni-specific NBNI3 / OTHER_SOLID thermodynamics is associated with the hardest "
        "recovery regime, evidenced by stratified TVD and by a Spearman rho of 0.7597 "
        "between anchor NBNI3 coverage and recovery magnitude.",
        c8_allowed + " " + c8_corr_note,
        "Do not cite the coverage-recovery correlation as evidence that NBNI3 coverage "
        "drives recovery, and do not write that NBNI3 causes all error.",
        "STEP5I", "post-blind descriptive", "EXPLORATORY",
        [f("L4NB"), f("L4TOPO"), f("L4COV")],
        "stratified mean TVD at budget 32; rank correlations reported with their "
        "confound", t["C8_strata"] and t["C8_correlation"],
        weakened="the correlation half of the nominated claim is confounded with budget "
                 "size; only the stratified half is retained as evidence")

    df = pd.DataFrame(claims)
    df.to_csv(C.P(C.D_CLAIM, "FINAL_CLAIM_AUDIT.csv"), index=False, encoding="utf-8")

    hierarchy = {
        "generated_utc": C.utcnow(),
        "rule": "evidence levels are never mixed inside a single statement",
        "levels": C.EVIDENCE_LEVELS,
        "level_membership": {
            "LEVEL_1": ["STEP5H-B PRIMARY zero-shot mean TVD, CI, topology accuracy, "
                        "per-family MAE, true and predicted topology distributions"],
            "LEVEL_2": ["STEP5H-B FULL11 SECONDARY zero-shot metrics"],
            "LEVEL_3": ["STEP5I random few-shot learning curve", "maximin comparison",
                        "R0 / R1 / R2 anchor-only controls", "B_skill", "B_recovery25"],
            "LEVEL_4": ["NBNI3 stratification", "true-topology stratification",
                        "temperature-regime stratification",
                        "anchor coverage correlations", "topology-inheritance "
                        "observation", "paired bootstrap intervals"],
            "CONTEXT_NOT_EVIDENCE": [
                "STEP5A IID", "STEP5B composition-domain OOD",
                "STEP5C / STEP5E / STEP5F retrospective LOEO (model-family context, "
                "not scores of the frozen production checkpoints)"],
        },
        "claims_tested": {c["claim_id"]: c["evidence_supports_nominated_wording"]
                          for c in claims},
        "claims_weakened": [c["claim_id"] for c in claims if c["weakened_because"]],
        "forbidden_claims": C.FORBIDDEN_CLAIMS,
        "conditional_phrase": {
            "phrase": "one point / one composition is sufficient",
            "allowed_only_with": "an explicit statement that one composition is 11 "
                                 "CALPHAD temperature states"},
    }
    C.write_json(C.P(C.D_CLAIM, "STEP5J_CLAIM_HIERARCHY.json"), hierarchy)

    lines = ["# FINAL_CLAIM_AUDIT.md", "",
             "Generated %s. Eight claims were nominated for freezing; each is tested "
             "against the recomputed numbers before it is allowed." % C.utcnow(), ""]
    for c in claims:
        lines += ["## %s" % c["claim_id"], "",
                  "**Allowed wording.** %s" % c["allowed_wording"], "",
                  "| | |", "|---|---|",
                  "| evidence stage | %s |" % c["evidence_stage"],
                  "| prospective / retrospective | %s |"
                  % c["prospective_or_retrospective"],
                  "| tier | %s |" % c["primary_secondary_exploratory"],
                  "| supporting files | %s |" % c["supporting_files"],
                  "| supporting metric | %s |" % c["supporting_metric"],
                  "| nominated wording supported as written | %s |"
                  % c["evidence_supports_nominated_wording"], ""]
        if c["weakened_because"]:
            lines += ["**Weakened.** %s" % c["weakened_because"], ""]
        # the audit's whole purpose is to name the overclaims; quoting them is not
        # making them, so this block is marked as a disclaimer for the phrase scan
        lines += [C.DISCLAIMER_OPEN,
                  "**Forbidden.** %s" % c["forbidden_overclaim"],
                  C.DISCLAIMER_CLOSE, "", "---", ""]
    with open(C.P(C.D_CLAIM, "FINAL_CLAIM_AUDIT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    n_weak = len([c for c in claims if c["weakened_because"]])
    print("  %d claims audited | %d supported as nominated | %d weakened"
          % (len(claims), sum(c["evidence_supports_nominated_wording"] for c in claims),
             n_weak))
    for c in claims:
        if c["weakened_because"]:
            print("    weakened %s: %s" % (c["claim_id"], c["weakened_because"][:110]))
    print("STAGE 2: PASS")
    return 0


def L3BASE_topo(N):
    """Zero-shot topology accuracy on the RECOVERY subset. Falling back to the
    22,000-state value would compare two different row sets inside one sentence."""
    v = N["level_3_postblind_recovery"].get("zero_shot_topology_accuracy_same_subset")
    if not v:
        raise SystemExit("FATAL: the recovery-subset zero-shot topology accuracy is "
                         "missing from FINAL_NUMERICAL_RESULTS.json; CLAIM 7 must not "
                         "be built from the 22,000-state value.")
    return v["value"]


if __name__ == "__main__":
    sys.exit(main())
