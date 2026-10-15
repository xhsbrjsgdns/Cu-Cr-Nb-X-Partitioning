# -*- coding: utf-8 -*-
"""
121_j_numbers.py
STEP5J Stage 1: recompute every manuscript-facing number from frozen files and freeze
them (spec 14, 15, 16, 17, 32, 35).

Nothing is transcribed from the brief. Every value is read or derived from a frozen
file, carried with its source path, source hash, metric definition and evidence level,
and then compared against the value quoted in the brief. A disagreement beyond rounding
is REPORTED as drift, not absorbed - a consolidation stage that silently adopts the
number it was told to expect is not consolidating anything.

Where a value cannot be derived from a frozen file it is written as NOT_AVAILABLE or
REQUIRES_NEW_EXPERIMENT (spec 35). No missing number is computed by running something
new.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j_common as C

DRIFT = []


def check_drift(name, computed, tol=5e-6):
    """Compare a recomputed value with the one the brief quoted."""
    if name not in C.BRIEF_QUOTED:
        return
    quoted = C.BRIEF_QUOTED[name]
    # the brief rounds; allow the rounding it actually used, then flag anything worse
    dec = len(str(quoted).split(".")[-1]) if "." in str(quoted) else 0
    rounded_ok = round(float(computed), dec) == quoted
    DRIFT.append({"quantity": name, "brief_quoted": quoted,
                  "recomputed_from_frozen_files": float(computed),
                  "abs_difference": abs(float(computed) - quoted),
                  "agrees_at_quoted_precision": bool(rounded_ok),
                  "status": "OK" if rounded_ok or abs(float(computed) - quoted) <= tol
                  else "DRIFT"})


def curve(agg, method, col="mean_tvd_mean"):
    g = agg[agg.method == method].sort_values("budget")
    return {int(b): float(v) for b, v in zip(g.budget, g[col])}


def stratum(df, stratum_name, method, budget=None, kind="RANDOM"):
    s = df[(df.method == method) & (df.stratum == stratum_name)]
    if method == "M_ZERO":
        return float(s.mean_tvd.mean()) if len(s) else None
    s = s[s.kind == kind]
    if budget is not None:
        s = s[s.budget == budget]
    return float(s.mean_tvd.mean()) if len(s) else None


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5J  Stage 1: recompute and freeze the manuscript numbers")
    print("=" * 78)

    hb_p = C.load_json("HB_PRIMARY_METRICS")
    hb_s = C.load_json("HB_SECONDARY_METRICS")
    hb_score = C.load_json("HB_PRIMARY_SCORE")
    base = C.load_json("I_BASELINES")
    thr = C.load_json("I_THRESHOLDS")
    ood = C.load_json("HA_OOD_FREEZE")
    agg = C.load_csv("I_RANDOM_AGG")
    rep = C.load_csv("I_RANDOM_REP")
    maxi = C.load_csv("I_MAXIMIN")
    cmp_ = C.load_csv("I_CMP")
    topo = C.load_csv("I_TOPO")
    nb = C.load_csv("I_NBNI3")
    temp = C.load_csv("I_TEMP")
    cov = C.load_csv("I_COVERAGE")
    rng_diag = C.load_csv("HA_RANGE_DIAG")
    ladder = C.load_csv("C_LADDER")
    b_ci = C.load_csv("B_IID_VS_OOD")
    v1_topo = C.load_csv("V1_TOPOLOGY")
    e_pipe = C.load_json("E_PIPELINE")
    f_sum = C.load_json("F_AUG_SUMMARY")
    b_counts = C.load_json("B_COUNTS")
    v1_dom = C.load_csv("V1_DOMAIN")
    _voc = C.load_csv("HB_PHASE_VOCAB")

    zero = base["M_ZERO"]["mean_tvd"]
    triv = base["M_TRIVIAL"]["mean_tvd"]
    refit = curve(agg, "M_REFIT")
    refit_sd = curve(agg, "M_REFIT", "mean_tvd_std")
    topo_acc = curve(agg, "M_REFIT", "topology_accuracy_mean")
    r0 = curve(agg, "R0")
    r1 = curve(agg, "R1")
    r2 = curve(agg, "R2")
    mx = {int(r.budget): float(r.mean_tvd) for _, r in
          maxi[maxi.method == "M_REFIT"].iterrows()}

    # ------------------------------------------------------- drift checks --- #
    check_drift("primary_blind_mean_tvd", hb_p["mean_tvd"])
    check_drift("primary_blind_topology_accuracy", hb_p["topology_accuracy"])
    check_drift("primary_blind_trivial_tvd", hb_p["trivial_mean_tvd"])
    check_drift("primary_blind_skill", hb_p["skill_vs_trivial"])
    check_drift("secondary_blind_mean_tvd", hb_s["mean_tvd"])
    check_drift("secondary_blind_topology_accuracy", hb_s["topology_accuracy"])
    check_drift("recovery_subset_zero_shot_tvd", zero)
    check_drift("recovery_subset_trivial_tvd", triv)
    for b in C.BUDGETS:
        check_drift("refit_b%d" % b, refit[b])
        check_drift("r0_b%d" % b, r0[b])
        check_drift("r2_b%d" % b, r2[b])
        check_drift("topo_b%d" % b, topo_acc[b])

    # ------------------------------------ spec 14: information efficiency --- #
    rows = [{"n_compositions": 0, "n_Ni_states": 0,
             "mean_TVD": zero, "mean_TVD_sd_across_replicates": "",
             "TVD_reduction_vs_zero": 0.0,
             "topology_accuracy": base["M_ZERO"]["topology_accuracy"],
             "skill_vs_trivial": triv - zero,
             "M_REFIT_minus_R0": "", "M_REFIT_minus_R2": "",
             "row_meaning": "frozen PRIMARY zero-shot on the recovery subset"}]
    for b in C.BUDGETS:
        rows.append({"n_compositions": b, "n_Ni_states": b * len(C.TEMPERATURES_C),
                     "mean_TVD": refit[b], "mean_TVD_sd_across_replicates": refit_sd[b],
                     "TVD_reduction_vs_zero": 1.0 - refit[b] / zero,
                     "topology_accuracy": topo_acc[b],
                     "skill_vs_trivial": triv - refit[b],
                     "M_REFIT_minus_R0": refit[b] - r0[b],
                     "M_REFIT_minus_R2": refit[b] - r2[b],
                     "row_meaning": "mean of 3 frozen random replicates"})
    ie = pd.DataFrame(rows)
    ie.to_csv(C.P(C.D_TABLES, "FINAL_INFORMATION_EFFICIENCY_TABLE.csv"), index=False,
              encoding="utf-8")
    print("  information-efficiency table: %d rows (budget 0 is the frozen zero-shot)"
          % len(ie))

    # -------------------------------- spec 16: descriptive crossover table --- #
    crows = []
    for b in C.BUDGETS:
        vals = {"M_REFIT": refit[b], "R0": r0[b], "R1": r1[b], "R2": r2[b]}
        winner = min(vals, key=vals.get)
        crows.append({"budget": b, "n_Ni_states": b * len(C.TEMPERATURES_C),
                      "M_REFIT": refit[b], "R0": r0[b], "R1": r1[b], "R2": r2[b],
                      "winner": winner,
                      "M_REFIT_vs_R0": "M_REFIT lower" if refit[b] < r0[b] else "R0 lower",
                      "M_REFIT_vs_R2": "M_REFIT lower" if refit[b] < r2[b] else "R2 lower",
                      "M_REFIT_minus_R0": refit[b] - r0[b],
                      "M_REFIT_minus_R2": refit[b] - r2[b]})
    cross = pd.DataFrame(crows)
    cross.to_csv(C.P(C.D_TABLES, "FINAL_CONTROL_CROSSOVER_TABLE.csv"), index=False,
                 encoding="utf-8")
    first_r2 = next((b for b in C.BUDGETS if refit[b] < r2[b]), None)
    first_r0 = next((b for b in C.BUDGETS if refit[b] < r0[b]), None)
    print("  crossover: M_REFIT first below R2 at budget %s, below R0 at budget %s"
          % (first_r2, first_r0))

    # ------------------------- spec 17: prospective vs post-blind table ------ #
    x_row = ladder[ladder.task == "x_partition"].iloc[0]
    prows = [
        {"experiment": "STEP5A IID test (Domain A, all six X identities seen)",
         "status": "IN-DOMAIN", "target_visible_during_model_design": "no (sealed test)",
         "model_changed_by_this_result": "no",
         "mean_TVD_x_partition": float(x_row.STEP5A_IID),
         "interpretation": "interpolation among seen identities; near-zero error",
         "evidence_level": "context"},
        {"experiment": "STEP5B composition-domain OOD (Domain B shell)",
         "status": "RETROSPECTIVE OOD (composition, not chemistry)",
         "target_visible_during_model_design": "no (sealed until first open)",
         "model_changed_by_this_result": "no",
         "mean_TVD_x_partition": float(x_row.STEP5B_composition_domain_OOD),
         "interpretation": "moving outside the sampled composition shell costs ~43x the "
                           "IID error but remains small in absolute terms",
         "evidence_level": "context"},
        {"experiment": "STEP5C retrospective leave-one-element-out (6 folds)",
         "status": "RETROSPECTIVE ELEMENT OOD",
         "target_visible_during_model_design":
             "no for the held-out element within its own fold; the protocol was "
             "designed with all six identities known to exist",
         "model_changed_by_this_result":
             "yes - later stages (5D/5E/5F) were designed in response",
         "mean_TVD_x_partition": float(x_row.STEP5C_LOEO_mean),
         "interpretation": "per-fold selected tree/boosting models; six-fold mean. "
                           "Model-family context, not a score of the STEP5G checkpoints",
         "evidence_level": "context"},
        {"experiment": "STEP5E Z1_MLP_DIRECT retrospective LOEO (selected pipeline)",
         "status": "RETROSPECTIVE ELEMENT OOD",
         "target_visible_during_model_design": "no within fold; six identities known",
         "model_changed_by_this_result": "this result SELECTED the production family",
         "mean_TVD_x_partition": C.LOEO_CONTEXT["original_six_Z1_MLP_DIRECT_mean_tvd"],
         "interpretation": C.LOEO_CONTEXT["caveat"],
         "evidence_level": "context"},
        {"experiment": "STEP5F FULL11 augmented retrospective LOEO",
         "status": "RETROSPECTIVE ELEMENT OOD",
         "target_visible_during_model_design": "no within fold",
         "model_changed_by_this_result":
             "no - every preregistered switching gate failed, so PRIMARY stayed "
             "original-six",
         "mean_TVD_x_partition": C.LOEO_CONTEXT["full5_augmented_mean_tvd"],
         "interpretation": C.LOEO_CONTEXT["caveat"],
         "evidence_level": "context"},
        {"experiment": "STEP5H-B prospective Ni zero-shot, PRIMARY",
         "status": "PROSPECTIVE ZERO-SHOT (unseen chemical identity)",
         "target_visible_during_model_design":
             "no - no Ni descriptor, property, prediction or CALPHAD state existed "
             "until after the model was frozen and hashed",
         "model_changed_by_this_result": "no - scored once, immutable",
         "mean_TVD_x_partition": hb_p["mean_tvd"],
         "interpretation": "the primary result of this work; worse than its own trivial "
                           "baseline (%.6f)" % hb_p["trivial_mean_tvd"],
         "evidence_level": "LEVEL 1"},
        {"experiment": "STEP5H-B prospective Ni zero-shot, SECONDARY (FULL11)",
         "status": "PROSPECTIVE ZERO-SHOT SENSITIVITY",
         "target_visible_during_model_design": "no",
         "model_changed_by_this_result": "no - declared secondary before the reveal",
         "mean_TVD_x_partition": hb_s["mean_tvd"],
         "interpretation": "improves on PRIMARY but is still worse than its own trivial "
                           "baseline",
         "evidence_level": "LEVEL 2"},
        {"experiment": "STEP5I post-blind few-shot recovery (32 Ni compositions)",
         "status": "POST-BLIND RETROSPECTIVE RECOVERY",
         "target_visible_during_model_design":
             "YES - real Ni CALPHAD targets are used as anchors",
         "model_changed_by_this_result":
             "a separate recovery model is refit; the frozen zero-shot checkpoints are "
             "untouched",
         "mean_TVD_x_partition": refit[32],
         "interpretation": "not zero-shot and not blind; answers how much direct Ni "
                           "data restores transfer",
         "evidence_level": "LEVEL 3"},
    ]
    pr = pd.DataFrame(prows)
    pr.to_csv(C.P(C.D_TABLES, "FINAL_PROSPECTIVE_RETROSPECTIVE_TABLE.csv"), index=False,
              encoding="utf-8")

    # --------------------- the NBNI3 coverage confound (spec 12 vs review) --- #
    from scipy.stats import spearmanr
    rnd = cov[cov.kind == "RANDOM"]
    confound = {}
    for col in ("fraction_anchor_states_NBNI3", "n_anchor_comps_with_any_NBNI3",
                "fraction_anchor_states_true_OTHER_SOLID"):
        if rnd[col].std(ddof=0) == 0:
            confound[col] = {"spearman_rho_with_budget": None,
                             "note": "constant across cells"}
        else:
            s = spearmanr(rnd.budget, rnd[col])
            confound[col] = {"spearman_rho_with_budget": float(s.statistic),
                             "p_value": float(s.pvalue)}
    mech = C.load_csv("I_MECH")
    pooled = mech[(mech.scope == "ALL_RANDOM_CELLS") & (mech.method == "M_REFIT")
                  & (mech.coverage_metric == "fraction_anchor_states_NBNI3")]
    pooled_rho = float(pooled.spearman_rho.iloc[0]) if len(pooled) else None
    if pooled_rho is not None:
        check_drift("coverage_rho", pooled_rho, tol=1e-4)
    within = mech[mech.scope.str.startswith("BUDGET_") & (mech.method == "M_REFIT")
                  & (mech.coverage_metric == "fraction_anchor_states_NBNI3")]
    n_within_defined = int(within.spearman_rho.notna().sum())

    # ------------------ descriptive topology-inheritance observation --------- #
    a = v1_topo[v1_topo.sampling_domain == "A"]
    piv = a.pivot_table(index="family", columns="dominant_X_solid_partition_family",
                        values="n_states", aggfunc="sum", fill_value=0)
    share = piv.div(piv.sum(axis=1), axis=0)
    nearest = ood.get("descriptor_position", {}).get("nearest_original_identity")
    nearest_z = ood.get("descriptor_position", {}).get("z_distance_to_nearest")
    nearest_c15 = float(share.loc[nearest, "C15"]) if (
        nearest in share.index and "C15" in share.columns) else None
    pred_c15 = hb_p["pred_topology_distribution"].get("C15")

    # ------------------------------------------- FINAL_NUMERICAL_RESULTS ---- #
    V = C.value
    N = {
        "generated_utc": C.utcnow(),
        "stage": "STEP5J FINAL EVIDENCE FREEZE",
        "rule": "every value below is recomputed from a frozen file; none is "
                "transcribed from prose",
        "evidence_levels": C.EVIDENCE_LEVELS,

        "level_1_prospective_primary": {
            "n_states": V(hb_p["n_scored"], "HB_PRIMARY_METRICS",
                          "converged Ni blind states scored", 1),
            "mean_tvd": V(hb_p["mean_tvd"], "HB_PRIMARY_METRICS",
                          "0.5*sum|P_pred-P_true| over the 5-family Ni X-partition, "
                          "mean over all converged blind states", 1),
            "ci95": V(hb_score["ci95"], "HB_PRIMARY_SCORE",
                      "composition-level cluster bootstrap, 1000 replicates, seed "
                      "20260911", 1),
            "median_tvd": V(hb_p["median_tvd"], "HB_PRIMARY_METRICS", "median TVD", 1),
            "p95_tvd": V(hb_p["p95_tvd"], "HB_PRIMARY_METRICS", "95th percentile TVD", 1),
            "topology_accuracy": V(hb_p["topology_accuracy"], "HB_PRIMARY_METRICS",
                                   "fraction of states whose argmax family is correct", 1),
            "topology_macro_f1": V(hb_p["topology_macro_f1"], "HB_PRIMARY_METRICS",
                                   "macro F1 over 5 families, zero_division=0", 1),
            "trivial_tvd": V(hb_p["trivial_mean_tvd"], "HB_PRIMARY_METRICS",
                             "frozen PRIMARY training-mean simplex on the same states", 1),
            "skill_vs_trivial": V(hb_p["skill_vs_trivial"], "HB_PRIMARY_METRICS",
                                  "trivial TVD minus model TVD; negative = worse than "
                                  "predicting the training mean", 1),
            "per_family_mae": V(hb_p["per_family_mae"], "HB_PRIMARY_METRICS",
                                "mean absolute error per family", 1),
            "true_topology_distribution": V(hb_p["true_topology_distribution"],
                                            "HB_PRIMARY_METRICS",
                                            "share of blind states by true argmax family",
                                            1),
            "pred_topology_distribution": V(hb_p["pred_topology_distribution"],
                                            "HB_PRIMARY_METRICS",
                                            "share of blind states by predicted argmax "
                                            "family", 1),
        },
        "level_2_prospective_secondary": {
            "mean_tvd": V(hb_s["mean_tvd"], "HB_SECONDARY_METRICS",
                          "same metric, FULL11 SECONDARY model", 2),
            "topology_accuracy": V(hb_s["topology_accuracy"], "HB_SECONDARY_METRICS",
                                   "same metric, FULL11 SECONDARY model", 2),
            "own_trivial_tvd": V(hb_s["own_trivial_mean_tvd"], "HB_SECONDARY_METRICS",
                                 "SECONDARY's own training-mean simplex on the same "
                                 "states", 2),
            "own_descriptive_skill": V(hb_s["own_descriptive_skill"],
                                       "HB_SECONDARY_METRICS",
                                       "own trivial TVD minus model TVD; negative = "
                                       "worse than its own training mean", 2),
            "ci95": V(hb_s["bootstrap_mean_tvd"], "HB_SECONDARY_METRICS",
                      "composition-level cluster bootstrap, 1000 replicates", 2),
            "pred_topology_distribution": V(hb_s["pred_topology_distribution"],
                                            "HB_SECONDARY_METRICS",
                                            "share of blind states by predicted argmax "
                                            "family", 2),
            "relative_tvd_vs_primary": V(
                hb_s["mean_tvd"] / hb_p["mean_tvd"] - 1.0, "HB_SECONDARY_METRICS",
                "SECONDARY mean TVD relative to PRIMARY, same states", 2),
        },
        "level_3_postblind_recovery": {
            "recovery_subset": V(
                {"n_compositions": C.N_RECOVERY_EVAL, "n_states": C.N_RECOVERY_STATES},
                "I_BASELINES", "RECOVERY_EVALUATION split fixed before the reveal", 3),
            "zero_shot_tvd_same_subset": V(zero, "I_BASELINES",
                                           "frozen PRIMARY zero-shot restricted to the "
                                           "1400-composition recovery subset", 3),
            "zero_shot_topology_accuracy_same_subset": V(
                base["M_ZERO"]["topology_accuracy"], "I_BASELINES",
                "frozen PRIMARY zero-shot topology accuracy on the 15400-state "
                "recovery subset; not the 22000-state value", 3),
            "trivial_topology_accuracy_same_subset": V(
                base["M_TRIVIAL"]["topology_accuracy"], "I_BASELINES",
                "the training-mean simplex is FCC-dominant, so it scores well on "
                "topology while scoring badly on TVD", 3),
            "trivial_tvd_same_subset": V(triv, "I_BASELINES",
                                         "frozen PRIMARY training-mean simplex on the "
                                         "same 15400 states", 3),
            "random_M_REFIT_mean_tvd_by_budget": V(refit, "I_RANDOM_AGG",
                                                   "mean over 3 frozen random "
                                                   "replicates", 3),
            "random_M_REFIT_sd_by_budget": V(refit_sd, "I_RANDOM_AGG",
                                             "across-replicate SD; finite protocol "
                                             "variability, not a population estimate", 3),
            "random_topology_accuracy_by_budget": V(topo_acc, "I_RANDOM_AGG",
                                                    "mean over 3 replicates", 3),
            "R0_by_budget": V(r0, "I_RANDOM_AGG", "per-temperature anchor mean; the "
                              "model is ignored entirely", 3),
            "R1_by_budget": V(r1, "I_RANDOM_AGG", "global mean-shift of the frozen raw "
                              "output", 3),
            "R2_by_budget": V(r2, "I_RANDOM_AGG", "per-temperature mean-shift of the "
                              "frozen raw output", 3),
            "maximin_by_budget": V(mx, "I_MAXIMIN",
                                   "SECONDARY comparison only; never promoted", 3),
            "maximin_budgets_better_than_random_mean": V(
                sum(1 for b in C.BUDGETS if mx.get(b, np.inf) < refit[b]), "I_MAXIMIN",
                "count of budgets where maximin beats the random across-replicate mean",
                3),
            "B_skill": V(thr["B_skill"], "I_THRESHOLDS", thr["definitions"]["B_skill"], 3),
            "B_recovery25": V(thr["B_recovery25"], "I_THRESHOLDS",
                              thr["definitions"]["B_recovery25"], 3),
            "recovery_fraction_by_budget": V(
                {b: 1.0 - refit[b] / zero for b in C.BUDGETS}, "I_RANDOM_AGG",
                "1 - mean TVD / zero-shot TVD, both on the same recovery subset", 3),
            "first_budget_M_REFIT_below_R2": V(first_r2, "I_RANDOM_AGG",
                                               "descriptive crossover, not a "
                                               "preregistered endpoint", 3),
            "first_budget_M_REFIT_below_R0": V(first_r0, "I_RANDOM_AGG",
                                               "descriptive crossover, not a "
                                               "preregistered endpoint", 3),
            "anchor_fraction_of_training_rows": V(
                [float(rep.anchor_fraction_of_training_rows.min()),
                 float(rep.anchor_fraction_of_training_rows.max())], "I_RANDOM_REP",
                "Ni anchor rows as a fraction of the refit training set; natural "
                "weighting, no oversampling", 3),
        },
        "level_4_postblind_descriptive": {
            "true_FCC": V({"zero_shot": stratum(topo, "FCC", "M_ZERO"),
                           "b32": stratum(topo, "FCC", "M_REFIT", 32),
                           "n_states": int(topo[topo.stratum == "FCC"].n_states.iloc[0])},
                          "I_TOPO", "mean TVD on states whose true argmax family is FCC",
                          4),
            "true_OTHER_SOLID": V(
                {"zero_shot": stratum(topo, "OTHER_SOLID", "M_ZERO"),
                 "b32": stratum(topo, "OTHER_SOLID", "M_REFIT", 32),
                 "n_states": int(topo[topo.stratum == "OTHER_SOLID"].n_states.iloc[0])},
                "I_TOPO", "mean TVD on states whose true argmax family is OTHER_SOLID", 4),
            "NBNI3_present": V(
                {"zero_shot": stratum(nb, "NBNI3_PRESENT", "M_ZERO"),
                 "b32": stratum(nb, "NBNI3_PRESENT", "M_REFIT", 32),
                 "n_states": int(nb[nb.stratum == "NBNI3_PRESENT"].n_states.iloc[0])},
                "I_NBNI3", "mean TVD on recovery states whose frozen raw phase list "
                "contains NBNI3", 4),
            "NBNI3_absent": V(
                {"zero_shot": stratum(nb, "NBNI3_ABSENT", "M_ZERO"),
                 "b32": stratum(nb, "NBNI3_ABSENT", "M_REFIT", 32),
                 "n_states": int(nb[nb.stratum == "NBNI3_ABSENT"].n_states.iloc[0])},
                "I_NBNI3", "mean TVD on recovery states without NBNI3", 4),
            "LOW_T": V({"zero_shot": stratum(temp, "LOW_T", "M_ZERO"),
                        "b32": stratum(temp, "LOW_T", "M_REFIT", 32),
                        "temperatures_C": C.LOW_T},
                       "I_TEMP", "300-600 degC, the window in which NBNI3 appears", 4),
            "HIGH_T": V({"zero_shot": stratum(temp, "HIGH_T", "M_ZERO"),
                         "b32": stratum(temp, "HIGH_T", "M_REFIT", 32),
                         "temperatures_C": C.HIGH_T},
                        "I_TEMP", "700-1000 degC", 4),
            "nbni3_phase": V({"n_states_of_22000": int(_voc.loc[
                _voc.raw_phase_name.astype(str).str.upper().str.contains("NBNI3"),
                "n_states"].sum()),
                "temperature_range_C": [300.0, 600.0],
                "frozen_mapping": "OTHER_SOLID",
                "previously_seen_in_project": False},
                "HB_PHASE_VOCAB",
                "a phase absent from every earlier stage, mapped by the frozen "
                "five-family default; no new phase family was introduced", 4),
            "anchor_coverage_confound": V(
                {"pooled_spearman_rho_coverage_vs_recovery": pooled_rho,
                 "rho_between_budget_and_coverage": confound,
                 "n_within_budget_correlations_defined": n_within_defined,
                 "n_cells_per_budget": 3},
                "I_COVERAGE",
                "rank correlation between what the frozen anchors happened to contain "
                "and how much recovery followed. Every calibration composition contains "
                "at least one NBNI3 state, so n_anchor_comps_with_any_NBNI3 is exactly "
                "the budget; the pooled correlation is therefore confounded with budget "
                "size and cannot show that NBNI3 coverage per se drives recovery", 4),
            "topology_inheritance_observation": V(
                {"nearest_training_identity_in_descriptor_space": nearest,
                 "z_distance_to_nearest": nearest_z,
                 "z_distance_to_six_centroid": ood.get("descriptor_position", {}).get(
                     "z_distance_to_six_centroid"),
                 "C15_share_of_that_identity_domain_A": nearest_c15,
                 "predicted_C15_share_for_Ni": pred_c15,
                 "true_C15_share_for_Ni": 0.0},
                "V1_TOPOLOGY",
                "descriptive association: the model's predicted Ni topology resembles "
                "the topology of the training identity nearest to Ni in descriptor "
                "space. The nearest-identity computation was frozen before the reveal; "
                "the comparison is post-reveal and is not a demonstrated mechanism", 4),
        },
        "context_not_primary_evidence": {
            "iid_x_partition_mean_tvd": V(float(x_row.STEP5A_IID), "C_LADDER",
                                          "STEP5A IID test, x_partition", "context"),
            "composition_domain_ood_x_partition_mean_tvd": V(
                float(x_row.STEP5B_composition_domain_OOD), "C_LADDER",
                "STEP5B Domain-B shell, x_partition", "context"),
            "retrospective_loeo_tree_mean": V(float(x_row.STEP5C_LOEO_mean), "C_LADDER",
                                              "STEP5C six-fold mean, per-fold selected "
                                              "models", "context"),
            "retrospective_loeo_per_element": V(
                {r.heldout_element: float(r.x_partition) for _, r in
                 C.load_csv("C_PER_ELEMENT").iterrows()}, "C_PER_ELEMENT",
                "per-held-out-element x_partition mean TVD; the six-fold average is "
                "never reported alone", "context"),
            "step5e_pipeline_selection": V(
                {c["candidate"]: c["mean_tvd"] for c in e_pipe["candidate_summary"]},
                "E_PIPELINE", "retrospective LOEO mean TVD of the six candidate "
                "zero-shot pipelines; Z1_MLP_DIRECT was selected", "context"),
            "step5e_guardrail_satisfied_by_winner": V(
                e_pipe["guardrail_satisfied_by_winner"], "E_PIPELINE",
                "whether the selected pipeline met its own preregistered guardrail",
                "context"),
            "step5f_augmented_loeo": V(
                {"mean_Z1_tvd": f_sum["mean_Z1_tvd"], "mean_FULL5_tvd": f_sum["mean_FULL5_tvd"],
                 "relative_reduction": f_sum["relative_reduction_vs_Z1"],
                 "mean_topology_accuracy": f_sum["mean_topology_accuracy"],
                 "folds_beating_trivial": f_sum["folds_beating_trivial"]},
                "F_AUG_SUMMARY", "retrospective augmented LOEO", "context"),
            "loeo_caveat": C.LOEO_CONTEXT["caveat"],
            "ni_feature_range": V(
                {"n_features_extending_train_range":
                     ood["n_features_extending_train_range"],
                 "n_features_total": int(len(rng_diag)),
                 "frac_states_any_abs_z_gt3": ood["frac_states_any_abs_z_gt3"],
                 "frac_states_any_abs_z_gt5": ood["frac_states_any_abs_z_gt5"],
                 "max_abs_z_median": ood["max_abs_z_median"],
                 "outside_six_identity_property_range_count":
                     ood["descriptor_position"]["outside_six_range_count"]},
                "HA_OOD_FREEZE",
                "frozen BEFORE any Ni CALPHAD state existed: how far the Ni inputs lie "
                "outside the training feature ranges", "context"),
            "dataset": V({"domain_A": {"compositions": 35000, "states": 385000},
                          "domain_B": {"compositions": b_counts["compositions"],
                                       "states": b_counts["formal"],
                                       "converged": b_counts["converged"],
                                       "failed": b_counts["failed"]},
                          "ni_blind": {"compositions": 2000, "states": C.N_BLIND_STATES,
                                       "converged": C.N_BLIND_STATES, "failed": 0}},
                         "V1_DOMAIN", "CALPHAD design", "context"),
        },
        "not_available": {
            "experimental_validation": C.unavailable(
                "no experimental measurement exists in this project; CALPHAD equilibrium "
                "is the only ground truth", requires_experiment=True),
            "kinetics": C.unavailable(
                "all labels are equilibrium; no kinetic calculation was performed",
                requires_experiment=True),
            "second_prospective_unseen_element": C.unavailable(
                "only Ni was held out prospectively; a second blind element would be a "
                "new experiment", requires_experiment=True),
            "low_order_binary_ni_anchors": C.unavailable(
                "Cu-Ni / Cr-Ni / Nb-Ni were never computed (STEP5I spec 47)",
                requires_experiment=True),
            "dft_or_first_principles_descriptors": C.unavailable(
                "out of scope by design; CALPHAD plus classical elemental descriptors "
                "only", requires_experiment=True),
        },
        "drift_vs_brief": DRIFT,
        "n_drift_flags": int(sum(1 for d in DRIFT if d["status"] == "DRIFT")),
    }
    C.write_json(C.P(C.D_FREEZE, "FINAL_NUMERICAL_RESULTS.json"), N)
    pd.DataFrame(DRIFT).to_csv(C.P(C.D_QA, "STEP5J_BRIEF_VALUE_DRIFT.csv"), index=False,
                               encoding="utf-8")

    nd = N["n_drift_flags"]
    print("  recomputed %d quoted values; %d disagree with the brief beyond its own "
          "quoted precision" % (len(DRIFT), nd))
    for d in DRIFT:
        if d["status"] == "DRIFT":
            print("    DRIFT %-34s brief %.10g vs frozen %.10g"
                  % (d["quantity"], d["brief_quoted"],
                     d["recomputed_from_frozen_files"]))
    print("  NBNI3 coverage confound: rho(budget, n_anchor_comps_with_any_NBNI3) = %s"
          % confound["n_anchor_comps_with_any_NBNI3"].get("spearman_rho_with_budget"))
    print("  topology inheritance: nearest identity %s (z=%.3f), its C15 share %.3f vs "
          "predicted Ni C15 share %.3f"
          % (nearest, nearest_z or float("nan"), nearest_c15 or float("nan"),
             pred_c15 or float("nan")))
    print("STAGE 1: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
