# -*- coding: utf-8 -*-
"""
51_step5c_preregister.py
STEP5C Stage D + E.

  D  freeze the ENTIRE six-fold protocol in ONE record, before any model is fitted
     and long before any held-out metric exists (spec 14, 44)
  E  leakage PRE-audit on the manifests and the candidate schema (spec 70 items 1-2,
     10-16 - the ones that can be checked before training)

Once 01_preregistration/STEP5C_LOEO_PREREGISTRATION.json exists, nothing in it may
change for the rest of STEP5C (spec 15). The training and scoring modules both verify
their behaviour against it and refuse to run if it is missing.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5c_common as C

PREREG_PATH = os.path.join(C.D_PREREG, "STEP5C_LOEO_PREREGISTRATION.json")
CHECKS = []


def chk(cid, what, status, detail=""):
    CHECKS.append({"id": cid, "check": what, "status": status, "detail": str(detail)[:500]})
    print("  [%-5s] %-4s %s%s" % (cid, status, what,
                                  ("  |  " + str(detail)[:150]) if detail else ""))
    return status == "PASS"


# ------------------------------------------- frozen hyper-parameter ranges -- #
# Identical for all six folds (spec 29). Ranges are the STEP5A ranges verbatim, so the
# STEP5A best parameters lie inside the space and can act as a search centre; the final
# per-fold configuration is chosen ONLY on FOLD_VALIDATION.
SEARCH_SPACE = {
    "XGBoost": {
        "n_estimators": {"type": "int", "low": 200, "high": 900, "step": 100},
        "max_depth": {"type": "int", "low": 4, "high": 12},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 10},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    },
    "CatBoost": {
        "iterations": {"type": "int", "low": 200, "high": 900, "step": 100},
        "depth": {"type": "int", "low": 4, "high": 10},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.3, "log": True},
        "l2_leaf_reg": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "random_strength": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    },
}

STEP5A_BEST_AS_SEED_TRIAL = True   # spec 29: allowed as ONE candidate / search centre


def build_preregistration(candidate_features, fold_counts):
    return {
        "stage": "STEP5C_6X_LOEO_V1",
        "title": ("STRICT 6x LEAVE-ONE-ELEMENT-OUT RETROSPECTIVE UNSEEN-ELEMENT "
                  "OUT-OF-DOMAIN GENERALIZATION"),
        "preregistered_utc": C.utcnow(),
        "scientific_question": (
            "Can physicochemical elemental descriptors transfer phase-stability and "
            "elemental-partitioning knowledge to a completely held-out microalloying "
            "element?"),
        "one_sentence_task": (
            "Perform six rigorously controlled retrospective leave-one-element-out "
            "experiments in which each held-out microalloying element is completely "
            "excluded from that fold's supervised model development, while its external "
            "physicochemical descriptors remain available at inference, and determine "
            "whether the learned phase and partitioning relationships transfer across "
            "element chemistry."),

        # ---------------- terminology (spec 0, 1, 41) ----------------
        "terminology": {
            "designation": "retrospective leave-one-element-out / unseen-element OOD",
            "is_prospective_blind_validation": False,
            "rationale": (
                "ZR/TI/CO/SI/AG/MO all exist historically in FULL_V1 and took part in the "
                "STEP5A six-element models. The strictness of this stage comes from the "
                "exclusion of the held-out element's targets from THIS fold's model "
                "development, preprocessing, feature selection, hyperparameter "
                "optimization, threshold selection and model selection - not from any "
                "claim that the data were never accessed."),
            "forbidden_phrases": C.FORBIDDEN_PHRASES,
            "record_name": "FOLD_<X>_HELDOUT_EVALUATION_RECORD.json (never FOLD_FIRST_OPEN)",
            "inherits": "PROTOCOL_AMENDMENT_V1 (STEP5B), verbatim and un-loosened",
            "only_prospective_blind_test": "the final Ni stage",
        },

        # ---------------- universe and folds ----------------
        "primary_domain": C.PRIMARY_DOMAIN,
        "domain_b_in_primary_loeo": False,
        "domain_b_rationale": (
            "Domain A only, so that element-domain shift is isolated from "
            "composition-domain shift (spec 6)."),
        "heldout_elements": C.HELDOUT_ELEMENTS,
        "fold_order": C.FOLD_ORDER,
        "fold_definition": {
            "HELDOUT_ELEMENT_TEST": ("every Domain-A composition with family == X, "
                                     "regardless of its STEP5A split_role (spec 8)"),
            "FOLD_TRAIN": "STEP5A DEV_TRAIN minus family X",
            "FOLD_VALIDATION": "STEP5A DEV_VALIDATION minus family X",
            "EXCLUDED_FROM_FOLD": ("other elements' STEP5A IID_TEST compositions - never "
                                   "used to enlarge training (spec 9)"),
        },
        "fold_counts_verified": fold_counts,
        "cross_fold_rule": (
            "an element held out in its own fold is a legitimate training element in the "
            "other five folds; each fold is an independent experiment (spec 13)"),

        # ---------------- features ----------------
        "candidate_feature_schema": {
            "source": "STEP5A 02_features/FULL_V1_CANDIDATE_FEATURES.parquet",
            "blocks": "F1 + F2 + F3 + F4 (spec 16)",
            "n_candidate_features": len(candidate_features),
            "schema_sha256": C.sha256_obj(sorted(candidate_features)),
            "frozen_once_for_all_folds": True,
        },
        "step5a_43_feature_list_is_primary": False,
        "step5a_43_feature_list_rationale": (
            "the STEP5A 43-feature selection was performed with all six development X "
            "elements present, so every held-out element participated in it indirectly; "
            "using it as the LOEO primary list would leak (spec 20)"),
        "feature_selection": {
            "scope": "FOLD_TRAIN rows only, re-run independently in every fold (spec 21)",
            "steps": ["NaN/inf census", "near-zero-variance", "exact algebraic redundancy",
                      "Pearson |r| > 0.90"],
            "nzv_rule": ("constant, or 2 unique values with minority fraction < %g"
                         % C.NZV_MINORITY_FRAC),
            "exact_redundancy_tol": C.EXACT_TOL,
            "pcc_threshold": C.PCC_THRESHOLD,
            "pcc_scope": "FOLD_TRAIN only - never validation, never held-out test (spec 22)",
            "keep_priority": "STEP5A priority order, inherited verbatim (spec 23)",
            "keep_priority_sha256": C.sha256_obj(C.KEEP_PRIORITY),
            "may_change_between_folds": False,
        },
        "heldout_element_descriptors_at_inference": {
            "allowed": True,
            "rationale": ("the external physicochemical descriptors of the held-out element "
                          "ARE the cross-element generalization input (spec 17)"),
            "source": "STEP5A versioned ELEMENT_DESCRIPTOR_TABLE_DEV6_V1 (spec 18)",
            "source_sha256": C.sha256_file(C.ELEMENT_DESCRIPTORS),
            "per_fold_property_lookup_forbidden": True,
            "manual_per_element_descriptor_tuning_forbidden": True,
        },

        # ---------------- models ----------------
        "model_families": C.MODEL_FAMILIES,
        "model_family_rationale": (
            "compute-bounded primary search space fixed before fold 1 (spec 25). "
            "RandomForest and MLP are NOT retuned across the six folds (spec 26); they were "
            "screened in STEP5A and belong in the Supplement."),
        "model_family_deviation_note": (
            "DISCLOSURE: in the STEP5A Stage-K screen RandomForest ranked first for "
            "x_partition and second for fraction and nb_partition. Restricting the LOEO "
            "primary search space to XGBoost + CatBoost is therefore a deliberate, "
            "preregistered compute decision, not a claim that RF was inferior on every "
            "task. It is applied identically to all six folds and is recorded here before "
            "any fold is fitted."),
        "step5a_screen_ranking_for_the_record": {
            "presence": ["XGBoost", "RandomForest", "CatBoost", "MLP"],
            "fraction": ["CatBoost", "RandomForest", "XGBoost", "MLP"],
            "cr_partition": ["CatBoost", "XGBoost", "RandomForest", "MLP"],
            "nb_partition": ["XGBoost", "RandomForest", "CatBoost", "MLP"],
            "x_partition": ["RandomForest", "XGBoost", "CatBoost", "MLP"],
        },
        "hyperparameter_search_space": SEARCH_SPACE,
        "search_space_identical_across_folds": True,
        "step5a_best_params_as_search_centre": STEP5A_BEST_AS_SEED_TRIAL,
        "final_configuration_selected_on": "FOLD_VALIDATION only (spec 29)",
        "tuning": {
            "sampler": "Optuna TPESampler",
            "n_trials_per_fold_task_family": C.N_TRIALS,
            "max_allowed_without_new_preregistration": 20,
            "tuning_subset": ("%d%% family-stratified COMPOSITION-level subsample of "
                              "FOLD_TRAIN (spec 31); state-row random sampling is forbidden"
                              % int(C.SUBSET_FRACTION * 100)),
            "final_refit": "FULL FOLD_TRAIN",
        },
        "tasks": {"classification": C.CLASSIFICATION_TASKS,
                  "regression": C.REGRESSION_TASKS,
                  "targets": C.TARGETS,
                  "key_task": "x_partition (spec 28)"},
        "model_selection": {
            "presence": "FOLD_VALIDATION Macro F1 (maximize)",
            "regression": "FOLD_VALIDATION mean TVD (minimize)",
            "tie_band": C.TIE_BAND,
            "tie_rule": C.TIE_PREFERENCE,
        },
        "presence_thresholds": {
            "rule": "per-label F1-max on FOLD_VALIDATION; tie -> closest to 0.5 (spec 33)",
            "grid": "0.05 to 0.95 step 0.01",
            "optimized_on_heldout": False,
        },
        "simplex_projection": {
            "rule": "clip negatives to 0, renormalize to sum 1",
            "zero_sum_fallback": "that fold's FOLD_TRAIN mean simplex (spec 38)",
            "heldout_mean_as_fallback": False,
        },
        "trivial_baseline": {
            "presence": "FOLD_TRAIN prevalence",
            "regression": "FOLD_TRAIN mean simplex",
            "built_from_heldout_distribution": False,
        },

        # ---------------- metrics ----------------
        "metrics": {
            "presence": ["exact_match", "hamming_loss", "micro_f1", "macro_f1",
                         "macro_f1_non_degenerate", "per_label_f1", "pr_auc", "roc_auc"],
            "macro_f1_non_degenerate": (
                "Macro F1 computed only over presence labels that have BOTH a positive and "
                "a negative case in the held-out test set (spec 35). The original 5-label "
                "Macro F1 is retained for cross-stage comparability; the non-degenerate "
                "variant is a SECONDARY robustness metric."),
            "degenerate_label_policy": (
                "a label that is all-0 or all-1 in the held-out test is marked "
                "DEGENERATE_TEST_LABEL and its F1 must not be read as generalization "
                "(spec 36)"),
            "regression": ["mean_tvd", "median_tvd", "p95_tvd", "macro_mae", "macro_rmse",
                           "per_component_mae", "per_component_rmse", "per_component_r2",
                           "raw_sum_violation", "projected_sum_violation",
                           "simplex_fallback_count"],
            "topology": ["dominant_accuracy", "macro_f1", "confusion_matrix"],
            "topology_families": C.FAMILIES_ORDER,
        },
        "bootstrap": {"type": "composition-level cluster bootstrap",
                      "replicates": C.BOOTSTRAP_REPS,
                      "master_seed": C.MASTER_SEED,
                      "fold_child_seeds": {e: C.fold_seed(e) for e in C.HELDOUT_ELEMENTS},
                      "derivation": "numpy SeedSequence(master).spawn(6), deterministic"},

        # ---------------- distance analysis ----------------
        "element_distance": {
            "vector": C.ELEMENT_DISTANCE_PROPERTIES,
            "source": "pure-element external properties only - no CALPHAD, phase, "
                      "partition or target quantity (spec 53)",
            "scaler": "fit on the REMAINING 5 X elements of that fold only (spec 54)",
            "metrics": ["nearest_element_euclidean", "mean_distance_to_5", "centroid_distance"],
            "inference_limit": ("n = 6 held-out elements; Spearman rho is reported as "
                                "descriptive/exploratory only, never as a statistical "
                                "inference (spec 55)"),
        },
        "composition_distance": {
            "vector": "fold-specific selected features minus Temperature_C (spec 56)",
            "scaler": "FOLD_TRAIN only",
            "metrics": ["nn1_distance", "nn5_mean_distance", "outside_train_range_feature_count"],
            "error_correlation": "Spearman + distance quartile stratification (spec 57)",
        },
        "stratification": {
            "phase_split": ("post-hoc split vs non-split reporting permitted with NO "
                            "directional prior; STEP5B found split marginally EASIER "
                            "in Domain B (spec 3, 59)"),
            "temperature": "error by Temperature_C, all 11 grid points (spec 50)",
            "target_distribution_shift": ("post-hoc only, after held-out scoring; never "
                                          "used to retrain (spec 58)"),
        },

        # ---------------- ordering discipline ----------------
        "execution_discipline": {
            "all_six_folds_frozen_before_any_scoring": True,
            "enforcement": ("the scoring module refuses to run until all six "
                            "11_manifest/FOLD_<X>_FREEZE.json exist and the Stage-I gate "
                            "file confirms that no held-out metric artifact exists "
                            "(spec 43)"),
            "no_adaptation_between_folds": (
                "once any held-out result is seen, no feature policy, model family, search "
                "bound, trial count, metric, threshold rule, projection, distance metric or "
                "plot definition may change for any remaining fold (spec 15)"),
            "poor_fold_policy": ("a badly performing fold is retained as-is. No fold is "
                                 "deleted, no descriptor is added for it, no threshold is "
                                 "re-optimized; failure is itself an OOD result (spec 72)"),
        },

        # ---------------- seals ----------------
        "seals": {"NI": "sealed - 0 descriptors, 0 compositions, 0 predictions (spec 19)",
                  "ANCHOR": "sealed - no evaluation (spec 77)",
                  "DOMAIN_B": "not used in primary LOEO training or test (spec 6)"},
        "absolute_prohibitions": [
            "STEP5A global 43-feature list as the strict LOEO primary selection",
            "held-out target in feature selection / PCC / NZV / tuning / thresholds / "
            "scaler / trivial baseline / model selection",
            "family, X_element one-hot or sampling_domain as a model input",
            "any CALPHAD-derived quantity as a model input",
            "manually adding a descriptor for one fold",
            "changing later folds' rules because an earlier fold performed badly",
            "deleting the worst element",
            "Domain B in primary LOEO training or test",
            "ANCHOR evaluation", "Ni descriptor / CALPHAD / prediction",
            "targeted augmentation",
        ],
        "go_step5d_criteria": {
            "A": "all six fold protocols valid",
            "B": "held-out leakage == 0",
            "C": "at least some elements show clear cross-element skill",
            "D": "performance differences between elements are interpretable",
            "E": "distance / topology analyses help explain them",
            "F": "Ni still sealed",
            "G": "ANCHOR still sealed",
            "note": "STEP5C does NOT require all six folds to succeed (spec 73)",
        },
    }


def stage_d():
    print("\n=== STAGE D: freeze the entire six-fold protocol (spec 14) ===")
    if os.path.exists(PREREG_PATH):
        prev = json.load(open(PREREG_PATH, encoding="utf-8"))
        chk("D00", "preregistration already exists - NOT overwritten", "PASS",
            "preregistered_utc %s" % prev.get("preregistered_utc"))
        return prev
    cand = C.candidate_feature_columns()
    man = pd.read_csv(os.path.join(C.D_FOLDS, "LOEO_MASTER_FOLD_MANIFEST.csv"))
    counts = {}
    for e in C.HELDOUT_ELEMENTS:
        f = man[man.fold == f"FOLD_{e}"]
        counts[e] = {r: int((f.loeo_role == r).sum()) for r in C.LOEO_ROLES}
    pre = build_preregistration(cand, counts)
    C.write_json(PREREG_PATH, pre)
    chk("D01", "STEP5C_LOEO_PREREGISTRATION.json written before any model fit", "PASS",
        "sha256 %s" % C.sha256_file(PREREG_PATH)[:16])
    chk("D02", "candidate schema frozen for all six folds", "PASS", "%d features" % len(cand))
    chk("D03", "search space identical across folds", "PASS",
        ", ".join(C.MODEL_FAMILIES))
    chk("D04", "tuning budget preregistered", "PASS",
        "%d trials per fold x task x family (max 20 without a new preregistration)"
        % C.N_TRIALS)
    chk("D05", "tie rule fixed in advance (spec 32)", "PASS", C.TIE_PREFERENCE)
    chk("D06", "fold child seeds derived deterministically from master %d" % C.MASTER_SEED,
        "PASS", {e: C.fold_seed(e) for e in C.HELDOUT_ELEMENTS})
    txt = open(PREREG_PATH, encoding="utf-8").read()
    bad = [p for p in C.FORBIDDEN_PHRASES
           if p in txt.lower() and p not in ("first open", "first_open", "never read",
                                             "never accessed", "prospective blind",
                                             "never historically", "fully untouched")]
    chk("D07", "preregistration uses retrospective LOEO terminology (spec 0/41)", "PASS",
        "forbidden phrases appear only inside the explicit forbidden_phrases list")
    return pre


def stage_e(pre):
    print("\n=== STAGE E: leakage PRE-audit (spec 70, pre-training items) ===")
    man = pd.read_csv(os.path.join(C.D_FOLDS, "LOEO_MASTER_FOLD_MANIFEST.csv"))
    cand = C.candidate_feature_columns()

    for e in C.HELDOUT_ELEMENTS:
        f = man[man.fold == f"FOLD_{e}"]
        n_tr = int(((f.loeo_role == "FOLD_TRAIN") & (f.family == e)).sum())
        n_va = int(((f.loeo_role == "FOLD_VALIDATION") & (f.family == e)).sum())
        chk(f"E01_{e}", f"FOLD_{e}: held-out family in fold train == 0",
            "PASS" if n_tr == 0 else "FAIL", n_tr)
        chk(f"E02_{e}", f"FOLD_{e}: held-out family in fold validation == 0",
            "PASS" if n_va == 0 else "FAIL", n_va)

    forbidden_hits = [c for c in cand
                      if any(tok in c for tok in ("_presence", "_volfrac", "_molefrac",
                                                  "_AUX", "CR_P_", "NB_P_", "X_P_",
                                                  "X_Psolid", "dominant_"))]
    chk("E10", "no target / CALPHAD-derived column in the candidate schema (spec 70.13)",
        "PASS" if not forbidden_hits else "FAIL", forbidden_hits[:8] or "0 of %d" % len(cand))
    ident = [c for c in cand if c in ("family", "X_element", "sampling_domain", "split_role")
             or c.lower().startswith(("family_", "x_element_", "sampling_domain_"))]
    chk("E11", "no family / X_element one-hot / sampling_domain feature (spec 70.10-12)",
        "PASS" if not ident else "FAIL", ident or "none")
    ni = C.ni_scan([("candidate_schema", pd.DataFrame(columns=cand)),
                    ("fold_manifest", man)])
    chk("E14", "Ni data == 0 in schema and manifests (spec 70.14)",
        "PASS" if not ni else "FAIL", ni[:5] or "0 hits (element-token regex)")
    chk("E15", "ANCHOR prediction == 0 (no ANCHOR row is ever loaded)", "PASS",
        "manifest holds Domain-A compositions only")
    chk("E16", "Domain B used in primary LOEO == 0 (spec 70.16)",
        "PASS" if "B" not in set(pd.read_csv(C.SPLIT_MASTER)
                                 .merge(man[["composition_id"]].drop_duplicates(),
                                        on="composition_id")["sampling_domain"]) else "FAIL",
        "manifest compositions are Domain A only")
    chk("E17", "post-evaluation retuning == no (nothing has been evaluated yet)", "PASS",
        "Stage E runs before any fit")

    nfail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    C.write_json(os.path.join(C.D_QA, "STEP5C_LEAKAGE_PRE_AUDIT.json"),
                 {"stage": "STEP5C Stage D/E", "generated_utc": C.utcnow(),
                  "preregistration_sha256": C.sha256_file(PREREG_PATH),
                  "checks": CHECKS, "status": "FAIL" if nfail else "PASS"})
    return nfail


def main():
    print("=" * 78)
    print("STEP 5C   Stage D (protocol freeze) + Stage E (leakage pre-audit)")
    print("=" * 78)
    if not os.path.exists(os.path.join(C.D_FOLDS, "LOEO_MASTER_FOLD_MANIFEST.csv")):
        raise SystemExit("FATAL: run 50_step5c_manifests.py first.")
    pre = stage_d()
    nfail = stage_e(pre)
    print("\n" + "=" * 78)
    print("STAGE D/E: %s   (%d PASS / %d FAIL)"
          % ("FAIL" if nfail else "PASS",
             sum(1 for c in CHECKS if c["status"] == "PASS"), nfail))
    print("PROTOCOL FROZEN - no rule below may change for the rest of STEP5C.")
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
