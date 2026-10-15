# -*- coding: utf-8 -*-
"""61_step5e_preregister.py — Stage B: freeze the ENTIRE STEP5E protocol.
Stage C: freeze the retrospective recovery splits and few-shot sequences
(target-free; uses only input feature geometry). No model is fit here."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5e_common as C

TOPO_CLF_GRID = C.TOPO_CLF_GRID


def build_prereg():
    import importlib.metadata as mdv
    versions = {p: mdv.version(p) for p in
                ["numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "catboost",
                 "torch", "optuna", "matminer", "pymatgen", "scipy"]}
    pre = {
        "stage": "STEP5E_XPART_REMEDY_V1",
        "title": "Preregistered retrospective X-partition remedy study",
        "preregistered_utc": C.utcnow(),
        "terminology": C.DISCLAIMER_OPEN
        + " This is a RETROSPECTIVE remedy / retrospective few-shot recovery study. "
          "ZR/TI/CO/SI/AG/MO targets historically exist and were evaluated in STEP5C; "
          "nothing here is prospective, blind, or never-seen. The prospective blind "
          "unseen-element test (Ni) remains sealed and is NOT part of STEP5E. "
        + C.DISCLAIMER_CLOSE,
        "primary_task_only": "X partition (spec 12); presence/fraction/Cr/Nb are NOT retrained",
        "folds": C.FOLDS,
        "fold_data_source": "STEP5C LOEO_MASTER_FOLD_MANIFEST.csv roles reused verbatim; "
                            "features from the frozen STEP5A candidate parquet (Domain A only)",
        "frozen_zero_shot_references": {
            "Z0_TREE_DIRECT": "STEP5C FOLD_<X>_HELDOUT_PREDICTIONS.parquet X_P_*_proj (never refit)",
            "Z1_MLP_DIRECT": "STEP5D 07_model_bias_diagnostic/FOLD_<X>/DIAGNOSTIC_HELDOUT_PREDICTIONS.parquet "
                             "MLP_* columns (never refit)"},
        "candidate_branches": {
            "Z2": "affinity-enhanced direct XGBoost (STEP5C fold XGB hyperparameters)",
            "Z3": "topology-first XGBoost on original fold features",
            "Z4": "affinity-enhanced topology-first XGBoost",
            "Z5": "affinity-enhanced direct MLP 128/64/5"},
        "affinity_descriptors": {
            "source": "matminer Miedema featurizer (deterministic implementation of "
                      "Miedema's model with the de Boer et al. 1988 parameter tables, "
                      "shipped inside matminer)",
            "quantity": "Miedema_deltaH_inter at equiatomic 50:50 composition "
                        "(intermetallic formation enthalpy, Miedema model)",
            "unit": "eV/atom",
            "composition_definition": "equiatomic 50:50 binary, identical for all six X "
                                      "and all three base partners (spec 21)",
            "required_pairs": ["X-Cu", "X-Cr", "X-Nb"],
            "features": {
                "A1": "H_XCu", "A2": "H_XCr", "A3": "H_XNb",
                "A4": "H_XCr - H_XCu", "A5": "H_XNb - H_XCu", "A6": "H_XNb - H_XCr",
                "A7": "min(H_XCu,H_XCr,H_XNb)", "A8": "max(...)", "A9": "range(...)"},
            "sign_convention": "taken verbatim from the source; no absolute values",
            "ni_rule": "the matminer parameter table may contain Ni; Ni pairs are NEVER "
                       "queried; descriptor table contains only ZR TI CO SI AG MO (spec 23)",
            "filtering": "finite + constant + exact-duplicate check only; no heldout-aware "
                         "selection; no new PCC (spec 27)"},
        "affinity_feature_filtering": "finite/constant/exact-duplicate only (spec 27)",
        "topology_first": {
            "principle": "96.696% of summed X-partition TVD comes from wrong-dominant-topology "
                         "states (STEP5D frozen finding); hierarchical target formulation",
            "stage1_target": "dominant family = argmax of TRUE X-partition vector over the five "
                             "core families; labels from FOLD_TRAIN only (spec 32)",
            "classifier": "XGBoost multiclass, fixed family; preregistered validation-only "
                          "grid: " + str(TOPO_CLF_GRID) + " selected on fold-validation "
                          "multiclass macro F1 (spec 33)",
            "stage2": "topology-conditional XGBoost regressor per dominant class k "
                      "(same hyperparameters as Z2), trained on FOLD_TRAIN states with "
                      "true dominant topology == k (spec 35)",
            "expert_support_thresholds": {"min_training_compositions": C.MIN_EXPERT_COMPS,
                                          "min_training_states": C.MIN_EXPERT_STATES,
                                          "else": "global direct-regressor fallback "
                                                  "(original features, Z2 hyperparameters)"},
            "routing": "hard routing: predict dominant topology -> route to the conditional "
                       "regressor -> simplex projection (spec 37)",
            "oracle": "ORACLE_TOPOLOGY_CONDITIONAL routes with TRUE topology; labelled "
                      "DIAGNOSTIC_ORACLE, excluded from candidate ranking (spec 39-40)"},
        "mlp_architecture": {"layers": [128, 64, 5], "activations": "ReLU/ReLU/linear",
                             "scaler": "StandardScaler on FOLD_TRAIN only",
                             "learning_rate_candidates": C.MLP_LRS,
                             "selection": "fold-validation only", "max_epochs": C.MLP_EPOCHS,
                             "early_stopping_patience": C.MLP_PATIENCE},
        "simplex_projection": "clip negatives to 0; renormalize to sum 1; sum==0 -> fold-train "
                              "mean simplex; never uses held-out labels (spec 30)",
        "zero_shot_primary_metrics": ["mean_tvd", "median_tvd", "p95_tvd", "macro_mae",
                                      "topology_accuracy", "topology_macro_f1",
                                      "positive-skill folds vs fold-train trivial"] +
                                     ["per-fold mean_tvd for every branch (spec 61: never "
                                      "average-only)"],
        "zero_shot_selection_rule": {
            "primary": "lowest 6-fold mean TVD among Z0..Z5",
            "guardrail": "candidate must beat the fold-train trivial baseline on >=3/6 folds "
                         "AND mean topology accuracy must not be below Z0's",
            "fallback": "if the lowest-TVD candidate fails the guardrail, take the lowest-TVD "
                        "candidate satisfying it; if none does, keep the Z0/Z1 baseline that "
                        "better satisfies the rule",
            "fixed_before_any_heldout_metric": True,
            "purpose": "this selection defines the FUTURE Ni zero-shot pipeline; few-shot "
                       "results never influence it (spec 43)"},
        "few_shot": {
            "purpose": "answer whether the failure is caused by missing direct chemical "
                       "anchors; NOT the final Ni model (spec 44, 78)",
            "split_rule": "for each held-out element, order its 5,250 Domain-A compositions "
                          "by SHA256('{element}|{composition_id}|20260908') ascending; the "
                          "lowest 30% (~1575) = CALIBRATION_POOL, the remaining 70% (~3675) = "
                          "RETROSPECTIVE_RECOVERY_EVAL; verified at freeze time (spec 46)",
            "eval_terminology": "RECOVERY_EVAL is only the internal non-calibration subset; "
                                "it is NOT historically untouched (STEP5C evaluated it) (spec 47)",
            "budgets": C.FEWSHOT_BUDGETS,
            "nested": True,
            "each_composition": "all available temperature rows enter calibration together; "
                                "MO@950 failures not interpolated (spec 48)",
            "random_strategy": f"{C.N_RANDOM_REPLICATES} independent replicate sequences, "
                               "deterministic SeedSequence children of the master seed; "
                               "nested prefixes of one 32-composition permutation (spec 50)",
            "targeted_strategy": "target-free greedy maximin in composition-level feature "
                                 "space (Temperature_C and fold-constant columns removed; "
                                 "StandardScaler fit on FOLD_TRAIN only); first pick = "
                                 "largest 1-NN distance to FOLD_TRAIN, then maximize the "
                                 "minimum distance to FOLD_TRAIN + already selected (spec 51-53)",
            "architectures": {"F0": "direct Tree formulation (fold features, Z2 hyperparameters)",
                              "F1": "affinity + topology-first XGBoost (Z4 configuration); "
                                    "degrades to topology-first if affinity BLOCKED"},
            "no_retuning": "no Optuna, no feature reselection, no threshold policy change "
                           "after calibration labels are added; only weights/trees are refit "
                           "on FOLD_TRAIN + selected calibration compositions (spec 56-57)",
            "evaluation": "metrics ONLY on RETROSPECTIVE_RECOVERY_EVAL (spec 58)",
            "metrics": ["mean_tvd", "median_tvd", "p95_tvd", "topology_accuracy",
                        "topology_macro_f1", "skill_vs_trivial",
                        "relative_TVD_reduction_vs_zero_shot_same_subset"],
            "b_skill": "smallest budget whose TVD < fold-train trivial TVD; >32 if never",
            "b_topo50": "smallest budget with topology accuracy >= 0.50; >32 if never",
            "random_vs_targeted": "per budget: targeted TVD, random mean/std, difference, "
                                  "targeted percentile among random replicates; no pseudo "
                                  "state-level significance tests (spec 63)"},
        "internal_success_criteria": {
            "zero_shot_practical": "6-fold mean TVD -10% vs Z0 AND topology accuracy improved "
                                   "AND >=3/6 folds beat trivial",
            "zero_shot_strong": "6-fold mean TVD -25% vs Z0 AND >=4/6 folds beat trivial",
            "few_shot_rapid": "<=8 compositions beat trivial AND relative TVD reduction >=25%",
            "note": "internal only, never lowered post hoc (spec 79-80)"},
        "seeds": {"master": C.MASTER_SEED, "derivation": "numpy SeedSequence([20260908, replicate]) "
                   "for random replicate r; split ordering via SHA256 text"},
        "seals": {"Ni": "zero descriptors/properties/pairs/compositions/predictions",
                  "ANCHOR": "no training/tuning/prediction/evaluation",
                  "DomainB": "not used for tuning, descriptor selection, few-shot "
                             "calibration, or zero-shot model selection",
                  "new_CALPHAD": "0 Thermo-Calc / TC-Python calls"},
        "immutability": "STEP5C primary predictions/metrics/freezes/features and STEP5D "
                        "diagnostic outputs are read-only; STEP5E only copies/references",
        "package_versions": versions,
    }
    return pre


def freeze_splits_and_sequences():
    man = C.load_fold_manifest()
    split_rows, seq_rows = [], []
    for e in C.HELDOUT_ELEMENTS:
        comps = sorted(man[(man.fold == f"FOLD_{e}")
                           & (man.loeo_role == "HELDOUT_ELEMENT_TEST")]["composition_id"])
        assert len(comps) == 5250, (e, len(comps))
        cal, ev = C.frozen_recovery_split(e, comps)
        assert len(cal) == 1575 and len(ev) == 3675, (e, len(cal), len(ev))
        assert not (set(cal) & set(ev)) and len(set(cal) | set(ev)) == 5250
        for c in cal:
            split_rows.append({"heldout_element": e, "composition_id": c,
                               "recovery_role": "CALIBRATION_POOL"})
        for c in ev:
            split_rows.append({"heldout_element": e, "composition_id": c,
                               "recovery_role": "RETROSPECTIVE_RECOVERY_EVAL"})
        # targeted maximin sequence: uses ONLY input geometry of fold-train + pool
        df, feats = C.load_fold_features_and_targets(
            f"FOLD_{e}", [ln.strip() for ln in open(C.C_FEATLIST(e), encoding="utf-8")])
        train = C.fold_rows(df, "FOLD_TRAIN")
        pool = df[df["composition_id"].isin(cal)].drop_duplicates("composition_id")
        # composition-level space: drop Temperature_C and columns constant within the fold
        comp_cols = [c for c in feats if c != "Temperature_C"]
        const_cols = [c for c in comp_cols
                      if train[c].nunique() <= 1 and pool[c].nunique() <= 1]
        use = [c for c in comp_cols if c not in const_cols]
        seq = C.maximin_sequence(e, pool["composition_id"].tolist(),
                                 train.drop_duplicates("composition_id")["composition_id"].tolist(),
                                 pool[use].to_numpy(float),
                                 train.drop_duplicates("composition_id")[use].to_numpy(float))
        assert len(seq) == 32 and len(set(seq)) == 32
        assert set(seq) <= set(cal)
        for rank, cid in enumerate(seq, start=1):
            seq_rows.append({"heldout_element": e, "strategy": "targeted_maximin",
                             "replicate": 0, "rank": rank, "composition_id": cid})
        for r in range(1, C.N_RANDOM_REPLICATES + 1):
            rseq = C.random_sequence(r, cal)[:32]   # PATCH_02: nested 32-prefix (spec 50)
            for rank, cid in enumerate(rseq, start=1):
                seq_rows.append({"heldout_element": e, "strategy": "random",
                                 "replicate": r, "rank": rank, "composition_id": cid})
        print(f"  {e}: cal {len(cal)} / eval {len(ev)}; targeted + "
              f"{C.N_RANDOM_REPLICATES} random sequences frozen", flush=True)
    sp = pd.DataFrame(split_rows)
    sq = pd.DataFrame(seq_rows)
    sp.to_parquet(os.path.join(C.D_FROZEN, "FROZEN_RECOVERY_SPLITS.parquet"), index=False)
    sq.to_parquet(os.path.join(C.D_FROZEN, "FROZEN_FEW_SHOT_SEQUENCES.parquet"), index=False)
    return sp, sq


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    pre = build_prereg()
    ppath = os.path.join(C.D_PREREG, "STEP5E_REMEDY_PREREGISTRATION.json")
    C.write_json(ppath, pre)
    sp, sq = freeze_splits_and_sequences()
    split_sha = C.sha256_file(os.path.join(C.D_FROZEN, "FROZEN_RECOVERY_SPLITS.parquet"))
    seq_sha = C.sha256_file(os.path.join(C.D_FROZEN, "FROZEN_FEW_SHOT_SEQUENCES.parquet"))
    C.write_json(os.path.join(C.D_PREREG, "STEP5E_PREREGISTRATION_HASH.json"), {
        "preregistration_sha256": C.sha256_file(ppath),
        "frozen_recovery_splits_sha256": split_sha,
        "frozen_few_shot_sequences_sha256": seq_sha,
        "generated_utc": C.utcnow(),
        "note": "splits and sequences frozen before ANY model of this stage is fit"})
    print("preregistration + frozen splits/sequences written")


if __name__ == "__main__":
    main()
