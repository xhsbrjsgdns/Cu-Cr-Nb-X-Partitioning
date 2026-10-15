# -*- coding: utf-8 -*-
"""
53_step5c_train.py
STEP5C Stage G + H: tune, select, refit and FREEZE all six folds.

NOTHING in this module reads a HELDOUT_ELEMENT_TEST row. Held-out targets are excluded
from tuning, model selection, thresholds, scalers, class weights, the trivial baseline
and the simplex fallback. The held-out element's EXTERNAL DESCRIPTORS are of course part
of the frozen feature schema - that is the generalization input, not leakage (spec 17).

Per fold x task:
  1. Optuna TPE, 15 trials per model family, fitted on a 20% family-stratified
     COMPOSITION-level subsample of FOLD_TRAIN, scored on FOLD_VALIDATION
  2. winner chosen on FOLD_VALIDATION with the preregistered tie rule
  3. refit on the FULL FOLD_TRAIN
  4. presence thresholds fitted on FOLD_VALIDATION
  5. 11_manifest/FOLD_<X>_FREEZE.json written

Run with --fold ZR to do one fold at a time; default does all six in the frozen order.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5c_common as C

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------- builders -- #
def make_predictor(model_name, task, params, seed):
    """fn(Xtr, Ytr, Xte) -> (n, 5) predictions. One model per family column, as STEP5A."""
    if model_name == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor
        base = dict(random_state=seed, n_jobs=-1, tree_method="hist")
        if os.environ.get("STEP5C_XGB_DEVICE", "cuda") != "cpu":
            base["device"] = os.environ.get("STEP5C_XGB_DEVICE", "cuda")
        base.update(params)
        n_est = base.pop("n_estimators", 400)

        def fn(Xtr, Ytr, Xte, return_models=False):
            P = np.zeros((len(Xte), Ytr.shape[1]))
            models = []
            for i in range(Ytr.shape[1]):
                col = Ytr[:, i]
                if task == "presence":
                    if len(np.unique(col)) < 2:
                        P[:, i] = col.mean(); models.append(("constant", float(col.mean())))
                        continue
                    m = XGBClassifier(n_estimators=n_est,
                                      scale_pos_weight=C.class_weight_positive(col), **base)
                    m.fit(Xtr, col); P[:, i] = m.predict_proba(Xte)[:, 1]
                else:
                    # PATCH_01: constant-target guard. In an LOEO fold-train the LIQUID
                    # component targets (LIQUID_volfrac, *_P_LIQUID) can be identically 0
                    # (LIQUID exists only in ZR-family states), which makes CatBoostRegressor
                    # raise "All train targets are equal". XGBoost trains such a column to a
                    # numerically identical constant prediction; the explicit guard makes the
                    # two families behave identically and is applied uniformly to all folds.
                    if len(np.unique(col)) < 2:
                        P[:, i] = float(col.mean()); models.append(("constant", float(col.mean())))
                        continue
                    m = XGBRegressor(n_estimators=n_est, **base)
                    m.fit(Xtr, col); P[:, i] = m.predict(Xte)
                models.append(m)
            return (P, models) if return_models else P
        return fn

    if model_name == "CatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        base = dict(random_seed=seed, verbose=0, thread_count=-1, allow_writing_files=False)
        base.update(params)

        def fn(Xtr, Ytr, Xte, return_models=False):
            P = np.zeros((len(Xte), Ytr.shape[1]))
            models = []
            for i in range(Ytr.shape[1]):
                col = Ytr[:, i]
                if task == "presence":
                    if len(np.unique(col)) < 2:
                        P[:, i] = col.mean(); models.append(("constant", float(col.mean())))
                        continue
                    m = CatBoostClassifier(loss_function="Logloss",
                                           scale_pos_weight=C.class_weight_positive(col),
                                           **base)
                    m.fit(Xtr, col); P[:, i] = m.predict_proba(Xte)[:, 1]
                else:
                    # PATCH_01: same constant-target guard as the XGBoost branch (see above).
                    if len(np.unique(col)) < 2:
                        P[:, i] = float(col.mean()); models.append(("constant", float(col.mean())))
                        continue
                    m = CatBoostRegressor(loss_function="RMSE", **base)
                    m.fit(Xtr, col); P[:, i] = m.predict(Xte)
                models.append(m)
            return (P, models) if return_models else P
        return fn

    raise ValueError("model family %r is not in the preregistered search space %s"
                     % (model_name, C.MODEL_FAMILIES))


def suggest(trial, model_name, space):
    p = {}
    for name, spec in space[model_name].items():
        if spec["type"] == "int":
            p[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                        step=spec.get("step", 1))
        else:
            p[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                          log=spec.get("log", False))
    return p


def fit_cost(model_name, params):
    """Deterministic complexity proxy used only by the preregistered tie rule (spec 32)."""
    if model_name == "XGBoost":
        return int(params.get("n_estimators", 0)) * int(params.get("max_depth", 0))
    return int(params.get("iterations", 0)) * int(params.get("depth", 0))


def pick_winner(task, cands):
    """Preregistered selection + tie rule. cands: [{model, value, params, ...}]."""
    band = C.TIE_BAND["presence" if task == "presence" else "regression"]
    if task == "presence":
        best = max(c["value"] for c in cands)
        pool = [c for c in cands if best - c["value"] <= band]
    else:
        best = min(c["value"] for c in cands)
        pool = [c for c in cands if c["value"] - best <= band]
    if len(pool) == 1:
        return pool[0], False
    pool = sorted(pool, key=lambda c: (fit_cost(c["model"], c["params"]),
                                       C.MODEL_FAMILIES.index(c["model"])))
    return pool[0], True


# ------------------------------------------------------------------ fold -- #
def run_fold(heldout, df, man, pre, force=False):
    fold = f"FOLD_{heldout}"
    freeze_path = os.path.join(C.D_MANIFEST, f"{fold}_FREEZE.json")
    if os.path.exists(freeze_path) and not force:
        print("  %s already frozen - skipping (use --force to redo)" % fold)
        return json.load(open(freeze_path, encoding="utf-8"))

    t0 = time.time()
    seed = C.fold_seed(heldout)
    feats = [ln.strip() for ln in open(
        os.path.join(C.D_FEAT, f"{fold}_SELECTED_FEATURES.txt"), encoding="utf-8")
        if ln.strip()]
    roles = man[man.fold == fold].set_index("composition_id")["loeo_role"]
    d = df.copy()
    d["loeo_role"] = d["composition_id"].map(roles)
    train_all = d[d.loeo_role == "FOLD_TRAIN"]
    val_all = d[d.loeo_role == "FOLD_VALIDATION"]

    # hard guards: the held-out element must be absent from BOTH development sets
    assert (train_all["family"] != heldout).all(), "held-out family in FOLD_TRAIN"
    assert (val_all["family"] != heldout).all(), "held-out family in FOLD_VALIDATION"
    assert set(train_all["sampling_domain"]) == {"A"} and set(val_all["sampling_domain"]) == {"A"}

    train_sub = C.subset_compositions(train_all, C.SUBSET_FRACTION, seed)
    print("  %s  seed %d | features %d | train %d rows / %d comps | sub %d rows | val %d rows"
          % (fold, seed, len(feats), len(train_all),
             train_all["composition_id"].nunique(), len(train_sub), len(val_all)))

    space = pre["hyperparameter_search_space"]
    fold_record = {"fold": fold, "heldout_element": heldout, "seed": seed,
                   "tasks": {}, "tuning_log": {}}

    for task in C.ALL_TASKS:
        tr_sub = C.task_rows(train_sub, task)
        tr_full = C.task_rows(train_all, task)
        va = C.task_rows(val_all, task)
        Xtr = tr_sub[feats].to_numpy(np.float64)
        Ytr = tr_sub[C.TARGETS[task]].to_numpy(np.float64)
        Xva = va[feats].to_numpy(np.float64)
        Yva = va[C.TARGETS[task]].to_numpy(np.float64)
        Yfull = tr_full[C.TARGETS[task]].to_numpy(np.float64)
        fb = None if task == "presence" else C.trivial_regression_simplex(Yfull)

        cands, log = [], {}
        for model_name in C.MODEL_FAMILIES:
            def objective(trial, model_name=model_name):
                params = suggest(trial, model_name, space)
                fn = make_predictor(model_name, task, params, seed)
                pred = fn(Xtr, Ytr, Xva)
                if task == "presence":
                    ths = C.fit_thresholds_f1(Yva.astype(int), pred)
                    score = C.classification_metrics(Yva.astype(int), pred, ths)["macro_f1"]
                    trial.set_user_attr("val_macro_f1", score)
                    trial.set_user_attr("val_thresholds", ths)
                    return score
                proj, _ = C.project_simplex(pred, fb)
                score = C.regression_metrics(Yva, pred, proj)["mean_tvd"]
                trial.set_user_attr("val_mean_tvd", score)
                return score

            study = optuna.create_study(
                direction="maximize" if task == "presence" else "minimize",
                sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=C.N_TRIALS, show_progress_bar=False)
            cands.append({"model": model_name, "value": float(study.best_value),
                          "params": dict(study.best_trial.params),
                          "user_attrs": dict(study.best_trial.user_attrs)})
            log[model_name] = {"best_value": float(study.best_value),
                               "best_params": dict(study.best_trial.params),
                               "n_trials": C.N_TRIALS}
            print("    %-13s %-12s best=%.6f" % (task, model_name, study.best_value),
                  flush=True)

        winner, tie = pick_winner(task, cands)
        fold_record["tuning_log"][task] = log

        # ---- refit on the FULL FOLD_TRAIN ----
        Xfull = tr_full[feats].to_numpy(np.float64)
        fn = make_predictor(winner["model"], task, winner["params"], seed)
        pred_va, models = fn(Xfull, Yfull, Xva, return_models=True)
        if task == "presence":
            ths = C.fit_thresholds_f1(Yva.astype(int), pred_va)
            val_metrics = C.classification_metrics(Yva.astype(int), pred_va, ths)
            val_headline = val_metrics["macro_f1"]
            trivial = C.trivial_classification_probs(Yfull).tolist()
        else:
            ths = None
            proj, nfb = C.project_simplex(pred_va, fb)
            val_metrics = C.regression_metrics(Yva, pred_va, proj, nfb)
            val_headline = val_metrics["mean_tvd"]
            trivial = fb.tolist()

        mdir = os.path.join(C.D_MODELS, fold, task)
        os.makedirs(mdir, exist_ok=True)
        mpath = os.path.join(mdir, "model_state.pkl")
        with open(mpath, "wb") as f:
            pickle.dump({"fold": fold, "heldout_element": heldout, "task": task,
                         "model_family": winner["model"], "params": winner["params"],
                         "features": feats, "thresholds": ths,
                         "trivial": trivial, "seed": seed, "models": models}, f)
        fold_record["tasks"][task] = {
            "model_family": winner["model"], "hyperparameters": winner["params"],
            "tie_applied": bool(tie),
            "validation_selection_metric": ("macro_f1" if task == "presence" else "mean_tvd"),
            "validation_value_on_subset_tuning": winner["value"],
            "validation_value_after_full_refit": float(val_headline),
            "thresholds": ths,
            "trivial_baseline": trivial,
            "simplex_fallback_row": (None if task == "presence" else fb.tolist()),
            "n_train_rows": int(len(tr_full)), "n_train_compositions":
                int(tr_full["composition_id"].nunique()),
            "n_validation_rows": int(len(va)), "n_validation_compositions":
                int(va["composition_id"].nunique()),
            "model_artifact": os.path.relpath(mpath, C.STEP5C).replace("\\", "/"),
            "model_artifact_sha256": C.sha256_file(mpath),
            "validation_metrics": val_metrics,
        }
        print("    -> %-13s winner %-9s val %s=%.6f%s"
              % (task, winner["model"],
                 "macroF1" if task == "presence" else "meanTVD", val_headline,
                 "  (tie rule applied)" if tie else ""), flush=True)

    feat_file = os.path.join(C.D_FEAT, f"{fold}_SELECTED_FEATURES.txt")
    freeze = {
        "fold": fold,
        "heldout_element": heldout,
        "frozen_utc": C.utcnow(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "terminology": ("retrospective leave-one-element-out; the held-out element's "
                        "targets are excluded from THIS fold's development"),
        "preregistration_sha256": C.sha256_file(
            os.path.join(C.D_PREREG, "STEP5C_LOEO_PREREGISTRATION.json")),
        "candidate_schema_hash": pre["candidate_feature_schema"]["schema_sha256"],
        "selected_features": feats,
        "n_selected_features": len(feats),
        "selected_features_sha256": C.sha256_obj(feats),
        "selected_features_file_sha256": C.sha256_file(feat_file),
        "pcc_log_sha256": C.sha256_file(
            os.path.join(C.D_FEAT, f"{fold}_PCC_REMOVAL.csv")),
        "nzv_log_sha256": C.sha256_file(
            os.path.join(C.D_FEAT, f"{fold}_NZV_REMOVAL.csv")),
        "random_seed": seed,
        "fold_train_compositions": int(train_all["composition_id"].nunique()),
        "fold_validation_compositions": int(val_all["composition_id"].nunique()),
        "fold_train_rows": int(len(train_all)),
        "fold_validation_rows": int(len(val_all)),
        "heldout_rows_used_in_development": 0,
        "projection_rule": pre["simplex_projection"],
        "threshold_rule": pre["presence_thresholds"]["rule"],
        "tasks": fold_record["tasks"],
        "tuning_log": fold_record["tuning_log"],
        "heldout_metrics_computed": False,
    }
    freeze["freeze_sha256"] = C.sha256_obj(
        {k: v for k, v in freeze.items() if k != "frozen_utc"})
    C.write_json(freeze_path, freeze)
    print("  %s FROZEN in %.1f s -> %s" % (fold, time.time() - t0,
                                           os.path.basename(freeze_path)))
    return freeze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=None, help="single held-out element, e.g. ZR")
    ap.add_argument("--force", action="store_true", help="re-run an already frozen fold")
    args = ap.parse_args()

    print("=" * 78)
    print("STEP 5C   Stage G + H: tune / select / refit / freeze")
    print("no HELDOUT_ELEMENT_TEST row is read anywhere in this module")
    print("=" * 78)
    prereg = os.path.join(C.D_PREREG, "STEP5C_LOEO_PREREGISTRATION.json")
    if not os.path.exists(prereg):
        raise SystemExit("FATAL: protocol not preregistered (run Stage D first).")
    pre = json.load(open(prereg, encoding="utf-8"))
    assert pre["model_families"] == C.MODEL_FAMILIES, "model families drifted from prereg"
    assert pre["tuning"]["n_trials_per_fold_task_family"] == C.N_TRIALS

    man = pd.read_csv(os.path.join(C.D_FOLDS, "LOEO_MASTER_FOLD_MANIFEST.csv"))
    all_feats = sorted({f for e in C.HELDOUT_ELEMENTS for f in
                        (ln.strip() for ln in open(
                            os.path.join(C.D_FEAT, f"FOLD_{e}_SELECTED_FEATURES.txt"),
                            encoding="utf-8")) if f})
    df, _ = C.load_domain_a(all_feats)

    todo = [args.fold.upper()] if args.fold else C.HELDOUT_ELEMENTS
    for heldout in todo:
        if heldout not in C.HELDOUT_ELEMENTS:
            raise SystemExit("unknown held-out element %r" % heldout)
        run_fold(heldout, df, man, pre, force=args.force)

    frozen = [e for e in C.HELDOUT_ELEMENTS
              if os.path.exists(os.path.join(C.D_MANIFEST, f"FOLD_{e}_FREEZE.json"))]
    print("\nfrozen folds: %d/6  %s" % (len(frozen), frozen))
    print("STAGE G/H: %s" % ("PASS" if len(frozen) == 6 else "INCOMPLETE"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
