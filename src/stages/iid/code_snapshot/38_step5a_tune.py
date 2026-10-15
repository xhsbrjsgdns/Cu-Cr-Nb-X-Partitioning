# -*- coding: utf-8 -*-
"""
38_step5a_tune.py
STEP5A Stage L: top-model tuning (sec.71).

For each task group, tune ONLY the top-2 model families selected in Stage K,
using Optuna TPE (seed 20260904), ~30 trials each, evaluated on
DEV_VALIDATION. Fitting during tuning uses the 20% composition subset of
DEV_TRAIN to keep compute bounded; the FINAL reported model is refit on the
FULL DEV_TRAIN in 39_step5a_development_freeze.py.

IID_TEST, Domain B and ANCHOR are never touched here.

Outputs:
  04_models/TUNING_RESULTS.json
  04_models/BEST_PARAMS.json
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5a_common as C
from importlib import import_module

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = C.MASTER_SEED
N_TRIALS = 30


# --------- per-family builders that take a params dict and return a predict fn ---------

def make_predictor(model_name, task, params):
    """Return fn(Xtr, Ytr, Xva) -> predictions, honoring task type."""
    if model_name == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor
        base = dict(random_state=SEED, n_jobs=-1, tree_method="hist", device="cuda")
        base.update(params)
        if task == "presence":
            n_est = base.pop("n_estimators", 400)

            def fn(Xtr, Ytr, Xva):
                P = np.zeros((len(Xva), Ytr.shape[1]))
                for i in range(Ytr.shape[1]):
                    col = Ytr[:, i]
                    if len(np.unique(col)) < 2:
                        P[:, i] = col.mean(); continue
                    m = XGBClassifier(n_estimators=n_est, **base)
                    m.fit(Xtr, col); P[:, i] = m.predict_proba(Xva)[:, 1]
                return P
            return fn
        else:
            n_est = base.pop("n_estimators", 400)

            def fn(Xtr, Ytr, Xva):
                P = np.zeros((len(Xva), Ytr.shape[1]))
                for i in range(Ytr.shape[1]):
                    m = XGBRegressor(n_estimators=n_est, **base)
                    m.fit(Xtr, Ytr[:, i]); P[:, i] = m.predict(Xva)
                return P
            return fn

    if model_name == "CatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        base = dict(random_seed=SEED, verbose=0, thread_count=-1)
        base.update(params)
        if task == "presence":
            def fn(Xtr, Ytr, Xva):
                P = np.zeros((len(Xva), Ytr.shape[1]))
                for i in range(Ytr.shape[1]):
                    col = Ytr[:, i]
                    if len(np.unique(col)) < 2:
                        P[:, i] = col.mean(); continue
                    w = C.class_weights_from_train(Ytr[:, i:i + 1])[0]
                    m = CatBoostClassifier(loss_function="Logloss", scale_pos_weight=w, **base)
                    m.fit(Xtr, col); P[:, i] = m.predict_proba(Xva)[:, 1]
                return P
            return fn
        else:
            def fn(Xtr, Ytr, Xva):
                P = np.zeros((len(Xva), Ytr.shape[1]))
                for i in range(Ytr.shape[1]):
                    m = CatBoostRegressor(loss_function="RMSE", **base)
                    m.fit(Xtr, Ytr[:, i]); P[:, i] = m.predict(Xva)
                return P
            return fn

    if model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        base = dict(random_state=SEED, n_jobs=-1)
        base.update(params)
        if task == "presence":
            def fn(Xtr, Ytr, Xva):
                P = np.zeros((len(Xva), Ytr.shape[1]))
                for i in range(Ytr.shape[1]):
                    col = Ytr[:, i]
                    if len(np.unique(col)) < 2:
                        P[:, i] = col.mean(); continue
                    m = RandomForestClassifier(class_weight="balanced_subsample", **base)
                    m.fit(Xtr, col); P[:, i] = m.predict_proba(Xva)[:, 1]
                return P
            return fn
        else:
            def fn(Xtr, Ytr, Xva):
                P = np.zeros((len(Xva), Ytr.shape[1]))
                for i in range(Ytr.shape[1]):
                    m = RandomForestRegressor(**base)
                    m.fit(Xtr, Ytr[:, i]); P[:, i] = m.predict(Xva)
                return P
            return fn

    if model_name == "MLP":
        # import the screen module by path (filename starts with a digit)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "screen_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "37_step5a_model_screen.py"))
        screen_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(screen_mod)

        def fn(Xtr, Ytr, Xva):
            return screen_mod.screen_mlp(task, Xtr, Ytr, Xva,
                                         epochs=params.get("epochs", 80),
                                         n_models=params.get("n_models", 1),
                                         hidden=tuple(params.get("hidden", (256, 128))),
                                         dropout=params.get("dropout", 0.1),
                                         lr=params.get("lr", 1e-3),
                                         batch_size=params.get("batch_size", 4096))
        return fn

    raise ValueError(model_name)


def suggest(trial, model_name, task):
    if model_name == "XGBoost":
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    elif model_name == "CatBoost":
        p = {
            "iterations": trial.suggest_int("iterations", 200, 900, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        }
    elif model_name == "RandomForest":
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [8, 16, 24, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8, 1.0]),
        }
    elif model_name == "MLP":
        h1 = trial.suggest_categorical("h1", [128, 256, 512])
        h2 = trial.suggest_categorical("h2", [64, 128, 256])
        p = {
            "hidden": [h1, h2],
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "epochs": trial.suggest_categorical("epochs", [60, 100, 150]),
            "batch_size": trial.suggest_categorical("batch_size", [2048, 4096]),
            "n_models": 1,
        }
    else:
        raise ValueError(model_name)
    return p


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    feats = C.primary_features()
    sel = json.load(open(os.path.join(C.MODELS, "SCREEN_SELECTION.json"), encoding="utf-8"))["selection"]

    df, _ = C.load_development(feature_cols=feats)
    train_all = df[df["split_role"] == "DEV_TRAIN"]
    val_all = df[df["split_role"] == "DEV_VALIDATION"]
    train_sub = C.subset_compositions(train_all)

    # precompute task arrays
    task_data = {}
    for task in C.ALL_TASKS:
        tr = C.task_rows(train_sub, task)
        va = C.task_rows(val_all, task)
        tr_full = C.task_rows(train_all, task)
        task_data[task] = {
            "Xtr": tr[feats].to_numpy(np.float64), "Xva": va[feats].to_numpy(np.float64),
            "Ytr": tr[C.TARGETS[task]].to_numpy(np.float64),
            "Yva": va[C.TARGETS[task]].to_numpy(np.float64),
            "fb": (lambda a: a / a.sum())(C.trivial_regression_pred(
                tr_full[C.TARGETS[task]].to_numpy(np.float64))) if task != "presence" else None,
        }

    best = {}
    tuning_log = {}
    for task in C.ALL_TASKS:
        d = task_data[task]
        best[task] = []
        tuning_log[task] = {}
        for model_name in sel[task]["top2"]:
            def objective(trial):
                params = suggest(trial, model_name, task)
                fn = make_predictor(model_name, task, params)
                t0 = time.time()
                pred = fn(d["Xtr"], d["Ytr"], d["Xva"])
                if task == "presence":
                    ths = C.fit_thresholds_f1(d["Yva"].astype(int), pred)
                    score = C.classification_metrics(d["Yva"].astype(int), pred, ths)["macro_f1"]
                    trial.set_user_attr("val_macro_f1", score)
                    trial.set_user_attr("val_thresholds", ths)
                    return score  # maximize
                else:
                    proj, _ = C.project_simplex(pred, d["fb"])
                    score = C.regression_metrics(d["Yva"], pred, proj)["mean_tvd_proj"]
                    trial.set_user_attr("val_mean_tvd", score)
                    return score  # minimize
            direction = "maximize" if task == "presence" else "minimize"
            sampler = optuna.samplers.TPESampler(seed=SEED)
            study = optuna.create_study(direction=direction, sampler=sampler)
            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            bp = study.best_trial.params
            if model_name == "MLP":
                bp = {"hidden": [bp["h1"], bp["h2"]], "dropout": bp["dropout"], "lr": bp["lr"],
                      "epochs": bp["epochs"], "batch_size": bp["batch_size"], "n_models": 1}
            ba = study.best_trial.user_attrs
            tuning_log[task][model_name] = {"best_value": study.best_value,
                                            "best_params_raw": study.best_trial.params,
                                            "best_params": bp if isinstance(bp, dict) else {},
                                            "user_attrs": {k: v for k, v in ba.items()},
                                            "n_trials": N_TRIALS}
            best[task].append({"model": model_name, "value": study.best_value,
                               "params": bp, "user_attrs": ba})
            print(f"  {task} / {model_name}: best={study.best_value:.5f} "
                  f"({time.strftime('%H:%M:%S')})", flush=True)

    with open(os.path.join(C.MODELS, "TUNING_RESULTS.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_utc": ts, "seed": SEED, "n_trials": N_TRIALS,
                   "tuning_log": tuning_log}, f, indent=2, default=str)

    # choose winner per task by validation metric
    winners = {}
    for task in C.ALL_TASKS:
        cands = best[task]
        if task == "presence":
            w = max(cands, key=lambda c: c["value"])
        else:
            w = min(cands, key=lambda c: c["value"])
        winners[task] = w
    with open(os.path.join(C.MODELS, "BEST_PARAMS.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_utc": ts, "winners": winners}, f, indent=2, default=str)
    print(json.dumps({t: {"model": winners[t]["model"], "value": winners[t]["value"]}
                      for t in C.ALL_TASKS}, indent=2))
    print("STAGE L: done")


if __name__ == "__main__":
    main()
