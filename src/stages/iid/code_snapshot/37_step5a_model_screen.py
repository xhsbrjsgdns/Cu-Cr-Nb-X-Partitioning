# -*- coding: utf-8 -*-
"""
37_step5a_model_screen.py
STEP5A Stage K: baseline model screening (sec.66-70).

Models: Random Forest, XGBoost, CatBoost, MLP (shared encoder + 5-output head)
+ trivial baselines (prevalence/majority; DEV_TRAIN mean vector).

Screening runs on the fixed family-stratified 20%-composition subset of
DEV_TRAIN and scores on full DEV_VALIDATION, with the PRIMARY feature list.
IID_TEST untouched. Domain B / ANCHOR untouched.

Outputs:
  05_results/BASELINE_MODEL_SCREEN.csv
  04_models/SCREEN_SELECTION.json   (top-2 model families per task group)
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5a_common as C

SEED = C.MASTER_SEED


# ---------------- model wrappers ----------------

def screen_xgb(task, Xtr, Ytr, Xva):
    from xgboost import XGBClassifier, XGBRegressor
    params = dict(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9,
                  colsample_bytree=0.9, random_state=SEED, n_jobs=-1,
                  tree_method="hist", device="cuda")
    P = np.zeros((len(Xva), Ytr.shape[1]))
    if task == "presence":
        for i in range(Ytr.shape[1]):
            col = Ytr[:, i]
            if len(np.unique(col)) < 2:
                P[:, i] = col.mean()
                continue
            m = XGBClassifier(**params)
            m.fit(Xtr, col)
            P[:, i] = m.predict_proba(Xva)[:, 1]
    else:
        for i in range(Ytr.shape[1]):
            m = XGBRegressor(**params)
            m.fit(Xtr, Ytr[:, i])
            P[:, i] = m.predict(Xva)
    return P


def screen_rf(task, Xtr, Ytr, Xva):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    P = np.zeros((len(Xva), Ytr.shape[1]))
    if task == "presence":
        for i in range(Ytr.shape[1]):
            col = Ytr[:, i]
            if len(np.unique(col)) < 2:
                P[:, i] = col.mean()
                continue
            m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                       class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1)
            m.fit(Xtr, col)
            P[:, i] = m.predict_proba(Xva)[:, 1]
    else:
        for i in range(Ytr.shape[1]):
            m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                      random_state=SEED, n_jobs=-1)
            m.fit(Xtr, Ytr[:, i])
            P[:, i] = m.predict(Xva)
    return P


def screen_cat(task, Xtr, Ytr, Xva):
    from catboost import CatBoostClassifier, CatBoostRegressor
    P = np.zeros((len(Xva), Ytr.shape[1]))
    common = dict(iterations=300, depth=8, learning_rate=0.1, random_seed=SEED,
                  verbose=0, thread_count=-1)
    if task == "presence":
        for i in range(Ytr.shape[1]):
            col = Ytr[:, i]
            if len(np.unique(col)) < 2:
                P[:, i] = col.mean()
                continue
            w = C.class_weights_from_train(Ytr[:, i:i + 1])[0]
            m = CatBoostClassifier(**common, loss_function="Logloss",
                                   scale_pos_weight=w)
            m.fit(Xtr, col)
            P[:, i] = m.predict_proba(Xva)[:, 1]
    else:
        for i in range(Ytr.shape[1]):
            m = CatBoostRegressor(**common, loss_function="RMSE")
            m.fit(Xtr, Ytr[:, i])
            P[:, i] = m.predict(Xva)
    return P


def screen_mlp(task, Xtr, Ytr, Xva, epochs=60, n_models=1, hidden=(256, 128), dropout=0.1,
               lr=1e-3, batch_size=4096, scaler=None, return_scaler=False):
    """Shared-encoder MLP with one 5-output head (sec.68). n_models>1 = different seeds, averaged."""
    import torch
    from sklearn.preprocessing import StandardScaler
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if scaler is None:
        scaler = StandardScaler().fit(Xtr)  # fit on TRAIN rows only
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)

    k = Ytr.shape[1]
    preds = []
    for mi in range(n_models):
        torch.manual_seed(SEED + mi)
        layers = []
        d_in = Xtr_s.shape[1]
        for h in hidden:
            layers += [torch.nn.Linear(d_in, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            d_in = h
        layers += [torch.nn.Linear(d_in, k)]
        model = torch.nn.Sequential(*layers).to(dev)
        xt = torch.from_numpy(Xtr_s).to(dev)
        xv = torch.from_numpy(Xva_s).to(dev)

        if task == "presence":
            yt = torch.from_numpy(Ytr.astype(np.float32)).to(dev)
            pos_w = torch.tensor(C.class_weights_from_train(Ytr), dtype=torch.float32, device=dev)
            lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
            out_act = torch.sigmoid
        else:
            yt = torch.from_numpy(Ytr.astype(np.float32)).to(dev)
            lossf = torch.nn.MSELoss()
            out_act = lambda z: z

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n = len(xt)
        g = torch.Generator(device="cpu"); g.manual_seed(SEED + mi)
        model.train()
        for ep in range(epochs):
            perm = torch.randperm(n, generator=g).to(dev)
            for j in range(0, n, batch_size):
                idx = perm[j:j + batch_size]
                opt.zero_grad()
                loss = lossf(model(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            preds.append(out_act(model(xv)).cpu().numpy())
    P = np.mean(preds, axis=0)
    if task == "presence":
        # constant labels: force prevalence (cannot be learned from BCE)
        for i in range(k):
            if len(np.unique(Ytr[:, i])) < 2:
                P[:, i] = Ytr[:, i].mean()
    if return_scaler:
        return P, scaler
    return P


SCREENERS = {"RandomForest": screen_rf, "XGBoost": screen_xgb, "CatBoost": screen_cat, "MLP": screen_mlp}


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    feats = C.primary_features()
    df, _ = C.load_development(feature_cols=feats)
    train_all = df[df["split_role"] == "DEV_TRAIN"]
    val_all = df[df["split_role"] == "DEV_VALIDATION"]
    train_sub = C.subset_compositions(train_all)
    print(f"screen train: {len(train_sub)} rows / {train_sub['composition_id'].nunique()} comps; "
          f"val: {len(val_all)} rows")

    rows = []
    # trivial baselines first (from FULL DEV_TRAIN, per sec.66)
    for task in C.ALL_TASKS:
        tr_full = C.task_rows(train_all, task)
        va = C.task_rows(val_all, task)
        Ytr = tr_full[C.TARGETS[task]].to_numpy(np.float64)
        Yva = va[C.TARGETS[task]].to_numpy(np.float64)
        if task == "presence":
            Ytr_i = Ytr.astype(int)
            tri = C.trivial_classification_probs(Ytr_i)
            prob = np.tile(tri, (len(Yva), 1))
            met = C.classification_metrics(Yva.astype(int), prob, [0.5] * 5)
            rows.append({"model": "TrivialPrevalence", "task": task, "stage": "screen",
                         "macro_f1": met["macro_f1"], "exact_match": met["exact_match"],
                         "micro_f1": met["micro_f1"], "seconds": 0.0})
            for fam, d in met["per_label"].items():
                rows.append({"model": "TrivialPrevalence", "task": f"{task}:{fam}", "stage": "screen",
                             "f1": d["f1"], "precision": d["precision"], "recall": d["recall"],
                             "support": d["support"], "pr_auc": d["pr_auc"]})
        else:
            fb = C.trivial_regression_pred(Ytr)
            fb = fb / fb.sum()
            pred = np.tile(fb, (len(Yva), 1))
            met = C.regression_metrics(Yva, pred, pred.copy())
            rows.append({"model": "TrivialMeanVector", "task": task, "stage": "screen",
                         "mean_tvd": met["mean_tvd_proj"], "macro_mae": met["macro_mae_proj"],
                         "seconds": 0.0})
            for fam, d in met["per_family"].items():
                rows.append({"model": "TrivialMeanVector", "task": f"{task}:{fam}", "stage": "screen",
                             "mae": d["mae_proj"], "rmse": d["rmse_proj"], "r2": d["r2_proj"],
                             "true_mean": d["true_mean"]})

    for model_name, fn in SCREENERS.items():
        for task in C.ALL_TASKS:
            tr = C.task_rows(train_sub, task)
            va = C.task_rows(val_all, task)
            Xtr = tr[feats].to_numpy(np.float64)
            Xva = va[feats].to_numpy(np.float64)
            t0 = time.time()
            if task == "presence":
                Ytr = tr[C.TARGETS[task]].to_numpy(int)
                Yva = va[C.TARGETS[task]].to_numpy(int)
                prob = fn(task, Xtr, Ytr.astype(np.float64), Xva)
                met = C.classification_metrics(Yva, prob, [0.5] * 5)
                rows.append({"model": model_name, "task": task, "stage": "screen",
                             "macro_f1": met["macro_f1"], "exact_match": met["exact_match"],
                             "micro_f1": met["micro_f1"], "hamming_loss": met["hamming_loss"],
                             "seconds": round(time.time() - t0, 1)})
                for fam, d in met["per_label"].items():
                    rows.append({"model": model_name, "task": f"{task}:{fam}", "stage": "screen",
                                 "f1": d["f1"], "precision": d["precision"], "recall": d["recall"],
                                 "support": d["support"], "pr_auc": d["pr_auc"], "roc_auc": d["roc_auc"]})
            else:
                Ytr = tr[C.TARGETS[task]].to_numpy(np.float64)
                Yva = va[C.TARGETS[task]].to_numpy(np.float64)
                raw = fn(task, Xtr, Ytr, Xva)
                fb = C.trivial_regression_pred(C.task_rows(train_all, task)[C.TARGETS[task]].to_numpy(np.float64))
                fb = fb / fb.sum()
                proj, n_fb = C.project_simplex(raw, fb)
                met = C.regression_metrics(Yva, raw, proj)
                rows.append({"model": model_name, "task": task, "stage": "screen",
                             "mean_tvd": met["mean_tvd_proj"], "macro_mae": met["macro_mae_proj"],
                             "raw_sum_violation_mean": met["raw_sum_violation_mean"],
                             "simplex_fallbacks": n_fb,
                             "seconds": round(time.time() - t0, 1)})
                for fam, d in met["per_family"].items():
                    rows.append({"model": model_name, "task": f"{task}:{fam}", "stage": "screen",
                                 "mae": d["mae_proj"], "rmse": d["rmse_proj"], "r2": d["r2_proj"],
                                 "true_mean": d["true_mean"]})
            print(f"  {model_name} x {task}: {rows[-1].get('seconds')} s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.RESULTS, "BASELINE_MODEL_SCREEN.csv"), index=False)

    # top-2 model families per task group (sec.71)
    metric_col = {"presence": "macro_f1"}
    selection = {}
    for task in C.ALL_TASKS:
        col = "macro_f1" if task == "presence" else "mean_tvd"
        sub = out[(out["task"] == task) & (out["stage"] == "screen")
                  & (~out["model"].str.startswith("Trivial"))]
        sub = sub.dropna(subset=[col])
        ranked = sub.sort_values(col, ascending=(task != "presence"))["model"].tolist()
        selection[task] = {"metric": col, "ranking": ranked, "top2": ranked[:2]}
    with open(os.path.join(C.MODELS, "SCREEN_SELECTION.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_utc": ts, "selection": selection,
                   "screen_train_rows": len(train_sub),
                   "screen_train_compositions": int(train_sub["composition_id"].nunique()),
                   "val_rows": len(val_all), "features_n": len(feats), "seed": SEED},
                  f, indent=2)
    print(json.dumps(selection, indent=2))
    print("STAGE K: done")


if __name__ == "__main__":
    main()
