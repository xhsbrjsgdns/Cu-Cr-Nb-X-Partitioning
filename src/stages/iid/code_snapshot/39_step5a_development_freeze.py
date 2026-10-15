# -*- coding: utf-8 -*-
"""
39_step5a_development_freeze.py
STEP5A Stage M + N: final model refit on FULL DEV_TRAIN, validation-based
threshold freezing, and DEVELOPMENT_FREEZE_V1.json (sec.75).

Per task group:
  - winner model family + tuned hyperparameters (Stage L, validation-selected)
  - refit on FULL DEV_TRAIN (sec.70)
  - MLP: StandardScaler fit on FULL DEV_TRAIN only
  - presence: per-label thresholds fit on DEV_VALIDATION (F1-max, tie -> 0.5)
  - regression: simplex fallback row = normalized DEV_TRAIN target mean
  - trivial baselines refit on FULL DEV_TRAIN
Artifacts are pickled/serialized into 04_models/final/ and hashed.

IID_TEST remains sealed: this script computes NO test metrics.

Outputs:
  04_models/final/<task>/...            (artifacts + meta)
  05_results/BASELINE_VALIDATION_METRICS.csv
  05_results/BASELINE_PER_TARGET_METRICS.csv   (stage=validation)
  05_results/LIQUID_VALIDATION_METRICS.csv
  09_manifest/DEVELOPMENT_FREEZE_V1.json
"""
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5a_common as C

SEED = C.MASTER_SEED
FINAL = os.path.join(C.MODELS, "final")
LIB_VERSIONS = None


def lib_versions():
    global LIB_VERSIONS
    if LIB_VERSIONS is None:
        import importlib.metadata as mdv
        vers = {}
        for pkg in ["python", "numpy", "pandas", "pyarrow", "scikit-learn", "xgboost",
                    "catboost", "torch", "optuna", "matminer", "pymatgen", "mendeleev",
                    "scipy", "matplotlib", "shap"]:
            try:
                vers[pkg] = sys.version.split()[0] if pkg == "python" else mdv.version(pkg)
            except Exception:
                vers[pkg] = None
        LIB_VERSIONS = vers
    return LIB_VERSIONS


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build_final_model(model_name, task, params):
    """Return (fit(Xtr,Ytr)->state, predict(state,Xva)->P, artifacts dict to pickle)."""
    if model_name == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor
        base = dict(random_state=SEED, n_jobs=-1, tree_method="hist", device="cuda")
        base.update(params)
        n_est = base.pop("n_estimators", 400)
        models = []

        def fit(Xtr, Ytr):
            models.clear()
            for i in range(Ytr.shape[1]):
                if task == "presence":
                    col = Ytr[:, i]
                    if len(np.unique(col)) < 2:
                        models.append(("const", float(col.mean())))
                        continue
                    m = XGBClassifier(n_estimators=n_est, **base)
                    m.fit(Xtr, col)
                else:
                    m = XGBRegressor(n_estimators=n_est, **base)
                    m.fit(Xtr, Ytr[:, i])
                models.append(("model", m))
            return models

        def predict(state, Xva):
            P = np.zeros((len(Xva), len(state)))
            for i, (kind, obj) in enumerate(state):
                if kind == "const":
                    P[:, i] = obj
                elif task == "presence":
                    P[:, i] = obj.predict_proba(Xva)[:, 1] if hasattr(obj, "predict_proba") and len(obj.classes_) > 1 else obj.classes_[0]
                else:
                    P[:, i] = obj.predict(Xva)
            return P
        return fit, predict

    if model_name == "CatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        base = dict(random_seed=SEED, verbose=0, thread_count=-1)
        base.update(params)
        models = []

        def fit(Xtr, Ytr):
            models.clear()
            for i in range(Ytr.shape[1]):
                if task == "presence":
                    col = Ytr[:, i]
                    if len(np.unique(col)) < 2:
                        models.append(("const", float(col.mean())))
                        continue
                    w = C.class_weights_from_train(Ytr[:, i:i + 1])[0]
                    m = CatBoostClassifier(loss_function="Logloss", scale_pos_weight=w, **base)
                    m.fit(Xtr, col)
                else:
                    m = CatBoostRegressor(loss_function="RMSE", **base)
                    m.fit(Xtr, Ytr[:, i])
                models.append(("model", m))
            return models

        def predict(state, Xva):
            P = np.zeros((len(Xva), len(state)))
            for i, (kind, obj) in enumerate(state):
                if kind == "const":
                    P[:, i] = obj
                elif task == "presence":
                    P[:, i] = obj.predict_proba(Xva)[:, 1]
                else:
                    P[:, i] = obj.predict(Xva)
            return P
        return fit, predict

    if model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        base = dict(random_state=SEED, n_jobs=-1)
        base.update(params)
        models = []

        def fit(Xtr, Ytr):
            models.clear()
            for i in range(Ytr.shape[1]):
                if task == "presence":
                    col = Ytr[:, i]
                    if len(np.unique(col)) < 2:
                        models.append(("const", float(col.mean())))
                        continue
                    m = RandomForestClassifier(class_weight="balanced_subsample", **base)
                    m.fit(Xtr, col)
                else:
                    m = RandomForestRegressor(**base)
                    m.fit(Xtr, Ytr[:, i])
                models.append(("model", m))
            return models

        def predict(state, Xva):
            P = np.zeros((len(Xva), len(state)))
            for i, (kind, obj) in enumerate(state):
                if kind == "const":
                    P[:, i] = obj
                elif task == "presence":
                    P[:, i] = obj.predict_proba(Xva)[:, 1]
                else:
                    P[:, i] = obj.predict(Xva)
            return P
        return fit, predict

    if model_name == "MLP":
        import torch
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "screen_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "37_step5a_model_screen.py"))
        screen_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(screen_mod)
        from sklearn.preprocessing import StandardScaler
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        state = {}

        def fit(Xtr, Ytr):
            scaler = StandardScaler().fit(Xtr)
            Xs = scaler.transform(Xtr).astype(np.float32)
            k = Ytr.shape[1]
            torch.manual_seed(SEED)
            layers = []
            d_in = Xs.shape[1]
            for h in params.get("hidden", [256, 128]):
                layers += [torch.nn.Linear(d_in, int(h)), torch.nn.ReLU(),
                           torch.nn.Dropout(float(params.get("dropout", 0.1)))]
                d_in = int(h)
            layers += [torch.nn.Linear(d_in, k)]
            model = torch.nn.Sequential(*layers).to(dev)
            xt = torch.from_numpy(Xs).to(dev)
            yt = torch.from_numpy(Ytr.astype(np.float32)).to(dev)
            if task == "presence":
                pos_w = torch.tensor(C.class_weights_from_train(Ytr), dtype=torch.float32, device=dev)
                lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
            else:
                lossf = torch.nn.MSELoss()
            opt = torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 1e-3)))
            n = len(xt)
            bs = int(params.get("batch_size", 4096))
            g = torch.Generator(device="cpu"); g.manual_seed(SEED)
            model.train()
            for ep in range(int(params.get("epochs", 100))):
                perm = torch.randperm(n, generator=g).to(dev)
                for j in range(0, n, bs):
                    idx = perm[j:j + bs]
                    opt.zero_grad()
                    loss = lossf(model(xt[idx]), yt[idx])
                    loss.backward()
                    opt.step()
            model.eval()
            state["scaler"] = scaler
            state["model"] = model
            state["const_cols"] = {i: float(Ytr[:, i].mean()) for i in range(k)
                                   if task == "presence" and len(np.unique(Ytr[:, i])) < 2}
            return state

        def predict(st, Xva):
            Xs = st["scaler"].transform(Xva).astype(np.float32)
            xv = torch.from_numpy(Xs).to(dev)
            with torch.no_grad():
                out = st["model"](xv)
                if task == "presence":
                    out = torch.sigmoid(out)
            P = out.cpu().numpy()
            for i, v in st.get("const_cols", {}).items():
                P[:, i] = v
            return P
        return fit, predict

    raise ValueError(model_name)


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(FINAL, exist_ok=True)
    feats = C.primary_features()
    winners = json.load(open(os.path.join(C.MODELS, "BEST_PARAMS.json"), encoding="utf-8"))["winners"]

    df, _ = C.load_development(feature_cols=feats)
    train_all = df[df["split_role"] == "DEV_TRAIN"]
    val_all = df[df["split_role"] == "DEV_VALIDATION"]
    print(f"FULL DEV_TRAIN rows: {len(train_all)}; DEV_VALIDATION rows: {len(val_all)}")

    freeze = {
        "freeze_id": "DEVELOPMENT_FREEZE_V1",
        "generated_utc": ts,
        "master_seed": SEED,
        "feature_set": "PRIMARY_FEATURE_LIST.txt",
        "n_features": len(feats),
        "features": feats,
        "feature_list_sha256": hashlib.sha256(
            open(os.path.join(C.FEAT, "PRIMARY_FEATURE_LIST.txt"), "rb").read()).hexdigest(),
        "pcc_removal_sha256": sha256_file(os.path.join(C.FEAT, "PCC_REMOVAL_LOG.csv")),
        "nzv_removal_sha256": sha256_file(os.path.join(C.FEAT, "NEAR_ZERO_VARIANCE_REMOVAL.csv")),
        "exact_redundancy_sha256": sha256_file(os.path.join(C.FEAT, "EXACT_REDUNDANCY_LOG.csv")),
        "library_versions": lib_versions(),
        "metric_definitions": {
            "presence": ["exact_match", "hamming_loss", "micro_f1", "macro_f1",
                         "per-label precision/recall/F1/ROC-AUC/PR-AUC at frozen thresholds"],
            "regression": ["per-family MAE/RMSE/R2 (raw+projected)", "macro MAE/RMSE",
                           "TVD=0.5*sum|f_pred-f_true| (mean/median/p95)", "raw/projected sum violation"],
            "threshold_rule": "per-label F1-maximizing on DEV_VALIDATION, tie -> closest to 0.5, frozen here",
            "projection_rule": "clip at 0 then renormalize to sum 1; sum==0 -> DEV_TRAIN mean simplex (counted)",
            "bootstrap": "composition-level cluster bootstrap, 1000 replicates, seed 20260904",
        },
        "tasks": {},
    }

    val_rows = []
    per_target_rows = []
    liquid_rows = []

    for task in C.ALL_TASKS:
        w = winners[task]
        model_name = w["model"]
        params = w["params"]
        tr = C.task_rows(train_all, task)
        va = C.task_rows(val_all, task)
        Xtr = tr[feats].to_numpy(np.float64)
        Ytr = tr[C.TARGETS[task]].to_numpy(np.float64)
        Xva = va[feats].to_numpy(np.float64)
        Yva = va[C.TARGETS[task]].to_numpy(np.float64)

        tdir = os.path.join(FINAL, task)
        os.makedirs(tdir, exist_ok=True)
        t0 = time.time()
        fit, predict = build_final_model(model_name, task, params)
        state = fit(Xtr, Ytr)
        fit_seconds = time.time() - t0

        # trivial baselines on FULL DEV_TRAIN
        if task == "presence":
            trivial_vec = C.trivial_classification_probs(Ytr.astype(int))
        else:
            tv = C.trivial_regression_pred(Ytr)
            trivial_vec = tv / tv.sum()

        with open(os.path.join(tdir, "model_state.pkl"), "wb") as f:
            pickle.dump({"state": state, "trivial": trivial_vec}, f)

        t0 = time.time()
        pred_va = predict(state, Xva)
        pred_seconds = time.time() - t0

        task_freeze = {
            "model_family": model_name,
            "hyperparameters": params,
            "train_rows": len(tr), "train_compositions": int(tr["composition_id"].nunique()),
            "val_rows": len(va), "n_features": len(feats),
            "fit_seconds": round(fit_seconds, 1),
            "val_predict_seconds": round(pred_seconds, 1),
            "seed": SEED,
        }

        if task == "presence":
            Yva_i = Yva.astype(int)
            ths = C.fit_thresholds_f1(Yva_i, pred_va)
            task_freeze["thresholds"] = dict(zip(C.FAMILIES_ORDER, ths))
            task_freeze["class_weights_train_only"] = C.class_weights_from_train(Ytr.astype(int))
            met = C.classification_metrics(Yva_i, pred_va, ths)
            met_default = C.classification_metrics(Yva_i, pred_va, [0.5] * 5)
            tri_prob = np.tile(trivial_vec, (len(Yva), 1))
            met_tri = C.classification_metrics(Yva_i, tri_prob, [0.5] * 5)
            val_rows.append({"stage": "validation", "task": task, "model": model_name,
                             "exact_match": met["exact_match"], "macro_f1": met["macro_f1"],
                             "micro_f1": met["micro_f1"], "hamming_loss": met["hamming_loss"],
                             "exact_match_default0.5": met_default["exact_match"],
                             "macro_f1_default0.5": met_default["macro_f1"],
                             "trivial_macro_f1": met_tri["macro_f1"],
                             "trivial_exact_match": met_tri["exact_match"]})
            for fam, d in met["per_label"].items():
                per_target_rows.append({"stage": "validation", "task": task, "model": model_name,
                                        "target": fam, **{k: v for k, v in d.items() if k != "note"}})
                if fam == "LIQUID":
                    liquid_rows.append({"stage": "validation", "model": model_name, "kind": "classification",
                                        **d})
            # also trivial per-label
            for fam, d in met_tri["per_label"].items():
                per_target_rows.append({"stage": "validation", "task": task, "model": "TrivialPrevalence",
                                        "target": fam, **{k: v for k, v in d.items() if k != "note"}})
        else:
            proj, n_fb = C.project_simplex(pred_va, trivial_vec)
            met = C.regression_metrics(Yva, pred_va, proj)
            tri_pred = np.tile(trivial_vec, (len(Yva), 1))
            met_tri = C.regression_metrics(Yva, tri_pred, tri_pred.copy())
            task_freeze["simplex_fallback_row"] = trivial_vec.tolist()
            task_freeze["val_simplex_fallbacks"] = n_fb
            val_rows.append({"stage": "validation", "task": task, "model": model_name,
                             "mean_tvd": met["mean_tvd_proj"], "median_tvd": met["median_tvd_proj"],
                             "p95_tvd": met["p95_tvd_proj"], "macro_mae": met["macro_mae_proj"],
                             "macro_rmse": met["macro_rmse_proj"],
                             "raw_sum_violation_mean": met["raw_sum_violation_mean"],
                             "proj_sum_violation_max": met["proj_sum_violation_max"],
                             "simplex_fallbacks": n_fb,
                             "trivial_mean_tvd": met_tri["mean_tvd_proj"],
                             "trivial_macro_mae": met_tri["macro_mae_proj"]})
            for fam, d in met["per_family"].items():
                per_target_rows.append({"stage": "validation", "task": task, "model": model_name,
                                        "target": fam, **d})
                if fam == "LIQUID" and task == "fraction":
                    liquid_rows.append({"stage": "validation", "model": model_name, "kind": "regression",
                                        "mae": d["mae_proj"], "rmse": d["rmse_proj"], "r2": d["r2_proj"],
                                        "true_mean": d["true_mean"], "true_std": d["true_std"]})
            for fam, d in met_tri["per_family"].items():
                per_target_rows.append({"stage": "validation", "task": task, "model": "TrivialMeanVector",
                                        "target": fam, **d})

        task_freeze["artifact_sha256"] = sha256_file(os.path.join(tdir, "model_state.pkl"))
        freeze["tasks"][task] = task_freeze
        print(f"  {task}: winner={model_name} fit={fit_seconds:.0f}s", flush=True)

    pd.DataFrame(val_rows).to_csv(os.path.join(C.RESULTS, "BASELINE_VALIDATION_METRICS.csv"), index=False)
    pd.DataFrame(per_target_rows).to_csv(os.path.join(C.RESULTS, "BASELINE_PER_TARGET_METRICS.csv"), index=False)
    pd.DataFrame(liquid_rows).to_csv(os.path.join(C.RESULTS, "LIQUID_VALIDATION_METRICS.csv"), index=False)

    freeze["sealed_sets_status"] = {
        "IID_TEST": "SEALED - no test metric computed by this script",
        "SEALED_DOMAIN_B": "UNOPENED - never loaded for training/validation/scaler/PCC/tuning",
        "SEALED_ANCHOR": "UNOPENED - composition count/family count/ID integrity only",
        "NI": "SEALED - zero descriptor rows, zero compositions, zero calculations",
    }
    man = os.path.join(C.STEP5A, "09_manifest", "DEVELOPMENT_FREEZE_V1.json")
    with open(man, "w", encoding="utf-8") as f:
        json.dump(freeze, f, indent=2, default=str)
    print(f"\nDEVELOPMENT_FREEZE_V1.json written: {man}")
    print(pd.DataFrame(val_rows)[["task", "model"]].to_string(index=False))
    print("STAGE M+N: done")


if __name__ == "__main__":
    main()
