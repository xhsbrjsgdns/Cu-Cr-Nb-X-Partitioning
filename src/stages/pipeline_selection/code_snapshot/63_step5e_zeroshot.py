# -*- coding: utf-8 -*-
"""63_step5e_zeroshot.py — Stage F+G: develop the zero-shot remedy branches
Z2/Z3/Z4/Z5 on FOLD_TRAIN + FOLD_VALIDATION ONLY and freeze all six folds.

NO HELD-OUT ROW IS SCORED HERE. Z0 (frozen STEP5C tree) and Z1 (frozen STEP5D
MLP diagnostic) are references loaded at evaluation time, never refit."""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5e_common as C

def xgb_regressors(params):
    from xgboost import XGBRegressor
    base = dict(random_state=C.MASTER_SEED, n_jobs=-1, tree_method="hist", device="cuda")
    base.update(params)
    return base, XGBRegressor


def fit_direct(train, val, feats, params, affinity):
    """Direct multi-output regression: one XGB per component (Z2-style)."""
    use = list(feats) + (C.AFF_COLS if affinity else [])
    base, Cls = xgb_regressors(params)
    Ytr = train[C.X_TARGETS].to_numpy(float)
    models, Pval = [], np.zeros((len(val), 5))
    for i, t in enumerate(C.X_TARGETS):
        m = Cls(**base)
        m.fit(train[use].to_numpy(float), Ytr[:, i])
        Pval[:, i] = m.predict(val[use].to_numpy(float))
        models.append(m)
    return models, use, Pval


def train_multiclass(train, val, feats, params):
    """Topology classifier (stage 1): argmax of TRUE X-partition on FOLD_TRAIN states."""
    from xgboost import XGBClassifier
    ytr = C.dominant_topology(train[C.X_TARGETS].to_numpy(float))
    best = None
    for grid in C.TOPO_CLF_GRID:
        m = XGBClassifier(random_state=C.MASTER_SEED, n_jobs=-1, tree_method="hist",
                          device="cuda", eval_metric="mlogloss", **grid)
        m.fit(train[feats].to_numpy(float), ytr)
        pv = m.predict_proba(val[feats].to_numpy(float))
        # selection metric: multiclass macro F1 on fold-validation dominant labels
        from sklearn.metrics import f1_score
        f1 = f1_score(C.dominant_topology(val[C.X_TARGETS].to_numpy(float)),
                      pv.argmax(axis=1), average="macro", zero_division=0)
        if best is None or f1 > best[0]:
            best = (f1, grid, m)
    return best[2], best[1], best[0]


def predict_topology_first(clf, experts, fallback_models, test, feats):
    """Hard routing (spec 37)."""
    route = clf.predict(test[feats].to_numpy(float))
    P = np.zeros((len(test), 5))
    route_counts = {}
    for k in range(5):
        idx = np.flatnonzero(route == k)
        route_counts[k] = int(len(idx))
        if not len(idx):
            continue
        src = experts.get(k, fallback_models)
        for i, m in enumerate(src):
            P[idx, i] = m.predict(test[feats].to_numpy(float)[idx])
    return P, route, route_counts


def train_mlp(train, val, feats, affinity):
    """Z5: 128/64/5 MLP on (fold features + affinity), scaler on FOLD_TRAIN only."""
    import torch
    from sklearn.preprocessing import StandardScaler
    use = list(feats) + C.AFF_COLS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = StandardScaler().fit(train[use].to_numpy(float))
    Xtr = scaler.transform(train[use].to_numpy(float)).astype(np.float32)
    Xva = scaler.transform(val[use].to_numpy(float)).astype(np.float32)
    Ytr = train[C.X_TARGETS].to_numpy(np.float32)
    Yva = val[C.X_TARGETS].to_numpy(np.float32)
    tv_trivial = C.trivial_simplex(Ytr)
    best = None
    for lr in C.MLP_LRS:
        torch.manual_seed(C.MASTER_SEED)
        model = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 5)).to(dev)
        xt = torch.from_numpy(Xtr).to(dev)
        yt = torch.from_numpy(Ytr).to(dev)
        xv = torch.from_numpy(Xva).to(dev)
        yv = torch.from_numpy(Yva).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n = len(xt)
        g = torch.Generator().manual_seed(C.MASTER_SEED)
        best_state, best_tvd, bad = None, np.inf, 0
        bs = 8192
        for ep in range(C.MLP_EPOCHS):
            model.train()
            perm = torch.randperm(n, generator=g).to(dev)
            for j in range(0, n, bs):
                idx = perm[j:j + bs]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                raw = model(xv).cpu().numpy()
            proj, _ = C.project_simplex(raw, tv_trivial)
            tvd = float(C.tvd(proj, Yva).mean())
            if tvd < best_tvd - 1e-9:
                best_tvd, bad = tvd, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= C.MLP_PATIENCE:
                    break
        if best is None or best_tvd < best[0]:
            model.load_state_dict(best_state)
            best = (best_tvd, lr, model, scaler)
    return {"scaler": best[3], "model": best[2], "lr": best[1],
            "val_mean_tvd": best[0], "features": use}


def process_fold(e):
    t0 = time.time()
    fold = f"FOLD_{e}"
    seed = C.MASTER_SEED
    feats = [ln.strip() for ln in open(C.C_FEATLIST(e), encoding="utf-8")]
    df, feats = C.load_fold_features_and_targets(fold, feats)
    df = C.attach_affinity(df)
    # spec 27: affinity finite/constant/duplicate check on FOLD_TRAIN only
    tr = C.fold_rows(df, "FOLD_TRAIN")
    tr = tr[tr.family != "BASELINE"]
    va = C.fold_rows(df, "FOLD_VALIDATION")
    va = va[va.family != "BASELINE"]
    for c in C.AFF_COLS:
        assert np.isfinite(tr[c].to_numpy(float)).all()
        assert tr[c].nunique() > 1, f"{fold}: {c} constant on fold-train"
    assert not pd.Index(feats + C.AFF_COLS).duplicated().any()
    params = C.fold_xgb_params(e)
    print(f"  {fold}: train {len(tr)} rows / {tr.composition_id.nunique()} comps | "
          f"val {len(va)} rows | features {len(feats)}+{len(C.AFF_COLS)} affinity", flush=True)

    out = {"fold": fold, "heldout_element": e, "seed": seed,
           "xgb_hyperparameters": params, "n_train_rows": int(len(tr)),
           "n_val_rows": int(len(va)), "features_original": feats,
           "affinity_columns": C.AFF_COLS, "branches": {}}
    arts = {}

    # ---- Z2: affinity direct XGBoost ----
    z2, _, z2_val = fit_direct(tr, va, feats, params, affinity=True)
    arts["z2_models"] = z2
    out["branches"]["Z2"] = {"val_mean_tvd": float(C.tvd(
        C.project_simplex(z2_val, C.trivial_simplex(tr[C.X_TARGETS].to_numpy(float)))[0],
        va[C.X_TARGETS].to_numpy(float)).mean())}
    # global direct (no affinity) fallback model, trained once
    gmodels, _, gval = fit_direct(tr, va, feats, params, affinity=False)
    arts["global_direct_models"] = gmodels
    out["branches"]["GLOBAL_DIRECT_fallback"] = {"val_mean_tvd": float(C.tvd(
        C.project_simplex(gval, C.trivial_simplex(tr[C.X_TARGETS].to_numpy(float)))[0],
        va[C.X_TARGETS].to_numpy(float)).mean())}

    # ---- Z3 / Z4: topology-first ----
    for tag, use_aff in [("Z3", False), ("Z4", True)]:
        f_list = list(feats) + (C.AFF_COLS if use_aff else [])
        clf, clf_grid, clf_f1 = train_multiclass(tr, va, f_list, params)
        dom_tr = C.dominant_topology(tr[C.X_TARGETS].to_numpy(float))
        experts = {}
        expert_report = {}
        for k, fam in enumerate(C.FAMILIES_ORDER):
            sub = tr[dom_tr == k]
            if sub.composition_id.nunique() >= C.MIN_EXPERT_COMPS and \
               len(sub) >= C.MIN_EXPERT_STATES:
                ms, _, _ = fit_direct(sub, va, feats, params, affinity=use_aff)
                experts[k] = ms
                expert_report[fam] = {"n_comps": int(sub.composition_id.nunique()),
                                      "n_states": int(len(sub)), "expert": True}
            else:
                expert_report[fam] = {"n_comps": int(sub.composition_id.nunique()),
                                      "n_states": int(len(sub)), "expert": False,
                                      "fallback": "global direct"}
        # branch-consistent fallback: same feature set as the branch (spec 36)
        if use_aff:
            fb_models, _, _ = fit_direct(tr, va, feats, params, affinity=True)
        else:
            fb_models = gmodels
        Pv, route, counts = predict_topology_first(clf, experts, fb_models, va, f_list)
        proj, _ = C.project_simplex(Pv, C.trivial_simplex(tr[C.X_TARGETS].to_numpy(float)))
        yva = va[C.X_TARGETS].to_numpy(float)
        out["branches"][tag] = {
            "val_mean_tvd": float(C.tvd(proj, yva).mean()),
            "val_topo_accuracy": float((route == C.dominant_topology(yva)).mean()),
            "clf_grid": clf_grid, "clf_val_macro_f1": clf_f1,
            "route_counts": counts, "experts": expert_report}
        arts[f"{tag}_clf"] = clf
        arts[f"{tag}_experts"] = experts
        arts[f"{tag}_fallback"] = fb_models
        arts[f"{tag}_features"] = f_list
        # oracle diagnostic is computed at heldout-evaluation time with TRUE routing

    # ---- Z5: affinity MLP ----
    z5 = train_mlp(tr, va, feats, affinity=True)
    arts["z5"] = z5
    out["branches"]["Z5"] = {"val_mean_tvd": z5["val_mean_tvd"], "chosen_lr": z5["lr"]}

    out["frozen_utc"] = C.utcnow()
    art_path = os.path.join(C.D_ZS, f"{fold}_REMEDY_ARTIFACTS.pkl")
    with open(art_path, "wb") as f:
        pickle.dump(arts, f)
    out["artifact_sha256"] = C.sha256_file(art_path)
    out["artifact_path"] = art_path
    C.write_json(os.path.join(C.D_MAN, f"{fold}_REMEDY_FREEZE.json"), out)
    print(f"  {fold} FROZEN in {time.time()-t0:.0f}s | val TVD " +
          " ".join(f"{k}={v.get('val_mean_tvd', float('nan')):.4f}"
                   for k, v in out["branches"].items()), flush=True)


def main():
    gate_ok = all(os.path.exists(os.path.join(C.D_MAN, f"FOLD_{e}_REMEDY_FREEZE.json"))
                  for e in C.HELDOUT_ELEMENTS)
    if gate_ok:
        print("all six remedy freezes already exist - nothing to do")
        return
    assert os.path.exists(os.path.join(C.D_PREREG, "STEP5E_REMEDY_PREREGISTRATION.json"))
    assert os.path.exists(os.path.join(C.D_AFF, "AFFINITY_BRANCH_STATUS.json"))
    status = C.load_json(os.path.join(C.D_AFF, "AFFINITY_BRANCH_STATUS.json"))
    assert status["status"] == "AVAILABLE" and not status["ni_scan_hits"]
    for e in C.HELDOUT_ELEMENTS:
        fp = os.path.join(C.D_MAN, f"FOLD_{e}_REMEDY_FREEZE.json")
        if os.path.exists(fp):
            print(f"  FOLD_{e} already frozen - skipping")
            continue
        process_fold(e)
    print("\nSTAGE F+G: all six folds frozen (no held-out row scored)")


if __name__ == "__main__":
    main()
