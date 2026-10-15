# -*- coding: utf-8 -*-
"""66_step5e_fewshot.py — Stage J+K+L: run the preregistered few-shot recovery
experiment. Sequences were frozen in Stage C (target-free). Calibration labels
come ONLY from the frozen CALIBRATION_POOL nested prefixes; metrics are computed
ONLY on RETROSPECTIVE_RECOVERY_EVAL. No hyperparameter retuning: the zero-shot
frozen configurations are refit (weights/trees only) on
FOLD_TRAIN + selected calibration compositions (spec 56-57)."""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5e_common as C


def refit_direct(train, feats, params):
    from xgboost import XGBRegressor
    base = dict(random_state=C.MASTER_SEED, n_jobs=-1, tree_method="hist", device="cuda")
    base.update(params)
    models = []
    Y = train[C.X_TARGETS].to_numpy(float)
    X = train[feats].to_numpy(float)
    for i in range(5):
        m = XGBRegressor(**base)
        m.fit(X, Y[:, i])
        models.append(m)
    return models


def refit_topology_first(train, feats, clf_grid, params):
    """Z4 configuration refit: classifier + conditional experts + affinity fallback."""
    from xgboost import XGBClassifier, XGBRegressor
    base = dict(random_state=C.MASTER_SEED, n_jobs=-1, tree_method="hist", device="cuda")
    base.update(params)
    y_dom = C.dominant_topology(train[C.X_TARGETS].to_numpy(float))
    clf = XGBClassifier(random_state=C.MASTER_SEED, n_jobs=-1, tree_method="hist",
                        device="cuda", eval_metric="mlogloss", **clf_grid)
    clf.fit(train[feats].to_numpy(float), y_dom)
    experts, report = {}, {}
    for k, fam in enumerate(C.FAMILIES_ORDER):
        sub = train[y_dom == k]
        ok = (sub.composition_id.nunique() >= C.MIN_EXPERT_COMPS
              and len(sub) >= C.MIN_EXPERT_STATES)
        report[fam] = {"n_comps": int(sub.composition_id.nunique()),
                       "n_states": int(len(sub)), "expert": bool(ok)}
        if ok:
            ms = []
            X = sub[feats].to_numpy(float)
            Y = sub[C.X_TARGETS].to_numpy(float)
            for i in range(5):
                m = XGBRegressor(**base)
                m.fit(X, Y[:, i])
                ms.append(m)
            experts[k] = ms
    fallback = None
    if len(experts) < 5:
        fb = []
        X = train[feats].to_numpy(float)
        Y = train[C.X_TARGETS].to_numpy(float)
        for i in range(5):
            m = XGBRegressor(**base)
            m.fit(X, Y[:, i])
            fb.append(m)
        fallback = fb
    return clf, experts, fallback, report


def predict_topo(clf, experts, fallback, X, feats):
    route = clf.predict(X[feats].to_numpy(float))
    P = np.zeros((len(X), 5))
    for k in range(5):
        idx = np.flatnonzero(route == k)
        if not len(idx):
            continue
        src = experts.get(k, fallback)
        Xk = X[feats].to_numpy(float)[idx]
        for i, m in enumerate(src):
            P[idx, i] = m.predict(Xk)
    return P


def process_fold(e, curves):
    fold = f"FOLD_{e}"
    fr = C.load_json(os.path.join(C.D_MAN, f"{fold}_REMEDY_FREEZE.json"))
    with open(fr["artifact_path"], "rb") as f:
        arts = pickle.load(f)
    params = fr["xgb_hyperparameters"]
    feats = fr["features_original"]
    clf_grid = fr["branches"]["Z4"]["clf_grid"]

    df, feats = C.load_fold_features_and_targets(fold, feats)
    df = C.attach_affinity(df)
    tr = C.fold_rows(df, "FOLD_TRAIN")
    tr = tr[tr.family != "BASELINE"]
    ho = C.fold_rows(df, "HELDOUT_ELEMENT_TEST")
    ho = ho[ho.family != "BASELINE"]

    splits = pd.read_parquet(os.path.join(C.D_FROZEN, "FROZEN_RECOVERY_SPLITS.parquet"))
    seqs = pd.read_parquet(os.path.join(C.D_FROZEN, "FROZEN_FEW_SHOT_SEQUENCES.parquet"))
    sp = splits[splits.heldout_element == e]
    cal_pool = set(sp[sp.recovery_role == "CALIBRATION_POOL"]["composition_id"])
    eval_ids = set(sp[sp.recovery_role == "RETROSPECTIVE_RECOVERY_EVAL"]["composition_id"])
    sq = seqs[seqs.heldout_element == e]
    assert len(sq) == (1 + C.N_RANDOM_REPLICATES) * 32

    ev = ho[ho.composition_id.isin(eval_ids)]
    assert len(ev) and not (eval_ids & cal_pool)
    y_ev = ev[C.X_TARGETS].to_numpy(float)
    trivial = C.trivial_simplex(tr[C.X_TARGETS].to_numpy(float))
    trivial_tvd = float(C.tvd(np.tile(trivial, (len(ev), 1)), y_ev).mean())

    # frozen zero-shot references on the SAME eval subset (spec 60)
    z4 = C.load_json(os.path.join(C.D_MAN, f"{fold}_REMEDY_FREEZE.json"))
    zs_ref = {}
    # F0 reference: frozen GLOBAL_DIRECT (original features) zero-shot model
    Xe0 = ev[feats].to_numpy(float)
    zs_ref["F0_DIRECT_TREE"] = np.column_stack(
        [m.predict(Xe0) for m in arts["global_direct_models"]])
    # F1 reference: frozen Z4 model predictions (already stored at Stage H)
    st = pd.read_parquet(os.path.join(C.D_ABL, f"{fold}_ZERO_SHOT_PREDICTIONS.parquet"))
    st = ev[["state_id"]].merge(st[["state_id"] + [f"Z4_AFFINITY_TOPOLOGY_FIRST_{f}"
                                                   for f in C.FAMILIES_ORDER]],
                                on="state_id", how="left", validate="1:1")
    zs_ref["F1_AFFINITY_TOPOLOGY_FIRST"] = st[
        [f"Z4_AFFINITY_TOPOLOGY_FIRST_{f}" for f in C.FAMILIES_ORDER]].to_numpy(float)
    zs_tvd = {k: float(C.tvd(v, y_ev).mean()) for k, v in zs_ref.items()}

    for (strategy, rep), grp in sq.groupby(["strategy", "replicate"]):
        ordered = grp.sort_values("rank")["composition_id"].tolist()
        assert len(ordered) == 32 and len(set(ordered)) == 32
        assert set(ordered) <= cal_pool
        for budget in C.FEWSHOT_BUDGETS:
            chosen = ordered[:budget]                      # nested prefix (spec 49)
            assert all(c in cal_pool for c in chosen)
            calib_rows = ho[ho.composition_id.isin(chosen)]
            assert len(calib_rows) > 0
            joined = pd.concat([tr, calib_rows], ignore_index=True)
            t0 = time.time()
            # ---- F0: direct tree ----
            m0 = refit_direct(joined, feats, params)
            P0 = np.column_stack([m.predict(ev[feats].to_numpy(float)) for m in m0])
            P0, _ = C.project_simplex(P0, trivial)
            # ---- F1: affinity + topology-first ----
            f1_feats = list(feats) + C.AFF_COLS
            clf, experts, fb, _ = refit_topology_first(joined, f1_feats, clf_grid, params)
            P1 = predict_topo(clf, experts, fb, ev, f1_feats)
            P1, _ = C.project_simplex(P1, trivial)
            for arch, P, zs in [("F0_DIRECT_TREE", P0, zs_ref["F0_DIRECT_TREE"]),
                                ("F1_AFFINITY_TOPOLOGY_FIRST", P1, zs_ref["F1_AFFINITY_TOPOLOGY_FIRST"])]:
                t = C.tvd(P, y_ev)
                curves.append({
                    "heldout": e, "architecture": arch, "strategy": strategy,
                    "replicate": int(rep), "budget": budget,
                    "n_calibration_compositions": len(chosen),
                    "n_calibration_rows": int(len(calib_rows)),
                    "n_eval_compositions": int(ev.composition_id.nunique()),
                    "n_eval_rows": int(len(ev)),
                    "mean_tvd": float(t.mean()), "median_tvd": float(np.median(t)),
                    "p95_tvd": float(np.percentile(t, 95)),
                    "topology_accuracy": C.topology_accuracy(y_ev, P),
                    "topology_macro_f1": C._topo_macro_f1(y_ev, P),
                    "trivial_tvd": trivial_tvd,
                    "beats_trivial": bool(t.mean() < trivial_tvd),
                    "relative_recovery": float(1 - t.mean() / zs_tvd[arch])})
            print(f"  {e} {strategy} r{rep} b{budget}: F0 {t.mean() if False else float(C.tvd(P0, y_ev).mean()):.4f} "
                  f"F1 {float(C.tvd(P1, y_ev).mean()):.4f} ({time.time()-t0:.0f}s)", flush=True)


def main():
    assert os.path.exists(os.path.join(C.D_FROZEN, "FROZEN_FEW_SHOT_SEQUENCES.parquet"))
    assert os.path.exists(os.path.join(C.D_ABL, "STEP5E_ZERO_SHOT_ABLATION.csv")), \
        "zero-shot held-out evaluation must be frozen before few-shot runs"
    curves = []
    for e in C.HELDOUT_ELEMENTS:
        process_fold(e, curves)
    out = pd.DataFrame(curves)
    out.to_csv(os.path.join(C.D_FEWSHOT, "STEP5E_FEWSHOT_RECOVERY_CURVES.csv"), index=False)
    print(f"\nSTAGE J+K+L: {len(out)} curve rows written")


if __name__ == "__main__":
    main()
