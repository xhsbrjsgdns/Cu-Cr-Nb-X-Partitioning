# -*- coding: utf-8 -*-
"""65_step5e_zeroshot_eval.py — Stage H+I: the FIRST held-out evaluation of the
zero-shot remedy branches, the preregistered selection rule, and the
oracle-topology diagnostic.

Z0 (frozen STEP5C tree) and Z1 (frozen STEP5D MLP) are loaded, never refit.
ORACLE_TOPOLOGY_CONDITIONAL routes with TRUE topology and is labelled
DIAGNOSTIC_ORACLE — never a candidate for ranking."""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5e_common as C

CANDIDATES = ["Z0_TREE_DIRECT", "Z1_MLP_DIRECT", "Z2_AFFINITY_XGB",
              "Z3_TOPOLOGY_FIRST", "Z4_AFFINITY_TOPOLOGY_FIRST", "Z5_AFFINITY_MLP"]


def predict_direct(models, X):
    return np.column_stack([m.predict(X) for m in models])


def evaluate_fold(e):
    fold = f"FOLD_{e}"
    fr = C.load_json(os.path.join(C.D_MAN, f"{fold}_REMEDY_FREEZE.json"))
    with open(fr["artifact_path"], "rb") as f:
        arts = pickle.load(f)
    feats = fr["features_original"]
    df, _ = C.load_fold_features_and_targets(fold, feats)
    df = C.attach_affinity(df)
    ho = C.fold_rows(df, "HELDOUT_ELEMENT_TEST")
    ho = ho[ho.family != "BASELINE"]
    tr = C.fold_rows(df, "FOLD_TRAIN")
    tr = tr[tr.family != "BASELINE"]
    y = ho[C.X_TARGETS].to_numpy(float)
    trivial = C.trivial_simplex(tr[C.X_TARGETS].to_numpy(float))
    Xo = ho[feats].to_numpy(float)
    Xa = ho[list(feats) + C.AFF_COLS].to_numpy(float)
    assert len(ho) in (57750, 56596), (fold, len(ho))  # MO loses 1154 failed rows

    preds = {}
    # Z0: frozen STEP5C tree projections
    z0 = C.load_frozen_z0(fold)
    z0 = ho[["state_id"]].merge(z0, on="state_id", how="left", validate="1:1")
    assert z0[[f"Z0_{f}" for f in C.FAMILIES_ORDER]].notna().all().all()
    preds["Z0_TREE_DIRECT"] = z0[[f"Z0_{f}" for f in C.FAMILIES_ORDER]].to_numpy(float)
    # Z1: frozen STEP5D MLP diagnostics, projected with the fold-train fallback
    z1 = C.load_frozen_z1(fold)
    z1 = ho[["state_id"]].merge(z1, on="state_id", how="left", validate="1:1")
    assert z1[[f"Z1_{f}" for f in C.FAMILIES_ORDER]].notna().all().all()
    raw1 = z1[[f"Z1_{f}" for f in C.FAMILIES_ORDER]].to_numpy(float)
    preds["Z1_MLP_DIRECT"], _ = C.project_simplex(raw1, trivial)
    # Z2
    preds["Z2_AFFINITY_XGB"], _ = C.project_simplex(predict_direct(arts["z2_models"], Xa), trivial)
    # Z3 / Z4 (hard routing)
    for tag in ["Z3", "Z4"]:
        P, route, counts = C.__dict__.get("_", None), None, None
        from importlib import import_module
        # reuse the routing helper from the training module logic
        P = np.zeros((len(ho), 5))
        rt = arts[f"{tag}_clf"].predict(Xo if tag == "Z3" else Xa)
        for k in range(5):
            idx = np.flatnonzero(rt == k)
            src = arts[f"{tag}_experts"].get(k, arts[f"{tag}_fallback"])
            Xref = Xo if tag == "Z3" else Xa
            for i, m in enumerate(src):
                P[idx, i] = m.predict(Xref[idx])
        preds[f"{tag}_TOPOLOGY_FIRST" if tag == "Z3" else f"{tag}_AFFINITY_TOPOLOGY_FIRST"], _ = \
            C.project_simplex(P, trivial)
    # Z5
    z5 = arts["z5"]
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xs = z5["scaler"].transform(Xa).astype(np.float32)
    with torch.no_grad():
        raw5 = z5["model"](torch.from_numpy(Xs).to(dev)).cpu().numpy()
    preds["Z5_AFFINITY_MLP"], _ = C.project_simplex(raw5, trivial)
    # ORACLE: true-topology routing with the Z4 conditional regressors (diagnostic)
    true_route = C.dominant_topology(y)
    Po = np.zeros((len(ho), 5))
    for k in range(5):
        idx = np.flatnonzero(true_route == k)
        src = arts["Z4_experts"].get(k, arts["Z4_fallback"])
        for i, m in enumerate(src):
            Po[idx, i] = m.predict(Xa[idx])
    preds["ORACLE_TOPOLOGY_CONDITIONAL"], _ = C.project_simplex(Po, trivial)

    rows = []
    for cand, Pm in preds.items():
        s = C.eval_summary(y, Pm, Pm, trivial)
        rows.append({
            "heldout": e, "candidate": cand,
            "backbone": {"Z0_TREE_DIRECT": "STEP5C Tree", "Z1_MLP_DIRECT": "STEP5D MLP",
                         "Z2_AFFINITY_XGB": "XGBoost", "Z3_TOPOLOGY_FIRST": "XGBoost",
                         "Z4_AFFINITY_TOPOLOGY_FIRST": "XGBoost",
                         "Z5_AFFINITY_MLP": "MLP",
                         "ORACLE_TOPOLOGY_CONDITIONAL": "XGBoost"}[cand],
            "affinity_used": cand in ("Z2_AFFINITY_XGB", "Z4_AFFINITY_TOPOLOGY_FIRST",
                                      "Z5_AFFINITY_MLP", "ORACLE_TOPOLOGY_CONDITIONAL"),
            "topology_first": cand in ("Z3_TOPOLOGY_FIRST", "Z4_AFFINITY_TOPOLOGY_FIRST",
                                       "ORACLE_TOPOLOGY_CONDITIONAL"),
            "diagnostic_oracle": cand == "ORACLE_TOPOLOGY_CONDITIONAL",
            "positive_skill": bool(s["mean_tvd"] < s["trivial_tvd"]),
            **s})
    out = pd.DataFrame(rows)
    pr = pd.DataFrame(Pm, columns=[f"LAST_{f}" for f in C.FAMILIES_ORDER])  # placeholder guard
    # persist per-fold predictions for every candidate
    store = ho[["state_id", "composition_id", "Temperature_C"]].copy()
    for cand, Pm in preds.items():
        for i, fam in enumerate(C.FAMILIES_ORDER):
            store[f"{cand}_{fam}"] = Pm[:, i]
        store[f"{cand}_true_topo"] = C.dominant_topology(y)
    for i, fam in enumerate(C.FAMILIES_ORDER):
        store[f"true_{fam}"] = y[:, i]
    store.to_parquet(os.path.join(C.D_ABL, f"{fold}_ZERO_SHOT_PREDICTIONS.parquet"), index=False)
    return out


def main():
    for e in C.HELDOUT_ELEMENTS:
        fp = os.path.join(C.D_MAN, f"FOLD_{e}_REMEDY_FREEZE.json")
        assert os.path.exists(fp), f"{fp} missing - run 63 first"
    man = os.path.join(C.D_MAN, "CODE_SNAPSHOT_MANIFEST_STEP5E.json")
    assert os.path.exists(man), "code snapshot missing - run 64 first"
    for entry in C.load_json(man)["files"]:
        assert C.sha256_file(entry["path"]) == entry["sha256"], entry["filename"]

    all_rows = []
    for e in C.HELDOUT_ELEMENTS:
        t0 = time.time()
        r = evaluate_fold(e)
        all_rows.append(r)
        print(f"  {e}: {len(r)} candidates scored in {time.time()-t0:.0f}s", flush=True)
    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(os.path.join(C.D_ABL, "STEP5E_ZERO_SHOT_PER_FOLD.csv"), index=False)

    # ---- ablation table (spec 72) ----
    abl = res[~res.diagnostic_oracle].copy()
    abl.to_csv(os.path.join(C.D_ABL, "STEP5E_ZERO_SHOT_ABLATION.csv"), index=False)
    res[res.diagnostic_oracle].to_csv(os.path.join(C.D_ABL, "STEP5E_ORACLE_TOPOLOGY_DIAGNOSTIC.csv"),
                                      index=False)

    # ---- preregistered selection rule (spec 42) ----
    summ = abl.groupby("candidate").agg(
        mean_tvd=("mean_tvd", "mean"),
        mean_topo_accuracy=("topology_accuracy", "mean"),
        folds_beat_trivial=("positive_skill", "sum")).reset_index()
    z0 = summ[summ.candidate == "Z0_TREE_DIRECT"].iloc[0]
    eligible = summ[(summ.folds_beat_trivial >= C.SELECTION_GUARDRAIL_MIN_FOLDS_BEAT_TRIVIAL)
                    & (summ.mean_topo_accuracy >= z0.mean_topo_accuracy)]
    pool = eligible if len(eligible) else summ
    best = pool.sort_values("mean_tvd").iloc[0]
    rec = {
        "recommended_zero_shot_pipeline": best.candidate,
        "rule": "lowest 6-fold mean TVD; guardrail >=3/6 folds beat fold-train trivial AND "
                "mean topology accuracy >= Z0 (spec 42); fixed before any held-out metric",
        "candidate_summary": summ.to_dict(orient="records"),
        "guardrail_satisfied_by_winner": bool(best.candidate in eligible.candidate.values),
        "selected_utc": C.utcnow(),
        "note": "defines the FUTURE Ni zero-shot pipeline; Ni itself remains sealed; few-shot "
                "results are generated after this selection and cannot change it (spec 43)"}
    C.write_json(os.path.join(C.D_MAN, "STEP5E_RECOMMENDED_ZERO_SHOT_PIPELINE.json"), rec)
    print(summ.to_string(index=False))
    print(f"\nRECOMMENDED ZERO-SHOT PIPELINE: {best.candidate}")
    print("STAGE H+I: done")


if __name__ == "__main__":
    main()
