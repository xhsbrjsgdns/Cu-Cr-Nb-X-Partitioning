# -*- coding: utf-8 -*-
"""
131_k_screen_predict.py
STEP5K Part B Stage 2: frozen-surrogate prediction over the virtual candidate
space + applicability filter (spec 22, 26).

Frozen surrogates (STEP5A final, never refit): fraction (CatBoost), cr_partition
(XGBoost), x_partition (XGBoost), loaded with the frozen
39_step5a_development_freeze.build_final_model convention and projected with the
frozen trivial fallback rows - the exact STEP5A IID-test inference path.

Features are built with the STEP5H-A featurization path whose bit-exact equality
with the historical STEP5A feature table was proven at source-consistency max
|delta| = 0 (STEP5H_A_SOURCE_CONSISTENCY_TEST.csv).

OOD applicability: mean 5-NN distance (600 C embedding) to STEP5A DEV_TRAIN
composition rows, StandardScaler fitted on the reference rows only, threshold
frozen at 1.5 in STEP5K_SCREENING_RULES_FREEZE.json BEFORE this stage runs.

NO Ni row is featurized or predicted here; the candidate table contains only the
six developed identities (asserted).
"""
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K Part B Stage 2: frozen-surrogate screening predictions")
    print("=" * 78)
    rules = json.load(open(os.path.join(C.D_SCREEN, "STEP5K_SCREENING_RULES_FREEZE.json"),
                           encoding="utf-8"))
    comps = pd.read_csv(os.path.join(C.D_SCREEN,
                                     "STEP5K_VIRTUAL_CANDIDATE_COMPOSITIONS.csv"))
    if C.sha256_file(os.path.join(C.D_SCREEN, "STEP5K_VIRTUAL_CANDIDATE_COMPOSITIONS.csv")) \
            != rules["candidate_space"]["compositions_sha256"]:
        raise SystemExit("FATAL: candidate table changed after the rules freeze.")
    assert comps.X_identity.isin(C.SIX).all() and len(comps) == 120000

    # ---- feature construction via the proven STEP5H-A path ------------------- #
    sys.path.insert(0, C.STEP5HA and os.path.join(C.STEP5HA, "code_snapshot"))
    M92 = load_module("h92", os.path.join(C.STEP5HA, "code_snapshot", "92_h_features.py"))
    a_desc = pd.read_parquet(os.path.join(C.STEP5A, "01_feature_sources",
                                          "ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.parquet"))
    need = ["AtomicNumber", "AtomicWeight", "Electronegativity", "VEC", "MeltingT",
            "FirstIonizationEnergy", "AtomicVolume_GS", "MetallicRadius", "Density"]
    desc = {r.element: {p: float(r[p]) for p in need} for _, r in a_desc.iterrows()}

    feats43 = [ln.strip() for ln in open(C.A_FEATLIST, encoding="utf-8") if ln.strip()]
    t0 = time.time()
    lvl_parts = []
    for el in C.SIX:
        sub = comps[comps.X_identity == el].reset_index(drop=True)
        lvl = M92.build_composition_features(sub, el, desc)
        lvl.insert(0, "candidate_id", sub.candidate_id.to_numpy())
        lvl_parts.append(lvl)
        print("  %s: composition features built (%.0f s cumulative)"
              % (el, time.time() - t0), flush=True)
    lvl = pd.concat(lvl_parts, ignore_index=True)
    missing = [f for f in feats43 if f != "Temperature_C" and f not in lvl.columns]
    assert not missing, missing

    # ---- OOD applicability (frozen rule) ------------------------------------- #
    from scipy.spatial import cKDTree
    from sklearn.preprocessing import StandardScaler
    ref_cols = list(dict.fromkeys(["state_id", "composition_id", "split_role"]
                                  + feats43))   # Temperature_C is feature #1
    ref = pd.read_parquet(os.path.join(C.STEP5A, "02_features",
                                       "FULL_V1_CANDIDATE_FEATURES.parquet"),
                          columns=ref_cols)
    ref = ref[(ref.split_role == "DEV_TRAIN") & (ref.Temperature_C == 600)]
    Xref = ref[feats43].to_numpy(float)
    scaler = StandardScaler().fit(Xref)
    tree = cKDTree(scaler.transform(Xref))
    cand600 = lvl.copy()
    cand600["Temperature_C"] = 600.0
    Zc = scaler.transform(cand600[feats43].to_numpy(float))
    dist, _ = tree.query(Zc, k=C.OOD_KNN, workers=-1)
    lvl["ood_knn5_distance"] = dist.mean(axis=1)
    n_in = int((lvl.ood_knn5_distance <= C.OOD_MAX_KNN_DIST).sum())
    print("  OOD filter: reference %d DEV_TRAIN compositions | inside trusted region "
          "%d / %d (%.1f%%) | median dist %.3f"
          % (len(ref), n_in, len(lvl), 100 * n_in / len(lvl),
             float(np.median(lvl.ood_knn5_distance))))

    # ---- state expansion + frozen surrogate predictions ---------------------- #
    freeze = json.load(open(os.path.join(C.STEP5A, "09_manifest",
                                         "DEVELOPMENT_FREEZE_V1.json"), encoding="utf-8"))
    sys.path.insert(0, C.STEP5A_CODE)
    M39 = load_module("a39", os.path.join(C.STEP5A_CODE,
                                          "39_step5a_development_freeze.py"))
    ids = comps[["candidate_id", "X_identity"]]
    lvl = lvl.merge(ids, on="candidate_id", how="left", validate="one_to_one")
    assert lvl.X_identity.notna().all()
    states = []
    for t in C.SCREEN_TEMPS:
        s = lvl[["candidate_id", "X_identity"] + [f for f in feats43
                                                  if f != "Temperature_C"]].copy()
        s["Temperature_C"] = float(t)
        states.append(s)
    st = pd.concat(states, ignore_index=True)
    # Temperature_C is itself feature #1 - deduplicate the selection list
    cols = list(dict.fromkeys(["candidate_id", "X_identity"] + feats43))
    st = st[cols]
    print("  state table: %d rows (%d compositions x %d temperatures)"
          % (len(st), len(lvl), len(C.SCREEN_TEMPS)))

    X = st[feats43].to_numpy(np.float64)
    assert np.isfinite(X).all()
    preds = {}
    for task in ("fraction", "cr_partition", "x_partition"):
        tf = freeze["tasks"][task]
        with open(os.path.join(C.STEP5A, "04_models", "final", task,
                               "model_state.pkl"), "rb") as f:
            blob = pickle.load(f)
        state, trivial = blob["state"], np.asarray(blob["trivial"], dtype=float)
        _, predict = M39.build_final_model(tf["model_family"], task, tf["hyperparameters"])
        t1 = time.time()
        raw = np.asarray(predict(state, X), dtype=float)
        P = np.clip(raw, 0.0, None)
        sums = P.sum(axis=1, keepdims=True)
        proj = np.where(sums > 0, P / np.where(sums > 0, sums, 1.0), trivial[None, :])
        preds[task] = proj
        print("  %s (%s): predicted %d rows in %.0f s | simplex fallbacks %d"
              % (task, tf["model_family"], len(X), time.time() - t1,
                 int((sums[:, 0] == 0).sum())), flush=True)

    for j, f in enumerate(C.FAMILIES_ORDER):
        st["frac_%s" % f] = preds["fraction"][:, j]
        st["Cr_P_%s" % f] = preds["cr_partition"][:, j]
        st["X_P_%s" % f] = preds["x_partition"][:, j]

    # drop the feature columns but KEEP Temperature_C (it doubles as a metadata
    # column for the downstream pivot; it is feature #1 of the frozen schema)
    drop_cols = [f for f in feats43 if f != "Temperature_C"]
    space = st.drop(columns=drop_cols).merge(
        lvl[["candidate_id", "ood_knn5_distance"]], on="candidate_id", how="left")
    out = os.path.join(C.D_SCREEN, "STEP5K_VIRTUAL_SCREENING_SPACE.parquet")
    space.to_parquet(out, index=False)

    C.write_json(os.path.join(C.D_SCREEN, "STEP5K_SCREENING_PREDICTION_QA.json"), {
        "generated_utc": C.utcnow(),
        "n_compositions": int(len(lvl)), "n_states": int(len(space)),
        "temperatures": C.SCREEN_TEMPS,
        "models": {t: {"family": freeze["tasks"][t]["model_family"],
                       "refit": False,
                       "artifact": os.path.join(C.STEP5A, "04_models", "final", t,
                                                "model_state.pkl")}
                   for t in preds},
        "feature_path_source_consistency": "STEP5H-A proven bit-exact (max |delta| 0)",
        "ood_reference_rows": int(len(ref)),
        "ood_inside_trusted": n_in,
        "ood_median_distance": float(np.median(lvl.ood_knn5_distance)),
        "ni_rows_featurized": 0, "ni_rows_predicted": 0,
        "space_sha256": C.sha256_file(out)})
    print("STAGE B2: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
