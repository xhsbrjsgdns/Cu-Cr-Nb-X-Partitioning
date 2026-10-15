# -*- coding: utf-8 -*-
"""
93_h_ood.py
STEP5H-A Stage 3: TARGET-FREE OOD DIAGNOSTICS, frozen BEFORE any Ni CALPHAD
exists (spec 17-22).

Descriptive only: these diagnostics NEVER alter model choice or any prediction
(spec 17). They may later be correlated with blind error, but no threshold may
be selected after the reveal (spec 22).

  D1 (spec 18) feature-range diagnostic vs PRIMARY training rows, all 43 features
  D2 (spec 19) standardized extremeness per Ni state (frozen PRIMARY scaler):
     max_abs_z, mean_abs_z, #|z|>3, #|z|>5
  D3 (spec 20) nearest-neighbour distances in the PRIMARY standardized space:
     1-NN and 5-NN-mean per state, per-composition means, per-temperature summary
  D4 (spec 21) Ni descriptor position vs the original-six X descriptor vectors:
     nearest original identity, centroid distance, outside-range count
     (interpretation only - no model/role change)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5h_common as C


def primary_train_features(feats):
    """PRIMARY production training rows' features, from the frozen STEP5D
    diagnostic table restricted to the frozen STEP5G train-ID list."""
    import joblib  # noqa: F401  (kept out; scaler loaded by caller if needed)
    ids = pd.read_csv(C.G_TRAIN_IDS)
    id_col = "state_id" if "state_id" in ids.columns else ids.columns[0]
    d = pd.read_parquet(C.D_DIAG_DATA, columns=["state_id"] + feats)
    tr = d[d.state_id.isin(set(ids[id_col]))].reset_index(drop=True)
    if len(tr) != len(ids):
        raise SystemExit("FATAL: PRIMARY train-ID list (%d) does not match the "
                         "diagnostic table rows (%d)" % (len(ids), len(tr)))
    return tr


def main():
    import joblib
    from scipy.spatial import cKDTree
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5H-A  Stage 3: target-free OOD diagnostics (descriptive only)")
    print("=" * 78)
    C.assert_no_tc_python()

    feats = C.feature_schema()
    ni = pd.read_parquet(os.path.join(C.D_FEAT, "STEP5H_A_NI_FEATURES.parquet"))
    Xni = ni[feats].to_numpy(float)
    tr = primary_train_features(feats)
    Xtr = tr[feats].to_numpy(float)
    print("  Ni states %d x %d | PRIMARY train rows %d" % (Xni.shape[0], len(feats), len(Xtr)))

    # ---- D1 feature-range diagnostic (spec 18) ------------------------------ #
    tr_min, tr_max = Xtr.min(axis=0), Xtr.max(axis=0)
    ni_min, ni_max = Xni.min(axis=0), Xni.max(axis=0)
    below = (Xni < tr_min[None, :]).mean(axis=0)
    above = (Xni > tr_max[None, :]).mean(axis=0)
    rng = pd.DataFrame({
        "feature": feats, "train_min": tr_min, "train_max": tr_max,
        "ni_min": ni_min, "ni_max": ni_max,
        "fraction_ni_below_train_min": below,
        "fraction_ni_above_train_max": above,
        "outside_fraction": below + above,
        "extends_range": ((ni_min < tr_min) | (ni_max > tr_max)).astype(int)})
    rng.to_csv(os.path.join(C.D_OOD, "STEP5H_A_NI_FEATURE_RANGE_DIAGNOSTICS.csv"),
               index=False, encoding="utf-8")
    n_ext = int(rng.extends_range.sum())
    print("  D1: %d of 43 features have Ni values outside the PRIMARY training range"
          % n_ext)

    # ---- D2 standardized extremeness (spec 19), frozen PRIMARY scaler ------- #
    C.verify_hash(C.G_PRIMARY_SCALER, C.FROZEN_HASHES["primary_scaler"], "PRIMARY scaler")
    scaler = joblib.load(C.G_PRIMARY_SCALER)
    if list(scaler.feature_names_in_) != feats:
        raise SystemExit("FATAL: frozen PRIMARY scaler feature order != schema")
    Z = scaler.transform(Xni)          # NO clipping / winsorizing / capping (spec 16)
    absZ = np.abs(Z)
    per_state = ni[["state_id", "composition_id", "Temperature_C", "fewshot_role"]].copy()
    per_state["max_abs_z"] = absZ.max(axis=1)
    per_state["mean_abs_z"] = absZ.mean(axis=1)
    per_state["n_features_abs_z_gt_3"] = (absZ > 3).sum(axis=1)
    per_state["n_features_abs_z_gt_5"] = (absZ > 5).sum(axis=1)
    frac3 = float((per_state.n_features_abs_z_gt_3 > 0).mean())
    frac5 = float((per_state.n_features_abs_z_gt_5 > 0).mean())
    print("  D2: states with any |z|>3: %.1f%% | any |z|>5: %.1f%% | max_abs_z "
          "median %.2f, p95 %.2f" % (100 * frac3, 100 * frac5,
                                     per_state.max_abs_z.median(),
                                     per_state.max_abs_z.quantile(0.95)))

    # ---- D3 nearest-neighbour distances (spec 20) --------------------------- #
    Ztr = scaler.transform(Xtr)
    tree = cKDTree(Ztr)
    dist, _ = tree.query(Z, k=5, workers=-1)
    per_state["nn1_distance"] = dist[:, 0]
    per_state["knn5_mean_distance"] = dist[:, :5].mean(axis=1)
    per_comp = (per_state.groupby("composition_id")
                .agg(nn1_mean=("nn1_distance", "mean"),
                     knn5_mean=("knn5_mean_distance", "mean"),
                     max_abs_z_mean=("max_abs_z", "mean"))
                .reset_index())
    per_temp = (per_state.groupby("Temperature_C")
                .agg(nn1_mean=("nn1_distance", "mean"),
                     nn1_p95=("nn1_distance", lambda s: float(np.percentile(s, 95))),
                     knn5_mean=("knn5_mean_distance", "mean"),
                     max_abs_z_mean=("max_abs_z", "mean"),
                     frac_states_any_z_gt3=("n_features_abs_z_gt_3",
                                            lambda s: float((s > 0).mean())))
                .reset_index())
    per_state.to_csv(os.path.join(C.D_OOD, "STEP5H_A_NI_OOD_DIAGNOSTICS.csv"),
                     index=False, encoding="utf-8")
    per_comp.to_csv(os.path.join(C.D_OOD, "STEP5H_A_NI_OOD_PER_COMPOSITION.csv"),
                    index=False, encoding="utf-8")
    per_temp.to_csv(os.path.join(C.D_OOD, "STEP5H_A_NI_OOD_PER_TEMPERATURE.csv"),
                    index=False, encoding="utf-8")
    print("  D3: nn1 distance median %.3f p95 %.3f | 5-NN mean median %.3f"
          % (per_state.nn1_distance.median(),
             per_state.nn1_distance.quantile(0.95),
             per_state.knn5_mean_distance.median()))

    # ---- D4 descriptor position (spec 21), interpretation only -------------- #
    desc = pd.read_parquet(C.A_DESC_TABLE).set_index("element")
    props = [p for p, _, _, _, _ in C.NI_REQUIRED_PROPERTIES if p in desc.columns]
    six = desc.loc[C.ORIGINAL_SIX, props].astype(float)
    nir = pd.read_parquet(os.path.join(C.D_ACCESS,
                                       "STEP5H_A_NI_DESCRIPTOR_ROW.parquet")).iloc[0]
    niv = np.array([float(nir[p]) for p in props])
    mu, sd = six.mean(axis=0).to_numpy(), six.std(axis=0, ddof=0).to_numpy()
    sd_safe = np.where(sd > 0, sd, 1.0)
    z_six = (six.to_numpy() - mu) / sd_safe
    z_ni = (niv - mu) / sd_safe
    d_six = np.linalg.norm(z_six - z_ni[None, :], axis=1)
    nearest = C.ORIGINAL_SIX[int(np.argmin(d_six))]
    centroid_dist = float(np.linalg.norm(z_six.mean(axis=0) - z_ni))
    outside = int(((niv < six.min(axis=0).to_numpy())
                   | (niv > six.max(axis=0).to_numpy())).sum())
    d4 = {"properties_used": props,
          "nearest_original_identity": nearest,
          "z_distance_to_nearest": float(d_six.min()),
          "z_distance_to_six_centroid": centroid_dist,
          "per_identity_z_distance": {e: float(x) for e, x in zip(C.ORIGINAL_SIX, d_six)},
          "outside_six_range_count": outside,
          "outside_six_range_properties":
              [p for p, v in zip(props, niv)
               if v < six[p].min() or v > six[p].max()],
          "role": "interpretation only; no model, role or selection change (spec 21)"}
    print("  D4: nearest original identity %s (z-dist %.2f) | centroid z-dist %.2f | "
          "%d/%d properties outside the six-element range"
          % (nearest, d_six.min(), centroid_dist, outside, len(props)))

    # ---- freeze (spec 22) ---------------------------------------------------- #
    C.write_json(os.path.join(C.D_OOD, "STEP5H_A_OOD_DIAGNOSTICS_FREEZE.json"), {
        "generated_utc": C.utcnow(),
        "frozen_before_any_ni_calphad": True,
        "descriptive_only": True,
        "must_never_alter_model_choice_or_predictions": True,
        "no_threshold_may_be_selected_after_reveal": True,
        "n_features_extending_train_range": n_ext,
        "features_extending_train_range": list(rng.loc[rng.extends_range == 1, "feature"]),
        "frac_states_any_abs_z_gt3": frac3,
        "frac_states_any_abs_z_gt5": frac5,
        "max_abs_z_median": float(per_state.max_abs_z.median()),
        "max_abs_z_p95": float(per_state.max_abs_z.quantile(0.95)),
        "nn1_median": float(per_state.nn1_distance.median()),
        "nn1_p95": float(per_state.nn1_distance.quantile(0.95)),
        "knn5_median": float(per_state.knn5_mean_distance.median()),
        "descriptor_position": d4,
        "files": {f: C.sha256_file(os.path.join(C.D_OOD, f)) for f in
                  ("STEP5H_A_NI_FEATURE_RANGE_DIAGNOSTICS.csv",
                   "STEP5H_A_NI_OOD_DIAGNOSTICS.csv",
                   "STEP5H_A_NI_OOD_PER_COMPOSITION.csv",
                   "STEP5H_A_NI_OOD_PER_TEMPERATURE.csv")}})
    print("STAGE 3: PASS (diagnostics frozen; no target, no threshold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
