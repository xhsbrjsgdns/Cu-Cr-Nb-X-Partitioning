# -*- coding: utf-8 -*-
"""
75_f5f_ablation.py
STEP5F Stage E: IDENTITY CONTRIBUTION ABLATION + Tree secondary check
(spec 44-51).

Preregistered configurations, all with the SAME frozen MLP setup as Stage D
(spec 48: no per-subset tuning):
  FULL5            BE SN MN LA P            (primary; metrics reused from 74)
  FULL5_MINUS_<X>  drop one identity        (5 configs)
  BASE_PLUS_<X>    add exactly one identity (5 configs)
Ablation answers ONLY scientific diagnosis (spec 49):
  A. does identity count itself matter?
  B. is the gain driven by the different-topology identities LA/P?
  C. do the FCC-like identities BE/SN/MN still add independent value?
FULL5 remains the production candidate; NO post-hoc subset selection (spec 50).

Tree secondary (spec 51): frozen Z0_TREE_DIRECT (STEP5C) vs TREE_FULL5 -
a direct multi-output XGBoost with the frozen per-fold STEP5C hyperparameters
trained on the augmented train set. Two model families, same direction check;
no further tree ablations.

Output:
  06_ablation/STEP5F_IDENTITY_ABLATION.csv
  06_ablation/STEP5F_TREE_SECONDARY.csv
  06_ablation/STEP5F_ABLATION_SUMMARY.json
"""
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5f_common as C


def load_stage_d():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "74_f5f_augmented_loeo.py")
    spec = importlib.util.spec_from_file_location("stage_d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CONFIGS = (["FULL5"]
           + ["FULL5_MINUS_%s" % x for x in C.SELECTED_ELEMENTS]
           + ["BASE_PLUS_%s" % x for x in C.SELECTED_ELEMENTS])


def new_elements_for_config(cfg):
    if cfg == "FULL5":
        return list(C.SELECTED_ELEMENTS)
    if cfg.startswith("FULL5_MINUS_"):
        return [x for x in C.SELECTED_ELEMENTS if x != cfg.split("_", 2)[2]]
    if cfg.startswith("BASE_PLUS_"):
        return [cfg.split("_", 2)[2]]
    raise ValueError(cfg)


def main():
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("STEP5F FULL  Stage E: identity ablation (11 configs x 6 folds) + Tree secondary")
    print("=" * 78)
    SD = load_stage_d()
    d, man = SD.load_original_data()
    feat_union = [c for c in d.columns
                  if not c.startswith("true_")
                  and c not in ("state_id", "composition_id", "family", "X_element",
                                "sampling_domain", "split_role")]
    new = SD.load_new_data(feat_union)

    # FULL5 fold metrics come from Stage D (identical protocol, no retraining)
    st_d = pd.read_csv(os.path.join(C.D_ML, "STEP5F_AUGMENTED_LOEO_METRICS.csv"))

    rows = []
    for cfg in CONFIGS:
        els = new_elements_for_config(cfg)
        reuse = (cfg == "FULL5")
        for e in C.HELDOUT_ELEMENTS:
            fold = "FOLD_%s" % e
            if reuse:
                r = st_d[st_d.fold == fold].iloc[0]
                rows.append({
                    "config": cfg, "new_elements": ";".join(els), "fold": fold,
                    "heldout_element": e,
                    "n_train_rows": int(r.n_train_rows_augmented),
                    "n_heldout_rows": int(r.n_heldout_rows),
                    "mean_tvd": r.FULL5_mean_tvd, "median_tvd": r.FULL5_median_tvd,
                    "p95_tvd": r.FULL5_p95_tvd,
                    "topology_accuracy": r.FULL5_topology_accuracy,
                    "topology_macro_f1": r.FULL5_topology_macro_f1,
                    "trivial_tvd": r.trivial_tvd,
                    "skill_vs_trivial": r.FULL5_skill_vs_trivial,
                    "beats_trivial": int(r.FULL5_beats_trivial),
                    "mean_tvd_vs_Z1": r.FULL5_mean_tvd - r.Z1_mean_tvd,
                    "train_seconds": 0.0, "reused_from_stage_D": 1})
                continue
            t0 = time.time()
            fr = json.load(open(C.C_FREEZE(e), encoding="utf-8"))
            fc = list(fr["selected_features"])
            baseline = np.array(fr["tasks"]["x_partition"]["trivial_baseline"], dtype=float)
            seed = SD.fold_seed(e)
            d2 = d.copy()
            d2["loeo_role"] = SD.attach_roles(d2, man, fold)
            tr_orig = d2[(d2.loeo_role == "FOLD_TRAIN") & (d2.family != e)
                         & (d2.family != "BASELINE")]
            va_orig = d2[(d2.loeo_role == "FOLD_VALIDATION") & (d2.family != e)]
            held_ids = set(pd.read_parquet(C.C_HELDOUT_PRED(e),
                                           columns=["state_id"]).state_id)
            held = d2[d2.state_id.isin(held_ids)]
            new_cols = (["state_id", "composition_id", "family"]
                        + [c for c in feat_union
                           if c not in ("state_id", "composition_id", "family")]
                        + C.TRUE_COLS)
            new_use = new[new.family.isin(els)][new_cols]
            tr_aug = pd.concat([tr_orig[new_cols], new_use], ignore_index=True)
            assert e not in set(tr_aug.family)
            art = SD.train_augmented_mlp(fold, e, tr_aug, va_orig, fc, baseline, seed, device)
            xt = torch.as_tensor(art["scaler"].transform(held[fc]),
                                 dtype=torch.float32, device=device)
            with torch.no_grad():
                raw = art["model"](xt).cpu().numpy()
            pred = C.project_simplex(raw, baseline)
            y_true = held[C.TRUE_COLS].to_numpy(float)
            m = C.metrics_block(y_true, pred, baseline)
            rows.append({
                "config": cfg, "new_elements": ";".join(els), "fold": fold,
                "heldout_element": e, "n_train_rows": int(len(tr_aug)),
                "n_heldout_rows": m["n_rows"],
                "mean_tvd": m["mean_tvd"], "median_tvd": m["median_tvd"],
                "p95_tvd": m["p95_tvd"], "topology_accuracy": m["topology_accuracy"],
                "topology_macro_f1": m["topology_macro_f1"], "trivial_tvd": m["trivial_tvd"],
                "skill_vs_trivial": m["skill_vs_trivial"],
                "beats_trivial": int(C.beats_trivial(m)),
                "mean_tvd_vs_Z1": m["mean_tvd"]
                                  - float(st_d.loc[st_d.fold == fold, "Z1_mean_tvd"].iloc[0]),
                "train_seconds": round(time.time() - t0, 1), "reused_from_stage_D": 0})
            print("  %-18s %-8s train %6d | mean TVD %.4f | topo %.3f | %.0f s"
                  % (cfg, fold, m["n_rows"], m["mean_tvd"], m["topology_accuracy"],
                     time.time() - t0), flush=True)

    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(C.D_ABL, "STEP5F_IDENTITY_ABLATION.csv"),
               index=False, encoding="utf-8")

    # ------------------------------------------------- Tree secondary (51) -- #
    print("\n--- Tree secondary: Z0_TREE_DIRECT (frozen) vs TREE_FULL5 ---")
    import joblib
    from xgboost import XGBRegressor
    trows = []
    for e in C.HELDOUT_ELEMENTS:
        fold = "FOLD_%s" % e
        t0 = time.time()
        fr = json.load(open(C.C_FREEZE(e), encoding="utf-8"))
        xt_task = fr["tasks"]["x_partition"]
        hp = dict(xt_task["hyperparameters"]
                  if xt_task["model_family"] == "XGBoost"
                  else xt_task["xgboost_branch_hyperparameters"])
        fc = list(fr["selected_features"])
        baseline = np.array(xt_task["trivial_baseline"], dtype=float)
        rs = int(fr["random_seed"])
        d2 = d.copy()
        d2["loeo_role"] = SD.attach_roles(d2, man, fold)
        tr_orig = d2[(d2.loeo_role == "FOLD_TRAIN") & (d2.family != e)
                     & (d2.family != "BASELINE")]
        held_ids = set(pd.read_parquet(C.C_HELDOUT_PRED(e),
                                       columns=["state_id"]).state_id)
        held = d2[d2.state_id.isin(held_ids)]
        new_cols = (["state_id", "composition_id", "family", "sampling_domain"]
                    + [c for c in feat_union
                       if c not in ("state_id", "composition_id", "family",
                                    "sampling_domain")]
                    + C.TRUE_COLS)
        new_use = new[new_cols]
        tr_aug = pd.concat([tr_orig[new_cols], new_use], ignore_index=True)
        base = dict(random_state=rs, n_jobs=-1, tree_method="hist", device="cuda")
        base.update(hp)
        yt = tr_aug[C.TRUE_COLS].to_numpy(float)
        P = np.zeros((len(held), 5))
        Xtr = tr_aug[fc].to_numpy(float)
        Xte = held[fc].to_numpy(float)
        models = []
        for i in range(5):
            m = XGBRegressor(**base)
            m.fit(Xtr, yt[:, i])
            P[:, i] = m.predict(Xte)
            models.append(m)
        joblib.dump(models, os.path.join(C.D_ABL, "FOLD_%s_TREE_FULL5.joblib" % e))
        pred = C.project_simplex(P, baseline)
        y_true = held[C.TRUE_COLS].to_numpy(float)
        m_t5 = C.metrics_block(y_true, pred, baseline)
        # frozen Z0 tree predictions (STEP5C), projected at freeze time
        zp = pd.read_parquet(C.C_HELDOUT_PRED(e))
        z0 = held[["state_id"]].merge(
            zp[["state_id"] + ["pred_proj_x_partition_%s" % f for f in C.FAMILIES_ORDER]],
            on="state_id", validate="one_to_one")
        z0_pred = z0[["pred_proj_x_partition_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)
        m_z0 = C.metrics_block(y_true, z0_pred, baseline)
        trows.append({"fold": fold, "heldout_element": e,
                      "Z0_mean_tvd": m_z0["mean_tvd"], "TREE_FULL5_mean_tvd": m_t5["mean_tvd"],
                      "Z0_topology_accuracy": m_z0["topology_accuracy"],
                      "TREE_FULL5_topology_accuracy": m_t5["topology_accuracy"],
                      "trivial_tvd": m_t5["trivial_tvd"],
                      "Z0_skill_vs_trivial": m_z0["skill_vs_trivial"],
                      "TREE_FULL5_skill_vs_trivial": m_t5["skill_vs_trivial"],
                      "same_direction_as_MLP": int(
                          (m_t5["mean_tvd"] < m_z0["mean_tvd"])
                          == (float(st_d.loc[st_d.fold == fold, "FULL5_mean_tvd"].iloc[0])
                              < float(st_d.loc[st_d.fold == fold, "Z1_mean_tvd"].iloc[0]))),
                      "train_seconds": round(time.time() - t0, 1)})
        print("  %-8s Z0 %.4f -> TREE_FULL5 %.4f | same direction as MLP: %s"
              % (fold, m_z0["mean_tvd"], m_t5["mean_tvd"],
                 bool(trows[-1]["same_direction_as_MLP"])), flush=True)
    trdf = pd.DataFrame(trows)
    trdf.to_csv(os.path.join(C.D_ABL, "STEP5F_TREE_SECONDARY.csv"),
                index=False, encoding="utf-8")

    # ------------------------------------------------------ summary --------- #
    piv = abl.groupby("config").agg(mean_tvd=("mean_tvd", "mean"),
                                    mean_topo=("topology_accuracy", "mean"),
                                    folds_beat=("beats_trivial", "sum")).reset_index()
    C.write_json(os.path.join(C.D_ABL, "STEP5F_ABLATION_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "configs": json.loads(piv.to_json(orient="records")),
        "tree_secondary": json.loads(trdf.to_json(orient="records")),
        "purpose": ("A identity-count effect; B LA/P differential-topology contribution; "
                    "C FCC-like independent value. Diagnosis only - FULL5 stays the "
                    "production candidate (spec 49, 50)"),
        "mlp_setup_frozen": True, "post_hoc_subset_selection": False})
    print("\n" + piv.to_string(index=False))
    print("STAGE E: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
