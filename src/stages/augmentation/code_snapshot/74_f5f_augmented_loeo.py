# -*- coding: utf-8 -*-
"""
74_f5f_augmented_loeo.py
STEP5F Stage D: AUGMENTED STRICT LOEO on the six frozen STEP5C heldout folds.

Design (spec 25-41):
  - the ONLY primary model is the frozen Z1_MLP_DIRECT formulation retrained on
    augmented data (AUGMENTED_MLP_FULL5): 128/64/5 MLP, raw-component MSE, Adam,
    batch 1024, <=200 epochs, patience 15 on raw validation MSE, per-fold seed,
    LR grid {1e-4, 3e-4, 1e-3} selected on PROJECTED validation TVD,
    StandardScaler on the (augmented) train rows, simplex projection with the
    FROZEN fold-train trivial row as fallback. No new Optuna / architecture /
    LR search (spec 30).
  - features: the frozen per-fold STEP5C selected list (spec 28, 29).
  - train = original FOLD_TRAIN rows (five old identities) + BE/SN/MN/LA/P
    new-element states (spec 31, 32).
  - validation = original FOLD_VALIDATION rows ONLY; new-element targets never
    enter validation or model selection (spec 33).
  - heldout test = the exact frozen STEP5C heldout compositions; no resplit
    (spec 34). The only change vs STEP5C/5E is training identity count 5 -> 10
    (spec 35).
  - Z1 (frozen STEP5D MLP diagnostic) and the frozen fold trivial baseline are
    loaded, never refit (spec 36, 38).
  - thresholds 0.602115 / 0.35 / 4-of-6 are applied as preregistered (spec 39-43).

Outputs (spec 57):
  05_augmented_ML/STEP5F_AUGMENTED_LOEO_METRICS.csv
  05_augmented_ML/STEP5F_AUGMENTED_LOEO_PER_ELEMENT.csv
  05_augmented_ML/STEP5F_FULL5_PREDICTIONS_<X>.parquet  (per original heldout X)
"""
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5f_common as C

MLP_LRS = [1e-4, 3e-4, 1e-3]        # frozen STEP5D preregistration
MLP_EPOCHS = 200
MLP_PATIENCE = 15
MLP_BATCH = 1024


def seed_all(s):
    import random
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def network(n):
    import torch
    from torch import nn
    return nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                         nn.Linear(64, 5))


def load_original_data():
    """Frozen STEP5D diagnostic table: Domain-A six identities, union features,
    true X-partition targets (read-only)."""
    d = pd.read_parquet(C.D_DIAG_DATA)
    man = pd.read_csv(C.C_MANIFEST)
    return d, man


def attach_roles(d, man, fold):
    m = man[man.fold == fold].set_index("composition_id").loeo_role
    return d.composition_id.map(m)


def load_new_data(feat_union):
    """New-element rows: features + targets, converged states only."""
    f = pd.read_parquet(os.path.join(C.D_PROC, "STEP5F_NEW_ELEMENT_CANDIDATE_FEATURES.parquet"))
    t = pd.read_parquet(os.path.join(C.D_PROC, "STEP5F_FULL_NEW_ELEMENT_DATA.parquet"),
                        columns=["state_id", "element", "state_converged"] + C.X_TARGETS)
    t = t[t.state_converged == 1].rename(
        columns=dict(zip(C.X_TARGETS, C.TRUE_COLS)))
    df = f.merge(t, on="state_id", validate="one_to_one")
    assert df[C.TRUE_COLS].notna().all().all(), "NaN target on a converged new state"
    assert df[feat_union].notna().all().all(), "NaN feature on a new state"
    return df


def fold_seed(e):
    """Frozen STEP5D preregistered per-fold seed."""
    import json
    p = json.load(open(os.path.join(C.STEP5D, "01_preregistration",
                                    "STEP5D_FAILURE_DIAGNOSIS_PREREGISTRATION.json"),
                       encoding="utf-8"))
    return int(p["models"]["seeds"][e])


def train_augmented_mlp(fold, e, tr, va, fc, baseline, seed, device):
    """Frozen Z1 protocol trained on the given (augmented) train/val frames."""
    import joblib
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    yt = tr[C.TRUE_COLS].to_numpy(float)
    yv = va[C.TRUE_COLS].to_numpy(float)
    scaler = StandardScaler().fit(tr[fc])
    xt_np = scaler.transform(tr[fc])
    xv_np = scaler.transform(va[fc])
    assert np.isfinite(xt_np).all() and np.isfinite(xv_np).all()
    xt = torch.as_tensor(xt_np, dtype=torch.float32, device=device)
    xv = torch.as_tensor(xv_np, dtype=torch.float32, device=device)
    yt_t = torch.as_tensor(yt, dtype=torch.float32, device=device)
    yv_t = torch.as_tensor(yv, dtype=torch.float32, device=device)
    best_mlp, lr_details = (np.inf, None, None), []
    with threadpool_limits(limits=2):
        for lr in MLP_LRS:
            seed_all(seed)
            model = network(xt.shape[1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            best_mse, best_epoch, best_state, wait = np.inf, 0, None, 0
            start = time.time()
            for epoch in range(1, MLP_EPOCHS + 1):
                model.train()
                order = torch.randperm(len(xt), device=device)
                for idx in order.split(MLP_BATCH):
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(model(xt[idx]), yt_t[idx])
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    raw = model(xv)
                    vmse = float(nn.functional.mse_loss(raw, yv_t).item())
                if vmse < best_mse - 1e-8:
                    best_mse, best_epoch = vmse, epoch
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                if wait >= MLP_PATIENCE:
                    break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                raw = model(xv).cpu().numpy()
            score = float(C.tvd(C.project_simplex(raw, baseline), yv).mean())
            detail = {"fold": fold, "learning_rate": lr, "validation_tvd": score,
                      "best_epoch": best_epoch, "epochs_run": epoch,
                      "best_validation_mse": best_mse,
                      "elapsed_seconds": round(time.time() - start, 1)}
            lr_details.append(detail)
            if score < best_mlp[0] - 1e-12:
                best_mlp = (score, detail, best_state)
    model = network(xt.shape[1]).to(device)
    model.load_state_dict(best_mlp[2])
    model.eval()
    return {"model": model, "scaler": scaler, "chosen_lr": best_mlp[1]["learning_rate"],
            "val_tvd": best_mlp[0], "lr_details": lr_details,
            "n_train_rows": int(len(tr)), "n_val_rows": int(len(va))}


def main():
    import torch
    import joblib
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("STEP5F FULL  Stage D: augmented strict LOEO (frozen Z1 protocol, FULL5)")
    print("device: %s" % device)
    print("=" * 78)

    d, man = load_original_data()
    feat_union = [c for c in d.columns
                  if not c.startswith("true_")
                  and c not in ("state_id", "composition_id", "family", "X_element",
                                "sampling_domain", "split_role")]
    new = load_new_data(feat_union)
    print("original rows %d | new-element rows %d (5 identities x 2000 comps x 11 T)"
          % (len(d), len(new)))

    rows, per_rows, structures = [], [], []
    for e in C.HELDOUT_ELEMENTS:
        fold = "FOLD_%s" % e
        t0 = time.time()
        fr = __import__("json").load(open(C.C_FREEZE(e), encoding="utf-8"))
        fc = list(fr["selected_features"])
        baseline = np.array(fr["tasks"]["x_partition"]["trivial_baseline"], dtype=float)
        seed = fold_seed(e)

        d2 = d.copy()
        d2["loeo_role"] = attach_roles(d2, man, fold)
        tr_orig = d2[(d2.loeo_role == "FOLD_TRAIN") & (d2.family != e)
                     & (d2.family != "BASELINE")]
        va_orig = d2[(d2.loeo_role == "FOLD_VALIDATION") & (d2.family != e)]
        assert set(tr_orig.family) == set(C.HELDOUT_ELEMENTS) - {e}
        held_ids = set(pd.read_parquet(C.C_HELDOUT_PRED(e),
                                       columns=["state_id"]).state_id)
        held = d2[d2.state_id.isin(held_ids)]
        assert set(held.family) == {e} and len(held) == len(held_ids)

        new_cols = (["state_id", "composition_id", "family", "sampling_domain"]
                    + [c for c in feat_union
                       if c not in ("state_id", "composition_id", "family",
                                    "sampling_domain")]
                    + C.TRUE_COLS)
        new_use = new[new_cols]
        tr_aug = pd.concat([tr_orig[new_cols], new_use], ignore_index=True)
        assert not set(tr_aug.composition_id) & set(va_orig.composition_id)
        assert not set(tr_aug.composition_id) & set(held.composition_id)
        # heldout-element targets must never be in train (spec 1, audit 17)
        assert e not in set(tr_aug.family)

        art = train_augmented_mlp(fold, e, tr_aug, va_orig, fc, baseline, seed, device)
        structures.append({
            "fold": fold, "heldout_identity": e,
            "train_identities": sorted(set(tr_aug.family)),
            "new_identities_added": sorted(C.SELECTED_ELEMENTS),
            "val_identities": sorted(set(va_orig.family.unique())),
            "heldout_target_in_train": int(e in set(tr_aug.family)),
            "heldout_target_in_validation": int(e in set(va_orig.family.unique())),
            "anchor_rows_in_train": int(tr_aug.composition_id.str.contains(
                "ANCHOR", case=False).sum()),
            "domain_B_rows_in_train": int((tr_aug.sampling_domain == "B").sum()),
            "n_train_rows_original": int(len(tr_orig)),
            "n_train_rows_augmented": int(len(tr_aug)),
            "n_new_rows": int(len(new_use)),
            "n_val_rows": int(len(va_orig)), "n_heldout_rows": int(len(held)),
            "feature_list_source": "STEP5C 03_features/FOLD_%s_SELECTED_FEATURES.txt" % e,
            "feature_list_sha256": C.sha256_file(C.C_FEATLIST(e)),
            "freeze_json_sha256": C.sha256_file(C.C_FREEZE(e)),
            "chosen_lr": art["chosen_lr"], "val_tvd": art["val_tvd"]})
        joblib.dump(art["scaler"], os.path.join(C.D_ML, "FOLD_%s_FULL5_scaler.joblib" % e))
        torch.save(art["model"].state_dict(),
                   os.path.join(C.D_ML, "FOLD_%s_FULL5_mlp.pt" % e))

        x_held = art["scaler"].transform(held[fc])
        xt = torch.as_tensor(x_held, dtype=torch.float32, device=device)
        with torch.no_grad():
            raw = art["model"](xt).cpu().numpy()
        pred = C.project_simplex(raw, baseline)
        y_true = held[C.TRUE_COLS].to_numpy(float)
        m_full5 = C.metrics_block(y_true, pred, baseline)

        # frozen Z1 (STEP5D MLP diagnostic) - loaded, never refit (spec 36)
        z1p = pd.read_parquet(C.D_Z1PRED(e))
        z1 = z1p[["state_id"] + ["MLP_%s" % f for f in C.FAMILIES_ORDER]].copy()
        z1 = held[["state_id"]].merge(z1, on="state_id", validate="one_to_one")
        z1_pred = z1[["MLP_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)
        m_z1 = C.metrics_block(y_true, z1_pred, baseline)

        out = held[["state_id", "composition_id", "Temperature_C"]].copy()
        for j, f in enumerate(C.FAMILIES_ORDER):
            out["true_%s" % f] = y_true[:, j]
            out["FULL5_%s" % f] = pred[:, j]
            out["Z1_%s" % f] = z1_pred[:, j]
        out.to_parquet(os.path.join(C.D_ML, "STEP5F_FULL5_PREDICTIONS_%s.parquet" % e),
                       index=False)

        rel_red = 1.0 - m_full5["mean_tvd"] / m_z1["mean_tvd"]
        rows.append({"fold": fold, "heldout_element": e,
                     "n_train_rows_original": int(len(tr_orig)),
                     "n_train_rows_augmented": int(len(tr_aug)),
                     "n_new_rows_added": int(len(new_use)),
                     "n_val_rows": int(len(va_orig)), "n_heldout_rows": int(len(held)),
                     "chosen_lr": art["chosen_lr"], "val_tvd": art["val_tvd"],
                     "Z1_mean_tvd": m_z1["mean_tvd"], "FULL5_mean_tvd": m_full5["mean_tvd"],
                     "Z1_median_tvd": m_z1["median_tvd"],
                     "FULL5_median_tvd": m_full5["median_tvd"],
                     "Z1_p95_tvd": m_z1["p95_tvd"], "FULL5_p95_tvd": m_full5["p95_tvd"],
                     "Z1_topology_accuracy": m_z1["topology_accuracy"],
                     "FULL5_topology_accuracy": m_full5["topology_accuracy"],
                     "Z1_topology_macro_f1": m_z1["topology_macro_f1"],
                     "FULL5_topology_macro_f1": m_full5["topology_macro_f1"],
                     "trivial_tvd": m_full5["trivial_tvd"],
                     "Z1_skill_vs_trivial": m_z1["skill_vs_trivial"],
                     "FULL5_skill_vs_trivial": m_full5["skill_vs_trivial"],
                     "relative_tvd_reduction_vs_Z1": rel_red,
                     "FULL5_beats_trivial": int(C.beats_trivial(m_full5)),
                     "train_seconds": round(time.time() - t0, 1)})
        pr = pd.DataFrame(art["lr_details"])          # lr_details already carry `fold`
        per_rows.append(pr)
        print("  %-8s train %6d rows (orig %d + new %d) | val %5d | LR %g | val TVD %.4f | "
              "heldout: Z1 %.4f -> FULL5 %.4f (%+.1f%%) | topo %.3f | %.0f s"
              % (fold, len(tr_aug), len(tr_orig), len(new_use), len(va_orig),
                 art["chosen_lr"], art["val_tvd"], m_z1["mean_tvd"], m_full5["mean_tvd"],
                 100 * (m_full5["mean_tvd"] / m_z1["mean_tvd"] - 1),
                 m_full5["topology_accuracy"], time.time() - t0), flush=True)

    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(C.D_ML, "STEP5F_AUGMENTED_LOEO_METRICS.csv"),
               index=False, encoding="utf-8")
    lrs = pd.concat(per_rows, ignore_index=True)
    lrs.to_csv(os.path.join(C.D_ML, "STEP5F_AUGMENTED_LOEO_LR_TRAILS.csv"),
               index=False, encoding="utf-8")

    # fold structures - audit evidence for identity/leakage/feature/hyperparameter
    # freeze (integrity items 15-20), captured at training time
    C.write_json(os.path.join(C.D_ML, "STEP5F_AUGMENTED_LOEO_FOLD_STRUCTURES.json"), {
        "generated_utc": C.utcnow(),
        "protocol": {"batch_size": MLP_BATCH, "max_epochs": MLP_EPOCHS,
                     "patience": MLP_PATIENCE, "learning_rates": MLP_LRS,
                     "source": "STEP5D preregistration (models.mlp), unchanged",
                     "protocol_sha256": C.sha256_file(os.path.join(
                         C.STEP5D, "01_preregistration",
                         "STEP5D_FAILURE_DIAGNOSIS_PREREGISTRATION.json"))},
        "folds": structures})

    mean_z1 = float(met.Z1_mean_tvd.mean())
    mean_f5 = float(met.FULL5_mean_tvd.mean())
    n_beat = int(met.FULL5_beats_trivial.sum())
    summary = {
        "generated_utc": C.utcnow(),
        "mean_Z1_tvd": mean_z1, "mean_FULL5_tvd": mean_f5,
        "relative_reduction_vs_Z1": 1.0 - mean_f5 / mean_z1,
        "mean_topology_accuracy": float(met.FULL5_topology_accuracy.mean()),
        "folds_beating_trivial": n_beat,
        "primary_success": bool(mean_f5 <= C.PRIMARY_SUCCESS_TVD),
        "topology_criterion": bool(met.FULL5_topology_accuracy.mean() >= C.TOPOLOGY_ACC_CRITERION),
        "skill_criterion": bool(n_beat >= C.FOLDS_BEAT_TRIVIAL_MIN),
        "strong_improvement": bool(mean_f5 <= C.STRONG_TVD
                                   and n_beat >= C.STRONG_FOLDS_BEAT),
        "thresholds": {"primary_success_tvd": C.PRIMARY_SUCCESS_TVD,
                       "topology_accuracy": C.TOPOLOGY_ACC_CRITERION,
                       "folds_beat_trivial": C.FOLDS_BEAT_TRIVIAL_MIN,
                       "strong_tvd": C.STRONG_TVD, "strong_folds": C.STRONG_FOLDS_BEAT},
    }
    C.write_json(os.path.join(C.D_ML, "STEP5F_AUGMENTED_LOEO_SUMMARY.json"), summary)
    print("\n  mean TVD: Z1 %.6f -> FULL5 %.6f  (rel. reduction %.1f%%)"
          % (mean_z1, mean_f5, 100 * (1 - mean_f5 / mean_z1)))
    print("  mean topology accuracy: %.4f | folds beating trivial: %d/6"
          % (summary["mean_topology_accuracy"], n_beat))
    print("  primary_success=%s topology_criterion=%s skill_criterion=%s strong=%s"
          % (summary["primary_success"], summary["topology_criterion"],
             summary["skill_criterion"], summary["strong_improvement"]))
    print("STAGE D: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
