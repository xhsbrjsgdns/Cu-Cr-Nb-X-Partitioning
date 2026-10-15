# -*- coding: utf-8 -*-
"""
82_g_train.py
STEP5G Stage 2: PRODUCTION TRAINING AND PIPELINE FREEZE (spec 8-30).

Two models are trained, frozen and hashed here. Nothing about either is chosen in
this stage: architecture, optimizer, batch size, epoch cap, patience, learning-rate
grid, selection rule, projection rule and feature schema all come in unchanged from
the frozen STEP5D/STEP5F protocol and the frozen STEP5A 43-feature list.

  PRIMARY   = ORIGINAL6_DIRECT_MLP
              train      : Domain-A DEV_TRAIN, six original identities (ZR TI CO SI AG MO)
              validation : Domain-A DEV_VALIDATION, same six identities
  SECONDARY = FULL11_DIRECT_MLP
              train      : the PRIMARY train rows + every STEP5F BE/SN/MN/LA/P state
              validation : the SAME original-six DEV_VALIDATION rows, unchanged

Why PRIMARY is the original six (spec 11): all four preregistered STEP5F switching
gates failed (mean TVD 0.710332 vs required <= 0.602115; topology 0.3323 vs required
>= 0.35; 3/6 folds beat trivial vs required >= 4/6; strong-improvement criterion not
met). The preregistered rule then keeps ORIGINAL6 as PRIMARY. FULL11 is carried as
SECONDARY because the augmentation did reduce mean LOEO TVD by 5.6% - a real but
insufficient effect, reported as such.

Not done here, deliberately (spec 14, 21, 30):
  - no NZV / PCC / SHAP / Optuna / LOEO feature selection of any kind
  - no element-balanced, identity-balanced or class-balanced row weighting; the
    SECONDARY training set is the natural union, so the five added identities are
    31% of its rows and that is reported rather than corrected
  - no architecture, learning-rate, seed or scaler search beyond the frozen LR grid
  - production use of BASE_PLUS_MN or any FULL5_MINUS_<X> subset is prohibited; those
    were diagnostics (spec 10) and are refused explicitly below

After this stage both models are IMMUTABLE (spec 30). No Ni descriptor, Ni property,
Ni prediction or Ni CALPHAD state exists or is read anywhere in this file.
"""
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_common as C
import g_contract as G

META = ["state_id", "composition_id", "family", "sampling_domain", "split_role"]


def seed_all(s):
    import random
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def network(n):
    """Frozen Z1 architecture - identical to 74_f5f_augmented_loeo.network()."""
    from torch import nn
    return nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                         nn.Linear(64, 5))


def train_frozen_mlp(role, tr, va, feats, fallback, seed, device):
    """Frozen Z1 protocol: StandardScaler on train rows, raw-component MSE, Adam,
    batch 1024, <= 200 epochs, patience 15 on RAW validation MSE, learning rate
    chosen on PROJECTED validation TVD. Nothing here is searched or tuned."""
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    yt = tr[C.TRUE_COLS].to_numpy(float)
    yv = va[C.TRUE_COLS].to_numpy(float)
    scaler = StandardScaler().fit(tr[feats])
    xt_np, xv_np = scaler.transform(tr[feats]), scaler.transform(va[feats])
    assert np.isfinite(xt_np).all() and np.isfinite(xv_np).all(), "non-finite feature"
    xt = torch.as_tensor(xt_np, dtype=torch.float32, device=device)
    xv = torch.as_tensor(xv_np, dtype=torch.float32, device=device)
    yt_t = torch.as_tensor(yt, dtype=torch.float32, device=device)
    yv_t = torch.as_tensor(yv, dtype=torch.float32, device=device)

    best = (np.inf, None, None)
    trail = []
    with threadpool_limits(limits=4):
        for lr in C.MLP_LRS:
            seed_all(seed)
            model = network(xt.shape[1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            best_mse, best_epoch, best_state, wait = np.inf, 0, None, 0
            t0 = time.time()
            for epoch in range(1, C.MLP_EPOCHS + 1):
                model.train()
                for idx in torch.randperm(len(xt), device=device).split(C.MLP_BATCH):
                    opt.zero_grad(set_to_none=True)
                    nn.functional.mse_loss(model(xt[idx]), yt_t[idx]).backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vmse = float(nn.functional.mse_loss(model(xv), yv_t).item())
                if vmse < best_mse - 1e-8:
                    best_mse, best_epoch, wait = vmse, epoch, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    wait += 1
                if wait >= C.MLP_PATIENCE:
                    break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                raw = model(xv).cpu().numpy()
            proj, n_fb = C.project_simplex_counted(raw, fallback)
            score = float(C.tvd(proj, yv).mean())
            d = {"role": role, "learning_rate": lr, "validation_tvd": score,
                 "best_validation_mse": best_mse, "best_epoch": best_epoch,
                 "epochs_run": epoch, "simplex_fallbacks": n_fb,
                 "elapsed_seconds": round(time.time() - t0, 1)}
            trail.append(d)
            print("    lr %-7g | best epoch %3d/%3d | val MSE %.6f | val TVD %.6f | %.0f s"
                  % (lr, best_epoch, epoch, best_mse, score, time.time() - t0), flush=True)
            if score < best[0] - 1e-12:
                best = (score, d, best_state)

    model = network(xt.shape[1]).to(device)
    model.load_state_dict(best[2])
    model.eval()
    with torch.no_grad():
        raw = model(xv).cpu().numpy()
    proj, n_fb = C.project_simplex_counted(raw, fallback)
    triv = np.repeat(np.asarray(fallback, float)[None, :], len(yv), axis=0)
    acc, f1 = C.topology_metrics(yv, proj)
    v = C.tvd(proj, yv)
    metrics = {"mean_tvd": float(v.mean()), "median_tvd": float(np.median(v)),
               "p95_tvd": float(np.quantile(v, 0.95)),
               "topology_accuracy": acc, "topology_macro_f1": f1,
               "trivial_tvd": float(C.tvd(triv, yv).mean()),
               "simplex_fallbacks": n_fb}
    metrics["skill_vs_trivial"] = metrics["trivial_tvd"] - metrics["mean_tvd"]
    return {"model": model, "scaler": scaler, "chosen_lr": best[1]["learning_rate"],
            "lr_trail": trail, "selection": best[1], "validation_metrics": metrics,
            "n_train_rows": int(len(tr)), "n_val_rows": int(len(va))}


def load_frames(feats):
    o = pd.read_parquet(C.D_DIAG_DATA, columns=META + feats + C.TRUE_COLS)
    o = o[o[C.TRUE_COLS].notna().all(axis=1)]
    tr = o[o.split_role == "DEV_TRAIN"].reset_index(drop=True)
    va = o[o.split_role == "DEV_VALIDATION"].reset_index(drop=True)

    nf = pd.read_parquet(C.F_NEWFEAT, columns=META + feats)
    nt = pd.read_parquet(C.F_NEWDATA,
                         columns=["state_id", "state_converged"] + C.X_TARGETS)
    nt = nt[nt.state_converged == 1].rename(columns=dict(zip(C.X_TARGETS, C.TRUE_COLS)))
    new = nf.merge(nt[["state_id"] + C.TRUE_COLS], on="state_id", validate="one_to_one")
    new = new[new[C.TRUE_COLS].notna().all(axis=1)].reset_index(drop=True)

    # every gate that could silently corrupt a production model, asserted here
    for name, f in (("PRIMARY train", tr), ("PRIMARY validation", va),
                    ("SECONDARY added", new)):
        assert not (f.family.astype(str) == "BASELINE").any(), "%s: BASELINE row" % name
        assert (f.sampling_domain.astype(str) == "A").all(), "%s: non-Domain-A row" % name
        assert not f.split_role.astype(str).str.contains("ANCHOR", case=False).any(), \
            "%s: ANCHOR row" % name
        assert not f.composition_id.astype(str).str.contains("ANCHOR", case=False).any()
        assert not f.split_role.astype(str).eq("IID_TEST").any(), "%s: IID_TEST row" % name
    assert set(tr.family) == set(C.ORIGINAL_SIX), sorted(set(tr.family))
    assert set(va.family) == set(C.ORIGINAL_SIX), sorted(set(va.family))
    assert set(new.family) == set(C.FULL5_ADDED), sorted(set(new.family))
    assert not set(tr.composition_id) & set(va.composition_id), "train/val composition overlap"
    assert not set(new.composition_id) & set(va.composition_id), "new/val composition overlap"
    assert tr.state_id.is_unique and va.state_id.is_unique and new.state_id.is_unique
    return tr, va, new


CARD = """# {title}

**Role:** {role}
**Model id:** `{mid}`
**Frozen (UTC):** {ts}
**Status after this document is written:** IMMUTABLE. No retraining, fine-tuning,
additional epochs, new seed, new scaler, new feature or new hyperparameter is
permitted for this model - and specifically none is permitted after any Ni elemental
descriptor has been accessed (spec 30).

## What it predicts

The five-family X-partition of the minor alloying element X in Cu-Cr-Nb-X at
equilibrium: `P_X^alpha = n_X^alpha / sum_beta n_X^beta`, with
`n_X^alpha = phase_moles_alpha * X_X^alpha`, over the frozen family order
{families}. The output is a 5-vector on the simplex. It is a thermodynamic
equilibrium quantity, not a mechanical or kinetic property, and no mechanical
property is derived from it.

## Training data

| | |
|---|---|
| identities seen in training | {train_ids} |
| training rows | {n_train} |
| training compositions | {n_train_comp} |
| validation identities | {val_ids} |
| validation rows | {n_val} |
| validation compositions | {n_val_comp} |
| temperatures | {temps} |
| domain | Domain A only; Domain B, ANCHOR and IID_TEST excluded |
| row weighting | natural (no element, identity or class balancing; spec 21) |
{extra_rows}

## Architecture and protocol (all inherited, none chosen here)

| | |
|---|---|
| architecture | Linear({nf}->128) - ReLU - Linear(128->64) - ReLU - Linear(64->5) |
| loss | mean squared error on the RAW 5 components |
| optimizer | Adam |
| batch size | {batch} |
| epoch cap | {epochs} |
| early stopping | patience {patience} on raw validation MSE |
| learning-rate grid | {lrs} |
| LR selection rule | lowest PROJECTED validation TVD |
| chosen learning rate | **{lr}** (best epoch {best_epoch} of {epochs_run}) |
| scaler | StandardScaler fitted on this model's training rows ONLY |
| projection | clip at 0, renormalize; zero-sum row -> this model's training-set mean simplex |
| seed | {seed} (derived as sha256("{master}|{role}") -> first 4 bytes) |
| protocol source | STEP5D preregistration, unchanged through STEP5F and STEP5G |

## Frozen simplex fallback row

`{fallback}`

## In-domain validation performance

These are IN-DOMAIN, seen-identity numbers on Domain-A DEV_VALIDATION. They are not
an out-of-distribution result and must never be quoted as one; the unseen-element
evidence for this pipeline is the STEP5C/STEP5E/STEP5F LOEO record, and the
prospective evidence does not exist yet.

| metric | value |
|---|---:|
| mean TVD | {mean_tvd:.6f} |
| median TVD | {median_tvd:.6f} |
| p95 TVD | {p95_tvd:.6f} |
| topology accuracy (argmax family) | {topology_accuracy:.4f} |
| topology macro F1 | {topology_macro_f1:.4f} |
| frozen trivial baseline TVD | {trivial_tvd:.6f} |
| skill vs trivial (positive = better) | {skill_vs_trivial:+.6f} |
| simplex fallbacks used | {simplex_fallbacks} |

## Known limits

- Unseen-element generalization is weak. The frozen retrospective leave-one-element-out
  record for this formulation is mean TVD {z1:.6f} (original six) and {f5:.6f} after
  the five-identity augmentation - both far above the in-domain numbers above, and the
  augmented result met none of its four preregistered switching gates.
<!-- DISCLAIMER_OPEN -->
- Held-out-element targets were excluded from the corresponding LOEO fold's model
  development, preprocessing, feature selection, hyperparameter optimization,
  threshold selection and model selection. That is a retrospective design; it is not
  a prospective blind validation.
<!-- DISCLAIMER_CLOSE -->
- The model has never seen Ni, and no Ni elemental descriptor, property, similarity
  score, prediction or CALPHAD state was read or produced at any point in its
  construction.

## Intended use

Zero-shot prediction of the X-partition for a chemically new X identity, scored
against CALPHAD equilibrium, under a preregistered protocol. It is a thermodynamic
screening model. It does not predict strength, hardness, ductility or conductivity,
and no such quantity may be inferred from its output.
"""


def write_card(path, **kw):
    with open(path, "w", encoding="utf-8") as f:
        f.write(CARD.format(**kw))


def main():
    import joblib
    import torch
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)

    # spec 10: refuse a diagnostic ablation subset as a production model, loudly.
    for bad in C.FORBIDDEN_PRODUCTION_MODELS:
        if os.environ.get("STEP5G_PRODUCTION_MODEL", "").upper() == bad:
            print("FATAL: %s is a STEP5F DIAGNOSTIC ablation configuration. Production "
                  "use of any STEP5F ablation subset is prohibited (spec 10)." % bad)
            return 2

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("STEP5G  Stage 2: production training and pipeline freeze")
    print("device: %s | torch %s" % (device, torch.__version__))
    print("=" * 78)

    feats = C.feature_schema()
    feat_sha = C.sha256_file(C.A_FEATLIST)
    if feat_sha != C.A_FEATLIST_SHA256:
        print("FATAL: the frozen 43-feature list no longer hashes to the value recorded "
              "in DEVELOPMENT_FREEZE_V1.json (%s != %s). The production schema is not "
              "the frozen schema; aborting before any model is written."
              % (feat_sha[:16], C.A_FEATLIST_SHA256[:16]))
        return 2

    if os.path.exists(os.path.join(C.D_FREEZE, "STEP5G_FINAL_PIPELINE_FREEZE.json")):
        raise RuntimeError("Production already frozen: refusing overwrite or retraining")
    tr, va, new = load_frames(feats)
    G.verify_master(tr); G.verify_master(va)
    source_hashes_before = G.source_hashes()
    print("PRIMARY   train %d rows / %d comps | validation %d rows / %d comps"
          % (len(tr), tr.composition_id.nunique(), len(va), va.composition_id.nunique()))
    print("SECONDARY train %d rows (+%d added-identity rows, %.1f%% of the set)"
          % (len(tr) + len(new), len(new), 100.0 * len(new) / (len(tr) + len(new))))

    G.prepare_schema(feats)

    jobs = [
        ("PRIMARY", C.PRIMARY_ROLE, tr, va, C.D_PRIMARY, "STEP5G_PRIMARY",
         "Original-six direct MLP (production PRIMARY)"),
        ("SECONDARY", C.SECONDARY_ROLE,
         pd.concat([tr, new], ignore_index=True), va, C.D_SECOND, "STEP5G_SECONDARY",
         "Full-eleven-identity direct MLP (production SECONDARY)"),
    ]
    freeze = {"freeze_id": "STEP5G_FINAL_PIPELINE_FREEZE_V1",
              "generated_utc": C.utcnow(),
              "primary_role": C.PRIMARY_ROLE, "secondary_role": C.SECONDARY_ROLE,
              "primary_selection_rule": (
                  "ORIGINAL6 remains PRIMARY because every preregistered STEP5F "
                  "switching gate failed; this is protocol-following, not preference."),
              "step5f_gates": C.FULL5_GATES,
              "forbidden_production_models": C.FORBIDDEN_PRODUCTION_MODELS,
              "feature_schema": {"n": len(feats), "sha256": feat_sha, "features": feats,
                                 "reselection_performed": False},
              "protocol": {"architecture": "Linear(n,128)-ReLU-Linear(128,64)-ReLU-"
                                           "Linear(64,5)",
                           "loss": "raw-component MSE", "optimizer": "Adam",
                           "batch_size": C.MLP_BATCH, "max_epochs": C.MLP_EPOCHS,
                           "patience": C.MLP_PATIENCE, "learning_rates": C.MLP_LRS,
                           "lr_selection": "lowest projected validation TVD",
                           "projection": "clip at 0, renormalize; zero-sum -> train mean "
                                         "simplex",
                           "row_weighting": "natural (spec 21)",
                           "master_seed": C.PRODUCTION_MASTER_SEED,
                           "source": "STEP5D preregistration, unchanged"},
              "library_versions": {"torch": torch.__version__,
                                   "numpy": np.__version__, "pandas": pd.__version__},
              "models": {}, "trails": []}

    for tag, role, trf, vaf, outdir, stem, title in jobs:
        print("\n%s (%s)" % (tag, role))
        seed = C.child_seed(C.PRODUCTION_MASTER_SEED, role)
        fallback = C.train_mean_simplex(trf[C.TRUE_COLS].to_numpy(float))
        art = train_frozen_mlp(role, trf, vaf, feats, fallback, seed, device)

        mpath = os.path.join(outdir, "%s_MODEL_CHECKPOINT.pt" % stem)
        spath = os.path.join(outdir, "%s_SCALER.joblib" % stem)
        torch.save({"state_dict": art["model"].state_dict(), "role": role,
                    "n_features": len(feats), "features": feats,
                    "family_order": C.FAMILIES_ORDER,
                    "fallback_row": fallback.tolist(), "seed": seed,
                    "chosen_lr": art["chosen_lr"],
                    "architecture": "Linear(%d,128)-ReLU-Linear(128,64)-ReLU-Linear(64,5)"
                                    % len(feats)}, mpath)
        joblib.dump(art["scaler"], spath)
        train_ids_path = os.path.join(outdir, stem + "_TRAIN_IDS.csv")
        val_ids_path = os.path.join(outdir, stem + "_VALIDATION_IDS.csv")
        train_ids_hash = G.record_ids(trf, train_ids_path)
        val_ids_hash = G.record_ids(vaf, val_ids_path)
        C.write_json(os.path.join(outdir, stem + "_SCALER_PARAMETERS.json"), {
            "mean_": art["scaler"].mean_, "scale_": art["scaler"].scale_,
            "var_": art["scaler"].var_, "feature_names_in_": feats,
            "feature_order": feats, "scaler_sha256": C.sha256_file(spath)})

        ids_tr = sorted(set(trf.family))
        extra = ""
        if tag == "SECONDARY":
            extra = ("| added identities | %s (STEP5F, %d rows = %.1f%% of training) |\n"
                     % (", ".join(sorted(C.FULL5_ADDED)), len(new),
                        100.0 * len(new) / len(trf)))
        write_card(os.path.join(outdir, "%s_MODEL_CARD.md" % stem),
                   title="%s_MODEL_CARD.md" % stem, role=role, mid=stem, ts=C.utcnow(),
                   families=", ".join(C.FAMILIES_ORDER),
                   train_ids=", ".join(ids_tr), n_train=len(trf),
                   n_train_comp=trf.composition_id.nunique(),
                   val_ids=", ".join(sorted(set(vaf.family))), n_val=len(vaf),
                   n_val_comp=vaf.composition_id.nunique(),
                   temps=", ".join(str(t) for t in C.TEMPERATURES_C) + " degC",
                   extra_rows=extra, nf=len(feats), batch=C.MLP_BATCH,
                   epochs=C.MLP_EPOCHS, patience=C.MLP_PATIENCE,
                   lrs=", ".join("%g" % x for x in C.MLP_LRS),
                   lr=art["chosen_lr"], best_epoch=art["selection"]["best_epoch"],
                   epochs_run=art["selection"]["epochs_run"], seed=seed,
                   master=C.PRODUCTION_MASTER_SEED,
                   fallback=", ".join("%.9f" % x for x in fallback),
                   z1=C.Z1_HISTORICAL_LOEO_MEAN_TVD, f5=C.FULL5_MEAN_TVD,
                   **art["validation_metrics"])

        card_path = os.path.join(outdir, stem + "_MODEL_CARD.md")
        with open(card_path, "a", encoding="utf-8") as card:
            card.write("\n## Artifact provenance\n\n")
            for label, value in {"model SHA256": C.sha256_file(mpath),
                "scaler SHA256": C.sha256_file(spath),
                "feature schema SHA256": C.sha256_file(os.path.join(C.D_FREEZE,"STEP5G_FEATURE_SCHEMA.csv")),
                "train-ID SHA256": train_ids_hash, "validation-ID SHA256": val_ids_hash,
                "software versions": G.runtime(), "Ni descriptor access at freeze": 0,
                "Ni target access": 0}.items(): card.write("- %s: `%s`\n" % (label,value))
            if tag == "SECONDARY": card.write("\n**SECONDARY ONLY: cannot replace PRIMARY based on Ni outcome.**\n")
        freeze["models"][role] = {
            "train_ids": train_ids_path, "train_ids_sha256": train_ids_hash,
            "validation_ids": val_ids_path, "validation_ids_sha256": val_ids_hash,
            "best_epoch": art["selection"]["best_epoch"],
            "epochs_trained": art["selection"]["epochs_run"],
            "tag": tag, "model_id": stem,
            "train_identities": ids_tr, "n_train_rows": int(len(trf)),
            "n_train_compositions": int(trf.composition_id.nunique()),
            "validation_identities": sorted(set(vaf.family)),
            "n_validation_rows": int(len(vaf)),
            "n_validation_compositions": int(vaf.composition_id.nunique()),
            "added_identity_rows": int(len(new)) if tag == "SECONDARY" else 0,
            "seed": seed, "chosen_learning_rate": art["chosen_lr"],
            "fallback_row": fallback.tolist(),
            "in_domain_validation_metrics": art["validation_metrics"],
            "checkpoint": mpath, "checkpoint_sha256": C.sha256_file(mpath),
            "scaler": spath, "scaler_sha256": C.sha256_file(spath),
            "immutable_from_utc": C.utcnow()}
        freeze["trails"].extend(art["lr_trail"])
        print("  chosen lr %g | in-domain validation mean TVD %.6f | topology %.4f"
              % (art["chosen_lr"], art["validation_metrics"]["mean_tvd"],
                 art["validation_metrics"]["topology_accuracy"]))

    pd.DataFrame(freeze["trails"]).to_csv(
        os.path.join(C.D_FREEZE, "STEP5G_LR_SELECTION_TRAIL.csv"), index=False,
        encoding="utf-8")
    assert source_hashes_before == G.source_hashes(), "Source data changed during training"
    freeze["source_hashes"] = source_hashes_before
    freeze["library_versions"] = G.runtime()
    freeze["feature_schema"]["csv_sha256"] = C.sha256_file(os.path.join(C.D_FREEZE,"STEP5G_FEATURE_SCHEMA.csv"))
    freeze["code_snapshot_hashes"] = {p:C.sha256_file(os.path.join(C.D_CODE,p)) for p in sorted(os.listdir(C.D_CODE)) if p.endswith(".py")}
    freeze["recovery_exception"] = "Sections 68-69 authorize separate post-blind recovery refits only; frozen zero-shot artifacts and roles remain immutable."
    C.write_json(os.path.join(C.D_FREEZE, "STEP5G_FINAL_PIPELINE_FREEZE.json"), freeze)

    with open(os.path.join(C.D_FREEZE, "STEP5G_FINAL_PIPELINE_FREEZE.md"), "w",
              encoding="utf-8") as f:
        f.write(freeze_md(freeze, feats))
    print("\nSTAGE 2: PASS - both production models are frozen and hashed.")
    return 0


def freeze_md(fz, feats):
    p = fz["models"][C.PRIMARY_ROLE]
    s = fz["models"][C.SECONDARY_ROLE]
    g = fz["step5f_gates"]
    return """# STEP5G_FINAL_PIPELINE_FREEZE.md

**Freeze id:** `{fid}`
**Frozen (UTC):** {ts}

From this document forward the production pipeline is closed. Retraining,
fine-tuning, further epochs, a new seed, a new scaler, a new feature or a new
hyperparameter is not permitted - and is specifically not permitted after any Ni
elemental descriptor has been accessed (spec 30).

## The decision

| role | model | why |
|---|---|---|
| **PRIMARY** | `{p_id}` ({p_role}) | Every preregistered STEP5F switching gate failed, so the preregistered rule keeps the original-six formulation. |
| **SECONDARY** | `{s_id}` ({s_role}) | The five-identity augmentation reduced mean LOEO TVD by 5.6% - real, but short of every gate. Carried as a declared secondary, not promoted. |

### The gates, as preregistered, with what actually happened

| gate | required | observed | result |
|---|---|---|---|
| primary TVD | <= {g1r} | {g1o} | **{g1res}** |
| topology accuracy | >= {g2r} | {g2o} | **{g2res}** |
| cross-element skill | >= {g3r} of 6 folds beat trivial | {g3o} of 6 | **{g3res}** |
| strong improvement | {g4r} | - | **{g4res}** |

Production use of `BASE_PLUS_MN`, `FULL5_MINUS_BE`, `FULL5_MINUS_LA`,
`FULL5_MINUS_SN`, `FULL5_MINUS_P` or any other STEP5F ablation subset is prohibited
(spec 10). BASE_PLUS_MN reaching 0.6917 is a diagnostic observation about which
identity carries the effect; selecting it as the production model after seeing that
number would be exactly the post-hoc selection the preregistration exists to prevent.

## Frozen feature schema

{nf} features, taken unchanged from STEP5A `SELECTED_FEATURES_F1_F2_F3_F4.txt`
(sha256 `{fsha}`). No NZV filtering, PCC filtering, feature selection, SHAP-based
selection, Optuna feature search or fold-specific selection was performed in STEP5G
(spec 14). Every one of the {nf} features was verified present in both the
original-six table and the new-element table before training began.

## Frozen protocol

| | |
|---|---|
| architecture | {arch} |
| loss | raw-component MSE |
| optimizer | Adam |
| batch size | {batch} |
| epoch cap / patience | {epochs} / {patience} (on raw validation MSE) |
| learning-rate grid | {lrs} |
| LR selection | lowest projected validation TVD |
| projection | clip at 0, renormalize; zero-sum row -> that model's training mean simplex |
| row weighting | natural; no element, identity or class balancing (spec 21) |
| master seed | {seed} |
| protocol origin | STEP5D preregistration, carried unchanged through STEP5F |

## Frozen artifacts

| role | rows (train / validation) | lr | checkpoint sha256 | scaler sha256 |
|---|---|---:|---|---|
| PRIMARY | {p_tr} / {p_va} | {p_lr} | `{p_csha}` | `{p_ssha}` |
| SECONDARY | {s_tr} / {s_va} | {s_lr} | `{s_csha}` | `{s_ssha}` |

## In-domain validation (seen identities - NOT an OOD result)

| metric | PRIMARY | SECONDARY |
|---|---:|---:|
| mean TVD | {p_tvd:.6f} | {s_tvd:.6f} |
| median TVD | {p_med:.6f} | {s_med:.6f} |
| p95 TVD | {p_p95:.6f} | {s_p95:.6f} |
| topology accuracy | {p_acc:.4f} | {s_acc:.4f} |
| topology macro F1 | {p_f1:.4f} | {s_f1:.4f} |
| trivial baseline TVD | {p_triv:.6f} | {s_triv:.6f} |

These numbers describe interpolation among identities the models have seen. The
unseen-element behaviour of this formulation is the LOEO record (mean TVD 0.752644
original-six, 0.710332 augmented), and no prospective evidence exists yet.

## Ni status at the moment of freezing

Ni elemental descriptor access: 0. Ni property access: 0. Ni similarity computation:
0. Ni prediction: 0. Ni CALPHAD / TC-Python state: 0. Ni phase, partition or target
access: 0. Neither model has seen nickel in any form.
""".format(
        fid=fz["freeze_id"], ts=fz["generated_utc"],
        p_id=p["model_id"], p_role=C.PRIMARY_ROLE,
        s_id=s["model_id"], s_role=C.SECONDARY_ROLE,
        g1r=g["primary_tvd"]["required_max"], g1o=g["primary_tvd"]["observed"],
        g1res=g["primary_tvd"]["result"],
        g2r=g["topology"]["required_min"], g2o=g["topology"]["observed"],
        g2res=g["topology"]["result"],
        g3r=g["cross_element_skill"]["required_min_folds"],
        g3o=g["cross_element_skill"]["observed_folds"],
        g3res=g["cross_element_skill"]["result"],
        g4r=g["strong"]["required"], g4res=g["strong"]["result"],
        nf=len(feats), fsha=fz["feature_schema"]["sha256"],
        arch=fz["protocol"]["architecture"], batch=C.MLP_BATCH, epochs=C.MLP_EPOCHS,
        patience=C.MLP_PATIENCE, lrs=", ".join("%g" % x for x in C.MLP_LRS),
        seed=C.PRODUCTION_MASTER_SEED,
        p_tr=p["n_train_rows"], p_va=p["n_validation_rows"],
        p_lr=p["chosen_learning_rate"], p_csha=p["checkpoint_sha256"],
        p_ssha=p["scaler_sha256"],
        s_tr=s["n_train_rows"], s_va=s["n_validation_rows"],
        s_lr=s["chosen_learning_rate"], s_csha=s["checkpoint_sha256"],
        s_ssha=s["scaler_sha256"],
        p_tvd=p["in_domain_validation_metrics"]["mean_tvd"],
        s_tvd=s["in_domain_validation_metrics"]["mean_tvd"],
        p_med=p["in_domain_validation_metrics"]["median_tvd"],
        s_med=s["in_domain_validation_metrics"]["median_tvd"],
        p_p95=p["in_domain_validation_metrics"]["p95_tvd"],
        s_p95=s["in_domain_validation_metrics"]["p95_tvd"],
        p_acc=p["in_domain_validation_metrics"]["topology_accuracy"],
        s_acc=s["in_domain_validation_metrics"]["topology_accuracy"],
        p_f1=p["in_domain_validation_metrics"]["topology_macro_f1"],
        s_f1=s["in_domain_validation_metrics"]["topology_macro_f1"],
        p_triv=p["in_domain_validation_metrics"]["trivial_tvd"],
        s_triv=s["in_domain_validation_metrics"]["trivial_tvd"])


if __name__ == "__main__":
    sys.exit(main())
