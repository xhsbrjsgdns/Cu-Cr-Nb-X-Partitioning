# -*- coding: utf-8 -*-
"""
111_i_refit.py
STEP5I Stage 1: PRIMARY_FORMULATION_REFIT (spec 10-19, 29).

24 independent refits: 3 random replicates x 6 budgets, plus the 6 maximin budgets
that serve as the SECONDARY comparison only (spec 28 - maximin is never promoted on
the basis of its result).

Each refit is the frozen PRIMARY formulation trained from scratch on
    the exact frozen PRIMARY original-six training rows  +  the anchor Ni rows
with:
    frozen PRIMARY scaler, never refit and never adapted to Ni   (spec 15)
    fixed learning rate 0.0003, never retuned                    (spec 13)
    Adam, raw-component MSE, batch 1024, <=200 epochs, patience 15
    early stopping on the frozen original-six DEV_VALIDATION rows ONLY, so the
        stopping rule carries no Ni signal whatsoever              (spec 14)
    a fresh model from the frozen PRIMARY production seed for every cell; no warm
        start from a smaller budget and no weight carried between replicates (spec 16)
    natural row weighting - no Ni oversampling, reweighting, topology balancing or
        temperature balancing                                      (spec 18)

Under natural weighting the anchors are between 0.005% (11 rows) and 0.145% (352 rows)
of the training set. That is a consequence of a protocol frozen before the reveal, and
this stage reports it rather than repairing it: the anchor fraction is written into
every cell's record so the learning curve can never be read without it.

RECOVERY_EVALUATION labels are never used for fitting, validation, early stopping or
model selection. The only thing they are used for is scoring, in the next stage.

Nothing frozen is overwritten: every artifact this stage writes is new and lives under
the STEP5I root.
"""
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i_common as C

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
    """Frozen PRIMARY architecture (spec 11). No architecture search."""
    from torch import nn
    return nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                         nn.Linear(64, 5))


def load_original_rows(feats):
    """The exact frozen PRIMARY train and validation rows, selected by the STEP5G
    ID files rather than by re-deriving a split."""
    tr_ids = pd.read_csv(C.G_PRIMARY_TRAIN_IDS)
    va_ids = pd.read_csv(C.G_PRIMARY_VAL_IDS)
    cols = META + feats + C.DIAG_TRUE_COLS
    d = pd.read_parquet(C.D_DIAG_DATA, columns=cols)
    tr = d[d.state_id.isin(set(tr_ids.state_id))].reset_index(drop=True)
    va = d[d.state_id.isin(set(va_ids.state_id))].reset_index(drop=True)
    if len(tr) != len(tr_ids) or len(va) != len(va_ids):
        raise SystemExit("FATAL: frozen PRIMARY id files select %d/%d rows, expected "
                         "%d/%d." % (len(tr), len(va), len(tr_ids), len(va_ids)))
    if set(tr.state_id) & set(va.state_id):
        raise SystemExit("FATAL: frozen train and validation rows overlap.")
    return tr, va


def train_cell(x_tr, y_tr, x_va, y_va, device):
    """Frozen PRIMARY formulation, one fixed learning rate, fresh initialization."""
    import torch
    from torch import nn
    from threadpoolctl import threadpool_limits

    xt = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_va, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_va, dtype=torch.float32, device=device)

    with threadpool_limits(limits=4):
        seed_all(C.REFIT_SEED)
        model = network(xt.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=C.REFIT_LR)
        best_mse, best_epoch, best_state, wait = np.inf, 0, None, 0
        history = []
        t0 = time.time()
        for epoch in range(1, C.REFIT_EPOCHS + 1):
            model.train()
            for idx in torch.randperm(len(xt), device=device).split(C.REFIT_BATCH):
                opt.zero_grad(set_to_none=True)
                nn.functional.mse_loss(model(xt[idx]), yt[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vmse = float(nn.functional.mse_loss(model(xv), yv).item())
            history.append({"epoch": epoch, "validation_mse": vmse})
            if vmse < best_mse - 1e-8:
                best_mse, best_epoch, wait = vmse, epoch, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                wait += 1
            if wait >= C.REFIT_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "state_dict": best_state, "best_epoch": best_epoch,
            "epochs_run": epoch, "best_validation_mse": best_mse,
            "seconds": round(time.time() - t0, 1), "history": history,
            "early_stopped": wait >= C.REFIT_PATIENCE}


def main():
    import joblib
    import torch
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("STEP5I  Stage 1: PRIMARY_FORMULATION_REFIT  (post-blind recovery)")
    print("device: %s | torch %s" % (device, torch.__version__))
    print("=" * 78)

    pre = C.fewshot_prereg()
    C.assert_protocol_matches_prereg(pre)
    for key, path in (("HB_TARGETS", C.HB_TARGETS),
                      ("HA_PRIMARY_PREDICTIONS", C.HA_PRED),
                      ("HA_MAXIMIN_SEQUENCE", C.HA_MAXIMIN_CSV),
                      ("G_RANDOM_SEQUENCES", C.G_SEQ_CSV),
                      ("G_PRIMARY_SCALER", C.G_PRIMARY_SCALER)):
        live = C.sha256_file(path)
        if live != C.FROZEN_HASHES[key]:
            print("FATAL: %s changed (%s != %s). Refusing to refit against a moved "
                  "frozen input." % (key, live[:16], C.FROZEN_HASHES[key][:16]))
            return 2

    gc = C.load_g_common()          # frozen projection, loaded once
    ni, feats = C.load_frozen_ni()
    tr_o, va_o = load_original_rows(feats)
    scaler = joblib.load(C.G_PRIMARY_SCALER)
    fallback = C.frozen_fallback_row()
    if list(getattr(scaler, "feature_names_in_", feats)) != list(feats):
        raise SystemExit("FATAL: frozen scaler feature order differs from the Ni "
                         "feature table order.")

    # transform ONCE with the frozen scaler; it is never refit (spec 15)
    x_tr_o = scaler.transform(tr_o[feats])
    x_va = scaler.transform(va_o[feats])
    y_tr_o = tr_o[C.DIAG_TRUE_COLS].to_numpy(float)
    y_va = va_o[C.DIAG_TRUE_COLS].to_numpy(float)
    ni_x_all = scaler.transform(ni[feats])
    ni_y_all = ni[C.TRUE_COLS].to_numpy(float)
    ni_index = {sid: i for i, sid in enumerate(ni.state_id)}

    ev = ni[ni.fewshot_role == "RECOVERY_EVALUATION"].reset_index(drop=True)
    if len(ev) != C.N_RECOVERY_STATES:
        raise SystemExit("FATAL: recovery evaluation has %d states, expected %d."
                         % (len(ev), C.N_RECOVERY_STATES))
    ev_rows = np.array([ni_index[s] for s in ev.state_id])
    x_ev = ni_x_all[ev_rows]
    ev_ids = set(ev.composition_id)

    print("original-six train %d rows | validation %d rows | Ni states %d "
          "(recovery evaluation %d)" % (len(tr_o), len(va_o), len(ni), len(ev)))
    print("frozen scaler: %s (never refit)" % os.path.basename(C.G_PRIMARY_SCALER))

    seqs = C.random_sequences()
    mx = C.maximin_sequence()
    cells = [("RANDOM", r, b, seqs[r]) for r in C.REPLICATES for b in C.BUDGETS]
    cells += [("MAXIMIN", 0, b, mx) for b in C.BUDGETS]

    manifest, coverage = [], []
    for kind, rep, budget, seq in cells:
        tag = C.cell_tag(kind, rep, budget)
        anc_ids = C.anchors(seq, budget)
        assert not (set(anc_ids) & ev_ids), "anchor leaked into recovery evaluation"
        anc = ni[ni.composition_id.isin(anc_ids)]
        assert len(anc) == budget * len(C.TEMPERATURES_C), \
            "%s: %d anchor states, expected %d" % (tag, len(anc),
                                                   budget * len(C.TEMPERATURES_C))
        anc_rows = np.array([ni_index[s] for s in anc.state_id])

        x_tr = np.vstack([x_tr_o, ni_x_all[anc_rows]])
        y_tr = np.vstack([y_tr_o, ni_y_all[anc_rows]])
        frac = len(anc_rows) / len(x_tr)

        art = train_cell(x_tr, y_tr, x_va, y_va, device)
        with torch.no_grad():
            raw = art["model"](torch.as_tensor(x_ev, dtype=torch.float32,
                                               device=device)).cpu().numpy()
        proj, n_fb = gc.project_simplex_counted(raw, fallback)

        outdir = C.D_RANDOM if kind == "RANDOM" else C.D_MAXIMIN
        pred = ev[["state_id", "composition_id", "Temperature_C"]].copy()
        for j, f in enumerate(C.FAMILIES_ORDER):
            pred["refit_raw_%s" % f] = raw[:, j]
            pred["refit_pred_%s" % f] = proj[:, j]
        pred["refit_topology"] = [C.FAMILIES_ORDER[k] for k in proj.argmax(axis=1)]
        ppath = os.path.join(outdir, "STEP5I_%s_PREDICTIONS.parquet" % tag)
        pred.to_parquet(ppath, index=False)

        mpath = os.path.join(outdir, "STEP5I_%s_MODEL.pt" % tag)
        torch.save({"state_dict": art["state_dict"], "tag": tag, "kind": kind,
                    "replicate": rep, "budget": budget, "anchor_ids": anc_ids,
                    "features": feats, "n_features": len(feats),
                    "family_order": C.FAMILIES_ORDER, "fallback_row": fallback.tolist(),
                    "seed": C.REFIT_SEED, "learning_rate": C.REFIT_LR,
                    "scaler_source": C.G_PRIMARY_SCALER,
                    "scaler_sha256": C.FROZEN_HASHES["G_PRIMARY_SCALER"],
                    "warm_started": False, "scaler_refit": False,
                    "architecture": "Linear(%d,128)-ReLU-Linear(128,64)-ReLU-"
                                    "Linear(64,5)" % len(feats),
                    "note": "POST-BLIND RECOVERY CHECKPOINT - not a zero-shot model"},
                   mpath)
        pd.DataFrame(art["history"]).assign(tag=tag).to_csv(
            os.path.join(outdir, "STEP5I_%s_VALIDATION_HISTORY.csv" % tag),
            index=False, encoding="utf-8")

        # spec 40: descriptive record of what the frozen anchors happened to contain
        coverage.append({
            "kind": kind, "replicate": rep, "budget": budget, "tag": tag,
            "n_anchor_compositions": budget,
            "n_anchor_states": int(len(anc)),
            "anchor_fraction_of_training_rows": float(frac),
            "n_anchor_comps_with_any_NBNI3": int(
                anc.groupby("composition_id").nbni3_present.max().sum()),
            "n_anchor_states_NBNI3_present": int(anc.nbni3_present.sum()),
            "fraction_anchor_states_NBNI3": float(anc.nbni3_present.mean()),
            "n_anchor_states_true_FCC": int((anc.true_topology == "FCC").sum()),
            "n_anchor_states_true_OTHER_SOLID": int(
                (anc.true_topology == "OTHER_SOLID").sum()),
            "fraction_anchor_states_true_OTHER_SOLID": float(
                (anc.true_topology == "OTHER_SOLID").mean()),
            "anchor_ids": ";".join(anc_ids)})

        manifest.append({
            "tag": tag, "kind": kind, "replicate": rep, "budget": budget,
            "n_anchor_compositions": budget, "n_anchor_states": int(len(anc)),
            "n_train_rows": int(len(x_tr)), "n_original_rows": int(len(x_tr_o)),
            "anchor_fraction_of_training_rows": float(frac),
            "n_validation_rows": int(len(x_va)),
            "validation_source": "frozen original-six DEV_VALIDATION (no Ni)",
            "learning_rate": C.REFIT_LR, "seed": C.REFIT_SEED,
            "best_epoch": art["best_epoch"], "epochs_run": art["epochs_run"],
            "best_validation_mse": art["best_validation_mse"],
            "early_stopped": art["early_stopped"],
            "train_seconds": art["seconds"],
            "simplex_fallbacks_on_recovery": int(n_fb),
            "scaler_refit": False, "warm_started": False,
            "predictions": ppath, "predictions_sha256": C.sha256_file(ppath),
            "checkpoint": mpath, "checkpoint_sha256": C.sha256_file(mpath)})

        print("  %-16s anchors %2d comps / %3d states (%.4f%% of train) | best epoch "
              "%3d/%3d | val MSE %.3e | %.0f s"
              % (tag, budget, len(anc), 100 * frac, art["best_epoch"],
                 art["epochs_run"], art["best_validation_mse"], art["seconds"]),
              flush=True)

    pd.DataFrame(manifest).to_csv(
        os.path.join(C.D_MANIFEST, "STEP5I_RECOVERY_CHECKPOINT_MANIFEST.csv"),
        index=False, encoding="utf-8")
    pd.DataFrame(coverage).to_csv(
        os.path.join(C.D_TOPO, "STEP5I_ANCHOR_TOPOLOGY_COVERAGE.csv"),
        index=False, encoding="utf-8")
    C.write_json(os.path.join(C.D_CONFIG, "STEP5I_REFIT_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "stage": "PRIMARY_FORMULATION_REFIT (post-blind retrospective recovery)",
        "n_cells": len(manifest),
        "n_random_cells": len(C.REPLICATES) * len(C.BUDGETS),
        "n_maximin_cells": len(C.BUDGETS),
        "protocol": {"learning_rate": C.REFIT_LR, "batch": C.REFIT_BATCH,
                     "max_epochs": C.REFIT_EPOCHS, "patience": C.REFIT_PATIENCE,
                     "seed": C.REFIT_SEED, "retuning": False,
                     "scaler": "frozen PRIMARY, never refit (spec 15)",
                     "validation": "frozen original-six DEV_VALIDATION only (spec 14)",
                     "weighting": "natural; no Ni oversampling (spec 18)",
                     "warm_start": False, "architecture_search": False,
                     "feature_selection": False},
        "anchor_fraction_range": [float(min(m["anchor_fraction_of_training_rows"]
                                            for m in manifest)),
                                  float(max(m["anchor_fraction_of_training_rows"]
                                            for m in manifest))],
        "total_train_seconds": float(sum(m["train_seconds"] for m in manifest)),
        "library_versions": {"torch": torch.__version__, "numpy": np.__version__,
                             "pandas": pd.__version__},
        "cells": manifest})
    print("\nSTAGE 1: PASS - %d recovery checkpoints written, none overwriting a "
          "frozen artifact." % len(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
