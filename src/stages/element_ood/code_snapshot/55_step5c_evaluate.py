# -*- coding: utf-8 -*-
"""
55_step5c_evaluate.py
STEP5C Stage J + K + L: score the six HELDOUT_ELEMENT_TEST sets, bootstrap, topology.

This is the FIRST module that reads a held-out target. It refuses to start unless the
Stage-I gate authorises scoring, i.e. unless all six folds were already frozen.

Per fold:
  05_predictions/FOLD_<X>_HELDOUT_PREDICTIONS.parquet     (spec 63)
  11_manifest/FOLD_<X>_HELDOUT_EVALUATION_RECORD.json     (spec 42; NEVER *_FIRST_OPEN)
Aggregate:
  06_metrics/LOEO_ALL_FOLDS_METRICS.csv                   (spec 64)
  06_metrics/LOEO_PER_TARGET_METRICS.csv                  (spec 65)
  06_metrics/LOEO_TOPOLOGY_METRICS.csv                    (spec 66)
  06_metrics/LOEO_ERROR_BY_TEMPERATURE.csv                (spec 50)
  06_metrics/LOEO_ERROR_BY_PHASE_SPLIT.csv                (spec 59, no directional prior)
  06_metrics/LOEO_TARGET_DISTRIBUTION_SHIFT.csv           (spec 58, post-hoc only)

MO failure policy (spec 45): FOLD_MO's Domain-A MO@950 CALPHAD failures are not
interpolated, not imputed and not deleted. Formal / converged / failed row counts are
reported and metrics are computed on converged rows only.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5c_common as C

TEMP_GRID = [300., 350., 400., 450., 500., 600., 700., 800., 900., 950., 1000.]
assert len(TEMP_GRID) == 11


def predict_with(bundle, X):
    """Re-run the stored per-family models. Mirrors the training predictor exactly."""
    P = np.zeros((len(X), len(C.FAMILIES_ORDER)))
    for i, m in enumerate(bundle["models"]):
        if isinstance(m, tuple) and m[0] == "constant":
            P[:, i] = m[1]
        elif bundle["task"] == "presence":
            P[:, i] = m.predict_proba(X)[:, 1]
        else:
            P[:, i] = m.predict(X)
    return P


def cluster_bootstrap(comp_ids, fn, reps, seed):
    """Composition-level cluster bootstrap (spec 62). fn(mask_index_array) -> float."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(comp_ids)
    idx_by_comp = {c: np.flatnonzero(comp_ids == c) for c in uniq}
    vals = []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_comp[c] for c in pick])
        try:
            v = fn(idx)
        except Exception:
            v = np.nan
        vals.append(v)
    vals = np.asarray(vals, dtype=np.float64)
    good = vals[np.isfinite(vals)]
    if not len(good):
        return None, None, vals
    return float(np.percentile(good, 2.5)), float(np.percentile(good, 97.5)), vals


def evaluate_fold(heldout, df, man, formal_counts):
    fold = f"FOLD_{heldout}"
    t0 = time.time()
    freeze = json.load(open(os.path.join(C.D_MANIFEST, f"{fold}_FREEZE.json"),
                            encoding="utf-8"))
    feats = freeze["selected_features"]
    seed = freeze["random_seed"]
    roles = man[man.fold == fold].set_index("composition_id")["loeo_role"]
    d = df.copy()
    d["loeo_role"] = d["composition_id"].map(roles)
    test = d[d.loeo_role == "HELDOUT_ELEMENT_TEST"].reset_index(drop=True)
    assert set(test["family"]) == {heldout}, "held-out test contains a foreign family"
    assert set(test["sampling_domain"]) == {"A"}, "non Domain-A row in the held-out test"

    n_comp = int(test["composition_id"].nunique())
    formal_rows = n_comp * len(TEMP_GRID)
    converged_rows = int(len(test))
    failed_rows = formal_rows - converged_rows
    print("  %s: %d compositions | formal rows %d | converged %d | FAILED %d (not imputed)"
          % (fold, n_comp, formal_rows, converged_rows, failed_rows))

    out_cols = {}
    metrics_rows, per_target_rows, topo = [], [], None
    boot_rows = []
    pred_frame = test[["state_id", "composition_id", "family", "Temperature_C",
                       "has_multiple_FCC_instances", "has_multiple_BCC_instances"]].copy()
    pred_frame["heldout_element"] = heldout

    for task in C.ALL_TASKS:
        sub = C.task_rows(test, task)
        if not len(sub):
            continue
        bundle = pickle.load(open(os.path.join(
            C.D_MODELS, fold, task, "model_state.pkl"), "rb"))
        assert bundle["features"] == feats, "feature drift between freeze and model"
        X = sub[feats].to_numpy(np.float64)
        Y = sub[C.TARGETS[task]].to_numpy(np.float64)
        raw = predict_with(bundle, X)
        rec = freeze["tasks"][task]

        if task == "presence":
            ths = np.asarray(rec["thresholds"], dtype=np.float64)
            m = C.classification_metrics(Y.astype(int), raw, ths)
            triv = np.asarray(rec["trivial_baseline"], dtype=np.float64)
            triv_prob = np.tile(triv, (len(Y), 1))
            triv_m = C.classification_metrics(Y.astype(int), triv_prob,
                                              np.full(len(C.FAMILIES_ORDER), 0.5))
            headline, triv_headline = m["macro_f1"], triv_m["macro_f1"]
            proj = (raw >= ths).astype(int)
            for i, fam in enumerate(C.FAMILIES_ORDER):
                pl = m["per_label"][fam]
                per_target_rows.append({
                    "heldout_element": heldout, "task": task, "target": fam,
                    "support": pl["support"], "n": pl["n"], "f1": pl["f1"],
                    "precision": pl["precision"], "recall": pl["recall"],
                    "pr_auc": pl["pr_auc"], "roc_auc": pl["roc_auc"],
                    "degenerate_test_label": pl["degenerate_test_label"],
                    "threshold": float(ths[i])})
            extra = {"macro_f1_non_degenerate": m["macro_f1_non_degenerate"],
                     "n_non_degenerate_labels": m["n_non_degenerate_labels"],
                     "degenerate_test_labels": ";".join(m["degenerate_test_labels"]),
                     "exact_match": m["exact_match"], "hamming_loss": m["hamming_loss"],
                     "micro_f1": m["micro_f1"]}

            def score_fn(idx, Y=Y, raw=raw, ths=ths):
                from sklearn.metrics import f1_score
                yp = (raw[idx] >= ths).astype(int)
                return float(f1_score(Y[idx].astype(int), yp, average="macro",
                                      zero_division=0))
        else:
            fb = np.asarray(rec["simplex_fallback_row"], dtype=np.float64)
            proj, nfb = C.project_simplex(raw, fb)
            m = C.regression_metrics(Y, raw, proj, nfb)
            triv = np.asarray(rec["trivial_baseline"], dtype=np.float64)
            triv_pred = np.tile(triv, (len(Y), 1))
            triv_m = C.regression_metrics(Y, triv_pred, triv_pred, 0)
            headline, triv_headline = m["mean_tvd"], triv_m["mean_tvd"]
            for fam in C.FAMILIES_ORDER:
                pf = m["per_family"][fam]
                per_target_rows.append({
                    "heldout_element": heldout, "task": task, "target": fam,
                    "mae": pf["mae"], "rmse": pf["rmse"], "r2": pf["r2"],
                    "true_mean": pf["true_mean"], "true_std": pf["true_std"]})
            extra = {"median_tvd": m["median_tvd"], "p95_tvd": m["p95_tvd"],
                     "macro_mae": m["macro_mae"], "macro_rmse": m["macro_rmse"],
                     "raw_sum_violation_mean": m["raw_sum_violation_mean"],
                     "raw_sum_violation_max": m["raw_sum_violation_max"],
                     "projected_sum_violation_max": m["projected_sum_violation_max"],
                     "simplex_fallbacks": m["simplex_fallbacks"]}

            def score_fn(idx, Y=Y, proj=proj):
                return float(C.tvd(proj[idx], Y[idx]).mean())

            if task == "x_partition":
                topo = C.topology_metrics(Y, proj)

        lo, hi, _ = cluster_bootstrap(sub["composition_id"].to_numpy(), score_fn,
                                      C.BOOTSTRAP_REPS, seed)
        skill = ((headline - triv_headline) if task == "presence"
                 else (triv_headline - headline))
        row = {"heldout_element": heldout, "task": task,
               "model": rec["model_family"],
               "metric": "macro_f1" if task == "presence" else "mean_tvd",
               "value": headline, "bootstrap_low": lo, "bootstrap_high": hi,
               "trivial_value": triv_headline, "skill_vs_trivial": skill,
               "n_train_comp": rec["n_train_compositions"],
               "n_val_comp": rec["n_validation_compositions"],
               "n_test_comp": int(sub["composition_id"].nunique()),
               "n_test_rows": int(len(sub)),
               "n_formal_test_rows": formal_rows,
               "n_failed_test_rows": failed_rows,
               "n_selected_features": len(feats)}
        row.update(extra)
        metrics_rows.append(row)
        boot_rows.append({"heldout_element": heldout, "task": task,
                          "metric": row["metric"], "value": headline,
                          "ci_low": lo, "ci_high": hi, "reps": C.BOOTSTRAP_REPS,
                          "seed": seed})
        print("    %-13s %-9s %s=%.6f  CI[%.6f, %.6f]  trivial %.6f"
              % (task, rec["model_family"], row["metric"], headline,
                 lo if lo is not None else float("nan"),
                 hi if hi is not None else float("nan"), triv_headline), flush=True)

        # prediction columns
        idx = sub.index
        for j, fam in enumerate(C.FAMILIES_ORDER):
            out_cols.setdefault(f"true_{task}_{fam}", pd.Series(np.nan, index=test.index))
            out_cols.setdefault(f"pred_raw_{task}_{fam}", pd.Series(np.nan, index=test.index))
            out_cols.setdefault(f"pred_proj_{task}_{fam}", pd.Series(np.nan, index=test.index))
            out_cols[f"true_{task}_{fam}"].loc[idx] = Y[:, j]
            out_cols[f"pred_raw_{task}_{fam}"].loc[idx] = raw[:, j]
            out_cols[f"pred_proj_{task}_{fam}"].loc[idx] = proj[:, j]
        if task == "x_partition":
            tcol = pd.Series(pd.NA, index=test.index, dtype="object")
            pcol = pd.Series(pd.NA, index=test.index, dtype="object")
            tcol.loc[idx] = [C.FAMILIES_ORDER[k] for k in Y.argmax(axis=1)]
            pcol.loc[idx] = [C.FAMILIES_ORDER[k] for k in proj.argmax(axis=1)]
            out_cols["topology_true"] = tcol
            out_cols["topology_pred"] = pcol

    for k, v in out_cols.items():
        pred_frame[k] = v.values
    ppath = os.path.join(C.D_PRED, f"{fold}_HELDOUT_PREDICTIONS.parquet")
    pred_frame.to_parquet(ppath, index=False)

    record = {
        "heldout_element": heldout,
        "fold": fold,
        "fold_freeze_hash": freeze["freeze_sha256"],
        "evaluation_timestamp": C.utcnow(),
        "test_compositions": n_comp,
        "formal_test_rows": formal_rows,
        "converged_test_rows": converged_rows,
        "failed_test_rows": failed_rows,
        "failure_policy": ("CALPHAD-failed states are not interpolated, not imputed and "
                           "not deleted; metrics are computed on converged rows only "
                           "(spec 45)"),
        "feature_hash": freeze["selected_features_sha256"],
        "model_hashes": {t: v["model_artifact_sha256"] for t, v in freeze["tasks"].items()},
        "prediction_file": os.path.relpath(ppath, C.STEP5C).replace("\\", "/"),
        "prediction_sha256": C.sha256_file(ppath),
        "statement": ("held-out targets excluded from this fold's development "
                      "(feature selection, PCC, NZV, scaler, tuning, thresholds, model "
                      "selection and trivial baseline)"),
        "terminology_note": ("retrospective leave-one-element-out evaluation; this is NOT "
                             "a claim that these data were never historically accessed, "
                             "and it is NOT a prospective blind validation (spec 0, 41)"),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    C.write_json(os.path.join(C.D_MANIFEST, f"{fold}_HELDOUT_EVALUATION_RECORD.json"),
                 record)
    return metrics_rows, per_target_rows, topo, boot_rows, pred_frame, record


def stratified_tables(preds):
    """spec 50 (temperature), 59 (phase split, no directional prior), 58 (target shift)."""
    temp_rows, split_rows = [], []
    for heldout, p in preds.items():
        p = p.copy()
        p["is_split"] = ((p.has_multiple_FCC_instances == 1)
                         | (p.has_multiple_BCC_instances == 1)).astype(int)
        for task in C.REGRESSION_TASKS:
            tc = [f"true_{task}_{f}" for f in C.FAMILIES_ORDER]
            pc = [f"pred_proj_{task}_{f}" for f in C.FAMILIES_ORDER]
            if tc[0] not in p.columns:
                continue
            ok = p[tc[0]].notna()
            q = p[ok]
            t = C.tvd(q[pc].to_numpy(np.float64), q[tc].to_numpy(np.float64))
            g = pd.DataFrame({"Temperature_C": q["Temperature_C"].values, "tvd": t,
                              "is_split": q["is_split"].values})
            agg = g.groupby("Temperature_C")["tvd"].agg(["mean", "median", "count"])
            for T, r in agg.iterrows():
                temp_rows.append({"heldout_element": heldout, "task": task,
                                  "Temperature_C": float(T), "mean_tvd": float(r["mean"]),
                                  "median_tvd": float(r["median"]), "n_rows": int(r["count"])})
            for s, r in g.groupby("is_split")["tvd"].agg(["mean", "median", "count"]).iterrows():
                split_rows.append({"heldout_element": heldout, "task": task,
                                   "stratum": "split" if s == 1 else "non_split",
                                   "mean_tvd": float(r["mean"]),
                                   "median_tvd": float(r["median"]),
                                   "n_rows": int(r["count"])})
        if "true_presence_FCC" in p.columns:
            tc = [f"true_presence_{f}" for f in C.FAMILIES_ORDER]
            pc = [f"pred_proj_presence_{f}" for f in C.FAMILIES_ORDER]
            ok = p[tc[0]].notna()
            q = p[ok]
            err = (q[tc].to_numpy() != q[pc].to_numpy()).mean(axis=1)
            g = pd.DataFrame({"Temperature_C": q["Temperature_C"].values, "err": err})
            for T, r in g.groupby("Temperature_C")["err"].agg(["mean", "count"]).iterrows():
                temp_rows.append({"heldout_element": heldout, "task": "presence",
                                  "Temperature_C": float(T),
                                  "mean_tvd": float(r["mean"]),   # label error rate
                                  "median_tvd": None, "n_rows": int(r["count"])})
    return pd.DataFrame(temp_rows), pd.DataFrame(split_rows)


def target_distribution_shift(df, man, preds):
    """spec 58: POST-HOC comparison only. Never used to retrain anything."""
    rows = []
    for heldout in C.HELDOUT_ELEMENTS:
        roles = man[man.fold == f"FOLD_{heldout}"].set_index("composition_id")["loeo_role"]
        d = df.copy()
        d["loeo_role"] = d["composition_id"].map(roles)
        tr = d[d.loeo_role == "FOLD_TRAIN"]
        te = d[d.loeo_role == "HELDOUT_ELEMENT_TEST"]
        for task in C.ALL_TASKS:
            cols = C.TARGETS[task]
            a = C.task_rows(tr, task)[cols].to_numpy(np.float64)
            b = C.task_rows(te, task)[cols].to_numpy(np.float64)
            if not len(a) or not len(b):
                continue
            for i, fam in enumerate(C.FAMILIES_ORDER):
                rows.append({"heldout_element": heldout, "task": task, "target": fam,
                             "fold_train_mean": float(a[:, i].mean()),
                             "heldout_mean": float(b[:, i].mean()),
                             "fold_train_std": float(a[:, i].std()),
                             "heldout_std": float(b[:, i].std()),
                             "mean_shift": float(b[:, i].mean() - a[:, i].mean())})
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("STEP 5C   Stage J + K + L: held-out scoring, bootstrap, topology")
    print("=" * 78)
    gate_path = os.path.join(C.D_MANIFEST, "STEP5C_SCORING_GATE.json")
    if not os.path.exists(gate_path):
        raise SystemExit("FATAL: Stage-I gate missing. Run 54_step5c_gate.py first - all six "
                         "folds must be frozen before any held-out set is scored (spec 43).")
    gate = json.load(open(gate_path, encoding="utf-8"))
    if not gate.get("scoring_authorized"):
        raise SystemExit("FATAL: Stage-I gate says scoring is NOT authorized:\n  "
                         + json.dumps([c for c in gate["checks"]
                                       if c["status"] == "FAIL"], indent=2))
    print("  Stage-I gate: scoring AUTHORIZED (all six folds frozen)")

    man = pd.read_csv(os.path.join(C.D_FOLDS, "LOEO_MASTER_FOLD_MANIFEST.csv"))
    all_feats = sorted({f for e in C.HELDOUT_ELEMENTS
                        for f in json.load(open(os.path.join(
                            C.D_MANIFEST, f"FOLD_{e}_FREEZE.json"),
                            encoding="utf-8"))["selected_features"]})
    df, _ = C.load_domain_a(all_feats)

    all_metrics, all_targets, all_topo, all_boot, preds, records = [], [], [], [], {}, {}
    for heldout in C.HELDOUT_ELEMENTS:
        m, pt, topo, boot, pf, rec = evaluate_fold(heldout, df, man, None)
        all_metrics += m
        all_targets += pt
        all_boot += boot
        preds[heldout] = pf
        records[heldout] = rec
        if topo:
            all_topo.append({"heldout_element": heldout,
                             "topology_accuracy": topo["accuracy"],
                             "macro_f1": topo["macro_f1"], "support": topo["support"],
                             "true_dominant_distribution": json.dumps(topo["true_distribution"]),
                             "predicted_distribution": json.dumps(topo["predicted_distribution"]),
                             "confusion_matrix": json.dumps(topo["confusion_matrix"]),
                             "confusion_labels": ";".join(topo["confusion_labels"])})

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(C.D_METRICS, "LOEO_ALL_FOLDS_METRICS.csv"), index=False, encoding="utf-8")
    pd.DataFrame(all_targets).to_csv(
        os.path.join(C.D_METRICS, "LOEO_PER_TARGET_METRICS.csv"), index=False, encoding="utf-8")
    pd.DataFrame(all_topo).to_csv(
        os.path.join(C.D_METRICS, "LOEO_TOPOLOGY_METRICS.csv"), index=False, encoding="utf-8")
    pd.DataFrame(all_boot).to_csv(
        os.path.join(C.D_METRICS, "LOEO_BOOTSTRAP_CI.csv"), index=False, encoding="utf-8")
    temp, split = stratified_tables(preds)
    temp.to_csv(os.path.join(C.D_METRICS, "LOEO_ERROR_BY_TEMPERATURE.csv"),
                index=False, encoding="utf-8")
    split.to_csv(os.path.join(C.D_METRICS, "LOEO_ERROR_BY_PHASE_SPLIT.csv"),
                 index=False, encoding="utf-8")
    target_distribution_shift(df, man, preds).to_csv(
        os.path.join(C.D_METRICS, "LOEO_TARGET_DISTRIBUTION_SHIFT.csv"),
        index=False, encoding="utf-8")

    C.write_json(os.path.join(C.D_METRICS, "LOEO_BOOTSTRAP_META.json"),
                 {"generated_utc": C.utcnow(), "type": "composition-level cluster bootstrap",
                  "replicates": C.BOOTSTRAP_REPS, "master_seed": C.MASTER_SEED,
                  "fold_child_seeds": {e: C.fold_seed(e) for e in C.HELDOUT_ELEMENTS}})
    print("\nSTAGE J/K/L: PASS - six held-out evaluation records written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
