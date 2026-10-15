# -*- coding: utf-8 -*-
"""
41_step5a_bootstrap.py
STEP5A Stage R: composition-level cluster bootstrap of the IID test metrics
(sec.78). 1000 replicates, seed 20260904. Each replicate samples
composition_ids WITH replacement from IID_TEST and takes ALL temperature rows
of the sampled compositions. Never state-row bootstrap.

All quantities are computed from the frozen IID_TEST_PREDICTIONS.parquet -
no model is retrained or re-evaluated.

Output: 05_results/BASELINE_BOOTSTRAP_CI.csv
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5a_common as C

N_REP = 1000
SEED = C.MASTER_SEED
FAMILIES = C.FAMILIES_ORDER


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    preds = pd.read_parquet(os.path.join(C.RESULTS, "IID_TEST_PREDICTIONS.parquet"))
    freeze = json.load(open(os.path.join(C.STEP5A, "09_manifest", "DEVELOPMENT_FREEZE_V1.json"),
                            encoding="utf-8"))
    ths = np.array([freeze["tasks"]["presence"]["thresholds"][f] for f in FAMILIES])

    comps = preds["composition_id"].values
    uniq = np.unique(comps)
    row_idx_by_comp = {c: np.flatnonzero(comps == c) for c in uniq}
    print(f"IID_TEST: {len(preds)} rows / {len(uniq)} compositions")

    from sklearn.metrics import f1_score, average_precision_score

    # pre-extract arrays
    prob = np.column_stack([preds[f"{f}_presence_prob"].values for f in FAMILIES])
    true_p = np.column_stack([preds[f"{f}_presence_true"].values for f in FAMILIES]).astype(int)
    frac_proj = np.column_stack([preds[f"{f}_frac_proj"].values for f in FAMILIES])
    frac_true = np.column_stack([preds[f"{f}_volfrac_true"].values for f in FAMILIES])
    part = {}
    for el, tag in [("Cr", "CR_P"), ("Nb", "NB_P"), ("X", "X_P")]:
        part[el] = (np.column_stack([preds[f"{f}_{tag}_proj"].values for f in FAMILIES]),
                    np.column_stack([preds[f"{tag}_{f}_true"].values for f in FAMILIES]))

    rng = np.random.default_rng(SEED)
    acc = {k: [] for k in ["presence_exact_match", "presence_macro_f1", "presence_micro_f1",
                           "liquid_f1", "liquid_pr_auc",
                           "fraction_mean_tvd", "fraction_macro_mae",
                           "cr_mean_tvd", "nb_mean_tvd", "x_mean_tvd"]}
    for b in range(N_REP):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([row_idx_by_comp[uniq[i]] for i in pick])
        p_, t_ = prob[idx], true_p[idx]
        pred_bin = (p_ >= ths).astype(int)
        acc["presence_exact_match"].append(float((pred_bin == t_).all(axis=1).mean()))
        acc["presence_macro_f1"].append(float(f1_score(t_, pred_bin, average="macro", zero_division=0)))
        acc["presence_micro_f1"].append(float(f1_score(t_, pred_bin, average="micro", zero_division=0)))
        liq_t, liq_p = t_[:, 4], p_[:, 4]
        if len(np.unique(liq_t)) > 1:
            acc["liquid_f1"].append(float(f1_score(liq_t, pred_bin[:, 4], zero_division=0)))
            acc["liquid_pr_auc"].append(float(average_precision_score(liq_t, liq_p)))
        else:
            acc["liquid_f1"].append(np.nan)
            acc["liquid_pr_auc"].append(np.nan)
        fp, ft = frac_proj[idx], frac_true[idx]
        acc["fraction_mean_tvd"].append(float(C.tvd(fp, ft).mean()))
        acc["fraction_macro_mae"].append(float(np.abs(fp - ft).mean()))
        for el, key in [("Cr", "cr_mean_tvd"), ("Nb", "nb_mean_tvd"), ("X", "x_mean_tvd")]:
            pr, tr = part[el]
            m = np.isfinite(tr[idx]).all(axis=1)  # X partition: skip BASELINE rows
            acc[key].append(float(C.tvd(pr[idx][m], tr[idx][m]).mean()))
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b+1}/{N_REP}", flush=True)

    rows = []
    point = {
        "presence_exact_match": float(((prob >= ths).astype(int) == true_p).all(axis=1).mean()),
        "presence_macro_f1": float(f1_score(true_p, (prob >= ths).astype(int), average="macro", zero_division=0)),
        "presence_micro_f1": float(f1_score(true_p, (prob >= ths).astype(int), average="micro", zero_division=0)),
        "fraction_mean_tvd": float(C.tvd(frac_proj, frac_true).mean()),
        "fraction_macro_mae": float(np.abs(frac_proj - frac_true).mean()),
    }
    for el, key in [("Cr", "cr_mean_tvd"), ("Nb", "nb_mean_tvd"), ("X", "x_mean_tvd")]:
        pr, tr = part[el]
        m = np.isfinite(tr).all(axis=1)
        point[key] = float(C.tvd(pr[m], tr[m]).mean())
    if len(np.unique(true_p[:, 4])) > 1:
        point["liquid_f1"] = float(f1_score(true_p[:, 4], (prob[:, 4] >= ths[4]).astype(int), zero_division=0))
        point["liquid_pr_auc"] = float(average_precision_score(true_p[:, 4], prob[:, 4]))

    for k, v in acc.items():
        arr = np.array(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        rows.append({"metric": k, "point_estimate": point.get(k),
                     "bootstrap_mean": float(arr.mean()),
                     "ci95_low": float(np.percentile(arr, 2.5)),
                     "ci95_high": float(np.percentile(arr, 97.5)),
                     "n_replicates": len(arr)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.RESULTS, "BASELINE_BOOTSTRAP_CI.csv"), index=False)
    meta = {"generated_utc": ts, "n_replicates": N_REP, "seed": SEED,
            "scheme": "composition-level cluster bootstrap over IID_TEST compositions; "
                      "all temperature rows of each sampled composition travel together",
            "n_compositions": int(len(uniq)), "n_rows": int(len(preds))}
    with open(os.path.join(C.RESULTS, "BOOTSTRAP_META.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(out.to_string(index=False))
    print("STAGE R: done")


if __name__ == "__main__":
    main()
