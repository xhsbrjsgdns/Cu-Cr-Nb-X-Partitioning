# -*- coding: utf-8 -*-
"""
43_step5a_figures.py
STEP5A Stage T: journal-draft figures (sec.93-94).

Rules honored:
  - parity panels say "IID_TEST only" and annotate N states / N compositions
  - no cosmetic point removal, no axis hiding of errors, minority phases shown
  - LIQUID gets its own figure
Outputs: 06_figures/Fig_S5A_01..10*.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5a_common as C

FIG = os.path.join(C.STEP5A, "06_figures")
FAMILIES = C.FAMILIES_ORDER
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def fig01_pipeline():
    sel = json.load(open(os.path.join(C.FEAT, "FEATURE_SELECTION_SUMMARY.json"), encoding="utf-8"))
    stages = ["candidates", "after NZV", "after exact\nredundancy", "after PCC\n|r|>0.90"]
    n0 = sel["candidate_features"]
    n1 = n0 - sel["nzv_removed"]
    n2 = n1 - sel["exact_redundancy_removed"]
    n3 = sel["selected_total"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].bar(stages, [n0, n1, n2, n3], color=["#888", "#4C72B0", "#55A868", "#C44E52"])
    for i, v in enumerate([n0, n1, n2, n3]):
        ax[0].text(i, v + 3, str(v), ha="center")
    ax[0].set_ylabel("features (DEV_TRAIN filtering)")
    ax[0].set_title("Feature pipeline (train-only)")
    blocks = ["F1", "F2", "F3", "F4"]
    counts = [len(sel["selected_F1"]), len(sel["selected_F2"]), len(sel["selected_F3"]), sel["selected_F4_n"]]
    ax[1].bar(blocks, counts, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    for i, v in enumerate(counts):
        ax[1].text(i, v + 0.4, str(v), ha="center")
    ax[1].set_ylabel("selected features")
    ax[1].set_title(f"Final {n3} features by block")
    fig.savefig(os.path.join(FIG, "Fig_S5A_01_feature_pipeline.png"))
    plt.close(fig)


def fig02_ablation():
    ab = pd.read_csv(os.path.join(C.RESULTS, "FEATURE_SET_ABLATION.csv"))
    main = ab[~ab["task"].str.contains(":")]
    sets = ["F1", "F1_F2", "F1_F2_F3", "F1_F2_F3_F4"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    p = main[main["task"] == "presence"].set_index("feature_set").reindex(sets)
    ax[0].plot(sets, p["macro_f1"], "o-", label="XGBoost")
    ax[0].axhline(p["trivial_macro_f1"].iloc[0], ls="--", c="gray", label="trivial (prevalence)")
    ax[0].set_ylabel("validation macro F1")
    ax[0].set_title("Task A: phase presence")
    ax[0].legend()
    for task, lab in [("fraction", "phase fraction"), ("cr_partition", "Cr partition"),
                      ("nb_partition", "Nb partition"), ("x_partition", "X partition")]:
        t = main[main["task"] == task].set_index("feature_set").reindex(sets)
        ax[1].plot(sets, t["mean_tvd"], "o-", label=lab)
        ax[1].axhline(t["trivial_mean_tvd"].iloc[0], ls=":", c="gray", lw=0.8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("validation mean TVD (log)")
    ax[1].set_title("Tasks B-E (dotted gray: trivial mean vector)")
    ax[1].legend()
    fig.savefig(os.path.join(FIG, "Fig_S5A_02_feature_ablation.png"))
    plt.close(fig)


def fig03_model_comparison():
    screen = pd.read_csv(os.path.join(C.RESULTS, "BASELINE_MODEL_SCREEN.csv"))
    test = pd.read_csv(os.path.join(C.RESULTS, "BASELINE_IID_TEST_METRICS.csv"))
    screen = screen[~screen["task"].str.contains(":")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    tasks = ["presence", "fraction", "cr_partition", "nb_partition", "x_partition"]
    for ax, task in zip(axes.flat, tasks):
        if task == "presence":
            s = screen[screen["task"] == task]
            t = test[test["task"] == task]
            models = s["model"].tolist()
            vals = s["macro_f1"].tolist()
            tmap = dict(zip(t["model"], t["macro_f1"]))
            ax.bar(range(len(models)), vals, color="#4C72B0", label="validation")
            ax.scatter(range(len(models)), [tmap.get(m, np.nan) for m in models],
                       color="#C44E52", zorder=3, label="IID test")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=20, ha="right", fontsize=7)
            ax.set_ylabel("macro F1")
        else:
            s = screen[screen["task"] == task]
            t = test[test["task"] == task]
            models = s["model"].tolist()
            vals = s["mean_tvd"].tolist()
            tmap = dict(zip(t["model"], t["mean_tvd"]))
            ax.bar(range(len(models)), vals, color="#4C72B0", label="validation")
            ax.scatter(range(len(models)), [tmap.get(m, np.nan) for m in models],
                       color="#C44E52", zorder=3, label="IID test")
            ax.set_yscale("log")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=20, ha="right", fontsize=7)
            ax.set_ylabel("mean TVD (log)")
        ax.set_title(task)
        ax.legend(fontsize=7)
    axes.flat[5].axis("off")
    fig.savefig(os.path.join(FIG, "Fig_S5A_03_model_comparison.png"))
    plt.close(fig)


def _parity(ax, ytrue, ypred, title, n_states, n_comp, loglog=False):
    m = np.isfinite(ytrue) & np.isfinite(ypred)
    yt, yp = ytrue[m], ypred[m]
    if loglog:
        lo = 1e-8
        yt = np.clip(yt, lo, None)
        yp = np.clip(yp, lo, None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, 1.5); ax.set_ylim(lo, 1.5)
    ax.scatter(yt, yp, s=2, alpha=0.25, color="#4C72B0", rasterized=True)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_xlim()[0], ax.get_xlim()[1]],
            "k-", lw=0.8)
    ax.set_xlabel("true"); ax.set_ylabel("predicted")
    ax.set_title(f"{title}\nIID_TEST only | N states={n_states:,} | N comp={n_comp:,}", fontsize=7.5)


def fig04_05_parity(preds):
    n_comp = preds["composition_id"].nunique()
    # Fig 04: phase fraction parity, 5 families (log axes for minority phases so
    # their error structure is visible; all points retained)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, fam in zip(axes, FAMILIES):
        yt = preds[f"{fam}_volfrac_true"].values
        yp = preds[f"{fam}_frac_proj"].values
        _parity(ax, yt, yp, f"{fam} volume fraction", len(preds), n_comp, loglog=(fam != "FCC"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Fig_S5A_04_phase_fraction_parity.png"))
    plt.close(fig)
    # Fig 05: partition parity Cr/Nb/X x 5 families (log-log)
    fig, axes = plt.subplots(3, 5, figsize=(17, 10.2))
    for r, (el, tag) in enumerate([("Cr", "CR_P"), ("Nb", "NB_P"), ("X", "X_P")]):
        for c, fam in enumerate(FAMILIES):
            yt = preds[f"{tag}_{fam}_true"].values
            yp = preds[f"{fam}_{tag}_proj"].values
            m = np.isfinite(yt)
            _parity(axes[r][c], yt[m], yp[m], f"{el} → {fam}", int(m.sum()),
                    preds.loc[m, "composition_id"].nunique(), loglog=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Fig_S5A_05_partition_parity.png"))
    plt.close(fig)


def fig06_error_by_T():
    t = pd.read_csv(os.path.join(C.RESULTS, "TEST_ERROR_BY_TEMPERATURE.csv"))
    t["T"] = t["stratum"].str.extract(r"T=(\d+)").astype(int)
    t = t.sort_values("T")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(t["T"], t["fraction_mean_tvd"], "o-", label="fraction TVD")
    ax.plot(t["T"], t["cr_mean_tvd"], "s-", label="Cr TVD")
    ax.plot(t["T"], t["nb_mean_tvd"], "^-", label="Nb TVD")
    ax.plot(t["T"], t["x_mean_tvd"], "d-", label="X TVD")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (C)"); ax.set_ylabel("mean TVD (log)")
    ax2 = ax.twinx()
    ax2.plot(t["T"], t["presence_macro_f1"], "k:", label="presence macro F1")
    ax2.set_ylabel("presence macro F1"); ax2.grid(False)
    ax.set_title("IID_TEST error by temperature")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    fig.savefig(os.path.join(FIG, "Fig_S5A_06_error_by_temperature.png"))
    plt.close(fig)


def fig07_error_by_family():
    t = pd.read_csv(os.path.join(C.RESULTS, "TEST_ERROR_BY_FAMILY.csv"))
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x - 0.2, t["fraction_mean_tvd"], 0.2, label="fraction TVD")
    ax.bar(x, t["cr_mean_tvd"], 0.2, label="Cr TVD")
    ax.bar(x + 0.2, t["nb_mean_tvd"], 0.2, label="Nb TVD")
    ax.bar(x + 0.4, t["x_mean_tvd"].fillna(0), 0.2, label="X TVD")
    ax.set_yscale("log")
    ax.set_xticks(x + 0.1); ax.set_xticklabels(t["stratum"].str.replace("family=", ""), rotation=20)
    ax.set_ylabel("mean TVD (log)")
    ax2 = ax.twinx()
    ax2.plot(x + 0.1, t["presence_macro_f1"], "k:o", ms=3, label="presence macro F1")
    ax2.set_ylabel("presence macro F1"); ax2.grid(False)
    ax.set_title("IID_TEST error by alloy family")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    fig.savefig(os.path.join(FIG, "Fig_S5A_07_error_by_family.png"))
    plt.close(fig)


def fig08_liquid(preds):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    yt = preds["LIQUID_presence_true"].values.astype(int)
    yp = preds["LIQUID_presence_prob"].values
    prec, rec, _ = precision_recall_curve(yt, yp)
    axes[0].step(rec, prec, where="post", color="#C44E52")
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision")
    axes[0].set_title(f"LIQUID presence PR (test)\nPR-AUC={average_precision_score(yt, yp):.4f}, "
                      f"support={int(yt.sum())}")
    ft = preds["LIQUID_volfrac_true"].values
    fp = preds["LIQUID_frac_proj"].values
    axes[1].scatter(ft, fp, s=4, alpha=0.4, color="#4C72B0")
    axes[1].plot([0, max(ft.max(), fp.max())], [0, max(ft.max(), fp.max())], "k-", lw=0.8)
    axes[1].set_xlabel("true LIQUID volfrac"); axes[1].set_ylabel("predicted")
    axes[1].set_title("LIQUID fraction parity (test only)")
    d = fp - ft
    axes[2].hist(d, bins=60, color="#55A868")
    axes[2].set_yscale("log")
    axes[2].axvline(0, color="k", lw=0.8)
    axes[2].set_xlabel("predicted - true LIQUID volfrac")
    axes[2].set_title(f"error dist (MAE={np.abs(d).mean():.2e})")
    fig.savefig(os.path.join(FIG, "Fig_S5A_08_liquid_performance.png"))
    plt.close(fig)


def fig09_simplex(preds):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (task, tag) in zip(axes, [("fraction", "frac"), ("cr_partition", "CR_P"),
                                      ("nb_partition", "NB_P"), ("x_partition", "X_P")]):
        raw = np.column_stack([preds[f"{f}_{tag}_raw"].values for f in FAMILIES])
        proj = np.column_stack([preds[f"{f}_{tag}_proj"].values for f in FAMILIES])
        m = np.isfinite(raw).all(axis=1)
        rs = raw[m].sum(axis=1)
        ax.hist(rs, bins=60, alpha=0.7, label="raw sum", color="#C44E52")
        ax.axvline(1.0, color="k", lw=0.8)
        ps = proj[m].sum(axis=1)
        ax.hist(ps, bins=np.linspace(1 - 1e-6, 1 + 1e-6, 41), alpha=0.7,
                label="projected sum", color="#4C72B0")
        ax.set_yscale("log")
        ax.set_title(f"{task}: simplex sums (test)\nraw |sum-1| mean={np.abs(rs-1).mean():.2e}")
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "Fig_S5A_09_simplex_constraint.png"))
    plt.close(fig)


def fig10_phase_boundary():
    t = pd.read_csv(os.path.join(C.RESULTS, "TEST_ERROR_BY_PHASE_SPLIT.csv"))
    keys = ["any_split_region", "no_split_region"]
    t2 = t[t["stratum"].isin(keys)].set_index("stratum").reindex(keys)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    metrics = ["fraction_mean_tvd", "cr_mean_tvd", "nb_mean_tvd", "x_mean_tvd"]
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, t2[metrics].values[0], 0.36, label="split region", color="#C44E52")
    ax.bar(x + 0.18, t2[metrics].values[1], 0.36, label="non-split region", color="#4C72B0")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_mean_tvd", "") for m in metrics])
    ax.set_ylabel("mean TVD (log)")
    ax.set_title("IID_TEST: miscibility-gap (split) vs non-split regions")
    ax.legend()
    fig.savefig(os.path.join(FIG, "Fig_S5A_10_phase_boundary_error.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    preds = pd.read_parquet(os.path.join(C.RESULTS, "IID_TEST_PREDICTIONS.parquet"))
    fig01_pipeline()
    fig02_ablation()
    fig03_model_comparison()
    fig04_05_parity(preds)
    fig06_error_by_T()
    fig07_error_by_family()
    fig08_liquid(preds)
    fig09_simplex(preds)
    fig10_phase_boundary()
    print("STAGE T: 10 figures written to 06_figures/")


if __name__ == "__main__":
    main()
