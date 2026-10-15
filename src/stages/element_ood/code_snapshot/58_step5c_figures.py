# -*- coding: utf-8 -*-
"""
58_step5c_figures.py
STEP5C Stage O: the ten LOEO figures (spec 69). PNG + PDF, no seaborn.
Every figure is drawn from the frozen metric tables - nothing is recomputed here.
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
import step5c_common as C

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
TASKS = ["presence", "fraction", "cr_partition", "nb_partition", "x_partition"]
NICE = {"presence": "Phase presence\n(Macro F1)", "fraction": "Phase fraction\n(mean TVD)",
        "cr_partition": "Cr partition\n(mean TVD)", "nb_partition": "Nb partition\n(mean TVD)",
        "x_partition": "X partition\n(mean TVD)"}
CAPTIONS = {}


def save(fig, name, caption):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.D_FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    CAPTIONS[name] = caption
    print("  wrote %s.png/.pdf" % name)


def fig01(met):
    piv = met.pivot_table(index="heldout_element", columns="task", values="value")
    piv = piv.reindex(C.HELDOUT_ELEMENTS)[TASKS]
    norm = piv.copy()
    for t in TASKS:                       # per-task min-max so one colour scale reads
        col = piv[t].astype(float)
        rng = col.max() - col.min()
        z = (col - col.min()) / rng if rng > 0 else col * 0
        norm[t] = (1 - z) if t == "presence" else z   # everywhere: darker = worse
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(norm.to_numpy(float), cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([NICE[t] for t in TASKS], fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            ax.text(j, i, "-" if pd.isna(v) else ("%.4f" % v), ha="center", va="center",
                    fontsize=7.5,
                    color="white" if norm.iloc[i, j] > 0.6 else "black")
    ax.set_title("STEP5C retrospective leave-one-element-out: held-out performance", fontsize=10)
    fig.colorbar(im, ax=ax, label="relative difficulty within task\n(darker = worse)")
    ax.grid(False)
    save(fig, "Fig_S5C_01_LOEO_heatmap",
         "Held-out performance of all six retrospective LOEO folds across the five tasks. "
         "Cell text is the raw metric; colour is min-max normalised within each task so "
         "that darker always means worse.")


def fig02(met):
    m = met[met.task == "presence"].set_index("heldout_element").reindex(C.HELDOUT_ELEMENTS)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(m))
    lo = (m["value"] - m["bootstrap_low"]).clip(lower=0).fillna(0)
    hi = (m["bootstrap_high"] - m["value"]).clip(lower=0).fillna(0)
    ax.bar(x - 0.2, m["value"], 0.36, yerr=[lo, hi], capsize=3, label="Macro F1 (5 labels)")
    if "macro_f1_non_degenerate" in m.columns:
        ax.bar(x + 0.2, m["macro_f1_non_degenerate"], 0.36,
               label="Macro F1, non-degenerate labels only")
    ax.plot(x, m["trivial_value"], "k^--", ms=5, lw=1, label="fold-train trivial baseline")
    ax.set_xticks(x); ax.set_xticklabels(m.index)
    ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("held-out element")
    ax.set_title("Phase presence under element-domain OOD", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "Fig_S5C_02_LOEO_presence_by_element",
         "Phase-presence Macro F1 for each held-out element with 95% composition-level "
         "cluster-bootstrap intervals. The non-degenerate variant excludes presence labels "
         "that carry only one class in that held-out test set.")


def fig03(met, cmp_):
    m = met[met.task == "fraction"].set_index("heldout_element").reindex(C.HELDOUT_ELEMENTS)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(m))
    ax.bar(x, m["value"], 0.55,
           yerr=[(m["value"] - m["bootstrap_low"]).clip(lower=0).fillna(0),
                 (m["bootstrap_high"] - m["value"]).clip(lower=0).fillna(0)],
           capsize=3, color="#4878a8")
    r = cmp_[cmp_.task == "fraction"]
    if len(r):
        if pd.notna(r.iloc[0]["STEP5A_IID"]):
            ax.axhline(r.iloc[0]["STEP5A_IID"], color="green", ls="--", lw=1,
                       label="STEP5A IID")
        if pd.notna(r.iloc[0]["STEP5B_composition_domain_OOD"]):
            ax.axhline(r.iloc[0]["STEP5B_composition_domain_OOD"], color="orange", ls="-.",
                       lw=1, label="STEP5B composition-domain OOD")
    ax.set_xticks(x); ax.set_xticklabels(m.index)
    ax.set_ylabel("mean TVD"); ax.set_xlabel("held-out element")
    ax.set_title("Phase-fraction error under element-domain OOD", fontsize=10)
    ax.legend(fontsize=7.5)
    save(fig, "Fig_S5C_03_LOEO_fraction_TVD",
         "Phase-fraction mean total variation distance per held-out element, with the IID "
         "and composition-domain OOD reference levels.")


def fig04(met):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    for ax, task, title in zip(axes, ["cr_partition", "nb_partition", "x_partition"],
                               ["Cr partition", "Nb partition", "X partition (key task)"]):
        m = met[met.task == task].set_index("heldout_element").reindex(C.HELDOUT_ELEMENTS)
        x = np.arange(len(m))
        ax.bar(x, m["value"], 0.6,
               yerr=[(m["value"] - m["bootstrap_low"]).clip(lower=0).fillna(0),
                     (m["bootstrap_high"] - m["value"]).clip(lower=0).fillna(0)],
               capsize=3, color="#a85a4a" if task == "x_partition" else "#4878a8")
        ax.plot(x, m["trivial_value"], "k^--", ms=4, lw=1, label="trivial")
        ax.set_xticks(x); ax.set_xticklabels(m.index, fontsize=8)
        ax.set_title(title, fontsize=9.5); ax.set_ylabel("mean TVD")
        ax.legend(fontsize=7)
    fig.suptitle("Elemental partition error under element-domain OOD", fontsize=10.5)
    save(fig, "Fig_S5C_04_LOEO_partition_TVD",
         "Cr, Nb and X partition mean TVD per held-out element with cluster-bootstrap "
         "intervals and the fold-train trivial baseline.")


def fig05(topo):
    n = len(topo)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.1 * rows))
    axes = np.atleast_1d(axes).ravel()
    labels = C.FAMILIES_ORDER
    for ax, (_, r) in zip(axes, topo.iterrows()):
        cm = np.array(json.loads(r["confusion_matrix"]), dtype=float)
        with np.errstate(invalid="ignore"):
            cmn = cm / np.where(cm.sum(axis=1, keepdims=True) > 0,
                                cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l[:4] for l in labels], fontsize=6.5, rotation=45)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l[:4] for l in labels], fontsize=6.5)
        ax.set_title("%s  acc %.3f  F1 %.3f"
                     % (r["heldout_element"], r["topology_accuracy"], r["macro_f1"]),
                     fontsize=8.5)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, "%d" % cm[i, j], ha="center", va="center", fontsize=5.5,
                            color="white" if cmn[i, j] > 0.5 else "black")
        ax.grid(False)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Dominant X-partition topology: true (rows) vs predicted (columns)",
                 fontsize=10.5)
    save(fig, "Fig_S5C_05_X_partition_topology_confusion",
         "Confusion matrices for the dominant X-partition family of each held-out element. "
         "Rows are the argmax of the true partition vector, columns the argmax of the "
         "projected prediction.")


def fig06(el, ev_el):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    e = el.set_index("heldout_element").reindex(C.HELDOUT_ELEMENTS)
    x = np.arange(len(e))
    axes[0].bar(x - 0.2, e["nearest_element_distance"], 0.38, label="nearest of the 5")
    axes[0].bar(x + 0.2, e["mean_distance_to_5"], 0.38, label="mean to all 5")
    axes[0].set_xticks(x); axes[0].set_xticklabels(e.index)
    axes[0].set_ylabel("standardised Euclidean distance")
    axes[0].set_title("Held-out element vs its 5 training elements", fontsize=9.5)
    axes[0].legend(fontsize=7.5)
    for i, el_ in enumerate(e.index):
        axes[0].annotate(e.loc[el_, "nearest_element"],
                         (i - 0.2, e.loc[el_, "nearest_element_distance"]),
                         ha="center", va="bottom", fontsize=6.5)
    met = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_ALL_FOLDS_METRICS.csv"))
    xp = met[met.task == "x_partition"].set_index("heldout_element").reindex(e.index)
    axes[1].scatter(e["nearest_element_distance"], xp["value"], s=45, c="#a85a4a")
    for el_ in e.index:
        axes[1].annotate(el_, (e.loc[el_, "nearest_element_distance"], xp.loc[el_, "value"]),
                         textcoords="offset points", xytext=(4, 3), fontsize=7.5)
    axes[1].set_xlabel("nearest-element descriptor distance")
    axes[1].set_ylabel("X partition mean TVD")
    rho = ev_el[(ev_el.task == "x_partition")
                & (ev_el.distance_metric == "nearest_element_distance")]
    sub = "n = 6, descriptive only"
    if len(rho) and pd.notna(rho.iloc[0]["spearman_rho"]):
        sub = "Spearman rho = %.3f (n = 6, DESCRIPTIVE ONLY)" % rho.iloc[0]["spearman_rho"]
    axes[1].set_title(sub, fontsize=9)
    save(fig, "Fig_S5C_06_element_descriptor_distance",
         "Left: standardised descriptor distance from each held-out element to the five "
         "training elements of its fold, scaler fitted on those five only. Right: X-partition "
         "error against that distance. With six points this is exploratory only and no "
         "statistical inference is drawn.")


def fig07(quart):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    q = quart[quart.task == "x_partition"]
    order = ["Q1", "Q2", "Q3", "Q4"]
    for e in C.HELDOUT_ELEMENTS:
        s = q[q.heldout_element == e].set_index("distance_quartile").reindex(order)
        axes[0].plot(order, s["mean_error"], "o-", ms=4, lw=1.2, label=e)
    axes[0].set_xlabel("composition 1-NN distance quartile (Q1 = nearest)")
    axes[0].set_ylabel("X partition mean TVD")
    axes[0].set_title("Error vs composition distance", fontsize=9.5)
    axes[0].legend(fontsize=7, ncol=2)
    ev = pd.read_csv(os.path.join(C.D_DIST, "LOEO_ERROR_DISTANCE.csv"))
    rho_col = ("spearman_rho_nn1" if "spearman_rho_nn1" in ev.columns
               else "spearman_rho")          # LOEO_ERROR_DISTANCE carries the _nn1 / _nn5 pair
    piv = ev.pivot_table(index="heldout_element", columns="task", values=rho_col)
    piv = piv.reindex(C.HELDOUT_ELEMENTS)
    im = axes[1].imshow(piv.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(piv.shape[1]))
    axes[1].set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=7.5)
    axes[1].set_yticks(range(piv.shape[0])); axes[1].set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                axes[1].text(j, i, "%.2f" % v, ha="center", va="center", fontsize=7)
    axes[1].set_title("Spearman rho: error vs 1-NN distance\n(blank = error constant, rho undefined)", fontsize=9)
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1])
    save(fig, "Fig_S5C_07_error_vs_composition_distance",
         "Left: X-partition error by composition-distance quartile within each held-out "
         "element. Right: Spearman correlation between per-composition error and 1-NN "
         "distance to that fold's training compositions.")


def fig08(cmp_):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5))
    r = cmp_[cmp_.task == "presence"]
    if len(r):
        r = r.iloc[0]
        vals = [r["STEP5A_IID"], r["STEP5B_composition_domain_OOD"], r["STEP5C_LOEO_mean"]]
        axes[0].bar(["IID\n(5A)", "composition OOD\n(5B)", "element OOD\n(5C mean)"],
                    vals, color=["#4a8", "#e8a33d", "#a85a4a"])
        # clip at 0: with near-identical folds the mean can sit a float epsilon below
        # the min, and matplotlib rejects a negative yerr
        lo_e = max(0.0, float(r["STEP5C_LOEO_mean"] - r["STEP5C_LOEO_min"]))
        hi_e = max(0.0, float(r["STEP5C_LOEO_max"] - r["STEP5C_LOEO_mean"]))
        axes[0].errorbar([2], [r["STEP5C_LOEO_mean"]], yerr=[[lo_e], [hi_e]],
                         fmt="none", ecolor="k", capsize=4)
        axes[0].set_ylabel("presence Macro F1"); axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Phase presence across three generalization layers", fontsize=9.5)
    reg = ["fraction", "cr_partition", "nb_partition", "x_partition"]
    x = np.arange(len(reg)); w = 0.26
    for k, (col, lab, c) in enumerate([("STEP5A_IID", "IID (5A)", "#4a8"),
                                       ("STEP5B_composition_domain_OOD",
                                        "composition OOD (5B)", "#e8a33d"),
                                       ("STEP5C_LOEO_mean", "element OOD (5C mean)",
                                        "#a85a4a")]):
        v = [float(cmp_[cmp_.task == t][col].iloc[0]) if len(cmp_[cmp_.task == t])
             and pd.notna(cmp_[cmp_.task == t][col].iloc[0]) else np.nan for t in reg]
        axes[1].bar(x + (k - 1) * w, v, w, label=lab, color=c)
    axes[1].set_xticks(x); axes[1].set_xticklabels(reg, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("mean TVD"); axes[1].set_yscale("log")
    axes[1].set_title("Regression error across three generalization layers", fontsize=9.5)
    axes[1].legend(fontsize=7.5)
    save(fig, "Fig_S5C_08_IID_vs_B_OOD_vs_LOEO",
         "The three-layer comparison: IID (STEP5A), composition-domain OOD (STEP5B) and "
         "element-domain OOD (STEP5C, mean over six folds with the min-max range shown).")


def fig09(freq):
    top = freq.sort_values(["selected_in_n_folds", "feature"],
                           ascending=[False, True]).head(45)
    fig, ax = plt.subplots(figsize=(7.0, max(3.5, 0.17 * len(top))))
    colors = {"F1": "#2b6", "F2": "#48a", "F3": "#a54", "F4": "#888", "OTHER": "#ccc"}
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["selected_in_n_folds"],
            color=[colors.get(b, "#ccc") for b in top["feature_block"]])
    ax.set_yticks(y); ax.set_yticklabels(top["feature"], fontsize=6.5)
    ax.set_xlabel("number of folds selecting the feature (out of 6)")
    ax.set_xlim(0, 6.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), fontsize=7.5, title="block", loc="lower right")
    ax.set_title("Feature-selection stability across the six LOEO folds", fontsize=10)
    save(fig, "Fig_S5C_09_feature_selection_frequency",
         "How often each candidate feature survived that fold's train-only filtering. "
         "Features selected in all six folds are stable descriptors; features selected in "
         "few folds are element-sensitive.")


def fig10(temp):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    xp = temp[temp.task == "x_partition"]
    for e in C.HELDOUT_ELEMENTS:
        s = xp[xp.heldout_element == e].sort_values("Temperature_C")
        axes[0].plot(s["Temperature_C"], s["mean_tvd"], "o-", ms=3.5, lw=1.2, label=e)
    axes[0].set_xlabel("Temperature (C)"); axes[0].set_ylabel("X partition mean TVD")
    axes[0].set_title("X partition error by temperature", fontsize=9.5)
    axes[0].legend(fontsize=7, ncol=2)
    for T in (300, 950, 1000):
        axes[0].axvline(T, color="grey", ls=":", lw=0.8)
    pr = temp[temp.task == "presence"]
    for e in C.HELDOUT_ELEMENTS:
        s = pr[pr.heldout_element == e].sort_values("Temperature_C")
        axes[1].plot(s["Temperature_C"], s["mean_tvd"], "o-", ms=3.5, lw=1.2, label=e)
    axes[1].set_xlabel("Temperature (C)"); axes[1].set_ylabel("presence label error rate")
    axes[1].set_title("Presence error by temperature", fontsize=9.5)
    axes[1].legend(fontsize=7, ncol=2)
    for T in (300, 950, 1000):
        axes[1].axvline(T, color="grey", ls=":", lw=0.8)
    save(fig, "Fig_S5C_10_LOEO_temperature_error",
         "Held-out error against the 11-point temperature grid for every fold. Dotted lines "
         "mark 300, 950 and 1000 C. For FOLD_MO the 950 C column contains only the MO states "
         "that converged; the CALPHAD failures are excluded from the metric, never imputed.")


def main():
    print("=" * 78)
    print("STEP 5C   Stage O: figures")
    print("=" * 78)
    met = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_ALL_FOLDS_METRICS.csv"))
    topo = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_TOPOLOGY_METRICS.csv"))
    cmp_ = pd.read_csv(os.path.join(C.D_METRICS, "IID_VS_COMPOSITION_OOD_VS_LOEO.csv"))
    freq = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_FEATURE_SELECTION_FREQUENCY.csv"))
    temp = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_ERROR_BY_TEMPERATURE.csv"))
    el = pd.read_csv(os.path.join(C.D_DIST, "LOEO_ELEMENT_DISTANCE.csv"))
    ev_el = pd.read_csv(os.path.join(C.D_DIST, "LOEO_ERROR_VS_ELEMENT_DISTANCE.csv"))
    quart = pd.read_csv(os.path.join(C.D_DIST, "LOEO_ERROR_BY_DISTANCE_QUARTILE.csv"))

    fig01(met); fig02(met); fig03(met, cmp_); fig04(met); fig05(topo)
    fig06(el, ev_el); fig07(quart); fig08(cmp_); fig09(freq); fig10(temp)

    with open(os.path.join(C.D_FIG, "FIGURE_CAPTIONS.txt"), "w", encoding="utf-8") as f:
        for k in sorted(CAPTIONS):
            f.write("%s\n%s\n\n" % (k, CAPTIONS[k]))
    print("\nSTAGE O: PASS - %d figures" % len(CAPTIONS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
