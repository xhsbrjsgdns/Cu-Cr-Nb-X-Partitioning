# -*- coding: utf-8 -*-
"""
116_i_figures.py
STEP5I Stage 6: the ten preregistered figures (spec 53).

Every panel carries the frozen zero-shot and trivial reference lines measured on the
SAME recovery subset, so no figure can be read as showing a recovery that is really a
change of subset. Budget 0 on the learning curves is the frozen zero-shot value.

No figure uses journal-success language, and Fig_I_10 in particular reports B_skill and
B_recovery25 as measured numbers, not as achievements.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i_common as C

PALETTE = {"M_REFIT": "#2f6f9f", "R0": "#c26a2a", "R1": "#7a9a3a", "R2": "#8b4f9e",
           "M_ZERO": "#444444", "M_TRIVIAL": "#999999"}
REP_COLORS = ["#2f6f9f", "#4f9ec4", "#1b4a6b"]


def finish(fig, ax, path, title, ylab, xlab="anchor compositions (budget)"):
    if isinstance(ax, plt.Axes):
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("    %s" % os.path.basename(path))


def budget_axis(ax, budgets):
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0] + list(budgets))
    ax.set_xticklabels(["0"] + [str(b) for b in budgets])


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5I  Stage 6: figures")
    print("=" * 78)

    base = json.load(open(os.path.join(C.D_METRICS, "STEP5I_RECOVERY_BASELINES.json"),
                          encoding="utf-8"))
    zero = base["M_ZERO"]["mean_tvd"]
    zero_topo = base["M_ZERO"]["topology_accuracy"]
    triv = base["M_TRIVIAL"]["mean_tvd"]
    triv_topo = base["M_TRIVIAL"]["topology_accuracy"]
    thr = json.load(open(os.path.join(C.D_METRICS, "STEP5I_BUDGET_THRESHOLDS.json"),
                         encoding="utf-8"))
    per_rep = pd.read_csv(os.path.join(C.D_METRICS,
                                       "STEP5I_RANDOM_FEWSHOT_PER_REPLICATE.csv"))
    agg = pd.read_csv(os.path.join(C.D_METRICS, "STEP5I_RANDOM_FEWSHOT_METRICS.csv"))
    maxi = pd.read_csv(os.path.join(C.D_METRICS, "STEP5I_MAXIMIN_FEWSHOT_METRICS.csv"))
    topo = pd.read_csv(os.path.join(C.D_TOPO, "STEP5I_TRUE_TOPOLOGY_STRATIFIED.csv"))
    nb = pd.read_csv(os.path.join(C.D_NBNI3, "STEP5I_NBNI3_STRATIFIED.csv"))
    tt = pd.read_csv(os.path.join(C.D_TOPO, "STEP5I_TEMPERATURE_REGIME_METRICS.csv"))
    cov = pd.read_csv(os.path.join(C.D_TOPO, "STEP5I_ANCHOR_TOPOLOGY_COVERAGE.csv"))
    B = C.BUDGETS
    rf = per_rep[per_rep.method == "M_REFIT"]

    # ---- 01 random learning curve, three sequences + mean ------------------ #
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, r in enumerate(C.REPLICATES):
        g = rf[rf.replicate == r].sort_values("budget")
        ax.plot([0] + list(g.budget), [zero] + list(g.mean_tvd), "o-", ms=4, lw=1.1,
                color=REP_COLORS[i], alpha=0.85, label="random replicate %d" % r)
    m = agg[agg.method == "M_REFIT"].sort_values("budget")
    ax.plot([0] + list(m.budget), [zero] + list(m.mean_tvd_mean), "s-", ms=6, lw=2.2,
            color="#12324a", label="mean of 3 replicates")
    ax.axhline(zero, color=PALETTE["M_ZERO"], ls="--", lw=1,
               label="frozen zero-shot (budget 0) %.4f" % zero)
    budget_axis(ax, B)
    finish(fig, ax, os.path.join(C.D_FIG, "Fig_I_01_random_learning_curve.png"),
           "M_REFIT recovery curve, RECOVERY_EVALUATION (%d states)\n"
           "post-blind retrospective recovery - not a zero-shot result"
           % C.N_RECOVERY_STATES, "mean TVD")

    # ---- 02 random vs trivial and zero-shot -------------------------------- #
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([0] + list(m.budget), [zero] + list(m.mean_tvd_mean), "s-", ms=6, lw=2,
            color=PALETTE["M_REFIT"], label="M_REFIT (mean of 3)")
    ax.fill_between(list(m.budget), list(m.mean_tvd_min), list(m.mean_tvd_max),
                    color=PALETTE["M_REFIT"], alpha=0.15, label="replicate range")
    ax.axhline(zero, color=PALETTE["M_ZERO"], ls="--", lw=1.4,
               label="M_ZERO frozen zero-shot %.4f" % zero)
    ax.axhline(triv, color=PALETTE["M_TRIVIAL"], ls=":", lw=1.6,
               label="M_TRIVIAL frozen train-mean %.4f" % triv)
    budget_axis(ax, B)
    finish(fig, ax, os.path.join(C.D_FIG, "Fig_I_02_random_vs_trivial.png"),
           "M_REFIT against the frozen references on the identical recovery subset",
           "mean TVD")

    # ---- 03 refit vs R0 / R1 / R2 ------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for meth in ("M_REFIT", "R0", "R1", "R2"):
        g = agg[agg.method == meth].sort_values("budget")
        ax.plot(g.budget, g.mean_tvd_mean, "o-", ms=5, lw=1.8, color=PALETTE[meth],
                label=meth)
        ax.fill_between(g.budget, g.mean_tvd_min, g.mean_tvd_max,
                        color=PALETTE[meth], alpha=0.12)
    ax.axhline(zero, color=PALETTE["M_ZERO"], ls="--", lw=1, label="M_ZERO")
    ax.axhline(triv, color=PALETTE["M_TRIVIAL"], ls=":", lw=1.4, label="M_TRIVIAL")
    budget_axis(ax, B)
    finish(fig, ax, os.path.join(C.D_FIG, "Fig_I_03_refit_vs_R0_R1_R2.png"),
           "Does refitting beat anchor-only correction?\n"
           "R0 ignores the model entirely", "mean TVD")

    # ---- 04 topology accuracy learning curve -------------------------------- #
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for meth in ("M_REFIT", "R0", "R2"):
        g = agg[agg.method == meth].sort_values("budget")
        ax.plot([0] + list(g.budget), [zero_topo] + list(g.topology_accuracy_mean),
                "o-", ms=5, lw=1.8, color=PALETTE[meth], label=meth)
    ax.axhline(zero_topo, color=PALETTE["M_ZERO"], ls="--", lw=1,
               label="zero-shot %.4f" % zero_topo)
    ax.axhline(triv_topo, color=PALETTE["M_TRIVIAL"], ls=":", lw=1.4,
               label="trivial %.4f" % triv_topo)
    budget_axis(ax, B)
    finish(fig, ax,
           os.path.join(C.D_FIG, "Fig_I_04_topology_accuracy_learning_curve.png"),
           "Topology (argmax family) accuracy on the recovery subset",
           "topology accuracy")

    # ---- 05 FCC vs OTHER_SOLID --------------------------------------------- #
    sub = topo[(topo.kind == "RANDOM") & (topo.method.isin(["M_REFIT", "R0"]))]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, stratum in zip(axes, ["FCC", "OTHER_SOLID"]):
        s = sub[sub.stratum == stratum]
        if not len(s):
            ax.text(0.5, 0.5, "no %s states" % stratum, ha="center", va="center")
            continue
        for meth in ("M_REFIT", "R0"):
            g = s[s.method == meth].groupby("budget").mean_tvd.agg(["mean", "min", "max"])
            ax.plot(g.index, g["mean"], "o-", ms=5, color=PALETTE[meth], label=meth)
            ax.fill_between(g.index, g["min"], g["max"], color=PALETTE[meth], alpha=0.12)
        zs = topo[(topo.stratum == stratum) & (topo.method == "M_ZERO")].mean_tvd.mean()
        ax.axhline(zs, color=PALETTE["M_ZERO"], ls="--", lw=1,
                   label="zero-shot %.4f" % zs)
        budget_axis(ax, B)
        ax.set_title("true topology = %s (n=%d states)"
                     % (stratum, int(s.n_states.iloc[0])), fontsize=10)
        ax.set_xlabel("anchor compositions (budget)")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("mean TVD")
    fig.suptitle("Recovery split by TRUE Ni topology (descriptive post-blind "
                 "stratification)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(C.D_FIG, "Fig_I_05_FCC_vs_OTHER_recovery.png"), dpi=160)
    plt.close(fig)
    print("    Fig_I_05_FCC_vs_OTHER_recovery.png")

    # ---- 06 NBNI3 present vs absent ---------------------------------------- #
    sub = nb[(nb.kind == "RANDOM") & (nb.method.isin(["M_REFIT", "R0"]))]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, stratum in zip(axes, ["NBNI3_PRESENT", "NBNI3_ABSENT"]):
        s = sub[sub.stratum == stratum]
        if not len(s):
            ax.text(0.5, 0.5, "no %s states" % stratum, ha="center", va="center")
            continue
        for meth in ("M_REFIT", "R0"):
            g = s[s.method == meth].groupby("budget").mean_tvd.agg(["mean", "min", "max"])
            ax.plot(g.index, g["mean"], "o-", ms=5, color=PALETTE[meth], label=meth)
            ax.fill_between(g.index, g["min"], g["max"], color=PALETTE[meth], alpha=0.12)
        zs = nb[(nb.stratum == stratum) & (nb.method == "M_ZERO")].mean_tvd.mean()
        ax.axhline(zs, color=PALETTE["M_ZERO"], ls="--", lw=1,
                   label="zero-shot %.4f" % zs)
        budget_axis(ax, B)
        ax.set_title("%s (n=%d states)" % (stratum, int(s.n_states.iloc[0])),
                     fontsize=10)
        ax.set_xlabel("anchor compositions (budget)")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("mean TVD")
    fig.suptitle("NBNI3 was unknown when the anchor sequences were frozen "
                 "(descriptive stratification)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(C.D_FIG, "Fig_I_06_NBNI3_present_absent_recovery.png"),
                dpi=160)
    plt.close(fig)
    print("    Fig_I_06_NBNI3_present_absent_recovery.png")

    # ---- 07 low vs high temperature ---------------------------------------- #
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for stratum, ls in (("LOW_T", "-"), ("HIGH_T", "--")):
        s = tt[(tt.kind == "RANDOM") & (tt.method == "M_REFIT")
               & (tt.stratum == stratum)]
        g = s.groupby("budget").mean_tvd.mean()
        zs = tt[(tt.stratum == stratum) & (tt.method == "M_ZERO")].mean_tvd.mean()
        line, = ax.plot([0] + list(g.index), [zs] + list(g.values), "o" + ls, ms=5,
                        lw=1.8, label="M_REFIT %s" % stratum)
        ax.axhline(zs, color=line.get_color(), ls=":", lw=1,
                   label="zero-shot %s %.4f" % (stratum, zs))
    budget_axis(ax, B)
    finish(fig, ax,
           os.path.join(C.D_FIG, "Fig_I_07_low_vs_high_temperature_recovery.png"),
           "LOW_T 300-600 degC (the NBNI3 window) vs HIGH_T 700-1000 degC",
           "mean TVD")

    # ---- 08 random vs maximin ---------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([0] + list(m.budget), [zero] + list(m.mean_tvd_mean), "s-", ms=6, lw=2,
            color=PALETTE["M_REFIT"], label="RANDOM (preregistered primary, mean of 3)")
    ax.fill_between(list(m.budget), list(m.mean_tvd_min), list(m.mean_tvd_max),
                    color=PALETTE["M_REFIT"], alpha=0.15)
    mm = maxi[maxi.method == "M_REFIT"].sort_values("budget")
    ax.plot([0] + list(mm.budget), [zero] + list(mm.mean_tvd), "^--", ms=6, lw=1.8,
            color="#b5651d", label="MAXIMIN (secondary comparison only)")
    ax.axhline(zero, color=PALETTE["M_ZERO"], ls="--", lw=1, label="M_ZERO")
    ax.axhline(triv, color=PALETTE["M_TRIVIAL"], ls=":", lw=1.4, label="M_TRIVIAL")
    budget_axis(ax, B)
    finish(fig, ax, os.path.join(C.D_FIG, "Fig_I_08_random_vs_maximin.png"),
           "Random is the preregistered primary strategy; maximin is never promoted\n"
           "on the basis of this comparison", "mean TVD")

    # ---- 09 anchor topology coverage ---------------------------------------- #
    rnd = cov[cov.kind == "RANDOM"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for i, r in enumerate(C.REPLICATES):
        g = rnd[rnd.replicate == r].sort_values("budget")
        axes[0].plot(g.budget, g.fraction_anchor_states_NBNI3, "o-", ms=5,
                     color=REP_COLORS[i], label="replicate %d" % r)
        axes[1].plot(g.budget, g.fraction_anchor_states_true_OTHER_SOLID, "o-", ms=5,
                     color=REP_COLORS[i], label="replicate %d" % r)
    for ax, lab in zip(axes, ["fraction of anchor states containing NBNI3",
                              "fraction of anchor states with true OTHER_SOLID"]):
        budget_axis(ax, B)
        ax.set_xlabel("anchor compositions (budget)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("What the frozen random sequences happened to contain\n"
                 "(recorded, never used to choose anchors)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(C.D_FIG, "Fig_I_09_anchor_topology_coverage.png"), dpi=160)
    plt.close(fig)
    print("    Fig_I_09_anchor_topology_coverage.png")

    # ---- 10 information efficiency summary ----------------------------------- #
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    g = agg[agg.method == "M_REFIT"].sort_values("budget")
    ax.plot([0] + list(g.budget), [zero] + list(g.mean_tvd_mean), "s-", ms=6, lw=2.2,
            color=PALETTE["M_REFIT"], label="M_REFIT (mean of 3)")
    g0 = agg[agg.method == "R0"].sort_values("budget")
    ax.plot(g0.budget, g0.mean_tvd_mean, "o-", ms=5, lw=1.6, color=PALETTE["R0"],
            label="R0 anchor-only control")
    ax.axhline(zero, color=PALETTE["M_ZERO"], ls="--", lw=1.2,
               label="M_ZERO %.4f" % zero)
    ax.axhline(triv, color=PALETTE["M_TRIVIAL"], ls=":", lw=1.6,
               label="M_TRIVIAL %.4f" % triv)
    ax.axhline(thr["recovery25_target_tvd"], color="#8b1a1a", ls="-.", lw=1.2,
               label="25%% reduction target %.4f" % thr["recovery25_target_tvd"])
    budget_axis(ax, B)
    txt = ("B_skill = %s\nB_recovery25 = %s\n(both measured on the same %d states)"
           % (thr["B_skill"], thr["B_recovery25"], C.N_RECOVERY_STATES))
    ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#bbbbbb"))
    finish(fig, ax,
           os.path.join(C.D_FIG, "Fig_I_10_information_efficiency_summary.png"),
           "Direct Ni anchors: how many, and does the network add anything?",
           "mean TVD")

    print("STAGE 6: PASS - 10 figures written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
