# -*- coding: utf-8 -*-
"""
124_j_figures.py
STEP5J Stage 4: the twelve publication figures (spec 20, 21).

Each figure is exported as PNG at 600 dpi, PDF and SVG with LIVE TEXT (Type 42 / no
path conversion), plus the exact source CSV that produced it, so a co-author can edit
the labels or re-plot without asking anyone for the numbers.

Layout discipline (spec 21): constrained layout everywhere so nothing is clipped;
annotation boxes are placed outside the data envelope rather than over the curves -
the STEP5I review noted that its Fig_I_10 threshold box sat on top of the low-TVD part
of the learning curve, and that is fixed here rather than repeated.

Figures that cannot be built from frozen files are not invented: the figure inventory
records them as NOT_AVAILABLE with the reason.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    # opaque legend: reference lines drawn across the axes must never strike through
    # legend text (the STEP5I review flagged exactly this class of defect)
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.facecolor": "white",
    "legend.edgecolor": "#d0d0d0", "legend.borderpad": 0.45,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.constrained_layout.use": True, "savefig.bbox": None,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j_common as C

INVENTORY = []
PAL = {"M_REFIT": "#2f6f9f", "R0": "#c26a2a", "R1": "#7a9a3a", "R2": "#8b4f9e",
       "zero": "#333333", "trivial": "#8a8a8a", "maximin": "#b5651d",
       "FCC": "#3a7ca5", "BCC": "#8c6d31", "C15": "#a4453f",
       "OTHER_SOLID": "#5b8c5a", "LIQUID": "#7a6ea8"}
REP_C = ["#2f6f9f", "#66a5c9", "#17435f"]


def save(fig, num, name, source_df, caption, level, sources, notes=""):
    stem = "Figure_%02d_%s" % (num, name)
    paths = {}
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        p = C.P(C.D_FIGS, "%s.%s" % (stem, ext))
        fig.savefig(p, **kw)
        paths[ext] = p
    plt.close(fig)
    csv_p = C.P(C.D_FIGS, "%s_SOURCE.csv" % stem)
    source_df.to_csv(csv_p, index=False, encoding="utf-8")
    INVENTORY.append({"figure": num, "name": name, "status": "OK", "caption": caption,
                      "evidence_level": level, "png": paths["png"], "pdf": paths["pdf"],
                      "svg": paths["svg"], "source_csv": csv_p,
                      "png_sha256": C.sha256_file(paths["png"]),
                      "source_csv_sha256": C.sha256_file(csv_p),
                      "sources": "; ".join(sources), "notes": notes})
    print("  Figure %2d  %s" % (num, name))


def skip(num, name, reason, requires_experiment=False):
    INVENTORY.append({"figure": num, "name": name,
                      "status": C.REQUIRES_NEW_EXPERIMENT if requires_experiment
                      else C.NOT_AVAILABLE,
                      "caption": "", "evidence_level": "", "png": "", "pdf": "",
                      "svg": "", "source_csv": "", "png_sha256": "",
                      "source_csv_sha256": "", "sources": "", "notes": reason})
    print("  Figure %2d  %s -> %s (%s)" % (num, name, C.NOT_AVAILABLE, reason))


def budget_axis(ax, budgets=None):
    budgets = budgets or C.BUDGETS
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0] + list(budgets))
    ax.set_xticklabels(["0"] + [str(b) for b in budgets])
    ax.set_xlabel("Ni anchor compositions (each = 11 CALPHAD states)")


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5J  Stage 4: publication figures (PNG 600 dpi + PDF + SVG + source CSV)")
    print("=" * 78)

    N = json.load(open(C.P(C.D_FREEZE, "FINAL_NUMERICAL_RESULTS.json"), encoding="utf-8"))
    L1, L2 = N["level_1_prospective_primary"], N["level_2_prospective_secondary"]
    L3, L4 = N["level_3_postblind_recovery"], N["level_4_postblind_descriptive"]
    CTX = N["context_not_primary_evidence"]
    zero = L3["zero_shot_tvd_same_subset"]["value"]
    triv = L3["trivial_tvd_same_subset"]["value"]
    zero_topo = L3["zero_shot_topology_accuracy_same_subset"]["value"]
    triv_topo = L3["trivial_topology_accuracy_same_subset"]["value"]
    refit = {int(k): v for k, v in L3["random_M_REFIT_mean_tvd_by_budget"]["value"].items()}
    tacc = {int(k): v for k, v in L3["random_topology_accuracy_by_budget"]["value"].items()}
    r0 = {int(k): v for k, v in L3["R0_by_budget"]["value"].items()}
    r1 = {int(k): v for k, v in L3["R1_by_budget"]["value"].items()}
    r2 = {int(k): v for k, v in L3["R2_by_budget"]["value"].items()}
    mx = {int(k): v for k, v in L3["maximin_by_budget"]["value"].items()}
    rep = C.load_csv("I_RANDOM_REP")
    B = C.BUDGETS

    # ---- Figure 1: workflow and evidence hierarchy -------------------------- #
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axis("off")
    ax.grid(False)
    stages = [
        ("FULL_V1", "550,000 CALPHAD states\n6 X identities + baseline", "#dce9f2"),
        ("STEP5A", "43-feature schema\nIID models frozen", "#dce9f2"),
        ("STEP5B", "composition-domain OOD\n(Domain B shell)", "#dce9f2"),
        ("STEP5C-E", "retrospective LOEO\nfailure diagnosis, pipeline choice", "#dce9f2"),
        ("STEP5F", "chemical-identity augmentation\n(+5 identities)", "#dce9f2"),
        ("STEP5G", "production freeze\nNi preregistered, still sealed", "#f6e2c8"),
        ("STEP5H-A", "Ni features + predictions\nfrozen BEFORE any Ni CALPHAD", "#f6e2c8"),
        ("STEP5H-B", "Ni CALPHAD reveal\nLEVEL 1 prospective result", "#f2cfcf"),
        ("STEP5I", "post-blind few-shot recovery\nLEVEL 3 + LEVEL 4", "#dcecdc"),
        ("STEP5J", "evidence freeze\nclaim audit", "#e8e8e8"),
    ]
    y = 0.93
    for i, (name, desc, col) in enumerate(stages):
        ax.add_patch(plt.Rectangle((0.02, y - 0.075), 0.40, 0.070, facecolor=col,
                                   edgecolor="#666666", linewidth=0.7))
        ax.text(0.04, y - 0.040, name, fontsize=8.5, fontweight="bold", va="center")
        ax.text(0.145, y - 0.040, desc, fontsize=7.4, va="center")
        if i < len(stages) - 1:
            ax.annotate("", xy=(0.22, y - 0.078), xytext=(0.22, y - 0.088),
                        arrowprops=dict(arrowstyle="-|>", color="#666666", lw=0.8))
        y -= 0.093
    lv = [("LEVEL 1", "prospective PRIMARY\nSTEP5H-B PRIMARY only", "#f2cfcf"),
          ("LEVEL 2", "prospective SECONDARY\nFULL11 sensitivity", "#f7dede"),
          ("LEVEL 3", "post-blind preregistered\nfew-shot, maximin, R0/R1/R2", "#dcecdc"),
          ("LEVEL 4", "post-blind descriptive\nNBNI3, topology, temperature", "#eef4ee")]
    ax.text(0.52, 0.95, "Evidence hierarchy (never mixed)", fontsize=9,
            fontweight="bold")
    yy = 0.86
    for name, desc, col in lv:
        ax.add_patch(plt.Rectangle((0.52, yy - 0.13), 0.46, 0.125, facecolor=col,
                                   edgecolor="#666666", linewidth=0.7))
        ax.text(0.54, yy - 0.037, name, fontsize=8.5, fontweight="bold")
        ax.text(0.54, yy - 0.085, desc, fontsize=7.4, va="center")
        yy -= 0.155
    ax.text(0.52, 0.20, "Retrospective LOEO (STEP5C/5E/5F) is model-family context,\n"
                        "not a score of the frozen production checkpoints.",
            fontsize=7.2, style="italic", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, 1, "workflow_and_evidence_hierarchy",
         pd.DataFrame([{"stage": s, "description": d.replace("\n", " ")}
                       for s, d, _ in stages]),
         "Workflow from the CALPHAD dataset to the prospective blind test, and the "
         "four evidence levels that are kept apart throughout.", "all levels",
         [C.cite("G_FREEZE_JSON")])

    # ---- Figure 2: topology landscape across known X identities ------------- #
    v1 = C.load_csv("V1_TOPOLOGY")
    a = v1[v1.sampling_domain == "A"]
    piv = a.pivot_table(index="family", columns="dominant_X_solid_partition_family",
                        values="n_states", aggfunc="sum", fill_value=0)
    share = piv.div(piv.sum(axis=1), axis=0).reindex(columns=[f for f in C.FAMILIES_ORDER
                                                             if f in piv.columns],
                                                     fill_value=0.0)
    ni_true = L1["true_topology_distribution"]["value"]
    ni_pred = L1["pred_topology_distribution"]["value"]
    share.loc["Ni (true, blind)"] = [ni_true.get(f, 0.0) for f in share.columns]
    share.loc["Ni (predicted)"] = [ni_pred.get(f, 0.0) for f in share.columns]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bottom = np.zeros(len(share))
    for f in share.columns:
        ax.bar(range(len(share)), share[f].to_numpy(), bottom=bottom, width=0.72,
               color=PAL.get(f, "#999999"), edgecolor="white", linewidth=0.6, label=f)
        bottom += share[f].to_numpy()
    ax.set_xticks(range(len(share)))
    ax.set_xticklabels(share.index, rotation=20, ha="right")
    ax.set_ylabel("share of states by dominant X-partition family")
    ax.set_ylim(0, 1)
    ax.axvline(len(share) - 2.5, color="#444444", lw=0.9, ls="--")
    ax.set_title("Dominant X-partition topology: six training identities (Domain A) "
                 "versus the blind Ni result")
    ax.legend(ncol=len(share.columns), fontsize=7.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, 2, "topology_landscape_across_identities", share.reset_index(),
         "Each training identity occupies a different region of topology space. Ni's "
         "true topology contains no C15-dominant state, yet the frozen model assigns "
         "C15 to most Ni states.", "context + LEVEL 1",
         [C.cite("V1_TOPOLOGY"), C.cite("HB_PRIMARY_METRICS")])

    # ---- Figure 3: IID vs Domain-B vs LOEO vs prospective -------------------- #
    lad = C.load_csv("C_LADDER")
    x = lad[lad.task == "x_partition"].iloc[0]
    bars = [("STEP5A\nIID test", float(x.STEP5A_IID), "#7fb3d3"),
            ("STEP5B\ncomposition OOD", float(x.STEP5B_composition_domain_OOD), "#5b93b8"),
            ("STEP5C LOEO\n(6-fold mean)", float(x.STEP5C_LOEO_mean), "#c9a227"),
            ("STEP5E Z1 LOEO\n(selected family)",
             C.LOEO_CONTEXT["original_six_Z1_MLP_DIRECT_mean_tvd"], "#c9a227"),
            ("STEP5F FULL5 LOEO\n(augmented)", C.LOEO_CONTEXT["full5_augmented_mean_tvd"],
             "#c9a227"),
            ("STEP5H-B Ni\nPROSPECTIVE", L1["mean_tvd"]["value"], "#b5453f"),
            ("STEP5H-B Ni\nSECONDARY", L2["mean_tvd"]["value"], "#d08a86")]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars],
           width=0.66, edgecolor="#444444", linewidth=0.5)
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v + 0.018, "%.4f" % v, ha="center", fontsize=7.5)
    ax.axhline(L1["trivial_tvd"]["value"], color=PAL["trivial"], ls=":", lw=1.4,
               label="PRIMARY trivial baseline %.4f" % L1["trivial_tvd"]["value"])
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.4)
    ax.set_ylabel("x-partition mean TVD")
    ax.set_ylim(0, 1.0)
    ax.set_title("Escalating generalization demand. Yellow bars are retrospective "
                 "model-family context, not checkpoint scores")
    ax.legend(loc="upper left")
    save(fig, 3, "IID_vs_compositionOOD_vs_LOEO_vs_prospective",
         pd.DataFrame([{"stage": b[0].replace("\n", " "), "x_partition_mean_tvd": b[1]}
                       for b in bars]),
         "The same metric across four increasingly demanding regimes. Only the red bars "
         "are prospective.", "context + LEVEL 1 + LEVEL 2",
         [C.cite("C_LADDER"), C.cite("HB_PRIMARY_METRICS")],
         notes=C.LOEO_CONTEXT["caveat"])

    # ---- Figure 4: LOEO failure diagnosis ----------------------------------- #
    per_el = C.load_csv("C_PER_ELEMENT").set_index("heldout_element")
    tasks = ["fraction", "nb_partition", "cr_partition", "x_partition"]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    w = 0.2
    for i, t in enumerate(tasks):
        ax.bar(np.arange(len(C.ORIGINAL_SIX)) + (i - 1.5) * w,
               [per_el.loc[e, t] for e in C.ORIGINAL_SIX], width=w, label=t,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(C.ORIGINAL_SIX)))
    ax.set_xticklabels(C.ORIGINAL_SIX)
    ax.set_xlabel("held-out X identity")
    ax.set_ylabel("mean TVD (retrospective LOEO)")
    ax.set_title("Retrospective LOEO: only the X-partition task collapses")
    ax.legend(ncol=4, fontsize=7.5)
    save(fig, 4, "LOEO_failure_diagnosis", per_el.reset_index(),
         "Phase-fraction and Cr/Nb partition survive element hold-out; the X-partition "
         "task does not. This is why the X-partition is the target of the prospective "
         "test.", "context", [C.cite("C_PER_ELEMENT")])

    # ---- Figure 5: STEP5F augmentation --------------------------------------- #
    fa = C.load_csv("F_AUG_METRICS")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    idx = np.arange(len(fa))
    ax.bar(idx - 0.2, fa.Z1_mean_tvd, width=0.4, label="Z1 original six", color="#8aa6bd")
    ax.bar(idx + 0.2, fa.FULL5_mean_tvd, width=0.4, label="FULL5 augmented",
           color="#3c6d99")
    ax.plot(idx, fa.trivial_tvd, "kd--", ms=5, lw=1.0, label="fold trivial baseline")
    ax.set_xticks(idx)
    ax.set_xticklabels(fa.heldout_element)
    ax.set_xlabel("held-out X identity")
    ax.set_ylabel("mean TVD")
    ax.set_title("Chemical-identity augmentation: real but insufficient "
                 "(3 of 6 folds beat trivial)")
    ax.legend(fontsize=7.5)
    save(fig, 5, "chemical_identity_augmentation", fa,
         "Adding five identities lowers the retrospective LOEO mean TVD but does not "
         "clear the preregistered switching gates.", "context",
         [C.cite("F_AUG_METRICS"), C.cite("F_AUG_SUMMARY")])

    # ---- Figure 6: prospective Ni zero-shot reveal ---------------------------- #
    per_T = C.load_csv("HB_PER_TEMP")
    tcol = next((c for c in per_T.columns if "emperature" in c), per_T.columns[0])
    vcol = next((c for c in per_T.columns if "mean_tvd" in c.lower()), None)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    ax = axes[0]
    ax.bar([0, 1], [L1["mean_tvd"]["value"], L2["mean_tvd"]["value"]],
           color=["#b5453f", "#d08a86"], width=0.55, edgecolor="#444444", linewidth=0.5)
    ci = L1["ci95"]["value"]
    ax.errorbar([0], [L1["mean_tvd"]["value"]],
                yerr=[[max(0.0, L1["mean_tvd"]["value"] - ci[0])],
                      [max(0.0, ci[1] - L1["mean_tvd"]["value"])]],
                fmt="none", ecolor="#222222", capsize=4, lw=1.0)
    ax.axhline(L1["trivial_tvd"]["value"], color=PAL["trivial"], ls=":", lw=1.5,
               label="PRIMARY trivial %.4f" % L1["trivial_tvd"]["value"])
    ax.axhline(L2["own_trivial_tvd"]["value"], color="#bdbdbd", ls="--", lw=1.2,
               label="SECONDARY trivial %.4f" % L2["own_trivial_tvd"]["value"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["PRIMARY\n(LEVEL 1)", "SECONDARY\n(LEVEL 2)"])
    ax.set_ylabel("mean TVD over 22,000 blind states")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)
    ax.set_title("Prospective Ni zero-shot")
    if vcol:
        axes[1].plot(per_T[tcol], per_T[vcol], "o-", color="#b5453f", ms=4)
        axes[1].axhline(L1["trivial_tvd"]["value"], color=PAL["trivial"], ls=":", lw=1.4)
        axes[1].set_xlabel("temperature (degC)")
        axes[1].set_ylabel("PRIMARY mean TVD")
        axes[1].set_title("Error is present at every temperature")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, C.NOT_AVAILABLE, ha="center")
    save(fig, 6, "prospective_Ni_zero_shot_reveal", per_T,
         "The primary result, with its frozen composition-level bootstrap interval, and "
         "its temperature profile. Both models sit above their own trivial baselines.",
         "LEVEL 1 and LEVEL 2",
         [C.cite("HB_PRIMARY_METRICS"), C.cite("HB_PER_TEMP")])

    # ---- Figure 7: true vs predicted Ni topology ------------------------------ #
    fams = C.FAMILIES_ORDER
    tv = [L1["true_topology_distribution"]["value"].get(f, 0.0) for f in fams]
    pv = [L1["pred_topology_distribution"]["value"].get(f, 0.0) for f in fams]
    sv = [L2["pred_topology_distribution"]["value"].get(f, 0.0) for f in fams]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    idx = np.arange(len(fams))
    ax.bar(idx - 0.26, tv, width=0.25, label="true (CALPHAD)", color="#3a7ca5")
    ax.bar(idx, pv, width=0.25, label="PRIMARY predicted", color="#b5453f")
    ax.bar(idx + 0.26, sv, width=0.25, label="SECONDARY predicted", color="#d08a86")
    nearest = L4["topology_inheritance_observation"]["value"]
    ax.axhline(nearest["C15_share_of_that_identity_domain_A"], color="#666666",
               ls="-.", lw=1.1,
               label="C15 share of %s, the training identity nearest Ni in descriptor "
                     "space" % nearest["nearest_training_identity_in_descriptor_space"])
    ax.set_xticks(idx)
    ax.set_xticklabels(fams)
    ax.set_ylabel("share of the 22,000 blind states")
    ax.set_title("Topology selection, not small partition error, is what failed")
    ax.legend(fontsize=7.2)
    save(fig, 7, "true_vs_predicted_Ni_topology",
         pd.DataFrame({"family": fams, "true": tv, "primary_predicted": pv,
                       "secondary_predicted": sv}),
         "No Ni state is C15-dominant, yet C15 is the model's modal prediction. The "
         "dashed line is the C15 share of the nearest training identity - a descriptive "
         "association, not a demonstrated mechanism.", "LEVEL 1 + LEVEL 4",
         [C.cite("HB_PRIMARY_METRICS"), C.cite("V1_TOPOLOGY"), C.cite("HA_OOD_FREEZE")])

    # ---- Figure 8: few-shot TVD learning curve -------------------------------- #
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, r in enumerate(C.REPLICATES):
        g = rep[(rep.method == "M_REFIT") & (rep.replicate == r)].sort_values("budget")
        ax.plot([0] + list(g.budget), [zero] + list(g.mean_tvd), "o-", ms=3.6, lw=1.0,
                color=REP_C[i], alpha=0.9, label="random replicate %d" % r)
    ax.plot([0] + B, [zero] + [refit[b] for b in B], "s-", ms=6, lw=2.2,
            color="#12324a", label="mean of 3 replicates")
    ax.axhline(zero, color=PAL["zero"], ls="--", lw=1.2,
               label="frozen zero-shot %.4f" % zero)
    ax.axhline(triv, color=PAL["trivial"], ls=":", lw=1.6,
               label="trivial baseline %.4f" % triv)
    budget_axis(ax)
    ax.set_ylabel("mean TVD on the 15,400-state recovery subset")
    ax.set_title("Post-blind few-shot recovery (uses real Ni labels; not a zero-shot "
                 "result)")
    ax.legend(fontsize=7.2, loc="upper right")
    save(fig, 8, "fewshot_TVD_learning_curve",
         rep[rep.method == "M_REFIT"][["replicate", "budget", "mean_tvd",
                                       "topology_accuracy"]],
         "Every replicate is shown alongside the mean. Budget 0 is the frozen zero-shot "
         "value on the identical subset.", "LEVEL 3",
         [C.cite("I_RANDOM_REP"), C.cite("I_BASELINES")])

    # ---- Figure 9: M_REFIT vs R0 / R1 / R2 ------------------------------------ #
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for meth, series in (("M_REFIT", refit), ("R0", r0), ("R1", r1), ("R2", r2)):
        ax.plot(B, [series[b] for b in B], "o-", ms=5, lw=1.8, color=PAL[meth],
                label=meth)
    ax.axhline(triv, color=PAL["trivial"], ls=":", lw=1.5, label="trivial baseline")
    first_r2 = L3["first_budget_M_REFIT_below_R2"]["value"]
    first_r0 = L3["first_budget_M_REFIT_below_R0"]["value"]
    for b, lab, col in ((first_r2, "first below R2", PAL["R2"]),
                        (first_r0, "first below R0", PAL["R0"])):
        if b:
            ax.axvline(b, color=col, ls="--", lw=0.9, alpha=0.7)
            ax.annotate(lab, xy=(b, ax.get_ylim()[1]), xytext=(0, -10),
                        textcoords="offset points", ha="center", fontsize=7, color=col)
    budget_axis(ax)
    ax.set_ylabel("mean TVD")
    ax.set_title("Anchor information dominates at small budgets; refitting takes over "
                 "from 8 anchors")
    ax.legend(fontsize=7.5)
    save(fig, 9, "refit_vs_R0_R1_R2",
         C.load_csv(C.P(C.D_TABLES, "FINAL_CONTROL_CROSSOVER_TABLE.csv")),
         "R0 discards the model entirely. Crossovers are post-blind descriptive "
         "observations, not preregistered endpoints.", "LEVEL 3",
         [C.cite("I_RANDOM_AGG"), C.cite("I_CONTROLS")])

    # ---- Figure 10: topology accuracy recovery -------------------------------- #
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot([0] + B, [zero_topo] + [tacc[b] for b in B], "s-", ms=6, lw=2.2,
            color="#2f6f9f", label="M_REFIT (mean of 3)")
    for i, r in enumerate(C.REPLICATES):
        g = rep[(rep.method == "M_REFIT") & (rep.replicate == r)].sort_values("budget")
        ax.plot(g.budget, g.topology_accuracy, "o--", ms=3.4, lw=0.8, color=REP_C[i],
                alpha=0.8, label="replicate %d" % r)
    ax.axhline(zero_topo, color=PAL["zero"], ls="--", lw=1.2,
               label="frozen zero-shot %.4f" % zero_topo)
    ax.axhline(triv_topo, color=PAL["trivial"], ls=":", lw=1.6,
               label="trivial baseline %.4f" % triv_topo)
    budget_axis(ax)
    ax.set_ylabel("topology accuracy (argmax family)")
    ax.set_ylim(0, 1.02)
    ax.set_title("The trivial baseline already beats the frozen model on topology; "
                 "one anchor composition overturns that")
    ax.legend(fontsize=7.2, loc="lower right")
    save(fig, 10, "topology_accuracy_recovery_curve",
         pd.DataFrame([{"budget": 0, "topology_accuracy": zero_topo,
                        "row": "frozen zero-shot"}]
                      + [{"budget": b, "topology_accuracy": tacc[b],
                          "row": "mean of 3 replicates"} for b in B]),
         "Topology accuracy, the quantity that collapsed in the blind test.", "LEVEL 3",
         [C.cite("I_RANDOM_AGG"), C.cite("I_BASELINES")])

    # ---- Figure 11: FCC vs OTHER_SOLID and NBNI3 ------------------------------ #
    topo = C.load_csv("I_TOPO")
    nbs = C.load_csv("I_NBNI3")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, (df, strata, title) in zip(
            axes, [(topo, ["FCC", "OTHER_SOLID"], "by true topology"),
                   (nbs, ["NBNI3_ABSENT", "NBNI3_PRESENT"], "by NBNI3 status")]):
        for s, ls in zip(strata, ["-", "--"]):
            sub = df[(df.kind == "RANDOM") & (df.method == "M_REFIT") & (df.stratum == s)]
            g = sub.groupby("budget").mean_tvd.mean()
            z = df[(df.stratum == s) & (df.method == "M_ZERO")].mean_tvd.mean()
            line, = ax.plot([0] + list(g.index), [z] + list(g.values), "o" + ls, ms=4.5,
                            lw=1.6, label=s)
            ax.axhline(z, color=line.get_color(), ls=":", lw=0.9, alpha=0.7)
        ax.axhline(triv, color=PAL["trivial"], ls=":", lw=1.5, label="trivial baseline")
        ax.set_yscale("log")
        budget_axis(ax)
        ax.set_title(title)
        ax.legend(fontsize=7.2)
    axes[0].set_ylabel("mean TVD (log scale)")
    fig.suptitle("The NBNI3 / OTHER_SOLID stratum is the hardest, but every stratum "
                 "ends far below trivial", fontsize=10)
    save(fig, 11, "FCC_vs_OTHER_and_NBNI3_recovery",
         pd.concat([topo.assign(stratification="TRUE_TOPOLOGY"),
                    nbs.assign(stratification="NBNI3")], ignore_index=True),
         "Descriptive post-blind stratification. The anchor sequences were frozen "
         "before NBNI3 was known to exist.", "LEVEL 4",
         [C.cite("I_TOPO"), C.cite("I_NBNI3")])

    # ---- Figure 12: information efficiency summary ---------------------------- #
    ie = C.load_csv(C.P(C.D_TABLES, "FINAL_INFORMATION_EFFICIENCY_TABLE.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot([0] + B, [zero] + [refit[b] for b in B], "s-", ms=6, lw=2.2,
            color=PAL["M_REFIT"], label="M_REFIT (mean of 3)")
    ax.plot(B, [r0[b] for b in B], "o-", ms=5, lw=1.5, color=PAL["R0"],
            label="R0 anchor-only control")
    ax.axhline(zero, color=PAL["zero"], ls="--", lw=1.3, label="zero-shot %.4f" % zero)
    ax.axhline(triv, color=PAL["trivial"], ls=":", lw=1.6, label="trivial %.4f" % triv)
    ax.set_yscale("log")
    budget_axis(ax)
    ax.set_ylabel("mean TVD (log scale)")
    ax.set_ylim(min(min(refit.values()), min(r0.values())) * 0.55, 1.25)
    # the annotation sits ABOVE the curves in the empty upper band, never over them
    ax.text(0.02, 0.03,
            "B_skill = %s composition   B_recovery25 = %s composition\n"
            "1 composition = 11 CALPHAD temperature states\n"
            "recovery vs zero-shot: %s"
            % (L3["B_skill"]["value"], L3["B_recovery25"]["value"],
               ",  ".join("b%d %.1f%%" % (b, 100 * (1 - refit[b] / zero)) for b in B)),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f6f6f6", ec="#bbbbbb", lw=0.6))
    ax.set_title("Information efficiency of direct Ni thermodynamic anchors")
    ax.legend(fontsize=7.5, loc="upper right")
    save(fig, 12, "information_efficiency_summary", ie,
         "How much direct Ni data it takes, and how much of it the network rather than "
         "the anchors provides.", "LEVEL 3",
         [C.cite("I_RANDOM_AGG"), C.cite("I_THRESHOLDS")],
         notes="Annotation placed in the lower-left empty band so it never overlaps the "
               "curves.")

    inv = pd.DataFrame(INVENTORY)
    inv.to_csv(C.P(C.D_FIGS, "STEP5J_FIGURE_INVENTORY.csv"), index=False,
               encoding="utf-8")
    print("STAGE 4: PASS - %d figures, %d exported"
          % (len(inv), int((inv.status == "OK").sum())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
