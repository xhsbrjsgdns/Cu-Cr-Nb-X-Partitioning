# -*- coding: utf-8 -*-
"""
76_f5f_figures.py
STEP5F Stage F: the ten preregistered figures (spec 65), matplotlib Agg only,
no styling experiments - plain, readable, JMST-suppement-level.

Fig_S5F_01 CALPHAD convergence            Fig_S5F_06 single-addition ablation
Fig_S5F_02 new-element topology           Fig_S5F_07 minus-one ablation
Fig_S5F_03 Z1 vs FULL5 per element        Fig_S5F_08 identity-count story
Fig_S5F_04 mean TVD before/after          Fig_S5F_09 error reduction by element
Fig_S5F_05 topology accuracy before/after Fig_S5F_10 final pipeline decision
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
import f5f_common as C

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})


def save(fig, name):
    p = os.path.join(C.D_FIG, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % name)


def main():
    met = pd.read_csv(os.path.join(C.D_ML, "STEP5F_AUGMENTED_LOEO_METRICS.csv"))
    abl = pd.read_csv(os.path.join(C.D_ABL, "STEP5F_IDENTITY_ABLATION.csv"))
    gate = pd.read_csv(os.path.join(C.D_CALQA, "STEP5F_FULL_CONVERGENCE_GATE.csv"))
    lo = json.load(open(os.path.join(C.D_ML, "STEP5F_AUGMENTED_LOEO_SUMMARY.json"),
                        encoding="utf-8"))
    data = pd.read_parquet(os.path.join(C.D_PROC, "STEP5F_FULL_NEW_ELEMENT_DATA.parquet"),
                           columns=["state_id", "element", "state_converged"] + C.X_TARGETS)
    print("=" * 78)
    print("STEP5F FULL  Stage F: figures")
    print("=" * 78)

    # 01 convergence
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.bar(gate.element, gate.converged_states, color="#4477aa", label="converged")
    ax.bar(gate.element, gate.failed_states, bottom=gate.converged_states,
           color="#ee6677", label="failed")
    ax.axhline(22000, ls="--", c="k", lw=0.8)
    ax.text(4.35, 22100, "22000 = 2000 x 11", fontsize=7)
    for _, r in gate.iterrows():
        ax.text(r.element, 22500, "%.4f" % r.convergence_rate, ha="center", fontsize=7)
    ax.set_ylim(0, 24000)
    ax.set_ylabel("states")
    ax.set_title("Fig_S5F_01  CALPHAD convergence per new element")
    ax.legend(fontsize=7)
    save(fig, "Fig_S5F_01_CALPHAD_convergence.png")

    # 02 new-element topology
    cv = data[data.state_converged == 1].copy()
    xp = cv[C.X_TARGETS].to_numpy(float)
    ok = np.isfinite(xp).all(axis=1) & (xp.sum(axis=1) > 0)
    cv = cv[ok]
    dom = np.array([C.FAMILIES_ORDER[i] for i in xp[ok].argmax(axis=1)], dtype=object)
    cv["dom"] = dom
    ct = pd.crosstab(cv.element, cv.dom, normalize="index").reindex(
        columns=C.FAMILIES_ORDER, fill_value=0.0)
    colors = {"FCC": "#4477aa", "BCC": "#66ccee", "C15": "#228833",
              "OTHER_SOLID": "#ccbb44", "LIQUID": "#ee6677"}
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bottom = np.zeros(len(ct))
    for f in C.FAMILIES_ORDER:
        ax.bar(ct.index, ct[f], bottom=bottom, label=f, color=colors[f])
        bottom += ct[f].to_numpy()
    ax.set_ylabel("fraction of converged states")
    ax.set_title("Fig_S5F_02  dominant X-partition topology (new elements)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "Fig_S5F_02_new_element_topology.png")

    # 03 Z1 vs FULL5 per element
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(met))
    ax.bar(x - 0.2, met.Z1_mean_tvd, 0.4, label="Z1 (frozen STEP5D)")
    ax.bar(x + 0.2, met.FULL5_mean_tvd, 0.4, label="FULL5 (augmented)")
    ax.set_xticks(x, met.heldout_element)
    ax.set_ylabel("heldout mean TVD")
    ax.set_title("Fig_S5F_03  Z1 vs FULL5 per original heldout element")
    ax.legend(fontsize=7)
    save(fig, "Fig_S5F_03_Z1_vs_FULL5_per_element.png")

    # 04 mean TVD before/after
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(["Z1", "FULL5"], [lo["mean_Z1_tvd"], lo["mean_FULL5_tvd"]],
           color=["#999999", "#4477aa"])
    ax.axhline(C.PRIMARY_SUCCESS_TVD, ls="--", c="#ee6677")
    ax.text(0.5, C.PRIMARY_SUCCESS_TVD + 0.008,
            "primary threshold %.6f" % C.PRIMARY_SUCCESS_TVD, ha="center",
            fontsize=7, color="#ee6677")
    ax.set_ylabel("6-fold mean TVD")
    ax.set_title("Fig_S5F_04  mean TVD before/after augmentation")
    save(fig, "Fig_S5F_04_mean_TVD_before_after.png")

    # 05 topology accuracy before/after
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(["Z1", "FULL5"], [met.Z1_topology_accuracy.mean(),
                             met.FULL5_topology_accuracy.mean()],
           color=["#999999", "#4477aa"])
    ax.axhline(C.TOPOLOGY_ACC_CRITERION, ls="--", c="#ee6677")
    ax.text(0.5, C.TOPOLOGY_ACC_CRITERION + 0.005,
            "criterion %.2f" % C.TOPOLOGY_ACC_CRITERION, ha="center", fontsize=7,
            color="#ee6677")
    ax.set_ylabel("mean topology accuracy")
    ax.set_title("Fig_S5F_05  topology accuracy before/after")
    save(fig, "Fig_S5F_05_topology_accuracy_before_after.png")

    # 06 single addition
    sing = {x: abl[abl.config == "BASE_PLUS_%s" % x].mean_tvd.mean()
            for x in C.SELECTED_ELEMENTS}
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(list(sing), list(sing.values()), color="#4477aa")
    ax.axhline(lo["mean_Z1_tvd"], ls="--", c="#999999", label="Z1")
    ax.axhline(lo["mean_FULL5_tvd"], ls="--", c="#ee6677", label="FULL5")
    ax.set_ylabel("6-fold mean TVD")
    ax.set_title("Fig_S5F_06  single-addition ablation (BASE_PLUS_X)")
    ax.legend(fontsize=7)
    save(fig, "Fig_S5F_06_single_addition_ablation.png")

    # 07 minus one
    mins = {x: abl[abl.config == "FULL5_MINUS_%s" % x].mean_tvd.mean()
            for x in C.SELECTED_ELEMENTS}
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(list(mins), list(mins.values()), color="#228833")
    ax.axhline(lo["mean_FULL5_tvd"], ls="--", c="#ee6677", label="FULL5")
    ax.set_ylabel("6-fold mean TVD")
    ax.set_title("Fig_S5F_07  minus-one ablation (FULL5_MINUS_X)")
    ax.legend(fontsize=7)
    save(fig, "Fig_S5F_07_minus_one_ablation.png")

    # 08 identity count story
    pts = [("5 (Z1)", 5, lo["mean_Z1_tvd"])]
    for x in C.SELECTED_ELEMENTS:
        pts.append(("+%s" % x, 6, float(abl[abl.config == "BASE_PLUS_%s" % x].mean_tvd.mean())))
    for x in C.SELECTED_ELEMENTS:
        pts.append(("FULL5-%s" % x, 9, float(abl[abl.config == "FULL5_MINUS_%s" % x].mean_tvd.mean())))
    pts.append(("10 (FULL5)", 10, lo["mean_FULL5_tvd"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.scatter(xs, ys, c=["#999999"] + ["#4477aa"] * 5 + ["#228833"] * 5 + ["#ee6677"],
               zorder=3)
    for lbl, xx, yy in pts:
        ax.annotate(lbl, (xx, yy), fontsize=6, rotation=45,
                    textcoords="offset points", xytext=(2, 3))
    ax.axhline(C.PRIMARY_SUCCESS_TVD, ls="--", c="#ee6677", lw=0.8)
    ax.set_xlabel("visible training X identities")
    ax.set_ylabel("6-fold mean TVD")
    ax.set_title("Fig_S5F_08  identity-count story")
    save(fig, "Fig_S5F_08_identity_count_story.png")

    # 09 error reduction by element
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    vals = 100 * met.relative_tvd_reduction_vs_Z1
    ax.bar(met.heldout_element, vals,
           color=["#228833" if v > 0 else "#ee6677" for v in vals])
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("TVD reduction vs Z1 (%)")
    ax.set_title("Fig_S5F_09  error reduction by element")
    save(fig, "Fig_S5F_09_error_reduction_by_element.png")

    # 10 final decision
    tree = pd.read_csv(os.path.join(C.D_ABL, "STEP5F_TREE_SECONDARY.csv"))
    rec = json.load(open(os.path.join(C.ROOT, "STEP5F_FULL_SUMMARY.json"),
                         encoding="utf-8"))["recommendation"] \
        if os.path.exists(os.path.join(C.ROOT, "STEP5F_FULL_SUMMARY.json")) else ""
    checks = [("primary TVD <= %.6f" % C.PRIMARY_SUCCESS_TVD, lo["primary_success"]),
              ("topology accuracy >= %.2f" % C.TOPOLOGY_ACC_CRITERION,
               lo["topology_criterion"]),
              (">= %d/6 beat trivial" % C.FOLDS_BEAT_TRIVIAL_MIN, lo["skill_criterion"]),
              ("strong improvement", lo["strong_improvement"]),
              ("tree same direction (>=4/6)", int(tree.same_direction_as_MLP.sum()) >= 4)]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.axis("off")
    y = 0.9
    ax.text(0.02, y + 0.08, "RECOMMENDATION: %s" % rec, fontsize=10,
            fontweight="bold", transform=ax.transAxes)
    for lbl, ok in checks:
        ax.text(0.02, y, ("[PASS] " if ok else "[FAIL] ") + lbl, fontsize=9,
                color="#228833" if ok else "#ee6677", transform=ax.transAxes)
        y -= 0.13
    ax.set_title("Fig_S5F_10  final pipeline decision")
    save(fig, "Fig_S5F_10_final_pipeline_decision.png")
    print("STAGE F: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
