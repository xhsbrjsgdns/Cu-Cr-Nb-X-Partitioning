# -*- coding: utf-8 -*-
"""
r3_60_figures.py
STEP5K-R3 stage 60: the four validation figures (spec 42).

Every panel carries a SOURCE csv. No decorative figure is produced: a panel is drawn
only when the data it needs exist. When nothing has converged, or when too few phase
compositions are matched to say anything, the stage writes FIGURE_STATUS.md naming
each panel that was not drawn and why, and draws nothing in its place.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 8.5,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
MIN_MATCHED_FOR_FIG3 = 12


def save(fig, name, src):
    fig.savefig(os.path.join(C.D_FIG, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(C.D_FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    src.to_csv(os.path.join(C.D_FIG, name + "_SOURCE.csv"), index=False,
               encoding="utf-8")
    print("    %s" % name)


def fig01(conv, raw):
    """Which phases each study reports, and which the description returns."""
    rows = []
    for sysname, g in conv.groupby("system"):
        exp, calc = {}, {}
        for _, r in g.iterrows():
            for p in str(r.experimental_raw_phase_set).split(";"):
                if p.strip():
                    exp[p.strip()] = exp.get(p.strip(), 0) + 1
            for p in str(r.calculated_raw_phase_set).split(";"):
                if p.strip():
                    calc[p.strip()] = calc.get(p.strip(), 0) + 1
        for p, n in exp.items():
            rows.append({"system": sysname, "phase": p, "side": "experimental",
                         "n_conditions": n})
        for p, n in calc.items():
            rows.append({"system": sysname, "phase": p, "side": "calculated",
                         "n_conditions": n})
    src = pd.DataFrame(rows)
    systems = sorted(src.system.unique())
    fig, axes = plt.subplots(1, len(systems), figsize=(3.4 * len(systems), 5.0))
    fig.subplots_adjust(wspace=0.95)
    axes = np.atleast_1d(axes)
    for ax, sysname in zip(axes, systems):
        g = src[src.system == sysname]
        phases = sorted(set(g.phase))
        y = np.arange(len(phases))
        e = [g[(g.phase == p) & (g.side == "experimental")].n_conditions.sum()
             for p in phases]
        c = [g[(g.phase == p) & (g.side == "calculated")].n_conditions.sum()
             for p in phases]
        ax.barh(y - 0.2, e, 0.38, color="#4c72b0", label="reported")
        ax.barh(y + 0.2, c, 0.38, color="#dd8452", label="calculated")
        ax.set_yticks(y)
        ax.set_yticklabels(phases, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(sysname, fontsize=8)
        ax.set_xlabel("conditions", fontsize=7)
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Phases reported by the studies and returned by the frozen "
                 "description (labels are not aligned by mapping here; see the "
                 "crosswalk)", fontsize=8.5)
    save(fig, "Fig_R3_01_phase_set_matrix", src)


def fig02(conv):
    """Condition-level agreement, one row per condition, grouped by system."""
    cols = ["raw_phase_Jaccard", "family_Jaccard"]
    g = conv.sort_values(["system", "temperature_C", "condition_id"]).copy()
    g["multiplicity"] = pd.to_numeric(g.multiplicity_match, errors="coerce")
    g["chemistry_within_tolerance"] = np.where(
        g.chemistry_gate == "WITHIN_TOLERANCE", 1.0,
        np.where(g.chemistry_gate.astype(str).str.len() > 0, 0.0, np.nan))
    show = cols + ["multiplicity", "chemistry_within_tolerance"]
    M = g[show].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.13 * len(g))))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(["raw-phase\nJaccard", "family\nJaccard", "phase\nmultiplicity",
                        "phase chemistry\nwithin tolerance"], fontsize=7)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.condition_id, fontsize=4.5)
    prev, bounds = None, []
    for i, s in enumerate(g.system):
        if s != prev:
            bounds.append((i, s))
            prev = s
    for i, s in bounds:
        ax.axhline(i - 0.5, color="k", lw=0.8)
        ax.text(len(show) - 0.4, i, s, fontsize=6, va="top")
    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.24,
                 label="agreement (white = not applicable)")
    ax.set_title("Condition-level agreement (white cells are not applicable)",
                 fontsize=8.5)
    save(fig, "Fig_R3_02_condition_agreement", g[["condition_id", "system",
                                                  "temperature_C"] + show])


def fig03(matched):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    pairs = []
    for _, r in matched.iterrows():
        e = json.loads(r.experimental_at_pct)
        c = json.loads(r.calculated_at_pct)
        for k in sorted(set(e) | set(c)):
            pairs.append({"condition_id": r.condition_id, "source_id": r.source_id,
                          "literature_phase": r.literature_phase,
                          "matched_calculated_phase": r.matched_calculated_phase,
                          "element": k, "experimental_at_pct": e.get(k, 0.0),
                          "calculated_at_pct": c.get(k, 0.0)})
    src = pd.DataFrame(pairs)
    for el, g in src.groupby("element"):
        ax.scatter(g.experimental_at_pct, g.calculated_at_pct, s=11, alpha=0.75,
                   linewidths=0, label=el)
    ax.plot([0, 100], [0, 100], "k--", lw=0.8)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("measured phase composition [at.%]")
    ax.set_ylabel("calculated phase composition [at.%]")
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_title("Matched phases only; matching is by structure then dominant\n"
                 "element, never by minimising this error", fontsize=8)
    save(fig, "Fig_R3_03_phase_compositions", src)


def fig04(ev):
    order = {"STRONG": 3, "MODERATE": 2, "LIMITED": 1, "NONE": 0}
    g = ev.copy()
    g["score"] = g.evidence_strength.map(order)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    colors = {"STRONG": "#55a868", "MODERATE": "#8fbc8f", "LIMITED": "#dd8452",
              "NONE": "#b0b0b0"}
    ax.barh(range(len(g)), g.score, color=[colors[s] for s in g.evidence_strength])
    for i, r in enumerate(g.itertuples()):
        ax.text(r.score + 0.05, i, " %s (%s)" % (r.evidence_strength,
                                                 r.evidence_strength_score),
                va="center", fontsize=7)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.system, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.2)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["NONE", "LIMITED", "MODERATE", "STRONG"], fontsize=7)
    ax.set_title("Consistency-evidence strength by system\n"
                 "(agreement contributes at most one of six points; independence "
                 "from the database assessment is not established)", fontsize=8)
    save(fig, "Fig_R3_04_evidence_strength",
         g[["system", "evidence_strength", "evidence_strength_score",
            "evidence_strength_parts", "n_calphad_converged"]])


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K-R3 stage 60: figures")
    print("=" * 78)
    C.verify_protected()

    cmp_ = pd.read_csv(os.path.join(C.D_CMP, "STEP5K_R3_CONDITION_COMPARISON.csv"))
    pcmp = pd.read_csv(os.path.join(C.D_PHCOMP,
                                    "STEP5K_R3_PHASE_COMPOSITION_COMPARISON.csv"))
    ev = pd.read_csv(os.path.join(C.D_EVID, "STEP5K_R3_SYSTEM_EVIDENCE.csv"))
    raw = pd.read_csv(os.path.join(C.D_RAW,
                                   "STEP5K_R3_EXTERNAL_CALPHAD_RAW_PHASES.csv"))
    conv = cmp_[cmp_.execution_status == "CONVERGED"]
    matched = pcmp[pcmp.match_status == "MATCHED"] if len(pcmp) else pcmp

    skipped = []
    if len(conv):
        fig01(conv, raw)
        fig02(conv)
    else:
        skipped += [("Fig_R3_01_phase_set_matrix",
                     "no converged calculation exists, so there is no calculated "
                     "phase set to place beside the reported one"),
                    ("Fig_R3_02_condition_agreement",
                     "no converged calculation exists, so no condition has an "
                     "agreement value")]
    if len(matched) >= MIN_MATCHED_FOR_FIG3:
        fig03(matched)
    else:
        skipped.append(("Fig_R3_03_phase_compositions",
                        "only %d matched phase compositions exist; the panel is "
                        "drawn only at %d or more, so that a handful of points is "
                        "not presented as a parity plot"
                        % (len(matched), MIN_MATCHED_FOR_FIG3)))
    if ev.n_calphad_converged.sum() > 0:
        fig04(ev)
    else:
        skipped.append(("Fig_R3_04_evidence_strength",
                        "every system is NONE or pending; a bar chart of zeros "
                        "would be decorative"))

    made = sorted(f for f in os.listdir(C.D_FIG) if f.endswith(".png"))
    status = ["# R3 figure status\n\n_Generated %s._\n\n" % C.utcnow()]
    status.append("Drawn: %s\n\n" % (", ".join(made) if made else "none"))
    if skipped:
        status.append("Not drawn, and why:\n\n")
        for name, why in skipped:
            status.append("- **%s** - %s\n" % (name, why))
        status.append("\nNo placeholder image was produced in their place.\n")
    open(os.path.join(C.D_FIG, "FIGURE_STATUS.md"), "w",
         encoding="utf-8").writelines(status)
    print("  drawn %d | skipped %d" % (len(made), len(skipped)))
    print("STAGE 60: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
