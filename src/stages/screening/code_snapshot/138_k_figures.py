# -*- coding: utf-8 -*-
"""
138_k_figures.py
STEP5K figures Fig_K_01..08 (spec 45-47).

Every figure writes its own SOURCE csv next to the image, so the panel can be
regenerated - or checked - without rerunning any model or any equilibrium. No figure
displays a Ni quantity, and no figure re-plots a frozen Ni blind result.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
FAM = C.FAMILIES_ORDER
COL = {"ZR": "#1f77b4", "TI": "#ff7f0e", "CO": "#2ca02c", "SI": "#d62728",
       "AG": "#9467bd", "MO": "#8c564b"}


def save(fig, name, src=None):
    p = os.path.join(C.D_FIG, name + ".png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    if src is not None:
        src.to_csv(os.path.join(C.D_FIG, name + "_SOURCE.csv"), index=False,
                   encoding="utf-8")
    print("    %s" % os.path.basename(p))


def fig01_domain_gap(ext):
    """Where the literature alloys sit relative to the sampled Domain A."""
    rows = []
    for eid, g in ext.groupby("ext_id"):
        w = {k: float(v) for k, v in
             (t.split("=") for t in g.composition_wt_pct.iloc[0].split(";"))}
        cr, nb = w.get("CR", 0.0), w.get("NB", 0.0)
        if nb <= 0:
            continue
        rows.append({"ext_id": eid, "S_CrNb": cr + nb,
                     "R_CrNb_atomic": (cr / C.M_CR) / (nb / C.M_NB)})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.add_patch(Rectangle((C.S_CRNB_RANGE[0], C.R_CRNB_RANGE[0]),
                           C.S_CRNB_RANGE[1] - C.S_CRNB_RANGE[0],
                           C.R_CRNB_RANGE[1] - C.R_CRNB_RANGE[0],
                           facecolor="#4c72b0", alpha=0.18, edgecolor="#4c72b0",
                           lw=1.4, label="FULL_V1 Domain A (sampled)"))
    ax.scatter(d.S_CrNb, d.R_CrNb_atomic, s=70, marker="D", color="#c44e52",
               zorder=5, label="literature composition (stated in the paper)")
    for _, r in d.iterrows():
        ax.annotate(r.ext_id.replace("EXT_", "").replace("_", " "),
                    (r.S_CrNb, r.R_CrNb_atomic), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)
    ax.axhline(2.0, color="#c44e52", ls=":", lw=1,
               label="stoichiometric Cr$_2$Nb ratio, R = 2")
    ax.set_xscale("log")
    ax.set_xlabel("$S_{CrNb}$ = w(Cr) + w(Nb)   [wt.%]")
    ax.set_ylabel("$R_{CrNb}$ (atomic)")
    ax.set_title("The sampled domain and the published alloys do not overlap")
    ax.legend(fontsize=7, loc="upper right")
    save(fig, "Fig_K_01_domain_gap", d)


def fig02_verdicts(val):
    c = val.verdict.value_counts()
    order = [v for v in ["QUALITATIVELY_CONSISTENT", "PARTIALLY_CONSISTENT",
                         "REASONABLE_EQUILIBRIUM_VS_NON_EQUILIBRIUM_DISCREPANCY",
                         "NOT_REPRODUCED_BY_EQUILIBRIUM_CALCULATION",
                         "NOT_DIRECTLY_COMPARABLE"] if v in c.index]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.barh(range(len(order)), [c[v] for v in order],
            color=["#55a868", "#8fbc8f", "#dd8452", "#c44e52", "#b0b0b0"][:len(order)])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([v.replace("_", " ").lower() for v in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("literature rows")
    ax.set_title("External validation outcome (CALPHAD layer only)")
    save(fig, "Fig_K_02_validation_verdicts",
         pd.DataFrame({"verdict": order, "n": [c[v] for v in order]}))


def fig03_ext_phase_vs_T(ext):
    ids = sorted(ext.ext_id.unique())
    n = len(ids)
    fig, axes = plt.subplots(1, n, figsize=(2.05 * n, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, eid in zip(axes, ids):
        g = ext[(ext.ext_id == eid) & (ext.converged == 1)].sort_values("Temperature_C")
        for f, c in zip(FAM, ["#4c72b0", "#dd8452", "#c44e52", "#55a868", "#937860"]):
            if ("frac_%s" % f) in g:
                ax.plot(g.Temperature_C, g["frac_%s" % f], marker="o", ms=2.5,
                        lw=1.2, color=c, label=f)
        ax.set_title(eid.replace("EXT_", "").replace("_", "\n"), fontsize=6.5)
        ax.set_xlabel("T [°C]", fontsize=7)
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel("phase-family volume fraction")
    axes[-1].legend(fontsize=6, loc="upper right")
    fig.suptitle("Equilibrium phase families at the stated literature compositions "
                 "(post-freeze, non-Ni)", fontsize=8.5)
    save(fig, "Fig_K_03_extval_phase_vs_T", ext)


def fig04_funnel(pf):
    steps = [("all candidates", pf["n_candidates"]),
             ("C: moderate C15 band", pf["constraint_counts"]["C_band"]),
             ("E: no liquid", pf["constraint_counts"]["E_liquid"]),
             ("F: inside OOD radius", pf["constraint_counts"]["F_ood"]),
             ("applicable (C and E and F)", pf["n_applicable"]),
             ("global Pareto front", pf["global_front"]),
             ("finalists", pf["finalists"])]
    d = pd.DataFrame(steps, columns=["stage", "n"])
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ax.barh(range(len(d)), d.n, color="#4c72b0")
    for i, v in enumerate(d.n):
        ax.text(v, i, " %d" % v, va="center", fontsize=7.5)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.stage, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xscale("symlog")
    ax.set_xlabel("candidates remaining (log scale)")
    ax.set_title("Screening funnel: constraints were frozen before any prediction")
    save(fig, "Fig_K_04_screening_funnel", d)


def fig05_pareto(par):
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    bg = par[par.pareto_rank > 1]
    ax.scatter(bg.Cr_FCC_950, bg.Delta_Cr_FCC, s=3, c=bg.Cr_OTHER_450,
               cmap="viridis_r", alpha=0.35, linewidths=0)
    fr = par[par.pareto_rank == 1].sort_values("Cr_FCC_950")
    ax.plot(fr.Cr_FCC_950, fr.Delta_Cr_FCC, color="#c44e52", lw=1.2, zorder=4)
    ax.scatter(fr.Cr_FCC_950, fr.Delta_Cr_FCC, s=22, color="#c44e52", zorder=5,
               label="non-dominated front")
    fin = par[par.selected_finalist == 1]
    ax.scatter(fin.Cr_FCC_950, fin.Delta_Cr_FCC, s=70, facecolors="none",
               edgecolors="k", lw=1.1, zorder=6, label="maximin finalists")
    sm = plt.cm.ScalarMappable(cmap="viridis_r",
                               norm=plt.Normalize(par.Cr_OTHER_450.min(),
                                                  par.Cr_OTHER_450.max()))
    fig.colorbar(sm, ax=ax, label="objective D: $P_{Cr}^{OTHER}$ at 450 °C (minimised)")
    ax.set_xlabel("objective A: $P_{Cr}^{FCC}$ at 950 °C (maximised)")
    ax.set_ylabel("objective B: $\\Delta P_{Cr}^{FCC}$, 950 - 450 °C (maximised)")
    ax.set_title("Non-dominated sorting - no weighted score was used")
    ax.legend(fontsize=7)
    save(fig, "Fig_K_05_pareto_front",
         par[["candidate_id", "X_identity", "Cr_FCC_950", "Delta_Cr_FCC",
              "Cr_OTHER_450", "pareto_rank", "selected_finalist"]])


def fig06_family_fairness(par, pf):
    d = pd.DataFrame({
        "X_identity": C.SIX,
        "applicable": [pf["applicable_per_family"].get(f, 0) for f in C.SIX],
        "family_front": [pf["family_front"].get(f, 0) for f in C.SIX],
        "global_front": [int(((par.X_identity == f) & (par.pareto_rank == 1)).sum())
                         for f in C.SIX],
        "finalists": [pf["per_family_finalists"].get(f, 0) for f in C.SIX]})
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    x = np.arange(len(C.SIX))
    for i, (col, lab) in enumerate([("applicable", "applicable"),
                                    ("family_front", "own-family front"),
                                    ("global_front", "on global front"),
                                    ("finalists", "selected finalists")]):
        ax.bar(x + (i - 1.5) * 0.2, d[col], 0.19, label=lab)
    ax.set_yscale("symlog")
    ax.set_xticks(x); ax.set_xticklabels(C.SIX)
    ax.set_ylabel("candidates (log scale)")
    ax.set_title("Family fairness: no identity silently fills the finalist list")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "Fig_K_06_family_fairness", d)


def fig07_diversity(par):
    fin = par[par.selected_finalist == 1]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    for ax, (a, b) in zip(axes, [("S_CrNb", "R_CrNb_atomic"),
                                 ("S_CrNb", "X_wt_pct"),
                                 ("R_CrNb_atomic", "X_wt_pct")]):
        ax.scatter(par[a], par[b], s=2, color="#c8c8c8", alpha=0.5, linewidths=0)
        for f in C.SIX:
            m = fin[fin.X_identity == f]
            ax.scatter(m[a], m[b], s=42, color=COL[f], edgecolors="k", lw=0.5,
                       label=f)
        ax.set_xlabel(a, fontsize=7.5); ax.set_ylabel(b, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
    axes[-1].legend(fontsize=6, ncol=2)
    fig.suptitle("Finalist spread in composition space (deterministic maximin)",
                 fontsize=9)
    save(fig, "Fig_K_07_finalist_diversity",
         fin[["candidate_id", "X_identity", "S_CrNb", "R_CrNb_atomic", "X_wt_pct",
              "diversity_rank"]])


def fig08_confirmation(conf):
    ok = conf[conf.converged == 1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for f, c in zip(FAM, ["#4c72b0", "#dd8452", "#c44e52", "#55a868", "#937860"]):
        a1.scatter(ok["calphad_frac_%s" % f], ok["pred_frac_%s" % f], s=12,
                   color=c, alpha=0.75, label=f, linewidths=0)
    lim = [0, 1.02]
    a1.plot(lim, lim, "k--", lw=0.8)
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("CALPHAD phase-family volume fraction")
    a1.set_ylabel("frozen surrogate prediction")
    a1.legend(fontsize=6.5, loc="upper left")
    a1.set_title("Finalists only - a selected, non-blind subset", fontsize=8.5)
    a2.hist(ok.TVD_family.dropna(), bins=18, color="#4c72b0", alpha=0.85)
    a2.axvline(ok.TVD_family.mean(), color="#c44e52", lw=1.3,
               label="mean %.4f" % ok.TVD_family.mean())
    a2.set_xlabel("family-vector TVD (prediction vs calculation)")
    a2.set_ylabel("states"); a2.legend(fontsize=7)
    a2.set_title("Not comparable to the prospective blind metric", fontsize=8.5)
    save(fig, "Fig_K_08_ml_vs_calphad", ok)


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K figures")
    print("=" * 78)
    ext = pd.read_csv(os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_STATES.csv"))
    val = pd.read_csv(os.path.join(C.D_EXTVAL,
                                   "STEP5K_EXTERNAL_LITERATURE_VALIDATION.csv"))
    par = pd.read_csv(os.path.join(C.D_SCREEN,
                                   "STEP5K_VIRTUAL_SCREENING_PARETO.csv"))
    pf = json.load(open(os.path.join(C.D_SCREEN, "STEP5K_PARETO_FREEZE.json"),
                        encoding="utf-8"))
    conf = pd.read_csv(os.path.join(C.D_CONFIRM,
                                    "STEP5K_SCREENING_CALPHAD_CONFIRMATION.csv"))
    for f, args in [(fig01_domain_gap, (ext,)), (fig02_verdicts, (val,)),
                    (fig03_ext_phase_vs_T, (ext,)), (fig04_funnel, (pf,)),
                    (fig05_pareto, (par,)), (fig06_family_fairness, (par, pf)),
                    (fig07_diversity, (par,)), (fig08_confirmation, (conf,))]:
        f(*args)
    made = sorted(x for x in os.listdir(C.D_FIG) if x.endswith(".png"))
    if len(made) != 8:
        raise SystemExit("FATAL: expected 8 figures, found %d: %s" % (len(made), made))
    print("FIGURES: PASS (%d panels, each with a SOURCE csv)" % len(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
