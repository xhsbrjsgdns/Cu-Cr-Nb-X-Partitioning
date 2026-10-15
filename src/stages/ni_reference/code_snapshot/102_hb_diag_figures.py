# -*- coding: utf-8 -*-
"""
102_hb_diag_figures.py
STEP5H-B Stage 5: PRIMARY-vs-SECONDARY sensitivity comparison (spec 41),
EXPLORATORY post-reveal diagnostics (spec 42, 43), composition-level error
descriptives (spec 45), temperature diagnostic (spec 44) and the ten figures
(spec 46).

Everything here is descriptive. No threshold fitting, no AUROC threshold
selection, no model modification, no model selection (spec 41, 43). The
historical LOEO numbers appear ONLY as clearly-labelled retrospective context
(spec 47), with the STEP5G_CLARIFICATION_01 wording: they are model-family
context, not scores of these exact production checkpoints.
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
import hb_common as C

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})
FAM_COLORS = {"FCC": "#4477aa", "BCC": "#66ccee", "C15": "#228833",
              "OTHER_SOLID": "#ccbb44", "LIQUID": "#ee6677"}


def save(fig, name):
    p = os.path.join(C.D_FIG, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % name)


def main():
    from scipy.stats import spearmanr
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5H-B  Stage 5: comparison, exploratory diagnostics, figures")
    print("=" * 78)
    for req in ("STEP5H_B_PRIMARY_SCORE_FREEZE.json",):
        if not os.path.exists(os.path.join(C.D_PSCORE, req)):
            raise SystemExit("FATAL: %s missing" % req)
    if not os.path.exists(os.path.join(C.D_SSCORE,
                                       "STEP5H_B_SECONDARY_SCORE_FREEZE.json")):
        raise SystemExit("FATAL: SECONDARY score freeze missing")

    pm = json.load(open(os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_BLIND_METRICS.json"),
                        encoding="utf-8"))
    sm = json.load(open(os.path.join(C.D_SSCORE,
                                     "STEP5H_B_SECONDARY_BLIND_METRICS.json"),
                        encoding="utf-8"))
    targets = pd.read_parquet(os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGETS.parquet"))
    perr = pd.read_parquet(os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_BLIND_ERRORS.parquet"))
    serr = pd.read_parquet(os.path.join(C.D_SSCORE,
                                        "STEP5H_B_SECONDARY_BLIND_ERRORS.parquet"))
    pt = pd.read_csv(os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_PER_TEMPERATURE.csv"))
    st = pd.read_csv(os.path.join(C.D_SSCORE, "STEP5H_B_SECONDARY_PER_TEMPERATURE.csv"))
    scored = targets[targets.calphad_status == "CONVERGED"]

    # ---- spec 41: sensitivity comparison (NO model selection) ---------------- #
    cmp_rows = [{
        "quantity": "mean_tvd", "PRIMARY": pm["mean_tvd"], "SECONDARY": sm["mean_tvd"],
        "difference_S_minus_P": sm["mean_tvd"] - pm["mean_tvd"],
        "relative_difference": (sm["mean_tvd"] - pm["mean_tvd"]) / pm["mean_tvd"]},
        {"quantity": "median_tvd", "PRIMARY": pm["median_tvd"],
         "SECONDARY": sm["median_tvd"],
         "difference_S_minus_P": sm["median_tvd"] - pm["median_tvd"],
         "relative_difference": (sm["median_tvd"] - pm["median_tvd"]) / pm["median_tvd"]},
        {"quantity": "p95_tvd", "PRIMARY": pm["p95_tvd"], "SECONDARY": sm["p95_tvd"],
         "difference_S_minus_P": sm["p95_tvd"] - pm["p95_tvd"],
         "relative_difference": (sm["p95_tvd"] - pm["p95_tvd"]) / pm["p95_tvd"]},
        {"quantity": "topology_accuracy", "PRIMARY": pm["topology_accuracy"],
         "SECONDARY": sm["topology_accuracy"],
         "difference_S_minus_P": sm["topology_accuracy"] - pm["topology_accuracy"],
         "relative_difference": np.nan},
        {"quantity": "macro_mae", "PRIMARY": pm["macro_mae"],
         "SECONDARY": sm["macro_mae"],
         "difference_S_minus_P": sm["macro_mae"] - pm["macro_mae"],
         "relative_difference": (sm["macro_mae"] - pm["macro_mae"]) / pm["macro_mae"]}]
    cmpdf = pd.DataFrame(cmp_rows)
    cmpdf.to_csv(os.path.join(C.D_POSTREV, "STEP5H_B_PRIMARY_VS_SECONDARY.csv"),
                 index=False, encoding="utf-8")
    per_t_cmp = pt.merge(st, on="Temperature_C", suffixes=("_P", "_S"))
    per_t_cmp.to_csv(os.path.join(C.D_POSTREV,
                                  "STEP5H_B_PER_TEMPERATURE_COMPARISON.csv"),
                     index=False, encoding="utf-8")

    # ---- spec 43: EXPLORATORY post-reveal diagnostics ------------------------ #
    ood = pd.read_csv(C.HA_OOD_STATE)
    dis = pd.read_csv(C.HA_DISAGREE)
    dj = perr.merge(ood[["state_id", "max_abs_z", "mean_abs_z", "nn1_distance",
                         "knn5_mean_distance"]], on="state_id", validate="1:1")
    dj = dj.merge(dis[["state_id", "tvd_primary_vs_secondary"]], on="state_id",
                  validate="1:1")
    corr = {}
    for col in ("max_abs_z", "mean_abs_z", "nn1_distance", "knn5_mean_distance",
                "tvd_primary_vs_secondary"):
        r = spearmanr(dj[col].to_numpy(float), dj.tvd.to_numpy(float))
        corr[col] = {"spearman_rho": float(r.statistic), "p_value": float(r.pvalue)}
    diag = pd.DataFrame([{"diagnostic": k, **v} for k, v in corr.items()])
    diag.insert(0, "label", "EXPLORATORY POST-REVEAL DIAGNOSTIC")
    diag.insert(1, "target", "PRIMARY per-state TVD")
    diag.to_csv(os.path.join(C.D_POSTREV, "STEP5H_B_POSTREVEAL_DIAGNOSTICS.csv"),
                index=False, encoding="utf-8")
    print(diag.to_string(index=False))

    # ---- spec 45: composition-level descriptives ----------------------------- #
    comp_err = (perr.groupby("composition_id")
                .agg(mean_tvd=("tvd", "mean"), max_tvd=("tvd", "max"),
                     topology_accuracy=("topology_correct", "mean"),
                     n_converged_T=("tvd", "size")).reset_index())
    comp_err.to_csv(os.path.join(C.D_POSTREV, "STEP5H_B_PRIMARY_ERROR_BY_COMPOSITION.csv"),
                    index=False, encoding="utf-8")

    # ---- figures (spec 46) ---------------------------------------------------- #
    # 01 true topology vs temperature
    tt = (scored.assign(topo=scored.true_topology)
          .groupby(["Temperature_C", "topo"]).size().unstack(fill_value=0))
    tt = tt.div(tt.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    bottom = np.zeros(len(tt))
    for f in C.FAMILIES_ORDER:
        if f in tt.columns:
            ax.bar(tt.index.astype(int), tt[f], bottom=bottom, label=f,
                   color=FAM_COLORS[f])
            bottom += tt[f].to_numpy()
    ax.set_xlabel("Temperature (degC)")
    ax.set_ylabel("fraction of converged states")
    ax.set_title("Fig_HB_01  TRUE Ni dominant topology vs temperature\n"
                 "(prospective blind CALPHAD targets)")
    ax.legend(fontsize=7)
    save(fig, "Fig_HB_01_true_Ni_topology_vs_temperature.png")

    # 02 primary predicted vs true topology (row-normalized confusion)
    conf = pd.crosstab(perr.true_topology, perr.pred_topology, normalize="index")
    for f in C.FAMILIES_ORDER:
        if f not in conf.columns:
            conf[f] = 0.0
    conf = conf.reindex(index=[f for f in C.FAMILIES_ORDER if f in conf.index])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(conf[C.FAMILIES_ORDER].to_numpy(float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(5), C.FAMILIES_ORDER, rotation=30, fontsize=7)
    ax.set_yticks(range(len(conf)), conf.index, fontsize=7)
    for i in range(len(conf)):
        for j in range(5):
            v = conf[C.FAMILIES_ORDER].to_numpy(float)[i, j]
            if v > 0.01:
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")
    ax.set_xlabel("PRIMARY predicted")
    ax.set_ylabel("true (CALPHAD)")
    ax.set_title("Fig_HB_02  PRIMARY predicted vs true topology\n(prospective Ni blind)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "Fig_HB_02_primary_predicted_vs_true_topology.png")

    # 03 PRIMARY TVD distribution
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist(perr.tvd, bins=80, color="#4477aa")
    ax.axvline(pm["mean_tvd"], c="#ee6677", ls="--",
               label="mean %.4f" % pm["mean_tvd"])
    ax.axvline(pm["median_tvd"], c="#228833", ls=":",
               label="median %.4f" % pm["median_tvd"])
    ax.set_xlabel("per-state TVD")
    ax.set_ylabel("states")
    ax.set_title("Fig_HB_03  PRIMARY blind TVD distribution (prospective Ni)")
    ax.legend(fontsize=7)
    save(fig, "Fig_HB_03_primary_TVD_distribution.png")

    # 04 PRIMARY TVD by temperature
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(pt.Temperature_C, pt.mean_tvd, "o-", label="PRIMARY mean TVD")
    ax.plot(pt.Temperature_C, pt.median_tvd, "s:", label="PRIMARY median TVD", c="#228833")
    ax.set_xlabel("Temperature (degC)")
    ax.set_ylabel("TVD")
    ax.set_title("Fig_HB_04  PRIMARY blind TVD by temperature (all 11 kept)")
    ax.legend(fontsize=7)
    save(fig, "Fig_HB_04_primary_TVD_by_temperature.png")

    # 05 PRIMARY vs SECONDARY TVD
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist([perr.tvd, serr.tvd], bins=60, label=["PRIMARY", "SECONDARY"],
            color=["#4477aa", "#ccbb44"])
    ax.set_xlabel("per-state TVD")
    ax.set_ylabel("states")
    ax.set_title("Fig_HB_05  PRIMARY vs SECONDARY blind TVD\n(sensitivity comparison - "
                 "NOT model selection)")
    ax.legend(fontsize=7)
    save(fig, "Fig_HB_05_primary_vs_secondary_TVD.png")

    # 06 topology accuracy by temperature
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(per_t_cmp.Temperature_C, per_t_cmp.topology_accuracy_P, "o-", label="PRIMARY")
    ax.plot(per_t_cmp.Temperature_C, per_t_cmp.topology_accuracy_S, "s-", label="SECONDARY")
    ax.set_xlabel("Temperature (degC)")
    ax.set_ylabel("topology accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Fig_HB_06  topology accuracy by temperature (blind)")
    ax.legend(fontsize=7)
    save(fig, "Fig_HB_06_primary_secondary_topology_accuracy.png")

    # 07 error vs pre-reveal distance
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax_, col, lab in ((axs[0], "nn1_distance", "pre-reveal 1-NN distance"),
                          (axs[1], "max_abs_z", "pre-reveal max |z|")):
        ax_.scatter(dj[col], dj.tvd, s=1, alpha=0.15, c="#4477aa")
        qb = pd.qcut(dj[col], 10, duplicates="drop")
        mx = dj.groupby(qb, observed=True).tvd.mean()
        ax_.plot([iv.mid for iv in mx.index], mx.values, "o-", c="#ee6677", ms=3,
                 label="decile mean")
        rho = corr[col]["spearman_rho"]
        ax_.set_xlabel(lab)
        ax_.set_ylabel("PRIMARY TVD")
        ax_.set_title("Spearman rho = %.3f" % rho, fontsize=8)
        ax_.legend(fontsize=6)
    fig.suptitle("Fig_HB_07  PRIMARY blind error vs PRE-REVEAL frozen diagnostics "
                 "(EXPLORATORY)", fontsize=9)
    save(fig, "Fig_HB_07_error_vs_pre_reveal_distance.png")

    # 08 error vs model disagreement
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.scatter(dj.tvd_primary_vs_secondary, dj.tvd, s=1, alpha=0.15, c="#228833")
    qb = pd.qcut(dj.tvd_primary_vs_secondary, 10, duplicates="drop")
    mx = dj.groupby(qb, observed=True).tvd.mean()
    ax.plot([iv.mid for iv in mx.index], mx.values, "o-", c="#ee6677", ms=3,
            label="decile mean")
    ax.set_xlabel("pre-reveal PRIMARY-vs-SECONDARY disagreement (TVD)")
    ax.set_ylabel("PRIMARY blind TVD")
    ax.set_title("Fig_HB_08  error vs model disagreement (EXPLORATORY)\nSpearman rho "
                 "= %.3f" % corr["tvd_primary_vs_secondary"]["spearman_rho"])
    ax.legend(fontsize=7)
    save(fig, "Fig_HB_08_error_vs_model_disagreement.png")

    # 09 true partition by temperature
    fam_cols = C.TRUE_COLS
    mt = scored.groupby("Temperature_C")[fam_cols].mean()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for j, f in enumerate(C.FAMILIES_ORDER):
        ax.plot(mt.index.astype(int), mt[fam_cols[j]], "o-", label=f,
                color=FAM_COLORS[f], ms=3)
    ax.set_xlabel("Temperature (degC)")
    ax.set_ylabel("mean true Ni partition P_Ni")
    ax.set_title("Fig_HB_09  TRUE Ni partition by temperature (blind CALPHAD)")
    ax.legend(fontsize=7)
    save(fig, "Fig_HB_09_true_partition_by_temperature.png")

    # 10 blind result vs historical context
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    bars = ["PRIMARY Ni blind\n(PROSPECTIVE, this checkpoint)",
            "SECONDARY Ni blind\n(PROSPECTIVE, sensitivity)",
            "original-six LOEO\n(RETROSPECTIVE family context)",
            "FULL5 LOEO\n(RETROSPECTIVE family context)"]
    vals = [pm["mean_tvd"], sm["mean_tvd"], C.HISTORICAL_LOEO_ORIGINAL6,
            C.HISTORICAL_LOEO_FULL5]
    cols = ["#4477aa", "#ccbb44", "#999999", "#bbbbbb"]
    ax.bar(bars, vals, color=cols)
    ax.errorbar(0, pm["mean_tvd"],
                yerr=[[pm["mean_tvd"] - pm["bootstrap_mean_tvd"]["ci95_low"]],
                      [pm["bootstrap_mean_tvd"]["ci95_high"] - pm["mean_tvd"]]],
                fmt="none", ecolor="k", capsize=4)
    ax.axhline(0.5, ls=":", c="#ee6677")
    ax.text(3.4, 0.505, "0.50 descriptive guide", ha="right", fontsize=6, color="#ee6677")
    ax.set_ylabel("mean TVD")
    ax.set_title("Fig_HB_10  blind result vs historical context\n(the two grey bars are "
                 "RETROSPECTIVE model-family values, NOT scores of these checkpoints - "
                 "STEP5G_CLARIFICATION_01)", fontsize=8)
    ax.tick_params(axis="x", labelsize=6.5)
    save(fig, "Fig_HB_10_blind_result_context_vs_historical_LOEO.png")

    C.write_json(os.path.join(C.D_POSTREV, "STEP5H_B_DIAGNOSTICS_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "label": "EXPLORATORY POST-REVEAL DIAGNOSTIC (spec 43); descriptive only; "
                 "no threshold fitted, no model modified, no model selected",
        "spearman_primary_tvd_vs": corr,
        "primary_vs_secondary": json.loads(cmpdf.to_json(orient="records")),
        "composition_error_descriptives": {
            "mean_of_mean_tvd": float(comp_err.mean_tvd.mean()),
            "worst_composition": comp_err.loc[comp_err.mean_tvd.idxmax()].to_dict(),
            "best_composition": comp_err.loc[comp_err.mean_tvd.idxmin()].to_dict()},
        "figures": sorted(os.listdir(C.D_FIG))})
    print("STAGE 5: PASS (descriptive only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
