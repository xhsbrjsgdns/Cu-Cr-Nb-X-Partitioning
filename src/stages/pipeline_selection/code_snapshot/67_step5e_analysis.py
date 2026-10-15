# -*- coding: utf-8 -*-
"""67_step5e_analysis.py — Stage M+N+O: learning curves, B_skill / B_topo50,
random-vs-targeted comparison, recovery-curve areas, effect decomposition and
per-element remedy metrics. Consumes only frozen prediction/curve files."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5e_common as C


def b_skill(rows, arch, strategy):
    """Smallest budget with mean TVD < fold-train trivial TVD (spec 61)."""
    sub = rows[(rows.architecture == arch) & (rows.strategy == strategy)]
    if strategy == "random":
        sub = sub.groupby("budget", as_index=False).agg(mean_tvd=("mean_tvd", "mean"),
                                                        trivial_tvd=("trivial_tvd", "first"))
    else:
        sub = sub[["budget", "mean_tvd", "trivial_tvd"]]
    for _, r in sub.sort_values("budget").iterrows():
        if r.mean_tvd < r.trivial_tvd:
            return int(r.budget)
    return 999


def b_topo50(rows, arch, strategy):
    sub = rows[(rows.architecture == arch) & (rows.strategy == strategy)]
    if strategy == "random":
        sub = sub.groupby("budget", as_index=False).agg(acc=("topology_accuracy", "mean"))
    else:
        sub = sub[["budget", "topology_accuracy"]].rename(columns={"topology_accuracy": "acc"})
    for _, r in sub.sort_values("budget").iterrows():
        if r.acc >= 0.50:
            return int(r.budget)
    return 999


def curve_area(rows, arch, strategy):
    """Normalized recovery-curve area over log2(budget); lower = more efficient (spec 64)."""
    sub = rows[(rows.architecture == arch) & (rows.strategy == strategy)]
    if strategy == "random":
        sub = sub.groupby("budget", as_index=False).mean_tvd.mean()
    else:
        sub = sub[["budget", "mean_tvd"]]
    sub = sub.sort_values("budget")
    x = np.log2(sub.budget.to_numpy(float))
    y = sub.mean_tvd.to_numpy(float)
    if len(x) < 2 or not np.isfinite(y).all():
        return np.nan
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def main():
    curves = pd.read_csv(os.path.join(C.D_FEWSHOT, "STEP5E_FEWSHOT_RECOVERY_CURVES.csv"))
    abl = pd.read_csv(os.path.join(C.D_ABL, "STEP5E_ZERO_SHOT_ABLATION.csv"))
    oracle = pd.read_csv(os.path.join(C.D_ABL, "STEP5E_ORACLE_TOPOLOGY_DIAGNOSTIC.csv"))
    elems = C.HELDOUT_ELEMENTS

    # ---------- random vs targeted (spec 63/74) ----------
    rt = []
    for e in elems:
        for arch in ["F0_DIRECT_TREE", "F1_AFFINITY_TOPOLOGY_FIRST"]:
            for b in C.FEWSHOT_BUDGETS:
                sel = curves[(curves.heldout == e) & (curves.architecture == arch)
                             & (curves.strategy == "random") & (curves.budget == b)]
                tgt = curves[(curves.heldout == e) & (curves.architecture == arch)
                             & (curves.strategy == "targeted_maximin")
                             & (curves.budget == b)]
                if not len(sel) or not len(tgt):
                    continue
                rmean, rstd = float(sel.mean_tvd.mean()), float(sel.mean_tvd.std(ddof=1))
                tv = float(tgt.mean_tvd.iloc[0])
                rt.append({"heldout": e, "architecture": arch, "budget": b,
                           "targeted_tvd": tv, "random_mean_tvd": rmean,
                           "random_std_tvd": rstd,
                           "targeted_minus_random_mean": tv - rmean,
                           "targeted_percentile_among_random":
                               float((sel.mean_tvd < tv).mean())})
    rt = pd.DataFrame(rt)
    rt.to_csv(os.path.join(C.D_FEWSHOT, "STEP5E_FEWSHOT_RANDOM_VS_TARGETED.csv"), index=False)

    # ---------- B_skill / B_topo50 / areas ----------
    bs = []
    for e in elems:
        for arch in ["F0_DIRECT_TREE", "F1_AFFINITY_TOPOLOGY_FIRST"]:
            sub = curves[(curves.heldout == e) & (curves.architecture == arch)]
            bs.append({"heldout": e, "architecture": arch,
                       "B_skill_random": b_skill(sub, arch, "random"),
                       "B_skill_targeted": b_skill(sub, arch, "targeted_maximin"),
                       "B_topo50_random": b_topo50(sub, arch, "random"),
                       "B_topo50_targeted": b_topo50(sub, arch, "targeted_maximin"),
                       "area_random": curve_area(sub, arch, "random"),
                       "area_targeted": curve_area(sub, arch, "targeted_maximin"),
                       "tvd_32shot_random_mean": float(
                           sub[(sub.strategy == "random") & (sub.budget == 32)].mean_tvd.mean()),
                       "tvd_32shot_targeted": float(
                           sub[(sub.strategy == "targeted_maximin") & (sub.budget == 32)]
                           .mean_tvd.iloc[0])})
    bs = pd.DataFrame(bs)
    bs.to_csv(os.path.join(C.D_FEWSHOT, "STEP5E_B_SKILL_B_TOPO50.csv"), index=False)

    # ---------- effect decomposition (spec 76) ----------
    per = abl.pivot_table(index="heldout", columns="candidate",
                          values="mean_tvd", aggfunc="first")
    ora = oracle.set_index("heldout")["mean_tvd"]
    dec = []
    for e in elems:
        z0, z1 = per.loc[e, "Z0_TREE_DIRECT"], per.loc[e, "Z1_MLP_DIRECT"]
        z2, z3 = per.loc[e, "Z2_AFFINITY_XGB"], per.loc[e, "Z3_TOPOLOGY_FIRST"]
        z4, z5 = per.loc[e, "Z4_AFFINITY_TOPOLOGY_FIRST"], per.loc[e, "Z5_AFFINITY_MLP"]
        f1_32 = curves[(curves.heldout == e) & (curves.architecture == "F1_AFFINITY_TOPOLOGY_FIRST")
                       & (curves.budget == 32)]
        f1_32_tvd = float(f1_32.mean_tvd.mean())
        dec.append({
            "heldout": e,
            "descriptor_effect_Z2_minus_Z0": z2 - z0,
            "topology_effect_Z3_minus_Z0": z3 - z0,
            "combined_effect_Z4_minus_Z0": z4 - z0,
            "smooth_descriptor_effect_Z5_minus_Z1": z5 - z1,
            "interaction_Z4_minus_Z2_minus_Z3_plus_Z0": z4 - z2 - z3 + z0,
            "oracle_topology_TVD": float(ora[e]),
            "oracle_gap_Z4_minus_ORACLE": z4 - float(ora[e]),
            "fewshot_F1_32shot_TVD": f1_32_tvd,
            "fewshot_effect_32shot_minus_Z4": f1_32_tvd - z4})
    dec = pd.DataFrame(dec)
    dec.to_csv(os.path.join(C.D_ABL, "STEP5E_REMEDY_EFFECT_DECOMPOSITION.csv"), index=False)

    # ---------- per-element remedy metrics (spec 75) ----------
    pe = []
    for e in elems:
        sub = abl[abl.heldout == e]
        best = sub.sort_values("mean_tvd").iloc[0]
        triv = float(sub.trivial_tvd.iloc[0])
        b = bs[bs.heldout == e].iloc[0]
        pe.append({
            "heldout": e,
            "best_zero_shot_candidate": best.candidate,
            "best_zero_shot_mean_tvd": float(best.mean_tvd),
            "zero_shot_improvement_vs_Z0": float(
                per.loc[e, "Z0_TREE_DIRECT"] - best.mean_tvd),
            "zero_shot_relative_improvement_vs_Z0": float(
                1 - best.mean_tvd / per.loc[e, "Z0_TREE_DIRECT"]),
            "B_skill_random": int(b.B_skill_random),
            "B_skill_targeted": int(b.B_skill_targeted),
            "B_topo50_targeted": int(b.B_topo50_targeted),
            "tvd_32shot_targeted": float(b.tvd_32shot_targeted),
            "trivial_tvd": triv})
    pe = pd.DataFrame(pe)
    pe.to_csv(os.path.join(C.D_ABL, "STEP5E_PER_ELEMENT_REMEDY_METRICS.csv"), index=False)
    print(pe.to_string(index=False))
    print("\nSTAGE M+N+O: analysis tables written")


if __name__ == "__main__":
    main()
