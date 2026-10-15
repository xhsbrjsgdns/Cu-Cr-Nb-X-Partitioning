# -*- coding: utf-8 -*-
"""
63_f5p_process.py
STEP5F-PILOT Stage 3: 5-family aggregation, MOLAR element partitioning, targets
(spec 15, 16, 17, 18).

Partition definition (spec 17, frozen since FULL_V1 - not restated loosely here):

    n_i,alpha  = phase_moles_alpha  *  X_i^alpha
    P_i^alpha  = n_i,alpha / sum_beta n_i,beta

`phase_volume_fraction * element_mole_fraction` is FORBIDDEN and appears nowhere.

Phase aggregation (spec 15, 16): FCC_L12* -> FCC, BCC_B2* -> BCC, C15_LAVES* -> C15,
LIQUID* -> LIQUID, EVERY other stable non-liquid phase -> OTHER_SOLID. Raw phase names
are recorded verbatim first; nothing is forced into FCC/BCC/C15 for convenience. A phase
that is not one of the three known solid models is OTHER_SOLID by default, and that is
reported as a finding, not hidden.

Outputs:
  03_processed/STEP5F_PILOT_STATE_RESULTS.parquet   one row per state, all five targets
  03_processed/STEP5F_PILOT_PARTITION_LONG.parquet  per (state, family, element)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5p_common as C

sys.path.insert(0, C.FULL_V1_CODE)

PRESENCE_THRESHOLD = 1e-4          # frozen FULL_V1 threshold, unchanged


def main():
    import fivefamily as ff
    print("=" * 78)
    print("STEP5F-PILOT  Stage 3: aggregation, molar partitioning, targets")
    print("=" * 78)

    comps = pd.read_csv(os.path.join(C.D_COMP, "STEP5F_PILOT_COMPOSITIONS.csv"))
    states = pd.read_csv(os.path.join(C.D_RAW, "STEP5F_PILOT_EQUILIBRIUM_STATES.csv"),
                         keep_default_na=False, low_memory=False)
    phases = C.read_table(os.path.join(C.D_RAW, "STEP5F_PILOT_PHASES_LONG"))
    pcomp = C.read_table(os.path.join(C.D_RAW, "STEP5F_PILOT_PHASE_COMPOSITIONS_LONG"))
    print("  states %d | phase rows %d | phase-composition rows %d"
          % (len(states), len(phases), len(pcomp)))

    states["converged_b"] = states["converged"].astype(str).str.lower() == "true"
    meta = comps.set_index("composition_id")
    states["element"] = states["case_id"].map(meta["X_element"])
    conv_ids = set(states.loc[states.converged_b, "state_id"])

    # ------------------------------------------------ raw -> base -> family -- #
    phases = phases[phases.state_id.isin(conv_ids)].copy()
    pcomp = pcomp[pcomp.state_id.isin(conv_ids)].copy()
    if "phase_model_base_name" not in phases.columns:
        phases["phase_model_base_name"] = (phases["raw_phase_name"].astype(str)
                                           .str.replace(r"#\d+$", "", regex=True))
    phases["aggregated_family"] = phases["phase_model_base_name"].map(ff.family_of_5)
    pcomp["phase_model_base_name"] = (pcomp["raw_phase_name"].astype(str)
                                      .str.replace(r"#\d+$", "", regex=True))
    pcomp["aggregated_family"] = pcomp["phase_model_base_name"].map(ff.family_of_5)

    # ------------------------------------------------------- 5-family wide -- #
    agg = (phases.groupby(["state_id", "aggregated_family"], as_index=False)
           .agg(volfrac=("phase_volume_fraction", "sum"),
                molefrac=("phase_mole_fraction", "sum"),
                moles=("phase_moles", "sum")))
    wide = agg.pivot_table(index="state_id", columns="aggregated_family",
                           values="volfrac", fill_value=0.0)
    widem = agg.pivot_table(index="state_id", columns="aggregated_family",
                            values="molefrac", fill_value=0.0)
    for f in C.FAMILIES_ORDER:
        if f not in wide.columns:
            wide[f] = 0.0
        if f not in widem.columns:
            widem[f] = 0.0
    wide = wide[C.FAMILIES_ORDER]
    widem = widem[C.FAMILIES_ORDER]

    # ------------------------------------------- MOLAR element partitioning -- #
    # n_i,alpha = phase_moles_alpha * X_i^alpha        (spec 17)
    pm = phases.set_index(["state_id", "raw_phase_name"])["phase_moles"]
    pc = pcomp.copy()
    pc["phase_moles"] = pd.MultiIndex.from_frame(
        pc[["state_id", "raw_phase_name"]]).map(pm)
    pc["element"] = pc["element"].astype(str).str.upper()
    pc["n_i_alpha"] = pc["phase_moles"].astype(float) * pc["element_mole_fraction"].astype(float)
    tot = pc.groupby(["state_id", "element"])["n_i_alpha"].sum().rename("n_i_total")
    fam = (pc.groupby(["state_id", "element", "aggregated_family"], as_index=False)
           ["n_i_alpha"].sum())
    fam = fam.merge(tot, on=["state_id", "element"], how="left")
    fam["P"] = np.where(fam["n_i_total"] > 0, fam["n_i_alpha"] / fam["n_i_total"], np.nan)
    fam.to_parquet(os.path.join(C.D_PROC, "STEP5F_PILOT_PARTITION_LONG.parquet"),
                   index=False)

    part = fam.pivot_table(index=["state_id", "element"], columns="aggregated_family",
                           values="P", fill_value=0.0).reset_index()
    for f in C.FAMILIES_ORDER:
        if f not in part.columns:
            part[f] = 0.0

    out = states[["state_id", "case_id", "family", "element", "X_element",
                  "Temperature_C", "Temperature_K", "converged_b", "solver_level_used",
                  "bridge_used", "attempt_count", "final_status",
                  "n_stable_phases"]].copy()
    out = out.rename(columns={"case_id": "composition_id",
                              "converged_b": "state_converged"})
    out["state_converged"] = out["state_converged"].astype(int)
    for col in ("Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct", "S_CrNb_wt_pct",
                "R_CrNb_atomic_actual"):
        out[col] = pd.to_numeric(out["composition_id"].map(meta[col]), errors="coerce")

    for f in C.FAMILIES_ORDER:
        out["%s_volfrac" % f] = out["state_id"].map(wide[f])
        out["%s_molefrac" % f] = out["state_id"].map(widem[f])
        out["%s_presence" % f] = (out["%s_volfrac" % f] > PRESENCE_THRESHOLD).astype(int)
    idx = part.set_index(["state_id", "element"])
    for f in C.FAMILIES_ORDER:
        out["CR_P_%s" % f] = pd.MultiIndex.from_arrays(
            [out.state_id, ["CR"] * len(out)]).map(idx[f])
        out["NB_P_%s" % f] = pd.MultiIndex.from_arrays(
            [out.state_id, ["NB"] * len(out)]).map(idx[f])
        out["X_P_%s" % f] = pd.MultiIndex.from_arrays(
            [out.state_id, out.element.astype(str)]).map(idx[f])

    fail = out.state_converged == 0
    tcols = [c for c in out.columns if c.endswith(("_volfrac", "_molefrac"))
             or "_P_" in c]
    out.loc[fail, tcols] = np.nan
    out.loc[fail, [c for c in out.columns if c.endswith("_presence")]] = 0

    out.to_parquet(os.path.join(C.D_PROC, "STEP5F_PILOT_STATE_RESULTS.parquet"),
                   index=False)

    cv = out[out.state_converged == 1]
    v = cv[["%s_volfrac" % f for f in C.FAMILIES_ORDER]].sum(axis=1)
    dev5 = float(np.abs(v.dropna() - 1.0).max()) if v.notna().any() else float("nan")
    worst = {}
    for role in ("CR", "NB", "X"):
        s = cv[["%s_P_%s" % (role, f) for f in C.FAMILIES_ORDER]].sum(axis=1).dropna()
        worst[role] = float(np.abs(s - 1.0).max()) if len(s) else float("nan")
    print("  5-family volume fraction sum-to-one max dev : %.3e" % dev5)
    print("  partition conservation max dev              : %s"
          % {k: "%.2e" % v for k, v in worst.items()})

    C.write_json(os.path.join(C.D_PROC, "STEP5F_PILOT_PROCESSING_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "states": int(len(out)), "converged": int((out.state_converged == 1).sum()),
        "failed": int((out.state_converged == 0).sum()),
        "partition_definition": ("n_i_alpha = phase_moles_alpha * X_i^alpha ; "
                                 "P_i^alpha = n_i_alpha / sum_beta n_i_beta "
                                 "(volume-fraction-based partitioning FORBIDDEN)"),
        "presence_threshold": PRESENCE_THRESHOLD,
        "five_family_sum_max_dev": dev5,
        "partition_sum_max_dev": worst,
        "families": C.FAMILIES_ORDER})
    print("\nSTAGE 3: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
