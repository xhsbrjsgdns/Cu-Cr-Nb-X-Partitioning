# -*- coding: utf-8 -*-
"""
23_fullv1_partitioning.py (from the Step 3C-validated 19_pilot2000_partitioning.py)

19_pilot2000_partitioning.py - Step 3C: 5-family aggregation, element partitioning,
liquid-aware topology and the ML target preview for the 550,000-state FULL CALPHAD V1.

Physics is IDENTICAL to Step 3B (15_pilot200_partitioning.py):
    n_i_alpha = phase_moles_alpha * X_i_alpha
    P_i_family = sum(n_i in family) / sum(n_i over all phases)
Volume fractions are NEVER used for elemental partitioning.

What is different: the per-state Python loop of Step 3B is replaced by vectorised
groupby/merge, because at 22000+ states the old per-state loop is itself O(N^2)-ish.  The
numbers it produces are unchanged (verified against the Pilot200 outputs).

Core 5-family labels stay frozen: FCC / BCC / C15 / OTHER_SOLID / LIQUID.

AUXILIARY / DIAGNOSTIC columns (spec section 6) - these are NOT core targets and must
never be summed together with the 5 core families (that would double-count ZRM5_C15B):
    ZRM5_C15B_volfrac
    C15_family_wide_volfrac      = C15_volfrac + ZRM5_C15B_volfrac
    <EL>_P_C15_family_wide       = P_<EL>_C15 + P_<EL>_ZRM5_C15B
Every auxiliary column name is listed in FULL_V1_AUXILIARY_COLUMNS.txt.

Reads the two shard SQLite result stores directly (they are the canonical store),
so no parquet dependency is needed to run this.
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1")
sys.path.insert(0, str(PROJECT_ROOT / "code_snapshot"))
import fivefamily as ff     # noqa: E402
import io_store             # noqa: E402

DIR_RAW = PROJECT_ROOT / "02_raw"
DIR_PROC = PROJECT_ROOT / "03_processed"
DIR_QA = PROJECT_ROOT / "04_qa"
FILE_COMPS = PROJECT_ROOT / "01_compositions" / "FULL_V1_COMPOSITIONS.csv"
PRESENCE_THRESHOLD = 1e-4
AUX_BASE = "ZRM5_C15B"          # spec section 6: stays in OTHER_SOLID, mirrored as auxiliary


def _argmax_label(df, prefix):
    """Column name (prefix stripped) of the row-wise max; empty string for all-NA rows."""
    out = pd.Series("", index=df.index, dtype=object)
    mask = df.notna().any(axis=1)
    if mask.any():
        out.loc[mask] = (df.loc[mask].idxmax(axis=1).astype(str)
                         .str.replace(prefix, "", regex=False))
    return out


def load_stores(shards=None):
    if shards is None:
        shards = sorted(int(p.stem.split("shard")[-1])
                        for p in DIR_RAW.glob("calculation_checkpoint_shard*.sqlite"))
    frames = {"state_result": [], "phase_result": [], "phase_composition_result": [],
              "agg_result": []}
    found = []
    for n in shards:
        db = DIR_RAW / ("calculation_checkpoint_shard%02d.sqlite" % n)
        if not db.exists():
            continue
        found.append(n)
        con = sqlite3.connect("file:%s?mode=ro" % db.as_posix(), uri=True)
        try:
            for t in frames:
                df = pd.read_sql_query("SELECT * FROM %s" % t, con)
                df["shard"] = n
                frames[t].append(io_store._retype(df))
        finally:
            con.close()
    if not found:
        raise SystemExit("no shard store found under %s" % DIR_RAW)
    out = {t: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for t, v in frames.items()}
    out["state_result"] = out["state_result"].drop_duplicates("state_id", keep="last")
    print("loaded shards %s: states=%d phases=%d phase_comps=%d agg=%d"
          % (found, len(out["state_result"]), len(out["phase_result"]),
             len(out["phase_composition_result"]), len(out["agg_result"])))
    return out


def main():
    for d in (DIR_RAW, DIR_PROC, DIR_QA):
        d.mkdir(parents=True, exist_ok=True)
    S = load_stores()
    states, phases, pcomp, agg = (S["state_result"], S["phase_result"],
                                  S["phase_composition_result"], S["agg_result"])
    comps = pd.read_csv(FILE_COMPS, dtype=str).fillna("")
    meta = comps.set_index("composition_id")

    states["converged_b"] = states["converged"].astype(str).str.lower() == "true"
    conv_ids = set(states.loc[states.converged_b, "state_id"])

    # ---------------------------------------------------------------- families -- #
    base_map = {b: ff.family_of_5(b) for b in phases["phase_model_base_name"].dropna().unique()}
    phases["aggregated_family"] = phases["phase_model_base_name"].map(base_map)

    # ------------------------------------------------- element partitioning ---- #
    # n_i_alpha = phase_moles_alpha * X_i_alpha   (strictly mole based)
    ph = phases[["state_id", "case_id", "raw_phase_name", "phase_model_base_name",
                 "aggregated_family", "phase_moles", "Temperature_C"]]
    pr = pcomp.merge(ph, on=["state_id", "raw_phase_name"], how="left",
                     suffixes=("", "_ph"))
    pr["element"] = pr["element"].astype(str).str.upper()
    pr["element_moles_in_phase"] = pr["phase_moles"] * pr["element_mole_fraction"]
    tot = (pr.groupby(["state_id", "element"])["element_moles_in_phase"]
             .transform("sum"))
    pr["element_partition_raw"] = np.where(tot > 0, pr["element_moles_in_phase"] / tot, np.nan)
    fam_of_state = states.set_index("state_id")["family"]
    pr["family"] = pr["state_id"].map(fam_of_state)
    pr["X_element"] = pr["case_id"].map(lambda c: (meta.loc[c, "X_element"]
                                                   if c in meta.index else ""))
    raw_cols = ["state_id", "case_id", "family", "X_element", "Temperature_C",
                "raw_phase_name", "phase_model_base_name", "aggregated_family",
                "phase_moles", "element", "element_mole_fraction",
                "element_moles_in_phase", "element_partition_raw"]
    part_raw = pr[[c for c in raw_cols if c in pr.columns]].copy()
    part_raw = part_raw.rename(columns={"case_id": "composition_id",
                                        "phase_model_base_name": "phase_model_base"})

    # core 5-family partition, wide
    g = (pr.groupby(["state_id", "element", "aggregated_family"])["element_moles_in_phase"]
           .sum().reset_index())
    gt = g.groupby(["state_id", "element"])["element_moles_in_phase"].transform("sum")
    g["P"] = np.where(gt > 0, g["element_moles_in_phase"] / gt, np.nan)
    pw = g.pivot_table(index=["state_id", "element"], columns="aggregated_family",
                       values="P", aggfunc="sum")
    for fam in ff.FAMILY_ORDER_5:
        if fam not in pw.columns:
            pw[fam] = 0.0
    pw = pw[ff.FAMILY_ORDER_5].fillna(0.0).reset_index()

    # AUXILIARY: partition into the ZRM5_C15B base alone (subset of OTHER_SOLID)
    gz = (pr[pr["phase_model_base_name"] == AUX_BASE]
          .groupby(["state_id", "element"])["element_moles_in_phase"].sum().reset_index()
          .rename(columns={"element_moles_in_phase": "n_aux"}))
    tot_el = (pr.groupby(["state_id", "element"])["element_moles_in_phase"].sum()
                .reset_index().rename(columns={"element_moles_in_phase": "n_tot"}))
    aux = tot_el.merge(gz, on=["state_id", "element"], how="left").fillna({"n_aux": 0.0})
    aux["P_AUX_ZRM5_C15B"] = np.where(aux["n_tot"] > 0, aux["n_aux"] / aux["n_tot"], np.nan)
    pw = pw.merge(aux[["state_id", "element", "P_AUX_ZRM5_C15B"]], on=["state_id", "element"],
                  how="left")
    pw["P_AUX_C15_family_wide"] = pw["C15"] + pw["P_AUX_ZRM5_C15B"].fillna(0.0)

    # wide per state: P_<EL>_<FAM>  (vectorised - a per-state Python loop would be
    # O(N^2) at 22000 states, which is exactly the defect Step 3C exists to remove)
    x_elements = sorted({str(x).upper() for x in meta["X_element"].unique() if str(x).strip()})
    keep_els = ["CR", "NB"] + [e for e in x_elements if e not in ("CR", "NB")]
    pwk = pw[pw["element"].isin(keep_els)].copy()
    val_cols = ff.FAMILY_ORDER_5 + ["P_AUX_C15_family_wide"]
    piv = pwk.pivot_table(index="state_id", columns="element", values=val_cols, aggfunc="sum")
    piv.columns = ["P_%s_%s" % (el, "C15_family_wide_AUX" if v == "P_AUX_C15_family_wide" else v)
                   for v, el in piv.columns]
    piv = piv.reset_index()
    for el in keep_els:
        cols = ["P_%s_%s" % (el, f) for f in ff.FAMILY_ORDER_5 if "P_%s_%s" % (el, f) in piv.columns]
        piv["P_%s_sum" % el] = piv[cols].sum(axis=1, min_count=1) if cols else np.nan
    smeta = states[["state_id", "case_id", "family", "Temperature_C"]].copy()
    part_agg = smeta.merge(piv, on="state_id", how="inner").rename(columns={"case_id": "composition_id"})
    part_agg["X_element"] = part_agg["composition_id"].map(
        lambda c: (meta.loc[c, "X_element"] if c in meta.index else ""))
    part_agg["sampling_domain"] = part_agg["composition_id"].map(
        lambda c: (meta.loc[c, "sampling_domain"] if c in meta.index else ""))
    front = ["state_id", "composition_id", "family", "X_element", "sampling_domain", "Temperature_C"]
    part_agg = part_agg[front + [c for c in part_agg.columns if c not in front]]

    # ------------------------------------------------- 5-family wide fractions -- #
    wide = io_store.wide_5family(agg, states["state_id"].tolist(), ff.FAMILY_ORDER_5)
    # AUXILIARY volume fractions
    z = (agg[agg["phase_model_base_name"] == AUX_BASE]
         .groupby("state_id")["phase_volume_fraction_total"].sum())
    wide["ZRM5_C15B_volfrac_AUX"] = wide["state_id"].map(z).fillna(0.0)
    wide["C15_family_wide_volfrac_AUX"] = (wide["C15_total_volume_fraction"]
                                           + wide["ZRM5_C15B_volfrac_AUX"])
    sigma = (agg[agg["phase_model_base_name"] == "SIGMA"]
             .groupby("state_id")["phase_volume_fraction_total"].sum())
    wide["SIGMA_volfrac_AUX"] = wide["state_id"].map(sigma).fillna(0.0)

    # -------------------------------------------------------------- ML preview -- #
    st = states[["state_id", "case_id", "family", "Temperature_C", "Temperature_K",
                 "converged_b", "solver_level_used", "bridge_used"]].copy()
    ml = st.rename(columns={"case_id": "composition_id", "converged_b": "state_converged"})
    ml["state_converged"] = ml["state_converged"].astype(int)
    for col in ("X_element", "sampling_domain", "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct",
                "X_wt_pct", "R_CrNb_atomic_actual", "S_CrNb_wt_pct"):
        ml[col] = ml["composition_id"].map(
            lambda c, col=col: (meta.loc[c, col] if c in meta.index else ""))
    for col in ("Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
                "R_CrNb_atomic_actual", "S_CrNb_wt_pct"):
        ml[col] = pd.to_numeric(ml[col], errors="coerce")
    ml = ml.rename(columns={"R_CrNb_atomic_actual": "CrNb_atomic_ratio"})
    ml = ml.merge(wide, on="state_id", how="left", suffixes=("", "_w"))

    for fam in ff.FAMILY_ORDER_5:
        ml["%s_volfrac" % fam] = ml["%s_total_volume_fraction" % fam]
        ml["%s_molefrac" % fam] = ml["%s_total_mole_fraction" % fam]
        ml["%s_presence" % fam] = (ml["%s_volfrac" % fam] > PRESENCE_THRESHOLD).astype(int)
    ml.loc[ml.state_converged == 0, [c for c in ml.columns
                                     if c.endswith(("_volfrac", "_molefrac"))]] = np.nan
    ml.loc[ml.state_converged == 0, [c for c in ml.columns if c.endswith("_presence")]] = 0
    ml["solid_volume_fraction"] = 1.0 - ml["LIQUID_volfrac"].fillna(0.0)
    ml.loc[ml.state_converged == 0, "solid_volume_fraction"] = np.nan
    ml["has_liquid"] = ml["LIQUID_presence"]

    npz = phases.groupby("state_id").size()
    ml["phase_count"] = ml["state_id"].map(npz).fillna(0).astype(int)
    fcc_n = (phases[phases.aggregated_family == "FCC"].groupby("state_id").size())
    bcc_n = (phases[phases.aggregated_family == "BCC"].groupby("state_id").size())
    ml["has_multiple_FCC_instances"] = (ml["state_id"].map(fcc_n).fillna(0) >= 2).astype(int)
    ml["has_multiple_BCC_instances"] = (ml["state_id"].map(bcc_n).fillna(0) >= 2).astype(int)

    # partition columns onto the ML frame, per element role
    pa = part_agg.set_index("state_id")
    for role, els in (("CR", None), ("NB", None)):
        for f in ff.FAMILY_ORDER_5:
            ml["%s_P_%s" % (role, f)] = ml["state_id"].map(pa["P_%s_%s" % (role, f)]) \
                if "P_%s_%s" % (role, f) in pa.columns else np.nan
        ml["%s_P_C15_family_wide_AUX" % role] = ml["state_id"].map(
            pa["P_%s_C15_family_wide_AUX" % role]) if "P_%s_C15_family_wide_AUX" % role in pa.columns else np.nan
    # X columns: the element differs per row, so assign per X-element group
    # (baseline rows have no X and stay NaN by construction - never 0).
    for f in ff.FAMILY_ORDER_5:
        ml["X_P_%s" % f] = np.nan
    ml["X_P_C15_family_wide_AUX"] = np.nan
    for xel in sorted({str(x).upper() for x in ml["X_element"].unique() if str(x).strip()}):
        m = ml["X_element"].astype(str).str.upper() == xel
        for f in ff.FAMILY_ORDER_5:
            c = "P_%s_%s" % (xel, f)
            if c in pa.columns:
                ml.loc[m, "X_P_%s" % f] = ml.loc[m, "state_id"].map(pa[c]).values
        c = "P_%s_C15_family_wide_AUX" % xel
        if c in pa.columns:
            ml.loc[m, "X_P_C15_family_wide_AUX"] = ml.loc[m, "state_id"].map(pa[c]).values

    xp = ml[["X_P_%s" % f for f in ff.FAMILY_ORDER_5]]
    ml["dominant_X_partition_family"] = _argmax_label(xp, "X_P_")
    xs = ml[["X_P_%s" % f for f in ff.SOLID_FAMILIES]]
    ssum = xs.sum(axis=1, min_count=1)
    for f in ff.SOLID_FAMILIES:
        ml["X_Psolid_%s" % f] = np.where(ssum > 0, ml["X_P_%s" % f] / ssum, np.nan)
    xsn = ml[["X_Psolid_%s" % f for f in ff.SOLID_FAMILIES]]
    ml["dominant_X_solid_partition_family"] = _argmax_label(xsn, "X_Psolid_")

    # ------------------------------------------------------------ liquid diag -- #
    liq = ml[(ml.state_converged == 1) & (ml.LIQUID_volfrac > 0)][
        ["state_id", "composition_id", "family", "sampling_domain", "Temperature_C",
         "LIQUID_volfrac", "LIQUID_molefrac", "X_element"]].copy()

    # ----------------------------------------------------------------- write --- #
    states_out = states.drop(columns=["converged_b"]).sort_values(["case_id", "Temperature_C"])
    states_out.to_csv(DIR_RAW / "FULL_V1_EQUILIBRIUM_STATES.csv", index=False, encoding="utf-8")
    f1 = io_store.write_table(phases, DIR_RAW / "FULL_V1_PHASES_LONG")
    f2 = io_store.write_table(pcomp, DIR_RAW / "FULL_V1_PHASE_COMPOSITIONS_LONG")
    f3 = io_store.write_table(part_raw, DIR_RAW / "FULL_V1_ELEMENT_PARTITIONING_RAW")
    f4 = io_store.write_table(agg, DIR_PROC / "FULL_V1_AGGREGATED_PHASES_LONG")
    f5 = io_store.write_table(wide, DIR_PROC / "FULL_V1_AGGREGATED_PHASES")
    f6 = io_store.write_table(part_agg, DIR_PROC / "FULL_V1_ELEMENT_PARTITIONING_AGGREGATED")
    f7 = io_store.write_table(ml, DIR_PROC / "FULL_V1_ML_TARGET_PREVIEW")
    liq.to_csv(DIR_QA / "FULL_V1_LIQUID_DIAGNOSTICS.csv", index=False, encoding="utf-8")

    aux_cols = sorted({c for c in list(ml.columns) + list(wide.columns) + list(part_agg.columns)
                       if c.endswith("_AUX") or "C15_family_wide" in c or c.startswith("ZRM5_")})
    (DIR_PROC / "FULL_V1_AUXILIARY_COLUMNS.txt").write_text(
        "# AUXILIARY / DIAGNOSTIC columns - NOT core 5-family targets.\n"
        "# Never sum these together with FCC/BCC/C15/OTHER_SOLID/LIQUID: ZRM5_C15B is\n"
        "# already counted inside OTHER_SOLID, so doing so would double-count it.\n"
        + "\n".join(aux_cols) + "\n", encoding="utf-8")

    summary = {"states": len(states_out), "converged": int(states["converged_b"].sum()),
               "phase_rows": len(phases), "phase_composition_rows": len(pcomp),
               "partition_raw_rows": len(part_raw), "agg_rows": len(agg),
               "partition_agg_rows": len(part_agg), "ml_rows": len(ml),
               "liquid_states": len(liq),
               "files": [f.name for f in (f1, f2, f3, f4, f5, f6, f7)],
               "auxiliary_columns": aux_cols,
               "parquet_available": io_store.have_parquet()}
    (DIR_PROC / "FULL_V1_PARTITIONING_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
