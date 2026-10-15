# -*- coding: utf-8 -*-
"""
32_step5a_features.py
STEP5A Stage G: candidate feature construction (F1-F4) for Domain-A
development rows only (DEV_TRAIN / DEV_VALIDATION / IID_TEST).

Leakage-safe by construction:
  - inputs: raw 4-decimal wt.% + temperature (as fed to Thermo-Calc),
    Stage-F element descriptor table, Magpie ElementProperty preset
  - NO CALPHAD target column, NO derived-descriptor column, NO family /
    X_element / sampling_domain / solver metadata enters the feature matrix
    (they are carried only as metadata columns)
  - Domain B and ANCHOR rows are NOT featurized in this stage (sealed)

Feature blocks:
  F1  raw composition + temperature (7 + has_X)
  F2  general composition-physics descriptors from atomic fractions (16)
      + temperature-normalized physics (1)
  F3  X-specific physicochemical descriptors and differences vs Cu/Cr/Nb (36)
  F4  Magpie ElementProperty preset stats (composition-level, 132)

Outputs:
  02_features/FULL_V1_CANDIDATE_FEATURES.parquet
  02_features/F1_COMPOSITION_LEVEL.parquet        (50k, sealed-set rows: F1 only)
  02_features/FEATURE_DICTIONARY.csv
  02_features/WT_TO_ATOMIC_CONVERSION_QA.md
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

FULL_V1 = r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1"
STEP5A = r"E:\Cu_CALPHAD_ML\04_ML\STEP5A_JMST_ML_FOUNDATION_V2"
FEAT = os.path.join(STEP5A, "02_features")
SRC = os.path.join(STEP5A, "01_feature_sources")
SPLITS = os.path.join(STEP5A, "03_splits")

R_GAS = 8.314462618  # J/(mol K)
DEV_ROLES = ["DEV_TRAIN", "DEV_VALIDATION", "IID_TEST"]
X_SYMBOL = {"ZR": "Zr", "TI": "Ti", "CO": "Co", "SI": "Si", "AG": "Ag", "MO": "Mo"}
P_SYMBOL = {"CU": "Cu", "CR": "Cr", "NB": "Nb", **X_SYMBOL}
CORE_P = ["CovalentRadius", "Electronegativity", "VEC", "MeltingT"]  # complete for all 9
F3_PROP = ["AtomicNumber", "AtomicWeight", "MetallicRadius", "Electronegativity", "VEC",
           "MeltingT", "Density", "FirstIonizationEnergy", "CohesiveEnergy"]
F2_MEAN_PROPS = [("Zbar", "AtomicNumber"), ("M_bar", "AtomicWeight"),
                 ("CovalentR_bar", "CovalentRadius"), ("chi_bar", "Electronegativity"),
                 ("VEC_mix", "VEC"), ("Tm_mix", "MeltingT"), ("density_bar", "Density"),
                 ("Ecoh_bar", "CohesiveEnergy"), ("IE1_bar", "FirstIonizationEnergy"),
                 ("EA_bar", "ElectronAffinity"), ("Mendeleev_bar", "MendeleevNumber"),
                 ("AtomicVol_bar", "AtomicVolume_GS")]


def build_f2_f3(comp_lvl: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    """comp_lvl: one row per composition with x_Cu x_Cr x_Nb x_X, family, X_element."""
    P = {e: desc.set_index("element").loc[e] for e in ["CU", "CR", "NB", "ZR", "TI", "CO", "SI", "AG", "MO"]}
    out = pd.DataFrame(index=comp_lvl.index)

    x = np.column_stack([comp_lvl["x_Cu"], comp_lvl["x_Cr"], comp_lvl["x_Nb"], comp_lvl["x_X"]])
    base_elems = ["CU", "CR", "NB"]
    # element property matrix per row: 4th element depends on X identity
    fam_sym = comp_lvl["family"].values  # BASELINE or ZR/TI/...
    x_elem_code = np.array([f if f != "BASELINE" else "CU" for f in fam_sym])

    def prop_matrix(prop):
        m = np.column_stack([np.full(len(comp_lvl), P[e][prop]) for e in base_elems])
        col4 = np.array([P[c][prop] for c in x_elem_code])
        return np.column_stack([m, col4])

    # weighted means over the 4 species
    for name, prop in F2_MEAN_PROPS:
        out[name] = (prop_matrix(prop) * x).sum(axis=1)

    # configurational entropy (x_i > 0 only)
    with np.errstate(divide="ignore", invalid="ignore"):
        xlogx = np.where(x > 0, x * np.log(x), 0.0)
    out["Delta_Smix"] = -R_GAS * xlogx.sum(axis=1)

    # size mismatch with metallic radius
    r = prop_matrix("MetallicRadius")
    r_bar = (r * x).sum(axis=1)
    out["r_bar"] = r_bar
    ratio = np.where(x > 0, 1.0 - r / r_bar[:, None], 0.0)
    out["delta_r"] = 100.0 * np.sqrt((x * ratio ** 2).sum(axis=1))

    # electronegativity dispersion
    chi = prop_matrix("Electronegativity")
    chi_bar = (chi * x).sum(axis=1)
    out["delta_chi"] = np.sqrt((x * (chi - chi_bar[:, None]) ** 2).sum(axis=1))

    # temperature-normalized physics (state level; added later via T_K)
    # ---- F3: X-specific descriptors and differences ----
    has_X = (comp_lvl["family"] != "BASELINE").values
    for p in F3_PROP:
        v = np.array([P[c][p] if h else 0.0 for c, h in zip(x_elem_code, has_X)])
        out[f"X_{p}"] = v
        for base, sym in [("Cu", "CU"), ("Cr", "CR"), ("Nb", "NB")]:
            out[f"Delta_{p}_X_{base}"] = v - (P[sym][p] * has_X)
    return out


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(FEAT, exist_ok=True)
    desc = pd.read_parquet(os.path.join(SRC, "ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.parquet"))
    split = pd.read_csv(os.path.join(SPLITS, "COMPOSITION_SPLIT_MASTER_V1.csv"))

    comp = pd.read_parquet(
        os.path.join(FULL_V1, "01_compositions", "FULL_V1_COMPOSITIONS.parquet"),
        columns=["composition_id", "family", "X_element", "sampling_domain",
                 "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
                 "S_CrNb_wt_pct", "R_CrNb_atomic_actual"])
    comp = comp.merge(split, on="composition_id", how="left", suffixes=("", "_m"))
    assert (comp["family"] == comp["family_m"]).all() and (comp["sampling_domain"] == comp["sampling_domain_m"]).all()
    comp = comp.drop(columns=["family_m", "sampling_domain_m"])

    # ---------------- wt% -> atomic fraction (4-decimal wt%, NOT *_exact) ----------------
    M = desc.set_index("element")["AtomicWeight"]
    w = comp[["Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct"]].astype(float)
    w["X_wt_pct"] = w["X_wt_pct"].fillna(0.0)
    # BASELINE hard rule: X_wt_pct = 0 (sec.27); propagate back so the F1 feature is 0, not NaN
    comp["X_wt_pct"] = w["X_wt_pct"]
    # BASELINE hard rule: X_wt_pct = 0
    base_mask = comp["family"] == "BASELINE"
    assert (w.loc[base_mask, "X_wt_pct"] == 0).all()
    mass = {"CU": M["CU"], "CR": M["CR"], "NB": M["NB"]}
    xmat = np.column_stack([w["Cu_wt_pct"] / mass["CU"], w["Cr_wt_pct"] / mass["CR"],
                            w["Nb_wt_pct"] / mass["NB"]])
    xM = np.array([M[f] if f != "BASELINE" else 1.0 for f in comp["family"]])
    nx = w["X_wt_pct"].values / xM
    nx[base_mask.values] = 0.0
    xmat = np.column_stack([xmat, nx])
    xsum = xmat.sum(axis=1)
    xat = xmat / xsum[:, None]
    comp["x_Cu"], comp["x_Cr"], comp["x_Nb"], comp["x_X"] = xat.T
    sum_dev = np.abs(xat.sum(axis=1) - 1.0).max()
    # wt% closure check
    wt_sum_dev = np.abs(w.sum(axis=1).values - 100.0).max()

    # ---------------- F2/F3 composition-level ----------------
    f2f3 = build_f2_f3(comp, desc)
    comp_lvl = pd.concat([comp[["composition_id", "family", "X_element", "sampling_domain",
                                "split_role", "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
                                "S_CrNb_wt_pct", "R_CrNb_atomic_actual",
                                "x_Cu", "x_Cr", "x_Nb", "x_X"]], f2f3], axis=1)
    comp_lvl["has_X"] = (comp_lvl["family"] != "BASELINE").astype(int)

    # ---------------- F4 Magpie (composition-level, atomic-fraction compositions) ----------------
    from pymatgen.core.composition import Composition
    from matminer.featurizers.composition import ElementProperty
    ep = ElementProperty.from_preset("magpie", impute_nan=False)
    comps = []
    for fam, xc in zip(comp_lvl["family"], comp_lvl[["x_Cu", "x_Cr", "x_Nb", "x_X"]].itertuples(index=False)):
        d = {"Cu": xc.x_Cu, "Cr": xc.x_Cr, "Nb": xc.x_Nb}
        if fam != "BASELINE":
            d[X_SYMBOL[fam]] = xc.x_X
        comps.append(Composition({k: v for k, v in d.items() if v > 0}))
    try:
        from tqdm import tqdm as _progress
    except ImportError:
        def _progress(it, **kw):
            return it
    feats = []
    for c in _progress(comps, desc="magpie", mininterval=30):
        feats.append(ep.featurize(c))
    f4 = pd.DataFrame(feats, columns=[c.replace("MagpieData ", "magpie_").replace(" ", "_") for c in ep.feature_labels()])
    assert f4.notna().all().all(), "Magpie preset produced NaN"
    comp_lvl = pd.concat([comp_lvl.reset_index(drop=True), f4.reset_index(drop=True)], axis=1)

    # F1 composition-level = raw wt.% columns (S/R/has_X)
    f1_cols = ["Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct", "S_CrNb_wt_pct",
               "R_CrNb_atomic_actual", "has_X"]
    f2_cols = [c for c in ["Zbar", "M_bar", "CovalentR_bar", "chi_bar", "VEC_mix", "Tm_mix",
                           "density_bar", "Ecoh_bar", "IE1_bar", "EA_bar", "Mendeleev_bar",
                           "AtomicVol_bar", "Delta_Smix", "r_bar", "delta_r", "delta_chi"] ]
    f3_cols = ([f"X_{p}" for p in F3_PROP]
               + [f"Delta_{p}_X_{b}" for p in F3_PROP for b in ("Cu", "Cr", "Nb")])
    assert all(c in comp_lvl.columns for c in f3_cols)
    f4_cols = [c for c in comp_lvl.columns if c.startswith("magpie_")]

    # ---------------- join to state level (development rows only) ----------------
    mlr = pd.read_parquet(
        os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet"),
        columns=["state_id", "composition_id", "Temperature_C"])
    comp_dev = comp_lvl[comp_lvl["split_role"].isin(DEV_ROLES)]
    assert (comp_dev["sampling_domain"] == "A").all()
    dev = mlr.merge(comp_dev, on="composition_id", how="inner")
    assert dev["split_role"].isin(DEV_ROLES).all(), "non-development row entered feature table"
    n_expected = 35_000 * 11 - 1_154  # Domain A failed states (MO@950) = 1154
    print(f"development state rows: {len(dev)} (expected {n_expected})")
    assert len(dev) == n_expected, (len(dev), n_expected)

    dev["Temperature_K"] = dev["Temperature_C"] + 273.15
    dev["T_over_Tm_mix"] = dev["Temperature_K"] / dev["Tm_mix"]

    meta_cols = ["state_id", "composition_id", "family", "X_element", "sampling_domain",
                 "split_role", "Temperature_C"]
    feature_cols = ["Temperature_C"] + f1_cols + f2_cols + ["T_over_Tm_mix"] + f3_cols + f4_cols
    cand = dev[meta_cols + ["Temperature_K"] + [c for c in feature_cols if c != "Temperature_C"]].copy()
    # Temperature_K kept as internal helper for physics formulas, NOT a candidate feature
    cand.to_parquet(os.path.join(FEAT, "FULL_V1_CANDIDATE_FEATURES.parquet"), index=False)

    # composition-level table for ALL 50k (F1 only meaningful for sealed rows; kept for STEP5B reuse)
    comp_lvl.to_parquet(os.path.join(FEAT, "COMPOSITION_FEATURES_ALL50K.parquet"), index=False)

    # ---------------- FEATURE_DICTIONARY.csv ----------------
    rows = []
    def add(name, block, formula, source, unit, uses_X, baseline, ood, leak, primary):
        rows.append({"feature_name": name, "feature_block": block, "formula": formula,
                     "source": source, "unit": unit, "uses_X_property": uses_X,
                     "baseline_policy": baseline, "OOD_safe": ood, "leakage_risk": leak,
                     "primary_candidate": primary})

    add("Temperature_C", "F1", "frozen 11-point grid value", "FULL_V1_ML_READY_CONVERGED", "degC",
        "no", "unchanged", "yes", "none", "yes")
    for c, f in [("Cu_wt_pct", "Cu balance wt%"), ("Cr_wt_pct", "Cr wt% (4-decimal, as fed to TC)"),
                 ("Nb_wt_pct", "Nb wt%"), ("X_wt_pct", "X solute wt%"),
                 ("S_CrNb_wt_pct", "wCr + wNb"), ("R_CrNb_atomic_actual", "(wCr/51.9961)/(wNb/92.9064) realised")]:
        add(c, "F1", f, "FULL_V1_COMPOSITIONS", "wt.%" if "wt" in c else "-",
            "yes" if c == "X_wt_pct" else "no",
            "X_wt_pct=0 for BASELINE" if c == "X_wt_pct" else "unchanged", "yes", "none", "yes")
    add("has_X", "F1", "1 if family != BASELINE else 0", "derived from family (metadata use only)", "-",
        "no", "0 for BASELINE", "yes", "none", "yes")
    f2_formulas = {
        "Zbar": "sum x_i Z_i", "M_bar": "sum x_i M_i", "CovalentR_bar": "sum x_i r_cov_i",
        "chi_bar": "sum x_i chi_i", "VEC_mix": "sum x_i VEC_i", "Tm_mix": "sum x_i Tm_i",
        "density_bar": "sum x_i rho_i", "Ecoh_bar": "sum x_i Ecoh_i", "IE1_bar": "sum x_i IE1_i",
        "EA_bar": "sum x_i EA_i", "Mendeleev_bar": "sum x_i MendelevNum_i",
        "AtomicVol_bar": "sum x_i V_GS_i",
        "Delta_Smix": "-R sum x_i ln x_i (x_i>0), R=8.314462618",
        "r_bar": "sum x_i r_metal_i",
        "delta_r": "100*sqrt(sum x_i (1-r_i/r_bar)^2)",
        "delta_chi": "sqrt(sum x_i (chi_i-chi_bar)^2)",
        "T_over_Tm_mix": "Temperature_K / Tm_mix (F2T dimensionless)",
    }
    units = {"Zbar": "-", "M_bar": "amu", "CovalentR_bar": "pm", "chi_bar": "Pauling", "VEC_mix": "count",
             "Tm_mix": "K", "density_bar": "g/cm^3", "Ecoh_bar": "eV/atom", "IE1_bar": "eV",
             "EA_bar": "eV", "Mendeleev_bar": "-", "AtomicVol_bar": "A^3/atom",
             "Delta_Smix": "J/(mol K)", "r_bar": "pm", "delta_r": "%", "delta_chi": "Pauling",
             "T_over_Tm_mix": "-"}
    for c in f2_cols + ["T_over_Tm_mix"]:
        add(c, "F2", f2_formulas[c], "Stage-F descriptor table (magpie/mendeleev/cohesive)", units[c],
            "yes", "x_X=0; X species absent from sums", "yes", "none", "yes")
    for c in f3_cols:
        is_delta = c.startswith("Delta_")
        p = c[2:] if not is_delta else c.split("_")[1]
        add(c, "F3",
            (f"X_{p} - {p}(Cu/Cr/Nb)" if is_delta else f"{p} of X element"),
            "Stage-F descriptor table", "see property", "yes",
            "0 for BASELINE (has_X=0 disambiguates)", "yes", "none", "yes")
    for c in f4_cols:
        stat, prop = c.replace("magpie_", "").split("_", 1) if c.count("_") >= 2 else ("", c)
        add(c, "F4", f"Magpie preset stat '{stat}' over property '{prop}', weighted by atomic fractions",
            f"matminer ElementProperty.from_preset('magpie')", "see Magpie", "yes",
            "X species absent for BASELINE", "yes", "none", "yes")
    pd.DataFrame(rows).to_csv(os.path.join(FEAT, "FEATURE_DICTIONARY.csv"), index=False)

    # ---------------- WT_TO_ATOMIC_CONVERSION_QA.md ----------------
    with open(os.path.join(FEAT, "WT_TO_ATOMIC_CONVERSION_QA.md"), "w", encoding="utf-8") as f:
        f.write("# WT.% -> ATOMIC FRACTION CONVERSION QA (sec.28)\n\n")
        f.write(f"generated_utc: {ts}\n\n")
        f.write("- Input: the 4-decimal wt.% columns actually fed to Thermo-Calc "
                "(Cu/Cr/Nb/X_wt_pct from FULL_V1_COMPOSITIONS.parquet). *_exact columns NOT used.\n")
        f.write("- Formula: n_i = w_i / M_i ; x_i = n_i / sum_j n_j.\n")
        f.write(f"- Atomic masses M_i from the Stage-F descriptor table (matminer MagpieData 'AtomicWeight'): "
                f"Cu {M['CU']}, Cr {M['CR']}, Nb {M['NB']}, Zr {M['ZR']}, Ti {M['TI']}, Co {M['CO']}, "
                f"Si {M['SI']}, Ag {M['AG']}, Mo {M['MO']} amu.\n")
        f.write(f"- BASELINE rows: X_wt_pct = 0 and x_X = 0 exactly ({int(base_mask.sum())} compositions).\n")
        f.write(f"- max |sum(wt%) - 100| = {wt_sum_dev:.3e}\n")
        f.write(f"- max |sum(x_i) - 1| = {sum_dev:.3e}  (tolerance 1e-12: "
                f"{'PASS' if sum_dev <= 1e-12 else 'FAIL'})\n")
        f.write(f"- rows written to FULL_V1_CANDIDATE_FEATURES.parquet: {len(cand)} "
                f"(development Domain-A converged states; expected {n_expected})\n\n")
        f.write("Sample (first 3 development compositions):\n\n")
        f.write(comp_lvl.head(3)[["composition_id", "family", "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct",
                                  "X_wt_pct", "x_Cu", "x_Cr", "x_Nb", "x_X"]].to_markdown(index=False))
        f.write("\n")
    assert sum_dev <= 1e-12, f"atomic fraction closure failed: {sum_dev}"

    summary = {
        "generated_utc": ts, "n_candidate_features": len(feature_cols),
        "n_F1": len(f1_cols) + 1, "n_F2": len(f2_cols) + 1, "n_F3": len(f3_cols), "n_F4": len(f4_cols),
        "F1": ["Temperature_C"] + f1_cols, "F2": f2_cols + ["T_over_Tm_mix"],
        "F3": f3_cols, "F4_n": len(f4_cols),
        "development_state_rows": len(cand), "wt_closure_max_dev": float(wt_sum_dev),
        "atomic_closure_max_dev": float(sum_dev),
    }
    with open(os.path.join(FEAT, "FEATURE_BUILD_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))
    print("STAGE G: PASS")


if __name__ == "__main__":
    main()
