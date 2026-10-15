# -*- coding: utf-8 -*-
"""
73_f5f_features.py
STEP5F Stage C2 (ML side): candidate features for the five new identities,
formula-IDENTICAL to STEP5A Stage G (32_step5a_features.py) - no feature
reselection, no new feature types (spec 28, 29).

Three build blocks:
  1. ELEMENT DESCRIPTOR TABLE EXTENSION
     The frozen STEP5A table (9 dev elements) is RECOMPUTED from the pinned
     sources (matminer MagpieData / mendeleev / CohesiveEnergyData) and must
     reproduce the frozen values exactly; then BE SN MN LA P are appended from
     the same sources, same policy (drop a property only if any element lacks
     it; none do - verified at runtime).
  2. COMPOSITION-LEVEL FEATURES (F1/F2/F3/F4) for the 10,000 new-element
     compositions, using the exact STEP5A formulas and the exact
     4-decimal wt.% -> atomic fraction conversion.
  3. EQUIVALENCE PROOF - a deterministic sample of existing FULL_V1
     compositions is pushed through THIS pipeline and compared against the
     frozen STEP5A candidate feature table; any drift aborts the stage.

Output:
  03_processed/STEP5F_NEW_ELEMENT_CANDIDATE_FEATURES.parquet
  00_config/STEP5F_ELEMENT_DESCRIPTOR_TABLE_14X_V1.{csv,parquet}
  08_qa/STEP5F_FEATURE_BUILD_SUMMARY.json

Ni is never queried against any descriptor source (spec 1).
"""
import importlib.metadata as md_ver
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5f_common as C

R_GAS = 8.314462618
DEV9 = ["CU", "CR", "NB", "ZR", "TI", "CO", "SI", "AG", "MO"]
NEW5 = C.SELECTED_ELEMENTS                                   # BE SN MN LA P
ALL14 = DEV9 + NEW5
P_SYMBOL = {"CU": "Cu", "CR": "Cr", "NB": "Nb", **C.X_SYMBOL}
F3_PROP = ["AtomicNumber", "AtomicWeight", "MetallicRadius", "Electronegativity", "VEC",
           "MeltingT", "Density", "FirstIonizationEnergy", "CohesiveEnergy"]
F2_MEAN_PROPS = [("Zbar", "AtomicNumber"), ("M_bar", "AtomicWeight"),
                 ("CovalentR_bar", "CovalentRadius"), ("chi_bar", "Electronegativity"),
                 ("VEC_mix", "VEC"), ("Tm_mix", "MeltingT"), ("density_bar", "Density"),
                 ("Ecoh_bar", "CohesiveEnergy"), ("IE1_bar", "FirstIonizationEnergy"),
                 ("EA_bar", "ElectronAffinity"), ("Mendeleev_bar", "MendeleevNumber"),
                 ("AtomicVol_bar", "AtomicVolume_GS")]
# (property, source_key, field_name) - identical to 31_step5a_element_descriptors.py
PROPERTIES = [
    ("AtomicNumber", "magpie", "Number"),
    ("AtomicWeight", "magpie", "AtomicWeight"),
    ("CovalentRadius", "magpie", "CovalentRadius"),
    ("Electronegativity", "magpie", "Electronegativity"),
    ("VEC", "magpie", "NValence"),
    ("MeltingT", "magpie", "MeltingT"),
    ("FirstIonizationEnergy", "magpie", "FirstIonizationEnergy"),
    ("ElectronAffinity", "magpie", "ElectronAffinity"),
    ("AtomicVolume_GS", "magpie", "GSvolume_pa"),
    ("MetallicRadius", "mendeleev", "atomic_radius"),
    ("Density", "mendeleev", "density"),
    ("MendeleevNumber", "mendeleev", "mendeleev_number"),
    ("CohesiveEnergy", "cohesive", "cohesive energy"),
]


def build_descriptor_rows(elements):
    from matminer.utils.data import MagpieData, CohesiveEnergyData
    from mendeleev import element as mend_element
    from pymatgen.core.periodic_table import Element
    magpie, cohesive = MagpieData(), CohesiveEnergyData()
    rows = {e: {"element": e} for e in elements}
    for prop, src, field in PROPERTIES:
        for e in elements:
            sym = P_SYMBOL[e]
            if src == "magpie":
                v = magpie.get_elemental_property(Element(sym), field)
            elif src == "cohesive":
                v = cohesive.get_elemental_property(sym, field)
            else:
                v = getattr(mend_element(sym), field)
            rows[e][prop] = None if v is None or v != v else float(v)
    return pd.DataFrame([rows[e] for e in elements])


def build_f2_f3(comp_lvl, desc):
    """Verbatim port of 32_step5a_features.build_f2_f3 (X-bearing rows only here)."""
    P = {e: desc.set_index("element").loc[e] for e in ALL14}
    out = pd.DataFrame(index=comp_lvl.index)
    x = np.column_stack([comp_lvl["x_Cu"], comp_lvl["x_Cr"], comp_lvl["x_Nb"],
                         comp_lvl["x_X"]])
    base_elems = ["CU", "CR", "NB"]
    # BASELINE rows carry the Cu property in the 4th column (verbatim STEP5A behaviour);
    # their x_X is 0 so the column never contributes.
    x_elem_code = np.array([f if f != "BASELINE" else "CU" for f in comp_lvl["family"].values])

    def prop_matrix(prop):
        m = np.column_stack([np.full(len(comp_lvl), P[e][prop]) for e in base_elems])
        col4 = np.array([P[c][prop] for c in x_elem_code])
        return np.column_stack([m, col4])

    for name, prop in F2_MEAN_PROPS:
        out[name] = (prop_matrix(prop) * x).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        xlogx = np.where(x > 0, x * np.log(x), 0.0)
    out["Delta_Smix"] = -R_GAS * xlogx.sum(axis=1)
    r = prop_matrix("MetallicRadius")
    r_bar = (r * x).sum(axis=1)
    out["r_bar"] = r_bar
    ratio = np.where(x > 0, 1.0 - r / r_bar[:, None], 0.0)
    out["delta_r"] = 100.0 * np.sqrt((x * ratio ** 2).sum(axis=1))
    chi = prop_matrix("Electronegativity")
    chi_bar = (chi * x).sum(axis=1)
    out["delta_chi"] = np.sqrt((x * (chi - chi_bar[:, None]) ** 2).sum(axis=1))
    has_X = (comp_lvl["family"] != "BASELINE").values
    for p in F3_PROP:
        v = np.array([P[c][p] if h else 0.0 for c, h in zip(x_elem_code, has_X)])
        out["X_%s" % p] = v
        for base, sym in [("Cu", "CU"), ("Cr", "CR"), ("Nb", "NB")]:
            out["Delta_%s_X_%s" % (p, base)] = v - (P[sym][p] * has_X)
    return out


def magpie_f4(comp_lvl):
    from pymatgen.core.composition import Composition
    from matminer.featurizers.composition import ElementProperty
    ep = ElementProperty.from_preset("magpie", impute_nan=False)
    comps = []
    for fam, xc in zip(comp_lvl["family"],
                       comp_lvl[["x_Cu", "x_Cr", "x_Nb", "x_X"]].itertuples(index=False)):
        d = {"Cu": xc.x_Cu, "Cr": xc.x_Cr, "Nb": xc.x_Nb}
        if fam != "BASELINE":
            d[C.X_SYMBOL[fam]] = xc.x_X
        comps.append(Composition({k: v for k, v in d.items() if v > 0}))
    feats = [ep.featurize(c) for c in comps]
    return pd.DataFrame(
        feats, columns=[c.replace("MagpieData ", "magpie_").replace(" ", "_")
                        for c in ep.feature_labels()])


def composition_level_table(comp):
    """comp: DataFrame with composition_id family X_element Cu/Cr/Nb/X_wt_pct
    S_CrNb_wt_pct R_CrNb_atomic_actual - formula-identical wt%->atomic conversion."""
    desc = pd.read_parquet(os.path.join(C.D_CONFIG,
                                        "STEP5F_ELEMENT_DESCRIPTOR_TABLE_14X_V1.parquet"))
    M = desc.set_index("element")["AtomicWeight"]
    w = comp[["Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct"]].astype(float).copy()
    w["X_wt_pct"] = w["X_wt_pct"].fillna(0.0)
    comp = comp.copy()
    comp["X_wt_pct"] = w["X_wt_pct"]   # propagate the filled value back (STEP5A sec.27)
    mass = {"CU": M["CU"], "CR": M["CR"], "NB": M["NB"]}
    xmat = np.column_stack([w["Cu_wt_pct"] / mass["CU"], w["Cr_wt_pct"] / mass["CR"],
                            w["Nb_wt_pct"] / mass["NB"]])
    xM = np.array([M[f] if f != "BASELINE" else 1.0 for f in comp["family"]])
    nx = w["X_wt_pct"].values / xM
    xmat = np.column_stack([xmat, nx])
    xat = xmat / xmat.sum(axis=1)[:, None]
    comp = comp.copy()
    comp["x_Cu"], comp["x_Cr"], comp["x_Nb"], comp["x_X"] = xat.T
    sum_dev = float(np.abs(xat.sum(axis=1) - 1.0).max())
    assert sum_dev <= 1e-12, sum_dev
    f2f3 = build_f2_f3(comp, desc)
    lvl = pd.concat([comp[["composition_id", "family", "X_element", "sampling_domain",
                           "split_role", "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct",
                           "X_wt_pct", "S_CrNb_wt_pct", "R_CrNb_atomic_actual",
                           "x_Cu", "x_Cr", "x_Nb", "x_X"]], f2f3], axis=1)
    lvl["has_X"] = (lvl["family"] != "BASELINE").astype(int)
    f4 = magpie_f4(lvl)
    assert f4.notna().all().all(), "Magpie preset produced NaN"
    lvl = pd.concat([lvl.reset_index(drop=True), f4.reset_index(drop=True)], axis=1)
    return lvl, desc, sum_dev


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5F FULL  Stage C2: candidate features for the five new identities")
    print("=" * 78)

    # ------- 1. descriptor table: reproduce frozen 9, then extend by 5 ----- #
    frozen = pd.read_parquet(C.STEP5A_DESC)
    repro = build_descriptor_rows(DEV9)
    fr = frozen.set_index("element").loc[DEV9]
    rp = repro.set_index("element").loc[DEV9]
    drift = []
    for col in rp.columns:
        a, b = fr[col].astype(float).to_numpy(), rp[col].astype(float).to_numpy()
        if not np.array_equal(a, b):
            drift.append(col)
    if drift:
        raise SystemExit("FATAL: descriptor reproduction drifted for %s - the frozen "
                         "STEP5A source values changed; aborting before any feature is "
                         "written." % drift)
    print("  descriptor reproduction of the frozen 9 elements: EXACT (13 properties)")
    new_rows = build_descriptor_rows(NEW5)
    nulls = new_rows.isna().sum().sum()
    assert nulls == 0, "missing descriptor values for the new elements: %s" % \
        new_rows.columns[new_rows.isna().any()].tolist()
    table = pd.concat([frozen, new_rows], ignore_index=True)
    table.to_csv(os.path.join(C.D_CONFIG, "STEP5F_ELEMENT_DESCRIPTOR_TABLE_14X_V1.csv"),
                 index=False)
    table.to_parquet(os.path.join(C.D_CONFIG, "STEP5F_ELEMENT_DESCRIPTOR_TABLE_14X_V1.parquet"),
                     index=False)
    print("  extended descriptor table: %d elements x %d properties"
          % (len(table), len(table.columns) - 1))

    # ------------- 2. composition-level features for the 10,000 ------------ #
    comp = pd.read_csv(os.path.join(C.D_COMP, "STEP5F_FULL_COMPOSITIONS.csv"))
    comp = comp.rename(columns={})
    lvl, _, sum_dev = composition_level_table(comp)
    print("  composition-level features built for %d compositions (atomic closure %.2e)"
          % (len(lvl), sum_dev))

    # -------- 3. equivalence proof against the frozen STEP5A table --------- #
    cand = pd.read_parquet(C.STEP5A_FEAT)
    fv1 = pd.read_parquet(os.path.join(C.FULL_V1, "01_compositions", "FULL_V1_COMPOSITIONS.parquet"),
                          columns=["composition_id", "family", "X_element", "sampling_domain",
                                   "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
                                   "S_CrNb_wt_pct", "R_CrNb_atomic_actual"])
    # probe only compositions the frozen candidate table actually covers
    # (Domain-A development rows; Domain B / ANCHOR were never featurized)
    fv1 = fv1[fv1.composition_id.isin(set(cand.composition_id))]
    rng = np.random.default_rng(20260910)
    sample = []
    for fam in ["ZR", "TI", "CO", "SI", "AG", "MO", "BASELINE"]:
        sub = fv1[fv1.family == fam]
        take = min(20, len(sub))
        sample.append(sub.iloc[rng.permutation(len(sub))[:take]])
    probe = pd.concat(sample, ignore_index=True)
    probe["split_role"] = "EQUIVALENCE_PROBE"
    probe["R_CrNb_atomic_actual"] = probe["R_CrNb_atomic_actual"].astype(float)
    probe_lvl, _, _ = composition_level_table(probe)
    one_state = (cand.sort_values("state_id").groupby("composition_id", as_index=False)
                 .first())
    ref = one_state.set_index("composition_id").loc[probe.composition_id]
    pl = probe_lvl.set_index("composition_id").loc[ref.index]
    feat_cols = [c for c in ref.columns if c in pl.columns
                 and pd.api.types.is_numeric_dtype(ref[c])
                 and pd.api.types.is_numeric_dtype(pl[c])]
    maxdev, bad = 0.0, []
    for col in feat_cols:
        a = pd.to_numeric(ref[col], errors="coerce").to_numpy(float)
        b = pd.to_numeric(pl[col], errors="coerce").to_numpy(float)
        if np.isnan(a).any() != np.isnan(b).any():
            bad.append(col)
            continue
        d = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
        if d > 1e-9:
            bad.append("%s(%.3e)" % (col, d))
        maxdev = max(maxdev, d)
    print("  equivalence probe: %d compositions x %d features | max |delta| = %.3e | bad: %s"
          % (len(probe), len(feat_cols), maxdev, bad or "none"))
    if bad:
        raise SystemExit("FATAL: feature pipeline is NOT formula-identical to STEP5A: %s" % bad)

    # -------------------- join to the 110,000 new states ------------------- #
    states = pd.read_parquet(os.path.join(C.D_PROC, "STEP5F_FULL_NEW_ELEMENT_DATA.parquet"),
                             columns=["state_id", "composition_id", "element", "X_element",
                                      "Temperature_C", "state_converged"])
    states = states.rename(columns={"element": "family"})
    states["sampling_domain"] = "A"
    states["split_role"] = "AUGMENTATION_TRAIN"
    keep_meta = ["state_id", "composition_id", "family", "X_element", "sampling_domain",
                 "split_role", "Temperature_C"]
    feats_cols = [c for c in lvl.columns if c not in
                  ("composition_id", "family", "X_element", "sampling_domain", "split_role")]
    out = states.merge(lvl[["composition_id"] + feats_cols], on="composition_id",
                       how="left", validate="m:1")
    assert out[feats_cols].notna().all().all(), "feature join left NaN rows"
    out["Temperature_K"] = out["Temperature_C"] + 273.15
    ordered = keep_meta + ["Temperature_K"] + [c for c in feats_cols if c != "Temperature_K"]
    out = out[ordered]
    out.to_parquet(os.path.join(C.D_PROC, "STEP5F_NEW_ELEMENT_CANDIDATE_FEATURES.parquet"),
                   index=False)

    # union of the frozen fold feature lists must be a subset
    union = []
    for e in C.HELDOUT_ELEMENTS:
        for ln in open(C.C_FEATLIST(e), encoding="utf-8"):
            f = ln.strip()
            if f and f not in union:
                union.append(f)
    missing = [f for f in union if f not in out.columns]
    assert not missing, "union feature missing: %s" % missing
    fin = out[union]
    assert np.isfinite(fin.to_numpy(float)).all(), "non-finite union feature for new rows"
    print("  STEP5F_NEW_ELEMENT_CANDIDATE_FEATURES.parquet: %d rows, %d feature columns; "
          "frozen 51-feature union present and finite" % (len(out), len(feats_cols)))

    hits = C.ni_scan([("features", out), ("descriptors", table)])
    C.write_json(os.path.join(C.D_QA, "STEP5F_FEATURE_BUILD_SUMMARY.json"), {
        "generated_utc": ts,
        "descriptor_reproduction_exact": True,
        "n_elements_descriptor_table": int(len(table)),
        "new_elements": NEW5,
        "n_compositions_featurized": int(len(lvl)),
        "atomic_closure_max_dev": sum_dev,
        "equivalence_probe_compositions": int(len(probe)),
        "equivalence_probe_max_abs_delta": maxdev,
        "union_features": len(union),
        "output_rows": int(len(out)),
        "matminer": md_ver.version("matminer"), "mendeleev": md_ver.version("mendeleev"),
        "ni_scan_hits": hits,
        "output_sha256": C.sha256_file(
            os.path.join(C.D_PROC, "STEP5F_NEW_ELEMENT_CANDIDATE_FEATURES.parquet"))})
    print("STAGE C2: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
