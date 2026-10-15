# -*- coding: utf-8 -*-
"""
92_h_features.py
STEP5H-A Stage 2: state-table expansion (spec 8) + EXACT 43-feature construction
(spec 9, 10) + feature validation (spec 11) + the source-consistency test against
historical frozen features (spec 12).

The feature formulas are the frozen STEP5A ones, character-for-character in
mathematics:
  - wt% -> atomic fraction with descriptor-table atomic masses
  - F2 weighted means / delta_r / delta_chi over the 4 species
  - F3 X_<p> and Delta_<p>_X_Cu / _Cr differences
  - F4 the frozen matminer ElementProperty.from_preset("magpie") implementation
No "equivalent implementation" is used: the SAME code path computes the six
existing-element reference compositions and must reproduce their historical
frozen feature values to <= 1e-10 before any Ni feature is accepted (spec 12).

Output: 02_ni_features/STEP5H_A_NI_FEATURES.parquet  (22000 x 43, exact schema
order, metadata columns first). No target column of any kind (spec 39 guard).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5h_common as C

# F3 properties needed by the 43-feature schema, and which base differences
F3_NEEDED = {
    "X_AtomicNumber": ("AtomicNumber", None),
    "X_VEC": ("VEC", None),
    "X_MeltingT": ("MeltingT", None),
    "Delta_AtomicNumber_X_Cu": ("AtomicNumber", "CU"),
    "Delta_Electronegativity_X_Cu": ("Electronegativity", "CU"),
    "Delta_VEC_X_Cu": ("VEC", "CU"),
    "Delta_VEC_X_Cr": ("VEC", "CR"),
    "Delta_MeltingT_X_Cu": ("MeltingT", "CU"),
    "Delta_MeltingT_X_Cr": ("MeltingT", "CR"),
    "Delta_Density_X_Cu": ("Density", "CU"),
    "Delta_FirstIonizationEnergy_X_Cu": ("FirstIonizationEnergy", "CU"),
    "Delta_FirstIonizationEnergy_X_Cr": ("FirstIonizationEnergy", "CR"),
}


def descriptor_lookup():
    """Descriptor values for CU/CR/NB + the six existing X from the FROZEN STEP5A
    table, the five STEP5F additions from the frozen 14X table, and NI from the
    stage-91 extraction. One source per property, identical to STEP5A policy."""
    a = pd.read_parquet(C.A_DESC_TABLE).set_index("element")
    f = pd.read_parquet(C.F5F_DESC_TABLE).set_index("element")
    ni = pd.read_parquet(os.path.join(C.D_ACCESS,
                                      "STEP5H_A_NI_DESCRIPTOR_ROW.parquet")).set_index("element")
    # the frozen 9 must be identical in both historical tables (they were
    # reproduced exactly in STEP5F; re-verify cheaply here)
    shared = [c for c in a.columns if c in f.columns]
    if not a.loc[[e for e in a.index if e in f.index], shared].equals(
            f.loc[[e for e in a.index if e in f.index], shared]):
        raise SystemExit("FATAL: STEP5A and STEP5F descriptor tables disagree on a "
                         "shared element/property - source drift; STOP.")
    need = ["AtomicNumber", "AtomicWeight", "Electronegativity", "VEC", "MeltingT",
            "FirstIonizationEnergy", "AtomicVolume_GS", "MetallicRadius", "Density"]
    table = {}
    for el in ["CU", "CR", "NB"] + C.ORIGINAL_SIX:
        table[el] = {p: float(a.loc[el, p]) for p in need}
    for el in ["BE", "SN", "MN", "LA", "P"]:
        table[el] = {p: float(f.loc[el, p]) for p in need}
    table["NI"] = {p: float(ni.loc["NI", p]) for p in need}
    return table


def build_composition_features(comp, x_code, desc):
    """The frozen STEP5A composition-level feature math for one X identity.

    comp: DataFrame with Cu_wt_pct / Cr_wt_pct / Nb_wt_pct / X_wt_pct columns.
    Returns the composition-level subset of the 43 features (Temperature_C is
    state-level and added by the caller).
    """
    from pymatgen.core.composition import Composition
    from matminer.featurizers.composition import ElementProperty

    P = {e: desc[e] for e in ("CU", "CR", "NB", x_code)}
    w = comp[["Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct"]].astype(float)
    xmat = np.column_stack([w["Cu_wt_pct"] / P["CU"]["AtomicWeight"],
                            w["Cr_wt_pct"] / P["CR"]["AtomicWeight"],
                            w["Nb_wt_pct"] / P["NB"]["AtomicWeight"],
                            w["X_wt_pct"] / P[x_code]["AtomicWeight"]])
    xat = xmat / xmat.sum(axis=1)[:, None]
    if float(np.abs(xat.sum(axis=1) - 1.0).max()) > 1e-12:
        raise SystemExit("FATAL: atomic closure failed")
    x_Cu, x_Cr, x_Nb, x_X = xat.T

    def prop_matrix(prop):
        return np.column_stack([np.full(len(comp), P["CU"][prop]),
                                np.full(len(comp), P["CR"][prop]),
                                np.full(len(comp), P["NB"][prop]),
                                np.full(len(comp), P[x_code][prop])])

    out = pd.DataFrame(index=comp.index)
    # F1
    out["Cr_wt_pct"] = w["Cr_wt_pct"].to_numpy()
    out["Nb_wt_pct"] = w["Nb_wt_pct"].to_numpy()
    out["X_wt_pct"] = w["X_wt_pct"].to_numpy()
    # the historical tables STORE R_CrNb_atomic_actual rounded to 6 decimals
    # (frozen FULL_V1 convention); use the stored column when present so the
    # consistency test compares like with like, else recompute + round identically
    if "R_CrNb_atomic_actual" in comp.columns:
        out["R_CrNb_atomic_actual"] = comp["R_CrNb_atomic_actual"].astype(float).to_numpy()
    else:
        out["R_CrNb_atomic_actual"] = np.round(
            ((w["Cr_wt_pct"] / C.M_CR) / (w["Nb_wt_pct"] / C.M_NB)).to_numpy(), 6)
    out["has_X"] = 1
    # F2 (frozen formulas)
    out["Zbar"] = (prop_matrix("AtomicNumber") * xat).sum(axis=1)
    out["density_bar"] = (prop_matrix("Density") * xat).sum(axis=1)
    out["AtomicVol_bar"] = (prop_matrix("AtomicVolume_GS") * xat).sum(axis=1)
    r = prop_matrix("MetallicRadius")
    r_bar = (r * xat).sum(axis=1)
    out["r_bar"] = r_bar
    ratio = np.where(xat > 0, 1.0 - r / r_bar[:, None], 0.0)
    out["delta_r"] = 100.0 * np.sqrt((xat * ratio ** 2).sum(axis=1))
    chi = prop_matrix("Electronegativity")
    chi_bar = (chi * xat).sum(axis=1)
    out["delta_chi"] = np.sqrt((xat * (chi - chi_bar[:, None]) ** 2).sum(axis=1))
    # F3 (frozen formulas; has_X = 1 for every row here)
    for name, (prop, base) in F3_NEEDED.items():
        v = P[x_code][prop]
        out[name] = v if base is None else v - P[base][prop]
    # F4: the frozen matminer preset, verbatim implementation
    ep = ElementProperty.from_preset("magpie", impute_nan=False)
    sym = C.PMG_SYMBOL[x_code]
    comps = [Composition({k: v for k, v in
                          {"Cu": xc, "Cr": xr, "Nb": xn, sym: xx}.items() if v > 0})
             for xc, xr, xn, xx in zip(x_Cu, x_Cr, x_Nb, x_X)]
    # verbatim STEP5A implementation: per-composition ep.featurize, then rename
    rows = [ep.featurize(c) for c in comps]
    cols = [c.replace("MagpieData ", "magpie_").replace(" ", "_")
            for c in ep.feature_labels()]
    feats = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c in out.columns:
            continue
        out[c] = feats[c].to_numpy()
    return out


def expand_states(comp):
    """Spec 8: each composition x the 11 frozen temperatures; state_id format
    NI_BLIND_000001__T0300C; fewshot_role / fewshot_rank preserved."""
    rows = []
    for t in C.TEMPERATURES_C:
        s = comp[["composition_id", "fewshot_role", "fewshot_rank"]].copy()
        s["Temperature_C"] = float(t)
        s["state_id"] = s.composition_id + "__T%04dC" % int(t)
        rows.append(s)
    st = pd.concat(rows, ignore_index=True)
    st = st.sort_values(["composition_id", "Temperature_C"]).reset_index(drop=True)
    return st[["state_id", "composition_id", "Temperature_C", "fewshot_role",
               "fewshot_rank"]]


def consistency_test(desc, feats43):
    """Spec 12: one frozen composition per existing element, at 300/700/1000 C,
    recomputed through THIS featurization path and compared against the
    historical frozen candidate feature table. Mismatch -> STOP before Ni."""
    hist = pd.read_parquet(C.A_CANDIDATE_FEATURES)
    comp_src = pd.read_parquet(
        os.path.join(C.MLROOT, "03_full_CALPHAD", "FULL_V1", "01_compositions",
                     "FULL_V1_COMPOSITIONS.parquet"),
        columns=["composition_id", "family", "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct",
                 "X_wt_pct", "R_CrNb_atomic_actual"])
    hist_ids = set(hist.state_id)
    rows, max_dev, bad = [], 0.0, []
    for el in C.ORIGINAL_SIX:
        sub = comp_src[comp_src.family == el].sort_values("composition_id")
        # first composition whose three reference states exist in the historical
        # development table ( Domain-A DEV rows only were featurized in STEP5A )
        cid = None
        for c in sub.composition_id:
            if all("%s__T%04dC" % (c, t) in hist_ids for t in C.CONSISTENCY_TEMPS):
                cid = c
                break
        if cid is None:
            bad.append("%s: no composition with all 3 reference states" % el)
            continue
        one = sub[sub.composition_id == cid].reset_index(drop=True)
        lvl = build_composition_features(one, el, desc)
        for t in C.CONSISTENCY_TEMPS:
            sid = "%s__T%04dC" % (cid, int(t))
            h = hist[hist.state_id == sid]
            if not len(h):
                bad.append("%s: historical state missing" % sid)
                continue
            h = h.iloc[0]
            dev = {"state_id": sid, "element": el}
            for f in feats43:
                if f == "Temperature_C":
                    v_new, v_hist = float(t), float(h[f])
                elif f in lvl.columns:
                    v_new, v_hist = float(lvl[f].iloc[0]), float(h[f])
                else:
                    bad.append("%s: feature %s not produced" % (sid, f))
                    continue
                d = abs(v_new - v_hist)
                rel = d / max(abs(v_hist), 1e-300) if v_hist != 0 else d
                dev[f] = d
                if max(d, min(rel, d)) > C.CONSISTENCY_TOL and d > C.CONSISTENCY_TOL:
                    bad.append("%s::%s dev %.3e (hist %.12g new %.12g)"
                               % (sid, f, d, v_hist, v_new))
                max_dev = max(max_dev, d)
            rows.append(dev)
    detail = pd.DataFrame(rows)
    return detail, max_dev, bad


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5H-A  Stage 2: state expansion, 43-feature construction, consistency test")
    print("=" * 78)
    C.assert_no_tc_python()

    feats43 = C.feature_schema()
    C.verify_hash(C.G_SCHEMA_CSV, C.FROZEN_HASHES["feature_schema_csv"], "feature schema")
    desc = descriptor_lookup()

    # ---- spec 12 FIRST: prove the path on existing elements before Ni ------- #
    detail, max_dev, bad = consistency_test(desc, feats43)
    detail.to_csv(os.path.join(C.D_QA, "STEP5H_A_SOURCE_CONSISTENCY_TEST.csv"),
                  index=False, encoding="utf-8")
    print("  source-consistency test: 6 elements x %d temperatures = %d states | "
          "max |delta| = %.3e | bad: %s"
          % (len(C.CONSISTENCY_TEMPS), len(detail), max_dev, bad or "none"))
    if bad or max_dev > C.CONSISTENCY_TOL:
        print("FATAL: this featurization path does NOT reproduce the historical "
              "frozen features. STOP BEFORE Ni prediction (spec 12).")
        for b in bad[:20]:
            print("   ", b)
        return 2

    # ---- Ni compositions -> composition-level features ---------------------- #
    comp = pd.read_csv(C.G_NI_COMP_CSV)
    C.verify_hash(C.G_NI_COMP_CSV, C.FROZEN_HASHES["ni_compositions_csv"],
                  "Ni compositions")
    lvl = build_composition_features(comp, "NI", desc)
    lvl.insert(0, "composition_id", comp.composition_id.to_numpy())

    # ---- spec 8: state expansion -------------------------------------------- #
    states = expand_states(comp)
    assert len(states) == C.N_STATES, len(states)
    df = states.merge(lvl, on="composition_id", how="left", validate="m:1")
    assert len(df) == C.N_STATES

    # ---- spec 9/11: exact schema order, validation -------------------------- #
    missing = [f for f in feats43 if f not in df.columns]
    if missing:
        print("FATAL: features missing from the construction: %s" % missing)
        return 2
    # Temperature_C is part of the 43-feature schema; keep it only once
    meta = ["state_id", "composition_id", "fewshot_role", "fewshot_rank"]
    df = df[meta + feats43]
    df = df.sort_values(["composition_id", "Temperature_C"]).reset_index(drop=True)
    X = df[feats43]
    checks = {
        "shape_22000x43": X.shape == (C.N_STATES, C.N_FEATURES),
        "all_numeric": all(pd.api.types.is_numeric_dtype(X[c]) for c in feats43),
        "no_nan": bool(X.notna().all().all()),
        "no_inf": bool(np.isfinite(X.to_numpy(float)).all()),
        "column_order_exact": list(X.columns) == feats43,
        "state_id_format": bool(df.state_id.str.fullmatch(
            r"NI_BLIND_\d{6}__T\d{4}C").all()),
        "state_id_unique": df.state_id.is_unique,
        "fewshot_roles_preserved":
            int((df.fewshot_role == "FEWSHOT_CALIBRATION").sum()) == 600 * 11
            and int((df.fewshot_role == "RECOVERY_EVALUATION").sum()) == 1400 * 11,
        "schema_hash_unchanged": C.sha256_file(C.G_SCHEMA_CSV)
                                 == C.FROZEN_HASHES["feature_schema_csv"],
    }
    guard_hits = C.target_column_guard(df, "NI_FEATURES")
    checks["target_column_guard_clean"] = not guard_hits

    out = os.path.join(C.D_FEAT, "STEP5H_A_NI_FEATURES.parquet")
    df.to_parquet(out, index=False)
    for k, v in checks.items():
        print("  [%-4s] %s" % ("PASS" if v else "FAIL", k))
    if guard_hits:
        print("  target-column guard hits: %s" % guard_hits)
    C.write_json(os.path.join(C.D_QA, "STEP5H_A_FEATURE_QA.json"), {
        "generated_utc": C.utcnow(),
        "n_states": int(len(df)), "n_features": len(feats43),
        "feature_order": feats43,
        "checks": checks,
        "consistency_test": {"states": int(len(detail)), "max_abs_delta": max_dev,
                             "tolerance": C.CONSISTENCY_TOL, "bad": bad,
                             "verdict": "PASS" if not bad else "FAIL"},
        "target_column_guard": guard_hits,
        "features_sha256": C.sha256_file(out),
        "status": "PASS" if all(checks.values()) and not bad else "FAIL"})
    ok = all(checks.values()) and not bad
    print("STAGE 2: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
