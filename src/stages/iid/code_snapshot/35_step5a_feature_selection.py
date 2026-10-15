# -*- coding: utf-8 -*-
"""
35_step5a_feature_selection.py
STEP5A Stage I: TRAIN-ONLY feature filtering (sec.41, 45-49).

Everything here uses DEV_TRAIN rows ONLY (no validation, no test, no Domain B,
no ANCHOR):
  1. NaN / inf census
  2. near-zero-variance removal            -> NEAR_ZERO_VARIANCE_REMOVAL.csv
  3. exact algebraic redundancy (|r|>=1-1e-10) -> EXACT_REDUNDANCY_LOG.csv
  4. Pearson |r| > 0.90 with fixed keep priority -> PCC_REMOVAL_LOG.csv
  5. SELECTED_FEATURES_F1 / F1_F2 / F1_F2_F3 / F1_F2_F3_F4 .txt
     PRIMARY_FEATURE_LIST.txt
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

STEP5A = r"E:\Cu_CALPHAD_ML\04_ML\STEP5A_JMST_ML_FOUNDATION_V2"
FEAT = os.path.join(STEP5A, "02_features")
QA = os.path.join(STEP5A, "07_qa")

META_COLS = ["state_id", "composition_id", "family", "X_element", "sampling_domain", "split_role"]
HELPER = ["Temperature_K"]

PCC_THRESHOLD = 0.90
EXACT_TOL = 1e-10

# keep priority (sec.47): raw F1 core first, then well-defined physical
# descriptors, then statistical derivatives (F4 last).
PRIORITY = (
    ["Temperature_C", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct", "S_CrNb_wt_pct",
     "R_CrNb_atomic_actual", "Cu_wt_pct", "has_X",
     "T_over_Tm_mix", "Delta_Smix", "delta_r", "delta_chi", "r_bar", "chi_bar",
     "VEC_mix", "Tm_mix", "Zbar", "M_bar", "CovalentR_bar", "density_bar",
     "Ecoh_bar", "IE1_bar", "EA_bar", "Mendeleev_bar", "AtomicVol_bar"]
    + [f"X_{p}" for p in ["AtomicNumber", "AtomicWeight", "MetallicRadius", "Electronegativity",
                          "VEC", "MeltingT", "Density", "FirstIonizationEnergy", "CohesiveEnergy"]]
    + [f"Delta_{p}_X_{b}" for p in ["AtomicNumber", "AtomicWeight", "MetallicRadius",
                                    "Electronegativity", "VEC", "MeltingT", "Density",
                                    "FirstIonizationEnergy", "CohesiveEnergy"]
       for b in ("Cu", "Cr", "Nb")]
)
STAT_ORDER = {"mean": 0, "avg_dev": 1, "minimum": 2, "maximum": 3, "range": 4, "mode": 5}


def f4_key(name):
    # magpie_<stat>_<Property>
    parts = name.split("_", 2)
    stat = parts[1] if len(parts) > 2 else "mean"
    prop = parts[2] if len(parts) > 2 else name
    return (STAT_ORDER.get(stat, 9), prop)


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cand = pd.read_parquet(os.path.join(FEAT, "FULL_V1_CANDIDATE_FEATURES.parquet"))
    feat_cols = [c for c in cand.columns if c not in META_COLS + HELPER]
    train = cand[cand["split_role"] == "DEV_TRAIN"]
    print(f"DEV_TRAIN rows: {len(train)}; candidate features: {len(feat_cols)}")

    X = train[feat_cols].to_numpy(dtype=np.float64)

    # ---------- 1. NaN / inf census (train) ----------
    nan_cols = {feat_cols[i]: int(np.isnan(X[:, i]).sum()) for i in range(len(feat_cols))
                if np.isnan(X[:, i]).any()}
    inf_cols = {feat_cols[i]: int(np.isinf(X[:, i]).sum()) for i in range(len(feat_cols))
                if np.isinf(X[:, i]).any()}
    print(f"NaN columns: {len(nan_cols)}; inf columns: {len(inf_cols)}")

    # ---------- 2. near-zero variance ----------
    nzv_rows = []
    keep1 = []
    for i, c in enumerate(feat_cols):
        v = X[:, i]
        v = v[np.isfinite(v)]
        std = v.std()
        uniq, counts = np.unique(v, return_counts=True)
        minority_frac = (counts.min() / counts.sum()) if len(uniq) > 1 else 0.0
        is_const = len(uniq) <= 1
        is_nzv = (not is_const) and len(uniq) == 2 and minority_frac < 1e-4
        if is_const or is_nzv:
            nzv_rows.append({"feature": c, "reason": "constant" if is_const else "near-zero-variance",
                             "n_unique": len(uniq), "std": float(std),
                             "minority_fraction": float(minority_frac)})
        else:
            keep1.append(c)
    pd.DataFrame(nzv_rows).to_csv(os.path.join(FEAT, "NEAR_ZERO_VARIANCE_REMOVAL.csv"), index=False)
    print(f"NZV removed: {len(nzv_rows)}; remaining: {len(keep1)}")

    # ---------- correlation matrix on remaining ----------
    cols = keep1
    Xk = train[cols].to_numpy(dtype=np.float64)
    Xk = Xk - Xk.mean(axis=0)
    sd = Xk.std(axis=0)
    C = (Xk.T @ Xk) / len(Xk)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = C / np.outer(sd, sd)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)

    # ---------- priority map ----------
    prio = {}
    for rank, c in enumerate(PRIORITY):
        if c in cols:
            prio[c] = rank
    f4_sorted = sorted([c for c in cols if c not in prio], key=f4_key)
    for rank, c in enumerate(f4_sorted):
        prio[c] = len(PRIORITY) + rank

    # ---------- 3+4. greedy removal (deterministic) ----------
    active = set(cols)
    exact_rows = []
    pcc_rows = []
    # precompute upper-triangle pairs sorted by |r| desc, deterministically
    iu = np.triu_indices(len(cols), k=1)
    order = np.argsort(-np.abs(R[iu]), kind="stable")
    for idx in order:
        i, j = iu[0][idx], iu[1][idx]
        r = R[i, j]
        a, b = cols[i], cols[j]
        if abs(r) <= PCC_THRESHOLD:
            break
        if a not in active or b not in active:
            continue
        kept, dropped = (a, b) if prio[a] < prio[b] else (b, a)
        active.discard(dropped)
        row = {"feature_A": a, "feature_B": b, "Pearson_r": float(r),
               "kept": kept, "dropped": dropped,
               "reason": f"|r|={abs(r):.6f} > {PCC_THRESHOLD}; keep priority {prio[kept]} < {prio[dropped]}"}
        if abs(abs(r) - 1.0) <= EXACT_TOL:
            row["reason"] = f"exact algebraic redundancy (|r| = {abs(r):.12f}); kept higher-priority {kept}"
            exact_rows.append(row)
        else:
            pcc_rows.append(row)

    pd.DataFrame(exact_rows).to_csv(os.path.join(FEAT, "EXACT_REDUNDANCY_LOG.csv"), index=False)
    pd.DataFrame(pcc_rows).to_csv(os.path.join(FEAT, "PCC_REMOVAL_LOG.csv"), index=False)
    selected = [c for c in cols if c in active]
    selected.sort(key=lambda c: (prio[c], c))
    print(f"exact redundancy removed: {len(exact_rows)}; PCC removed: {len(pcc_rows)}; "
          f"selected: {len(selected)}")

    # ---------- 5. feature-set files ----------
    blocks = json.load(open(os.path.join(FEAT, "FEATURE_BUILD_SUMMARY.json"), encoding="utf-8"))
    f1 = [c for c in selected if c in blocks["F1"]]
    f2 = [c for c in selected if c in blocks["F2"]]
    f3 = [c for c in selected if c in set(blocks["F3"])]
    f4 = [c for c in selected if c.startswith("magpie_")]
    assert len(f1) + len(f2) + len(f3) + len(f4) == len(selected), (
        len(f1), len(f2), len(f3), len(f4), len(selected))

    def write_list(name, cols_):
        with open(os.path.join(FEAT, name), "w", encoding="utf-8") as f:
            f.write("\n".join(cols_) + "\n")

    write_list("SELECTED_FEATURES_F1.txt", f1)
    write_list("SELECTED_FEATURES_F1_F2.txt", f1 + f2)
    write_list("SELECTED_FEATURES_F1_F2_F3.txt", f1 + f2 + f3)
    write_list("SELECTED_FEATURES_F1_F2_F3_F4.txt", f1 + f2 + f3 + f4)
    write_list("PRIMARY_FEATURE_LIST.txt", selected)

    summary = {
        "generated_utc": ts, "dev_train_rows": len(train),
        "candidate_features": len(feat_cols), "nan_columns": nan_cols, "inf_columns": inf_cols,
        "nzv_removed": len(nzv_rows), "exact_redundancy_removed": len(exact_rows),
        "pcc_removed": len(pcc_rows), "pcc_threshold": PCC_THRESHOLD,
        "selected_total": len(selected),
        "selected_F1": f1, "selected_F2": f2, "selected_F3": f3, "selected_F4_n": len(f4),
        "selected_F4": f4,
        "train_only": True,
    }
    with open(os.path.join(FEAT, "FEATURE_SELECTION_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(QA, "FEATURE_ENGINEERING_QA.md"), "w", encoding="utf-8") as f:
        f.write("# FEATURE ENGINEERING QA (sec.96)\n\n")
        f.write(f"generated_utc: {ts}\n\n")
        f.write(f"- candidate feature count: {len(feat_cols)} "
                f"(F1 {blocks['n_F1']}, F2 {blocks['n_F2']}, F3 {blocks['n_F3']}, F4 {blocks['n_F4']})\n")
        f.write(f"- DEV_TRAIN rows used for ALL filtering: {len(train)} (train-only hard rule, sec.41)\n")
        f.write(f"- missingness: NaN columns {len(nan_cols)} {nan_cols if nan_cols else ''}; "
                f"inf columns {len(inf_cols)} {inf_cols if inf_cols else ''}\n")
        f.write(f"- NZV removed: {len(nzv_rows)} (see NEAR_ZERO_VARIANCE_REMOVAL.csv)\n")
        f.write(f"- exact algebraic redundancy removed: {len(exact_rows)} (EXACT_REDUNDANCY_LOG.csv)\n")
        f.write(f"- PCC |r|>0.90 removed: {len(pcc_rows)} (PCC_REMOVAL_LOG.csv)\n")
        f.write(f"- final selected feature count: {len(selected)} "
                f"(F1 {len(f1)}, F2 {len(f2)}, F3 {len(f3)}, F4 {len(f4)})\n")
        f.write("- units: see FEATURE_DICTIONARY.csv\n")
        f.write("- descriptor source: Stage-F table (matminer MagpieData / mendeleev / CohesiveEnergyData); "
                "Magpie ElementProperty preset for F4\n")
        f.write("- Ni scan: 02_features + feature-selection outputs contain no Ni token "
                "(verified in Stage-H audit and final Stage-V scan)\n\n")
        f.write("## Selected features\n\n")
        f.write("### F1 (raw composition + temperature)\n\n" + ", ".join(f1) + "\n\n")
        f.write("### F2 (general composition physics)\n\n" + ", ".join(f2) + "\n\n")
        f.write("### F3 (X-specific descriptors)\n\n" + ", ".join(f3) + "\n\n")
        f.write(f"### F4 (Magpie, {len(f4)} selected)\n\n" + ", ".join(f4) + "\n")

    print("STAGE I: PASS")


if __name__ == "__main__":
    main()
