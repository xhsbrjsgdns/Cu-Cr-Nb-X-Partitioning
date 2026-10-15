# -*- coding: utf-8 -*-
"""
34_step5a_leakage_audit.py
STEP5A Stage H: leakage audit #1 (pre-modeling gate). If FAIL -> STOP.

Checks (name-level + structural + value-level where cheap):
  H1  candidate feature parquet contains only dev rows (Domain A)
  H2  no forbidden column (sec.12) inside the feature matrix
  H3  no target / AUX / derived-descriptor column inside the feature matrix
  H4  family / X_element / sampling_domain / split_role appear ONLY as metadata
  H5  no X one-hot column
  H6  no feature column is an exact copy of a target column (value check on DEV_TRAIN)
  H7  split disjointness re-verified from the written master file
  H8  FEATURE_DICTIONARY covers every feature column
  H9  Ni token scan over 02_features text outputs (token-bounded)
  H10 NaN / inf scan of the feature matrix
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

FULL_V1 = r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1"
STEP5A = r"E:\Cu_CALPHAD_ML\04_ML\STEP5A_JMST_ML_FOUNDATION_V2"
FEAT = os.path.join(STEP5A, "02_features")
SPLITS = os.path.join(STEP5A, "03_splits")
QA = os.path.join(STEP5A, "07_qa")

META_COLS = ["state_id", "composition_id", "family", "X_element", "sampling_domain", "split_role"]
HELPER_COLS = ["Temperature_K"]
FORBIDDEN = {"state_id", "composition_id", "family", "X_element", "sampling_domain",
             "state_converged", "solver_level_used", "bridge_used", "attempt_count",
             "phase_count", "has_multiple_FCC_instances", "has_multiple_BCC_instances",
             "has_liquid", "BCC_split_closure_C", "FCC_split_closure_C",
             "BCC_split_closure_censored", "LIQUID_onset_C", "C15_volfrac_450",
             "C15_volfrac_600", "C15_volfrac_max", "Delta_C15_600_450", "X_Cr_FCC_450",
             "X_Cr_FCC_950", "Delta_XCr_FCC_950_450", "dominant_X_partition_family",
             "dominant_X_solid_partition_family"}
TARGET_PATTERNS = [r"_presence$", r"_volfrac", r"_molefrac", r"^(CR|NB|X)_P_", r"^X_Psolid_",
                   r"_AUX$", r"_total_volume_fraction$", r"_total_mole_fraction$", r"_n_instances$"]

failures = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(name)
    print(f"[{status}] {name} {detail}")
    return {"check": name, "status": status, "detail": str(detail)[:500]}


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []
    cand = pd.read_parquet(os.path.join(FEAT, "FULL_V1_CANDIDATE_FEATURES.parquet"))
    feat_cols = [c for c in cand.columns if c not in META_COLS + HELPER_COLS]

    results.append(check("H1 only development Domain-A rows",
                         cand["split_role"].isin(["DEV_TRAIN", "DEV_VALIDATION", "IID_TEST"]).all()
                         and (cand["sampling_domain"] == "A").all(), f"{len(cand)} rows"))
    results.append(check("H2 no forbidden column in feature matrix",
                         not (set(feat_cols) & FORBIDDEN), str(set(feat_cols) & FORBIDDEN)))
    tgt_hits = [c for c in feat_cols if any(re.search(p, c) for p in TARGET_PATTERNS)]
    results.append(check("H3 no target/AUX/derived column in feature matrix", not tgt_hits, str(tgt_hits)))
    results.append(check("H4 family/X_element/sampling_domain/split_role are metadata only",
                         all(c in META_COLS for c in ["family", "X_element", "sampling_domain", "split_role"]
                             if c in cand.columns)
                         and not any(c in feat_cols for c in ["family", "X_element", "sampling_domain", "split_role"])))
    onehot = [c for c in feat_cols if re.search(r"(^is_|_onehot|^ZR$|^TI$|^CO$|^SI$|^AG$|^MO$)", c)]
    results.append(check("H5 no X one-hot column", not onehot, str(onehot)))

    # H6 value check: no feature exactly equals a target on DEV_TRAIN rows
    train = cand[cand["split_role"] == "DEV_TRAIN"]
    mlr = pd.read_parquet(
        os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet"),
        columns=["state_id", "Temperature_C",
                 "FCC_volfrac", "BCC_volfrac", "C15_volfrac", "OTHER_SOLID_volfrac", "LIQUID_volfrac",
                 "FCC_presence", "BCC_presence", "C15_presence", "OTHER_SOLID_presence", "LIQUID_presence",
                 "CR_P_FCC", "CR_P_BCC", "CR_P_C15", "CR_P_OTHER_SOLID", "CR_P_LIQUID",
                 "NB_P_FCC", "NB_P_BCC", "NB_P_C15", "NB_P_OTHER_SOLID", "NB_P_LIQUID",
                 "X_P_FCC", "X_P_BCC", "X_P_C15", "X_P_OTHER_SOLID", "X_P_LIQUID"])
    tr = train.merge(mlr, on="state_id", suffixes=("", "_t"))
    assert len(tr) == len(train)
    exact_copy = []
    constant_match = []
    tcols = [x for x in mlr.columns if x != "state_id" and x != "Temperature_C"]
    for c in feat_cols:
        if c == "Temperature_C":
            continue
        v = tr[c].values.astype(float)
        feat_const = np.nanstd(v) == 0.0
        for t in tcols:
            tv = tr[t].values.astype(float)
            if np.isnan(tv).any():
                continue  # NaN targets (e.g. X_P_* on BASELINE) cannot be copied by a finite feature
            if np.max(np.abs(v - tv)) < 1e-15:
                if feat_const:
                    # constant feature vs constant target on DEV_TRAIN: accidental degeneracy,
                    # NOT information transfer. The feature carries zero variance and is removed
                    # by the Stage-I near-zero-variance filter before any model sees it.
                    constant_match.append((c, t))
                else:
                    exact_copy.append((c, t))
    results.append(check("H6 no non-constant feature is an exact copy of a target (DEV_TRAIN values)",
                         not exact_copy,
                         f"exact copies: {exact_copy[:5]}; constant-feature/constant-target accidental "
                         f"matches (deferred to Stage-I NZV removal): {sorted(set(c for c, _ in constant_match))}"))

    master = pd.read_csv(os.path.join(SPLITS, "COMPOSITION_SPLIT_MASTER_V1.csv"))
    sets = {r: set(master.loc[master["split_role"] == r, "composition_id"])
            for r in ["DEV_TRAIN", "DEV_VALIDATION", "IID_TEST", "SEALED_DOMAIN_B", "SEALED_ANCHOR"]}
    disjoint = (len(sets["DEV_TRAIN"] & sets["DEV_VALIDATION"]) == 0
                and len(sets["DEV_TRAIN"] & sets["IID_TEST"]) == 0
                and len(sets["DEV_VALIDATION"] & sets["IID_TEST"]) == 0
                and len((sets["DEV_TRAIN"] | sets["DEV_VALIDATION"] | sets["IID_TEST"]) & sets["SEALED_DOMAIN_B"]) == 0
                and len((sets["DEV_TRAIN"] | sets["DEV_VALIDATION"] | sets["IID_TEST"]) & sets["SEALED_ANCHOR"]) == 0)
    results.append(check("H7 split disjointness re-verified", disjoint))
    # candidate feature rows never contain sealed compositions
    sealed_ids = sets["SEALED_DOMAIN_B"] | sets["SEALED_ANCHOR"]
    results.append(check("H7b no sealed composition in candidate features",
                         len(set(cand["composition_id"]) & sealed_ids) == 0))

    dic = pd.read_csv(os.path.join(FEAT, "FEATURE_DICTIONARY.csv"))
    missing_dic = set(feat_cols) - set(dic["feature_name"])
    results.append(check("H8 FEATURE_DICTIONARY covers all feature columns", not missing_dic,
                         str(sorted(missing_dic)[:10])))

    ni_hits = []
    for folder in [FEAT]:
        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            if os.path.isfile(p) and fn.endswith((".md", ".csv", ".json", ".txt")):
                txt = open(p, encoding="utf-8", errors="ignore").read()
                for m in re.finditer(r"(?<![A-Za-z])(?:NI|Ni)(?![A-Za-z])", txt):
                    ctx = txt[max(0, m.start() - 20):m.end() + 20]
                    if not re.search(r"NIOBIUM|DOMINANT", ctx, re.IGNORECASE):
                        ni_hits.append((fn, ctx))
    results.append(check("H9 Ni token scan 02_features text outputs == 0 hits", not ni_hits, str(ni_hits[:5])))

    num = cand[feat_cols]
    results.append(check("H10 no NaN in feature matrix", not num.isna().any().any(),
                         str(num.columns[num.isna().any()].tolist()[:10])))
    results.append(check("H10b no inf in feature matrix",
                         not np.isinf(num.values.astype(float)).any()))

    overall = "PASS" if not failures else "FAIL"
    audit = {"stage": "STEP5A Stage H leakage audit #1", "generated_utc": ts,
             "status": overall, "n_feature_columns": len(feat_cols),
             "feature_columns": feat_cols, "checks": results, "failures": failures}
    with open(os.path.join(QA, "LEAKAGE_AUDIT_STAGE_H.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    with open(os.path.join(QA, "LEAKAGE_AUDIT.md"), "w", encoding="utf-8") as f:
        f.write("# LEAKAGE AUDIT (Stage H, pre-modeling gate)\n\n")
        f.write(f"generated_utc: {ts}\n\nSTATUS: **{overall}**\n\n")
        f.write("| check | status | detail |\n|---|---|---|\n")
        for r in results:
            f.write(f"| {r['check']} | {r['status']} | {r['detail'][:160]} |\n")
        f.write("\nThis is leakage audit #1. The final audit (sec.95, 15 items) runs after "
                "the one-time IID test as LEAKAGE_AUDIT_FINAL.md.\n")
    print(f"\nSTAGE H LEAKAGE AUDIT #1: {overall}")
    if overall == "FAIL":
        print("STOP — do not proceed to modeling.")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
