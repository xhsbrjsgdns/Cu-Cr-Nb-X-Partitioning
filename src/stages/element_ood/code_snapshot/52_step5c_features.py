# -*- coding: utf-8 -*-
"""
52_step5c_features.py
STEP5C Stage F: per-fold, FOLD_TRAIN-ONLY feature pipelines (spec 21-24).

Every fold restarts from the frozen 193-feature candidate schema. The STEP5A
43-feature list is NOT used as the LOEO primary list, because that selection saw all
six development X elements and therefore saw every held-out element indirectly (spec 20).

Only FOLD_TRAIN rows are touched: no FOLD_VALIDATION row, no HELDOUT_ELEMENT_TEST row,
no Domain B row, no ANCHOR row.

Per fold outputs (spec 24):
  03_features/FOLD_<X>_SELECTED_FEATURES.txt
  03_features/FOLD_<X>_NZV_REMOVAL.csv
  03_features/FOLD_<X>_EXACT_REDUNDANCY.csv
  03_features/FOLD_<X>_PCC_REMOVAL.csv
  03_features/FOLD_<X>_FEATURE_PROVENANCE.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5c_common as C


def f4_key(name):
    parts = name.split("_", 2)
    stat = parts[1] if len(parts) > 2 else "mean"
    prop = parts[2] if len(parts) > 2 else name
    return (C.STAT_ORDER.get(stat, 9), prop)


def priority_map(cols):
    """STEP5A keep priority, inherited verbatim (spec 23). Fixed - never fold-dependent."""
    prio = {}
    for rank, c in enumerate(C.KEEP_PRIORITY):
        if c in cols:
            prio[c] = rank
    rest = sorted([c for c in cols if c not in prio], key=f4_key)
    for rank, c in enumerate(rest):
        prio[c] = len(C.KEEP_PRIORITY) + rank
    return prio


def select_features_for_fold(train_df, feat_cols, heldout):
    """Run the four filtering steps on FOLD_TRAIN rows only."""
    X = train_df[feat_cols].to_numpy(dtype=np.float64)
    nan_cols = {feat_cols[i]: int(np.isnan(X[:, i]).sum())
                for i in range(len(feat_cols)) if np.isnan(X[:, i]).any()}
    inf_cols = {feat_cols[i]: int(np.isinf(X[:, i]).sum())
                for i in range(len(feat_cols)) if np.isinf(X[:, i]).any()}

    # 1. near-zero variance
    nzv_rows, keep1 = [], []
    for i, c in enumerate(feat_cols):
        v = X[:, i]
        v = v[np.isfinite(v)]
        uniq, counts = np.unique(v, return_counts=True)
        minority = (counts.min() / counts.sum()) if len(uniq) > 1 else 0.0
        is_const = len(uniq) <= 1
        is_nzv = (not is_const) and len(uniq) == 2 and minority < C.NZV_MINORITY_FRAC
        if is_const or is_nzv:
            nzv_rows.append({"feature": c,
                             "reason": "constant" if is_const else "near-zero-variance",
                             "n_unique": int(len(uniq)), "std": float(v.std()),
                             "minority_fraction": float(minority),
                             "heldout_element": heldout})
        else:
            keep1.append(c)

    # 2/3. correlation matrix on FOLD_TRAIN only
    Xk = train_df[keep1].to_numpy(dtype=np.float64)
    Xk = Xk - Xk.mean(axis=0)
    sd = Xk.std(axis=0)
    Cov = (Xk.T @ Xk) / len(Xk)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = Cov / np.outer(sd, sd)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)

    prio = priority_map(keep1)
    active = set(keep1)
    exact_rows, pcc_rows = [], []
    iu = np.triu_indices(len(keep1), k=1)
    order = np.argsort(-np.abs(R[iu]), kind="stable")
    for idx in order:
        i, j = iu[0][idx], iu[1][idx]
        r = float(R[i, j])
        if abs(r) <= C.PCC_THRESHOLD:
            break
        a, b = keep1[i], keep1[j]
        if a not in active or b not in active:
            continue
        kept, dropped = (a, b) if prio[a] < prio[b] else (b, a)
        active.discard(dropped)
        row = {"feature_A": a, "feature_B": b, "Pearson_r": r, "kept": kept,
               "dropped": dropped, "heldout_element": heldout,
               "reason": "|r|=%.6f > %.2f; keep priority %d < %d"
                         % (abs(r), C.PCC_THRESHOLD, prio[kept], prio[dropped])}
        if abs(abs(r) - 1.0) <= C.EXACT_TOL:
            row["reason"] = ("exact algebraic redundancy (|r| = %.12f); kept higher-priority %s"
                             % (abs(r), kept))
            exact_rows.append(row)
        else:
            pcc_rows.append(row)

    selected = [c for c in keep1 if c in active]
    selected.sort(key=lambda c: (prio[c], c))
    return selected, nzv_rows, exact_rows, pcc_rows, nan_cols, inf_cols


def main():
    print("=" * 78)
    print("STEP 5C   Stage F: per-fold FOLD_TRAIN-only feature pipelines")
    print("=" * 78)
    prereg_path = os.path.join(C.D_PREREG, "STEP5C_LOEO_PREREGISTRATION.json")
    if not os.path.exists(prereg_path):
        raise SystemExit("FATAL: the protocol must be preregistered first (Stage D).")
    pre = json.load(open(prereg_path, encoding="utf-8"))
    assert pre["step5a_43_feature_list_is_primary"] is False

    man = pd.read_csv(os.path.join(C.D_FOLDS, "LOEO_MASTER_FOLD_MANIFEST.csv"))
    cand = C.candidate_feature_columns()
    print("  candidate schema: %d features (frozen, identical for all folds)" % len(cand))
    df, feat_cols = C.load_domain_a(cand)
    print("  Domain-A rows loaded: %d (Domain B / ANCHOR never read)" % len(df))

    blocks = json.load(open(C.FEATURE_BUILD_SUMMARY, encoding="utf-8"))
    f1s, f2s, f3s = set(blocks["F1"]), set(blocks["F2"]), set(blocks["F3"])

    summary = {}
    for heldout in C.HELDOUT_ELEMENTS:
        fold = f"FOLD_{heldout}"
        roles = man[man.fold == fold].set_index("composition_id")["loeo_role"]
        d = df.copy()
        d["loeo_role"] = d["composition_id"].map(roles)
        train = d[d.loeo_role == "FOLD_TRAIN"]
        assert (train["family"] != heldout).all(), "held-out family leaked into FOLD_TRAIN"
        assert set(train["sampling_domain"]) == {"A"}, "non-Domain-A row in FOLD_TRAIN"

        sel, nzv, exact, pcc, nan_cols, inf_cols = select_features_for_fold(
            train, feat_cols, heldout)

        base = os.path.join(C.D_FEAT, f"FOLD_{heldout}")
        with open(base + "_SELECTED_FEATURES.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sel) + "\n")
        pd.DataFrame(nzv).to_csv(base + "_NZV_REMOVAL.csv", index=False, encoding="utf-8")
        pd.DataFrame(exact).to_csv(base + "_EXACT_REDUNDANCY.csv", index=False,
                                   encoding="utf-8")
        pd.DataFrame(pcc).to_csv(base + "_PCC_REMOVAL.csv", index=False, encoding="utf-8")

        f1 = [c for c in sel if c in f1s]
        f2 = [c for c in sel if c in f2s]
        f3 = [c for c in sel if c in f3s]
        f4 = [c for c in sel if c.startswith("magpie_")]
        prov = {
            "fold": fold, "heldout_element": heldout,
            "generated_utc": C.utcnow(),
            "scope": "FOLD_TRAIN rows only (spec 21)",
            "fold_train_rows": int(len(train)),
            "fold_train_compositions": int(train["composition_id"].nunique()),
            "fold_train_families": sorted(set(train["family"])),
            "heldout_element_rows_used": 0,
            "validation_rows_used": 0,
            "heldout_test_rows_used": 0,
            "candidate_features": len(feat_cols),
            "candidate_schema_sha256": C.sha256_obj(sorted(feat_cols)),
            "nan_columns": nan_cols, "inf_columns": inf_cols,
            "nzv_removed": len(nzv),
            "exact_redundancy_removed": len(exact),
            "pcc_removed": len(pcc), "pcc_threshold": C.PCC_THRESHOLD,
            "selected_total": len(sel),
            "selected_by_block": {"F1": f1, "F2": f2, "F3": f3, "F4_n": len(f4), "F4": f4},
            "keep_priority_sha256": C.sha256_obj(C.KEEP_PRIORITY),
            "keep_priority_changed_for_this_fold": False,
            "selected_features_sha256": C.sha256_obj(sel),
            "selected_features_file_sha256": C.sha256_file(base + "_SELECTED_FEATURES.txt"),
            "pcc_log_sha256": C.sha256_file(base + "_PCC_REMOVAL.csv"),
            "nzv_log_sha256": C.sha256_file(base + "_NZV_REMOVAL.csv"),
            "step5a_43_list_used": False,
        }
        C.write_json(base + "_FEATURE_PROVENANCE.json", prov)
        summary[heldout] = {"n_selected": len(sel), "n_F1": len(f1), "n_F2": len(f2),
                            "n_F3": len(f3), "n_F4": len(f4),
                            "nzv_removed": len(nzv), "exact_removed": len(exact),
                            "pcc_removed": len(pcc),
                            "fold_train_rows": int(len(train)),
                            "features_sha256": prov["selected_features_sha256"]}
        print("  %-8s selected %3d  (F1 %d | F2 %d | F3 %2d | F4 %2d)   "
              "NZV %2d  exact %2d  PCC %2d   train rows %d"
              % (fold, len(sel), len(f1), len(f2), len(f3), len(f4),
                 len(nzv), len(exact), len(pcc), len(train)))

    # cross-fold stability (spec 51/52) - by-fold table now, frequency table at report time
    rows = []
    for heldout in C.HELDOUT_ELEMENTS:
        sel = [ln.strip() for ln in open(
            os.path.join(C.D_FEAT, f"FOLD_{heldout}_SELECTED_FEATURES.txt"),
            encoding="utf-8") if ln.strip()]
        for f in sel:
            rows.append({"heldout_element": heldout, "feature": f,
                         "feature_block": ("F1" if f in f1s else "F2" if f in f2s
                                           else "F3" if f in f3s else
                                           "F4" if f.startswith("magpie_") else "OTHER")})
    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(os.path.join(C.D_METRICS, "LOEO_FEATURE_SELECTION_BY_FOLD.csv"),
                   index=False, encoding="utf-8")

    freq = (by_fold.assign(v=1)
            .pivot_table(index=["feature", "feature_block"], columns="heldout_element",
                         values="v", aggfunc="sum", fill_value=0).reset_index())
    for e in C.HELDOUT_ELEMENTS:
        if e not in freq.columns:
            freq[e] = 0
    freq["selected_in_n_folds"] = freq[C.HELDOUT_ELEMENTS].sum(axis=1)
    freq = freq[["feature", "selected_in_n_folds"] + C.HELDOUT_ELEMENTS + ["feature_block"]]
    freq = freq.sort_values(["selected_in_n_folds", "feature"],
                            ascending=[False, True]).reset_index(drop=True)
    freq.to_csv(os.path.join(C.D_METRICS, "LOEO_FEATURE_SELECTION_FREQUENCY.csv"),
                index=False, encoding="utf-8")
    print("\n  feature stability: %d distinct features; %d selected in all six folds"
          % (len(freq), int((freq.selected_in_n_folds == 6).sum())))

    C.write_json(os.path.join(C.D_FEAT, "STEP5C_FEATURE_STAGE_SUMMARY.json"),
                 {"stage": "STEP5C Stage F", "generated_utc": C.utcnow(),
                  "candidate_features": len(feat_cols),
                  "per_fold": summary,
                  "identical_selection_across_folds": len(
                      {v["features_sha256"] for v in summary.values()}) == 1,
                  "step5a_43_feature_list_used_as_primary": False})
    print("\nSTAGE F: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
