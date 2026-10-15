# -*- coding: utf-8 -*-
"""
56_step5c_distance.py
STEP5C Stage M: element-descriptor distance and composition-level distance (spec 53-57).

Element distance (spec 53-55)
  vector  : PURE-ELEMENT external properties only - no CALPHAD, no phase, no partition,
            no target quantity of any kind
  scaler  : fitted on the REMAINING FIVE X elements of that fold only
  outputs : nearest-element, mean-to-5 and centroid Euclidean distances
  caveat  : n = 6 held-out elements. Spearman rho is DESCRIPTIVE / EXPLORATORY only and
            must never be presented as a statistical inference.

Composition distance (spec 56-57)
  vector  : that fold's selected features MINUS Temperature_C, at composition level
  scaler  : FOLD_TRAIN only
  outputs : 1-NN distance, 5-NN mean distance, outside-train-range feature count,
            Spearman correlation with each error, and quartile-stratified errors
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5c_common as C


def element_distance():
    print("\n--- element descriptor distance (spec 53-55) ---")
    tab = pd.read_csv(C.ELEMENT_DESCRIPTORS)
    tab["element"] = tab["element"].astype(str).str.upper()
    hits = C.ni_scan([("element_descriptor_table", tab)])
    assert not hits, "Ni token found in the element descriptor table: %s" % hits[:3]
    tab = tab[tab["element"].isin(C.HELDOUT_ELEMENTS)].set_index("element")
    props = [p for p in C.ELEMENT_DISTANCE_PROPERTIES if p in tab.columns]
    rows = []
    for heldout in C.HELDOUT_ELEMENTS:
        others = [e for e in C.HELDOUT_ELEMENTS if e != heldout]
        Tr = tab.loc[others, props].to_numpy(np.float64)
        mu, sd = Tr.mean(axis=0), Tr.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)                  # scaler: remaining 5 only (spec 54)
        Z = (Tr - mu) / sd
        z = (tab.loc[heldout, props].to_numpy(np.float64) - mu) / sd
        dists = np.linalg.norm(Z - z, axis=1)
        rows.append({
            "heldout_element": heldout,
            "nearest_element": others[int(np.argmin(dists))],
            "nearest_element_distance": float(dists.min()),
            "mean_distance_to_5": float(dists.mean()),
            "max_distance_to_5": float(dists.max()),
            "centroid_distance": float(np.linalg.norm(z - Z.mean(axis=0))),
            "training_elements": ";".join(others),
            "per_element_distance": json.dumps(
                {o: round(float(v), 6) for o, v in zip(others, dists)}),
            "n_properties": len(props),
            "scaler_scope": "remaining 5 X elements of this fold only",
        })
        print("  %-3s nearest %-3s d=%.3f | mean %.3f | centroid %.3f"
              % (heldout, rows[-1]["nearest_element"], dists.min(), dists.mean(),
                 rows[-1]["centroid_distance"]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.D_DIST, "LOEO_ELEMENT_DISTANCE.csv"),
               index=False, encoding="utf-8")
    return out


def composition_distance(df, man):
    print("\n--- composition-level distance (spec 56) ---")
    frames = []
    for heldout in C.HELDOUT_ELEMENTS:
        fold = f"FOLD_{heldout}"
        fr = json.load(open(os.path.join(C.D_MANIFEST, f"{fold}_FREEZE.json"),
                            encoding="utf-8"))
        feats = [f for f in fr["selected_features"] if f != "Temperature_C"]  # spec 56
        roles = man[man.fold == fold].set_index("composition_id")["loeo_role"]
        d = df.copy()
        d["loeo_role"] = d["composition_id"].map(roles)
        # composition level: one row per composition (features are T-independent here)
        comp = (d.groupby(["composition_id", "family", "loeo_role"], as_index=False)[feats]
                .mean())
        tr = comp[comp.loeo_role == "FOLD_TRAIN"]
        te = comp[comp.loeo_role == "HELDOUT_ELEMENT_TEST"]
        A = tr[feats].to_numpy(np.float64)
        mu, sd = A.mean(axis=0), A.std(axis=0)          # scaler: FOLD_TRAIN only
        sd = np.where(sd > 0, sd, 1.0)
        Az = (A - mu) / sd
        B = te[feats].to_numpy(np.float64)
        Bz = (B - mu) / sd
        lo, hi = A.min(axis=0), A.max(axis=0)
        outside = ((B < lo) | (B > hi)).sum(axis=1)
        # chunked 1-NN / 5-NN to bound memory
        nn1 = np.empty(len(Bz)); nn5 = np.empty(len(Bz))
        step = 512
        for s in range(0, len(Bz), step):
            blk = Bz[s:s + step]
            dd = np.sqrt(((blk[:, None, :] - Az[None, :, :]) ** 2).sum(axis=2))
            part = np.partition(dd, 5, axis=1)
            nn1[s:s + step] = part[:, 0]
            nn5[s:s + step] = part[:, :5].mean(axis=1)
        frames.append(pd.DataFrame({
            "heldout_element": heldout,
            "composition_id": te["composition_id"].values,
            "family": te["family"].values,
            "nn1_distance": nn1, "nn5_mean_distance": nn5,
            "outside_train_range_feature_count": outside,
            "n_features_used": len(feats)}))
        print("  %-3s 1-NN median %.3f | 5-NN median %.3f | outside-range median %d"
              % (heldout, np.median(nn1), np.median(nn5), int(np.median(outside))))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(os.path.join(C.D_DIST, "LOEO_COMPOSITION_DISTANCE.parquet"), index=False)
    return out


def _spearman(x, y):
    """Spearman rho, or NaN when either input is constant (rho is undefined there,
    not zero). Returning 0 would silently invent a 'no relationship' finding."""
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan"), float("nan"), "undefined: constant input"
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue), ""


def error_vs_distance(comp_dist):
    print("\n--- error vs composition distance (spec 57) ---")
    rows, quart = [], []
    for heldout in C.HELDOUT_ELEMENTS:
        p = pd.read_parquet(os.path.join(
            C.D_PRED, f"FOLD_{heldout}_HELDOUT_PREDICTIONS.parquet"))
        cd = comp_dist[comp_dist.heldout_element == heldout].set_index("composition_id")
        per_comp = {}
        for task in C.REGRESSION_TASKS:
            tc = [f"true_{task}_{f}" for f in C.FAMILIES_ORDER]
            pc = [f"pred_proj_{task}_{f}" for f in C.FAMILIES_ORDER]
            if tc[0] not in p.columns:
                continue
            q = p[p[tc[0]].notna()]
            t = C.tvd(q[pc].to_numpy(np.float64), q[tc].to_numpy(np.float64))
            per_comp[task] = pd.Series(t, index=q["composition_id"].values) \
                .groupby(level=0).mean()
        tc = [f"true_presence_{f}" for f in C.FAMILIES_ORDER]
        pc = [f"pred_proj_presence_{f}" for f in C.FAMILIES_ORDER]
        if tc[0] in p.columns:
            q = p[p[tc[0]].notna()]
            e = (q[tc].to_numpy() != q[pc].to_numpy()).mean(axis=1)
            per_comp["presence"] = pd.Series(e, index=q["composition_id"].values) \
                .groupby(level=0).mean()
        for task, s in per_comp.items():
            common = cd.index.intersection(s.index)
            d1 = cd.loc[common, "nn1_distance"].to_numpy(np.float64)
            d5 = cd.loc[common, "nn5_mean_distance"].to_numpy(np.float64)
            err = s.loc[common].to_numpy(np.float64)
            r1, p1, n1 = _spearman(d1, err)
            r5, p5, n5 = _spearman(d5, err)
            rows.append({"heldout_element": heldout, "task": task,
                         "metric": "label_error_rate" if task == "presence" else "mean_tvd",
                         "n_compositions": int(len(common)),
                         "spearman_rho_nn1": r1, "spearman_p_nn1": p1,
                         "spearman_rho_nn5": r5, "spearman_p_nn5": p5,
                         "note": n1 or n5})
            qs = pd.qcut(pd.Series(d1, index=common), 4, labels=["Q1", "Q2", "Q3", "Q4"],
                         duplicates="drop")
            for lab, g in pd.DataFrame({"q": qs, "err": err}, index=common).groupby("q",
                                                                                   observed=True):
                quart.append({"heldout_element": heldout, "task": task,
                              "distance_quartile": str(lab),
                              "mean_error": float(g["err"].mean()),
                              "median_error": float(g["err"].median()),
                              "n_compositions": int(len(g))})
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(C.D_DIST, "LOEO_ERROR_DISTANCE.csv"), index=False, encoding="utf-8")
    pd.DataFrame(quart).to_csv(
        os.path.join(C.D_DIST, "LOEO_ERROR_BY_DISTANCE_QUARTILE.csv"),
        index=False, encoding="utf-8")
    return ev


def error_vs_element_distance(el_dist):
    """spec 55: n = 6. DESCRIPTIVE ONLY."""
    met = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_ALL_FOLDS_METRICS.csv"))
    rows = []
    for task in C.ALL_TASKS:
        m = met[met.task == task].set_index("heldout_element")["value"]
        e = el_dist.set_index("heldout_element")
        common = [x for x in C.HELDOUT_ELEMENTS if x in m.index and x in e.index]
        if len(common) < 3:
            continue
        for dcol in ("nearest_element_distance", "mean_distance_to_5", "centroid_distance"):
            rho, pv, note = _spearman(e.loc[common, dcol].to_numpy(np.float64),
                                      m.loc[common].to_numpy(np.float64))
            rows.append({"task": task, "distance_metric": dcol, "n": len(common),
                         "spearman_rho": rho, "spearman_p": pv, "note": note,
                         "interpretation_limit": ("DESCRIPTIVE / EXPLORATORY ONLY - n = 6 "
                                                  "held-out elements; not a statistical "
                                                  "inference (spec 55)")})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.D_DIST, "LOEO_ERROR_VS_ELEMENT_DISTANCE.csv"),
               index=False, encoding="utf-8")
    return out


def main():
    print("=" * 78)
    print("STEP 5C   Stage M: distance analysis")
    print("=" * 78)
    man = pd.read_csv(os.path.join(C.D_FOLDS, "LOEO_MASTER_FOLD_MANIFEST.csv"))
    all_feats = sorted({f for e in C.HELDOUT_ELEMENTS
                        for f in json.load(open(os.path.join(
                            C.D_MANIFEST, f"FOLD_{e}_FREEZE.json"),
                            encoding="utf-8"))["selected_features"]})
    df, _ = C.load_domain_a(all_feats)
    el = element_distance()
    cd = composition_distance(df, man)
    error_vs_distance(cd)
    error_vs_element_distance(el)
    print("\nSTAGE M: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
