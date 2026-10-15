# -*- coding: utf-8 -*-
"""
95_h_maximin.py
STEP5H-A Stage 5: POST-BLIND MAXIMIN PREPARATION (spec 32-36) - the target-free
geometry ONLY. The preregistered SECONDARY few-shot strategy needs its anchor
sequence fixed while no Ni target exists; this stage computes and freezes it.

Geometry (spec 32): frozen PRIMARY scaler, the 43 production inputs, 11 rows per
composition flattened in temperature order -> 43 x 11 = 473 dimensions.
Reference set (spec 33): original PRIMARY train compositions having all 11
temperature rows. No target information, no Ni CALPHAD.
Algorithm (spec 34): within the frozen 600 FEWSHOT_CALIBRATION compositions,
select a NESTED sequence of 32 - first maximizes nearest distance to the PRIMARY
train compositions, then sequentially maximizes the minimum distance to (train +
already-selected). Ties: lexicographically smallest composition_id.
Freeze (spec 35): STEP5H_A_MAXIMIN_SEQUENCE.csv, 32 IDs in rank order, SHA256
frozen; the sequence cannot change after the Ni CALPHAD reveal.
Random sequences (spec 36): NOT regenerated - the three STEP5G-frozen sequences
are verified by hash instead.
NO few-shot execution here (spec 37): no labels, no refit, no B_skill, no
B_recovery25.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5h_common as C


def composition_vectors(feats, frame, temp_col="Temperature_C"):
    """Flatten per-composition standardized feature rows (temperature order)
    into 473-dim vectors. temp_col must hold the REAL temperatures - after
    standardization the Temperature_C feature column itself holds z-scores."""
    out, order = {}, []
    for cid, g in frame.groupby("composition_id", sort=False):
        g = g.sort_values(temp_col)
        if len(g) != len(C.TEMPERATURES_C):
            continue
        if list(g[temp_col].astype(int)) != C.TEMPERATURES_C:
            continue
        out[cid] = g[feats].to_numpy(float).ravel()
        order.append(cid)
    return out, order


def main():
    import joblib
    from scipy.spatial import cKDTree
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5H-A  Stage 5: target-free maximin sequence freeze + random-sequence check")
    print("=" * 78)
    C.assert_no_tc_python()

    feats = C.feature_schema()
    C.verify_hash(C.G_PRIMARY_SCALER, C.FROZEN_HASHES["primary_scaler"], "PRIMARY scaler")
    scaler = joblib.load(C.G_PRIMARY_SCALER)
    if list(scaler.feature_names_in_) != feats:
        raise SystemExit("FATAL: scaler feature order != frozen schema")

    # ---- reference: PRIMARY train compositions with all 11 rows (spec 33) --- #
    ids = pd.read_csv(C.G_TRAIN_IDS)
    id_col = "state_id" if "state_id" in ids.columns else ids.columns[0]
    # Temperature_C is itself one of the 43 features - deduplicate the column list
    cols = list(dict.fromkeys(["state_id", "composition_id"] + feats))
    d = pd.read_parquet(C.D_DIAG_DATA, columns=cols)
    tr = d[d.state_id.isin(set(ids[id_col]))]
    tr_std = tr.copy()
    tr_std["T_real"] = tr["Temperature_C"].to_numpy()   # survives standardization
    tr_std[feats] = scaler.transform(tr[feats].to_numpy(float))
    ref_vecs, ref_ids = composition_vectors(feats, tr_std, temp_col="T_real")
    ref_mat = np.stack([ref_vecs[c] for c in ref_ids])
    print("  reference: %d PRIMARY train compositions with all 11 temperatures "
          "(%d dims)" % (len(ref_ids), ref_mat.shape[1]))

    # ---- candidates: frozen 600 FEWSHOT_CALIBRATION compositions (spec 34) -- #
    ni = pd.read_parquet(os.path.join(C.D_FEAT, "STEP5H_A_NI_FEATURES.parquet"))
    cal_ids = sorted(ni.loc[ni.fewshot_role == "FEWSHOT_CALIBRATION",
                            "composition_id"].unique())
    if len(cal_ids) != 600:
        raise SystemExit("FATAL: %d calibration compositions, expected 600" % len(cal_ids))
    ni_std = ni.copy()
    ni_std["T_real"] = ni["Temperature_C"].to_numpy()
    ni_std[feats] = scaler.transform(ni[feats].to_numpy(float))
    cand_vecs, cand_order = composition_vectors(feats, ni_std, temp_col="T_real")
    cand_ids = [c for c in cal_ids if c in cand_vecs]
    if len(cand_ids) != 600:
        raise SystemExit("FATAL: only %d calibration compositions have full 11-T "
                         "vectors" % len(cand_ids))
    cand_mat = np.stack([cand_vecs[c] for c in cand_ids])

    # ---- nested greedy maximin with lexicographic tie-break ------------------ #
    tree_ref = cKDTree(ref_mat)
    d0, _ = tree_ref.query(cand_mat, k=1, workers=-1)
    chosen, min_d, sel_dist = [], d0.copy(), []
    while len(chosen) < 32:
        best = np.max(min_d)
        tied = [i for i in np.flatnonzero(min_d == best)]
        pick = sorted(tied, key=lambda i: cand_ids[i])[0]   # lexicographic smallest id
        chosen.append(int(pick))
        sel_dist.append(float(min_d[pick]))
        min_d[pick] = -np.inf
        if len(chosen) < 32:
            dists = np.linalg.norm(cand_mat - cand_mat[pick][None, :], axis=1)
            min_d = np.minimum(min_d, dists)
            min_d[np.array(chosen)] = -np.inf
    seq_ids = [cand_ids[i] for i in chosen]
    seq = pd.DataFrame({"rank": range(1, 33), "composition_id": seq_ids,
                        "strategy": "TARGET_FREE_MAXIMIN_473DIM",
                        "min_distance_at_selection": sel_dist})
    out = os.path.join(C.D_MAXIMIN, "STEP5H_A_MAXIMIN_SEQUENCE.csv")
    seq.to_csv(out, index=False, encoding="utf-8")
    seq_sha = C.sha256_file(out)
    print("  maximin sequence frozen: 32 IDs, rank 1 = %s | sha256 %s"
          % (seq_ids[0], seq_sha[:16]))

    # ---- random sequences: verify unchanged, DO NOT regenerate (spec 36) ---- #
    C.verify_hash(C.G_SEQ_CSV, C.FROZEN_HASHES["fewshot_sequences_csv"],
                  "STEP5G nested random sequences")
    few = json.load(open(C.G_FEWSHOT_JSON, encoding="utf-8"))
    seq_csv = pd.read_csv(C.G_SEQ_CSV)
    json_ok = all(
        seq_csv[(seq_csv.replicate == int(r))].sort_values("rank").composition_id.tolist()
        == v for r, v in few["sequences"].items())
    if not json_ok:
        raise SystemExit("FATAL: random sequence CSV content != preregistration JSON")
    print("  random nested sequences verified UNCHANGED (hash + content match); "
          "not regenerated")

    C.write_json(os.path.join(C.D_MAXIMIN, "STEP5H_A_MAXIMIN_FREEZE.json"), {
        "generated_utc": C.utcnow(),
        "purpose": "SECONDARY few-shot comparison anchor sequence (spec 66-67 of "
                   "STEP5G); SECONDARY COMPARISON ONLY, not preferred",
        "geometry": {"scaler": "frozen PRIMARY (hash-verified)",
                     "features": len(feats), "temperatures": len(C.TEMPERATURES_C),
                     "dimensions": len(feats) * len(C.TEMPERATURES_C),
                     "flattening": "temperature order, per composition",
                     "distance": "euclidean"},
        "reference_set": {"source": "PRIMARY train compositions with all 11 "
                                    "temperature rows", "n": int(len(ref_ids)),
                          "target_information_used": False,
                          "ni_calphad_used": False},
        "candidates": {"pool": "frozen 600 FEWSHOT_CALIBRATION compositions",
                       "n": len(cand_ids)},
        "algorithm": "nested greedy maximin; tie -> lexicographically smallest "
                     "composition_id",
        "sequence": seq_ids,
        "sequence_file": out,
        "sequence_sha256": seq_sha,
        "immutable_after_ni_calphad_reveal": True,
        "random_sequences": {"file": C.G_SEQ_CSV,
                             "sha256": C.FROZEN_HASHES["fewshot_sequences_csv"],
                             "regenerated": False,
                             "content_matches_preregistration_json": True},
        "fewshot_executed_here": False,
        "labels_used": 0, "b_skill_computed": False, "b_recovery25_computed": False})
    print("STAGE 5: PASS (no label, no refit, no B_skill/B_recovery25)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
