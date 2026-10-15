# -*- coding: utf-8 -*-
"""
94_h_predict.py
STEP5H-A Stage 4: PRIMARY zero-shot prediction + IMMEDIATE freeze (spec 23-26),
then SECONDARY prediction + freeze (spec 27-29), then target-free disagreement
(spec 30) and the prediction distribution summary (spec 31).

Binding order (spec 26, 27, 57-of-STEP5G): PRIMARY is predicted and frozen
BEFORE SECONDARY is loaded. Both predictions cover the EXACT same 22,000 states.
Neither model is retrained; neither scaler is refit; standardized OOD values are
NOT clipped, winsorized, capped or recentered (spec 16). No target exists yet,
so nothing here can compute an error, and nothing does (spec 37, 45).
"""
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5h_common as C


def network(n):
    from torch import nn
    return nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                         nn.Linear(64, 5))


def predict_role(which, feats, ni, expect_role):
    """Verify the frozen checkpoint + scaler EXACTLY (spec 13-15), then predict."""
    import joblib
    import torch
    ck, ck_path, sc_path = C.load_frozen_checkpoint(which)
    if ck["role"] != expect_role:
        raise SystemExit("FATAL: %s checkpoint role is %r, expected %r"
                         % (which, ck["role"], expect_role))
    if list(ck["features"]) != feats or ck["n_features"] != C.N_FEATURES:
        raise SystemExit("FATAL: %s checkpoint feature list/order != frozen schema"
                         % which)
    arch = ck["architecture"]
    if arch != "Linear(43,128)-ReLU-Linear(128,64)-ReLU-Linear(64,5)":
        raise SystemExit("FATAL: unexpected architecture string %r" % arch)
    scaler = joblib.load(sc_path)
    if list(scaler.feature_names_in_) != feats:
        raise SystemExit("FATAL: %s scaler feature order != frozen schema" % which)

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")     # CPU for bit-stable inference of a frozen model
    model = network(len(feats)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    X = ni[feats].to_numpy(float)
    Z = scaler.transform(X)          # frozen scaler; NO clipping/capping (spec 16)
    with torch.no_grad():
        raw = model(torch.as_tensor(Z, dtype=torch.float32, device=device)).numpy()
    raw = raw.astype(np.float64)
    fallback = np.asarray(ck["fallback_row"], dtype=float)
    proj, n_fb = C.project_simplex(raw, fallback)
    return raw, proj, n_fb, ck, arch


def build_frame(ni, raw, proj, n_fb, prefix):
    out = ni[["state_id", "composition_id", "Temperature_C", "fewshot_role"]].copy()
    for j, f in enumerate(C.FAMILIES_ORDER):
        out["raw_%s" % f] = raw[:, j]
    for j, f in enumerate(C.FAMILIES_ORDER):
        out["pred_%s" % f] = proj[:, j]
    out["pred_topology"] = [C.FAMILIES_ORDER[i] for i in proj.argmax(axis=1)]
    out["simplex_fallback"] = (raw.clip(min=0).sum(axis=1) == 0).astype(int)
    assert int(out.simplex_fallback.sum()) == n_fb
    return out


def qa_and_freeze(out, which, path_parquet, path_freeze):
    """Spec 25 QA + spec 26/29 immediate freeze."""
    checks = {
        "rows_22000": len(out) == C.N_STATES,
        "state_id_unique": out.state_id.is_unique,
        "all_finite": bool(np.isfinite(out[[c for c in out.columns
                                            if c.startswith(("raw_", "pred_"))
                                            and c != "pred_topology"]]
                                       .to_numpy(float)).all()),
        "projected_non_negative": bool((out[["pred_%s" % f for f in C.FAMILIES_ORDER]]
                                        .to_numpy(float) >= 0).all()),
        "projected_sum_one_1e12": bool(np.abs(
            out[["pred_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)
            .sum(axis=1) - 1.0).max() <= 1e-12),
        "target_column_guard_clean": not C.target_column_guard(out, "%s_PREDICTIONS" % which),
    }
    guard = C.target_column_guard(out, "%s_PREDICTIONS" % which)
    if guard:
        raise SystemExit("FATAL: target-like column in prediction artifact: %s" % guard)
    if not all(checks.values()):
        raise SystemExit("FATAL: %s prediction QA failed: %s" % (which, checks))
    out.to_parquet(path_parquet, index=False)
    schema_hash = C.sha256_text("|".join(out.columns))
    C.write_json(path_freeze, {
        "role": which,
        "generated_utc": C.utcnow(),
        "file": path_parquet,
        "sha256": C.sha256_file(path_parquet),
        "file_size_bytes": os.path.getsize(path_parquet),
        "row_count": int(len(out)),
        "schema_columns": list(out.columns),
        "schema_sha256": schema_hash,
        "simplex_fallback_count": int(out.simplex_fallback.sum()),
        "max_projected_sum_deviation": float(np.abs(
            out[["pred_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)
            .sum(axis=1) - 1.0).max()),
        "qa_checks": checks,
        "immutable_from_utc": C.utcnow(),
        "no_target_column_present": True})
    return checks, schema_hash


def distribution_summary(out, label):
    """Spec 31: distribution summary WITHOUT any accuracy terminology."""
    pred = out[["pred_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)
    topo = out.pred_topology.value_counts(normalize=True)
    by_t = (out.groupby("Temperature_C").pred_topology
            .value_counts(normalize=True).unstack(fill_value=0.0))
    return {
        "model": label,
        "mean_predicted_simplex": {f: float(pred[:, j].mean())
                                   for j, f in enumerate(C.FAMILIES_ORDER)},
        "median_component_values": {f: float(np.median(pred[:, j]))
                                    for j, f in enumerate(C.FAMILIES_ORDER)},
        "modal_topology_distribution": {k: float(v) for k, v in topo.items()},
        "topology_by_temperature": json.loads(by_t.round(6).to_json(orient="index")),
        "simplex_fallback_count": int(out.simplex_fallback.sum()),
        "extreme_probability_fractions": {
            "pred_max_component_gt_0.95": float((pred.max(axis=1) > 0.95).mean()),
            "pred_max_component_gt_0.99": float((pred.max(axis=1) > 0.99).mean()),
            "all_components_lt_0.05": float((pred.max(axis=1) < 0.05).mean())},
        "note": "descriptive distribution only; no accuracy terminology (spec 31)"}


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5H-A  Stage 4: PRIMARY -> freeze -> SECONDARY -> freeze -> disagreement")
    print("=" * 78)
    C.assert_no_tc_python()
    feats = C.feature_schema()
    ni = pd.read_parquet(os.path.join(C.D_FEAT, "STEP5H_A_NI_FEATURES.parquet"))
    if len(ni) != C.N_STATES:
        raise SystemExit("FATAL: Ni feature table has %d rows" % len(ni))

    # ---------------- PRIMARY first (spec 23-26) ----------------------------- #
    t0 = time.time()
    raw_p, proj_p, fb_p, ck_p, arch = predict_role(
        "PRIMARY", feats, ni, "ORIGINAL6_DIRECT_MLP")
    out_p = build_frame(ni, raw_p, proj_p, fb_p, "PRIMARY")
    p_parq = os.path.join(C.D_PRIMPRED, "STEP5H_A_PRIMARY_NI_PREDICTIONS.parquet")
    p_freeze = os.path.join(C.D_PRIMPRED, "STEP5H_A_PRIMARY_PREDICTION_FREEZE.json")
    checks_p, schema_p = qa_and_freeze(out_p, "PRIMARY", p_parq, p_freeze)
    print("  PRIMARY: 22000 predictions frozen | fallbacks %d | sha256 %s | %.1f s"
          % (fb_p, C.sha256_file(p_parq)[:16], time.time() - t0))

    # ---------------- SECONDARY only after PRIMARY freeze (spec 27-29) ------- #
    if not os.path.exists(p_freeze):
        raise SystemExit("FATAL: PRIMARY freeze missing")
    t0 = time.time()
    raw_s, proj_s, fb_s, ck_s, _ = predict_role(
        "SECONDARY", feats, ni, "FULL11_DIRECT_MLP")
    out_s = build_frame(ni, raw_s, proj_s, fb_s, "SECONDARY")
    s_parq = os.path.join(C.D_SECPRED, "STEP5H_A_SECONDARY_NI_PREDICTIONS.parquet")
    s_freeze = os.path.join(C.D_SECPRED, "STEP5H_A_SECONDARY_PREDICTION_FREEZE.json")
    checks_s, schema_s = qa_and_freeze(out_s, "SECONDARY", s_parq, s_freeze)
    print("  SECONDARY: 22000 predictions frozen | fallbacks %d | sha256 %s | %.1f s"
          % (fb_s, C.sha256_file(s_parq)[:16], time.time() - t0))

    # identical state coverage
    if list(out_p.state_id) != list(out_s.state_id):
        raise SystemExit("FATAL: PRIMARY and SECONDARY state_id sets differ")

    # ---------------- disagreement, descriptive only (spec 30) --------------- #
    Pp = out_p[["pred_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)
    Ps = out_s[["pred_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)
    tvd_ps = 0.5 * np.abs(Pp - Ps).sum(axis=1)
    topo_agree = (out_p.pred_topology == out_s.pred_topology).astype(int)
    dis = out_p[["state_id", "composition_id", "Temperature_C"]].copy()
    dis["tvd_primary_vs_secondary"] = tvd_ps
    dis["topology_agreement"] = topo_agree
    dis.to_csv(os.path.join(C.D_DIS, "STEP5H_A_PRIMARY_SECONDARY_DISAGREEMENT.csv"),
               index=False, encoding="utf-8")
    by_t = dis.groupby("Temperature_C").agg(
        mean_tvd=("tvd_primary_vs_secondary", "mean"),
        topology_agreement=("topology_agreement", "mean")).reset_index()
    by_c = dis.groupby("composition_id").agg(
        mean_tvd=("tvd_primary_vs_secondary", "mean"),
        topology_agreement=("topology_agreement", "mean")).reset_index()
    by_t.to_csv(os.path.join(C.D_DIS, "STEP5H_A_DISAGREEMENT_BY_TEMPERATURE.csv"),
                index=False, encoding="utf-8")
    by_c.to_csv(os.path.join(C.D_DIS, "STEP5H_A_DISAGREEMENT_BY_COMPOSITION.csv"),
                index=False, encoding="utf-8")
    print("  disagreement: mean TVD(P,S) %.4f | topology agreement %.4f "
          "(descriptive only - lower disagreement is NOT correctness)"
          % (tvd_ps.mean(), topo_agree.mean()))

    # ---------------- distribution summaries (spec 31) ----------------------- #
    summ = [distribution_summary(out_p, "PRIMARY"),
            distribution_summary(out_s, "SECONDARY")]
    C.write_json(os.path.join(C.D_CONFIG, "STEP5H_A_PREDICTION_DISTRIBUTIONS.json"), {
        "generated_utc": C.utcnow(), "summaries": summ,
        "disagreement": {"mean_tvd_primary_vs_secondary": float(tvd_ps.mean()),
                         "median_tvd": float(np.median(tvd_ps)),
                         "p95_tvd": float(np.percentile(tvd_ps, 95)),
                         "topology_agreement_rate": float(topo_agree.mean())},
        "interpretation_note": ("target-free descriptive statistics; no accuracy "
                                "terminology, no correctness interpretation")})

    C.write_json(os.path.join(C.D_CONFIG, "STEP5H_A_PREDICTION_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "primary": {"freeze": json.load(open(p_freeze, encoding="utf-8")),
                    "architecture": arch, "checkpoint_role": ck_p["role"],
                    "chosen_lr": ck_p["chosen_lr"]},
        "secondary": {"freeze": json.load(open(s_freeze, encoding="utf-8")),
                      "checkpoint_role": ck_s["role"], "chosen_lr": ck_s["chosen_lr"]},
        "state_ids_identical": True,
        "qa_primary": checks_p, "qa_secondary": checks_s,
        "schema_sha256_primary": schema_p, "schema_sha256_secondary": schema_s,
        "no_target_accessed": True, "no_error_computed": True,
        "no_model_retrained": True, "no_scaler_refit": True})
    print("STAGE 4: PASS - both prediction files immutable from now")
    return 0


if __name__ == "__main__":
    sys.exit(main())
