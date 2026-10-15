# -*- coding: utf-8 -*-
"""
98_hb_calphad.py
STEP5H-B Stage 1: the prospective Ni CALPHAD blind reveal - 2,000 frozen
compositions x 11 temperatures = 22,000 formal states (spec 2, 4-10, 18, 19).

Binding order inside every shard process:
  1. HARD PRE-RUN GATE (spec 2): the four frozen hashes + the frozen runner hash
     are verified EXACTLY; any mismatch stops before anything else happens.
  2. FIRST TARGET ACCESS RECORD (spec 4): written (atomically, once) IMMEDIATELY
     BEFORE any Thermo-Calc session is opened.
  3. only then is TCPython started and the frozen FULL_V1 runner reused
     byte-identically (spec 5, 6): same escalation, same temperature bridge,
     same retry policy (spec 19), same checkpoint store, same 5-family
     aggregation. The runner is NOT rewritten.

No prediction-guided CALPHAD (spec 9): the composition set is the frozen table
in its frozen order; the stride shard split is by composition index only (the
same technical load-balancing used in FULL_V1/STEP5F); predictions, OOD
distances, disagreements, predicted topologies and few-shot roles are never
read by this module. All 22,000 states are treated equally.

Failure policy (spec 18): failures keep their rows with verbatim exceptions;
nothing is interpolated, imputed, temperature-copied, fabricated or deleted.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hb_common as C


def load_frozen_runner():
    spec = importlib.util.spec_from_file_location("fullv1_runner_hb", C.FROZEN_RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_temperature_file():
    p = os.path.join(C.D_CONFIG, "STEP5H_B_TEMPERATURES.csv")
    if not os.path.exists(p):
        pd.DataFrame({"Temperature_C": C.TEMPERATURES_C,
                      "Temperature_K": [t + 273.15 for t in C.TEMPERATURES_C]}
                     ).to_csv(p, index=False, encoding="utf-8")
    return p


def ensure_first_access_record():
    """Spec 4: create the record BEFORE the first Thermo-Calc call, atomically
    and exactly once across concurrent shards."""
    path = os.path.join(C.D_ACCESS, "STEP5H_B_FIRST_TARGET_ACCESS.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    import tc_python
    try:
        tcpy_version = getattr(tc_python, "__version__", None) or \
            str(importlib.util.find_spec("tc_python").origin)
    except Exception:
        tcpy_version = "unknown"
    import importlib.metadata as md
    try:
        tcpy_version = md.version("tc-python")
    except Exception:
        pass
    rec = {
        "timestamp_utc": C.utcnow(),
        "stage": "STEP5H-B",
        "pipeline_freeze_sha256": C.FROZEN["step5g_pipeline_freeze"],
        "composition_sha256": C.FROZEN["ni_compositions_csv"],
        "PRIMARY_prediction_sha256": C.FROZEN["primary_predictions"],
        "SECONDARY_prediction_sha256": C.FROZEN["secondary_predictions"],
        "metric_preregistration_sha256": C.sha256_file(C.G_METRIC_JSON),
        "calphad_runner_path": C.FROZEN_RUNNER,
        "calphad_runner_sha256": C.FROZEN["frozen_runner"],
        "thermo_calc_version": "Thermo-Calc 2025b (calculation engine banner is "
                               "captured in the shard logs)",
        "tc_python_version": str(tcpy_version),
        "database_version": "TCCU6 Cu-Alloys v6.2 (frozen FULL_V1 database; "
                            "DATABASE_NAME='TCCU6' inside the frozen runner stack)",
        "settings": {"pressure_bar": 1.0, "system_amount_mol": 1.0,
                     "composition_unit": "wt.%", "balance": "Cu",
                     "mode": "equilibrium only"},
        "statement": "Both PRIMARY and SECONDARY Ni zero-shot predictions were "
                     "frozen before the first Ni CALPHAD target was generated.",
        "evidence": {
            "primary_frozen_utc": json.load(
                open(C.HA_PRIMARY_FREEZE, encoding="utf-8"))["immutable_from_utc"],
            "secondary_frozen_utc": json.load(
                open(C.HA_SECONDARY_FREEZE, encoding="utf-8"))["immutable_from_utc"]},
    }
    tmp = path + ".tmp.%d" % os.getpid()
    C.write_json(tmp, rec)
    try:
        os.rename(tmp, path)          # atomic on the same filesystem
    except OSError:
        os.remove(tmp)                # another shard won the race; use theirs
    return json.load(open(path, encoding="utf-8"))


def build_failure_map(store, comps):
    states = store.frame("state_result")
    if not len(states):
        return pd.DataFrame(columns=["element", "composition_id", "state_id",
                                     "Temperature_C", "stage", "exception"])
    conv = states["converged"].astype(str).str.lower() == "true"
    f = states[~conv].copy().rename(columns={"case_id": "composition_id"})
    f["element"] = "NI"
    meta = comps.set_index("composition_id")
    for col in ("Cr_wt_pct", "Nb_wt_pct", "X_wt_pct", "S_CrNb_wt_pct",
                "R_CrNb_atomic_actual"):
        f[col] = pd.to_numeric(f["composition_id"].map(meta[col]), errors="coerce")
    f["stage"] = "equilibrium"
    f["exception"] = (f.get("original_error_message", pd.Series("", index=f.index))
                      .astype(str).replace({"": None})
                      .fillna(f.get("error_message", pd.Series("", index=f.index))
                              .astype(str)))
    keep = ["element", "composition_id", "state_id", "Temperature_C", "stage",
            "solver_level_used", "bridge_used", "bridge_temperatures_C",
            "attempt_count", "final_status", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
            "S_CrNb_wt_pct", "R_CrNb_atomic_actual", "exception"]
    return f[[c for c in keep if c in f.columns]].sort_values(
        ["composition_id", "Temperature_C"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=None, help="debug cap")
    args = ap.parse_args()
    n, nsh = args.shard, args.shards

    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)

    # ---- spec 2: HARD PRE-RUN GATE ------------------------------------------ #
    C.hard_gate()
    print("hard pre-run gate: 5/5 frozen hashes verified EXACTLY")

    # ---- spec 4: FIRST TARGET ACCESS RECORD, before any TC call ------------- #
    rec = ensure_first_access_record()
    print("first target access record: %s" % rec["timestamp_utc"])

    R = load_frozen_runner()
    comps_all = pd.read_csv(C.G_NI_COMP_CSV)      # frozen order, no resampling
    if len(comps_all) != C.N_COMPOSITIONS:
        raise SystemExit("FATAL: frozen composition table has %d rows" % len(comps_all))
    ids = comps_all["composition_id"].tolist()
    subset = set(ids[(n - 1)::nsh])               # technical stride split only
    comps = comps_all[comps_all.composition_id.isin(subset)].copy()
    temps = R.load_temperatures(write_temperature_file())
    cases = [R.make_case(r) for _, r in comps.iterrows()]
    for cse in cases:
        if cse["elements_string"] != "CU,CR,NB,NI":
            raise SystemExit("FATAL: unexpected element set %r" % cse["elements_string"])

    log = Path(C.D_LOGS) / ("STEP5H_B_RUN_shard%02d.log" % n)
    logger = R.lib.setup_logging(log, name="hb_s%d" % n)
    print("=" * 78)
    print("STEP5H-B  Stage 1: Ni CALPHAD blind reveal (frozen FULL_V1 runner, unedited)")
    print("shard %d/%d: %d compositions x %d temperatures = %d states"
          % (n, nsh, len(cases), len(temps), len(cases) * len(temps)))
    print("=" * 78)

    db = Path(C.D_RAW) / ("STEP5H_B_checkpoint_shard%02d.sqlite" % n)
    progress_csv = Path(C.D_RAW) / ("STEP5H_B_checkpoint_shard%02d.csv" % n)
    t0 = time.time()
    stats, store, cp, timing = R.run_shard(
        cases, temps, db, log, logger=logger, max_states=args.max_states,
        progress_every=1000, progress_csv=progress_csv, label="hb shard %d" % n)
    R.export_shard(store, cp, {"raw": Path(C.D_RAW), "proc": Path(C.D_TARGETS)},
                   "STEP5H_B_shard%02d_" % n, logger, progress_csv=progress_csv)

    fmap = build_failure_map(store, comps_all)
    fmap.to_csv(Path(C.D_CALQA) / ("STEP5H_B_FAILURE_MAP_shard%02d.csv" % n),
                index=False, encoding="utf-8")
    n_ok, n_res, missing = store.consistency()
    states = store.frame("state_result")
    per_t = {}
    if len(states):
        st = states.copy()
        st["ok"] = st["converged"].astype(str).str.lower() == "true"
        for t, g in st.groupby("Temperature_C"):
            per_t[str(int(round(float(t))))] = {"states": int(len(g)),
                                                "converged": int(g.ok.sum()),
                                                "failed": int((~g.ok).sum())}
    C.write_json(Path(C.D_LOGS) / ("STEP5H_B_RUNTIME_shard%02d.json" % n), {
        "generated_utc": C.utcnow(),
        "frozen_runner": "code_snapshot/22_fullv1_run.py (FULL_V1, unedited)",
        "frozen_runner_sha256": C.FROZEN["frozen_runner"],
        "database": "TCCU6", "shard": n, "shards": nsh,
        "planned_states": int(len(cases) * len(temps)),
        "stats": stats, "timing": {k: round(v, 2) for k, v in timing.items()},
        "wall_seconds": round(time.time() - t0, 1),
        "checkpoint_success": int(n_ok), "stored_results": int(n_res),
        "success_without_result": missing[:20],
        "per_temperature": per_t, "failure_rows": int(len(fmap)),
        "first_target_access_utc": rec["timestamp_utc"],
        "prediction_guided_calphad": False,
        "failure_policy": ("failures retained with verbatim exceptions; never "
                           "interpolated, imputed, temperature-copied, fabricated "
                           "or silently deleted (spec 18)")})
    print("\n  shard %d wall %.1f min | failed %d | consistency: SUCCESS %d / results "
          "%d / missing %d" % (n, (time.time() - t0) / 60, stats["failed"], n_ok,
                               n_res, len(missing)))
    if missing:
        print("  WARNING: %d SUCCESS states without a stored result" % len(missing))
    print("RUN_OK shard %d" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
