# -*- coding: utf-8 -*-
"""
71_f5f_run.py
STEP5F Stage B: the CALPHAD run - 9,750 new compositions x 11 temperatures =
107,250 new solver states (spec 8, 9, 21).

This module does NOT re-implement the equilibrium loop. It imports the FROZEN
FULL_V1 runner `22_fullv1_run.py` and calls its `run_shard` / `export_shard`,
so the thermodynamic settings (TCCU6, 1 bar, 1 mol, wt.%, Cu balance,
equilibrium only), the Level 0-3 solver escalation, the adaptive temperature
bridge, the single-transaction SQLite store and the 5-family aggregation are
the exact code that produced the FULL_V1 database (spec 9). Nothing in
code_snapshot/ of FULL_V1 is edited.

The 250 pilot compositions are NOT re-run (spec 6): they are merged at the
processing stage from the pilot's own frozen parquet.

Failure policy (spec 21): failures are retained by the frozen store with their
verbatim Thermo-Calc error; nothing is interpolated or dropped silently.

Resumable: already-SUCCESS states are skipped by the frozen checkpoint, so an
interrupted run is restarted by simply running this again.

Sharding: FULL_V1 itself ran as 2 concurrent shard processes; this stage uses
the same stride split (--shard k --shards n). A composition's 11 temperatures
always stay inside one shard.
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5f_common as C


def load_frozen_runner():
    path = os.path.join(C.FULL_V1_CODE, "22_fullv1_run.py")
    if not os.path.exists(path):
        raise SystemExit("FATAL: frozen runner not found at %s" % path)
    spec = importlib.util.spec_from_file_location("fullv1_runner_f5f", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_temperature_file():
    p = os.path.join(C.D_CONFIG, "STEP5F_FULL_TEMPERATURES.csv")
    if not os.path.exists(p):
        pd.DataFrame({"Temperature_C": C.TEMPERATURES_C,
                      "Temperature_K": [t + 273.15 for t in C.TEMPERATURES_C]}
                     ).to_csv(p, index=False, encoding="utf-8")
    return p


def build_failure_map(store, comps):
    states = store.frame("state_result")
    if not len(states):
        return pd.DataFrame(columns=["element", "composition_id", "state_id",
                                     "Temperature_C", "stage", "exception"])
    conv = states["converged"].astype(str).str.lower() == "true"
    f = states[~conv].copy().rename(columns={"case_id": "composition_id"})
    meta = comps.set_index("composition_id")
    f["element"] = f["composition_id"].map(meta["X_element"])
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
        ["element", "composition_id", "Temperature_C"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=None, help="debug cap")
    args = ap.parse_args()
    n, nsh = args.shard, args.shards

    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    R = load_frozen_runner()

    comp_path = os.path.join(C.D_COMP, "STEP5F_FULL_NEW_COMPOSITIONS.csv")
    if not os.path.exists(comp_path):
        raise SystemExit("FATAL: run 70_f5f_compositions.py first.")
    comps_all = pd.read_csv(comp_path)
    ids = comps_all["composition_id"].tolist()
    # stride split as in FULL_V1: keeps families balanced across shards while a
    # composition's 11 temperatures stay in one process
    subset = set(ids[(n - 1)::nsh])
    comps = comps_all[comps_all.composition_id.isin(subset)].copy()
    temp_path = write_temperature_file()
    temps = R.load_temperatures(temp_path)
    cases = [R.make_case(r) for _, r in comps.iterrows()]

    log = Path(C.D_LOGS) / ("STEP5F_FULL_RUN_shard%02d.log" % n)
    logger = R.lib.setup_logging(log, name="f5f_s%d" % n)
    print("=" * 78)
    print("STEP5F FULL  Stage B: CALPHAD run  (frozen FULL_V1 runner reused verbatim)")
    print("shard %d/%d: %d compositions x %d temperatures = %d states"
          % (n, nsh, len(cases), len(temps), len(cases) * len(temps)))
    print("elements: %s" % " ".join(sorted(set(comps.X_element))))
    print("=" * 78)

    db = Path(C.D_RAW) / ("STEP5F_FULL_checkpoint_shard%02d.sqlite" % n)
    progress_csv = Path(C.D_RAW) / ("STEP5F_FULL_checkpoint_shard%02d.csv" % n)
    t0 = time.time()
    stats, store, cp, timing = R.run_shard(
        cases, temps, db, log, logger=logger, max_states=args.max_states,
        progress_every=1000, progress_csv=progress_csv, label="f5f shard %d" % n)

    R.export_shard(store, cp, {"raw": Path(C.D_RAW), "proc": Path(C.D_PROC)},
                   "STEP5F_FULL_shard%02d_" % n, logger, progress_csv=progress_csv)

    fmap = build_failure_map(store, comps_all)
    fmap.to_csv(Path(C.D_CALQA) / ("STEP5F_FULL_FAILURE_MAP_shard%02d.csv" % n),
                index=False, encoding="utf-8")
    n_ok, n_res, missing = store.consistency()
    states = store.frame("state_result")
    per_el = {}
    if len(states):
        st = states.copy()
        st["element"] = st["case_id"].map(
            comps_all.set_index("composition_id")["X_element"])
        st["ok"] = st["converged"].astype(str).str.lower() == "true"
        for el, g in st.groupby("element"):
            per_el[str(el)] = {"states": int(len(g)), "converged": int(g.ok.sum()),
                               "failed": int((~g.ok).sum()),
                               "convergence_rate": float(g.ok.mean())}
    C.write_json(Path(C.D_LOGS) / ("STEP5F_FULL_RUNTIME_shard%02d.json" % n), {
        "generated_utc": C.utcnow(),
        "frozen_runner": "code_snapshot/22_fullv1_run.py (FULL_V1, unedited)",
        "frozen_runner_sha256": C.sha256_file(
            os.path.join(C.FULL_V1_CODE, "22_fullv1_run.py")),
        "database": "TCCU6", "shard": n, "shards": nsh,
        "planned_states": int(len(cases) * len(temps)),
        "stats": stats, "timing": {k: round(v, 2) for k, v in timing.items()},
        "wall_seconds": round(time.time() - t0, 1),
        "checkpoint_success": int(n_ok), "stored_results": int(n_res),
        "success_without_result": missing[:20], "per_element": per_el,
        "failure_rows": int(len(fmap)),
        "failure_policy": ("failures retained with their verbatim Thermo-Calc error; "
                           "never interpolated, fabricated or silently dropped (spec 21)")})
    print("\n  shard %d wall %.1f min | failed %d | consistency: SUCCESS %d / results %d / missing %d"
          % (n, (time.time() - t0) / 60, stats["failed"], n_ok, n_res, len(missing)))
    for el in sorted(per_el):
        d = per_el[el]
        print("    %-3s %5d states | converged %5d | failed %3d | %.4f"
              % (el, d["states"], d["converged"], d["failed"], d["convergence_rate"]))
    if missing:
        print("  WARNING: %d states marked SUCCESS without a stored result" % len(missing))
    print("RUN_OK shard %d" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
