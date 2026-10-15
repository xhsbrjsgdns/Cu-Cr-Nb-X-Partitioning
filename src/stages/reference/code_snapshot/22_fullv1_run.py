# -*- coding: utf-8 -*-
"""
22_fullv1_run.py - STEP 4 Stage H runner: 50,000 compositions x 11 T = 550,000 states.

Byte-identical in physics and I/O to 18_pilot2000_run.py (Step 3C, PASS WITH WARNINGS);
only paths, file prefixes and the composition table differ. dry_run_lib / solver_escalation /
fivefamily / io_store are reused from code_snapshot/ unchanged (SHA256 in
06_manifest/CODE_SNAPSHOT_MANIFEST.json).

Original Step 3C header follows.

18_pilot2000_run.py - Step 3C runner: 2000 compositions x 11 temperatures = 22000 states.

Physics is UNCHANGED from Step 3B and is not reimplemented here:
    dry_run_lib.create_calculation / run_single_equilibrium / extract_result   (Step 2 validated)
    solver_escalation.run_direct_levels / run_bridge                          (Step 3B validated)
    fivefamily.family_of_5                                                    (Step 3B frozen)

Only the persistence layer is new (io_store.py) - see that module for why.

Per shard:
  * ONE process, ONE TC-Python session (Level-2 rebuilds it in place)
  * ONE SQLite file holding BOTH the checkpoint and the raw results, so a state's
    result rows and its SUCCESS flag commit in the same transaction
  * bulk exports (long tables + the wide 5-family table) are built ONCE at the end
  * a composition's 11 temperatures are never split across processes

Usage:
    python 18_pilot2000_run.py --shard 1 [--shards 2] [--max-states N]
"""

import argparse
import json
import sys
import time as time_mod
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1")
sys.path.insert(0, str(PROJECT_ROOT / "code_snapshot"))
import dry_run_lib as lib          # noqa: E402
import solver_escalation as se     # noqa: E402
import fivefamily as ff            # noqa: E402
import io_store                    # noqa: E402

COMP_TBL = PROJECT_ROOT / "01_compositions" / "FULL_V1_COMPOSITIONS.csv"
TEMP_FILE = PROJECT_ROOT / "00_config" / "FULL_V1_TEMPERATURES.csv"
MAX_ATTEMPTS = 3
PROGRESS_EVERY = 1000


# --------------------------------------------------------------------------- #
def load_temperatures(path=TEMP_FILE):
    df = pd.read_csv(path)
    return [(float(r["Temperature_C"]), float(r["Temperature_K"])) for _, r in df.iterrows()]


def make_case(row):
    """Identical to 14_pilot200_run.make_case (Step 3B validated)."""
    xel = (str(row["X_element"]) or "").strip().upper()
    wt = {"CU": float(row["Cu_wt_pct"]), "CR": float(row["Cr_wt_pct"]), "NB": float(row["Nb_wt_pct"])}
    if xel:
        wt[xel] = float(row["X_wt_pct"])
    total = sum(wt.values())
    return {
        "case_id": row["composition_id"], "family": row["family"], "x_element": xel,
        "elements": ["CU", "CR", "NB"] + ([xel] if xel else []),
        "wt_pct": wt, "wt_fraction": {k: v / 100.0 for k, v in wt.items()},
        "elements_string": ",".join(["CU", "CR", "NB"] + ([xel] if xel else [])),
        "composition_sum_wt_pct": total,
        "composition_sum_wt_fraction": total / 100.0,
        "composition_sum_ok": abs(total - 100.0) < 0.01,
        "sampling_domain": row["sampling_domain"],
    }


def agg5(case, state, phases):
    """5-family aggregation - identical to 14_pilot200_run.agg5 (raw names untouched)."""
    groups = {}
    for ph in phases:
        base = ph["phase_model_base_name"]
        fam = ff.family_of_5(base)
        g = groups.setdefault((fam, base), {
            "state_id": state["state_id"], "case_id": case["case_id"],
            "Temperature_C": state["Temperature_C"], "family": fam,
            "phase_model_base_name": base, "n_instances": 0,
            "phase_moles_total": 0.0, "phase_mole_fraction_total": 0.0,
            "phase_volume_fraction_total": 0.0, "phase_mass_fraction_total": 0.0,
        })
        g["n_instances"] += 1
        for k in ("phase_moles_total", "phase_mole_fraction_total",
                  "phase_volume_fraction_total", "phase_mass_fraction_total"):
            v = ph[k.replace("_total", "")]
            if v is not None and v == v:
                g[k] += v
    return list(groups.values())


def neighbor_status(cp, case_id):
    smap = cp.status_map()
    return {int(round(float(tc))): (st == "SUCCESS")
            for (cid, tc), st in smap.items() if cid == case_id}


def failed_state_row(case, tc, tk, state_id, info):
    return {
        "state_id": state_id, "case_id": case["case_id"], "family": case["family"],
        "X_element": case["x_element"] or "", "elements": case["elements_string"],
        "Temperature_C": round(tc, 4), "Temperature_K": round(tk, 4),
        "Pressure_bar": 1.0, "Pressure_Pa": 100000.0, "system_size_mol": 1.0,
        "converged": False, "error_message": str(info["error_message"]),
        "calculation_time_seconds": None, "n_stable_phases": 0,
        "sum_phase_mole_fraction": None, "sum_phase_volume_fraction": None,
        "composition_sum_wt_pct": round(case["composition_sum_wt_pct"], 6),
        "composition_sum_wt_fraction": round(case["composition_sum_wt_pct"] / 100.0, 8),
        "composition_sum_ok": True, "phase_fraction_sum_ok": None,
        "solver_level_used": None, "bridge_used": 0, "bridge_temperatures_C": "",
        "attempt_count": int(info["attempt_count"]),
        "original_error_message": str(info["error_message"]), "final_status": "FAILED",
    }


# --------------------------------------------------------------------------- #
def run_shard(cases, temps, db_path: Path, log_path: Path, logger=None,
              max_states=None, progress_every=PROGRESS_EVERY, progress_csv: Path = None,
              label="shard"):
    """
    Core loop. Returns (stats, store, cp, timing).
    Caller owns export; this function only computes and persists.
    """
    logger = logger or lib.setup_logging(log_path, name=label)
    logger.info("=" * 78)
    logger.info("FULL_V1 %s started - NEW I/O layer (sqlite result store, batched export)", label)
    logger.info("=" * 78)

    cp = lib.CheckpointStore(db_path, logger)
    store = io_store.ResultStore(cp, logger)
    interrupted = cp.resolve_interrupted()
    if interrupted:
        logger.warning("resolved %d RUNNING rows from a previous crash -> INTERRUPTED "
                       "(they will be recomputed)", len(interrupted))

    holder = se.SessionHolder(logger)
    holder.start()

    stats = {"planned": len(cases) * len(temps), "skipped": 0, "level0": 0, "level1": 0,
             "level2": 0, "level3": 0, "failed": 0, "dropped": 0}
    timing = {"t_calc": 0.0, "t_persist": 0.0, "t_progress": 0.0}
    wall_start = time_mod.time()
    done = 0
    try:
        for case in cases:
            for tc, tk in temps:
                state_id = "%s__T%04dC" % (case["case_id"], int(round(tc)))
                info = cp.get_status(state_id)
                if info and info["status"] == "SUCCESS":
                    stats["skipped"] += 1
                    continue
                if info and info["status"] == "FAILED" and info["attempt_count"] >= MAX_ATTEMPTS:
                    stats["dropped"] += 1
                    continue

                cp.mark_running(state_id, case["case_id"], tc)
                t0 = time_mod.perf_counter()
                result, level, first_error = se.run_direct_levels(holder, case, tc, tk,
                                                                  state_id, logger)
                bridges, bridge_used = [], 0
                if not result["state"]["converged"]:
                    result2, ok, bridges, _err = se.run_bridge(
                        holder, case, tc, tk, state_id, logger,
                        neighbor_status(cp, case["case_id"]))
                    if ok:
                        result, level, bridge_used = result2, 3, 1
                    else:
                        level = 3
                t_calc = time_mod.perf_counter() - t0
                timing["t_calc"] += t_calc

                st = result["state"]
                st["solver_level_used"] = level
                st["bridge_used"] = bridge_used
                st["bridge_temperatures_C"] = ";".join(str(int(t)) for t in bridges) if bridges else ""
                st["attempt_count"] = (info["attempt_count"] + 1) if info else 1
                st["original_error_message"] = first_error or ""
                st["final_status"] = "SUCCESS" if st["converged"] else "FAILED"
                st["calculation_time_seconds"] = round(t_calc, 4)

                agg_rows = agg5(case, st, result["phases"]) if st["converged"] else []
                t1 = time_mod.perf_counter()
                store.persist_state(st, result["phases"], result["compositions"], agg_rows,
                                    "SUCCESS" if st["converged"] else "FAILED")
                timing["t_persist"] += time_mod.perf_counter() - t1

                if st["converged"]:
                    stats["level%d" % level] += 1
                else:
                    stats["failed"] += 1
                done += 1

                if progress_every and done % progress_every == 0:
                    t2 = time_mod.perf_counter()
                    if progress_csv:
                        cp.export_csv(progress_csv)
                    el = time_mod.time() - wall_start
                    logger.info("PROGRESS %d/%d done | %.2f s/state | calc %.0f%% | ETA %.1f h",
                                done, stats["planned"] - stats["skipped"], el / done,
                                100 * timing["t_calc"] / max(el, 1e-9),
                                (el / done) * (stats["planned"] - stats["skipped"] - done) / 3600.0)
                    timing["t_progress"] += time_mod.perf_counter() - t2

                if max_states and done >= max_states:
                    raise StopIteration
    except StopIteration:
        logger.info("stopping early: --max-states reached")
    finally:
        holder.shutdown()
    timing["wall"] = time_mod.time() - wall_start
    timing["done"] = done
    return stats, store, cp, timing


def export_shard(store, cp, dirs, prefix, logger, progress_csv=None):
    """Bulk export - ONCE, after the loop. This is where _wide() used to live per-state."""
    t0 = time_mod.perf_counter()
    raw, proc = dirs["raw"], dirs["proc"]
    raw.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    states = store.frame("state_result").sort_values(["case_id", "Temperature_C"])
    phases = store.frame("phase_result")
    pcomp = store.frame("phase_composition_result")
    agg = store.frame("agg_result")

    states.to_csv(raw / (prefix + "EQUILIBRIUM_STATES.csv"), index=False, encoding="utf-8")
    p1 = io_store.write_table(phases, raw / (prefix + "PHASES_LONG"))
    p2 = io_store.write_table(pcomp, raw / (prefix + "PHASE_COMPOSITIONS_LONG"))
    p3 = io_store.write_table(agg, proc / (prefix + "AGGREGATED_PHASES_LONG"))
    wide = io_store.wide_5family(agg, states.state_id.tolist(), ff.FAMILY_ORDER_5)
    wide = wide.merge(states[["state_id", "case_id", "Temperature_C"]], on="state_id", how="left")
    wide.to_csv(proc / (prefix + "AGGREGATED_PHASES.csv"), index=False, encoding="utf-8")
    if progress_csv:
        cp.export_csv(progress_csv)
    dt = time_mod.perf_counter() - t0
    logger.info("export done in %.1f s -> %s | %s | %s", dt, p1.name, p2.name, p3.name)
    return {"export_seconds": round(dt, 2), "n_states": len(states), "n_phase_rows": len(phases),
            "n_phase_composition_rows": len(pcomp), "n_agg_rows": len(agg),
            "phases_file": p1.name, "phase_compositions_file": p2.name, "agg_long_file": p3.name}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=2)   # 4 only after 22b benchmark PASS
    ap.add_argument("--max-states", type=int, default=None)
    ap.add_argument("--comps", type=str, default=str(COMP_TBL))
    ap.add_argument("--outroot", type=str, default=None,
                    help="redirect all outputs under this root (benchmark use only; "
                         "production always uses the FULL_V1 tree)")
    args, _ = ap.parse_known_args()
    n, nsh = args.shard, args.shards

    root = Path(args.outroot) if args.outroot else PROJECT_ROOT
    dirs = {"raw": root / "02_raw", "proc": root / "03_processed",
            "qa": root / "04_qa", "log": root / "05_logs"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    prefix = "FULL_V1_shard%02d_" % n
    logger = lib.setup_logging(dirs["log"] / ("FULL_V1_RUN_shard%02d.log" % n),
                               name="fullv1_s%d" % n)

    comps = pd.read_csv(args.comps, dtype=str).fillna("")
    ids = comps["composition_id"].tolist()
    total = len(ids)
    # Round-robin (stride) split rather than a contiguous block: the composition table is
    # ordered by family, so a block split would put all 300 MO compositions - the only
    # family with a known solver-hard region (Mo @ 950 C) - into one shard and leave the
    # two processes badly unbalanced.  A stride split gives each shard half of every
    # family and domain, while still keeping a composition's 11 temperatures together
    # in one process (spec section 15).
    subset = set(ids[(n - 1)::nsh])
    cases = [make_case(r) for _, r in comps.iterrows() if r["composition_id"] in subset]
    logger.info("shard %d/%d: %d compositions of %d (stride split, families balanced)",
                n, nsh, len(cases), total)
    fam_mix = {}
    for cse in cases:
        fam_mix[cse["family"]] = fam_mix.get(cse["family"], 0) + 1
    logger.info("shard %d family mix: %s", n, fam_mix)
    temps = load_temperatures()
    logger.info("temperature grid: %s", [int(t[0]) for t in temps])

    db = dirs["raw"] / ("calculation_checkpoint_shard%02d.sqlite" % n)
    progress_csv = dirs["raw"] / ("calculation_checkpoint_shard%02d.csv" % n)

    stats, store, cp, timing = run_shard(
        cases, temps, db, dirs["log"] / ("FULL_V1_RUN_shard%02d.log" % n), logger=logger,
        max_states=args.max_states, progress_csv=progress_csv, label="shard %d" % n)

    exp = export_shard(store, cp, dirs, prefix, logger, progress_csv=progress_csv)
    n_ok, n_res, missing = store.consistency()
    if missing:
        logger.error("CONSISTENCY FAIL: %d SUCCESS states without a raw result: %s",
                     len(missing), missing[:10])
    else:
        logger.info("consistency OK: %d SUCCESS checkpoints == %d stored results", n_ok, n_res)

    meta = {
        "shard": n, "shards": nsh, "compositions": len(cases),
        "wall_seconds": round(timing["wall"], 2),
        "thermocalc_seconds": round(timing["t_calc"], 2),
        "persist_seconds": round(timing["t_persist"], 2),
        "progress_snapshot_seconds": round(timing["t_progress"], 2),
        "export_seconds": exp["export_seconds"],
        "io_seconds_total": round(timing["t_persist"] + timing["t_progress"]
                                  + exp["export_seconds"], 2),
        "states_computed": timing["done"],
        "states_per_second": round(timing["done"] / max(timing["wall"], 1e-9), 4),
        "stats": stats, "export": exp,
        "checkpoint_success_count": n_ok, "stored_result_count": n_res,
        "consistency_missing": missing,
        "parquet_available": io_store.have_parquet(),
        "wall_start_utc": time_mod.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time_mod.gmtime(time_mod.time() - timing["wall"])),
    }
    (dirs["log"] / ("FULL_V1_RUNTIME_shard%02d.json" % n)).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    cp.close()
    logger.info("shard %d done: %s", n, stats)
    logger.info("wall %.1f s | Thermo-Calc %.1f s (%.1f%%) | I/O %.1f s (%.1f%%)",
                timing["wall"], timing["t_calc"], 100 * timing["t_calc"] / max(timing["wall"], 1e-9),
                meta["io_seconds_total"], 100 * meta["io_seconds_total"] / max(timing["wall"], 1e-9))
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
