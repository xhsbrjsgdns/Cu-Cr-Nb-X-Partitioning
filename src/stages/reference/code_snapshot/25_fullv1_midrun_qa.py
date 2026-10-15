# -*- coding: utf-8 -*-
"""
25_fullv1_midrun_qa.py - STEP 4 Stage I: read-only MIDRUN QA SNAPSHOT.

Safe to run at any time while the shards are running: it opens each shard's SQLite
store READ-ONLY (mode=ro) and recomputes nothing. It never touches the checkpoint,
never re-runs a state, and never writes into 02_raw / 03_processed.

Checks (spec sections 40-41):
    convergence so far, failure clustering, phase vocabulary / new phases,
    checkpoint vs raw-result consistency, Ni occurrence, temperature completeness,
    partition conservation (on a sample), bridge-temperature leakage.

STOP conditions (spec 41) are evaluated explicitly; if any fires the script prints
MIDRUN_STOP and writes MIDRUN_FAILURE_INVESTIGATION.md.

Usage:  python 25_fullv1_midrun_qa.py [--sample 20000]
"""

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1")
sys.path.insert(0, str(PROJECT_ROOT / "code_snapshot"))
import fivefamily as ff        # noqa: E402
import io_store                # noqa: E402

DIR_RAW = PROJECT_ROOT / "02_raw"
DIR_QA = PROJECT_ROOT / "04_qa"
TEMPS = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]
N_STATES = 550000
NI_PAT = r"(?i)(?<![a-z])ni(?![0-9a-z])"
KNOWN_HARD = ("MO", 950.0)


def shard_dbs():
    return sorted(DIR_RAW.glob("calculation_checkpoint_shard*.sqlite"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000,
                    help="states to sample for the conservation check (0 = all)")
    args, _ = ap.parse_known_args()
    dbs = shard_dbs()
    if not dbs:
        print("no shard store yet under %s" % DIR_RAW)
        return 0

    cps, states, phases, stop, warn = [], [], [], [], []
    for db in dbs:
        n = int(db.stem.split("shard")[-1])
        # ONE deferred read transaction => all three reads see the SAME WAL snapshot.
        # Without this, reading a live store gives each query its own snapshot, and states
        # the shard commits between two queries appear in the later table but not the
        # earlier one - a pure read-skew artefact that looks like an inconsistency.
        con = sqlite3.connect("file:%s?mode=ro" % db.as_posix(), uri=True,
                              isolation_level=None)
        try:
            con.execute("BEGIN DEFERRED")
            cp = pd.read_sql_query("SELECT state_id, case_id, Temperature_C, status, "
                                   "error_message FROM state_checkpoint", con)
            st = pd.read_sql_query("SELECT state_id, case_id, family, Temperature_C, converged, "
                                   "solver_level_used, bridge_used, bridge_temperatures_C, "
                                   "error_message FROM state_result", con)
            ph = pd.read_sql_query("SELECT DISTINCT phase_model_base_name, raw_phase_name "
                                   "FROM phase_result", con)
            con.execute("COMMIT")
        finally:
            con.close()
        for d in (cp, st, ph):
            d["shard"] = n
        cps.append(cp); states.append(st); phases.append(ph)
    cp = pd.concat(cps, ignore_index=True)
    st = pd.concat(states, ignore_index=True)
    ph = pd.concat(phases, ignore_index=True)
    st["ok"] = st["converged"].astype(str).str.lower() == "true"

    done = len(cp[cp.status.isin(["SUCCESS", "FAILED"])])
    n_ok = int(st.ok.sum())
    n_fail = int((~st.ok).sum())
    rate = 100.0 * n_ok / max(done, 1)

    # --- consistency: SUCCESS without raw result, and converged raw without SUCCESS --
    ok_cp = set(cp.loc[cp.status == "SUCCESS", "state_id"])
    ok_raw = set(st.loc[st.ok, "state_id"])
    miss_a = sorted(ok_cp - ok_raw)[:20]
    miss_b = sorted(ok_raw - ok_cp)[:20]
    # SUCCESS-without-result is structurally impossible (result rows and the SUCCESS flag
    # commit in ONE transaction), so any occurrence is a hard STOP.
    if miss_a:
        stop.append("checkpoint/raw inconsistency: %d SUCCESS checkpoints have no raw result "
                    "(structurally impossible - investigate) %s"
                    % (len(ok_cp - ok_raw), miss_a[:5]))
    # The other direction can only come from read skew, which the single snapshot above
    # removes; if it still shows up it is worth a warning, not a stop.
    if miss_b:
        warn.append("%d converged raw results whose checkpoint is not yet SUCCESS "
                    "(expected 0 with the snapshot read; harmless if the run is live) %s"
                    % (len(ok_raw - ok_cp), miss_b[:5]))

    # --- Ni ------------------------------------------------------------------- #
    ni_hits = [c for c in st.columns
               if st[c].astype(str).str.contains(NI_PAT, regex=True).any()]
    ni_hits += [c for c in ph.columns
                if ph[c].astype(str).str.contains(NI_PAT, regex=True).any()]
    if ni_hits:
        stop.append("Ni/NI found in columns %s" % ni_hits)

    # --- temperature leakage --------------------------------------------------- #
    tset = sorted({int(round(float(t))) for t in st.Temperature_C.dropna()})
    leaked = [t for t in tset if t not in TEMPS]
    if leaked:
        stop.append("bridge temperature leaked into states: %s" % leaked)

    # --- failure clustering ---------------------------------------------------- #
    fails = st[~st.ok]
    by_ft = (fails.groupby(["family", "Temperature_C"]).size().sort_values(ascending=False)
             if len(fails) else pd.Series(dtype=int))
    unknown = {k: int(v) for k, v in by_ft.items() if (k[0], float(k[1])) != KNOWN_HARD}
    if sum(unknown.values()) > max(20, 0.001 * max(done, 1)):
        stop.append("failure cluster outside the frozen Mo@950 region: %s"
                    % dict(list(unknown.items())[:8]))

    # --- phase vocabulary ------------------------------------------------------ #
    bases = sorted(ph.phase_model_base_name.dropna().unique())
    known = {"BCC_B2", "C15_LAVES", "CR3SI", "CU4TI1", "FCC_L12", "LIQUID",
             "MU_PHASE", "SIGMA", "ZRM5_C15B"}
    new_bases = [b for b in bases if b not in known]

    # --- conservation on a sample --------------------------------------------- #
    cons = {}
    if n_ok:
        # A shared, ORDERED CTE of state_ids drives both fetches. Applying LIMIT to each
        # query separately (the earlier version) selected DIFFERENT state sets - the first
        # N phase rows cover far fewer states than the first N distinct state_ids - leaving
        # phase-composition rows with no phase_moles to join to. Their element totals then
        # came out 0 and the partition sum read as 0 instead of 1, i.e. a spurious max
        # deviation of exactly 1.0 on data that is in fact conserved to ~1e-16.
        con = sqlite3.connect("file:%s?mode=ro" % dbs[0].as_posix(), uri=True,
                              isolation_level=None)
        try:
            con.execute("BEGIN DEFERRED")
            cte = ("WITH s AS (SELECT state_id FROM state_result WHERE converged='True' "
                   "ORDER BY state_id%s) "
                   % (" LIMIT %d" % args.sample if args.sample else ""))
            pr = pd.read_sql_query(
                cte + "SELECT p.state_id, p.raw_phase_name, p.phase_moles "
                      "FROM phase_result p JOIN s ON p.state_id = s.state_id", con)
            pc = pd.read_sql_query(
                cte + "SELECT c.state_id, c.raw_phase_name, c.element, c.element_mole_fraction "
                      "FROM phase_composition_result c JOIN s ON c.state_id = s.state_id", con)
            con.execute("COMMIT")
        finally:
            con.close()
        pr = io_store._retype(pr); pc = io_store._retype(pc)
        m = pc.merge(pr, on=["state_id", "raw_phase_name"], how="left")
        cons["sampled_states"] = int(pr.state_id.nunique())
        cons["orphan_composition_rows"] = int(m.phase_moles.isna().sum())
        if cons["orphan_composition_rows"]:
            warn.append("%d sampled phase-composition rows had no matching phase row; the "
                        "conservation number below is not trustworthy"
                        % cons["orphan_composition_rows"])
        m["n_i"] = m.phase_moles * m.element_mole_fraction
        tot = m.groupby(["state_id", "element"])["n_i"].transform("sum")
        p = (m.n_i / tot).where(tot > 0)
        g = p.groupby([m.state_id, m.element]).sum()
        cons["partition_sum_max_dev"] = float((g - 1).abs().max())
        xs = pc.groupby(["state_id", "raw_phase_name"])["element_mole_fraction"].sum()
        cons["phase_composition_sum_max_dev"] = float((xs - 1).abs().max())
        if cons["partition_sum_max_dev"] > 1e-6:
            stop.append("partition conservation broken: max dev %.3e"
                        % cons["partition_sum_max_dev"])

    # --- per-composition temperature completeness (finished compositions only) -- #
    fin = cp[cp.status.isin(["SUCCESS", "FAILED"])]
    got = fin.groupby("case_id")["Temperature_C"].nunique()
    partial = int((got < 11).sum())

    snap = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shards": [int(d.stem.split("shard")[-1]) for d in dbs],
        "states_done": done, "of_planned": N_STATES,
        "progress_pct": round(100.0 * done / N_STATES, 3),
        "converged": n_ok, "failed": n_fail, "convergence_rate_so_far": round(rate, 4),
        "failures_by_family": fails.groupby("family").size().to_dict() if len(fails) else {},
        "failures_by_temperature": (fails.groupby("Temperature_C").size().to_dict()
                                    if len(fails) else {}),
        "failures_outside_Mo950": unknown,
        "checkpoint_status_counts": cp.status.value_counts().to_dict(),
        "success_without_raw_result": len(ok_cp - ok_raw),
        "raw_result_without_success": len(ok_raw - ok_cp),
        "phase_bases_seen": bases, "new_phase_bases_vs_pilot2000": new_bases,
        "temperatures_seen": tset, "leaked_temperatures": leaked,
        "compositions_in_progress_partial": partial,
        "conservation_sample": cons,
        "ni_occurrences": len(ni_hits),
        "solver_levels": st.solver_level_used.value_counts(dropna=False).to_dict(),
        "bridge_used": int(pd.to_numeric(st.bridge_used, errors="coerce").fillna(0).sum()),
        "stop_conditions": stop, "warnings": warn,
    }
    DIR_QA.mkdir(parents=True, exist_ok=True)
    out = DIR_QA / "FULL_V1_MIDRUN_QA_SNAPSHOT.json"
    out.write_text(json.dumps(snap, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("MIDRUN SNAPSHOT  %s" % snap["timestamp_utc"])
    print("  progress      : %d / %d states (%.2f%%)" % (done, N_STATES, snap["progress_pct"]))
    print("  convergence   : %d ok / %d failed = %.3f%%" % (n_ok, n_fail, rate))
    print("  failures      : %s" % snap["failures_by_family"])
    print("  outside Mo@950: %s" % (unknown or "none"))
    print("  phase bases   : %s" % bases)
    print("  new vs Pilot  : %s" % (new_bases or "none"))
    print("  cp/raw consist: SUCCESS-no-result=%d, result-no-SUCCESS=%d"
          % (snap["success_without_raw_result"], snap["raw_result_without_success"]))
    print("  Ni occurrences: %d" % len(ni_hits))
    print("  conservation  : %s" % cons)
    if new_bases:
        (DIR_QA / "FULL_V1_NEW_PHASE_DISCOVERY_MIDRUN.md").write_text(
            "# NEW_PHASE_DISCOVERED (midrun)\n\n"
            "新出现的 phase base（相对 Pilot2000 的 9 个）：\n\n"
            + "\n".join("- `%s` → 默认归入 OTHER_SOLID（规范 §42：不删除、raw 完整保存、"
                        "不立刻改核心 5-family）" % b for b in new_bases) + "\n",
            encoding="utf-8")
    if warn:
        print()
        for w in warn:
            print("  WARN: %s" % w)
    if stop:
        (DIR_QA / "MIDRUN_FAILURE_INVESTIGATION.md").write_text(
            "# MIDRUN_FAILURE_INVESTIGATION.md\n\n**触发时间**：%s\n\n"
            "以下 STOP 条件被触发（规范 §41），应暂停新的计算并调查：\n\n"
            % snap["timestamp_utc"]
            + "\n".join("%d. %s" % (i, s) for i, s in enumerate(stop, 1))
            + "\n\n快照：`FULL_V1_MIDRUN_QA_SNAPSHOT.json`\n", encoding="utf-8")
        print()
        print("MIDRUN_STOP: %s" % " | ".join(stop))
        return 2
    print()
    print("MIDRUN_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
