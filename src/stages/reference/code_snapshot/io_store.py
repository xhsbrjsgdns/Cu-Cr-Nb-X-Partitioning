# -*- coding: utf-8 -*-
"""
io_store.py - Step 3C I/O performance layer for the Pilot2000 runner.

WHY THIS EXISTS
---------------
Pilot200 measured a hard O(N^2) wall-time defect: 14_pilot200_run.py called save()
after every state, and save() rewrote all four growing CSVs plus an O(n*m) _wide().
Measured per-state wall was  0.36 + 3.0e-3 * N  seconds (N = states already done);
the intercept is Thermo-Calc, the slope is pure I/O.  At 11000 states/shard that is
~50 h/shard, ~98% of it wasted.

WHAT THIS CHANGES
-----------------
Nothing about the physics.  dry_run_lib / solver_escalation / fivefamily are reused
untouched.  Only the persistence layer changes:

  * the canonical durable store is SQLite (same file as the checkpoint),
    written with ONE transaction per state -> O(1) per state, no growth;
  * the result rows and the checkpoint status flip commit in the SAME transaction,
    so "checkpoint says SUCCESS but the raw result is missing" is structurally
    impossible (spec section 2.E is satisfied by construction, not by a later check);
  * the wide 5-family table and all bulk exports are built ONCE, after the shard
    finishes (or in finally), via a single pivot_table - never inside the loop.

Crash/resume semantics are strictly stronger than Pilot200's:
  - a state is SUCCESS only if its result rows are already committed;
  - a crash mid-state leaves status RUNNING -> resolve_interrupted() -> recomputed;
  - no completed state is ever lost or duplicated (state_id is the primary key).
"""

import gzip
import json
import math
import sqlite3
import time
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1

# Column schemas mirror dry_run_lib.extract_result() + the runner's solver metadata.
STATE_COLS = [
    "state_id", "case_id", "family", "X_element", "elements",
    "Temperature_C", "Temperature_K", "Pressure_bar", "Pressure_Pa", "system_size_mol",
    "converged", "error_message", "calculation_time_seconds", "n_stable_phases",
    "sum_phase_mole_fraction", "sum_phase_volume_fraction",
    "composition_sum_wt_pct", "composition_sum_wt_fraction", "composition_sum_ok",
    "phase_fraction_sum_ok",
    "solver_level_used", "bridge_used", "bridge_temperatures_C", "attempt_count",
    "original_error_message", "final_status",
]
PHASE_COLS = [
    "state_id", "case_id", "Temperature_C", "Temperature_K", "phase_index",
    "raw_phase_name", "phase_model_base_name", "phase_moles", "phase_mole_fraction",
    "phase_volume_fraction", "phase_mass_fraction", "is_stable_numeric",
    "is_present_1e4", "note",
]
PCOMP_COLS = [
    "state_id", "case_id", "Temperature_C", "raw_phase_name", "element",
    "element_mole_fraction", "element_mass_fraction",
]
AGG_COLS = [
    "state_id", "case_id", "Temperature_C", "family", "phase_model_base_name",
    "n_instances", "phase_moles_total", "phase_mole_fraction_total",
    "phase_volume_fraction_total", "phase_mass_fraction_total",
]
TABLES = {
    "state_result": (STATE_COLS, "state_id"),
    "phase_result": (PHASE_COLS, None),
    "phase_composition_result": (PCOMP_COLS, None),
    "agg_result": (AGG_COLS, None),
}


def _q(name):
    return '"%s"' % str(name).replace('"', '""')


class ResultStore:
    """
    Durable per-state result store sharing the checkpoint's SQLite connection.

    Sharing the connection is deliberate: it makes the result INSERTs and the
    checkpoint UPDATE a single atomic transaction.  Do not give this its own
    connection - that would reintroduce the "success without result" window.
    """

    def __init__(self, checkpoint_store, logger):
        self.cp = checkpoint_store
        self.logger = logger
        self.conn = checkpoint_store._conn          # intentional: shared transaction scope
        self._ensure_schema()

    # ---------------------------------------------------------------- schema -- #
    def _ensure_schema(self):
        c = self.conn
        for table, (cols, pk) in TABLES.items():
            # NOTE: columns are declared with NO type, i.e. BLOB affinity, so SQLite
            # stores each value in its natural storage class. Declaring them TEXT would
            # round-trip floats through %!.15g and lose the last ULP.
            defs = []
            for col in cols:
                d = _q(col)
                if pk and col == pk:
                    d = "%s PRIMARY KEY" % _q(col)
                defs.append(d)
            c.execute("CREATE TABLE IF NOT EXISTS %s (%s)" % (table, ", ".join(defs)))
        c.execute("CREATE INDEX IF NOT EXISTS ix_phase_state ON phase_result(state_id)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_pcomp_state ON phase_composition_result(state_id)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_agg_state ON agg_result(state_id)")
        c.execute("CREATE TABLE IF NOT EXISTS store_meta (k TEXT PRIMARY KEY, v TEXT)")
        c.execute("INSERT OR REPLACE INTO store_meta (k, v) VALUES ('schema_version', ?)",
                  (str(SCHEMA_VERSION),))
        c.commit()

    def _widen(self, table, keys):
        """Add any column the caller supplies that the table does not have yet."""
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(%s)" % table)}
        for k in keys:
            if k not in have:
                self.conn.execute("ALTER TABLE %s ADD COLUMN %s" % (table, _q(k)))
                self.logger.warning("result store: new column %s.%s (schema drift)", table, k)

    # ----------------------------------------------------------------- write -- #
    @staticmethod
    def _val(v):
        if v is None:
            return None
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, bool):
            return "True" if v else "False"
        return v

    def _insert(self, table, rows, replace=True):
        if not rows:
            return
        keys = list(rows[0].keys())
        self._widen(table, keys)
        verb = "INSERT OR REPLACE INTO" if replace else "INSERT INTO"
        sql = "%s %s (%s) VALUES (%s)" % (verb, table, ", ".join(_q(k) for k in keys),
                                          ", ".join("?" * len(keys)))
        self.conn.executemany(sql, [tuple(self._val(r.get(k)) for k in keys) for r in rows])

    def persist_state(self, state, phases, compositions, agg_rows, mark):
        """
        Persist one state's complete result AND its checkpoint status in ONE transaction.

        mark: 'SUCCESS' or 'FAILED'.  Returns only after COMMIT, so once this returns
        the state is durable; if the process dies inside it, nothing is committed and
        the state stays RUNNING -> INTERRUPTED -> recomputed on resume.
        """
        sid = state["state_id"]
        try:
            # replace any partial rows from an interrupted earlier attempt
            for t in ("phase_result", "phase_composition_result", "agg_result"):
                self.conn.execute("DELETE FROM %s WHERE state_id=?" % t, (sid,))
            self._insert("state_result", [state])
            self._insert("phase_result", phases, replace=False)
            self._insert("phase_composition_result", compositions, replace=False)
            self._insert("agg_result", agg_rows, replace=False)
            if mark == "SUCCESS":
                self.conn.execute(
                    "UPDATE state_checkpoint SET status='SUCCESS', end_time=? WHERE state_id=?",
                    (time.strftime("%Y-%m-%dT%H:%M:%S"), sid))
            else:
                self.conn.execute(
                    "UPDATE state_checkpoint SET status='FAILED', end_time=?, error_message=? "
                    "WHERE state_id=?",
                    (time.strftime("%Y-%m-%dT%H:%M:%S"),
                     str(state.get("error_message", ""))[:2000], sid))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------ consistency -- #
    def consistency(self):
        """(n_success_checkpoint, n_state_result, missing_ids) - spec section 2.E."""
        cur = self.conn.execute("SELECT state_id FROM state_checkpoint WHERE status='SUCCESS'")
        ok_ids = {r[0] for r in cur}
        cur = self.conn.execute("SELECT state_id FROM state_result WHERE converged='True'")
        res_ids = {r[0] for r in cur}
        return len(ok_ids), len(res_ids), sorted(ok_ids - res_ids)

    # ------------------------------------------------------------------ read -- #
    def frame(self, table):
        df = pd.read_sql_query("SELECT * FROM %s" % table, self.conn)
        return _retype(df)


NUMERIC_HINT = ("Temperature_", "Pressure_", "system_size_mol", "phase_", "sum_", "n_",
                "composition_sum_wt", "calculation_time", "element_mo", "element_ma",
                "solver_level_used", "bridge_used", "attempt_count", "is_")
BOOL_COLS = {"converged", "composition_sum_ok", "phase_fraction_sum_ok",
             "is_stable_numeric", "is_present_1e4"}


def _retype(df):
    for c in df.columns:
        if c in BOOL_COLS:
            continue
        if any(c.startswith(h) or h in c for h in NUMERIC_HINT):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().any() or df[c].isna().all():
                df[c] = conv
    return df


# --------------------------------------------------------------------------- #
# Bulk export - called ONCE per shard (never inside the calculation loop)
# --------------------------------------------------------------------------- #
def have_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def write_table(df, path_noext: Path, prefer_parquet=True):
    """Write a long table once. Parquet when available, else gzipped CSV. Returns the path."""
    path_noext.parent.mkdir(parents=True, exist_ok=True)
    if prefer_parquet and have_parquet():
        p = path_noext.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    p = path_noext.with_suffix(".csv.gz")
    with gzip.open(p, "wt", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)
    return p


def read_table(path_noext: Path):
    """Read back whichever form write_table produced."""
    for suf, fn in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv), (".csv", pd.read_csv)):
        p = path_noext.with_suffix(suf)
        if p.exists():
            return fn(p)
    raise FileNotFoundError("no table at %s(.parquet|.csv.gz|.csv)" % path_noext)


def wide_5family(agg_long: pd.DataFrame, state_ids, family_order):
    """
    The O(n*m) `_wide()` of 14_pilot200_run.py, replaced by ONE pivot_table.
    Called once per shard after the loop - this is the single biggest win.
    """
    out = pd.DataFrame({"state_id": list(state_ids)})
    if len(agg_long):
        pv = agg_long.pivot_table(index="state_id", columns="family",
                                  values=["phase_mole_fraction_total",
                                          "phase_volume_fraction_total"],
                                  aggfunc="sum", fill_value=0.0)
        cnt = agg_long.pivot_table(index="state_id", columns="family",
                                   values="n_instances", aggfunc="sum", fill_value=0)
    else:
        pv = cnt = None
    for fam in family_order:
        for src, dst in (("phase_mole_fraction_total", "%s_total_mole_fraction" % fam),
                         ("phase_volume_fraction_total", "%s_total_volume_fraction" % fam)):
            if pv is not None and (src, fam) in pv.columns:
                out[dst] = out.state_id.map(pv[(src, fam)]).fillna(0.0)
            else:
                out[dst] = 0.0
        if cnt is not None and fam in cnt.columns:
            out["%s_n_instances" % fam] = out.state_id.map(cnt[fam]).fillna(0).astype(int)
        else:
            out["%s_n_instances" % fam] = 0
    return out
