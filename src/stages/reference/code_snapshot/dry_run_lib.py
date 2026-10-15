# -*- coding: utf-8 -*-
"""
dry_run_lib.py - Shared engine for Step 2 / Step 2.5 equilibrium dry runs (Thermo-Calc 2025b + TC-Python + TCCU6).

This module is the single place that:
  * loads the case / temperature configuration (00_config/*.csv),
  * runs one equilibrium per (case, temperature) through TC-Python and extracts ALL stable
    phases with amounts and compositions (identical logic that was validated in Step 2),
  * manages a durable, atomic checkpoint store (SQLite, WAL) to enable resume across runs,
  * writes the unified raw / aggregated outputs incrementally (crash-safe).

Checkpoint design
-----------------
  Store : <out_root>/02_raw/calculation_checkpoint.sqlite  (WAL mode; per-state transactions)
  Table : state_checkpoint(state_id TEXT PRIMARY KEY, case_id, composition_id, Temperature_C,
                           status TEXT, attempt_count INT, start_time TEXT, end_time TEXT,
                           error_message TEXT)
  Status values: PENDING / RUNNING / SUCCESS / FAILED / INTERRUPTED (transient)
  * SUCCESS   -> skipped on resume
  * FAILED    -> retried while attempt_count < MAX_ATTEMPTS, otherwise skipped (recorded)
  * PENDING   -> computed
  * RUNNING   -> at startup converted to INTERRUPTED and recomputed (attempt count preserved)
An auditable CSV snapshot of the table is exported next to the database after every state
(<out_root>/02_raw/calculation_checkpoint.csv, written atomically via tmp+replace).

TC-Python API facts verified on this machine (tc_python 2025.2.30, Thermo-Calc 2025b):
  session.select_database_and_elements("TCCU6", [...]) .get_system() -> System
  system.with_single_equilibrium_calculation(default_conditions=False) -> SingleEquilibriumCalculation
  calc.set_condition("T", K) / ("P", 100000.0) / ("N", 1.0) / ("W(CR)", wf)   (Cu = balance, not set)
  result = calc.calculate()  -> SingleEquilibriumTempResult
  result.get_stable_phases() -> raw stable phase instance names (e.g. 'BCC_B2#2', 'CU4TI1#1')
  result.get_value_of("NPM(...)") / "NP(...)" / "VPV(...)" / "BPW(...)" / "X(phase,el)" / "W(phase,el)"
"""

import argparse
import logging
import math
import os
import re
import sqlite3
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & constants (all centralised here)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(r"E:\Cu_CALPHAD_ML\02_pilot_2000")

DATABASE_NAME = "TCCU6"
PRESSURE_BAR = 1.0
PRESSURE_PA = 100000.0
SYSTEM_SIZE_MOL = 1.0
BALANCE_ELEMENT = "CU"
NON_STABLE_NUMERIC_THRESHOLD = 1e-12
PRESENCE_THRESHOLD = 1e-4
MAX_ATTEMPTS = 2            # retry policy for FAILED states

FAMILY_MAP = {
    "FCC_L12": "FCC",
    "BCC_B2": "BCC",
    "C15_LAVES": "C15",
}

STATE_HEADER = [
    "state_id", "case_id", "family", "X_element", "elements", "Temperature_C", "Temperature_K",
    "Pressure_bar", "Pressure_Pa", "system_size_mol", "converged", "error_message",
    "calculation_time_seconds", "n_stable_phases", "sum_phase_mole_fraction",
    "sum_phase_volume_fraction", "composition_sum_wt_pct", "composition_sum_wt_fraction",
    "composition_sum_ok", "phase_fraction_sum_ok",
]
PHASE_HEADER = [
    "state_id", "case_id", "Temperature_C", "Temperature_K", "phase_index", "raw_phase_name",
    "phase_model_base_name", "phase_moles", "phase_mole_fraction", "phase_volume_fraction",
    "phase_mass_fraction", "is_stable_numeric", "is_present_1e4", "note",
]
COMPOSITION_HEADER = [
    "state_id", "case_id", "Temperature_C", "raw_phase_name", "element",
    "element_mole_fraction", "element_mass_fraction",
]
AGGREGATED_HEADER = [
    "state_id", "case_id", "Temperature_C", "family", "phase_model_base_name", "n_instances",
    "phase_moles_total", "phase_mole_fraction_total", "phase_volume_fraction_total",
    "phase_mass_fraction_total",
]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_file: Path, name="dry_run") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(str(log_file), encoding="utf-8", mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- #
# Configuration loading / validation
# --------------------------------------------------------------------------- #
def load_config(cases_file: Path, temperatures_file: Path):
    """Load case and temperature config CSVs (no fixed-case-count assumption)."""
    cases_df = pd.read_csv(cases_file, dtype=str).fillna("")
    temps_df = pd.read_csv(temperatures_file, dtype=str).fillna("")

    required_case_cols = {"case_id", "family", "X_element", "Cu_wt_pct", "Cr_wt_pct",
                          "Nb_wt_pct", "X_wt_pct", "composition_source", "purpose"}
    missing = required_case_cols - set(cases_df.columns)
    if missing:
        raise ValueError("cases CSV missing columns: %s" % sorted(missing))
    if len(set(cases_df["case_id"])) != len(cases_df):
        raise ValueError("Duplicate case_id in cases CSV")

    if {"Temperature_C", "Temperature_K"} - set(temps_df.columns):
        raise ValueError("temperatures CSV must have columns Temperature_C, Temperature_K")
    temps = []
    for _, row in temps_df.iterrows():
        tc, tk = float(row["Temperature_C"]), float(row["Temperature_K"])
        if abs(tk - (tc + 273.15)) > 1e-9:
            raise ValueError("Temperature_K inconsistent with Temperature_C+273.15: %s/%s" % (tc, tk))
        temps.append((tc, tk))
    return cases_df, temps


def build_case(case_row) -> dict:
    """Resolve one config row into a case dict with per-element wt fractions."""
    wt = OrderedDict()
    wt[BALANCE_ELEMENT] = float(case_row["Cu_wt_pct"])
    wt["CR"] = float(case_row["Cr_wt_pct"])
    wt["NB"] = float(case_row["Nb_wt_pct"])
    x_el = (case_row["X_element"] or "").strip().upper()
    wt_pct_x = case_row["X_wt_pct"].strip()
    if x_el:
        if not wt_pct_x:
            raise ValueError("X_element '%s' given but X_wt_pct empty for %s" % (x_el, case_row["case_id"]))
        wt[x_el] = float(wt_pct_x)
    total = sum(wt.values())
    cu_balance = 100.0 - sum(v for k, v in wt.items() if k != BALANCE_ELEMENT)
    case = {
        "case_id": case_row["case_id"],
        "family": case_row["family"],
        "x_element": x_el,
        "elements": [BALANCE_ELEMENT, "CR", "NB"] + ([x_el] if x_el else []),
        "wt_pct": wt,
        "cu_balance_wt_pct": cu_balance,
        "wt_fraction": {k: v / 100.0 for k, v in wt.items()},
        "composition_source": case_row["composition_source"],
        "purpose": case_row["purpose"],
        "composition_sum_wt_pct": total,
        "composition_sum_ok": abs(total - 100.0) < 0.01,
        "elements_string": ",".join([BALANCE_ELEMENT, "CR", "NB"] + ([x_el] if x_el else [])),
    }
    case["wt_fraction"][BALANCE_ELEMENT] = cu_balance / 100.0
    return case


def validate_cases(cases_df, logger) -> list:
    resolved = [build_case(c) for _, c in cases_df.iterrows()]
    for case in resolved:
        if not case["composition_sum_ok"]:
            raise ValueError("Case %s: wt%% sum = %.4f (expected 100 within 0.01)"
                             % (case["case_id"], case["composition_sum_wt_pct"]))
        logger.info("Case %-14s elements=%-22s wt%%sum=%.4f  OK",
                    case["case_id"], ",".join(case["elements"]), case["composition_sum_wt_pct"])
    return resolved


def save_compositions_csv(cases, path: Path) -> None:
    rows = []
    for case in cases:
        for el in case["elements"]:
            rows.append({
                "case_id": case["case_id"], "family": case["family"],
                "x_element": case["x_element"] or "", "element": el,
                "wt_pct": round(case["wt_pct"][el], 6),
                "wt_fraction": round(case["wt_fraction"][el], 8),
                "is_balance": el == BALANCE_ELEMENT,
            })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


# --------------------------------------------------------------------------- #
# TC-Python equilibrium & extraction (validated in Step 2)
# --------------------------------------------------------------------------- #
def build_calculation(calc, case: dict, temperature_k: float) -> None:
    calc.set_condition("T", float(temperature_k))
    calc.set_condition("P", float(PRESSURE_PA))
    calc.set_condition("N", float(SYSTEM_SIZE_MOL))
    for el in case["elements"]:
        if el == BALANCE_ELEMENT:
            continue
        calc.set_condition("W(%s)" % el, float(case["wt_fraction"][el]))


def get_value_quiet(result, expression: str, default=math.nan):
    try:
        v = float(result.get_value_of(expression))
        return v if math.isfinite(v) else default
    except Exception:  # noqa: BLE001 - one missing quantity must not kill the state
        return default


def get_phase_model_base_name(raw_phase_name: str) -> str:
    return re.sub(r"#\d+$", "", str(raw_phase_name))


def create_calculation(session, case: dict, temperature_k: float):
    """Create a fresh system + calculation with all conditions set (reusable by bridging)."""
    system = session.select_database_and_elements(DATABASE_NAME, case["elements"]).get_system()
    calc = system.with_single_equilibrium_calculation(default_conditions=False)
    build_calculation(calc, case, temperature_k)
    return calc


def extract_result(case: dict, temperature_c: float, temperature_k: float, state_id: str,
                   result, logger: logging.Logger, t0: float = None) -> dict:
    """Extract all stable phases + amounts + compositions from a calculated result."""
    state = {
        "state_id": state_id, "case_id": case["case_id"], "family": case["family"],
        "X_element": case["x_element"] or "", "elements": case["elements_string"],
        "Temperature_C": round(temperature_c, 4), "Temperature_K": round(temperature_k, 4),
        "Pressure_bar": PRESSURE_BAR, "Pressure_Pa": PRESSURE_PA,
        "system_size_mol": SYSTEM_SIZE_MOL, "converged": True, "error_message": "",
        "calculation_time_seconds": round((time.perf_counter() - t0), 4) if t0 else None,
        "n_stable_phases": None,
        "sum_phase_mole_fraction": None, "sum_phase_volume_fraction": None,
        "composition_sum_wt_pct": round(case["composition_sum_wt_pct"], 6),
        "composition_sum_wt_fraction": round(case["composition_sum_wt_pct"] / 100.0, 8),
        "composition_sum_ok": bool(case["composition_sum_ok"]),
        "phase_fraction_sum_ok": None,
    }
    phases, compositions = [], []
    stable_phases = list(result.get_stable_phases())
    components = list(result.get_components())
    state["n_stable_phases"] = len(stable_phases)
    sum_npm, sum_vpv = 0.0, 0.0
    for idx, raw_phase in enumerate(stable_phases, start=1):
        npm = get_value_quiet(result, "NPM(%s)" % raw_phase)
        np_ = get_value_quiet(result, "NP(%s)" % raw_phase)
        vpv = get_value_quiet(result, "VPV(%s)" % raw_phase)
        bpw = get_value_quiet(result, "BPW(%s)" % raw_phase)
        sum_npm += npm if not math.isnan(npm) else 0.0
        sum_vpv += vpv if not math.isnan(vpv) else 0.0
        note = ""
        if math.isnan(npm) or math.isnan(vpv):
            note = "quantity NaN"
        phases.append({
            "state_id": state_id, "case_id": case["case_id"],
            "Temperature_C": round(temperature_c, 4), "Temperature_K": round(temperature_k, 4),
            "phase_index": idx, "raw_phase_name": raw_phase,
            "phase_model_base_name": get_phase_model_base_name(raw_phase),
            "phase_moles": np_, "phase_mole_fraction": npm,
            "phase_volume_fraction": vpv, "phase_mass_fraction": bpw,
            "is_stable_numeric": bool(not math.isnan(npm) and npm > NON_STABLE_NUMERIC_THRESHOLD),
            "is_present_1e4": bool(not math.isnan(npm) and npm > PRESENCE_THRESHOLD),
            "note": note,
        })
        for el in components:
            compositions.append({
                "state_id": state_id, "case_id": case["case_id"],
                "Temperature_C": round(temperature_c, 4), "raw_phase_name": raw_phase,
                "element": el,
                "element_mole_fraction": get_value_quiet(result, "X(%s,%s)" % (raw_phase, el)),
                "element_mass_fraction": get_value_quiet(result, "W(%s,%s)" % (raw_phase, el)),
            })
    state["sum_phase_mole_fraction"] = round(sum_npm, 10)
    state["sum_phase_volume_fraction"] = round(sum_vpv, 10)
    state["phase_fraction_sum_ok"] = bool(abs(sum_npm - 1.0) < 1e-5 and abs(sum_vpv - 1.0) < 1e-3)
    logger.info("  OK  %-14s %-22s T=%5.1fC  phases=%s", case["case_id"], state_id,
                temperature_c, ", ".join("%s(%.4g)" % (p["raw_phase_name"], p["phase_mole_fraction"])
                                         for p in phases))
    if not state["phase_fraction_sum_ok"]:
        state["error_message"] = ("phase fraction sum off: sumNPM=%.6g sumVPV=%.6g"
                                  % (sum_npm, sum_vpv))
    return {"state": state, "phases": phases, "compositions": compositions}


def run_single_equilibrium(session, case: dict, temperature_c: float, temperature_k: float,
                           state_id: str, logger: logging.Logger) -> dict:
    """
    Run one equilibrium; extract all stable phases + amounts + compositions.
    Returns {'state': row, 'phases': [...], 'compositions': [...]}; never raises.
    """
    state = {
        "state_id": state_id, "case_id": case["case_id"], "family": case["family"],
        "X_element": case["x_element"] or "", "elements": case["elements_string"],
        "Temperature_C": round(temperature_c, 4), "Temperature_K": round(temperature_k, 4),
        "Pressure_bar": PRESSURE_BAR, "Pressure_Pa": PRESSURE_PA,
        "system_size_mol": SYSTEM_SIZE_MOL, "converged": False, "error_message": "",
        "calculation_time_seconds": None, "n_stable_phases": 0,
        "sum_phase_mole_fraction": None, "sum_phase_volume_fraction": None,
        "composition_sum_wt_pct": round(case["composition_sum_wt_pct"], 6),
        "composition_sum_wt_fraction": round(case["composition_sum_wt_pct"] / 100.0, 8),
        "composition_sum_ok": bool(case["composition_sum_ok"]),
        "phase_fraction_sum_ok": None,
    }
    phases, compositions = [], []
    t0 = time.perf_counter()
    try:
        result = create_calculation(session, case, temperature_k).calculate()
        extracted = extract_result(case, temperature_c, temperature_k, state_id, result, logger, t0=t0)
        return extracted
    except Exception as e:  # noqa: BLE001 - one failing state must not kill the batch
        state["converged"] = False
        state["error_message"] = "%s: %s" % (type(e).__name__, str(e))
        state["calculation_time_seconds"] = round(time.perf_counter() - t0, 4)
        logger.error("  FAIL %-14s %-22s  %s", case["case_id"], state_id, state["error_message"])
        logger.debug("  detail: %s", traceback.format_exc())
    return {"state": state, "phases": phases, "compositions": compositions}


def aggregate_phase_families(case: dict, state: dict, phases: list) -> list:
    """Aggregated layer: FCC / BCC / C15 / OTHER per state (raw names untouched)."""
    groups = OrderedDict()
    for ph in phases:
        base = ph["phase_model_base_name"]
        family = FAMILY_MAP.get(base, "OTHER")
        g = groups.setdefault((family, base), {
            "state_id": state["state_id"], "case_id": case["case_id"],
            "Temperature_C": state["Temperature_C"], "family": family,
            "phase_model_base_name": base, "n_instances": 0,
            "phase_moles_total": 0.0, "phase_mole_fraction_total": 0.0,
            "phase_volume_fraction_total": 0.0, "phase_mass_fraction_total": 0.0,
        })
        g["n_instances"] += 1
        for k in ("phase_moles_total", "phase_mole_fraction_total",
                  "phase_volume_fraction_total", "phase_mass_fraction_total"):
            v = ph[k.replace("_total", "")]
            if v is not None and not math.isnan(v):
                g[k] += v
    return list(groups.values())


# --------------------------------------------------------------------------- #
# Checkpoint store (SQLite, WAL, atomic per-state transactions)
# --------------------------------------------------------------------------- #
class CheckpointStore:
    """Durable per-state checkpoint store on SQLite (WAL + transactions)."""

    def __init__(self, db_path: Path, logger: logging.Logger):
        self.db_path = Path(db_path)
        self.logger = logger
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS state_checkpoint (
                state_id       TEXT PRIMARY KEY,
                case_id        TEXT NOT NULL,
                composition_id TEXT,
                Temperature_C  REAL NOT NULL,
                status         TEXT NOT NULL,
                attempt_count  INTEGER NOT NULL DEFAULT 0,
                start_time     TEXT,
                end_time       TEXT,
                error_message  TEXT
            )""")
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    # -- status query ------------------------------------------------------- #
    def get_status(self, state_id: str):
        cur = self._conn.execute("SELECT status, attempt_count, error_message FROM state_checkpoint "
                                 "WHERE state_id=?", (state_id,))
        row = cur.fetchone()
        return {"status": row[0], "attempt_count": row[1], "error_message": row[2]} if row else None

    def all_rows(self, states_only=True):
        if states_only:
            return self._conn.execute("SELECT state_id, case_id, Temperature_C, status, "
                                      "attempt_count, start_time, end_time, error_message "
                                      "FROM state_checkpoint ORDER BY case_id, Temperature_C").fetchall()
        return self._conn.execute("SELECT * FROM state_checkpoint ORDER BY case_id, Temperature_C").fetchall()

    def status_map(self):
        """{(case_id, Temperature_C rounded 4): status} for the whole store."""
        cur = self._conn.execute("SELECT case_id, Temperature_C, status FROM state_checkpoint")
        return {(str(r[0]), round(float(r[1]), 4)): r[2] for r in cur.fetchall()}

    # -- transitions (each is a transaction; atomic) ------------------------ #
    def seed_success(self, state_id, case_id, temperature_c, end_time=None):
        self._conn.execute(
            "INSERT OR IGNORE INTO state_checkpoint "
            "(state_id, case_id, composition_id, Temperature_C, status, attempt_count, start_time, end_time, error_message) "
            "VALUES (?,?,?,?,'SUCCESS',1,NULL,?,'seeded from existing states CSV')",
            (state_id, case_id, case_id, float(temperature_c), end_time or time.strftime("%Y-%m-%dT%H:%M:%S")))
        self._conn.commit()

    def mark_running(self, state_id, case_id, temperature_c=0.0):
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._conn.execute("""
            INSERT INTO state_checkpoint (state_id, case_id, composition_id, Temperature_C, status, attempt_count, start_time)
            VALUES (?,?,?,?,'RUNNING',1,?)
            ON CONFLICT(state_id) DO UPDATE SET
                status='RUNNING', start_time=?, attempt_count=attempt_count+1, error_message=NULL""",
                           (state_id, case_id, case_id, float(temperature_c), now, now))
        self._conn.commit()

    def mark_success(self, state_id):
        self._conn.execute("UPDATE state_checkpoint SET status='SUCCESS', end_time=? WHERE state_id=?",
                           (time.strftime("%Y-%m-%dT%H:%M:%S"), state_id))
        self._conn.commit()

    def mark_failed(self, state_id, error_message):
        self._conn.execute("UPDATE state_checkpoint SET status='FAILED', end_time=?, error_message=? "
                           "WHERE state_id=?", (time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                 str(error_message)[:2000], state_id))
        self._conn.commit()

    def resolve_interrupted(self):
        """RUNNING rows from a previous crashed run -> INTERRUPTED (will be recomputed)."""
        cur = self._conn.execute("SELECT state_id FROM state_checkpoint WHERE status='RUNNING'")
        ids = [r[0] for r in cur.fetchall()]
        for sid in ids:
            self._conn.execute("UPDATE state_checkpoint SET status='INTERRUPTED', error_message="
                               "'interrupted by previous run termination' WHERE state_id=?", (sid,))
        self._conn.commit()
        return ids

    def export_csv(self, csv_path: Path):
        """Auditable CSV snapshot, atomically replaced (tmp -> os.replace)."""
        tmp = csv_path.with_suffix(".tmp")
        df = pd.DataFrame(self.all_rows(), columns=["state_id", "case_id", "Temperature_C", "status",
                                                    "attempt_count", "start_time", "end_time",
                                                    "error_message"])
        df.to_csv(tmp, index=False, encoding="utf-8")
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno()) if False else None
        with open(tmp, "rb+") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp), str(csv_path))


# --------------------------------------------------------------------------- #
# Unified output writing (raw + aggregated), incremental (crash-safe)
# --------------------------------------------------------------------------- #
class UnifiedOutput:
    """Holds all rows collected so far and rewrites the unified CSVs after every state.

    `prefix` allows separate runs (e.g. DRY_RUN_* vs MINI20_*) in the same directory layout.
    """

    def __init__(self, out_root: Path, logger: logging.Logger, prefix: str = "DRY_RUN_",
                 subdir: str = ""):
        self.logger = logger
        self.prefix = prefix
        self.dir_raw = out_root / "02_raw" / subdir
        self.dir_processed = out_root / "03_processed" / subdir
        self.dir_compositions = out_root / "01_compositions" / subdir
        for d in (self.dir_raw, self.dir_processed, self.dir_compositions):
            d.mkdir(parents=True, exist_ok=True)
        self.file_states = self.dir_raw / (prefix + "EQUILIBRIUM_STATES.csv")
        self.file_phases = self.dir_raw / (prefix + "PHASES_LONG.csv")
        self.file_comps = self.dir_raw / (prefix + "PHASE_COMPOSITIONS_LONG.csv")
        self.file_agg_long = self.dir_processed / (prefix + "AGGREGATED_PHASES_LONG.csv")
        self.file_agg = self.dir_processed / (prefix + "AGGREGATED_PHASES.csv")
        self.state_rows, self.phase_rows, self.comp_rows, self.agg_rows = [], [], [], []

    def load_existing(self):
        """Load previously saved rows (so unified CSVs merge old + new)."""
        for attr, path in (("state_rows", self.file_states), ("phase_rows", self.file_phases),
                           ("comp_rows", self.file_comps), ("agg_rows", self.file_agg_long)):
            if path.exists():
                df = pd.read_csv(path, keep_default_na=False)
                setattr(self, attr, df.to_dict("records"))
                self.logger.debug("Loaded %d existing rows from %s", len(getattr(self, attr)), path.name)

    def add_state(self, result: dict):
        self.state_rows.append(result["state"])
        self.phase_rows.extend(result["phases"])
        self.comp_rows.extend(result["compositions"])

    def add_aggregated(self, agg: list):
        self.agg_rows.extend(agg)

    def save(self):
        pd.DataFrame(self.state_rows, columns=STATE_HEADER).to_csv(self.file_states, index=False, encoding="utf-8")
        pd.DataFrame(self.phase_rows, columns=PHASE_HEADER).to_csv(self.file_phases, index=False, encoding="utf-8")
        pd.DataFrame(self.comp_rows, columns=COMPOSITION_HEADER).to_csv(self.file_comps, index=False, encoding="utf-8")
        if self.agg_rows:
            pd.DataFrame(self.agg_rows, columns=AGGREGATED_HEADER).to_csv(self.file_agg_long, index=False, encoding="utf-8")
        self.save_wide()
        self.logger.debug("Incremental save: %d states, %d phases, %d comps",
                          len(self.state_rows), len(self.phase_rows), len(self.comp_rows))

    def save_wide(self):
        if not self.agg_rows:
            return
        agg_df = pd.DataFrame(self.agg_rows)
        wide = []
        seen = OrderedDict()
        for st in self.state_rows:
            sub = agg_df[agg_df["state_id"] == st["state_id"]]
            row = {"state_id": st["state_id"], "case_id": st["case_id"],
                   "Temperature_C": st["Temperature_C"],
                   "FCC_total_mole_fraction": 0.0, "FCC_total_volume_fraction": 0.0,
                   "BCC_total_mole_fraction": 0.0, "BCC_total_volume_fraction": 0.0,
                   "C15_total_mole_fraction": 0.0, "C15_total_volume_fraction": 0.0,
                   "OTHER_total_mole_fraction": 0.0, "OTHER_total_volume_fraction": 0.0,
                   "OTHER_phase_bases": ""}
            for _, g in sub.iterrows():
                fam = g["family"]
                if fam in ("FCC", "BCC", "C15"):
                    row["%s_total_mole_fraction" % fam] = float(g["phase_mole_fraction_total"])
                    row["%s_total_volume_fraction" % fam] = float(g["phase_volume_fraction_total"])
                elif fam == "OTHER":
                    row["OTHER_total_mole_fraction"] += float(g["phase_mole_fraction_total"])
                    row["OTHER_total_volume_fraction"] += float(g["phase_volume_fraction_total"])
                    bases = row["OTHER_phase_bases"].split(";") if row["OTHER_phase_bases"] else []
                    if g["phase_model_base_name"] not in bases:
                        bases.append(g["phase_model_base_name"])
                    row["OTHER_phase_bases"] = ";".join(bases)
            seen[st["state_id"]] = row
        wide = [v for _, v in seen.items()]
        pd.DataFrame(wide).to_csv(self.file_agg, index=False, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main runner with checkpoint / resume
# --------------------------------------------------------------------------- #
def seed_checkpoint_from_states_csv(cp: CheckpointStore, states_csv: Path):
    """Seed SUCCESS rows for states that finished in a previous run (resume capability)."""
    if not states_csv.exists():
        return 0
    df = pd.read_csv(states_csv, keep_default_na=False)
    n = 0
    for _, row in df.iterrows():
        if str(row["converged"]).lower() == "true":
            cp.seed_success(str(row["state_id"]), str(row["case_id"]), float(row["Temperature_C"]))
            n += 1
    return n


def run_all(session, cases, temps, out_root: Path, cp: CheckpointStore, logger,
            max_states: int = None, simulate_crash_after: int = None) -> dict:
    """
    Run all (case x temperature) states respecting the checkpoint store.
    `session` is the opened TC-Python session (SetUp object).
    Returns summary dict; `simulate_crash_after` is only used by the resume test
    (it intentionally aborts the process with os._exit after N executed states,
    leaving one RUNNING checkpoint row behind to emulate a hard crash).
    """
    output = UnifiedOutput(out_root, logger)
    output.load_existing()

    interrupted = cp.resolve_interrupted()
    if interrupted:
        logger.warning("Recovered %d INTERRUPTED state(s) from previous run: %s",
                       len(interrupted), interrupted)

    planned = []
    for case in cases:
        for temperature_c, temperature_k in temps:
            planned.append((case, temperature_c, temperature_k,
                            "%s__T%04dC" % (case["case_id"], int(round(temperature_c)))))

    stats = {"planned": len(planned), "skipped": 0, "computed": 0, "failed": 0, "dropped": 0,
             "interrupted_recovered": len(interrupted)}
    executed = 0
    for case, temperature_c, temperature_k, state_id in planned:
        info = cp.get_status(state_id)
        if info and info["status"] == "SUCCESS":
            stats["skipped"] += 1
            logger.info("  SKIP %-14s %-22s (already SUCCESS)", case["case_id"], state_id)
            continue
        if info and info["status"] == "FAILED" and info["attempt_count"] >= MAX_ATTEMPTS:
            stats["dropped"] += 1
            logger.error("  DROP %-14s %-22s (FAILED after %d attempts: %s)",
                         case["case_id"], state_id, info["attempt_count"], info["error_message"])
            continue

        cp.mark_running(state_id, case["case_id"], temperature_c)
        if simulate_crash_after is not None and executed >= simulate_crash_after:
            # deterministic crash simulation: the current state stays RUNNING in the DB
            logger.critical("SIMULATED CRASH after %d executed states (state %s left RUNNING)",
                            executed, state_id)
            output.save()
            cp.export_csv(out_root / "02_raw" / "calculation_checkpoint.csv")
            os._exit(1)

        result = run_single_equilibrium(session, case, temperature_c, temperature_k,
                                        state_id, logger)
        executed += 1
        if result["state"]["converged"]:
            cp.mark_success(state_id)
            stats["computed"] += 1
            output.add_state(result)
            output.add_aggregated(aggregate_phase_families(case, result["state"], result["phases"]))
        else:
            cp.mark_failed(state_id, result["state"]["error_message"])
            stats["failed"] += 1
        output.save()
        cp.export_csv(out_root / "02_raw" / "calculation_checkpoint.csv")
        if max_states is not None and stats["computed"] >= max_states:
            logger.info("Reached --max-states=%d; stopping cleanly.", max_states)
            break
    return stats
