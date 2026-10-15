# -*- coding: utf-8 -*-
"""
solver_escalation.py - Step 3B solver escalation for equilibrium states (Levels 0-3).

Level 0: direct equilibrium calculation (fresh calculation object)
Level 1: rebuild calculation object + retry (same TCPython session)
Level 2: rebuild the independent TC-Python session (stop_api_server + start_api_server)
Level 3: adaptive temperature bridging from the nearest successful grid neighbor
         (bridge temperatures are ONLY used for convergence, never stored as states)

Per-state record fields produced by this module (merged into the state row):
  solver_level_used    0/1/2/3
  bridge_used          0/1
  bridge_temperatures_C   e.g. "900" or "900;925;940;945" (empty for levels 0-2)
  attempt_count         total attempts (counter managed by the caller via checkpoint)
  original_error_message   first error message of the failure chain
"""

import time as time_mod

import fivefamily as ff  # noqa: F401 (available for callers)
import dry_run_lib as lib

# official grid neighbors used by the adaptive bridge
GRID_TEMPERATURES_C = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]
BRIDGE_SCHEDULE = {
    950: {"from_lower": [950],   # [950] means: only the finer list below is used if direct fails
          "lower_steps": [900, 925, 940, 945, 950],
          "upper_steps": [1000, 975, 960, 955, 950]},
}


class SessionHolder:
    """Owns a live TC-Python session; can be destroyed and rebuilt (Level 2)."""

    def __init__(self, logger):
        self.logger = logger
        self.session = None

    def start(self):
        from tc_python import start_api_server
        from tc_python.server import SetUp
        start_api_server()
        self.session = SetUp(False)
        self.logger.info("TC-Python API server started (new)")
        return self.session

    def rebuild(self):
        from tc_python import stop_api_server
        self.logger.info("TC-Python API server REBUILD (stop + start)")
        try:
            stop_api_server()
        except Exception as e:  # noqa: BLE001
            self.logger.warning("stop_api_server warning: %r", e)
        self.session = None
        return self.start()

    def shutdown(self):
        from tc_python import stop_api_server
        self.logger.info("TC-Python API server shutdown (clean exit)")
        try:
            stop_api_server()
        except Exception as e:  # noqa: BLE001
            self.logger.warning("stop_api_server warning: %r", e)
        self.session = None


def _fresh_attempt(session, case, tc, tk, state_id, logger):
    """One direct attempt (Level 0/1/2 body): fresh calculation object in the given session."""
    return lib.run_single_equilibrium(session, case, tc, tk, state_id, logger)


def run_direct_levels(holder: SessionHolder, case, tc, tk, state_id, logger):
    """
    Levels 0 -> 1 -> 2. Returns (result, level_used, first_error).
    level_used = level at which the state SUCCEEDED, or 2 if all failed.
    """
    first_error = None
    for level in (0, 1, 2):
        if level == 2:
            holder.rebuild()
        result = _fresh_attempt(holder.session, case, tc, tk, state_id, logger)
        if result["state"]["converged"]:
            return result, level, first_error
        if first_error is None:
            first_error = result["state"]["error_message"]
        logger.warning("level %d failed for %s: %s", level, state_id, result["state"]["error_message"])
    return result, 2, first_error


def run_bridge(holder: SessionHolder, case, tc, tk, state_id, logger,
               neighbor_status: dict):
    """
    Level 3: adaptive temperature bridging (variant chain, first success wins):
      v1: direct continuation from the nearest successful neighbor  [start, target]
      v2: finer stepping from that neighbor
      v3: direct from the other-side nearest successful neighbor (if exists)
      v4: finer stepping from the other side
    Empirical note (Mini20 Mo-950C): intermediate bridge temperatures (e.g. 925 C) may
    themselves be poorly converging while a direct 2-point continuation succeeds - hence
    the direct variant is tried FIRST. At most 4 variants are attempted (nearest side
    first, then the other side).
    """
    candidates = []
    lower_ok = [t for t in reversed(GRID_TEMPERATURES_C) if t < tc and neighbor_status.get(t)]
    upper_ok = [t for t in GRID_TEMPERATURES_C if t > tc and neighbor_status.get(t)]
    near_lower = lower_ok[:1]      # closest successful neighbor below
    near_upper = upper_ok[-1:] if upper_ok else []   # closest successful neighbor above
    for start in (near_lower + near_upper):
        candidates.append([start, tc])                       # direct (v1 / v3)
    for start in near_lower:
        steps = [_t for _t in (900, 925, 940, 945) if start < _t < tc]
        candidates.append([start] + steps + [tc])            # finer from lower (v2)
    for start in near_upper:
        steps = [_t for _t in (1000, 975, 960, 955) if tc < _t < start]
        candidates.append([start] + steps + [tc])            # finer from upper (v4)

    seen = set()
    for schedule in candidates:
        key = tuple(schedule)
        if key in seen:
            continue
        seen.add(key)
        extracted, bridges_used, err = _bridge_execute_steps(holder, case, tc, tk, state_id,
                                                             logger, schedule)
        if extracted is not None and extracted["state"]["converged"]:
            return extracted, True, bridges_used, None
        logger.warning("bridge variant %s failed for %s (%s)", schedule, state_id,
                       (err or "not converged")[:120])
    return None, False, [], "all bridge variants failed"


def _bridge_execute_steps(holder, case, tc, tk, state_id, logger, schedule):
    """Step through `schedule` (first = start neighbor, last = target) with continuation."""
    bridges_used = []
    try:
        calc = lib.create_calculation(holder.session, case, schedule[0] + 273.15)
        t0 = time_mod.perf_counter()
        r = calc.calculate()  # start neighbor (must converge here)
        if schedule[0] != tc:
            bridges_used.append(schedule[0])
        for t in schedule[1:]:
            calc.set_condition("T", t + 273.15)
            r = calc.calculate()
            if t != tc:
                bridges_used.append(t)
        extracted = lib.extract_result(case, tc, tk, state_id, r, logger, t0=t0)
        if not extracted["state"]["phase_fraction_sum_ok"]:
            extracted["state"]["error_message"] = extracted["state"]["error_message"] or "phase sum off"
        return extracted, bridges_used, None
    except Exception as e:  # noqa: BLE001
        logger.warning("bridge failed for %s: %s", state_id, str(e)[:160])
        return None, bridges_used, "bridge: %s" % str(e)[:160]
