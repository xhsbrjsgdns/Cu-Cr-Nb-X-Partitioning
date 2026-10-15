# -*- coding: utf-8 -*-
"""
run_r3.py
STEP5K-R3: execute the frozen source-specific requests. POST-FREEZE, NON-Ni.

This is the earlier runner with five additions and no behavioural change otherwise:
  * it refuses to start unless the engine smoke test has passed, so the 120 states
    can only follow a proven engine, never precede one;
  * it stamps 07_calphad_raw/ENGINE_PROVENANCE.json with the live engine's identity,
    which the downstream stages require - a checkpoint produced by anything other
    than this runner on a real engine cannot enter the scientific tables;
  * the 471 protected artifacts are re-hashed before the first calculation and again
    at the end, so the run cannot silently coexist with a change to a frozen result;
  * `--limit N` runs only the first N pending requests, so the engine can be proved
    on a handful of states before 120 are committed;
  * the solver level that actually succeeded is recorded on every state.

It resumes from its own checkpoint: requests already present in STATES.csv are
skipped, so an interrupted run continues instead of restarting.

Nothing here trains, tunes, or predicts. Every state is an equilibrium calculation at
an exact literature composition and an exact experimental temperature.
"""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
import pandas as pd

O = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(O / "code_snapshot"))
sys.path.insert(0, r"E:/Cu_CALPHAD_ML/02_pilot_2000/code")

from dry_run_lib import (create_calculation, extract_result, setup_logging,  # noqa
                         PHASE_HEADER, COMPOSITION_HEADER)
from fivefamily import family_of_5                                # noqa: E402
from tc_python import TCPython, SingleEquilibriumOptions          # noqa: E402
import tc_python                                                  # noqa: E402

ALLOWED = {"CU", "CR", "NB", "AG", "MO", "CO", "SI", "TI"}
SMOKE = "14_qa/STEP5K_R3_ENGINE_SMOKE_TEST.json"
PROVENANCE = "07_calphad_raw/ENGINE_PROVENANCE.json"
# deterministic, pre-declared escalation: (max iterations, global grid points)
ESCALATION = [(500, 2000), (2000, 10000), (5000, 30000), (10000, 100000)]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_protected(tag):
    rows = json.loads((O / "00_config" / "protected_before.json").read_text(
        encoding="utf-8"))
    bad = [r["path"] for r in rows
           if (not os.path.exists(r["path"])) or sha(r["path"]) != r["sha256"]]
    if bad:
        raise SystemExit("FATAL (%s): %d protected artifacts changed, first %s. R3 "
                         "must not alter any frozen Ni, model, OOD, recovery or "
                         "screening result." % (tag, len(bad), bad[:3]))
    print("  [%s] %d protected artifacts re-hashed, all unchanged" % (tag, len(rows)))
    return len(rows)


def require_smoke_test():
    p = O / SMOKE
    if not p.exists():
        raise SystemExit(
            "FATAL: %s does not exist. Part C runs only after the engine smoke test "
            "passes. Run code_snapshot/r3_00_engine_diagnostic.py with the "
            "Thermo-Calc Python first." % SMOKE)
    d = json.loads(p.read_text(encoding="utf-8"))
    ok_status = ("PASS", "ENGINE_OK_CONDITION_DID_NOT_CONVERGE")
    if d.get("status") not in ok_status or not d.get("engine_usable", True):
        raise SystemExit(
            "FATAL: the engine smoke test did not pass (status=%s). The 120-state "
            "batch must not be started on an engine that cannot initialise, load "
            "TCCU6, calculate and return phase amounts and compositions. See "
            "14_qa/STEP5K_R3_ENGINE_DIAGNOSTIC.md." % d.get("status"))
    print("  smoke test %s (%s), engine proven on %s"
          % (d.get("status"), d.get("generated_utc"),
             d["smoke_test_condition"]["condition_id"]))
    if d.get("status") != "PASS":
        print("  NOTE: the engine is usable but the smoke-test condition itself did "
              "not converge; that is recorded and the batch proceeds.")
    return d


def stamp_provenance(smoke, n_requests):
    """Identify the engine that produced this checkpoint. The downstream stages
    require this file, so a checkpoint written by anything else - a stand-in used for
    testing, say - cannot be mistaken for a scientific result."""
    import platform
    rec = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": "Thermo-Calc 2025b",
        "tc_python_version": tc_python.__version__,
        "tc_python_path": os.path.dirname(tc_python.__file__),
        "database": "TCCU6",
        "python_interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "host": platform.node(),
        "platform": platform.platform(),
        "produced_by": "code_snapshot/run_r3.py",
        "real_engine": True,
        "synthetic": False,
        "smoke_test_utc": smoke.get("generated_utc"),
        "smoke_test_condition": smoke["smoke_test_condition"]["condition_id"],
        "smoke_test_status": smoke.get("status"),
        "n_requests_in_batch": n_requests,
        "note": "written by the runner on the live engine; downstream stages refuse "
                "to build scientific tables from a checkpoint without it"}
    (O / PROVENANCE).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    seal = json.loads((O / "03_inclusion_freeze" / "INPUT_FREEZE.json").read_text(
        encoding="utf-8"))
    request_path = O / "06_calphad_requests" / "STEP5K_R3_CALPHAD_REQUESTS.csv"
    assert sha(request_path) == seal["request_sha256"], "request table changed"
    assert sha(O / "03_inclusion_freeze"
               / "STEP5K_R3_CONDITION_INCLUSION_FREEZE.md") == seal["rules_sha256"], \
        "inclusion rules changed"
    for p, h in json.loads((O / "00_config" / "helper_hashes.json").read_text(
            encoding="utf-8")).items():
        assert sha(p) == h, "frozen helper changed: %s" % p
    assert tc_python.__version__ == "2025.2.30", tc_python.__version__
    smoke = require_smoke_test()
    verify_protected("before")

    requests = pd.read_csv(request_path).to_dict("records")
    assert len(requests) <= 120, "state budget exceeded"

    D = O / "07_calphad_raw"
    D.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(O / "16_logs" / "equilibrium.log", name="R3")
    states, phases, comps, done = [], [], [], set()
    if (D / "STATES.csv").exists():
        states = pd.read_csv(D / "STATES.csv").to_dict("records")
        done = {r["state_id"] for r in states}
        if (D / "PHASES.csv").exists():
            phases = pd.read_csv(D / "PHASES.csv").to_dict("records")
        if (D / "PHASE_COMPOSITIONS.csv").exists():
            comps = pd.read_csv(D / "PHASE_COMPOSITIONS.csv").to_dict("records")
        print("  resuming: %d states already on disk" % len(done))

    stamp_provenance(smoke, len(requests))
    todo = [q for q in requests if q["state_id"] not in done]
    if limit is not None:
        todo = todo[:limit]
        print("  --limit %d: running %d of %d pending requests"
              % (limit, len(todo), len(requests) - len(done)))

    os.chdir(D)
    # The session is created before the loop, so a startup failure can never be
    # written out as 120 failed equilibria.
    with TCPython() as session:
        for i, q in enumerate(todo, 1):
            w = json.loads(q["composition_wt_pct"])
            assert set(w) <= ALLOWED, "element outside the permitted set: %s" % set(w)
            assert abs(sum(w.values()) - 100) < 1e-8, "composition does not close"
            elements = [k for k, v in w.items() if v > 0]
            assert "CU" in elements
            case = dict(
                case_id=q["condition_id"], family="R3_POSTFREEZE_EXTERNAL",
                x_element="" if q["X_identity"] == "NB_BASE" else q["X_identity"],
                elements=elements, elements_string=",".join(elements),
                wt_fraction={k: v / 100 for k, v in w.items()},
                composition_sum_wt_pct=sum(w.values()), composition_sum_ok=True)

            final, errors, level = None, [], 0
            for level, (iterations, grid) in enumerate(ESCALATION):
                t = time.perf_counter()
                a = dict(state_id=q["state_id"], condition_id=q["condition_id"],
                         level=level, iterations=iterations,
                         global_grid_points=grid,
                         started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime()))
                try:
                    calc = create_calculation(session, case, q["temperature_K"])
                    calc.enable_global_minimization()
                    calc.with_options(
                        SingleEquilibriumOptions()
                        .set_max_no_of_iterations(iterations)
                        .set_global_minimization_max_grid_points(grid))
                    result = calc.calculate()
                    final = extract_result(case, q["temperature_C"],
                                           q["temperature_K"], q["state_id"],
                                           result, logger, t0=t)
                    a.update(converged=True, error="")
                except Exception as exc:                          # noqa: BLE001
                    err = type(exc).__name__ + ": " + str(exc)
                    errors.append(err)
                    a.update(converged=False, error=err)
                a["seconds"] = time.perf_counter() - t
                with (D / "ATTEMPTS.jsonl").open("a", encoding="utf8") as f:
                    f.write(json.dumps(a) + "\n")
                if final:
                    break
                if "Unrecoverable" in a.get("error", ""):
                    raise RuntimeError(
                        "backend crashed during a request; the attempt is preserved "
                        "and the run stops for infrastructure diagnosis rather than "
                        "recording 120 fabricated failures")
            if final is None:
                final = {"state": dict(state_id=q["state_id"],
                                       case_id=q["condition_id"], converged=False,
                                       error_message=" | ".join(errors),
                                       Temperature_C=q["temperature_C"],
                                       Temperature_K=q["temperature_K"],
                                       n_stable_phases=None),
                         "phases": [], "compositions": []}
            final["state"].update(
                condition_id=q["condition_id"], source_id=q["source_id"],
                system=q["system"], source_type="POSTFREEZE_R3_NONNI",
                input_composition_wt_pct=json.dumps(w), solver_level=level,
                prior_attempt_errors=json.dumps(errors))
            states.append(final["state"])
            for p in final["phases"]:
                p["family_5"] = family_of_5(p["phase_model_base_name"])
            phases.extend(final["phases"])
            comps.extend(final["compositions"])
            pd.DataFrame(states).to_csv(D / "STATES.csv", index=False)
            pd.DataFrame(phases, columns=PHASE_HEADER + ["family_5"]).to_csv(
                D / "PHASES.csv", index=False)
            pd.DataFrame(comps, columns=COMPOSITION_HEADER).to_csv(
                D / "PHASE_COMPOSITIONS.csv", index=False)
            print("CHECKPOINT %d/%d  %s  converged=%s  level=%d"
                  % (i, len(todo), q["condition_id"],
                     final["state"]["converged"], level), flush=True)

    verify_protected("after")
    ok = sum(1 for s in states if s.get("converged"))
    print("  %d states on disk | converged %d | failed %d"
          % (len(states), ok, len(states) - ok))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        (O / "16_logs" / "RUN_STOP_ERROR.txt").write_text(traceback.format_exc(),
                                                          encoding="utf8")
        raise
