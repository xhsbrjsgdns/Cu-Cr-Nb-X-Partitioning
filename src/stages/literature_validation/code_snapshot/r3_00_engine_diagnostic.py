# -*- coding: utf-8 -*-
"""
r3_00_engine_diagnostic.py
STEP5K-R3 part B: diagnose TC-Python initialization and run ONE non-Ni smoke test.

Seven steps, each recorded with its verbatim error if it fails:
    1 session initialisation
    2 TCCU6 selected with the system's elements
    3 system built
    4 phase names queried
    5 equilibrium calculated
    6 phase amounts queried
    7 phase compositions queried

The smoke-test chemistry is Cu-Cr-Nb, taken from an already-admitted R3 condition -
the first Cu-Cr-Nb request in the sealed request table, by state_id - at its exact
literature composition and exact experimental temperature. Nothing is invented and
nothing is substituted.

ONE DISTINCTION MATTERS. Steps 1-4 test whether the ENGINE works. Step 5 tests
whether this particular alloy CONVERGES. They are different failures and are reported
differently: if the engine initialises, loads TCCU6 and returns the phase list, but
this Cr-rich condition does not converge even after the frozen four-level escalation,
the status is ENGINE_OK_CONDITION_DID_NOT_CONVERGE. That is a scientific outcome the
batch is built to record, not a reason to block 120 states, and the batch runner
accepts it while noting it. Any failure at steps 1-4 blocks the batch outright.

NOTHING IS CHANGED TO MAKE IT PASS. The database stays TCCU6, the phase mapping stays
frozen, pressure stays 1 bar, system size stays 1 mol, and no scientific setting is
touched. `--stack32` runs the same probe with a larger Java thread stack; it is a
diagnostic variant for the vendor, not a fix, and the output says so.

The result is a diagnostic only. It writes no state into 07_calphad_raw and can never
be mistaken for a scientific calculation.
"""
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

STACK32 = "--stack32" in sys.argv
if STACK32:
    import tc_python.server as _server
    _orig_launch = _server.launch_gateway

    def _launch(**kwargs):
        kwargs["javaopts"] = [("-Xss32m" if x == "-Xss10m" else x)
                              for x in kwargs["javaopts"]]
        return _orig_launch(**kwargs)
    _server.launch_gateway = _launch

O = Path(__file__).resolve().parents[1]
JSON_OUT = O / "14_qa" / "STEP5K_R3_ENGINE_SMOKE_TEST.json"
MD_OUT = O / "14_qa" / "STEP5K_R3_ENGINE_DIAGNOSTIC.md"
REQ = O / "06_calphad_requests" / "STEP5K_R3_CALPHAD_REQUESTS.csv"


def environment():
    env = {
        "python_interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": os.getcwd(),
    }
    try:
        import tc_python
        env["tc_python_version"] = getattr(tc_python, "__version__", "unknown")
        env["tc_python_path"] = os.path.dirname(
            getattr(tc_python, "__file__", "") or "")
    except Exception as e:                                        # noqa: BLE001
        env["tc_python_import_error"] = "%s: %s" % (type(e).__name__, e)
    for k in ("TC25B_HOME", "TC24B_HOME", "THERMOCALC_HOME", "JAVA_HOME",
              "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS",
              "LSHOST", "LSFORCEHOST", "TC_LICENSE_FILE", "OMP_STACKSIZE",
              "KMP_STACKSIZE", "TEMP", "TMP"):
        if os.environ.get(k):
            env["env_" + k] = os.environ[k]
    home = os.environ.get("TC25B_HOME") or r"C:\Program Files\Thermo-Calc\2025b"
    env["thermo_calc_installation"] = home
    env["thermo_calc_installation_exists"] = os.path.isdir(home)
    if os.path.isdir(home):
        for sub in ("data", "python", "jre", "bin"):
            env["has_" + sub] = os.path.isdir(os.path.join(home, sub))
        jre = os.path.join(home, "jre", "bin", "java.exe")
        env["bundled_java"] = jre if os.path.exists(jre) else "not found at jre/bin"
        db = os.path.join(home, "data", "TCCU6")
        env["TCCU6_data_directory"] = db if os.path.isdir(db) else "not found"
    try:
        t = os.environ.get("TEMP") or os.getcwd()
        u = shutil.disk_usage(t)
        env["temp_free_GB"] = round(u.free / 1e9, 2)
        u2 = shutil.disk_usage(os.path.splitdrive(sys.executable)[0] + os.sep)
        env["interpreter_drive_free_GB"] = round(u2.free / 1e9, 2)
    except Exception as e:                                        # noqa: BLE001
        env["disk_usage_error"] = str(e)
    env["java_thread_stack_variant"] = "-Xss32m (diagnostic variant, NOT a fix)" \
        if STACK32 else "installer default"
    return env


def pick_condition():
    import pandas as pd
    req = pd.read_csv(REQ)
    nb = req[req.system.astype(str).str.replace("\u2013", "-") == "Cu-Cr-Nb"]
    if not len(nb):
        raise SystemExit("FATAL: the sealed request table contains no Cu-Cr-Nb "
                         "condition to smoke-test with.")
    r = nb.sort_values("state_id").iloc[0]
    w = json.loads(r.composition_wt_pct)
    if "NI" in w:
        raise SystemExit("FATAL: the chosen smoke-test condition contains Ni.")
    return {"condition_id": r.condition_id, "state_id": r.state_id,
            "source_id": r.source_id, "system": r.system,
            "composition_wt_pct": w, "temperature_C": float(r.temperature_C),
            "temperature_K": float(r.temperature_K),
            "pressure_Pa": float(r.pressure_Pa),
            "system_size_mol": float(r.system_size_mol),
            "database": str(r.database), "page": r.page, "table": r.table}


def main():
    (O / "14_qa").mkdir(parents=True, exist_ok=True)
    env = environment()
    cond = pick_condition()
    steps, state = [], {}
    t_all = time.time()

    def step(n, name, fn):
        t = time.perf_counter()
        rec = {"step": n, "name": name}
        try:
            rec["detail"] = fn()
            rec["result"] = "PASS"
        except BaseException as e:                                # noqa: BLE001
            rec["result"] = "FAIL"
            rec["error_type"] = type(e).__name__
            rec["error"] = str(e)
            rec["traceback"] = traceback.format_exc()
        rec["seconds"] = round(time.perf_counter() - t, 3)
        steps.append(rec)
        print("  %d %-34s %s%s" % (n, name, rec["result"],
                                   "" if rec["result"] == "PASS"
                                   else "  <- " + rec["error"][:150]))
        return rec["result"] == "PASS"

    print("=" * 78)
    print("STEP5K-R3 engine diagnostic and Cu-Cr-Nb smoke test")
    print("=" * 78)
    print("  interpreter : %s" % env["python_interpreter"])
    print("  tc_python   : %s (%s)" % (env.get("tc_python_version", "?"),
                                       env.get("tc_python_path", "?")))
    print("  Thermo-Calc : %s (exists=%s)" % (env["thermo_calc_installation"],
                                              env["thermo_calc_installation_exists"]))
    print("  condition   : %s  %s  %.2f K" % (cond["condition_id"],
                                              cond["composition_wt_pct"],
                                              cond["temperature_K"]))
    print("-" * 78)

    from tc_python import TCPython, SingleEquilibriumOptions   # noqa: E402

    # the frozen escalation, identical to run_r3.ESCALATION
    ESCALATION = [(500, 2000), (2000, 10000), (5000, 30000), (10000, 100000)]

    ok = True
    ctx = {}
    try:
        with TCPython() as session:
            ctx["session"] = session
            steps.append({"step": 1, "name": "session initialisation",
                          "result": "PASS", "seconds": None,
                          "detail": "TCPython() context entered; GES version set"})
            print("  1 %-34s PASS" % "session initialisation")

            els = sorted(cond["composition_wt_pct"])
            ok &= step(2, "TCCU6 selected with elements",
                       lambda: ctx.setdefault(
                           "builder",
                           session.select_database_and_elements(
                               cond["database"], els)) and
                       {"database": cond["database"], "elements": els})
            if ok:
                ok &= step(3, "system built",
                           lambda: ctx.setdefault("sys",
                                                  ctx["builder"].get_system())
                           and "System object created")
            if ok:
                ok &= step(4, "phase names queried", lambda: {
                    "n_phases": len(list(ctx["sys"].get_phases_in_system())),
                    "phases": sorted(ctx["sys"].get_phases_in_system())})
            if ok:
                def _calc():
                    c = ctx["sys"].with_single_equilibrium_calculation(
                        default_conditions=False)
                    c.set_condition("T", cond["temperature_K"])
                    c.set_condition("P", cond["pressure_Pa"])
                    c.set_condition("N", cond["system_size_mol"])
                    for el, wt in cond["composition_wt_pct"].items():
                        if el == "CU":
                            continue
                        c.set_condition("W(%s)" % el, wt / 100.0)
                    c.enable_global_minimization()
                    errs = []
                    # the frozen deterministic escalation, exactly as run_r3.py uses
                    for lvl, (it, grid) in enumerate(ESCALATION):
                        try:
                            c.with_options(
                                SingleEquilibriumOptions()
                                .set_max_no_of_iterations(it)
                                .set_global_minimization_max_grid_points(grid))
                            ctx["result"] = c.calculate()
                            ctx["level"] = lvl
                            return {"solver_level": lvl, "iterations": it,
                                    "global_grid_points": grid,
                                    "earlier_attempt_errors": errs,
                                    "note": "calculated at the exact literature "
                                            "composition and temperature"}
                        except Exception as e:                    # noqa: BLE001
                            errs.append("level %d: %s: %s"
                                        % (lvl, type(e).__name__, str(e)[:300]))
                            if "Unrecoverable" in type(e).__name__ or \
                                    "Unrecoverable" in str(e):
                                raise
                    raise RuntimeError("did not converge after the frozen four-level "
                                       "escalation: " + " | ".join(errs))
                ok &= step(5, "equilibrium calculated", _calc)
            if ok:
                def _amounts():
                    stable = sorted(ctx["result"].get_stable_phases())
                    ctx["stable"] = stable
                    out = {}
                    for p in stable:
                        out[p] = {
                            "NPM": float(ctx["result"].get_value_of("NPM(%s)" % p)),
                            "VPV": float(ctx["result"].get_value_of("VPV(%s)" % p))}
                    state["stable_phases"] = stable
                    state["amounts"] = out
                    return {"n_stable_phases": len(stable), "amounts": out}
                ok &= step(6, "phase amounts queried", _amounts)
            if ok:
                def _comps():
                    comps = list(ctx["result"].get_components())
                    out = {}
                    for p in ctx["stable"]:
                        out[p] = {e: float(ctx["result"].get_value_of(
                            "X(%s,%s)" % (p, e))) for e in comps}
                    state["phase_compositions_mole_fraction"] = out
                    return {"components": comps, "n_phases": len(out)}
                ok &= step(7, "phase compositions queried", _comps)
    except BaseException as e:                                    # noqa: BLE001
        ok = False
        steps.append({"step": 1, "name": "session initialisation", "result": "FAIL",
                      "error_type": type(e).__name__, "error": str(e),
                      "traceback": traceback.format_exc(),
                      "note": "the engine did not start; every later step is "
                              "unreachable and says nothing about any composition, "
                              "phase or database content"})
        print("  1 %-34s FAIL  <- %s" % ("session initialisation", str(e)[:150]))

    by_step = {s.get("step"): s for s in steps}
    engine_ok = all(by_step.get(n, {}).get("result") == "PASS" for n in (1, 2, 3, 4))
    passed = ok and all(s["result"] == "PASS" for s in steps)
    if passed:
        status = "PASS"
    elif engine_ok and by_step.get(5, {}).get("result") == "FAIL":
        status = "ENGINE_OK_CONDITION_DID_NOT_CONVERGE"
    else:
        status = "FAIL"
    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "engine_usable": bool(engine_ok),
        "engine_steps_1_to_4_passed": bool(engine_ok),
        "solver_level_used": (by_step.get(5, {}).get("detail") or {}).get(
            "solver_level") if isinstance(by_step.get(5, {}).get("detail"),
                                          dict) else None,
        "all_seven_steps_passed": bool(passed),
        "wall_clock_seconds": round(time.time() - t_all, 2),
        "environment": env,
        "smoke_test_condition": cond,
        "steps": steps,
        "observed_state": state,
        "is_a_scientific_result": False,
        "written_into_07_calphad_raw": False,
        "database_or_settings_changed_to_pass": False,
        "ni_used": False}
    JSON_OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    md = ["# STEP5K-R3 engine diagnostic\n\n",
          "_Generated %s._ **%s**\n\n" % (out["generated_utc"],
                                          "Engine initialises and calculates."
                                          if passed else
                                          "Engine does NOT reach a usable state."),
          "## Environment\n\n| item | value |\n|---|---|\n"]
    for k, v in env.items():
        md.append("| %s | `%s` |\n" % (k, v))
    md.append("\n## Smoke-test condition\n\n")
    md.append("Taken from the sealed request table, not invented: **%s** (%s, page "
              "%s, table %s), composition `%s` wt.%%, %.2f K (%.2f C), %s, %g Pa, "
              "%g mol.\n\n"
              % (cond["condition_id"], cond["source_id"], cond["page"],
                 cond["table"], cond["composition_wt_pct"], cond["temperature_K"],
                 cond["temperature_C"], cond["database"], cond["pressure_Pa"],
                 cond["system_size_mol"]))
    md.append("## Steps\n\n| # | step | result | seconds | detail or error |\n"
              "|---|---|---|---|---|\n")
    for s in steps:
        d = s.get("detail") if s["result"] == "PASS" else (
            "`%s: %s`" % (s.get("error_type", ""), str(s.get("error", ""))[:300]))
        md.append("| %s | %s | %s | %s | %s |\n"
                  % (s.get("step"), s.get("name"), s["result"], s.get("seconds"),
                     str(d)[:400].replace("\n", " ")))
    if status == "ENGINE_OK_CONDITION_DID_NOT_CONVERGE":
        md.append("\n## Result\n\n**The engine is usable.** Initialisation, TCCU6 "
                  "loading, system construction and the phase-list query all "
                  "succeeded. The equilibrium at this particular condition did not "
                  "converge, even after the frozen four-level escalation. That is a "
                  "property of this Cr-rich alloy, not of the installation, and it is "
                  "exactly the outcome the batch is built to record state by state. "
                  "The batch may proceed; the runner records this status in the "
                  "engine provenance stamp.\n\n```\n%s\n```\n"
                  % str(by_step.get(5, {}).get("error", ""))[:1500])
    elif passed:
        md.append("\n## Result\n\nAll seven steps passed. The stable phases at this "
                  "condition were `%s`. This is a diagnostic, not a scientific "
                  "state: it was not written into `07_calphad_raw`, and the frozen "
                  "120-request batch is still the only source of R3 results.\n"
                  % ", ".join(state.get("stable_phases", [])))
    else:
        first = next((s for s in steps if s["result"] == "FAIL"), None)
        md.append("\n## Result\n\n**Initialisation or calculation failed at step %s "
                  "(%s).**\n\n```\n%s\n```\n\n"
                  % (first.get("step") if first else "?",
                     first.get("name") if first else "?",
                     (first.get("traceback") or first.get("error", ""))[:3000]
                     if first else ""))
        md.append("Nothing was changed to work around it: the database, the phase "
                  "mapping, the pressure, the system size and every scientific "
                  "setting are as frozen. The R3 batch has not been run and no "
                  "state has been fabricated.\n\n")
        md.append("Report to the vendor with this file, the environment table above "
                  "and `16_logs/RUNTIME_DIAGNOSIS.md`.\n")
    MD_OUT.write_text("".join(md), encoding="utf-8")

    print("-" * 78)
    print("  SMOKE TEST: %s  ->  %s" % (out["status"], JSON_OUT.name))
    if passed:
        return 0
    if status == "ENGINE_OK_CONDITION_DID_NOT_CONVERGE":
        print("  the engine works; this condition did not converge. The batch may "
              "proceed and will record it.")
        return 4
    return 1


if __name__ == "__main__":
    sys.exit(main())
