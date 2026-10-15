# -*- coding: utf-8 -*-
"""
60_f5p_probe.py
STEP5F-PILOT Stage 0: TCCU6 ELEMENT AVAILABILITY PROBE.

Before 450 compositions are generated and 4,950 equilibria are queued, establish the one
fact that decides whether the pilot is even possible for each candidate: is the element
present in TCCU6 at all, and does a Cu-Cr-Nb-X system build and converge once?

This costs nine systems and nine equilibria - seconds - and turns "the run died three
hours in" into "La is not in this database, here is the verdict".

An element that fails here is NOT quietly dropped. It is recorded with the verbatim
exception and carried into the report as a feasibility result in its own right.

Ni is never probed. It is not in the candidate list and this module refuses to accept it.
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5p_common as C

sys.path.insert(0, C.FULL_V1_CODE)

PROBE_T_C = 800.0                 # mid-grid, away from both ends
PROBE_S, PROBE_R, PROBE_X = 1.3, 4.0, 0.10   # centre of the frozen pilot ranges


def build_probe_case(element):
    w_cr, w_nb = C.invert_s_r(PROBE_S, PROBE_R)
    w_cr, w_nb = round(w_cr, C.WT_DECIMALS), round(w_nb, C.WT_DECIMALS)
    return {
        "case_id": "F5P_PROBE_%s" % element, "family": element,
        "X_element": element, "X_wt_pct": "%.4f" % PROBE_X,
        "Cu_wt_pct": 100.0 - w_cr - w_nb - PROBE_X,
        "Cr_wt_pct": w_cr, "Nb_wt_pct": w_nb,
        "composition_source": "STEP5F_PILOT probe", "purpose": "element availability probe",
    }


def classify(exc_text):
    t = exc_text.lower()
    if any(k in t for k in ("not found in database", "no such element", "element not",
                            "unknown element", "is not in the database",
                            "not present in database", "cannot find element")):
        return "NOT_IN_DATABASE"
    if "license" in t or "licence" in t:
        return "LICENSE_ERROR"
    return "ERROR"


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    bad = [e for e in C.PILOT_ELEMENTS if e in C.FORBIDDEN_ELEMENTS]
    if bad:
        raise SystemExit("FATAL: forbidden element in the pilot list: %s" % bad)

    print("=" * 78)
    print("STEP5F-PILOT  Stage 0: TCCU6 element availability probe")
    print("nine candidates: %s" % " ".join(C.PILOT_ELEMENTS))
    print("Ni is sealed and is not probed.")
    print("=" * 78)

    import dry_run_lib as L
    from tc_python import TCPython

    rows = []
    t0 = time.time()
    with TCPython() as session:
        for element in C.PILOT_ELEMENTS:
            rec = {"element": element, "database": L.DATABASE_NAME,
                   "probe_temperature_C": PROBE_T_C,
                   "probe_S_CrNb_wt_pct": PROBE_S, "probe_R_CrNb_atomic": PROBE_R,
                   "probe_X_wt_pct": PROBE_X}
            case = L.build_case(build_probe_case(element))
            t1 = time.time()
            try:
                calc = L.create_calculation(session, case, PROBE_T_C + 273.15)
                res = calc.calculate()
                phases = list(res.get_stable_phases())
                rec.update({"status": "AVAILABLE", "system_built": True,
                            "equilibrium_converged": True,
                            "n_stable_phases": len(phases),
                            "stable_phases": ";".join(sorted(phases)),
                            "exception_type": "", "exception": ""})
                print("  %-3s AVAILABLE     %d phases: %s"
                      % (element, len(phases), ", ".join(sorted(phases))))
            except Exception as exc:                                    # noqa: BLE001
                txt = "%s: %s" % (type(exc).__name__, exc)
                rec.update({"status": classify(txt), "system_built": False,
                            "equilibrium_converged": False, "n_stable_phases": 0,
                            "stable_phases": "",
                            "exception_type": type(exc).__name__,
                            "exception": txt[:1500]})
                print("  %-3s %-14s %s" % (element, rec["status"], txt[:110]))
                with open(os.path.join(C.D_LOGS, "PROBE_TRACEBACKS.log"), "a",
                          encoding="utf-8") as fh:
                    fh.write("\n===== %s =====\n%s\n" % (element, traceback.format_exc()))
            rec["probe_seconds"] = round(time.time() - t1, 2)
            rows.append(rec)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(C.D_QA, "STEP5F_PILOT_ELEMENT_PROBE.csv")
    df.to_csv(out, index=False, encoding="utf-8")

    available = sorted(df.loc[df.status == "AVAILABLE", "element"])
    blocked = sorted(df.loc[df.status != "AVAILABLE", "element"])
    hits = C.ni_scan([("probe_table", df)])
    summary = {
        "stage": "STEP5F-PILOT Stage 0 element probe",
        "generated_utc": C.utcnow(),
        "database": L.DATABASE_NAME,
        "probe_conditions": {"T_C": PROBE_T_C, "S_CrNb_wt_pct": PROBE_S,
                             "R_CrNb_atomic": PROBE_R, "X_wt_pct": PROBE_X,
                             "pressure_bar": 1.0, "system_size_mol": 1.0,
                             "basis": "wt.%, Cu balance"},
        "candidates": C.PILOT_ELEMENTS,
        "available": available,
        "not_available": blocked,
        "not_available_detail": {r["element"]: {"status": r["status"],
                                                "exception": r["exception"]}
                                 for r in rows if r["status"] != "AVAILABLE"},
        "n_available": len(available), "n_candidates": len(C.PILOT_ELEMENTS),
        "ni_probed": False, "ni_hits": hits,
        "elapsed_seconds": round(time.time() - t0, 1),
        "note": ("an element that is not available in TCCU6 cannot be part of a "
                 "Cu-Cr-Nb-X CALPHAD augmentation at all; that is a feasibility RESULT, "
                 "not a bug, and it is reported rather than worked around"),
    }
    C.write_json(os.path.join(C.D_QA, "STEP5F_PILOT_ELEMENT_PROBE.json"), summary)

    print("\n" + "=" * 78)
    print("AVAILABLE (%d/%d): %s" % (len(available), len(C.PILOT_ELEMENTS),
                                     " ".join(available) or "none"))
    if blocked:
        print("NOT AVAILABLE:      %s" % " ".join(blocked))
        print("\nThese elements cannot enter the pilot run. The remaining stages will")
        print("run on the available elements and the report will state exactly why.")
    print("Ni scan: %s" % ("CLEAN" if not hits else hits))
    print("=" * 78)
    if not available:
        print("PROBE_FAILED - no candidate is usable")
        return 1
    print("PROBE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
