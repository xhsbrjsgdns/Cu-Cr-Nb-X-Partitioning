# -*- coding: utf-8 -*-
"""
probe_database.py
STEP5K-R3 metadata probe. CALCULATES NO EQUILIBRIUM.

It opens each Cu-Cr-X system, records the phases the installed TCCU6 description
actually contains and their sublattice constituents, and - where the installed
TC-Python exposes it - the database's own reference or bibliography text. The phase
list is what lets stage 10 resolve the crosswalk without guessing; the reference text,
if any, is what could settle whether the six source papers are part of the assessment
this comparison is being checked against.

Extends the earlier probe with: the reference attempt, a written status file even when
startup fails, and `--stack32`, which was already present, kept unchanged.

Usage (Thermo-Calc Python):
    python probe_database.py            metadata only
    python probe_database.py --stack32  same, with a larger Java thread stack
"""
import json
import os
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

if "--stack32" in sys.argv:
    import tc_python.server as server
    _orig = server.launch_gateway

    def _launch(**kwargs):
        kwargs["javaopts"] = [("-Xss32m" if x == "-Xss10m" else x)
                              for x in kwargs["javaopts"]]
        return _orig(**kwargs)
    server.launch_gateway = _launch

from tc_python import TCPython                                    # noqa: E402

O = Path(__file__).resolve().parents[1]
ELEMENTS = ["NB", "AG", "MO", "CO", "SI", "TI"]
STATUS = O / "16_logs" / "PROBE_STATUS.json"


def try_references(sys_obj, builder, session):
    """The database's reference list is not part of the documented TC-Python surface
    and differs between releases. Every plausible accessor is attempted defensively;
    whatever comes back is recorded verbatim, and nothing is inferred from silence."""
    found, tried = {}, []
    for owner, label in ((sys_obj, "system"), (builder, "builder"),
                         (session, "session")):
        for name in dir(owner):
            low = name.lower()
            if not any(k in low for k in ("reference", "bibliograph", "citation",
                                          "database_info", "databaseinfo")):
                continue
            tried.append("%s.%s" % (label, name))
            try:
                v = getattr(owner, name)
                v = v() if callable(v) else v
                found["%s.%s" % (label, name)] = str(v)[:20000]
            except Exception as e:                                # noqa: BLE001
                found["%s.%s" % (label, name)] = "ERROR: %s" % str(e)[:200]
    return {"accessors_tried": tried, "results": found}


def main():
    (O / "16_logs").mkdir(parents=True, exist_ok=True)
    (O / "05_phase_crosswalk").mkdir(parents=True, exist_ok=True)
    rows, refs = [], {}
    with TCPython() as session:
        for x in ELEMENTS:
            builder = session.select_database_and_elements("TCCU6", ["CU", "CR", x])
            sysobj = builder.get_system()
            for name in sysobj.get_phases_in_system():
                p = sysobj.get_phase_object(name)
                rows.append({
                    "system": "CU-CR-" + x, "phase": name,
                    "sublattices": [
                        {"sites": a.get_nr_of_sites(),
                         "constituents": sorted(z.get_name()
                                                for z in a.get_constituents())}
                        for a in p.get_sublattices()]})
            if not refs:
                refs = try_references(sysobj, builder, session)
    (O / "05_phase_crosswalk" / "DATABASE_PHASE_METADATA.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    (O / "05_phase_crosswalk" / "DATABASE_REFERENCE_PROBE.json").write_text(
        json.dumps(refs, indent=2), encoding="utf-8")
    STATUS.write_text(json.dumps({
        "status": "OK", "phase_records": len(rows),
        "systems": sorted({r["system"] for r in rows}),
        "distinct_phases": sorted({r["phase"] for r in rows}),
        "reference_accessors_tried": refs.get("accessors_tried", []),
        "equilibrium_requests": 0}, indent=2), encoding="utf-8")
    print("Database phase metadata rows:", len(rows), "| equilibrium requests: 0")
    print("Distinct phases:", len(({r["phase"] for r in rows})))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as exc:                                  # noqa: BLE001
        (O / "16_logs").mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps({
            "status": "FAILED", "error": "%s: %s" % (type(exc).__name__, str(exc)),
            "traceback": traceback.format_exc(),
            "equilibrium_requests": 0,
            "note": "the engine did not start; this says nothing about any "
                    "composition, phase or database content"}, indent=2),
            encoding="utf-8")
        raise
