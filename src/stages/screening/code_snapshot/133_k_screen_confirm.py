# -*- coding: utf-8 -*-
"""
133_k_screen_confirm.py
STEP5K Part B Stage 4: POST-FREEZE, NON-Ni CALPHAD confirmation of the screening
finalists (spec 29-31).

Scope guard, restated because it is the one rule this stage could break:
  * the finalists are ZR/TI/CO/SI/AG/MO only - the stage refuses to start if any
    other identity, Ni included, appears in the candidate file;
  * nothing here is fed back into any model. These states are a descriptive
    equilibrium check on compositions the frozen surrogate itself selected, and
    the comparison stage labels them as such (spec 13, 34);
  * no frozen Ni blind artifact is read for writing or rewritten - `protect_baseline`
    hashes them before the first equilibrium and re-verifies afterwards.

The equilibrium engine is NOT re-implemented here. `dry_run_lib` and
`solver_escalation` are imported from the frozen FULL_V1 code snapshot by path, so
these states are produced by literally the same runner, database, conditions and
Level 0-3 escalation as the 550,000-state FULL_V1 set.

Failed states are written with converged=0 and their error message. They are never
deleted, never interpolated, and never replaced by a neighbouring temperature.
"""
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C

sys.path.insert(0, C.FULL_V1_CODE)
import dry_run_lib as lib            # noqa: E402  frozen FULL_V1 runner
import fivefamily as ff              # noqa: E402  frozen 5-family mapping
import solver_escalation as esc      # noqa: E402  frozen Level 0-3 escalation

CAND = os.path.join(C.D_SCREEN, "STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv")
OUT_STATES = os.path.join(C.D_CONFIRM, "STEP5K_CONFIRM_STATES_RAW.csv")
OUT_PHASES = os.path.join(C.D_CONFIRM, "STEP5K_CONFIRM_PHASES_RAW.csv")
OUT_PCOMP = os.path.join(C.D_CONFIRM, "STEP5K_CONFIRM_PHASE_COMPOSITIONS_RAW.csv")
OUT_PROC = os.path.join(C.D_CONFIRM, "STEP5K_CONFIRM_PROCESSED.csv")
OUT_LOG = os.path.join(C.D_CONFIRM, "STEP5K_CONFIRM_RUN_LOG.json")


def build_cases(cand):
    cases = []
    for _, r in cand.iterrows():
        x = str(r.X_identity).strip().upper()
        if x not in C.SIX:
            raise SystemExit("FATAL: candidate %s carries identity %r, which is not "
                             "one of the six developed identities %s. STEP5K screening "
                             "is restricted to those (spec 17) and must never run Ni."
                             % (r.candidate_id, x, C.SIX))
        cases.append(lib.build_case({
            "case_id": str(r.candidate_id),
            "family": "STEP5K_SCREEN_%s" % x,
            "X_element": x,
            "X_wt_pct": "%.6f" % float(r.X_wt_pct),
            "Cu_wt_pct": float(r.Cu_wt_pct),
            "Cr_wt_pct": float(r.Cr_wt_pct),
            "Nb_wt_pct": float(r.Nb_wt_pct),
            "composition_source": "STEP5K_VIRTUAL_SCREENING",
            "purpose": "post-freeze non-Ni screening confirmation",
        }))
    return cases


def partition_from_state(phases, comps):
    """Frozen FULL_V1 partitioning, restated for one state.

        n_i_alpha = phase_moles_alpha * X_i_alpha
        P_i_family = sum_alpha in family n_i_alpha / sum_all n_i_alpha

    Mole based, exactly as in 23_fullv1_partitioning.py. Volume-fraction
    partitioning is forbidden and is not computed anywhere in this file.
    """
    moles = {p["raw_phase_name"]: p["phase_moles"] for p in phases}
    fam = {p["raw_phase_name"]: ff.family_of_5(p["phase_model_base_name"])
           for p in phases}
    n = {}
    for c in comps:
        el = str(c["element"]).upper()
        ph = c["raw_phase_name"]
        m, x = moles.get(ph), c["element_mole_fraction"]
        if m is None or m != m or x != x:
            continue
        n.setdefault(el, {}).setdefault(fam[ph], 0.0)
        n[el][fam[ph]] += m * x
    out = {}
    for el, d in n.items():
        tot = sum(d.values())
        for f in ff.FAMILY_ORDER_5:
            out[(el, f)] = (d.get(f, 0.0) / tot) if tot > 0 else float("nan")
    return out


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    logger = lib.setup_logging(os.path.join(C.D_LOGS, "STEP5K_CONFIRM.log"),
                               name="step5k_confirm")
    print("=" * 78)
    print("STEP5K Part B Stage 4: POST-FREEZE NON-Ni CALPHAD confirmation")
    print("=" * 78)

    baseline = C.protect_baseline()
    print("  protected frozen artifacts hashed: %d" % len(baseline))

    cand = pd.read_csv(CAND)
    cases = build_cases(cand)
    temps = list(C.CONFIRM_TEMPS)
    n_states = len(cases) * len(temps)
    if len(cases) > C.MAX_FINALISTS_TOTAL:
        raise SystemExit("FATAL: %d finalists exceeds the frozen cap of %d."
                         % (len(cases), C.MAX_FINALISTS_TOTAL))
    if n_states > 120:
        raise SystemExit("FATAL: %d states exceeds the frozen cap of 120." % n_states)
    print("  %d finalists x %d temperatures = %d equilibrium states"
          % (len(cases), len(temps), n_states))

    holder = esc.SessionHolder(logger)
    holder.start()
    S, P, PC, rows = [], [], [], []
    t_start = time.time()
    try:
        for ci, case in enumerate(cases, 1):
            neighbour_ok = {}
            for tc in temps:
                sid = "%s_T%04d" % (case["case_id"], tc)
                res, level, first_err = esc.run_direct_levels(
                    holder, case, float(tc), float(tc) + 273.15, sid, logger)
                bridge_used, bridges = 0, []
                if not res["state"]["converged"]:
                    br, ok, bridges, _ = esc.run_bridge(
                        holder, case, float(tc), float(tc) + 273.15, sid, logger,
                        neighbour_ok)
                    if ok and br is not None and br["state"]["converged"]:
                        res, level, bridge_used = br, 3, 1
                st = dict(res["state"])
                st["solver_level_used"] = level
                st["bridge_used"] = bridge_used
                st["bridge_temperatures_C"] = ";".join(str(b) for b in bridges)
                st["original_error_message"] = first_err or ""
                neighbour_ok[tc] = bool(st["converged"])
                S.append(st)
                P.extend(res["phases"])
                PC.extend(res["compositions"])

                row = {"candidate_id": case["case_id"], "X_identity": case["x_element"],
                       "Temperature_C": tc, "state_id": sid,
                       "converged": int(bool(st["converged"])),
                       "solver_level_used": level, "bridge_used": bridge_used,
                       "error_message": st["error_message"],
                       "n_stable_phases": st["n_stable_phases"]}
                if st["converged"]:
                    agg = lib.aggregate_phase_families(case, st, res["phases"])
                    vol = {f: 0.0 for f in ff.FAMILY_ORDER_5}
                    for g in agg:
                        vol[ff.family_of_5(g["phase_model_base_name"])] += \
                            g["phase_volume_fraction_total"]
                    for f in ff.FAMILY_ORDER_5:
                        row["calphad_frac_%s" % f] = vol[f]
                    pa = partition_from_state(res["phases"], res["compositions"])
                    for f in ff.FAMILY_ORDER_5:
                        row["calphad_Cr_P_%s" % f] = pa.get(("CR", f), float("nan"))
                        row["calphad_X_P_%s" % f] = pa.get(
                            (case["x_element"], f), float("nan"))
                else:
                    for f in ff.FAMILY_ORDER_5:
                        row["calphad_frac_%s" % f] = float("nan")
                        row["calphad_Cr_P_%s" % f] = float("nan")
                        row["calphad_X_P_%s" % f] = float("nan")
                rows.append(row)
            print("  [%2d/%2d] %s  %s" % (ci, len(cases), case["case_id"],
                                          "".join("." if neighbour_ok[t] else "X"
                                                  for t in temps)))
    finally:
        holder.shutdown()

    pd.DataFrame(S).to_csv(OUT_STATES, index=False, encoding="utf-8")
    pd.DataFrame(P).to_csv(OUT_PHASES, index=False, encoding="utf-8")
    pd.DataFrame(PC).to_csv(OUT_PCOMP, index=False, encoding="utf-8")
    proc = pd.DataFrame(rows)
    proc.to_csv(OUT_PROC, index=False, encoding="utf-8")

    moved = C.verify_protected(baseline)
    if moved:
        raise SystemExit("FATAL: protected frozen artifacts changed during this "
                         "stage: %s" % moved)

    n_ok = int(proc.converged.sum())
    C.write_json(OUT_LOG, {
        "generated_utc": C.utcnow(),
        "stage": "STEP5K Part B Stage 4 - post-freeze non-Ni screening confirmation",
        "engine": "Thermo-Calc TCCU6, frozen FULL_V1 dry_run_lib + solver_escalation "
                  "imported by path from %s" % C.FULL_V1_CODE,
        "conditions": {"P_Pa": lib.PRESSURE_PA, "N_mol": lib.SYSTEM_SIZE_MOL,
                       "basis": "wt.%, Cu balance", "database": lib.DATABASE_NAME},
        "temperatures_C": temps,
        "n_finalists": len(cases), "n_states": n_states,
        "n_converged": n_ok, "n_failed": n_states - n_ok,
        "identities": sorted({c["x_element"] for c in cases}),
        "ni_states_run": 0,
        "failed_states_are_kept_not_deleted_or_interpolated": True,
        "partition_formula": "n_i_alpha = phase_moles * X_i_alpha ; "
                             "P_i_family = sum_family n / sum_all n (mole based; "
                             "volume-fraction partitioning is forbidden)",
        "used_to_retrain_or_retune_anything": False,
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "protected_artifacts_unchanged": True})
    print("  converged %d / %d states in %.1f s" % (n_ok, n_states, time.time() - t_start))
    if n_ok == 0:
        raise SystemExit("FATAL: no confirmation state converged; nothing to compare.")
    print("STAGE B4: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
