# -*- coding: utf-8 -*-
"""
135_k_extval_calphad.py
STEP5K Part A Stage 2: POST-FREEZE, NON-Ni CALPHAD for the literature compositions
that are actually stated in the papers.

Why this stage exists. FULL_V1's Domain A samples S_CrNb = 0.6-2.0 wt.% - the dilute
microalloying regime. Every Cu-Cr-Nb composition that the retrieved literature states
explicitly (Cu-3Cr-0.5Nb; GRCop-42; GRCop-84) sits far above that window, and the
Cu-Zr rows carry no Nb at all. No frozen FULL_V1 state is therefore a legitimate
stand-in for any of them, and pretending otherwise would be the exact failure this
stage is designed to avoid. So the comparison is made on new, clearly labelled
POST-FREEZE equilibrium states instead.

What this stage does and does not claim.
  * It checks the frozen thermodynamic description against reported phase
    identities. It is a CALPHAD-LAYER plausibility check.
  * It is NOT a validation of the machine-learning surrogate. These compositions are
    outside the surrogate's training domain and NO model prediction is made for them
    anywhere in STEP5K.
  * It never asserts that CALPHAD is ground truth. Verdicts come from the fixed
    vocabulary: qualitatively consistent / partially consistent / reasonable
    equilibrium-vs-non-equilibrium discrepancy / not directly comparable.
  * Nothing here is fed back into any model, feature set, hyperparameter, topology
    class or Ni interpretation (spec 13).

Composition provenance. Every composition below is quoted from its paper. Nothing is
invented, and nothing is silently converted:
  * EXT_01 Cu-3Cr-0.5Nb - the LIT_17 abstract states "(wt.%)" explicitly.
  * EXT_02/03 GRCop-42 / GRCop-84 - the LIT_18 abstract does NOT state the basis. The
    NASA definition of the designation is ATOMIC per cent (NASA/TM-2005-213566,
    "GRCop-84 (Cu-8 at.% Cr-4 at.% Nb)"), so the at.% reading is computed as primary
    and converted to wt.% here; the wt.% reading of the same digits is ALSO computed
    and kept, labelled, because the paper itself does not settle it. Neither is
    asserted as the paper's meaning.
  * EXT_04/06 Cu-0.21Cr-0.20Zr and Cu-0.06Zr - LIT_04 states wt.%.
  * EXT_05 Cu-0.13Zr - LIT_05 states wt.%.
  * Cu-0.3%Cr (LIT_19) is deliberately ABSENT: the available metadata does not
    distinguish wt.% from at.%, and spec 5 forbids inferring it.
"""
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

GRID_C = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]   # frozen FULL_V1 grid
M = {"CU": 63.546, "CR": 51.9961, "NB": 92.9064, "ZR": 91.224}


def at_to_wt(at):
    """at.% dict (balance CU implied) -> wt.% dict. Used only for the GRCop rows."""
    full = dict(at)
    full["CU"] = 100.0 - sum(at.values())
    mass = {k: v * M[k] for k, v in full.items()}
    tot = sum(mass.values())
    return {k: 100.0 * v / tot for k, v in mass.items()}


def spec(case_id, paper_id, note, wt, basis):
    """Build a case dict in the frozen FULL_V1 shape from an explicit wt.% dict.

    dry_run_lib.build_case always injects CR and NB; several literature alloys have
    no Nb at all, and declaring an element at W = 0 is not the same calculation as
    not declaring it. So the case dict is assembled directly here, with exactly the
    elements the paper names, and handed to the same frozen create_calculation /
    extract_result code path.
    """
    wt = {k: float(v) for k, v in wt.items() if float(v) > 0.0}
    cu = 100.0 - sum(wt.values())
    els = ["CU"] + sorted(wt)
    full = dict(wt); full["CU"] = cu
    return {"case_id": case_id, "family": "STEP5K_EXTVAL", "x_element": "",
            "elements": els, "wt_pct": full, "cu_balance_wt_pct": cu,
            "wt_fraction": {k: v / 100.0 for k, v in full.items()},
            "composition_source": "%s (%s)" % (paper_id, basis),
            "purpose": note, "composition_sum_wt_pct": sum(full.values()),
            "composition_sum_ok": abs(sum(full.values()) - 100.0) < 0.01,
            "elements_string": ",".join(els)}


def external_set():
    g84 = at_to_wt({"CR": 8.0, "NB": 4.0})
    g42 = at_to_wt({"CR": 4.0, "NB": 2.0})
    return [
        spec("EXT_01_Cu3Cr05Nb", "LIT_17",
             "Nb-lean sintered Cu-Cr-Nb; paper reports Cr-based and Cr2Nb precipitates",
             {"CR": 3.0, "NB": 0.5}, "wt.% stated in the abstract"),
        spec("EXT_02_GRCop42_at", "LIT_18",
             "GRCop-42 read as Cu-4at.%Cr-2at.%Nb (NASA designation), converted to wt.%",
             {"CR": g42["CR"], "NB": g42["NB"]}, "at.% designation -> wt.%"),
        spec("EXT_03_GRCop84_at", "LIT_18",
             "GRCop-84 read as Cu-8at.%Cr-4at.%Nb (NASA designation), converted to wt.%",
             {"CR": g84["CR"], "NB": g84["NB"]}, "at.% designation -> wt.%"),
        spec("EXT_02b_GRCop42_wt", "LIT_18",
             "GRCop-42 digits read as wt.% instead - alternative reading, not asserted",
             {"CR": 4.0, "NB": 2.0}, "digits read as wt.% (alternative)"),
        spec("EXT_03b_GRCop84_wt", "LIT_18",
             "GRCop-84 digits read as wt.% instead - alternative reading, not asserted",
             {"CR": 8.0, "NB": 4.0}, "digits read as wt.% (alternative)"),
        spec("EXT_04_Cu021Cr020Zr", "LIT_04",
             "Cu-Cr-Zr, no Nb; paper reports equilibrium Cu5Zr after 450 C / 1 h",
             {"CR": 0.21, "ZR": 0.20}, "wt.% stated in the abstract"),
        spec("EXT_05_Cu013Zr", "LIT_05",
             "binary Cu-Zr; paper reports Cu5Zr as the equilibrium phase, 450-950 C",
             {"ZR": 0.13}, "wt.% stated in the abstract"),
        spec("EXT_06_Cu006Zr", "LIT_04",
             "binary Cu-Zr, dilute; two-stage decomposition to equilibrium Cu5Zr",
             {"ZR": 0.06}, "wt.% stated in the abstract"),
    ]


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    logger = lib.setup_logging(os.path.join(C.D_LOGS, "STEP5K_EXTVAL_CALPHAD.log"),
                               name="step5k_extval")
    print("=" * 78)
    print("STEP5K Part A Stage 2: POST-FREEZE NON-Ni literature-composition CALPHAD")
    print("=" * 78)
    baseline = C.protect_baseline()
    print("  protected frozen artifacts hashed: %d" % len(baseline))

    cases = external_set()
    for c in cases:
        if "NI" in c["elements"]:
            raise SystemExit("FATAL: Ni appears in external case %s." % c["case_id"])
        if not c["composition_sum_ok"]:
            raise SystemExit("FATAL: %s wt%% sums to %.4f" % (c["case_id"],
                                                             c["composition_sum_wt_pct"]))
        print("  %-22s %s" % (c["case_id"], ", ".join(
            "%s %.4f" % (e, c["wt_pct"][e]) for e in c["elements"])))
    n_states = len(cases) * len(GRID_C)
    print("  %d compositions x %d temperatures = %d equilibrium states"
          % (len(cases), len(GRID_C), n_states))

    holder = esc.SessionHolder(logger)
    holder.start()
    S, P, PC, rows = [], [], [], []
    t0 = time.time()
    try:
        for case in cases:
            ok_map = {}
            for tc in GRID_C:
                sid = "%s_T%04d" % (case["case_id"], tc)
                res, level, first_err = esc.run_direct_levels(
                    holder, case, float(tc), float(tc) + 273.15, sid, logger)
                bridge_used, bridges = 0, []
                if not res["state"]["converged"]:
                    br, ok, bridges, _ = esc.run_bridge(
                        holder, case, float(tc), float(tc) + 273.15, sid, logger, ok_map)
                    if ok and br is not None and br["state"]["converged"]:
                        res, level, bridge_used = br, 3, 1
                st = dict(res["state"])
                st.update({"solver_level_used": level, "bridge_used": bridge_used,
                           "bridge_temperatures_C": ";".join(str(b) for b in bridges),
                           "original_error_message": first_err or ""})
                ok_map[tc] = bool(st["converged"])
                S.append(st); P.extend(res["phases"]); PC.extend(res["compositions"])

                row = {"ext_id": case["case_id"], "paper_id":
                       case["composition_source"].split(" (")[0],
                       "composition_wt_pct": ";".join(
                           "%s=%.6f" % (e, case["wt_pct"][e]) for e in case["elements"]),
                       "composition_basis_note": case["composition_source"],
                       "purpose": case["purpose"], "Temperature_C": tc,
                       "state_id": sid, "converged": int(bool(st["converged"])),
                       "solver_level_used": level, "bridge_used": bridge_used,
                       "error_message": st["error_message"],
                       "n_stable_phases": st["n_stable_phases"]}
                if st["converged"]:
                    agg = lib.aggregate_phase_families(case, st, res["phases"])
                    vol = {f: 0.0 for f in ff.FAMILY_ORDER_5}
                    bases = {}
                    for g in agg:
                        fam = ff.family_of_5(g["phase_model_base_name"])
                        vol[fam] += g["phase_volume_fraction_total"]
                        bases[g["phase_model_base_name"]] = \
                            g["phase_volume_fraction_total"]
                    for f in ff.FAMILY_ORDER_5:
                        row["frac_%s" % f] = vol[f]
                    row["stable_phase_bases"] = ";".join(
                        "%s=%.6g" % (k, v) for k, v in
                        sorted(bases.items(), key=lambda kv: -kv[1]))
                    row["raw_stable_phases"] = ";".join(
                        sorted({p["raw_phase_name"] for p in res["phases"]}))
                else:
                    for f in ff.FAMILY_ORDER_5:
                        row["frac_%s" % f] = float("nan")
                    row["stable_phase_bases"] = ""
                    row["raw_stable_phases"] = ""
                rows.append(row)
            print("  %-22s %s" % (case["case_id"],
                                  "".join("." if ok_map[t] else "X" for t in GRID_C)))
    finally:
        holder.shutdown()

    pd.DataFrame(S).to_csv(os.path.join(C.D_EXTCAL,
                                        "STEP5K_EXTVAL_STATES_RAW.csv"),
                           index=False, encoding="utf-8")
    pd.DataFrame(P).to_csv(os.path.join(C.D_EXTCAL,
                                        "STEP5K_EXTVAL_PHASES_RAW.csv"),
                           index=False, encoding="utf-8")
    pd.DataFrame(PC).to_csv(os.path.join(C.D_EXTCAL,
                                         "STEP5K_EXTVAL_PHASE_COMPOSITIONS_RAW.csv"),
                            index=False, encoding="utf-8")
    proc = pd.DataFrame(rows)
    proc.to_csv(os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_STATES.csv"),
                index=False, encoding="utf-8")

    moved = C.verify_protected(baseline)
    if moved:
        raise SystemExit("FATAL: protected frozen artifacts changed: %s" % moved)

    n_ok = int(proc.converged.sum())
    C.write_json(os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_LOG.json"), {
        "generated_utc": C.utcnow(),
        "stage": "STEP5K Part A Stage 2 - post-freeze non-Ni literature-composition CALPHAD",
        "engine": "frozen FULL_V1 dry_run_lib + solver_escalation from %s" % C.FULL_V1_CODE,
        "database": lib.DATABASE_NAME, "P_Pa": lib.PRESSURE_PA,
        "N_mol": lib.SYSTEM_SIZE_MOL, "temperatures_C": GRID_C,
        "n_compositions": len(cases), "n_states": n_states, "n_converged": n_ok,
        "n_failed_kept_not_deleted": n_states - n_ok,
        "ni_states_run": 0,
        "is_a_calphad_layer_check_not_an_ML_validation": True,
        "ml_predictions_made_for_these_compositions": 0,
        "used_to_retrain_or_retune_anything": False,
        "compositions": [{"ext_id": c["case_id"],
                          "wt_pct": {e: round(c["wt_pct"][e], 6) for e in c["elements"]},
                          "provenance": c["composition_source"],
                          "purpose": c["purpose"]} for c in cases],
        "excluded_on_purpose": {
            "LIT_19 Cu-0.3%Cr": "available metadata does not distinguish wt.% from "
                                "at.%; spec 5 forbids inferring the basis",
            "LIT_10 Cu-Cr-Zr-Zn-0.45Fe-0.2Ti": "Cr, Zr and Zn contents are not stated, "
                                               "so the composition cannot be built "
                                               "without inventing numbers"},
        "wall_clock_seconds": round(time.time() - t0, 1),
        "protected_artifacts_unchanged": True})
    print("  converged %d / %d states in %.1f s" % (n_ok, n_states, time.time() - t0))
    print("STAGE A2: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
