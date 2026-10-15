# -*- coding: utf-8 -*-
"""
r3_20_postprocess.py
STEP5K-R3 stage 20: turn the runner's checkpoint into the canonical R3 tables.

Three rules govern this stage.

  * A request with no calculated state is PENDING. Pending is not CALPHAD_FAILED and
    is never evidence that a phase is absent (spec 28, 37). Until the engine runs,
    every row is PENDING and the QA note says so in those words.
  * Raw phase instances are preserved with their instance numbers and are written out
    BEFORE any family aggregation, because the aggregation merges exactly the
    multiplicity that Cu-Cr-Ag and Cu-Cr-Mo are here to test (spec 15).
  * Nothing is dropped. Failed states keep their Thermo-Calc error text; states that
    needed solver escalation keep the level they succeeded at, so a reader can see
    which results depended on it.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C

FROZEN_FAMILY = {"FCC_L12": "FCC", "BCC_B2": "BCC", "C15_LAVES": "C15",
                 "LIQUID": "LIQUID"}


def family_of(raw):
    return FROZEN_FAMILY.get(C.base_phase(raw), "OTHER_SOLID")


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K-R3 stage 20: canonical CALPHAD tables")
    print("=" * 78)
    n_prot, _, _ = C.verify_protected()
    print("  %d protected artifacts re-hashed, all unchanged" % n_prot)

    I = C.read_inputs()
    req = I["requests"].copy()
    states, phases, comps = C.raw_states()

    # A checkpoint may only become a scientific table if the runner stamped it on a
    # live engine. Anything else - a stand-in used while the engine was down, a
    # hand-edited file - is refused here rather than quietly flowing into the
    # comparison, the figures and the manuscript text.
    prov_p = os.path.join(C.D_RAW, "ENGINE_PROVENANCE.json")
    provenance = None
    if states is not None and len(states):
        if not os.path.exists(prov_p):
            raise SystemExit(
                "FATAL: 07_calphad_raw/STATES.csv exists but "
                "ENGINE_PROVENANCE.json does not. Only run_r3.py writes that stamp, "
                "and only on a live Thermo-Calc engine. Refusing to build scientific "
                "tables from an unattributed checkpoint.")
        provenance = C.load_json(prov_p)
        if not provenance.get("real_engine") or provenance.get("synthetic"):
            raise SystemExit(
                "FATAL: the checkpoint is stamped synthetic or non-engine "
                "(%s). Synthetic results must never enter the scientific tables."
                % prov_p)
        print("  engine provenance: %s, tc_python %s, host %s, %s"
              % (provenance.get("engine"), provenance.get("tc_python_version"),
                 provenance.get("host"), provenance.get("generated_utc")))

    if states is None or not len(states):
        print("  no calculated state exists yet - writing %d PENDING rows" % len(req))
        states = pd.DataFrame(columns=["state_id", "converged"])
        phases = pd.DataFrame()
        comps = pd.DataFrame()

    st = states.set_index("state_id") if len(states) else pd.DataFrame()
    rows, raw_rows, fam_rows = [], [], []
    for _, q in req.iterrows():
        sid = q.state_id
        r = {"state_id": sid, "condition_id": q.condition_id,
             "source_id": q.source_id, "system": q.system,
             "X_identity": q.X_identity,
             "input_composition_wt_pct": q.composition_wt_pct,
             "temperature_C": q.temperature_C, "temperature_K": q.temperature_K,
             "pressure_Pa": q.pressure_Pa, "system_size_mol": q.system_size_mol,
             "database": q.database}
        if len(st) and sid in st.index:
            s = st.loc[sid]
            conv = bool(s.get("converged", False))
            r["execution_status"] = "CONVERGED" if conv else "FAILED"
            r["converged"] = int(conv)
            r["solver_level"] = s.get("solver_level", np.nan)
            r["error_message"] = s.get("error_message", "")
            r["n_stable_phases"] = s.get("n_stable_phases", np.nan)
            r["sum_phase_mole_fraction"] = s.get("sum_phase_mole_fraction", np.nan)
            r["sum_phase_volume_fraction"] = s.get("sum_phase_volume_fraction",
                                                   np.nan)
            r["phase_fraction_sum_ok"] = s.get("phase_fraction_sum_ok", np.nan)
        else:
            r.update(execution_status="PENDING", converged=np.nan,
                     solver_level=np.nan, error_message="",
                     n_stable_phases=np.nan, sum_phase_mole_fraction=np.nan,
                     sum_phase_volume_fraction=np.nan,
                     phase_fraction_sum_ok=np.nan)

        p = phases[phases.state_id == sid] if len(phases) else pd.DataFrame()
        present = p[p.get("phase_mole_fraction", pd.Series(dtype=float))
                    > C.PRESENCE] if len(p) else p
        fam_counts = {f: 0 for f in C.FAMILIES}
        fam_vol = {f: 0.0 for f in C.FAMILIES}
        fam_mol = {f: 0.0 for f in C.FAMILIES}
        for _, ph in present.iterrows():
            fam = family_of(ph.raw_phase_name)
            fam_counts[fam] += 1
            v = ph.get("phase_volume_fraction", np.nan)
            m = ph.get("phase_mole_fraction", np.nan)
            fam_vol[fam] += 0.0 if pd.isna(v) else float(v)
            fam_mol[fam] += 0.0 if pd.isna(m) else float(m)
            raw_rows.append({
                "state_id": sid, "condition_id": q.condition_id,
                "source_id": q.source_id, "system": q.system,
                "temperature_C": q.temperature_C,
                "raw_phase_name": ph.raw_phase_name,
                "phase_model_base_name": C.base_phase(ph.raw_phase_name),
                "frozen_family": fam,
                "phase_mole_fraction": m, "phase_volume_fraction": v,
                "phase_moles": ph.get("phase_moles", np.nan),
                "is_present_1e4": 1})
        for f in C.FAMILIES:
            r["n_%s_raw_instances" % f] = fam_counts[f]
            r["frac_%s" % f] = fam_vol[f] if r["execution_status"] == "CONVERGED" \
                else np.nan
            fam_rows.append({"state_id": sid, "condition_id": q.condition_id,
                             "system": q.system, "family": f,
                             "n_raw_instances": fam_counts[f],
                             "volume_fraction": fam_vol[f]
                             if r["execution_status"] == "CONVERGED" else np.nan,
                             "mole_fraction": fam_mol[f]
                             if r["execution_status"] == "CONVERGED" else np.nan,
                             "execution_status": r["execution_status"]})
        r["calculated_raw_phase_set"] = ";".join(sorted(
            {C.base_phase(x) for x in present.raw_phase_name})) if len(present) else ""
        r["calculated_raw_instances"] = ";".join(sorted(
            present.raw_phase_name.astype(str))) if len(present) else ""
        r["calculated_family_set"] = ";".join(
            [f for f in C.FAMILIES if fam_counts[f] > 0])
        rows.append(r)

    out = pd.DataFrame(rows)
    raw = pd.DataFrame(raw_rows, columns=[
        "state_id", "condition_id", "source_id", "system", "temperature_C",
        "raw_phase_name", "phase_model_base_name", "frozen_family",
        "phase_mole_fraction", "phase_volume_fraction", "phase_moles",
        "is_present_1e4"])
    fam = pd.DataFrame(fam_rows)

    parq = os.path.join(C.D_PROC, "STEP5K_R3_EXTERNAL_CALPHAD.parquet")
    try:
        out.to_parquet(parq, index=False)
    except Exception as e:                                        # noqa: BLE001
        raise SystemExit("FATAL: could not write the canonical parquet (%s). "
                         "pyarrow is required by the ML interpreter." % e)
    raw.to_csv(os.path.join(C.D_RAW, "STEP5K_R3_EXTERNAL_CALPHAD_RAW_PHASES.csv"),
               index=False, encoding="utf-8")
    fam.to_csv(os.path.join(C.D_PROC, "STEP5K_R3_EXTERNAL_CALPHAD_FAMILY.csv"),
               index=False, encoding="utf-8")
    if len(comps):
        comps.to_csv(os.path.join(C.D_RAW,
                                  "STEP5K_R3_EXTERNAL_CALPHAD_PHASE_COMPOSITIONS.csv"),
                     index=False, encoding="utf-8")

    n = len(out)
    n_conv = int((out.execution_status == "CONVERGED").sum())
    n_fail = int((out.execution_status == "FAILED").sum())
    n_pend = int((out.execution_status == "PENDING").sum())
    esc = out[(out.execution_status == "CONVERGED") & (out.solver_level > 0)]
    by_sys = out.groupby("system").execution_status.value_counts().unstack(
        fill_value=0)

    C.write_json(os.path.join(C.D_PROC, "STEP5K_R3_EXTERNAL_CALPHAD_FREEZE.json"), {
        "generated_utc": C.utcnow(),
        "stage": "STEP5K-R3 stage 20 - canonical external CALPHAD tables",
        "engine": "Thermo-Calc 2025b / TC-Python 2025.2.30 / TCCU6 Cu-Alloys v6.2",
        "conditions": {"pressure_Pa": float(req.pressure_Pa.iloc[0]),
                       "system_size_mol": float(req.system_size_mol.iloc[0]),
                       "basis": "wt.%, Cu balance", "equilibrium_only": True},
        "n_requests": n, "n_converged": n_conv, "n_failed": n_fail,
        "n_pending": n_pend,
        "n_converged_needing_solver_escalation": int(len(esc)),
        "escalated_state_ids": sorted(esc.state_id.astype(str)),
        "per_system": by_sys.to_dict(),
        "elements_used": sorted({e for c in req.composition_wt_pct
                                 for e in json.loads(c)}),
        "ni_states": 0,
        "pending_is_not_failure": ("a PENDING row means no calculation has been "
                                   "submitted; it is never read as phase absence and "
                                   "never scored as CALPHAD_FAILED"),
        "raw_instances_preserved_before_aggregation": True,
        "presence_threshold_mole_fraction": C.PRESENCE,
        "frozen_family_map": FROZEN_FAMILY,
        "request_sha256": I["seal"]["request_sha256"],
        "engine_provenance": provenance,
        "results_are_from_a_live_thermo_calc_engine": bool(provenance),
        "used_for_training_or_tuning": False})

    qa = ["# R3 external CALPHAD QA\n\n",
          "_Generated %s._\n\n" % C.utcnow(),
          "| | n |\n|---|---|\n",
          "| requests | %d |\n| converged | %d |\n| failed | %d |\n| pending | %d |\n"
          % (n, n_conv, n_fail, n_pend),
          "\nPer system:\n\n", by_sys.to_markdown() if len(by_sys) else "(none)\n",
          "\n\n"]
    if n_pend:
        qa.append("**%d requests are PENDING.** No equilibrium has been submitted for "
                  "them. A pending row is not a failed calculation and says nothing "
                  "about whether a phase is stable; the raw-phase, family and "
                  "phase-composition tables are empty for those rows by construction, "
                  "and no agreement metric is computed from them.\n\n" % n_pend)
    if n_fail:
        qa.append("**%d calculations failed.** They are kept with their Thermo-Calc "
                  "error text and scored CALPHAD_FAILED; none was removed, "
                  "interpolated, or replaced by a neighbouring temperature.\n\n"
                  % n_fail)
    if len(esc):
        qa.append("**%d converged states required solver escalation above level 0.** "
                  "They are listed in the freeze file so that any result depending on "
                  "escalation can be identified.\n\n" % len(esc))
    if n_conv:
        bad = out[(out.execution_status == "CONVERGED")
                  & (out.phase_fraction_sum_ok == False)]          # noqa: E712
        qa.append("Phase-fraction conservation failed on %d converged states.\n\n"
                  % len(bad))
    open(os.path.join(C.D_QA, "STEP5K_R3_EXTERNAL_CALPHAD_QA.md"), "w",
         encoding="utf-8").writelines(qa)

    print("  %d requests -> converged %d | failed %d | pending %d"
          % (n, n_conv, n_fail, n_pend))
    print("  raw phase instances written: %d" % len(raw))
    print("STAGE 20: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
