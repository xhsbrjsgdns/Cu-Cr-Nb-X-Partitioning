# -*- coding: utf-8 -*-
"""
72_f5f_process.py
STEP5F Stage C: 5-family aggregation, MOLAR element partitioning, pilot merge,
CALPHAD QA gate (spec 16-22).

Partition definition (spec 19, frozen since FULL_V1):
    n_i,alpha = phase_moles_alpha * X_i^alpha
    P_i^alpha = n_i,alpha / sum_beta n_i,beta
volume-fraction partitioning is FORBIDDEN and appears nowhere.

Aggregation (spec 16, 17, 18): FCC_L12* -> FCC, BCC_B2* -> BCC, C15_LAVES* -> C15,
LIQUID* -> LIQUID, every other non-liquid raw phase -> OTHER_SOLID. Raw names are
recorded verbatim first; nothing is forced into FCC/BCC/C15 by name guessing.
The four pilot-known new bases (CU6LA_HT, M2P, M3P, MP) keep their raw names.

Pilot merge (spec 6): the 250 pilot compositions (50 x 5 elements x 11 T) are
loaded from the frozen STEP5F-PILOT parquet and concatenated - never recomputed.

CALPHAD technical gate (spec 22): per-element convergence >= 98% PASS,
95-98% WARN, < 95% FAIL (stop before ML).

Outputs:
  03_processed/STEP5F_FULL_NEW_ELEMENT_DATA.parquet   110,000 state rows, all targets
  03_processed/STEP5F_FULL_PARTITION_LONG.parquet     per (state, element, family)
  04_calphad_qa/STEP5F_FULL_FAILURE_MAP.csv
  04_calphad_qa/STEP5F_FULL_PHASE_VOCABULARY.csv
  04_calphad_qa/STEP5F_FULL_PHASE_FAMILY_MAPPING.csv
  04_calphad_qa/STEP5F_FULL_CALPHAD_QA.md
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5f_common as C

sys.path.insert(0, C.FULL_V1_CODE)

PRESENCE_THRESHOLD = 1e-4          # frozen FULL_V1 threshold, unchanged


def md_table(df, max_rows=None):
    """Minimal markdown table writer (TC python lacks tabulate)."""
    d = df if max_rows is None else df.head(max_rows)
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in d.columns) + " |")
    return chr(10).join(lines)



def read_table(path):
    """Read a table written by io_store.write_table: pass the exact path (with
    extension) or the stem; .parquet / .csv.gz / .csv are all handled."""
    for suf, fn in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv),
                    (".csv", pd.read_csv)):
        p = path if path.endswith(suf) and os.path.isfile(path) else path + suf
        if os.path.isfile(p):
            return fn(p)
    raise FileNotFoundError(path + ".{parquet|csv.gz|csv}")


def load_shard_tables():
    states, phases, pcomp = [], [], []
    for s in sorted(glob.glob(os.path.join(C.D_RAW, "STEP5F_FULL_shard*_EQUILIBRIUM_STATES.csv"))):
        states.append(pd.read_csv(s, keep_default_na=False, low_memory=False))
    for pat, tgt in (("STEP5F_FULL_shard*_PHASES_LONG.*", phases),
                     ("STEP5F_FULL_shard*_PHASE_COMPOSITIONS_LONG.*", pcomp)):
        for p in sorted(glob.glob(os.path.join(C.D_RAW, pat))):
            if p.endswith((".csv", ".sqlite")) or "_checkpoint" in os.path.basename(p):
                continue
            tgt.append(read_table(p))
    if not states:
        raise SystemExit("FATAL: no exported shard tables under 02_raw_CALPHAD - run 71 first.")
    return (pd.concat(states, ignore_index=True),
            pd.concat(phases, ignore_index=True) if phases else pd.DataFrame(),
            pd.concat(pcomp, ignore_index=True) if pcomp else pd.DataFrame())


def process_new_states(states, phases, pcomp, comps):
    import fivefamily as ff
    states["converged_b"] = states["converged"].astype(str).str.lower() == "true"
    meta = comps.set_index("composition_id")
    states["element"] = states["case_id"].map(meta["X_element"])
    conv_ids = set(states.loc[states.converged_b, "state_id"])

    phases = phases[phases.state_id.isin(conv_ids)].copy()
    pcomp = pcomp[pcomp.state_id.isin(conv_ids)].copy()
    if "phase_model_base_name" not in phases.columns:
        phases["phase_model_base_name"] = (phases["raw_phase_name"].astype(str)
                                           .str.replace(r"#\d+$", "", regex=True))
    phases["aggregated_family"] = phases["phase_model_base_name"].map(ff.family_of_5)
    pcomp["phase_model_base_name"] = (pcomp["raw_phase_name"].astype(str)
                                      .str.replace(r"#\d+$", "", regex=True))
    pcomp["aggregated_family"] = pcomp["phase_model_base_name"].map(ff.family_of_5)

    agg = (phases.groupby(["state_id", "aggregated_family"], as_index=False)
           .agg(volfrac=("phase_volume_fraction", "sum"),
                molefrac=("phase_mole_fraction", "sum"),
                moles=("phase_moles", "sum")))
    wide = agg.pivot_table(index="state_id", columns="aggregated_family",
                           values="volfrac", fill_value=0.0)
    widem = agg.pivot_table(index="state_id", columns="aggregated_family",
                            values="molefrac", fill_value=0.0)
    for f in C.FAMILIES_ORDER:
        if f not in wide.columns:
            wide[f] = 0.0
        if f not in widem.columns:
            widem[f] = 0.0
    wide, widem = wide[C.FAMILIES_ORDER], widem[C.FAMILIES_ORDER]

    # MOLAR element partitioning (spec 19)
    pm = phases.set_index(["state_id", "raw_phase_name"])["phase_moles"]
    pc = pcomp.copy()
    pc["phase_moles"] = pd.MultiIndex.from_frame(
        pc[["state_id", "raw_phase_name"]]).map(pm)
    pc["element"] = pc["element"].astype(str).str.upper()
    pc["n_i_alpha"] = pc["phase_moles"].astype(float) * pc["element_mole_fraction"].astype(float)
    tot = pc.groupby(["state_id", "element"])["n_i_alpha"].sum().rename("n_i_total")
    fam = (pc.groupby(["state_id", "element", "aggregated_family"], as_index=False)
           ["n_i_alpha"].sum()).merge(tot, on=["state_id", "element"], how="left")
    fam["P"] = np.where(fam["n_i_total"] > 0, fam["n_i_alpha"] / fam["n_i_total"], np.nan)
    fam.to_parquet(os.path.join(C.D_PROC, "STEP5F_FULL_PARTITION_LONG.parquet"), index=False)

    part = fam.pivot_table(index=["state_id", "element"], columns="aggregated_family",
                           values="P", fill_value=0.0).reset_index()
    for f in C.FAMILIES_ORDER:
        if f not in part.columns:
            part[f] = 0.0

    out = states[["state_id", "case_id", "element", "Temperature_C", "Temperature_K",
                  "converged_b", "solver_level_used", "bridge_used", "attempt_count",
                  "final_status", "n_stable_phases"]].copy()
    out = out.rename(columns={"case_id": "composition_id", "converged_b": "state_converged"})
    out["state_converged"] = out["state_converged"].astype(int)
    out["X_element"] = out["element"]
    out["source"] = "STEP5F_FULL_new"
    for col in ("Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct", "S_CrNb_wt_pct",
                "R_CrNb_atomic_actual"):
        out[col] = pd.to_numeric(out["composition_id"].map(meta[col]), errors="coerce")

    for f in C.FAMILIES_ORDER:
        out["%s_volfrac" % f] = out["state_id"].map(wide[f])
        out["%s_molefrac" % f] = out["state_id"].map(widem[f])
        out["%s_presence" % f] = (out["%s_volfrac" % f].fillna(0) > PRESENCE_THRESHOLD).astype(int)
    idx = part.set_index(["state_id", "element"])
    for f in C.FAMILIES_ORDER:
        out["CR_P_%s" % f] = pd.MultiIndex.from_arrays(
            [out.state_id, ["CR"] * len(out)]).map(idx[f])
        out["NB_P_%s" % f] = pd.MultiIndex.from_arrays(
            [out.state_id, ["NB"] * len(out)]).map(idx[f])
        out["X_P_%s" % f] = pd.MultiIndex.from_arrays(
            [out.state_id, out.element.astype(str)]).map(idx[f])

    fail = out.state_converged == 0
    tcols = [c for c in out.columns if c.endswith(("_volfrac", "_molefrac")) or "_P_" in c]
    out.loc[fail, tcols] = np.nan
    out.loc[fail, [c for c in out.columns if c.endswith("_presence")]] = 0
    return out, phases, pcomp


def align_pilot_schema(pilot, new):
    """Project the pilot parquet onto the new-state schema (identical columns),
    then tag the provenance of both frames."""
    cols = [c for c in new.columns if c != "source"]
    miss = [c for c in cols if c not in pilot.columns]
    if miss:
        raise SystemExit("FATAL: pilot parquet lacks columns %s - schema drift" % miss)
    p = pilot[cols].copy()
    p["source"] = "STEP5F_PILOT_reused"
    return p


def main():
    import fivefamily as ff
    print("=" * 78)
    print("STEP5F FULL  Stage C: aggregation, molar partitioning, pilot merge, QA gate")
    print("=" * 78)
    states, phases, pcomp = load_shard_tables()
    comps = pd.read_csv(os.path.join(C.D_COMP, "STEP5F_FULL_NEW_COMPOSITIONS.csv"))
    print("  new states %d | phase rows %d | phase-composition rows %d"
          % (len(states), len(phases), len(pcomp)))

    new, ph_conv, _ = process_new_states(states, phases, pcomp, comps)
    cv = new[new.state_converged == 1]
    v = cv[["%s_volfrac" % f for f in C.FAMILIES_ORDER]].sum(axis=1)
    dev5 = float(np.abs(v.dropna() - 1.0).max())
    worst = {}
    for role in ("CR", "NB", "X"):
        s = cv[["%s_P_%s" % (role, f) for f in C.FAMILIES_ORDER]].sum(axis=1).dropna()
        worst[role] = float(np.abs(s - 1.0).max()) if len(s) else float("nan")
    print("  5-family volume sum max dev: %.3e | partition conservation: %s"
          % (dev5, {k: "%.2e" % x for k, x in worst.items()}))

    # ------------------------------------------------------- pilot merge -- #
    pilot = pd.read_parquet(C.PILOT_STATE_RESULTS)
    pilot = pilot[pilot.element.isin(C.SELECTED_ELEMENTS)].copy()
    assert pilot.composition_id.nunique() == 250 and len(pilot) == 2750, \
        (pilot.composition_id.nunique(), len(pilot))
    pilot_al = align_pilot_schema(pilot, new)
    both = pd.concat([new, pilot_al], ignore_index=True)
    both["source"] = both["source"].fillna("STEP5F_PILOT_reused")
    both.to_parquet(os.path.join(C.D_PROC, "STEP5F_FULL_NEW_ELEMENT_DATA.parquet"), index=False)
    print("  merged: %d new states + %d pilot states = %d rows -> STEP5F_FULL_NEW_ELEMENT_DATA.parquet"
          % (len(new), len(pilot_al), len(both)))
    assert len(both) == C.N_FORMAL_STATES, (len(both), C.N_FORMAL_STATES)

    # ---------------------------------------------- phase vocabulary (18) -- #
    ph = ph_conv.copy()
    ph["element"] = ph["state_id"].map(new.set_index("state_id")["element"])
    voc = (ph.groupby(["element", "raw_phase_name", "phase_model_base_name",
                       "aggregated_family"], as_index=False)
           .agg(n_states=("state_id", "nunique"),
                Tmin=("Temperature_C", "min"), Tmax=("Temperature_C", "max"),
                min_volfrac=("phase_volume_fraction", "min"),
                max_volfrac=("phase_volume_fraction", "max")))
    voc.to_csv(os.path.join(C.D_CALQA, "STEP5F_FULL_PHASE_VOCABULARY.csv"),
               index=False, encoding="utf-8")
    mapping = (ph.groupby(["phase_model_base_name", "aggregated_family"], as_index=False)
               .agg(n_states=("state_id", "nunique"),
                    elements=("element", lambda s: ";".join(sorted(set(map(str, s)))))))
    mapping["is_known_solid_model"] = mapping["phase_model_base_name"].isin(
        ["FCC_L12", "BCC_B2", "C15_LAVES", "LIQUID"]).astype(int)
    mapping["mapped_by_default_to_OTHER_SOLID"] = (
        mapping.aggregated_family == "OTHER_SOLID").astype(int)
    mapping.to_csv(os.path.join(C.D_CALQA, "STEP5F_FULL_PHASE_FAMILY_MAPPING.csv"),
                   index=False, encoding="utf-8")
    known = {"FCC_L12", "BCC_B2", "C15_LAVES", "LIQUID"}
    pilot_known_new = {"CU6LA_HT", "M2P", "M3P", "MP"}
    new_bases = sorted(set(mapping.phase_model_base_name) - known - pilot_known_new)
    unmapped = mapping[~mapping.aggregated_family.isin(C.FAMILIES_ORDER)]
    print("  phase vocabulary: %d raw names, %d bases; unexpected new bases: %s"
          % (voc.raw_phase_name.nunique(), mapping.phase_model_base_name.nunique(),
             new_bases or "none"))

    # ------------------------------------------------------ failure map -- #
    fmaps = sorted(glob.glob(os.path.join(C.D_CALQA, "STEP5F_FULL_FAILURE_MAP_shard*.csv")))
    fmap = pd.concat([pd.read_csv(p) for p in fmaps], ignore_index=True) if fmaps else pd.DataFrame()
    fmap.to_csv(os.path.join(C.D_CALQA, "STEP5F_FULL_FAILURE_MAP.csv"),
                index=False, encoding="utf-8")

    # ------------------------------------- per-element convergence gate -- #
    rows = []
    for el, g in both.groupby("element"):
        n_formal = C.N_TOTAL_PER_ELEMENT * len(C.TEMPERATURES_C)
        conv = int((g.state_converged == 1).sum())
        rate = conv / n_formal
        status = ("PASS" if rate >= C.GATE_WARN_BELOW
                  else "WARN" if rate >= C.GATE_FAIL_BELOW else "FAIL")
        rows.append({"element": el, "compositions": C.N_TOTAL_PER_ELEMENT,
                     "formal_states": n_formal, "converged_states": conv,
                     "failed_states": n_formal - conv, "convergence_rate": rate,
                     "gate_status": status})
    feas = pd.DataFrame(rows)
    feas.to_csv(os.path.join(C.D_CALQA, "STEP5F_FULL_CONVERGENCE_GATE.csv"),
                index=False, encoding="utf-8")
    print(feas.to_string(index=False))

    liq = (both[both.state_converged == 1]
           .assign(has_liq=lambda d: (d.LIQUID_volfrac.fillna(0) > PRESENCE_THRESHOLD).astype(int))
           .groupby(["element", "Temperature_C"], as_index=False)
           .agg(states=("state_id", "size"), liquid_states=("has_liq", "sum")))
    liq["liquid_fraction"] = liq.liquid_states / liq.states.replace(0, np.nan)
    liq.to_csv(os.path.join(C.D_CALQA, "STEP5F_FULL_LIQUID_CHECK.csv"),
               index=False, encoding="utf-8")

    gate_fail = feas[feas.gate_status == "FAIL"]
    gate_warn = feas[feas.gate_status == "WARN"]
    hits = C.ni_scan([("new_element_data", both), ("vocabulary", voc),
                      ("failure_map", fmap)])
    C.write_json(os.path.join(C.D_PROC, "STEP5F_FULL_PROCESSING_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "states_total": int(len(both)),
        "states_new": int(len(new)), "states_pilot_reused": int(len(pilot_al)),
        "converged": int((both.state_converged == 1).sum()),
        "failed": int((both.state_converged == 0).sum()),
        "partition_definition": ("n_i_alpha = phase_moles_alpha * X_i^alpha ; "
                                 "P_i^alpha = n_i_alpha / sum_beta n_i_beta "
                                 "(volume-fraction-based partitioning FORBIDDEN)"),
        "presence_threshold": PRESENCE_THRESHOLD,
        "five_family_sum_max_dev": dev5,
        "partition_sum_max_dev": worst,
        "phase_family_mapping": json.loads(mapping.to_json(orient="records")),
        "n_raw_phase_names": int(voc.raw_phase_name.nunique()),
        "n_model_bases": int(mapping.phase_model_base_name.nunique()),
        "unexpected_new_bases_beyond_pilot": new_bases,
        "unmapped_bases": json.loads(unmapped.to_json(orient="records")),
        "convergence_gate": json.loads(feas.to_json(orient="records")),
        "gate_failures": json.loads(gate_fail.to_json(orient="records")),
        "gate_warnings": json.loads(gate_warn.to_json(orient="records")),
        "failure_rows": int(len(fmap)),
        "ni_scan_hits": hits,
        "new_element_data_sha256": C.sha256_file(
            os.path.join(C.D_PROC, "STEP5F_FULL_NEW_ELEMENT_DATA.parquet"))})

    # ------------------------------------------------- CALPHAD QA md ----- #
    with open(os.path.join(C.D_CALQA, "STEP5F_FULL_CALPHAD_QA.md"), "w", encoding="utf-8") as f:
        f.write("# STEP5F_FULL_CALPHAD_QA.md\n\n")
        f.write("generated_utc: %s\n\n" % C.utcnow())
        f.write("- states: %d new + %d pilot reused = %d (expected %d)\n"
                % (len(new), len(pilot_al), len(both), C.N_FORMAL_STATES))
        f.write("- converged %d / failed %d\n"
                % (int((both.state_converged == 1).sum()),
                   int((both.state_converged == 0).sum())))
        f.write("- partition definition: molar (spec 19); conservation max dev %s\n"
                % {k: "%.2e" % x for k, x in worst.items()})
        f.write("- 5-family volume sum max dev: %.3e\n\n" % dev5)
        f.write("## convergence gate (spec 22)\n\n" + md_table(feas) + "\n\n")
        f.write("## phase family mapping\n\n" + md_table(mapping) + "\n\n")
        f.write("unexpected new bases beyond the pilot-known four: %s\n\n"
                % (", ".join("`%s`" % b for b in new_bases) or "none"))
        f.write("## failure map\n\n%d retained failure rows (spec 21)\n" % len(fmap))
        if len(fmap):
            f.write("\n" + md_table(fmap, 50) + "\n")
        f.write("\nNi token scan: %s\n" % (hits or "CLEAN"))

    print("\nSTAGE C: %s" % ("FAIL" if (len(gate_fail) or unmapped.size or hits)
                             else "PASS"))
    return 1 if (len(gate_fail) or unmapped.size or hits) else 0


if __name__ == "__main__":
    sys.exit(main())
