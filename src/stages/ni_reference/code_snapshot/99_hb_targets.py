# -*- coding: utf-8 -*-
"""
99_hb_targets.py
STEP5H-B Stage 2: target construction + TARGET FREEZE + CALPHAD QA (spec 10-18,
21-24). Runs BEFORE any scoring (spec 22, 23).

Processing is the frozen FULL_V1/STEP5F logic:
  - raw phase names preserved verbatim (suffixes #1/#2 carry no identity, spec 13)
  - frozen five-family mapping (spec 11); any unknown raw phase -> OTHER_SOLID
    automatically, with its raw name preserved and documented (spec 12)
  - MOLAR Ni partition (spec 14): n_Ni,alpha = phase_moles_alpha * X_Ni^alpha;
    P_Ni^alpha = n_Ni,alpha / sum_beta n_Ni,beta. The volume-fraction x mole
    fraction form is FORBIDDEN (spec 15) and appears nowhere.
  - conservation QA (spec 16) and five-family volume QA (spec 17)
  - failed states keep their rows with NaN targets and full failure metadata
    (spec 18); all 22,000 formal states remain represented

Outputs:
  03_processed_targets/STEP5H_B_NI_TARGETS.parquet
  03_processed_targets/STEP5H_B_NI_TARGET_FREEZE.json
  04_phase_vocab/STEP5H_B_PHASE_VOCABULARY.csv
  04_phase_vocab/STEP5H_B_PHASE_FAMILY_MAPPING.csv
  05_calphad_qa/STEP5H_B_CALPHAD_QA.md, STEP5H_B_FAILURE_MAP.csv, ...
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hb_common as C

sys.path.insert(0, C.FULL_V1_CODE)

PRESENCE_THRESHOLD = 1e-4
# raw phase model bases already seen anywhere in the frozen project history
KNOWN_BASES = {"FCC_L12", "BCC_B2", "C15_LAVES", "LIQUID",
               "CU6LA_HT", "M2P", "M3P", "MP"}


def read_table(path):
    for suf, fn in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv),
                    (".csv", pd.read_csv)):
        p = path if path.endswith(suf) and os.path.isfile(path) else path + suf
        if os.path.isfile(p):
            return fn(p)
    raise FileNotFoundError(path)


def md_table(df, max_rows=None):
    d = df if max_rows is None else df.head(max_rows)
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in d.columns) + " |")
    return chr(10).join(lines)


def main():
    import fivefamily as ff
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5H-B  Stage 2: target construction, freeze, CALPHAD QA")
    print("=" * 78)

    # ---- load shard exports -------------------------------------------------- #
    states_l, phases_l, pcomp_l = [], [], []
    for s in sorted(glob.glob(os.path.join(C.D_RAW, "STEP5H_B_shard*_EQUILIBRIUM_STATES.csv"))):
        states_l.append(pd.read_csv(s, keep_default_na=False, low_memory=False))
    for pat, tgt in (("STEP5H_B_shard*_PHASES_LONG.*", phases_l),
                     ("STEP5H_B_shard*_PHASE_COMPOSITIONS_LONG.*", pcomp_l)):
        for p in sorted(glob.glob(os.path.join(C.D_RAW, pat))):
            if "_checkpoint" in os.path.basename(p):
                continue
            tgt.append(read_table(p))
    if not states_l:
        raise SystemExit("FATAL: no exported shard tables - run 98 first.")
    states = pd.concat(states_l, ignore_index=True)
    phases = pd.concat(phases_l, ignore_index=True)
    pcomp = pd.concat(pcomp_l, ignore_index=True)
    print("  store rows: states %d | phase rows %d | phase-composition rows %d"
          % (len(states), len(phases), len(pcomp)))

    # ---- formal state grid: EVERY formal state remains represented ----------- #
    comps = pd.read_csv(C.G_NI_COMP_CSV)
    grid = comps[["composition_id", "fewshot_role"]].merge(
        pd.DataFrame({"Temperature_C": [float(t) for t in C.TEMPERATURES_C]}),
        how="cross")
    grid["state_id"] = [("%s__T%04dC" % (c, int(t_)))
                        for c, t_ in zip(grid.composition_id, grid.Temperature_C)]
    assert len(grid) == C.N_STATES and grid.state_id.is_unique

    states["converged_b"] = states["converged"].astype(str).str.lower() == "true"
    st = states[["state_id", "case_id", "Temperature_C", "converged_b",
                 "solver_level_used", "bridge_used", "attempt_count", "final_status",
                 "n_stable_phases", "error_message", "original_error_message"]].copy()
    st = st.rename(columns={"case_id": "composition_id"})
    st["calphad_status"] = np.where(st.converged_b, "CONVERGED", "FAILED")
    t = grid.merge(st.drop(columns=["Temperature_C"]), on=["state_id", "composition_id"],
                   how="left", validate="1:1")
    n_missing = int(t.calphad_status.isna().sum())
    if n_missing:
        t.loc[t.calphad_status.isna(), "calphad_status"] = "NOT_IN_STORE"
    print("  formal grid 22000 | joined from store: %d | missing from store: %d"
          % (C.N_STATES - n_missing, n_missing))

    # ---- raw phase names + frozen five-family mapping ------------------------ #
    conv_ids = set(states.loc[states.converged_b, "state_id"])
    ph = phases[phases.state_id.isin(conv_ids)].copy()
    pc = pcomp[pcomp.state_id.isin(conv_ids)].copy()
    for f_ in (ph, pc):
        f_["phase_model_base_name"] = (f_["raw_phase_name"].astype(str)
                                       .str.replace(r"#\d+$", "", regex=True))
        f_["aggregated_family"] = f_["phase_model_base_name"].map(ff.family_of_5)
    unmapped = sorted(set(ph.loc[ph.aggregated_family.isna(), "phase_model_base_name"]))
    if unmapped:
        raise SystemExit("FATAL: frozen fivefamily mapping returned NaN for %s - "
                         "STOP and document (spec 12 expects OTHER_SOLID default)."
                         % unmapped)
    raw_names = (ph.groupby("state_id")["raw_phase_name"]
                 .apply(lambda s: ";".join(sorted(set(s)))).rename("raw_phase_names"))
    t = t.merge(raw_names, on="state_id", how="left")
    t["raw_phase_names"] = t["raw_phase_names"].fillna("")

    # ---- 5-family volume aggregation ----------------------------------------- #
    agg = (ph.groupby(["state_id", "aggregated_family"], as_index=False)
           .agg(volfrac=("phase_volume_fraction", "sum"),
                moles=("phase_moles", "sum")))
    wide = agg.pivot_table(index="state_id", columns="aggregated_family",
                           values="volfrac", fill_value=0.0)
    for f_ in C.FAMILIES_ORDER:
        if f_ not in wide.columns:
            wide[f_] = 0.0
    wide = wide[C.FAMILIES_ORDER]

    # ---- MOLAR Ni partition (spec 14; volume-fraction form FORBIDDEN) --------- #
    pm = ph.set_index(["state_id", "raw_phase_name"])["phase_moles"]
    pc = pc.copy()
    pc["phase_moles"] = pd.MultiIndex.from_frame(
        pc[["state_id", "raw_phase_name"]]).map(pm)
    pc["element"] = pc["element"].astype(str).str.upper()
    pc_ni = pc[pc.element == "NI"].copy()
    pc_ni["n_Ni_alpha"] = (pc_ni["phase_moles"].astype(float)
                           * pc_ni["element_mole_fraction"].astype(float))
    tot = pc_ni.groupby("state_id")["n_Ni_alpha"].sum().rename("n_Ni_total")
    fam = (pc_ni.groupby(["state_id", "aggregated_family"], as_index=False)["n_Ni_alpha"]
           .sum().merge(tot, on="state_id", how="left"))
    fam["P_Ni"] = np.where(fam["n_Ni_total"] > 0,
                           fam["n_Ni_alpha"] / fam["n_Ni_total"], np.nan)
    part = fam.pivot_table(index="state_id", columns="aggregated_family",
                           values="P_Ni", fill_value=0.0)
    for f_ in C.FAMILIES_ORDER:
        if f_ not in part.columns:
            part[f_] = 0.0
    part = part[C.FAMILIES_ORDER]

    for j, f_ in enumerate(C.FAMILIES_ORDER):
        t["true_%s" % f_] = t.state_id.map(part[f_])
        t["%s_volfrac" % f_] = t.state_id.map(wide[f_]).fillna(0.0)
    failed = t.calphad_status != "CONVERGED"
    t.loc[failed, C.TRUE_COLS] = np.nan
    t["true_topology"] = np.where(
        ~failed,
        [C.FAMILIES_ORDER[i] for i in C.topology_of(t.loc[~failed, C.TRUE_COLS]
                                                    .to_numpy(float))], "")

    # ---- conservation QA (spec 16, 17) --------------------------------------- #
    cv = t[~failed]
    psum = cv[C.TRUE_COLS].sum(axis=1)
    cons_dev = float(np.abs(psum - 1.0).max())
    vsum = cv[["%s_volfrac" % f for f in C.FAMILIES_ORDER]].sum(axis=1)
    vol_dev = float(np.abs(vsum - 1.0).max())
    neg = float(cv[C.TRUE_COLS].to_numpy(float).min())
    print("  Ni partition conservation max dev: %.3e | family volume sum max dev: "
          "%.3e | min component %.1e" % (cons_dev, vol_dev, neg))

    # ---- vocabulary + mapping ------------------------------------------------ #
    ph["is_ni_state"] = True
    voc = (ph.groupby(["raw_phase_name", "phase_model_base_name", "aggregated_family"],
                      as_index=False)
           .agg(n_states=("state_id", "nunique"),
                Tmin=("Temperature_C", "min"), Tmax=("Temperature_C", "max"),
                min_volfrac=("phase_volume_fraction", "min"),
                max_volfrac=("phase_volume_fraction", "max")))
    voc["previously_seen_in_project"] = voc.phase_model_base_name.isin(KNOWN_BASES).astype(int)
    voc.to_csv(os.path.join(C.D_VOCAB, "STEP5H_B_PHASE_VOCABULARY.csv"),
               index=False, encoding="utf-8")
    mapping = (ph.groupby(["phase_model_base_name", "aggregated_family"], as_index=False)
               .agg(n_states=("state_id", "nunique")))
    mapping["previously_seen_in_project"] = \
        mapping.phase_model_base_name.isin(KNOWN_BASES).astype(int)
    mapping["mapped_by_frozen_default_to_OTHER_SOLID"] = \
        (mapping.aggregated_family == "OTHER_SOLID").astype(int)
    mapping.to_csv(os.path.join(C.D_VOCAB, "STEP5H_B_PHASE_FAMILY_MAPPING.csv"),
                   index=False, encoding="utf-8")
    new_bases = sorted(set(mapping.phase_model_base_name) - KNOWN_BASES)
    print("  phase vocabulary: %d raw names, %d bases | previously unseen bases: %s"
          % (voc.raw_phase_name.nunique(), mapping.phase_model_base_name.nunique(),
             new_bases or "none"))

    # ---- failure map + by temperature / region ------------------------------- #
    fmaps = sorted(glob.glob(os.path.join(C.D_CALQA, "STEP5H_B_FAILURE_MAP_shard*.csv")))
    fmap = pd.concat([pd.read_csv(p) for p in fmaps], ignore_index=True) if fmaps \
        else pd.DataFrame()
    fmap.to_csv(os.path.join(C.D_CALQA, "STEP5H_B_FAILURE_MAP.csv"),
                index=False, encoding="utf-8")
    fail_t = t[t.calphad_status != "CONVERGED"].groupby("Temperature_C").size()
    fc = t[t.calphad_status != "CONVERGED"].merge(
        comps[["composition_id", "S_CrNb_wt_pct", "X_wt_pct",
               "R_CrNb_atomic_actual"]], on="composition_id", how="left")
    fail_region = {}
    if len(fc):
        fc["S_q"] = pd.qcut(fc.S_CrNb_wt_pct, 4, duplicates="drop")
        fc["X_q"] = pd.qcut(fc.X_wt_pct, 4, duplicates="drop")
        fail_region = {"by_S_quartile": fc.groupby("S_q", observed=True).size().to_dict(),
                       "by_X_quartile": fc.groupby("X_q", observed=True).size().to_dict()}

    # ---- write the target table ---------------------------------------------- #
    cols = (["state_id", "composition_id", "Temperature_C", "fewshot_role",
             "calphad_status", "raw_phase_names"] + C.TRUE_COLS + ["true_topology"]
            + ["%s_volfrac" % f for f in C.FAMILIES_ORDER]
            + ["solver_level_used", "bridge_used", "attempt_count", "final_status",
               "n_stable_phases", "error_message", "original_error_message"])
    t = t[[c for c in cols if c in t.columns]]
    out = os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGETS.parquet")
    t.to_parquet(out, index=False)

    n_conv = int((t.calphad_status == "CONVERGED").sum())
    n_fail = int((t.calphad_status == "FAILED").sum())
    schema_hash = C.sha256_text("|".join(t.columns))
    partition_def = ("n_Ni_alpha = phase_moles_alpha * X_Ni^alpha ; P_Ni^alpha = "
                     "n_Ni_alpha / sum_beta n_Ni_beta ; family order FCC, BCC, C15, "
                     "OTHER_SOLID, LIQUID ; volume-fraction partitioning FORBIDDEN")
    freeze = {
        "freeze_id": "STEP5H_B_NI_TARGET_FREEZE_V1",
        "generated_utc": C.utcnow(),
        "file": out,
        "sha256": C.sha256_file(out),
        "row_count": int(len(t)),
        "n_formal": C.N_STATES, "n_converged": n_conv, "n_failed": n_fail,
        "n_not_in_store": n_missing,
        "convergence_rate": n_conv / C.N_STATES,
        "schema_columns": list(t.columns), "schema_sha256": schema_hash,
        "phase_vocabulary_sha256": C.sha256_file(
            os.path.join(C.D_VOCAB, "STEP5H_B_PHASE_VOCABULARY.csv")),
        "phase_family_mapping_sha256": C.sha256_file(
            os.path.join(C.D_VOCAB, "STEP5H_B_PHASE_FAMILY_MAPPING.csv")),
        "partition_definition": partition_def,
        "partition_definition_sha256": C.sha256_text(partition_def),
        "conservation_max_dev": cons_dev,
        "family_volume_sum_max_dev": vol_dev,
        "min_target_component": neg,
        "previously_unseen_bases": new_bases,
        "failure_by_temperature": {str(int(k)): int(v) for k, v in fail_t.items()},
        "failure_by_composition_region": fail_region,
        "immutable_from_utc": C.utcnow(),
        "modification_prohibited_after_this_point": True,
    }
    C.write_json(os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGET_FREEZE.json"), freeze)

    # ---- CALPHAD QA md (spec 24) --------------------------------------------- #
    with open(os.path.join(C.D_CALQA, "STEP5H_B_CALPHAD_QA.md"), "w",
              encoding="utf-8") as f:
        f.write("# STEP5H_B_CALPHAD_QA.md\n\ngenerated_utc: %s\n\n" % C.utcnow())
        f.write("## counts\n\n- formal states: %d\n- converged: %d\n- failed: %d\n"
                "- convergence rate: %.4f\n\n" % (C.N_STATES, n_conv, n_fail,
                                                  n_conv / C.N_STATES))
        f.write("## failure by temperature\n\n")
        fbt = pd.DataFrame({"Temperature_C": fail_t.index.astype(int),
                            "failed": fail_t.values}) if len(fail_t) else \
            pd.DataFrame({"Temperature_C": [], "failed": []})
        f.write((md_table(fbt) if len(fbt) else "no failures") + "\n\n")
        f.write("## failure by composition region\n\n%s\n\n"
                % (json.dumps(fail_region, indent=1, default=str) if fail_region
                   else "no failures"))
        f.write("## raw phase vocabulary\n\n" + md_table(voc) + "\n\n")
        f.write("## phase-family mapping\n\n" + md_table(mapping) + "\n\n")
        f.write("previously unseen raw phase bases: %s\n\n"
                % (", ".join("`%s`" % b for b in new_bases) or "none"))
        f.write("Every unseen base kept its exact raw name and was mapped by the "
                "frozen automatic rule: not FCC_L12*/BCC_B2*/C15_LAVES*/LIQUID* -> "
                "OTHER_SOLID (spec 12). No new family was invented; prediction "
                "performance was NOT inspected during mapping.\n\n")
        f.write("## QA\n\n- Ni partition conservation max |sum-1|: %.3e\n"
                "- five-family volume sum max dev: %.3e\n- min target component: %.1e\n"
                "- failed states retained with NaN targets: %d\n- interpolation / "
                "imputation performed: NONE\n" % (cons_dev, vol_dev, neg, n_fail))

    C.write_json(os.path.join(C.D_CALQA, "STEP5H_B_CALPHAD_QA.json"), {
        "generated_utc": C.utcnow(), "n_formal": C.N_STATES, "n_converged": n_conv,
        "n_failed": n_fail, "convergence_rate": n_conv / C.N_STATES,
        "failure_by_temperature": freeze["failure_by_temperature"],
        "failure_by_composition_region": fail_region,
        "conservation_max_dev": cons_dev, "family_volume_sum_max_dev": vol_dev,
        "previously_unseen_bases": new_bases,
        "target_sha256": freeze["sha256"]})
    print("  targets frozen: %d rows | converged %d | failed %d | rate %.4f"
          % (len(t), n_conv, n_fail, n_conv / C.N_STATES))
    print("STAGE 2: PASS (target freeze written BEFORE any scoring)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
