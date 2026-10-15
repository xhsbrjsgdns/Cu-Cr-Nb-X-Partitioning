# -*- coding: utf-8 -*-
"""
30_step5a_audit.py
STEP5A Stage A + Stage B formal audit.

Stage A: read FULL_V1 final status
  - POSTPROCESS RECOVERY STATUS == PASS, 75 PASS / 0 WARN / 0 FAIL
  - BCC_B2#3: label states == 4 (MO, 950 C), simultaneous >=3 BCC instances == 0
Stage B: hash the 6 frozen core deliverables, confirm unchanged.

Also emits 07_qa/INPUT_COLUMN_AUDIT.md (authoritative column roles, from the
actual parquet schema + FULL_V1_SCHEMA.json).

READ-ONLY with respect to 03_full_CALPHAD/FULL_V1.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq

FULL_V1 = r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1"
STEP5A = r"E:\Cu_CALPHAD_ML\04_ML\STEP5A_JMST_ML_FOUNDATION_V2"
QA = os.path.join(STEP5A, "07_qa")
MANIFEST = os.path.join(STEP5A, "09_manifest")

EXPECTED_ROWS = {
    "03_processed/FULL_V1_ML_TARGET_ALL.parquet": 550_000,
    "03_processed/FULL_V1_ML_READY_CONVERGED.parquet": 548_432,
    "01_compositions/FULL_V1_COMPOSITIONS.parquet": 50_000,
    "03_processed/FULL_V1_DERIVED_DESCRIPTORS.parquet": 50_000,
    "04_qa/FULL_V1_FAILURE_MAP.parquet": 1_568,
}

CORE_FROZEN = {
    "ML_TARGET_ALL",
    "ML_READY_CONVERGED",
    "DERIVED_DESCRIPTORS",
    "FAILURE_MAP",
    "COMPOSITIONS_PARQUET",
    "TOPOLOGY_SUMMARY",
}

failures = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(name)
    print(f"[{status}] {name} {detail}")
    return {"check": name, "status": status, "detail": detail}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    results = []
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---------------- Stage A: recovery status ----------------
    rec_path = os.path.join(FULL_V1, "04_qa", "FULL_V1_POSTPROCESS_RECOVERY_SUMMARY.json")
    rec = json.load(open(rec_path, encoding="utf-8"))
    results.append(check("A1 recovery status == PASS", rec["status"] == "PASS", rec["status"]))
    checks = rec["checks"]
    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    results.append(check(
        "A2 recovery checks 75 PASS / 0 WARN / 0 FAIL",
        (len(checks), n_pass, n_warn, n_fail) == (75, 75, 0, 0),
        f"total={len(checks)} pass={n_pass} warn={n_warn} fail={n_fail}"))

    st = rec["split_totals"]
    results.append(check("A3 BCC_B2#3 label states == 4", st["BCC_B2_3_label_states"] == 4,
                         str(st["bcc3_states"])))
    results.append(check("A4 states with >=3 simultaneous BCC instances == 0",
                         st["BCC_ge3_simultaneous"] == 0))
    results.append(check("A5 every BCC_B2#3 state has exactly 2 simultaneous instances",
                         all(v == 2 for v in st["bcc3_instance_counts"].values()),
                         str(st["bcc3_instance_counts"])))
    results.append(check("A6 all BCC3 states MO @ 950 C",
                         all("MO" in s and "T0950C" in s for s in st["bcc3_states"])))
    results.append(check("A7 thermo_calc_invoked == false", rec["thermo_calc_invoked"] is False))
    results.append(check("A8 Ni occurrences == 0", rec["nickel_occurrences"] == 0))

    # failure pattern from the frozen recovery summary
    results.append(check("A9 converged == 548432", rec["converged"] == 548_432))
    results.append(check("A10 failed == 1568", rec["failed"] == 1_568))
    results.append(check("A11 failed families == [MO]", rec["failed_families"] == ["MO"],
                         str(rec["failed_families"])))
    results.append(check("A12 failed temperatures == [950]", rec["failed_temperatures_C"] == [950.0],
                         str(rec["failed_temperatures_C"])))
    conv = 100.0 * rec["converged"] / (rec["converged"] + rec["failed"])
    results.append(check("A13 convergence == 99.7149%", abs(conv - 99.7149) < 1e-3, f"{conv:.4f}%"))
    results.append(check("A14 11 temperatures frozen grid",
                         rec["temperatures_present_C"] == [300.0, 350.0, 400.0, 450.0, 500.0,
                                                          600.0, 700.0, 800.0, 900.0, 950.0, 1000.0]))

    # ---------------- Stage B: row counts + hashes ----------------
    for rel, expected in EXPECTED_ROWS.items():
        p = os.path.join(FULL_V1, rel.replace("/", os.sep))
        n = pq.ParquetFile(p).metadata.num_rows
        results.append(check(f"B rows {rel} == {expected}", n == expected, f"found {n}"))

    hash_rows = pd.read_csv(os.path.join(FULL_V1, "06_manifest", "FULL_V1_POSTPROCESS_FILE_HASHES.csv"))
    hash_report = []
    for _, r in hash_rows.iterrows():
        p = os.path.join(FULL_V1, r["path"].replace("/", os.sep))
        actual = sha256(p)
        size = os.path.getsize(p)
        ok = (actual == r["sha256"]) and (size == int(r["bytes"]))
        core = r["deliverable"] in CORE_FROZEN
        hash_report.append({
            "deliverable": r["deliverable"], "path": r["path"], "core_frozen": core,
            "expected_sha256": r["sha256"], "actual_sha256": actual,
            "expected_bytes": int(r["bytes"]), "actual_bytes": size,
            "match": ok,
        })
        results.append(check(f"B hash {r['deliverable']} unchanged", ok, actual[:16]))

    # ML_READY quick structural facts (read-only)
    mlr = pd.read_parquet(
        os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet"),
        columns=["state_id", "composition_id", "family", "sampling_domain", "Temperature_C",
                 "state_converged", "X_element"],
    )
    results.append(check("B ML_READY state_id unique", mlr["state_id"].is_unique))
    results.append(check("B ML_READY composition_id unique == 50000",
                         mlr["composition_id"].nunique() == 50_000))
    results.append(check("B ML_READY all converged", (mlr["state_converged"] == 1).all()))
    rows_per_comp = mlr.groupby("composition_id").size()
    n10 = int((rows_per_comp == 10).sum())
    n11 = int((rows_per_comp == 11).sum())
    results.append(check("B ML_READY rows/composition == 11 (or 10 for MO@950 FAILED)",
                         set(rows_per_comp.unique()) <= {10, 11} and n10 + n11 == 50_000,
                         f"11 rows: {n11}, 10 rows: {n10}"))
    mo10 = mlr[mlr["composition_id"].map(rows_per_comp) == 10]
    results.append(check("B all 10-row compositions are MO", (mo10["family"] == "MO").all()))
    dom_counts = mlr.drop_duplicates("composition_id")["sampling_domain"].value_counts().to_dict()
    results.append(check("B composition domain counts A35000/B10000/ANCHOR5000",
                         dom_counts == {"A": 35_000, "B": 10_000, "ANCHOR": 5_000}, str(dom_counts)))
    del mlr

    # ---------------- INPUT_COLUMN_AUDIT.md ----------------
    schema = json.load(open(os.path.join(FULL_V1, "06_manifest", "FULL_V1_SCHEMA.json"), encoding="utf-8"))
    schema_roles = {c["name"]: c["role"] for c in schema["files"]["FULL_V1_ML_TARGET_ALL"]["columns"]}
    actual_cols = pq.read_schema(
        os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet")).names

    FORBIDDEN_FEATURES = {
        "state_id", "composition_id", "family", "X_element", "sampling_domain",
        "state_converged", "solver_level_used", "bridge_used",
        "phase_count", "has_multiple_FCC_instances", "has_multiple_BCC_instances",
        "has_liquid",
    }
    FEATURE_COLS = {"Temperature_C", "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
                    "S_CrNb_wt_pct", "CrNb_atomic_ratio"}
    PRESENCE_TARGETS = {f"{f}_presence" for f in ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]}
    FRACTION_TARGETS = {f"{f}_volfrac" for f in ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]}
    PART_TARGETS = {f"{e}_P_{f}" for e in ["CR", "NB", "X"]
                    for f in ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]}

    audit_rows = []
    for c in actual_cols:
        srole = schema_roles.get(c, "(not in schema; extra column in parquet)")
        if c in FEATURE_COLS:
            role, feat, tgt, meta, excl = "feature (raw composition/temperature)", "YES", "no", "no", ""
            if c == "Temperature_C":
                excl = "Temperature_K excluded as redundant (exact linear map); Temperature_C kept per PCC priority"
        elif c == "Temperature_K":
            role, feat, tgt, meta, excl = "derived from Temperature_C", "no", "no", "no", \
                "exactly linear with Temperature_C; usable inside physics formulas (T_over_Tm_mix) but not as a separate raw feature"
        elif c in PRESENCE_TARGETS:
            role, feat, tgt, meta, excl = "TARGET Task A (presence)", "no", "YES", "no", ""
        elif c in FRACTION_TARGETS:
            role, feat, tgt, meta, excl = "TARGET Task B (phase volume fraction)", "no", "YES", "no", ""
        elif c in PART_TARGETS:
            role, feat, tgt, meta, excl = "TARGET Tasks C/D/E (elemental partition)", "no", "YES", "no", ""
        elif c.endswith("_AUX"):
            role, feat, tgt, meta, excl = "auxiliary diagnostic", "no", "no", "YES", \
                "AUX: diagnostics/post-hoc only; never a feature, never a core target, never summed with the 5 families"
        elif c.endswith("_molefrac") or c.endswith("_total_volume_fraction") or c.endswith("_total_mole_fraction") \
                or c.endswith("_n_instances"):
            role, feat, tgt, meta, excl = "internal duplicate/derivative of a target", "no", "no", "YES", \
                "duplicate of *_volfrac targets or instance counts; excluded from features (target-derived)"
        elif c in FORBIDDEN_FEATURES:
            if c in ("family", "X_element"):
                role, feat, tgt, meta, excl = "metadata", "no", "no", "YES", \
                    ("family: split stratification / error reporting / QA only, never a feature; "
                     "X_element: element-property lookup + reporting only, never one-hot") if c in ("family", "X_element") else ""
            else:
                role, feat, tgt, meta, excl = "forbidden metadata/solver/leakage column", "no", "no", "YES", \
                    "explicitly forbidden as model input (task book sec.12)"
        elif c in ("solid_volume_fraction", "dominant_X_partition_family",
                   "dominant_X_solid_partition_family"):
            role, feat, tgt, meta, excl = "target-derived diagnostic", "no", "no", "YES", \
                "derived from targets; forbidden as feature; dominant-family labels NOT a primary task (sec.65)"
        elif c.startswith("X_Psolid_"):
            role, feat, tgt, meta, excl = "solid-renormalised X partition (secondary target view)", "no", "no", "YES", \
                "not a primary Step5A target; primary targets are X_P_* over 5 families"
        else:
            role, feat, tgt, meta, excl = f"schema role: {srole}", "no", "no", "YES", "unclassified - excluded by default"
        audit_rows.append({
            "actual_column_name": c, "schema_role": srole, "role": role,
            "source": "FULL_V1_ML_READY_CONVERGED.parquet",
            "used_as_feature": feat, "used_as_target": tgt, "used_as_metadata": meta,
            "excluded_reason": excl,
        })

    comp_cols = pq.read_schema(
        os.path.join(FULL_V1, "01_compositions", "FULL_V1_COMPOSITIONS.parquet")).names
    for c in comp_cols:
        if c in ("composition_id", "family", "X_element", "sampling_domain"):
            continue  # already audited above
        used_feat = "YES" if c in ("Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
                                   "S_CrNb_wt_pct") else "no"
        excl = ""
        if c.endswith("_exact"):
            excl = "full-precision pre-rounding value; sec.28 requires the 4-decimal wt.% actually fed to Thermo-Calc"
        elif c in ("Cu_at_pct", "Cr_at_pct", "Nb_at_pct", "X_at_pct"):
            excl = "STEP5A recomputes atomic fractions from the 4-decimal wt.% itself (WT_TO_ATOMIC_CONVERSION_QA.md)"
        elif c in ("sampling_method", "seed", "sobol_skip", "sobol_global_index",
                   "R_CrNb_atomic_target", "R_CrNb_atomic_error", "anchor_source",
                   "anchor_perturbation_pct_Cr", "anchor_perturbation_pct_Nb",
                   "anchor_perturbation_pct_X", "composition_sum_wt_pct"):
            excl = "sampling/provenance metadata, not a feature"
        audit_rows.append({
            "actual_column_name": c, "schema_role": "composition-table column", "role": "composition metadata/feature source",
            "source": "FULL_V1_COMPOSITIONS.parquet",
            "used_as_feature": used_feat, "used_as_target": "no", "used_as_metadata": "no" if used_feat == "YES" else "YES",
            "excluded_reason": excl,
        })

    dd_cols = pq.read_schema(
        os.path.join(FULL_V1, "03_processed", "FULL_V1_DERIVED_DESCRIPTORS.parquet")).names
    for c in dd_cols:
        audit_rows.append({
            "actual_column_name": c, "schema_role": "derived descriptor",
            "role": "CALPHAD-derived descriptor",
            "source": "FULL_V1_DERIVED_DESCRIPTORS.parquet",
            "used_as_feature": "no", "used_as_target": "no", "used_as_metadata": "YES",
            "excluded_reason": "sec.13: CALPHAD-derived -> target leakage if used as Step5A model input; post-hoc interpretation/error stratification only",
        })

    pd.DataFrame(audit_rows).to_csv(os.path.join(QA, "INPUT_COLUMN_AUDIT.csv"), index=False, encoding="utf-8")

    n_feat = sum(1 for r in audit_rows if r["used_as_feature"] == "YES")
    n_tgt = sum(1 for r in audit_rows if r["used_as_target"] == "YES")
    with open(os.path.join(QA, "INPUT_COLUMN_AUDIT.md"), "w", encoding="utf-8") as f:
        f.write("# STEP5A INPUT COLUMN AUDIT\n\n")
        f.write(f"generated_utc: {ts}\n\n")
        f.write("Authoritative rule (task book sec.9): actual column names come from\n"
                "`FULL_V1_SCHEMA.json` + the real parquet schemas. Machine-readable version:\n"
                "`INPUT_COLUMN_AUDIT.csv` (same folder).\n\n")
        f.write(f"- ML_READY_CONVERGED columns audited: {len(actual_cols)}\n")
        f.write(f"- COMPOSITIONS columns audited: {len(comp_cols)}\n")
        f.write(f"- DERIVED_DESCRIPTORS columns audited (all excluded from ML input): {len(dd_cols)}\n")
        f.write(f"- columns marked used_as_feature=YES: {n_feat}\n")
        f.write(f"- columns marked used_as_target=YES: {n_tgt}\n\n")
        f.write("## Raw feature columns allowed from the frozen tables (F1 base)\n\n")
        f.write("Temperature_C, Cu_wt_pct, Cr_wt_pct, Nb_wt_pct, X_wt_pct, S_CrNb_wt_pct, CrNb_atomic_ratio\n\n")
        f.write("(Temperature_K may appear inside physics formulas such as T_over_Tm_mix but is not a separate raw feature.)\n\n")
        f.write("## Primary targets (actual names)\n\n")
        f.write("- Presence (Task A): FCC_presence, BCC_presence, C15_presence, OTHER_SOLID_presence, LIQUID_presence\n")
        f.write("- Phase volume fractions (Task B): FCC_volfrac, BCC_volfrac, C15_volfrac, OTHER_SOLID_volfrac, LIQUID_volfrac\n")
        f.write("- Cr partition (Task C): CR_P_FCC, CR_P_BCC, CR_P_C15, CR_P_OTHER_SOLID, CR_P_LIQUID\n")
        f.write("- Nb partition (Task D): NB_P_FCC, NB_P_BCC, NB_P_C15, NB_P_OTHER_SOLID, NB_P_LIQUID\n")
        f.write("- X partition (Task E): X_P_FCC, X_P_BCC, X_P_C15, X_P_OTHER_SOLID, X_P_LIQUID (BASELINE rows NaN, excluded)\n\n")
        f.write("## Hard exclusions\n\n")
        f.write("- All *_AUX columns: diagnostics only (sec.7).\n")
        f.write("- All *_molefrac / *_total_* / *_n_instances: internal duplicates or instance counts.\n")
        f.write("- state_id, composition_id, family, X_element, sampling_domain, state_converged, solver_level_used, bridge_used (sec.12).\n")
        f.write("- phase_count, has_multiple_FCC_instances, has_multiple_BCC_instances, has_liquid: post-hoc stratification metadata only (sec.12, sec.80).\n")
        f.write("- dominant_X_partition_family / dominant_X_solid_partition_family: not primary targets (sec.65).\n")
        f.write("- X_Psolid_*: secondary solid-renormalised view, not a primary Step5A target.\n")
        f.write("- FULL_V1_DERIVED_DESCRIPTORS (every column): CALPHAD-derived, leakage risk (sec.13).\n")
        f.write("- *_exact columns: sec.28 requires the 4-decimal wt.% actually fed to Thermo-Calc.\n")

    # ---------------- persist Stage A/B audit ----------------
    overall = "PASS" if not failures else "FAIL"
    audit = {
        "stage": "STEP5A Stage A + Stage B audit",
        "generated_utc": ts,
        "status": overall,
        "n_checks": len(results),
        "n_fail": len(failures),
        "failures": failures,
        "checks": results,
        "core_file_hashes": hash_report,
        "bcc3_final_interpretation": {
            "raw_label_BCC_B2#3_states": 4,
            "all_MO_950C": True,
            "simultaneous_instances_per_state": 2,
            "states_with_ge3_simultaneous_BCC": 0,
            "note": "#n is a Thermo-Calc composition-set label, NOT a physical phase ordinal. Frozen; STEP5A does not modify.",
        },
    }
    os.makedirs(MANIFEST, exist_ok=True)
    with open(os.path.join(MANIFEST, "STEP5A_STAGE_AB_AUDIT.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"\nSTAGE A+B AUDIT: {overall} ({len(results)} checks, {len(failures)} failures)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
