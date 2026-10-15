# -*- coding: utf-8 -*-
"""
91_h_access.py
STEP5H-A Stage 1: frozen-composition verification + QA (spec 6, 7), the FIRST
Ni ACCESS RECORD written IMMEDIATELY BEFORE the first Ni elemental-property read
(spec 1), and the minimized Ni descriptor extraction with its access ledger
(spec 2, 3, 4).

Order of operations in this file is the order of operations in reality:
  1. verify the frozen composition table hash EXACTLY (spec 6; mismatch -> STOP)
  2. re-run the composition QA (spec 7; nothing is redrawn or altered)
  3. write STEP5H_A_NI_FIRST_DESCRIPTOR_ACCESS.json BEFORE any property read
  4. only then read the ten Ni elemental quantities required by the frozen
     43-feature schema, from the frozen STEP5A approved sources only, recording
     every access in STEP5H_A_NI_PROPERTY_ACCESS_LEDGER.csv

No Thermo-Calc, no tc_python, no Ni target of any kind (spec 5, 38).
"""
import importlib.metadata as md_ver
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5h_common as C

CHECKS = []


def chk(cid, what, ok, detail=""):
    CHECKS.append({"id": cid, "check": what, "status": "PASS" if ok else "FAIL",
                   "detail": str(detail)[:400]})
    print("  [%-4s] %-4s %s%s" % (cid, "PASS" if ok else "FAIL", what,
                                  ("  |  " + str(detail)[:140]) if detail else ""))
    return ok


def composition_qa(comp):
    """Spec 7 recheck. The table is read-only here; nothing is redrawn."""
    ok = True
    ok &= chk("H01", "exactly 2000 rows", len(comp) == C.N_COMPOSITIONS, len(comp))
    ok &= chk("H02", "composition_id unique, NI_BLIND_000001..002000",
              comp.composition_id.nunique() == len(comp)
              and bool(comp.composition_id.str.fullmatch(r"NI_BLIND_\d{6}").all())
              and comp.composition_id.iloc[0] == "NI_BLIND_000001"
              and comp.composition_id.iloc[-1] == "NI_BLIND_002000",
              "%s .. %s" % (comp.composition_id.iloc[0], comp.composition_id.iloc[-1]))
    ok &= chk("H03", "few-shot roles preserved: 600 calibration / 1400 evaluation",
              int((comp.fewshot_role == "FEWSHOT_CALIBRATION").sum()) == 600
              and int((comp.fewshot_role == "RECOVERY_EVALUATION").sum()) == 1400,
              comp.fewshot_role.value_counts().to_dict())
    ok &= chk("H04", "S_CrNb in frozen range 0.6-2.0 wt.%",
              bool(comp.S_CrNb_wt_pct.between(0.6 - 5e-4, 2.0 + 5e-4).all()),
              "[%.4f, %.4f]" % (comp.S_CrNb_wt_pct.min(), comp.S_CrNb_wt_pct.max()))
    ok &= chk("H05", "R_CrNb_atomic_actual in frozen range 2.0-6.5",
              bool(comp.R_CrNb_atomic_actual.between(2.0 - 5e-3, 6.5 + 5e-3).all()),
              "[%.4f, %.4f]" % (comp.R_CrNb_atomic_actual.min(),
                                comp.R_CrNb_atomic_actual.max()))
    ok &= chk("H06", "Ni wt.% in frozen range 0.02-0.20",
              bool(comp.X_wt_pct.between(0.02 - 5e-5, 0.20 + 5e-5).all()),
              "[%.4f, %.4f]" % (comp.X_wt_pct.min(), comp.X_wt_pct.max()))
    ok &= chk("H07", "Cu balance valid: all weights close to 100, Cu > 0",
              float(np.abs(comp.composition_sum_wt_pct - 100.0).max()) < 1e-6
              and bool((comp.Cu_wt_pct > 0).all()),
              "max |sum-100| = %.2e | min Cu %.4f"
              % (float(np.abs(comp.composition_sum_wt_pct - 100).max()),
                 comp.Cu_wt_pct.min()))
    ok &= chk("H08", "all numerical values finite",
              bool(comp.select_dtypes(include=[np.number]).notna().all().all())
              and bool(np.isfinite(comp.select_dtypes(include=[np.number])
                                   .to_numpy(float)).all()),
              "no NaN / inf in any numeric column")
    ok &= chk("H09", "no duplicate composition signature (Cr, Nb, X)",
              int(comp.duplicated(["Cr_wt_pct", "Nb_wt_pct", "X_wt_pct"]).sum()) == 0,
              "0 duplicates")
    return ok


def extract_ni_descriptors():
    """Read ONLY the ten required Ni quantities, from ONLY the frozen STEP5A
    approved sources (spec 2, 3, 4). No web, no manual entry, no other database."""
    from matminer.utils.data import MagpieData
    from mendeleev import element as mend_element
    from pymatgen.core.periodic_table import Element

    magpie = MagpieData()
    ts = C.utcnow()
    rows = []
    values = {}
    for prop, src, field, unit, required_by in C.NI_REQUIRED_PROPERTIES:
        if src == "magpie":
            v = magpie.get_elemental_property(Element("Ni"), field)
            pkg, ver = "matminer MagpieData (magpie.data.json)", md_ver.version("matminer")
        else:
            v = getattr(mend_element("Ni"), field)
            pkg, ver = "mendeleev built-in elemental database", md_ver.version("mendeleev")
        if v is None or (isinstance(v, float) and v != v):
            raise SystemExit("FATAL: approved source returned no value for Ni %s. "
                             "STEP5A missing-value policy: the feature cannot be "
                             "hand-estimated; STOP." % prop)
        values[prop] = float(v)
        rows.append({"source": pkg, "source_version": ver, "property": prop,
                     "raw_field": field, "unit": unit, "value": float(v),
                     "required_by_features": required_by,
                     "first_access_timestamp": ts, "status": "EXTRACTED"})
    # the frozen matminer preset (F4 features) accesses the SAME magpie repository
    # internally when featurizing a Ni-bearing Composition; recorded as one line,
    # value NOT extracted here individually (spec 4 minimization)
    rows.append({
        "source": "matminer ElementProperty.from_preset('magpie') [aggregate access]",
        "source_version": md_ver.version("matminer"),
        "property": "MAGPIE_PRESET_AGGREGATE",
        "raw_field": "(preset-internal; identical frozen implementation as STEP5A)",
        "unit": "-", "value": np.nan,
        "required_by_features": "the 19 magpie_* columns of the frozen 43-feature "
                                "schema, computed at featurization time (stage 92)",
        "first_access_timestamp": ts,
        "status": "ACCESSED_VIA_FROZEN_PRESET_AT_FEATURIZATION"})
    return pd.DataFrame(rows), values, ts


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5H-A  Stage 1: composition verification, first-access record, Ni extraction")
    print("=" * 78)
    C.assert_no_tc_python()

    # ---- spec 6: exact frozen-hash verification of the composition table ---- #
    C.verify_hash(C.G_NI_COMP_CSV, C.FROZEN_HASHES["ni_compositions_csv"],
                  "STEP5G_NI_BLIND_COMPOSITIONS.csv")
    print("  composition table sha256 matches the frozen value exactly")
    comp = pd.read_csv(C.G_NI_COMP_CSV)
    ok = composition_qa(comp)
    pd.DataFrame(CHECKS).to_csv(os.path.join(C.D_QA, "STEP5H_A_COMPOSITION_QA.csv"),
                                index=False, encoding="utf-8")
    if not ok:
        print("FATAL: composition QA failed; the frozen table may be corrupt. STOP.")
        return 2

    fz = json.load(open(C.G_FREEZE_JSON, encoding="utf-8"))
    Pm = fz["models"]["ORIGINAL6_DIRECT_MLP"]
    Sm = fz["models"]["FULL11_DIRECT_MLP"]

    # ---- spec 1: FIRST ACCESS RECORD, written BEFORE the first property read -- #
    access_path = os.path.join(C.D_ACCESS, "STEP5H_A_NI_FIRST_DESCRIPTOR_ACCESS.json")
    if os.path.exists(access_path):
        first = json.load(open(access_path, encoding="utf-8"))
        print("  first-access record already exists (written %s); extraction will be "
              "re-run read-only and must match the ledger" % first["timestamp_utc"])
    else:
        first = {
            "timestamp_utc": C.utcnow(),
            "stage": "STEP5H-A",
            "pipeline_freeze_id": fz["freeze_id"],
            "pipeline_freeze_sha256": C.sha256_file(C.G_FREEZE_JSON),
            "ni_composition_table_sha256": C.FROZEN_HASHES["ni_compositions_csv"],
            "primary_checkpoint_sha256": C.FROZEN_HASHES["primary_checkpoint"],
            "secondary_checkpoint_sha256": C.FROZEN_HASHES["secondary_checkpoint"],
            "primary_scaler_sha256": C.FROZEN_HASHES["primary_scaler"],
            "secondary_scaler_sha256": C.FROZEN_HASHES["secondary_scaler"],
            "feature_schema_sha256": C.FROZEN_HASHES["feature_schema_csv"],
            "feature_list_sha256": C.FROZEN_HASHES["feature_list_txt"],
            "approved_property_sources": [
                {"name": "matminer MagpieData (magpie.data.json elemental repository)",
                 "version": None, "usage": "9 of the 10 required Ni quantities + the "
                                           "frozen ElementProperty('magpie') preset"},
                {"name": "mendeleev built-in elemental database",
                 "version": None, "usage": "MetallicRadius (atomic_radius), Density"},
            ],
            "excluded_sources": ["web search", "manual entry", "Wikipedia",
                                 "Materials Project", "DFT", "alternate databases",
                                 "ad-hoc physical-property tables"],
            "properties_to_be_accessed": [p[0] for p in C.NI_REQUIRED_PROPERTIES],
            "properties_deliberately_not_accessed": C.NI_NOT_EXTRACTED,
            "statement": "No Ni CALPHAD target or equilibrium result has been accessed.",
            "tc_python_imported": False,
        }
        C.write_json(access_path, first)
        print("  first-access record written at %s - BEFORE any property read"
              % first["timestamp_utc"])

    # ---- spec 2-4: the extraction itself, with the ledger -------------------- #
    ledger, values, ts = extract_ni_descriptors()
    import matminer, mendeleev
    for src in first["approved_property_sources"]:
        src["version"] = (matminer.__version__ if "matminer" in src["name"]
                          else md_ver.version("mendeleev"))
    if not os.path.exists(access_path + ".sealed"):
        C.write_json(access_path, first)      # versions filled in, same record
        open(access_path + ".sealed", "w").write(ts)
    ledger.to_csv(os.path.join(C.D_ACCESS, "STEP5H_A_NI_PROPERTY_ACCESS_LEDGER.csv"),
                  index=False, encoding="utf-8")

    ni_row = pd.DataFrame([{"element": "NI", **values}])
    ni_row.to_parquet(os.path.join(C.D_ACCESS, "STEP5H_A_NI_DESCRIPTOR_ROW.parquet"),
                      index=False)

    print("\n  Ni descriptor row (the ONLY Ni quantities read):")
    for _, r in ledger[ledger.status == "EXTRACTED"].iterrows():
        print("    %-24s %-10s = %s" % (r["property"], r["unit"], r["value"]))
    print("  NOT extracted (no frozen 43-feature column requires them): %s"
          % ", ".join(C.NI_NOT_EXTRACTED))
    print("  ledger: %d rows -> STEP5H_A_NI_PROPERTY_ACCESS_LEDGER.csv" % len(ledger))

    hits = C.ni_scan_frame(ledger, "LEDGER", allowed_cols=("property", "raw_field",
                                                           "required_by_features",
                                                           "source", "status"))
    C.write_json(os.path.join(C.D_CONFIG, "STEP5H_A_ACCESS_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "first_access_record": first,
        "n_properties_extracted": int((ledger.status == "EXTRACTED").sum()),
        "properties_extracted": values,
        "properties_not_extracted": C.NI_NOT_EXTRACTED,
        "ledger_sha256": C.sha256_file(
            os.path.join(C.D_ACCESS, "STEP5H_A_NI_PROPERTY_ACCESS_LEDGER.csv")),
        "ni_descriptor_row_sha256": C.sha256_file(
            os.path.join(C.D_ACCESS, "STEP5H_A_NI_DESCRIPTOR_ROW.parquet")),
        "composition_qa": CHECKS,
        "tc_python_imported": False,
        "ni_calphad_or_target_accessed": 0})
    print("STAGE 1: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
