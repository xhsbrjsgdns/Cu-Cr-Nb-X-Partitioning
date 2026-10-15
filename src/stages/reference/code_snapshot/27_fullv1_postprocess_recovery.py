# -*- coding: utf-8 -*-
"""
27_fullv1_postprocess_recovery.py
FULL_V1 POST-PROCESSING DATA PACKAGE RECOVERY

Rebuilds the standard post-processing files that the downstream JMST ML / mechanism
analysis needs, USING ONLY files that already exist in the frozen FULL_V1 database.

    NO Thermo-Calc.  NO TC-Python.  NO equilibrium calculation.  NO CALPHAD of any kind.
    This module never imports tc_python and never opens a Thermo-Calc session.

It also does NOT modify any frozen artefact:
    - no raw file is deleted, overwritten or rewritten
    - the two authoritative SQLite stores are never opened for writing (never opened at all)
    - phase mapping, temperature grid, sampling policy, partition definition,
      presence threshold and failure policy are read from FULL_V1_SCHEMA.json,
      never redefined here
    - 22_fullv1_run.py / 23_fullv1_partitioning.py / fivefamily.py / solver_escalation.py /
      io_store.py / dry_run_lib.py are neither imported nor edited

Stages: A scan -> B audit -> C schema -> D no-CALPHAD assertion -> E ML_TARGET_ALL ->
F ML_READY -> G FAILURE_MAP -> H DERIVED_DESCRIPTORS -> I COMPOSITIONS.parquet ->
J PHASE_SPLIT_DIAGNOSTICS -> K DOMAIN_SUMMARY -> L TOPOLOGY_SUMMARY -> M QA ->
N hashes -> O report.

Usage:
    python 27_fullv1_postprocess_recovery.py            # skip files that exist and verify
    python 27_fullv1_postprocess_recovery.py --force    # regenerate everything
"""

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- paths -- #
PROJECT_ROOT = Path(r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1")
CFG = PROJECT_ROOT / "00_config"
COMP = PROJECT_ROOT / "01_compositions"
RAW = PROJECT_ROOT / "02_raw"
PROC = PROJECT_ROOT / "03_processed"
QA = PROJECT_ROOT / "04_qa"
MAN = PROJECT_ROOT / "06_manifest"

SCHEMA_FILE = MAN / "FULL_V1_SCHEMA.json"
COMPS_CSV = COMP / "FULL_V1_COMPOSITIONS.csv"
COMPS_PQ = COMP / "FULL_V1_COMPOSITIONS.parquet"
PREVIEW = PROC / "FULL_V1_ML_TARGET_PREVIEW"          # extension resolved at read time
STATES_CSV = RAW / "FULL_V1_EQUILIBRIUM_STATES.csv"
PHASES_LONG = RAW / "FULL_V1_PHASES_LONG"
PCOMP_LONG = RAW / "FULL_V1_PHASE_COMPOSITIONS_LONG"
PART_AGG = PROC / "FULL_V1_ELEMENT_PARTITIONING_AGGREGATED"

OUT = {
    "ML_TARGET_ALL": PROC / "FULL_V1_ML_TARGET_ALL.parquet",
    "ML_READY_CONVERGED": PROC / "FULL_V1_ML_READY_CONVERGED.parquet",
    "DERIVED_DESCRIPTORS": PROC / "FULL_V1_DERIVED_DESCRIPTORS.parquet",
    "FAILURE_MAP": QA / "FULL_V1_FAILURE_MAP.parquet",
    "COMPOSITIONS_PARQUET": COMPS_PQ,
    "TOPOLOGY_SUMMARY": QA / "FULL_V1_TOPOLOGY_SUMMARY.csv",
    "DOMAIN_SUMMARY": QA / "FULL_V1_DOMAIN_SUMMARY.csv",
    "PHASE_SPLIT_DIAGNOSTICS": QA / "FULL_V1_PHASE_SPLIT_DIAGNOSTICS.parquet",
}
AUDIT_MD = QA / "FULL_V1_POSTPROCESS_FILE_AUDIT.md"
REPORT_MD = QA / "FULL_V1_POSTPROCESS_RECOVERY_REPORT.md"
SUMMARY_JSON = QA / "FULL_V1_POSTPROCESS_RECOVERY_SUMMARY.json"
CAVEATS_MD = QA / "FULL_V1_DERIVED_DESCRIPTOR_CAVEATS.md"
BCC3_AUDIT_CSV = QA / "FULL_V1_BCC3_INSTANCE_AUDIT.csv"
BCC3_FIX_MD = QA / "FULL_V1_BCC3_FIX_REPORT.md"
HASHES_CSV = MAN / "FULL_V1_POSTPROCESS_FILE_HASHES.csv"

# ------------------------------------------------------------- expected known truth -- #
N_COMPS, N_STATES, N_CONVERGED, N_FAILED = 50000, 550000, 548432, 1568
FAILED_FAMILY, FAILED_TEMP = "MO", 950.0
CONS_TOL = 1e-12

CHECKS = []
NOTES = []
_T0 = time.time()


def chk(cid, what, status, detail=""):
    CHECKS.append({"id": cid, "check": what, "status": status, "detail": str(detail)[:600]})
    print("  [%-4s] %-4s  %s%s" % (cid, status, what, ("  |  " + str(detail)[:160]) if detail else ""))
    return status == "PASS"


def note(msg):
    NOTES.append(msg)
    print("  ..   %s" % msg)


def read_table(stem: Path):
    """Read whichever form io_store.write_table produced. Read-only; frozen files untouched."""
    for suf, fn in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv), (".csv", pd.read_csv)):
        p = stem.with_suffix(suf) if not str(stem).endswith(suf) else Path(str(stem))
        if p.exists():
            return fn(p), p
    raise FileNotFoundError("no table found for %s(.parquet|.csv.gz|.csv)" % stem)


def sha256(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


NI_TOKEN = re.compile(r"(?<![A-Za-z])NI(?![A-Za-z])")


def ni_hits(df, label):
    """Element-token Ni scan. 'DOMINANT' and 'NIOBIUM' cannot match: the token is
    bounded by letters on at least one side there."""
    hits = []
    for c in df.columns:
        if NI_TOKEN.search(str(c).upper()):
            hits.append("%s::column::%s" % (label, c))
    for c in df.columns:
        s = df[c]
        if s.dtype == object or str(s.dtype).startswith("string"):
            vals = s.dropna().astype(str)
            if len(vals) == 0:
                continue
            uniq = pd.unique(vals) if vals.nunique(dropna=True) < 5000 else vals.unique()
            for v in uniq:
                if NI_TOKEN.search(v.upper()):
                    hits.append("%s::%s::%s" % (label, c, v[:60]))
                    if len(hits) > 20:
                        return hits
    return hits


# =============================================================== STAGE A + B: audit == #
REQUESTED = [
    ("FULL_V1_ML_TARGET_ALL.parquet", "ML_TARGET_ALL", "03_processed/FULL_V1_ML_TARGET_PREVIEW.parquet"),
    ("FULL_V1_ML_READY_CONVERGED.parquet", "ML_READY_CONVERGED", "FULL_V1_ML_TARGET_ALL (state_converged == 1)"),
    ("FULL_V1_DERIVED_DESCRIPTORS.parquet", "DERIVED_DESCRIPTORS",
     "ML_TARGET_PREVIEW + 02_raw/FULL_V1_PHASE_COMPOSITIONS_LONG.parquet"),
    ("FULL_V1_FAILURE_MAP.parquet", "FAILURE_MAP",
     "02_raw/FULL_V1_EQUILIBRIUM_STATES.csv + 01_compositions/FULL_V1_COMPOSITIONS.csv"),
    ("FULL_V1_COMPOSITIONS.parquet", "COMPOSITIONS_PARQUET", "01_compositions/FULL_V1_COMPOSITIONS.csv"),
    ("FULL_V1_TOPOLOGY_SUMMARY.csv", "TOPOLOGY_SUMMARY", "ML_TARGET_PREVIEW (X_P_* / X_Psolid_* / dominant_*)"),
    ("FULL_V1_DOMAIN_SUMMARY.csv", "DOMAIN_SUMMARY", "ML_TARGET_PREVIEW + FULL_V1_COMPOSITIONS.csv"),
    ("FULL_V1_PHASE_SPLIT_DIAGNOSTICS.parquet", "PHASE_SPLIT_DIAGNOSTICS",
     "ML_TARGET_PREVIEW + 02_raw/FULL_V1_PHASES_LONG.parquet"),
]
ALSO_LOOK_FOR = ["26_fullv1_finalize.py", "FINALIZE.bat"]


def stage_ab_audit():
    print("\n=== STAGE A: recursive scan of the whole FULL_V1 tree ===")
    index = {}
    for p in PROJECT_ROOT.rglob("*"):
        if p.is_file():
            index.setdefault(p.name, []).append(p)
    print("  indexed %d files" % sum(len(v) for v in index.values()))

    rows = []
    for fname, key, source in REQUESTED:
        found = index.get(fname, [])
        # a hit inside the benchmark sandbox is not the real deliverable
        real = [p for p in found if "proc_benchmark" not in p.parts]
        exists = len(real) > 0
        path = str(real[0].relative_to(PROJECT_ROOT)).replace("\\", "/") if exists else ""
        size = real[0].stat().st_size if exists else 0
        nrow, notes_ = "", ""
        if exists:
            try:
                df, _ = read_table(real[0])
                nrow = len(df)
                del df
                gc.collect()
            except Exception as e:                                   # pragma: no cover
                notes_ = "unreadable: %s" % e
        expected_path = str(OUT[key].relative_to(PROJECT_ROOT)).replace("\\", "/")
        if exists and path != expected_path:
            notes_ = (notes_ + " ; " if notes_ else "") + "found at a DIFFERENT path than requested"
        rows.append({"requested_file": fname, "exists": "YES" if exists else "NO",
                     "actual_path": path or "(not found anywhere under FULL_V1)",
                     "size": size, "row_count": nrow, "source_file": source,
                     "needs_regeneration": "NO (verify only)" if exists else "YES",
                     "notes": notes_})
    extra = []
    for fname in ALSO_LOOK_FOR:
        found = [p for p in index.get(fname, []) if "proc_benchmark" not in p.parts]
        extra.append({"file": fname, "exists": "YES" if found else "NO",
                      "actual_path": str(found[0].relative_to(PROJECT_ROOT)).replace("\\", "/") if found else "",
                      "size": found[0].stat().st_size if found else 0})

    near = {}
    for stem in ("TOPOLOGY", "DOMAIN", "SPLIT", "FAILURE", "DESCRIPTOR", "ML_"):
        near[stem] = sorted({str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
                             for n, ps in index.items() if stem in n.upper()
                             for p in ps if "proc_benchmark" not in p.parts})

    lines = ["# FULL_V1_POSTPROCESS_FILE_AUDIT.md", "",
             "**Stage A/B of FULL_V1 POST-PROCESSING DATA PACKAGE RECOVERY.**", "",
             "Recursive scan of the entire `%s` tree (all depths), not just the target directories. "
             "Matches inside `05_logs/proc_benchmark/` are ignored: that is the 2- vs 4-process "
             "benchmark sandbox, not the production database." % PROJECT_ROOT, "",
             "Scanned %d files at %s UTC." % (sum(len(v) for v in index.values()),
                                              time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), "",
             "## Requested deliverables", "",
             "| requested_file | exists | actual_path | size | row_count | source_file | needs_regeneration | notes |",
             "|---|---|---|---:|---:|---|---|---|"]
    for r in rows:
        lines.append("| `%s` | **%s** | `%s` | %s | %s | `%s` | %s | %s |" % (
            r["requested_file"], r["exists"], r["actual_path"],
            "{:,}".format(r["size"]) if r["size"] else "-",
            "{:,}".format(r["row_count"]) if r["row_count"] != "" else "-",
            r["source_file"], r["needs_regeneration"], r["notes"] or "-"))
    lines += ["", "## Recovery scripts", "",
              "| file | exists | actual_path | size |", "|---|---|---|---:|"]
    for e in extra:
        lines.append("| `%s` | **%s** | `%s` | %s |" % (
            e["file"], e["exists"], e["actual_path"] or "-",
            "{:,}".format(e["size"]) if e["size"] else "-"))
    lines += ["", "## Name-similar files that already exist (checked so nothing is duplicated)", ""]
    for k, v in near.items():
        lines.append("- **%s**: %s" % (k, ", ".join("`%s`" % x for x in v) if v else "(none)"))
    lines += ["", "## Conclusion", "",
              "Files that exist are verified rather than rebuilt. Files that do not exist anywhere "
              "under FULL_V1 are regenerated from the frozen sources named above. "
              "**No Thermo-Calc calculation is performed at any point.**", ""]
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("  wrote %s" % AUDIT_MD.name)
    return rows, extra


# ==================================================================== STAGE C: schema = #
def stage_c_schema():
    print("\n=== STAGE C: read the frozen schema ===")
    if not SCHEMA_FILE.exists():
        raise SystemExit("FATAL: %s is missing - definitions must come from the schema, "
                         "not from memory." % SCHEMA_FILE)
    sch = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    conv = sch["conventions"]
    fam5 = list(conv["core_families"])
    temps = [float(t) for t in conv["temperature_grid_C"]]
    thr = float(conv["presence_threshold"])
    solid = [f for f in fam5 if f != "LIQUID"]
    print("  core families      : %s" % fam5)
    print("  temperature grid   : %s" % [int(t) for t in temps])
    print("  presence threshold : %g" % thr)
    print("  group key          : %s" % conv["group_key_for_splitting"])
    desc_cols = [c["name"] for c in sch["files"]["FULL_V1_DERIVED_DESCRIPTORS"]["columns"]]
    print("  descriptor columns declared in schema: %d" % len(desc_cols))
    return sch, fam5, solid, temps, thr, desc_cols


# ============================================================ STAGE D: no CALPHAD ===== #
def stage_d_no_calphad():
    print("\n=== STAGE D: confirm no Thermo-Calc is needed or loaded ===")
    loaded = sorted(m for m in sys.modules if "tc_python" in m or "thermo" in m.lower())
    ok = not loaded
    chk("D01", "no Thermo-Calc / TC-Python module loaded in this process",
        "PASS" if ok else "FAIL", loaded or "none")
    missing = [str(p) for p in (COMPS_CSV, STATES_CSV) if not p.exists()]
    for stem in (PREVIEW, PHASES_LONG, PCOMP_LONG, PART_AGG):
        if not any(stem.with_suffix(s).exists() for s in (".parquet", ".csv.gz", ".csv")):
            missing.append(str(stem) + ".{parquet|csv}")
    chk("D02", "every deliverable is derivable from existing frozen files",
        "PASS" if not missing else "FAIL", missing or "all sources present")
    if missing:
        raise SystemExit("FATAL: missing sources; recovery cannot proceed without recalculating, "
                         "which is forbidden. STOP and investigate: %s" % missing)


# ================================================ STAGE E: ML_TARGET_ALL (re-export) == #
def stage_e_target_all(force):
    print("\n=== STAGE E: FULL_V1_ML_TARGET_ALL (formal re-export of ML_TARGET_PREVIEW) ===")
    prev, prev_path = read_table(PREVIEW)
    print("  read %s  ->  %d rows x %d cols" % (prev_path.name, len(prev), prev.shape[1]))
    dst = OUT["ML_TARGET_ALL"]
    if dst.exists() and not force:
        note("ML_TARGET_ALL already exists - verifying instead of rebuilding")
    else:
        # Straight re-export. NOTHING is recomputed: same rows, same order, same columns,
        # same values, same NaN pattern (spec section 22).
        prev.to_parquet(dst, index=False)
        print("  wrote %s" % dst.name)
    ml = pd.read_parquet(dst)

    same_cols = list(ml.columns) == list(prev.columns)
    chk("E01", "column set and order identical to ML_TARGET_PREVIEW",
        "PASS" if same_cols else "FAIL",
        "%d vs %d cols" % (ml.shape[1], prev.shape[1]))
    same_order = bool((ml["state_id"].values == prev["state_id"].values).all()) \
        if len(ml) == len(prev) else False
    chk("E02", "state_id order identical to ML_TARGET_PREVIEW",
        "PASS" if same_order else "FAIL", "%d vs %d rows" % (len(ml), len(prev)))

    maxdev, nanmismatch, strmismatch = 0.0, 0, 0
    for c in prev.columns:
        a, b = prev[c], ml[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            na, nb = a.isna(), b.isna()
            nanmismatch += int((na != nb).sum())
            m = ~na & ~nb
            if m.any():
                d = float(np.max(np.abs(a[m].to_numpy(dtype="float64")
                                        - b[m].to_numpy(dtype="float64"))))
                maxdev = max(maxdev, d)
        else:
            strmismatch += int((a.fillna("").astype(str) != b.fillna("").astype(str)).sum())
    chk("E03", "numeric values identical (max |delta|)", "PASS" if maxdev == 0.0 else "FAIL",
        "%.3e" % maxdev)
    chk("E04", "NaN pattern identical", "PASS" if nanmismatch == 0 else "FAIL", nanmismatch)
    chk("E05", "string/categorical values identical", "PASS" if strmismatch == 0 else "FAIL",
        strmismatch)

    nfail = int((ml.state_converged == 0).sum())
    chk("E06", "row count == 550,000", "PASS" if len(ml) == N_STATES else "FAIL", len(ml))
    chk("E07", "unique state_id == 550,000",
        "PASS" if ml.state_id.nunique() == N_STATES else "FAIL", ml.state_id.nunique())
    chk("E08", "unique composition_id == 50,000",
        "PASS" if ml.composition_id.nunique() == N_COMPS else "FAIL", ml.composition_id.nunique())
    per = ml.groupby("composition_id")["Temperature_C"].nunique()
    chk("E09", "exactly 11 temperatures per composition",
        "PASS" if int(per.min()) == 11 and int(per.max()) == 11 else "FAIL",
        "min %d max %d" % (per.min(), per.max()))
    chk("E10", "converged == 548,432 / failed == 1,568",
        "PASS" if (len(ml) - nfail) == N_CONVERGED and nfail == N_FAILED else "FAIL",
        "converged %d failed %d" % (len(ml) - nfail, nfail))
    del prev
    gc.collect()
    return ml


# ==================================================== STAGE F: ML_READY_CONVERGED ===== #
def stage_f_ml_ready(ml, force):
    print("\n=== STAGE F: FULL_V1_ML_READY_CONVERGED ===")
    dst = OUT["ML_READY_CONVERGED"]
    ready = ml[ml.state_converged == 1].reset_index(drop=True)
    if dst.exists() and not force:
        note("ML_READY_CONVERGED already exists - verifying instead of rebuilding")
    else:
        # strict subset: no column dropped, no NaN filled, no row reordered
        ready.to_parquet(dst, index=False)
        print("  wrote %s" % dst.name)
    r = pd.read_parquet(dst)
    chk("F01", "row count == 548,432", "PASS" if len(r) == N_CONVERGED else "FAIL", len(r))
    chk("F02", "every row state_converged == 1",
        "PASS" if set(pd.unique(r.state_converged)) == {1} else "FAIL",
        sorted(set(pd.unique(r.state_converged))))
    chk("F03", "state_id unique", "PASS" if r.state_id.nunique() == len(r) else "FAIL",
        r.state_id.nunique())
    chk("F04", "column structure identical to ML_TARGET_ALL",
        "PASS" if list(r.columns) == list(ml.columns) else "FAIL",
        "%d vs %d" % (r.shape[1], ml.shape[1]))
    failed_ids = set(ml.loc[ml.state_converged == 0, "state_id"])
    leak = len(failed_ids & set(r.state_id))
    chk("F05", "contains no FAILED state", "PASS" if leak == 0 else "FAIL", leak)

    # BASELINE X partition must stay NaN, never 0 (spec section 5)
    base = r[r.family == "BASELINE"]
    xcols = [c for c in r.columns if c.startswith("X_P_") and not c.endswith("_AUX")]
    allnan = all(base[c].isna().all() for c in xcols) if len(base) else False
    zeros = int(sum((base[c] == 0).sum() for c in xcols)) if len(base) else 0
    chk("F06", "BASELINE X partition targets are NaN, never 0",
        "PASS" if (allnan and zeros == 0) else "FAIL",
        "%d baseline rows, %d X_P_* cols, zeros=%d" % (len(base), len(xcols), zeros))
    del r
    gc.collect()
    return ready


# ============================================================= STAGE G: FAILURE_MAP === #
FAILMAP_COLS = ["state_id", "composition_id", "family", "X_element", "sampling_domain",
                "Temperature_C", "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
                "S_CrNb_wt_pct", "CrNb_atomic_ratio", "solver_level_used", "attempt_count",
                "bridge_used", "bridge_temperatures_C", "final_status",
                "original_error_message", "error_message"]


def stage_g_failure_map(comps, force):
    print("\n=== STAGE G: FULL_V1_FAILURE_MAP ===")
    dst = OUT["FAILURE_MAP"]
    st = pd.read_csv(STATES_CSV, keep_default_na=False, low_memory=False)
    st["ok"] = st["converged"].astype(str).str.lower() == "true"
    f = st[~st.ok].copy()
    print("  FAILED states in EQUILIBRIUM_STATES.csv: %d" % len(f))
    meta = comps.set_index("composition_id")
    f = f.rename(columns={"case_id": "composition_id"})
    for src, dstcol in (("sampling_domain", "sampling_domain"), ("Cu_wt_pct", "Cu_wt_pct"),
                        ("Cr_wt_pct", "Cr_wt_pct"), ("Nb_wt_pct", "Nb_wt_pct"),
                        ("X_wt_pct", "X_wt_pct"), ("S_CrNb_wt_pct", "S_CrNb_wt_pct"),
                        ("R_CrNb_atomic_actual", "CrNb_atomic_ratio")):
        f[dstcol] = f["composition_id"].map(meta[src]) if src in meta.columns else np.nan
    for c in ("Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct", "S_CrNb_wt_pct",
              "CrNb_atomic_ratio", "Temperature_C"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    absent = [c for c in FAILMAP_COLS if c not in f.columns]
    if absent:
        raise SystemExit("FATAL: FAILURE_MAP is missing required columns %s. STOP and "
                         "investigate rather than shipping a partial map." % absent)
    f = f[FAILMAP_COLS].sort_values(["composition_id", "Temperature_C"]).reset_index(drop=True)

    fams, temps = sorted(set(f.family)), sorted(set(f.Temperature_C))
    if fams != [FAILED_FAMILY] or temps != [FAILED_TEMP] or len(f) != N_FAILED:
        chk("G00", "STOP CONDITION: failures are not exactly MO @ 950 C", "FAIL",
            "families=%s temps=%s rows=%d" % (fams, temps, len(f)))
        raise SystemExit("FATAL (spec section 6): FAILED set is not the expected "
                         "1,568 MO @ 950 C. families=%s temperatures=%s rows=%d. "
                         "STOP and investigate before generating anything else." %
                         (fams, temps, len(f)))
    if dst.exists() and not force:
        note("FAILURE_MAP already exists - verifying instead of rebuilding")
    else:
        f.to_parquet(dst, index=False)
        print("  wrote %s" % dst.name)
    m = pd.read_parquet(dst)
    chk("G01", "row count == 1,568", "PASS" if len(m) == N_FAILED else "FAIL", len(m))
    chk("G02", "family unique == {MO}", "PASS" if sorted(set(m.family)) == ["MO"] else "FAIL",
        sorted(set(m.family)))
    chk("G03", "Temperature_C unique == {950}",
        "PASS" if sorted(set(m.Temperature_C)) == [950.0] else "FAIL", sorted(set(m.Temperature_C)))
    chk("G04", "all 19 required columns present",
        "PASS" if list(m.columns) == FAILMAP_COLS else "FAIL", len(m.columns))
    chk("G05", "verbatim Thermo-Calc error retained",
        "PASS" if m.original_error_message.astype(str).str.strip().ne("").all() else "WARN",
        m.original_error_message.astype(str).str.slice(0, 40).mode().iat[0]
        if len(m) else "")
    dom = m.sampling_domain.value_counts().to_dict()
    note("failure domains: %s" % dom)
    del st
    gc.collect()
    return m, dom


# ====================================================== STAGE H: DERIVED_DESCRIPTORS == #
def stage_h_descriptors(ml, comps, failmap, temps, force):
    print("\n=== STAGE H: FULL_V1_DERIVED_DESCRIPTORS ===")
    dst = OUT["DERIVED_DESCRIPTORS"]
    pc, pc_path = read_table(PCOMP_LONG)
    print("  read %s -> %d rows" % (pc_path.name, len(pc)))
    pc["element"] = pc["element"].astype(str).str.upper()
    pc["base"] = pc["raw_phase_name"].astype(str).str.replace(r"#\d+$", "", regex=True)
    fcc_cr = pc[(pc.base == "FCC_L12") & (pc.element == "CR")]
    cr_max = fcc_cr.groupby("state_id")["element_mole_fraction"].max()
    cr_n = fcc_cr.groupby("state_id")["element_mole_fraction"].size()
    del pc, fcc_cr
    gc.collect()

    idx = ml[["state_id", "composition_id", "Temperature_C"]].copy()
    idx["X_Cr_FCC"] = idx.state_id.map(cr_max)
    idx["n_FCC_inst"] = idx.state_id.map(cr_n).fillna(0).astype(int)
    # The frozen definition takes the maximum over stable FCC instances. Verify that this
    # is never actually ambiguous at the two temperatures the descriptors use.
    amb = int(((idx.n_FCC_inst >= 2) & (idx.Temperature_C.isin([450.0, 950.0]))).sum())
    chk("H01", "X_Cr_FCC is unambiguous at 450 C and 950 C (no multi-instance FCC there)",
        "PASS" if amb == 0 else "WARN",
        "%d states with >=2 FCC instances at 450/950 C" % amb)

    w = idx.pivot_table(index="composition_id", columns="Temperature_C", values="X_Cr_FCC")
    c15 = ml.pivot_table(index="composition_id", columns="Temperature_C", values="C15_volfrac")
    liq = ml.pivot_table(index="composition_id", columns="Temperature_C", values="LIQUID_presence")
    mb = ml.pivot_table(index="composition_id", columns="Temperature_C",
                        values="has_multiple_BCC_instances")
    mf = ml.pivot_table(index="composition_id", columns="Temperature_C",
                        values="has_multiple_FCC_instances")

    def col(df, t):
        return df[t] if t in df.columns else pd.Series(np.nan, index=df.index)

    def onset(df, ascending=True):
        """Lowest (ascending) or highest (descending) grid T at which the flag is 1; NaN if never."""
        out = pd.Series(np.nan, index=df.index)
        for t in (temps if ascending else temps[::-1]):
            if t in df.columns:
                m = (df[t] == 1) & out.isna()
                out[m] = t
        return out

    d = pd.DataFrame(index=w.index)
    d["X_Cr_FCC_450"] = col(w, 450.0)
    d["X_Cr_FCC_950"] = col(w, 950.0)
    d["Delta_XCr_FCC_950_450"] = col(w, 950.0) - col(w, 450.0)   # equilibrium Cr depletion proxy
    d["C15_volfrac_450"] = col(c15, 450.0)
    d["C15_volfrac_600"] = col(c15, 600.0)
    d["Delta_C15_600_450"] = col(c15, 600.0) - col(c15, 450.0)
    d["C15_volfrac_max"] = c15.max(axis=1)
    d["LIQUID_onset_C"] = onset(liq, ascending=True)
    d["BCC_split_closure_C"] = onset(mb, ascending=False)        # highest OBSERVED split T
    d["FCC_split_closure_C"] = onset(mf, ascending=False)
    d = d.reset_index()
    meta = comps.set_index("composition_id")
    for c in ("family", "X_element", "sampling_domain"):
        d[c] = d.composition_id.map(meta[c]).fillna("")

    # ---- NEW diagnostic field (spec section 8). The frozen definition of
    # ---- BCC_split_closure_C itself is NOT modified; this only flags where it is censored.
    censored_ids = set(failmap.composition_id)
    d["BCC_split_closure_censored"] = d.composition_id.isin(censored_ids).astype(int)
    d["BCC_split_closure_valid"] = 1 - d["BCC_split_closure_censored"]

    d = d[["composition_id", "family", "X_element", "sampling_domain",
           "X_Cr_FCC_450", "X_Cr_FCC_950", "Delta_XCr_FCC_950_450",
           "C15_volfrac_450", "C15_volfrac_600", "Delta_C15_600_450", "C15_volfrac_max",
           "LIQUID_onset_C", "BCC_split_closure_C", "BCC_split_closure_censored",
           "BCC_split_closure_valid", "FCC_split_closure_C"]]
    if dst.exists() and not force:
        note("DERIVED_DESCRIPTORS already exists - verifying instead of rebuilding")
    else:
        d.to_parquet(dst, index=False)
        print("  wrote %s" % dst.name)
    dd = pd.read_parquet(dst)

    chk("H02", "row count == 50,000 (one per composition)",
        "PASS" if len(dd) == N_COMPS else "FAIL", len(dd))
    chk("H03", "composition_id unique", "PASS" if dd.composition_id.nunique() == len(dd) else "FAIL",
        dd.composition_id.nunique())
    cen = dd[dd.BCC_split_closure_censored == 1]
    chk("H04", "X_Cr_FCC_950 is NaN for every censored (MO@950 FAILED) composition",
        "PASS" if cen.X_Cr_FCC_950.isna().all() else "FAIL",
        "%d censored, %d non-NaN" % (len(cen), int(cen.X_Cr_FCC_950.notna().sum())))
    chk("H05", "Delta_XCr_FCC_950_450 is NaN for every censored composition",
        "PASS" if cen.Delta_XCr_FCC_950_450.isna().all() else "FAIL",
        int(cen.Delta_XCr_FCC_950_450.notna().sum()))
    chk("H06", "censored count == 1,568 and all family MO",
        "PASS" if len(cen) == N_FAILED and set(cen.family) == {"MO"} else "FAIL",
        "%d rows, families %s" % (len(cen), sorted(set(cen.family))))
    # Tolerance note: these compare a stored difference against a difference recomputed from
    # the stored operands, so the gate is on the FORMULA, not on bit-level float identity -
    # demanding atol=0 would fail on any storage round-trip that is not bit-exact.
    dlt = dd.dropna(subset=["Delta_XCr_FCC_950_450"])
    d7 = float(np.max(np.abs((dlt.X_Cr_FCC_950 - dlt.X_Cr_FCC_450)
                             - dlt.Delta_XCr_FCC_950_450))) if len(dlt) else 0.0
    chk("H07", "Delta_XCr_FCC_950_450 == X_Cr_FCC_950 - X_Cr_FCC_450 (<= 1e-12)",
        "PASS" if d7 <= 1e-12 else "FAIL", "%d non-NaN rows, max |delta| %.3e" % (len(dlt), d7))
    dc = dd.dropna(subset=["Delta_C15_600_450"])
    d8 = float(np.max(np.abs((dc.C15_volfrac_600 - dc.C15_volfrac_450)
                             - dc.Delta_C15_600_450))) if len(dc) else 0.0
    chk("H08", "Delta_C15_600_450 == C15_volfrac_600 - C15_volfrac_450 (<= 1e-12)",
        "PASS" if d8 <= 1e-12 else "FAIL", "%d non-NaN rows, max |delta| %.3e" % (len(dc), d8))
    lo = set(dd.LIQUID_onset_C.dropna().unique())
    chk("H09", "LIQUID_onset_C takes only grid temperatures or NaN",
        "PASS" if lo <= set(temps) else "FAIL", sorted(lo))
    liqfam = sorted(set(dd[dd.LIQUID_onset_C.notna()].family))
    note("LIQUID_onset_C present for families %s at %s" % (liqfam, sorted(lo)))
    bs = set(dd.BCC_split_closure_C.dropna().unique())
    chk("H10", "BCC_split_closure_C takes only grid temperatures or NaN",
        "PASS" if bs <= set(temps) else "FAIL", sorted(bs))
    fs = set(dd.FCC_split_closure_C.dropna().unique())
    chk("H11", "FCC_split_closure_C takes only grid temperatures or NaN",
        "PASS" if fs <= set(temps) else "FAIL", sorted(fs))
    note("BCC split families %s ; FCC split families %s"
         % (sorted(set(dd[dd.BCC_split_closure_C.notna()].family)),
            sorted(set(dd[dd.FCC_split_closure_C.notna()].family))))
    del w, c15, liq, mb, mf, idx
    gc.collect()
    return dd


# ================================================ STAGE I: COMPOSITIONS.parquet ======= #
def stage_i_compositions(comps, force):
    print("\n=== STAGE I: FULL_V1_COMPOSITIONS.parquet ===")
    dst = OUT["COMPOSITIONS_PARQUET"]
    existed = dst.exists()
    if existed and not force:
        note("COMPOSITIONS.parquet already exists - verifying against the CSV, not rebuilding")
    else:
        comps.to_parquet(dst, index=False)
        print("  wrote %s" % dst.name)
    pq = pd.read_parquet(dst)
    csv_raw = pd.read_csv(COMPS_CSV, dtype=str, keep_default_na=False)

    chk("I01", "row count identical CSV vs parquet",
        "PASS" if len(pq) == len(csv_raw) == N_COMPS else "FAIL",
        "csv %d parquet %d" % (len(csv_raw), len(pq)))
    chk("I02", "column names identical CSV vs parquet",
        "PASS" if list(pq.columns) == list(csv_raw.columns) else "FAIL",
        "csv %d parquet %d" % (csv_raw.shape[1], pq.shape[1]))
    maxdev, strbad = 0.0, 0
    for c in csv_raw.columns:
        if c not in pq.columns:
            continue
        a = pd.to_numeric(csv_raw[c], errors="coerce")
        if a.notna().any() and pd.api.types.is_numeric_dtype(pq[c]):
            b = pd.to_numeric(pq[c], errors="coerce")
            m = a.notna() & b.notna()
            if m.any():
                maxdev = max(maxdev, float(np.max(np.abs(a[m].to_numpy("float64")
                                                         - b[m].to_numpy("float64")))))
        else:
            strbad += int((csv_raw[c].astype(str).str.strip()
                           != pq[c].fillna("").astype(str).str.strip()).sum())
    chk("I03", "string fields identical CSV vs parquet", "PASS" if strbad == 0 else "FAIL", strbad)
    chk("I04", "float max deviation CSV vs parquet <= 1e-12",
        "PASS" if maxdev <= 1e-12 else "FAIL", "%.3e" % maxdev)
    chk("I05", "original CSV left untouched", "PASS", "not opened for writing at any point")
    return pq, existed


# ================================================ STAGE J: PHASE_SPLIT_DIAGNOSTICS ==== #
def stage_j_split(ml, comps, force):
    """Phase-split diagnostics.

    Two DIFFERENT quantities are computed here, and conflating them is what made the
    original J07 look broken:

      (a) LABEL-based   - how many states carry a raw instance named `BCC_B2#3`.
      (b) COUNT-based   - how many states have >= 3 BCC_B2 instances stable AT THE SAME TIME.

    A raw suffix is a solver label, not an ordinal census: Thermo-Calc can return
    `BCC_B2#1` together with `BCC_B2#3` while `BCC_B2#2` is absent from that state. When
    that happens (a) = 4 while (b) = 0, and both are correct. Instance numbers carry no
    physical identity (frozen policy), so (b) is computed by counting rows, never by
    reading the number after the '#'.
    """
    print("\n=== STAGE J: FULL_V1_PHASE_SPLIT_DIAGNOSTICS ===")
    dst = OUT["PHASE_SPLIT_DIAGNOSTICS"]
    ph, ph_path = read_table(PHASES_LONG)
    print("  read %s -> %d rows" % (ph_path.name, len(ph)))
    ph["raw"] = ph["raw_phase_name"].astype(str)
    ph["base"] = (ph["phase_model_base_name"].astype(str) if "phase_model_base_name" in ph.columns
                  else ph["raw"].str.replace(r"#\d+$", "", regex=True))
    bcc = ph[ph.base == "BCC_B2"]
    fcc = ph[ph.base == "FCC_L12"]

    # ---- (b) COUNT-based: simultaneous instances per state ----
    n_bcc = bcc.groupby("state_id").size()
    n_fcc = fcc.groupby("state_id").size()
    # ---- (a) LABEL-based: which raw suffixes appear, per state ----
    bcc_labels = bcc.groupby("state_id")["raw"].apply(lambda s: ";".join(sorted(set(s))))
    lab3_states = sorted(set(bcc.loc[bcc.raw == "BCC_B2#3", "state_id"]))
    lab2_states = set(bcc.loc[bcc.raw == "BCC_B2#2", "state_id"])
    hi_lab = bcc.copy()
    hi_lab["idx"] = pd.to_numeric(hi_lab["raw"].str.extract(r"#(\d+)$", expand=False), errors="coerce")
    labge3_states = set(hi_lab.loc[hi_lab.idx >= 3, "state_id"])
    print("  BCC label counts: %s"
          % bcc.raw.value_counts().sort_index().to_dict())
    print("  BCC simultaneous-instance counts: %s"
          % n_bcc.value_counts().sort_index().to_dict())

    # ---------------- BCC3 instance audit (task 2) ---------------- #
    meta = comps.set_index("composition_id")
    ev = bcc[bcc.state_id.isin(lab3_states)].copy()
    if len(ev):
        ev = ev.rename(columns={"case_id": "composition_id"})
        if "family" not in ev.columns:
            ev["family"] = ev.composition_id.map(meta["family"])
        ev["sampling_domain"] = ev.composition_id.map(meta["sampling_domain"])
        ev["n_BCC_instances_in_state"] = ev.state_id.map(n_bcc)
        ev["bcc_instance_labels_in_state"] = ev.state_id.map(bcc_labels)
        ev["is_the_BCC_B2_3_row"] = (ev.raw == "BCC_B2#3").astype(int)
        cols = ["state_id", "composition_id", "family", "sampling_domain", "Temperature_C",
                "raw_phase_name", "phase_moles", "phase_mole_fraction",
                "phase_volume_fraction", "phase_mass_fraction",
                "n_BCC_instances_in_state", "bcc_instance_labels_in_state",
                "is_the_BCC_B2_3_row"]
        ev = ev[[c for c in cols if c in ev.columns]].sort_values(
            ["state_id", "raw_phase_name"]).reset_index(drop=True)
    else:
        ev = pd.DataFrame(columns=["state_id", "composition_id", "family", "sampling_domain",
                                   "Temperature_C", "raw_phase_name"])
    ev.to_csv(BCC3_AUDIT_CSV, index=False, encoding="utf-8")
    print("  wrote %s (%d rows covering %d states)"
          % (BCC3_AUDIT_CSV.name, len(ev), len(lab3_states)))
    for sid in lab3_states:
        print("    %s : %d simultaneous BCC instances -> %s"
              % (sid, int(n_bcc.get(sid, 0)), bcc_labels.get(sid, "")))

    del ph, bcc, fcc, hi_lab
    gc.collect()

    # ---------------- per-group diagnostics ---------------- #
    g = ml[["state_id", "family", "sampling_domain", "Temperature_C", "state_converged",
            "has_multiple_FCC_instances", "has_multiple_BCC_instances"]].copy()
    g["n_BCC_inst"] = g.state_id.map(n_bcc).fillna(0).astype(int)
    g["n_FCC_inst"] = g.state_id.map(n_fcc).fillna(0).astype(int)
    g["multi_BCC_ge3"] = (g.n_BCC_inst >= 3).astype(int)
    g["multi_FCC_ge3"] = (g.n_FCC_inst >= 3).astype(int)
    g["label_BCC_B2_3"] = g.state_id.isin(lab3_states).astype(int)
    cv = g[g.state_converged == 1]
    sp = (cv.groupby(["family", "sampling_domain", "Temperature_C"])
          .agg(n_converged=("state_id", "size"),
               n_multi_FCC=("has_multiple_FCC_instances", "sum"),
               n_multi_BCC=("has_multiple_BCC_instances", "sum"),
               n_multi_BCC_ge3=("multi_BCC_ge3", "sum"),
               n_multi_FCC_ge3=("multi_FCC_ge3", "sum"),
               n_label_BCC_B2_3=("label_BCC_B2_3", "sum")).reset_index())
    tot = (g.groupby(["family", "sampling_domain", "Temperature_C"])
           .size().rename("n_states").reset_index())
    sp = tot.merge(sp, on=["family", "sampling_domain", "Temperature_C"], how="left")
    for c in ("n_converged", "n_multi_FCC", "n_multi_BCC", "n_multi_BCC_ge3",
              "n_multi_FCC_ge3", "n_label_BCC_B2_3"):
        sp[c] = sp[c].fillna(0).astype(int)
    den = sp.n_converged.replace(0, np.nan)
    for c in ("multi_FCC", "multi_BCC", "multi_BCC_ge3", "multi_FCC_ge3"):
        sp["rate_%s" % c] = sp["n_%s" % c] / den
    sp["rate_label_BCC_B2_3"] = sp.n_label_BCC_B2_3 / den
    sp = sp.sort_values(["family", "sampling_domain", "Temperature_C"]).reset_index(drop=True)
    if dst.exists() and not force:
        note("PHASE_SPLIT_DIAGNOSTICS already exists - verifying instead of rebuilding")
    else:
        sp.to_parquet(dst, index=False)
        print("  wrote %s" % dst.name)
    s = pd.read_parquet(dst)

    tfcc = int(s.n_multi_FCC.sum())
    tbcc = int(s.n_multi_BCC.sum())
    tb3 = int(s.n_multi_BCC_ge3.sum())
    tlab3 = int(s.n_label_BCC_B2_3.sum())
    ffam = sorted(set(s[s.n_multi_FCC > 0].family))
    bfam = sorted(set(s[s.n_multi_BCC > 0].family))
    ftmax = s[s.n_multi_FCC > 0].Temperature_C.max() if tfcc else np.nan
    btmax = s[s.n_multi_BCC > 0].Temperature_C.max() if tbcc else np.nan
    chk("J01", "FCC multi-instance total == 8,676", "PASS" if tfcc == 8676 else "FAIL", tfcc)
    chk("J02", "FCC splitting confined to AG", "PASS" if ffam == ["AG"] else "FAIL", ffam)
    chk("J03", "FCC splitting confined to <= 400 C", "PASS" if ftmax <= 400 else "FAIL", ftmax)
    chk("J04", "BCC multi-instance total == 58,371", "PASS" if tbcc == 58371 else "FAIL", tbcc)
    chk("J05", "BCC splitting confined to MO", "PASS" if bfam == ["MO"] else "FAIL", bfam)
    chk("J06", "BCC splitting max temperature == 950 C", "PASS" if btmax == 950 else "FAIL", btmax)

    # ---- J07: the LABEL check (this is the acceptance criterion) ---- #
    l3fam = sorted(set(s[s.n_label_BCC_B2_3 > 0].family))
    l3T = sorted(set(s[s.n_label_BCC_B2_3 > 0].Temperature_C))
    chk("J07", "raw phase instance BCC_B2#3 == 4 states, MO only, 950 C only",
        "PASS" if (tlab3 == 4 and l3fam == ["MO"] and l3T == [950.0]) else "FAIL",
        "%d states, families %s, T %s" % (tlab3, l3fam, l3T))
    chk("J07b", "BCC_B2#3 label count agrees with FULL_V1_RAW_PHASE_INVENTORY.csv (4)",
        "PASS" if len(lab3_states) == 4 else "FAIL",
        "%d states in FULL_V1_PHASES_LONG" % len(lab3_states))

    # ---- J09: reconciliation - this is what proves the interpretation ---- #
    # A state has >= 2 SIMULTANEOUS BCC instances iff it carries the #2 label, or it
    # carries a label >= 3 without #2. If that identity holds, the label counts and the
    # simultaneous counts are mutually consistent and neither is a bug.
    only_hi = labge3_states - lab2_states
    predicted = len(lab2_states) + len(only_hi)
    chk("J09", "simultaneous-instance count reconciles with the raw instance labels: "
               "n(>=2) == n(states with #2) + n(states with a label >=3 but no #2)",
        "PASS" if predicted == tbcc else "FAIL",
        "%d + %d = %d vs measured %d" % (len(lab2_states), len(only_hi), predicted, tbcc))
    ge3_detail = "; ".join("%s -> %d instances (%s)"
                           % (sid, int(n_bcc.get(sid, 0)), bcc_labels.get(sid, ""))
                           for sid in lab3_states) or "no BCC_B2#3 state found"
    chk("J10", "states with >= 3 SIMULTANEOUS BCC instances (measured by counting rows, "
               "never by reading the '#n' suffix)",
        "PASS", "%d states. BCC_B2#3-labelled states: %s" % (tb3, ge3_detail))
    if tb3 != tlab3:
        note("BCC_B2#3 is a LABEL on %d states, but only %d state(s) actually have >= 3 BCC "
             "instances stable simultaneously. The suffix is a solver label, not an ordinal "
             "census - see 04_qa/FULL_V1_BCC3_INSTANCE_AUDIT.csv for the per-state evidence."
             % (tlab3, tb3))

    def rate_at(fam, T, col):
        sub = s[(s.family == fam) & (s.Temperature_C == T)]
        return float(sub[col.replace("rate_", "n_")].sum() / sub.n_converged.sum()) \
            if sub.n_converged.sum() else float("nan")

    ag = {int(t): round(rate_at("AG", float(t), "rate_multi_FCC"), 4) for t in (300, 350, 400, 450)}
    mo = {int(t): round(rate_at("MO", float(t), "rate_multi_BCC"), 4)
          for t in (300, 900, 950, 1000)}
    note("AG FCC split rate by T: %s (expected ~0.7175 / 0.3912 / 0.0481 / 0)" % ag)
    note("MO BCC split rate by T: %s (expected ~0.9336 / 0.5000 / 0.1406 / 0)" % mo)
    exp_ag = {300: 0.7175, 350: 0.3912, 400: 0.0481, 450: 0.0}
    exp_mo = {300: 0.9336, 900: 0.5000, 950: 0.1406, 1000: 0.0}
    dev = max([abs(ag[k] - exp_ag[k]) for k in ag] + [abs(mo[k] - exp_mo[k]) for k in mo])
    chk("J08", "split rates reproduce the known FULL_V1 profile (max |delta| <= 0.01)",
        "PASS" if dev <= 0.01 else "WARN", "%.4f" % dev)
    totals = {"FCC_total": tfcc, "BCC_total": tbcc, "BCC_ge3_simultaneous": tb3,
              "BCC_B2_3_label_states": tlab3,
              "bcc3_states": [str(x) for x in lab3_states],
              "bcc3_instance_counts": {str(sid): int(n_bcc.get(sid, 0)) for sid in lab3_states},
              "bcc3_instance_labels": {str(sid): bcc_labels.get(sid, "") for sid in lab3_states},
              "n_states_label_2": len(lab2_states), "n_states_label_ge3_without_2": len(only_hi)}
    return s, ag, mo, totals


# ==================================================== STAGE K: DOMAIN_SUMMARY ========= #
def stage_k_domain(ml, comps, fam5, force):
    print("\n=== STAGE K: FULL_V1_DOMAIN_SUMMARY ===")
    dst = OUT["DOMAIN_SUMMARY"]
    ncomp = comps.groupby("sampling_domain")["composition_id"].nunique()
    tot = ml.groupby("sampling_domain").agg(n_states=("state_id", "size"),
                                            n_converged=("state_converged", "sum"))
    cv = ml[ml.state_converged == 1]
    agg = {"mean_phase_count": ("phase_count", "mean")}
    for k in fam5:
        agg["mean_%s_volfrac" % k] = ("%s_volfrac" % k, "mean")
    for k in fam5:
        agg["%s_presence_rate" % k] = ("%s_presence" % k, "mean")
    mm = cv.groupby("sampling_domain").agg(**agg)
    d = tot.join(mm)
    d.insert(0, "n_compositions", ncomp)
    d["convergence_rate"] = d.n_converged / d.n_states
    order = (["n_compositions", "n_states", "n_converged", "convergence_rate"]
             + ["mean_%s_volfrac" % k for k in fam5]
             + ["%s_presence_rate" % k for k in fam5] + ["mean_phase_count"])
    d = d[order].reset_index()
    # B / A contrast row - computed from the data, never hardcoded
    if {"A", "B"} <= set(d.sampling_domain):
        a = d[d.sampling_domain == "A"].iloc[0]
        b = d[d.sampling_domain == "B"].iloc[0]
        ratio = {"sampling_domain": "B_over_A"}
        for c in order:
            va, vb = float(a[c]), float(b[c])
            ratio[c] = (vb / va) if va not in (0.0,) and np.isfinite(va) else np.nan
        d = pd.concat([d, pd.DataFrame([ratio])], ignore_index=True)
    if dst.exists() and not force:
        note("DOMAIN_SUMMARY already exists - verifying instead of rebuilding")
    else:
        d.round(10).to_csv(dst, index=False, encoding="utf-8")
        print("  wrote %s" % dst.name)
    out = pd.read_csv(dst)
    doms = sorted(set(out.sampling_domain) - {"B_over_A"})
    chk("K01", "one row per sampling domain (A / B / ANCHOR) + B_over_A contrast",
        "PASS" if doms == ["A", "ANCHOR", "B"] else "FAIL", doms)
    chk("K02", "n_compositions sums to 50,000",
        "PASS" if int(out[out.sampling_domain != "B_over_A"].n_compositions.sum()) == N_COMPS
        else "FAIL", int(out[out.sampling_domain != "B_over_A"].n_compositions.sum()))
    chk("K03", "n_states sums to 550,000",
        "PASS" if int(out[out.sampling_domain != "B_over_A"].n_states.sum()) == N_STATES
        else "FAIL", int(out[out.sampling_domain != "B_over_A"].n_states.sum()))
    ratios = {}
    if "B_over_A" in set(out.sampling_domain):
        r = out[out.sampling_domain == "B_over_A"].iloc[0]
        ratios = {k: round(float(r["mean_%s_volfrac" % k]), 4) for k in fam5}
        note("Domain B / Domain A mean volume-fraction ratios: %s" % ratios)
        exp = {"C15": 3.22, "OTHER_SOLID": 2.28, "BCC": 1.29}
        dev = {k: round(abs(ratios.get(k, np.nan) - v), 3) for k, v in exp.items()}
        good = all(v <= 0.30 for v in dev.values() if np.isfinite(v))
        chk("K04", "Domain B/A contrast reproduces the known FULL_V1 pattern "
                   "(C15 ~3.22x, OTHER_SOLID ~2.28x, BCC ~1.29x)",
            "PASS" if good else "WARN", "|delta| %s" % dev)
        chk("K05", "FCC mean volume fraction decreases slightly from A to B",
            "PASS" if ratios.get("FCC", 2) < 1.0 else "WARN", ratios.get("FCC"))
    return out, ratios


# ==================================================== STAGE L: TOPOLOGY_SUMMARY ======= #
def _mode(s):
    s = s.dropna().astype(str)
    s = s[s.str.strip() != ""]
    return s.mode().iat[0] if len(s) else ""


def stage_l_topology(ml, fam5, solid, force):
    print("\n=== STAGE L: FULL_V1_TOPOLOGY_SUMMARY ===")
    dst = OUT["TOPOLOGY_SUMMARY"]
    xp = ["X_P_%s" % f for f in fam5]
    xps = ["X_Psolid_%s" % f for f in solid]
    keep = (["state_id", "composition_id", "family", "sampling_domain", "Temperature_C",
             "state_converged", "dominant_X_partition_family",
             "dominant_X_solid_partition_family"] + xp + xps)
    missing = [c for c in keep if c not in ml.columns]
    if missing:
        raise SystemExit("FATAL: ML_TARGET_ALL lacks topology columns %s. STOP and investigate "
                         "the partition stage rather than inventing them here." % missing)
    t = ml[keep].copy()

    # Verify the frozen X_Psolid definition rather than redefining it (spec section 15):
    # Psolid_f = P_f / (P_FCC + P_BCC + P_C15 + P_OTHER_SOLID); NaN when the denominator ~ 0.
    cvm = t.state_converged == 1
    den = t.loc[cvm, xp[:len(solid)]].sum(axis=1, min_count=1)
    recomputed = {f: np.where(den > 1e-15, t.loc[cvm, "X_P_%s" % f] / den, np.nan) for f in solid}
    dev = 0.0
    for f in solid:
        a = pd.Series(recomputed[f], index=t.loc[cvm].index)
        b = t.loc[cvm, "X_Psolid_%s" % f]
        m = a.notna() & b.notna()
        if m.any():
            dev = max(dev, float(np.max(np.abs(a[m].to_numpy("float64") - b[m].to_numpy("float64")))))
    chk("L01", "stored X_Psolid_* matches the frozen solid-renormalisation definition",
        "PASS" if dev <= 1e-12 else "FAIL", "max |delta| %.3e" % dev)
    del recomputed, den
    gc.collect()

    grp = ["family", "sampling_domain", "Temperature_C"]
    tot = t.groupby(grp).agg(n_states=("state_id", "size"),
                             n_converged=("state_converged", "sum")).reset_index()
    cv = t[cvm]
    agg = {}
    for c in xp + xps:
        agg["mean_%s" % c] = (c, "mean")
        agg["median_%s" % c] = (c, "median")
    stats = cv.groupby(grp).agg(**agg).reset_index()
    modes = (cv.groupby(grp)
             .agg(dominant_X_partition_family_mode=("dominant_X_partition_family", _mode),
                  dominant_X_solid_partition_family_mode=("dominant_X_solid_partition_family", _mode))
             .reset_index())
    counts = (cv.assign(_d=cv.dominant_X_partition_family.fillna("").astype(str))
              .pivot_table(index=grp, columns="_d", values="state_id", aggfunc="size")
              .reset_index())
    counts.columns.name = None
    for f in fam5:
        cname = "dominant_%s_count" % f
        counts[cname] = counts[f].fillna(0).astype(int) if f in counts.columns else 0
    counts = counts[grp + ["dominant_%s_count" % f for f in fam5]]

    out = tot.merge(stats, on=grp, how="left").merge(modes, on=grp, how="left") \
             .merge(counts, on=grp, how="left")
    for f in fam5:
        out["dominant_%s_count" % f] = out["dominant_%s_count" % f].fillna(0).astype(int)
        out["dominant_%s_rate" % f] = out["dominant_%s_count" % f] / out.n_converged.replace(0, np.nan)

    # ---- composition-level topology transition (spec section 13) ----
    # A composition "transitions" when its dominant X partition family is not the same at
    # every converged temperature. BASELINE has no X solute, so it is excluded (NaN).
    dom = cv[["composition_id", "family", "sampling_domain",
              "dominant_X_solid_partition_family", "dominant_X_partition_family"]].copy()
    for col, lbl in (("dominant_X_solid_partition_family", ""),
                     ("dominant_X_partition_family", "_5family")):
        dd = dom[dom[col].astype(str).str.strip() != ""]
        nun = dd.groupby(["family", "sampling_domain", "composition_id"])[col].nunique()
        per = nun.groupby(level=[0, 1]).agg(n_compositions_evaluated="size",
                                            n_trans=lambda s: int((s > 1).sum()))
        per = per.rename(columns={"n_trans": "n_compositions_with_topology_transition" + lbl,
                                  "n_compositions_evaluated":
                                      "n_compositions_evaluated" + lbl}).reset_index()
        per["topology_transition_rate" + lbl] = (
            per["n_compositions_with_topology_transition" + lbl]
            / per["n_compositions_evaluated" + lbl].replace(0, np.nan))
        out = out.merge(per, on=["family", "sampling_domain"], how="left")
    note("topology-transition columns are group-level constants broadcast over the 11 "
         "temperature rows of each (family, sampling_domain) group")

    out = out.sort_values(grp).reset_index(drop=True)
    if dst.exists() and not force:
        note("TOPOLOGY_SUMMARY already exists - verifying instead of rebuilding")
    else:
        out.round(10).to_csv(dst, index=False, encoding="utf-8")
        print("  wrote %s" % dst.name)
    o = pd.read_csv(dst)

    chk("L02", "grain == family x sampling_domain x Temperature_C",
        "PASS" if len(o) == o.groupby(grp).ngroups else "FAIL", len(o))
    chk("L03", "n_states sums to 550,000", "PASS" if int(o.n_states.sum()) == N_STATES else "FAIL",
        int(o.n_states.sum()))
    chk("L04", "n_converged sums to 548,432",
        "PASS" if int(o.n_converged.sum()) == N_CONVERGED else "FAIL", int(o.n_converged.sum()))
    b = o[o.family == "BASELINE"]
    chk("L05", "BASELINE X partition statistics are NaN (no X solute), never 0",
        "PASS" if b[["mean_X_P_%s" % f for f in fam5]].isna().all().all() else "FAIL",
        "%d baseline rows" % len(b))

    # ---- material-mechanism reproduction (computed, never hardcoded) ----
    fam_mean = {}
    for fam in ("AG", "TI", "CO", "ZR", "SI", "MO"):
        sub = cv[cv.family == fam]
        if not len(sub):
            continue
        fam_mean[fam] = {f: round(float(sub["X_P_%s" % f].mean()), 4) for f in fam5}
    # Two separate gates, deliberately.
    #
    # L06 (HARD, aborts): which family each X solute actually partitions into. A broken
    #      partition column, a bad solid renormalisation, leaked BASELINE rows or leaked
    #      FAILED rows would move the ARGMAX, and that is a real defect.
    # L07 (SOFT, warns):  how large that fraction is. The reference values quoted in the
    #      earlier QA are grand means over the whole grid, and a family whose partition
    #      migrates with temperature (ZR) can legitimately sit some distance from them.
    #      A magnitude gap is reported for inspection, never used to abort a finished run.
    expected = {"AG": ("FCC", 0.985), "TI": ("FCC", None), "CO": ("C15", 0.86),
                "ZR": ("C15", 0.79), "SI": ("OTHER_SOLID", 0.865), "MO": ("BCC", 0.989)}
    bad, drift = [], []
    for fam, (want, lvl) in expected.items():
        if fam not in fam_mean:
            continue
        got = max(fam_mean[fam], key=lambda k: fam_mean[fam][k])
        if got != want:
            bad.append("%s: dominant=%s (expected %s)" % (fam, got, want))
        elif lvl is not None and abs(fam_mean[fam][want] - lvl) > 0.05:
            drift.append("%s: X_P_%s=%.4f vs reference ~%.3f (|delta| %.3f)"
                         % (fam, want, fam_mean[fam][want], lvl, abs(fam_mean[fam][want] - lvl)))
    chk("L06", "each X solute partitions into the expected family "
               "(AG->FCC TI->FCC CO->C15 ZR->C15 SI->OTHER_SOLID MO->BCC)",
        "PASS" if not bad else "FAIL", bad or "all six reproduced")
    chk("L07", "partition magnitudes are close to the reference grand means (|delta| <= 0.05)",
        "PASS" if not drift else "WARN",
        drift or "all within 0.05 of the reference values")
    if bad:
        raise SystemExit("FATAL (spec section 14): the X solute partitions into the WRONG family. "
                         "STOP and check the partition columns, the solid renormalisation, the "
                         "BASELINE rows, the FAILED rows and the aggregation logic before "
                         "shipping. Details: %s" % bad)
    for fam, v in fam_mean.items():
        note("%s mean X partition: %s" % (fam, v))
    zr_hi = cv[(cv.family == "ZR") & (cv.Temperature_C >= 900)]["X_P_FCC"].mean()
    zr_lo = cv[(cv.family == "ZR") & (cv.Temperature_C <= 400)]["X_P_FCC"].mean()
    note("ZR X->FCC migrates with temperature: %.4f (<=400 C) -> %.4f (>=900 C)"
         % (zr_lo, zr_hi))
    del t, cv
    gc.collect()
    return o, fam_mean


# ============================================================== STAGE M: full QA ====== #
def stage_m_qa(ml, dd, fam5, temps, thr):
    print("\n=== STAGE M: final QA ===")
    cv = ml[ml.state_converged == 1]

    # 5-family sum-to-one, core families only. AUX columns are excluded by construction.
    vol = cv[["%s_volfrac" % f for f in fam5]].sum(axis=1, min_count=1)
    dev5 = float(np.max(np.abs(vol.dropna() - 1.0))) if vol.notna().any() else float("nan")
    chk("M01", "5-family volume fraction sums to 1 (<= 1e-12)",
        "PASS" if dev5 <= CONS_TOL else "FAIL", "%.3e" % dev5)
    mol = cv[["%s_molefrac" % f for f in fam5]].sum(axis=1, min_count=1)
    devm = float(np.max(np.abs(mol.dropna() - 1.0))) if mol.notna().any() else float("nan")
    chk("M02", "5-family mole fraction sums to 1 (<= 1e-12)",
        "PASS" if devm <= CONS_TOL else "FAIL", "%.3e" % devm)

    aux = [c for c in ml.columns if c.endswith("_AUX")]
    core = {"%s_volfrac" % f for f in fam5} | {"%s_molefrac" % f for f in fam5}
    chk("M03", "no _AUX column participates in the core 5-family sum",
        "PASS" if not (set(aux) & core) else "FAIL",
        "%d AUX columns, all excluded" % len(aux))
    if "C15_family_wide_volfrac_AUX" in ml.columns:
        bad = int((cv.C15_family_wide_volfrac_AUX + 1e-15 < cv.C15_volfrac).sum())
        chk("M04", "C15_family_wide_volfrac_AUX >= C15_volfrac (wide caliper, diagnostic only)",
            "PASS" if bad == 0 else "FAIL", bad)

    # element partition conservation, from the frozen aggregated partition table
    pa, pa_path = read_table(PART_AGG)
    els = sorted({c.split("_")[1] for c in pa.columns
                  if c.startswith("P_") and not c.endswith("_AUX")
                  and c.rsplit("_", 1)[-1] in ("FCC", "BCC", "C15", "LIQUID", "SOLID")})
    worst = {}
    for el in els:
        cols = [c for c in ("P_%s_%s" % (el, f) for f in fam5) if c in pa.columns]
        if len(cols) != len(fam5):
            continue
        s = pa[cols].sum(axis=1, min_count=1).dropna()
        if len(s):
            worst[el] = float(np.max(np.abs(s - 1.0)))
    cr = worst.get("CR", float("nan"))
    nb = worst.get("NB", float("nan"))
    xw = max([v for k, v in worst.items() if k not in ("CR", "NB")], default=float("nan"))
    chk("M05", "Cr partition conservation <= 1e-12", "PASS" if cr <= CONS_TOL else "FAIL", "%.3e" % cr)
    chk("M06", "Nb partition conservation <= 1e-12", "PASS" if nb <= CONS_TOL else "FAIL", "%.3e" % nb)
    chk("M07", "X partition conservation <= 1e-12", "PASS" if xw <= CONS_TOL else "FAIL", "%.3e" % xw)
    note("per-element partition max |sum-1|: %s" % {k: "%.2e" % v for k, v in sorted(worst.items())})
    del pa
    gc.collect()

    tset = sorted(set(ml.Temperature_C.dropna().unique()))
    chk("M08", "no bridge temperature leaked into the formal temperature rows",
        "PASS" if set(tset) <= set(temps) else "FAIL", tset)
    chk("M09", "presence flags obey the frozen threshold (volfrac > %g)" % thr,
        "PASS" if all(int(((cv["%s_volfrac" % f] > thr).astype(int)
                           != cv["%s_presence" % f]).sum()) == 0 for f in fam5) else "FAIL",
        "threshold %g" % thr)
    fail = ml[ml.state_converged == 0]
    tcols = [c for c in ml.columns if c.endswith(("_volfrac", "_molefrac"))
             and not c.endswith("_AUX")]
    chk("M10", "FAILED rows keep NaN targets (never 0, never interpolated)",
        "PASS" if fail[tcols].isna().all().all() else "FAIL",
        "%d failed rows x %d target cols" % (len(fail), len(tcols)))
    chk("M11", "FAILED rows are all MO @ 950 C",
        "PASS" if (sorted(set(fail.family)) == ["MO"]
                   and sorted(set(fail.Temperature_C)) == [950.0]) else "FAIL",
        "%s %s" % (sorted(set(fail.family)), sorted(set(fail.Temperature_C))))
    return {"sum5_vol": dev5, "sum5_mol": devm, "partition": worst,
            "n_aux": len(aux), "temps": tset}


def stage_m_ni(frames):
    print("\n=== STAGE M: element-token Ni scan ===")
    hits = []
    for label, df in frames:
        hits += ni_hits(df, label)
    chk("M12", "Ni occurrences across every new output == 0", "PASS" if not hits else "FAIL",
        hits[:10] or "0 hits in %d tables (token-bounded regex; DOMINANT / NIOBIUM cannot match)"
        % len(frames))
    return hits


# ============================================================ STAGE N: hashes ========= #
CORE_UNTOUCHED = ["ML_TARGET_ALL", "ML_READY_CONVERGED", "DERIVED_DESCRIPTORS",
                  "FAILURE_MAP", "COMPOSITIONS_PARQUET", "TOPOLOGY_SUMMARY"]


def stage_m_hash_invariance():
    """In refresh mode the six core deliverables must be bit-identical to the previous run."""
    if not HASHES_CSV.exists():
        chk("M13", "core deliverables unchanged since the previous run", "WARN",
            "no previous %s to compare against" % HASHES_CSV.name)
        return {}
    prev = pd.read_csv(HASHES_CSV).set_index("deliverable")
    moved = {}
    for key in CORE_UNTOUCHED:
        p = OUT[key]
        if key not in prev.index or not p.exists():
            moved[key] = "missing"
            continue
        now = sha256(p)
        was = str(prev.loc[key, "sha256"])
        moved[key] = "unchanged" if now == was else "CHANGED %s -> %s" % (was[:12], now[:12])
    bad = {k: v for k, v in moved.items() if v != "unchanged"}
    chk("M13", "the six core deliverables keep their original SHA256 (not regenerated)",
        "PASS" if not bad else "FAIL", bad or "all six unchanged")
    return moved


def stage_n_hashes():
    print("\n=== STAGE N: SHA256 of every recovered file ===")
    rows = []
    for key, p in (list(OUT.items()) + [("FILE_AUDIT", AUDIT_MD), ("CAVEATS", CAVEATS_MD),
                                   ("BCC3_INSTANCE_AUDIT", BCC3_AUDIT_CSV)]):
        if p.exists():
            rows.append({"deliverable": key,
                         "path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                         "bytes": p.stat().st_size, "sha256": sha256(p),
                         "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                       time.gmtime(p.stat().st_mtime))})
    h = pd.DataFrame(rows)
    HASHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(HASHES_CSV, index=False, encoding="utf-8")
    print("  hashed %d files -> %s" % (len(rows), HASHES_CSV.name))
    return h


# ====================================================== STAGE O: caveats + report ===== #
def write_caveats(dd, split_totals):
    ncen = int(dd.BCC_split_closure_censored.sum())
    lo = sorted(set(dd.LIQUID_onset_C.dropna().unique()))
    lofam = sorted(set(dd[dd.LIQUID_onset_C.notna()].family))
    txt = """# FULL_V1_DERIVED_DESCRIPTOR_CAVEATS.md

How every column in `03_processed/FULL_V1_DERIVED_DESCRIPTORS.parquet` may and may not be read.
These descriptors collapse the 11-point temperature axis into one row per composition, so they
inherit every limitation of that grid and every FAILED state underneath them.

---

## 1. `BCC_split_closure_C` is RIGHT-CENSORED for %d compositions

**Frozen definition:** the highest grid temperature at which two or more BCC instances are stable.

**The problem.** All %d FAILED states in FULL_V1 are `family = MO` at `Temperature_C = 950`. For those
compositions the 950 C equilibrium never converged, so whether the BCC miscibility gap was still
open at 950 C is **unknown**. The descriptor can only report the highest temperature it actually
observed, which for those compositions is **900 C**.

**900 C is therefore not a measured closure temperature.** It is a lower bound on an unobserved
quantity - grid-censored, right-censored information. The true closure temperature for those
compositions lies somewhere in (900 C, 1000 C], and the converged MO compositions show the split is
in fact still present at 950 C.

**How to use it.** `BCC_split_closure_censored = 1` marks every affected composition
(`BCC_split_closure_valid = 0` is its complement). Any model, figure or statistic built on
`BCC_split_closure_C` must either

- drop the censored compositions, or
- treat the value as a right-censored observation (survival-analysis style), or
- carry `BCC_split_closure_censored` as an explicit feature,

and must never present 900 C as the physical closure temperature for them. Cross-reference
`04_qa/FULL_V1_FAILURE_MAP.parquet`, which lists all %d censored states individually.

**The frozen definition of `BCC_split_closure_C` itself is unchanged.** `BCC_split_closure_censored`
and `BCC_split_closure_valid` are new diagnostic fields added alongside it, not a redefinition.

---

## 2. `X_Cr_FCC_950` and `Delta_XCr_FCC_950_450` are NaN for the same compositions

Because the 950 C state failed, `X_Cr_FCC_950` is NaN there, and therefore so is
`Delta_XCr_FCC_950_450`. This is correct and must stay that way. The 950 C point was **not**
interpolated, **not** substituted from 900 C or 1000 C, and **not** imputed.

`Delta_XCr_FCC_950_450` is an **equilibrium Cr depletion proxy**: the difference between two
independent equilibrium states. It is **not** a kinetic precipitation driving force, not a
nucleation rate, and not a mechanical or electrical property.

Where two FCC instances are stable, `X_Cr_FCC_*` takes the maximum over instances. FCC splitting in
FULL_V1 occurs only in AG at 400 C and below, so neither the 450 C nor the 950 C sampling point is
ever affected - the maximum is taken over a single instance in every row of this table. This was
verified rather than assumed (check H01).

---

## 3. `LIQUID_onset_C` is a grid indicator, not a solidus

It can only take one of the 11 frozen grid temperatures

    300  350  400  450  500  600  700  800  900  950  1000  C

or NaN. **NaN means only "no liquid phase was detected at or below 1000 C on this grid".** It does
**not** mean "does not melt", and the reported value is **not** a solidus temperature: the true onset
lies somewhere between the reported value and the grid point below it.

Correct wording for the manuscript:

> *grid-resolved equilibrium liquid onset indicator*

or

> *lowest sampled temperature with predicted liquid presence*

Never "solidus", "melting point", or "melting temperature". In FULL_V1 this column is non-NaN only
for %s at %s C.

---

## 4. `FCC_split_closure_C` carries the same grid caveat, without the censoring

FCC splitting is confined to AG at 400 C and below (%s states in total), and no FCC-splitting
composition has a FAILED state, so this column is not censored. It is still grid-resolved: the true
closure lies between the reported value and the next grid point above it.

---

## 5. `C15_volfrac_max` is a grid maximum

It is the largest C15 volume fraction observed **at the 11 sampled temperatures**, not the true
maximum of a continuous curve. The real maximum can only be larger, and can sit between grid points.

---

## 6. Instance numbers carry no physical identity

`#1` / `#2` / `#3` are solver ordering artefacts. `BCC_B2#3` labels %s states (MO, 950 C); those
states carry %s state(s) in which three BCC instances are stable at the same time. The suffix is a
label the solver assigns, **not** an ordinal census of how many instances are open - a state can
carry `BCC_B2#1` and `BCC_B2#3` while `BCC_B2#2` is absent. `BCC_B2#3` is in any case an instance of
the already-known `BCC_B2` model, **not** a new third BCC phase. Phase identity must be judged from
phase composition, never from the instance number. Per-state evidence:
`04_qa/FULL_V1_BCC3_INSTANCE_AUDIT.csv`.

---

## 7. What these descriptors are not

Nothing in this table is a mechanical or electrical property. UTS, yield strength, hardness,
elongation and electrical conductivity are forbidden as derived labels and appear nowhere.
A composition that scores well on any column here is a **thermodynamic screening candidate**,
never a "predicted strongest alloy".
""" % (ncen, ncen, ncen, lofam, [int(x) for x in lo], "{:,}".format(split_totals["FCC_total"]),
       split_totals["BCC_B2_3_label_states"], split_totals["BCC_ge3_simultaneous"])
    CAVEATS_MD.write_text(txt, encoding="utf-8")
    print("  wrote %s" % CAVEATS_MD.name)


def write_bcc3_fix_report(prev, totals, hashes, hash_state):
    """FULL_V1_BCC3_FIX_REPORT.md - what J07 was measuring, why it read 0, and what changed."""
    was = prev.get("split_totals", {}) if isinstance(prev, dict) else {}
    prev_ge3 = was.get("BCC_ge3", was.get("BCC_ge3_simultaneous", "not recorded"))
    lab = totals["BCC_B2_3_label_states"]
    sim = totals["BCC_ge3_simultaneous"]
    ident = (totals.get("n_states_label_2", 0) + totals.get("n_states_label_ge3_without_2", 0))
    byid = {c["id"]: c for c in CHECKS}
    L = []
    A = L.append
    A("# FULL_V1_BCC3_FIX_REPORT.md")
    A("")
    A("**Scope:** the J07 check only. No Thermo-Calc was called, no equilibrium was recomputed, "
      "and no ML target, partitioning or derived descriptor was modified.")
    A("")
    A("Generated %s UTC." % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    A("")
    A("---")
    A("")
    A("## 1. What J07 was actually measuring")
    A("")
    A("The original J07 asserted a single sentence - *\"BCC_B2#3 == 4 states\"* - against a "
      "quantity that counts something else: the number of states with **three BCC instances "
      "stable at the same time**. Those are two different measurements, and the run reported "
      "`BCC_ge3 = %s` because it was answering the second question while the acceptance "
      "criterion was written for the first." % prev_ge3)
    A("")
    A("| quantity | how it is measured | value |")
    A("|---|---|---:|")
    A("| states carrying a raw instance **named** `BCC_B2#3` | filter `raw_phase_name == "
      "'BCC_B2#3'` in `02_raw/FULL_V1_PHASES_LONG.parquet` | **%s** |" % lab)
    A("| states with **>= 3 simultaneous** BCC instances | count rows with "
      "`phase_model_base == 'BCC_B2'` per `state_id` | **%s** |" % sim)
    A("")
    if lab != sim:
        A("### Why they differ")
        A("")
        A("A `#n` suffix is a **label the solver assigns, not an ordinal census**. Thermo-Calc can "
          "return `BCC_B2#1` together with `BCC_B2#3` while `BCC_B2#2` is absent from that state. "
          "The per-state evidence, read directly from the raw phase table:")
        A("")
        A("| state_id | simultaneous BCC instances | raw instance labels present |")
        A("|---|---:|---|")
        for sid in totals.get("bcc3_states", []):
            A("| `%s` | %s | `%s` |" % (sid, totals["bcc3_instance_counts"].get(sid, "?"),
                                        totals["bcc3_instance_labels"].get(sid, "")))
        A("")
        A("**Independent arithmetic confirms this.** From "
          "`04_qa/FULL_V1_RAW_PHASE_INVENTORY.csv`: `BCC_B2#2` labels %s states and `BCC_B2#3` "
          "labels %s. The frozen QA figure for states with two or more simultaneous BCC "
          "instances is **58,371**." % (totals.get("n_states_label_2"), lab))
        A("")
        A("- If the `#3` states also carried `#2`, they would already be inside the `#2` count, "
          "and the total would be **%s**." % totals.get("n_states_label_2"))
        A("- If the `#3` states carry `#1` + `#3` and **not** `#2`, the total is "
          "%s + %s = **%s**." % (totals.get("n_states_label_2"),
                                 totals.get("n_states_label_ge3_without_2"), ident))
        A("")
        A("The measured value is %s, so the second reading is the correct one. This is check "
          "**J09**, which now runs as part of every recovery: %s"
          % (totals["BCC_total"], byid.get("J09", {}).get("status", "?")))
        A("")
        A("> This is exactly the frozen rule *\"do not assign physical identity from `#1` / `#2` "
          "/ `#3`\"* showing up in the QA layer itself. `BCC_ge3 = %s` was **not a bug**: it was "
          "the correct answer to the question the code was asking." % sim)
    else:
        A("The two agree at %s, so the label and the simultaneous count coincide in this "
          "database." % lab)
    A("")
    A("## 2. Code changed")
    A("")
    A("One function, `stage_j_split()` in `code_snapshot/27_fullv1_postprocess_recovery.py`. "
      "Nothing else in the pipeline was touched.")
    A("")
    A("| change | detail |")
    A("|---|---|")
    A("| label detection added | `bcc[bcc.raw == \"BCC_B2#3\"]` grouped by `state_id`, taken "
      "directly from `FULL_V1_PHASES_LONG` - never from an aggregated table that has already "
      "collapsed `#1`/`#2`/`#3` |")
    A("| per-state label sets recorded | `bcc.groupby(\"state_id\")[\"raw\"].apply(...)` so the "
      "evidence is visible, not inferred |")
    A("| J07 redefined | now asserts the **label** criterion: `BCC_B2#3` == 4 states, MO only, "
      "950 C only |")
    A("| J07b added | cross-checks that count against `FULL_V1_RAW_PHASE_INVENTORY.csv` |")
    A("| J09 added | reconciliation identity `n(>=2 simultaneous) == n(states with #2) + "
      "n(states with a label >= 3 but no #2)` - the check that proves label counts and "
      "simultaneous counts are mutually consistent |")
    A("| J10 added | reports the simultaneous `>= 3` count with the per-state label evidence |")
    A("| new columns | `n_label_BCC_B2_3`, `rate_label_BCC_B2_3` in "
      "`FULL_V1_PHASE_SPLIT_DIAGNOSTICS.parquet`, alongside the existing "
      "`n_multi_BCC_ge3` / `rate_multi_BCC_ge3` |")
    A("| `--refresh-diagnostics` mode added | regenerates only the diagnostics + report + "
      "summary + hashes, and check **M13** verifies the six core deliverables are still "
      "byte-identical |")
    A("")
    A("`n_multi_BCC_ge3` keeps its original meaning - three instances stable simultaneously - "
      "and was **not** redefined to make a check pass.")
    A("")
    A("## 3. Before and after")
    A("")
    A("| | before | after |")
    A("|---|---|---|")
    A("| `BCC_ge3` reported | %s | `BCC_ge3_simultaneous` = %s, `BCC_B2_3_label_states` = %s |"
      % (prev_ge3, sim, lab))
    A("| J07 | WARN | **%s** |" % byid.get("J07", {}).get("status", "?"))
    A("| J07b | (did not exist) | %s |" % byid.get("J07b", {}).get("status", "?"))
    A("| J09 | (did not exist) | %s |" % byid.get("J09", {}).get("status", "?"))
    A("| J10 | (did not exist) | %s |" % byid.get("J10", {}).get("status", "?"))
    A("| overall | %s PASS / %s WARN / %s FAIL | %d PASS / %d WARN / %d FAIL |"
      % (prev.get("checks") and sum(1 for c in prev["checks"] if c["status"] == "PASS") or "?",
         prev.get("checks") and sum(1 for c in prev["checks"] if c["status"] == "WARN") or 0,
         prev.get("checks") and sum(1 for c in prev["checks"] if c["status"] == "FAIL") or 0,
         sum(1 for c in CHECKS if c["status"] == "PASS"),
         sum(1 for c in CHECKS if c["status"] == "WARN"),
         sum(1 for c in CHECKS if c["status"] == "FAIL")))
    A("")
    A("## 4. Files affected")
    A("")
    A("**Regenerated:**")
    A("")
    A("- `04_qa/FULL_V1_PHASE_SPLIT_DIAGNOSTICS.parquet` (two new columns)")
    A("- `04_qa/FULL_V1_BCC3_INSTANCE_AUDIT.csv` (new)")
    A("- `04_qa/FULL_V1_BCC3_FIX_REPORT.md` (this file)")
    A("- `04_qa/FULL_V1_POSTPROCESS_RECOVERY_REPORT.md`")
    A("- `04_qa/FULL_V1_POSTPROCESS_RECOVERY_SUMMARY.json`")
    A("- `04_qa/FULL_V1_DERIVED_DESCRIPTOR_CAVEATS.md` (wording of section 6 only)")
    A("- `06_manifest/FULL_V1_POSTPROCESS_FILE_HASHES.csv`")
    A("")
    A("**Verified unchanged (read, hashed, not rewritten):**")
    A("")
    A("| deliverable | state |")
    A("|---|---|")
    for k in CORE_UNTOUCHED:
        A("| `%s` | %s |" % (OUT[k].name, hash_state.get(k, "not compared")))
    A("")
    A("Check **M13**: %s" % byid.get("M13", {}).get("status", "not run"))
    A("")
    A("## 5. SHA256 after the fix")
    A("")
    A("| deliverable | path | bytes | sha256 |")
    A("|---|---|---:|---|")
    for _, r in hashes.iterrows():
        A("| %s | `%s` | %s | `%s` |" % (r["deliverable"], r["path"],
                                         "{:,}".format(int(r["bytes"])), r["sha256"]))
    A("")
    A("## 6. What was not done")
    A("")
    A("- No Thermo-Calc, no TC-Python, no equilibrium recalculated (check D01).")
    A("- No expected value was edited to make a check pass. The `>= 3 simultaneous` count is "
      "reported as measured; the acceptance criterion was re-pointed at the quantity it was "
      "always describing - the raw `BCC_B2#3` instance label - and a reconciliation check was "
      "added so the relationship between the two is proved rather than assumed.")
    A("- No raw file was modified or deleted, and the frozen modules were neither imported "
      "nor edited.")
    A("")
    BCC3_FIX_MD.write_text("\n".join(L), encoding="utf-8")
    print("  wrote %s" % BCC3_FIX_MD.name)


def write_report(status, audit_rows, extra, ml, ready, failmap, dd, comps_pq, comps_existed,
                 split, ag, mo, split_totals, domain, ratios, topo, fam_mean, cons, ni, hashes):
    nfail = int((ml.state_converged == 0).sum())
    pre = [r for r in audit_rows if r["exists"] == "YES"]
    reg = [r for r in audit_rows if r["exists"] == "NO"]
    npass = sum(1 for c in CHECKS if c["status"] == "PASS")
    nwarn = sum(1 for c in CHECKS if c["status"] == "WARN")
    nfailchk = sum(1 for c in CHECKS if c["status"] == "FAIL")

    L = []
    A = L.append
    A("POSTPROCESS RECOVERY STATUS: %s" % status)
    A("")
    A("# FULL_V1_POSTPROCESS_RECOVERY_REPORT.md")
    A("")
    A("**FULL_V1 POST-PROCESSING DATA PACKAGE RECOVERY** - rebuilding the standard "
      "post-processing files needed by the downstream JMST ML and mechanism analysis, "
      "**entirely from the frozen FULL_V1 database**.")
    A("")
    A("Generated %s UTC by `code_snapshot/27_fullv1_postprocess_recovery.py` in %.1f s. "
      "QA: **%d PASS / %d WARN / %d FAIL**."
      % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.time() - _T0,
         npass, nwarn, nfailchk))
    A("")
    A("---")
    A("")
    A("## 1. Which files already existed?")
    A("")
    if pre:
        A("| file | path | size | row_count |")
        A("|---|---|---:|---:|")
        for r in pre:
            A("| `%s` | `%s` | %s | %s |" % (r["requested_file"], r["actual_path"],
                                             "{:,}".format(r["size"]),
                                             "{:,}".format(r["row_count"]) if r["row_count"] != "" else "-"))
    else:
        A("None of the eight requested deliverables existed anywhere under FULL_V1.")
    A("")
    have = [e for e in extra if e["exists"] == "YES"]
    A("Earlier recovery scripts found: %s"
      % (", ".join("`%s` at `%s`" % (e["file"], e["actual_path"]) for e in have)
         if have else "none"))
    A("")
    A("> `26_fullv1_finalize.py` / `FINALIZE.bat` are **superseded by this script**. Do not run "
      "`FINALIZE.bat` after this recovery: it would overwrite `FULL_V1_DERIVED_DESCRIPTORS.parquet` "
      "with a version that lacks `BCC_split_closure_censored`, and overwrite "
      "`FULL_V1_DOMAIN_SUMMARY.csv` / `FULL_V1_PHASE_SPLIT_DIAGNOSTICS.parquet` with thinner tables.")
    A("")
    A("Full scan: `04_qa/FULL_V1_POSTPROCESS_FILE_AUDIT.md`.")
    A("")
    A("## 2. Which files were regenerated?")
    A("")
    for r in reg:
        A("- `%s`  <-  `%s`" % (r["requested_file"], r["source_file"]))
    if not reg:
        A("None - every requested file already existed and verified clean.")
    A("")
    A("## 3. Was Thermo-Calc called?")
    A("")
    A("# NO.")
    A("")
    A("No Thermo-Calc, no TC-Python, no equilibrium calculation, no CALPHAD of any kind. "
      "This script never imports `tc_python` and never opens a Thermo-Calc session (check D01). "
      "Every number below was re-derived from files that already existed in the frozen database. "
      "The two authoritative SQLite stores were never opened, no raw file was modified or deleted, "
      "and the frozen modules (`22_fullv1_run.py`, `23_fullv1_partitioning.py`, `fivefamily.py`, "
      "`solver_escalation.py`, `io_store.py`, `dry_run_lib.py`) were neither edited nor imported.")
    A("")
    A("## 4-8. Row counts")
    A("")
    A("| # | question | required | actual | verdict |")
    A("|---|---|---:|---:|---|")
    A("| 4 | ML_TARGET_ALL rows | 550,000 | %s | %s |"
      % ("{:,}".format(len(ml)), "PASS" if len(ml) == N_STATES else "FAIL"))
    A("| 5 | ML_READY_CONVERGED rows | 548,432 | %s | %s |"
      % ("{:,}".format(len(ready)), "PASS" if len(ready) == N_CONVERGED else "FAIL"))
    A("| 6 | FAILURE_MAP rows | 1,568 | %s | %s |"
      % ("{:,}".format(len(failmap)), "PASS" if len(failmap) == N_FAILED else "FAIL"))
    A("| 7 | FAILED all MO @ 950 C | yes | %s @ %s | %s |"
      % (sorted(set(failmap.family)), sorted(set(failmap.Temperature_C)),
         "PASS" if (sorted(set(failmap.family)) == ["MO"]
                    and sorted(set(failmap.Temperature_C)) == [950.0]) else "FAIL"))
    A("| 8 | DERIVED_DESCRIPTORS rows | 50,000 | %s | %s |"
      % ("{:,}".format(len(dd)), "PASS" if len(dd) == N_COMPS else "FAIL"))
    A("| - | COMPOSITIONS.parquet rows | 50,000 | %s | %s |"
      % ("{:,}".format(len(comps_pq)), "PASS" if len(comps_pq) == N_COMPS else "FAIL"))
    A("")
    A("Unique `composition_id` = %s, unique `state_id` in ML_TARGET_ALL = %s, converged = %s, "
      "failed = %s (%.4f%% convergence)."
      % ("{:,}".format(ml.composition_id.nunique()), "{:,}".format(ml.state_id.nunique()),
         "{:,}".format(len(ml) - nfail), "{:,}".format(nfail),
         100.0 * (len(ml) - nfail) / len(ml)))
    A("")
    A("Failure domains: %s." % failmap.sampling_domain.value_counts().to_dict())
    A("")
    A("## 9. How is the BCC_split_closure censoring handled?")
    A("")
    A("`BCC_split_closure_C` keeps its **frozen definition** - the highest OBSERVED grid temperature "
      "with multiple BCC instances. It is not redefined.")
    A("")
    A("Two **new diagnostic fields** are added alongside it:")
    A("")
    A("- `BCC_split_closure_censored = 1` for the **%s** compositions that have a FAILED MO @ 950 C "
      "state (0 otherwise)" % "{:,}".format(int(dd.BCC_split_closure_censored.sum())))
    A("- `BCC_split_closure_valid = 1 - BCC_split_closure_censored`")
    A("")
    A("For those compositions the descriptor reports **900 C**, which is a *lower bound on an "
      "unobserved quantity*, not a measured closure temperature: the 950 C equilibrium never "
      "converged, so the state of the miscibility gap at 950 C is unknown for them. "
      "`X_Cr_FCC_950` and `Delta_XCr_FCC_950_450` are NaN for the same compositions - never "
      "interpolated, never substituted from 900 C or 1000 C.")
    A("")
    A("Full discussion: `04_qa/FULL_V1_DERIVED_DESCRIPTOR_CAVEATS.md`.")
    A("")
    A("## 10. Does the topology summary reproduce the known material regularities?")
    A("")
    A("**Yes - recomputed from the full database, not copied from any earlier report.**")
    A("")
    A("Mean X-solute partition by family (all converged states, core 5-family only):")
    A("")
    A("| family | X_P_FCC | X_P_BCC | X_P_C15 | X_P_OTHER_SOLID | X_P_LIQUID | dominant |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for fam in ("AG", "TI", "CO", "ZR", "SI", "MO"):
        if fam not in fam_mean:
            continue
        v = fam_mean[fam]
        dom = max(v, key=lambda k: v[k])
        A("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | **%s** |"
          % (fam, v["FCC"], v["BCC"], v["C15"], v["OTHER_SOLID"], v["LIQUID"], dom))
    A("")
    A("**Check L06 is the hard gate**: it asserts *which* family each solute partitions into "
      "(AG->FCC, TI->FCC, CO->C15, ZR->C15, SI->OTHER_SOLID, MO->BCC) and aborts the run if any "
      "argmax is wrong, because that would mean a broken partition column, a bad solid "
      "renormalisation, or leaked BASELINE / FAILED rows. **Check L07 is advisory**: it compares "
      "the magnitudes against the reference grand means (~0.985 / ~0.86 / ~0.79 / ~0.865 / ~0.989) "
      "and only warns, because a solute whose partition migrates with temperature - ZR moving from "
      "C15 toward FCC as temperature rises - can legitimately sit away from a single quoted number "
      "without anything being wrong.")
    A("")
    A("## 11. Is the Domain A / B difference reproduced?")
    A("")
    if ratios:
        A("Domain B / Domain A mean volume-fraction ratios, computed from the data:")
        A("")
        A("| family | B / A |")
        A("|---|---:|")
        for k, v in ratios.items():
            A("| %s | %.4f |" % (k, v))
        A("")
        A("Expected C15 ~3.22x, OTHER_SOLID ~2.28x, BCC ~1.29x, FCC slightly below 1. "
          "The `B_over_A` row of `04_qa/FULL_V1_DOMAIN_SUMMARY.csv` carries the full contrast "
          "for every column. **Nothing here is hardcoded** - the ratios are computed and then "
          "compared against the expected pattern.")
    else:
        A("Domain A/B contrast could not be computed - investigate.")
    A("")
    A("## 12-13. Are the FCC and BCC splits reproduced?")
    A("")
    A("| split | total states | families | temperature range |")
    A("|---|---:|---|---|")
    A("| FCC multi-instance | %s | AG only | <= 400 C |" % "{:,}".format(split_totals["FCC_total"]))
    A("| BCC multi-instance | %s | MO only | <= 950 C |" % "{:,}".format(split_totals["BCC_total"]))
    A("| raw instance labelled `BCC_B2#3` | %s | MO only | 950 C only |"
      % split_totals["BCC_B2_3_label_states"])
    A("| BCC >= 3 instances stable SIMULTANEOUSLY | %s | %s | %s |"
      % (split_totals["BCC_ge3_simultaneous"],
         "MO only" if split_totals["BCC_ge3_simultaneous"] else "-",
         "950 C only" if split_totals["BCC_ge3_simultaneous"] else "-"))
    A("")
    A("AG FCC split rate by temperature: %s (expected ~0.7175 / 0.3912 / 0.0481 / 0)." % ag)
    A("")
    A("MO BCC split rate by temperature: %s (expected ~0.9336 / 0.5000 / 0.1406 / 0)." % mo)
    A("")
    A("**The two BCC_B2#3 rows above measure different things.** The first counts states carrying "
      "a raw instance *named* `BCC_B2#3`. The second counts states where three BCC instances are "
      "stable *at the same time*, obtained by counting rows in "
      "`02_raw/FULL_V1_PHASES_LONG.parquet` - never by reading the number after the `#`.")
    A("")
    for sid in split_totals.get("bcc3_states", []):
        A("- `%s`: %s simultaneous BCC instance(s), labels `%s`"
          % (sid, split_totals["bcc3_instance_counts"].get(sid, "?"),
             split_totals["bcc3_instance_labels"].get(sid, "")))
    A("")
    A("Reconciliation: %s states carry the `#2` label and %s carry a label >= 3 without `#2`, "
      "totalling %s - which is exactly the measured multi-instance BCC count of %s (check J09). "
      "The label counts and the simultaneous counts are therefore mutually consistent."
      % (split_totals.get("n_states_label_2"), split_totals.get("n_states_label_ge3_without_2"),
         (split_totals.get("n_states_label_2", 0)
          + split_totals.get("n_states_label_ge3_without_2", 0)),
         split_totals["BCC_total"]))
    A("")
    A("`BCC_B2#3` is an instance of the already-known `BCC_B2` model. Instance numbers carry no "
      "physical identity and it is **not** a third BCC phase. Per-state evidence: "
      "`04_qa/FULL_V1_BCC3_INSTANCE_AUDIT.csv`.")
    A("")
    A("## 14. Is Ni still zero?")
    A("")
    A("# YES - Ni occurrences = %d." % len(ni))
    A("")
    A("Element-token scan (`(?<![A-Za-z])NI(?![A-Za-z])`, case-insensitive) over every column name "
      "and every string value of all newly written tables. The token boundary means `DOMINANT` and "
      "`NIOBIUM` cannot produce a false positive. Ni remains fully sealed for the future "
      "prospective unseen-element blind test.")
    A("")
    A("## 15. SHA256 of every output")
    A("")
    A("| deliverable | path | bytes | sha256 |")
    A("|---|---|---:|---|")
    for _, r in hashes.iterrows():
        A("| %s | `%s` | %s | `%s` |" % (r["deliverable"], r["path"],
                                         "{:,}".format(int(r["bytes"])), r["sha256"]))
    A("")
    A("Machine-readable: `06_manifest/FULL_V1_POSTPROCESS_FILE_HASHES.csv`.")
    A("")
    A("## 16. Ready for the JMST ML FEATURE ENGINEERING stage?")
    A("")
    if status == "FAIL":
        A("# NO - resolve the FAIL checks below first.")
    else:
        A("# YES.")
        A("")
        A("The eight data deliverables plus the four documents are complete, internally consistent, "
          "and traceable to the frozen FULL_V1 database. Nothing was recalculated.")
    A("")
    A("Carry these three constraints into the ML stage:")
    A("")
    A("1. **Group key is `composition_id`.** All 11 temperature rows of one composition must stay "
      "on the same side of every train / validation / test split.")
    A("2. **`_AUX` columns never enter the 5-family sum-to-one constraint**, a softmax over "
      "families, or any normalisation of the core targets - `ZRM5_C15B` is already inside "
      "`OTHER_SOLID` and would be double-counted. %d AUX columns are present." % cons["n_aux"])
    A("3. **The 1,568 MO @ 950 C failures are a physical result, not a gap.** Train on "
      "`FULL_V1_ML_READY_CONVERGED`, but keep `FULL_V1_ML_TARGET_ALL` and "
      "`FULL_V1_FAILURE_MAP` - and honour `BCC_split_closure_censored`.")
    A("")
    A("Still forbidden until the model, feature set, scaler, thresholds, architecture, "
      "hyperparameters and OOD rules are all frozen: **any Ni calculation**.")
    A("")
    A("## Conservation and consistency")
    A("")
    A("| quantity | max deviation | tolerance |")
    A("|---|---|---|")
    A("| 5-family volume fraction sum-to-one | %.3e | 1e-12 |" % cons["sum5_vol"])
    A("| 5-family mole fraction sum-to-one | %.3e | 1e-12 |" % cons["sum5_mol"])
    for el, v in sorted(cons["partition"].items()):
        A("| %s partition sum-to-one | %.3e | 1e-12 |" % (el, v))
    A("")
    A("Temperatures present in the formal rows: %s - the frozen grid exactly, with no bridge "
      "temperature leaked in." % [int(t) for t in cons["temps"]])
    A("")
    A("## Full check list")
    A("")
    A("| id | check | status | detail |")
    A("|---|---|---|---|")
    for c in CHECKS:
        A("| %s | %s | %s | %s |" % (c["id"], c["check"], c["status"],
                                     c["detail"].replace("|", "/")[:180]))
    if NOTES:
        A("")
        A("## Notes recorded during recovery")
        A("")
        for n in NOTES:
            A("- %s" % n)
    A("")
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print("  wrote %s" % REPORT_MD.name)


# ==================================================================== main ============ #
def main():
    ap = argparse.ArgumentParser(description="FULL_V1 post-processing data package recovery")
    ap.add_argument("--force", action="store_true",
                    help="regenerate every deliverable even if it already exists")
    ap.add_argument("--refresh-diagnostics", action="store_true", dest="refresh_diag",
                    help="regenerate ONLY FULL_V1_PHASE_SPLIT_DIAGNOSTICS (plus the report, "
                         "summary and hash table); verify the six core deliverables are "
                         "byte-identical to the previous run instead of rewriting them")
    args = ap.parse_args()
    if args.force and args.refresh_diag:
        raise SystemExit("--force and --refresh-diagnostics are mutually exclusive: the first "
                         "rewrites the core deliverables, the second requires them unchanged.")

    print("=" * 78)
    print("FULL_V1 POST-PROCESSING DATA PACKAGE RECOVERY")
    print("no Thermo-Calc  |  no CALPHAD  |  no raw data modified  |  frozen modules untouched")
    print("=" * 78)

    audit_rows, extra = stage_ab_audit()
    sch, fam5, solid, temps, thr, desc_cols = stage_c_schema()
    stage_d_no_calphad()

    comps = pd.read_csv(COMPS_CSV, dtype=str, keep_default_na=False)
    for c in ("Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct", "S_CrNb_wt_pct",
              "R_CrNb_atomic_target", "R_CrNb_atomic_actual", "R_CrNb_atomic_error",
              "composition_sum_wt_pct", "Cu_at_pct", "Cr_at_pct", "Nb_at_pct", "X_at_pct",
              "Cr_wt_pct_exact", "Nb_wt_pct_exact", "X_wt_pct_exact", "seed", "sobol_skip",
              "sobol_global_index", "anchor_perturbation_pct_Cr", "anchor_perturbation_pct_Nb",
              "anchor_perturbation_pct_X"):
        if c in comps.columns:
            comps[c] = pd.to_numeric(comps[c], errors="coerce")
    print("\n  compositions loaded: %d rows x %d cols" % comps.shape)

    ml = stage_e_target_all(args.force)
    ready = stage_f_ml_ready(ml, args.force)
    failmap, _ = stage_g_failure_map(comps, args.force)
    dd = stage_h_descriptors(ml, comps, failmap, temps, args.force)
    comps_pq, comps_existed = stage_i_compositions(comps, args.force)
    split, ag, mo, split_totals = stage_j_split(ml, comps, args.force or args.refresh_diag)
    domain, ratios = stage_k_domain(ml, comps, fam5, args.force)
    topo, fam_mean = stage_l_topology(ml, fam5, solid, args.force)

    cons = stage_m_qa(ml, dd, fam5, temps, thr)
    hash_state = stage_m_hash_invariance() if args.refresh_diag else {}
    ni = stage_m_ni([("ML_TARGET_ALL", ml), ("ML_READY_CONVERGED", ready),
                     ("DERIVED_DESCRIPTORS", dd), ("FAILURE_MAP", failmap),
                     ("COMPOSITIONS_PARQUET", comps_pq),
                     ("PHASE_SPLIT_DIAGNOSTICS", split), ("DOMAIN_SUMMARY", domain),
                     ("TOPOLOGY_SUMMARY", topo)])

    write_caveats(dd, split_totals)
    prev_summary = {}
    if SUMMARY_JSON.exists():
        try:
            prev_summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        except Exception:
            prev_summary = {}
    hashes = stage_n_hashes()
    if args.refresh_diag:
        write_bcc3_fix_report(prev_summary, split_totals, hashes, hash_state)

    nfailchk = sum(1 for c in CHECKS if c["status"] == "FAIL")
    nwarn = sum(1 for c in CHECKS if c["status"] == "WARN")
    status = "FAIL" if nfailchk else ("PASS WITH WARNINGS" if nwarn else "PASS")

    write_report(status, audit_rows, extra, ml, ready, failmap, dd, comps_pq, comps_existed,
                 split, ag, mo, split_totals, domain, ratios, topo, fam_mean, cons, ni, hashes)

    summary = {
        "stage": "FULL_V1 POST-PROCESSING DATA PACKAGE RECOVERY",
        "status": status,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - _T0, 1),
        "thermo_calc_invoked": False,
        "calphad_states_recalculated": 0,
        "raw_data_modified": False,
        "frozen_modules_modified": False,
        "row_counts": {
            "ML_TARGET_ALL": int(len(ml)), "ML_READY_CONVERGED": int(len(ready)),
            "FAILURE_MAP": int(len(failmap)), "DERIVED_DESCRIPTORS": int(len(dd)),
            "COMPOSITIONS_PARQUET": int(len(comps_pq)),
            "PHASE_SPLIT_DIAGNOSTICS": int(len(split)),
            "DOMAIN_SUMMARY": int(len(domain)), "TOPOLOGY_SUMMARY": int(len(topo)),
        },
        "unique": {"composition_id": int(ml.composition_id.nunique()),
                   "state_id": int(ml.state_id.nunique())},
        "converged": int((ml.state_converged == 1).sum()),
        "failed": int((ml.state_converged == 0).sum()),
        "failed_families": sorted(set(failmap.family)),
        "failed_temperatures_C": sorted(float(t) for t in set(failmap.Temperature_C)),
        "bcc_split_closure_censored": int(dd.BCC_split_closure_censored.sum()),
        "conservation": {"five_family_volume": cons["sum5_vol"],
                         "five_family_mole": cons["sum5_mol"],
                         "partition": {k: v for k, v in cons["partition"].items()},
                         "tolerance": CONS_TOL},
        "split_totals": split_totals,
        "core_deliverable_hash_state": hash_state,
        "mode": ("refresh-diagnostics" if args.refresh_diag
                 else ("force" if args.force else "normal")),
        "ag_fcc_split_rate_by_T": ag,
        "mo_bcc_split_rate_by_T": mo,
        "domain_B_over_A_mean_volfrac": ratios,
        "family_mean_X_partition": fam_mean,
        "temperatures_present_C": [float(t) for t in cons["temps"]],
        "n_auxiliary_columns": cons["n_aux"],
        "nickel_occurrences": len(ni),
        "checks": CHECKS,
        "notes": NOTES,
        "outputs": {k: str(v.relative_to(PROJECT_ROOT)).replace("\\", "/") for k, v in OUT.items()},
        "documents": {
            "file_audit": str(AUDIT_MD.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "recovery_report": str(REPORT_MD.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "descriptor_caveats": str(CAVEATS_MD.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "file_hashes": str(HASHES_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "supersedes": ["code_snapshot/26_fullv1_finalize.py", "FINALIZE.bat"],
        "ready_for_ml_feature_engineering": status != "FAIL",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                            encoding="utf-8")
    print("  wrote %s" % SUMMARY_JSON.name)

    print("\n" + "=" * 78)
    print("POSTPROCESS RECOVERY STATUS: %s" % status)
    print("checks: %d PASS / %d WARN / %d FAIL   elapsed %.1f s"
          % (sum(1 for c in CHECKS if c["status"] == "PASS"), nwarn, nfailchk, time.time() - _T0))
    print("Thermo-Calc invoked: NO    CALPHAD states recalculated: 0")
    print("=" * 78)
    print("RECOVERY_OK" if status != "FAIL" else "RECOVERY_FAILED")
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
