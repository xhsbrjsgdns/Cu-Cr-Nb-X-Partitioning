# -*- coding: utf-8 -*-
"""
26_fullv1_finalize.py - STEP 4 Stages N + P: derived descriptors, failure map,
ML_TARGET_ALL / ML_READY_CONVERGED, file hashes and the data manifest.

Reads only what already exists on disk. Recomputes no Thermo-Calc and touches no raw data.

Produces:
  03_processed/FULL_V1_ML_TARGET_ALL.parquet          all 550,000 rows incl. FAILED (NaN targets)
  03_processed/FULL_V1_ML_READY_CONVERGED.parquet     converged rows only
  03_processed/FULL_V1_DERIVED_DESCRIPTORS.parquet    per-composition thermodynamic descriptors
  04_qa/FULL_V1_FAILURE_MAP.parquet                   every FAILED state + composition + solver metadata
  04_qa/FULL_V1_PHASE_SPLIT_DIAGNOSTICS.parquet       FCC/BCC instance-count statistics
  04_qa/FULL_V1_DOMAIN_SUMMARY.csv                    Domain A/B/ANCHOR contrast
  06_manifest/FULL_V1_FILE_HASHES.csv                 SHA256 of every deliverable
  06_manifest/FULL_V1_DATA_MANIFEST.json              row counts, sizes, hashes, schema pointer

Naming note (spec section 47): the descriptors below are EQUILIBRIUM quantities.
Delta_XCr_FCC is an "equilibrium Cr depletion proxy" - NOT a kinetic precipitation
driving force, and nothing here is a mechanical or electrical property.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1")
sys.path.insert(0, str(PROJECT_ROOT / "code_snapshot"))
import io_store  # noqa: E402

RAW, PROC = PROJECT_ROOT / "02_raw", PROJECT_ROOT / "03_processed"
QA, MAN = PROJECT_ROOT / "04_qa", PROJECT_ROOT / "06_manifest"
COMPS = PROJECT_ROOT / "01_compositions" / "FULL_V1_COMPOSITIONS.csv"


def main():
    for d in (PROC, QA, MAN):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ml = io_store.read_table(PROC / "FULL_V1_ML_TARGET_PREVIEW")
    comps = pd.read_csv(COMPS, dtype=str).fillna("")
    for c in ("X_element", "sampling_domain"):
        if c in ml.columns:
            ml[c] = ml[c].fillna("").astype(str).str.strip()
    print("ML rows: %d  (converged %d)" % (len(ml), int(ml.state_converged.sum())))

    # ---------------- ML_TARGET_ALL / ML_READY (spec 37) --------------------------- #
    ml.to_parquet(PROC / "FULL_V1_ML_TARGET_ALL.parquet", index=False)
    ready = ml[ml.state_converged == 1].reset_index(drop=True)
    ready.to_parquet(PROC / "FULL_V1_ML_READY_CONVERGED.parquet", index=False)
    print("ML_TARGET_ALL %d rows | ML_READY_CONVERGED %d rows" % (len(ml), len(ready)))

    # ---------------- failure map (spec 52) ---------------------------------------- #
    st = pd.read_csv(RAW / "FULL_V1_EQUILIBRIUM_STATES.csv", keep_default_na=False)
    st["ok"] = st["converged"].astype(str).str.lower() == "true"
    f = st[~st.ok].copy()
    meta = comps.set_index("composition_id")
    for col in ("sampling_domain", "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
                "S_CrNb_wt_pct", "R_CrNb_atomic_actual"):
        f[col] = f["case_id"].map(meta[col]).fillna("")
    keep = ["state_id", "case_id", "family", "X_element", "sampling_domain", "Temperature_C",
            "Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct", "S_CrNb_wt_pct",
            "R_CrNb_atomic_actual", "solver_level_used", "bridge_used",
            "bridge_temperatures_C", "attempt_count", "final_status",
            "original_error_message", "error_message"]
    f = f[[c for c in keep if c in f.columns]].rename(columns={"case_id": "composition_id"})
    f.to_parquet(QA / "FULL_V1_FAILURE_MAP.parquet", index=False)
    print("failure map: %d rows" % len(f))

    # ---------------- phase-split diagnostics (spec 45) ---------------------------- #
    sp = (ml[ml.state_converged == 1]
          .groupby(["family", "sampling_domain", "Temperature_C"])
          .agg(n_states=("state_id", "size"),
               multi_FCC=("has_multiple_FCC_instances", "sum"),
               multi_BCC=("has_multiple_BCC_instances", "sum")).reset_index())
    sp["p_multi_FCC"] = sp.multi_FCC / sp.n_states
    sp["p_multi_BCC"] = sp.multi_BCC / sp.n_states
    sp.to_parquet(QA / "FULL_V1_PHASE_SPLIT_DIAGNOSTICS.parquet", index=False)

    # ---------------- domain summary (spec 36) ------------------------------------- #
    fams = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
    dom = (ml[ml.state_converged == 1].groupby("sampling_domain")
           .agg(**{"n_states": ("state_id", "size"), "mean_phase_count": ("phase_count", "mean")},
                **{f"mean_{k}_volfrac": (f"{k}_volfrac", "mean") for k in fams},
                **{f"presence_{k}": (f"{k}_presence", "mean") for k in fams}))
    dom.round(8).to_csv(QA / "FULL_V1_DOMAIN_SUMMARY.csv", encoding="utf-8")

    # ---------------- derived thermodynamic descriptors (spec 47) ------------------ #
    # EQUILIBRIUM quantities only. Delta_XCr_FCC is an equilibrium Cr depletion proxy,
    # NOT a kinetic driving force. No mechanical/electrical property is derived here.
    pc = io_store.read_table(RAW / "FULL_V1_PHASE_COMPOSITIONS_LONG")
    pc["element"] = pc["element"].astype(str).str.upper()
    pc["base"] = pc["raw_phase_name"].astype(str).str.replace(r"#\d+$", "", regex=True)
    cr_fcc = (pc[(pc.base == "FCC_L12") & (pc.element == "CR")]
              .groupby("state_id")["element_mole_fraction"].max())
    idx = ml[["state_id", "composition_id", "Temperature_C"]].copy()
    idx["X_Cr_FCC"] = idx.state_id.map(cr_fcc)
    w = idx.pivot_table(index="composition_id", columns="Temperature_C", values="X_Cr_FCC")
    c15 = ml.pivot_table(index="composition_id", columns="Temperature_C", values="C15_volfrac")
    liq = ml.pivot_table(index="composition_id", columns="Temperature_C", values="LIQUID_presence")
    mb = ml.pivot_table(index="composition_id", columns="Temperature_C",
                        values="has_multiple_BCC_instances")
    mf = ml.pivot_table(index="composition_id", columns="Temperature_C",
                        values="has_multiple_FCC_instances")
    T = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]

    def col(df, t):
        return df[t] if t in df.columns else pd.Series(np.nan, index=df.index)

    def onset(df, ascending=True):
        """lowest (or highest) grid T at which the flag is 1; NaN if never."""
        out = pd.Series(np.nan, index=df.index)
        order = T if ascending else T[::-1]
        for t in order:
            if t in df.columns:
                m = (df[t] == 1) & out.isna()
                out[m] = t
        return out

    d = pd.DataFrame(index=w.index)
    d["X_Cr_FCC_450"] = col(w, 450)
    d["X_Cr_FCC_950"] = col(w, 950)
    d["Delta_XCr_FCC_950_450"] = col(w, 950) - col(w, 450)     # equilibrium Cr depletion proxy
    d["C15_volfrac_450"] = col(c15, 450)
    d["C15_volfrac_600"] = col(c15, 600)
    d["Delta_C15_600_450"] = col(c15, 600) - col(c15, 450)
    d["C15_volfrac_max"] = c15.max(axis=1)
    d["LIQUID_onset_C"] = onset(liq, ascending=True)
    d["BCC_split_closure_C"] = onset(mb, ascending=False)      # highest T still split
    d["FCC_split_closure_C"] = onset(mf, ascending=False)
    d = d.reset_index()
    for c in ("family", "X_element", "sampling_domain"):
        d[c] = d.composition_id.map(meta[c]).fillna("")
    d.to_parquet(PROC / "FULL_V1_DERIVED_DESCRIPTORS.parquet", index=False)
    print("derived descriptors: %d compositions x %d columns" % d.shape)

    # ---------------- hashes + manifest (spec 36, 53) ------------------------------ #
    targets = []
    for sub in ("01_compositions", "02_raw", "03_processed", "04_qa", "06_manifest",
                "code_snapshot", "00_config"):
        for p in sorted((PROJECT_ROOT / sub).glob("*")):
            if (p.is_file() and p.suffix.lower() not in (".shm", ".wal")
                    and p.name != "FULL_V1_FILE_HASHES.csv"):   # self-exclusion: keeps re-runs deterministic
                targets.append(p)
    rows = []
    for p in targets:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        rows.append({"path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                     "bytes": p.stat().st_size, "sha256": h.hexdigest(),
                     "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime(p.stat().st_mtime))})
    hs = pd.DataFrame(rows)
    hs.to_csv(MAN / "FULL_V1_FILE_HASHES.csv", index=False, encoding="utf-8")
    total = int(hs.bytes.sum())

    manifest = {
        "database_version": "FULL_V1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compositions": int(len(comps)), "target_states": int(len(ml)),
        "converged": int(ml.state_converged.sum()),
        "failed": int(len(ml) - ml.state_converged.sum()),
        "row_counts": {
            "FULL_V1_COMPOSITIONS": int(len(comps)),
            "FULL_V1_EQUILIBRIUM_STATES": int(len(st)),
            "FULL_V1_ML_TARGET_ALL": int(len(ml)),
            "FULL_V1_ML_READY_CONVERGED": int(len(ready)),
            "FULL_V1_FAILURE_MAP": int(len(f)),
            "FULL_V1_DERIVED_DESCRIPTORS": int(len(d)),
        },
        "total_bytes": total, "total_GB": round(total / 1024 ** 3, 3),
        "n_files": len(rows),
        "hashes_file": "06_manifest/FULL_V1_FILE_HASHES.csv",
        "schema_file": "06_manifest/FULL_V1_SCHEMA.json",
        "freeze_policy": "06_manifest/FULL_V1_FREEZE_POLICY.md",
        "canonical_store": "02_raw/calculation_checkpoint_shard0{1,2}.sqlite "
                           "(result rows + checkpoint status commit in one transaction)",
        "raw_retained": True, "failures_retained": True,
    }
    (MAN / "FULL_V1_DATA_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("hashed %d files, %.2f GB, in %.1f s" % (len(rows), total / 1024 ** 3, time.time() - t0))
    print("FINALIZE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
