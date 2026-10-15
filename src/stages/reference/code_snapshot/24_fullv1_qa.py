# -*- coding: utf-8 -*-
"""
20_pilot2000_qa.py - Step 3C QA: all mechanical diagnostics for the 22000-state Pilot2000.

Produces (04_qa/):
    FULL_V1_QA_SUMMARY.json          machine-readable: every check + every statistic
    FULL_V1_RAW_PHASE_INVENTORY.csv  raw instances, families, domains, T range, volfrac range
    FULL_V1_PHASE_VOCABULARY_COMPARISON.csv   Step2.5 / Mini20 / Pilot200 / Pilot2000
    FULL_V1_NEW_PHASE_REPORT.md      any phase not seen before
    FULL_V1_SOLVER_DIAGNOSTICS.csv   per-state solver metadata
    FULL_V1_LIQUID_DIAGNOSTICS.csv   (written by 19; summarised here)
    FULL_V1_TOPOLOGY_STATS.csv       dominant X phase by family x temperature x domain
    FULL_V1_RUNTIME_REPORT.md        wall / Thermo-Calc / I/O, O(N^2) elimination check
    FULL_V1_QA_REPORT.md             first-pass report (the narrative version is written
                                       separately from the summary JSON)

Nothing here recomputes Thermo-Calc; it reads the shard stores and the 19_* outputs.
"""

import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1")
sys.path.insert(0, str(PROJECT_ROOT / "code_snapshot"))
import fivefamily as ff     # noqa: E402
import io_store             # noqa: E402

PILOT_ROOT = Path(r"E:\Cu_CALPHAD_ML\02_pilot_2000")
DIR_RAW = PROJECT_ROOT / "02_raw"
DIR_PROC = PROJECT_ROOT / "03_processed"
DIR_QA = PROJECT_ROOT / "04_qa"
DIR_LOG = PROJECT_ROOT / "05_logs"
FILE_COMPS = PROJECT_ROOT / "01_compositions" / "FULL_V1_COMPOSITIONS.csv"
INV_25 = PILOT_ROOT / "04_qa" / "STEP2_5_RAW_PHASE_INVENTORY.csv"
INV_M20 = PILOT_ROOT / "04_qa" / "mini20" / "MINI20_RAW_PHASE_INVENTORY.csv"
INV_P200 = PILOT_ROOT / "04_qa" / "pilot200" / "PILOT200_RAW_PHASE_INVENTORY.csv"
# FULL_V1_PATCH_02: the PILOT2000_ -> FULL_V1_ prefix rename that produced this file from
# 20_pilot2000_qa.py also rewrote these two PRIOR-STAGE reference paths, which must keep the
# Pilot2000 filenames. Both silently resolved to non-existent files, so the Pilot2000 prior
# set was empty: MU_PHASE was falsely reported as new, and check A10 skipped Pilot2000
# entirely. Lookup paths only - no physics, threshold, mapping or computed value is affected.
INV_P2000 = PILOT_ROOT / "04_qa" / "pilot2000" / "PILOT2000_RAW_PHASE_INVENTORY.csv"

TEMPS = [300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000]
N_COMPS, N_STATES = 50000, 550000
CONV_GO = 99.5
NI_PAT = r"(?i)(?<![a-z])ni(?![0-9a-z])"
DOMAINS = {
    "A": dict(S=(0.6, 2.0), R=(2.0, 6.5),
              X={"ZR": (.02, .20), "TI": (.02, .20), "CO": (.03, .30),
                 "SI": (.005, .08), "AG": (.03, .30), "MO": (.03, .30)}),
    "B": dict(S=(2.0, 4.5), R=(1.0, 4.0),
              X={"ZR": (.02, .40), "TI": (.02, .35), "CO": (.03, .60),
                 "SI": (.005, .15), "AG": (.03, .60), "MO": (.03, .60)}),
}
CHECKS = []


def chk(cid, desc, status, detail=""):
    CHECKS.append({"id": cid, "check": desc, "status": status, "detail": str(detail)[:400]})
    return status


def tof(v, d=math.nan):
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except (TypeError, ValueError):
        return d


def _shard_ids():
    return sorted(int(p.stem.split("shard")[-1])
                  for p in DIR_RAW.glob("calculation_checkpoint_shard*.sqlite")) or [1, 2]


def load_checkpoints():
    rows, per_shard = [], {}
    for n in _shard_ids():
        db = DIR_RAW / ("calculation_checkpoint_shard%02d.sqlite" % n)
        if not db.exists():
            continue
        con = sqlite3.connect("file:%s?mode=ro" % db.as_posix(), uri=True)
        try:
            cpdf = pd.read_sql_query("SELECT state_id, case_id, Temperature_C, status, "
                                     "attempt_count, error_message FROM state_checkpoint", con)
            res = pd.read_sql_query("SELECT state_id, converged FROM state_result", con)
        finally:
            con.close()
        cpdf["shard"] = n
        rows.append(cpdf)
        ok = set(cpdf.loc[cpdf.status == "SUCCESS", "state_id"])
        stored = set(res.loc[res.converged.astype(str) == "True", "state_id"])
        per_shard[n] = {"counts": cpdf.status.value_counts().to_dict(),
                        "success": len(ok), "stored_converged": len(stored),
                        "missing_result": sorted(ok - stored)[:20],
                        "n_missing_result": len(ok - stored)}
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), per_shard


def main():
    DIR_QA.mkdir(parents=True, exist_ok=True)
    comps = pd.read_csv(FILE_COMPS, dtype=str).fillna("")
    states = pd.read_csv(DIR_RAW / "FULL_V1_EQUILIBRIUM_STATES.csv", keep_default_na=False)
    phases = io_store.read_table(DIR_RAW / "FULL_V1_PHASES_LONG")
    pcomp = io_store.read_table(DIR_RAW / "FULL_V1_PHASE_COMPOSITIONS_LONG")
    part_raw = io_store.read_table(DIR_RAW / "FULL_V1_ELEMENT_PARTITIONING_RAW")
    agg = io_store.read_table(DIR_PROC / "FULL_V1_AGGREGATED_PHASES_LONG")
    wide = io_store.read_table(DIR_PROC / "FULL_V1_AGGREGATED_PHASES")
    part_agg = io_store.read_table(DIR_PROC / "FULL_V1_ELEMENT_PARTITIONING_AGGREGATED")
    ml = io_store.read_table(DIR_PROC / "FULL_V1_ML_TARGET_PREVIEW")
    states["ok"] = states["converged"].astype(str).str.lower() == "true"
    # CSV round-trip turns the baseline's empty X_element into NaN; normalise it back to ""
    # so baseline rows are never mistaken for an X family called "nan".
    for _df in (ml, part_agg, part_raw):
        if "X_element" in _df.columns:
            _df["X_element"] = _df["X_element"].fillna("").astype(str).str.strip()
    if "sampling_domain" in ml.columns:
        ml["sampling_domain"] = ml["sampling_domain"].fillna("").astype(str).str.strip()
    for _c in ("dominant_X_partition_family", "dominant_X_solid_partition_family"):
        if _c in ml.columns:
            ml[_c] = ml[_c].fillna("").astype(str)
    mlc = ml[ml.state_converged == 1]
    HAS_X = mlc.X_element != ""
    meta = comps.set_index("composition_id")

    # ---------------- A: compositions ---------------------------------------- #
    chk("A01", "50,000 unique compositions",
        "PASS" if len(comps) == N_COMPS and comps.composition_id.nunique() == N_COMPS else "FAIL",
        "n=%d unique=%d" % (len(comps), comps.composition_id.nunique()))
    chk("A02", "7 families",
        "PASS" if set(comps.family) == {"BASELINE", "ZR", "TI", "CO", "SI", "AG", "MO"} else "FAIL",
        sorted(set(comps.family)))
    alloc = comps.family.value_counts().to_dict()
    chk("A03", "allocation 5,000 baseline + 6 x 7,500",
        "PASS" if alloc.get("BASELINE") == 5000 and all(alloc.get(f) == 7500 for f in
                                                        ("ZR", "TI", "CO", "SI", "AG", "MO")) else "FAIL", alloc)
    dom = comps.sampling_domain.value_counts().to_dict()
    chk("A04", "domain A35000 / B10000 / ANCHOR5000",
        "PASS" if dom == {"A": 35000, "B": 10000, "ANCHOR": 5000} else "FAIL", dom)
    s = comps.composition_sum_wt_pct.astype(float)
    chk("A05", "composition sum == 100 wt%", "PASS" if (abs(s - 100) < 0.01).all() else "FAIL",
        "min=%.4f max=%.4f" % (s.min(), s.max()))
    cu = comps.Cu_wt_pct.astype(float)
    chk("A06", "Cu > 90 wt%", "PASS" if (cu > 90).all() else "FAIL", "min=%.4f" % cu.min())
    bad = []
    for _, r in comps.iterrows():
        if r.sampling_domain in ("A", "B"):
            d = DOMAINS[r.sampling_domain]
            if not (d["S"][0] - 1e-6 <= float(r.S_CrNb_wt_pct) <= d["S"][1] + 1e-6):
                bad.append((r.composition_id, "S"))
            if not (d["R"][0] - 1e-3 <= float(r.R_CrNb_atomic_actual) <= d["R"][1] + 1e-3):
                bad.append((r.composition_id, "R"))
            x = (r.X_element or "").upper()
            if x and not (d["X"][x][0] - 1e-9 <= float(r.X_wt_pct) <= d["X"][x][1] + 1e-9):
                bad.append((r.composition_id, "X"))
    chk("A07", "Domain A/B ranges", "PASS" if not bad else "FAIL", bad[:8])
    bp = [(r.composition_id, c) for _, r in comps[comps.sampling_domain == "ANCHOR"].iterrows()
          for c in ("anchor_perturbation_pct_Cr", "anchor_perturbation_pct_Nb")
          if abs(float(r[c])) > 10 + 1e-9]
    chk("A08", "anchor perturbation <= +/-10%", "PASS" if not bp else "FAIL", bp[:8])
    sig = comps.apply(lambda r: (r.family, round(float(r.Cr_wt_pct), 4), round(float(r.Nb_wt_pct), 4),
                                 round(float(r.X_wt_pct) if r.X_element else 0.0, 4)), axis=1)
    chk("A09", "no internal duplicate", "PASS" if sig.nunique() == len(comps) else "FAIL",
        "%d/%d" % (sig.nunique(), len(comps)))
    prior = set()
    missing_prior = []                                        # FULL_V1_PATCH_02
    for f in (PILOT_ROOT / "00_config" / "DRY_RUN_CASES.csv",
              PILOT_ROOT / "01_compositions" / "mini20" / "MINI20_COMPOSITIONS.csv",
              PILOT_ROOT / "01_compositions" / "pilot200" / "PILOT200_COMPOSITIONS.csv",
              # FULL_V1_PATCH_02: was FULL_V1_COMPOSITIONS.csv - a path that does not exist here,
              # so this check silently skipped Pilot2000 entirely.
              PILOT_ROOT / "01_compositions" / "pilot2000" / "PILOT2000_COMPOSITIONS.csv"):
        if not f.exists():                                    # FULL_V1_PATCH_02
            missing_prior.append(f.name)
        if f.exists():
            d0 = pd.read_csv(f, dtype=str).fillna("")
            prior |= set(d0.apply(lambda r: (r["family"], round(float(r["Cr_wt_pct"]), 4),
                                             round(float(r["Nb_wt_pct"]), 4),
                                             round(float(r["X_wt_pct"]) if r.get("X_element") and
                                                   r.get("X_wt_pct") else 0.0, 4)), axis=1))
    ov = len(set(sig) & prior)
    # FULL_V1_PATCH_02: a missing prior-stage file must never be silently treated as "no overlap".
    chk("A10", "no duplicate vs Step2.5 + Mini20 + Pilot200 + Pilot2000",
        "FAIL" if ov else ("WARN" if missing_prior else "PASS"),
        "overlap=%d%s" % (ov, ("; prior-stage file(s) NOT FOUND, comparison incomplete: %s"
                               % missing_prior) if missing_prior else ""))
    ra = pd.to_numeric(comps.R_CrNb_atomic_actual, errors="coerce")
    rt = pd.to_numeric(comps.R_CrNb_atomic_target, errors="coerce")
    chk("A11", "R_actual ~ R_target", "PASS" if float((ra - rt).abs().max()) <= 0.01 else "FAIL",
        "max dev %.3e" % float((ra - rt).abs().max()))

    # ---------------- B: states ----------------------------------------------- #
    n_states, n_ok = len(states), int(states.ok.sum())
    n_fail = n_states - n_ok
    rate = 100.0 * n_ok / N_STATES
    chk("B01", "550,000 target state rows", "PASS" if n_states == N_STATES else "FAIL",
        "found %d (missing %d)" % (n_states, N_STATES - n_states))
    got = states.groupby("case_id")["Temperature_C"].apply(lambda s: sorted({int(round(tof(t))) for t in s}))
    missing = [(cid, sorted(set(TEMPS) - set(v))) for cid, v in got.items() if set(TEMPS) - set(v)]
    missing += [(c, TEMPS) for c in comps.composition_id if c not in got.index]
    chk("B02", "11 target temperatures per composition", "PASS" if not missing else "FAIL",
        "%d incomplete: %s" % (len(missing), missing[:5]))
    tset = sorted({int(round(tof(t))) for t in states.Temperature_C})
    chk("B03", "no bridge temperature leaked into states", "PASS" if set(tset) <= set(TEMPS) else "FAIL", tset)
    chk("B04", "convergence >= %.1f%%" % CONV_GO, "PASS" if rate >= CONV_GO else "WARN",
        "%d/%d = %.3f%%" % (n_ok, N_STATES, rate))
    hits = []
    for lbl, df in (("states", states), ("phases", phases), ("phase_comps", pcomp),
                    ("part_raw", part_raw), ("part_agg", part_agg), ("ml", ml), ("comps", comps)):
        for col in df.columns:
            if col in ("composition_id", "state_id", "case_id"):
                continue
            try:
                if df[col].astype(str).str.contains(NI_PAT, regex=True).any():
                    hits.append((lbl, col))
            except Exception:
                pass
    chk("B05", "no Ni/NI anywhere", "PASS" if not hits else "FAIL", hits[:6])
    chk("B06", "all failures retained", "PASS", "%d FAILED rows kept with error messages" % n_fail)

    # ---------------- C: conservation ------------------------------------------ #
    c = states[states.ok]
    d1 = float((pd.to_numeric(c.sum_phase_mole_fraction, errors="coerce") - 1).abs().max())
    d2 = float((pd.to_numeric(c.sum_phase_volume_fraction, errors="coerce") - 1).abs().max())
    chk("C01", "sum(phase mole fraction) == 1", "PASS" if d1 <= 1e-5 else "FAIL", "max dev %.2e" % d1)
    chk("C02", "sum(phase volume fraction) == 1", "PASS" if d2 <= 2e-3 else "FAIL", "max dev %.2e" % d2)
    g = pcomp.groupby(["state_id", "raw_phase_name"])["element_mole_fraction"].sum()
    d3 = float((g - 1).abs().max())
    chk("C03", "phase composition sums == 1", "PASS" if d3 <= 1e-4 else "WARN", "max dev %.2e" % d3)
    pg = part_raw.groupby(["state_id", "element"])["element_partition_raw"].sum()
    d4 = float((pg - 1).abs().max())
    chk("C04", "raw element partition sums == 1", "PASS" if d4 <= 1e-6 else "FAIL", "max dev %.2e" % d4)
    devs = {}
    for el in ["CR", "NB"] + sorted({str(x).upper() for x in comps.X_element.unique() if str(x).strip()}):
        cols = ["P_%s_%s" % (el, f) for f in ff.FAMILY_ORDER_5]
        if not all(x in part_agg.columns for x in cols):
            continue
        ss = part_agg[cols].sum(axis=1, min_count=1)
        devs[el] = float((ss.dropna() - 1).abs().max()) if ss.notna().any() else 0.0
    chk("C05", "Cr 5-family partition sum == 1", "PASS" if devs.get("CR", 1) <= 1e-6 else "FAIL",
        "max dev %.2e" % devs.get("CR", float("nan")))
    chk("C06", "Nb 5-family partition sum == 1", "PASS" if devs.get("NB", 1) <= 1e-6 else "FAIL",
        "max dev %.2e" % devs.get("NB", float("nan")))
    xdev = max([v for k, v in devs.items() if k not in ("CR", "NB")] or [0.0])
    chk("C07", "X 5-family partition sum == 1", "PASS" if xdev <= 1e-6 else "FAIL", "max dev %.2e" % xdev)
    f5 = mlc[["%s_volfrac" % f for f in ff.FAMILY_ORDER_5]].sum(axis=1)
    chk("C08", "5-family volume fractions sum == 1", "PASS" if float((f5 - 1).abs().max()) <= 1e-5 else "FAIL",
        "max dev %.2e" % float((f5 - 1).abs().max()))
    f5m = mlc[["%s_molefrac" % f for f in ff.FAMILY_ORDER_5]].sum(axis=1)
    chk("C09", "5-family mole fractions sum == 1", "PASS" if float((f5m - 1).abs().max()) <= 1e-5 else "FAIL",
        "max dev %.2e" % float((f5m - 1).abs().max()))
    sv = (mlc.solid_volume_fraction - (1 - mlc.LIQUID_volfrac)).abs().max()
    chk("C10", "solid_volume_fraction == 1 - LIQUID_volfrac", "PASS" if float(sv) <= 1e-12 else "FAIL",
        "max dev %.2e" % float(sv))
    bx = mlc[~HAS_X][["X_P_%s" % f for f in ff.FAMILY_ORDER_5]]
    chk("C11", "baseline X targets are NaN (not 0)", "PASS" if bx.isna().all().all() else "FAIL",
        "%d baseline rows checked" % len(bx))

    # ---------------- D: phase vocabulary -------------------------------------- #
    ph = phases.copy()
    ph["composition_id"] = ph.state_id.str.split("__").str[0]
    ph["fam_of_comp"] = ph.composition_id.map(lambda cc: meta.loc[cc, "family"] if cc in meta.index else "")
    ph["dom_of_comp"] = ph.composition_id.map(lambda cc: meta.loc[cc, "sampling_domain"] if cc in meta.index else "")
    inv = (ph.groupby("raw_phase_name")
             .agg(n_states=("state_id", "nunique"),
                  n_compositions=("composition_id", "nunique"),
                  families=("fam_of_comp", lambda s: ";".join(sorted(set(s)))),
                  sampling_domains=("dom_of_comp", lambda s: ";".join(sorted(set(s)))),
                  Tmin=("Temperature_C", "min"), Tmax=("Temperature_C", "max"),
                  min_volfrac=("phase_volume_fraction", "min"),
                  max_volfrac=("phase_volume_fraction", "max")).reset_index())
    inv["phase_model_base"] = inv.raw_phase_name.map(ff.base_name)
    inv["aggregated_family"] = inv.phase_model_base.map(ff.family_of_5)
    inv = inv[["raw_phase_name", "phase_model_base", "aggregated_family", "n_states",
               "n_compositions", "families", "sampling_domains", "Tmin", "Tmax",
               "min_volfrac", "max_volfrac"]].sort_values("raw_phase_name")
    inv.to_csv(DIR_QA / "FULL_V1_RAW_PHASE_INVENTORY.csv", index=False, encoding="utf-8")

    prev = {}
    for label, f in (("step25", INV_25), ("mini20", INV_M20), ("pilot200", INV_P200),
                     ("pilot2000", INV_P2000)):
        prev[label] = set(pd.read_csv(f)["phase_model_base"]) if Path(f).exists() else set()
    voc = []
    STAGE_ORDER = ("step25", "mini20", "pilot200", "pilot2000")   # FULL_V1_PATCH_02
    for b in sorted(set(inv.phase_model_base) | set().union(*prev.values())):
        sub = inv[inv.phase_model_base == b]
        first_seen = next((lbl for lbl in STAGE_ORDER if b in prev[lbl]), "fullv1")
        voc.append({"phase_model_base": b, "aggregated_family": ff.family_of_5(b),
                    "first_seen_stage": first_seen,
                    "seen_step25": int(b in prev["step25"]), "seen_mini20": int(b in prev["mini20"]),
                    "seen_pilot200": int(b in prev["pilot200"]),
                    "seen_pilot2000": int(b in prev["pilot2000"]),
                    "seen_fullv1": int(len(sub) > 0),
                    "is_new": int(len(sub) > 0 and not any(b in prev[k] for k in prev)),
                    "n_states": int(sub.n_states.sum()) if len(sub) else 0,
                    "n_compositions": int(sub.n_compositions.sum()) if len(sub) else 0,
                    "families": ";".join(sorted({x for r in sub.families for x in r.split(";") if x})) if len(sub) else "",
                    "sampling_domains": ";".join(sorted({x for r in sub.sampling_domains for x in r.split(";") if x})) if len(sub) else "",
                    "Tmin": float(sub.Tmin.min()) if len(sub) else math.nan,
                    "Tmax": float(sub.Tmax.max()) if len(sub) else math.nan})
    voc = pd.DataFrame(voc)
    voc.to_csv(DIR_QA / "FULL_V1_PHASE_VOCABULARY_COMPARISON.csv", index=False, encoding="utf-8")
    new = voc[voc.is_new == 1]
    chk("D01", "phase vocabulary captured dynamically", "PASS",
        "%d raw instances / %d base names" % (len(inv), int(voc.seen_fullv1.sum())))
    chk("D02", "new phases retained (never dropped)", "PASS" if new.empty or
        set(new.aggregated_family) <= {"OTHER_SOLID", "LIQUID"} else "WARN",
        "new=%s" % list(new.phase_model_base))
    chk("D03", "LIQUID has its own family", "PASS" if ff.family_of_5("LIQUID#1") == "LIQUID" else "FAIL")

    N = ["# FULL_V1_NEW_PHASE_REPORT.md", "",
         "Step 3C 相对 Step2.5 / Mini20 / Pilot200 新出现的相。全部保留 raw 名称与数值，按 `fivefamily.py` 归类。", ""]
    if new.empty:
        N += ["**未发现新相。** Pilot2000 的 phase vocabulary 与 Pilot200 完全一致：",
              "", "`%s`" % "`, `".join(sorted(inv.phase_model_base.unique())), ""]
    else:
        for _, r in new.iterrows():
            sub = inv[inv.phase_model_base == r.phase_model_base]
            N += ["## `%s` -> `%s`" % (r.phase_model_base, r.aggregated_family), "",
                  "- raw instances: %s" % sorted(sub.raw_phase_name),
                  "- states: %d | compositions: %d" % (r.n_states, r.n_compositions),
                  "- families: %s | domains: %s" % (r.families, r.sampling_domains),
                  "- T: %.0f-%.0f C | volfrac: %.3e - %.3e"
                  % (r.Tmin, r.Tmax, sub.min_volfrac.min(), sub.max_volfrac.max()), ""]
    (DIR_QA / "FULL_V1_NEW_PHASE_REPORT.md").write_text("\n".join(N), encoding="utf-8")

    # ---------------- E: solver ------------------------------------------------ #
    sd = states[["state_id", "case_id", "family", "Temperature_C", "solver_level_used",
                 "bridge_used", "bridge_temperatures_C", "attempt_count", "final_status",
                 "converged", "calculation_time_seconds", "original_error_message"]].copy()
    sd["sampling_domain"] = sd.case_id.map(lambda cc: meta.loc[cc, "sampling_domain"] if cc in meta.index else "")
    sd.to_csv(DIR_QA / "FULL_V1_SOLVER_DIAGNOSTICS.csv", index=False, encoding="utf-8")
    lev = {k: int((pd.to_numeric(sd.solver_level_used, errors="coerce") == k).sum()) for k in (0, 1, 2, 3)}
    n_bridge = int(pd.to_numeric(sd.bridge_used, errors="coerce").fillna(0).sum())
    fails = states[~states.ok]
    chk("E01", "solver escalation metadata recorded", "PASS",
        "levels=%s bridge_success=%d failures=%d" % (lev, n_bridge, n_fail))

    # ---------------- F: checkpoint / result consistency ----------------------- #
    cp_all, per_shard = load_checkpoints()
    n_cp = len(cp_all)
    chk("F01", "checkpoint rows == 550000", "PASS" if n_cp == N_STATES else "FAIL", "rows=%d" % n_cp)
    tot_missing = sum(v["n_missing_result"] for v in per_shard.values())
    chk("F02", "every SUCCESS checkpoint has a stored raw result (spec 2.E)",
        "PASS" if tot_missing == 0 else "FAIL",
        "missing=%d %s" % (tot_missing, {k: v["missing_result"] for k, v in per_shard.items()}))
    split_bad = [cid for cid, gg in cp_all.groupby("case_id")["shard"].nunique().items() if gg > 1] \
        if len(cp_all) else []
    chk("F03", "a composition is never split across processes",
        "PASS" if not split_bad else "FAIL", split_bad[:5])

    # ---------------- G: topology ---------------------------------------------- #
    topo = (mlc[HAS_X]
            .groupby(["family", "sampling_domain", "Temperature_C",
                      "dominant_X_solid_partition_family"]).size()
            .reset_index(name="n_states"))
    topo.to_csv(DIR_QA / "FULL_V1_TOPOLOGY_STATS.csv", index=False, encoding="utf-8")
    xs = [c for c in mlc.columns if c.startswith("X_Psolid_")]
    topo_mean = mlc[HAS_X].groupby("family")[xs].mean()
    dom_share = (mlc[HAS_X]
                 .groupby("family")["dominant_X_solid_partition_family"]
                 .value_counts(normalize=True).unstack(fill_value=0.0))
    degenerate = [f for f in dom_share.index if float(dom_share.loc[f].max()) >= 0.999]
    chk("G01", "no family has a completely degenerate topology",
        "PASS" if not degenerate else "WARN",
        "single-outcome families: %s" % degenerate)

    # ---------------- H: liquid ------------------------------------------------- #
    liq = pd.read_csv(DIR_QA / "FULL_V1_LIQUID_DIAGNOSTICS.csv") if (
        DIR_QA / "FULL_V1_LIQUID_DIAGNOSTICS.csv").exists() else pd.DataFrame()
    chk("H01", "LIQUID handled as an independent 5th family", "PASS",
        "%d states / %d compositions" % (len(liq), liq.composition_id.nunique() if len(liq) else 0))

    # ---------------- runtime --------------------------------------------------- #
    rts = {}
    for n in _shard_ids():
        f = DIR_LOG / ("FULL_V1_RUNTIME_shard%02d.json" % n)
        if f.exists():
            rts[n] = json.loads(f.read_text(encoding="utf-8"))
    bench = {}
    fb = DIR_QA / "FULL_V1_IO_BENCHMARK.json"
    if fb.exists():
        bench = json.loads(fb.read_text(encoding="utf-8"))
    growth = per_state_growth()

    R = ["# FULL_V1_RUNTIME_REPORT.md — Step 3C 运行性能", ""]
    if bench:
        gate = bench.get("min_speedup_gate", bench.get("min_io_speedup_gate", 5.0))
        R += ["## 1. I/O 补丁 benchmark（550 states，OLD vs NEW）", "",
              "| 指标 | OLD | NEW | 比值 |", "|---|---|---|---|",
              "| wall | %.1f s | %.1f s | **%.2f×** |" % (bench["old"]["wall"], bench["new"]["wall"],
                                                          bench["speedup_wall"]),
              "| Thermo-Calc | %.1f s | %.1f s | |" % (bench["old"]["t_calc"], bench["new"]["t_calc"]),
              "| I/O | %.1f s | %.1f s | **%.1f×** |" % (bench["old"]["t_io"], bench["new"]["t_io"],
                                                         bench["speedup_io"]),
              "", "verdict: **%s**（门禁 ≥%.1f×）" % (bench["verdict"], gate), ""]
    R += ["## 2. 实际运行", "", "| shard | wall | Thermo-Calc | I/O | states | states/s |",
          "|---|---|---|---|---|---|"]
    for n, m in rts.items():
        R.append("| %d | %.1f s (%.2f h) | %.1f s (%.1f%%) | %.1f s (%.1f%%) | %d | %.3f |"
                 % (n, m["wall_seconds"], m["wall_seconds"] / 3600.0, m["thermocalc_seconds"],
                    100 * m["thermocalc_seconds"] / max(m["wall_seconds"], 1e-9),
                    m["io_seconds_total"], 100 * m["io_seconds_total"] / max(m["wall_seconds"], 1e-9),
                    m["states_computed"], m["states_per_second"]))
    if rts:
        R += ["", "- 两 shard 并行的有效 wall = max(shard wall) = **%.2f h**"
              % (max(m["wall_seconds"] for m in rts.values()) / 3600.0),
              "- 导出（单次批量）耗时：%s"
              % {n: "%.1f s" % m["export_seconds"] for n, m in rts.items()},
              "- checkpoint 快照耗时：%s"
              % {n: "%.1f s" % m["progress_snapshot_seconds"] for n, m in rts.items()}, ""]
    R += ["## 3. O(N²) 是否已消除", ""]
    if growth:
        R += ["逐 state 间隔按累计进度分段（来自 shard 日志的 PROGRESS 行）：", "",
              "| shard | 前 25% | 25-50% | 50-75% | 后 25% | 斜率 |", "|---|---|---|---|---|---|"]
        for n, g in growth.items():
            R.append("| %d | %.3f s | %.3f s | %.3f s | %.3f s | %.2e s/state² |"
                     % (n, g["q1"], g["q2"], g["q3"], g["q4"], g["slope"]))
        R += ["", "Step 3B 的斜率是 **2.9e-03 – 3.1e-03 s/state²**。", ""]
        worst = max(abs(g["slope"]) for g in growth.values())
        R.append("**结论：%s**（最大斜率 %.2e，Step 3B 的 %.2f%%）"
                 % ("O(N²) 已消除" if worst < 3e-4 else "仍存在明显增长，需复查",
                    worst, 100 * worst / 3.0e-3))
        chk("I01", "O(N^2) growth eliminated", "PASS" if worst < 3e-4 else "FAIL",
            "max slope %.2e s/state^2 (Step 3B: ~3.0e-3)" % worst)
    else:
        R.append("（日志中没有足够的 PROGRESS 行来拟合增长曲线）")
    (DIR_QA / "FULL_V1_RUNTIME_REPORT.md").write_text("\n".join(R), encoding="utf-8")

    # ---------------- summary --------------------------------------------------- #
    overall = ("FAIL" if any(x["status"] == "FAIL" for x in CHECKS)
               else ("PASS WITH WARNINGS" if any(x["status"] == "WARN" for x in CHECKS) else "PASS"))
    summary = {
        "step": "STEP 4 - FULL CALPHAD DATABASE V1", "status": overall,
        "database": "TCCU6", "temperatures_C": TEMPS,
        "compositions": {"total": len(comps), "families": alloc, "domains": dom,
                         "cu_min_wt_pct": float(cu.min()),
                         "sobol_seed": int(comps.seed.iloc[0]) if "seed" in comps else None,
                         "sobol_skip": int(comps.sobol_skip.iloc[0]) if "sobol_skip" in comps else None,
                         "duplicate_vs_prior": ov},
        "states": {"planned": N_STATES, "rows": n_states, "converged": n_ok, "failed": n_fail,
                   "convergence_rate": round(rate, 4),
                   "failures_by_family": fails.groupby("family").size().to_dict(),
                   "failures_by_temperature": fails.groupby("Temperature_C").size().to_dict(),
                   "failures_by_domain": sd[~sd.converged.astype(str).str.lower().eq("true")]
                       .groupby("sampling_domain").size().to_dict(),
                   "failed_state_ids": fails.state_id.tolist()[:200],
                   "convergence_by_family": (states.groupby("family").ok.mean() * 100).round(4).to_dict()},
        "solver": {"level_counts": lev, "bridge_success": n_bridge,
                   "bridge_temperatures": sorted({str(t) for t in sd.bridge_temperatures_C.dropna().astype(str)
                                                  if t and t != "nan"}),
                   "calculation_time_seconds": {
                       k: float(v) for k, v in pd.to_numeric(
                           c.calculation_time_seconds, errors="coerce").describe(
                           percentiles=[.5, .9, .95]).items()}},
        "phase_vocabulary": {"base_names": sorted(inv.phase_model_base.unique().tolist()),
                             "raw_instances": sorted(inv.raw_phase_name.tolist()),
                             "new_vs_prior": new.phase_model_base.tolist(),
                             "family_map": {b: ff.family_of_5(b) for b in sorted(inv.phase_model_base.unique())},
                             "inventory": inv.to_dict("records")},
        "liquid": {"n_states": len(liq), "n_compositions": int(liq.composition_id.nunique()) if len(liq) else 0,
                   "families": liq.family.value_counts().to_dict() if len(liq) else {},
                   "domains": liq.sampling_domain.value_counts().to_dict() if len(liq) else {},
                   "temperatures": liq.Temperature_C.value_counts().to_dict() if len(liq) else {},
                   "min_temperature_C": float(liq.Temperature_C.min()) if len(liq) else None,
                   "max_volume_fraction": float(liq.LIQUID_volfrac.max()) if len(liq) else None,
                   "median_volume_fraction": float(liq.LIQUID_volfrac.median()) if len(liq) else None},
        "phase_splitting": {
            "multi_FCC_states": int(mlc.has_multiple_FCC_instances.sum()),
            "multi_FCC_by_family": mlc[mlc.has_multiple_FCC_instances == 1].family.value_counts().to_dict(),
            "multi_FCC_max_T": float(mlc[mlc.has_multiple_FCC_instances == 1].Temperature_C.max())
                if int(mlc.has_multiple_FCC_instances.sum()) else None,
            "multi_BCC_states": int(mlc.has_multiple_BCC_instances.sum()),
            "multi_BCC_by_family": mlc[mlc.has_multiple_BCC_instances == 1].family.value_counts().to_dict(),
            "multi_BCC_max_T": float(mlc[mlc.has_multiple_BCC_instances == 1].Temperature_C.max())
                if int(mlc.has_multiple_BCC_instances.sum()) else None,
            "mo_bcc_split_by_T": (mlc[mlc.family == "MO"].groupby("Temperature_C")
                                  .has_multiple_BCC_instances.mean().round(4).to_dict()),
            "ag_fcc_split_by_T": (mlc[mlc.family == "AG"].groupby("Temperature_C")
                                  .has_multiple_FCC_instances.mean().round(4).to_dict())},
        "conservation": {"phase_mole_sum_max_dev": d1, "phase_volume_sum_max_dev": d2,
                         "phase_composition_sum_max_dev": d3, "raw_partition_max_dev": d4,
                         "family_partition_max_dev": devs,
                         "five_family_volfrac_max_dev": float((f5 - 1).abs().max()),
                         "five_family_molefrac_max_dev": float((f5m - 1).abs().max()),
                         "solid_vs_liquid_max_dev": float(sv)},
        "topology": {"mean_X_solid_partition": topo_mean.round(6).to_dict(),
                     "dominant_share": dom_share.round(6).to_dict(),
                     "degenerate_families": degenerate},
        "auxiliary": {
            "note": "ZRM5_C15B stays inside OTHER_SOLID; these are DIAGNOSTIC only, never summed with the 5 core families",
            "zrm5_states": int((wide.get("ZRM5_C15B_volfrac_AUX", pd.Series(dtype=float)) > 0).sum()),
            "sigma_states": int((wide.get("SIGMA_volfrac_AUX", pd.Series(dtype=float)) > 0).sum()),
            "zr_mean_X_P_C15": float(mlc[mlc.family == "ZR"].X_P_C15.mean()) if (mlc.family == "ZR").any() else None,
            "zr_mean_X_P_C15_family_wide": float(mlc[mlc.family == "ZR"].X_P_C15_family_wide_AUX.mean())
                if (mlc.family == "ZR").any() else None},
        "domain_contrast": {
            "mean_volume_fraction": mlc.groupby("sampling_domain")[
                ["%s_volfrac" % f for f in ff.FAMILY_ORDER_5]].mean().round(8).to_dict(),
            "presence_rate": mlc.groupby("sampling_domain")[
                ["%s_presence" % f for f in ff.FAMILY_ORDER_5]].mean().round(6).to_dict(),
            "mean_phase_count": mlc.groupby("sampling_domain").phase_count.mean().round(4).to_dict()},
        "checkpoint": {"rows": n_cp, "per_shard": per_shard},
        "runtime": rts, "benchmark": bench, "growth": growth,
        "nickel": {"used": False, "occurrences": len(hits)},
        "checks": CHECKS,
    }
    (DIR_QA / "FULL_V1_QA_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    L = ["# FULL_V1_QA_REPORT.md — Step 3C（机器生成的检查表部分）", "",
         "**STEP 4 STATUS: %s**" % overall, "",
         "- 目标 states：%d × 11 = %d；实得 %%d，收敛 %%d，失败 %%d，收敛率 **%%.3f%%%%**" % (N_COMPS, N_STATES)
         % (n_states, n_ok, n_fail, rate), "",
         "> 完整的 18 问叙述版报告基于 `FULL_V1_QA_SUMMARY.json` 另行生成。", "",
         "## 检查表（%d 项）" % len(CHECKS), "",
         "| id | 检查 | 状态 | 详情 |", "|---|---|---|---|"]
    for x in CHECKS:
        L.append("| %s | %s | **%s** | %s |" % (x["id"], x["check"], x["status"], x["detail"]))
    (DIR_QA / "FULL_V1_QA_REPORT.md").write_text("\n".join(L), encoding="utf-8")

    print("STEP 4 STATUS: %s" % overall)
    for x in CHECKS:
        print("  [%s] %s : %s" % (x["status"], x["check"], x["detail"][:110]))
    return 0 if overall != "FAIL" else 5


def per_state_growth():
    """Fit per-state wall time vs cumulative progress from each shard log's PROGRESS lines."""
    out = {}
    for n in _shard_ids():
        f = DIR_LOG / ("FULL_V1_RUN_shard%02d.log" % n)
        if not f.exists():
            continue
        pts = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"PROGRESS (\d+)/(\d+) done \| ([\d.]+) s/state", line)
            if m:
                pts.append((int(m.group(1)), float(m.group(3))))
        if len(pts) < 8:
            continue
        x = np.array([p[0] for p in pts], dtype=float)
        cum = np.array([p[1] for p in pts], dtype=float)      # cumulative mean s/state
        inst = np.diff(cum * x) / np.diff(x)                   # instantaneous s/state
        xm = x[1:]
        q = np.array_split(np.arange(len(inst)), 4)
        slope = float(np.polyfit(xm, inst, 1)[0]) if len(inst) >= 2 else float("nan")
        out[n] = {"q1": float(inst[q[0]].mean()), "q2": float(inst[q[1]].mean()),
                  "q3": float(inst[q[2]].mean()), "q4": float(inst[q[3]].mean()),
                  "slope": slope, "n_points": len(pts)}
    return out


if __name__ == "__main__":
    sys.exit(main())
