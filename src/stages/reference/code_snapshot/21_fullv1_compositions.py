# -*- coding: utf-8 -*-
"""
21_fullv1_compositions.py - STEP 4 Stage E: 50,000 unique compositions for FULL CALPHAD V1.

Sampling POLICY is unchanged from Step 3B/3C and is not re-derived here:
  * variables are S_CrNb = wCr + wNb and R_CrNb_atomic = nCr/nNb (never independent Cr/Nb)
  * Domain A / B ranges, X ranges, anchor centres and +/-10% neighbourhoods: frozen
  * Sobol: scipy.stats.qmc.Sobol(d=3, scramble=True, seed=20260903)
  * wt.% written to 4 decimals (this is what every previous stage fed to Thermo-Calc,
    so it stays; full-precision values are kept alongside as *_exact provenance columns)

Only the COUNTS and the SOBOL OFFSET change. The offset is derived from the source of
the three earlier generators, not guessed - see FULL_V1_SOBOL_CONTINUATION_AUDIT.md:

    stage      domain A            domain B
    Mini20     [0, 8)              [0, 8)
    Pilot200   [8, 264)            [8, 136)
    Pilot2000  [512, 2048)         [512, 1024)
    -> highest index ever consumed = 2048

FULL V1 therefore skips 2048 and draws ONE block of 65536 (2^16, keeps the sequence's
balance property and avoids scipy's non-power-of-2 warning):

    Domain A  = points[    0 : 35000 ]      -> global index [ 2048,  37048)
    Domain B  = points[35000 : 45000 ]      -> global index [37048,  47048)
    spare     = points[45000 : 65536 ]      -> collision replacements only

Domain A and Domain B use DISJOINT segments. Earlier stages had B's points as a subset
of A's (same seed, same skip); that produced no duplicate compositions because the S
ranges are disjoint, but it meant B was a re-mapping of A rather than independent
sampling. Full V1 fixes that; this changes no frozen policy, only the offset bookkeeping.

Anchor perturbation RNG stream: Mini20 used default_rng(SEED), Pilot200 SEED+1,
Pilot2000 SEED+2 -> Full V1 uses SEED+3.

Usage:
    python 21_fullv1_compositions.py            # generate + QA
    python 21_fullv1_compositions.py --verify   # QA only, on the existing file
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1")
PILOT_ROOT = Path(r"E:\Cu_CALPHAD_ML\02_pilot_2000")
DIR_COMPS = PROJECT_ROOT / "01_compositions"
DIR_QA = PROJECT_ROOT / "04_qa"
FILE_CSV = DIR_COMPS / "FULL_V1_COMPOSITIONS.csv"
FILE_PARQUET = DIR_COMPS / "FULL_V1_COMPOSITIONS.parquet"
FILE_QA = DIR_QA / "FULL_V1_COMPOSITION_QA_REPORT.md"
FILE_QA_JSON = DIR_QA / "FULL_V1_COMPOSITION_QA_SUMMARY.json"

PRIOR_FILES = [
    ("Step2.5", PILOT_ROOT / "00_config" / "DRY_RUN_CASES.csv"),
    ("Mini20", PILOT_ROOT / "01_compositions" / "mini20" / "MINI20_COMPOSITIONS.csv"),
    ("Pilot200", PILOT_ROOT / "01_compositions" / "pilot200" / "PILOT200_COMPOSITIONS.csv"),
    ("Pilot2000", PILOT_ROOT / "01_compositions" / "pilot2000" / "PILOT2000_COMPOSITIONS.csv"),
]

SEED = 20260903
SOBOL_SKIP = 2048
SOBOL_BLOCK = 65536
ANCHOR_RNG_OFFSET = 3
M_CR = 51.9961
M_NB = 92.9064
ATOMIC_WEIGHTS = {"CU": 63.546, "CR": M_CR, "NB": M_NB, "TI": 47.867, "SI": 28.0855,
                  "AG": 107.8682, "MO": 95.95, "ZR": 91.224, "CO": 58.9332}

DOMAINS = {
    "A": {"S_min": 0.60, "S_max": 2.00, "R_min": 2.00, "R_max": 6.50,
          "X_range": {"ZR": (0.02, 0.20), "TI": (0.02, 0.20), "CO": (0.03, 0.30),
                      "SI": (0.005, 0.08), "AG": (0.03, 0.30), "MO": (0.03, 0.30)}},
    "B": {"S_min": 2.00, "S_max": 4.50, "R_min": 1.00, "R_max": 4.00,
          "X_range": {"ZR": (0.02, 0.40), "TI": (0.02, 0.35), "CO": (0.03, 0.60),
                      "SI": (0.005, 0.15), "AG": (0.03, 0.60), "MO": (0.03, 0.60)}},
}
FAMILY_ORDER = ["ZR", "TI", "CO", "SI", "AG", "MO"]
X_ANCHORS = {
    "ZR": {"Cr": 2.0000, "Nb": 1.3500, "X": 0.1500},
    "TI": {"Cr": 1.6500, "Nb": 1.3500, "X": 0.1200},
    "CO": {"Cr": 2.1289, "Nb": 1.3167, "X": 0.1392},
    "SI": {"Cr": 2.1304, "Nb": 1.3177, "X": 0.0664},
    "AG": {"Cr": 2.1264, "Nb": 1.3152, "X": 0.2545},
    "MO": {"Cr": 2.1270, "Nb": 1.3156, "X": 0.2264},
}
BASE_ANCHOR = {"Cr": 0.4700, "Nb": 0.1600, "X": 0.0}

ALLOCS = {("A", "BASELINE"): 3500, ("B", "BASELINE"): 1000, ("ANCHOR", "BASELINE"): 500}
for _f in FAMILY_ORDER:
    ALLOCS[("A", _f)] = 5250
    ALLOCS[("B", _f)] = 1500
    ALLOCS[("ANCHOR", _f)] = 750
N_TOTAL = sum(ALLOCS.values())          # 50000
NEED_A = sum(ALLOCS[("A", k)] for k in ["BASELINE"] + FAMILY_ORDER)   # 35000
NEED_B = sum(ALLOCS[("B", k)] for k in ["BASELINE"] + FAMILY_ORDER)   # 10000
NI_PAT = r"(?i)(?<![a-z])ni(?![0-9a-z])"

STATS = {"sobol_collisions": 0, "anchor_collisions": 0, "spare_used": 0}


def sig(family, cr, nb, x):
    return (family, round(cr, 4), round(nb, 4), round(x, 4))


def load_prior_signatures():
    sigs, sources = set(), {}
    for label, f in PRIOR_FILES:
        if not Path(f).exists():
            sources[label] = "MISSING"
            continue
        df = pd.read_csv(f, dtype=str).fillna("")
        n = 0
        for _, r in df.iterrows():
            xel = (r.get("X_element", "") or "").strip().upper()
            x = float(r["X_wt_pct"]) if (xel and str(r.get("X_wt_pct", "")).strip()) else 0.0
            sigs.add(sig(r["family"], float(r["Cr_wt_pct"]), float(r["Nb_wt_pct"]), x))
            n += 1
        sources[label] = n
    return sigs, sources


def w_from_sr(s, r_target):
    w_nb = s / (1.0 + r_target * (M_CR / M_NB))
    return s - w_nb, w_nb


def r_actual(w_cr, w_nb):
    return (w_cr / M_CR) / (w_nb / M_NB)


def at_pct(wt_pct):
    moles = {e: w / ATOMIC_WEIGHTS[e] for e, w in wt_pct.items() if e in ATOMIC_WEIGHTS}
    tot = sum(moles.values())
    return {e: 100.0 * m / tot for e, m in moles.items()}


def make_row(cid, family, xel, domain, method, cr, nb, x, anchor_info=None, r_target=None,
             sobol_index=None):
    """wt.% rounded to 4 decimals is the Thermo-Calc input (frozen since Mini20);
    *_exact columns keep the pre-rounding double for provenance."""
    w = {"CU": None, "CR": round(cr, 4), "NB": round(nb, 4)}
    if xel:
        w[xel] = round(x, 4)
    solutes = sum(v for k, v in w.items() if k != "CU")
    w["CU"] = round(100.0 - solutes, 4)
    at = at_pct(w)
    ra = r_actual(w["CR"], w["NB"])
    return {
        "composition_id": cid, "family": family, "X_element": xel or "",
        "sampling_domain": domain, "sampling_method": method, "seed": SEED,
        "sobol_skip": SOBOL_SKIP,
        "sobol_global_index": ("" if sobol_index is None else int(sobol_index)),
        "Cu_wt_pct": w["CU"], "Cr_wt_pct": w["CR"], "Nb_wt_pct": w["NB"],
        "X_wt_pct": (w[xel] if xel else ""),
        "S_CrNb_wt_pct": round(w["CR"] + w["NB"], 4),
        "R_CrNb_atomic_target": (round(r_target, 6) if r_target else ""),
        "R_CrNb_atomic_actual": round(ra, 6),
        "R_CrNb_atomic_error": (round(ra - r_target, 8) if r_target else ""),
        "anchor_source": (anchor_info["source"] if anchor_info else ""),
        "anchor_perturbation_pct_Cr": (anchor_info["cr"] if anchor_info else ""),
        "anchor_perturbation_pct_Nb": (anchor_info["nb"] if anchor_info else ""),
        "anchor_perturbation_pct_X": (anchor_info["x"] if anchor_info else ""),
        "composition_sum_wt_pct": round(w["CU"] + solutes, 4),
        "Cu_at_pct": round(at["CU"], 6), "Cr_at_pct": round(at["CR"], 6),
        "Nb_at_pct": round(at["NB"], 6),
        "X_at_pct": (round(at[xel], 6) if xel else ""),
        # full-precision provenance (spec section 27) - NOT the Thermo-Calc input
        "Cr_wt_pct_exact": repr(float(cr)), "Nb_wt_pct_exact": repr(float(nb)),
        "X_wt_pct_exact": (repr(float(x)) if xel else ""),
    }


def generate():
    from scipy.stats import qmc
    prior, sources = load_prior_signatures()
    used = set(prior)
    rows = []

    seq = qmc.Sobol(d=3, scramble=True, seed=SEED)
    seq.random(SOBOL_SKIP)
    pts = seq.random(SOBOL_BLOCK)
    seg = {"A": (0, NEED_A), "B": (NEED_A, NEED_A + NEED_B)}
    spare_ptr = [NEED_A + NEED_B]

    def next_spare():
        i = spare_ptr[0]
        spare_ptr[0] += 1
        STATS["spare_used"] += 1
        if i >= SOBOL_BLOCK:
            raise RuntimeError("Sobol spare pool exhausted - increase SOBOL_BLOCK")
        return i, pts[i]

    for domain in ("A", "B"):
        d = DOMAINS[domain]
        lo, hi = seg[domain]
        idx = lo
        for k in FAMILY_ORDER + ["BASE"]:
            fam = "BASELINE" if k == "BASE" else k
            for j in range(ALLOCS[(domain, fam)]):
                while True:
                    if idx < hi:
                        i, q = idx, pts[idx]
                        idx += 1
                    else:
                        i, q = next_spare()
                    q_s, q_r, q_x = float(q[0]), float(q[1]), float(q[2])
                    s = d["S_min"] + q_s * (d["S_max"] - d["S_min"])
                    r_t = d["R_min"] + q_r * (d["R_max"] - d["R_min"])
                    w_cr, w_nb = w_from_sr(s, r_t)
                    if k == "BASE":
                        cid = "FV1_BASE_%s_%05d" % (domain, j + 1)
                        row = make_row(cid, "BASELINE", None, domain,
                                       "scipy.stats.qmc.Sobol(scramble=True)", w_cr, w_nb, 0.0,
                                       r_target=r_t, sobol_index=SOBOL_SKIP + i)
                    else:
                        xmin, xmax = d["X_range"][k]
                        x = xmin + q_x * (xmax - xmin)
                        cid = "FV1_%s_%s_%05d" % (k, domain, j + 1)
                        row = make_row(cid, k, k, domain,
                                       "scipy.stats.qmc.Sobol(scramble=True)", w_cr, w_nb, x,
                                       r_target=r_t, sobol_index=SOBOL_SKIP + i)
                    s2 = sig(row["family"], row["Cr_wt_pct"], row["Nb_wt_pct"],
                             float(row["X_wt_pct"]) if row["X_element"] else 0.0)
                    if s2 in used:
                        # 4-decimal grid collision: take the next Sobol point, never abort.
                        STATS["sobol_collisions"] += 1
                        continue
                    used.add(s2)
                    rows.append(row)
                    break

    rng = np.random.default_rng(SEED + ANCHOR_RNG_OFFSET)
    for fam in FAMILY_ORDER + ["BASELINE"]:
        a = X_ANCHORS[fam] if fam != "BASELINE" else BASE_ANCHOR
        xel = fam if fam != "BASELINE" else None
        src = ("Anchor_%s" % fam) if fam != "BASELINE" else "Cu-0.47Cr-0.16Nb manual anchor"
        for j in range(1, ALLOCS[("ANCHOR", fam)] + 1):
            for _try in range(500):
                p_cr = float(rng.uniform(-0.10, 0.10))
                p_nb = float(rng.uniform(-0.10, 0.10))
                p_x = float(rng.uniform(-0.10, 0.10)) if xel else 0.0
                cid = ("FV1_%s_ANCHOR_%05d" % (fam, j)) if xel else ("FV1_BASE_ANCHOR_%05d" % j)
                row = make_row(cid, fam, xel, "ANCHOR", "anchor_neighborhood_perturbation",
                               a["Cr"] * (1 + p_cr), a["Nb"] * (1 + p_nb),
                               (a["X"] * (1 + p_x)) if xel else 0.0,
                               anchor_info={"source": src, "cr": round(100 * p_cr, 6),
                                            "nb": round(100 * p_nb, 6),
                                            "x": (round(100 * p_x, 6) if xel else "")})
                s2 = sig(row["family"], row["Cr_wt_pct"], row["Nb_wt_pct"],
                         float(row["X_wt_pct"]) if xel else 0.0)
                if s2 in used:
                    STATS["anchor_collisions"] += 1
                    continue
                used.add(s2)
                rows.append(row)
                break
            else:
                raise RuntimeError("no unique anchor neighbour for %s #%d" % (fam, j))

    df = pd.DataFrame(rows)
    DIR_COMPS.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILE_CSV, index=False, encoding="utf-8")
    try:
        df.to_parquet(FILE_PARQUET, index=False)
    except Exception as e:                                   # noqa: BLE001
        print("parquet not written (%r) - CSV is authoritative" % (e,))
    STATS["sobol_consumed_end_index"] = SOBOL_SKIP + max(spare_ptr[0], NEED_A + NEED_B)
    return df, sources


# --------------------------------------------------------------------------- #
def run_qa(df):
    checks = []
    add = lambda d, ok, dt="": checks.append({"desc": d, "ok": bool(ok), "detail": str(dt)[:300]})
    prior, sources = load_prior_signatures()

    add("total compositions == 50,000", len(df) == N_TOTAL, len(df))
    add("composition_id unique", df.composition_id.nunique() == len(df), df.composition_id.nunique())
    fams = set(df.family)
    add("7 families complete", fams == {"BASELINE", "ZR", "TI", "CO", "SI", "AG", "MO"}, sorted(fams))
    add("baseline == 5,000", int((df.family == "BASELINE").sum()) == 5000,
        int((df.family == "BASELINE").sum()))
    xf = {f: int((df.family == f).sum()) for f in FAMILY_ORDER}
    add("each X family == 7,500", all(v == 7500 for v in xf.values()), xf)
    dom = df.sampling_domain.value_counts().to_dict()
    add("domain A35000 / B10000 / ANCHOR5000",
        dom == {"A": 35000, "B": 10000, "ANCHOR": 5000}, dom)
    per = df.groupby(["family", "sampling_domain"]).size().unstack(fill_value=0)
    ok_per = all(int(per.loc[f, d]) == ALLOCS[(d, f)]
                 for f in ["BASELINE"] + FAMILY_ORDER for d in ("A", "B", "ANCHOR"))
    add("per-family 70/20/10 split exact", ok_per, per.to_dict())

    bad = []
    A = df[df.sampling_domain.isin(["A", "B"])]
    for domain in ("A", "B"):
        d = DOMAINS[domain]
        sub = df[df.sampling_domain == domain]
        s = sub.S_CrNb_wt_pct.astype(float)
        r = sub.R_CrNb_atomic_actual.astype(float)
        if not ((s >= d["S_min"] - 1e-6) & (s <= d["S_max"] + 1e-6)).all():
            bad.append((domain, "S", float(s.min()), float(s.max())))
        if not ((r >= d["R_min"] - 1e-3) & (r <= d["R_max"] + 1e-3)).all():
            bad.append((domain, "R", float(r.min()), float(r.max())))
        for xel in FAMILY_ORDER:
            ss = sub[sub.X_element == xel]
            if len(ss):
                xv = ss.X_wt_pct.astype(float)
                lo, hi = d["X_range"][xel]
                if not ((xv >= lo - 1e-9) & (xv <= hi + 1e-9)).all():
                    bad.append((domain, xel, float(xv.min()), float(xv.max())))
    add("Domain A/B ranges satisfied", not bad, bad[:8])

    anc = df[df.sampling_domain == "ANCHOR"]
    pcr = anc.anchor_perturbation_pct_Cr.astype(float).abs().max()
    pnb = anc.anchor_perturbation_pct_Nb.astype(float).abs().max()
    px = pd.to_numeric(anc.anchor_perturbation_pct_X, errors="coerce").abs().max()
    add("anchor perturbation <= +/-10%", max(pcr, pnb, (px if px == px else 0)) <= 10 + 1e-9,
        "max |Cr|=%.4f |Nb|=%.4f |X|=%.4f" % (pcr, pnb, px if px == px else 0))

    sums = df.composition_sum_wt_pct.astype(float)
    add("composition sum == 100 wt%", bool((abs(sums - 100) < 0.01).all()),
        "min=%.4f max=%.4f" % (sums.min(), sums.max()))
    cu = df.Cu_wt_pct.astype(float)
    add("Cu > 90 wt%", bool((cu > 90).all()), "min=%.4f max=%.4f" % (cu.min(), cu.max()))
    solutes_ok = bool((df.Cr_wt_pct.astype(float) > 0).all()
                      and (df.Nb_wt_pct.astype(float) > 0).all()
                      and (pd.to_numeric(df.X_wt_pct, errors="coerce").dropna() > 0).all())
    add("all solutes > 0", solutes_ok, "")

    ra = pd.to_numeric(df.R_CrNb_atomic_actual, errors="coerce")
    rt = pd.to_numeric(df.R_CrNb_atomic_target, errors="coerce")
    dev = float((ra - rt).abs().max())
    add("R_actual ~ R_target (<=0.01)", dev <= 0.01, "max |R_error| = %.3e" % dev)

    sigs = df.apply(lambda r: sig(r.family, float(r.Cr_wt_pct), float(r.Nb_wt_pct),
                                  float(r.X_wt_pct) if r.X_element else 0.0), axis=1)
    add("no internal duplicate", sigs.nunique() == len(df), "%d/%d" % (sigs.nunique(), len(df)))
    overlap = sorted(set(sigs) & prior)
    add("no duplicate vs Step2.5 + Mini20 + Pilot200 + Pilot2000", len(overlap) == 0,
        "prior=%s overlap=%d %s" % (sources, len(overlap), overlap[:3]))

    ni_cols = [c for c in df.columns
               if df[c].astype(str).str.contains(NI_PAT, regex=True).any()]
    add("no Ni / NI anywhere", not ni_cols, ni_cols)
    num = df[["Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "S_CrNb_wt_pct"]].astype(float).values
    add("no NaN / Inf / negative", bool(np.isfinite(num).all() and (num >= 0).all()), "")

    idx = pd.to_numeric(df.sobol_global_index, errors="coerce").dropna()
    add("Sobol indices strictly above every prior stage (>= 2048)",
        bool(len(idx) == 0 or idx.min() >= SOBOL_SKIP),
        "min=%d max=%d (prior max consumed = 2048)" % (idx.min(), idx.max()) if len(idx) else "n/a")
    return checks, sources


def write_report(df, checks, sources):
    ok_all = all(c["ok"] for c in checks)
    per = df.groupby(["family", "sampling_domain"]).size().unstack(fill_value=0)
    L = ["# FULL_V1_COMPOSITION_QA_REPORT.md — STEP 4 Stage F", "",
         "**COMPOSITION QA STATUS: %s**" % ("PASS" if ok_all else "FAIL"), "",
         "- 生成器：`21_fullv1_compositions.py`",
         "- Sobol：`scipy.stats.qmc.Sobol(d=3, scramble=True, seed=%d)`，skip=**%d**，"
         "单块抽取 %d 点（2^16）" % (SEED, SOBOL_SKIP, SOBOL_BLOCK),
         "- Domain A = block[0:%d]（全局索引 [%d, %d)）；Domain B = block[%d:%d]（[%d, %d)）—— **两段不相交**"
         % (NEED_A, SOBOL_SKIP, SOBOL_SKIP + NEED_A, NEED_A, NEED_A + NEED_B,
            SOBOL_SKIP + NEED_A, SOBOL_SKIP + NEED_A + NEED_B),
         "- Anchor 扰动 RNG：`default_rng(%d)`（Mini20=%d，Pilot200=%d，Pilot2000=%d）"
         % (SEED + ANCHOR_RNG_OFFSET, SEED, SEED + 1, SEED + 2),
         "- 先前阶段 composition 计数：%s" % sources,
         "- 4 位小数网格碰撞：Sobol %d 次，Anchor %d 次；备用点池用掉 %d 个"
         % (STATS["sobol_collisions"], STATS["anchor_collisions"], STATS["spare_used"]),
         "", "| # | 检查项 | 状态 | 详情 |", "|---|--------|------|------|"]
    for i, c in enumerate(checks, 1):
        L.append("| %d | %s | **%s** | %s |" % (i, c["desc"], "PASS" if c["ok"] else "FAIL",
                                                c["detail"]))
    L += ["", "## 分配", "", "| family | A | B | ANCHOR | 合计 |", "|---|---|---|---|---|"]
    for f in ["BASELINE"] + FAMILY_ORDER:
        L.append("| %s | %d | %d | %d | %d |"
                 % (f, per.loc[f].get("A", 0), per.loc[f].get("B", 0),
                    per.loc[f].get("ANCHOR", 0), int(per.loc[f].sum())))
    L.append("| **总计** | %d | %d | %d | **%d** |"
             % (int(per["A"].sum()), int(per["B"].sum()), int(per["ANCHOR"].sum()), len(df)))
    L += ["", "## 组成分布（wt.%）", "",
          "| 变量 | min | median | max |", "|---|---|---|---|"]
    for col in ("Cu_wt_pct", "Cr_wt_pct", "Nb_wt_pct", "S_CrNb_wt_pct", "R_CrNb_atomic_actual"):
        v = pd.to_numeric(df[col], errors="coerce")
        L.append("| %s | %.4f | %.4f | %.4f |" % (col, v.min(), v.median(), v.max()))
    for xel in FAMILY_ORDER:
        v = pd.to_numeric(df[df.X_element == xel].X_wt_pct, errors="coerce")
        L.append("| X_wt_pct (%s) | %.4f | %.4f | %.4f |" % (xel, v.min(), v.median(), v.max()))
    L += ["", "## 前 10 行示例", "", "```"]
    for _, r in df.head(10).iterrows():
        L.append("  %-22s %-9s %-7s S=%-8s R=%-9s Cr=%-9s Nb=%-9s X=%-8s Cu=%s"
                 % (r.composition_id, r.family, r.sampling_domain, r.S_CrNb_wt_pct,
                    r.R_CrNb_atomic_actual, r.Cr_wt_pct, r.Nb_wt_pct, r.X_wt_pct, r.Cu_wt_pct))
    L += ["```", "",
          "> wt.% 以 4 位小数写出，这是 Mini20 起每个阶段实际喂给 Thermo-Calc 的口径（规范 §27：",
          "> 先审计现有行为并保持一致）。未取整的双精度值另存于 `*_exact` 列作为溯源，不作为计算输入。",
          "", "**若 FAIL：停止，禁止开始 550,000-state 计算。**"]
    DIR_QA.mkdir(parents=True, exist_ok=True)
    FILE_QA.write_text("\n".join(L), encoding="utf-8")
    FILE_QA_JSON.write_text(json.dumps(
        {"status": "PASS" if ok_all else "FAIL", "n": len(df), "checks": checks,
         "sobol": {"seed": SEED, "skip": SOBOL_SKIP, "block": SOBOL_BLOCK,
                   "domain_A_global_index": [SOBOL_SKIP, SOBOL_SKIP + NEED_A],
                   "domain_B_global_index": [SOBOL_SKIP + NEED_A, SOBOL_SKIP + NEED_A + NEED_B],
                   "anchor_rng": SEED + ANCHOR_RNG_OFFSET},
         "collisions": STATS, "prior_counts": sources,
         "allocation": per.to_dict()}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args, _ = ap.parse_known_args()
    if args.verify and FILE_CSV.exists():
        df = pd.read_csv(FILE_CSV, dtype=str).fillna("")
        print("verifying existing %s (%d rows)" % (FILE_CSV.name, len(df)))
        _, sources = load_prior_signatures()
        if not FILE_PARQUET.exists():
            # the CSV may have been produced where pyarrow was unavailable
            try:
                pd.read_csv(FILE_CSV).to_parquet(FILE_PARQUET, index=False)
                print("wrote %s" % FILE_PARQUET.name)
            except Exception as e:                            # noqa: BLE001
                print("parquet still not written (%r) - CSV is authoritative" % (e,))
    else:
        df, sources = generate()
        print("generated %d compositions | collisions: %s" % (len(df), STATS))
    checks, sources = run_qa(df)
    ok = write_report(df, checks, sources)
    for c in checks:
        print("  [%s] %s : %s" % ("PASS" if c["ok"] else "FAIL", c["desc"], c["detail"][:110]))
    print()
    print("COMPOSITION_QA_VERDICT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
