# -*- coding: utf-8 -*-
"""
22b_fullv1_proc_benchmark.py - STEP 4 Stage G (optional, non-blocking):
2-process vs 4-process benchmark before the 550,000-state production run.

Spec section 34: production defaults to 2 processes (validated in Pilot200 + Pilot2000).
4 processes may ONLY be used if this benchmark shows
    * >= MIN_SPEEDUP relative to 2 processes, AND
    * numerically identical results, AND
    * no license / server conflict, AND
    * checkpoint + raw-result consistency intact.
Anything else -> stay at 2 processes. This script never touches the production tree.

Exit code 0 = 4 processes recommended, 1 = keep 2 processes (also on any error).
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1")
sys.path.insert(0, str(PROJECT_ROOT / "code_snapshot"))
import io_store  # noqa: E402

N_COMPS = 80
MIN_SPEEDUP = 1.5
FLOAT_TOL = 1e-12
BENCH = PROJECT_ROOT / "05_logs" / "proc_benchmark"
REPORT = PROJECT_ROOT / "04_qa" / "FULL_V1_PROC_BENCHMARK.md"
RUNNER = PROJECT_ROOT / "code_snapshot" / "22_fullv1_run.py"
COMPS = PROJECT_ROOT / "01_compositions" / "FULL_V1_COMPOSITIONS.csv"


def subset_csv(dest: Path, n=N_COMPS):
    df = pd.read_csv(COMPS, dtype=str).fillna("")
    per = max(1, n // df.family.nunique())
    sub = df.groupby("family", sort=False).head(per).head(n).reset_index(drop=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(dest, index=False, encoding="utf-8")
    return sub


def run_arm(nproc, root: Path, comps_csv: Path):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    procs = []
    for k in range(1, nproc + 1):
        procs.append(subprocess.Popen(
            [sys.executable, str(RUNNER), "--shard", str(k), "--shards", str(nproc),
             "--comps", str(comps_csv), "--outroot", str(root)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        time.sleep(2)                       # stagger TC-Python server starts
    errs = []
    for k, p in enumerate(procs, 1):
        _, err = p.communicate()
        if p.returncode != 0:
            errs.append("shard %d exit %d: %s" % (k, p.returncode,
                                                  (err or b"").decode("utf-8", "replace")[-400:]))
    return time.time() - t0, errs


def collect(root: Path, nproc):
    st, ph = [], []
    for k in range(1, nproc + 1):
        f = root / "02_raw" / ("FULL_V1_shard%02d_EQUILIBRIUM_STATES.csv" % k)
        if f.exists():
            st.append(pd.read_csv(f, keep_default_na=False))
        try:
            ph.append(io_store.read_table(root / "02_raw"
                                          / ("FULL_V1_shard%02d_PHASES_LONG" % k)))
        except FileNotFoundError:
            pass
    st = pd.concat(st, ignore_index=True) if st else pd.DataFrame()
    ph = pd.concat(ph, ignore_index=True) if ph else pd.DataFrame()
    return st.sort_values("state_id").reset_index(drop=True), \
        ph.sort_values(["state_id", "raw_phase_name"]).reset_index(drop=True)


def cmp_num(a, b, cols, label, diffs):
    if len(a) != len(b):
        diffs.append("%s: row count %d vs %d" % (label, len(a), len(b)))
        return
    for c in cols:
        if c not in a.columns or c not in b.columns:
            continue
        na = pd.to_numeric(a[c], errors="coerce").astype("float64")
        nb = pd.to_numeric(b[c], errors="coerce").astype("float64")
        d = (na - nb).abs()
        scale = pd.concat([na.abs(), nb.abs()], axis=1).max(axis=1).replace(0, 1.0)
        bad = int(((d / scale).fillna(0.0) > FLOAT_TOL).sum())
        nanmis = int((na.isna() != nb.isna()).sum())
        if bad or nanmis:
            diffs.append("%s.%s: %d differ, %d NaN-pattern" % (label, c, bad, nanmis))


def main():
    BENCH.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    comps_csv = BENCH / "subset_compositions.csv"
    sub = subset_csv(comps_csv)
    print("benchmark subset: %d compositions x 11 T = %d states" % (len(sub), len(sub) * 11))

    print("--- arm 1/2: 2 processes ---")
    t2, e2 = run_arm(2, BENCH / "p2", comps_csv)
    print("    2-process wall = %.1f s%s" % (t2, "  ERRORS" if e2 else ""))
    print("--- arm 2/2: 4 processes ---")
    t4, e4 = run_arm(4, BENCH / "p4", comps_csv)
    print("    4-process wall = %.1f s%s" % (t4, "  ERRORS" if e4 else ""))

    diffs = []
    s2, p2 = collect(BENCH / "p2", 2)
    s4, p4 = collect(BENCH / "p4", 4)
    cmp_num(s2, s4, ["Temperature_C", "sum_phase_mole_fraction", "sum_phase_volume_fraction",
                     "n_stable_phases"], "STATES", diffs)
    cmp_num(p2, p4, ["phase_moles", "phase_mole_fraction", "phase_volume_fraction",
                     "phase_mass_fraction"], "PHASES", diffs)
    if len(s2) and set(s2.state_id) != set(s4.state_id):
        diffs.append("state_id sets differ")
    conv2 = int((s2["converged"].astype(str).str.lower() == "true").sum()) if len(s2) else 0
    conv4 = int((s4["converged"].astype(str).str.lower() == "true").sum()) if len(s4) else 0
    if conv2 != conv4:
        diffs.append("converged count %d vs %d" % (conv2, conv4))

    speedup = t2 / max(t4, 1e-9)
    errs = e2 + e4
    ok = (not errs) and (not diffs) and speedup >= MIN_SPEEDUP and len(s4) == len(sub) * 11
    verdict = "USE_4_PROCESS" if ok else "KEEP_2_PROCESS"

    L = ["# FULL_V1_PROC_BENCHMARK.md — STEP 4 Stage G（可选，非阻塞）", "",
         "**PROC_BENCHMARK_VERDICT: %s**" % verdict, "",
         "- 样本：%d compositions × 11 T = **%d states**（分层取自正式 50k 表，不写入生产目录）"
         % (len(sub), len(sub) * 11),
         "- 门禁：4-process 相对 2-process 加速 ≥ %.1f× **且** 数值完全一致 **且** 无进程错误" % MIN_SPEEDUP,
         "- 规范 §30/§34：默认生产用 2 进程；未经本 benchmark 通过，禁止用 4 进程跑 50k。", "",
         "## 1. 时间", "", "| 进程数 | wall | 相对 2-process |", "|---|---|---|",
         "| 2 | %.1f s | 1.00× |" % t2,
         "| 4 | %.1f s | **%.2f×** |" % (t4, speedup), "",
         "## 2. 数值一致性", ""]
    if diffs:
        L += ["**不一致：**", ""] + ["- `%s`" % d for d in diffs]
    else:
        L += ["**完全一致。** states（相分数和、稳定相数）与 phases（相摩尔数、摩尔/体积/质量分数）"
              "逐值比对，相对容差 %.0e，NaN 模式一致；收敛计数 %d = %d。" % (FLOAT_TOL, conv2, conv4)]
    L += ["", "## 3. 进程错误", ""]
    L += (["- `%s`" % e for e in errs] if errs else ["无。两条臂的所有子进程均以 0 退出，"
                                                     "未出现 license / server 冲突。"])
    L += ["", "## 4. 结论", ""]
    if ok:
        L.append("**%s** — 加速 %.2f×（门禁 ≥%.1f×），数值完全一致，无进程错误。"
                 "允许用 4 进程执行 550,000-state 生产运行。" % (verdict, speedup, MIN_SPEEDUP))
    else:
        why = []
        if errs:
            why.append("出现进程错误")
        if diffs:
            why.append("数值不一致")
        if speedup < MIN_SPEEDUP:
            why.append("加速 %.2f× < 门禁 %.1f×" % (speedup, MIN_SPEEDUP))
        if len(s4) != len(sub) * 11:
            why.append("4-process 只产出 %d / %d 个 state" % (len(s4), len(sub) * 11))
        L.append("**%s** — %s。生产运行继续使用 2 进程（Pilot200 与 Pilot2000 已验证的配置）。"
                 % (verdict, "；".join(why)))
    REPORT.write_text("\n".join(L), encoding="utf-8")
    (REPORT.parent / "FULL_V1_PROC_BENCHMARK.json").write_text(json.dumps(
        {"verdict": verdict, "n_compositions": len(sub), "n_states": len(sub) * 11,
         "wall_2proc": t2, "wall_4proc": t4, "speedup": speedup, "min_speedup_gate": MIN_SPEEDUP,
         "diffs": diffs, "errors": errs, "converged_2": conv2, "converged_4": conv4},
        indent=2, default=str), encoding="utf-8")
    print()
    print("PROC_BENCHMARK_VERDICT: %s  (speedup %.2fx, diffs %d, errors %d)"
          % (verdict, speedup, len(diffs), len(errs)))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:                                    # noqa: BLE001
        print("PROC_BENCHMARK_VERDICT: KEEP_2_PROCESS  (benchmark itself failed: %r)" % (e,))
        sys.exit(1)
