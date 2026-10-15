# -*- coding: utf-8 -*-
"""
33_step5a_master_split.py
STEP5A Stage D + Stage E.

Stage D: deterministic family-stratified composition-level split of all 50,000
FULL_V1 compositions (master seed 20260904):
  Domain B   -> SEALED_DOMAIN_B   (10,000)
  ANCHOR     -> SEALED_ANCHOR     (5,000)
  Domain A   -> DEV_TRAIN 24,500 / DEV_VALIDATION 5,250 / IID_TEST 5,250
     BASELINE 3,500: Train 2,450 / Val 525 / Test 525
     each X family 5,250: Train 3,675; Val/Test per the fixed schedule
       ZR 788/787, TI 787/788, CO 788/787, SI 787/788, AG 788/787, MO 787/788
  Within each family: sort composition_id, permute with rng(20260904),
  first n_train -> DEV_TRAIN, next n_val -> DEV_VALIDATION, rest -> IID_TEST.
  No target information is used; no reshuffling based on targets.

Stage E: seal QA - Domain B and ANCHOR are confirmed by composition count,
family count and ID integrity only; no metrics, no predictions.

Outputs:
  03_splits/COMPOSITION_SPLIT_MASTER_V1.csv   (50,000 rows)
  03_splits/IID_SPLIT_QA.md
  03_splits/SPLIT_MASTER_SUMMARY.json
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FULL_V1 = r"E:\Cu_CALPHAD_ML\03_full_CALPHAD\FULL_V1"
STEP5A = r"E:\Cu_CALPHAD_ML\04_ML\STEP5A_JMST_ML_FOUNDATION_V2"
SPLITS = os.path.join(STEP5A, "03_splits")
MASTER_SEED = 20260904

VAL_TEST_SCHEDULE = {
    # family: (n_val, n_test) - fixed by task book sec.21
    "ZR": (788, 787),
    "TI": (787, 788),
    "CO": (788, 787),
    "SI": (787, 788),
    "AG": (788, 787),
    "MO": (787, 788),
}
N_TRAIN_X = 3675
BASELINE_ALLOC = (2450, 525, 525)  # train, val, test

failures = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(name)
    print(f"[{status}] {name} {detail}")
    return {"check": name, "status": status, "detail": str(detail)}


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    comp = pd.read_parquet(os.path.join(FULL_V1, "01_compositions", "FULL_V1_COMPOSITIONS.parquet"),
                           columns=["composition_id", "family", "sampling_domain"])
    results = []
    results.append(check("compositions == 50000", len(comp) == 50_000, len(comp)))
    results.append(check("composition_id unique", comp["composition_id"].is_unique))

    roles = pd.Series(index=comp.index, dtype=object)
    dom = comp["sampling_domain"]
    roles[dom == "B"] = "SEALED_DOMAIN_B"
    roles[dom == "ANCHOR"] = "SEALED_ANCHOR"

    rng = np.random.default_rng(MASTER_SEED)
    domain_a = comp[dom == "A"]
    for family in ["BASELINE", "ZR", "TI", "CO", "SI", "AG", "MO"]:
        ids = np.sort(domain_a.loc[domain_a["family"] == family, "composition_id"].values)
        perm = rng.permutation(ids)
        if family == "BASELINE":
            n_tr, n_va, n_te = BASELINE_ALLOC
        else:
            n_va, n_te = VAL_TEST_SCHEDULE[family]
            n_tr = N_TRAIN_X
        assert len(perm) == n_tr + n_va + n_te, (family, len(perm))
        roles[comp["composition_id"].isin(perm[:n_tr])] = "DEV_TRAIN"
        roles[comp["composition_id"].isin(perm[n_tr:n_tr + n_va])] = "DEV_VALIDATION"
        roles[comp["composition_id"].isin(perm[n_tr + n_va:])] = "IID_TEST"

    master = comp.copy()
    master["split_role"] = roles
    assert master["split_role"].notna().all()

    os.makedirs(SPLITS, exist_ok=True)
    out_csv = os.path.join(SPLITS, "COMPOSITION_SPLIT_MASTER_V1.csv")
    master.to_csv(out_csv, index=False)

    # ---------------- verification ----------------
    counts = master["split_role"].value_counts().to_dict()
    expected = {"DEV_TRAIN": 24_500, "DEV_VALIDATION": 5_250, "IID_TEST": 5_250,
                "SEALED_DOMAIN_B": 10_000, "SEALED_ANCHOR": 5_000}
    results.append(check("split_role totals exact", counts == expected, counts))

    dev = master[master["split_role"].isin(["DEV_TRAIN", "DEV_VALIDATION", "IID_TEST"])]
    results.append(check("every development composition is Domain A",
                         (dev["sampling_domain"] == "A").all()))
    results.append(check("development covers all 35000 Domain A compositions",
                         len(dev) == 35_000))

    sets = {r: set(master.loc[master["split_role"] == r, "composition_id"]) for r in expected}
    results.append(check("Train ∩ Val == 0", len(sets["DEV_TRAIN"] & sets["DEV_VALIDATION"]) == 0))
    results.append(check("Train ∩ IID Test == 0", len(sets["DEV_TRAIN"] & sets["IID_TEST"]) == 0))
    results.append(check("Val ∩ IID Test == 0", len(sets["DEV_VALIDATION"] & sets["IID_TEST"]) == 0))
    dev_union = sets["DEV_TRAIN"] | sets["DEV_VALIDATION"] | sets["IID_TEST"]
    results.append(check("development ∩ Domain B == 0", len(dev_union & sets["SEALED_DOMAIN_B"]) == 0))
    results.append(check("development ∩ Anchor == 0", len(dev_union & sets["SEALED_ANCHOR"]) == 0))
    all_ids = set(master["composition_id"])
    results.append(check("each composition appears exactly once",
                         len(all_ids) == 50_000 and sum(len(s) for s in sets.values()) == 50_000))

    # per-family dev allocation
    fam_tab = master[master["split_role"].isin(["DEV_TRAIN", "DEV_VALIDATION", "IID_TEST"])] \
        .pivot_table(index="family", columns="split_role", values="composition_id", aggfunc="count", fill_value=0)
    ok_base = (fam_tab.loc["BASELINE", "DEV_TRAIN"] == 2450
               and fam_tab.loc["BASELINE", "DEV_VALIDATION"] == 525
               and fam_tab.loc["BASELINE", "IID_TEST"] == 525)
    results.append(check("BASELINE 2450/525/525", ok_base, fam_tab.loc["BASELINE"].to_dict()))
    ok_x = True
    for fam, (n_va, n_te) in VAL_TEST_SCHEDULE.items():
        ok_x &= (fam_tab.loc[fam, "DEV_TRAIN"] == 3675
                 and fam_tab.loc[fam, "DEV_VALIDATION"] == n_va
                 and fam_tab.loc[fam, "IID_TEST"] == n_te)
    results.append(check("X families 3675 train + fixed val/test schedule", ok_x, fam_tab.to_dict()))

    # sealed-set family counts (Stage E: ID integrity only, no metrics)
    b_tab = master[master["split_role"] == "SEALED_DOMAIN_B"]["family"].value_counts().to_dict()
    results.append(check("Domain B family counts BASELINE=1000, each X=1500",
                         b_tab == {"BASELINE": 1_000, "ZR": 1_500, "TI": 1_500, "CO": 1_500,
                                   "SI": 1_500, "AG": 1_500, "MO": 1_500}, b_tab))
    a_tab = master[master["split_role"] == "SEALED_ANCHOR"]["family"].value_counts().to_dict()
    results.append(check("ANCHOR family counts BASELINE=500, each X=750",
                         a_tab == {"BASELINE": 500, "ZR": 750, "TI": 750, "CO": 750,
                                   "SI": 750, "AG": 750, "MO": 750}, a_tab))
    results.append(check("Domain B IDs all contain _B_ marker",
                         master.loc[master["split_role"] == "SEALED_DOMAIN_B", "composition_id"]
                         .str.contains("_B_").all()))
    results.append(check("ANCHOR IDs all contain _ANCHOR_ marker",
                         master.loc[master["split_role"] == "SEALED_ANCHOR", "composition_id"]
                         .str.contains("_ANCHOR_").all()))
    results.append(check("Domain A dev IDs all contain _A_ marker",
                         dev["composition_id"].str.contains("_A_").all()))

    # state-level inheritance check against ML_READY (read-only)
    mlr = pd.read_parquet(os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet"),
                          columns=["state_id", "composition_id", "family", "sampling_domain"])
    joined = mlr.merge(master[["composition_id", "split_role"]], on="composition_id", how="left")
    results.append(check("every ML_READY state inherits a split_role", joined["split_role"].notna().all()))
    # each composition's states all share one role (guaranteed by construction at composition level)
    per_comp_roles = joined.groupby("composition_id")["split_role"].nunique()
    results.append(check("each composition's states share exactly one split_role",
                         (per_comp_roles == 1).all()))
    dev_state_counts = joined[joined["split_role"].isin(["DEV_TRAIN", "DEV_VALIDATION", "IID_TEST"])] \
        .groupby("split_role").size().to_dict()
    results.append(check("no Domain B state tagged as development",
                         not any(joined[(joined["sampling_domain"] == "B")]["split_role"]
                                 .isin(["DEV_TRAIN", "DEV_VALIDATION", "IID_TEST"]))))
    results.append(check("no ANCHOR state tagged as development",
                         not any(joined[(joined["sampling_domain"] == "ANCHOR")]["split_role"]
                                 .isin(["DEV_TRAIN", "DEV_VALIDATION", "IID_TEST"]))))

    sha = hashlib.sha256(open(out_csv, "rb").read()).hexdigest()

    # ---------------- IID_SPLIT_QA.md ----------------
    fam_tab_dev = joined[joined["split_role"].isin(["DEV_TRAIN", "DEV_VALIDATION", "IID_TEST"])] \
        .pivot_table(index="family", columns="split_role", values="state_id", aggfunc="count", fill_value=0)
    with open(os.path.join(SPLITS, "IID_SPLIT_QA.md"), "w", encoding="utf-8") as f:
        f.write("# IID SPLIT QA — COMPOSITION_SPLIT_MASTER_V1\n\n")
        f.write(f"generated_utc: {ts}\n\n")
        f.write(f"master seed: {MASTER_SEED}\n\n")
        f.write("Method: per family, composition_ids sorted, permuted with numpy "
                f"default_rng({MASTER_SEED}) (one shared stream, families processed in order "
                "BASELINE, ZR, TI, CO, SI, AG, MO); first n_train -> DEV_TRAIN, next n_val -> "
                "DEV_VALIDATION, remainder -> IID_TEST. No target information was consulted; "
                "no reshuffling based on targets.\n\n")
        f.write("## Composition-level counts\n\n")
        for r, n in expected.items():
            f.write(f"- {r}: {counts.get(r)} (expected {n})\n")
        f.write(f"- total: {sum(counts.values())}\n\n")
        f.write("## Disjointness\n\n")
        for c in results:
            if "∩" in c["check"] or "exactly once" in c["check"]:
                f.write(f"- {c['check']}: {c['status']}\n")
        f.write("\n## Composition-level allocation by family (Domain A)\n\n")
        f.write(fam_tab.to_markdown() + "\n\n")
        f.write("## State-level inheritance (ML_READY_CONVERGED)\n\n")
        f.write(f"- development states by role: {dev_state_counts}\n")
        f.write(f"- MO compositions with 10 converged states (950 C FAILED): their remaining "
                "10 states all inherit the composition's single split_role — never split apart.\n\n")
        f.write("State rows per family x role (development):\n\n")
        f.write(fam_tab_dev.to_markdown() + "\n\n")
        f.write("## Seal QA (Stage E)\n\n")
        f.write("- Domain B: composition count, family count and ID integrity confirmed only. "
                "No predictions, no metrics, no scaler/feature/tuning contact. SEALED for STEP5B.\n")
        f.write("- ANCHOR: composition count, family count and ID integrity confirmed only. "
                "ANCHOR is a CALPHAD thermodynamic anchor domain, NOT an experimental validation set. SEALED.\n\n")
        f.write(f"## Master file hash\n\n- COMPOSITION_SPLIT_MASTER_V1.csv SHA256: {sha}\n")

    summary = {
        "stage": "STEP5A Stage D + E",
        "generated_utc": ts,
        "master_seed": MASTER_SEED,
        "status": "PASS" if not failures else "FAIL",
        "counts": counts,
        "family_allocation_compositions": fam_tab.to_dict(),
        "development_state_rows": dev_state_counts,
        "master_csv_sha256": sha,
        "checks": results,
        "failures": failures,
    }
    with open(os.path.join(SPLITS, "SPLIT_MASTER_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSTAGE D+E SPLIT: {'PASS' if not failures else 'FAIL'} "
          f"({len(results)} checks, {len(failures)} failures)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
