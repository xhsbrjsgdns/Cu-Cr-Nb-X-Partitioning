# -*- coding: utf-8 -*-
"""
42_step5a_error_analysis.py
STEP5A Stage S: IID error stratification (sec.79-81). Runs only AFTER the test
was opened by 40_step5a_iid_test.py.

Strata:
  TEST_ERROR_BY_FAMILY.csv           (alloy family: BASELINE/ZR/TI/CO/SI/AG/MO)
  TEST_ERROR_BY_TEMPERATURE.csv      (11-point grid)
  TEST_ERROR_BY_PHASE_SPLIT.csv      (multi-instance FCC/BCC regions vs not;
                                      flags are post-hoc metadata ONLY)
  TEST_ERROR_BY_COMPOSITION_RANGE.csv(S_CrNb and X_wt_pct quartile bins)

No TEST_ERROR_BY_DOMAIN file is produced (Domain B stays sealed).
BCC_B2#3 states (4, all MO@950) are annotated diagnostically only.
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5a_common as C

FULL_V1 = C.FULL_V1
FAMILIES = C.FAMILIES_ORDER


def stratum_metrics(sub, ths):
    from sklearn.metrics import f1_score
    out = {"n_states": len(sub), "n_compositions": sub["composition_id"].nunique()}
    prob = np.column_stack([sub[f"{f}_presence_prob"].values for f in FAMILIES])
    tp = np.column_stack([sub[f"{f}_presence_true"].values for f in FAMILIES]).astype(int)
    m = np.isfinite(prob).all(axis=1)
    if m.any():
        pb = (prob[m] >= ths).astype(int)
        out["presence_exact_match"] = float((pb == tp[m]).all(axis=1).mean())
        out["presence_macro_f1"] = float(f1_score(tp[m], pb, average="macro", zero_division=0))
        out["liquid_f1"] = float(f1_score(tp[m][:, 4], pb[:, 4], zero_division=0))
    fp = np.column_stack([sub[f"{f}_frac_proj"].values for f in FAMILIES])
    ft = np.column_stack([sub[f"{f}_volfrac_true"].values for f in FAMILIES])
    out["fraction_mean_tvd"] = float(C.tvd(fp, ft).mean())
    out["fraction_macro_mae"] = float(np.abs(fp - ft).mean())
    for i, fam in enumerate(FAMILIES):
        out[f"fraction_mae_{fam}"] = float(np.abs(fp[:, i] - ft[:, i]).mean())
    for el, tag in [("cr", "CR_P"), ("nb", "NB_P"), ("x", "X_P")]:
        pr = np.column_stack([sub[f"{f}_{tag}_proj"].values for f in FAMILIES])
        tr = np.column_stack([sub[f"{tag}_{f}_true"].values for f in FAMILIES])
        mm = np.isfinite(tr).all(axis=1)
        out[f"{el}_mean_tvd"] = float(C.tvd(pr[mm], tr[mm]).mean()) if mm.any() else np.nan
        out[f"{el}_n"] = int(mm.sum())
    return out


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    assert os.path.exists(os.path.join(C.RESULTS, "IID_TEST_PREDICTIONS.parquet")), \
        "run 40_step5a_iid_test.py first"
    preds = pd.read_parquet(os.path.join(C.RESULTS, "IID_TEST_PREDICTIONS.parquet"))
    freeze = json.load(open(os.path.join(C.STEP5A, "09_manifest", "DEVELOPMENT_FREEZE_V1.json"),
                            encoding="utf-8"))
    ths = np.array([freeze["tasks"]["presence"]["thresholds"][f] for f in FAMILIES])

    # join stratification metadata (post-hoc only)
    diag = pd.read_parquet(os.path.join(FULL_V1, "03_processed", "FULL_V1_ML_READY_CONVERGED.parquet"),
                           columns=["state_id", "has_multiple_FCC_instances", "has_multiple_BCC_instances"])
    preds = preds.merge(diag, on="state_id", how="left", validate="1:1")
    compfeat = pd.read_parquet(os.path.join(C.FEAT, "FULL_V1_CANDIDATE_FEATURES.parquet"),
                               columns=["composition_id", "S_CrNb_wt_pct", "X_wt_pct"]).drop_duplicates()
    preds = preds.merge(compfeat, on="composition_id", how="left", validate="m:1")

    # BCC_B2#3 diagnostic annotation (4 states overall; in IID_TEST possibly fewer)
    bcc3_states = ["FV1_MO_ANCHOR_00535__T0950C", "FV1_MO_ANCHOR_00687__T0950C",
                   "FV1_MO_A_00696__T0950C", "FV1_MO_A_03181__T0950C"]
    n_bcc3_in_test = int(preds["state_id"].isin(bcc3_states).sum())

    rows = []
    for fam, sub in preds.groupby("family"):
        rows.append({"stratum": f"family={fam}", **stratum_metrics(sub, ths)})
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "TEST_ERROR_BY_FAMILY.csv"), index=False)

    rows = []
    for t, sub in preds.groupby("Temperature_C"):
        rows.append({"stratum": f"T={t:g}C", **stratum_metrics(sub, ths)})
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "TEST_ERROR_BY_TEMPERATURE.csv"), index=False)

    rows = []
    groups = {
        "multi_FCC_instance(AG<=400C)": preds["has_multiple_FCC_instances"] == 1,
        "single_FCC_instance": preds["has_multiple_FCC_instances"] == 0,
        "multi_BCC_instance(MO<=950C)": preds["has_multiple_BCC_instances"] == 1,
        "single_BCC_instance": preds["has_multiple_BCC_instances"] == 0,
        "any_split_region": (preds["has_multiple_FCC_instances"] == 1) | (preds["has_multiple_BCC_instances"] == 1),
        "no_split_region": (preds["has_multiple_FCC_instances"] == 0) & (preds["has_multiple_BCC_instances"] == 0),
    }
    for name, mask in groups.items():
        sub = preds[mask]
        if len(sub):
            rows.append({"stratum": name, **stratum_metrics(sub, ths)})
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "TEST_ERROR_BY_PHASE_SPLIT.csv"), index=False)

    rows = []
    preds["S_bin"] = pd.qcut(preds["S_CrNb_wt_pct"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    for b, sub in preds.groupby("S_bin", observed=True):
        rows.append({"stratum": f"S_CrNb={b}", "range_min": sub["S_CrNb_wt_pct"].min(),
                     "range_max": sub["S_CrNb_wt_pct"].max(), **stratum_metrics(sub, ths)})
    xsub = preds[preds["X_wt_pct"] > 0].copy()
    xsub["X_bin"] = pd.qcut(xsub["X_wt_pct"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    for b, sub in xsub.groupby("X_bin", observed=True):
        rows.append({"stratum": f"X_wt_pct={b}", "range_min": sub["X_wt_pct"].min(),
                     "range_max": sub["X_wt_pct"].max(), **stratum_metrics(sub, ths)})
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "TEST_ERROR_BY_COMPOSITION_RANGE.csv"), index=False)

    meta = {"generated_utc": ts, "bcc3_labelled_states_in_iid_test": n_bcc3_in_test,
            "note": "BCC_B2#3 states are diagnostic annotation only (never a class, feature, or "
                    "physical phase). has_multiple_* flags are post-hoc stratification metadata only."}
    with open(os.path.join(C.RESULTS, "ERROR_STRATIFICATION_META.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"BCC_B2#3 states inside IID_TEST: {n_bcc3_in_test}")
    print("STAGE S: done")


if __name__ == "__main__":
    main()
