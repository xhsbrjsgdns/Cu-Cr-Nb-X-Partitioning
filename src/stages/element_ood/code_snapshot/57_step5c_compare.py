# -*- coding: utf-8 -*-
"""
57_step5c_compare.py
STEP5C Stage N: the three-layer generalization comparison (spec 60, 61).

  STEP5A  IID                       (Domain A held-out compositions, all six X elements seen)
  STEP5B  composition-domain OOD    (Domain B extrapolation shell, all six X elements seen)
  STEP5C  element-domain OOD        (Domain A, one X element entirely held out)

Spec 61: the LOEO layer is reported PER ELEMENT as well as mean / median / min / max.
An average over six folds alone would hide exactly the element-to-element differences
this stage exists to measure.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5c_common as C

A_METRICS = os.path.join(C.STEP5A, "05_results", "BASELINE_IID_TEST_METRICS.csv")
B_METRICS = os.path.join(C.STEP5B, "04_metrics", "DOMAIN_B_OOD_METRICS.csv")

HEADLINE = {"presence": "macro_f1", "fraction": "mean_tvd", "cr_partition": "mean_tvd",
            "nb_partition": "mean_tvd", "x_partition": "mean_tvd"}


def iid_values():
    a = pd.read_csv(A_METRICS)
    a = a[a["stage"] == "test"] if "stage" in a.columns else a
    out = {}
    for task, metric in HEADLINE.items():
        r = a[a.task == task]
        if len(r) and metric in r.columns and pd.notna(r.iloc[0][metric]):
            out[task] = float(r.iloc[0][metric])
    return out


def domain_b_values():
    b = pd.read_csv(B_METRICS)
    b = b[~b.model.astype(str).str.startswith("Trivial")]
    out = {}
    for task, metric in HEADLINE.items():
        r = b[b.task == task]
        if len(r) and metric in r.columns and pd.notna(r.iloc[0][metric]):
            out[task] = float(r.iloc[0][metric])
    return out


def main():
    print("=" * 78)
    print("STEP 5C   Stage N: IID vs composition-OOD vs element-OOD (LOEO)")
    print("=" * 78)
    iid, dom_b = iid_values(), domain_b_values()
    loeo = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_ALL_FOLDS_METRICS.csv"))

    rows = []
    for task, metric in HEADLINE.items():
        sub = loeo[loeo.task == task]
        if not len(sub):
            continue
        v = sub.set_index("heldout_element")["value"]
        higher_is_better = (task == "presence")
        rec = {
            "task": task, "metric": metric,
            "STEP5A_IID": iid.get(task),
            "STEP5B_composition_domain_OOD": dom_b.get(task),
            "STEP5C_LOEO_mean": float(v.mean()),
            "STEP5C_LOEO_median": float(v.median()),
            "STEP5C_LOEO_min": float(v.min()),
            "STEP5C_LOEO_max": float(v.max()),
            "STEP5C_LOEO_best_element": str(v.idxmax() if higher_is_better else v.idxmin()),
            "STEP5C_LOEO_worst_element": str(v.idxmin() if higher_is_better else v.idxmax()),
            "higher_is_better": higher_is_better,
        }
        for e in C.HELDOUT_ELEMENTS:
            rec[f"STEP5C_{e}"] = float(v[e]) if e in v.index else None
        if rec["STEP5A_IID"] is not None:
            if higher_is_better:
                rec["LOEO_retention_vs_IID"] = rec["STEP5C_LOEO_mean"] / rec["STEP5A_IID"]
            else:
                rec["LOEO_error_inflation_vs_IID"] = (rec["STEP5C_LOEO_mean"]
                                                      / rec["STEP5A_IID"]) \
                    if rec["STEP5A_IID"] else None
        if rec["STEP5B_composition_domain_OOD"] is not None:
            b = rec["STEP5B_composition_domain_OOD"]
            if higher_is_better:
                rec["LOEO_vs_composition_OOD_delta"] = rec["STEP5C_LOEO_mean"] - b
                rec["LOEO_harder_than_composition_OOD"] = bool(rec["STEP5C_LOEO_mean"] < b)
            else:
                rec["LOEO_vs_composition_OOD_delta"] = rec["STEP5C_LOEO_mean"] - b
                rec["LOEO_harder_than_composition_OOD"] = bool(rec["STEP5C_LOEO_mean"] > b)
        rows.append(rec)
        print("  %-13s IID %-10s  B-OOD %-10s  LOEO mean %-10s  [%s .. %s]"
              % (task,
                 ("%.6f" % rec["STEP5A_IID"]) if rec["STEP5A_IID"] is not None else "-",
                 ("%.6f" % rec["STEP5B_composition_domain_OOD"])
                 if rec["STEP5B_composition_domain_OOD"] is not None else "-",
                 "%.6f" % rec["STEP5C_LOEO_mean"],
                 "%.6f" % rec["STEP5C_LOEO_min"], "%.6f" % rec["STEP5C_LOEO_max"]))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.D_METRICS, "IID_VS_COMPOSITION_OOD_VS_LOEO.csv"),
               index=False, encoding="utf-8")

    # spec 61: never report the six-fold average alone
    per_el = loeo.pivot_table(index="heldout_element", columns="task", values="value")
    per_el = per_el.reindex(C.HELDOUT_ELEMENTS)
    per_el.to_csv(os.path.join(C.D_METRICS, "LOEO_PER_ELEMENT_HEADLINE.csv"),
                  encoding="utf-8")
    C.write_json(os.path.join(C.D_METRICS, "LOEO_COMPARISON_META.json"),
                 {"generated_utc": C.utcnow(),
                  "layers": {"STEP5A": "IID (Domain A, all six X elements seen)",
                             "STEP5B": "composition-domain OOD (Domain B shell)",
                             "STEP5C": "element-domain OOD (one X element held out)"},
                  "step5a_source": A_METRICS, "step5b_source": B_METRICS,
                  "reporting_rule": ("per-element values are reported alongside "
                                     "mean/median/min/max; the six-fold average is never "
                                     "reported alone (spec 61)")})
    print("\nSTAGE N: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
