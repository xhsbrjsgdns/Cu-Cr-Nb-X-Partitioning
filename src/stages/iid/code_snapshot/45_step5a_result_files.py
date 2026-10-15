# -*- coding: utf-8 -*-
"""
45_step5a_result_files.py
STEP5A: derive the per-task result files required by sec.91 from the already
frozen artifacts (pure reshaping; NO recomputation, NO new model contact):

  05_results/PHASE_PRESENCE_METRICS.csv
  05_results/PHASE_FRACTION_METRICS.csv
  05_results/CR_PARTITION_METRICS.csv
  05_results/NB_PARTITION_METRICS.csv
  05_results/X_PARTITION_METRICS.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5a_common as C

TASK_FILE = {"fraction": "PHASE_FRACTION_METRICS.csv",
             "cr_partition": "CR_PARTITION_METRICS.csv",
             "nb_partition": "NB_PARTITION_METRICS.csv",
             "x_partition": "X_PARTITION_METRICS.csv"}


def main():
    per = pd.read_csv(os.path.join(C.RESULTS, "BASELINE_PER_TARGET_METRICS.csv"))
    head = pd.read_csv(os.path.join(C.RESULTS, "BASELINE_IID_TEST_METRICS.csv"))
    val = pd.read_csv(os.path.join(C.RESULTS, "BASELINE_VALIDATION_METRICS.csv"))
    boot = pd.read_csv(os.path.join(C.RESULTS, "BASELINE_BOOTSTRAP_CI.csv"))

    # presence
    pres_head = pd.concat([val[val["task"] == "presence"], head[head["task"] == "presence"]],
                          ignore_index=True)
    pres_per = per[per["task"] == "presence"]
    out = pd.concat([
        pres_head.assign(granularity="task_level"),
        pres_per.assign(granularity="per_label"),
        boot[boot["metric"].isin(["presence_exact_match", "presence_macro_f1",
                                  "presence_micro_f1", "liquid_f1", "liquid_pr_auc"])]
        .assign(granularity="bootstrap_ci", task="presence"),
    ], ignore_index=True)
    out.to_csv(os.path.join(C.RESULTS, "PHASE_PRESENCE_METRICS.csv"), index=False)

    for task, fname in TASK_FILE.items():
        t_head = pd.concat([val[val["task"] == task], head[head["task"] == task]],
                           ignore_index=True).assign(granularity="task_level")
        t_per = per[per["task"] == task].assign(granularity="per_family")
        ci_map = {"fraction": "fraction_mean_tvd", "cr_partition": "cr_mean_tvd",
                  "nb_partition": "nb_mean_tvd", "x_partition": "x_mean_tvd"}[task]
        t_ci = boot[boot["metric"] == ci_map].assign(granularity="bootstrap_ci", task=task)
        pd.concat([t_head, t_per, t_ci], ignore_index=True).to_csv(
            os.path.join(C.RESULTS, fname), index=False)
    print("sec.91 per-task result files written:")
    for f in ["PHASE_PRESENCE_METRICS.csv", *TASK_FILE.values()]:
        print(" ", f)


if __name__ == "__main__":
    main()
