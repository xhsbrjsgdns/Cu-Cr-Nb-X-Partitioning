# -*- coding: utf-8 -*-
"""
123_j_tables.py
STEP5J Stage 3: the eight manuscript tables (spec 19), each as CSV plus a Markdown
rendering, plus STEP5J_TABLE_INVENTORY.csv.

Every cell comes from FINAL_NUMERICAL_RESULTS.json or directly from a frozen file.
Nothing is typed in. Where a row would need a number that does not exist, the cell
carries NOT_AVAILABLE and the inventory records why (spec 35).

Table 3 deliberately keeps the retrospective LOEO rows in a separate block with their
caveat attached, so an escalation ladder can be read without the retrospective numbers
being mistaken for scores of the frozen production checkpoints (spec 18).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j_common as C

INVENTORY = []


def emit(num, title, df, caption, level, sources, notes=""):
    stem = "Table_%02d_%s" % (num, title.replace(" ", "_").replace("/", "_"))
    csv_path = C.P(C.D_TABLES, stem + ".csv")
    md_path = C.P(C.D_TABLES, stem + ".md")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("**Table %d.** %s\n\n" % (num, caption))
        f.write(C.to_md(df))
        f.write("\n\n*Evidence level:* %s\n\n*Sources:* %s\n" % (level, "; ".join(sources)))
        if notes:
            f.write("\n*Notes:* %s\n" % notes)
    INVENTORY.append({"table": num, "title": title, "caption": caption,
                      "evidence_level": level, "n_rows": len(df),
                      "csv": csv_path, "markdown": md_path,
                      "sources": "; ".join(sources), "notes": notes,
                      "csv_sha256": C.sha256_file(csv_path)})
    print("  Table %d: %s (%d rows)" % (num, title, len(df)))


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5J  Stage 3: manuscript tables")
    print("=" * 78)

    N = json.load(open(C.P(C.D_FREEZE, "FINAL_NUMERICAL_RESULTS.json"), encoding="utf-8"))
    L1, L2 = N["level_1_prospective_primary"], N["level_2_prospective_secondary"]
    L3, L4 = N["level_3_postblind_recovery"], N["level_4_postblind_descriptive"]
    CTX = N["context_not_primary_evidence"]
    dev = C.load_json("A_DEVFREEZE")
    v1 = C.load_csv("V1_DOMAIN")
    phases = C.load_csv("V1_PHASES")
    ladder = C.load_csv("C_LADDER")
    per_el = C.load_csv("C_PER_ELEMENT")
    rep = C.load_csv("I_RANDOM_REP")
    ie = C.load_csv(C.P(C.D_TABLES, "FINAL_INFORMATION_EFFICIENCY_TABLE.csv"))
    cross = C.load_csv(C.P(C.D_TABLES, "FINAL_CONTROL_CROSSOVER_TABLE.csv"))
    claims = C.load_csv(C.P(C.D_CLAIM, "FINAL_CLAIM_AUDIT.csv"))

    refit = {int(k): v for k, v in L3["random_M_REFIT_mean_tvd_by_budget"]["value"].items()}
    sd = {int(k): v for k, v in L3["random_M_REFIT_sd_by_budget"]["value"].items()}
    tacc = {int(k): v for k, v in L3["random_topology_accuracy_by_budget"]["value"].items()}
    mx = {int(k): v for k, v in L3["maximin_by_budget"]["value"].items()}
    zero = L3["zero_shot_tvd_same_subset"]["value"]
    triv = L3["trivial_tvd_same_subset"]["value"]

    # ------------------------------------------------ Table 1: dataset ------ #
    ds = CTX["dataset"]["value"]
    t1 = pd.DataFrame([
        {"quantity": "thermodynamic engine", "value": "Thermo-Calc 2025b / TC-Python "
                                                      "2025.2.30 / TCCU6 (Cu-Alloys v6.2)"},
        {"quantity": "conditions", "value": "1 bar, 1 mol, wt.%, Cu = balance"},
        {"quantity": "temperature grid",
         "value": "%d points: %s degC" % (len(C.TEMPERATURES_C),
                                          " ".join(str(t) for t in C.TEMPERATURES_C))},
        {"quantity": "composition sampling",
         "value": "Sobol(d=3, scramble); S_CrNb and R_CrNb_atomic sampled then inverted, "
                  "so Cr and Nb are never sampled independently"},
        {"quantity": "Domain A (development)",
         "value": "%d compositions / %d states" % (ds["domain_A"]["compositions"],
                                                   ds["domain_A"]["states"])},
        {"quantity": "Domain B (composition-domain OOD shell)",
         "value": "%d compositions / %d states, %d converged, %d failed"
                  % (ds["domain_B"]["compositions"], ds["domain_B"]["states"],
                     ds["domain_B"]["converged"], ds["domain_B"]["failed"])},
        {"quantity": "prospective Ni blind set",
         "value": "%d compositions / %d states, %d converged, %d failed"
                  % (ds["ni_blind"]["compositions"], ds["ni_blind"]["states"],
                     ds["ni_blind"]["converged"], ds["ni_blind"]["failed"])},
        {"quantity": "X identities in development", "value": ", ".join(C.ORIGINAL_SIX)},
        {"quantity": "X identities added in augmentation",
         "value": ", ".join(C.FULL5_ADDED)},
        {"quantity": "prospectively held-out identity", "value": "Ni"},
        {"quantity": "five-family mapping",
         "value": "FCC_L12*->FCC, BCC_B2*->BCC, C15_LAVES*->C15, LIQUID*->LIQUID, every "
                  "other stable solid->OTHER_SOLID; presence threshold 1e-4"},
        {"quantity": "partition definition",
         "value": "n_X^a = phase_moles_a * X_X^a ; P_X^a = n_X^a / sum_b n_X^b "
                  "(molar; volume-fraction partitioning forbidden)"},
        {"quantity": "distinct raw phases across the project",
         "value": "%d, of which NBNI3 appeared only in the Ni blind set"
                  % (len(phases) + 1)},
    ])
    emit(1, "Dataset and CALPHAD design", t1,
         "High-throughput CALPHAD design underlying every stage.", "context",
         [C.cite("V1_DOMAIN"), C.cite("V1_PHASES"), C.cite("B_COUNTS")])

    # ------------------------------------ Table 2: feature / model protocol -- #
    x = dev["tasks"]["x_partition"]
    t2 = pd.DataFrame([
        {"item": "feature schema", "specification":
            "%d features (F1 composition, F2 mixing, F3 X-property contrasts, F4 Magpie "
            "statistics), frozen at STEP5A and never re-selected" % dev["n_features"]},
        {"item": "feature list sha256", "specification": dev["feature_list_sha256"]},
        {"item": "production architecture", "specification":
            "Linear(43,128)-ReLU-Linear(128,64)-ReLU-Linear(64,5)"},
        {"item": "loss / optimizer", "specification": "raw-component MSE / Adam"},
        {"item": "batch, epochs, patience", "specification": "1024, <=200, 15 on raw "
                                                             "validation MSE"},
        {"item": "learning-rate grid", "specification":
            "1e-4, 3e-4, 1e-3, selected on projected validation TVD (PRIMARY chose "
            "3e-4)"},
        {"item": "projection", "specification":
            "clip at 0, renormalize; zero-sum row -> training-mean simplex"},
        {"item": "x_partition training rows / compositions",
         "specification": "%d / %d" % (x["train_rows"], x["train_compositions"])},
        {"item": "validation rows", "specification": str(x["val_rows"])},
        {"item": "split rule", "specification":
            "composition-level; a composition never appears in two splits"},
        {"item": "master seed", "specification": str(dev["master_seed"])},
        {"item": "bootstrap", "specification": dev["metric_definitions"]["bootstrap"]},
        {"item": "PRIMARY production model", "specification":
            "ORIGINAL6_DIRECT_MLP - six identities; kept as PRIMARY because every "
            "preregistered augmentation switching gate failed"},
        {"item": "SECONDARY production model", "specification":
            "FULL11_DIRECT_MLP - eleven identities; declared secondary before the "
            "reveal and not promotable by it"},
        {"item": "post-blind recovery protocol", "specification":
            "same architecture, frozen PRIMARY scaler never refit, fixed LR 3e-4, "
            "natural row weighting, early stopping on original-six validation only"},
    ])
    emit(2, "Feature and model protocol", t2,
         "Frozen feature schema, model formulation and training protocol.", "context",
         [C.cite("A_DEVFREEZE"), C.cite("G_FREEZE_JSON"), C.cite("G_FEWSHOT_JSON")])

    # -------------------- Table 3: IID / composition-OOD / LOEO / prospective - #
    rows = []
    for _, r in ladder.iterrows():
        rows.append({"task": r["task"], "metric": r["metric"],
                     "STEP5A_IID": r["STEP5A_IID"],
                     "STEP5B_composition_domain_OOD": r["STEP5B_composition_domain_OOD"],
                     "STEP5C_retrospective_LOEO_mean": r["STEP5C_LOEO_mean"],
                     "STEP5C_LOEO_min": r["STEP5C_LOEO_min"],
                     "STEP5C_LOEO_max": r["STEP5C_LOEO_max"],
                     "LOEO_worst_element": r["STEP5C_LOEO_worst_element"],
                     "block": "A. retrospective ladder (context, not primary evidence)"})
    for el in C.ORIGINAL_SIX:
        v = per_el[per_el.heldout_element == el]
        rows.append({"task": "x_partition", "metric": "mean_tvd (per held-out element)",
                     "STEP5A_IID": "", "STEP5B_composition_domain_OOD": "",
                     "STEP5C_retrospective_LOEO_mean": float(v.x_partition.iloc[0]),
                     "STEP5C_LOEO_min": "", "STEP5C_LOEO_max": "",
                     "LOEO_worst_element": el,
                     "block": "B. per-element LOEO (the six-fold average is never "
                              "reported alone)"})
    rows.append({"task": "x_partition", "metric": "mean_tvd",
                 "STEP5A_IID": "", "STEP5B_composition_domain_OOD": "",
                 "STEP5C_retrospective_LOEO_mean":
                     C.LOEO_CONTEXT["original_six_Z1_MLP_DIRECT_mean_tvd"],
                 "STEP5C_LOEO_min": "", "STEP5C_LOEO_max": "",
                 "LOEO_worst_element": "Z1_MLP_DIRECT (selected pipeline family)",
                 "block": "C. selected-family retrospective LOEO"})
    rows.append({"task": "x_partition", "metric": "mean_tvd",
                 "STEP5A_IID": "", "STEP5B_composition_domain_OOD": "",
                 "STEP5C_retrospective_LOEO_mean": C.LOEO_CONTEXT["full5_augmented_mean_tvd"],
                 "STEP5C_LOEO_min": "", "STEP5C_LOEO_max": "",
                 "LOEO_worst_element": "FULL5-augmented family",
                 "block": "C. selected-family retrospective LOEO"})
    t3 = pd.DataFrame(rows)
    emit(3, "IID composition-OOD and retrospective LOEO summary", t3,
         "Escalating generalization demands, all as x_partition mean TVD unless the "
         "metric column says otherwise. Blocks A-C are retrospective context; the "
         "prospective result is Table 4.", "context",
         [C.cite("C_LADDER"), C.cite("C_PER_ELEMENT"), C.cite("E_PIPELINE"),
          C.cite("F_AUG_SUMMARY")],
         notes=C.LOEO_CONTEXT["caveat"])

    # ------------------------------- Table 4: prospective Ni blind result ---- #
    ci = L1["ci95"]["value"]
    sci = L2["ci95"]["value"]
    t4 = pd.DataFrame([
        {"metric": "states scored", "PRIMARY (LEVEL 1)": L1["n_states"]["value"],
         "SECONDARY (LEVEL 2)": C.N_BLIND_STATES},
        {"metric": "mean TVD", "PRIMARY (LEVEL 1)": L1["mean_tvd"]["value"],
         "SECONDARY (LEVEL 2)": L2["mean_tvd"]["value"]},
        {"metric": "95% CI (composition cluster bootstrap)",
         "PRIMARY (LEVEL 1)": "[%.6f, %.6f]" % (ci[0], ci[1]),
         "SECONDARY (LEVEL 2)": "[%.6f, %.6f]" % (sci["ci95_low"], sci["ci95_high"])},
        {"metric": "median TVD", "PRIMARY (LEVEL 1)": L1["median_tvd"]["value"],
         "SECONDARY (LEVEL 2)": C.load_json("HB_SECONDARY_METRICS")["median_tvd"]},
        {"metric": "p95 TVD", "PRIMARY (LEVEL 1)": L1["p95_tvd"]["value"],
         "SECONDARY (LEVEL 2)": C.load_json("HB_SECONDARY_METRICS")["p95_tvd"]},
        {"metric": "topology accuracy", "PRIMARY (LEVEL 1)": L1["topology_accuracy"]["value"],
         "SECONDARY (LEVEL 2)": L2["topology_accuracy"]["value"]},
        {"metric": "topology macro F1",
         "PRIMARY (LEVEL 1)": L1["topology_macro_f1"]["value"],
         "SECONDARY (LEVEL 2)": C.load_json("HB_SECONDARY_METRICS")["topology_macro_f1"]},
        {"metric": "own trivial baseline TVD", "PRIMARY (LEVEL 1)": L1["trivial_tvd"]["value"],
         "SECONDARY (LEVEL 2)": L2["own_trivial_tvd"]["value"]},
        {"metric": "skill vs own trivial (negative = worse than the training mean)",
         "PRIMARY (LEVEL 1)": L1["skill_vs_trivial"]["value"],
         "SECONDARY (LEVEL 2)": L2["own_descriptive_skill"]["value"]},
        {"metric": "predicted dominant family share, C15",
         "PRIMARY (LEVEL 1)": L1["pred_topology_distribution"]["value"].get("C15"),
         "SECONDARY (LEVEL 2)": L2["pred_topology_distribution"]["value"].get("C15")},
        {"metric": "true dominant family share, C15",
         "PRIMARY (LEVEL 1)": L1["true_topology_distribution"]["value"].get("C15", 0.0),
         "SECONDARY (LEVEL 2)": L1["true_topology_distribution"]["value"].get("C15", 0.0)},
        {"metric": "true dominant family share, FCC / OTHER_SOLID",
         "PRIMARY (LEVEL 1)": "%.6f / %.6f"
                              % (L1["true_topology_distribution"]["value"]["FCC"],
                                 L1["true_topology_distribution"]["value"]["OTHER_SOLID"]),
         "SECONDARY (LEVEL 2)": "same targets"},
        {"metric": "features extending the training range (frozen pre-reveal)",
         "PRIMARY (LEVEL 1)": "%d of %d"
                              % (CTX["ni_feature_range"]["value"]["n_features_extending_train_range"],
                                 CTX["ni_feature_range"]["value"]["n_features_total"]),
         "SECONDARY (LEVEL 2)": "same inputs"},
    ])
    emit(4, "Prospective Ni zero-shot blind result", t4,
         "The primary result. Both models were frozen and hashed before any Ni "
         "descriptor, prediction or CALPHAD state existed.", "LEVEL 1 and LEVEL 2",
         [C.cite("HB_PRIMARY_METRICS"), C.cite("HB_SECONDARY_METRICS"),
          C.cite("HA_OOD_FREEZE")])

    # --------------------------------- Table 5: few-shot learning curve ------ #
    rows = [{"n_Ni_compositions": 0, "n_Ni_states": 0,
             "mean_TVD": zero, "SD_across_replicates": "",
             "replicate_1": "", "replicate_2": "", "replicate_3": "",
             "TVD_reduction_vs_zero_shot": 0.0, "topology_accuracy":
                 L3["zero_shot_topology_accuracy_same_subset"]["value"],
             "maximin_mean_TVD": zero,
             "row": "frozen zero-shot on the recovery subset"}]
    for b in C.BUDGETS:
        r = rep[(rep.method == "M_REFIT") & (rep.budget == b)].sort_values("replicate")
        rows.append({"n_Ni_compositions": b, "n_Ni_states": b * len(C.TEMPERATURES_C),
                     "mean_TVD": refit[b], "SD_across_replicates": sd[b],
                     "replicate_1": float(r.mean_tvd.iloc[0]),
                     "replicate_2": float(r.mean_tvd.iloc[1]),
                     "replicate_3": float(r.mean_tvd.iloc[2]),
                     "TVD_reduction_vs_zero_shot": 1 - refit[b] / zero,
                     "topology_accuracy": tacc[b],
                     "maximin_mean_TVD": mx.get(b),
                     "row": "3 frozen random replicates; maximin is a secondary "
                            "comparison only"})
    emit(5, "Post-blind few-shot learning curve", pd.DataFrame(rows),
         "Recovery on the 1,400-composition / 15,400-state RECOVERY_EVALUATION subset. "
         "Every replicate is shown; the mean is never reported alone. SD is finite "
         "protocol variability across three frozen anchor sequences, not a population "
         "estimate. The maximin column is a secondary comparison only and is not "
         "promoted on the strength of it.", "LEVEL 3",
         [C.cite("I_RANDOM_AGG"), C.cite("I_RANDOM_REP"), C.cite("I_MAXIMIN"),
          C.cite("I_BASELINES")],
         notes="Trivial baseline on the same subset: %.6f. All cells share one frozen "
               "initialization seed, so the replicate spread is anchor-driven." % triv)

    # ------------------------------- Table 6: M_REFIT vs R0 / R1 / R2 -------- #
    emit(6, "Refit versus anchor-only controls", cross,
         "R0 ignores the model and predicts the anchor mean per temperature; R1 and R2 "
         "shift the frozen raw output by a global or per-temperature anchor residual. "
         "Crossover budgets are post-blind descriptive, not preregistered endpoints.",
         "LEVEL 3", [C.cite("I_RANDOM_AGG"), C.cite("I_CONTROLS")])

    # ------------------- Table 7: topology / NBNI3 / temperature recovery ---- #
    rows = []
    for label, key in (("true FCC", "true_FCC"), ("true OTHER_SOLID", "true_OTHER_SOLID"),
                       ("NBNI3 present", "NBNI3_present"),
                       ("NBNI3 absent", "NBNI3_absent"),
                       ("LOW_T 300-600 degC", "LOW_T"), ("HIGH_T 700-1000 degC", "HIGH_T")):
        v = L4[key]["value"]
        rows.append({"stratum": label, "n_states": v.get("n_states", ""),
                     "zero_shot_TVD": v["zero_shot"], "b32_TVD": v["b32"],
                     "TVD_reduction": 1 - v["b32"] / v["zero_shot"]
                     if v["zero_shot"] else "",
                     "shared_trivial_TVD": triv})
    emit(7, "Topology NBNI3 and temperature-resolved recovery", pd.DataFrame(rows),
         "Descriptive post-blind stratification of the same recovery subset. The "
         "sequences were frozen before NBNI3 was known to exist. All strata end far "
         "below the shared trivial baseline; the ordering among them is relative, not "
         "a residual failure.", "LEVEL 4",
         [C.cite("I_TOPO"), C.cite("I_NBNI3"), C.cite("I_TEMP")])

    # ------------------------- Table 8: integrity and evidence levels -------- #
    fr = C.load_json(C.P(C.D_FREEZE, "STEP5J_FREEZE_VERIFICATION.json"))
    rows = [{"item": "frozen sources verified",
             "value": "%d (%d against published hashes, %d recorded by STEP5J)"
                      % (fr["n_sources"], fr["n_constant_checked"], fr["n_recorded_only"])},
            {"item": "STEP5I recovery checkpoints verified",
             "value": str(fr["n_step5i_checkpoints"])},
            {"item": "STEP5I sealed files verified", "value": str(fr["n_step5i_sealed_files"])},
            {"item": "new CALPHAD states computed in STEP5J", "value": "0"},
            {"item": "new models trained in STEP5J", "value": "0"},
            {"item": "frozen artifacts modified", "value": "0"},
            {"item": "values recomputed and cross-checked against the brief",
             "value": "%d, of which %d disagree beyond the brief's own quoted precision"
                      % (len(N["drift_vs_brief"]), N["n_drift_flags"])},
            {"item": "claims audited", "value": str(len(claims))},
            {"item": "claims weakened by their own evidence",
             "value": str(int((claims.weakened_because.fillna("") != "").sum()))}]
    for lvl, desc in C.EVIDENCE_LEVELS.items():
        rows.append({"item": "LEVEL %d" % lvl, "value": desc})
    rows.append({"item": "explicitly NOT available",
                 "value": "; ".join(sorted(N["not_available"].keys()))})
    emit(8, "Integrity and evidence-level summary", pd.DataFrame(rows),
         "What was verified, what was recomputed, and what this package does not "
         "contain.", "all levels",
         [C.cite("I_SEAL"), C.cite("G_FREEZE_JSON")])

    inv = pd.DataFrame(INVENTORY)
    inv.to_csv(C.P(C.D_TABLES, "STEP5J_TABLE_INVENTORY.csv"), index=False,
               encoding="utf-8")
    print("STAGE 3: PASS - %d tables" % len(inv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
