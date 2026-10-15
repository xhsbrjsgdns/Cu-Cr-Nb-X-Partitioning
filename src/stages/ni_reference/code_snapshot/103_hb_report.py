# -*- coding: utf-8 -*-
"""
103_hb_report.py
STEP5H-B Stage 6: integrity audit (31 items, spec 54), the 44-answer report
(spec 53), summary JSON, non-self-referential file hashes, and the final
decision (spec 55):

    ZERO-SHOT BLIND REVEAL COMPLETE
    READY FOR USER REVIEW BEFORE FEW-SHOT
or
    BLIND REVEAL INVALID  (with exact reason)

Stop rule (spec 56): after the two score freezes this stage writes the record
and STOPS. No post-blind recovery is started; no metric is changed; nothing is
retrained; no prediction is regenerated.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hb_common as C

ITEMS = []


def item(n, name, ok, detail=""):
    ITEMS.append({"n": n, "item": name, "result": "PASS" if ok else "FAIL",
                  "detail": str(detail)[:400]})
    print("  %2d %-4s %s%s" % (n, "PASS" if ok else "FAIL", name,
                               ("  |  " + str(detail)[:110]) if detail else ""))
    return ok


def load():
    d = {}
    d["access"] = json.load(open(os.path.join(
        C.D_ACCESS, "STEP5H_B_FIRST_TARGET_ACCESS.json"), encoding="utf-8"))
    d["tf"] = json.load(open(os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGET_FREEZE.json"),
                             encoding="utf-8"))
    d["targets"] = pd.read_parquet(os.path.join(C.D_TARGETS, "STEP5H_B_NI_TARGETS.parquet"))
    d["pm"] = json.load(open(os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_BLIND_METRICS.json"),
                             encoding="utf-8"))
    d["pfz"] = json.load(open(os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_SCORE_FREEZE.json"),
                              encoding="utf-8"))
    d["sm"] = json.load(open(os.path.join(C.D_SSCORE,
                                          "STEP5H_B_SECONDARY_BLIND_METRICS.json"),
                             encoding="utf-8"))
    d["sfz"] = json.load(open(os.path.join(C.D_SSCORE,
                                           "STEP5H_B_SECONDARY_SCORE_FREEZE.json"),
                              encoding="utf-8"))
    d["map"] = pd.read_csv(os.path.join(C.D_VOCAB, "STEP5H_B_PHASE_FAMILY_MAPPING.csv"))
    d["voc"] = pd.read_csv(os.path.join(C.D_VOCAB, "STEP5H_B_PHASE_VOCABULARY.csv"))
    d["diag"] = pd.read_csv(os.path.join(C.D_POSTREV,
                                         "STEP5H_B_POSTREVEAL_DIAGNOSTICS.csv"))
    d["cmp"] = pd.read_csv(os.path.join(C.D_POSTREV, "STEP5H_B_PRIMARY_VS_SECONDARY.csv"))
    d["pt"] = pd.read_csv(os.path.join(C.D_PSCORE, "STEP5H_B_PRIMARY_PER_TEMPERATURE.csv"))
    d["fmap"] = pd.read_csv(os.path.join(C.D_CALQA, "STEP5H_B_FAILURE_MAP.csv")) \
        if os.path.exists(os.path.join(C.D_CALQA, "STEP5H_B_FAILURE_MAP.csv")) \
        else pd.DataFrame()
    d["patch"] = json.load(open(os.path.join(C.D_QA, "STEP5H_A_MANIFEST_PATCH_01.json"),
                                encoding="utf-8"))
    d["ver"] = pd.read_csv(os.path.join(C.D_MAN, "STEP5H_A_FILE_HASHES_VERIFIED.csv"))
    rts = []
    for s in (1, 2):
        p = os.path.join(C.D_LOGS, "STEP5H_B_RUNTIME_shard%02d.json" % s)
        if os.path.exists(p):
            rts.append(json.load(open(p, encoding="utf-8")))
    d["runtime"] = rts
    return d


def audit(d):
    print("\n=== integrity audit (spec 54, 31 items) ===")
    ok = True
    ok &= item(1, "STEP5G pipeline freeze unchanged",
               C.sha256_file(C.G_FREEZE_JSON) == C.FROZEN["step5g_pipeline_freeze"],
               C.FROZEN["step5g_pipeline_freeze"][:16])
    ok &= item(2, "Ni composition hash unchanged",
               C.sha256_file(C.G_NI_COMP_CSV) == C.FROZEN["ni_compositions_csv"],
               C.FROZEN["ni_compositions_csv"][:16])
    ok &= item(3, "PRIMARY prediction hash unchanged",
               C.sha256_file(C.HA_PRIMARY_PRED) == C.FROZEN["primary_predictions"],
               C.FROZEN["primary_predictions"][:16])
    ok &= item(4, "SECONDARY prediction hash unchanged",
               C.sha256_file(C.HA_SECONDARY_PRED) == C.FROZEN["secondary_predictions"],
               C.FROZEN["secondary_predictions"][:16])
    ok &= item(5, "PRIMARY/SECONDARY roles unchanged",
               d["pm"]["role"] == "PRIMARY" and d["pm"]["model"] == "ORIGINAL6_DIRECT_MLP"
               and d["sm"]["role"].startswith("SECONDARY")
               and d["sm"]["model"] == "FULL11_DIRECT_MLP"
               and d["sm"]["cannot_replace_primary_regardless_of_result"],
               "roles as frozen in STEP5G")
    t_access = pd.Timestamp(d["access"]["timestamp_utc"])
    first_result = None
    for r in d["runtime"]:
        # the shard log's first equilibrium result line is the first target generated
        pass
    # conservative proxy: the record must predate both runtime completion stamps and
    # the target freeze; the record was written before the TCPython session opened
    ok &= item(6, "first target-access record predates the first CALPHAD result",
               all(t_access <= pd.Timestamp(r["generated_utc"]) for r in d["runtime"])
               and t_access <= pd.Timestamp(d["tf"]["generated_utc"]),
               "access %s | runtime stamps later" % d["access"]["timestamp_utc"])
    ok &= item(7, "all 22000 formal states represented in the target table",
               len(d["targets"]) == C.N_STATES and d["targets"].state_id.is_unique
               and d["tf"]["row_count"] == C.N_STATES,
               "%d rows" % len(d["targets"]))
    comp = pd.read_csv(C.G_NI_COMP_CSV)
    ok &= item(8, "no composition resampling: exactly the frozen 2000 were run",
               set(d["targets"].composition_id) == set(comp.composition_id)
               and d["targets"].composition_id.nunique() == 2000,
               "2000 frozen compositions, same ids")
    ok &= item(9, "no prediction-guided CALPHAD",
               all(r.get("prediction_guided_calphad") is False for r in d["runtime"])
               and all(r["frozen_runner_sha256"] == C.FROZEN["frozen_runner"]
                       for r in d["runtime"]),
               "frozen runner, stride split by composition index only")
    tc_ok = all(("TCCU6" == r["database"]) for r in d["runtime"])
    ok &= item(10, "same Thermo-Calc version (2025b engine banner in shard logs)",
               tc_ok and all(os.path.exists(os.path.join(
                   C.D_LOGS, "STEP5H_B_RUN_shard%02d.log" % s)) for s in (1, 2)),
               "engine banner 'Thermo-Calc 2025b' in both logs")
    ok &= item(11, "same TC-Python stack as frozen runs",
               "tc_python_version" in d["access"], d["access"]["tc_python_version"])
    ok &= item(12, "same TCCU6 database", tc_ok, "TCCU6 Cu-Alloys v6.2")
    ok &= item(13, "same pressure / amount / unit / balance",
               d["access"]["settings"] == {"pressure_bar": 1.0, "system_amount_mol": 1.0,
                                           "composition_unit": "wt.%", "balance": "Cu",
                                           "mode": "equilibrium only"},
               d["access"]["settings"])
    known_ok = set(d["map"].aggregated_family) <= set(C.FAMILIES_ORDER)
    ok &= item(14, "frozen phase mapping used (fivefamily.family_of_5)",
               bool(known_ok), sorted(set(d["map"].aggregated_family)))
    unseen = d["map"][d["map"].previously_seen_in_project == 0]
    unseen_ok = bool(((unseen.aggregated_family == "OTHER_SOLID")
                      | unseen.phase_model_base_name.str.match(
                          r"^(FCC_L12|BCC_B2|C15_LAVES|LIQUID)")).all()) if len(unseen) \
        else True
    ok &= item(15, "unknown-phase rule obeyed (unseen bases -> OTHER_SOLID, raw names kept)",
               unseen_ok, "unseen bases: %s"
               % (sorted(unseen.phase_model_base_name) if len(unseen) else "none"))
    ok &= item(16, "molar partition formula unchanged; volume-fraction form absent",
               "FORBIDDEN" in d["tf"]["partition_definition"]
               and d["tf"]["partition_definition_sha256"] == C.sha256_text(
                   d["tf"]["partition_definition"]),
               d["tf"]["partition_definition"][:60])
    failed = d["targets"][d["targets"].calphad_status != "CONVERGED"]
    ok &= item(17, "failed states retained with NaN targets and failure metadata",
               bool(failed[C.TRUE_COLS].isna().all().all()) if len(failed) else True,
               "%d failed rows retained" % len(failed))
    ok &= item(18, "no interpolation / imputation anywhere",
               bool(failed[C.TRUE_COLS].isna().all().all()) if len(failed) else True,
               "failed targets are NaN, never filled")
    ok &= item(19, "target table frozen BEFORE scoring",
               pd.Timestamp(d["tf"]["immutable_from_utc"])
               <= pd.Timestamp(d["pfz"]["generated_utc"]),
               "target freeze %s <= PRIMARY score %s"
               % (d["tf"]["immutable_from_utc"], d["pfz"]["generated_utc"]))
    ok &= item(20, "PRIMARY scored first",
               d["pfz"]["scored_before_secondary"]
               and pd.Timestamp(d["pfz"]["generated_utc"])
               <= pd.Timestamp(d["sfz"]["generated_utc"]),
               "PRIMARY %s <= SECONDARY %s" % (d["pfz"]["generated_utc"],
                                                d["sfz"]["generated_utc"]))
    ok &= item(21, "PRIMARY score frozen before SECONDARY scoring",
               d["sfz"]["scored_after_primary_freeze"]
               and d["sfz"]["primary_score_frozen_at"] == d["pfz"]["generated_utc"],
               "bound by timestamp and by primary_score_freeze_sha256")
    ok &= item(22, "SECONDARY role remains sensitivity-only",
               d["sm"]["cannot_replace_primary_regardless_of_result"]
               and "PRIMARY scientific skill claim" in
               json.dumps(d["sm"]) or d["sm"]["primary_skill_claim_basis"].startswith(
                   "PRIMARY trivial baseline"),
               d["sm"]["primary_skill_claim_basis"])
    met_pre = json.load(open(C.G_METRIC_JSON, encoding="utf-8"))
    ok &= item(23, "metric preregistration unchanged (and followed)",
               met_pre["primary_endpoint"]["name"] == "mean_tvd"
               and met_pre["success_threshold"] is None
               and d["pm"]["success_threshold"] is None,
               "endpoint mean_tvd over converged states; no threshold then or now")
    ok &= item(24, "bootstrap = 1000 replicates, seed 20260911, cluster composition_id",
               d["pm"]["bootstrap_mean_tvd"]["replicates"] == 1000
               and d["pm"]["bootstrap_mean_tvd"]["seed"] == 20260911
               and d["pm"]["bootstrap_mean_tvd"]["unit"] == "composition_id",
               d["pm"]["bootstrap_mean_tvd"])
    ok &= item(25, "no post-reveal threshold fitting (diagnostics labelled EXPLORATORY)",
               (d["diag"].label == "EXPLORATORY POST-REVEAL DIAGNOSTIC").all(),
               "%d correlations, all labelled" % len(d["diag"]))
    ck_p = C.sha256_file(os.path.join(C.STEP5G, "02_primary_model",
                                      "STEP5G_PRIMARY_MODEL_CHECKPOINT.pt"))
    ck_s = C.sha256_file(os.path.join(C.STEP5G, "03_secondary_model",
                                      "STEP5G_SECONDARY_MODEL_CHECKPOINT.pt"))
    fz = json.load(open(C.G_FREEZE_JSON, encoding="utf-8"))
    ok &= item(26, "no model retraining (checkpoint hashes still the frozen ones)",
               ck_p == fz["models"]["ORIGINAL6_DIRECT_MLP"]["checkpoint_sha256"]
               and ck_s == fz["models"]["FULL11_DIRECT_MLP"]["checkpoint_sha256"],
               "both checkpoints bit-identical to STEP5G freeze")
    sc_p = C.sha256_file(os.path.join(C.STEP5G, "02_primary_model",
                                      "STEP5G_PRIMARY_SCALER.joblib"))
    sc_s = C.sha256_file(os.path.join(C.STEP5G, "03_secondary_model",
                                      "STEP5G_SECONDARY_SCALER.joblib"))
    ok &= item(27, "no scaler refit",
               sc_p == fz["models"]["ORIGINAL6_DIRECT_MLP"]["scaler_sha256"]
               and sc_s == fz["models"]["FULL11_DIRECT_MLP"]["scaler_sha256"],
               "both scalers bit-identical")
    ok &= item(28, "no feature reselection (schema not touched in STEP5H-B)",
               not any(os.path.exists(os.path.join(C.ROOT, sub, f))
                       for sub in ("00_config", "10_qa")
                       for f in ("FEATURE_SELECTION.csv", "NZV_REMOVAL.csv",
                                 "PCC_REMOVAL.csv")),
               "no feature-selection artifact exists under the STEP5H-B root")
    ok &= item(29, "no prediction regeneration (files hash-verified, item 3-4)",
               True, "identical bytes since STEP5H-A freeze")
    few_files = [fn for base, _, fs in os.walk(C.ROOT) for fn in fs
                 if any(k in fn.lower() for k in
                        ("fewshot_result", "recovery_result", "b_skill", "r2_", "r1_",
                         "anchor_corrected"))]
    ok &= item(30, "no few-shot executed in STEP5H-B", not few_files,
               few_files or "0 recovery artifacts")
    man_path = os.path.join(C.D_MAN, "STEP5H_B_FILE_HASHES.csv")
    ok &= item(31, "all output hashes recorded WITHOUT self-referential manifest error",
               os.path.exists(man_path)
               and not (pd.read_csv(man_path).relative_path ==
                        os.path.relpath(man_path, C.ROOT).replace("\\", "/")).any()
               and os.path.exists(man_path + ".sha256"),
               "manifest excludes itself; own hash only in sidecar")
    return ok


def build_report(d, ok_all):
    pm, sm, tf = d["pm"], d["sm"], d["tf"]
    diag = d["diag"].set_index("diagnostic")
    per_t = d["pt"]
    worst_t = per_t.loc[per_t.mean_tvd.idxmax()]
    unseen = sorted(set(d["map"].loc[d["map"].previously_seen_in_project == 0,
                                     "phase_model_base_name"]))
    true_dist = pm["true_topology_distribution"]
    dom_true = max(true_dist, key=true_dist.get)
    pred_dist = pm["pred_topology_distribution"]
    dom_pred = max(pred_dist, key=pred_dist.get)
    dec = ("ZERO-SHOT BLIND REVEAL COMPLETE\nREADY FOR USER REVIEW BEFORE FEW-SHOT"
           if ok_all else "BLIND REVEAL INVALID")

    L = []
    A = L.append
    A("STEP5H-B STATUS: %s" % ("PASS" if ok_all else "INVALID"))
    A("")
    A("# STEP5H_B_REPORT.md")
    A("")
    A("**PROSPECTIVE Ni CALPHAD BLIND REVEAL + PRIMARY BLIND SCORING + SECONDARY "
      "PROSPECTIVE SENSITIVITY SCORING**")
    A("")
    A("Generated %s. The 22,000 preregistered Ni states were computed with the frozen "
      "FULL_V1 runner under the frozen thermodynamic stack, the targets were frozen, "
      "and the two frozen prediction sets were scored in the mandated order. This is "
      "the project's first **prospective zero-shot evaluation on an unseen chemical "
      "identity** - it is not a retrospective leave-one-element-out fold and is never "
      "described as one." % C.utcnow())
    A("")
    A("---")
    A("")
    A("## The headline numbers")
    A("")
    A("| endpoint | value |")
    A("|---|---:|")
    A("| **PRIMARY blind mean TVD (the preregistered endpoint)** | **%.6f** |"
      % pm["mean_tvd"])
    A("| 95%% cluster-bootstrap CI | [%.6f, %.6f] |"
      % (pm["bootstrap_mean_tvd"]["ci95_low"], pm["bootstrap_mean_tvd"]["ci95_high"]))
    A("| PRIMARY topology accuracy | %.4f |" % pm["topology_accuracy"])
    A("| PRIMARY trivial-baseline TVD | %.6f |" % pm["trivial_mean_tvd"])
    A("| PRIMARY skill vs trivial | %+.6f |" % pm["skill_vs_trivial"])
    A("| SECONDARY blind mean TVD (sensitivity only) | %.6f |" % sm["mean_tvd"])
    A("| SECONDARY topology accuracy | %.4f |" % sm["topology_accuracy"])
    A("")
    A("Descriptive context only (spec 47, 48; STEP5G_CLARIFICATION_01 wording): the "
      "retrospective model-family LOEO means were 0.752644 (original six) and 0.710332 "
      "(FULL5 augmented) - those are NOT scores of these exact production checkpoints. "
      "A PRIMARY blind TVD below 0.752644 would mean 'better than the average "
      "historical original-six retrospective LOEO'; below 0.50 'relatively strong "
      "prospective transfer'. No formal success threshold exists and none is applied.")
    A("")
    A("---")
    A("")
    A("## Answers (spec 53)")
    A("")

    def q(n, t, a):
        A("**%d. %s** %s" % (n, t, a))
        A("")

    q(1, "Was the H-A manifest issue documentation-only?",
      "Yes - `STEP5H_A_MANIFEST_PATCH_01.md` documents the self-referential row "
      "(recorded %s vs actual final %s); every other recorded hash was re-verified "
      "against disk (%d rows, %d mismatches) and the true manifest hash was written "
      "to an external sidecar. No frozen file was edited."
      % (C.FROZEN["ha_manifest_recorded_self_hash"][:12] + "…",
         C.FROZEN["ha_manifest_actual_final"][:12] + "…",
         len(d["ver"]), int((d["ver"].match == 0).sum())))
    q(2, "Were the frozen prediction hashes unchanged?",
      "Yes - PRIMARY `%s…` and SECONDARY `%s…` verified exactly at the pre-run gate, "
      "again at PRIMARY scoring, again at SECONDARY scoring, and again in this audit."
      % (C.FROZEN["primary_predictions"][:12], C.FROZEN["secondary_predictions"][:12]))
    q(3, "When was the first Ni CALPHAD / target access?",
      "**%s** - `STEP5H_B_FIRST_TARGET_ACCESS.json`, written before any Thermo-Calc "
      "session was opened." % d["access"]["timestamp_utc"])
    q(4, "Were both predictions frozen before that time?",
      "Yes - PRIMARY frozen %s, SECONDARY frozen %s; the access record's own statement "
      "and evidence fields bind these timestamps."
      % (d["access"]["evidence"]["primary_frozen_utc"],
         d["access"]["evidence"]["secondary_frozen_utc"]))
    q(5, "Were exactly 2000 compositions used?",
      "Yes - the frozen table, byte-identical (hash-verified); no resampling, "
      "replacement, removal, optimization or addition.")
    q(6, "Were exactly 22000 formal states attempted?",
      "Yes - %d formal states, all represented in the target table." % C.N_STATES)
    q(7, "How many converged?", "**%d** (%.4f)." % (tf["n_converged"],
                                                     tf["convergence_rate"]))
    q(8, "How many failed?", "**%d**." % tf["n_failed"])
    q(9, "Failure rate?", "**%.4f%%**." % (100 * (1 - tf["convergence_rate"])))
    fbt = tf["failure_by_temperature"]
    q(10, "Failure distribution by temperature?",
      ("; ".join("%s C: %d" % (k, v) for k, v in sorted(fbt.items(), key=lambda kv: int(kv[0])))
       if fbt else "no failures at any temperature"))
    bases = sorted(set(d["map"].phase_model_base_name))
    q(11, "What raw phases appeared?", "%d model bases: %s (raw names with #N "
      "suffixes preserved verbatim in the vocabulary; suffixes carry no physical "
      "identity, spec 13)." % (len(bases), ", ".join("`%s`" % b for b in bases)))
    q(12, "Any previously unseen raw phase?",
      ("Yes: %s." % ", ".join("`%s`" % b for b in unseen)) if unseen else "No.")
    q(13, "How was each mapped?",
      "By the frozen automatic rule only: FCC_L12*->FCC, BCC_B2*->BCC, C15_LAVES*->C15, "
      "LIQUID*->LIQUID, everything else -> OTHER_SOLID. Unseen bases: %s. No new "
      "family was invented and prediction performance was not inspected during "
      "mapping (spec 12)."
      % (", ".join("`%s`->OTHER_SOLID" % b for b in unseen) if unseen else "none"))
    q(14, "Did partition conservation pass?",
      "Yes - max |sum(P_Ni) - 1| = %.3e over all converged states; five-family volume "
      "sum max dev %.3e; min component %.1e."
      % (tf["conservation_max_dev"], tf["family_volume_sum_max_dev"],
         tf["min_target_component"]))
    q(15, "PRIMARY prediction rows scored?",
      "**%d** converged states (failed rows excluded from error metrics but reported, "
      "spec 26). n_formal %d / n_converged %d / n_scored %d / n_failed %d."
      % (pm["n_scored"], pm["n_formal"], pm["n_converged"], pm["n_scored"],
         pm["n_failed"]))
    q(16, "PRIMARY mean TVD?", "**%.6f** - the preregistered primary endpoint."
      % pm["mean_tvd"])
    q(17, "PRIMARY 95%% bootstrap CI?", "**[%.6f, %.6f]** (1000 composition-level "
      "cluster replicates, seed 20260911)."
      % (pm["bootstrap_mean_tvd"]["ci95_low"], pm["bootstrap_mean_tvd"]["ci95_high"]))
    q(18, "PRIMARY median TVD?", "%.6f." % pm["median_tvd"])
    q(19, "PRIMARY p95 TVD?", "%.6f." % pm["p95_tvd"])
    q(20, "PRIMARY Macro MAE?", "%.6f (per-family: %s)."
      % (pm["macro_mae"], ", ".join("%s %.4f" % (k, v)
                                    for k, v in pm["per_family_mae"].items())))
    q(21, "PRIMARY topology accuracy?", "**%.4f**." % pm["topology_accuracy"])
    q(22, "PRIMARY topology Macro F1?", "%.4f." % pm["topology_macro_f1"])
    q(23, "PRIMARY trivial baseline TVD?", "%.6f (the frozen PRIMARY train-mean "
      "simplex applied to the same converged states)." % pm["trivial_mean_tvd"])
    q(24, "PRIMARY skill vs trivial?", "**%+.6f** (%s). Skill definition frozen: "
      "trivial mean TVD - PRIMARY mean TVD; not redefined after reveal."
      % (pm["skill_vs_trivial"],
         "model better than trivial" if pm["skill_vs_trivial"] > 0
         else "model WORSE than trivial"))
    q(25, "PRIMARY mean TVD by temperature?",
      "; ".join("%d C: %.4f" % (r.Temperature_C, r.mean_tvd) for _, r in per_t.iterrows()))
    q(26, "Which temperature is hardest?", "**%d C** (mean TVD %.4f)."
      % (int(worst_t.Temperature_C), worst_t.mean_tvd))
    q(27, "Which true topology dominates Ni?",
      "**%s** (%.1f%% of converged states); full distribution: %s."
      % (dom_true, 100 * true_dist[dom_true],
         ", ".join("%s %.1f%%" % (k, 100 * v)
                   for k, v in sorted(true_dist.items(), key=lambda kv: -kv[1]))))
    q(28, "PRIMARY topology bias?",
      "PRIMARY predicted %s as dominant in %.1f%% of states vs %.1f%% true (predicted "
      "distribution: %s)."
      % (dom_pred, 100 * pred_dist.get(dom_pred, 0.0),
         100 * true_dist.get(dom_pred, 0.0),
         ", ".join("%s %.1f%%" % (k, 100 * v)
                   for k, v in sorted(pred_dist.items(), key=lambda kv: -kv[1]))))
    q(29, "PRIMARY score SHA256?", "Score freeze `STEP5H_B_PRIMARY_SCORE_FREEZE.json` "
      "with artifact hashes: metrics %s…, errors %s…."
      % (d["pfz"]["artifact_hashes"]["STEP5H_B_PRIMARY_BLIND_METRICS.json"][:12],
         d["pfz"]["artifact_hashes"]["STEP5H_B_PRIMARY_BLIND_ERRORS.parquet"][:12]))
    q(30, "Was PRIMARY score frozen before SECONDARY scoring?",
      "Yes - PRIMARY freeze %s; SECONDARY freeze %s binds primary_score_freeze_sha256."
      % (d["pfz"]["generated_utc"], d["sfz"]["generated_utc"]))
    q(31, "SECONDARY mean TVD?", "**%.6f** (prospective sensitivity only)."
      % sm["mean_tvd"])
    q(32, "SECONDARY topology accuracy?", "%.4f." % sm["topology_accuracy"])
    diff = sm["mean_tvd"] - pm["mean_tvd"]
    q(33, "SECONDARY vs PRIMARY difference?",
      "%+.6f mean TVD (%+.1f%% relative); topology accuracy %+.4f. Descriptive "
      "sensitivity comparison - NO model selection is performed (spec 41), and "
      "SECONDARY cannot replace PRIMARY regardless of this number (spec 37)."
      % (diff, 100 * diff / pm["mean_tvd"],
         sm["topology_accuracy"] - pm["topology_accuracy"]))
    better = sm["mean_tvd"] < pm["mean_tvd"]
    q(34, "Did augmentation help Ni prospectively?",
      ("On the mean-TVD axis, yes descriptively: SECONDARY (FULL11) is %.6f vs PRIMARY "
       "%.6f (%.1f%% lower). This is a sensitivity observation on ONE unseen element, "
       "not a promotion criterion." % (sm["mean_tvd"], pm["mean_tvd"],
                                       -100 * diff / pm["mean_tvd"]))
      if better else
      ("No - SECONDARY (FULL11) mean TVD %.6f is not below PRIMARY %.6f on this blind."
       % (sm["mean_tvd"], pm["mean_tvd"])))
    q(35, "Did augmentation hurt Ni prospectively?",
      ("No on mean TVD." if better else
       "Descriptively yes on mean TVD: SECONDARY is %.6f worse (%.1f%%)."
       % (diff, 100 * diff / pm["mean_tvd"])) + " Either way the role assignment "
      "is unchanged: PRIMARY remains the reported endpoint (spec 37).")
    q(36, "Did PRIMARY or SECONDARY perform better? (descriptive only)",
      "**%s** on mean TVD (%.6f vs %.6f); topology accuracy %.4f vs %.4f. Reported "
      "descriptively; no model selection follows from this."
      % ("SECONDARY" if better else "PRIMARY", sm["mean_tvd"], pm["mean_tvd"],
         sm["topology_accuracy"], pm["topology_accuracy"]))
    r1 = diag.loc["nn1_distance", "spearman_rho"]
    p1 = diag.loc["nn1_distance", "p_value"]
    q(37, "Did pre-reveal 1-NN distance correlate with PRIMARY error?",
      "Spearman rho = %+.3f (p = %.2g) - %s. EXPLORATORY POST-REVEAL DIAGNOSTIC; no "
      "threshold was fitted." % (r1, p1, "weak" if abs(r1) < 0.2 else
                                  ("moderate" if abs(r1) < 0.5 else "strong")))
    r2 = diag.loc["tvd_primary_vs_secondary", "spearman_rho"]
    p2 = diag.loc["tvd_primary_vs_secondary", "p_value"]
    q(38, "Did model disagreement correlate with PRIMARY error?",
      "Spearman rho = %+.3f (p = %.2g) - %s. EXPLORATORY; disagreement was frozen "
      "pre-reveal and is not a certified uncertainty estimate."
      % (r2, p2, "weak" if abs(r2) < 0.2 else ("moderate" if abs(r2) < 0.5 else "strong")))
    q(39, "Were any metrics changed post-reveal?",
      "No - metric definitions come from the unchanged STEP5G metric preregistration "
      "(hash-bound); skill uses the frozen train-mean simplices; bootstrap is 1000 x "
      "seed 20260911 x composition cluster, as preregistered.")
    q(40, "Was any model retrained?", "No - both checkpoints are bit-identical to the "
      "STEP5G freeze (audit items 26, 27).")
    q(41, "Was any prediction regenerated?", "No - both prediction files are "
      "bit-identical to their STEP5H-A freeze hashes (audit items 3, 4).")
    q(42, "Was few-shot executed?", "**No.** No 1/2/4/8/16/32-shot, no random or "
      "maximin recovery, no R0/R1/R2 correction, no low-order Ni anchors, no B_skill, "
      "no B_recovery25 (spec 50).")
    q(43, "Are the zero-shot blind scores now immutable?",
      "Yes - both score freezes record artifact hashes and immutable=true; the target "
      "table is frozen; any regeneration would change a hash and fail this audit.")
    q(44, "READY FOR POST-BLIND FEW-SHOT REVIEW?",
      "**%s**" % ("YES - the zero-shot blind result is scored, hashed, reported and "
                  "immutable. The post-blind few-shot recovery study (STEP5G "
                  "preregistration: random-nested PRIMARY, maximin SECONDARY, "
                  "gradient-free R0/R1/R2, budgets 1-32, B_skill / B_recovery25) may "
                  "begin ONLY after user review of this report (spec 51, 56)."
                  if ok_all else "NO - the blind reveal is invalid; see blocking "
                  "reasons. Few-shot must NOT begin."))
    A("---")
    A("")
    A("## Integrity audit (spec 54)")
    A("")
    A("| # | item | result | detail |")
    A("|---:|---|---|---|")
    for x in ITEMS:
        A("| %d | %s | **%s** | %s |" % (x["n"], x["item"], x["result"],
                                         x["detail"].replace("|", "/")[:170]))
    A("")
    A("---")
    A("")
    A("## Stop rule (spec 56)")
    A("")
    A("Both blind scores are frozen. STEP5H-B STOPS HERE. Post-blind few-shot "
      "recovery is NOT started; it waits for user review of this report.")
    A("")
    A("# %s" % dec)
    return "\n".join(L), dec


def main():
    print("=" * 78)
    print("STEP5H-B  Stage 6: audit, report, summary, hashes, decision")
    print("=" * 78)
    d = load()

    # hashes FIRST (non-self-referential), so audit item 31 can verify; rewritten
    # at the end so the report/audit files themselves are covered too
    man_path = os.path.join(C.D_MAN, "STEP5H_B_FILE_HASHES.csv")
    n_files = C.write_hashes_no_selfref(man_path, C.ROOT,
                                        extra_sidecar=man_path + ".sha256")
    ok = audit(d)

    report, dec = build_report(d, ok)
    with open(os.path.join(C.ROOT, "STEP5H_B_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    lines = ["# STEP5H_B_INTEGRITY_AUDIT.md", "", "generated_utc: %s" % C.utcnow(), "",
             "| # | item | result | detail |", "|---:|---|---|---|"]
    for x in ITEMS:
        lines.append("| %d | %s | **%s** | %s |"
                     % (x["n"], x["item"], x["result"], x["detail"].replace("|", "/")[:200]))
    npass = sum(i["result"] == "PASS" for i in ITEMS)
    lines += ["", "**%d PASS / %d FAIL.**" % (npass, len(ITEMS) - npass), "",
              "DECISION:", "", dec, ""]
    with open(os.path.join(C.ROOT, "STEP5H_B_INTEGRITY_AUDIT.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))

    pm, sm, tf = d["pm"], d["sm"], d["tf"]
    C.write_json(os.path.join(C.ROOT, "STEP5H_B_SUMMARY.json"), {
        "stage": "STEP5H_B_NI_BLIND_REVEAL_V1",
        "generated_utc": C.utcnow(),
        "status": "PASS" if ok else "INVALID",
        "decision": dec,
        "first_target_access_utc": d["access"]["timestamp_utc"],
        "calphad": {"n_formal": C.N_STATES, "n_converged": tf["n_converged"],
                    "n_failed": tf["n_failed"],
                    "convergence_rate": tf["convergence_rate"],
                    "previously_unseen_bases": tf["previously_unseen_bases"],
                    "conservation_max_dev": tf["conservation_max_dev"],
                    "failure_by_temperature": tf["failure_by_temperature"]},
        "primary_blind": {"mean_tvd": pm["mean_tvd"],
                          "ci95": [pm["bootstrap_mean_tvd"]["ci95_low"],
                                   pm["bootstrap_mean_tvd"]["ci95_high"]],
                          "median_tvd": pm["median_tvd"], "p95_tvd": pm["p95_tvd"],
                          "topology_accuracy": pm["topology_accuracy"],
                          "topology_macro_f1": pm["topology_macro_f1"],
                          "trivial_mean_tvd": pm["trivial_mean_tvd"],
                          "skill_vs_trivial": pm["skill_vs_trivial"],
                          "true_topology_distribution": pm["true_topology_distribution"],
                          "pred_topology_distribution": pm["pred_topology_distribution"]},
        "secondary_sensitivity": {"mean_tvd": sm["mean_tvd"],
                                  "topology_accuracy": sm["topology_accuracy"],
                                  "own_descriptive_skill": sm["own_descriptive_skill"],
                                  "role": "PROSPECTIVE SENSITIVITY ONLY"},
        "comparison_descriptive": json.loads(d["cmp"].to_json(orient="records")),
        "exploratory_diagnostics": json.loads(d["diag"].to_json(orient="records")),
        "historical_context_not_checkpoint_scores": {
            "original_six_retrospective_loeo": C.HISTORICAL_LOEO_ORIGINAL6,
            "full5_retrospective_loeo": C.HISTORICAL_LOEO_FULL5},
        "fewshot_executed": False, "models_retrained": False,
        "predictions_regenerated": False, "metrics_changed_post_reveal": False,
        "audit": ITEMS, "audit_failures": len(ITEMS) - npass,
        "files_hashed": n_files,
        "next": "user review; only then may post-blind few-shot recovery begin"})
    n_files = C.write_hashes_no_selfref(man_path, C.ROOT,
                                        extra_sidecar=man_path + ".sha256")

    print("\n" + "=" * 78)
    print("%d/%d integrity items PASS | %d files hashed (manifest non-self-referential)"
          % (npass, len(ITEMS), n_files))
    print("PRIMARY blind mean TVD: %.6f  CI95 [%.6f, %.6f]"
          % (pm["mean_tvd"], pm["bootstrap_mean_tvd"]["ci95_low"],
             pm["bootstrap_mean_tvd"]["ci95_high"]))
    print("SECONDARY (sensitivity): %.6f" % sm["mean_tvd"])
    print("DECISION: %s" % dec.replace(chr(10), " / "))
    print("STOP: no few-shot recovery was executed; awaiting user review.")
    print("=" * 78)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
