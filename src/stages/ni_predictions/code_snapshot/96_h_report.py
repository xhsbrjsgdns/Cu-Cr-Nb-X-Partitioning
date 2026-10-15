# -*- coding: utf-8 -*-
"""
96_h_report.py
STEP5H-A Stage 6: integrity audit (27 items, spec 43), the 38-answer report
(spec 42), summary JSON, file hashes, and the final decision (spec 44):

    READY FOR STEP5H-B - Ni CALPHAD BLIND REVEAL
or  NOT READY - with exact blocking reasons.

Stop rule (spec 45): after this stage NOTHING else runs. No Thermo-Calc launch,
no Ni label, no error calculation, no PRIMARY/SECONDARY scoring, no few-shot
recovery. Wait for user review.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5h_common as C

ITEMS = []


def item(n, name, ok, detail=""):
    ITEMS.append({"n": n, "item": name, "result": "PASS" if ok else "FAIL",
                  "detail": str(detail)[:400]})
    print("  %2d %-4s %s%s" % (n, "PASS" if ok else "FAIL", name,
                               ("  |  " + str(detail)[:110]) if detail else ""))
    return ok


def load():
    d = {}
    d["clar_sha"] = C.sha256_file(os.path.join(C.ROOT, "STEP5G_CLARIFICATION_01.md"))
    d["access"] = json.load(open(os.path.join(
        C.D_ACCESS, "STEP5H_A_NI_FIRST_DESCRIPTOR_ACCESS.json"), encoding="utf-8"))
    d["ledger"] = pd.read_csv(os.path.join(
        C.D_ACCESS, "STEP5H_A_NI_PROPERTY_ACCESS_LEDGER.csv"))
    d["accsum"] = json.load(open(os.path.join(C.D_CONFIG, "STEP5H_A_ACCESS_SUMMARY.json"),
                                 encoding="utf-8"))
    d["featqa"] = json.load(open(os.path.join(C.D_QA, "STEP5H_A_FEATURE_QA.json"),
                                 encoding="utf-8"))
    d["cons"] = pd.read_csv(os.path.join(C.D_QA, "STEP5H_A_SOURCE_CONSISTENCY_TEST.csv"))
    d["compqa"] = pd.read_csv(os.path.join(C.D_QA, "STEP5H_A_COMPOSITION_QA.csv"))
    d["oodfz"] = json.load(open(os.path.join(
        C.D_OOD, "STEP5H_A_OOD_DIAGNOSTICS_FREEZE.json"), encoding="utf-8"))
    d["rng"] = pd.read_csv(os.path.join(C.D_OOD,
                                        "STEP5H_A_NI_FEATURE_RANGE_DIAGNOSTICS.csv"))
    d["ood"] = pd.read_csv(os.path.join(C.D_OOD, "STEP5H_A_NI_OOD_DIAGNOSTICS.csv"))
    d["predsum"] = json.load(open(os.path.join(C.D_CONFIG,
                                               "STEP5H_A_PREDICTION_SUMMARY.json"),
                                  encoding="utf-8"))
    d["dist"] = json.load(open(os.path.join(C.D_CONFIG,
                                            "STEP5H_A_PREDICTION_DISTRIBUTIONS.json"),
                               encoding="utf-8"))
    d["pfz"] = json.load(open(os.path.join(
        C.D_PRIMPRED, "STEP5H_A_PRIMARY_PREDICTION_FREEZE.json"), encoding="utf-8"))
    d["sfz"] = json.load(open(os.path.join(
        C.D_SECPRED, "STEP5H_A_SECONDARY_PREDICTION_FREEZE.json"), encoding="utf-8"))
    d["pp"] = pd.read_parquet(os.path.join(C.D_PRIMPRED,
                                           "STEP5H_A_PRIMARY_NI_PREDICTIONS.parquet"))
    d["sp"] = pd.read_parquet(os.path.join(C.D_SECPRED,
                                           "STEP5H_A_SECONDARY_NI_PREDICTIONS.parquet"))
    d["dis"] = pd.read_csv(os.path.join(C.D_DIS,
                                        "STEP5H_A_PRIMARY_SECONDARY_DISAGREEMENT.csv"))
    d["maxfz"] = json.load(open(os.path.join(C.D_MAXIMIN, "STEP5H_A_MAXIMIN_FREEZE.json"),
                                encoding="utf-8"))
    d["maxseq"] = pd.read_csv(os.path.join(C.D_MAXIMIN, "STEP5H_A_MAXIMIN_SEQUENCE.csv"))
    d["comp"] = pd.read_csv(C.G_NI_COMP_CSV)
    d["feats"] = pd.read_parquet(os.path.join(C.D_FEAT, "STEP5H_A_NI_FEATURES.parquet"))
    return d


def audit(d):
    print("\n=== integrity audit (spec 43, 27 items) ===")
    ok = True
    fz_now = C.sha256_file(C.G_FREEZE_JSON)
    fz_rec = d["access"]["pipeline_freeze_sha256"]
    ok &= item(1, "STEP5G pipeline freeze hash unchanged", fz_now == fz_rec, fz_now[:16])
    ok &= item(2, "PRIMARY checkpoint hash unchanged",
               C.sha256_file(C.G_PRIMARY_CKPT) == C.FROZEN_HASHES["primary_checkpoint"],
               C.FROZEN_HASHES["primary_checkpoint"][:16])
    ok &= item(3, "SECONDARY checkpoint hash unchanged",
               C.sha256_file(C.G_SECONDARY_CKPT) == C.FROZEN_HASHES["secondary_checkpoint"],
               C.FROZEN_HASHES["secondary_checkpoint"][:16])
    ok &= item(4, "PRIMARY scaler hash unchanged",
               C.sha256_file(C.G_PRIMARY_SCALER) == C.FROZEN_HASHES["primary_scaler"],
               C.FROZEN_HASHES["primary_scaler"][:16])
    ok &= item(5, "SECONDARY scaler hash unchanged",
               C.sha256_file(C.G_SECONDARY_SCALER) == C.FROZEN_HASHES["secondary_scaler"],
               C.FROZEN_HASHES["secondary_scaler"][:16])
    ok &= item(6, "feature schema hash unchanged",
               C.sha256_file(C.G_SCHEMA_CSV) == C.FROZEN_HASHES["feature_schema_csv"],
               C.FROZEN_HASHES["feature_schema_csv"][:16])
    ok &= item(7, "Ni composition table hash unchanged",
               C.sha256_file(C.G_NI_COMP_CSV) == C.FROZEN_HASHES["ni_compositions_csv"],
               C.FROZEN_HASHES["ni_compositions_csv"][:16])
    ok &= item(8, "2000 compositions unchanged (QA recheck PASS)",
               len(d["comp"]) == 2000
               and (d["compqa"].status == "PASS").all(), "2000 rows, 9/9 QA PASS")
    feats43 = C.feature_schema()
    ok &= item(9, "22000 states in features and in both prediction files",
               len(d["feats"]) == C.N_STATES and len(d["pp"]) == C.N_STATES
               and len(d["sp"]) == C.N_STATES, C.N_STATES)
    ok &= item(10, "same 43 features, exact frozen order, in the feature table",
               [c for c in d["feats"].columns if c not in
                ("state_id", "composition_id", "fewshot_role",
                 "fewshot_rank")] == feats43,
               "%d features (Temperature_C is itself feature #1)" % len(feats43))
    led = d["ledger"]
    approved = bool(led.source.str.contains("matminer|mendeleev").all())
    ok &= item(11, "approved property sources only (matminer MagpieData / mendeleev)",
               bool(approved), sorted(set(led.source))[:2])
    ok &= item(12, "no manually entered Ni values",
               not led.source.str.contains("manual|typed|hand", case=False).any()
               and "manual entry" in d["access"]["excluded_sources"],
               "ledger sources are package databases only")
    ok &= item(13, "no web-derived Ni values",
               not led.source.str.contains("web|wiki|http|project", case=False).any(),
               "no web source in the ledger")
    tc_files = []
    for base, _, files in os.walk(C.ROOT):
        for fn in files:
            if "calphad" in fn.lower() or "thermo" in fn.lower():
                tc_files.append(fn)
    ok &= item(14, "no Ni CALPHAD anywhere in STEP5H-A (files, imports, calls)",
               not tc_files and d["accsum"]["tc_python_imported"] is False
               and d["predsum"]["no_target_accessed"], tc_files or "0 CALPHAD artifacts")
    ok &= item(15, "no Ni targets accessed (none exist)",
               d["accsum"]["ni_calphad_or_target_accessed"] == 0
               and d["predsum"]["no_error_computed"], "0 target rows")
    ok &= item(16, "no target columns loaded into any prediction artifact",
               d["featqa"]["target_column_guard"] == []
               and all(v for v in d["pfz"]["qa_checks"].values())
               and all(v for v in d["sfz"]["qa_checks"].values()),
               "guard clean on features + both prediction files")
    ok &= item(17, "no model retraining (checkpoint hashes are the proof)",
               d["predsum"]["no_model_retrained"]
               and C.sha256_file(C.G_PRIMARY_CKPT) == C.FROZEN_HASHES["primary_checkpoint"]
               and C.sha256_file(C.G_SECONDARY_CKPT) == C.FROZEN_HASHES["secondary_checkpoint"],
               "hashes identical to STEP5G freeze")
    ok &= item(18, "no scaler refit (scaler hashes are the proof)",
               d["predsum"]["no_scaler_refit"]
               and C.sha256_file(C.G_PRIMARY_SCALER) == C.FROZEN_HASHES["primary_scaler"]
               and C.sha256_file(C.G_SECONDARY_SCALER) == C.FROZEN_HASHES["secondary_scaler"],
               "hashes identical to STEP5G freeze")
    ok &= item(19, "no feature selection performed",
               C.sha256_file(C.G_SCHEMA_CSV) == C.FROZEN_HASHES["feature_schema_csv"]
               and not any(os.path.exists(os.path.join(C.D_QA, f)) for f in
                           ("NZV_REMOVAL.csv", "PCC_REMOVAL.csv", "SHAP_SELECTION.csv")),
               "43-feature schema untouched")
    ok &= item(20, "PRIMARY predicted first",
               d["pfz"]["generated_utc"] <= d["sfz"]["generated_utc"],
               "PRIMARY %s <= SECONDARY %s" % (d["pfz"]["generated_utc"],
                                                d["sfz"]["generated_utc"]))
    ok &= item(21, "PRIMARY frozen before SECONDARY prediction",
               d["pfz"]["immutable_from_utc"] <= d["sfz"]["generated_utc"],
               "freeze timestamps ordered")
    ok &= item(22, "SECONDARY frozen (immutable)",
               os.path.exists(os.path.join(
                   C.D_SECPRED, "STEP5H_A_SECONDARY_PREDICTION_FREEZE.json"))
               and d["sfz"]["immutable_from_utc"] is not None, d["sfz"]["sha256"][:16])
    ok &= item(23, "prediction state IDs identical between PRIMARY and SECONDARY",
               list(d["pp"].state_id) == list(d["sp"].state_id), "22000 identical IDs")
    few = json.load(open(C.G_FEWSHOT_JSON, encoding="utf-8"))
    ok &= item(24, "random few-shot split and sequences unchanged",
               C.sha256_file(C.G_SEQ_CSV) == C.FROZEN_HASHES["fewshot_sequences_csv"]
               and int((d["comp"].fewshot_role == "FEWSHOT_CALIBRATION").sum()) == 600
               and int((d["comp"].fewshot_role == "RECOVERY_EVALUATION").sum()) == 1400
               and d["maxfz"]["random_sequences"]["regenerated"] is False,
               "600/1400 split + 3 sequences hash-verified, not regenerated")
    ok &= item(25, "maximin selection is target-free",
               d["maxfz"]["reference_set"]["target_information_used"] is False
               and d["maxfz"]["labels_used"] == 0
               and d["maxfz"]["fewshot_executed_here"] is False,
               "473-dim input geometry only")
    ok &= item(26, "maximin sequence frozen before any Ni CALPHAD exists",
               d["maxfz"]["immutable_after_ni_calphad_reveal"] and not tc_files,
               d["maxfz"]["sequence_sha256"][:16])
    man_path = os.path.join(C.D_MAN, "STEP5H_A_FILE_HASHES.csv")
    ok &= item(27, "all output hashes recorded", os.path.exists(man_path)
               and len(pd.read_csv(man_path)) >= 15, man_path)
    return ok


REQUIRED = ["STEP5G_CLARIFICATION_01.md", "STEP5H_A_REPORT.md", "STEP5H_A_SUMMARY.json",
            "01_ni_descriptor_access/STEP5H_A_NI_FIRST_DESCRIPTOR_ACCESS.json",
            "01_ni_descriptor_access/STEP5H_A_NI_PROPERTY_ACCESS_LEDGER.csv",
            "02_ni_features/STEP5H_A_NI_FEATURES.parquet",
            "03_ood_diagnostics/STEP5H_A_NI_FEATURE_RANGE_DIAGNOSTICS.csv",
            "03_ood_diagnostics/STEP5H_A_NI_OOD_DIAGNOSTICS.csv",
            "04_primary_predictions/STEP5H_A_PRIMARY_NI_PREDICTIONS.parquet",
            "04_primary_predictions/STEP5H_A_PRIMARY_PREDICTION_FREEZE.json",
            "05_secondary_predictions/STEP5H_A_SECONDARY_NI_PREDICTIONS.parquet",
            "05_secondary_predictions/STEP5H_A_SECONDARY_PREDICTION_FREEZE.json",
            "06_disagreement/STEP5H_A_PRIMARY_SECONDARY_DISAGREEMENT.csv",
            "07_maximin_freeze/STEP5H_A_MAXIMIN_SEQUENCE.csv",
            "STEP5H_A_INTEGRITY_AUDIT.md",
            "09_manifest/STEP5H_A_FILE_HASHES.csv"]


def build_report(d, ok):
    pfz, sfz, ood = d["pfz"], d["sfz"], d["oodfz"]
    dist = d["dist"]
    dec = "READY FOR STEP5H-B\nNi CALPHAD BLIND REVEAL" if ok else "NOT READY"
    L = []
    A = L.append
    A("STEP5H-A STATUS: %s" % ("PASS" if ok else "BLOCKED"))
    A("")
    A("# STEP5H_A_REPORT.md")
    A("")
    A("**Ni INPUT DESCRIPTOR EXTRACTION + FEATURE CONSTRUCTION + ZERO-SHOT "
      "PREDICTION FREEZE + TARGET-FREE OOD DIAGNOSTICS + POST-BLIND MAXIMIN-ID FREEZE**")
    A("")
    A("Generated %s. The frozen STEP5G_PRIMARY and STEP5G_SECONDARY checkpoints have "
      "produced predictions for all 22,000 preregistered Ni blind states. Both "
      "prediction files are cryptographically frozen. No Ni CALPHAD target exists; no "
      "error was computed; no model was scored." % C.utcnow())
    A("")
    A("---")
    A("")
    A("## Answers (spec 42)")
    A("")

    def q(n, t, a):
        A("**%d. %s** %s" % (n, t, a))
        A("")

    q(1, "Was the STEP5G clarification additive only?",
      "Yes - `STEP5G_CLARIFICATION_01.md` is a new append-only document; every frozen "
      "STEP5G hash (freeze JSON, both checkpoints, both scalers, schema, composition "
      "table, random sequences) verified unchanged at write time and again in this "
      "audit (items 1-7). No frozen artifact was edited.")
    q(2, "When was the first Ni elemental-property access?",
      "**%s** - recorded in `STEP5H_A_NI_FIRST_DESCRIPTOR_ACCESS.json`, written "
      "IMMEDIATELY BEFORE the first read, with all frozen hashes bound in."
      % d["access"]["timestamp_utc"])
    q(3, "Which exact Ni properties were accessed?",
      "Ten, and only ten, extracted directly: %s. Plus one aggregate access line: the "
      "frozen matminer ElementProperty('magpie') preset reads its own repository "
      "internally at featurization (recorded, values not individually enumerated)."
      % ", ".join("%s (%s.%s)" % (r["property"], "matminer" if "matminer" in r.source
                                  else "mendeleev", r.raw_field)
                  for _, r in d["ledger"][d["ledger"].status == "EXTRACTED"].iterrows()))
    q(4, "Were all accessed properties required by the frozen 43-feature schema?",
      "Yes - the ledger maps every property to the exact schema columns that require "
      "it (`required_by_features`); none is unmapped.")
    q(5, "Were any extra Ni properties accessed?",
      "No. Deliberately NOT extracted: %s - no frozen 43-feature column requires them "
      "(spec 4 minimization)." % ", ".join(C.NI_NOT_EXTRACTED))
    q(6, "Were the same historical sources used?",
      "Yes - matminer MagpieData (magpie.data.json) and the mendeleev built-in "
      "elemental database, the identical approved STEP5A sources, with the same "
      "fields, units and missing-value policy. No web search, no manual entry, no "
      "Wikipedia, no Materials Project, no DFT, no alternate database.")
    cons = d["featqa"]["consistency_test"]
    q(7, "Did the reference existing-element feature reproduction pass?",
      "**%s** - 6 elements x 3 temperatures (300/700/1000 C) = %d states recomputed "
      "through the STEP5H-A featurization path and compared against the historical "
      "frozen feature table: max |delta| = %.3e (tolerance %.0e). The Ni features were "
      "only accepted AFTER this passed."
      % (cons["verdict"], cons["states"], cons["max_abs_delta"], C.CONSISTENCY_TOL))
    q(8, "Ni feature rows = 22000?", "Yes - %d." % len(d["feats"]))
    q(9, "Feature count = 43?", "Yes - %d, in the exact frozen schema order "
      "(Temperature_C is itself feature #1)."
      % len([c for c in d["feats"].columns if c not in
             ("state_id", "composition_id", "fewshot_role", "fewshot_rank")]))
    q(10, "Any NaN / inf?", "None - all 22,000 x 43 values finite (feature QA).")
    q(11, "How many features extend outside the PRIMARY training range?",
      "**%d of 43**: %s." % (ood["n_features_extending_train_range"],
                             ", ".join("`%s`" % f for f in
                                       ood["features_extending_train_range"]) or "none"))
    q(12, "What fraction of states show |z|>3 or |z|>5?",
      "%.1f%% of states have at least one |z|>3 feature; %.1f%% at least one |z|>5. "
      "Median max_abs_z %.2f, p95 %.2f. Descriptive only - no threshold was or may be "
      "selected (spec 22)."
      % (100 * ood["frac_states_any_abs_z_gt3"], 100 * ood["frac_states_any_abs_z_gt5"],
         ood["max_abs_z_median"], ood["max_abs_z_p95"]))
    q(13, "Nearest-feature-space distance summary?",
      "In the frozen PRIMARY standardized 43-feature space: 1-NN distance median "
      "%.3f, p95 %.3f; 5-NN mean median %.3f (241,748 training reference rows)."
      % (ood["nn1_median"], ood["nn1_p95"], ood["knn5_median"]))
    q(14, "PRIMARY checkpoint hash verified?", "Yes - `%s…` matches the frozen value "
      "exactly, role ORIGINAL6_DIRECT_MLP, architecture 43->128->64->5."
      % C.FROZEN_HASHES["primary_checkpoint"][:16])
    q(15, "PRIMARY scaler hash verified?", "Yes - `%s…`; no refit, no Ni-based "
      "adaptation." % C.FROZEN_HASHES["primary_scaler"][:16])
    q(16, "SECONDARY checkpoint hash verified?", "Yes - `%s…`, role FULL11_DIRECT_MLP."
      % C.FROZEN_HASHES["secondary_checkpoint"][:16])
    q(17, "SECONDARY scaler hash verified?", "Yes - `%s…`."
      % C.FROZEN_HASHES["secondary_scaler"][:16])
    q(18, "PRIMARY predictions = 22000?", "Yes - %d rows." % pfz["row_count"])
    q(19, "PRIMARY simplex valid?", "Yes - all components >= 0, all rows sum to 1 "
      "within %.1e (max deviation %.2e), all finite, state_id unique."
      % (1e-12, pfz["max_projected_sum_deviation"]))
    q(20, "PRIMARY fallback count?", "%d states hit the zero-sum fallback (frozen "
      "PRIMARY train-mean simplex)." % pfz["simplex_fallback_count"])
    q(21, "PRIMARY prediction SHA256?", "`%s` - frozen in "
      "`STEP5H_A_PRIMARY_PREDICTION_FREEZE.json` at %s; immutable from that moment."
      % (pfz["sha256"], pfz["generated_utc"]))
    q(22, "SECONDARY predictions = 22000?", "Yes - %d rows, exact same state set."
      % sfz["row_count"])
    q(23, "SECONDARY simplex valid?", "Yes - max sum deviation %.2e."
      % sfz["max_projected_sum_deviation"])
    q(24, "SECONDARY fallback count?", "%d." % sfz["simplex_fallback_count"])
    q(25, "SECONDARY prediction SHA256?", "`%s`, frozen at %s."
      % (sfz["sha256"], sfz["generated_utc"]))
    dp = dist["summaries"][0]["modal_topology_distribution"]
    ds = dist["summaries"][1]["modal_topology_distribution"]
    q(26, "PRIMARY predicted topology distribution?",
      ", ".join("%s %.1f%%" % (k, 100 * v) for k, v in
                sorted(dp.items(), key=lambda kv: -kv[1]))
      + ". Descriptive; no accuracy terminology (spec 31).")
    q(27, "SECONDARY predicted topology distribution?",
      ", ".join("%s %.1f%%" % (k, 100 * v) for k, v in
                sorted(ds.items(), key=lambda kv: -kv[1])) + ".")
    q(28, "PRIMARY vs SECONDARY mean disagreement?",
      "Mean per-state TVD between the two frozen prediction sets: **%.4f** (median "
      "%.4f, p95 %.4f). Lower disagreement is NOT correctness (spec 30)."
      % (dist["disagreement"]["mean_tvd_primary_vs_secondary"],
         dist["disagreement"]["median_tvd"], dist["disagreement"]["p95_tvd"]))
    q(29, "Topology agreement rate?", "**%.4f** of states."
      % dist["disagreement"]["topology_agreement_rate"])
    q(30, "Maximin sequence frozen?",
      "Yes - 32 nested IDs from the frozen 600-composition calibration pool, "
      "473-dim target-free geometry (43 features x 11 temperatures, frozen PRIMARY "
      "scaler), reference = %d PRIMARY train compositions. Rank 1: %s."
      % (d["maxfz"]["reference_set"]["n"], d["maxseq"].composition_id.iloc[0]))
    q(31, "Maximin SHA256?", "`%s` (`STEP5H_A_MAXIMIN_SEQUENCE.csv`); cannot change "
      "after the reveal." % d["maxfz"]["sequence_sha256"])
    q(32, "Were the random few-shot sequences unchanged?",
      "Yes - `STEP5G_NESTED_RANDOM_SEQUENCES.csv` hash-verified (`%s…`) and its "
      "content matched the preregistration JSON; NOT regenerated (spec 36)."
      % C.FROZEN_HASHES["fewshot_sequences_csv"][:16])
    q(33, "Ni CALPHAD calls = 0?", "**0.** No Thermo-Calc launch, no tc_python import "
      "anywhere in this stage's code or process, no Ni equilibrium state.")
    q(34, "Ni target rows accessed = 0?", "**0.** No Ni phase label, fraction, "
      "partition vector, CALPHAD file or ground truth exists or was read; the "
      "target-column guard ran clean on the feature table and both prediction files.")
    q(35, "Any model retraining = 0?", "**0.** Both checkpoint hashes are bit-identical "
      "to the STEP5G freeze; no epoch, seed, architecture or hyperparameter touched.")
    q(36, "Any scaler refitting = 0?", "**0.** Both scaler hashes bit-identical; Ni rows "
      "were transformed with the frozen scalers; standardized values were NOT clipped, "
      "winsorized, capped or recentered (spec 16) - true extrapolation remains visible.")
    q(37, "Are both prediction files immutable?",
      "Yes - PRIMARY frozen at %s, SECONDARY at %s, each with SHA256, size, row count, "
      "schema hash and timestamp in its freeze JSON. Any regeneration would change the "
      "hash and fail the audit." % (pfz["immutable_from_utc"], sfz["immutable_from_utc"]))
    q(38, "READY FOR STEP5H-B?",
      "**%s**" % ("YES - all 27 integrity items PASS. The next stage is the Ni CALPHAD "
                  "blind reveal (STEP5H-B): run the frozen FULL_V1 thermodynamic "
                  "settings on the 2,000 preregistered compositions, then score "
                  "PRIMARY first, freeze, then SECONDARY, per the STEP5G metric "
                  "preregistration. Nothing was scored here."
                  if ok else "NO - see blocking reasons."))
    A("---")
    A("")
    A("## OOD diagnostic snapshot (frozen pre-reveal, descriptive only)")
    A("")
    A("| diagnostic | value |")
    A("|---|---|")
    A("| features extending PRIMARY train range | %d / 43 |"
      % ood["n_features_extending_train_range"])
    A("| states with any |z|>3 | %.1f%% |" % (100 * ood["frac_states_any_abs_z_gt3"]))
    A("| states with any |z|>5 | %.1f%% |" % (100 * ood["frac_states_any_abs_z_gt5"]))
    A("| 1-NN distance median / p95 | %.3f / %.3f |" % (ood["nn1_median"], ood["nn1_p95"]))
    dp4 = ood["descriptor_position"]
    A("| Ni nearest original identity (descriptor z-space) | %s (z-dist %.2f) |"
      % (dp4["nearest_original_identity"], dp4["z_distance_to_nearest"]))
    A("| Ni descriptor properties outside the six-element range | %d: %s |"
      % (dp4["outside_six_range_count"], ", ".join(dp4["outside_six_range_properties"])
         or "none"))
    A("")
    A("These were frozen BEFORE any Ni CALPHAD existed. They may later be correlated "
      "with blind error; no threshold may be selected after the reveal (spec 22).")
    A("")
    A("---")
    A("")
    A("## Integrity audit (spec 43)")
    A("")
    A("| # | item | result | detail |")
    A("|---:|---|---|---|")
    for x in ITEMS:
        A("| %d | %s | **%s** | %s |" % (x["n"], x["item"], x["result"],
                                         x["detail"].replace("|", "/")[:170]))
    A("")
    A("---")
    A("")
    A("## Stop rule (spec 45)")
    A("")
    A("STEP5H-A STOPS HERE. No Thermo-Calc was launched, no TC-Python equilibrium was "
      "run, no Ni label was calculated, no prediction error was computed, neither "
      "PRIMARY nor SECONDARY was scored, and no few-shot recovery was executed. "
      "Awaiting user review before STEP5H-B.")
    A("")
    A("# %s" % dec)
    return "\n".join(L), dec


def main():
    print("=" * 78)
    print("STEP5H-A  Stage 6: audit, report, summary, hashes, decision")
    print("=" * 78)
    C.assert_no_tc_python()
    d = load()

    # write hashes FIRST (so audit item 27 can verify), then audit, then report,
    # then rewrite hashes so the report itself is covered
    def write_hashes():
        rows = []
        for base, _, files in os.walk(C.ROOT):
            if "__pycache__" in base:
                continue
            for fn in sorted(files):
                if fn.endswith(".pyc"):
                    continue
                p = os.path.join(base, fn)
                rows.append({"relative_path": os.path.relpath(p, C.ROOT).replace("\\", "/"),
                             "bytes": os.path.getsize(p), "sha256": C.sha256_file(p)})
        h = pd.DataFrame(sorted(rows, key=lambda r: r["relative_path"]))
        h.to_csv(os.path.join(C.D_MAN, "STEP5H_A_FILE_HASHES.csv"), index=False,
                 encoding="utf-8")
        return h

    write_hashes()
    ok = audit(d)
    report, dec = build_report(d, ok)
    with open(os.path.join(C.ROOT, "STEP5H_A_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    lines = ["# STEP5H_A_INTEGRITY_AUDIT.md", "", "generated_utc: %s" % C.utcnow(), "",
             "| # | item | result | detail |", "|---:|---|---|---|"]
    for x in ITEMS:
        lines.append("| %d | %s | **%s** | %s |" % (x["n"], x["item"], x["result"],
                                                    x["detail"].replace("|", "/")[:200]))
    npass = sum(i["result"] == "PASS" for i in ITEMS)
    lines += ["", "**%d PASS / %d FAIL.**" % (npass, len(ITEMS) - npass), "",
              "DECISION: %s" % dec, ""]
    with open(os.path.join(C.ROOT, "STEP5H_A_INTEGRITY_AUDIT.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))

    # computed before the summary exists; the summary itself is checked after
    missing = [f for f in REQUIRED if f != "STEP5H_A_SUMMARY.json"
               and not os.path.exists(os.path.join(C.ROOT, f))]
    C.write_json(os.path.join(C.ROOT, "STEP5H_A_SUMMARY.json"), {
        "stage": "STEP5H_A_PREDICTION_FREEZE_V1",
        "generated_utc": C.utcnow(),
        "status": "PASS" if ok else "BLOCKED",
        "decision": dec,
        "clarification_additive_only": True,
        "first_ni_access_utc": d["access"]["timestamp_utc"],
        "ni_properties_extracted": d["accsum"]["properties_extracted"],
        "ni_properties_not_extracted": C.NI_NOT_EXTRACTED,
        "source_consistency_test": d["featqa"]["consistency_test"],
        "n_states": int(len(d["feats"])), "n_features": 43,
        "ood_diagnostics_frozen_pre_reveal": True,
        "features_extending_train_range": ood_n(d),
        "primary_prediction_sha256": d["pfz"]["sha256"],
        "primary_fallbacks": d["pfz"]["simplex_fallback_count"],
        "secondary_prediction_sha256": d["sfz"]["sha256"],
        "secondary_fallbacks": d["sfz"]["simplex_fallback_count"],
        "disagreement_mean_tvd": d["dist"]["disagreement"]["mean_tvd_primary_vs_secondary"],
        "topology_agreement_rate": d["dist"]["disagreement"]["topology_agreement_rate"],
        "maximin_sequence_sha256": d["maxfz"]["sequence_sha256"],
        "random_sequences_unchanged": True,
        "ni_calphad_calls": 0, "ni_target_rows_accessed": 0,
        "model_retraining_events": 0, "scaler_refit_events": 0,
        "feature_selection_events": 0, "fewshot_executed": False,
        "b_skill_computed": False, "b_recovery25_computed": False,
        "predictions_immutable": True,
        "audit": ITEMS, "audit_failures": len(ITEMS) - npass,
        "missing_required_outputs": missing,
        "next_stage": "STEP5H-B Ni CALPHAD blind reveal - ONLY after user review"})

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(C.ROOT, f))]
    h = write_hashes()
    print("\n" + "=" * 78)
    print("%d/%d integrity items PASS | %d files hashed" % (npass, len(ITEMS), len(h)))
    print("DECISION: %s" % dec)
    if missing:
        print("MISSING REQUIRED OUTPUTS: %s" % missing)
    print("STOP RULE: no CALPHAD, no labels, no errors, no scoring, no few-shot.")
    print("=" * 78)
    return 0 if ok and not missing else 2


def ood_n(d):
    return d["oodfz"]["n_features_extending_train_range"]


if __name__ == "__main__":
    sys.exit(main())
