# -*- coding: utf-8 -*-
"""
110_i_preflight.py
STEP5I Stage 0: verify every frozen input before anything is computed.

The zero-shot result is the scientific asset of this project. If any of it has moved,
STEP5I must stop rather than quietly recover against a different target. Every check
here is read-only and takes seconds.

  I01  frozen STEP5H-B Ni targets hash exactly as the brief states
  I02  frozen STEP5H-A PRIMARY zero-shot predictions hash exactly as the brief states
  I03  frozen maximin sequence hashes exactly as the brief states
  I04  frozen nested random sequences hash as recorded in STEP5G / STEP5H-A
  I05  frozen STEP5G PRIMARY checkpoint and scaler unchanged
  I06  frozen STEP5H-B PRIMARY and SECONDARY blind scores unchanged
  I07  the STEP5I protocol agrees with the frozen STEP5G preregistration
  I08  split is exactly 600 FEWSHOT_CALIBRATION / 1400 RECOVERY_EVALUATION, and the
       split rule reproduces it independently from composition ids alone
  I09  RECOVERY_EVALUATION is exactly 15,400 converged states (1400 x 11)
  I10  all 96 random anchor ids and all 32 maximin ids lie inside the calibration pool
  I11  nested prefixes hold: budget 1 subset of 2 subset of 4 ... subset of 32
  I12  no anchor composition is ever a recovery-evaluation composition
  I13  frozen PRIMARY train / validation ID files are present, disjoint, and carry no
       Ni row
  I14  the 43 features exist in both the Ni feature table and the original-six table
  I15  NBNI3 is present in the frozen phase vocabulary with the frozen OTHER_SOLID
       mapping, and appears only between 300 and 600 degC

Exit 0 = GO, 2 = NO-GO.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i_common as C

CHECKS = []


def parquet_columns(path):
    """Schema only - never materialize a large table just to ask for its names."""
    import pyarrow.parquet as pq
    return list(pq.ParquetFile(path).schema_arrow.names)


def rec(cid, name, ok, detail):
    CHECKS.append({"id": cid, "check": name, "result": "PASS" if ok else "FAIL",
                   "detail": str(detail)[:600]})
    print("  [%s] %-4s %s" % ("PASS" if ok else "FAIL", cid, name))
    if not ok:
        print("         -> %s" % detail)
    return ok


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5I  Stage 0: frozen-input pre-flight (read-only)")
    print("=" * 78)

    required = [C.G_COMMON, C.G_FREEZE_JSON, C.G_FEWSHOT_JSON, C.G_SEQ_CSV,
                C.G_PRIMARY_CKPT, C.G_PRIMARY_SCALER, C.G_PRIMARY_TRAIN_IDS,
                C.G_PRIMARY_VAL_IDS, C.HA_PRED, C.HA_FEATURES, C.HA_MAXIMIN_CSV,
                C.HB_TARGETS, C.HB_PRIMARY_SCORE, C.HB_SECONDARY_SCORE,
                C.HB_PHASE_VOCAB, C.D_DIAG_DATA]
    absent = [p for p in required if not os.path.exists(p)]
    if absent:
        print("NO-GO: required frozen input(s) absent:")
        for a in absent:
            print("   %s" % a)
        return 2

    gc = C.load_g_common()
    ok = True

    # ---- I01-I05 hashes -------------------------------------------------- #
    live = {"HB_TARGETS": C.sha256_file(C.HB_TARGETS),
            "HA_PRIMARY_PREDICTIONS": C.sha256_file(C.HA_PRED),
            "HA_MAXIMIN_SEQUENCE": C.sha256_file(C.HA_MAXIMIN_CSV),
            "G_RANDOM_SEQUENCES": C.sha256_file(C.G_SEQ_CSV),
            "G_PRIMARY_CHECKPOINT": C.sha256_file(C.G_PRIMARY_CKPT),
            "G_PRIMARY_SCALER": C.sha256_file(C.G_PRIMARY_SCALER),
            "G_SECONDARY_CHECKPOINT": C.sha256_file(C.G_SECONDARY_CKPT)}
    for cid, key, label in (("I01", "HB_TARGETS", "STEP5H-B Ni targets"),
                            ("I02", "HA_PRIMARY_PREDICTIONS",
                             "STEP5H-A PRIMARY zero-shot predictions"),
                            ("I03", "HA_MAXIMIN_SEQUENCE", "maximin anchor sequence"),
                            ("I04", "G_RANDOM_SEQUENCES", "nested random sequences"),
                            ("I05", "G_PRIMARY_CHECKPOINT",
                             "STEP5G PRIMARY checkpoint")):
        ok &= rec(cid, "%s sha256 unchanged" % label,
                  live[key] == C.FROZEN_HASHES[key],
                  "live %s | expected %s" % (live[key][:16], C.FROZEN_HASHES[key][:16]))
    ok &= rec("I05b", "STEP5G PRIMARY scaler and SECONDARY checkpoint unchanged",
              live["G_PRIMARY_SCALER"] == C.FROZEN_HASHES["G_PRIMARY_SCALER"]
              and live["G_SECONDARY_CHECKPOINT"]
              == C.FROZEN_HASHES["G_SECONDARY_CHECKPOINT"],
              {k: v[:16] for k, v in live.items() if "SCALER" in k or "SECONDARY" in k})

    # ---- I06 frozen blind scores ----------------------------------------- #
    ps = json.load(open(C.HB_PRIMARY_SCORE, encoding="utf-8"))
    ss = json.load(open(C.HB_SECONDARY_SCORE, encoding="utf-8"))
    sec_tvd = ss.get("mean_tvd", ss.get("all_secondary_metrics", {}).get("mean_tvd"))
    ok &= rec("I06", "frozen PRIMARY and SECONDARY blind scores unchanged",
              abs(ps["mean_tvd"] - C.ZERO_SHOT_FULL_MEAN_TVD) < 1e-12
              and abs(ps["trivial_baseline_tvd"] - C.ZERO_SHOT_FULL_TRIVIAL_TVD) < 1e-12
              and sec_tvd is not None
              and abs(float(sec_tvd) - C.SECONDARY_FULL_MEAN_TVD) < 1e-6
              and ps["primary_prediction_sha256"]
              == C.FROZEN_HASHES["HA_PRIMARY_PREDICTIONS"]
              and ps["target_sha256"] == C.FROZEN_HASHES["HB_TARGETS"],
              "PRIMARY %.10f | SECONDARY %s | trivial %.10f"
              % (ps["mean_tvd"], sec_tvd, ps["trivial_baseline_tvd"]))

    # ---- I07 protocol vs preregistration ---------------------------------- #
    pre = C.fewshot_prereg()
    try:
        C.assert_protocol_matches_prereg(pre)
        agree, why = True, ("LR %.4g, batch %d, <=%d epochs, patience %d, seed %d, "
                            "budgets %s nested, 3 replicates, "
                            "PRIMARY_FORMULATION_REFIT"
                            % (C.REFIT_LR, C.REFIT_BATCH, C.REFIT_EPOCHS,
                               C.REFIT_PATIENCE, C.REFIT_SEED, C.BUDGETS))
    except SystemExit as exc:
        agree, why = False, str(exc)
    ok &= rec("I07", "STEP5I protocol matches the frozen STEP5G preregistration",
              agree, why)

    # ---- I08 / I09 split and recovery size --------------------------------- #
    tgt = pd.read_parquet(C.HB_TARGETS, columns=["state_id", "composition_id",
                                                 "Temperature_C", "fewshot_role",
                                                 "calphad_status"])
    roles = tgt.groupby("composition_id").fewshot_role.nunique()
    counts = tgt.drop_duplicates("composition_id").fewshot_role.value_counts().to_dict()
    cal_ids = sorted(set(tgt.loc[tgt.fewshot_role == "FEWSHOT_CALIBRATION",
                                 "composition_id"]))
    ev_ids = sorted(set(tgt.loc[tgt.fewshot_role == "RECOVERY_EVALUATION",
                                "composition_id"]))
    # independent reproduction from composition ids alone, using the frozen rule
    all_ids = sorted(set(tgt.composition_id))
    repro = sorted(all_ids, key=gc.fewshot_split_role)
    repro_cal = set(repro[:C.N_CALIBRATION])
    ok &= rec("I08", "split is 600 / 1400 and reproduces from the frozen rule",
              counts.get("FEWSHOT_CALIBRATION") == C.N_CALIBRATION
              and counts.get("RECOVERY_EVALUATION") == C.N_RECOVERY_EVAL
              and int(roles.max()) == 1 and repro_cal == set(cal_ids),
              "%s | one role per composition: %s | independent reproduction matches: %s"
              % (counts, int(roles.max()) == 1, repro_cal == set(cal_ids)))

    ev = tgt[tgt.fewshot_role == "RECOVERY_EVALUATION"]
    conv = ev[ev.calphad_status.astype(str).str.upper().isin(
        ["SUCCESS", "CONVERGED", "OK"])] if "calphad_status" in ev.columns else ev
    per_comp_T = ev.groupby("composition_id").Temperature_C.nunique()
    ok &= rec("I09", "RECOVERY_EVALUATION is exactly 15,400 states (1400 x 11), all "
                     "converged",
              len(ev) == C.N_RECOVERY_STATES and len(conv) == C.N_RECOVERY_STATES
              and int(per_comp_T.min()) == 11 and int(per_comp_T.max()) == 11,
              "%d states | converged %d | temperatures per composition %d-%d"
              % (len(ev), len(conv), per_comp_T.min(), per_comp_T.max()))

    # ---- I10 / I11 / I12 anchors ------------------------------------------- #
    seqs = C.random_sequences()
    mx = C.maximin_sequence()
    cal = set(cal_ids)
    outside = sorted({c for s in seqs.values() for c in s if c not in cal}
                     | {c for c in mx if c not in cal})
    ok &= rec("I10", "every frozen anchor id lies inside the calibration pool",
              not outside, outside or "96 random + 32 maximin ids, all in the 600")

    nest_bad = []
    for label, s in list(("RANDOM_R%d" % r, seqs[r]) for r in C.REPLICATES) + \
            [("MAXIMIN", mx)]:
        for a, b in zip(C.BUDGETS, C.BUDGETS[1:]):
            if set(s[:a]) - set(s[:b]):
                nest_bad.append("%s: budget %d not nested in %d" % (label, a, b))
        if len(set(s)) != 32:
            nest_bad.append("%s: sequence is not 32 unique ids" % label)
    ok &= rec("I11", "nested prefixes hold for all four sequences (1 in 2 in 4 ... 32)",
              not nest_bad, nest_bad or "1 c 2 c 4 c 8 c 16 c 32 verified 4x")

    ok &= rec("I12", "no anchor composition is a recovery-evaluation composition",
              not ({c for s in seqs.values() for c in s} & set(ev_ids))
              and not (set(mx) & set(ev_ids)),
              "anchor pool and evaluation set are disjoint by construction")

    # ---- I13 frozen PRIMARY id files --------------------------------------- #
    tr_ids = pd.read_csv(C.G_PRIMARY_TRAIN_IDS)
    va_ids = pd.read_csv(C.G_PRIMARY_VAL_IDS)
    ni_in_ids = int(tr_ids.state_id.astype(str).str.contains("NI_BLIND").sum()
                    + va_ids.state_id.astype(str).str.contains("NI_BLIND").sum())
    ok &= rec("I13", "frozen PRIMARY train/validation ID files present, disjoint, Ni-free",
              len(tr_ids) == C.PRIMARY_TRAIN_ROWS
              and len(va_ids) == C.PRIMARY_VAL_ROWS
              and not (set(tr_ids.state_id) & set(va_ids.state_id)) and ni_in_ids == 0,
              "train %d / validation %d rows | shared ids %d | Ni rows %d"
              % (len(tr_ids), len(va_ids),
                 len(set(tr_ids.state_id) & set(va_ids.state_id)), ni_in_ids))

    # ---- I14 features in both tables --------------------------------------- #
    ni_cols = parquet_columns(C.HA_FEATURES)
    orig_cols = parquet_columns(C.D_DIAG_DATA)
    feats = C.feature_schema()
    miss_ni = [f for f in feats if f not in ni_cols]
    miss_o = [f for f in feats if f not in orig_cols]
    ok &= rec("I14", "the 43 frozen production features exist in both the Ni table and "
                     "the original-six table",
              len(feats) == C.N_FEATURES and not miss_ni and not miss_o,
              "schema %d features | missing from Ni table: %s | missing from the "
              "original-six table: %s"
              % (len(feats), miss_ni or "none", miss_o or "none"))

    # ---- I15 NBNI3 --------------------------------------------------------- #
    voc = pd.read_csv(C.HB_PHASE_VOCAB)
    nb = voc[voc.raw_phase_name.astype(str).str.upper().str.contains(C.NBNI3_RAW_NAME)]
    ok &= rec("I15", "NBNI3 present in the frozen vocabulary, mapped to OTHER_SOLID, "
                     "300-600 degC only",
              len(nb) >= 1 and set(nb.aggregated_family) == {"OTHER_SOLID"}
              and int(nb.n_states.sum()) == C.NBNI3_N_STATES_FROZEN
              and float(nb.Tmin.min()) == C.NBNI3_TEMPERATURE_RANGE[0]
              and float(nb.Tmax.max()) == C.NBNI3_TEMPERATURE_RANGE[1],
              "%d states, %s, %.0f-%.0f degC (mapping is frozen and not changed here)"
              % (int(nb.n_states.sum()), sorted(set(nb.aggregated_family)),
                 nb.Tmin.min(), nb.Tmax.max()) if len(nb) else "NBNI3 absent")

    pd.DataFrame(CHECKS).to_csv(os.path.join(C.D_QA, "STEP5I_PREFLIGHT_CHECKS.csv"),
                                index=False, encoding="utf-8")
    C.write_json(os.path.join(C.D_CONFIG, "STEP5I_PREFLIGHT.json"), {
        "generated_utc": C.utcnow(),
        "verdict": "GO" if ok else "NO-GO",
        "stage_type": "POST-BLIND RETROSPECTIVE RECOVERY",
        "not": ["zero-shot", "blind", "prospective zero-shot"],
        "frozen_hashes_live": live,
        "frozen_hashes_expected": C.FROZEN_HASHES,
        "split": counts,
        "n_recovery_states": int(len(ev)),
        "random_sequences": seqs, "maximin_sequence": mx,
        "budgets": C.BUDGETS, "replicates": C.REPLICATES,
        "refit": {"learning_rate": C.REFIT_LR, "batch": C.REFIT_BATCH,
                  "max_epochs": C.REFIT_EPOCHS, "patience": C.REFIT_PATIENCE,
                  "seed": C.REFIT_SEED, "scaler": "frozen PRIMARY, never refit",
                  "validation": "frozen original-six DEV_VALIDATION, no Ni"},
        "checks": CHECKS})
    C.write_json(os.path.join(C.D_FROZEN, "STEP5I_FROZEN_INPUT_VERIFICATION.json"), {
        "generated_utc": C.utcnow(), "hashes": live, "expected": C.FROZEN_HASHES,
        "all_match": all(live[k] == C.FROZEN_HASHES[k] for k in C.FROZEN_HASHES)})

    print("-" * 78)
    print("PRE-FLIGHT VERDICT: %s" % ("GO" if ok else "NO-GO"))
    if not ok:
        print("STEP5I stops here. No model was refit and no frozen artifact was touched.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
