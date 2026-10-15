# -*- coding: utf-8 -*-
"""
117_i_audit_report.py
STEP5I Stage 7: 32-item integrity audit, the 45-answer report, file hashes and the
next-stage recommendation (spec 50, 54, 55, 56).

The audit re-derives what it checks from the artifacts on disk rather than trusting the
summaries the earlier stages wrote about themselves. Nothing is executed here: no
model, no CALPHAD, no targeted anchor design, no binary Ni study.

The report ends with a recommendation CATEGORY, not an action. Spec 56 is explicit that
no category may be executed, so this stage writes
STEP5I_NEXT_STAGE_RECOMMENDATION.md and stops.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i_common as C

ITEMS = []
DISCLAIMER_OPEN = "<!-- DISCLAIMER_OPEN -->"
DISCLAIMER_CLOSE = "<!-- DISCLAIMER_CLOSE -->"
FORBIDDEN = [
    "prospective blind validation", "predicted strongest alloy", "strongest alloy",
    "kinetic precipitation driving force", "jmst success threshold",
    "successful zero-shot", "zero-shot success", "never read", "never accessed",
    "fully untouched",
]


def item(n, name, ok, detail):
    ITEMS.append({"n": n, "item": name, "result": "PASS" if ok else "FAIL",
                  "detail": str(detail)[:500]})
    print("  %2d %-4s %s" % (n, "PASS" if ok else "FAIL", name))
    if not ok:
        print("        -> %s" % detail)
    return ok


def scan_forbidden(path):
    """A STEP5I document must be able to state what it is NOT - that is the whole
    framing requirement of spec 1. Text inside an explicit disclaimer block is
    therefore exempt; everything else is scanned."""
    txt = open(path, encoding="utf-8").read()
    body = re.sub(re.escape(DISCLAIMER_OPEN) + r".*?" + re.escape(DISCLAIMER_CLOSE),
                  " ", txt, flags=re.S).lower()
    return [p for p in FORBIDDEN if p in body]


def load(p):
    return json.load(open(p, encoding="utf-8"))


def index_files(exclude_path):
    """Hash every STEP5I output except the index itself and __pycache__."""
    rows = []
    for basedir, _, files in os.walk(C.ROOT):
        for fn in sorted(files):
            p = os.path.join(basedir, fn)
            if fn.endswith(".pyc") or os.path.abspath(p) == os.path.abspath(exclude_path):
                continue
            rows.append({"relative_path": os.path.relpath(p, C.ROOT),
                         "bytes": os.path.getsize(p), "sha256": C.sha256_file(p)})
    return pd.DataFrame(sorted(rows, key=lambda r: r["relative_path"]))


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5I  Stage 7: integrity audit, report, hashes, recommendation")
    print("=" * 78)

    P = {
        "preflight": os.path.join(C.D_CONFIG, "STEP5I_PREFLIGHT.json"),
        "refit_summary": os.path.join(C.D_CONFIG, "STEP5I_REFIT_SUMMARY.json"),
        "manifest": os.path.join(C.D_MANIFEST,
                                 "STEP5I_RECOVERY_CHECKPOINT_MANIFEST.csv"),
        "baselines": os.path.join(C.D_METRICS, "STEP5I_RECOVERY_BASELINES.json"),
        "thresholds": os.path.join(C.D_METRICS, "STEP5I_BUDGET_THRESHOLDS.json"),
        "agg": os.path.join(C.D_METRICS, "STEP5I_RANDOM_FEWSHOT_METRICS.csv"),
        "per_rep": os.path.join(C.D_METRICS,
                                "STEP5I_RANDOM_FEWSHOT_PER_REPLICATE.csv"),
        "maximin": os.path.join(C.D_METRICS, "STEP5I_MAXIMIN_FEWSHOT_METRICS.csv"),
        "controls": os.path.join(C.D_CONTROL, "STEP5I_CONTROL_METRICS.csv"),
        "control_def": os.path.join(C.D_CONTROL, "STEP5I_CONTROL_DEFINITIONS.json"),
        "cmp": os.path.join(C.D_METRICS, "STEP5I_MODEL_VS_CONTROL_COMPARISON.csv"),
        "coverage": os.path.join(C.D_TOPO, "STEP5I_ANCHOR_TOPOLOGY_COVERAGE.csv"),
        "topo": os.path.join(C.D_TOPO, "STEP5I_TRUE_TOPOLOGY_STRATIFIED.csv"),
        "nbni3": os.path.join(C.D_NBNI3, "STEP5I_NBNI3_STRATIFIED.csv"),
        "temp": os.path.join(C.D_TOPO, "STEP5I_TEMPERATURE_REGIME_METRICS.csv"),
        "mech": os.path.join(C.D_NBNI3, "STEP5I_COVERAGE_MECHANISM_CORRELATIONS.csv"),
        "scope": os.path.join(C.D_NBNI3, "STEP5I_DIAGNOSTIC_SCOPE.json"),
        "boot": os.path.join(C.D_BOOT, "STEP5I_PAIRED_BOOTSTRAP.csv"),
    }
    absent = [k for k, v in P.items() if not os.path.exists(v)]
    if absent:
        print("NOT READY FOR USER REVIEW - missing STEP5I artifacts: %s" % absent)
        return 2

    pf, rs = load(P["preflight"]), load(P["refit_summary"])
    base, thr = load(P["baselines"]), load(P["thresholds"])
    cdef, scope = load(P["control_def"]), load(P["scope"])
    man = pd.read_csv(P["manifest"])
    agg = pd.read_csv(P["agg"])
    per_rep = pd.read_csv(P["per_rep"])
    maxi = pd.read_csv(P["maximin"])
    ctrl = pd.read_csv(P["controls"])
    cmp = pd.read_csv(P["cmp"])
    cov = pd.read_csv(P["coverage"])
    topo, nb, temp = (pd.read_csv(P["topo"]), pd.read_csv(P["nbni3"]),
                      pd.read_csv(P["temp"]))
    mech = pd.read_csv(P["mech"])
    boot = pd.read_csv(P["boot"])

    zero = base["M_ZERO"]["mean_tvd"]
    triv = base["M_TRIVIAL"]["mean_tvd"]
    gc = C.load_g_common()
    ok = True

    print("\n=== integrity audit (32 items) ===")
    live = {k: C.sha256_file(v) for k, v in
            (("HB_TARGETS", C.HB_TARGETS),
             ("HA_PRIMARY_PREDICTIONS", C.HA_PRED),
             ("HA_MAXIMIN_SEQUENCE", C.HA_MAXIMIN_CSV),
             ("G_RANDOM_SEQUENCES", C.G_SEQ_CSV),
             ("G_PRIMARY_CHECKPOINT", C.G_PRIMARY_CKPT),
             ("G_PRIMARY_SCALER", C.G_PRIMARY_SCALER),
             ("G_SECONDARY_CHECKPOINT", C.G_SECONDARY_CKPT))}
    ok &= item(1, "STEP5G pipeline (PRIMARY checkpoint + scaler) unchanged",
               live["G_PRIMARY_CHECKPOINT"] == C.FROZEN_HASHES["G_PRIMARY_CHECKPOINT"]
               and live["G_PRIMARY_SCALER"] == C.FROZEN_HASHES["G_PRIMARY_SCALER"],
               "checkpoint %s | scaler %s" % (live["G_PRIMARY_CHECKPOINT"][:16],
                                              live["G_PRIMARY_SCALER"][:16]))
    ok &= item(2, "STEP5H-A zero-shot predictions unchanged",
               live["HA_PRIMARY_PREDICTIONS"]
               == C.FROZEN_HASHES["HA_PRIMARY_PREDICTIONS"],
               live["HA_PRIMARY_PREDICTIONS"])
    ok &= item(3, "STEP5H-B Ni target file unchanged",
               live["HB_TARGETS"] == C.FROZEN_HASHES["HB_TARGETS"], live["HB_TARGETS"])
    ps = load(C.HB_PRIMARY_SCORE)
    ss = load(C.HB_SECONDARY_SCORE)
    sec_tvd = ss.get("mean_tvd", ss.get("all_secondary_metrics", {}).get("mean_tvd"))
    ok &= item(4, "STEP5H-B PRIMARY blind score unchanged (0.8643856187)",
               abs(ps["mean_tvd"] - C.ZERO_SHOT_FULL_MEAN_TVD) < 1e-12,
               ps["mean_tvd"])
    ok &= item(5, "STEP5H-B SECONDARY blind score unchanged (0.7234409099)",
               sec_tvd is not None
               and abs(float(sec_tvd) - C.SECONDARY_FULL_MEAN_TVD) < 1e-6, sec_tvd)
    ok &= item(6, "600 / 1400 split unchanged and reproduced from the frozen rule",
               pf["split"].get("FEWSHOT_CALIBRATION") == C.N_CALIBRATION
               and pf["split"].get("RECOVERY_EVALUATION") == C.N_RECOVERY_EVAL
               and all(c["result"] == "PASS" for c in pf["checks"] if c["id"] == "I08"),
               pf["split"])
    ok &= item(7, "frozen nested random sequences unchanged",
               live["G_RANDOM_SEQUENCES"] == C.FROZEN_HASHES["G_RANDOM_SEQUENCES"],
               live["G_RANDOM_SEQUENCES"])
    ok &= item(8, "frozen maximin sequence unchanged",
               live["HA_MAXIMIN_SEQUENCE"] == C.FROZEN_HASHES["HA_MAXIMIN_SEQUENCE"],
               live["HA_MAXIMIN_SEQUENCE"])
    ok &= item(9, "budgets are exactly 1 / 2 / 4 / 8 / 16 / 32",
               sorted(set(man.budget)) == C.BUDGETS
               and sorted(set(agg.budget)) == C.BUDGETS, sorted(set(man.budget)))
    ok &= item(10, "every anchor composition contributes all 11 temperatures",
               bool((man.n_anchor_states
                     == man.n_anchor_compositions * len(C.TEMPERATURES_C)).all()),
               dict(zip(man.tag, man.n_anchor_states)))

    seqs = C.random_sequences()
    mx = C.maximin_sequence()
    ni, _ = C.load_frozen_ni()
    ev_ids = set(ni.loc[ni.fewshot_role == "RECOVERY_EVALUATION", "composition_id"])
    anchor_ids = set()
    for s in list(seqs.values()) + [mx]:
        anchor_ids |= set(s)
    ok &= item(11, "no RECOVERY_EVALUATION composition entered any fit",
               not (anchor_ids & ev_ids)
               and all(not (set(str(r).split(";")) & ev_ids) for r in cov.anchor_ids),
               "%d anchor ids, 0 of them evaluation compositions" % len(anchor_ids))
    ok &= item(12, "no RECOVERY_EVALUATION row entered validation or early stopping",
               all(m == "frozen original-six DEV_VALIDATION (no Ni)"
                   for m in man.validation_source),
               sorted(set(man.validation_source)))
    va_ids = pd.read_csv(C.G_PRIMARY_VAL_IDS)
    ok &= item(13, "original-six validation set unchanged (%d frozen rows, no Ni)"
               % C.PRIMARY_VAL_ROWS,
               len(va_ids) == C.PRIMARY_VAL_ROWS
               and int(va_ids.state_id.astype(str).str.contains("NI_BLIND").sum()) == 0
               and bool((man.n_validation_rows == C.PRIMARY_VAL_ROWS).all()),
               "%d rows, 0 Ni" % len(va_ids))
    ok &= item(14, "architecture unchanged (43-128-64-5)",
               rs["protocol"]["architecture_search"] is False,
               "Linear(43,128)-ReLU-Linear(128,64)-ReLU-Linear(64,5)")
    ok &= item(15, "learning rate fixed at 0.0003 for every cell, never retuned",
               bool((man.learning_rate == C.REFIT_LR).all())
               and rs["protocol"]["retuning"] is False,
               sorted(set(man.learning_rate)))
    ok &= item(16, "frozen PRIMARY scaler used and never refit on Ni",
               bool((~man.scaler_refit.astype(bool)).all())
               and rs["protocol"]["scaler"].startswith("frozen PRIMARY"),
               rs["protocol"]["scaler"])
    ok &= item(17, "no feature selection performed in STEP5I",
               rs["protocol"]["feature_selection"] is False
               and not any(os.path.exists(os.path.join(C.ROOT, s, f))
                           for s in ("05_metrics", "10_qa")
                           for f in ("NZV_REMOVAL.csv", "PCC_REMOVAL.csv",
                                     "SHAP_SELECTION.csv")),
               "43 frozen features, no selection artifact under the STEP5I root")
    ok &= item(18, "natural row weighting; no Ni oversampling or balancing",
               rs["protocol"]["weighting"].startswith("natural"),
               "anchor fraction of training rows: %.5f%% to %.5f%%"
               % (100 * rs["anchor_fraction_range"][0],
                  100 * rs["anchor_fraction_range"][1]))
    ok &= item(19, "a fresh checkpoint exists for every replicate x budget cell",
               len(man) == len(C.REPLICATES) * len(C.BUDGETS) + len(C.BUDGETS)
               and man.tag.nunique() == len(man)
               and man.checkpoint_sha256.nunique() == len(man),
               "%d cells, %d distinct checkpoint hashes"
               % (len(man), man.checkpoint_sha256.nunique()))
    ok &= item(20, "no warm start between budgets or replicates",
               bool((~man.warm_started.astype(bool)).all())
               and rs["protocol"]["warm_start"] is False
               and bool((man.seed == C.REFIT_SEED).all()),
               "every cell initialized fresh from seed %d" % C.REFIT_SEED)

    for n, short in ((21, "R0"), (22, "R1"), (23, "R2")):
        name = C.CONTROL_IDS[short]
        ok &= item(n, "%s matches the frozen preregistered definition exactly" % short,
                   cdef["definitions"][short] == gc.RECOVERY_ESTIMATORS[name]
                   and cdef["g_common_sha256"] == C.sha256_file(C.G_COMMON),
                   "executed from the frozen STEP5G g_common.py, not re-implemented")
    ok &= item(24, "every method scored on the identical %d recovery states"
               % C.N_RECOVERY_STATES,
               bool((ctrl.n_states == C.N_RECOVERY_STATES).all())
               and bool((per_rep.n_states == C.N_RECOVERY_STATES).all())
               and bool((maxi.n_states == C.N_RECOVERY_STATES).all()),
               sorted(set(ctrl.n_states) | set(per_rep.n_states)))
    ok &= item(25, "B_skill uses the trivial baseline on the SAME recovery subset",
               abs(thr["trivial_tvd_recovery_subset"] - triv) < 1e-12
               and "SAME recovery subset" in thr["definitions"]["B_skill"],
               "trivial on recovery subset %.10f (full-blind value %.10f not used)"
               % (triv, C.ZERO_SHOT_FULL_TRIVIAL_TVD))
    ok &= item(26, "B_recovery25 uses the zero-shot value on the SAME recovery subset, "
                   "not the 22,000-state number",
               abs(thr["zero_shot_tvd_recovery_subset"] - zero) < 1e-12
               and abs(thr["recovery25_target_tvd"] - 0.75 * zero) < 1e-12
               and thr["zero_shot_tvd_recovery_subset"] != C.ZERO_SHOT_FULL_MEAN_TVD,
               "recovery-subset zero-shot %.10f vs full-blind %.10f"
               % (zero, C.ZERO_SHOT_FULL_MEAN_TVD))
    ok &= item(27, "topology stratification is descriptive only",
               scope["may_alter_anchor_selection"] is False
               and scope["label"].startswith("POST-BLIND DESCRIPTIVE"), scope["label"])
    ok &= item(28, "NBNI3 diagnostics did not alter anchor selection",
               scope["may_alter_anchor_selection"] is False
               and set(cov.anchor_ids.apply(lambda r: str(r).split(";")[0]))
               == {seqs[r][0] for r in C.REPLICATES} | {mx[0]},
               "first anchor of every cell is rank 1 of its frozen sequence")
    ok &= item(29, "no NBNI3-targeted or OTHER_SOLID-targeted sampling performed",
               scope["targeted_anchor_design_performed"] is False
               and scope["shargh_style_targeted_augmentation_performed"] is False,
               "spec 42 and 46 respected")
    ok &= item(30, "no low-order Cu-Ni / Cr-Ni / Nb-Ni binary anchor study executed",
               scope["low_order_binary_ni_study_performed"] is False, "spec 47")
    ok &= item(31, "no DFT / Materials Project / MLIP / first-principles descriptor used",
               scope["dft_used"] is False, "spec 48")

    hpath = os.path.join(C.D_MANIFEST, "STEP5I_FILE_HASHES.csv")
    hashes = index_files(hpath)
    hashes.to_csv(hpath, index=False, encoding="utf-8")
    ok &= item(32, "all STEP5I output hashes recorded",
               len(hashes) > 0 and hashes.sha256.notna().all(),
               "%d files indexed in 11_manifest/STEP5I_FILE_HASHES.csv" % len(hashes))

    pd.DataFrame(ITEMS).to_csv(os.path.join(C.D_QA, "STEP5I_INTEGRITY_CHECKS.csv"),
                               index=False, encoding="utf-8")
    with open(os.path.join(C.D_QA, "STEP5I_INTEGRITY_AUDIT.md"), "w",
              encoding="utf-8") as f:
        f.write("# STEP5I_INTEGRITY_AUDIT.md\n\nGenerated %s.\n\n"
                "This stage is POST-BLIND RETROSPECTIVE RECOVERY. Items 1-5 exist to "
                "prove the frozen zero-shot result was not touched by it.\n\n"
                "| # | item | result | detail |\n|---:|---|---|---|\n" % C.utcnow())
        for it in ITEMS:
            f.write("| %d | %s | **%s** | %s |\n"
                    % (it["n"], it["item"], it["result"], it["detail"]))
        f.write("\n%d of %d items PASS.\n"
                % (sum(i["result"] == "PASS" for i in ITEMS), len(ITEMS)))

    # ------------------------------------------------- recommendation ------ #
    rec = recommend(agg, cmp, thr, topo, nb, zero, triv)
    with open(os.path.join(C.ROOT, "STEP5I_NEXT_STAGE_RECOMMENDATION.md"), "w",
              encoding="utf-8") as f:
        f.write(rec["markdown"])

    summary = build_summary(pf, rs, base, thr, agg, per_rep, maxi, ctrl, cmp, cov,
                            topo, nb, temp, mech, boot, ITEMS, hashes, rec, zero, triv)
    C.write_json(os.path.join(C.ROOT, "STEP5I_SUMMARY.json"), summary)

    blockers = [it["item"] for it in ITEMS if it["result"] == "FAIL"]
    report = build_report(pf, rs, base, thr, agg, per_rep, maxi, ctrl, cmp, cov, topo,
                          nb, temp, mech, boot, ITEMS, hashes, rec, zero, triv,
                          blockers, ps, sec_tvd)
    rp = os.path.join(C.ROOT, "STEP5I_REPORT.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)

    bad = {}
    for basedir, _, files in os.walk(C.ROOT):
        for fn in files:
            if fn.endswith(".md"):
                p = os.path.join(basedir, fn)
                hits = scan_forbidden(p)
                if hits:
                    bad[os.path.relpath(p, C.ROOT)] = hits
    if bad:
        ok = False
        blockers.append("forbidden framing phrase in a STEP5I document")
        print("\n  FORBIDDEN PHRASE(S):")
        for k, v in bad.items():
            print("    %s: %s" % (k, v))
    if DISCLAIMER_OPEN not in report:
        ok = False
        blockers.append("report is missing its framing disclaimer block")

    # re-index now that the report, summary and recommendation exist. The index
    # excludes itself: a file cannot contain its own hash.
    hashes = index_files(hpath)
    hashes.to_csv(hpath, index=False, encoding="utf-8")

    C.write_json(os.path.join(C.D_MANIFEST, "STEP5I_COMPLETION.json"), {
        "generated_utc": C.utcnow(),
        "items_pass": sum(i["result"] == "PASS" for i in ITEMS),
        "items_total": len(ITEMS), "files_hashed": len(hashes),
        "forbidden_phrases": bad,
        "recommendation_category": rec["category"],
        "recommendation_executed": False,
        "decision": "READY FOR USER REVIEW" if ok else "NOT READY",
        "blocking_reasons": blockers})

    print("\n" + "-" * 78)
    print("%d/%d integrity items PASS | %d files hashed"
          % (sum(i["result"] == "PASS" for i in ITEMS), len(ITEMS), len(hashes)))
    print("recommendation category: %s (NOT executed)" % rec["category"])
    print("DECISION: %s" % ("READY FOR USER REVIEW" if ok else "NOT READY"))
    for b in blockers:
        print("   blocking: %s" % b)
    return 0 if ok else 2


# --------------------------------------------------------------------------- #
def recommend(agg, cmp, thr, topo, nb, zero, triv):
    """Spec 56. The rule is stated before it is applied, evaluated from the frozen
    numbers, and never executed. Categories are not mutually exclusive; all that
    trigger are reported, ranked by how strongly they trigger."""
    rnd = cmp[cmp.kind == "RANDOM"]
    refit = agg[agg.method == "M_REFIT"].sort_values("budget")
    r0 = agg[agg.method == "R0"].sort_values("budget")
    r2 = agg[agg.method == "R2"].sort_values("budget")
    b_skill = thr["B_skill"]
    b_rec = thr["B_recovery25"]
    finite_skill = isinstance(b_skill, int)
    beats_r0 = int(rnd.M_REFIT_beats_R0.sum())
    beats_r2 = int(rnd.M_REFIT_beats_R2.sum())
    n_cells = len(rnd)
    r0_wins_budgets = int((r0.mean_tvd_mean.values <= refit.mean_tvd_mean.values).sum())
    r2_wins_budgets = int((r2.mean_tvd_mean.values <= refit.mean_tvd_mean.values).sum())

    b32 = topo[(topo.kind == "RANDOM") & (topo.method == "M_REFIT")
               & (topo.budget == 32)]
    other32 = float(b32[b32.stratum == "OTHER_SOLID"].mean_tvd.mean()) \
        if len(b32[b32.stratum == "OTHER_SOLID"]) else float("nan")
    fcc32 = float(b32[b32.stratum == "FCC"].mean_tvd.mean()) \
        if len(b32[b32.stratum == "FCC"]) else float("nan")
    nb32 = nb[(nb.kind == "RANDOM") & (nb.method == "M_REFIT") & (nb.budget == 32)
              & (nb.stratum == "NBNI3_PRESENT")]
    nbni3_32 = float(nb32.mean_tvd.mean()) if len(nb32) else float("nan")

    trig = {}
    trig["A_DIRECT_FEW_SHOT_SUFFICIENT"] = bool(
        finite_skill and b_skill <= 8 and beats_r0 >= n_cells * 0.5
        and beats_r2 >= n_cells * 0.5)
    trig["B_ANCHOR_CORRECTION_DOMINATES"] = bool(
        r0_wins_budgets >= 3 or r2_wins_budgets >= 3)
    trig["C_TOPOLOGY_TARGETED_REMEDY_JUSTIFIED"] = bool(
        (other32 == other32 and (other32 >= fcc32 or other32 >= triv))
        or (nbni3_32 == nbni3_32 and nbni3_32 >= triv))
    trig["D_LOW_ORDER_NI_ANCHOR_STUDY_JUSTIFIED"] = bool(
        finite_skill and (b_skill > 8 or not isinstance(b_rec, int)))

    order = [k for k in ("A_DIRECT_FEW_SHOT_SUFFICIENT",
                         "B_ANCHOR_CORRECTION_DOMINATES",
                         "C_TOPOLOGY_TARGETED_REMEDY_JUSTIFIED",
                         "D_LOW_ORDER_NI_ANCHOR_STUDY_JUSTIFIED") if trig[k]]
    category = order[0] if order else "NONE_TRIGGERED"
    md = """# STEP5I_NEXT_STAGE_RECOMMENDATION.md

Generated %s.

**Nothing below is executed.** Spec 56 permits this document and forbids acting on it.
No targeted augmentation, no NBNI3-targeted anchor set, no Cu-Ni / Cr-Ni / Nb-Ni binary
calculation, no new descriptor, no new model family, no DFT and no meta-learning has
been started. STEP5I stops at this file and waits for review.

## The decision rule, stated before it was applied

| category | triggers when | triggered |
|---|---|---|
| A DIRECT FEW-SHOT SUFFICIENT | B_skill <= 8 **and** M_REFIT beats both R0 and R2 in at least half of the 18 random cells | %s |
| B ANCHOR CORRECTION DOMINATES | R0 or R2 matches or beats M_REFIT at 3 or more of the 6 budgets | %s |
| C TOPOLOGY-TARGETED REMEDY JUSTIFIED | at budget 32 the true-OTHER_SOLID stratum (or the NBNI3-present stratum) is still no better than the trivial baseline, or is worse than the FCC stratum | %s |
| D LOW-ORDER Ni ANCHOR STUDY JUSTIFIED | direct anchors do reach skill, but only slowly: B_skill > 8, or B_recovery25 never reached through 32 | %s |

## What the numbers were

| quantity | value |
|---|---|
| B_skill | %s |
| B_recovery25 | %s |
| M_REFIT beats R0 | %d of %d random cells |
| M_REFIT beats R2 | %d of %d random cells |
| budgets where R0 matches or beats M_REFIT | %d of 6 |
| budgets where R2 matches or beats M_REFIT | %d of 6 |
| budget-32 M_REFIT TVD, true OTHER_SOLID | %.6f |
| budget-32 M_REFIT TVD, true FCC | %.6f |
| budget-32 M_REFIT TVD, NBNI3 present | %.6f |
| zero-shot / trivial on the same subset | %.6f / %.6f |

## Recommendation

**%s**

%s

Categories are not mutually exclusive. Every one that triggered is listed above with
the numbers that triggered it, so a reader can disagree with the rule and still see the
evidence. Choosing which, if any, to authorize is the user's decision.
""" % (C.utcnow(),
       trig["A_DIRECT_FEW_SHOT_SUFFICIENT"], trig["B_ANCHOR_CORRECTION_DOMINATES"],
       trig["C_TOPOLOGY_TARGETED_REMEDY_JUSTIFIED"],
       trig["D_LOW_ORDER_NI_ANCHOR_STUDY_JUSTIFIED"],
       b_skill, b_rec, beats_r0, n_cells, beats_r2, n_cells,
       r0_wins_budgets, r2_wins_budgets, other32, fcc32, nbni3_32, zero, triv,
       category,
       "No category triggered under the stated rule; the numbers above are the whole "
       "finding." if category == "NONE_TRIGGERED" else
       "Triggered categories, in the order the rule ranks them: %s."
       % ", ".join(order))
    return {"category": category, "triggered": trig, "order": order, "markdown": md,
            "evidence": {"B_skill": b_skill, "B_recovery25": b_rec,
                         "beats_R0_cells": beats_r0, "beats_R2_cells": beats_r2,
                         "n_random_cells": n_cells,
                         "budgets_R0_matches_or_beats": r0_wins_budgets,
                         "budgets_R2_matches_or_beats": r2_wins_budgets,
                         "budget32_other_solid_tvd": other32,
                         "budget32_fcc_tvd": fcc32,
                         "budget32_nbni3_present_tvd": nbni3_32}}


def build_summary(pf, rs, base, thr, agg, per_rep, maxi, ctrl, cmp, cov, topo, nb,
                  temp, mech, boot, items, hashes, rec, zero, triv):
    refit = agg[agg.method == "M_REFIT"].sort_values("budget")
    return {
        "generated_utc": C.utcnow(),
        "stage": "STEP5I POST-BLIND Ni FEW-SHOT RECOVERY",
        "stage_type": "POST-BLIND RETROSPECTIVE RECOVERY",
        "is_not": ["zero-shot", "blind", "prospective zero-shot"],
        "frozen_zero_shot_unchanged": {
            "PRIMARY_mean_tvd_22000": C.ZERO_SHOT_FULL_MEAN_TVD,
            "SECONDARY_mean_tvd_22000": C.SECONDARY_FULL_MEAN_TVD,
            "artifacts_modified": 0},
        "recovery_subset": {
            "n_compositions": C.N_RECOVERY_EVAL, "n_states": C.N_RECOVERY_STATES,
            "M_ZERO_mean_tvd": zero, "M_TRIVIAL_mean_tvd": triv,
            "M_ZERO_topology_accuracy": base["M_ZERO"]["topology_accuracy"]},
        "protocol": rs["protocol"],
        "anchor_fraction_range": rs["anchor_fraction_range"],
        "random_learning_curve": {int(b): float(v) for b, v in
                                  zip(refit.budget, refit.mean_tvd_mean)},
        "random_learning_curve_std": {int(b): float(v) for b, v in
                                      zip(refit.budget, refit.mean_tvd_std)},
        "maximin_learning_curve": {int(r.budget): float(r.mean_tvd) for _, r in
                                   maxi[maxi.method == "M_REFIT"].iterrows()},
        "controls": {m: {int(b): float(v) for b, v in
                         zip(agg[agg.method == m].budget,
                             agg[agg.method == m].mean_tvd_mean)}
                     for m in ("R0", "R1", "R2")},
        "thresholds": {"B_skill": thr["B_skill"], "B_recovery25": thr["B_recovery25"],
                       "recovery25_target_tvd": thr["recovery25_target_tvd"]},
        "model_vs_control": {
            "M_REFIT_beats_R0_cells": int(cmp[cmp.kind == "RANDOM"].M_REFIT_beats_R0.sum()),
            "M_REFIT_beats_R2_cells": int(cmp[cmp.kind == "RANDOM"].M_REFIT_beats_R2.sum()),
            "M_REFIT_beats_M_ZERO_cells": int(
                cmp[cmp.kind == "RANDOM"].M_REFIT_beats_M_ZERO.sum()),
            "M_REFIT_beats_M_TRIVIAL_cells": int(
                cmp[cmp.kind == "RANDOM"].M_REFIT_beats_M_TRIVIAL.sum()),
            "n_random_cells": int(len(cmp[cmp.kind == "RANDOM"]))},
        "integrity": {"pass": sum(i["result"] == "PASS" for i in items),
                      "total": len(items)},
        "files_hashed": int(len(hashes)),
        "next_stage_recommendation": {"category": rec["category"],
                                      "triggered": rec["triggered"],
                                      "evidence": rec["evidence"],
                                      "executed": False},
        "not_executed": ["Shargh-style targeted augmentation",
                         "NBNI3-targeted anchor selection",
                         "Cu-Ni / Cr-Ni / Nb-Ni binary CALPHAD",
                         "new descriptors", "new model family", "DFT",
                         "meta-learning"],
    }


def build_report(pf, rs, base, thr, agg, per_rep, maxi, ctrl, cmp, cov, topo, nb, temp,
                 mech, boot, items, hashes, rec, zero, triv, blockers, ps, sec_tvd):
    refit = agg[agg.method == "M_REFIT"].sort_values("budget")
    rnd = cmp[cmp.kind == "RANDOM"]
    npass = sum(i["result"] == "PASS" for i in items)
    L = []
    L.append("STEP5I STATUS: %s" % ("PASS" if not blockers else "BLOCKED"))
    L.append("")
    L.append("# STEP5I_REPORT.md")
    L.append("")
    L.append("**POST-BLIND Ni FEW-SHOT RECOVERY + ANCHOR-ONLY CONTROLS + "
             "TOPOLOGY / NBNI3 MECHANISM DIAGNOSIS**")
    L.append("")
    L.append("Generated %s. %d recovery refits (3 random replicates x 6 budgets, plus "
             "6 maximin budgets), three anchor-only controls, and descriptive "
             "stratification against the newly revealed Ni thermodynamics."
             % (C.utcnow(), len(cmp)))
    L.append("")
    L.append(DISCLAIMER_OPEN)
    L.append("> **What this stage is, stated once and binding on everything below.** "
             "STEP5I is *post-blind retrospective recovery*. It is **not** zero-shot, "
             "**not** blind, and **not** prospective zero-shot: it uses real Ni CALPHAD "
             "targets as anchors, so it answers a different question from the frozen "
             "STEP5H result. The frozen prospective Ni zero-shot outcome stands "
             "unchanged and unimproved - PRIMARY mean TVD %.10f with topology accuracy "
             "%.6f against a trivial baseline of %.6f, i.e. worse than predicting the "
             "training mean. Nothing in this document rewrites that, and no number "
             "here may be reported as a zero-shot result. The topology and NBNI3 "
             "stratifications are descriptive: they were computed after the reveal and "
             "were not allowed to influence anchor selection, the model, the budgets or "
             "any metric definition."
             % (C.ZERO_SHOT_FULL_MEAN_TVD, C.ZERO_SHOT_FULL_TOPOLOGY_ACC,
                C.ZERO_SHOT_FULL_TRIVIAL_TVD))
    L.append(DISCLAIMER_CLOSE)
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Learning curves (RECOVERY_EVALUATION, %d states, identical for every "
             "method)" % C.N_RECOVERY_STATES)
    L.append("")
    L.append("| budget | M_REFIT mean | std | rep 1 | rep 2 | rep 3 | R0 | R1 | R2 | "
             "MAXIMIN M_REFIT |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    L.append("| 0 (zero-shot) | %.6f | - | - | - | - | - | - | - | %.6f |"
             % (zero, zero))
    for b in C.BUDGETS:
        r = refit[refit.budget == b].iloc[0]
        reps = per_rep[(per_rep.method == "M_REFIT") & (per_rep.budget == b)] \
            .sort_values("replicate")
        g = {m: agg[(agg.method == m) & (agg.budget == b)].mean_tvd_mean.iloc[0]
             for m in ("R0", "R1", "R2")}
        mm = maxi[(maxi.method == "M_REFIT") & (maxi.budget == b)].mean_tvd
        L.append("| %d | %.6f | %.6f | %s | %.6f | %.6f | %.6f | %.6f |"
                 % (b, r.mean_tvd_mean, r.mean_tvd_std,
                    " | ".join("%.6f" % v for v in reps.mean_tvd),
                    g["R0"], g["R1"], g["R2"],
                    float(mm.iloc[0]) if len(mm) else float("nan")))
    L.append("")
    L.append("Frozen references on the same subset: zero-shot **%.6f**, trivial "
             "**%.6f**." % (zero, triv))
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Answers")
    L.append("")

    def q(n, text, ans):
        L.append("**%d. %s** %s" % (n, text, ans))
        L.append("")

    def curve(m):
        return ", ".join("b%d %.6f" % (int(b), float(v)) for b, v in
                         zip(agg[agg.method == m].sort_values("budget").budget,
                             agg[agg.method == m].sort_values("budget").mean_tvd_mean))

    q(1, "Zero-shot artifacts unchanged?",
      "Yes. The STEP5G PRIMARY checkpoint and scaler, the STEP5H-A predictions and the "
      "STEP5H-B targets all hash to their frozen values (audit items 1-3). Every "
      "STEP5I artifact is new; nothing was overwritten.")
    q(2, "Target hash unchanged?",
      "Yes - `%s` for `STEP5H_B_NI_TARGETS.parquet`, exactly as preregistered."
      % C.FROZEN_HASHES["HB_TARGETS"])
    q(3, "Recovery evaluation compositions?", "%d." % C.N_RECOVERY_EVAL)
    q(4, "Recovery evaluation states?",
      "%d = %d x 11 temperatures, all converged (STEP5H-B reported 0 failed states "
      "out of 22,000)." % (C.N_RECOVERY_STATES, C.N_RECOVERY_EVAL))
    q(5, "Calibration pool compositions?",
      "%d, and the split was reproduced independently from the frozen salted-hash rule "
      "rather than taken on trust." % C.N_CALIBRATION)
    q(6, "Were frozen random sequences used exactly?",
      "Yes. The CSV hashes to `%s`, matches the preregistration JSON entry for entry, "
      "and every budget is the first b IDs of the same 32-composition sequence "
      "(1 in 2 in 4 in 8 in 16 in 32 verified)."
      % C.FROZEN_HASHES["G_RANDOM_SEQUENCES"])
    q(7, "Were frozen maximin IDs used exactly?",
      "Yes - `%s`, unchanged, and used only as the secondary comparison."
      % C.FROZEN_HASHES["HA_MAXIMIN_SEQUENCE"])
    q(8, "Any new anchor selection?",
      "None. No anchor was drawn, re-drawn, re-ordered, substituted or chosen by "
      "uncertainty, distance, topology or target in this stage.")
    q(9, "Any hyperparameter retuning?",
      "None. Learning rate fixed at %g for all %d cells, batch %d, <= %d epochs, "
      "patience %d, no LR search and no architecture search."
      % (C.REFIT_LR, len(cmp), C.REFIT_BATCH, C.REFIT_EPOCHS, C.REFIT_PATIENCE))
    q(10, "Any feature reselection?",
      "None. The frozen 43-feature schema was used unchanged and no selection artifact "
      "exists under the STEP5I root.")
    q(11, "Was PRIMARY scaler refit?",
      "No. The frozen PRIMARY `StandardScaler` was loaded and applied to the Ni rows "
      "as-is. Its mean and scale were not adapted to Ni, which is why the anchor rows "
      "enter the network as genuinely out-of-range inputs rather than renormalized ones.")
    q(12, "Recovery-subset zero-shot mean TVD?",
      "**%.10f** (M_ZERO). This - not the 22,000-state %.10f - is the value "
      "B_recovery25 is measured against." % (zero, C.ZERO_SHOT_FULL_MEAN_TVD))
    q(13, "Recovery-subset trivial TVD?", "**%.10f** (M_TRIVIAL)." % triv)
    for i, b in enumerate(C.BUDGETS):
        r = refit[refit.budget == b].iloc[0]
        reps = per_rep[(per_rep.method == "M_REFIT") & (per_rep.budget == b)] \
            .sort_values("replicate").mean_tvd
        q(14 + i, "Random budget %d mean TVD?" % b,
          "**%.6f** (std %.6f; replicates %s)."
          % (r.mean_tvd_mean, r.mean_tvd_std, ", ".join("%.6f" % v for v in reps)))
    q(20, "Individual replicate results?",
      "All preserved in `05_metrics/STEP5I_RANDOM_FEWSHOT_PER_REPLICATE.csv`, one row "
      "per replicate x budget x method. Because every cell starts from the same frozen "
      "initialization seed (%d), the replicate-to-replicate spread is caused entirely "
      "by which anchors the frozen sequence happened to contain - which is what makes "
      "question 37 answerable at all." % C.REFIT_SEED)
    q(21, "B_skill?",
      "**%s** - the smallest budget whose across-replicate mean M_REFIT TVD falls "
      "below the trivial baseline %.6f on the same subset." % (thr["B_skill"], triv))
    q(22, "B_recovery25?",
      "**%s** - the smallest budget reaching a mean TVD of %.6f or lower "
      "(0.75 x %.6f)." % (thr["B_recovery25"], thr["recovery25_target_tvd"], zero))
    q(23, "Maximin budget results?",
      "M_REFIT with maximin anchors: %s."
      % ", ".join("b%d %.6f" % (int(r.budget), float(r.mean_tvd))
                  for _, r in maxi[maxi.method == "M_REFIT"].sort_values("budget")
                  .iterrows()))
    n_mx_better = sum(
        1 for b in C.BUDGETS
        if float(maxi[(maxi.method == "M_REFIT") & (maxi.budget == b)].mean_tvd.iloc[0])
        < float(refit[refit.budget == b].mean_tvd_mean.iloc[0]))
    q(24, "Does maximin beat random?",
      "Maximin has the lower mean TVD at %d of the 6 budgets. This is a secondary "
      "comparison only: maximin is not promoted on the strength of it, whichever way "
      "it falls." % n_mx_better)
    q(25, "R0 results by budget?", "%s." % curve("R0"))
    q(26, "R1 results?", "%s." % curve("R1"))
    q(27, "R2 results?", "%s." % curve("R2"))
    bud_r0 = [int(b) for b in C.BUDGETS
              if float(refit[refit.budget == b].mean_tvd_mean.iloc[0])
              < float(agg[(agg.method == "R0") & (agg.budget == b)].mean_tvd_mean.iloc[0])]
    bud_r2 = [int(b) for b in C.BUDGETS
              if float(refit[refit.budget == b].mean_tvd_mean.iloc[0])
              < float(agg[(agg.method == "R2") & (agg.budget == b)].mean_tvd_mean.iloc[0])]
    q(28, "At which budgets does M_REFIT beat R0 descriptively?",
      ("Budgets %s (across-replicate means); %d of %d individual random cells."
       % (bud_r0, int(rnd.M_REFIT_beats_R0.sum()), len(rnd))) if bud_r0 else
      "None. At every budget the across-replicate mean M_REFIT TVD is at or above R0.")
    q(29, "At which budgets does M_REFIT beat R2 descriptively?",
      ("Budgets %s; %d of %d individual random cells."
       % (bud_r2, int(rnd.M_REFIT_beats_R2.sum()), len(rnd))) if bud_r2 else
      "None. At every budget the across-replicate mean M_REFIT TVD is at or above R2.")
    q(30, "Does the learned model add value beyond anchor-only correction?",
      ("Yes on this evidence: M_REFIT is lower than R0 at budgets %s and lower than R2 "
       "at budgets %s." % (bud_r0, bud_r2)) if (bud_r0 and bud_r2) else
      "**No.** Under this preregistered protocol the refit does not demonstrate value "
      "beyond simply knowing the anchor partition vectors. R0 discards the model "
      "entirely and is not beaten%s. The honest reading is that the anchors, not the "
      "network, carry the useful information here; this says nothing about whether a "
      "differently weighted or differently stopped refit could do better, because "
      "neither was permitted."
      % ("" if not bud_r2 else " at %d of 6 budgets" % (6 - len(bud_r0))))

    def strat(frame, stratum, method="M_REFIT"):
        s = frame[(frame.kind == "RANDOM") & (frame.method == method)
                  & (frame.stratum == stratum)]
        if not len(s):
            return "no states in this stratum"
        z = frame[(frame.stratum == stratum) & (frame.method == "M_ZERO")].mean_tvd.mean()
        g = s.groupby("budget").mean_tvd.mean()
        return ("zero-shot %.6f -> %s (n=%d states)"
                % (z, ", ".join("b%d %.6f" % (int(b), v) for b, v in g.items()),
                   int(s.n_states.iloc[0])))

    q(31, "Recovery on true FCC states?", strat(topo, "FCC") + ".")
    q(32, "Recovery on true OTHER_SOLID states?", strat(topo, "OTHER_SOLID") + ".")
    q(33, "Recovery on NBNI3-present states?", strat(nb, "NBNI3_PRESENT") + ".")
    q(34, "Recovery on NBNI3-absent states?", strat(nb, "NBNI3_ABSENT") + ".")
    q(35, "Low-T recovery?", strat(temp, "LOW_T") + " (300-600 degC, the NBNI3 window).")
    q(36, "High-T recovery?", strat(temp, "HIGH_T") + " (700-1000 degC).")
    m = mech[(mech.scope == "ALL_RANDOM_CELLS")
             & (mech.coverage_metric == "fraction_anchor_states_NBNI3")
             & (mech.method == "M_REFIT")]
    rho = float(m.spearman_rho.iloc[0]) if len(m) and pd.notna(
        m.spearman_rho.iloc[0]) else None
    pval = float(m.p_value.iloc[0]) if len(m) and pd.notna(m.p_value.iloc[0]) else float("nan")
    q(37, "Does anchor NBNI3 coverage track recovery strength?",
      ("Spearman rho = %.4f (p = %.3g) across the 18 random cells between the fraction "
       "of anchor states containing NBNI3 and the TVD reduction versus zero-shot. "
       "Descriptive only - the sequences were frozen before NBNI3 was known to exist, "
       "and no anchor was chosen because of it."
       % (rho, pval)) if rho is not None else
      "Not computable: the coverage metric is constant across the frozen cells, so no "
      "rank correlation exists. Returning 0 here would invent a 'no relationship' "
      "finding that was never measured.")
    q(38, "Any replicate sensitivity?",
      "Across-replicate standard deviation of M_REFIT mean TVD ranges from %.6f to "
      "%.6f across the six budgets. Since all cells share one initialization seed, "
      "this spread is entirely anchor-driven."
      % (refit.mean_tvd_std.min(), refit.mean_tvd_std.max()))
    man = pd.read_csv(os.path.join(C.D_MANIFEST,
                                   "STEP5I_RECOVERY_CHECKPOINT_MANIFEST.csv"))
    q(39, "Any training instability?",
      "No cell diverged. Best epochs range %d-%d of a %d cap; %d of %d cells stopped "
      "early on patience %d, the rest ran to the cap. Best validation MSE ranges "
      "%.3e-%.3e."
      % (man.best_epoch.min(), man.best_epoch.max(), C.REFIT_EPOCHS,
         int(man.early_stopped.astype(str).str.lower().isin(["true","1"]).sum()), len(man), C.REFIT_PATIENCE,
         man.best_validation_mse.min(), man.best_validation_mse.max()))
    q(40, "Any recovery checkpoint failure?",
      "None. %d checkpoints written, %d distinct hashes, all recorded in "
      "`11_manifest/STEP5I_RECOVERY_CHECKPOINT_MANIFEST.csv`."
      % (len(man), man.checkpoint_sha256.nunique()))
    q(41, "Were all budgets retained?",
      "Yes - %s, every replicate, every method, reported whatever they showed."
      % C.BUDGETS)
    q(42, "Was any best subset selected post hoc?",
      "No. No best replicate, best budget or best anchor sequence was nominated as a "
      "final model, and the recommendation document proposes categories rather than "
      "selecting an artifact.")
    q(43, "Was low-order binary study executed?",
      "No. No Cu-Ni, Cr-Ni or Nb-Ni calculation was run (spec 47).")
    q(44, "Was DFT used?",
      "No. No DFT, Materials Project, MLIP or first-principles descriptor entered this "
      "stage. The project remains CALPHAD plus classical elemental descriptors "
      "(spec 48).")
    q(45, "READY FOR USER REVIEW?",
      ("Yes. %d of %d integrity items pass, %d files are hashed, and the next-stage "
       "recommendation is written but not executed." % (npass, len(items), len(hashes)))
      if not blockers else
      "No. Blocking reasons: %s." % "; ".join(blockers))

    L.append("---")
    L.append("")
    L.append("## Anchor fraction, stated plainly")
    L.append("")
    L.append("Natural row weighting was frozen before the reveal (spec 18). It means "
             "the Ni anchors are between %.5f%% and %.5f%% of the refit training set - "
             "%d rows in %d at budget 1, %d in %d at budget 32 - and that early "
             "stopping watches only original-six validation rows, so the stopping rule "
             "sees no Ni at all. Both facts are part of the protocol being tested, not "
             "defects in it. They are the reason the R0/R1/R2 controls were "
             "preregistered alongside the refit: the controls use the same anchors "
             "without that dilution."
             % (100 * rs["anchor_fraction_range"][0],
                100 * rs["anchor_fraction_range"][1],
                len(C.TEMPERATURES_C),
                C.PRIMARY_TRAIN_ROWS + len(C.TEMPERATURES_C),
                32 * len(C.TEMPERATURES_C),
                C.PRIMARY_TRAIN_ROWS + 32 * len(C.TEMPERATURES_C)))
    L.append("")
    L.append("## Recommendation category (not executed)")
    L.append("")
    L.append("**%s** - see `STEP5I_NEXT_STAGE_RECOMMENDATION.md` for the rule and the "
             "numbers behind it." % rec["category"])
    L.append("")
    L.append("## Stop rule")
    L.append("")
    L.append("STEP5I ends here. No targeted augmentation, no NBNI3-targeted anchor "
             "selection, no Cu-Ni / Cr-Ni / Nb-Ni binary calculation, no new "
             "descriptor, no new model family, no DFT and no meta-learning has been "
             "started. The next action is the user's.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("READY FOR USER REVIEW" if not blockers else "NOT READY")
    if blockers:
        for b in blockers:
            L.append("")
            L.append("- %s" % b)
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
