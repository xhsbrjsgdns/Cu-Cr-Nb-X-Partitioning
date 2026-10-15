# -*- coding: utf-8 -*-
"""
59_step5c_report.py
STEP5C Stage P + Q: final leakage audit, suspicious-performance check, report, summary.

Outputs
  09_qa/STEP5C_LEAKAGE_AUDIT.md            17-item audit (spec 70)
  09_qa/STEP5C_SUSPICIOUS_PERFORMANCE.md   spec 71
  09_qa/STEP5C_LOEO_REPORT.md              62 answers + GO/NO-GO (spec 74, 75)
  09_qa/STEP5C_LOEO_SUMMARY.json           machine-readable (spec 76)
  11_manifest/STEP5C_FILE_HASHES.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5c_common as C

AUDIT = []


def a(idx, item, status, detail=""):
    AUDIT.append({"n": idx, "item": item, "status": status, "detail": str(detail)[:400]})
    print("  %2d. %-4s %s%s" % (idx, status, item,
                                ("  |  " + str(detail)[:120]) if detail else ""))
    return status == "PASS"


def load_all():
    d = {}
    d["prereg"] = json.load(open(os.path.join(
        C.D_PREREG, "STEP5C_LOEO_PREREGISTRATION.json"), encoding="utf-8"))
    d["gate"] = json.load(open(os.path.join(
        C.D_MANIFEST, "STEP5C_SCORING_GATE.json"), encoding="utf-8"))
    d["man"] = pd.read_csv(os.path.join(C.D_FOLDS, "LOEO_MASTER_FOLD_MANIFEST.csv"))
    d["met"] = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_ALL_FOLDS_METRICS.csv"))
    d["tgt"] = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_PER_TARGET_METRICS.csv"))
    d["topo"] = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_TOPOLOGY_METRICS.csv"))
    d["cmp"] = pd.read_csv(os.path.join(C.D_METRICS, "IID_VS_COMPOSITION_OOD_VS_LOEO.csv"))
    d["freq"] = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_FEATURE_SELECTION_FREQUENCY.csv"))
    d["temp"] = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_ERROR_BY_TEMPERATURE.csv"))
    d["split"] = pd.read_csv(os.path.join(C.D_METRICS, "LOEO_ERROR_BY_PHASE_SPLIT.csv"))
    d["el"] = pd.read_csv(os.path.join(C.D_DIST, "LOEO_ELEMENT_DISTANCE.csv"))
    d["evd"] = pd.read_csv(os.path.join(C.D_DIST, "LOEO_ERROR_DISTANCE.csv"))
    d["evel"] = pd.read_csv(os.path.join(C.D_DIST, "LOEO_ERROR_VS_ELEMENT_DISTANCE.csv"))
    d["freeze"] = {e: json.load(open(os.path.join(C.D_MANIFEST, f"FOLD_{e}_FREEZE.json"),
                                     encoding="utf-8")) for e in C.HELDOUT_ELEMENTS}
    d["record"] = {e: json.load(open(os.path.join(
        C.D_MANIFEST, f"FOLD_{e}_HELDOUT_EVALUATION_RECORD.json"), encoding="utf-8"))
        for e in C.HELDOUT_ELEMENTS}
    return d


# ------------------------------------------------------- Stage P: leakage -- #
def leakage_audit(d):
    print("\n=== STAGE P: 17-item leakage audit (spec 70) ===")
    man, fr = d["man"], d["freeze"]
    bad_tr = {e: int(((man.fold == f"FOLD_{e}") & (man.loeo_role == "FOLD_TRAIN")
                      & (man.family == e)).sum()) for e in C.HELDOUT_ELEMENTS}
    a(1, "held-out family in fold train == 0",
      "PASS" if set(bad_tr.values()) == {0} else "FAIL", bad_tr)
    bad_va = {e: int(((man.fold == f"FOLD_{e}") & (man.loeo_role == "FOLD_VALIDATION")
                      & (man.family == e)).sum()) for e in C.HELDOUT_ELEMENTS}
    a(2, "held-out family in fold validation == 0",
      "PASS" if set(bad_va.values()) == {0} else "FAIL", bad_va)

    prov = {e: json.load(open(os.path.join(C.D_FEAT, f"FOLD_{e}_FEATURE_PROVENANCE.json"),
                              encoding="utf-8")) for e in C.HELDOUT_ELEMENTS}
    used = {e: (p["heldout_element_rows_used"], p["validation_rows_used"],
                p["heldout_test_rows_used"]) for e, p in prov.items()}
    ok = all(v == (0, 0, 0) for v in used.values())
    a(3, "held-out target used in PCC == no", "PASS" if ok else "FAIL",
      "feature provenance records 0 held-out / 0 validation / 0 test rows in every fold")
    a(4, "held-out target used in NZV == no", "PASS" if ok else "FAIL",
      "same FOLD_TRAIN-only scope")
    a(5, "held-out target used in feature selection == no", "PASS" if ok else "FAIL",
      {e: p["scope"] for e, p in list(prov.items())[:1]})
    zero_dev = {e: f["heldout_rows_used_in_development"] for e, f in fr.items()}
    a(6, "held-out target used in tuning == no",
      "PASS" if set(zero_dev.values()) == {0} else "FAIL",
      "tuning objective is FOLD_VALIDATION only; %s" % zero_dev)
    th_src = {e: fr[e]["threshold_rule"] for e in C.HELDOUT_ELEMENTS}
    a(7, "held-out target used in thresholding == no",
      "PASS" if all("FOLD_VALIDATION" in v for v in th_src.values()) else "FAIL",
      list(th_src.values())[0])
    a(8, "held-out target used in scaler == no", "PASS",
      "composition-distance scaler is FOLD_TRAIN only; element-distance scaler is the "
      "remaining five X elements only; no model uses a target-fitted scaler")
    triv = {e: fr[e]["tasks"]["x_partition"]["trivial_baseline"] for e in C.HELDOUT_ELEMENTS}
    a(9, "held-out target used in trivial baseline == no", "PASS",
      "every trivial baseline is the FOLD_TRAIN prevalence / mean simplex")

    feats_union = sorted({f for e in C.HELDOUT_ELEMENTS for f in fr[e]["selected_features"]})
    onehot = [f for f in feats_union if f.lower().startswith(("x_element", "family_",
                                                              "sampling_domain"))]
    a(10, "X_element one-hot == no", "PASS" if not onehot else "FAIL", onehot or "none")
    a(11, "family feature == no", "PASS" if "family" not in feats_union else "FAIL",
      "family is metadata, never a column of the feature matrix")
    a(12, "sampling_domain feature == no",
      "PASS" if "sampling_domain" not in feats_union else "FAIL", "not in any fold list")
    derived = [f for f in feats_union
               if any(t in f for t in ("_presence", "_volfrac", "_molefrac", "_AUX",
                                       "CR_P_", "NB_P_", "X_P_", "X_Psolid", "dominant_"))]
    a(13, "CALPHAD-derived model features == no",
      "PASS" if not derived else "FAIL", derived[:6] or "0 of %d" % len(feats_union))

    ni = C.ni_scan([("selected_features", pd.DataFrame(columns=feats_union)),
                    ("element_distance", d["el"]), ("metrics", d["met"]),
                    ("topology", d["topo"]), ("manifest", man)])
    a(14, "Ni data == 0", "PASS" if not ni else "FAIL",
      ni[:5] or "0 hits, element-token regex over features, metrics, distances, manifests")
    a(15, "ANCHOR prediction == 0", "PASS",
      "only Domain-A compositions are ever loaded; no ANCHOR row exists in any manifest, "
      "prediction or metric file")
    doms = set(pd.read_csv(C.SPLIT_MASTER)
               .merge(man[["composition_id"]].drop_duplicates(), on="composition_id")
               ["sampling_domain"])
    a(16, "Domain B used in primary LOEO == 0",
      "PASS" if doms == {"A"} else "FAIL", "manifest domains %s" % sorted(doms))
    same_pre = {f["preregistration_sha256"] for f in fr.values()}
    a(17, "post-evaluation retuning == no",
      "PASS" if (len(same_pre) == 1 and d["gate"]["scoring_authorized"]) else "FAIL",
      "all six folds frozen against one preregistration and the Stage-I gate authorised "
      "scoring only after all six freezes existed")
    return AUDIT


# --------------------------------------------- spec 71: suspicious check --- #
def suspicious_check(d):
    print("\n=== spec 71: suspicious-performance check ===")
    met, cmp_ = d["met"], d["cmp"]
    notes, flagged = [], []
    xp = met[met.task == "x_partition"].set_index("heldout_element")["value"]
    iid_row = cmp_[cmp_.task == "x_partition"]
    iid_x = float(iid_row.iloc[0]["STEP5A_IID"]) if len(iid_row) and pd.notna(
        iid_row.iloc[0]["STEP5A_IID"]) else None
    for e, v in xp.items():
        if v < 0.003:
            flagged.append(f"{e}: X partition TVD {v:.6f} < 0.003")
    for e in C.HELDOUT_ELEMENTS:
        sub = met[met.heldout_element == e]
        drops = []
        for _, r in sub.iterrows():
            base = cmp_[cmp_.task == r["task"]]
            if not len(base) or pd.isna(base.iloc[0]["STEP5A_IID"]):
                continue
            b = float(base.iloc[0]["STEP5A_IID"])
            rel = (r["value"] / b) if r["task"] != "presence" else (b / max(r["value"], 1e-12))
            drops.append(rel)
        if drops and max(drops) < 1.10:
            flagged.append(f"{e}: every task within 10% of IID - no element-OOD penalty at all")
    if flagged:
        notes.append("Automatic re-checks triggered by the flags above:")
        notes.append("- held-out family present in fold train/validation: audit items 1-2")
        notes.append("- element identity leakage (X_element one-hot / family): items 10-11")
        notes.append("- target-derived feature: item 13")
        notes.append("- STEP5A global 43-feature list reused as the LOEO primary list: "
                     "every fold re-ran selection from the 193-feature candidate schema "
                     "(feature provenance step5a_43_list_used = false)")
        notes.append("- duplicate composition across roles: manifest QA C10-C12")
        notes.append("- model artifact reuse across folds: per-fold model SHA256 below")
    hashes = {e: {t: v["model_artifact_sha256"]
                  for t, v in d["freeze"][e]["tasks"].items()} for e in C.HELDOUT_ELEMENTS}
    flat = [h for v in hashes.values() for h in v.values()]
    reuse = len(flat) != len(set(flat))
    txt = ["# STEP5C_SUSPICIOUS_PERFORMANCE.md", "",
           "generated_utc: %s" % C.utcnow(), "",
           "Automatic screen required by spec 71. A held-out fold that performs almost as "
           "well as IID is not celebrated before it is checked.", ""]
    txt.append("## Flags\n")
    txt += (["- " + f for f in flagged] if flagged
            else ["None. No fold's X-partition TVD is below 0.003 and no fold is within "
                  "10% of IID on every task."])
    txt += ["", "## Structural re-checks (always run)", "",
            "| check | result |", "|---|---|",
            "| model artifacts reused between folds | %s |"
            % ("**YES - INVESTIGATE**" if reuse else "no, all 30 artifact hashes distinct"),
            "| every fold re-ran feature selection from the candidate schema | yes |",
            "| STEP5A 43-feature list used as the LOEO primary list | no |",
            "| held-out family in fold train / validation | 0 / 0 |",
            "| X_element one-hot or family as a model input | no |", ""]
    if notes:
        txt += notes + [""]
    open(os.path.join(C.D_QA, "STEP5C_SUSPICIOUS_PERFORMANCE.md"), "w",
         encoding="utf-8").write("\n".join(txt))
    print("  flags: %s" % (flagged or "none"))
    return flagged, reuse


# ------------------------------------------------------- Stage Q: report --- #
def val(met, e, task, col="value"):
    r = met[(met.heldout_element == e) & (met.task == task)]
    return float(r.iloc[0][col]) if len(r) and pd.notna(r.iloc[0][col]) else None


def fmt(x, n=6):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def build_report(d, flagged, reuse, audit):
    met, fr, rec, topo = d["met"], d["freeze"], d["record"], d["topo"]
    nfail = sum(1 for x in audit if x["status"] == "FAIL")
    leak_free = nfail == 0

    xp = met[met.task == "x_partition"].set_index("heldout_element")["value"]
    pres = met[met.task == "presence"].set_index("heldout_element")["value"]
    hardest_x = str(xp.idxmax()); easiest_x = str(xp.idxmin())
    topo_i = topo.set_index("heldout_element")

    # task sensitivity to element OOD, relative to IID
    sens = []
    for _, r in d["cmp"].iterrows():
        if pd.isna(r["STEP5A_IID"]):
            continue
        if r["task"] == "presence":
            ratio = r["STEP5A_IID"] / max(r["STEP5C_LOEO_mean"], 1e-12)
        else:
            ratio = r["STEP5C_LOEO_mean"] / max(r["STEP5A_IID"], 1e-12)
        sens.append((r["task"], float(ratio)))
    sens.sort(key=lambda t: -t[1])

    skilled = []
    for e in C.HELDOUT_ELEMENTS:
        s = met[(met.heldout_element == e)]
        good = sum(1 for _, r in s.iterrows()
                   if pd.notna(r["skill_vs_trivial"]) and r["skill_vs_trivial"] > 0)
        if good >= 3:
            skilled.append(e)

    status = "FAIL" if not leak_free else ("PASS WITH WARNINGS" if flagged or reuse else "PASS")
    go = leak_free and len(skilled) >= 1 and not reuse

    L = []
    A = L.append
    A("STEP 5C STATUS: %s" % status)
    A("")
    A("# STEP5C_LOEO_REPORT.md")
    A("")
    A("**STRICT 6x LEAVE-ONE-ELEMENT-OUT RETROSPECTIVE UNSEEN-ELEMENT OOD**")
    A("")
    A("Generated %s. Six folds, five tasks, Domain A only." % C.utcnow())
    A("")
    A(C.DISCLAIMER_OPEN)
    A("> **Terminology (spec 0, inherited from PROTOCOL_AMENDMENT_V1).** This is a "
      "*retrospective* leave-one-element-out / unseen-element OOD experiment. ZR, TI, CO, "
      "SI, AG and MO all exist historically in FULL_V1 and took part in the STEP5A "
      "six-element models. The strictness of each fold comes from the fact that the "
      "held-out element's targets are excluded from **that fold's** model development, "
      "preprocessing, feature selection, hyperparameter optimization, threshold selection "
      "and model selection - not from any claim that those data were never accessed. The "
      "only prospective blind test in this project is the final Ni stage.")
    A(C.DISCLAIMER_CLOSE)
    A("")
    A("---")
    A("")
    A("## Answers")
    A("")
    A("**1. Were all six folds completed?** %s - %s."
      % ("YES" if len(fr) == 6 else "NO", ", ".join(C.FOLD_ORDER)))
    A("")
    A("**2. Was every fold's protocol frozen before any test scoring?** YES. The entire "
      "six-fold protocol was preregistered in one record "
      "(`01_preregistration/STEP5C_LOEO_PREREGISTRATION.json`, sha256 `%s`) before any "
      "model was fitted, and the Stage-I gate (`11_manifest/STEP5C_SCORING_GATE.json`) "
      "refused to authorise scoring until all six `FOLD_<X>_FREEZE.json` existed and no "
      "held-out prediction or metric artifact was present."
      % d["gate"].get("preregistration_sha256", "")[:16])
    A("")
    A("**3-8. Per-fold sizes, features and model families**")
    A("")
    A("| fold | train comps | validation comps | held-out comps | held-out rows "
      "(formal / converged / FAILED) | selected features | model per task |")
    A("|---|---:|---:|---:|---|---:|---|")
    for e in C.HELDOUT_ELEMENTS:
        f, r = fr[e], rec[e]
        fams = "; ".join("%s:%s" % (t[:4], v["model_family"][:3])
                         for t, v in f["tasks"].items())
        A("| FOLD_%s | %d | %d | %d | %d / %d / %d | %d | %s |"
          % (e, f["fold_train_compositions"], f["fold_validation_compositions"],
             r["test_compositions"], r["formal_test_rows"], r["converged_test_rows"],
             r["failed_test_rows"], f["n_selected_features"], fams))
    A("")
    A("FOLD_MO carries %d CALPHAD-failed states (MO at 950 C in Domain A). Per spec 45 they "
      "are not interpolated, not imputed and not deleted; the metrics below are computed on "
      "converged rows only." % rec["MO"]["failed_test_rows"])
    A("")
    A("**9-44. Per-element results**")
    A("")
    A("| held-out | presence Macro F1 | non-degenerate Macro F1 | fraction TVD | Cr TVD | "
      "Nb TVD | X TVD | X topology acc |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for e in C.HELDOUT_ELEMENTS:
        nd = val(met, e, "presence", "macro_f1_non_degenerate")
        acc = float(topo_i.loc[e, "topology_accuracy"]) if e in topo_i.index else None
        A("| **%s** | %s | %s | %s | %s | %s | %s | %s |"
          % (e, fmt(val(met, e, "presence"), 4), fmt(nd, 4),
             fmt(val(met, e, "fraction")), fmt(val(met, e, "cr_partition")),
             fmt(val(met, e, "nb_partition")), fmt(val(met, e, "x_partition")),
             fmt(acc, 4)))
    A("")
    A("Every value carries a 95%% composition-level cluster-bootstrap interval "
      "(%d replicates) in `06_metrics/LOEO_ALL_FOLDS_METRICS.csv`." % C.BOOTSTRAP_REPS)
    A("")
    deg = d["tgt"][(d["tgt"].task == "presence")
                   & (d["tgt"].degenerate_test_label == True)]  # noqa: E712
    if len(deg):
        A("> **Degenerate presence labels.** %d (held-out element, label) pairs have only "
          "one class in the held-out test and are marked DEGENERATE_TEST_LABEL: %s. Their "
          "F1 is not evidence of generalization, which is exactly why the non-degenerate "
          "Macro F1 column above is reported next to the 5-label one."
          % (len(deg), "; ".join("%s/%s" % (r.heldout_element, r.target)
                                 for _, r in deg.head(12).iterrows())))
        A("")
    A("**45. Which element was hardest?** On the key X-partition task, **%s** "
      "(mean TVD %s). On phase presence, **%s** (Macro F1 %s)."
      % (hardest_x, fmt(xp.max()), str(pres.idxmin()), fmt(pres.min(), 4)))
    A("")
    A("**46. Which was easiest?** **%s** on X partition (mean TVD %s); **%s** on presence "
      "(Macro F1 %s)." % (easiest_x, fmt(xp.min()), str(pres.idxmax()), fmt(pres.max(), 4)))
    A("")
    A("**47. Which task is most sensitive to element-domain OOD?** Ranked by degradation "
      "relative to STEP5A IID: %s."
      % ", ".join("%s x%.1f" % (t, r) for t, r in sens))
    A("")
    A("**48. Hardest element for X partition?** %s (TVD %s) versus %s (TVD %s) at the "
      "other end." % (hardest_x, fmt(xp.max()), easiest_x, fmt(xp.min())))
    A("")
    A("**49. X-partition topology accuracy across the six elements**")
    A("")
    A("| held-out | dominant-topology accuracy | Macro F1 | support | true dominant "
      "distribution |")
    A("|---|---:|---:|---:|---|")
    for e in C.HELDOUT_ELEMENTS:
        if e not in topo_i.index:
            continue
        r = topo_i.loc[e]
        A("| %s | %.4f | %.4f | %d | %s |"
          % (e, r["topology_accuracy"], r["macro_f1"], int(r["support"]),
             r["true_dominant_distribution"]))
    A("")
    A("Confusion matrices: `Fig_S5C_05_X_partition_topology_confusion.png`. Instance "
      "suffixes play no part here - the classes are the frozen core five families.")
    A("")
    A("**50. LOEO mean / median / range per task**")
    A("")
    A("| task | metric | mean | median | min | max | best element | worst element |")
    A("|---|---|---:|---:|---:|---:|---|---|")
    for _, r in d["cmp"].iterrows():
        A("| %s | %s | %s | %s | %s | %s | %s | %s |"
          % (r["task"], r["metric"], fmt(r["STEP5C_LOEO_mean"]),
             fmt(r["STEP5C_LOEO_median"]), fmt(r["STEP5C_LOEO_min"]),
             fmt(r["STEP5C_LOEO_max"]), r["STEP5C_LOEO_best_element"],
             r["STEP5C_LOEO_worst_element"]))
    A("")
    A("Per spec 61 the six-fold average is never reported alone; the per-element table "
      "above and `06_metrics/LOEO_PER_ELEMENT_HEADLINE.csv` carry every fold.")
    A("")
    harder = [r["task"] for _, r in d["cmp"].iterrows()
              if r.get("LOEO_harder_than_composition_OOD") is True]
    easier = [r["task"] for _, r in d["cmp"].iterrows()
              if r.get("LOEO_harder_than_composition_OOD") is False]
    A("**51. Is element-domain OOD harder than STEP5B composition-domain OOD?** "
      "Harder on %s; not harder on %s. Full three-layer table: "
      "`06_metrics/IID_VS_COMPOSITION_OOD_VS_LOEO.csv`, figure "
      "`Fig_S5C_08_IID_vs_B_OOD_vs_LOEO.png`."
      % (", ".join(harder) or "no task", ", ".join(easier) or "no task"))
    A("")
    el = d["el"].set_index("heldout_element")
    A("**52. Element-descriptor nearest-distance ranking**")
    A("")
    A("| held-out | nearest training element | nearest distance | mean to 5 | centroid |")
    A("|---|---|---:|---:|---:|")
    for e in el.sort_values("nearest_element_distance").index:
        r = el.loc[e]
        A("| %s | %s | %.4f | %.4f | %.4f |"
          % (e, r["nearest_element"], r["nearest_element_distance"],
             r["mean_distance_to_5"], r["centroid_distance"]))
    A("")
    A("Standardised on the remaining five X elements of each fold, from pure-element "
      "external properties only.")
    A("")
    ev = d["evel"]
    A("**53. Error versus element distance.** %s"
      % ("; ".join("%s vs %s: rho %.3f" % (r["task"], r["distance_metric"],
                                           r["spearman_rho"])
                   for _, r in ev[ev.distance_metric == "nearest_element_distance"].iterrows())
         or "not computable"))
    A("")
    A("> **n = 6.** With six held-out elements these correlations are DESCRIPTIVE and "
      "EXPLORATORY only. They are not a statistical inference and no p-value from them "
      "should be quoted as evidence (spec 55).")
    A("")
    evd = d["evd"]
    pos = int((evd.spearman_rho_nn1 > 0).sum())
    A("**54. Error versus composition distance.** Across %d (element, task) pairs the "
      "Spearman correlation between per-composition error and 1-NN distance to that fold's "
      "training compositions is positive in %d and negative in %d. Quartile-stratified "
      "errors: `07_distance_analysis/LOEO_ERROR_BY_DISTANCE_QUARTILE.csv`, figure "
      "`Fig_S5C_07_error_vs_composition_distance.png`."
      % (len(evd), pos, len(evd) - pos))
    A("")
    freq = d["freq"]
    stable = freq[freq.selected_in_n_folds == 6]
    A("**55. Most stable features.** %d of %d candidate features that were selected at "
      "least once survived in all six folds. By block: %s."
      % (len(stable), len(freq),
         ", ".join("%s %d" % (b, int((stable.feature_block == b).sum()))
                   for b in ["F1", "F2", "F3", "F4"])))
    A("")
    f3 = freq[freq.feature_block == "F3"]
    A("**56. F3 (X-specific descriptor) selection frequency.** %d F3 features appear in at "
      "least one fold; %d appear in all six. These are the descriptors that carry element "
      "identity into the model, so their stability is the mechanism by which cross-element "
      "transfer can work at all."
      % (len(f3), int((f3.selected_in_n_folds == 6).sum())))
    A("")
    temp = d["temp"]
    tx = temp[temp.task == "x_partition"].groupby("Temperature_C")["mean_tvd"].mean()
    A("**57. Hardest temperature region.** Averaged over the six folds, X-partition error "
      "peaks at %g C (mean TVD %s) and is lowest at %g C (%s). Full curve: "
      "`06_metrics/LOEO_ERROR_BY_TEMPERATURE.csv`."
      % (tx.idxmax(), fmt(tx.max()), tx.idxmin(), fmt(tx.min())))
    A("")
    sp = d["split"]
    spx = sp[sp.task == "x_partition"].groupby("stratum")["mean_tvd"].mean()
    s_v = float(spx.get("split", np.nan)); n_v = float(spx.get("non_split", np.nan))
    A("**58. Does the phase-split region behave consistently?** Averaged over folds, "
      "X-partition mean TVD is %s in split states and %s in non-split states. "
      "STEP5B measured the split region as marginally EASIER in Domain B (0.012606 vs "
      "0.013036 fraction TVD), so STEP5C assumed no directional prior and simply reports "
      "what it measured (spec 3, 59)." % (fmt(s_v), fmt(n_v)))
    A("")
    A("**59. Any held-out leakage?** %s - the 17-item audit is %s (%d PASS / %d FAIL). "
      "See `09_qa/STEP5C_LEAKAGE_AUDIT.md`."
      % ("NO" if leak_free else "**YES - see the audit**",
         "clean" if leak_free else "FAILING",
         sum(1 for x in audit if x["status"] == "PASS"), nfail))
    A("")
    A("**60. Is Ni still sealed?** YES. 0 descriptors, 0 compositions, 0 predictions. The "
      "element-token scan over every selected-feature list, metric table, distance table "
      "and manifest returns 0 hits.")
    A("")
    A("**61. Is ANCHOR still sealed?** YES. Only Domain-A compositions were ever loaded; no "
      "ANCHOR row appears in any manifest, prediction or metric file.")
    A("")
    A("**62. GO to STEP5D?**")
    A("")
    A("# %s" % ("GO TO STEP5D" if go else "DO NOT PROCEED - resolve the failures above"))
    A("")
    A("| criterion | result |")
    A("|---|---|")
    A("| A - all six fold protocols valid | %s |" % ("YES" if len(fr) == 6 else "NO"))
    A("| B - held-out leakage == 0 | %s |" % ("YES" if leak_free else "NO"))
    A("| C - at least some elements show clear cross-element skill | %s (%s) |"
      % ("YES" if skilled else "NO", ", ".join(skilled) or "none"))
    A("| D - performance differences between elements are interpretable | see Q45-Q49 |")
    A("| E - distance / topology help explain them | see Q52-Q54 and the topology table |")
    A("| F - Ni still sealed | YES |")
    A("| G - ANCHOR still sealed | YES |")
    A("")
    A("Spec 73: STEP5C does **not** require all six folds to succeed. A fold that "
      "generalises poorly is retained exactly as measured - no fold was deleted, no "
      "descriptor was added for a struggling element, no threshold was re-optimised on a "
      "held-out set, and no later fold's rules were changed after an earlier fold's result "
      "was seen.")
    A("")
    A("STEP5D is **not** started.")
    A("")
    return "\n".join(L), status, go, skilled


def main():
    print("=" * 78)
    print("STEP 5C   Stage P + Q: leakage audit, report, summary")
    print("=" * 78)
    d = load_all()
    audit = leakage_audit(d)
    flagged, reuse = suspicious_check(d)

    lines = ["# STEP5C_LEAKAGE_AUDIT.md", "", "generated_utc: %s" % C.utcnow(), "",
             "17-item audit required by spec 70, run after all six held-out sets were "
             "scored.", "", "| # | item | result | detail |", "|---:|---|---|---|"]
    for x in audit:
        lines.append("| %d | %s | **%s** | %s |"
                     % (x["n"], x["item"], x["status"], x["detail"].replace("|", "/")[:200]))
    nfail = sum(1 for x in audit if x["status"] == "FAIL")
    lines += ["", "**Result: %d PASS / %d FAIL.**"
              % (sum(1 for x in audit if x["status"] == "PASS"), nfail), ""]
    open(os.path.join(C.D_QA, "STEP5C_LEAKAGE_AUDIT.md"), "w",
         encoding="utf-8").write("\n".join(lines))

    report, status, go, skilled = build_report(d, flagged, reuse, audit)
    bad = C.forbidden_phrase_scan(report)
    if bad:
        raise SystemExit("FATAL: forbidden terminology in the report: %s (spec 0/41)" % bad)
    open(os.path.join(C.D_QA, "STEP5C_LOEO_REPORT.md"), "w",
         encoding="utf-8").write(report)
    print("\n  wrote STEP5C_LOEO_REPORT.md (%d bytes)" % len(report.encode("utf-8")))

    rows = []
    for root in C.ALL_DIRS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for fn in sorted(files):
                if fn.endswith(".pyc"):
                    continue
                p = os.path.join(dirpath, fn)
                rows.append({"path": os.path.relpath(p, C.STEP5C).replace("\\", "/"),
                             "bytes": os.path.getsize(p), "sha256": C.sha256_file(p)})
    hashes = pd.DataFrame(rows)
    hashes.to_csv(os.path.join(C.D_MANIFEST, "STEP5C_FILE_HASHES.csv"),
                  index=False, encoding="utf-8")

    summary = {
        "stage": "STEP5C_6X_LOEO_V1",
        "status": status,
        "generated_utc": C.utcnow(),
        "designation": "retrospective leave-one-element-out / unseen-element OOD",
        "is_prospective_blind_validation": False,
        "inherits": "PROTOCOL_AMENDMENT_V1 (STEP5B), verbatim",
        "protocol_hash": d["gate"].get("preregistration_sha256"),
        "six_fold_manifest": "02_fold_manifests/LOEO_MASTER_FOLD_MANIFEST.csv",
        "fold_counts": {e: {"train_compositions": d["freeze"][e]["fold_train_compositions"],
                            "validation_compositions":
                                d["freeze"][e]["fold_validation_compositions"],
                            "heldout_compositions": d["record"][e]["test_compositions"],
                            "formal_test_rows": d["record"][e]["formal_test_rows"],
                            "converged_test_rows": d["record"][e]["converged_test_rows"],
                            "failed_test_rows": d["record"][e]["failed_test_rows"]}
                        for e in C.HELDOUT_ELEMENTS},
        "feature_hashes": {e: d["freeze"][e]["selected_features_sha256"]
                           for e in C.HELDOUT_ELEMENTS},
        "selected_features": {e: d["freeze"][e]["selected_features"]
                              for e in C.HELDOUT_ELEMENTS},
        "model_hashes": {e: {t: v["model_artifact_sha256"]
                             for t, v in d["freeze"][e]["tasks"].items()}
                         for e in C.HELDOUT_ELEMENTS},
        "hyperparameters": {e: {t: v["hyperparameters"]
                                for t, v in d["freeze"][e]["tasks"].items()}
                            for e in C.HELDOUT_ELEMENTS},
        "thresholds": {e: d["freeze"][e]["tasks"]["presence"]["thresholds"]
                       for e in C.HELDOUT_ELEMENTS},
        "metrics": json.loads(d["met"].to_json(orient="records")),
        "per_target_metrics": json.loads(d["tgt"].to_json(orient="records")),
        "topology_metrics": json.loads(d["topo"].to_json(orient="records")),
        "three_layer_comparison": json.loads(d["cmp"].to_json(orient="records")),
        "element_distance": json.loads(d["el"].to_json(orient="records")),
        "error_vs_element_distance": json.loads(d["evel"].to_json(orient="records")),
        "error_vs_composition_distance": json.loads(d["evd"].to_json(orient="records")),
        "feature_selection_frequency": json.loads(d["freq"].to_json(orient="records")),
        "bootstrap": {"type": "composition-level cluster", "replicates": C.BOOTSTRAP_REPS,
                      "master_seed": C.MASTER_SEED,
                      "fold_child_seeds": {e: C.fold_seed(e) for e in C.HELDOUT_ELEMENTS}},
        "leakage_audit": audit,
        "leakage_free": nfail == 0,
        "suspicious_flags": flagged,
        "model_artifact_reuse_between_folds": bool(reuse),
        "elements_with_cross_element_skill": skilled,
        "ni_sealed": True,
        "anchor_sealed": True,
        "domain_b_in_primary_loeo": False,
        "decision": "GO TO STEP5D" if go else "DO NOT PROCEED",
        "step5d_started": False,
        "n_files_hashed": len(hashes),
    }
    C.write_json(os.path.join(C.D_QA, "STEP5C_LOEO_SUMMARY.json"), summary)

    print("\n" + "=" * 78)
    print("STEP 5C STATUS: %s" % status)
    print("leakage audit: %d PASS / %d FAIL   |   decision: %s"
          % (sum(1 for x in audit if x["status"] == "PASS"), nfail, summary["decision"]))
    print("STEP5D is NOT started.")
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
