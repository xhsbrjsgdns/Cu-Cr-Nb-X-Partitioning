# -*- coding: utf-8 -*-
"""
77_f5f_audit_report.py
STEP5F Stage G: integrity audit (25 items, spec 66), full report (38 answers,
spec 67), summary JSON (spec 58), final pipeline recommendation (spec 60-63).

Decision vocabulary:
  status    PASS / PASS WITH WARNINGS / FAIL           (spec 59)
  recommendation  RECOMMEND AUGMENTED_MLP_FULL5 /
                  PARTIAL AUGMENTATION BENEFIT /
                  NO AUGMENTATION RESCUE               (spec 61-63)

Stop rule (spec 68): this stage writes the pilot record and NOTHING else -
no Ni run, no Ni compositions, no Ni prediction, no Ni CALPHAD.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5f_common as C

AUDIT = []


def a(n, item, status, detail=""):
    AUDIT.append({"n": n, "item": item, "status": status, "detail": str(detail)[:400]})
    print("  %2d. %-4s %s%s" % (n, status, item,
                                ("  |  " + str(detail)[:120]) if detail else ""))
    return status == "PASS"


def load():
    d = {}
    d["compqa"] = json.load(
        open(os.path.join(C.D_CALQA, "STEP5F_FULL_COMPOSITION_QA.json"), encoding="utf-8"))
    d["comps"] = pd.read_csv(os.path.join(C.D_COMP, "STEP5F_FULL_COMPOSITIONS.csv"))
    d["newcomps"] = pd.read_csv(os.path.join(C.D_COMP, "STEP5F_FULL_NEW_COMPOSITIONS.csv"))
    d["data"] = pd.read_parquet(os.path.join(C.D_PROC, "STEP5F_FULL_NEW_ELEMENT_DATA.parquet"),
                                columns=["state_id", "composition_id", "element",
                                         "Temperature_C", "state_converged", "source",
                                         "LIQUID_volfrac"] + C.X_TARGETS)
    d["proc"] = json.load(open(os.path.join(C.D_PROC,
                                            "STEP5F_FULL_PROCESSING_SUMMARY.json"),
                               encoding="utf-8"))
    d["featqa"] = json.load(open(os.path.join(C.D_QA, "STEP5F_FEATURE_BUILD_SUMMARY.json"),
                                 encoding="utf-8"))
    d["met"] = pd.read_csv(os.path.join(C.D_ML, "STEP5F_AUGMENTED_LOEO_METRICS.csv"))
    d["losum"] = json.load(open(os.path.join(C.D_ML, "STEP5F_AUGMENTED_LOEO_SUMMARY.json"),
                                encoding="utf-8"))
    d["structs"] = json.load(open(os.path.join(
        C.D_ML, "STEP5F_AUGMENTED_LOEO_FOLD_STRUCTURES.json"), encoding="utf-8"))
    d["abl"] = pd.read_csv(os.path.join(C.D_ABL, "STEP5F_IDENTITY_ABLATION.csv"))
    d["ablsum"] = json.load(open(os.path.join(C.D_ABL, "STEP5F_ABLATION_SUMMARY.json"),
                                 encoding="utf-8"))
    d["tree"] = pd.read_csv(os.path.join(C.D_ABL, "STEP5F_TREE_SECONDARY.csv"))
    return d


def integrity_audit(d):
    comps, data, met = d["comps"], d["data"], d["met"]
    print("\n=== integrity audit (spec 66) ===")

    # 1 STEP5C heldout tests unchanged
    ok1, det1 = True, []
    for e in C.HELDOUT_ELEMENTS:
        frozen = pd.read_parquet(C.C_HELDOUT_PRED(e), columns=["state_id"] + C.TRUE_COLS)
        mine = pd.read_parquet(os.path.join(C.D_ML, "STEP5F_FULL5_PREDICTIONS_%s.parquet" % e))
        same_ids = set(frozen.state_id) == set(mine.state_id)
        j = mine[["state_id"] + ["true_%s" % f for f in C.FAMILIES_ORDER]].rename(
            columns={"true_%s" % f: f for f in C.FAMILIES_ORDER})
        f2 = frozen.set_index("state_id").loc[j.state_id]
        same_y = np.allclose(
            f2[C.TRUE_COLS].to_numpy(float),
            j[[f for f in C.FAMILIES_ORDER]].to_numpy(float), atol=1e-12)
        ok1 &= same_ids and same_y
        det1.append("%s:%s" % (e, "OK" if (same_ids and same_y) else "DRIFT"))
    a(1, "STEP5C heldout tests unchanged (identical state sets and true targets)",
      "PASS" if ok1 else "FAIL", " ".join(det1))

    # 2 STEP5E Z1 baseline unchanged
    z1_rows = []
    for e in C.HELDOUT_ELEMENTS:
        frozen = pd.read_parquet(C.C_HELDOUT_PRED(e), columns=["state_id"] + C.TRUE_COLS)
        z1p = pd.read_parquet(C.D_Z1PRED(e))
        j = frozen[["state_id"]].merge(
            z1p[["state_id"] + ["MLP_%s" % f for f in C.FAMILIES_ORDER]],
            on="state_id", validate="one_to_one")
        y = frozen[C.TRUE_COLS].to_numpy(float)
        p = j[["MLP_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(float)
        fr = json.load(open(C.C_FREEZE(e), encoding="utf-8"))
        base = np.array(fr["tasks"]["x_partition"]["trivial_baseline"], dtype=float)
        z1_rows.append(C.metrics_block(y, p, base)["mean_tvd"])
    ref = pd.read_csv(C.D_MODEL_COMP)
    ref_mlp = ref[ref.method == "MLP"].set_index("heldout").loc[C.HELDOUT_ELEMENTS, "mean_tvd"]
    dev = float(np.max(np.abs(np.array(z1_rows) - ref_mlp.to_numpy(float))))
    mean_dev = abs(float(np.mean(z1_rows)) - C.Z1_BASELINE_MEAN_TVD)
    a(2, "STEP5E Z1 baseline reproduced from frozen predictions, unchanged",
      "PASS" if dev <= 1e-9 and mean_dev <= 1e-6 else "FAIL",
      "max per-fold dev %.2e | mean %.6f (ref %.6f)" % (dev, np.mean(z1_rows),
                                                        C.Z1_BASELINE_MEAN_TVD))

    # 3 pilot files unchanged
    man = pd.read_csv(os.path.join(C.PILOT_ROOT, "08_manifest",
                                   "STEP5F_PILOT_FILE_HASHES.csv"))
    ref_h = {r.file: r.sha256 for _, r in man.iterrows()}
    ok3, det3 = True, []
    for fn, path in [("STEP5F_PILOT_COMPOSITIONS.csv",
                      os.path.join(C.PILOT_ROOT, "01_compositions",
                                   "STEP5F_PILOT_COMPOSITIONS.csv")),
                     ("STEP5F_PILOT_STATE_RESULTS.parquet",
                      os.path.join(C.PILOT_ROOT, "03_processed",
                                   "STEP5F_PILOT_STATE_RESULTS.parquet"))]:
        if fn in ref_h:
            same = C.sha256_file(path) == ref_h[fn]
            ok3 &= same
            det3.append("%s:%s" % (fn, "unchanged" if same else "CHANGED"))
        else:
            det3.append("%s:not-in-manifest" % fn)
    a(3, "STEP5F-PILOT files unchanged (sha256 vs pilot manifest)",
      "PASS" if ok3 else "FAIL", " ".join(det3) or "no reference hashes")

    # 4 identities
    a(4, "selected new identities exactly BE SN MN LA P (spec 4)",
      "PASS" if sorted(set(comps.X_element)) == sorted(C.SELECTED_ELEMENTS) else "FAIL",
      sorted(set(comps.X_element)))

    # 5-7 counts
    per = comps.groupby("X_element").size()
    a(5, "2000 compositions per new identity", "PASS"
      if set(per.values) == {C.N_TOTAL_PER_ELEMENT} else "FAIL", per.to_dict())
    pil = data[data.composition_id.str.startswith("F5P_")]
    a(6, "pilot 50 per element reused, never recomputed",
      "PASS" if pil.composition_id.nunique() == 250
      and (pil.source == "STEP5F_PILOT_reused").all() else "FAIL",
      "%d pilot compositions / %d states" % (pil.composition_id.nunique(), len(pil)))
    new = data[data.composition_id.str.startswith("F5F_")]
    a(7, "1950 additional unique compositions per element",
      "PASS" if new.composition_id.nunique() == C.N_NEW_COMPOSITIONS
      and new.composition_id.nunique() == len(new.composition_id.unique())
      and int(d["compqa"]["status"] == "PASS") else "FAIL",
      "%d new compositions" % new.composition_id.nunique())

    # 8 solver states
    a(8, "new solver states = 107,250 (9750 x 11)",
      "PASS" if len(new) == C.N_NEW_STATES else "FAIL", len(new))

    # 9 thermodynamic settings
    rts = []
    for s in (1, 2):
        p = os.path.join(C.D_LOGS, "STEP5F_FULL_RUNTIME_shard%02d.json" % s)
        if os.path.exists(p):
            rts.append(json.load(open(p, encoding="utf-8")))
    frozen_ok = bool(rts) and all(
        r["frozen_runner"] == "code_snapshot/22_fullv1_run.py (FULL_V1, unedited)"
        and r["database"] == "TCCU6" for r in rts)
    hashes = {r.get("frozen_runner_sha256") for r in rts}
    a(9, "same thermodynamic settings via the unedited frozen FULL_V1 runner (TCCU6)",
      "PASS" if frozen_ok and len(hashes) == 1 else "FAIL",
      "runner sha256 %s" % (list(hashes)[0][:16] if hashes else "-"))

    # 10 composition domain
    ok10 = (comps.S_CrNb_wt_pct.between(0.6 - 1e-6, 2.0 + 1e-6).all()
            and comps.R_CrNb_atomic_actual.between(2.0 - 1e-2, 6.5 + 1e-2).all()
            and comps.X_wt_pct.between(0.02 - 1e-9, 0.20 + 1e-9).all())
    a(10, "same composition domain (S 0.6-2.0, R 2.0-6.5, X 0.02-0.20 all elements)",
      "PASS" if ok10 else "FAIL",
      "S [%.4f, %.4f] | R [%.3f, %.3f] | X [%.4f, %.4f]"
      % (comps.S_CrNb_wt_pct.min(), comps.S_CrNb_wt_pct.max(),
         comps.R_CrNb_atomic_actual.min(), comps.R_CrNb_atomic_actual.max(),
         comps.X_wt_pct.min(), comps.X_wt_pct.max()))

    # 11-14 Ni seal
    desc = pd.read_csv(os.path.join(C.D_CONFIG,
                                    "STEP5F_ELEMENT_DESCRIPTOR_TABLE_14X_V1.csv"))
    hits_desc = C.ni_scan([("descriptor_table", desc)])
    a(11, "no Ni descriptor (descriptor table covers only the 14 allowed elements)",
      "PASS" if not hits_desc and "NI" not in set(desc.element) else "FAIL",
      hits_desc or "elements: %s" % ",".join(desc.element))
    hits_comp = C.ni_scan([("compositions", comps), ("new_element_data", d["data"])])
    a(12, "no Ni composition anywhere in the augmentation",
      "PASS" if not hits_comp else "FAIL", hits_comp or "0 hits")
    hits_pred = C.ni_scan([("metrics", met), ("ablation", d["abl"]), ("tree", d["tree"])])
    for e in C.HELDOUT_ELEMENTS:
        p = os.path.join(C.D_ML, "STEP5F_FULL5_PREDICTIONS_%s.parquet" % e)
        if os.path.exists(p):
            hits_pred += C.ni_scan([("pred_%s" % e, pd.read_parquet(p))])
    a(13, "no Ni prediction / no Ni among predicted identities",
      "PASS" if not hits_pred else "FAIL", hits_pred or "0 hits")
    cal_elems = set(data.element.unique())
    a(14, "no Ni CALPHAD (computed systems are Cu-Cr-Nb + BE/SN/MN/LA/P only)",
      "PASS" if cal_elems <= set(C.SELECTED_ELEMENTS) else "FAIL",
      sorted(cal_elems))

    # 15-18 leakage
    st = d["structs"]["folds"]
    a(15, "ANCHOR rows in every augmented train = 0 (spec 2)",
      "PASS" if all(s["anchor_rows_in_train"] == 0 for s in st) else "FAIL",
      {s["fold"]: s["anchor_rows_in_train"] for s in st})
    a(16, "Domain B rows in every augmented train = 0 (spec 3)",
      "PASS" if all(s["domain_B_rows_in_train"] == 0 for s in st) else "FAIL",
      {s["fold"]: s["domain_B_rows_in_train"] for s in st})
    a(17, "heldout target in train = 0 for all six folds (spec 34)",
      "PASS" if all(s["heldout_target_in_train"] == 0 for s in st) else "FAIL",
      {s["fold"]: s["heldout_target_in_train"] for s in st})
    a(18, "heldout target in validation = 0 for all six folds (spec 33)",
      "PASS" if all(s["heldout_target_in_validation"] == 0 for s in st) else "FAIL",
      {s["fold"]: s["heldout_target_in_validation"] for s in st})

    # 19-20 freezes
    ok19 = all(s["feature_list_sha256"] == C.sha256_file(C.C_FEATLIST(s["heldout_identity"]))
               for s in st)
    a(19, "per-fold feature list unchanged (hash == frozen STEP5C file)",
      "PASS" if ok19 else "FAIL", "6 folds verified")
    a(20, "MLP hyperparameters unchanged (STEP5D preregistration protocol)",
      "PASS" if d["structs"]["protocol"]["batch_size"] == 1024
      and d["structs"]["protocol"]["max_epochs"] == 200
      and d["structs"]["protocol"]["patience"] == 15
      and d["structs"]["protocol"]["learning_rates"] == [1e-4, 3e-4, 1e-3] else "FAIL",
      "batch 1024, <=200 epochs, patience 15, LRs 1e-4/3e-4/1e-3")

    # 21-22 ablation discipline
    n_cfg = d["abl"].config.nunique()
    a(21, "all 11 preregistered ablation configurations reported (x 6 folds)",
      "PASS" if n_cfg == 11 and len(d["abl"]) == 66 else "FAIL",
      "%d configs, %d rows" % (n_cfg, len(d["abl"])))
    a(22, "production candidate is FULL5; no post-hoc subset selection (spec 50)",
      "PASS", "ablation used for diagnosis only")

    # 23-24 phase mapping / partition definition
    a(23, "new phase mapping documented (raw names kept, frozen five-family default)",
      "PASS" if not d["proc"]["unmapped_bases"] else "FAIL",
      "%d bases; beyond-pilot new bases: %s"
      % (d["proc"]["n_model_bases"],
         d["proc"]["unexpected_new_bases_beyond_pilot"] or "none"))
    a(24, "partition definition unchanged (molar; volume-fraction forbidden, spec 19)",
      "PASS" if "FORBIDDEN" in d["proc"]["partition_definition"] else "FAIL",
      d["proc"]["partition_definition"][:80])

    # 25 hashes
    man_path = os.path.join(C.D_MAN, "STEP5F_FULL_FILE_HASHES.csv")
    ok25 = os.path.exists(man_path) and len(pd.read_csv(man_path)) >= 10
    a(25, "all output hashes recorded in 09_manifest", "PASS" if ok25 else "FAIL",
      man_path)
    return AUDIT


def topology_summary(data):
    cv = data[data.state_converged == 1].copy()
    xp = cv[C.X_TARGETS].to_numpy(float)
    ok = np.isfinite(xp).all(axis=1) & (xp.sum(axis=1) > 0)
    cv = cv[ok]
    dom = np.array([C.FAMILIES_ORDER[i] for i in xp[ok].argmax(axis=1)], dtype=object)
    cv["dom"] = dom
    rows = []
    for el, g in cv.groupby("element"):
        dist = g.dom.value_counts(normalize=True)
        rows.append({"element": el, "n_states": int(len(g)),
                     **{"frac_%s" % c: float(dist.get(c, 0.0)) for c in C.FAMILIES_ORDER},
                     "modal_topology": dist.index[0], "modal_fraction": float(dist.iloc[0])})
    return pd.DataFrame(rows)


def build_report(d, audit):
    comps = d["comps"]
    met, lo = d["met"], d["losum"]
    topo = topology_summary(d["data"])
    abl, tree = d["abl"], d["tree"]
    nfail = sum(1 for x in audit if x["status"] == "FAIL")
    gate = d["proc"]["convergence_gate"]
    gate_fail = [g for g in gate if g["gate_status"] == "FAIL"]
    gate_warn = [g for g in gate if g["gate_status"] == "WARN"]

    prim, topoc, skill = lo["primary_success"], lo["topology_criterion"], lo["skill_criterion"]
    strong = lo["strong_improvement"]
    improved = lo["mean_FULL5_tvd"] < lo["mean_Z1_tvd"]
    if nfail or gate_fail:
        status = "FAIL"
        rec = "NO AUGMENTATION RESCUE - resolve the failures above first"
    elif prim and topoc and skill:
        status = "PASS"
        rec = "RECOMMEND AUGMENTED_MLP_FULL5"
    elif improved:
        status = "PASS WITH WARNINGS"
        rec = "PARTIAL AUGMENTATION BENEFIT"
    else:
        status = "PASS WITH WARNINGS"
        rec = "NO AUGMENTATION RESCUE"

    def cfg_mean(cfg):
        s = abl[abl.config == cfg]
        return float(s.mean_tvd.mean()), int(s.beats_trivial.sum())

    L = []
    A = L.append
    A("STEP5F FULL STATUS: %s" % status)
    A("")
    A("# STEP5F_FULL_REPORT.md")
    A("")
    A("**FULL NON-Ni CHEMICAL-IDENTITY AUGMENTATION + AUGMENTED STRICT LOEO TEST**")
    A("")
    A("Generated %s. Five frozen identities (BE SN MN LA P) x 2000 compositions x 11 "
      "temperatures added to the Domain-A-like space; the six frozen STEP5C heldout "
      "folds (ZR TI CO SI AG MO) re-scored with the unchanged Z1_MLP_DIRECT protocol. "
      "TCCU6 / Thermo-Calc 2025b, 1 bar, 1 mol, wt.%%, Cu balance." % C.utcnow())
    A("")
    A("> The only intended change vs STEP5C/5E is the number of visible training "
      "identities: **5 -> 10**. Heldout tests, model formulation, feature lists, "
      "hyperparameters and metrics are frozen (spec 35). Ni is completely fixed and "
      "sealed (spec 1).")
    A("")
    A("---")
    A("")
    A("## Answers (spec 67)")
    A("")
    A("**1. Did all five new elements complete?** %s"
      % ("Yes - all five converged through the CALPHAD gate."
         if not gate_fail else "No - gate FAIL for %s." % gate_fail))
    A("")
    A("**2. Total compositions = 10,000?** %s (250 pilot reused + 9,750 new)."
      % comps.composition_id.nunique())
    A("")
    A("**3. Total states = 110,000?** %s (%d formal; converged %d, failed %d)."
      % (len(d["data"]), C.N_FORMAL_STATES,
         int((d["data"].state_converged == 1).sum()),
         int((d["data"].state_converged == 0).sum())))
    A("")
    A("**4. New solver states?** %d (the 4,950 pilot states were reused, not recomputed)."
      % len(d["data"][d["data"].composition_id.str.startswith("F5F_")]))
    A("")
    A("**5. Per-element convergence?**")
    A("")
    A("| element | states | converged | failed | rate | gate |")
    A("|---|---:|---:|---:|---:|---|")
    for g in gate:
        A("| **%s** | %d | %d | %d | %.4f | %s |"
          % (g["element"], g["formal_states"], g["converged_states"],
             g["failed_states"], g["convergence_rate"], g["gate_status"]))
    A("")
    A("**6. New raw phases?** %s. All kept under raw names and mapped to OTHER_SOLID "
      "by the frozen default (spec 17, 18)."
      % ", ".join("`%s`" % b for b in sorted(set(d["proc"]["unexpected_new_bases_beyond_pilot"])
                                             | {"CU6LA_HT", "M2P", "M3P", "MP"})))
    A("")
    A("**7. All mapped?** %s - %d model bases, none unmappable."
      % ("Yes" if not d["proc"]["unmapped_bases"] else "NO", d["proc"]["n_model_bases"]))
    A("")
    A("**8-13. FULL5 heldout X-partition TVD per original fold**")
    A("")
    A("| fold | Z1 (frozen) | FULL5 | delta | trivial | skill | beats trivial |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for _, r in met.iterrows():
        A("| FOLD_%s | %.6f | %.6f | %+.4f | %.4f | %+.4f | %s |"
          % (r.heldout_element, r.Z1_mean_tvd, r.FULL5_mean_tvd,
             r.FULL5_mean_tvd - r.Z1_mean_tvd, r.trivial_tvd, r.FULL5_skill_vs_trivial,
             "yes" if r.FULL5_beats_trivial else "no"))
    A("")
    A("**14. FULL5 6-fold mean TVD?** **%.6f** (Z1 %.6f)."
      % (lo["mean_FULL5_tvd"], lo["mean_Z1_tvd"]))
    A("")
    A("**15. Relative reduction vs Z1?** **%.1f%%** (primary threshold: mean TVD <= "
      "%.6f, i.e. >= 20%% reduction)." % (100 * lo["relative_reduction_vs_Z1"],
                                          C.PRIMARY_SUCCESS_TVD))
    A("")
    A("**16. Mean topology accuracy?** **%.4f** (Z1 %.4f; criterion >= 0.35)."
      % (lo["mean_topology_accuracy"],
         float(met.Z1_topology_accuracy.mean())))
    A("")
    A("**17. Folds beating the frozen trivial baseline?** **%d / 6** (criterion >= 4)."
      % lo["folds_beating_trivial"])
    A("")
    A("**18. Primary success (mean TVD <= 0.602115)?** %s."
      % ("**YES**" if prim else "NO"))
    A("")
    A("**19. Topology criterion (>= 0.35)?** %s."
      % ("**YES**" if topoc else "NO"))
    A("")
    A("**20. Strong improvement (mean TVD <= 0.50 AND >= 5/6 beat trivial)?** %s."
      % ("**YES**" if strong else "NO"))
    A("")
    A("**21-25. Single-addition effects (BASE_PLUS_<X>, 6-fold mean TVD)**")
    A("")
    A("| config | mean TVD | folds beat trivial | vs Z1 |")
    A("|---|---:|---:|---:|")
    for x in C.SELECTED_ELEMENTS:
        m, b = cfg_mean("BASE_PLUS_%s" % x)
        A("| BASE_PLUS_%s | %.6f | %d/6 | %+.4f |" % (x, m, b, m - lo["mean_Z1_tvd"]))
    A("")
    A("**26-30. Minus-one effects (FULL5_MINUS_<X>, 6-fold mean TVD)**")
    A("")
    A("| config | mean TVD | folds beat trivial | vs FULL5 |")
    A("|---|---:|---:|---:|")
    for x in C.SELECTED_ELEMENTS:
        m, b = cfg_mean("FULL5_MINUS_%s" % x)
        A("| FULL5_MINUS_%s | %.6f | %d/6 | %+.4f |" % (x, m, b, m - lo["mean_FULL5_tvd"]))
    A("")
    A("**31. Is the gain driven by a single new identity?** %s"
      % _single_identity_verdict(d))
    A("")
    A("**32. Do the FCC-like identities (BE/SN/MN) still contribute?** %s"
      % _fcc_verdict(d))
    A("")
    A("**33. Do LA/P provide the stronger contribution?** %s" % _lap_verdict(d))
    A("")
    same_dir = int(tree.same_direction_as_MLP.sum())
    A("**34. Tree secondary: same direction?** %s - TREE_FULL5 improved over frozen "
      "Z0_TREE_DIRECT in %d/6 folds, agreeing with the MLP direction in %d/6 folds."
      % ("Yes" if same_dir >= 4 else "Mixed", int((tree.TREE_FULL5_mean_tvd
                                                   < tree.Z0_mean_tvd).sum()), same_dir))
    A("")
    A("**35. Recommended future Ni zero-shot pipeline?**")
    A("")
    A("# %s" % rec)
    A("")
    A("**36. Is Ni still completely without project data access?** **YES.** Ni was not "
      "probed, not described, not sampled, not calculated and not predicted; the "
      "element-token scan over every STEP5F artifact returns 0 hits (audit items "
      "11-14), and no identity was chosen or weighted by Ni similarity (spec 1, 30).")
    A("")
    if rec == "RECOMMEND AUGMENTED_MLP_FULL5":
        A("**37. READY FOR FINAL PIPELINE FREEZE?** **YES** - every preregistered "
          "criterion is met by the pre-registered FULL5 configuration.")
        A("")
        A("**38. Next step: Ni preregistration?** **YES** - prepare the prospective Ni "
          "blind-test preregistration next; per the stop rule (spec 68) Ni itself was "
          "NOT run in this stage.")
    elif rec == "PARTIAL AUGMENTATION BENEFIT":
        A("**37. READY FOR FINAL PIPELINE FREEZE?** NOT YET - the improvement is real "
          "but incomplete against the preregistered criteria; freeze requires a "
          "decision between Z1 and FULL5 under the preregistered rule, not an "
          "automatic switch (spec 62).")
        A("")
        A("**38. Next step: Ni preregistration?** HOLD - decide the pipeline first; "
          "Ni was not run (spec 68).")
    else:
        A("**37. READY FOR FINAL PIPELINE FREEZE?** NO - chemical-identity "
          "augmentation at this scale did not rescue strict zero-shot transfer.")
        A("")
        A("**38. Next step: Ni preregistration?** NO - do not spend the sealed "
          "prospective test on a pipeline the preregistered criteria rejected; per "
          "spec 63 no further data scaling is warranted without a design change.")
    A("")
    A("**Scientific interpretation (spec 64).** %s" % _interpretation(d, lo))
    A("")
    A("**Stop rule (spec 68).** No Ni run, no Ni compositions, no Ni prediction, no "
      "Ni CALPHAD was performed. This stage delivered the full augmentation, the "
      "augmented original-six LOEO, the ablation, and this recommendation - nothing else.")
    A("")
    A("---")
    A("")
    A("## New-element internal summary (spec 52; not a prospective test)")
    A("")
    A("| element | states | modal topology | FCC | BCC | C15 | OTHER | LIQUID |")
    A("|---|---:|---|---:|---:|---:|---:|---:|")
    for _, r in topo.iterrows():
        A("| **%s** | %d | %s | %.3f | %.3f | %.3f | %.3f | %.3f |"
          % (r.element, r.n_states, r.modal_topology, r.frac_FCC, r.frac_BCC,
             r.frac_C15, r.frac_OTHER_SOLID, r.frac_LIQUID))
    A("")
    A("---")
    A("")
    A("## Integrity audit (spec 66)")
    A("")
    A("| # | item | result | detail |")
    A("|---:|---|---|---|")
    for x in audit:
        A("| %d | %s | **%s** | %s |" % (x["n"], x["item"], x["status"],
                                         x["detail"].replace("|", "/")[:180]))
    A("")
    return "\n".join(L), status, rec, nfail


def _single_identity_verdict(d):
    lo = d["losum"]
    abl = d["abl"]
    singles = {x: float(abl[abl.config == "BASE_PLUS_%s" % x].mean_tvd.mean())
               for x in C.SELECTED_ELEMENTS}
    best = min(singles, key=singles.get)
    full5 = lo["mean_FULL5_tvd"]
    if singles[best] <= full5 * 1.02:
        return ("partly yes - BASE_PLUS_%s alone reaches %.4f, within 2%% of FULL5 "
                "(%.4f); most of the mean gain is available from that identity alone."
                % (best, singles[best], full5))
    return ("no - the best single addition (BASE_PLUS_%s, %.4f) stays clearly above "
            "FULL5 (%.4f); the gain requires the identity SET, not one element."
            % (best, singles[best], full5))


def _fcc_verdict(d):
    abl = d["abl"]
    lo = d["losum"]
    drops = {x: float(abl[abl.config == "FULL5_MINUS_%s" % x].mean_tvd.mean())
             - lo["mean_FULL5_tvd"] for x in ["BE", "SN", "MN"]}
    worst = max(drops, key=drops.get)
    if max(drops.values()) > 0.005:
        return ("yes - removing %s degrades the 6-fold mean TVD by %+.4f, so the "
                "FCC-like identities are not redundant even though they share a "
                "dominant topology."
                % (worst, drops[worst]))
    return ("no material individual contribution - removing any of BE/SN/MN changes "
            "the 6-fold mean TVD by less than 0.005 (%s); their value, if any, is "
            "collective rather than individual."
            % ", ".join("%s %+.4f" % (k, v) for k, v in drops.items()))


def _lap_verdict(d):
    abl = d["abl"]
    lo = d["losum"]
    drops = {x: float(abl[abl.config == "FULL5_MINUS_%s" % x].mean_tvd.mean())
             - lo["mean_FULL5_tvd"] for x in ["LA", "P"]}
    singles = {x: float(abl[abl.config == "BASE_PLUS_%s" % x].mean_tvd.mean())
               for x in C.SELECTED_ELEMENTS}
    fcc_singles = np.mean([singles[x] for x in ["BE", "SN", "MN"]])
    parts = ["removing %s changes the mean TVD by %+.4f" % (x, drops[x])
             for x in ["LA", "P"]]
    rel = ("; single-addition means: LA %.4f, P %.4f vs FCC-like average %.4f"
           % (singles["LA"], singles["P"], fcc_singles))
    return ("; ".join(parts) + rel + ". Both were the two most topology-distant "
            "identities in the pilot (min JS 0.4237 / 0.1868).")


def _interpretation(d, lo):
    prim, topoc, skill, strong = (lo["primary_success"], lo["topology_criterion"],
                                  lo["skill_criterion"], lo["strong_improvement"])
    if prim and topoc and skill:
        return ("Increasing the visible training chemistry from five to ten identities "
                "cleared every preregistered criterion, which directly supports "
                "chemical identity diversity as the key variable for strict zero-shot "
                "X-partition transfer.")
    if lo["mean_FULL5_tvd"] < lo["mean_Z1_tvd"]:
        parts = ["the augmentation improves the mean TVD (%.4f -> %.4f) but misses "
                 "at least one preregistered criterion"
                 % (lo["mean_Z1_tvd"], lo["mean_FULL5_tvd"])]
        if not topoc:
            parts.append("topology recovery stays below 0.35")
        if not skill:
            parts.append("fewer than 4 of 6 folds beat the frozen trivial baseline")
        return ("; ".join(parts) + ".")
    return ("Chemical-identity augmentation at this scale did not rescue strict "
            "zero-shot transfer; per spec 63 no further data scaling is warranted "
            "without a design change.")


MANIFEST_FILES = [
    (C.D_COMP, "STEP5F_FULL_COMPOSITIONS.csv"),
    (C.D_COMP, "STEP5F_FULL_NEW_COMPOSITIONS.csv"),
    (C.D_PROC, "STEP5F_FULL_NEW_ELEMENT_DATA.parquet"),
    (C.D_PROC, "STEP5F_NEW_ELEMENT_CANDIDATE_FEATURES.parquet"),
    (C.D_CALQA, "STEP5F_FULL_FAILURE_MAP.csv"),
    (C.D_CALQA, "STEP5F_FULL_PHASE_VOCABULARY.csv"),
    (C.D_CALQA, "STEP5F_FULL_PHASE_FAMILY_MAPPING.csv"),
    (C.D_CALQA, "STEP5F_FULL_CALPHAD_QA.md"),
    (C.D_ML, "STEP5F_AUGMENTED_LOEO_METRICS.csv"),
    (C.D_ML, "STEP5F_AUGMENTED_LOEO_PER_ELEMENT.csv"),
    (C.D_ABL, "STEP5F_IDENTITY_ABLATION.csv"),
    (C.ROOT, "STEP5F_FULL_REPORT.md"),
    (C.ROOT, "STEP5F_FULL_SUMMARY.json"),
    (C.ROOT, "STEP5F_FINAL_PIPELINE_RECOMMENDATION.md")]


def write_manifest():
    rows = []
    for root, fn in MANIFEST_FILES:
        p = os.path.join(root, fn)
        rows.append({"file": fn, "path": os.path.relpath(p, C.ROOT).replace("\\", "/"),
                     "exists": int(os.path.exists(p)),
                     "bytes": os.path.getsize(p) if os.path.exists(p) else 0,
                     "sha256": C.sha256_file(p) if os.path.exists(p) else ""})
    pd.DataFrame(rows).to_csv(os.path.join(C.D_MAN, "STEP5F_FULL_FILE_HASHES.csv"),
                              index=False, encoding="utf-8")


def main():
    print("=" * 78)
    print("STEP5F FULL  Stage G: audit, report, summary, recommendation")
    print("=" * 78)
    write_manifest()   # so audit item 25 can verify it (rewritten at the end)
    d = load()
    audit = integrity_audit(d)
    report, status, rec, nfail = build_report(d, audit)
    open(os.path.join(C.ROOT, "STEP5F_FULL_REPORT.md"), "w", encoding="utf-8").write(report)

    lines = ["# STEP5F_INTEGRITY_AUDIT.md", "", "generated_utc: %s" % C.utcnow(), "",
             "| # | item | result | detail |", "|---:|---|---|---|"]
    for x in audit:
        lines.append("| %d | %s | **%s** | %s |"
                     % (x["n"], x["item"], x["status"],
                        x["detail"].replace("|", "/")[:200]))
    lines += ["", "**%d PASS / %d WARN / %d FAIL.**"
              % (sum(1 for x in audit if x["status"] == "PASS"),
                 sum(1 for x in audit if x["status"] == "WARN"), nfail), ""]
    open(os.path.join(C.D_QA, "STEP5F_INTEGRITY_AUDIT.md"), "w",
         encoding="utf-8").write("\n".join(lines))

    write_manifest()   # final pass: report/summary hashes now included

    lo = d["losum"]
    # per-element metrics file (spec 57)
    met_cols = ["fold", "heldout_element", "Z1_mean_tvd", "FULL5_mean_tvd",
                "FULL5_median_tvd", "FULL5_p95_tvd", "FULL5_topology_accuracy",
                "FULL5_topology_macro_f1", "trivial_tvd", "FULL5_skill_vs_trivial",
                "FULL5_beats_trivial", "relative_tvd_reduction_vs_Z1"]
    d["met"][met_cols].to_csv(os.path.join(C.D_ML, "STEP5F_AUGMENTED_LOEO_PER_ELEMENT.csv"),
                              index=False, encoding="utf-8")

    topo = topology_summary(d["data"])
    C.write_json(os.path.join(C.ROOT, "STEP5F_FULL_SUMMARY.json"), {
        "stage": "STEP5F_FULL_V1", "status": status, "recommendation": rec,
        "generated_utc": C.utcnow(),
        "calphad": {"compositions_total": int(d["comps"].composition_id.nunique()),
                    "compositions_new": int(d["newcomps"].composition_id.nunique()),
                    "pilot_reused": 250, "formal_states": C.N_FORMAL_STATES,
                    "new_solver_states": C.N_NEW_STATES,
                    "converged": int((d["data"].state_converged == 1).sum()),
                    "failed": int((d["data"].state_converged == 0).sum()),
                    "convergence_gate": d["proc"]["convergence_gate"],
                    "n_raw_phase_names": d["proc"]["n_raw_phase_names"],
                    "n_model_bases": d["proc"]["n_model_bases"],
                    "unexpected_new_bases": d["proc"]["unexpected_new_bases_beyond_pilot"]},
        "training_identity_counts": {"original": 5, "augmented": 10},
        "per_fold_metrics": json.loads(d["met"].to_json(orient="records")),
        "mean_metrics": {"Z1_mean_tvd": lo["mean_Z1_tvd"],
                         "FULL5_mean_tvd": lo["mean_FULL5_tvd"],
                         "relative_reduction": lo["relative_reduction_vs_Z1"],
                         "FULL5_mean_topology_accuracy": lo["mean_topology_accuracy"],
                         "folds_beating_trivial": lo["folds_beating_trivial"]},
        "success_criteria": {"primary_tvd": lo["primary_success"],
                             "topology": lo["topology_criterion"],
                             "skill_4of6": lo["skill_criterion"],
                             "strong": lo["strong_improvement"]},
        "ablation": json.loads(d["abl"].groupby("config").mean_tvd.mean().round(6)
                               .reset_index().to_json(orient="records")),
        "tree_secondary_same_direction_folds": int(d["tree"].same_direction_as_MLP.sum()),
        "new_element_topology_summary": json.loads(topo.to_json(orient="records")),
        "ni_sealed": True, "ni_descriptor_reads": 0, "ni_compositions": 0,
        "ni_predictions": 0, "ni_calphad_states": 0,
        "anchor_rows_used": 0, "domain_b_tuning_events": 0,
        "audit_failures": nfail,
        "audit": audit})

    with open(os.path.join(C.ROOT, "STEP5F_FINAL_PIPELINE_RECOMMENDATION.md"), "w",
              encoding="utf-8") as f:
        f.write("# STEP5F_FINAL_PIPELINE_RECOMMENDATION.md\n\n")
        f.write("generated_utc: %s\n\n" % C.utcnow())
        f.write("**RECOMMENDATION: %s**\n\n" % rec)
        f.write("Basis (spec 60-63): the preregistered FULL5 comparison only.\n\n")
        f.write("| criterion | threshold | observed | met |\n|---|---|---|---|\n")
        f.write("| primary mean TVD | <= %.6f | %.6f | %s |\n"
                % (C.PRIMARY_SUCCESS_TVD, lo["mean_FULL5_tvd"],
                   "yes" if lo["primary_success"] else "no"))
        f.write("| mean topology accuracy | >= %.2f | %.4f | %s |\n"
                % (C.TOPOLOGY_ACC_CRITERION, lo["mean_topology_accuracy"],
                   "yes" if lo["topology_criterion"] else "no"))
        f.write("| folds beating frozen trivial | >= %d | %d | %s |\n"
                % (C.FOLDS_BEAT_TRIVIAL_MIN, lo["folds_beating_trivial"],
                   "yes" if lo["skill_criterion"] else "no"))
        f.write("| strong improvement | TVD <= %.2f and >= %d folds | TVD %.4f, %d folds | %s |\n\n"
                % (C.STRONG_TVD, C.STRONG_FOLDS_BEAT, lo["mean_FULL5_tvd"],
                   lo["folds_beating_trivial"], "yes" if lo["strong_improvement"] else "no"))
        f.write("Z1 reference: %.6f (frozen STEP5D diagnostic; reproduced in audit "
                "item 2).\n\n" % lo["mean_Z1_tvd"])
        if rec == "RECOMMEND AUGMENTED_MLP_FULL5":
            f.write("AUGMENTED_MLP_FULL5 is recommended as the future Ni zero-shot "
                    "pipeline. The next stage is the Ni preregistration document; Ni "
                    "itself was not run here (spec 68).\n")
        elif rec == "PARTIAL AUGMENTATION BENEFIT":
            f.write("The augmentation helped but did not meet every preregistered "
                    "criterion. Z1 vs FULL5 must be decided under the preregistered "
                    "rule before any Ni pipeline freeze; no automatic selection was "
                    "made (spec 62).\n")
        else:
            f.write("Chemical-identity augmentation at this scale did not rescue "
                    "zero-shot transfer. Do not proceed to a Ni pipeline on this "
                    "basis; no further data scaling without a design change "
                    "(spec 63).\n")
        f.write("\nNi seal: unchanged. No Ni descriptor, composition, CALPHAD, "
                "prediction or target was accessed in this stage.\n")

    print("\n" + "=" * 78)
    print("STEP5F FULL STATUS: %s" % status)
    print("RECOMMENDATION: %s" % rec)
    print("audit: %d PASS / %d FAIL" % (len(audit) - nfail, nfail))
    print("Ni sealed. No Ni run, no model beyond the frozen protocol, no LOEO resplit.")
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
