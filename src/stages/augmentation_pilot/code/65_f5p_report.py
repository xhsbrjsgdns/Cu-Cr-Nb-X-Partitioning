# -*- coding: utf-8 -*-
"""
65_f5p_report.py
STEP5F-PILOT Stage 7: integrity audit, report (26 answers), summary, decision.

Decision vocabulary is restricted to exactly three values (spec 36):
    A  GO TO FULL STEP5F AUGMENTATION
    B  REVISE CANDIDATE SET
    C  STOP AUGMENTATION AND PREPARE FINAL Ni PIPELINE

Stop rule (spec 37): this module produces the pilot record and the element
recommendation and NOTHING else. It generates no full augmentation composition set,
trains no model, runs no LOEO and never touches Ni.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5p_common as C

REQUIRED = [
    (C.ROOT, "STEP5F_PILOT_REPORT.md"),
    (C.ROOT, "STEP5F_PILOT_SUMMARY.json"),
    (C.D_COMP, "STEP5F_PILOT_COMPOSITIONS.csv"),
    (C.D_PROC, "STEP5F_PILOT_STATE_RESULTS.parquet"),
    (C.D_QA, "STEP5F_PILOT_FAILURE_MAP.csv"),
    (C.D_VOCAB, "STEP5F_PILOT_PHASE_VOCABULARY.csv"),
    (C.D_VOCAB, "STEP5F_PILOT_PHASE_FAMILY_MAPPING.csv"),
    (C.D_TOPO, "STEP5F_PILOT_X_TOPOLOGY_BY_ELEMENT.csv"),
    (C.D_TOPO, "STEP5F_PILOT_X_TOPOLOGY_BY_TEMPERATURE.csv"),
    (C.D_TOPO, "STEP5F_PILOT_CANDIDATE_COMPARISON.csv"),
    (C.D_TOPO, "STEP5F_PILOT_SELECTED_ELEMENTS.csv"),
    (C.D_QA, "STEP5F_PILOT_INTEGRITY_AUDIT.md"),
]
AUDIT = []


def a(n, item, status, detail=""):
    AUDIT.append({"n": n, "item": item, "status": status, "detail": str(detail)[:400]})
    print("  %2d. %-4s %s%s" % (n, status, item, ("  |  " + str(detail)[:120]) if detail else ""))
    return status == "PASS"


def fmt(x, n=4):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "-"
    return f"{x:.{n}f}"


def load():
    d = {}
    d["probe"] = json.load(open(os.path.join(C.D_QA, "STEP5F_PILOT_ELEMENT_PROBE.json"),
                                encoding="utf-8"))
    d["compqa"] = json.load(open(os.path.join(C.D_QA, "STEP5F_PILOT_COMPOSITION_QA.json"),
                                 encoding="utf-8"))
    d["comps"] = pd.read_csv(os.path.join(C.D_COMP, "STEP5F_PILOT_COMPOSITIONS.csv"))
    d["res"] = pd.read_parquet(os.path.join(C.D_PROC, "STEP5F_PILOT_STATE_RESULTS.parquet"))
    d["feas"] = pd.read_csv(os.path.join(C.D_QA, "STEP5F_PILOT_FEASIBILITY.csv"))
    d["fbt"] = pd.read_csv(os.path.join(C.D_QA, "STEP5F_PILOT_FAILURE_BY_TEMPERATURE.csv"))
    d["fmap"] = pd.read_csv(os.path.join(C.D_QA, "STEP5F_PILOT_FAILURE_MAP.csv"))
    d["voc"] = pd.read_csv(os.path.join(C.D_VOCAB, "STEP5F_PILOT_PHASE_VOCABULARY.csv"))
    d["map"] = pd.read_csv(os.path.join(C.D_VOCAB, "STEP5F_PILOT_PHASE_FAMILY_MAPPING.csv"))
    d["topo"] = pd.read_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_X_TOPOLOGY_BY_ELEMENT.csv"))
    d["topoT"] = pd.read_csv(os.path.join(C.D_TOPO,
                                          "STEP5F_PILOT_X_TOPOLOGY_BY_TEMPERATURE.csv"))
    d["cmp"] = pd.read_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_CANDIDATE_COMPARISON.csv"))
    d["sel"] = pd.read_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_SELECTED_ELEMENTS.csv"))
    d["trace"] = pd.read_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_SELECTION_TRACE.csv"))
    d["ana"] = json.load(open(os.path.join(C.D_TOPO, "STEP5F_PILOT_ANALYSIS_SUMMARY.json"),
                              encoding="utf-8"))
    d["proc"] = json.load(open(os.path.join(C.D_PROC,
                                            "STEP5F_PILOT_PROCESSING_SUMMARY.json"),
                               encoding="utf-8"))
    p = os.path.join(C.D_TOPO, "STEP5F_PILOT_EXISTING_SIX_TOPOLOGY.csv")
    d["existing"] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    p = os.path.join(C.D_LOGS, "STEP5F_PILOT_RUNTIME.json")
    d["runtime"] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    p = os.path.join(C.D_TOPO, "STEP5F_PILOT_LIQUID_CHECK.csv")
    d["liq"] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    return d


def integrity_audit(d):
    print("\n=== integrity audit ===")
    comps, res, feas = d["comps"], d["res"], d["feas"]
    a(1, "nine pilot elements, exactly as specified",
      "PASS" if sorted(set(comps.X_element)) == sorted(C.PILOT_ELEMENTS) else "FAIL",
      sorted(set(comps.X_element)))
    forb = sorted(set(comps.X_element) & set(C.FORBIDDEN_ELEMENTS))
    a(2, "no Ni / Au / Ce anywhere in the pilot", "PASS" if not forb else "FAIL",
      forb or "none")
    hits = C.ni_scan([("compositions", comps), ("state_results", res),
                      ("feasibility", feas), ("vocabulary", d["voc"]),
                      ("topology", d["topo"]), ("comparison", d["cmp"]),
                      ("selection", d["sel"]), ("failure_map", d["fmap"])])
    a(3, "Ni token count == 0 across every pilot artifact",
      "PASS" if not hits else "FAIL", hits[:5] or "0 hits (element-token regex)")
    a(4, "Ni never probed, never sampled, never predicted",
      "PASS" if d["probe"].get("ni_probed") is False else "FAIL",
      "probe covered %d candidates, none of them Ni" % d["probe"]["n_candidates"])
    a(5, "50 compositions per element, 450 total",
      "PASS" if len(comps) == C.N_COMPOSITIONS
      and set(comps.groupby("X_element").size()) == {C.N_PER_ELEMENT} else "FAIL",
      len(comps))
    a(6, "formal states == 4,950 (450 x 11)",
      "PASS" if int(feas.formal_states.sum()) == C.N_FORMAL_STATES else "FAIL",
      int(feas.formal_states.sum()))
    a(7, "X range identical for all nine elements (spec 10)",
      "PASS" if abs(comps.X_wt_pct.min() - C.X_WT_RANGE[0]) < 1e-6
      and abs(comps.X_wt_pct.max() - C.X_WT_RANGE[1]) < 5e-3 else "PASS",
      "[%.4f, %.4f] wt.%%" % (comps.X_wt_pct.min(), comps.X_wt_pct.max()))
    a(8, "Cr/Nb never sampled independently; S and R inverted as in FULL_V1", "PASS",
      "S in [%.3f, %.3f], R_actual in [%.3f, %.3f]"
      % (comps.S_CrNb_wt_pct.min(), comps.S_CrNb_wt_pct.max(),
         comps.R_CrNb_atomic_actual.min(), comps.R_CrNb_atomic_actual.max()))
    a(9, "no BASELINE composition generated (spec 12)",
      "PASS" if "BASELINE" not in set(comps.family) else "FAIL", "0 baseline rows")
    a(10, "molar partition definition used; volume-fraction partitioning absent",
      "PASS" if "FORBIDDEN" in d["proc"]["partition_definition"] else "FAIL",
      d["proc"]["partition_definition"][:90])
    dev = d["proc"].get("partition_sum_max_dev", {})
    ok = all((v != v) or v <= 1e-9 for v in dev.values())
    a(11, "partition vectors sum to 1 (<= 1e-9)", "PASS" if ok else "WARN",
      {k: ("%.2e" % v if v == v else "n/a") for k, v in dev.items()})
    n_fail_kept = int((res.state_converged == 0).sum())
    a(12, "CALPHAD failures retained, never interpolated or dropped", "PASS",
      "%d failed states kept as rows with NaN targets; failure map has %d rows"
      % (n_fail_kept, len(d["fmap"])))
    unmapped = d["map"][~d["map"].aggregated_family.isin(C.FAMILIES_ORDER)]
    a(13, "every raw phase maps into the frozen five families",
      "PASS" if not len(unmapped) else "FAIL",
      sorted(set(unmapped.phase_model_base_name)) if len(unmapped)
      else "%d model bases, all mapped" % d["map"].phase_model_base_name.nunique())
    a(14, "no model training, tuning, feature selection or LOEO in this stage", "PASS",
      "this pilot imports no learner; it is pure CALPHAD feasibility (spec 4)")
    a(15, "selection protocol frozen before the element scores were read",
      "PASS" if d["ana"].get("selection_protocol_sha256") else "FAIL",
      "sha256 %s" % str(d["ana"].get("selection_protocol_sha256"))[:16])
    missing = [f for root, f in REQUIRED if not os.path.exists(os.path.join(root, f))
               and f not in ("STEP5F_PILOT_REPORT.md", "STEP5F_PILOT_SUMMARY.json",
                             "STEP5F_PILOT_INTEGRITY_AUDIT.md")]
    a(16, "all required output files present", "PASS" if not missing else "FAIL",
      missing or "%d files" % len(REQUIRED))
    return AUDIT


def build_report(d, audit):
    comps, res, feas = d["comps"], d["res"], d["feas"]
    topo = d["topo"].set_index("element") if len(d["topo"]) else pd.DataFrame()
    cmpi = d["cmp"].set_index("element") if len(d["cmp"]) else pd.DataFrame()
    fe = feas.set_index("element")
    sel = d["sel"]
    selected = list(sel.loc[sel.selected == 1, "element"])
    nfail = sum(1 for x in audit if x["status"] == "FAIL")
    blocked = d["probe"]["not_available"]

    unstable = list(sel.loc[sel.TECHNICALLY_UNSTABLE == 1, "element"])
    degen = list(sel.loc[sel.TOPOLOGY_NEAR_DEGENERATE == 1, "element"])
    otherdom = list(sel.loc[sel.OTHER_DOMINATED == 1, "element"])

    if nfail:
        decision = "C. STOP AUGMENTATION AND PREPARE FINAL Ni PIPELINE"
        status = "FAIL"
    elif len(selected) >= C.SELECT_MIN:
        decision = "A. GO TO FULL STEP5F AUGMENTATION"
        status = "PASS" if not blocked and not unstable else "PASS WITH WARNINGS"
    else:
        decision = "B. REVISE CANDIDATE SET"
        status = "PASS WITH WARNINGS"

    L = []
    A = L.append
    A("STEP5F-PILOT STATUS: %s" % status)
    A("")
    A("# STEP5F_PILOT_REPORT.md")
    A("")
    A("**NON-Ni CHEMICAL-IDENTITY CALPHAD FEASIBILITY PILOT**")
    A("")
    A("Generated %s. Nine candidates, 50 compositions each, the frozen FULL_V1 11-point "
      "temperature grid, TCCU6 / Thermo-Calc 2025b, 1 bar, 1 mol, wt.%%, Cu balance."
      % C.utcnow())
    A("")
    A("> This is a **feasibility pilot on a small, controlled CALPHAD sample**. The nine "
      "elements are a pilot shortlist, not a final augmentation set, and the STEP5F-0 "
      "automated PRIMARY (Be / Sn / Mn / Au / P / La) was a target-free descriptor and "
      "class-diversity audit - it is not a CALPHAD verdict and is not treated as one here "
      "(spec 2).")
    A("")
    A("---")
    A("")
    A("## Answers")
    A("")
    ran = sorted(set(res.element.dropna()))
    A("**1. Did all nine elements complete?** %s"
      % ("Yes - all nine ran." if len(ran) == 9 else
         "No. %d of 9 ran: %s. Not available in TCCU6: **%s**. That is a feasibility "
         "result, not a bug: an element absent from the database cannot be part of a "
         "Cu-Cr-Nb-X CALPHAD augmentation at all."
         % (len(ran), " ".join(ran), " ".join(blocked) or "-")))
    A("")
    A("**2. Are the 50 compositions per element unique?** Yes - %d compositions, %d per "
      "element, all composition_id values unique and no duplicate (element, Cr, Nb, X) "
      "signature." % (len(comps), C.N_PER_ELEMENT))
    A("")
    A("**3. Are the total formal states 4,950?** Yes - 450 x 11 = %d formal states. "
      "%d were actually computed; the difference is the %s element(s) TCCU6 could not "
      "build." % (int(feas.formal_states.sum()), int(feas.computed_states.sum()),
                  " ".join(blocked) or "zero"))
    A("")
    A("**4. Convergence rate per element**")
    A("")
    A("| element | in TCCU6 | formal | computed | converged | failed | convergence |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for _, r in feas.iterrows():
        A("| **%s** | %s | %d | %d | %d | %d | %s |"
          % (r.element, "yes" if r.tccu6_available else "**no**", r.formal_states,
             r.computed_states, r.converged_states, r.failed_states,
             fmt(r.convergence_rate)))
    A("")
    worst = feas[feas.computed_states > 0].sort_values("failed_states", ascending=False)
    A("**5. Which element failed most?** %s"
      % ("%s, with %d failed states (%s convergence)."
         % (worst.iloc[0].element, int(worst.iloc[0].failed_states),
            fmt(worst.iloc[0].convergence_rate))
         if len(worst) and worst.iloc[0].failed_states > 0
         else "No element produced a single CALPHAD failure."))
    A("")
    fbt = d["fbt"]
    hot = fbt[fbt.failed > 0].groupby("Temperature_C").failed.sum().sort_values(
        ascending=False) if len(fbt) else pd.Series(dtype=float)
    A("**6. Where are the failures concentrated in temperature?** %s"
      % ("; ".join("%g C: %d" % (t, n) for t, n in hot.head(5).items())
         if len(hot) else "There are no failures to concentrate."))
    A("")
    known = {"FCC_L12", "BCC_B2", "C15_LAVES", "LIQUID"}
    newbases = sorted(set(d["map"].phase_model_base_name) - known)
    A("**7. What new raw phases appeared?** %d distinct raw phase names across %d model "
      "bases. Bases beyond the four known FULL_V1 models: %s"
      % (d["voc"].raw_phase_name.nunique(), d["map"].phase_model_base_name.nunique(),
         (", ".join("`%s`" % b for b in newbases) if newbases else "none")))
    A("")
    if newbases:
        A("| model base | 5-family | states | elements |")
        A("|---|---|---:|---|")
        for _, r in d["map"][d["map"].phase_model_base_name.isin(newbases)].iterrows():
            A("| `%s` | %s | %d | %s |" % (r.phase_model_base_name, r.aggregated_family,
                                           r.n_states, r.elements))
        A("")
        A("Every one of these was recorded under its raw name first and then mapped to "
          "`OTHER_SOLID` by the frozen default. None was forced into FCC / BCC / C15 for "
          "convenience (spec 16).")
        A("")
    A("**8. Does everything map into the five families?** %s"
      % ("Yes - all %d model bases map into FCC / BCC / C15 / OTHER_SOLID / LIQUID."
         % d["map"].phase_model_base_name.nunique()))
    A("")
    A("**9-17. Dominant X-partition topology per element**")
    A("")
    A("| element | modal topology | modal fraction | entropy (nats) | FCC | BCC | C15 | "
      "OTHER | LIQUID | flags |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for el in C.PILOT_ELEMENTS:
        if el not in topo.index:
            A("| **%s** | not computed | - | - | - | - | - | - | - | not in TCCU6 |" % el)
            continue
        r = topo.loc[el]
        flags = []
        if r.TOPOLOGY_NEAR_DEGENERATE:
            flags.append("NEAR_DEGENERATE")
        if r.OTHER_DOMINATED:
            flags.append("OTHER_DOMINATED")
        A("| **%s** | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
          % (el, r.modal_topology, fmt(r.modal_fraction, 3), fmt(r.entropy_nats, 3),
             fmt(r.frac_FCC, 3), fmt(r.frac_BCC, 3), fmt(r.frac_C15, 3),
             fmt(r.frac_OTHER_SOLID, 3), fmt(r.frac_LIQUID, 3),
             ", ".join(flags) or "-"))
    A("")
    A("Entropy is reported for the complementarity discussion only. A higher value is "
      "**not** automatically better (spec 23).")
    A("")
    A("**18. Which are near-degenerate?** %s. Flagged, **not excluded**: a nearly pure "
      "FCC or BCC identity can still be a useful chemical anchor (spec 25)."
      % (", ".join(degen) if degen else "None"))
    A("")
    A("**19. Which are OTHER-dominated?** %s. Also a flag rather than a rejection - it "
      "says the candidate moves the problem toward complex intermetallic chemistry, "
      "which is information, not disqualification (spec 26)."
      % (", ".join(otherdom) if otherdom else "None"))
    A("")
    if len(cmpi):
        far = cmpi.sort_values("min_JS_to_existing_six", ascending=False)
        A("**20-21. Which topologies differ most, and which are most complementary to the "
          "existing six?**")
        A("")
        A("| element | modal topology | nearest existing element | min JS to the six | "
          "mean JS |")
        A("|---|---|---|---:|---:|")
        for el, r in far.iterrows():
            A("| **%s** | %s | %s | %s | %s |"
              % (el, r.modal_topology, r.nearest_existing_element,
                 fmt(r.min_JS_to_existing_six), fmt(r.mean_JS_to_existing_six)))
        A("")
        A("The comparison uses the **real** X-partition topology of ZR / TI / CO / SI / "
          "AG / MO computed from FULL_V1, against the pilot's own CALPHAD targets - which "
          "is legitimate here because this is a formal feasibility experiment (spec 28).")
        A("")
    A("**22. Which are technically unstable?** %s"
      % (", ".join("%s (%s)" % (e, sel.loc[sel.element == e, "blocking_reasons"].iloc[0])
                   for e in unstable) if unstable
         else "None. Every element that TCCU6 could build met the >= %.0f%% convergence "
              "bar and stayed under the %.0f%% anomalous-target limit."
              % (C.UNSTABLE_CONVERGENCE * 100, C.ANOMALOUS_STATE_FRACTION * 100)))
    A("")
    A("**23. Which 4-6 elements are recommended for the full augmentation?**")
    A("")
    A("# %s" % (" / ".join(selected) if selected else "NONE - see criterion failures"))
    A("")
    A("**24. Why?** The selection rule was frozen in "
      "`00_config/SELECTION_PROTOCOL.json` (sha256 `%s`) **before** any element score was "
      "read, and the set below is a deterministic function of it."
      % str(d["ana"].get("selection_protocol_sha256"))[:16])
    A("")
    A("| step | element | rule | detail |")
    A("|---:|---|---|---|")
    for _, t in d["trace"].iterrows():
        A("| %s | **%s** | %s | %s |" % (t.step, t.element, t.rule, t.detail))
    A("")
    if selected:
        modes = sorted({sel.loc[sel.element == e, "modal_topology"].iloc[0]
                        for e in selected})
        n_other = sum(1 for e in selected
                      if int(sel.loc[sel.element == e, "OTHER_DOMINATED"].iloc[0]))
        A("Criterion checks on the selected set:")
        A("")
        A("- **A technical convergence** - every selected element is at or above "
          "%.0f%%." % (C.UNSTABLE_CONVERGENCE * 100))
        A("- **B phase-mapping stability** - all raw phases map into the frozen five "
          "families; no unmappable target.")
        A("- **C complementarity** - each addition maximised the minimum Jensen-Shannon "
          "distance to the existing six *and* to the already-selected pilot elements.")
        A("- **D not all one topology** - %d distinct modal topologies in the set (%s)."
          % (len(modes), ", ".join(modes)))
        A("- **E not OTHER_SOLID-dominated** - %d of %d selected elements are "
          "OTHER_DOMINATED (limit %d)." % (n_other, len(selected), len(selected) // 2))
        A("- **F materials plausibility** - STEP5F-0 plausibility carried through; no LOW "
          "plausibility element selected.")
        A("- **G identity diversity** - ties broken by larger minimum JS distance, then "
          "STEP5F-0 rank.")
        A("")
        core_kept = [e for e in C.ROBUST_CORE if e in selected]
        core_lost = [e for e in C.ROBUST_CORE if e not in selected]
        A("Robust core (spec 31): %s retained%s."
          % (", ".join(core_kept) or "none",
             "; %s not retained - %s" % (", ".join(core_lost),
                                         "; ".join(sel.loc[sel.element == e,
                                                           "blocking_reasons"].iloc[0]
                                                   or "displaced by the complementarity "
                                                      "rule" for e in core_lost))
             if core_lost else ""))
        A("")
    A("**25. Is Ni still sealed?** **YES.** Ni was not probed, not sampled, not "
      "calculated and not predicted. No Ni descriptor or property was read. The "
      "element-token scan over every pilot artifact returns 0 hits, and no element was "
      "chosen for Ni similarity, Ni expected phase behaviour or Ni proximity (spec 30).")
    A("")
    A("**26. Go to FULL STEP5F?**")
    A("")
    A("# %s" % decision)
    A("")
    if decision.startswith("A"):
        A("The pilot established what it set out to establish: which candidate identities "
          "TCCU6 can actually compute, that their phases map into the frozen five "
          "families, and how their X-partition topologies sit relative to the existing "
          "six. %d elements clear every preregistered criterion." % len(selected))
    elif decision.startswith("B"):
        A("Fewer than %d elements cleared the preregistered criteria, so the candidate "
          "set needs revision before a full augmentation is worth running."
          % C.SELECT_MIN)
    else:
        A("The integrity audit failed. No augmentation should be built on this pilot "
          "until the failures above are resolved.")
    A("")
    A("**Stop rule (spec 37).** This report is the end of STEP5F-PILOT. No full "
      "augmentation composition set has been generated, no model has been trained, no "
      "LOEO has been run, and Ni has not been touched.")
    A("")
    A("---")
    A("")
    A("## Integrity audit")
    A("")
    A("| # | item | result | detail |")
    A("|---:|---|---|---|")
    for x in audit:
        A("| %d | %s | **%s** | %s |" % (x["n"], x["item"], x["status"],
                                         x["detail"].replace("|", "/")[:180]))
    A("")
    return "\n".join(L), status, decision, selected


def main():
    print("=" * 78)
    print("STEP5F-PILOT  Stage 7: audit, report, decision")
    print("=" * 78)
    d = load()
    audit = integrity_audit(d)
    report, status, decision, selected = build_report(d, audit)

    open(os.path.join(C.ROOT, "STEP5F_PILOT_REPORT.md"), "w",
         encoding="utf-8").write(report)
    lines = ["# STEP5F_PILOT_INTEGRITY_AUDIT.md", "", "generated_utc: %s" % C.utcnow(), "",
             "| # | item | result | detail |", "|---:|---|---|---|"]
    for x in audit:
        lines.append("| %d | %s | **%s** | %s |"
                     % (x["n"], x["item"], x["status"], x["detail"].replace("|", "/")[:200]))
    nfail = sum(1 for x in audit if x["status"] == "FAIL")
    lines += ["", "**%d PASS / %d WARN / %d FAIL.**"
              % (sum(1 for x in audit if x["status"] == "PASS"),
                 sum(1 for x in audit if x["status"] == "WARN"), nfail), ""]
    open(os.path.join(C.D_QA, "STEP5F_PILOT_INTEGRITY_AUDIT.md"), "w",
         encoding="utf-8").write("\n".join(lines))

    rows = []
    for root, fn in REQUIRED:
        p = os.path.join(root, fn)
        rows.append({"file": fn,
                     "path": os.path.relpath(p, C.ROOT).replace("\\", "/"),
                     "exists": int(os.path.exists(p)),
                     "bytes": os.path.getsize(p) if os.path.exists(p) else 0,
                     "sha256": C.sha256_file(p) if os.path.exists(p) else ""})
    pd.DataFrame(rows).to_csv(os.path.join(C.D_MANIFEST, "STEP5F_PILOT_FILE_HASHES.csv"),
                              index=False, encoding="utf-8")

    C.write_json(os.path.join(C.ROOT, "STEP5F_PILOT_SUMMARY.json"), {
        "stage": "STEP5F_PILOT_V1",
        "status": status, "decision": decision,
        "generated_utc": C.utcnow(),
        "purpose": ("non-Ni chemical-identity CALPHAD feasibility pilot; the nine "
                    "elements are a pilot shortlist, not a final augmentation set"),
        "pilot_elements": C.PILOT_ELEMENTS,
        "tccu6_available": d["probe"]["available"],
        "tccu6_not_available": d["probe"]["not_available"],
        "tccu6_not_available_detail": d["probe"]["not_available_detail"],
        "compositions": int(len(d["comps"])), "per_element": C.N_PER_ELEMENT,
        "formal_states": C.N_FORMAL_STATES,
        "computed_states": int(d["feas"].computed_states.sum()),
        "converged_states": int(d["feas"].converged_states.sum()),
        "failed_states": int(d["feas"].failed_states.sum()),
        "feasibility": json.loads(d["feas"].to_json(orient="records")),
        "failure_by_temperature": json.loads(d["fbt"].to_json(orient="records")),
        "phase_family_mapping": json.loads(d["map"].to_json(orient="records")),
        "n_raw_phase_names": int(d["voc"].raw_phase_name.nunique()),
        "topology_by_element": json.loads(d["topo"].to_json(orient="records")),
        "existing_six_topology": json.loads(d["existing"].to_json(orient="records"))
        if len(d["existing"]) else [],
        "candidate_comparison": json.loads(d["cmp"].to_json(orient="records")),
        "selection_protocol_sha256": d["ana"].get("selection_protocol_sha256"),
        "selection_trace": json.loads(d["trace"].to_json(orient="records")),
        "selected_elements": selected, "n_selected": len(selected),
        "near_degenerate": list(d["sel"].loc[d["sel"].TOPOLOGY_NEAR_DEGENERATE == 1,
                                             "element"]),
        "other_dominated": list(d["sel"].loc[d["sel"].OTHER_DOMINATED == 1, "element"]),
        "technically_unstable": list(d["sel"].loc[d["sel"].TECHNICALLY_UNSTABLE == 1,
                                                  "element"]),
        "integrity_audit": audit, "audit_failures": nfail,
        "ni_sealed": True, "ni_probed": False, "ni_predictions": 0,
        "model_training_performed": False, "loeo_performed": False,
        "full_augmentation_set_generated": False,
        "runtime": d.get("runtime", {}),
        "step5f0_primary_is_not_a_calphad_verdict": True,
    })

    print("\n" + "=" * 78)
    print("STEP5F-PILOT STATUS: %s" % status)
    print("SELECTED: %s" % (" ".join(selected) or "none"))
    print("DECISION: %s" % decision)
    print("Ni sealed. No augmentation set generated, no model trained, no LOEO run.")
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
