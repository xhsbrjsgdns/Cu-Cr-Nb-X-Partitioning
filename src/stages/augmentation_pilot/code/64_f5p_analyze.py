# -*- coding: utf-8 -*-
"""
64_f5p_analyze.py
STEP5F-PILOT Stages 4-6: feasibility metrics, phase vocabulary, topology,
complementarity, and the PREREGISTERED element selection.

Covers spec 20-29 and 31-32.

The selection rule (spec 29) is written into 00_config/SELECTION_PROTOCOL.json BEFORE the
element scores are read, and the selection itself is a deterministic function of that
file. Nothing is chosen by eye.

Two thresholds are diagnostic FLAGS, not automatic rejections (spec 25, 26):
  TOPOLOGY_NEAR_DEGENERATE  >99% of states share one dominant topology
  OTHER_DOMINATED           >90% of states dominated by OTHER_SOLID
A pure-FCC or pure-BCC identity can still be a valuable chemical anchor, and an
OTHER_SOLID-heavy element is a finding about intermetallic chemistry, not a failure.

Entropy (spec 23) is reported. High entropy is NOT treated as automatically better.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5p_common as C

SELECTION_PROTOCOL = {
    "preregistered_utc": None,
    "target_set_size": C.SELECT_TARGET, "min": C.SELECT_MIN, "max": C.SELECT_MAX,
    "criteria": {
        "A_technical_convergence": "convergence rate >= %.2f, else TECHNICALLY_UNSTABLE"
                                   % C.UNSTABLE_CONVERGENCE,
        "B_database_phase_mapping_stability": ("every raw phase maps into the frozen five "
                                               "families; unmappable or pathological "
                                               "target fraction < %.2f"
                                               % C.ANOMALOUS_STATE_FRACTION),
        "C_chemical_topology_complementarity": ("greedy max-min Jensen-Shannon distance of "
                                                "the dominant X-partition topology "
                                                "distribution, against the existing six "
                                                "elements AND the already-selected pilot "
                                                "elements"),
        "D_not_all_same_topology": ("the selected set must contain at least two distinct "
                                    "modal topologies"),
        "E_not_dominated_by_OTHER_SOLID": ("at most floor(n/2) selected elements may be "
                                           "OTHER_DOMINATED"),
        "F_materials_plausibility": ("STEP5F-0 plausibility_level is carried through; "
                                     "LOW plausibility cannot be selected"),
        "G_maximize_chemical_identity_diversity": (
            "a candidate that contributes a modal topology NOT already present in the "
            "selected set is preferred over one that duplicates an existing modal "
            "topology, before the JS ordering is consulted. Without this, a greedy "
            "max-min rule can add an element whose topology distribution is identical "
            "to one already selected (JS = 0), which adds no chemical identity at all."),
    },
    "core_retention": {
        "elements": C.ROBUST_CORE,
        "rule": ("Be / Sn / Mn are seeded FIRST if they are eligible AND convergence "
                 ">= %.2f. This is a preference, not an absolute requirement (spec 31); "
                 "a core element that is TECHNICALLY_UNSTABLE is not selected."
                 % C.CORE_RETENTION_CONVERGENCE),
    },
    "forbidden": ["selection based on Ni similarity, Ni expected phase behaviour or Ni "
                  "proximity (spec 30)",
                  "treating high topology entropy as automatically better (spec 23)",
                  "auto-excluding a near-degenerate or OTHER-dominated element (spec 25, 26)"],
    "tie_break_order": ["contributes a modal topology not yet in the selected set",
                        "larger minimum JS distance to the existing six and the "
                        "already-selected set",
                        "higher convergence rate", "STEP5F-0 rank", "alphabetical"],
}


def freeze_protocol():
    p = os.path.join(C.D_CONFIG, "SELECTION_PROTOCOL.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    proto = dict(SELECTION_PROTOCOL)
    proto["preregistered_utc"] = C.utcnow()
    C.write_json(p, proto)
    print("  selection protocol frozen -> %s" % os.path.basename(p))
    return proto


# ------------------------------------------------------------- feasibility -- #
def feasibility(res, comps, probe):
    rows = []
    for el in C.PILOT_ELEMENTS:
        n_comp = int((comps.X_element == el).sum())
        formal = n_comp * len(C.TEMPERATURES_C)
        g = res[res.element == el]
        conv = int((g.state_converged == 1).sum())
        computed = int(len(g))
        avail = el in probe["available"]
        rows.append({
            "element": el, "tccu6_available": int(avail),
            "probe_status": next((r["status"] for r in [] if False), None)
            or ("AVAILABLE" if avail else "NOT_AVAILABLE"),
            "compositions": n_comp, "formal_states": formal,
            "computed_states": computed, "converged_states": conv,
            "failed_states": computed - conv,
            "not_attempted_states": formal - computed,
            "convergence_rate": (conv / computed) if computed else float("nan"),
        })
    return pd.DataFrame(rows)


def failure_by_temperature(res):
    rows = []
    for el, g in res.groupby("element"):
        for t, gg in g.groupby("Temperature_C"):
            rows.append({"element": el, "Temperature_C": float(t),
                         "states": int(len(gg)),
                         "failed": int((gg.state_converged == 0).sum()),
                         "failure_rate": float((gg.state_converged == 0).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------- phase vocabulary -- #
def phase_vocabulary(phases, res):
    import fivefamily as ff
    p = phases.copy()
    if "phase_model_base_name" not in p.columns:
        p["phase_model_base_name"] = (p["raw_phase_name"].astype(str)
                                      .str.replace(r"#\d+$", "", regex=True))
    p["element"] = p["state_id"].map(res.set_index("state_id")["element"])
    p["aggregated_family"] = p["phase_model_base_name"].map(ff.family_of_5)
    voc = (p.groupby(["element", "raw_phase_name", "phase_model_base_name",
                      "aggregated_family"], as_index=False)
           .agg(n_states=("state_id", "nunique"),
                Tmin=("Temperature_C", "min"), Tmax=("Temperature_C", "max"),
                min_volfrac=("phase_volume_fraction", "min"),
                max_volfrac=("phase_volume_fraction", "max")))
    mapping = (p.groupby(["phase_model_base_name", "aggregated_family"], as_index=False)
               .agg(n_states=("state_id", "nunique"),
                    elements=("element", lambda s: ";".join(sorted(set(map(str, s)))))))
    mapping["is_known_solid_model"] = mapping["phase_model_base_name"].isin(
        ["FCC_L12", "BCC_B2", "C15_LAVES", "LIQUID"]).astype(int)
    mapping["mapped_by_default_to_OTHER_SOLID"] = (
        (mapping.aggregated_family == "OTHER_SOLID")).astype(int)
    return voc, mapping


# ------------------------------------------------------------- topology --- #
def topology_tables(res):
    cv = res[(res.state_converged == 1)].copy()
    xp = cv[["X_P_%s" % f for f in C.FAMILIES_ORDER]]
    ok = xp.notna().all(axis=1) & (xp.sum(axis=1) > 0)
    cv = cv[ok]
    cv["dominant_topology"] = C.dominant_topology(
        cv[["X_P_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(np.float64))

    by_el = []
    for el, g in cv.groupby("element"):
        dist = C.topology_distribution(g.dominant_topology)
        row = {"element": el, "n_states": int(len(g)),
               "entropy_nats": C.shannon_entropy(dist),
               "entropy_normalized": C.shannon_entropy(dist, base="classes"),
               "modal_topology": max(dist, key=dist.get),
               "modal_fraction": max(dist.values())}
        for c in C.TOPOLOGY_CLASSES:
            row["frac_%s" % c] = dist[c]
        for f in C.FAMILIES_ORDER:
            row["mean_X_P_%s" % f] = float(g["X_P_%s" % f].mean())
        row["TOPOLOGY_NEAR_DEGENERATE"] = int(row["modal_fraction"] > C.NEAR_DEGENERATE_FRACTION)
        row["OTHER_DOMINATED"] = int(dist["OTHER_SOLID"] > C.OTHER_DOMINATED_FRACTION)
        by_el.append(row)
    by_el = pd.DataFrame(by_el)

    by_T = []
    for (el, t), g in cv.groupby(["element", "Temperature_C"]):
        dist = C.topology_distribution(g.dominant_topology)
        row = {"element": el, "Temperature_C": float(t), "n_states": int(len(g)),
               "modal_topology": max(dist, key=dist.get),
               "entropy_nats": C.shannon_entropy(dist)}
        for c in C.TOPOLOGY_CLASSES:
            row["frac_%s" % c] = dist[c]
        for f in C.FAMILIES_ORDER:
            row["mean_X_P_%s" % f] = float(g["X_P_%s" % f].mean())
        by_T.append(row)
    by_T = pd.DataFrame(by_T).sort_values(["element", "Temperature_C"])

    liq = (res[res.state_converged == 1]
           .assign(has_liq=lambda d: (d.LIQUID_volfrac.fillna(0) > 1e-4).astype(int))
           .groupby(["element", "Temperature_C"], as_index=False)
           .agg(states=("state_id", "size"), liquid_states=("has_liq", "sum")))
    liq["liquid_fraction"] = liq.liquid_states / liq.states.replace(0, np.nan)
    return by_el, by_T, liq, cv


# ------------------------------------------------------ existing six ------ #
def existing_six_topology():
    """Real X-partition topology of ZR/TI/CO/SI/AG/MO from FULL_V1 (spec 28)."""
    cols = ["state_id", "family"] + ["X_P_%s" % f for f in C.FAMILIES_ORDER]
    try:
        ml = pd.read_parquet(C.ML_READY, columns=cols)
    except Exception as exc:                                    # noqa: BLE001
        print("  WARNING: could not read FULL_V1 targets (%s); the comparison against the "
              "existing six will be reported as unavailable" % exc)
        return pd.DataFrame()
    ml = ml[ml.family.isin(C.EXISTING_SIX)]
    xp = ml[["X_P_%s" % f for f in C.FAMILIES_ORDER]]
    ok = xp.notna().all(axis=1) & (xp.sum(axis=1) > 0)
    ml = ml[ok]
    ml["dominant_topology"] = C.dominant_topology(
        ml[["X_P_%s" % f for f in C.FAMILIES_ORDER]].to_numpy(np.float64))
    rows = []
    for el, g in ml.groupby("family"):
        dist = C.topology_distribution(g.dominant_topology)
        row = {"element": el, "source": "FULL_V1", "n_states": int(len(g)),
               "modal_topology": max(dist, key=dist.get),
               "modal_fraction": max(dist.values()),
               "entropy_nats": C.shannon_entropy(dist)}
        for c in C.TOPOLOGY_CLASSES:
            row["frac_%s" % c] = dist[c]
        rows.append(row)
    return pd.DataFrame(rows)


def complementarity(by_el, existing):
    """JS distance of each pilot element's topology distribution to the existing six."""
    rows = []
    ex = {r["element"]: {c: r["frac_%s" % c] for c in C.TOPOLOGY_CLASSES}
          for _, r in existing.iterrows()} if len(existing) else {}
    for _, r in by_el.iterrows():
        p = {c: r["frac_%s" % c] for c in C.TOPOLOGY_CLASSES}
        d = {k: C.jensen_shannon(p, q) for k, q in ex.items()}
        rows.append({
            "element": r["element"], "modal_topology": r["modal_topology"],
            "modal_fraction": r["modal_fraction"],
            "entropy_nats": r["entropy_nats"],
            "nearest_existing_element": (min(d, key=d.get) if d else ""),
            "min_JS_to_existing_six": (min(d.values()) if d else float("nan")),
            "mean_JS_to_existing_six": (float(np.mean(list(d.values()))) if d
                                        else float("nan")),
            "per_existing_JS": json.dumps({k: round(v, 6) for k, v in d.items()}),
            "TOPOLOGY_NEAR_DEGENERATE": int(r["TOPOLOGY_NEAR_DEGENERATE"]),
            "OTHER_DOMINATED": int(r["OTHER_DOMINATED"]),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- selection -- #
def anomaly_fraction(res, el):
    """Fraction of that element's COMPUTED states whose targets are unusable.

    'Unusable' means: not converged, or a converged state whose X-partition vector is
    missing or does not sum to 1. This is criterion B's pathological-target measure
    (spec 32) and is computed, never assumed.
    """
    g = res[res.element == el]
    if not len(g):
        return float("nan")
    bad = (g.state_converged == 0)
    cv = g[g.state_converged == 1]
    if len(cv):
        xp = cv[["X_P_%s" % f for f in C.FAMILIES_ORDER]]
        s = xp.sum(axis=1)
        badtgt = xp.isna().any(axis=1) | (np.abs(s - 1.0) > 1e-6)
        n_bad = int(bad.sum() + badtgt.sum())
    else:
        n_bad = int(bad.sum())
    return n_bad / len(g)


def select_elements(proto, feas, comp, by_el, existing, f5f0_rank, f5f0_plaus, res):
    ex_dists = {r["element"]: {c: r["frac_%s" % c] for c in C.TOPOLOGY_CLASSES}
                for _, r in existing.iterrows()} if len(existing) else {}
    pilot_dists = {r["element"]: {c: r["frac_%s" % c] for c in C.TOPOLOGY_CLASSES}
                   for _, r in by_el.iterrows()}
    fe = feas.set_index("element")
    ce = comp.set_index("element") if len(comp) else pd.DataFrame()

    trace, eligibility = [], {}
    for el in C.PILOT_ELEMENTS:
        conv = float(fe.loc[el, "convergence_rate"]) if el in fe.index else float("nan")
        anom = anomaly_fraction(res, el)
        avail = bool(fe.loc[el, "tccu6_available"]) if el in fe.index else False
        reasons = []
        if not avail:
            reasons.append("not available in TCCU6")
        if not (conv == conv) or conv < C.UNSTABLE_CONVERGENCE:
            reasons.append("convergence %.4f < %.2f" % (conv if conv == conv else float("nan"),
                                                        C.UNSTABLE_CONVERGENCE))
        if anom == anom and anom > C.ANOMALOUS_STATE_FRACTION:
            reasons.append("anomalous/unmappable target fraction %.4f > %.2f"
                           % (anom, C.ANOMALOUS_STATE_FRACTION))
        if str(f5f0_plaus.get(el, "")).upper() == "LOW":
            reasons.append("STEP5F-0 plausibility LOW")
        eligibility[el] = {
            "convergence_rate": conv, "anomalous_fraction": anom,
            "tccu6_available": avail,
            "TECHNICALLY_UNSTABLE": bool(reasons and reasons[0] != "STEP5F-0 plausibility LOW"
                                         and any("convergence" in r or "anomalous" in r
                                                 or "TCCU6" in r for r in reasons)),
            "eligible": not reasons, "blocking_reasons": reasons,
            "modal_topology": (ce.loc[el, "modal_topology"] if el in ce.index else ""),
            "OTHER_DOMINATED": int(ce.loc[el, "OTHER_DOMINATED"]) if el in ce.index else 0,
            "TOPOLOGY_NEAR_DEGENERATE": (int(ce.loc[el, "TOPOLOGY_NEAR_DEGENERATE"])
                                         if el in ce.index else 0),
        }

    eligible = [e for e in C.PILOT_ELEMENTS if eligibility[e]["eligible"]]
    selected = []

    # ---- core retention first (spec 31) ----
    for el in C.ROBUST_CORE:
        if el in eligible and eligibility[el]["convergence_rate"] >= C.CORE_RETENTION_CONVERGENCE:
            selected.append(el)
            trace.append({"step": len(selected), "element": el, "rule": "core retention",
                          "detail": "Be/Sn/Mn robust core, convergence %.4f >= %.2f"
                                    % (eligibility[el]["convergence_rate"],
                                       C.CORE_RETENTION_CONVERGENCE),
                          "min_JS": ""})

    def min_js(el, chosen):
        ref = dict(ex_dists)
        for c in chosen:
            ref["pilot:" + c] = pilot_dists.get(c, {})
        p = pilot_dists.get(el, {})
        if not ref or not p:
            return 0.0
        return min(C.jensen_shannon(p, q) for q in ref.values())

    def would_break_E(el, chosen):
        n = len(chosen) + 1
        others = sum(1 for c in chosen + [el] if eligibility[c]["OTHER_DOMINATED"])
        return others > n // 2

    # ---- greedy complementarity (criteria C, D, E, G) ----
    while len(selected) < proto["target_set_size"]:
        pool = [e for e in eligible if e not in selected]
        if not pool:
            break
        have_modes = {eligibility[c]["modal_topology"] for c in selected}

        def key(el):
            # criterion G first: a duplicate modal topology adds no chemical identity,
            # however the JS numbers happen to tie.
            return (int(eligibility[el]["modal_topology"] not in have_modes),
                    min_js(el, selected),
                    eligibility[el]["convergence_rate"],
                    -f5f0_rank.get(el, 99), el)

        allowed = [e for e in pool if not would_break_E(e, selected)]
        note = ""
        if not allowed:
            allowed = pool
            note = "criterion E relaxed: no non-OTHER_DOMINATED candidate remained"
        scored = sorted((key(e) for e in allowed), reverse=True)
        best = scored[0]
        el = best[-1]
        selected.append(el)
        trace.append({
            "step": len(selected), "element": el,
            "rule": ("greedy complementarity: new modal topology" if best[0]
                     else "greedy complementarity: max-min JS")
                    + ((" (" + note + ")") if note else ""),
            "detail": "modal %s%s; min JS to existing six + already-selected = %.4f"
                      % (eligibility[el]["modal_topology"],
                         " (NEW to the set)" if best[0] else " (already represented)",
                         best[1]),
            "min_JS": round(best[1], 6)})

    # ---- criterion D: at least two distinct modal topologies ----
    modes = {eligibility[e]["modal_topology"] for e in selected}
    if len(selected) >= 2 and len(modes) < 2:
        for el in sorted([e for e in eligible if e not in selected],
                         key=lambda e: (-min_js(e, selected), f5f0_rank.get(e, 99))):
            if eligibility[el]["modal_topology"] not in modes:
                if len(selected) >= proto["max"]:
                    drop = selected[-1]
                    selected = selected[:-1]
                    trace.append({"step": len(selected), "element": drop,
                                  "rule": "criterion D swap-out",
                                  "detail": "removed to make room for a second modal "
                                            "topology", "min_JS": ""})
                selected.append(el)
                trace.append({"step": len(selected), "element": el,
                              "rule": "criterion D enforcement",
                              "detail": "added to guarantee >= 2 distinct modal topologies",
                              "min_JS": round(min_js(el, selected[:-1]), 6)})
                break

    selected = selected[:proto["max"]]
    return selected, eligibility, trace


def main():
    print("=" * 78)
    print("STEP5F-PILOT  Stages 4-6: feasibility, vocabulary, topology, selection")
    print("=" * 78)
    sys.path.insert(0, C.FULL_V1_CODE)
    proto = freeze_protocol()

    probe = json.load(open(os.path.join(C.D_QA, "STEP5F_PILOT_ELEMENT_PROBE.json"),
                           encoding="utf-8"))
    comps = pd.read_csv(os.path.join(C.D_COMP, "STEP5F_PILOT_COMPOSITIONS.csv"))
    res = pd.read_parquet(os.path.join(C.D_PROC, "STEP5F_PILOT_STATE_RESULTS.parquet"))
    try:
        phases = C.read_table(os.path.join(C.D_RAW, "STEP5F_PILOT_PHASES_LONG"))
    except FileNotFoundError as exc:
        raise SystemExit("FATAL: no PHASES_LONG table from the run (%s)" % exc)

    feas = feasibility(res, comps, probe)
    feas.to_csv(os.path.join(C.D_QA, "STEP5F_PILOT_FEASIBILITY.csv"),
                index=False, encoding="utf-8")
    print("\n--- technical feasibility (spec 20) ---")
    for _, r in feas.iterrows():
        print("  %-3s available %d | formal %4d | computed %4d | converged %4d | "
              "failed %3d | rate %s"
              % (r.element, r.tccu6_available, r.formal_states, r.computed_states,
                 r.converged_states, r.failed_states,
                 "n/a" if r.convergence_rate != r.convergence_rate
                 else "%.4f" % r.convergence_rate))

    fbt = failure_by_temperature(res)
    fbt.to_csv(os.path.join(C.D_QA, "STEP5F_PILOT_FAILURE_BY_TEMPERATURE.csv"),
               index=False, encoding="utf-8")

    voc, mapping = phase_vocabulary(phases, res)
    voc.to_csv(os.path.join(C.D_VOCAB, "STEP5F_PILOT_PHASE_VOCABULARY.csv"),
               index=False, encoding="utf-8")
    mapping.to_csv(os.path.join(C.D_VOCAB, "STEP5F_PILOT_PHASE_FAMILY_MAPPING.csv"),
                   index=False, encoding="utf-8")
    print("\n--- phase vocabulary (spec 21) ---")
    print("  %d distinct raw phase names, %d model bases"
          % (voc.raw_phase_name.nunique(), mapping.phase_model_base_name.nunique()))

    by_el, by_T, liq, cv = topology_tables(res)
    by_el.to_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_X_TOPOLOGY_BY_ELEMENT.csv"),
                 index=False, encoding="utf-8")
    by_T.to_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_X_TOPOLOGY_BY_TEMPERATURE.csv"),
                index=False, encoding="utf-8")
    liq.to_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_LIQUID_CHECK.csv"),
               index=False, encoding="utf-8")
    print("\n--- X partition topology (spec 22-25) ---")
    for _, r in by_el.iterrows():
        print("  %-3s modal %-12s %.3f | entropy %.3f | degenerate %d | OTHER-dom %d"
              % (r.element, r.modal_topology, r.modal_fraction, r.entropy_nats,
                 r.TOPOLOGY_NEAR_DEGENERATE, r.OTHER_DOMINATED))

    existing = existing_six_topology()
    if len(existing):
        existing.to_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_EXISTING_SIX_TOPOLOGY.csv"),
                        index=False, encoding="utf-8")
    comp = complementarity(by_el, existing)
    comp = comp.merge(feas[["element", "convergence_rate", "tccu6_available"]],
                      on="element", how="left")
    comp.to_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_CANDIDATE_COMPARISON.csv"),
                index=False, encoding="utf-8")

    f5f0_rank, f5f0_plaus = {}, {}
    sel_path = os.path.join(C.STEP5F0, "04_set_selection",
                            "STEP5F0_SELECTED_NONNI_ELEMENTS.csv")
    if os.path.exists(sel_path):
        s0 = pd.read_csv(sel_path)
        s0["EL"] = s0.element.astype(str).str.upper()
        f5f0_rank = dict(zip(s0.EL, s0["rank"]))
        f5f0_plaus = dict(zip(s0.EL, s0.plausibility_level))

    selected, eligibility, trace = select_elements(
        proto, feas, comp, by_el, existing, f5f0_rank, f5f0_plaus, res)

    sel_rows = []
    for el in C.PILOT_ELEMENTS:
        e = eligibility[el]
        crow = comp[comp.element == el]
        sel_rows.append({
            "element": el, "selected": int(el in selected),
            "selection_order": (selected.index(el) + 1) if el in selected else "",
            "tccu6_available": int(e["tccu6_available"]),
            "convergence_rate": e["convergence_rate"],
            "anomalous_fraction": e["anomalous_fraction"],
            "TECHNICALLY_UNSTABLE": int(e["TECHNICALLY_UNSTABLE"]),
            "eligible": int(e["eligible"]),
            "blocking_reasons": "; ".join(e["blocking_reasons"]),
            "modal_topology": e["modal_topology"],
            "TOPOLOGY_NEAR_DEGENERATE": e["TOPOLOGY_NEAR_DEGENERATE"],
            "OTHER_DOMINATED": e["OTHER_DOMINATED"],
            "min_JS_to_existing_six": (float(crow.min_JS_to_existing_six.iloc[0])
                                       if len(crow) else float("nan")),
            "nearest_existing_element": (crow.nearest_existing_element.iloc[0]
                                         if len(crow) else ""),
            "robust_core": int(el in C.ROBUST_CORE),
            "step5f0_rank": f5f0_rank.get(el, ""),
            "step5f0_plausibility": f5f0_plaus.get(el, ""),
        })
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_SELECTED_ELEMENTS.csv"),
                  index=False, encoding="utf-8")
    pd.DataFrame(trace).to_csv(os.path.join(C.D_TOPO, "STEP5F_PILOT_SELECTION_TRACE.csv"),
                               index=False, encoding="utf-8")

    print("\n--- preregistered selection (spec 29-32) ---")
    for t in trace:
        print("  %d. %-3s  %s  %s" % (t["step"], t["element"], t["rule"], t["detail"]))
    print("  SELECTED (%d): %s" % (len(selected), " ".join(selected) or "none"))

    C.write_json(os.path.join(C.D_TOPO, "STEP5F_PILOT_ANALYSIS_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "selection_protocol_sha256": C.sha256_file(
            os.path.join(C.D_CONFIG, "SELECTION_PROTOCOL.json")),
        "selected": selected, "n_selected": len(selected),
        "eligibility": eligibility, "trace": trace,
        "entropy_note": ("entropy is reported for chemical-complementarity discussion "
                         "only; a higher value is NOT automatically better (spec 23)"),
        "flag_note": ("TOPOLOGY_NEAR_DEGENERATE and OTHER_DOMINATED are diagnostic flags, "
                      "not automatic exclusions (spec 25, 26)")})
    print("\nSTAGES 4-6: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
