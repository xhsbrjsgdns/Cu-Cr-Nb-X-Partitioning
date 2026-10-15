# -*- coding: utf-8 -*-
"""
r3_50_evidence.py
STEP5K-R3 stage 50: per-system evidence table (spec 40, 41) and the mandatory
database-independence note (spec 39).

NO SINGLE GLOBAL ACCURACY IS PRODUCED ANYWHERE IN THIS STAGE (spec 38). The six
studies differ in method, temperature range, alloy count and phase-identification
quality; one headline percentage over them would be meaningless and would invite
exactly the overclaim the manuscript must avoid. Everything is reported per system,
per source and per temperature.

EVIDENCE STRENGTH IS NOT AN AGREEMENT PERCENTAGE (spec 41). It is a five-part score
over breadth, temperature coverage, measured phase chemistry, mapping quality and
agreement, written out below, and every row carries the independence caveat as a
standing limitation.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C

# The installed Thermo-Calc tree, read-only, for the documentary independence check.
# R3_TC_ROOT overrides it only so the stage can be exercised off-machine; on the
# analysis machine the environment variable is unset and the default path is used.
TC_ROOT = os.environ.get("R3_TC_ROOT", r"C:\Program Files\Thermo-Calc\2025b")
D_OPEN = "<!-- DISCLAIMER_BLOCK_START -->"
D_CLOSE = "<!-- DISCLAIMER_BLOCK_END -->"
SCAN_EXT = (".txt", ".html", ".htm", ".tdb", ".md", ".csv", ".xml", ".json")
SCAN_MAX_BYTES = 80 * 1024 * 1024


def strength(n_conv, n_temps, n_comp, agree_frac, unmappable_frac):
    """Five parts, each stated. Deliberately not a function of agreement alone."""
    parts = {
        "breadth (converged conditions)": 2 if n_conv >= 10 else 1 if n_conv >= 5
        else 0,
        "temperature coverage": 1 if n_temps >= 3 else 0,
        "measured phase chemistry": 1 if n_comp >= 10 else 0,
        "phase-identity mapping quality": 1 if unmappable_frac <= 0.25 else 0,
        "agreement": 1 if (agree_frac is not None and agree_frac >= 0.7) else 0}
    total = sum(parts.values())
    label = ("NONE" if n_conv == 0 else "STRONG" if total >= 5
             else "MODERATE" if total >= 3 else "LIMITED")
    return label, total, parts


def scan_database_documentation(dois, authors):
    """Read-only search of the installed Thermo-Calc tree for the six sources. A hit
    shows the database description cites that study, which makes the comparison
    circular for that system. A miss shows only that the accessible documentation
    does not name it - never that the assessment is independent."""
    found, scanned, skipped = {}, 0, 0
    if not os.path.isdir(TC_ROOT):
        return {"scanned_files": 0, "root_present": False, "hits": {},
                "note": "the Thermo-Calc installation directory was not reachable "
                        "from this interpreter; no documentary check was possible"}
    budget = SCAN_MAX_BYTES
    pats = {k: re.compile(re.escape(k), re.I) for k in list(dois) + list(authors)}
    for root, _, files in os.walk(TC_ROOT):
        for f in files:
            if not f.lower().endswith(SCAN_EXT):
                continue
            p = os.path.join(root, f)
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            if size > 25 * 1024 * 1024 or size > budget:
                skipped += 1
                continue
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                skipped += 1
                continue
            budget -= size
            scanned += 1
            for key, rx in pats.items():
                if rx.search(txt):
                    found.setdefault(key, []).append(os.path.relpath(p, TC_ROOT))
            if budget <= 0:
                break
        if budget <= 0:
            break
    return {"scanned_files": scanned, "skipped_files": skipped,
            "root_present": True, "hits": {k: v[:5] for k, v in found.items()}}


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K-R3 stage 50: system evidence and database independence")
    print("=" * 78)
    n_prot, _, _ = C.verify_protected()
    print("  %d protected artifacts re-hashed, all unchanged" % n_prot)

    I = C.read_inputs()
    cmp_ = pd.read_csv(os.path.join(C.D_CMP, "STEP5K_R3_CONDITION_COMPARISON.csv"))
    pcmp = pd.read_csv(os.path.join(C.D_PHCOMP,
                                    "STEP5K_R3_PHASE_COMPOSITION_COMPARISON.csv"))
    dom = I["domain"]
    man = I["manifest"]

    rows = []
    for system in C.SYSTEM_ROWS:
        if system.startswith("GRCop"):
            g = cmp_[cmp_.source_id == "S6_GRCOP"]
            pc = pcmp[pcmp.source_id == "S6_GRCOP"]
        else:
            g = cmp_[(cmp_.system == system) & (cmp_.source_id != "S6_GRCOP")]
            pc = pcmp[pcmp.condition_id.isin(g.condition_id)]
        conv = g[g.execution_status == "CONVERGED"]
        vc = conv.verdict.value_counts().to_dict()
        n_conv = len(conv)
        n_strong = vc.get("STRONGLY_CONSISTENT", 0)
        n_cons = vc.get("CONSISTENT", 0)
        n_part = vc.get("PARTIALLY_CONSISTENT", 0)
        n_disc = vc.get("DISCREPANT", 0)
        agree = ((n_strong + n_cons) / n_conv) if n_conv else None
        unmap = (g.unmappable_experimental_phases.fillna("").astype(str)
                 .str.len().gt(0).mean() if len(g) else 1.0)
        temps = g.temperature_C.dropna().nunique()
        matched = pc[pc.match_status == "MATCHED"] if len(pc) else pc
        label, total, parts = strength(n_conv, temps, len(matched), agree, unmap)
        if system.startswith("GRCop"):
            label = "LIMITED" if len(g) else "NONE"

        rj = conv.raw_phase_Jaccard.mean() if n_conv else np.nan
        fj = conv.family_Jaccard.mean() if n_conv else np.nan
        topo = conv.topology_agreement.replace("", np.nan).dropna()
        rows.append({
            "system": system,
            "n_source_conditions": int(len(g)),
            "n_directly_admitted": int(g.direct_eligible.fillna(0).sum()),
            "n_selected": int(g.selected.fillna(0).sum()),
            "n_calphad_submitted": int((g.execution_status.isin(
                ["CONVERGED", "FAILED"])).sum()),
            "n_calphad_converged": n_conv,
            "n_calphad_failed": int((g.execution_status == "FAILED").sum()),
            "n_pending": int((g.execution_status == "PENDING").sum()),
            "n_strongly_consistent": n_strong, "n_consistent": n_cons,
            "n_partially_consistent": n_part, "n_discrepant": n_disc,
            "n_not_comparable": vc.get("NOT_COMPARABLE", 0),
            "temperatures_C": ";".join(str(t) for t in
                                       sorted(g.temperature_C.dropna().unique())),
            "raw_phase_support": ("mean raw-phase Jaccard %.3f over %d converged"
                                  % (rj, n_conv)) if n_conv else "pending",
            "family_phase_support": ("mean family Jaccard %.3f; exact family set "
                                     "match in %d of %d"
                                     % (fj, int(conv.family_exact_set_match.sum()),
                                        n_conv)) if n_conv else "pending",
            "phase_composition_support": (
                "%d matched phases, MAE %.2f at.%%, worst %.2f at.%%"
                % (len(matched), matched.MAE_at_pct.mean(),
                   matched.max_abs_error_at_pct.max()))
            if len(matched) else "no matched phase compositions",
            "topology_support": ("element-enrichment direction agrees in %d of %d "
                                 "assessable conditions"
                                 % (int(topo.astype(float).sum()), len(topo)))
            if len(topo) else "not assessable",
            "ML_domain_relevance": ";".join(sorted(
                dom[dom.condition_id.isin(g.condition_id)]
                .TARGET_DOMAIN_RELEVANCE.dropna().unique())) or "n/a",
            "system_relation_class": ";".join(sorted(
                g.system_relation_class.dropna().astype(str).unique())) or "n/a",
            "evidence_strength": label,
            "evidence_strength_score": "%d/6" % total,
            "evidence_strength_parts": "; ".join("%s=%d" % kv
                                                 for kv in parts.items()),
            "remaining_limitation": ""})

    ev = pd.DataFrame(rows)

    # ------------------------------------------------ independence (spec 39) -- #
    dois = [str(d) for d in man.doi.dropna().unique()]
    authors = ["Qiu", "Wang", "Liu", "Zhang", "Li", "Seltzman"]
    scan = scan_database_documentation(dois, authors)
    doi_hits = {d: scan["hits"].get(d, []) for d in dois if d in scan["hits"]}

    limits = {
        "Cu-Cr-Nb": "baseline chemistry, but every condition lies outside the "
                    "sampled FULL_V1 composition domain, and the source is itself a "
                    "CALPHAD assessment of this ternary",
        "Cu-Cr-Ag": "adjacent ternary; carries no Nb and therefore no Laves "
                    "chemistry; the source is itself a CALPHAD assessment",
        "Cu-Cr-Mo": "adjacent ternary at 1100-1300 C, far above the 300-1000 C grid "
                    "the models were built on; the source is itself a CALPHAD "
                    "assessment",
        "Cu-Cr-Co": "adjacent ternary only. It says nothing about Co-Nb Laves "
                    "topology, which is the behaviour the surrogate actually "
                    "predicts for Co-bearing alloys",
        "Cu-Cr-Si": "adjacent ternary; several reported Cr-Si and Cu-Si phases have "
                    "no counterpart in the installed description",
        "Cu-Cr-Ti": "adjacent ternary; the ternary phase tau1 has no supported "
                    "database equivalent and is excluded from the set metric",
        "GRCop (qualitative context)":
            "gas-atomised and laser-powder-bed-fused material, not an equilibrium "
            "study; used only to show that Cr2Nb-containing chemistry is real in "
            "engineering Cu-Cr-Nb alloys",
    }
    caveat = ("independence from the TCCU6 assessment is not established, so this is "
              "a source-grounded consistency check, not independent validation")
    ev["remaining_limitation"] = [
        "%s; %s" % (limits.get(s, ""), caveat) for s in ev.system]
    ev.to_csv(os.path.join(C.D_EVID, "STEP5K_R3_SYSTEM_EVIDENCE.csv"),
              index=False, encoding="utf-8")

    assess = man[man.source_role == "DIRECT_PHASE_EQUILIBRIUM"]
    note = [
        "# STEP5K-R3 - database independence\n\n",
        "_Generated %s._\n\n" % C.utcnow(),
        "## The short answer\n\n",
        D_OPEN + "\n",
        "**These comparisons are source-grounded experimental consistency checks. "
        "They are not statistically independent validation of TCCU6, and the "
        "manuscript must not describe them as such.**\n",
        D_CLOSE + "\n\n",
        "## Why\n\n",
        "Four of the five direct-phase-equilibrium sources are not experimental "
        "papers alone. They are combined experimental *and thermodynamic-assessment* "
        "studies of exactly the ternary being tested:\n\n",
        "| source | system | title carries its own CALPHAD assessment |\n"
        "|---|---|---|\n"]
    for _, r in assess.iterrows():
        own = bool(re.search(r"thermodynamic (calculation|modeling)|CALPHAD",
                             str(r.title), re.I))
        note.append("| %s | %s | %s |\n" % (r.source_id, r.experimental_system,
                                            "yes" if own else "no"))
    note.append(
        "\nA commercial Cu-alloys description is built from exactly this literature. "
        "Where a source both measured the phase equilibria and fitted a description "
        "to them, and that description was subsequently incorporated, the calculation "
        "is partly reproducing its own input. Agreement then measures internal "
        "consistency, which is worth reporting and worth nothing as independent "
        "confirmation.\n\n")
    note.append("## What was checked\n\n")
    if not scan["root_present"]:
        note.append("The installed Thermo-Calc tree was not reachable from this "
                    "interpreter, so no documentary check of the database's "
                    "reference list was possible.\n\n")
    else:
        note.append("A read-only search of %d files under the installed Thermo-Calc "
                    "2025b tree looked for each source's DOI and first-author "
                    "surname.\n\n" % scan["scanned_files"])
        if doi_hits:
            note.append("**Direct citations were found**, which settles the question "
                        "for those sources:\n\n")
            for d, where in doi_hits.items():
                note.append("- `%s` appears in %s\n" % (d, ", ".join(where)))
            note.append("\n")
        else:
            note.append("No source DOI was found in the accessible files. That is "
                        "not evidence of independence: commercial database "
                        "documentation is largely encrypted or not shipped as "
                        "searchable text, and the assessment history is not "
                        "published with the binary. The question stays open, and the "
                        "caveat therefore stands unchanged.\n\n")
    note.append("## What may be claimed\n\n")
    note.append("Safe: *source-grounded experimental consistency checks*; "
                "*materials-level plausibility checks*; *the frozen thermodynamic "
                "description reproduces the phase assemblages these studies "
                "report*.\n\n")
    note.append(D_OPEN + "\n")
    note.append("Not safe: *TCCU6 was independently experimentally validated*; "
                "*the machine-learning model was experimentally validated*; any "
                "single headline experimental-accuracy percentage.\n")
    note.append(D_CLOSE + "\n\n")
    note.append("## The one partial exception\n\n")
    note.append("Source 5 (Cu-Cr-Ti, Li et al. 2021) is titled as an experimental "
                "phase-equilibria investigation and does not present its own "
                "assessment of the ternary. It is the least circular of the six, and "
                "still cannot be called independent, because nothing here establishes "
                "whether the installed description already used it.\n")
    open(os.path.join(C.D_EVID, "STEP5K_R3_DATABASE_INDEPENDENCE_NOTE.md"), "w",
         encoding="utf-8").writelines(note)

    C.write_json(os.path.join(C.D_EVID, "STEP5K_R3_EVIDENCE_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "no_single_global_accuracy_reported": True,
        "per_system": ev.set_index("system").to_dict("index"),
        "strength_rule": "five parts - breadth, temperature coverage, measured phase "
                         "chemistry, mapping quality, agreement - scored 0-6; STRONG "
                         ">= 5, MODERATE 3-4, LIMITED 1-2, NONE when nothing "
                         "converged. Agreement contributes at most one of six points.",
        "database_independence": {
            "established": False,
            "documentary_scan": scan,
            "doi_hits": doi_hits,
            "statement": caveat}})

    print(ev[["system", "n_calphad_converged", "n_strongly_consistent",
              "n_consistent", "n_partially_consistent", "n_discrepant",
              "evidence_strength"]].to_string(index=False))
    print("STAGE 50: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
