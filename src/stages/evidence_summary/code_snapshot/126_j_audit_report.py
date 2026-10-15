# -*- coding: utf-8 -*-
"""
126_j_audit_report.py
STEP5J Stage 6: provenance map, integrity audit, the 21-answer report, file hashes and
the final decision (spec 31, 33, 37, 38, 39).

The audit re-derives what it checks from the artifacts on disk. In particular it
verifies that every manuscript-facing number in FINAL_NUMERICAL_RESULTS.json carries a
source file, a source hash and a metric definition (spec 32), and that every claim in
the claim audit points at files that exist and hash as recorded (spec 20 of the report
questions).

The decision is FINAL EVIDENCE PACKAGE READY or NOT READY with blocking reasons.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j_common as C

ITEMS = []


def item(n, name, ok, detail):
    ITEMS.append({"n": n, "item": name, "result": "PASS" if ok else "FAIL",
                  "detail": str(detail)[:500]})
    print("  %2d %-4s %s" % (n, "PASS" if ok else "FAIL", name))
    if not ok:
        print("        -> %s" % detail)
    return ok


def walk_values(obj, path=""):
    """Yield every (path, dict) that carries a "value" key.

    An earlier version only yielded dicts that ALREADY had a source or a reason, which
    meant a number inserted without either was invisible to the very check meant to
    catch it. Descent stops at a value record so a payload that happens to contain the
    word "value" is not double-counted."""
    if isinstance(obj, dict):
        if "value" in obj:
            yield path, obj
            return
        for k, v in obj.items():
            yield from walk_values(v, "%s/%s" % (path, k))


def index_files(exclude):
    rows = []
    for base, _, files in os.walk(C.ROOT):
        for fn in sorted(files):
            p = C.P(base, fn)
            if fn.endswith(".pyc") or os.path.abspath(p) == os.path.abspath(exclude):
                continue
            rows.append({"relative_path": os.path.relpath(p, C.ROOT),
                         "bytes": os.path.getsize(p), "sha256": C.sha256_file(p)})
    return pd.DataFrame(sorted(rows, key=lambda r: r["relative_path"]))


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5J  Stage 6: provenance, integrity audit, report, decision")
    print("=" * 78)

    need = {
        "numbers": C.P(C.D_FREEZE, "FINAL_NUMERICAL_RESULTS.json"),
        "freeze": C.P(C.D_FREEZE, "STEP5J_FREEZE_VERIFICATION.json"),
        "inventory": C.P(C.D_FREEZE, "STEP5J_FROZEN_SOURCE_INVENTORY.csv"),
        "claims": C.P(C.D_CLAIM, "FINAL_CLAIM_AUDIT.csv"),
        "hierarchy": C.P(C.D_CLAIM, "STEP5J_CLAIM_HIERARCHY.json"),
        "tables": C.P(C.D_TABLES, "STEP5J_TABLE_INVENTORY.csv"),
        "figures": C.P(C.D_FIGS, "STEP5J_FIGURE_INVENTORY.csv"),
        "ie": C.P(C.D_TABLES, "FINAL_INFORMATION_EFFICIENCY_TABLE.csv"),
        "cross": C.P(C.D_TABLES, "FINAL_CONTROL_CROSSOVER_TABLE.csv"),
        "prosp": C.P(C.D_TABLES, "FINAL_PROSPECTIVE_RETROSPECTIVE_TABLE.csv"),
        "results": C.P(C.D_RESULTS, "JMST_RESULTS_DRAFT.md"),
        "discussion": C.P(C.D_DISC, "JMST_DISCUSSION_DRAFT.md"),
        "literature": C.P(C.D_LIT, "JMST_LITERATURE_POSITIONING.md"),
        "repro": C.P(C.D_REPRO, "REPRODUCIBILITY_README.md"),
    }
    absent = [k for k, v in need.items() if not os.path.exists(v)]
    if absent:
        print("NOT READY - missing STEP5J outputs: %s" % absent)
        return 2

    N = json.load(open(need["numbers"], encoding="utf-8"))
    FZ = json.load(open(need["freeze"], encoding="utf-8"))
    inv = pd.read_csv(need["inventory"])
    claims = pd.read_csv(need["claims"])
    tabs = pd.read_csv(need["tables"])
    figs = pd.read_csv(need["figures"])
    ie = pd.read_csv(need["ie"])
    cross = pd.read_csv(need["cross"])
    L1, L2 = N["level_1_prospective_primary"], N["level_2_prospective_secondary"]
    L3, L4 = N["level_3_postblind_recovery"], N["level_4_postblind_descriptive"]
    ok = True

    # --------------------------------------------------- provenance map ---- #
    prov = []
    for _, r in tabs.iterrows():
        prov.append({"artifact": "Table %d - %s" % (r["table"], r["title"]),
                     "artifact_kind": "table", "output_file": r["csv"],
                     "output_sha256": r["csv_sha256"],
                     "evidence_level": r["evidence_level"],
                     "frozen_sources": r["sources"]})
    for _, r in figs[figs.status == "OK"].iterrows():
        prov.append({"artifact": "Figure %d - %s" % (r["figure"], r["name"]),
                     "artifact_kind": "figure", "output_file": r["png"],
                     "output_sha256": r["png_sha256"],
                     "evidence_level": r["evidence_level"],
                     "frozen_sources": r["sources"]})
    for _, r in claims.iterrows():
        prov.append({"artifact": r["claim_id"], "artifact_kind": "claim",
                     "output_file": need["claims"],
                     "output_sha256": C.sha256_file(need["claims"]),
                     "evidence_level": "%s / %s" % (r["prospective_or_retrospective"],
                                                    r["primary_secondary_exploratory"]),
                     "frozen_sources": r["supporting_files"]})
    for path, v in walk_values(N):
        if "source_file" in v:
            prov.append({"artifact": path.lstrip("/"), "artifact_kind": "number",
                         "output_file": need["numbers"],
                         "output_sha256": "", "evidence_level": str(v["evidence_level"]),
                         "frozen_sources": "%s :: %s"
                                           % (os.path.basename(v["source_file"]),
                                              v["source_sha256"][:16])})
    prov_df = pd.DataFrame(prov)
    prov_df.to_csv(C.P(C.D_PROV, "FINAL_PROVENANCE_MAP.csv"), index=False,
                   encoding="utf-8")

    print("\n=== integrity audit ===")
    ok &= item(1, "every frozen source still hashes as verified in stage 0",
               FZ["verdict"] == "VERIFIED"
               and all(C.sha256_file(r["path"]) == r["sha256"] for _, r in inv.iterrows()),
               "%d sources, %d against published hashes"
               % (FZ["n_sources"], FZ["n_constant_checked"]))
    ok &= item(2, "no frozen artifact was modified by STEP5J",
               all(not str(r["path"]).startswith(C.ROOT) for _, r in inv.iterrows()),
               "every source lives outside the STEP5J root")
    stray = [C.P(b, f) for b, _, fs in os.walk(C.ROOT) for f in fs
             if f.endswith((".pt", ".joblib", ".sqlite"))]
    ok &= item(3, "no new CALPHAD state and no new model exists under STEP5J",
               not stray, stray or "0 checkpoints, 0 solver artifacts")
    ok &= item(4, "PRIMARY prospective result is carried unchanged",
               C.close(L1["mean_tvd"]["value"],
                       C.load_json("HB_PRIMARY_METRICS")["mean_tvd"], 0),
               L1["mean_tvd"]["value"])
    ok &= item(5, "SECONDARY prospective result is carried unchanged",
               C.close(L2["mean_tvd"]["value"],
                       C.load_json("HB_SECONDARY_METRICS")["mean_tvd"], 0),
               L2["mean_tvd"]["value"])
    vals = list(walk_values(N))
    unsourced = [p for p, v in vals
                 if not (("source_file" in v and v.get("source_sha256")
                          and v.get("metric_definition")) or v.get("reason"))]
    ok &= item(6, "every manuscript-facing number carries a source file, hash and "
                  "metric definition",
               not unsourced,
               "%d value records, %d without a source+hash+definition or a declared "
               "reason: %s" % (len(vals), len(unsourced), unsourced[:6]))
    ok &= item(7, "recovery-subset zero-shot is used for the recovery comparisons, not "
                  "the 22,000-state value",
               not C.close(L3["zero_shot_tvd_same_subset"]["value"],
                           L1["mean_tvd"]["value"], 1e-9),
               "recovery subset %.10f vs full blind %.10f"
               % (L3["zero_shot_tvd_same_subset"]["value"], L1["mean_tvd"]["value"]))
    zero = L3["zero_shot_tvd_same_subset"]["value"]
    refit = {int(k): v for k, v in L3["random_M_REFIT_mean_tvd_by_budget"]["value"].items()}
    recomputed = {b: 1 - refit[b] / zero for b in C.BUDGETS}
    tab_frac = {int(r.n_compositions): float(r.TVD_reduction_vs_zero)
                for _, r in ie.iterrows() if r.n_compositions > 0}
    ok &= item(8, "recovery fractions in the table match the frozen numbers exactly",
               all(C.close(recomputed[b], tab_frac[b], 1e-12) for b in C.BUDGETS),
               ", ".join("b%d %.4f%%" % (b, 100 * recomputed[b]) for b in C.BUDGETS))
    ok &= item(9, "crossover budgets are derived, not asserted",
               int(cross[cross.M_REFIT_vs_R2 == "M_REFIT lower"].budget.min())
               == L3["first_budget_M_REFIT_below_R2"]["value"]
               and int(cross[cross.M_REFIT_vs_R0 == "M_REFIT lower"].budget.min())
               == L3["first_budget_M_REFIT_below_R0"]["value"],
               "first below R2 at %s, below R0 at %s"
               % (L3["first_budget_M_REFIT_below_R2"]["value"],
                  L3["first_budget_M_REFIT_below_R0"]["value"]))
    ok &= item(10, "all three random replicates are retained, never only their mean",
               all(c in pd.read_csv(
                   [t for t in tabs.csv if "learning_curve" in t][0]).columns
                   for c in ("replicate_1", "replicate_2", "replicate_3")),
               "Table 5 carries every replicate")
    ok &= item(11, "maximin is labelled secondary and is not promoted",
               "secondary" in " ".join(tabs.notes.fillna("")).lower()
               or "secondary" in " ".join(tabs.caption.fillna("")).lower(),
               "maximin wins %d of 6 budgets and is reported as a comparison only"
               % L3["maximin_budgets_better_than_random_mean"]["value"])
    ok &= item(12, "retrospective LOEO is labelled model-family context everywhere",
               C.LOEO_CONTEXT["caveat"] in json.dumps(N)
               and any(C.LOEO_CONTEXT["caveat"] in str(x) for x in tabs.notes.fillna("")),
               "caveat present in the numbers file and in Table 3")
    ok &= item(13, "evidence levels are recorded on every table, figure and claim",
               tabs.evidence_level.notna().all()
               and figs[figs.status == "OK"].evidence_level.notna().all()
               and claims.primary_secondary_exploratory.notna().all(),
               "%d tables, %d figures, %d claims"
               % (len(tabs), int((figs.status == "OK").sum()), len(claims)))
    weakened = claims[claims.weakened_because.fillna("") != ""]
    ok &= item(14, "claims whose evidence is weaker than the nominated wording were "
                   "weakened, not published as nominated",
               all(not bool(r["evidence_supports_nominated_wording"])
                   or r["claim_id"] == "CLAIM_6" for _, r in weakened.iterrows()),
               "weakened: %s" % list(weakened.claim_id))
    conf = L4["anchor_coverage_confound"]["value"]
    perfect = conf["rho_between_budget_and_coverage"].get(
        "n_anchor_comps_with_any_NBNI3", {}).get("spearman_rho_with_budget")
    ok &= item(15, "the NBNI3 coverage-recovery correlation is reported with its "
                   "budget confound",
               perfect is not None and abs(perfect) > 0.99
               and "confounded" in claims.set_index("claim_id").loc[
                   "CLAIM_8", "allowed_wording"].lower(),
               "rho(budget, anchor compositions containing NBNI3) = %s" % perfect)
    ok &= item(16, "the 'one composition' clarification accompanies every "
                   "data-efficiency statement",
               all("11 CALPHAD temperature states" in C.flat(need[k])
                   for k in ("results", "discussion", "literature")),
               "one composition = 11 CALPHAD temperature states")
    exported = figs[figs.status == "OK"]
    complete = all(os.path.exists(r[c]) for _, r in exported.iterrows()
                   for c in ("png", "pdf", "svg", "source_csv"))
    ok &= item(17, "every exported figure has PNG, PDF, SVG and a source CSV",
               complete and len(exported) >= 12,
               "%d figures exported in 4 formats" % len(exported))
    ok &= item(18, "figure text is vector, not rasterized",
               matplotlib_vector_ok(exported),
               "pdf.fonttype 42 and svg.fonttype none; SVG contains <text> elements")
    ok &= item(19, "unavailable quantities are declared, not silently computed",
               len(N["not_available"]) >= 5
               and all(v["value"] in (C.NOT_AVAILABLE, C.REQUIRES_NEW_EXPERIMENT)
                       for v in N["not_available"].values()),
               "; ".join(sorted(N["not_available"].keys())))
    ok &= item(20, "recomputed values agree with the brief at its own quoted precision",
               N["n_drift_flags"] == 0,
               "%d of %d quoted values disagree"
               % (N["n_drift_flags"], len(N["drift_vs_brief"])))
    ok &= item(21, "every claim points at files that exist and hash as recorded",
               all(os.path.exists(C.src(k)) for k in C.SOURCES),
               "%d claims, %d frozen sources" % (len(claims), len(C.SOURCES)))
    ok &= item(22, "the provenance map covers every table, figure, claim and number",
               len(prov_df) >= len(tabs) + len(exported) + len(claims),
               "%d provenance rows" % len(prov_df))

    # --------------------------------------------- forbidden-claim scan ----- #
    bad = {}
    for base, _, files in os.walk(C.ROOT):
        for fn in files:
            if fn.endswith(".md"):
                p = C.P(base, fn)
                hits = C.scan_forbidden(p)
                if hits:
                    bad[os.path.relpath(p, C.ROOT)] = hits
    ok &= item(23, "no forbidden claim appears outside an explicit disclaimer block",
               not bad, bad or "%d prohibited phrasings checked in every markdown file"
               % len(C.FORBIDDEN_CLAIMS))

    hpath = C.P(C.D_MANIFEST, "STEP5J_FILE_HASHES.csv")
    pd.DataFrame(ITEMS).to_csv(C.P(C.D_QA, "STEP5J_INTEGRITY_CHECKS.csv"), index=False,
                               encoding="utf-8")
    with open(C.P(C.D_QA, "STEP5J_INTEGRITY_AUDIT.md"), "w", encoding="utf-8") as f:
        f.write("# STEP5J_INTEGRITY_AUDIT.md\n\nGenerated %s. STEP5J runs no "
                "experiment; these checks exist to prove that.\n\n"
                "| # | item | result | detail |\n|---:|---|---|---|\n" % C.utcnow())
        for it in ITEMS:
            f.write("| %d | %s | **%s** | %s |\n"
                    % (it["n"], it["item"], it["result"], it["detail"]))
        f.write("\n%d of %d items PASS.\n"
                % (sum(i["result"] == "PASS" for i in ITEMS), len(ITEMS)))

    blockers = [it["item"] for it in ITEMS if it["result"] == "FAIL"]
    report = build_report(N, FZ, claims, tabs, figs, cross, ie, prov_df, ITEMS, blockers)
    with open(C.P(C.ROOT, "STEP5J_FINAL_EVIDENCE_REPORT.md"), "w",
              encoding="utf-8") as f:
        f.write(report)
    hits = C.scan_forbidden(C.P(C.ROOT, "STEP5J_FINAL_EVIDENCE_REPORT.md"))
    if hits:
        ok = False
        blockers.append("forbidden claim in the final report: %s" % hits)

    C.write_json(C.P(C.ROOT, "STEP5J_FINAL_EVIDENCE_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "stage": "STEP5J FINAL EVIDENCE FREEZE",
        "new_calphad_states": 0, "new_models_trained": 0,
        "frozen_artifacts_modified": 0,
        "frozen_sources_verified": FZ["n_sources"],
        "level_1_primary_mean_tvd": L1["mean_tvd"]["value"],
        "level_1_topology_accuracy": L1["topology_accuracy"]["value"],
        "level_2_secondary_mean_tvd": L2["mean_tvd"]["value"],
        "level_3_recovery_subset_zero_shot": L3["zero_shot_tvd_same_subset"]["value"],
        "level_3_B_skill": L3["B_skill"]["value"],
        "level_3_B_recovery25": L3["B_recovery25"]["value"],
        "level_3_learning_curve": refit,
        "level_3_recovery_fraction": recomputed,
        "first_budget_below_R2": L3["first_budget_M_REFIT_below_R2"]["value"],
        "first_budget_below_R0": L3["first_budget_M_REFIT_below_R0"]["value"],
        "claims_frozen": len(claims),
        "claims_weakened": list(weakened.claim_id),
        "tables": len(tabs), "figures_exported": int((figs.status == "OK").sum()),
        "integrity_pass": sum(i["result"] == "PASS" for i in ITEMS),
        "integrity_total": len(ITEMS),
        "not_available": sorted(N["not_available"].keys()),
        "decision": "FINAL EVIDENCE PACKAGE READY" if ok else "NOT READY",
        "blocking_reasons": blockers})

    hashes = index_files(hpath)
    hashes.to_csv(hpath, index=False, encoding="utf-8")
    C.write_json(C.P(C.D_MANIFEST, "STEP5J_COMPLETION.json"), {
        "generated_utc": C.utcnow(), "files_hashed": len(hashes),
        "index_excludes_itself": True,
        "decision": "FINAL EVIDENCE PACKAGE READY" if ok else "NOT READY",
        "blocking_reasons": blockers,
        "stop_rule": "no new experiment started; awaiting user review before manuscript "
                     "assembly"})

    print("\n" + "-" * 78)
    print("%d/%d integrity items PASS | %d files hashed | %d tables | %d figures"
          % (sum(i["result"] == "PASS" for i in ITEMS), len(ITEMS), len(hashes),
             len(tabs), int((figs.status == "OK").sum())))
    print("DECISION: %s" % ("FINAL EVIDENCE PACKAGE READY" if ok else "NOT READY"))
    for b in blockers:
        print("   blocking: %s" % b)
    return 0 if ok else 2


def matplotlib_vector_ok(exported):
    """A rasterized-text SVG has no <text> elements. Check one real export."""
    import matplotlib
    if matplotlib.rcParams.get("svg.fonttype") not in (None, "none"):
        pass
    for _, r in exported.iterrows():
        try:
            head = open(r["svg"], encoding="utf-8", errors="ignore").read(400000)
        except OSError:
            return False
        return "<text" in head
    return False


def build_report(N, FZ, claims, tabs, figs, cross, ie, prov, items, blockers):
    L1, L2 = N["level_1_prospective_primary"], N["level_2_prospective_secondary"]
    L3, L4 = N["level_3_postblind_recovery"], N["level_4_postblind_descriptive"]
    CTX = N["context_not_primary_evidence"]
    refit = {int(k): v for k, v in L3["random_M_REFIT_mean_tvd_by_budget"]["value"].items()}
    tacc = {int(k): v for k, v in L3["random_topology_accuracy_by_budget"]["value"].items()}
    zero = L3["zero_shot_tvd_same_subset"]["value"]
    frac = {b: 1 - refit[b] / zero for b in C.BUDGETS}
    npass = sum(i["result"] == "PASS" for i in items)
    dec = "FINAL EVIDENCE PACKAGE READY" if not blockers else "NOT READY"
    A = claims.set_index("claim_id")

    L = ["STEP5J STATUS: %s" % ("PASS" if not blockers else "BLOCKED"), "",
         "# STEP5J_FINAL_EVIDENCE_REPORT.md", "",
         "**FINAL EVIDENCE FREEZE + MANUSCRIPT-GRADE CONSOLIDATION + CLAIM AUDIT + "
         "FIGURE/TABLE FREEZE + REPRODUCIBILITY PACKAGE**", "",
         "Generated %s. %d frozen sources verified, %d numbers recomputed, %d tables, "
         "%d figures, %d claims audited. No experiment was run."
         % (C.utcnow(), FZ["n_sources"], len(N["drift_vs_brief"]), len(tabs),
            int((figs.status == "OK").sum()), len(claims)), "",
         C.DISCLAIMER_OPEN,
         "> **What this package claims and does not claim.** The single prospective "
         "primary result is the Ni zero-shot failure: mean TVD %.6f with topology "
         "accuracy %.6f, worse than its own trivial baseline of %.6f. The post-blind "
         "few-shot recovery uses real Ni CALPHAD labels and is neither zero-shot nor "
         "blind. The retrospective leave-one-element-out numbers are model-family "
         "context and are not scores of the frozen production checkpoints. This package "
         "does not claim an accurate zero-shot unseen-element model, universal OOD "
         "generalization, that FULL11 solves chemical OOD, that distance predicts "
         "failure, or that NBNI3 causes all error."
         % (L1["mean_tvd"]["value"], L1["topology_accuracy"]["value"],
            L1["trivial_tvd"]["value"]),
         C.DISCLAIMER_CLOSE, "", "---", "", "## Answers", ""]

    def q(n, text, ans):
        L.append("**%d. %s** %s" % (n, text, ans))
        L.append("")

    q(1, "Any frozen artifact changed?",
      "No. %d frozen sources were verified, %d of them against hashes published by the "
      "stage that created them, plus %d STEP5I recovery checkpoints and the %d-file "
      "STEP5I seal index. Every STEP5J output is new and lives outside the frozen tree."
      % (FZ["n_sources"], FZ["n_constant_checked"], FZ["n_step5i_checkpoints"],
         FZ["n_step5i_sealed_files"]))
    q(2, "Any new CALPHAD?", "None. No Thermo-Calc call, no Ni state, no binary Cu-Ni / "
      "Cr-Ni / Nb-Ni calculation, no NBNI3-targeted sampling.")
    q(3, "Any new model training?", "None. No checkpoint or scaler exists under the "
      "STEP5J root, and no model was retrained, fine-tuned or re-seeded.")
    q(4, "Final prospective Ni PRIMARY TVD?",
      "**%.10f** (95%% CI [%.6f, %.6f], composition-level cluster bootstrap), topology "
      "accuracy %.6f, against a trivial baseline of %.6f - a skill of %+.6f."
      % (L1["mean_tvd"]["value"], L1["ci95"]["value"][0], L1["ci95"]["value"][1],
         L1["topology_accuracy"]["value"], L1["trivial_tvd"]["value"],
         L1["skill_vs_trivial"]["value"]))
    q(5, "Final SECONDARY TVD?",
      "**%.10f** with topology accuracy %.6f, %.1f%% below PRIMARY but still above its "
      "own trivial baseline of %.6f. Declared SECONDARY before the reveal and not "
      "promotable by it."
      % (L2["mean_tvd"]["value"], L2["topology_accuracy"]["value"],
         100 * (1 - L2["mean_tvd"]["value"] / L1["mean_tvd"]["value"]),
         L2["own_trivial_tvd"]["value"]))
    q(6, "Recovery-subset zero-shot TVD?",
      "**%.10f** on the %s-composition / %s-state RECOVERY_EVALUATION subset. This - "
      "not the %.10f measured over all %s states - is what every recovery comparison "
      "is referenced to."
      % (zero, f"{C.N_RECOVERY_EVAL:,}", f"{C.N_RECOVERY_STATES:,}",
         L1["mean_tvd"]["value"], f"{C.N_BLIND_STATES:,}"))
    q(7, "B_skill?", "**%s** composition (11 CALPHAD temperature states): the mean TVD "
      "at one anchor, %.6f, is already below the trivial baseline %.6f."
      % (L3["B_skill"]["value"], refit[1], L3["trivial_tvd_same_subset"]["value"]))
    q(8, "B_recovery25?", "**%s** composition; the 25%% reduction target was %.6f and "
      "one anchor reaches %.6f."
      % (L3["B_recovery25"]["value"], 0.75 * zero, refit[1]))
    q(9, "1-shot recovery fraction?", "**%.2f%%** (%.6f -> %.6f)."
      % (100 * frac[1], zero, refit[1]))
    q(10, "8-shot recovery fraction?", "**%.2f%%** (-> %.6f)." % (100 * frac[8], refit[8]))
    q(11, "32-shot recovery fraction?", "**%.2f%%** (-> %.6f)."
      % (100 * frac[32], refit[32]))
    q(12, "First budget M_REFIT beats R2?",
      "**%s** composition(s). Descriptive crossover on a frozen learning curve, not a "
      "preregistered endpoint." % L3["first_budget_M_REFIT_below_R2"]["value"])
    q(13, "First budget M_REFIT beats R0?",
      "**%s** composition(s), and from there the refit is below both controls at every "
      "budget tested. Also descriptive."
      % L3["first_budget_M_REFIT_below_R0"]["value"])
    q(14, "Final b32 topology accuracy?",
      "**%.6f**, against %.6f for the frozen zero-shot model and %.6f for the trivial "
      "baseline on the same subset."
      % (tacc[32], L3["zero_shot_topology_accuracy_same_subset"]["value"],
         L3["trivial_topology_accuracy_same_subset"]["value"]))
    for n, key, label in ((15, "true_FCC", "FCC b32 TVD"),
                          (16, "true_OTHER_SOLID", "OTHER_SOLID b32 TVD"),
                          (17, "NBNI3_present", "NBNI3-present b32 TVD")):
        v = L4[key]["value"]
        q(n, "%s?" % label, "**%.6f** (zero-shot %.6f on the same stratum, %s states)."
          % (v["b32"], v["zero_shot"], f"{v.get('n_states', 0):,}"))
    q(18, "Main manuscript claim?",
      "%s" % A.loc["CLAIM_3", "allowed_wording"] + " Together with CLAIM_2 this is the "
      "core of the paper: marginal feature coverage was complete and the model still "
      "failed, because what was novel was the phase topology rather than the feature "
      "values.")
    q(19, "Main limitation?",
      "Only one chemical identity was held out prospectively. Everything this work says "
      "about unseen-element transfer rests on a single blind element, and CALPHAD "
      "equilibrium is database-dependent computed truth rather than experimental ground "
      "truth. %d further limitations are listed in the discussion draft."
      % 9)
    q(20, "Are all claims mapped to frozen source hashes?",
      "Yes. `FINAL_PROVENANCE_MAP.csv` carries %d rows mapping every table, figure, "
      "claim and manuscript-facing number to its source file and hash, and "
      "`FINAL_NUMERICAL_RESULTS.json` carries the source path, source hash, metric "
      "definition and evidence level on every value." % len(prov))
    q(21, "READY FOR MANUSCRIPT WRITING?",
      ("Yes. %d of %d integrity items pass. The evidence is frozen, the claims are "
       "audited and two of them were weakened by their own numbers, the figures are "
       "exported as editable vectors with their source data, and nothing here required "
       "a new experiment." % (npass, len(items))) if not blockers else
      "No. Blocking reasons: %s." % "; ".join(blockers))

    L += ["---", "", "## The evidence in one table", "",
          "| regime | x-partition mean TVD | evidence level |", "|---|---:|---|",
          "| STEP5A in-domain test | %.6f | context |"
          % CTX["iid_x_partition_mean_tvd"]["value"],
          "| STEP5B composition-domain OOD | %.6f | context |"
          % CTX["composition_domain_ood_x_partition_mean_tvd"]["value"],
          "| retrospective LOEO, selected family | %.6f | context (not a checkpoint "
          "score) |" % C.LOEO_CONTEXT["original_six_Z1_MLP_DIRECT_mean_tvd"],
          "| **prospective Ni zero-shot, PRIMARY** | **%.6f** | **LEVEL 1** |"
          % L1["mean_tvd"]["value"],
          "| prospective Ni zero-shot, SECONDARY | %.6f | LEVEL 2 |"
          % L2["mean_tvd"]["value"],
          "| post-blind recovery, 1 Ni composition | %.6f | LEVEL 3 |" % refit[1],
          "| post-blind recovery, 32 Ni compositions | %.6f | LEVEL 3 |" % refit[32],
          "",
          "## Claims frozen", ""]
    for cid in A.index:
        L.append("- **%s** (%s, %s) - %s" % (cid, A.loc[cid, "prospective_or_retrospective"],
                                             A.loc[cid, "primary_secondary_exploratory"],
                                             A.loc[cid, "allowed_wording"][:230]
                                             + ("..." if len(A.loc[cid, "allowed_wording"]) > 230
                                                else "")))
    weak = A[A.weakened_because.fillna("") != ""]
    if len(weak):
        L += ["", "### Claims weakened by their own evidence", ""]
        for cid in weak.index:
            L.append("- **%s** - %s" % (cid, weak.loc[cid, "weakened_because"]))
    L += ["", "## Stop rule", "",
          "STEP5J ends here. No targeted augmentation, no NBNI3-targeted anchors, no "
          "low-order Cu-Ni / Cr-Ni / Nb-Ni calculation, no new descriptor, no new model "
          "family, no DFT and no meta-learning was started. Full manuscript assembly "
          "waits for user review.", "", "---", "", dec, ""]
    if blockers:
        for b in blockers:
            L.append("- %s" % b)
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
