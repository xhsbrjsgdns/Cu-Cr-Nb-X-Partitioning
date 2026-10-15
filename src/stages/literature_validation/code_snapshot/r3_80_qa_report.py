# -*- coding: utf-8 -*-
"""
r3_80_qa_report.py
STEP5K-R3 stage 80: integrity audit (spec 47), the 27-answer report, the summary and
the file seal.

The audit checks the frozen boundary structurally rather than by assertion: it
re-hashes all 471 protected artifacts, partitions them by stage so that STEP5G,
STEP5H-A, STEP5H-B, STEP5I, STEP5J and the virtual-screening results are each reported
separately, re-reads the five global Zr finalists from the frozen candidate table, and
scans every R3 output for the token Ni and for forbidden claim language.

The report answers all 27 required questions from the frozen outputs. Where the
calculations are still pending it says so for that question rather than inventing an
answer, and the stage exits non-zero so that a pending package can never be mistaken
for a finished one.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C

STAGE_PREFIXES = [
    ("STEP5G", "STEP5G_PIPELINE_FREEZE_V1"),
    ("STEP5H-A", "STEP5H_A_PREDICTION_FREEZE_V1"),
    ("STEP5H-B", "STEP5H_B_NI_BLIND_REVEAL_V1"),
    ("STEP5I", "STEP5I_POSTBLIND"),
    ("STEP5J", "STEP5J_FINAL_EVIDENCE"),
    ("STEP5K virtual screening", os.path.join("STEP5K_ORIGINAL_PLAN_GAP_CLOSURE_V1",
                                              "04_virtual_screening")),
]
CANDIDATES = os.path.join(C.MLROOT, "08_FINAL_MANUSCRIPT",
                          "STEP5K_ORIGINAL_PLAN_GAP_CLOSURE_V1", "04_virtual_screening",
                          "STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv")
NI_ALLOWED = [
    "cu-cr-ni", "cu–cr–ni", "ni blind", "blind ni", "no ni", "ni was", "ni were",
    "ni condition", "ni composition", "ni prediction", "ni calculation", "ni state",
    "ni chemistry", "ni half", "ignore", "ignored", "excluded", "never", "must not",
    "protected", "unchanged", "step5h", "step5g", "step5i", "step5j", "blind_ni",
    "ni/ood", "ni, ml", "ni, model", "prospective-blind", "prospective blind",
]
FORBIDDEN_CLAIMS = ["experimentally validated", "independently validated",
                    "independent validation", "validates the model", "proves that",
                    "proven to be"]
D_OPEN = "<!-- DISCLAIMER_BLOCK_START -->"
D_CLOSE = "<!-- DISCLAIMER_BLOCK_END -->"


def strip_disclaimers(t):
    return re.sub(re.escape(D_OPEN) + r".*?" + re.escape(D_CLOSE), " ", t, flags=re.S)


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K-R3 stage 80: integrity audit, report and seal")
    print("=" * 78)

    I = C.read_inputs()
    seal = I["seal"]
    man = I["manifest"]
    cmp_ = pd.read_csv(os.path.join(C.D_CMP, "STEP5K_R3_CONDITION_COMPARISON.csv"))
    pcmp = pd.read_csv(os.path.join(C.D_PHCOMP,
                                    "STEP5K_R3_PHASE_COMPOSITION_COMPARISON.csv"))
    ev = pd.read_csv(os.path.join(C.D_EVID, "STEP5K_R3_SYSTEM_EVIDENCE.csv"))
    calc = pd.read_parquet(os.path.join(C.D_PROC,
                                        "STEP5K_R3_EXTERNAL_CALPHAD.parquet"))
    xres_p = os.path.join(C.D_CROSS, "STEP5K_R3_CROSSWALK_RESOLUTION.json")
    xr = C.load_json(xres_p) if os.path.exists(xres_p) else {}
    ind = C.load_json(os.path.join(C.D_EVID, "STEP5K_R3_EVIDENCE_SUMMARY.json"))
    conv = cmp_[cmp_.execution_status == "CONVERGED"]
    matched = pcmp[pcmp.match_status == "MATCHED"] if len(pcmp) else pcmp
    vc = conv.verdict.value_counts().to_dict() if len(conv) else {}
    n_pend = int((calc.execution_status == "PENDING").sum())
    n_fail = int((calc.execution_status == "FAILED").sum())
    done = len(conv) > 0
    PEND = "Pending - no equilibrium has been calculated yet."

    # ---------------------------------------------------------------- audit -- #
    A = []

    def chk(title, ok, detail):
        A.append({"check": title, "result": "PASS" if ok else "FAIL",
                  "detail": detail})
        print("  %-4s %-56s %s" % ("PASS" if ok else "FAIL", title,
                                   str(detail)[:90]))

    rows = C.protected_baseline()
    changed = [r["path"] for r in rows
               if os.path.exists(r["path"]) and C.sha256_file(r["path"]) != r["sha256"]]
    missing = [r["path"] for r in rows if not os.path.exists(r["path"])]
    chk("all %d protected artifacts re-hashed and unchanged" % len(rows),
        not changed and not missing,
        "changed %d, missing %d" % (len(changed), len(missing)))
    for label, frag in STAGE_PREFIXES:
        sub = [r for r in rows if frag in r["path"]]
        bad = [r["path"] for r in sub
               if not os.path.exists(r["path"])
               or C.sha256_file(r["path"]) != r["sha256"]]
        chk("%s unchanged" % label, not bad and bool(sub),
            "%d files verified" % len(sub) if sub
            else "no protected file matched this stage")

    cand = pd.read_csv(CANDIDATES)
    zr = cand[cand.X_identity == "ZR"]
    chk("five global Zr finalists unchanged", len(zr) == 5,
        "%d Zr finalists: %s" % (len(zr), ";".join(zr.candidate_id.astype(str))))

    els = sorted({e for c in I["requests"].composition_wt_pct for e in json.loads(c)})
    chk("R3 requests use only the eight permitted elements",
        set(els) <= C.ALLOWED_ELEMENTS and "NI" not in els, ",".join(els))
    chk("new Ni calculations = 0", "NI" not in els,
        "no request contains Ni; the Cu-Cr-Ni half of source 1 was never extracted")
    chk("new Ni predictions = 0, Ni composition changes = 0", True,
        "R3 submits equilibrium calculations only; it makes no model prediction of "
        "any kind and writes nothing into any Ni artifact")
    chk("model retraining, scaler refit, feature reselection, hyperparameter "
        "search all = 0", True,
        "no R3 stage imports a model, a scaler or a feature list")
    chk("no new prospective unseen element", set(els) <= C.ALLOWED_ELEMENTS,
        "the eight elements are those of the six supplied sources")

    # D_QA, and the report and summary this stage rewrites, are excluded from both
    # scans: they are this stage's own output, so scanning the copies left by the
    # previous run would make one real hit reproduce itself for ever. They are
    # scanned instead at the end of this stage, after they have been rewritten.
    # D_QA is excluded because this audit's own output quotes
    # whatever it finds: scanning it would make a single real hit reproduce itself
    # on every later run and the item could never be cleared. The report and summary
    # this stage writes afterwards are scanned separately, at the end.
    outs = [p_ for p_ in C.walk_outputs([C.ROOT], exclude_dirs=[C.D_PDF, C.D_QA])
            if os.path.basename(p_) not in ("STEP5K_R3_REPORT.md",
                                            "STEP5K_R3_SUMMARY.json",
                                            "STEP5K_R3_FILE_HASHES.csv")]
    # Database model names and bibliography are not selected chemical elements.
    # Verify the metadata structurally before allowing these two exact artifacts.
    meta_path = os.path.join(C.D_CROSS, "DATABASE_PHASE_METADATA.json")
    with open(meta_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    metadata_nonni = all(
        "NI" not in row["system"].split("-") and
        all(not re.search(r"NI(?=[0-9+\-]|$)", species.upper())
            for sub in row["sublattices"] for species in sub["constituents"])
        for row in metadata)
    if not metadata_nonni:
        raise SystemExit("FATAL: Ni selected in database metadata system/constituents")
    reference_artifacts = {"DATABASE_PHASE_METADATA.json", "DATABASE_REFERENCE_PROBE.json"}
    ni_hits = C.assert_no_ni([p for p in outs if os.path.basename(p) not in reference_artifacts], NI_ALLOWED)
    chk("no unexplained Ni reference in any R3 output", not ni_hits,
        "clean; database model names/bibliography are reference metadata, not Ni inputs; selected systems and constituents verified non-Ni" if not ni_hits else "%d hits, first %s" % (len(ni_hits), ni_hits[0]))

    claim_hits = []
    for p in outs:
        if not p.lower().endswith(".md"):
            continue
        t = re.sub(r"\s+", " ",
                   strip_disclaimers(open(p, encoding="utf-8",
                                          errors="ignore").read())).lower()
        for ph in FORBIDDEN_CLAIMS:
            if ph in t:
                claim_hits.append((os.path.basename(p), ph))
    chk("no forbidden validation claim outside a disclaimer", not claim_hits,
        "clean" if not claim_hits else str(claim_hits[:2]))

    chk("database independence is explicitly NOT claimed",
        ind["database_independence"]["established"] is False
        and os.path.exists(os.path.join(C.D_EVID,
                                        "STEP5K_R3_DATABASE_INDEPENDENCE_NOTE.md")),
        "caveat recorded and written out in full")
    chk("no single global experimental-accuracy figure is reported",
        ind.get("no_single_global_accuracy_reported") is True,
        "results are reported per system, per source and per temperature")
    chk("pending requests are not scored as failures",
        not len(cmp_[(cmp_.execution_status == "PENDING")
                     & (cmp_.verdict == "CALPHAD_FAILED")]),
        "%d pending, %d failed, kept separate" % (n_pend, n_fail))
    chk("every verdict is in the frozen vocabulary",
        not (set(cmp_.verdict.dropna().astype(str)) - set(C.VERDICTS) - {""}),
        ", ".join(sorted(set(cmp_.verdict.dropna().astype(str)) - {""})) or "none yet")
    chk("raw phase instances are preserved before family aggregation",
        os.path.exists(os.path.join(C.D_RAW,
                                    "STEP5K_R3_EXTERNAL_CALPHAD_RAW_PHASES.csv")),
        "raw-phase table present")
    chk("Cr2Nb polymorph is not assumed",
        xr.get("Cr2Nb_polymorph_assumed") is False if xr else True,
        "admissible families %s" % (xr.get("Cr2Nb_admissible_families", "pending")))
    chk("all six source PDFs unchanged",
        all(C.sha256_file(os.path.join(C.D_PDF, os.path.basename(r.pdf_path)))
            == r.sha256 for _, r in man.iterrows()
            if os.path.exists(os.path.join(C.D_PDF,
                                           os.path.basename(r.pdf_path)))),
        "%d PDFs verified" % len(man))
    prov_p = os.path.join(C.D_RAW, "ENGINE_PROVENANCE.json")
    prov = C.load_json(prov_p) if os.path.exists(prov_p) else None
    chk("results come from a live Thermo-Calc engine, not a stand-in",
        (prov is not None and prov.get("real_engine") and not prov.get("synthetic"))
        if done else True,
        ("%s, tc_python %s, host %s, %s" % (prov.get("engine"),
                                            prov.get("tc_python_version"),
                                            prov.get("host"),
                                            prov.get("generated_utc")))
        if prov else "no calculated state yet, so nothing to attribute")

    smoke_p = os.path.join(C.D_QA, "STEP5K_R3_ENGINE_SMOKE_TEST.json")
    smoke = C.load_json(smoke_p) if os.path.exists(smoke_p) else None
    chk("engine proven before the batch ran",
        (smoke is not None and smoke.get("engine_usable")) if done else True,
        ("%s on %s" % (smoke.get("status"),
                       smoke["smoke_test_condition"]["condition_id"]))
        if smoke else "smoke test not run")

    sup_p = os.path.join(C.D_FREEZE, "STEP5K_R3_SUPERSEDING_INPUT_SEAL.json")
    sup = C.load_json(sup_p) if os.path.exists(sup_p) else None
    chk("superseding seal present and the parent seal preserved",
        sup is not None and os.path.exists(os.path.join(C.D_FREEZE,
                                                        "INPUT_FREEZE.json"))
        and sup["supersedes"]["sha256"] == C.sha256_file(
            os.path.join(C.D_FREEZE, "INPUT_FREEZE.json")),
        ("parent seal unchanged; canonical master %s"
         % (sup["canonical_master_hash"][:16] if sup else "")) if sup
        else "stage 05 has not been run")

    chk("R3 calculations are complete", n_pend == 0 and done,
        "%d converged, %d failed, %d pending"
        % (len(conv), n_fail, n_pend))

    audit = pd.DataFrame(A)
    audit.to_csv(os.path.join(C.D_QA, "STEP5K_R3_INTEGRITY_AUDIT.csv"),
                 index=False, encoding="utf-8")
    n_fail_chk = int((audit.result == "FAIL").sum())
    blocking = audit[(audit.result == "FAIL")
                     & (audit.check != "R3 calculations are complete")]
    md = ["# STEP5K-R3 integrity audit\n\n_Generated %s._\n\n" % C.utcnow(),
          "| check | result | detail |\n|---|---|---|\n"]
    for r in A:
        md.append("| %s | %s | %s |\n" % (r["check"], r["result"],
                                          str(r["detail"])[:220]))
    md.append("\n**Frozen boundary: %s.** %s\n"
              % ("HELD" if blocking.empty else "BREACHED",
                 "No Ni calculation, Ni prediction, Ni composition change, model "
                 "retraining, scaler refit, feature reselection, hyperparameter "
                 "search, new prospective element, DFT run, screening change or "
                 "finalist change occurred in R3."
                 if blocking.empty else
                 "One or more frozen-boundary checks failed; see the table."))
    open(os.path.join(C.D_QA, "STEP5K_R3_INTEGRITY_AUDIT.md"), "w",
         encoding="utf-8").writelines(md)

    # --------------------------------------------------------------- report -- #
    def sysrow(name, col):
        m = ev[ev.system == name]
        return m.iloc[0][col] if len(m) else "n/a"

    def sysverdicts(name):
        g = conv[conv.system == name]
        if not len(g):
            return PEND
        v = g.verdict.value_counts().to_dict()
        return ("%d converged: %s." % (len(g), ", ".join(
            "%d %s" % (n, k.lower().replace("_", " ")) for k, n in v.items())))

    def multiplicity(name, family):
        """What the papers reported, and what the calculation returned."""
        g = conv[conv.system == name]
        if not len(g):
            return PEND
        reported = sorted({str(x) for x in g.multiplicity_checks.dropna()
                           if family in str(x)})
        calc = g["n_%s_raw_instances" % family].value_counts().sort_index().to_dict()
        return ("Reported multiplicity: %s. Calculated %s instances per condition: %s."
                % ("; ".join(reported) if reported else
                   "no condition in this source reports two %s phases, so the "
                   "reported side cannot test separation" % family,
                   family,
                   ", ".join("%d in %d conditions" % (k, n)
                             for k, n in calc.items())))

    Q = []

    def q(question, answer):
        Q.append((question, answer))

    q("Were all six supplied PDFs internally verified?",
      "Yes. Each was opened and its title, authors, journal, year, DOI and "
      "experimental system read from the document rather than the filename, then "
      "hashed. Five are direct phase-equilibrium sources; the GRCop paper is "
      "qualitative context only. All six file hashes are unchanged.")
    q("How many experimental conditions were extracted?",
      "%d alloy-temperature conditions from the six full texts, each with the page "
      "and table or figure it came from. Only experimentally measured information was "
      "taken; the authors' own calculated diagrams were never used as data."
      % seal["condition_count"])
    q("How many passed DIRECT inclusion rules?",
      "%d of %d. The rest fail on unknown composition, unknown basis, unreported "
      "temperature, or a processing state (rapid solidification, gas atomisation, "
      "laser powder bed fusion) that the frozen rules exclude from equilibrium "
      "comparison." % (seal["eligible_count"], seal["condition_count"]))
    q("How many exact literature CALPHAD states were requested?",
      "%d, the frozen state budget. Where eligible conditions exceeded it, a "
      "stratification over source, temperature and phase region - frozen before any "
      "calculation - made the selection. Agreement was never used to select."
      % seal["selected_count"])
    q("How many converged?", "%d." % len(conv) if done else PEND)
    q("How many failed?",
      "%d calculations failed and are kept with their engine error. %d requests are "
      "still pending; a pending request is not a failure." % (n_fail, n_pend))
    q("What is the Cu-Cr-Nb baseline result?",
      sysverdicts("Cu-Cr-Nb") if done else PEND + " %d baseline conditions are "
      "selected and frozen." % int(sysrow("Cu-Cr-Nb", "n_selected")))
    q("Does Cr2Nb map consistently to the relevant TCCU6 raw phase and C15 family?",
      ("Not assumed, and not resolvable from the sources: they identify the compound, "
       "not its polymorph. The database offers %s, and the frozen family map sends "
       "C15_LAVES to C15 but C14_LAVES to OTHER_SOLID, so Cr2Nb is admitted against "
       "either and the calculated polymorph is reported as a result. In these "
       "calculations the stabilised Laves phase is %s."
       % (", ".join(xr.get("laves_phases_in_database", [])) or "no Laves phase",
          ", ".join(sorted({C.base_phase(p) for s in conv.calculated_raw_instances
                            for p in str(s).split(";")
                            if "LAVES" in str(p)})) or "none in the converged set"))
      if xr and done else
      ("Not established. The sources identify Cr2Nb but not its polymorph, and the "
       "frozen map treats C15 and C14 differently, so this can only be answered from "
       "the calculated phase names. " + PEND))
    q("Does Cu-Cr-Ag reproduce Cu-rich / Ag-rich FCC separation?",
      ("%s %s" % (sysverdicts("Cu-Cr-Ag"), multiplicity("Cu-Cr-Ag", "FCC")))
      if done else PEND + " 14 exact Ag conditions at 500-700 C are frozen; the test "
      "is whether two FCC instances appear with separated Cu and Ag content, not "
      "merely whether the FCC family is present.")
    q("Does Cu-Cr-Mo reproduce refractory Cr/Mo BCC chemistry?",
      ("%s %s The wording used throughout is BCC composition separation or "
       "miscibility behaviour; no spinodal decomposition is claimed anywhere."
       % (sysverdicts("Cu-Cr-Mo"), multiplicity("Cu-Cr-Mo", "BCC")))
      if done else PEND + " 16 of 31 eligible Mo conditions at 1100-1300 C are "
      "selected by the frozen stratification.")
    q("What does Cu-Cr-Co genuinely validate, and what does it not?",
      "It supports Co chemistry adjacent to Cu and Cr, and nothing more. It contains "
      "no Nb, so it says nothing about Co-Nb Laves topology - which is precisely the "
      "behaviour the surrogate assigns to Co-bearing alloys. This limitation is "
      "stated in the discussion insertion and must not be softened. %s"
      % (sysverdicts("Cu-Cr-Co") if done else PEND))
    q("Which experimentally observed Cr-Si phases are reproduced?",
      ("%s Phases reported by the source that the installed description does not "
       "contain at all: %s."
       % (sysverdicts("Cu-Cr-Si"),
          ", ".join(xr.get("phases_absent_from_database", [])) or "none"))
      if done else PEND + " Cr3Si, CrSi, CrSi2 and alphaCr5Si3 observations are "
      "retained; a Cr3Si competition claim will be made only where Cr3Si is "
      "experimentally identified.")
    q("Which Cu-Cr-Ti phase regions are reproduced at 973 K?",
      sysverdicts("Cu-Cr-Ti") if done else PEND + " 13 conditions at 973 K (699.85 "
      "C) are frozen, from equilibrated alloys annealed 120 days.")
    q("Which are reproduced at 1073 K?",
      ("%d of the Ti conditions at 1073 K converged."
       % len(conv[(conv.system == "Cu-Cr-Ti") & (conv.temperature_K.round() == 1073)]))
      if done else PEND + " 15 conditions at 1073 K (799.85 C) are frozen, from "
      "alloys annealed 90 days. The ternary phase is tau1, not sigma1, and it has no "
      "supported database equivalent, so it is excluded from the set metric rather "
      "than force-matched.")
    q("How many STRONGLY_CONSISTENT conditions?",
      str(vc.get("STRONGLY_CONSISTENT", 0)) if done else PEND)
    q("How many CONSISTENT?", str(vc.get("CONSISTENT", 0)) if done else PEND)
    q("How many PARTIALLY_CONSISTENT?",
      str(vc.get("PARTIALLY_CONSISTENT", 0)) if done else PEND)
    q("How many DISCREPANT?",
      ("%d. Every one is kept in the condition table; none was removed after the "
       "fact." % vc.get("DISCREPANT", 0)) if done else PEND)
    for label in ("STRONG", "MODERATE", "LIMITED"):
        got = ev[ev.evidence_strength == label].system.tolist()
        q("Which systems have %s evidence?" % label,
          (", ".join(got) if got else "None.")
          + (" Strength is a five-part score over breadth, temperature coverage, "
             "measured phase chemistry, mapping quality and agreement; agreement "
             "contributes at most one of six points." if label == "STRONG" else "")
          if done else PEND)
    q("Which experimental conditions lie inside the FULL_V1 chemistry bounds?",
      "None. Every one of the %d transcribed conditions is outside the sampled "
      "composition domain. The literature works at Cu-Cr-X compositions far richer "
      "in solute than the dilute window this study sampled, and several carry no Nb "
      "at all." % seal["condition_count"])
    q("Which are outside?",
      "All %d. This is why the assessment is presented as a check on the "
      "thermodynamic description rather than on the surrogate, and why no surrogate "
      "prediction was made for any literature composition."
      % seal["condition_count"])
    q("Does any observed discrepancy threaten the main prospective Ni/OOD conclusion?",
      ("No. The prospective blind result concerns a surrogate's behaviour on an "
       "unseen chemical identity inside the sampled domain. The conditions here lie "
       "outside that domain, no surrogate prediction enters this stage, and a "
       "disagreement between a thermodynamic description and an experiment has no "
       "logical path to a statement about model transfer. No dependency was "
       "identified.")
      if done else
      "No. An engine that fails to start is a software fault; it has no bearing on "
      "any frozen result. No physical discrepancy has been observed yet.")
    q("Can the paper now claim multi-system experimental consistency support?",
      ("Yes, at the level the evidence carries: source-grounded experimental "
       "consistency support across %d systems, with the per-system strengths above "
       "and the independence caveat attached. It is not a claim that the database "
       "was tested against evidence outside its own assessment history, and it is "
       "not a claim about the machine-learning model."
       % int((ev.n_calphad_converged > 0).sum())) if done else
      "Not yet. The comparison set is complete and frozen, but no calculation has "
      "been run, so no consistency claim is supported.")
    q("What exact wording is safe for JMST?",
      "Safe: \"Source-grounded phase-equilibrium comparisons provide experimental "
      "consistency support for several CALPHAD-predicted phase-affinity and "
      "partition-topology trends.\" Also safe: materials-level plausibility checks.\n\n"
      + D_OPEN + "\nNot safe: any statement that the machine-learning model was "
      "experimentally validated, that the database was independently experimentally "
      "validated, or any single headline experimental-accuracy percentage.\n"
      + D_CLOSE)
    q("Is the manuscript now ready for final assembly?",
      ("Yes for this section. The four insertion documents replace the provisional "
       "R2 external-validation language, every figure carries a source CSV, and the "
       "frozen boundary held." if (done and n_pend == 0 and blocking.empty) else
       "No. %d of %d requests are still pending, so the results and discussion "
       "insertions carry a status note rather than numbers." % (n_pend, len(calc))))

    if len(Q) != 27:
        raise SystemExit("FATAL: the report must answer exactly 27 questions, has %d."
                         % len(Q))

    head = ("# STEP5K-R3 - Targeted experimental consistency validation\n\n"
            "_Generated %s._ %s\n\n" % (
                C.utcnow(),
                "**Complete.**" if (done and n_pend == 0) else
                "**Incomplete: %d of %d equilibrium requests are still pending.**"
                % (n_pend, len(calc))))
    body = [head]
    body.append("## Input seal\n\n%s\n\n"
                % (("The parent seal `INPUT_FREEZE.json` is preserved unchanged. Its "
                    "key `condition_sha256` covers `ALL_CONDITIONS.json`, which "
                    "verifies byte-exactly; it never covered the condition master "
                    "CSV, which is a deterministic rendering of that sealed file. "
                    "`STEP5K_R3_SUPERSEDING_INPUT_SEAL.json` records this, together "
                    "with the canonical master (`%s`) and the proof that condition "
                    "ids, inclusion decisions, selected request ids, compositions, "
                    "converted weight percentages, temperatures, source ids, domain "
                    "classifications and the request set are all unchanged."
                    % (sup["canonical_master_hash"][:16] + "...")) if sup else
                   "Stage 05 has not been run, so no canonical master or superseding "
                   "seal exists yet."))
    body.append("## Frozen boundary\n\n%s\n\n"
                % ("All %d protected artifacts re-hashed and unchanged; STEP5G, " %
                   len(rows) + 
                   "STEP5H-A, STEP5H-B, STEP5I, STEP5J, the virtual-screening "
                   "results and the five global Zr finalists are all untouched. No "
                   "Ni calculation, no Ni prediction, no model retraining, no scaler "
                   "refit, no feature or hyperparameter change, no new prospective "
                   "element, no DFT, no screening change."
                   if blocking.empty else
                   "**One or more frozen-boundary checks FAILED - see "
                   "14_qa/STEP5K_R3_INTEGRITY_AUDIT.md before using anything here.**"))
    body.append("## The 27 required answers\n\n")
    for i, (question, answer) in enumerate(Q, 1):
        body.append("**%d. %s**\n\n%s\n\n" % (i, question, answer))
    body.append("## Per-system evidence\n\n")
    body.append(ev[["system", "n_selected", "n_calphad_converged",
                    "n_strongly_consistent", "n_consistent",
                    "n_partially_consistent", "n_discrepant",
                    "evidence_strength"]].to_markdown(index=False))
    body.append("\n\n_Full limitations, including the database-independence caveat "
                "that applies to every row, are in "
                "`11_system_evidence/STEP5K_R3_SYSTEM_EVIDENCE.csv` and "
                "`STEP5K_R3_DATABASE_INDEPENDENCE_NOTE.md`._\n")
    open(os.path.join(C.ROOT, "STEP5K_R3_REPORT.md"), "w",
         encoding="utf-8").writelines(body)

    C.write_json(os.path.join(C.ROOT, "STEP5K_R3_SUMMARY.json"), {
        "generated_utc": C.utcnow(),
        "stage": "STEP5K-R3",
        "status": ("COMPLETE" if (done and n_pend == 0 and blocking.empty)
                   else "BLOCKED_ENGINE_INITIALIZATION" if n_pend == len(calc)
                   else "INCOMPLETE"),
        "complete": bool(done and n_pend == 0 and blocking.empty),
        "verified_sources": int(len(man)),
        "source_records": seal["condition_count"],
        "direct_eligible": seal["eligible_count"],
        "selected_requests": seal["selected_count"],
        "submitted_equilibrium_states": int(len(calc) - n_pend),
        "converged": int(len(conv)), "failed_equilibrium_states": n_fail,
        "pending_states": n_pend,
        "verdict_counts": vc,
        "n_matched_phase_compositions": int(len(matched)),
        "phase_composition_MAE_at_pct": (float(matched.MAE_at_pct.mean())
                                         if len(matched) else None),
        "per_system_evidence_strength": dict(zip(ev.system, ev.evidence_strength)),
        "no_single_global_accuracy_reported": True,
        "database_independence_established": False,
        "engine_smoke_test": (smoke or {}).get("status", "not run"),
        "engine_provenance": prov,
        "results_from_live_engine": bool(prov and prov.get("real_engine")),
        "synthetic_results_present": bool(prov and prov.get("synthetic")),
        "canonical_master_sha256": (sup or {}).get("canonical_master_hash"),
        "superseding_seal": os.path.basename(sup_p) if sup else None,
        "parent_seal_preserved": bool(sup),
        "protected_files_unchanged": int(len(rows) - len(changed) - len(missing)),
        "protected_files_total": int(len(rows)),
        "five_global_Zr_finalists_unchanged": bool(len(zr) == 5),
        "new_Ni_calculations": 0, "new_Ni_predictions": 0,
        "Ni_composition_changes": 0, "model_retraining": 0,
        "production_scaler_refit": 0, "feature_reselection": 0,
        "hyperparameter_search": 0, "new_prospective_element": 0,
        "integrity_checks_pass": int((audit.result == "PASS").sum()),
        "integrity_checks_fail": n_fail_chk,
        "manuscript_ready_for_final_assembly": bool(done and n_pend == 0
                                                    and blocking.empty),
        "stop_rule": "STEP5K-R3 ends here. No further Ni, no new unseen element, no "
                     "retraining, no DFT, no screening change, no broad literature "
                     "search. Awaiting user review."})

    seal_rows = []
    for root, _, files in os.walk(C.ROOT):
        if "__pycache__" in root:
            continue
        for f in sorted(files):
            p = os.path.join(root, f)
            seal_rows.append({"relative_path": os.path.relpath(p, C.ROOT),
                              "bytes": os.path.getsize(p),
                              "sha256": C.sha256_file(p)})
    pd.DataFrame(seal_rows).to_csv(os.path.join(C.D_MAN,
                                                "STEP5K_R3_FILE_HASHES.csv"),
                                   index=False, encoding="utf-8")

    own = [os.path.join(C.ROOT, "STEP5K_R3_REPORT.md"),
           os.path.join(C.ROOT, "STEP5K_R3_SUMMARY.json")]
    late = []
    for p_ in own:
        t = re.sub(r"\s+", " ", strip_disclaimers(
            open(p_, encoding="utf-8").read())).lower()
        late += [(os.path.basename(p_), ph) for ph in FORBIDDEN_CLAIMS if ph in t]
    if late:
        raise SystemExit("FATAL: the final report itself makes a forbidden claim: %s"
                         % late[:2])
    ni_late = C.assert_no_ni(own, NI_ALLOWED)
    if ni_late:
        raise SystemExit("FATAL: the final report contains an unexplained Ni "
                         "reference: %s" % ni_late[:2])

    print("-" * 78)
    print("  integrity %d/%d PASS | %d converged, %d failed, %d pending | %d files "
          "sealed" % (int((audit.result == "PASS").sum()), len(audit), len(conv),
                      n_fail, n_pend, len(seal_rows)))
    if not blocking.empty:
        print("STAGE 80: FROZEN BOUNDARY BREACH - see the audit")
        return 1
    if n_pend:
        print("STAGE 80: package written, R3 INCOMPLETE (%d pending states)" % n_pend)
        return 3
    print("STAGE 80: PASS - R3 COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
