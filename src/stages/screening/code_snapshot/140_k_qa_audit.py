# -*- coding: utf-8 -*-
"""
140_k_qa_audit.py
STEP5K integrity audit - 26 items (spec 49).

Every item is a machine check against a file on disk. An item that cannot be
evaluated because its input is missing is FAIL, never SKIP: a missing artifact is
exactly the failure mode an audit exists to catch. The stage exits non-zero if any
item fails, so the runner cannot report success over a broken package.
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C
import k_scan as SC

# Ni is a subject of discussion in STEP5K prose - the whole point of several
# paragraphs is to say that no Ni was computed. These substrings mark that
# legitimate use; anything else carrying the token NI is a finding.
NI_ALLOWED = [
    "ni blind", "blind ni", "ni zero-shot", "no ni", "ni was", "ni were",
    "ni_states_run", "excluded, and no chemical identity", "ni composition",
    "ni target", "ni prediction", "ni result", "ni interpretation", "ni is",
    "ni state", "ni artifact", "ni anchor", "must not", "never", "excluded",
    "forbidden_x", "cu-cr-ni", "ni_guard", "no_ni_no_unseen_x", "ni-bearing",
    "ni identity", "ni few-shot", "ni blind artifact", "cu–cr–ni",
    # references to the earlier, frozen stages that legitimately DID compute Ni.
    # These are historical citations in the plan audit, not new computation; the
    # question of whether STEP5K itself produced a Ni state is answered separately
    # and structurally by items 5, 6 and 7, which read identities and state counts
    # rather than prose.
    "step5h", "step5g", "step5i", "step5j", "blind_ni", "07_final_blind",
]
REQUIRED = [
    (C.D_CONFIG, "STEP5K_DOI_VERIFICATION.json"),
    (C.D_LIT, "STEP5K_LITERATURE_RAW.csv"),
    (C.D_LIT, "STEP5K_LITERATURE_SEARCH_LOG.md"),
    (C.D_EXTVAL, "STEP5K_EXTERNAL_LITERATURE_VALIDATION.csv"),
    (C.D_EXTVAL, "STEP5K_EXTERNAL_VALIDATION_REPORT.md"),
    (C.D_EXTVAL, "STEP5K_EXTERNAL_VALIDATION_REFERENCES.csv"),
    (C.D_EXTVAL, "STEP5K_EXTERNAL_VALIDATION_SUMMARY.json"),
    (C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_STATES.csv"),
    (C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_LOG.json"),
    (C.D_SCREEN, "STEP5K_SCREENING_RULES_FREEZE.json"),
    (C.D_SCREEN, "STEP5K_VIRTUAL_CANDIDATE_COMPOSITIONS.csv"),
    (C.D_SCREEN, "STEP5K_VIRTUAL_SCREENING_PARETO.csv"),
    (C.D_SCREEN, "STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv"),
    (C.D_SCREEN, "STEP5K_PARETO_FREEZE.json"),
    (C.D_CONFIRM, "STEP5K_CONFIRM_PROCESSED.csv"),
    (C.D_CONFIRM, "STEP5K_SCREENING_CALPHAD_CONFIRMATION.csv"),
    (C.D_CONFIRM, "STEP5K_CONFIRMATION_SUMMARY.json"),
    (C.D_AUDIT, "STEP5K_ORIGINAL_PLAN_COMPLETION_AUDIT.csv"),
    (C.D_AUDIT, "STEP5K_ORIGINAL_PLAN_COMPLETION_SUMMARY.json"),
    (C.D_INSERT, "INSERT_1_METHODS_EXTERNAL_VALIDATION.md"),
    (C.D_INSERT, "INSERT_2_RESULTS_EXTERNAL_VALIDATION.md"),
    (C.D_INSERT, "INSERT_3_METHODS_VIRTUAL_SCREENING.md"),
    (C.D_INSERT, "INSERT_4_RESULTS_VIRTUAL_SCREENING.md"),
]
VERDICTS = {"QUALITATIVELY_CONSISTENT", "PARTIALLY_CONSISTENT",
            "REASONABLE_EQUILIBRIUM_VS_NON_EQUILIBRIUM_DISCREPANCY",
            "NOT_REPRODUCED_BY_EQUILIBRIUM_CALCULATION", "NOT_DIRECTLY_COMPARABLE"}


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K integrity audit (26 items)")
    print("=" * 78)
    R = []

    def item(n, title, fn):
        try:
            ok, detail = fn()
        except Exception as e:                                   # noqa: BLE001
            ok, detail = False, "%s: %s" % (type(e).__name__, str(e)[:220])
        R.append({"item": n, "title": title, "result": "PASS" if ok else "FAIL",
                  "detail": detail})
        print("  %2d %-4s %-58s %s" % (n, "PASS" if ok else "FAIL", title,
                                       str(detail)[:110]))

    # ---- 1-3 structure and protected artifacts ----------------------------- #
    item(1, "all 13 STEP5K directories exist",
         lambda: (all(os.path.isdir(d) for d in C.ALL_DIRS),
                  "%d/%d" % (sum(os.path.isdir(d) for d in C.ALL_DIRS),
                             len(C.ALL_DIRS))))

    def _req():
        miss = [f for d, f in REQUIRED if not os.path.exists(os.path.join(d, f))]
        return (not miss, "missing: %s" % miss if miss
                else "%d required artifacts present" % len(REQUIRED))
    item(2, "every required STEP5K artifact exists", _req)

    def _prot():
        bad = []
        for k, (p, expected) in C.PROTECTED_HASHES.items():
            if C.sha256_file(p) != expected:
                bad.append(k)
        return (not bad, "frozen Ni blind artifacts unchanged" if not bad
                else "CHANGED: %s" % bad)
    item(3, "frozen Ni blind artifacts byte-identical", _prot)

    # ---- 4-7 the Ni prohibition -------------------------------------------- #
    def _ni_text():
        hits = SC.ni_scan(SC.walk_files([C.D_EXTVAL, C.D_EXTCAL, C.D_SCREEN,
                                         C.D_CONFIRM, C.D_INSERT, C.D_AUDIT,
                                         C.D_FIG, C.D_CONFIG, C.ROOT],
                                        exclude_dirs=[C.D_QA]), NI_ALLOWED)
        return (not hits, "clean" if not hits
                else "%d unexplained NI tokens, first: %s" % (len(hits), hits[0]))
    item(4, "no unexplained Ni token in any STEP5K output", _ni_text)

    def _ni_screen():
        c = pd.read_csv(os.path.join(C.D_SCREEN,
                                     "STEP5K_VIRTUAL_CANDIDATE_COMPOSITIONS.csv"))
        bad = sorted(set(c.X_element.astype(str).str.upper()) - set(C.SIX))
        return (not bad, "identities %s" % sorted(set(c.X_element.astype(str)))
                if not bad else "unexpected identities %s" % bad)
    item(5, "screening space contains only the six developed identities", _ni_screen)

    item(6, "confirmation ran zero Ni states",
         lambda: (jload(os.path.join(C.D_CONFIRM,
                                     "STEP5K_CONFIRM_RUN_LOG.json"))["ni_states_run"] == 0,
                  "ni_states_run = 0"))
    item(7, "external CALPHAD ran zero Ni states",
         lambda: (jload(os.path.join(C.D_EXTCAL,
                                     "STEP5K_EXTVAL_CALPHAD_LOG.json"))["ni_states_run"] == 0,
                  "ni_states_run = 0"))

    # ---- 8-14 the screening protocol --------------------------------------- #
    rules_p = os.path.join(C.D_SCREEN, "STEP5K_SCREENING_RULES_FREEZE.json")
    item(8, "screening rules frozen before any prediction",
         lambda: (jload(rules_p)["frozen_before_any_screening_prediction"] is True,
                  "flag true"))

    def _rules_hash():
        pf = jload(os.path.join(C.D_SCREEN, "STEP5K_PARETO_FREEZE.json"))
        h = C.sha256_file(rules_p)
        return (pf["rules_sha256"] == h,
                "ranked against rules %s" % h[:16])
    item(9, "the ranking used the rules file that is on disk now", _rules_hash)

    def _band():
        b = jload(rules_p)["moderate_c15_band"]
        return (("FULL_V1" in b["source"] and b["band_low"] < b["band_high"]
                 and b["n_nonzero_states"] > 0),
                "[%.9g, %.9g] from %d training states"
                % (b["band_low"], b["band_high"], b["n_nonzero_states"]))
    item(10, "moderate C15 band derived from training data, pre-ranking", _band)

    def _noscore():
        pf = jload(os.path.join(C.D_SCREEN, "STEP5K_PARETO_FREEZE.json"))
        par = pd.read_csv(os.path.join(C.D_SCREEN,
                                       "STEP5K_VIRTUAL_SCREENING_PARETO.csv"))
        cols = [c for c in par.columns if "score" in c.lower() or "weight" in c.lower()]
        return (pf["no_weighted_score_used"] is True and not cols,
                "no score/weight column; flag true" if not cols
                else "found %s" % cols)
    item(11, "no weighted score anywhere in the ranking", _noscore)

    def _sweep():
        b = jload(os.path.join(C.D_SCREEN,
                               "STEP5K_PARETO_FREEZE.json"))["bruteforce_check"]
        return (b["identical"] and b["front_size_sweep"] == b["front_size_bruteforce"],
                "front %d == brute force %d on %d sampled"
                % (b["front_size_sweep"], b["front_size_bruteforce"], b["n_sampled"]))
    item(12, "non-dominated sort verified against brute force", _sweep)

    def _fair():
        pf = jload(os.path.join(C.D_SCREEN, "STEP5K_PARETO_FREEZE.json"))
        per = pf["per_family_finalists"]
        return (max(per.values()) <= C.FINALISTS_PER_FAMILY[1]
                and sum(per.values()) <= C.MAX_FINALISTS_TOTAL
                and bool(pf["family_front"]),
                "per family %s, total %d" % (per, sum(per.values())))
    item(13, "family fairness: no identity fills the finalist list", _fair)

    def _cap():
        conf = pd.read_csv(os.path.join(C.D_CONFIRM, "STEP5K_CONFIRM_PROCESSED.csv"))
        return (len(conf) <= 120 and conf.candidate_id.nunique() <= C.MAX_FINALISTS_TOTAL,
                "%d states for %d finalists (caps 120 / %d)"
                % (len(conf), conf.candidate_id.nunique(), C.MAX_FINALISTS_TOTAL))
    item(14, "confirmation stayed inside the frozen state budget", _cap)

    # ---- 15-19 literature integrity ---------------------------------------- #
    dv = jload(os.path.join(C.D_CONFIG, "STEP5K_DOI_VERIFICATION.json"))
    item(15, "every DOI independently re-verified against Crossref",
         lambda: (dv["n_resolved"] == dv["n_dois"] and not dv["unresolved_dois"],
                  "%d/%d resolved" % (dv["n_resolved"], dv["n_dois"])))
    item(16, "every title matches the registered Crossref record",
         lambda: (dv["n_title_match"] == dv["n_dois"],
                  "%d/%d titles, %d/%d journals, %d/%d years"
                  % (dv["n_title_match"], dv["n_dois"], dv["n_journal_match"],
                     dv["n_dois"], dv["n_year_match"], dv["n_dois"])))

    def _prov():
        log = jload(os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_LOG.json"))
        bad = [c["ext_id"] for c in log["compositions"]
               if not str(c["provenance"]).startswith("LIT_")]
        return (not bad, "%d compositions, all traced to a LIT_ record"
                % len(log["compositions"]) if not bad else "untraced: %s" % bad)
    item(17, "no invented composition: each traces to a literature record", _prov)

    item(18, "compositions that could not be pinned down are excluded, with reason",
         lambda: (bool(jload(os.path.join(C.D_EXTCAL,
                                          "STEP5K_EXTVAL_CALPHAD_LOG.json"))
                       ["excluded_on_purpose"]),
                  "exclusions documented"))

    def _vocab():
        v = pd.read_csv(os.path.join(C.D_EXTVAL,
                                     "STEP5K_EXTERNAL_LITERATURE_VALIDATION.csv"))
        bad = sorted(set(v.verdict) - VERDICTS)
        return (not bad, "verdicts %s" % sorted(set(v.verdict)) if not bad
                else "outside vocabulary: %s" % bad)
    item(19, "verdicts drawn only from the closed vocabulary", _vocab)

    # ---- 20-22 claim discipline -------------------------------------------- #
    def _forbidden():
        # D_QA is excluded because this audit's own output quotes whatever it finds:
        # scanning it would make a single real finding reproduce itself on every
        # subsequent run and would make the item impossible to clear. Every file
        # that states a claim is in scope. JSON declaration subtrees (the rules
        # freeze lists the forbidden words in order to forbid them) are skipped by
        # key path inside forbidden_scan, not by filename.
        hits = SC.forbidden_scan(SC.walk_files([C.D_INSERT, C.D_EXTVAL, C.D_SCREEN,
                                                C.D_CONFIRM, C.D_AUDIT, C.D_FIG,
                                                C.D_CONFIG, C.D_LIT, C.ROOT],
                                               exclude_dirs=[C.D_QA]),
                                 C.FORBIDDEN_CANDIDATE_WORDS)
        return (not hits, "clean outside disclaimer and declaration contexts"
                if not hits else "%d hits, first %s" % (len(hits), hits[0][:2]))
    item(20, "no forbidden claim phrase outside a marked disclaimer", _forbidden)

    def _nonblind():
        t = SC.normalise(open(os.path.join(
            C.D_INSERT, "INSERT_4_RESULTS_VIRTUAL_SCREENING.md"),
            encoding="utf-8").read()).lower()
        need = ["not be read as a generalisation estimate", "not comparable to the "
                "prospective blind", "selected, non-blind subset"]
        miss = [n for n in need if n not in t]
        return (not miss, "all three caveats present" if not miss else "missing %s" % miss)
    item(21, "the confirmation result is labelled selected and non-blind", _nonblind)

    def _noretrain():
        flags = []
        for p, k in [(os.path.join(C.D_CONFIRM, "STEP5K_CONFIRM_RUN_LOG.json"),
                      "used_to_retrain_or_retune_anything"),
                     (os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_LOG.json"),
                      "used_to_retrain_or_retune_anything"),
                     (os.path.join(C.D_CONFIRM, "STEP5K_CONFIRMATION_SUMMARY.json"),
                      "used_to_retrain_reselect_or_rerank"),
                     (os.path.join(C.D_EXTVAL,
                                   "STEP5K_EXTERNAL_VALIDATION_SUMMARY.json"),
                      "any_literature_state_used_to_retrain_reselect_or_reinterpret")]:
            flags.append(jload(p)[k] is False)
        return (all(flags), "all four no-feedback flags are false")
    item(22, "nothing computed in STEP5K fed back into any model", _noretrain)

    # ---- 23-24 failed states and the partition formula --------------------- #
    def _failkept():
        a = pd.read_csv(os.path.join(C.D_CONFIRM, "STEP5K_CONFIRM_PROCESSED.csv"))
        b = pd.read_csv(os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_STATES.csv"))
        la = jload(os.path.join(C.D_CONFIRM, "STEP5K_CONFIRM_RUN_LOG.json"))
        lb = jload(os.path.join(C.D_EXTCAL, "STEP5K_EXTVAL_CALPHAD_LOG.json"))
        return (len(a) == la["n_states"] and len(b) == lb["n_states"],
                "confirm %d/%d rows, extval %d/%d rows - failures kept"
                % (len(a), la["n_states"], len(b), lb["n_states"]))
    item(23, "failed CALPHAD states kept, not deleted or interpolated", _failkept)

    item(24, "partitioning recorded as mole based, not volume based",
         lambda: ("phase_moles" in jload(os.path.join(
             C.D_CONFIRM, "STEP5K_CONFIRM_RUN_LOG.json"))["partition_formula"]
             and "volume-fraction partitioning is forbidden" in jload(os.path.join(
                 C.D_CONFIRM, "STEP5K_CONFIRM_RUN_LOG.json"))["partition_formula"],
             "n_i = phase_moles * X_i"))

    # ---- 25-26 plan audit and figures -------------------------------------- #
    def _plan():
        a = pd.read_csv(os.path.join(C.D_AUDIT,
                                     "STEP5K_ORIGINAL_PLAN_COMPLETION_AUDIT.csv"))
        s = jload(os.path.join(C.D_AUDIT,
                               "STEP5K_ORIGINAL_PLAN_COMPLETION_SUMMARY.json"))
        p24 = a.loc[a.plan_item_id == "P24", "status"].iloc[0]
        p14 = a.loc[a.plan_item_id == "P14", "status"].iloc[0]
        ok = (len(a) == 37 and p24 == "COMPLETED_WITH_REVISED_PROTOCOL"
              and p14 != "COMPLETED"
              and "formula" in s["literal_completion"]
              and "formula" in s["scientific_completion"]
              and "does not measure the quality" in s["WHAT_THESE_NUMBERS_ARE_NOT"])
        return (ok, "37 rows | P24 %s | P14 %s | literal %.1f%% | scientific %.1f%%"
                % (p24, p14, s["literal_completion"]["percent"],
                   s["scientific_completion"]["percent"]))
    item(25, "plan audit did not rewrite history; both formulas stated", _plan)

    def _figs():
        png = sorted(os.path.basename(p) for p in glob.glob(
            os.path.join(C.D_FIG, "Fig_K_*.png")))
        src = [p for p in png
               if os.path.exists(os.path.join(C.D_FIG, p[:-4] + "_SOURCE.csv"))]
        return (len(png) == 8 and len(src) == 8,
                "%d figures, %d with SOURCE csv" % (len(png), len(src)))
    item(26, "eight figures, each regenerable from its SOURCE csv", _figs)

    df = pd.DataFrame(R)
    df.to_csv(os.path.join(C.D_QA, "STEP5K_INTEGRITY_AUDIT.csv"), index=False,
              encoding="utf-8")
    n_fail = int((df.result == "FAIL").sum())
    C.write_json(os.path.join(C.D_QA, "STEP5K_INTEGRITY_AUDIT.json"), {
        "generated_utc": C.utcnow(), "n_items": len(df),
        "n_pass": int((df.result == "PASS").sum()), "n_fail": n_fail,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
        "failures": df[df.result == "FAIL"].to_dict("records"),
        "items": df.to_dict("records")})
    print("-" * 78)
    print("INTEGRITY AUDIT: %d/%d PASS -> %s"
          % (len(df) - n_fail, len(df), "PASS" if n_fail == 0 else "FAIL"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
