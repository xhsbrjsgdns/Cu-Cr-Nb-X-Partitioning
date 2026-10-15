# -*- coding: utf-8 -*-
"""
r3_05_repair_seal.py
STEP5K-R3 part A: establish the canonical condition master and a superseding seal.

WHAT THIS STAGE FOUND, stated before anything else, because it corrects an earlier
claim of mine. There is no provenance drift. The parent seal's key `condition_sha256`
does not cover the condition master CSV at all - it covers
`02_source_extraction/ALL_CONDITIONS.json`, which is what the generator hashed:

    condition_sha256 = sha(O/'02_source_extraction/ALL_CONDITIONS.json')

That file verifies byte-exactly against the parent seal. The master CSV was never
sealed, and it never changed; an earlier analysis of mine compared the seal against
the wrong file and reported a drift that does not exist. The key name is misleading,
and this stage supersedes it with an explicitly named one rather than editing it.

The old seal and the old discrepancy record are preserved untouched. Nothing is
deleted and nothing is rewritten.

WHAT IS PROVEN HERE. The master CSV is a deterministic rendering of the sealed JSON:
this stage regenerates it from `ALL_CONDITIONS.json` with the generator's own
construction and compares. Columns are then classified three ways - transcribed from
the sources, derived by the generator and sealed, derived only at render time - so
that a reader can see exactly which numbers are evidence and which are bookkeeping.
Nine invariants are checked against the byte-valid sealed artifacts; any failure stops
the stage.
"""
import io
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C

ALL_COND = os.path.join(C.D_EXTRACT, "ALL_CONDITIONS.json")
MASTER = os.path.join(C.D_EXTRACT, "STEP5K_R3_EXPERIMENTAL_CONDITIONS_MASTER.csv")
CANON = os.path.join(C.D_EXTRACT, "STEP5K_R3_CANONICAL_MASTER.csv")
PROV = os.path.join(C.D_EXTRACT, "STEP5K_R3_CANONICAL_MASTER_PROVENANCE.json")
SUPER = os.path.join(C.D_FREEZE, "STEP5K_R3_SUPERSEDING_INPUT_SEAL.json")
PARENT = os.path.join(C.D_FREEZE, "INPUT_FREEZE.json")
DISCREPANCY = os.path.join(C.D_QA, "STEP5K_R3_INPUT_SEAL_DISCREPANCY.json")
PER_SOURCE = ["ag_mo", "nb_co_grcop", "si_ti"]
# The generator applies exactly one transformation to transcribed text before
# sealing: e['system'] = e['system'].replace('EN DASH', '-'). The per-source files
# print the system name with en dashes, as the journals do. It is named here so the
# comparison below tests evidence rather than punctuation, and so that the one
# permitted transformation is on the record rather than absorbed silently.
GENERATOR_NORMALISATIONS = {
    "system": "replace U+2013 EN DASH with '-' (assemble_inputs.py line: "
              "e['system']=e['system'].replace('\u2013','-'))"}


def normalise_like_generator(key, value):
    if key == "system" and isinstance(value, str):
        return value.replace("\u2013", "-")
    return value

# Added by the generator when it rendered the CSV; absent from the sealed JSON.
RENDER_TIME = ["Cu_original", "Cr_original", "Nb_original", "X_original",
               "phase_1", "phase_2", "phase_3", "phase_4",
               "measured_phase_compositions_available"]
# Computed by the generator BEFORE the JSON was sealed, so they are inside the seal
# but are still derived quantities rather than transcribed evidence.
DERIVED_AND_SEALED = ["composition_wt_pct", "conversion_valid",
                      "TARGET_DOMAIN_RELEVANCE", "system_relation_class"]


def build_master(E):
    """The generator's own master construction, restated exactly."""
    rows = []
    for e in E:
        x = e["X_identity"]
        c = e.get("composition") or {}
        valid = e.get("conversion_valid")
        m = dict(e)
        for k, el in (("Cu_original", "CU"), ("Cr_original", "CR"),
                      ("Nb_original", "NB"), ("X_original", x)):
            m[k] = c.get(el, 0) if valid else c.get(el)
        for i in range(4):
            m["phase_%d" % (i + 1)] = e["phases"][i] if i < len(e["phases"]) else ""
        m["measured_phase_compositions_available"] = bool(e["measured_phases"])
        rows.append(m)
    return [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                 else v) for k, v in r.items()} for r in rows]


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K-R3 stage 05: canonical master and superseding seal")
    print("=" * 78)
    n_prot, _, _ = C.verify_protected()
    print("  %d protected artifacts re-hashed, all unchanged" % n_prot)

    parent = C.load_json(PARENT)
    req_p = os.path.join(C.D_REQ, "STEP5K_R3_CALPHAD_REQUESTS.csv")
    rules_p = os.path.join(C.D_FREEZE, "STEP5K_R3_CONDITION_INCLUSION_FREEZE.md")
    conv_p = os.path.join(C.D_CONV, "STEP5K_R3_COMPOSITION_CONVERSIONS.csv")
    dom_p = os.path.join(C.D_CONV, "STEP5K_R3_DOMAIN_RELEVANCE.csv")
    sel_p = os.path.join(C.D_FREEZE, "CONDITION_SELECTION.csv")

    h = {k: C.sha256_file(p) for k, p in
         (("all_conditions_json", ALL_COND), ("master_csv", MASTER),
          ("request_table", req_p), ("inclusion_freeze", rules_p),
          ("conversion_table", conv_p), ("domain_table", dom_p),
          ("condition_selection", sel_p))}

    parent_ok = {
        "request_sha256": h["request_table"] == parent["request_sha256"],
        "rules_sha256": h["inclusion_freeze"] == parent["rules_sha256"],
        "condition_sha256_vs_ALL_CONDITIONS_json":
            h["all_conditions_json"] == parent["condition_sha256"],
        "condition_sha256_vs_master_csv":
            h["master_csv"] == parent["condition_sha256"]}
    for k in ("request_sha256", "rules_sha256",
              "condition_sha256_vs_ALL_CONDITIONS_json"):
        if not parent_ok[k]:
            raise SystemExit(
                "FATAL: the parent seal does not verify on %s. The sealed inputs are "
                "not intact and nothing downstream may proceed." % k)
    print("  parent seal verifies: request, inclusion rules, and ALL_CONDITIONS.json")
    print("  the parent key `condition_sha256` covers ALL_CONDITIONS.json, not the "
          "master CSV (%s)" % ("as expected" if not
                               parent_ok["condition_sha256_vs_master_csv"] else
                               "and also happens to match the CSV"))

    E = C.load_json(ALL_COND)
    rows = build_master(E)
    df = pd.DataFrame(rows)
    df.to_csv(CANON, index=False, encoding="utf-8-sig")
    canon_sha = C.sha256_file(CANON)

    disk = open(MASTER, "rb").read()
    canon = open(CANON, "rb").read()
    identical = disk == canon
    # If the bytes differ, say precisely how. Content equality is checked column by
    # column; a pure line-terminator difference is named as such rather than left as
    # an unexplained byte delta.
    a = pd.read_csv(io.BytesIO(canon), encoding="utf-8-sig", dtype=str, keep_default_na=False)
    b = pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    same_cols = list(a.columns) == list(b.columns)
    differing = [c for c in a.columns if c in b.columns and not a[c].equals(b[c])]
    content_identical = bool(same_cols and len(a) == len(b) and not differing)
    if identical:
        byte_reason = "byte-identical"
    elif content_identical:
        nl = ("line terminators differ (%d rows + header = %d lines, %d byte delta)"
              % (len(b), len(b) + 1, abs(len(disk) - len(canon)))
              if abs(len(disk) - len(canon)) == len(b) + 1 else
              "identical cell content, different serialisation")
        byte_reason = "content-identical; " + nl
    else:
        byte_reason = ("CONTENT DIFFERS: %d columns differ%s"
                       % (len(differing),
                          "" if same_cols else "; column sets or order differ"))
    print("  canonical master vs on-disk master: %s" % byte_reason)
    if not content_identical:
        raise SystemExit(
            "FATAL: the master CSV is not a faithful rendering of the sealed "
            "ALL_CONDITIONS.json - %s. Columns differing: %s. This is a real "
            "provenance failure and stops the stage."
            % (byte_reason, differing[:8]))

    # ---- column classification --------------------------------------------- #
    sealed_keys = sorted({k for e in E for k in e})
    classification = {}
    for col in a.columns:
        if col in RENDER_TIME:
            cls, why = "DERIVED_AT_RENDER_TIME", (
                "added when the CSV was written from the sealed JSON; not part of "
                "the sealed record")
        elif col in DERIVED_AND_SEALED:
            cls, why = "DERIVED_AND_SEALED", (
                "computed by the generator from transcribed evidence, then written "
                "into ALL_CONDITIONS.json before the seal was taken")
        elif col in sealed_keys:
            cls, why = "SOURCE_TRANSCRIBED", (
                "transcribed from the printed source and carried into the sealed "
                "record unchanged")
        else:
            cls, why = "UNCLASSIFIED", "not present in the sealed record"
        classification[col] = {"class": cls, "reason": why}
    unclassified = [c for c, v in classification.items()
                    if v["class"] == "UNCLASSIFIED"]
    if unclassified:
        raise SystemExit("FATAL: %d master columns are not accounted for by the "
                         "sealed record: %s" % (len(unclassified), unclassified))

    # ---- did the sealed JSON alter anything transcribed? -------------------- #
    per_source_check = {"performed": False}
    src_rows = []
    for folder in PER_SOURCE:
        p = os.path.join(C.D_EXTRACT, folder, "conditions.json")
        if os.path.exists(p):
            src_rows += C.load_json(p)
    if src_rows:
        by_id = {e["condition_id"]: e for e in E}
        mismatch = []
        for s in src_rows:
            t = by_id.get(s["condition_id"])
            if t is None:
                mismatch.append((s["condition_id"], "missing from the sealed record"))
                continue
            for k, v in s.items():
                if k in t and t[k] != normalise_like_generator(k, v):
                    mismatch.append((s["condition_id"], k))
        per_source_check = {
            "performed": True, "n_source_records": len(src_rows),
            "n_mismatched_fields": len(mismatch), "first_mismatches": mismatch[:8],
            "conclusion": ("every field transcribed in the per-source extraction "
                           "files is carried into the sealed record unchanged; the "
                           "sealed record adds derived keys, applies the one named "
                           "normalisation above, and alters nothing else"
                           if not mismatch else
                           "the sealed record differs from the per-source extraction")}
        if mismatch:
            raise SystemExit("FATAL: the sealed record changed transcribed evidence: "
                             "%s" % mismatch[:5])
        print("  per-source extraction -> sealed record: %d records, 0 altered fields"
              % len(src_rows))

    # ---- the nine invariants ------------------------------------------------ #
    req = pd.read_csv(req_p)
    sel = pd.read_csv(sel_p)
    conv = pd.read_csv(conv_p)
    dom = pd.read_csv(dom_p)
    cm = pd.read_csv(CANON, encoding="utf-8-sig")
    ids = set(cm.condition_id)
    inv = {}
    inv["condition IDs unchanged"] = (
        ids == {e["condition_id"] for e in E} and cm.condition_id.is_unique)
    inv["inclusion decisions unchanged"] = bool(
        (cm.set_index("condition_id").direct_eligible.astype(bool)
         == sel.set_index("condition_id").direct_eligible.astype(bool)).all())
    inv["selected request IDs unchanged"] = (
        set(sel.loc[sel.selected.astype(bool), "condition_id"])
        == set(req.condition_id))
    cmi = cm.set_index("condition_id")
    cvi = conv.set_index("condition_id")
    comp_ok = trueconv_ok = True
    for cid in req.condition_id:
        a_ = {k.upper(): round(float(v), 9)
              for k, v in json.loads(cmi.loc[cid, "composition_wt_pct"]).items()}
        b_ = {k.upper(): round(float(v), 9) for k, v in json.loads(
            req.loc[req.condition_id == cid, "composition_wt_pct"].iloc[0]).items()}
        c_ = {k.upper(): round(float(v), 9) for k, v in json.loads(
            cvi.loc[cid, "converted_composition_wt_pct"]).items()}
        comp_ok &= (a_ == b_)
        trueconv_ok &= (a_ == c_)
    inv["compositions unchanged (master vs sealed request table)"] = bool(comp_ok)
    inv["converted wt.% values unchanged (master vs conversion table)"] = \
        bool(trueconv_ok)
    inv["temperatures unchanged"] = bool(
        (cmi.loc[req.condition_id, "temperature_K"].to_numpy()
         == req.temperature_K.to_numpy()).all()
        and (cmi.loc[req.condition_id, "temperature_C"].to_numpy()
             == req.temperature_C.to_numpy()).all())
    inv["source IDs unchanged"] = bool(
        (cmi.loc[req.condition_id, "source_id"].to_numpy()
         == req.source_id.to_numpy()).all())
    inv["domain classifications unchanged"] = bool(
        (cmi.TARGET_DOMAIN_RELEVANCE.reindex(dom.condition_id).to_numpy()
         == dom.TARGET_DOMAIN_RELEVANCE.to_numpy()).all()
        and (cmi.system_relation_class.reindex(dom.condition_id).to_numpy()
             == dom.system_relation_class.to_numpy()).all())
    inv["request set unchanged"] = (
        len(req) == parent["selected_count"] and set(req.condition_id) <= ids
        and h["request_table"] == parent["request_sha256"])
    for k, v in inv.items():
        print("    %-62s %s" % (k, "OK" if v else "FAILED"))
    if not all(inv.values()):
        raise SystemExit("FATAL: invariant(s) failed: %s"
                         % [k for k, v in inv.items() if not v])

    # ---- artifacts ---------------------------------------------------------- #
    C.write_json(PROV, {
        "generated_utc": C.utcnow(),
        "canonical_master": os.path.basename(CANON),
        "canonical_master_sha256": canon_sha,
        "regenerated_from": "02_source_extraction/ALL_CONDITIONS.json",
        "regenerated_from_sha256": h["all_conditions_json"],
        "generator": "code_snapshot/assemble_inputs.py, master construction restated "
                     "verbatim in r3_05_repair_seal.build_master",
        "on_disk_master_sha256": h["master_csv"],
        "canonical_vs_on_disk": byte_reason,
        "content_identical": content_identical,
        "provenance_chain": [
            "printed source PDF (hash in STEP5K_R3_SOURCE_MANIFEST.csv)",
            "02_source_extraction/<source>/conditions.json (manual transcription)",
            "02_source_extraction/ALL_CONDITIONS.json (SEALED by the parent seal)",
            "02_source_extraction/STEP5K_R3_EXPERIMENTAL_CONDITIONS_MASTER.csv "
            "(deterministic rendering; never itself sealed)"],
        "per_source_to_sealed_record_check": per_source_check,
        "generator_normalisations": GENERATOR_NORMALISATIONS,
        "column_classification": classification,
        "column_class_counts": pd.Series(
            [v["class"] for v in classification.values()]).value_counts().to_dict(),
        "scientific_columns": sorted(c for c, v in classification.items()
                                     if v["class"] == "SOURCE_TRANSCRIBED"),
        "derived_columns_inside_the_seal": sorted(
            c for c, v in classification.items()
            if v["class"] == "DERIVED_AND_SEALED"),
        "derived_columns_added_at_render_time": sorted(
            c for c, v in classification.items()
            if v["class"] == "DERIVED_AT_RENDER_TIME"),
        "invariants": inv})

    C.write_json(SUPER, {
        "seal_id": "STEP5K_R3_SUPERSEDING_INPUT_SEAL",
        "generated_utc": C.utcnow(),
        "supersedes": {"file": "03_inclusion_freeze/INPUT_FREEZE.json",
                       "sha256": C.sha256_file(PARENT),
                       "frozen_utc": parent.get("frozen_utc"),
                       "status": "PRESERVED UNCHANGED; not edited, not deleted"},
        "parent_seal_recorded_condition_sha256": parent["condition_sha256"],
        "parent_key_actually_covers":
            "02_source_extraction/ALL_CONDITIONS.json",
        "original_master_recorded_hash": None,
        "original_master_recorded_hash_note":
            "there is none. The master CSV was never covered by the parent seal; the "
            "key named condition_sha256 hashes ALL_CONDITIONS.json, as the generator "
            "source shows.",
        "current_pre_repair_master_hash": h["master_csv"],
        "canonical_master_hash": canon_sha,
        "exact_reason_for_apparent_drift":
            "no drift occurred. An earlier check of mine compared the parent seal's "
            "condition_sha256 against the master CSV; that key covers "
            "ALL_CONDITIONS.json, which verifies byte-exactly. The master CSV is a "
            "deterministic rendering of that sealed file and is %s."
            % byte_reason,
        "columns_added_or_changed_relative_to_the_sealed_record": {
            "added_at_render_time": sorted(
                c for c, v in classification.items()
                if v["class"] == "DERIVED_AT_RENDER_TIME"),
            "changed": []},
        "proof_scientific_request_content_unchanged": inv,
        "hashes": {
            "ALL_CONDITIONS.json (parent-sealed)": h["all_conditions_json"],
            "STEP5K_R3_CANONICAL_MASTER.csv": canon_sha,
            "STEP5K_R3_EXPERIMENTAL_CONDITIONS_MASTER.csv": h["master_csv"],
            "STEP5K_R3_CALPHAD_REQUESTS.csv": h["request_table"],
            "STEP5K_R3_CONDITION_INCLUSION_FREEZE.md": h["inclusion_freeze"],
            "STEP5K_R3_COMPOSITION_CONVERSIONS.csv": h["conversion_table"],
            "STEP5K_R3_DOMAIN_RELEVANCE.csv": h["domain_table"],
            "CONDITION_SELECTION.csv": h["condition_selection"]},
        "parent_seal_verification": parent_ok,
        "preserved_discrepancy_record": (
            os.path.relpath(DISCREPANCY, C.ROOT) if os.path.exists(DISCREPANCY)
            else "not present on this machine; the misidentification it recorded is "
                 "described here instead"),
        "no_scientific_content_was_altered_by_this_stage": True})

    print("  wrote %s" % os.path.basename(CANON))
    print("  wrote %s" % os.path.basename(PROV))
    print("  wrote %s" % os.path.basename(SUPER))
    print("STAGE 05: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
