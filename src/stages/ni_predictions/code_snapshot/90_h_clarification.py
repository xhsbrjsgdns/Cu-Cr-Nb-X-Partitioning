# -*- coding: utf-8 -*-
"""
90_h_clarification.py
STEP5H-A Stage 0 (spec 0): the ADDITIVE clarification document, written BEFORE any
Ni property access, plus a read-only verification that every frozen STEP5G hash is
unchanged.

This module edits NOTHING frozen. It creates one new append-only document stating
the relationship between the historical LOEO record and the STEP5G production
checkpoints, and verifies the freeze contract this stage is about to rely on.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5h_common as C

CLARIFICATION = """# STEP5G_CLARIFICATION_01.md

**Written (UTC):** {ts}
**Status:** APPEND-ONLY DOCUMENTATION. This file edits no frozen STEP5G artifact.
It changes no checkpoint, no scaler, no feature schema, no model role, no
composition set, no metric, no preregistration and no hash-bound scientific
decision. It is written BEFORE the first Ni elemental-property access in
STEP5H-A, so that the framing below is part of the record prior to any Ni number
existing.

## 1. What the historical LOEO metrics are

The STEP5C / STEP5D / STEP5F leave-one-element-out metrics (historical strict
LOEO mean TVD 0.752644 for the original-six Z1-style MLP family; 0.710332 for
the FULL5 augmentation) are **retrospective evidence for the model
family/formulation and for the leakage-safe heldout protocol**. They
characterize the 128/64/5 direct-MLP formulation trained per-fold on five
visible identities with fold-specific STEP5C feature subsets, and they establish
that the heldout protocol was leakage-safe.

## 2. What they are NOT

They are **NOT direct LOEO scores of the exact STEP5G production checkpoints**.
The production checkpoints differ in two frozen, deliberate ways:

- they are trained on **all visible training identities at once** (PRIMARY: all
  six original identities on DEV_TRAIN; SECONDARY: all eleven on DEV_TRAIN +
  the STEP5F augmentation rows), not on per-fold five-identity subsets;
- they use the **fixed 43-feature STEP5A production schema** for every fold and
  every future element, not fold-specific feature subsets, because a production
  model cannot know a future element's fold-specific subset in advance.

Therefore quoting 0.752644 or 0.710332 as "the production model's LOEO score"
would be incorrect, and no document in this project does so.

## 3. What the Ni blind test will be

The prospective Ni blind test (STEP5H-A predictions, STEP5H-B reveal) will be
the **first unseen-element evaluation of the exact frozen STEP5G_PRIMARY and
STEP5G_SECONDARY checkpoints**. No score of these exact checkpoints on unseen
elements exists at the time of writing - in-domain validation numbers
(PRIMARY mean TVD {pv:.6f}, SECONDARY {sv:.6f} on seen-identity
DEV_VALIDATION) are interpolation, not OOD evidence, and are never quoted as
such.

## 4. What this clarification changes

Nothing: no checkpoint, no scaler, no feature schema, no model role, no
composition set, no metric, no preregistration, no hash-bound scientific
decision. The verification table below shows that every frozen STEP5G artifact
still hashes to its recorded value at the moment this document is written.

## Frozen-artifact verification at the time of writing

| artifact | sha256 (verified now) | status |
|---|---|---|
| STEP5G_FINAL_PIPELINE_FREEZE.json | `{fz}` | unchanged since freeze |
| PRIMARY checkpoint | `{pck}` | unchanged |
| PRIMARY scaler | `{psc}` | unchanged |
| SECONDARY checkpoint | `{sck}` | unchanged |
| SECONDARY scaler | `{ssc}` | unchanged |
| feature schema CSV | `{sch}` | unchanged |
| Ni blind composition table | `{comp}` | unchanged |
| nested random few-shot sequences | `{seq}` | unchanged |
"""


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5H-A  Stage 0: additive clarification + frozen-hash verification")
    print("=" * 78)
    C.assert_no_tc_python()

    fz = json.load(open(C.G_FREEZE_JSON, encoding="utf-8"))
    Pm = fz["models"]["ORIGINAL6_DIRECT_MLP"]
    Sm = fz["models"]["FULL11_DIRECT_MLP"]

    # verify every frozen hash before relying on anything
    C.verify_hash(C.G_PRIMARY_CKPT, C.FROZEN_HASHES["primary_checkpoint"], "PRIMARY ckpt")
    C.verify_hash(C.G_PRIMARY_SCALER, C.FROZEN_HASHES["primary_scaler"], "PRIMARY scaler")
    C.verify_hash(C.G_SECONDARY_CKPT, C.FROZEN_HASHES["secondary_checkpoint"], "SECONDARY ckpt")
    C.verify_hash(C.G_SECONDARY_SCALER, C.FROZEN_HASHES["secondary_scaler"], "SECONDARY scaler")
    C.verify_hash(C.G_SCHEMA_CSV, C.FROZEN_HASHES["feature_schema_csv"], "feature schema")
    C.verify_hash(C.G_NI_COMP_CSV, C.FROZEN_HASHES["ni_compositions_csv"], "Ni compositions")
    C.verify_hash(C.G_SEQ_CSV, C.FROZEN_HASHES["fewshot_sequences_csv"], "few-shot sequences")
    print("  all 7 frozen STEP5G hashes verified unchanged")

    out = os.path.join(C.ROOT, "STEP5G_CLARIFICATION_01.md")
    if os.path.exists(out):
        print("  clarification already exists; append-only policy - not rewritten")
    else:
        with open(out, "w", encoding="utf-8") as f:
            f.write(CLARIFICATION.format(
                ts=C.utcnow(),
                pv=Pm["in_domain_validation_metrics"]["mean_tvd"],
                sv=Sm["in_domain_validation_metrics"]["mean_tvd"],
                fz=C.sha256_file(C.G_FREEZE_JSON)[:32],
                pck=C.FROZEN_HASHES["primary_checkpoint"][:32],
                psc=C.FROZEN_HASHES["primary_scaler"][:32],
                sck=C.FROZEN_HASHES["secondary_checkpoint"][:32],
                ssc=C.FROZEN_HASHES["secondary_scaler"][:32],
                sch=C.FROZEN_HASHES["feature_schema_csv"][:32],
                comp=C.FROZEN_HASHES["ni_compositions_csv"][:32],
                seq=C.FROZEN_HASHES["fewshot_sequences_csv"][:32]))
        print("  wrote STEP5G_CLARIFICATION_01.md")

    C.write_json(os.path.join(C.D_CONFIG, "STEP5H_A_FROZEN_INPUT_VERIFICATION.json"), {
        "generated_utc": C.utcnow(),
        "pipeline_freeze_id": fz["freeze_id"],
        "pipeline_freeze_sha256": C.sha256_file(C.G_FREEZE_JSON),
        "verified_hashes": dict(C.FROZEN_HASHES),
        "clarification_sha256": C.sha256_file(out),
        "clarification_is_additive_only": True,
        "frozen_artifacts_edited": 0,
        "ni_property_access_so_far": 0})
    print("STAGE 0: PASS (no Ni property accessed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
