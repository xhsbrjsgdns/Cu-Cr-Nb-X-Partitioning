# -*- coding: utf-8 -*-
"""
83_g_ni_prereg.py
STEP5G Stage 3: the prospective Ni blind test - COMPOSITIONS AND PREREGISTRATION ONLY
(spec 31-45).

What this stage produces: a table of 2,000 Cu-Cr-Nb-Ni compositions in which "NI" is
a SYMBOL in a column, and a preregistration document that fixes, before any Ni number
exists, exactly what will be computed and exactly how it will be scored.

What this stage does NOT do, and what the code below is written to make impossible:
  - read a Ni elemental descriptor or any Ni elemental property
  - compute a Ni-Cu / Ni-Cr / Ni-Nb Miedema value or any Ni similarity
  - featurize a Ni composition
  - predict anything for Ni
  - run Thermo-Calc / TC-Python on a Ni system
  - touch a Ni phase label, partition label or target
The composition inversion uses the atomic masses of CHROMIUM and NIOBIUM only
(spec 38). That is the whole point: the Ni weight percent is a composition variable
the sampler chose, not a property of nickel. Because no Ni atomic mass is available
here, atomic fractions are deliberately NOT computed and NOT stored - featurization
belongs to STEP5H-A before the CALPHAD blind reveal. The absence of an atomic-closure check in
this stage's QA is therefore intentional, and is recorded as such rather than quietly
skipped.

The 2,000 compositions are also, in advance, split into the post-blind few-shot
calibration and recovery-evaluation sets by a salted hash of the composition id
(spec 61), so that split cannot be redrawn once the blind result is known.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_common as C
import g_contract as G

CHECKS = []


def chk(cid, what, ok, detail=""):
    CHECKS.append({"id": cid, "check": what, "status": "PASS" if ok else "FAIL",
                   "detail": str(detail)[:400]})
    print("  [%-4s] %-4s %s%s" % (cid, "PASS" if ok else "FAIL", what,
                                  ("  |  " + str(detail)[:150]) if detail else ""))
    return ok


def generate():
    """Sobol d=3 scramble, one deterministic stream, identical rule to FULL_V1 and
    STEP5F. S and R are sampled and inverted; Cr and Nb are never sampled
    independently."""
    seed = C.child_seed(C.NI_MASTER_SEED, "NI_BLIND_COMPOSITIONS")
    sob = qmc.Sobol(d=3, scramble=True, seed=seed)
    rows, seen, tries = [], set(), 0
    while len(rows) < C.NI_N_COMPOSITIONS and tries < 2_000_000:
        p = sob.random(1)[0]
        tries += 1
        s = C.S_CRNB_RANGE[0] + p[0] * (C.S_CRNB_RANGE[1] - C.S_CRNB_RANGE[0])
        r = C.R_CRNB_RANGE[0] + p[1] * (C.R_CRNB_RANGE[1] - C.R_CRNB_RANGE[0])
        x = C.NI_WT_RANGE[0] + p[2] * (C.NI_WT_RANGE[1] - C.NI_WT_RANGE[0])
        w_cr_e, w_nb_e = C.invert_s_r(s, r)          # M_Cr and M_Nb only (spec 38)
        w_cr = round(w_cr_e, C.WT_DECIMALS)
        w_nb = round(w_nb_e, C.WT_DECIMALS)
        w_x = round(x, C.WT_DECIMALS)
        key = (w_cr, w_nb, w_x)
        if key in seen or w_cr <= 0 or w_nb <= 0 or w_x <= 0:
            continue
        seen.add(key)
        cu = round(100.0 - w_cr - w_nb - w_x, 6)
        cid = C.NI_ID_FORMAT % (len(rows) + 1)
        rows.append({
            "composition_id": cid,
            "family": "NI", "X_element": "NI",
            "sampling_domain": "A",
            "split_role": "NI_BLIND_PROSPECTIVE",
            "composition_source": "STEP5G_prospective_Ni_blind",
            "sampling_method": "Sobol(d=3, scramble=True)",
            "master_seed": C.NI_MASTER_SEED, "stream_seed": seed,
            "sobol_index": tries - 1,
            "Cu_wt_pct": cu, "Cr_wt_pct": w_cr, "Nb_wt_pct": w_nb, "X_wt_pct": w_x,
            "S_CrNb_wt_pct": round(w_cr + w_nb, 6),
            "R_CrNb_atomic_target": round(r, 6),
            "R_CrNb_atomic_actual": round(C.actual_ratio(w_cr, w_nb), 6),
            "S_CrNb_target": round(s, 6),
            "composition_sum_wt_pct": round(cu + w_cr + w_nb + w_x, 9),
            "n_temperatures": len(C.TEMPERATURES_C),
            "fewshot_role": None,           # filled below, deterministically
            "fewshot_rank": None,
        })
    if len(rows) < C.NI_N_COMPOSITIONS:
        raise SystemExit("FATAL: only %d unique compositions after %d Sobol points"
                         % (len(rows), tries))
    df = pd.DataFrame(rows)
    print("  %d Sobol points -> %d unique compositions (stream seed %d)"
          % (tries, len(df), seed))

    # deterministic, salted, composition-level few-shot split (spec 61), fixed NOW
    order = np.argsort([C.fewshot_split_role(c) for c in df.composition_id],
                       kind="stable")
    role = np.array(["RECOVERY_EVALUATION"] * len(df), dtype=object)
    role[order[:C.N_CALIBRATION]] = "FEWSHOT_CALIBRATION"
    rank = np.empty(len(df), dtype=int)
    rank[order] = np.arange(len(df))
    df["fewshot_role"] = role
    df["fewshot_rank"] = rank
    return df, seed, tries


PREREG = """# STEP5G_NI_BLIND_PREREGISTRATION

**Written (UTC):** {ts}
**Status:** the compositions below exist; no Ni thermodynamic state, label, target or
prediction exists. This document is written BEFORE any of them can.

## What is being tested

Whether a model that has never seen nickel - in any descriptor, property, similarity
score, label or calculation - can predict the five-family equilibrium X-partition of
nickel in Cu-Cr-Nb-Ni. This is a **prospective zero-shot test on an unseen chemical
identity**. It is not a retrospective leave-one-element-out fold, and the two must not
be described in the same words.

## The design, fixed here

| | |
|---|---|
| compositions | {n} (`NI_BLIND_000001` - `NI_BLIND_{nlast}`) |
| temperatures | {temps} degC |
| formal states | {states} |
| sampler | Sobol(d=3, scramble=True), stream seed {seed} derived from master seed {master} |
| Sobol points drawn | {tries} (duplicates at 4 decimals skipped, sequence continued) |
| S_CrNb | {s0} - {s1} wt.% |
| R_CrNb_atomic | {r0} - {r1} |
| Ni | {x0} - {x1} wt.% |
| Cu | balance |
| composition inversion | `w_Nb = S / (1 + R * M_Cr / M_Nb)`, `w_Cr = S - w_Nb`, using the atomic masses of Cr ({mcr}) and Nb ({mnb}) ONLY |
| thermodynamics (STEP5H) | Thermo-Calc 2025b, TCCU6, 1 bar, 1 mol, wt.%, Cu balance - the unedited frozen FULL_V1 runner |

The composition domain is identical to Domain A and to the STEP5F augmentation
domain. Nothing about it was chosen using Ni similarity, expected Ni phase behaviour,
or proximity of Ni to any training element - no such quantity was computed.

## What is predicted, and by which frozen model

The five-family X-partition `P_Ni^alpha = n_Ni^alpha / sum_beta n_Ni^beta` with
`n_Ni^alpha = phase_moles_alpha * X_Ni^alpha`, over the frozen family order
{families}. Volume-fraction-based partition is forbidden and is not used anywhere.

| role | model | checkpoint sha256 |
|---|---|---|
| PRIMARY | `{p_id}` ({p_role}) | `{p_sha}` |
| SECONDARY | `{s_id}` ({s_role}) | `{s_sha}` |

Both were frozen in `STEP5G_FINAL_PIPELINE_FREEZE.json` before this file was written,
and neither may be retrained, fine-tuned, re-seeded, re-scaled or given a new feature
afterwards (spec 30). PRIMARY is the reported result; SECONDARY is reported alongside
it, not selected between after the fact.

## Order of operations, binding

1. This document and the composition table are hashed. (done in this stage)
2. STEP5H-A reads ONLY approved Ni INPUT properties required by the 43-feature schema;
   records first-access UTC; builds features and freezes PRIMARY and SECONDARY predictions
   as parquet/CSV with SHA256. STOP. No CALPHAD or Ni target access is allowed in STEP5H-A.
3. STEP5H-B may run Ni CALPHAD ONLY after BOTH prediction hashes exist and verify.
4. Score PRIMARY first and freeze its score/hash; then score SECONDARY and freeze its score/hash.
5. Only after both zero-shot score files are frozen may the preregistered post-blind recovery run.
6. Report the result whatever it is. No post-hoc success threshold is allowed.

Failure handling follows the frozen FULL_V1 policy: a state that fails to converge
after the full solver escalation is RECORDED AS FAILED. It is never deleted,
interpolated, replaced by a neighbouring temperature, or silently dropped, and the
failure count is reported with the result.

## Ni status at the moment this file is written

| | |
|---|---|
| Ni symbolic composition rows | {n} (allowed) |
| Ni elemental descriptor access | 0 |
| Ni elemental property access | 0 |
| Ni Miedema / similarity computation | 0 |
| Ni prediction | 0 |
| Ni CALPHAD / TC-Python state | 0 |
| Ni phase / partition / target access | 0 |

## Post-blind few-shot split, fixed in advance

The {n} compositions are already partitioned at composition level into
{ncal} `FEWSHOT_CALIBRATION` and {nev} `RECOVERY_EVALUATION` by the frozen rule
`{splitrule}`. The split is in the composition table shipped with
this document, so it cannot be redrawn after the blind result is seen. No
recovery-evaluation composition may ever be used as an anchor.
"""


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5G  Stage 3: prospective Ni blind - compositions + preregistration")
    print("=" * 78)

    fz_path = os.path.join(C.D_FREEZE, "STEP5G_FINAL_PIPELINE_FREEZE.json")
    if not os.path.exists(fz_path):
        print("FATAL: the production pipeline must be frozen before the Ni blind is "
              "preregistered. Run 82_g_train.py first.")
        return 2
    import json
    fz = json.load(open(fz_path, encoding="utf-8"))
    p = fz["models"][C.PRIMARY_ROLE]
    s = fz["models"][C.SECONDARY_ROLE]

    for model in (p,s):
        for kind in ("checkpoint","scaler"):
            assert C.sha256_file(model[kind]) == model[kind+"_sha256"]
    if os.path.exists(os.path.join(C.D_NIPRE,"STEP5G_NI_BLIND_PREREGISTRATION.json")):
        raise RuntimeError("Ni preregistration already exists: refusing overwrite")
    df, seed, tries = generate()

    print("\n=== composition QA ===")
    ok = True
    ok &= chk("N01", "exactly %d compositions" % C.NI_N_COMPOSITIONS,
              len(df) == C.NI_N_COMPOSITIONS, len(df))
    ok &= chk("N02", "composition_id unique and formatted NI_BLIND_000001..",
              df.composition_id.nunique() == len(df)
              and bool(df.composition_id.str.fullmatch(C.NI_ID_REGEX).all()),
              "%s .. %s" % (df.composition_id.iloc[0], df.composition_id.iloc[-1]))
    ok &= chk("N03", "no duplicate (Cr, Nb, X) key at 4 decimals",
              not df.duplicated(["Cr_wt_pct", "Nb_wt_pct", "X_wt_pct"]).any(),
              int(df.duplicated(["Cr_wt_pct", "Nb_wt_pct", "X_wt_pct"]).sum()))
    ok &= chk("N04", "S_CrNb within the frozen Domain-A range",
              bool((df.S_CrNb_wt_pct >= C.S_CRNB_RANGE[0] - 5e-4).all()
                   and (df.S_CrNb_wt_pct <= C.S_CRNB_RANGE[1] + 5e-4).all()),
              "[%.4f, %.4f]" % (df.S_CrNb_wt_pct.min(), df.S_CrNb_wt_pct.max()))
    ok &= chk("N05", "R_CrNb_atomic_actual within the frozen range",
              bool((df.R_CrNb_atomic_actual >= C.R_CRNB_RANGE[0] - 5e-3).all()
                   and (df.R_CrNb_atomic_actual <= C.R_CRNB_RANGE[1] + 5e-3).all()),
              "[%.4f, %.4f]" % (df.R_CrNb_atomic_actual.min(),
                                df.R_CrNb_atomic_actual.max()))
    ok &= chk("N06", "Ni wt.% within 0.02-0.20, identical to every added identity",
              bool((df.X_wt_pct >= C.NI_WT_RANGE[0] - 5e-5).all()
                   and (df.X_wt_pct <= C.NI_WT_RANGE[1] + 5e-5).all()),
              "[%.4f, %.4f]" % (df.X_wt_pct.min(), df.X_wt_pct.max()))
    ok &= chk("N07", "Cu is the balance and the four weights close to 100",
              float(np.abs(df.composition_sum_wt_pct - 100.0).max()) < 1e-6,
              "max |sum-100| = %.2e" % float(np.abs(df.composition_sum_wt_pct - 100).max()))
    ok &= chk("N08", "R_actual reproduces R_target after 4-decimal rounding",
              float((df.R_CrNb_atomic_actual - df.R_CrNb_atomic_target).abs().max()) < 0.02,
              "max |R_actual - R_target| = %.4f"
              % float((df.R_CrNb_atomic_actual - df.R_CrNb_atomic_target).abs().max()))
    ok &= chk("N09", "no elemental descriptor, property or feature column is present",
              not [c for c in df.columns if c.lower().startswith(
                  ("magpie_", "delta_", "x_atomic", "x_vec", "x_melting", "x_density",
                   "x_electroneg")) or c in ("x_Cu", "x_Cr", "x_Nb", "x_X")],
              "columns: %d, all compositional/bookkeeping" % len(df.columns))
    ok &= chk("N10", "Ni token appears ONLY as the symbolic X identity",
              set(df.family) == {"NI"} and set(df.X_element) == {"NI"}
              and not [c for c in df.columns if C.NI_TOKEN.search(c.upper())],
              "family/X_element = NI; no Ni-named column")
    ok &= chk("N11", "few-shot split fixed at composition level, %d / %d"
              % (C.N_CALIBRATION, C.N_RECOVERY_EVAL),
              int((df.fewshot_role == "FEWSHOT_CALIBRATION").sum()) == C.N_CALIBRATION
              and int((df.fewshot_role == "RECOVERY_EVALUATION").sum()) == C.N_RECOVERY_EVAL,
              df.fewshot_role.value_counts().to_dict())
    ok &= chk("N12", "formal state count = compositions x 11 temperatures",
              C.NI_N_COMPOSITIONS * len(C.TEMPERATURES_C) == C.NI_FORMAL_STATES,
              C.NI_FORMAL_STATES)
    ok &= chk("N13", "no composition collides with an existing project composition id",
              not df.composition_id.str.startswith(("FV1", "F5F", "F5P", "PILOT")).any(),
              "prefix NI_BLIND_ is unused elsewhere")

    out_csv = os.path.join(C.D_NIPRE, "STEP5G_NI_BLIND_COMPOSITIONS.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    pd.DataFrame(CHECKS).to_csv(
        os.path.join(C.D_QA, "STEP5G_NI_COMPOSITION_QA.csv"), index=False,
        encoding="utf-8")

    doc = PREREG.format(
        ts=C.utcnow(), n=C.NI_N_COMPOSITIONS, nlast="%06d" % C.NI_N_COMPOSITIONS,
        temps=" ".join(str(t) for t in C.TEMPERATURES_C), ntemp=len(C.TEMPERATURES_C),
        states=C.NI_FORMAL_STATES, seed=seed, master=C.NI_MASTER_SEED, tries=tries,
        s0=C.S_CRNB_RANGE[0], s1=C.S_CRNB_RANGE[1],
        r0=C.R_CRNB_RANGE[0], r1=C.R_CRNB_RANGE[1],
        x0=C.NI_WT_RANGE[0], x1=C.NI_WT_RANGE[1], mcr=C.M_CR, mnb=C.M_NB,
        families=", ".join(C.FAMILIES_ORDER),
        p_id=p["model_id"], p_role=C.PRIMARY_ROLE, p_sha=p["checkpoint_sha256"],
        s_id=s["model_id"], s_role=C.SECONDARY_ROLE, s_sha=s["checkpoint_sha256"],
        ncal=C.N_CALIBRATION, nev=C.N_RECOVERY_EVAL,
        splitrule=C.FEWSHOT_SPLIT_RULE)
    md = os.path.join(C.D_NIPRE, "STEP5G_NI_BLIND_PREREGISTRATION.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(doc)

    C.write_json(os.path.join(C.D_NIPRE, "STEP5G_NI_BLIND_PREREGISTRATION.json"), {
        "preregistration_id": "STEP5G_NI_BLIND_V1",
        "generated_utc": C.utcnow(),
        "execution_order": G.BLIND_ORDER,
        "future_calphad": G.CALPHAD,
        "bootstrap": G.BOOTSTRAP,
        "primary_endpoint": "PRIMARY mean TVD on converged Ni states",
        "secondary_metrics": ["median TVD", "p95 TVD", "Macro MAE", "topology accuracy", "topology macro F1", "per-temperature mean TVD", "per-temperature topology accuracy"],
        "framing": "prospective zero-shot test on an unseen chemical identity",
        "not_framing": ["retrospective leave-one-element-out",
                        "post-blind few-shot", "in-domain validation"],
        "n_compositions": C.NI_N_COMPOSITIONS,
        "temperatures_C": C.TEMPERATURES_C,
        "formal_states": C.NI_FORMAL_STATES,
        "sampler": {"method": "Sobol(d=3, scramble=True)", "master_seed": C.NI_MASTER_SEED,
                    "stream_seed": seed, "points_drawn": tries,
                    "S_CrNb_range": list(C.S_CRNB_RANGE),
                    "R_CrNb_atomic_range": list(C.R_CRNB_RANGE),
                    "Ni_wt_pct_range": list(C.NI_WT_RANGE),
                    "inversion_masses": {"M_Cr": C.M_CR, "M_Nb": C.M_NB},
                    "atomic_fractions_computed": False,
                    "atomic_fractions_reason": (
                        "computing them needs the Ni atomic mass, which is a Ni "
                        "elemental property; featurization is deferred to STEP5H")},
        "target_definition": ("P_Ni^alpha = n_Ni^alpha / sum_beta n_Ni^beta with "
                              "n_Ni^alpha = phase_moles_alpha * X_Ni^alpha; "
                              "volume-fraction partition forbidden"),
        "family_order": C.FAMILIES_ORDER,
        "models": {"PRIMARY": {"id": p["model_id"], "role": C.PRIMARY_ROLE,
                               "checkpoint_sha256": p["checkpoint_sha256"],
                               "scaler_sha256": p["scaler_sha256"],
                               "fallback_row": p["fallback_row"]},
                   "SECONDARY": {"id": s["model_id"], "role": C.SECONDARY_ROLE,
                                 "checkpoint_sha256": s["checkpoint_sha256"],
                                 "scaler_sha256": s["scaler_sha256"],
                                 "fallback_row": s["fallback_row"]}},
        "pipeline_freeze_sha256": C.sha256_file(fz_path),
        "failure_policy": ("failed states are recorded as FAILED and reported; never "
                           "deleted, interpolated, temperature-substituted or dropped"),
        "ni_status": {"symbolic_composition_rows": C.NI_N_COMPOSITIONS,
                      "elemental_descriptor_access": 0, "property_access": 0,
                      "similarity_computation": 0, "prediction": 0,
                      "calphad_state": 0, "target_access": 0},
        "fewshot_split": {"salt": C.FEWSHOT_SPLIT_SALT,
                          "rule": C.FEWSHOT_SPLIT_RULE,
                          "n_calibration": C.N_CALIBRATION,
                          "n_recovery_evaluation": C.N_RECOVERY_EVAL,
                          "level": "composition"},
        "composition_table": out_csv,
        "composition_table_sha256": C.sha256_file(out_csv),
        "qa": CHECKS,
        "qa_verdict": "PASS" if ok else "FAIL"})

    print("\n  wrote %s (%d rows)" % (os.path.basename(out_csv), len(df)))
    print("STAGE 3: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
