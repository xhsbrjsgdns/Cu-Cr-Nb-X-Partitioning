# -*- coding: utf-8 -*-
"""
112_i_controls.py
STEP5I Stage 2: M_ZERO, M_TRIVIAL and the R0 / R1 / R2 anchor-only controls
(spec 20-25).

The controls exist to answer one question (spec 25): does refitting the network learn
transferable Ni chemistry, or does it merely reproduce what a handful of Ni anchor
partition vectors already say? R0 is the sharpest form of that question - it discards
the model completely and predicts the anchor mean at each temperature. If M_REFIT does
not beat R0, the refit has demonstrated no value beyond the anchors under this
protocol, and the report is required to say so plainly.

The three estimators are NOT re-implemented here. `g_common.recovery_apply` is imported
from the frozen STEP5G code snapshot and executed, so what runs is the preregistered
definition rather than a fresh transcription of it.

  M_ZERO    frozen STEP5H-A PRIMARY predictions, restricted to the recovery subset.
            Never regenerated (spec 20). This is the number B_recovery25 is measured
            against - not the 22,000-state 0.8643856187, which is a different subset
            (spec 34).
  M_TRIVIAL frozen PRIMARY training-mean simplex on the same rows (spec 21).
  R0/R1/R2  computed per (replicate, budget) and per maximin budget, from the frozen
            anchor true targets and the frozen PRIMARY RAW zero-shot outputs.

Every method is scored on the identical 15,400 states (spec 31).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i_common as C


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5I  Stage 2: M_ZERO / M_TRIVIAL baselines and R0 / R1 / R2 controls")
    print("=" * 78)

    gc = C.load_g_common()
    ni, feats = C.load_frozen_ni()
    fallback = C.frozen_fallback_row()

    ev = ni[ni.fewshot_role == "RECOVERY_EVALUATION"].reset_index(drop=True)
    if len(ev) != C.N_RECOVERY_STATES:
        raise SystemExit("FATAL: recovery evaluation has %d states, expected %d."
                         % (len(ev), C.N_RECOVERY_STATES))
    y_ev = ev[C.TRUE_COLS].to_numpy(float)
    raw_ev = ev[C.RAW_COLS].to_numpy(float)
    pred_ev = ev[C.PRED_COLS].to_numpy(float)
    temps_ev = ev.Temperature_C.to_numpy()

    # ---------------------------------------------------------- M_ZERO ----- #
    m_zero = C.metrics_block(y_ev, pred_ev, fallback)
    m_zero_tvd = m_zero["mean_tvd"]
    m_trivial = C.metrics_block(
        y_ev, np.repeat(fallback[None, :], len(y_ev), axis=0), fallback)
    print("  M_ZERO    (frozen PRIMARY zero-shot, recovery subset): mean TVD %.10f | "
          "topology %.6f" % (m_zero_tvd, m_zero["topology_accuracy"]))
    print("  M_TRIVIAL (frozen PRIMARY train-mean simplex, same rows): mean TVD %.10f"
          % m_trivial["mean_tvd"])
    print("  full 22,000-state zero-shot for context only: %.10f  (NOT used for "
          "B_recovery25)" % C.ZERO_SHOT_FULL_MEAN_TVD)

    baselines = {
        "generated_utc": C.utcnow(),
        "subset": "RECOVERY_EVALUATION, %d compositions x 11 T = %d states"
                  % (C.N_RECOVERY_EVAL, C.N_RECOVERY_STATES),
        "M_ZERO": m_zero,
        "M_ZERO_source": "frozen STEP5H_A_PRIMARY_NI_PREDICTIONS.parquet, "
                         "restricted to the recovery subset; never regenerated",
        "M_TRIVIAL": m_trivial,
        "M_TRIVIAL_definition": "frozen PRIMARY training-mean simplex %s"
                                % fallback.tolist(),
        "full_blind_reference_do_not_substitute": {
            "mean_tvd_22000_states": C.ZERO_SHOT_FULL_MEAN_TVD,
            "topology_accuracy_22000_states": C.ZERO_SHOT_FULL_TOPOLOGY_ACC,
            "trivial_tvd_22000_states": C.ZERO_SHOT_FULL_TRIVIAL_TVD,
            "note": "spec 34: B_recovery25 uses the recovery-subset M_ZERO above, "
                    "not this value"},
    }
    C.write_json(os.path.join(C.D_METRICS, "STEP5I_RECOVERY_BASELINES.json"), baselines)

    # per-state baseline TVDs, reused by every later stage so no method is ever
    # scored on a different row set
    base = ev[["state_id", "composition_id", "Temperature_C", "true_topology",
               "nbni3_present", "t_regime"]].copy()
    base["tvd_M_ZERO"] = C.per_state_tvd(y_ev, pred_ev)
    base["tvd_M_TRIVIAL"] = C.per_state_tvd(
        y_ev, np.repeat(fallback[None, :], len(y_ev), axis=0))
    base["topo_correct_M_ZERO"] = (y_ev.argmax(1) == pred_ev.argmax(1)).astype(int)
    base["topo_correct_M_TRIVIAL"] = (
        y_ev.argmax(1) == np.repeat(fallback[None, :], len(y_ev), axis=0).argmax(1)
    ).astype(int)
    base.to_parquet(os.path.join(C.D_METRICS, "STEP5I_RECOVERY_BASELINE_PER_STATE.parquet"),
                    index=False)

    # ---------------------------------------------------- R0 / R1 / R2 ----- #
    seqs = C.random_sequences()
    mx = C.maximin_sequence()
    cells = [("RANDOM", r, b, seqs[r]) for r in C.REPLICATES for b in C.BUDGETS]
    cells += [("MAXIMIN", 0, b, mx) for b in C.BUDGETS]

    rows, per_state = [], []
    for kind, rep, budget, seq in cells:
        tag = C.cell_tag(kind, rep, budget)
        anc_ids = C.anchors(seq, budget)
        anc = ni[ni.composition_id.isin(anc_ids)]
        assert len(anc) == budget * len(C.TEMPERATURES_C)
        assert not (set(anc_ids) & set(ev.composition_id)), \
            "anchor composition is also a recovery-evaluation composition"
        a_true = anc[C.TRUE_COLS].to_numpy(float)
        a_raw = anc[C.RAW_COLS].to_numpy(float)
        a_temps = anc.Temperature_C.to_numpy()
        eff = anc.groupby("Temperature_C").size().to_dict()

        cell = base[["state_id"]].copy()
        for short in ("R0", "R1", "R2"):
            name = C.CONTROL_IDS[short]
            corrected = gc.recovery_apply(name, raw_ev, temps_ev, a_raw, a_true,
                                          a_temps, fallback)
            m = C.metrics_block(y_ev, corrected, fallback, zero_shot_tvd=m_zero_tvd)
            rows.append(dict(kind=kind, replicate=rep, budget=budget, tag=tag,
                             method=short, estimator=name,
                             n_anchor_compositions=budget,
                             n_anchor_states=int(len(anc)), **m))
            cell["tvd_%s" % short] = C.per_state_tvd(y_ev, corrected)
            cell["topo_correct_%s" % short] = (
                y_ev.argmax(1) == corrected.argmax(1)).astype(int)
        cell.insert(0, "tag", tag)
        per_state.append(cell)
        print("  %-16s R0 %.6f | R1 %.6f | R2 %.6f   (anchor states per T: %s)"
              % (tag,
                 rows[-3]["mean_tvd"], rows[-2]["mean_tvd"], rows[-1]["mean_tvd"],
                 sorted(set(eff.values()))))

    ctrl = pd.DataFrame(rows)
    ctrl.to_csv(os.path.join(C.D_CONTROL, "STEP5I_CONTROL_METRICS.csv"), index=False,
                encoding="utf-8")
    pd.concat(per_state, ignore_index=True).to_parquet(
        os.path.join(C.D_CONTROL, "STEP5I_CONTROL_PER_STATE.parquet"), index=False)

    C.write_json(os.path.join(C.D_CONTROL, "STEP5I_CONTROL_DEFINITIONS.json"), {
        "generated_utc": C.utcnow(),
        "source": "executed from the frozen STEP5G code snapshot g_common.py, not "
                  "re-implemented",
        "g_common_sha256": C.sha256_file(C.G_COMMON),
        "definitions": {k: gc.RECOVERY_ESTIMATORS[v]
                        for k, v in C.CONTROL_IDS.items()},
        "anchor_raw_source": "frozen STEP5H-A PRIMARY raw (unprojected) outputs",
        "anchor_true_source": "frozen STEP5H-B Ni targets",
        "evaluation_rows": C.N_RECOVERY_STATES,
        "all_methods_same_rows": True})
    print("\nSTAGE 2: PASS - %d control cells scored on the identical %d states."
          % (len(ctrl), C.N_RECOVERY_STATES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
