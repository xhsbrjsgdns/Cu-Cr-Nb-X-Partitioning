# -*- coding: utf-8 -*-
"""
132_k_screen_pareto.py
STEP5K Part B Stage 3: applicability filter -> objectives -> non-dominated sorting
(global + per-family) -> deterministic maximin diversity selection (spec 23-28).

NO weighted score is computed anywhere. Objectives A/B/D drive the domination
ordering; C (moderate C15 band) and E (liquid avoidance) are hard constraints taken
from the PRE-FROZEN rules file; F is the hard OOD applicability filter. Family
fairness: per-family fronts are reported beside the global front, and the round-robin
finalist selection cannot be filled by one family (spec 27).

Note on the sorting algorithm. The first version of this stage compared every pair
inside every front, which is O(k^2) per front and does not finish at 10^5 candidates.
It is replaced here by an exact O(n log n) sweep: rows are deduplicated and sorted
lexicographically descending on all three maximization-oriented objectives, then a
Fenwick tree over the compressed second objective carries the running maximum of the
third. Because the sort guarantees every already-inserted row has objective 1 greater
or equal, and duplicates were removed, a row is dominated exactly when some inserted
row has objective 2 and objective 3 both greater or equal. The result is identical to
brute force - `verify_against_bruteforce()` checks that on a random sample every run,
and the stage aborts if the two ever disagree.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_common as C

MAX_FRONTS = 50          # ranks beyond this are labelled MAX_FRONTS+1, not silently 1
BRUTE_CHECK_N = 1500     # sample size for the every-run equivalence check
SCALE = (1.4, 4.5, 0.18)  # Domain-A spans of (S_CrNb, R_CrNb_atomic, X_wt_pct)


# ------------------------------------------------------- non-dominated sort - #
class _BIT:
    """Fenwick tree of prefix maxima (1-indexed)."""

    def __init__(self, n):
        self.n = n
        self.t = np.full(n + 1, -np.inf)

    def update(self, i, v):
        while i <= self.n:
            if v > self.t[i]:
                self.t[i] = v
            i += i & (-i)

    def query(self, i):
        m = -np.inf
        while i > 0:
            if self.t[i] > m:
                m = self.t[i]
            i -= i & (-i)
        return m


def _front_mask(V):
    """Exact non-dominated mask for maximization-oriented V (n x 3)."""
    n = len(V)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if V.shape[1] != 3:
        raise ValueError("this sweep is written for exactly three objectives")
    # deduplicate: identical objective vectors can never dominate one another
    uniq, inverse = np.unique(V, axis=0, return_inverse=True)
    m = len(uniq)
    order = np.lexsort((-uniq[:, 2], -uniq[:, 1], -uniq[:, 0]))
    # compress objective 1 so that a PREFIX of the BIT means "objective 1 >= x"
    o1 = uniq[:, 1]
    vals = np.unique(o1)
    rank = (len(vals) - np.searchsorted(vals, o1)).astype(int)   # 1 = largest
    bit = _BIT(len(vals))
    dominated = np.zeros(m, dtype=bool)
    # One at a time in full lexicographic-descending order. Every row already
    # inserted has objective 0 greater or equal; among those the BIT returns the
    # largest objective 2 seen with objective 1 greater or equal. Because exact
    # duplicates were removed above, "greater or equal on all three" is strict
    # domination, so the test below is exact - no separate tie-group pass is
    # needed, and rows tying on objective 0 are still ordered by (obj1, obj2)
    # descending, which is the order domination inside a tie group can only run.
    for idx in order:
        if bit.query(rank[idx]) >= uniq[idx, 2]:
            dominated[idx] = True
        bit.update(rank[idx], uniq[idx, 2])
    return ~dominated[inverse]


def nondominated_rank(F, maximize, max_fronts=MAX_FRONTS):
    """Front index (1 = best). Rows still unassigned after max_fronts get
    max_fronts + 1, which the outputs record as 'deeper than the computed depth'."""
    V = np.asarray(F, dtype=float) * np.where(np.asarray(maximize), 1.0, -1.0)[None, :]
    n = len(V)
    rank = np.full(n, max_fronts + 1, dtype=int)
    idx = np.arange(n)
    r = 1
    while len(idx) and r <= max_fronts:
        m = _front_mask(V[idx])
        rank[idx[m]] = r
        idx = idx[~m]
        r += 1
    return rank


def verify_against_bruteforce(F, maximize, n=BRUTE_CHECK_N, seed=20260913):
    """Front 1 by sweep must equal front 1 by exhaustive pairwise comparison."""
    rng = np.random.default_rng(seed)
    take = rng.permutation(len(F))[:min(n, len(F))]
    V = np.asarray(F, float)[take] * np.where(np.asarray(maximize), 1.0, -1.0)[None, :]
    fast = _front_mask(V)
    slow = np.ones(len(V), dtype=bool)
    for a in range(len(V)):
        ge = (V >= V[a]).all(axis=1)
        gt = (V > V[a]).any(axis=1)
        slow[a] = not bool((ge & gt).any())
    return bool((fast == slow).all()), int(fast.sum()), int(slow.sum()), int(len(V))


# ------------------------------------------------------- maximin diversity -- #
class Maximin:
    """Deterministic maximin bookkeeping (frozen rule: "within a family pick the
    Pareto member maximizing the minimum scaled distance to already-picked
    finalists; ties -> lexicographically smallest candidate_id").

    The running minimum distance to the selected set is kept incrementally, so a
    pick costs O(pool) and not O(pool^2). The first version recomputed the whole
    ordering to take one element, which is what made this stage intractable.
    """

    def __init__(self, ids, coords):
        order = np.argsort(np.asarray(ids, dtype=object))   # lexicographic ties
        self.ids = [ids[i] for i in order]
        self.P = np.asarray(coords, float)[order]
        self.pos = {c: i for i, c in enumerate(self.ids)}
        self.d = np.full(len(self.ids), np.inf)             # min dist to selected
        self.taken = np.zeros(len(self.ids), dtype=bool)

    def pick(self, pool_ids):
        """Best remaining member of pool_ids, or None if the pool is exhausted."""
        cand = np.array([self.pos[c] for c in pool_ids if not self.taken[self.pos[c]]],
                        dtype=int)
        if not len(cand):
            return None
        cand.sort()                       # ids are stored lexicographically sorted,
        best = cand[np.argmax(self.d[cand])]   # so argmax takes the lex-smallest tie
        self.taken[best] = True
        self.d = np.minimum(self.d, np.linalg.norm(self.P - self.P[best], axis=1))
        self.d[best] = -np.inf
        return self.ids[best]


# --------------------------------------------------------------------- main - #
def pivot_objectives(space):
    """One row per candidate, with the temperature-resolved quantities the frozen
    objectives need. Every lookup is an explicit reindex on candidate_id so nothing
    depends on two pandas operations happening to sort the same way."""
    ids = space[["candidate_id", "X_identity", "ood_knn5_distance"]].drop_duplicates(
        "candidate_id").sort_values("candidate_id").reset_index(drop=True)
    key = ids.candidate_id

    def at(col, temp):
        s = space.loc[space.Temperature_C == float(temp), ["candidate_id", col]]
        if len(s) != len(key):
            raise SystemExit("FATAL: %s at %s degC has %d rows, expected %d."
                             % (col, temp, len(s), len(key)))
        return s.set_index("candidate_id")[col].reindex(key).to_numpy(float)

    obj = ids.copy()
    obj["Cr_FCC_950"] = at("Cr_P_FCC", 950)
    obj["Cr_FCC_450"] = at("Cr_P_FCC", 450)
    obj["Delta_Cr_FCC"] = obj.Cr_FCC_950 - obj.Cr_FCC_450
    obj["C15_450"] = at("frac_C15", 450)
    obj["C15_600"] = at("frac_C15", 600)
    obj["Cr_OTHER_450"] = at("Cr_P_OTHER_SOLID", 450)
    liq = (space.groupby("candidate_id")["frac_LIQUID"].max().reindex(key)
           .to_numpy(float))
    obj["LIQUID_max"] = liq
    if obj.isna().any().any():
        raise SystemExit("FATAL: NaN in the pivoted objective table.")
    return obj


def main():
    for d in C.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print("STEP5K Part B Stage 3: applicability -> Pareto -> finalists")
    print("=" * 78)

    rules = json.load(open(os.path.join(C.D_SCREEN,
                                        "STEP5K_SCREENING_RULES_FREEZE.json"),
                           encoding="utf-8"))
    if not rules.get("frozen_before_any_screening_prediction"):
        raise SystemExit("FATAL: the screening rules were not frozen before "
                         "prediction; ranking against them would be post-hoc.")
    band = rules["moderate_c15_band"]
    space = pd.read_parquet(os.path.join(C.D_SCREEN,
                                         "STEP5K_VIRTUAL_SCREENING_SPACE.parquet"))
    comps = pd.read_csv(os.path.join(C.D_SCREEN,
                                     "STEP5K_VIRTUAL_CANDIDATE_COMPOSITIONS.csv"))
    obj = pivot_objectives(space).merge(
        comps[["candidate_id", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct", "Cu_wt_pct",
               "S_CrNb", "R_CrNb_atomic"]], on="candidate_id", validate="one_to_one")

    c_pass = (obj.C15_450.between(band["band_low"], band["band_high"])
              & obj.C15_600.between(band["band_low"], band["band_high"]))
    e_pass = obj.LIQUID_max <= C.LIQUID_TOLERANCE
    f_pass = obj.ood_knn5_distance <= C.OOD_MAX_KNN_DIST
    applicable = c_pass & e_pass & f_pass
    obj["constraint_C_pass"] = c_pass.astype(int)
    obj["constraint_E_pass"] = e_pass.astype(int)
    obj["filter_F_pass"] = f_pass.astype(int)
    obj["applicable"] = applicable.astype(int)
    print("  applicability: C-band %d | E-liquid %d | F-OOD %d | ALL %d of %d"
          % (int(c_pass.sum()), int(e_pass.sum()), int(f_pass.sum()),
             int(applicable.sum()), len(obj)))
    per_fam_app = obj[obj.applicable == 1].X_identity.value_counts().to_dict()
    print("  applicable per family: %s" % {f: int(per_fam_app.get(f, 0)) for f in C.SIX})
    if int(applicable.sum()) == 0:
        raise SystemExit("FATAL: no candidate satisfies the pre-frozen constraints. "
                         "The constraints are frozen and must not be relaxed here; "
                         "report the empty set instead.")

    app = obj[obj.applicable == 1].copy().reset_index(drop=True)
    MAXIMIZE = [True, True, False]          # A up, B up, D down
    F = app[["Cr_FCC_950", "Delta_Cr_FCC", "Cr_OTHER_450"]].to_numpy(float)

    same, nf, ns, nchk = verify_against_bruteforce(F, MAXIMIZE)
    print("  sweep vs brute force on %d sampled candidates: %s (front %d vs %d)"
          % (nchk, "identical" if same else "DISAGREE", nf, ns))
    if not same:
        raise SystemExit("FATAL: the fast non-dominated sweep disagrees with brute "
                         "force. Refusing to rank candidates with an unverified "
                         "algorithm.")

    app["pareto_rank"] = nondominated_rank(F, MAXIMIZE)
    fam_rank = np.zeros(len(app), dtype=int)
    for fam in C.SIX:
        m = (app.X_identity == fam).to_numpy()
        if m.sum():
            fam_rank[m] = nondominated_rank(F[m], MAXIMIZE)
    app["family_pareto_rank"] = fam_rank
    front = app[app.pareto_rank == 1]
    fam_front_counts = {f: int(((app.X_identity == f)
                                & (app.family_pareto_rank == 1)).sum()) for f in C.SIX}
    print("  global Pareto front: %d | per-family fronts: %s"
          % (len(front), fam_front_counts))

    # ---- deterministic maximin finalists, round robin over families ---------- #
    scaled = np.column_stack([app.S_CrNb / SCALE[0], app.R_CrNb_atomic / SCALE[1],
                              app.X_wt_pct / SCALE[2]])
    front_ids = front.candidate_id.tolist()
    idx_of = {cid: i for i, cid in enumerate(app.candidate_id)}
    # Finalist pool per the amended diversity rule: each family's OWN Pareto
    # front; a family front with fewer than `lo` members is extended with that
    # family's next family_pareto_rank values so the preregistered 3-per-family
    # minimum is reachable (the global front alone can be dominated by one
    # family - here it is ZR-only - which would violate family fairness).
    ext_ids, ext_added = [], []
    lo, hi = C.FINALISTS_PER_FAMILY
    for fam in C.SIX:
        fm = app.loc[app.X_identity == fam].sort_values(
            ["family_pareto_rank", "candidate_id"])
        got = fm.loc[fm.family_pareto_rank == 1, "candidate_id"].tolist()
        if len(got) < lo:
            extra = fm.loc[fm.family_pareto_rank > 1, "candidate_id"].tolist()
            ext_added.extend(extra[:lo - len(got)])
            got = got + extra[:lo - len(got)]
        ext_ids.extend(got)
    pool_ids = sorted(set(front_ids) | set(ext_ids))
    mm = Maximin(pool_ids, scaled[[idx_of[c] for c in pool_ids]])
    fam_pool = {f: app.loc[(app.X_identity == f)
                           & (app.candidate_id.isin(pool_ids)),
                           "candidate_id"].tolist()
                for f in C.SIX}
    selected, order_no, per_family = [], [], {f: 0 for f in C.SIX}
    for _rnd in range(hi):
        if len(selected) >= C.MAX_FINALISTS_TOTAL:
            break
        for fam in C.SIX:
            if len(selected) >= C.MAX_FINALISTS_TOTAL:
                break
            if per_family[fam] >= hi:
                continue
            pick = mm.pick(fam_pool[fam])
            if pick is None:
                continue
            selected.append(pick)
            order_no.append(len(selected))
            per_family[fam] += 1
    shortfall = {f: lo - per_family[f] for f in C.SIX if per_family[f] < lo}
    app["selected_finalist"] = app.candidate_id.isin(selected).astype(int)
    app["diversity_rank"] = (app.candidate_id.map(dict(zip(selected, order_no)))
                             .fillna(0).astype(int))
    print("  finalist pool: %d (family fronts + rank extensions for %s)"
          % (len(pool_ids), [f for f, n in
                             {f: int((app.X_identity == f).sum()
                                     and app.loc[(app.X_identity == f)
                                                 & (app.family_pareto_rank == 1)].shape[0])
                              for f in C.SIX}.items() if n < lo] or "none"))
    print("  finalists: %d (round-robin maximin; per family %s)%s"
          % (len(selected), per_family,
             "  SHORTFALL %s" % shortfall if shortfall else ""))

    keep = ["candidate_id", "X_identity", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
            "Cu_wt_pct", "S_CrNb", "R_CrNb_atomic",
            "Cr_FCC_950", "Cr_FCC_450", "Delta_Cr_FCC", "C15_450", "C15_600",
            "Cr_OTHER_450", "LIQUID_max", "ood_knn5_distance",
            "constraint_C_pass", "constraint_E_pass", "filter_F_pass", "applicable",
            "pareto_rank", "family_pareto_rank", "diversity_rank", "selected_finalist"]
    app[keep].to_csv(os.path.join(C.D_SCREEN, "STEP5K_VIRTUAL_SCREENING_PARETO.csv"),
                     index=False, encoding="utf-8")

    sel = app[app.selected_finalist == 1].sort_values("diversity_rank").copy()
    sel = sel.rename(columns={"Cr_FCC_950": "pred_Cr_FCC_950",
                              "Cr_FCC_450": "pred_Cr_FCC_450",
                              "Delta_Cr_FCC": "pred_Delta_Cr_FCC",
                              "C15_450": "pred_C15_450", "C15_600": "pred_C15_600",
                              "Cr_OTHER_450": "pred_Cr_OTHER",
                              "ood_knn5_distance": "OOD_distance"})
    sel["CALPHAD_confirmed"] = 0        # set by the confirmation stage
    cand_cols = ["candidate_id", "X_identity", "Cr_wt_pct", "Nb_wt_pct", "X_wt_pct",
                 "Cu_wt_pct", "S_CrNb", "R_CrNb_atomic", "pred_Cr_FCC_950",
                 "pred_Cr_FCC_450", "pred_Delta_Cr_FCC", "pred_C15_450",
                 "pred_C15_600", "pred_Cr_OTHER", "OOD_distance", "pareto_rank",
                 "family_pareto_rank", "diversity_rank", "selected_finalist",
                 "CALPHAD_confirmed"]
    sel[cand_cols].to_csv(os.path.join(C.D_SCREEN,
                                       "STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv"),
                          index=False, encoding="utf-8")

    C.write_json(os.path.join(C.D_SCREEN, "STEP5K_PARETO_FREEZE.json"), {
        "generated_utc": C.utcnow(),
        "rules_sha256": C.sha256_file(os.path.join(
            C.D_SCREEN, "STEP5K_SCREENING_RULES_FREEZE.json")),
        "rules_frozen_before_prediction": True,
        "n_candidates": int(len(obj)), "n_applicable": int(len(app)),
        "applicable_per_family": {f: int(per_fam_app.get(f, 0)) for f in C.SIX},
        "constraint_counts": {"C_band": int(c_pass.sum()),
                              "E_liquid": int(e_pass.sum()),
                              "F_ood": int(f_pass.sum())},
        "objectives": {"A": "maximize Cr_FCC_950", "B": "maximize Delta_Cr_FCC",
                       "D": "minimize Cr_OTHER_450"},
        "sorting_algorithm": "exact O(n log n) lexicographic sweep with a Fenwick "
                             "prefix-maximum tree; verified against brute force on "
                             "%d sampled candidates every run" % nchk,
        "bruteforce_check": {"n_sampled": nchk, "identical": same,
                             "front_size_sweep": nf, "front_size_bruteforce": ns},
        "max_fronts_computed": MAX_FRONTS,
        "rank_beyond_max_fronts_is_labelled": MAX_FRONTS + 1,
        "global_front": int(len(front)),
        "family_front": fam_front_counts,
        "finalists": int(len(selected)), "per_family_finalists": per_family,
        "per_family_shortfall_vs_minimum_%d" % lo: shortfall,
        "diversity_method": rules["diversity_selection"]["method"],
        "no_weighted_score_used": True,
        "pareto_file_sha256": C.sha256_file(os.path.join(
            C.D_SCREEN, "STEP5K_VIRTUAL_SCREENING_PARETO.csv")),
        "candidates_file_sha256": C.sha256_file(os.path.join(
            C.D_SCREEN, "STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv"))})
    print("STAGE B3: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
