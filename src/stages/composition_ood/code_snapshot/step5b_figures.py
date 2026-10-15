"""Frozen STEP5B figure contract; Python/matplotlib quantitative grids.

Purpose: describe IID/OOD predictive performance and covariate-domain coverage,
without presupposing improvement, degradation, or a causal shift explanation.
Evidence: 01 paired primary metrics; 02 phase-fraction parity; 03 elemental
partition parity; 04/05 subgroup errors; 06 frozen-feature shifts; 07 domain
distances; 08 composition-aggregated distance/error association; 09 composition
coverage; 10 liquid presence and fraction diagnostics. Each panel has one role.

All task-eligible converged supplied rows are retained. X partition is undefined
for BASELINE alloys: those rows are explicitly excluded from X scoring and X
parity, with NaN retained in aligned row errors and composition summaries.
No winsorization, quantile clipping,
target-dependent filtering, model fitting, or filesystem data reads occur here.
Finite masks exclude only unavailable coordinates and their counts are reported.
Scatter marks are rasterized in PDF; axes/text remain editable. Outputs are white
background 300-dpi PNG and PDF, with a plain-text caption/data-contract manifest.
Continuous-error summaries are TVD = 0.5*sum(abs(predicted-true)) over five phases;
presence is five-label macro F1 (zero denominator -> 0). Subgroup summaries use
all rows, whereas Fig 08 averages row errors within composition before plotting.
No confidence bands are drawn: rows at different temperatures are not independent
composition replicates. Counts refer to supplied evaluation rows unless indicated.
Optional counts maps formal/converged/failed to row counts; failed equilibrium
rows are context only and cannot enter target-scored panels.
"""

from pathlib import Path
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASKS = ("presence", "phase_fraction", "CR_P", "NB_P", "X_P")
PHASES = ("FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID")
ALLOY_FAMILIES = ("BASELINE", "ZR", "TI", "CO", "SI", "AG", "MO")
LABELS = ("Phase presence", "Phase fraction", "Cr partition", "Nb partition", "X partition")
COLORS = {"Train": "#7F8B94", "IID": "#477A9D", "B": "#C27D48"}


def _phases(df):
    phases = [c.removesuffix("_volfrac_true") for c in df if c.endswith("_volfrac_true")]
    if set(phases) != set(PHASES):
        raise ValueError(f"Expected exactly five phase-fraction targets; found {phases}")
    return list(PHASES)


def _pair(df, phases, task):
    if task == "presence":
        a, b = [f"{p}_presence_true" for p in phases], [f"{p}_presence_pred" for p in phases]
    elif task == "phase_fraction":
        a, b = [f"{p}_volfrac_true" for p in phases], [f"{p}_frac_proj" for p in phases]
    else:
        a, b = [f"{task}_{p}_true" for p in phases], [f"{p}_{task}_proj" for p in phases]
    return df[a].to_numpy(float), df[b].to_numpy(float)


def _score(df, phases, task):
    if task == 'X_P':
        df = df.loc[df.family != 'BASELINE']
    if len(df) == 0:
        return float('nan')
    a, b = _pair(df, phases, task)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Non-finite scored predictions/targets in {task}; resolve upstream")
    if task == "presence":
        tp = ((a == 1) & (b == 1)).sum(0)
        den = (a == 1).sum(0) + (b == 1).sum(0)
        return float(np.divide(2 * tp, den, out=np.zeros(5), where=den != 0).mean())
    return float(np.abs(b - a).sum(1).mean() / 2)


def _row_errors(df, phases):
    result = df[["composition_id", "family", "Temperature_C"]].copy()
    for task in TASKS:
        a, b = _pair(df, phases, task)
        result[task] = (a != b).mean(1) if task == "presence" else np.abs(b - a).sum(1) / 2
        if task == 'X_P':
            result.loc[df.family == 'BASELINE', task] = np.nan
    if "nn1_distance" in df:
        result["nn1_distance"] = df["nn1_distance"].to_numpy()
    return result


def _cloud(ax, x, y, color=None, parity=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[valid], y[valid], s=9 if valid.sum() < 2000 else 3,
               alpha=.6 if valid.sum() < 2000 else .24, color=color or COLORS["B"],
               linewidths=0, rasterized=True)
    if parity and valid.any():
        lo = min(0, x[valid].min(), y[valid].min())
        hi = max(1, x[valid].max(), y[valid].max())
        ax.plot([lo, hi], [lo, hi], color="#555555", lw=.8, ls="--")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
    ax.text(.03, .97, f"n = {valid.sum():,}" + (f"; missing = {(~valid).sum():,}" if (~valid).any() else ""),
            transform=ax.transAxes, va="top", fontsize=7)


def make_figures(ood_df, iid_df, train_comp, iid_comp, b_comp, metrics_df,
                 comparison_df, shift_df, strata_dict, outdir, counts=None):
    """Export ten figure pairs and captions; consumes passed data only.

    metrics_df, comparison_df and strata_dict are accepted for pipeline API
    consistency; primary/subgroup figures recompute fixed metrics from supplied
    predictions so plot definitions remain unambiguous. No data are modified.
    Distance must be in the frozen training-standardized feature space.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    phases = _phases(ood_df)
    errors = _row_errors(ood_df, phases)
    for df in (ood_df, iid_df):
        for task in TASKS:
            _score(df, phases, task)
    captions, paths = [], []
    context = (f"B formal = {counts.get('formal', 'NA'):,}; converged = {counts.get('converged', len(ood_df)):,}; "
               f"failed = {counts.get('failed', 'NA'):,}" if counts and all(k in counts for k in ('formal','failed'))
               else f"B scored converged rows = {len(ood_df):,}; compositions = {ood_df.composition_id.nunique():,}")

    def save(fig, number, slug, title, caption):
        fig.get_layout_engine().set(rect=(0, .065, 1, .935))
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.text(.5, .008, context, ha="center", va="bottom", fontsize=7, color="#555555")
        for i, ax in enumerate(fig.axes):
            ax.text(-.10, 1.05, string.ascii_lowercase[i], transform=ax.transAxes,
                    fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=7)
        stem = f"Fig_S5B_{number:02d}_{slug}"
        for ext in ("png", "pdf"):
            path = outdir / f"{stem}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            paths.append(str(path))
        plt.close(fig)
        captions.append(f"{stem}\n{caption}\n{context}\n")

    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
             "axes.linewidth": .7, "legend.frameon": False, "pdf.fonttype": 42,
             "svg.fonttype": "none", "figure.constrained_layout.use": True}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 5, figsize=(12.8, 3.2))
        for ax, task, label in zip(axes, TASKS, LABELS):
            vals = [_score(df, phases, task) for df in (iid_df, ood_df)]
            ax.bar([0, 1], vals, color=[COLORS['IID'], COLORS['B']], width=.6)
            ax.set(xticks=[0, 1], xticklabels=["IID", "B OOD"], title=label,
                   ylabel="Macro F1 (higher is better)" if task == "presence" else "Mean TVD (lower is better)")
            ax.set_ylim(0, max(1 if task == "presence" else .01, max(vals) * 1.2))
            for j, v in enumerate(vals):
                ax.text(j, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        save(fig, 1, "IID_vs_OOD", "IID and B OOD performance", "Five fixed primary tasks, evaluated over all supplied task-eligible converged rows. X partition excludes BASELINE, where it is undefined. Presence: macro F1 across five labels, zero denominator set to zero. Regressions: mean five-component TVD. Bar baselines are zero; no uncertainty intervals implied.")

        fig, axes = plt.subplots(1, 5, figsize=(12.8, 3.4))
        a, b = _pair(ood_df, phases, "phase_fraction")
        for j, (ax, p) in enumerate(zip(axes, phases)):
            _cloud(ax, a[:, j], b[:, j], parity=True)
            ax.set(title=p, xlabel="True fraction", ylabel="Projected fraction")
        save(fig, 2, "OOD_phase_fraction_parity", "B OOD phase-fraction parity", "All converged rows; rasterized scatter, no point trimming. Dashed lines mark identity; each axis includes zero, one, and any out-of-range predictions.")

        fig, axes = plt.subplots(3, 5, figsize=(12.8, 8.4))
        for row, task in enumerate(TASKS[2:]):
            eligible = ood_df.loc[ood_df.family != 'BASELINE'] if task == 'X_P' else ood_df
            a, b = _pair(eligible, phases, task)
            for j, p in enumerate(phases):
                ax = axes[row, j]
                _cloud(ax, a[:, j], b[:, j], parity=True)
                ax.set(title=f"{task}: {p}", xlabel="True partition", ylabel="Projected partition")
        save(fig, 3, "OOD_partition_parity", "B OOD elemental-partition parity", "Rows show Cr, Nb, and X partition tasks; columns show the five phase families. X panels exclude BASELINE because X partition is undefined; n denotes eligible rows. All other supplied converged rows and zero-valued components remain visible; identity lines are references.")

        for number, group, slug in [(4, "family", "OOD_error_by_family"), (5, "Temperature_C", "OOD_error_by_temperature")]:
            fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.8))
            groups = list(ood_df.groupby(group, sort=True, observed=True))
            if group == 'family':
                groups.sort(key=lambda item: (ALLOY_FAMILIES.index(item[0]) if item[0] in ALLOY_FAMILIES else len(ALLOY_FAMILIES), str(item[0])))
            for ax, task, label in zip(axes, TASKS, LABELS):
                values = [1 - _score(g, phases, task) if task == "presence" else _score(g, phases, task) for _, g in groups]
                positions = np.arange(len(groups))
                if group == "Temperature_C":
                    positions = np.asarray([k for k, _ in groups], float)
                    ax.plot(positions, values, "o-", ms=3, color=COLORS['B'], lw=1)
                    ax.set_xlabel("Temperature (°C)")
                else:
                    ax.bar(positions, values, color=COLORS['B'])
                    ax.set_xticks(positions, [f"{k}\n(n={len(g):,})" for k, g in groups], rotation=45, ha="right")
                ax.set(title=label, ylabel="1 − macro F1" if task == "presence" else "Mean TVD")
                if group == 'family' and task == 'X_P':
                    for position, value in zip(positions, values):
                        if not np.isfinite(value):
                            ax.text(position, 0, 'N/A', ha='center', va='bottom', fontsize=7)
                ax.set_ylim(bottom=0)
            save(fig, number, slug, f"B OOD error by {'alloy family' if group == 'family' else 'temperature'}", "All task-eligible converged rows grouped by " + group + "; lower is better. X partition excludes BASELINE and its family bar is N/A; displayed family n is total converged rows. Presence is one minus group macro F1; other tasks use mean TVD. Temperature lines only connect evaluated levels; no interpolation or independent-row confidence intervals are asserted.")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
        shifts = shift_df.copy()
        shifts['_rank'] = shifts.standardized_mean_shift.abs()
        shifts = shifts.sort_values('_rank', ascending=False, na_position='last').head(15).iloc[::-1]
        names = shifts.feature.astype(str).tolist()
        axes[0].barh(names, shifts.standardized_mean_shift.replace([np.inf, -np.inf], np.nan), color=COLORS['B'])
        for position, value in enumerate(shifts.standardized_mean_shift):
            if not np.isfinite(value):
                axes[0].text(0, position, 'undefined (train SD = 0 or missing)', fontsize=6, va='center')
        axes[0].axvline(0, color="#555555", lw=.7)
        axes[0].set_xlabel("(B mean − train mean) / train SD")
        axes[1].barh(names, shifts.fraction_outside_train_range, color=COLORS['IID'])
        axes[1].set(xlim=(0, 1), xlabel="B fraction outside training range")
        for ax in axes:
            ax.tick_params(axis='y', labelsize=7)
        save(fig, 6, "feature_shift", "Frozen-feature shift: train to B", "Up to 15 features ranked by absolute standardized mean shift, using the supplied frozen-feature audit. Signed mean shifts and training-range exceedance are descriptive, not causal. Nonfinite standardized shifts (for example zero training variance) have no finite bar; range exceedance remains reported. Feature ranking limits display only, never scored samples.")

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        for ax, feature, label in zip(axes, ['nn1_distance', 'nn5_mean_distance'], ['Nearest-training distance', 'Mean five-neighbor distance']):
            any_data = False
            for frame, name in [(iid_comp, 'IID'), (b_comp, 'B')]:
                if feature in frame:
                    v = frame[feature].to_numpy(float)
                    v = np.sort(v[np.isfinite(v)])
                    if len(v):
                        ax.step(v, np.arange(1, len(v) + 1) / len(v), where='post', label=f"{name} (n={len(v):,})", color=COLORS[name])
                        any_data = True
            ax.set(xlabel=label, ylabel="Composition ECDF", ylim=(0, 1.02))
            if any_data:
                ax.legend(fontsize=7)
            else:
                ax.text(.5, .5, "Distance unavailable", transform=ax.transAxes, ha='center')
        save(fig, 7, "OOD_distance_distribution", "Distance to frozen training domain", "Empirical cumulative distributions count each composition once. Distances are supplied by the upstream frozen training-standardized feature audit. No log transform or tail truncation; panels explicitly mark unavailable fields.")

        if 'nn1_distance' not in errors:
            errors = errors.merge(b_comp[['composition_id', 'nn1_distance']], on='composition_id', how='left', validate='many_to_one')
        by_comp = errors.groupby('composition_id', observed=True)[list(TASKS) + ['nn1_distance']].mean()
        fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.4))
        for ax, task, label in zip(axes, TASKS, LABELS):
            eligible_comp = by_comp.loc[by_comp[task].notna()] if task == 'X_P' else by_comp
            _cloud(ax, eligible_comp.nn1_distance, eligible_comp[task])
            ax.set(title=label, xlabel="Nearest-training distance", ylabel="Mean label error" if task == 'presence' else "Composition mean TVD")
            ax.set_ylim(bottom=0)
        save(fig, 8, "error_vs_OOD_distance", "Composition error versus domain distance", "Each dot is one task-eligible B composition, averaging all available converged temperature rows. X partition excludes BASELINE compositions because X is undefined. Presence uses mean binary-label disagreement (not one minus macro F1); regressions use mean TVD. Association is descriptive. No fitted trend, point filtering beyond task eligibility, or causal inference.")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        frames = [(train_comp, 'Train'), (iid_comp, 'IID'), (b_comp, 'B')]
        finite_s = np.concatenate([d.S_CrNb_wt_pct.to_numpy(float) for d, _ in frames])
        finite_r = np.concatenate([d.R_CrNb_atomic_actual.to_numpy(float) for d, _ in frames])
        def limits(v):
            v = v[np.isfinite(v)]
            if not len(v):
                return (0, 1)
            pad = max(float(np.ptp(v)) * .04, .01)
            return v.min() - pad, v.max() + pad
        for ax, (df, name) in zip(axes, frames):
            _cloud(ax, df.S_CrNb_wt_pct, df.R_CrNb_atomic_actual, color=COLORS[name])
            ax.set(title=name, xlabel="Cr + Nb (wt.%)", ylabel="Actual atomic Cr/Nb ratio",
                   xlim=limits(finite_s), ylim=limits(finite_r))
        save(fig, 9, "composition_domain_map", "Composition-domain coverage", "Train, IID, and B shown on common axes in total Cr+Nb weight percent and actual atomic Cr/Nb ratio. Each dot is a supplied composition; all finite coordinates and full ranges retained. This two-dimensional projection does not replace the full-feature distance audit.")

        liquid = next((p for p in phases if p.upper() == 'LIQUID'), None)
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
        if liquid is None:
            raise ValueError("LIQUID phase is required for Fig_S5B_10")
        truth = ood_df[f'{liquid}_presence_true'].to_numpy(int)
        pred = ood_df[f'{liquid}_presence_pred'].to_numpy(int)
        cm = np.array([[( (truth == i) & (pred == j)).sum() for j in (0, 1)] for i in (0, 1)])
        axes[0].imshow(cm, cmap='Blues', vmin=0)
        for (i, j), v in np.ndenumerate(cm):
            axes[0].text(j, i, f'{v:,}', ha='center', va='center', color='white' if v > cm.max() / 2 else '#222222')
        axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Absent', 'Present'], yticklabels=['Absent', 'Present'], xlabel='Predicted presence', ylabel='True presence', title='Presence counts')
        _cloud(axes[1], ood_df[f'{liquid}_volfrac_true'], ood_df[f'{liquid}_frac_proj'], parity=True)
        axes[1].set(xlabel='True liquid fraction', ylabel='Projected liquid fraction', title='Fraction parity')
        for field, label, color in [(f'{liquid}_presence_true', 'True prevalence', COLORS['IID']), (f'{liquid}_presence_pred', 'Predicted prevalence', COLORS['B'])]:
            g = ood_df.groupby('Temperature_C', observed=True)[field].mean()
            axes[2].plot(g.index, g, 'o-', ms=3, label=label, color=color)
        axes[2].set(xlabel='Temperature (°C)', ylabel='Presence fraction', ylim=(-.02, 1.02), title='Liquid prevalence')
        axes[2].legend(fontsize=7)
        save(fig, 10, "LIQUID_OOD", "Liquid-phase diagnostics in B OOD", "Liquid presence confusion matrix uses frozen binary predictions; cells show raw counts. Fraction parity includes absent-liquid zeros and all converged rows. Temperature prevalence uses observed converged rows per level, with no implied paired-composition balance.")
    (outdir / 'FIGURE_CAPTIONS.txt').write_text('\n'.join(captions), encoding='utf-8')
    return paths
