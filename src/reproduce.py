"""Portable checks and stage launcher for the Cu-Cr-Nb-X manuscript release.

Only standard-library packages are needed for verify/prepare. Summary uses
numpy, pandas and pyarrow. Scientific source modules are in src/stages/.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

RELEASE = Path(__file__).resolve().parents[1]
FAMILIES = ['FCC', 'BCC', 'C15', 'OTHER_SOLID', 'LIQUID']


def manifest():
    return json.loads((RELEASE / 'data/manifest.json').read_text(encoding='utf-8'))


def sha_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(2**20), b''):
        h.update(block)
    return h.hexdigest()


@contextmanager
def source_stream(entry):
    path = RELEASE / entry['storage_path']
    if entry['compression'] == 'gzip-parts':
        with tempfile.TemporaryFile() as merged:
            for part in entry['storage_parts']:
                with (RELEASE/part).open('rb') as f:
                    shutil.copyfileobj(f, merged)
            merged.seek(0)
            with gzip.GzipFile(fileobj=merged, mode='rb') as stream:
                yield stream
    else:
        with (gzip.open(path, 'rb') if entry['compression'] == 'gzip' else path.open('rb')) as stream:
            yield stream


def verify():
    m = manifest()
    checked = {}
    for e in m['data_files']:
        key = e['storage_path']
        if key not in checked:
            with source_stream(e) as stream:
                checked[key] = sha_stream(stream)
        if checked[key] != e['sha256']:
            raise ValueError(f'Checksum mismatch: {key}')
    for e in m['source_files']:
        with (RELEASE / e['storage_path']).open('rb') as stream:
            if sha_stream(stream) != e['sha256']:
                raise ValueError(f'Source checksum mismatch: {e["storage_path"]}')
        compile((RELEASE / e['storage_path']).read_text(encoding='utf-8-sig'), e['storage_path'], 'exec')
    report = {'data_paths_verified': len(m['data_files']),
              'unique_data_files_verified': len(checked),
              'source_files_verified_and_compiled': len(m['source_files']),
              'result': 'PASS'}
    print(json.dumps(report, indent=2))
    return report


def data_entry(original):
    return next(e for e in manifest()['data_files'] if e['original_path'] == original)


def read_table(original, **kwargs):
    import pandas as pd
    entry = data_entry(original)
    with source_stream(entry) as stream:
        if original.endswith('.parquet'):
            return pd.read_parquet(stream, **kwargs)
        return pd.read_csv(stream, float_precision='round_trip', **kwargs)


def summary(output):
    """Recompute results from individual predictions/reference vectors, not prose."""
    import numpy as np
    import pandas as pd
    P = FAMILIES
    values = {}
    checks = []

    def check(name, ok, actual):
        checks.append({'check': name, 'passed': bool(ok), 'actual': actual})
        if not ok:
            raise AssertionError(f'{name}: {actual}')

    reference = '03_full_CALPHAD/FULL_V1/'
    d = read_table(reference + '03_processed/FULL_V1_ML_TARGET_ALL.parquet',
                   columns=['state_id', 'composition_id', 'family', 'Temperature_C', 'state_converged'])
    good = d.state_converged.astype(bool)
    counts = [len(d), int(good.sum()), int((~good).sum()), int(d.composition_id.nunique())]
    check('Reference population, including numerical failures', counts == [550000, 548432, 1568, 50000], counts)
    check('Failures are Mo at 950 C', (d.loc[~good, 'family'].eq('MO') & d.loc[~good, 'Temperature_C'].eq(950)).all(), int((~good).sum()))
    values['reference_counts'] = counts
    del d

    iid = read_table('04_ML/STEP5A_JMST_ML_FOUNDATION_V2/05_results/IID_TEST_PREDICTIONS.parquet')
    pred = iid[[f'{p}_X_P_proj' for p in P]].to_numpy(float)
    true = iid[[f'X_P_{p}_true' for p in P]].to_numpy(float)
    mask = np.isfinite(true).all(axis=1)
    iid_tvd = float((0.5 * abs(pred[mask] - true[mask]).sum(axis=1)).mean())
    check('IID X-partition TVD agrees at submitted precision', abs(iid_tvd - .001640) < .0000005, iid_tvd)
    values['iid_x_tvd'] = iid_tvd
    # Recompute label-wise F1 without loading trained estimators.
    yt = iid[[f'{p}_presence_true' for p in P]].to_numpy(bool)
    yp = iid[[f'{p}_presence_pred' for p in P]].to_numpy(bool)
    tp = (yt & yp).sum(axis=0); fp = (~yt & yp).sum(axis=0); fn = (yt & ~yp).sum(axis=0)
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(5, dtype=float), where=(2*tp+fp+fn) != 0)
    values['iid_presence_macro_f1'] = float(f1.mean())
    check('IID macro-F1 agrees at submitted precision', round(float(f1.mean()), 4) == .9951, float(f1.mean()))
    del iid

    folds = {}; wrong_mass = 0.; total_mass = 0.
    for element in ['ZR', 'TI', 'CO', 'SI', 'AG', 'MO']:
        d = read_table(f'05_OOD/STEP5C_6X_LOEO_V1/05_predictions/FOLD_{element}_HELDOUT_PREDICTIONS.parquet')
        y = d[[f'true_x_partition_{p}' for p in P]].to_numpy(float)
        p = d[[f'pred_proj_x_partition_{f}' for f in P]].to_numpy(float)
        e = .5 * abs(p-y).sum(axis=1)
        folds[element] = float(e.mean())
        wrong = y.argmax(axis=1) != p.argmax(axis=1)
        wrong_mass += float(e[wrong].sum()); total_mass += float(e.sum())
    values['loeo_x_tvd_by_element'] = folds
    values['loeo_unweighted_six_fold_mean'] = float(np.mean(list(folds.values())))
    values['loeo_topology_mismatch_error_share'] = wrong_mass / total_mass
    check('LOEO unweighted six-fold TVD', round(values['loeo_unweighted_six_fold_mean'], 4) == .8505, values['loeo_unweighted_six_fold_mean'])
    check('Pooled wrong-topology error share', round(100*wrong_mass/total_mass, 1) == 96.7, wrong_mass/total_mass)

    hb = '07_FINAL_BLIND/STEP5H_B_NI_BLIND_REVEAL_V1/'
    ha = '07_FINAL_BLIND/STEP5H_A_PREDICTION_FREEZE_V1/'
    labels = read_table(hb + '03_processed_targets/STEP5H_B_NI_TARGETS.parquet')
    pred = read_table(ha + '04_primary_predictions/STEP5H_A_PRIMARY_NI_PREDICTIONS.parquet')
    d = pred.merge(labels, on='state_id', suffixes=('_pred', '_reference'), validate='one_to_one')
    yp = d[[f'pred_{f}' for f in P]].to_numpy(float)
    yt = d[[f'true_{f}' for f in P]].to_numpy(float)
    tvd = .5 * abs(yp-yt).sum(axis=1)
    values['ni_blind_tvd'] = float(tvd.mean())
    values['ni_blind_topology_accuracy'] = float((yp.argmax(axis=1) == yt.argmax(axis=1)).mean())
    check('Ni blind population', len(d) == 22000 and labels.composition_id.nunique() == 2000, len(d))
    check('Ni blind TVD', round(float(tvd.mean()), 4) == .8644, float(tvd.mean()))
    ranges = read_table(ha + '03_ood_diagnostics/STEP5H_A_NI_FEATURE_RANGE_DIAGNOSTICS.csv')
    check('All 43 Ni features within training marginal ranges', len(ranges) == 43 and ranges.extends_range.sum() == 0 and ranges.outside_fraction.sum() == 0, len(ranges))

    means = {}; recovery_rows = []
    for budget, target in [(1,.2482), (8,.0507), (32,.0272)]:
        scores = []; eval_ids = None
        for rep in [1,2,3]:
            p = read_table(f'07_FINAL_BLIND/STEP5I_POSTBLIND_FEWSHOT_RECOVERY_V1/02_random_refit/STEP5I_RANDOM_R{rep}_B{budget}_PREDICTIONS.parquet')
            ids = set(p.state_id)
            if eval_ids is None: eval_ids = ids
            check(f'Recovery B{budget} R{rep} same evaluation IDs', ids == eval_ids and len(ids) == 15400, len(ids))
            q = p.merge(labels, on='state_id', validate='one_to_one')
            e = .5 * abs(q[[f'refit_pred_{f}' for f in P]].to_numpy(float) - q[[f'true_{f}' for f in P]].to_numpy(float)).sum(axis=1)
            scores.append(float(e.mean()))
            recovery_rows.append({'budget':budget, 'replicate':rep, 'mean_tvd':scores[-1]})
        means[str(budget)] = float(np.mean(scores))
        check(f'Ni refit TVD B{budget}', round(means[str(budget)], 4) == target, means[str(budget)])
    values['ni_refit_tvd'] = means

    si = 'JMST_SUPPLEMENTARY_MATERIAL/review/data/'
    lit = read_table(si + 'Supplementary_Data_S20.csv')
    selected = lit[lit.selected.eq(1)]
    counts = selected.verdict.value_counts().to_dict()
    check('Literature comparison population', len(selected) == 120, len(selected))
    check('Literature grades', counts == {'STRONGLY_CONSISTENT':61, 'CONSISTENT':20, 'PARTIALLY_CONSISTENT':34, 'DISCREPANT':5}, counts)
    values['literature_grades'] = counts
    candidates = read_table(si + 'Supplementary_Data_S22.csv')
    values['screening_compositions'] = len(candidates)
    check('Screening candidate population', len(candidates) == 120000, len(candidates))
    finalists = read_table(si + 'Supplementary_Data_S23.csv')
    check('Finalist count', len(finalists) == 5, len(finalists))
    report = {'result':'PASS', 'recomputed_values':values, 'checks':checks,
              'scope':'Recomputed from released reference and prediction files; not a retraining or a fresh CALPHAD calculation.'}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(recovery_rows).to_csv(output/'ni_recovery_recomputed.csv', index=False)
    pd.DataFrame([{'element':k,'mean_tvd':v} for k,v in folds.items()]).to_csv(output/'loeo_recomputed.csv', index=False)
    print(json.dumps(report, indent=2))
    return report


def prepare(workspace):
    """Restore original paths in an isolated directory; preserve data bytes."""
    workspace = Path(workspace).resolve()
    if workspace == RELEASE or workspace in RELEASE.parents or RELEASE in workspace.parents:
        raise ValueError('Use a workspace outside the release directory.')
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError('Workspace must be empty; existing research results will not be overwritten.')
    workspace.mkdir(parents=True, exist_ok=True)
    m = manifest()
    for e in m['data_files']:
        p = workspace/e['original_path']; p.parent.mkdir(parents=True, exist_ok=True)
        with source_stream(e) as reader, p.open('wb') as writer:
            shutil.copyfileobj(reader, writer)
    root = workspace.as_posix()
    changes = []
    for e in m['source_files']:
        p = workspace/e['original_path']; p.parent.mkdir(parents=True, exist_ok=True)
        original = (RELEASE/e['storage_path']).read_text(encoding='utf-8-sig')
        updated = original
        for old in ['E:\\\\Cu_CALPHAD_ML', 'E:\\Cu_CALPHAD_ML', 'E:/Cu_CALPHAD_ML']:
            updated = updated.replace(old, '__RELEASE_WORKSPACE_ROOT__')
        updated = updated.replace('__RELEASE_WORKSPACE_ROOT__', root)
        p.write_text(updated, encoding='utf-8')
        if original != updated: changes.append(e['original_path'])
    metadata_changes = []
    for p in workspace.rglob('*'):
        if p.is_file() and p.suffix in ['.csv','.json','.txt'] and p.stat().st_size < 16*2**20:
            text = p.read_text(encoding='utf-8-sig')
            updated = text
            for old in ['E:\\\\Cu_CALPHAD_ML', 'E:\\Cu_CALPHAD_ML', 'E:/Cu_CALPHAD_ML']:
                updated = updated.replace(old, '__RELEASE_WORKSPACE_ROOT__')
            updated = updated.replace('__RELEASE_WORKSPACE_ROOT__', root)
            if text != updated:
                p.write_text(updated, encoding='utf-8'); metadata_changes.append(p.relative_to(workspace).as_posix())
    (workspace/'RELEASE_PORTABILITY.json').write_text(json.dumps({'source_path_only_rewrites':changes, 'metadata_path_only_rewrites':metadata_changes, 'original_sources_preserved_in':str(RELEASE/'src/stages'), 'note':'Historical source/metadata hashes differ after path substitution. All unmodified originals and their checksums remain in the release; numerical arrays are unchanged. Historical provenance audits should be read as records, not regenerated with different paths.'}, indent=2), encoding='utf-8')
    print(f'Prepared {workspace}; {len(changes)} source files received path-only substitutions.')


def run_stage(name, workspace, extra):
    m = manifest()
    matches = [e for e in m['source_files'] if e['storage_path'] == name or Path(e['storage_path']).name == name]
    if len(matches) != 1: raise ValueError('Use a unique script name or the full src/stages path from list-stages.')
    workspace = Path(workspace).resolve()
    if not (workspace/'RELEASE_PORTABILITY.json').exists(): raise ValueError('Run prepare first.')
    script = workspace/matches[0]['original_path']
    env = os.environ.copy(); env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONPATH'] = str(script.parent) + os.pathsep + env.get('PYTHONPATH','')
    if extra and extra[0] == '--': extra = extra[1:]
    subprocess.run([sys.executable, str(script), *extra], cwd=workspace, env=env, check=True)


def inference(output):
    """Rerun both frozen production models using only release files, on CPU."""
    import joblib
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    features = read_table('07_FINAL_BLIND/STEP5H_A_PREDICTION_FREEZE_V1/02_ni_features/STEP5H_A_NI_FEATURES.parquet')
    results = []
    for role, folder in [('PRIMARY','02_primary_model'),('SECONDARY','03_secondary_model')]:
        prefix = f'07_FINAL_BLIND/STEP5G_PIPELINE_FREEZE_V1/{folder}/STEP5G_{role}'
        with source_stream(data_entry(prefix+'_MODEL_CHECKPOINT.pt')) as f:
            ck = torch.load(f, map_location='cpu', weights_only=False)
        with source_stream(data_entry(prefix+'_SCALER.joblib')) as f:
            scaler = joblib.load(f)
        cols = list(ck['features'])
        if len(cols) != 43 or list(scaler.feature_names_in_) != cols:
            raise AssertionError('Frozen feature schema/scaler mismatch')
        # Identical architecture, dtype and projection to 94_h_predict.py.
        model = nn.Sequential(nn.Linear(43,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,5))
        model.load_state_dict(ck['state_dict']); model.eval()
        with torch.no_grad():
            raw = model(torch.as_tensor(scaler.transform(features[cols].to_numpy(float)),dtype=torch.float32)).numpy().astype(float)
        y = np.maximum(raw,0); sums = y.sum(axis=1); zero = sums == 0
        y[~zero] /= sums[~zero,None]; y[zero] = np.asarray(ck['fallback_row'])
        folder_pred = '04_primary_predictions' if role == 'PRIMARY' else '05_secondary_predictions'
        old = read_table(f'07_FINAL_BLIND/STEP5H_A_PREDICTION_FREEZE_V1/{folder_pred}/STEP5H_A_{role}_NI_PREDICTIONS.parquet')
        old = old.set_index('state_id').loc[features.state_id]
        maximum = float(abs(y-old[[f'pred_{p}' for p in FAMILIES]].to_numpy(float)).max())
        raw_max = float(abs(raw-old[[f'raw_{p}' for p in FAMILIES]].to_numpy(float)).max())
        if maximum > 1e-7 or raw_max > 1e-7:
            raise AssertionError(f'{role} inference differs: raw={raw_max}, projected={maximum}')
        results.append({'role':role,'states':len(features),'maximum_raw_difference':raw_max,'maximum_projected_difference':maximum,'passed':True})
    report={'result':'PASS','models':results,'scope':'Fresh frozen-checkpoint CPU inference; no model training or CALPHAD rerun.'}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'inference.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('verify'); sub.add_parser('list-stages')
    p = sub.add_parser('summary'); p.add_argument('--output', default='reproduced_results')
    p = sub.add_parser('inference'); p.add_argument('--output', default='reproduced_results')
    p = sub.add_parser('prepare'); p.add_argument('--workspace', required=True)
    p = sub.add_parser('run'); p.add_argument('script'); p.add_argument('--workspace', required=True); p.add_argument('extra', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.command == 'verify': verify()
    elif args.command == 'summary': summary(args.output)
    elif args.command == 'inference': inference(args.output)
    elif args.command == 'prepare': prepare(args.workspace)
    elif args.command == 'run': run_stage(args.script,args.workspace,args.extra)
    else:
        for e in manifest()['source_files']: print(e['storage_path'])


if __name__ == '__main__':
    main()
