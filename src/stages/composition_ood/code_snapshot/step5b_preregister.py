"""Write STEP5B_PREREGISTRATION_V1.json. Must run AFTER step5b_manifest.py and BEFORE
any Domain-B target read. All hashes are of files as they exist at this moment."""
import json
from step5b_core import A, O, now, sha, load_json, freeze_mod

def main():
    assert not (O / '03_predictions' / 'DOMAIN_B_OOD_PREDICTIONS.parquet').exists()
    pre_path = O / '01_preregistration' / 'STEP5B_PREREGISTRATION_V1.json'
    original_ts = None
    if pre_path.exists():
        # PATCH_01 resume: preserve the ORIGINAL preregistration timestamp so the
        # pre-open ordering (prereg < first-open record) stays literally true; only the
        # document content (key placement + refreshed hashes) is updated.
        original_ts = load_json(pre_path)['timestamp_utc']
    freeze = load_json(A / '09_manifest' / 'DEVELOPMENT_FREEZE_V1.json')
    cfg = load_json(O / '00_config' / 'PREOPEN_CONFIG.json')
    features = freeze['features']
    assert features == cfg['features'] and len(features) == 43

    frozen_artifacts = [
        dict(path=str(A / '09_manifest' / 'DEVELOPMENT_FREEZE_V1.json'),
             sha256=sha(A / '09_manifest' / 'DEVELOPMENT_FREEZE_V1.json')),
        dict(path=str(A / '02_features' / 'PRIMARY_FEATURE_LIST.txt'),
             sha256=sha(A / '02_features' / 'PRIMARY_FEATURE_LIST.txt')),
        dict(path=str(A / '02_features' / 'PCC_REMOVAL_LOG.csv'),
             sha256=sha(A / '02_features' / 'PCC_REMOVAL_LOG.csv')),
        dict(path=str(A / '02_features' / 'NEAR_ZERO_VARIANCE_REMOVAL.csv'),
             sha256=sha(A / '02_features' / 'NEAR_ZERO_VARIANCE_REMOVAL.csv')),
        dict(path=str(A / '02_features' / 'EXACT_REDUNDANCY_LOG.csv'),
             sha256=sha(A / '02_features' / 'EXACT_REDUNDANCY_LOG.csv')),
    ]
    for t in ['presence', 'fraction', 'cr_partition', 'nb_partition', 'x_partition']:
        p = A / f'04_models/final/{t}/model_state.pkl'
        assert sha(p) == freeze['tasks'][t]['artifact_sha256']
        frozen_artifacts.append(dict(path=str(p), sha256=sha(p)))
    prepared = []
    for rel in ['02_inputs/DOMAIN_B_FEATURES_FROZEN43.parquet',
                '02_inputs/DEV_TRAIN_COMPOSITION_CONTEXT.parquet',
                '02_inputs/IID_COMPOSITION_CONTEXT.parquet',
                '02_inputs/DOMAIN_B_COMPOSITION_CONTEXT.parquet',
                '05_shift_analysis/DOMAIN_B_OOD_DISTANCE.parquet',
                '05_shift_analysis/FEATURE_SHIFT_A_TO_B.csv',
                '00_config/DISTANCE_SCALER_DEV_TRAIN_ONLY.json',
                '00_config/PREOPEN_CONFIG.json']:
        prepared.append(dict(path=str(O / rel), sha256=sha(O / rel)))

    pre = dict(
        protocol='STEP5B prospective Domain-B composition-domain OOD evaluation',
        protocol_amendment_approved=True,
        protocol_amendment_document='01_preregistration/PROTOCOL_AMENDMENT_V1.md',
        retained_warning='historical STEP5A full-table target materialization means the '
                         'literal never-read condition is permanently FALSE; seal is defined '
                         'by PROTOCOL_AMENDMENT_V1 (never used for development/tuning/evaluation, '
                         'no pre-open B artifact); STEP5B status capped at PASS WITH WARNINGS',
        timestamp_utc=original_ts or now(),
        features=features,
        bins=cfg['bins'],
        feature_list_sha256=freeze['feature_list_sha256'],
        development_freeze_sha256=sha(A / '09_manifest' / 'DEVELOPMENT_FREEZE_V1.json'),
        frozen_artifacts=frozen_artifacts,
        prepared_inputs=prepared,
        code_manifest_sha256=sha(O / '09_manifest' / 'CODE_SNAPSHOT_MANIFEST_STEP5B.json'),
        package_versions=freeze_mod.lib_versions(),
        models={t: dict(family=freeze['tasks'][t]['model_family'],
                        hyperparameters=freeze['tasks'][t]['hyperparameters'],
                        artifact_sha256=freeze['tasks'][t]['artifact_sha256']) for t in freeze['tasks']},
        classification_thresholds=dict(zip(['FCC', 'BCC', 'C15', 'OTHER_SOLID', 'LIQUID'],
                                           cfg['thresholds'])),
        secondary_threshold_report='default 0.5 reported as secondary only; primary results '
                                   'use the frozen thresholds above',
        projection_rules='clip negatives to 0; renormalize to sum 1; if sum==0 use the frozen '
                         'DEV_TRAIN mean simplex (counted); never use Domain-B labels',
        random_seeds=dict(master=20260904, step5b_bootstrap=cfg['seed']),
        bootstrap=dict(scheme='composition-level cluster bootstrap, all temperature rows of a '
                              'sampled composition travel together; IID and OOD draws use '
                              'independent SeedSequence children (not paired)',
                       n_reps=cfg['n_bootstrap'], seed=cfg['seed']),
        metrics=dict(
            presence=['Exact Match', 'Hamming Loss', 'Micro F1', 'Macro F1',
                      'per-family support/Precision/Recall/F1/ROC-AUC/PR-AUC',
                      'LIQUID reported separately'],
            regression=['mean/median/p95 TVD', 'Macro MAE', 'Macro RMSE',
                        'per-family MAE/RMSE/R2', 'raw sum violation mean/max',
                        'projected sum violation mean/max', 'simplex fallback count']),
        stratification_policy=dict(
            fixed_before_open=True,
            groups=['family', 'Temperature_C', 'S_CrNb_wt_pct quartile bins',
                    'R_CrNb_atomic_actual quartile bins', 'X_wt_pct quartile bins',
                    'phase split vs non-split (post-hoc metadata only)',
                    'OOD nn1-distance quartiles'],
            bin_edges=cfg['bins'],
            no_post_hoc_groups='adding a group after seeing results is forbidden'),
        ood_distance_definition=dict(
            level='composition (never state rows; 11 temperatures per composition are not '
                  'independent samples)',
            features='the 42 frozen composition-dependent features (43 minus Temperature_C)',
            scaler='StandardScaler fit on DEV_TRAIN compositions only; diagnostic use only, '
                   'never a model transform',
            distances=['1-NN Euclidean in scaled space to DEV_TRAIN compositions',
                       '5-NN mean Euclidean', 'outside-train-range feature count']),
        traffic_lights=dict(
            note='internal project criteria, not JMST official standards; decided before open; '
                 'results never trigger retuning',
            presence_macro_f1=dict(GREEN='>=0.90', YELLOW='0.75-0.90', RED='<0.75'),
            fraction_mean_tvd=dict(GREEN='<=0.005', YELLOW='0.005-0.02', RED='>0.02'),
            partition_mean_tvd=dict(GREEN='<=0.03', YELLOW='0.03-0.10', RED='>0.10')),
        trivial_baselines='STEP5A DEV_TRAIN-derived only (prevalence / mean simplex vectors '
                          'stored in the frozen artifacts); Domain-B means are never used',
        go_no_go=dict(
            A='no data leakage', B='STEP5A frozen artifacts unchanged',
            C='first-open protocol valid', D='all primary tasks clearly beat frozen trivial '
            'baselines', E='no FAILED interpolation', F='Anchor sealed', G='Ni sealed',
            H='OOD degradation quantified and explainable by shift/distance',
            yellow_rule='YELLOW on a task does not auto NO-GO if skill remains and shift explains it'),
        prohibitions=['retraining', 'Domain B tuning/threshold/PCC/feature-selection/scaler',
                      'removing hard B samples', 'interpolating MO@950 FAILED',
                      'family/X_element/sampling_domain as features', 'one-hot element IDs',
                      'CALPHAD/target-derived descriptors', 'ANCHOR predictions or metrics',
                      'any Ni operation', '6xLOEO', 'targeted/random augmentation'],
    )
    (O / '01_preregistration' / 'STEP5B_PREREGISTRATION_V1.json').write_text(
        json.dumps(pre, indent=2, ensure_ascii=False), encoding='utf-8')
    print('preregistration written:', sha(O / '01_preregistration' / 'STEP5B_PREREGISTRATION_V1.json'))


if __name__ == '__main__':
    main()
