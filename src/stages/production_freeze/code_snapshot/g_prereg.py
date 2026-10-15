"""Freeze blind metrics and post-blind recovery; no Ni descriptor/target access."""
import json
import os
import pandas as pd
import g_common as C
import g_contract as G

def main():
    fzp=os.path.join(C.D_FREEZE,'STEP5G_FINAL_PIPELINE_FREEZE.json')
    prep=os.path.join(C.D_NIPRE,'STEP5G_NI_BLIND_PREREGISTRATION.json')
    comp_path=os.path.join(C.D_NIPRE,'STEP5G_NI_BLIND_COMPOSITIONS.csv')
    fz=json.load(open(fzp)); pre=json.load(open(prep))
    assert pre['pipeline_freeze_sha256']==C.sha256_file(fzp)
    assert pre['composition_table_sha256']==C.sha256_file(comp_path)
    comp=pd.read_csv(comp_path); assert G.valid_split(comp)
    primary=fz['models'][C.PRIMARY_ROLE]
    common={'generated_utc':C.utcnow(), 'pipeline_freeze_sha256':C.sha256_file(fzp),
            'ni_blind_preregistration_sha256':C.sha256_file(prep),
            'composition_sha256':C.sha256_file(comp_path), 'execution_order':G.BLIND_ORDER}
    metric={**common, 'preregistration_id':'STEP5G_METRIC_V2',
        'primary_endpoint':{'name':'mean_tvd','decision_model':C.PRIMARY_ROLE,
                            'definition':'0.5 * sum(abs(pred-true)), mean over converged states'},
        'family_order':C.FAMILIES_ORDER,
        'secondary_endpoints':['median_tvd','p95_tvd','macro_mae','topology_accuracy','topology_macro_f1',
            'per_temperature_mean_tvd','per_temperature_topology_accuracy','per_family_mae',
            'skill_vs_primary_trivial','simplex_fallback_count'],
        'definitions':{'topology':'argmax family; ties resolved by frozen family order',
            'macro_f1':'all five labels, zero_division=0', 'p95':'numpy quantile 0.95, linear',
            'macro_mae':'mean absolute error over all five components and converged states',
            'skill':'PRIMARY frozen train-mean TVD minus model TVD; positive is better',
            'fallback':'count rows whose sum AFTER negative clipping equals zero'},
        'baselines':{role:m['fallback_row'] for role,m in fz['models'].items()},
        'uncertainty':G.BOOTSTRAP,'failure_policy':G.CALPHAD['failure_policy'],
        'success_threshold':None,'roles_immutable':True,
        'scoring_order':'PRIMARY score+hash, SECONDARY score+hash, only then post-blind recovery'}
    pool=comp.loc[comp.fewshot_role=='FEWSHOT_CALIBRATION','composition_id'].tolist()
    draws=G.nested_draws(pool)
    rows=[{'replicate':r,'rank':rank,'composition_id':cid} for r,seq in draws.items() for rank,cid in enumerate(seq,1)]
    seq_path=os.path.join(C.D_FEWSHOT,'STEP5G_NESTED_RANDOM_SEQUENCES.csv')
    pd.DataFrame(rows).to_csv(seq_path,index=False)
    few={**common,'preregistration_id':'STEP5G_POSTBLIND_FEWSHOT_V2',
        'name':'POST-BLIND FEW-SHOT RECOVERY','alternative_name':'THERMODYNAMIC ANCHOR RECOVERY',
        'is_not':['zero-shot','blind','prospective zero-shot'],
        'runs_only_after':'both PRIMARY and SECONDARY blind scores and hashes are frozen',
        'split':{'rule':C.FEWSHOT_SPLIT_RULE,'n_calibration':600,'n_recovery_evaluation':1400,
                 'may_be_redrawn':False,'all_temperatures_grouped':True},
        'evaluation_note':'RECOVERY_EVALUATION labels were part of initial blind scoring; this is a post-blind recovery evaluation subset.',
        'anchor_selection_primary':'RANDOM_NESTED',
        'random_rationale':'STEP5E found maximin worse than random at 1-8 shots.',
        'master_seed':20260912,'replicates':3,'budgets_compositions':[1,2,4,8,16,32],
        'budgets_nested':True,'sequence_seed_rule':'sha256(20260912|FEWSHOT_SEQ_<replicate>), first four bytes big-endian; numpy default_rng permutation of sorted calibration IDs',
        'sequences':draws,'sequence_file':seq_path,'sequence_sha256':C.sha256_file(seq_path),
        'budget_rule':'first b composition IDs of each 32-composition sequence; all 11 temperatures grouped',
        'anchor_selection_secondary':{
            'strategy':'TARGET_FREE_MAXIMIN','status':'SECONDARY COMPARISON ONLY, not preferred',
            'geometry':'At STEP5H-A, use frozen PRIMARY scaler and 43 inputs. Flatten 11 standardized rows per composition in temperature order into 473 coordinates. Euclidean distance. Reference only original PRIMARY train compositions with all 11 input rows available.',
            'selection':'First calibration composition maximizes nearest distance to PRIMARY training compositions. Subsequently maximize minimum distance to training and selected compositions. Lexicographically smallest composition_id breaks exact ties. One nested sequence of 32.',
            'uses_targets':False,'uses_errors':False,'uses_topology_labels':False,
            'execution':'Defer geometry until Ni inputs are available; freeze selected IDs before Ni CALPHAD.'},
        'primary_estimator':'PRIMARY_FORMULATION_REFIT',
        'refit':{'data':'exact PRIMARY original train IDs plus selected Ni calibration compositions; no recovery-eval labels used for fitting or stopping',
            'artifact_policy':'new post-blind recovery checkpoint, never overwrite or replace either frozen zero-shot checkpoint',
            'initialization':'fresh PRIMARY architecture from PRIMARY production child seed; do not carry weights from another budget',
            'architecture':fz['protocol']['architecture'],'optimizer':'Adam','loss':'raw-component MSE',
            'learning_rate':primary['chosen_learning_rate'],'batch_size':1024,'max_epochs':200,
            'patience':15,'seed':primary['seed'],
            'validation':'exact original-six PRIMARY DEV_VALIDATION IDs; early stopping on raw MSE; no Ni validation or LR search',
            'scaler':'frozen PRIMARY scaler; never refit scaler on Ni',
            'projection':'clip negatives, renormalize; zero-sum uses frozen PRIMARY training-mean simplex',
            'weighting':'natural row weighting','retuning':False,
            'scope_resolution':'Sections 30/58 protect zero-shot artifacts; sections 68/69 permit only separately named post-blind recovery weight refits.'},
        'secondary_controls':C.RECOVERY_ESTIMATORS,
        'control_interpretation':'Report paired TVD difference versus R0. Failure to outperform R0 means no demonstrated improvement beyond anchors under this protocol; do not infer universal absence of model information.',
        'metrics':['mean_tvd','median_tvd','p95_tvd','topology_accuracy',
                   'relative_tvd_reduction_vs_PRIMARY_zero_shot_same_subset','skill_vs_PRIMARY_trivial_same_subset'],
        'evaluation':'same converged RECOVERY_EVALUATION rows for all methods and budgets; report each replicate plus mean and spread',
        'B_skill':'smallest budget with mean TVD across three random replicates below PRIMARY training-mean trivial on the same subset; >32 if none',
        'B_recovery25':'smallest budget whose across-replicate mean TVD is reduced >=25% vs PRIMARY zero-shot on same subset; >32 if none',
        'anchor_failure_policy':'Do not replace selected compositions. Use converged anchor states only; report effective states per temperature. If no anchor exists at a temperature, mean-shift controls use zero shift and R0 uses frozen PRIMARY trivial. No fabricated labels.',
        'optional_low_order_anchor_study':{'status':'OPTIONAL SECONDARY ONLY',
            'comparison':'quaternary direct Ni anchors vs Cu-Ni / Cr-Ni / Nb-Ni low-order anchors',
            'requirements':'separate user authorization after zero-shot scores; clean uniformly defined CALPHAD protocol',
            'framework':'existing CALPHAD only, no DFT',
            'name':'THERMODYNAMICALLY ANCHORED FEW-SHOT / LOW-ORDER TRANSFER','executed':False},'executed':False}
    for directory,name,data in [(C.D_METRIC,'STEP5G_METRIC_PREREGISTRATION',metric),
                                (C.D_FEWSHOT,'STEP5G_POSTBLIND_FEWSHOT_PREREGISTRATION',few)]:
        path=os.path.join(directory,name+'.json')
        if os.path.exists(path): raise RuntimeError('Refusing overwrite of preregistration '+path)
        C.write_json(path,data)
        with open(os.path.join(directory,name+'.md'),'w',encoding='utf-8') as f:
            f.write('# '+name+'\n\nFrozen before any Ni descriptor or target access.\n\n```json\n'+json.dumps(data,indent=2)+'\n```\n')
    print('STAGE 4 PASS: blind protocol and nested recovery preregistered; no recovery executed.')
    return 0

if __name__=='__main__': raise SystemExit(main())
