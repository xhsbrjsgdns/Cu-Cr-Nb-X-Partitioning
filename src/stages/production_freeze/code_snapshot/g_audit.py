"""Verify disk artifacts against source data and the frozen prospective contract."""
import importlib.util
import json
import os
import pathlib
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler
import g_common as C
import g_contract as G

def load(p): return json.load(open(p,encoding='utf-8'))
def main():
    items=[]
    def check(name,ok,detail=''):
        items.append(dict(item=name,result='PASS' if ok else 'FAIL',detail=str(detail)))
        print(('%-4s '%items[-1]['result'])+name,flush=True)
    fzp=os.path.join(C.D_FREEZE,'STEP5G_FINAL_PIPELINE_FREEZE.json')
    fz=load(fzp)
    prep=os.path.join(C.D_NIPRE,'STEP5G_NI_BLIND_PREREGISTRATION.json'); pre=load(prep)
    met=load(os.path.join(C.D_METRIC,'STEP5G_METRIC_PREREGISTRATION.json'))
    few=load(os.path.join(C.D_FEWSHOT,'STEP5G_POSTBLIND_FEWSHOT_PREREGISTRATION.json'))
    cp=os.path.join(C.D_NIPRE,'STEP5G_NI_BLIND_COMPOSITIONS.csv'); comp=pd.read_csv(cp)
    feats=C.feature_schema()
    check('Historical 43-feature list hash',C.sha256_file(C.A_FEATLIST)==C.A_FEATLIST_SHA256)
    schema_path=os.path.join(C.D_FREEZE,'STEP5G_FEATURE_SCHEMA.csv'); schema=pd.read_csv(schema_path)
    check('Schema names/order/definitions/units/transforms frozen',schema.feature.tolist()==feats and schema['order'].tolist()==list(range(1,44)) and all(k in schema for k in ['formula','unit','transform_rule']) and C.sha256_file(schema_path)==fz['feature_schema']['csv_sha256'])
    check('Source datasets unchanged since training',fz['source_hashes']==G.source_hashes())
    check('Code unchanged since production freeze',all(C.sha256_file(os.path.join(C.D_CODE,p))==h for p,h in fz['code_snapshot_hashes'].items()))
    check('PRIMARY ORIGINAL6 / SECONDARY FULL11 fixed',fz['primary_role']==C.PRIMARY_ROLE and fz['secondary_role']==C.SECONDARY_ROLE and len(fz['models'])==2)
    check('No diagnostic ablation selected',set(fz['models'])=={C.PRIMARY_ROLE,C.SECONDARY_ROLE})
    check('All four switch gates failed; ORIGINAL6 retained by rule',fz['step5f_gates']==C.FULL5_GATES)
    spec=importlib.util.spec_from_file_location('train82',os.path.join(C.D_CODE,'82_g_train.py'))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    tr,va,new=mod.load_frames(feats);G.verify_master(tr);G.verify_master(va)
    check('PRIMARY reconciles exact STEP5A split roles',len(tr)==241748 and len(va)==51802 and tr.composition_id.nunique()==22050)
    frames={C.PRIMARY_ROLE:tr,C.SECONDARY_ROLE:pd.concat([tr,new],ignore_index=True)}
    for role,frame in frames.items():
        m=fz['models'][role]
        check(role+' identities',set(frame.family)==set(C.ORIGINAL_SIX if role==C.PRIMARY_ROLE else C.FULL11))
        for key,expected in [('train_ids',frame),('validation_ids',va)]:
            ids=pd.read_csv(m[key]); cols=ids.columns.tolist()
            check(role+' '+key+' exact membership/hash',C.sha256_file(m[key])==m[key+'_sha256'] and ids.equals(expected[cols].sort_values('state_id').reset_index(drop=True)))
        check(role+' no IID/Domain B/ANCHOR/baseline',set(frame.sampling_domain)=={'A'} and not frame.split_role.eq('IID_TEST').any() and not frame.composition_id.str.contains('ANCHOR').any() and 'BASELINE' not in set(frame.family))
        check(role+' disjoint train/validation compositions',not set(frame.composition_id)&set(va.composition_id))
        for kind in ['checkpoint','scaler']:
            check(role+' '+kind+' hash',C.sha256_file(m[kind])==m[kind+'_sha256'])
        scaler=joblib.load(m['scaler']); expected=StandardScaler().fit(frame[feats])
        check(role+' scaler train-only fit and feature order',list(scaler.feature_names_in_)==feats and all(np.allclose(getattr(scaler,k),getattr(expected,k),rtol=1e-12,atol=1e-12) for k in ['mean_','var_','scale_']))
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=True)
        check(role+' checkpoint architecture/schema/role',ck['features']==feats and ck['role']==role and tuple(ck['state_dict']['0.weight'].shape)==(128,43) and tuple(ck['state_dict']['2.weight'].shape)==(64,128) and tuple(ck['state_dict']['4.weight'].shape)==(5,64))
        check(role+' fallback from own training targets',np.allclose(ck['fallback_row'],C.train_mean_simplex(frame[C.TRUE_COLS]),rtol=1e-12,atol=1e-12))
        check(role+' seed and allowed LR',m['seed']==C.child_seed(20260911,role) and m['chosen_learning_rate'] in C.MLP_LRS)
        trail=[r for r in fz['trails'] if r['role']==role]
        chosen=min(trail,key=lambda r:r['validation_tvd'])
        check(role+' LR selection trail complete and correct',len(trail)==3 and sorted(r['learning_rate'] for r in trail)==C.MLP_LRS and chosen['learning_rate']==m['chosen_learning_rate'] and chosen['best_epoch']==m['best_epoch'])
        model=mod.network(43); model.load_state_dict(ck['state_dict']); model.eval()
        with torch.no_grad(): raw=model(torch.as_tensor(scaler.transform(va[feats]),dtype=torch.float32)).numpy()
        score=float(C.tvd(C.project_simplex(raw,ck['fallback_row']),va[C.TRUE_COLS].to_numpy()).mean())
        check(role+' saved checkpoint reproduces validation TVD',abs(score-m['in_domain_validation_metrics']['mean_tvd'])<1e-6,score)
    check('Both models same exact validation IDs',fz['models'][C.PRIMARY_ROLE]['validation_ids_sha256']==fz['models'][C.SECONDARY_ROLE]['validation_ids_sha256'])
    check('Ni prereg binds production freeze and both models',pre['pipeline_freeze_sha256']==C.sha256_file(fzp) and all(pre['models'][tag]['checkpoint_sha256']==fz['models'][role]['checkpoint_sha256'] for tag,role in [('PRIMARY',C.PRIMARY_ROLE),('SECONDARY',C.SECONDARY_ROLE)]))
    check('Ni composition hash / 2000 / 22000',C.sha256_file(cp)==pre['composition_table_sha256'] and len(comp)==2000 and pre['formal_states']==22000 and pre['temperatures_C']==C.TEMPERATURES_C)
    check('Ni compositions unique',comp.composition_id.is_unique and not comp.duplicated(['Cr_wt_pct','Nb_wt_pct','X_wt_pct']).any())
    check('Symbolic Ni only; legal IDs accepted',not G.symbolic_scan(comp))
    check('No Ni properties used in inversion',pre['sampler']['inversion_masses']=={'M_Cr':C.M_CR,'M_Nb':C.M_NB} and pre['sampler']['atomic_fractions_computed'] is False)
    check('Composition weights close and within ranges',np.allclose(comp[['Cu_wt_pct','Cr_wt_pct','Nb_wt_pct','X_wt_pct']].sum(axis=1),100,atol=1e-8) and comp.S_CrNb_wt_pct.between(.6-5e-4,2+5e-4).all() and comp.R_CrNb_atomic_actual.between(2-5e-3,6.5+5e-3).all() and comp.X_wt_pct.between(.02,.20).all())
    check('Exact full-SHA256 600/1400 split recomputed',G.valid_split(comp))
    pool=comp.loc[comp.fewshot_role=='FEWSHOT_CALIBRATION','composition_id'].tolist()
    draws=G.nested_draws(pool)
    check('Three nested random sequences and exact budgets',few['sequences']==draws and few['budgets_nested'] is True and few['budgets_compositions']==[1,2,4,8,16,32] and few['replicates']==3 and few['anchor_selection_primary']=='RANDOM_NESTED')
    seq=pd.read_csv(few['sequence_file'])
    check('Shipped random sequences match prereg',C.sha256_file(few['sequence_file'])==few['sequence_sha256'] and all(seq.loc[seq.replicate==int(r)].sort_values('rank').composition_id.tolist()==v for r,v in draws.items()))
    check('Maximin secondary target-free only',few['anchor_selection_secondary']['strategy']=='TARGET_FREE_MAXIMIN' and few['anchor_selection_secondary']['uses_targets'] is False)
    check('Post-blind independent refit / immutable zero-shot',few['primary_estimator']=='PRIMARY_FORMULATION_REFIT' and few['refit']['retuning'] is False and few['refit']['learning_rate']==fz['models'][C.PRIMARY_ROLE]['chosen_learning_rate'])
    check('Recovery metrics/B_skill/B_recovery25 frozen',all(k in few for k in ['B_skill','B_recovery25','metrics','anchor_failure_policy']))
    check('Optional low-order branch not executed',few['optional_low_order_anchor_study']['executed'] is False and few['executed'] is False)
    check('H-A prediction hashes before H-B CALPHAD; ordered scores',pre['execution_order']==met['execution_order']==few['execution_order']==G.BLIND_ORDER)
    check('Exact future CALPHAD versions/mapping/partition/failure policy',pre['future_calphad']==G.CALPHAD)
    check('Bootstrap exactly 1000 compositions, seed 20260911',met['uncertainty']==G.BOOTSTRAP)
    check('Blind metric contract and no success threshold',met['success_threshold'] is None and met['primary_endpoint']['decision_model']==C.PRIMARY_ROLE and 'per_temperature_topology_accuracy' in met['secondary_endpoints'] and 'macro_mae' in met['secondary_endpoints'])
    check('All preregistrations bind same freeze/compositions',all(d['pipeline_freeze_sha256']==C.sha256_file(fzp) and d['composition_sha256']==C.sha256_file(cp) and d['ni_blind_preregistration_sha256']==C.sha256_file(prep) for d in [met,few]))
    patch=load(os.path.join(C.D_QA,'STEP5F_REPORT_PATCH_01.json'))
    before=pathlib.Path(patch['pre_patch_copy']).read_text(encoding='utf-8')
    after=pathlib.Path(C.F_REPORT).read_text(encoding='utf-8')
    exact_bytes=pathlib.Path(patch['pre_patch_copy']).read_bytes().replace(C.PATCH_FIND.encode(),C.PATCH_REPLACE.encode(),1)==pathlib.Path(C.F_REPORT).read_bytes()
    check('Report patch exact substring only and both hashes',exact_bytes and before.replace(C.PATCH_FIND,C.PATCH_REPLACE,1)==after and C.sha256_file(C.F_REPORT)==patch['sha256_after'] and C.sha256_file(patch['pre_patch_copy'])==patch['sha256_before'])
    snapshot=load(os.path.join(C.D_CONFIG,'STEP5F_SCIENCE_BEFORE.json'))
    check('STEP5F scientific files unchanged',all(C.sha256_file(p)==h for p,h in snapshot.items()),str(len(snapshot))+' files')
    pf=load(os.path.join(C.D_CONFIG,'STEP5G_PREFLIGHT.json'))
    check('Real preflight clean',pf['verdict']=='GO' and not pf['ni_scan_hits'])
    check('Ni access counters zero and no forbidden output files',all(v==0 for k,v in pre['ni_status'].items() if k!='symbolic_composition_rows') and not any('prediction' in p.name.lower() or 'equilibrium' in p.name.lower() for p in pathlib.Path(C.ROOT).rglob('*') if p.suffix in ['.parquet','.pt']))
    blockers=[i['item'] for i in items if i['result']=='FAIL']; decision='NOT READY' if blockers else 'READY FOR STEP5H-A'
    pd.DataFrame(items).to_csv(os.path.join(C.D_QA,'STEP5G_INTEGRITY_CHECKS.csv'),index=False)
    with open(os.path.join(C.D_QA,'STEP5G_INTEGRITY_AUDIT.md'),'w',encoding='utf-8') as f:
        f.write('# Integrity audit\n\n'+decision+'\n\n| Check | Result | Detail |\n|---|---|---|\n')
        for i in items:f.write('| {item} | {result} | {detail} |\n'.format(**i))
        f.write('\nEvidence bounds: the stage code/data and output artifacts were inspected; zero counters are workflow declarations, not a machine-wide historical access monitor.\n')
    report=build_report(fz,met,few,decision,blockers,len(items))
    pathlib.Path(C.ROOT,'STEP5G_FINAL_REPORT.md').write_text(report,encoding='utf-8')
    C.write_json(os.path.join(C.D_MANIFEST,'STEP5G_COMPLETION.json'),dict(generated_utc=C.utcnow(),decision=decision,items_pass=len(items)-len(blockers),items_total=len(items),blocking_reasons=blockers,scope='STEP5G only; no Ni descriptor, prediction, CALPHAD, target or recovery execution',manifest_policy='Final artifacts included. Index/sidecar excluded to avoid self-reference; active logs excluded and separately archived after process completion.'))
    count=G.write_manifest()
    print('%s | %d/%d checks | %d files hashed'%(decision,len(items)-len(blockers),len(items),count))
    return 2 if blockers else 0

def build_report(fz,met,few,decision,blockers,n):
    p=fz['models'][C.PRIMARY_ROLE];s=fz['models'][C.SECONDARY_ROLE]
    answers=[('Was the STEP5F patch report-only?','Yes; exact substring diff, backup and hashes verified.'),
      ('PRIMARY production role?',C.PRIMARY_ROLE),('SECONDARY production role?',C.SECONDARY_ROLE),
      ('Any ablation subset selected?','No; diagnostic only, including BASE_PLUS_MN 0.6917. Selecting it after seeing its score would be post-hoc.'),
      ('Exact feature count?',43),('PRIMARY identities?',p['train_identities']),('SECONDARY identities?',s['train_identities']),
      ('PRIMARY train rows/compositions?',(p['n_train_rows'],p['n_train_compositions'])),
      ('SECONDARY train rows/compositions?',(s['n_train_rows'],s['n_train_compositions'])),
      ('Validation rows/compositions?',(p['n_validation_rows'],p['n_validation_compositions'])),
      ('PRIMARY selected LR?',p['chosen_learning_rate']),('SECONDARY selected LR?',s['chosen_learning_rate']),
      ('PRIMARY best epoch?',p['best_epoch']),('SECONDARY best epoch?',s['best_epoch']),
      ('PRIMARY model SHA256?',p['checkpoint_sha256']),('SECONDARY model SHA256?',s['checkpoint_sha256']),
      ('Feature schema SHA256?',fz['feature_schema']['csv_sha256']),
      ('PRIMARY scaler SHA256?',p['scaler_sha256']),('SECONDARY scaler SHA256?',s['scaler_sha256']),
      ('Ni descriptors accessed?',0),('Ni properties accessed?',0),('Ni predictions created?',0),
      ('Ni CALPHAD states generated?',0),('Ni targets accessed?',0),('Ni compositions?',2000),
      ('Formal future Ni states?',22000),('Ni Sobol seed?',{'master':20260911,'stream':C.child_seed(20260911,'NI_BLIND_COMPOSITIONS')}),
      ('Blind primary endpoint?',met['primary_endpoint']),('Bootstrap policy?',G.BOOTSTRAP),
      ('Calibration pool size?',600),('Recovery-eval size?',1400),('Few-shot budgets?',[1,2,4,8,16,32]),
      ('Why random primary?',few['random_rationale']),('Why maximin secondary?','STEP5E did not show systematic superiority; secondary comparison only.'),
      ('Low-order branch optional only?','Yes; separate authorization after zero-shot scoring; no DFT.'),
      ('Any Ni label generated?',0),('Are production models immutable?','Yes; separate post-blind recovery refits are explicitly preregistered under sections 68-69 and never replace frozen zero-shot models.'),
      ('READY FOR STEP5H-A?',decision)]
    text='# STEP5G FINAL REPORT\n\n'+decision+'\n\nFULL5 improved historical LOEO mean TVD by 5.62% (0.752644 to 0.710332), but failed all four preregistered switching gates. ORIGINAL6 remains PRIMARY by protocol; this is not a claim that augmentation has no benefit.\n\n'
    for k,(q,a) in enumerate(answers,1): text+='**%d. %s** %s\n\n'%(k,q,a)
    text+='STEP5H-A: only approved input properties, first-access UTC, two frozen predictions and hashes, then STOP. STEP5H-B: CALPHAD only after both prediction hashes verify. PRIMARY score/hash precedes SECONDARY score/hash; recovery follows both.\n\nAtomic fractions intentionally absent because Ni atomic mass is not accessed. No atomic-closure QA is claimed.\n\nHistorical LOEO is retrospective; held-out targets were excluded from the corresponding fold development. No mechanical properties are predicted. Delta_XCr_FCC is an equilibrium Cr depletion proxy.\n\nAudit checks: %d. See the integrity audit for evidence bounds.\n\n'%n
    return text+'\n'.join(blockers)+'\n'+decision+'\n'
