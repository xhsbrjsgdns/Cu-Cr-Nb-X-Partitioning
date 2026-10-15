"""Read-only consistency checks on assembled frozen-data extracts."""
from pathlib import Path
import json,re,csv,hashlib,ast
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parents[1]
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def d(n):return pd.read_csv(O/f'review/data/Supplementary_Data_S{n}.csv',float_precision='round_trip')
H=js(O/'audit/INPUT_HASHES.json');M=js(O/'audit/DATA_MANIFEST.json');F=js(O/'audit/FACTS.json')
checks=[]
def test(name,actual,expected,tol=0):
 ok=bool(np.isclose(actual,expected,rtol=0,atol=tol)) if isinstance(actual,(float,int,np.number)) else actual==expected
 checks.append(dict(check=name,actual=actual,expected=expected,absolute_tolerance=tol,status='PASS' if ok else 'FAIL'))
q=d(1);c=d(5)
test('reference_unique_compositions',c.composition_id.nunique(),50000);test('nominal_states',int(q.nominal_states.sum()),550000);test('converged_states',int(q.converged_states.sum()),548432);test('failed_states',int(q.failed_states.sum()),1568)
test('failure_identity',q.loc[q.failed_states.gt(0),'family'].unique().tolist(),['MO']);test('failure_temperature',q.loc[q.failed_states.gt(0),'Temperature_C'].unique().tolist(),[950])
test('reference_composition_IDs_unique',bool(c.composition_id.is_unique),True)
for k,n in {'DEV_TRAIN':24500,'DEV_VALIDATION':5250,'IID_TEST':5250}.items():
 test('split_'+k,int(c.split_role.eq(k).sum()),n)
f=d(4);test('feature_count',len(f),193);test('retained_features',int(f.selection_status.eq('retained').sum()),43)
for el,n in zip(['ZR','TI','CO','SI','AG','MO'],[39,38,39,38,38,37]):test('LOEO_features_'+el,int(f['LOEO_'+el].sum()),n)
eta=d(8);test('eta_states',sorted(eta.n_states.unique().tolist()),[345346]);test('eta_compositions',sorted(eta.n_compositions.unique().tolist()),[31500]);test('SS_decomposition_max_residual',float((eta.SS_total-eta.SS_between-eta.SS_within).abs().max()),0.,1e-7)
for task,v in [('fraction',.0774608057359036),('cr_partition',.2046470581679638),('nb_partition',.0417232727455628),('x_partition',.885454877763191)]:test('state_equal_eta_'+task,float(eta.loc[eta.task.eq(task)&eta.weighting.eq('state_equal'),'eta2_vec'].iloc[0]),v,1e-12)
e=d(9);pooled=float(e.loc[e.topology_status.eq('TOPOLOGY_WRONG'),'sum_tvd'].sum()/e.sum_tvd.sum());g=e.pivot(index='heldout_element',columns='topology_status',values='sum_tvd').fillna(0);macro=float((g.TOPOLOGY_WRONG/g.sum(axis=1)).mean());test('wrong_topology_pooled',pooled,.9669648302160682,1e-12);test('wrong_topology_macro',macro,.96370730644,1e-10)
cm=d(7);cm=cm[cm.source_table.eq('LOEO_TOPOLOGY_METRICS')];pop=[]
for _,r in cm.iterrows():
 a=np.asarray(json.loads(r.confusion_matrix));pop.append(int(a.sum()));n=int(e.loc[e.heldout_element.eq(r.heldout_element),'n_rows'].sum());test('topology_group_completeness_'+r.heldout_element,n,int(a.sum()))
 if not (e.heldout_element.eq(r.heldout_element)&e.topology_status.eq('TOPOLOGY_CORRECT')).any():test('omitted_correct_group_is_empty_'+r.heldout_element,int(np.trace(a)),0)
test('LOEO_population',sum(pop),345346)
oracle=d(10);test('oracle_macro',float(oracle.mean_tvd.mean()),.3528401003198626,1e-12);test('oracle_pooled',float(np.average(oracle.mean_tvd,weights=oracle.n_rows)),.3522303724903578,1e-12)
ni=d(11);seq=d(12);test('Ni_total',len(ni),2000);test('Ni_unique',ni.composition_id.nunique(),2000);test('Ni_role_counts',sorted(ni.fewshot_role.value_counts().tolist()),[600,1400]);test('sequence_lengths',seq.groupby('replicate').size().tolist(),[32,32,32])
cal=set(ni.loc[ni.fewshot_role.eq('FEWSHOT_CALIBRATION'),'composition_id']);rec=set(ni.loc[ni.fewshot_role.eq('RECOVERY_EVALUATION'),'composition_id']);test('Ni_calibration_eval_overlap',len(cal&rec),0);test('anchors_all_calibration',set(seq.composition_id)<=cal,True)
raw=d(13);summary=d(14);new=raw.groupby(['method','estimator','budget']).mean_tvd.agg(['mean','std']).reset_index().merge(summary,on=['method','estimator','budget']);test('sequence_mean_max_difference',float((new['mean']-new.mean_tvd).abs().max()),0,1e-14);test('sequence_sd_max_difference',float((new['std']-new.sd_tvd).abs().max()),0,1e-14)
test('recovery_budgets',sorted(raw.budget.unique().tolist()),[1,2,4,8,16,32]);test('recovery_state_count',sorted(raw.n_states.unique().tolist()),[15400])
master=d(17);sel=d(18);con=d(20);ph=d(21);pair=d(28)
test('literature_archived',len(master),149);test('literature_eligible',int(sel.direct_eligible.sum()),135);test('literature_selected',int(sel.selected.sum()),120);test('literature_coordinates',int(master.has_composition_coordinates.sum()),145)
grades=con[con.selected.eq(1)].verdict.value_counts();test('four_grades',[int(grades.get(k,0)) for k in ['STRONGLY_CONSISTENT','CONSISTENT','PARTIALLY_CONSISTENT','DISCREPANT']],[61,20,34,5]);test('matched_phase_instances',int(ph.match_status.eq('MATCHED').sum()),248);test('composition_pairs',len(pair),744)
official=float(ph.loc[ph.match_status.eq('MATCHED'),'MAE_at_pct'].mean());serialized=float(pair.absolute_error_at_pct.mean());test('stored_phase_MAE',official,2.1373191915903766,1e-12);test('serialized_pair_MAE',serialized,2.137320474199152,1e-12)
sc=d(22);test('screen_compositions',sc.candidate_id.nunique(),120000)
for k,n in [('C15_pass',57266),('liquid_pass',119917),('distance_pass',108402),('applicable',52944)]:test('screen_'+k,int(sc[k].sum()),n)
test('screen_applicable_states',int(sc.applicable.sum())*4,211776)
pareto_path=R/'08_FINAL_MANUSCRIPT/STEP5K_ORIGINAL_PLAN_GAP_CLOSURE_V1/04_virtual_screening/STEP5K_VIRTUAL_SCREENING_PARETO.csv'
pareto=pd.read_csv(pareto_path);test('actual_Pareto_input_ID_set',set(pareto.candidate_id)==set(sc.loc[sc.applicable,'candidate_id']),True)
cf=d(24);ids=d(23);pairs=d(25);test('confirmed_candidates',cf.candidate_id.nunique(),5);test('confirmation_states',len(cf),20);test('confirmation_temperatures',sorted(cf.Temperature_C.unique().tolist()),[450,600,950,1000]);test('confirmation_component_pairs',len(pairs),400);test('confirmation_ID_link',set(cf.candidate_id)==set(ids.candidate_id),True)
for task,col in [('phase_fraction','TVD_fraction'),('Cr_partition','TVD_Cr'),('Nb_partition','TVD_Nb'),('X_partition','TVD_X')]:
 a=pairs[pairs.task.eq(task)].copy();a['diff']=(a.surrogate-a.thermodynamic_reference).abs();a=a.groupby('state_id')['diff'].sum()/2;z=cf.set_index('state_id')[col];test('confirmation_TVD_'+task,float((a-z).abs().max()),0,1e-12)
path_hits=[]
for m in M:
 p=O/'review/data'/m['file'];rows=pd.read_csv(p);test('rows_'+m['file'],len(rows),m['rows'])
 for col in rows.select_dtypes(include=['object','str']).columns:
  hit=rows[col].astype(str).str.contains(r'(?i)[A-Z]:[\\/]|Users[\\/]',regex=True,na=False)
  if hit.any():path_hits.append(dict(file=m['file'],column=col,rows=int(hit.sum())))
test('public_CSV_local_paths',len(path_hits),0)
# Add actually read evidence records; record capture scope without backdating them.
extra=[R/'04_ML/STEP5A_JMST_ML_FOUNDATION_V2/04_models/TUNING_RESULTS.json',R/'07_FINAL_BLIND/STEP5H_B_NI_BLIND_REVEAL_V1/01_first_target_access/STEP5H_B_FIRST_TARGET_ACCESS.json',R/'07_FINAL_BLIND/STEP5H_B_NI_BLIND_REVEAL_V1/03_processed_targets/STEP5H_B_NI_TARGET_FREEZE.json']
extra.append(pareto_path)
extra_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in extra}
(O/'audit/ADDITIONAL_EVIDENCE_HASHES.json').write_text(json.dumps(extra_hashes,indent=2),encoding='utf-8')
hashchecks=[]
for path,before in H.items():
 p=Path(path);hh=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):hh.update(b)
 hashchecks.append(dict(path=path,before=before,after=hh.hexdigest(),unchanged=hh.hexdigest()==before))
test('protected_source_changes',sum(not x['unchanged'] for x in hashchecks),0)
out={'status':'PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL','checks':checks,'phase_composition_precision':{'stored_full_precision_instance_MAE':official,'serialized_dictionary_pair_MAE':serialized,'absolute_difference':abs(official-serialized)},'local_path_hits':path_hits,'protected_files':len(H),'hashes':hashchecks}
(O/'audit/NUMERICAL_CONSISTENCY_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('Checks:',len(checks),'failures:',[x for x in checks if x['status']=='FAIL'])
# Scientific requirement coverage, with honest source limitations.
coverage=[
('S1.A','Sampling cross counts, bounds, seeds, rounding, stable IDs','Tables S1–S2; Data S1,S5','VERIFIED',''),
('S1.A','Historical scientific origin of all seven perturbation centers','Table S2; Data S5 anchor_source','UNRESOLVED','Generator supplies numerical centers and labels; no publication or selection rationale tied to each center.'),
('S1.B','Convergence and missing continuous targets; zero convenience flags','Fig. S1; Data S1','VERIFIED',''),
('S1.C','Raw-instance mapping, family balance, threshold and argmax','Table S3; Data S2','VERIFIED',''),
('S1.D','Real worked elemental balance','Table S4; Data S3','VERIFIED',''),
('S2.A','43 exact descriptors and 193 candidate decisions','Table S5; Data S4,S30,S31','VERIFIED_WITH_LIMITATION','The bundled Magpie README verifies OQMD provenance and energy/volume units but omits the GSmagmom unit. No magnetic-moment unit is inferred.'),
('S2.B','193→43 filters, priorities and six fold memberships','Data S4; static code index','VERIFIED','No historical selection rerun.'),
('S2.C','Composition splits, baseline exclusion, state counts','Table S6; Data S5–S7','VERIFIED',''),
('S2.D','IID winners and distinct model protocol parameters','Tables S7–S8; Data S26','VERIFIED_WITH_LIMITATION','Explicit saved parameters and inspected constructor calls provided. Matching PyTorch constructor defaults and initialization were separately resolved from version 2.12.0+cu132; other absent settings are not claimed as frozen run metadata.'),
('S3.A','Task, family and temperature results','Table S9; Data S6–S7','VERIFIED',''),
('S3.B','Common representation and six-element Spearman diagnosis','Section S3.1','VERIFIED',''),
('S3.C','Weighted full-vector eta2, all sums of squares','Table S10; Data S8','VERIFIED',''),
('S3.D','Correct/wrong counts and TVD mass, pooled/macro','Table S11; Data S9','VERIFIED','Empty correct group represented as zero count/sum after checking confusion trace.'),
('S3.E','Reference-routed affinity conditional oracle','Table S12; Data S10,S26','VERIFIED','Not a single-factor routing ablation.'),
('S4.A','Internal model, prediction, target and score timeline','Section S4.1; protected freeze/access records','VERIFIED_LOCAL_ONLY','No external preregistration asserted.'),
('S4.B','All Ni IDs, calibration/evaluation split and anchor sequences','Data S11–S12','VERIFIED',''),
('S4.C','Composition-cluster bootstrap and distinct populations','Table S13; Section S4.1','VERIFIED',''),
('S4.D','All budgets/replicates and descriptive SD; strata','Table S14; Data S13–S16','VERIFIED',''),
('S5.A','149/135/120/145 populations and source locators','Data S17–S19,S27','VERIFIED','Four coordinate records remain unresolved and retained.'),
('S5.B','Structure, enrichment/multiplicity, four grades and gates','Tables S15–S16; Data S20–S21','VERIFIED',''),
('S5.C','248 matched instances and 744 component pairs','Data S21,S28','VERIFIED_WITH_PRECISION_NOTE','Full-precision stored MAE differs by 1.28e-6 at.% from rounded serialized composition dictionaries.'),
('S6.A','Actual screening bounds, objectives and directions','Section S6.1; Data S22','VERIFIED',''),
('S6.B','All nine 5NN-definition items and applicability rules','Section S6.2; Data S22','VERIFIED','Reference population 24500 includes baseline; old comment 22050 is inaccurate.'),
('S6.C','Independent pass counts and composition-level intersection','Table S17; Data S22','VERIFIED',''),
('S6.D','Stable Table3 mapping, surrogate/direct objectives and 20 states','Tables S18–S19; Data S23–S25,S29','VERIFIED','Later 88-state expansion not mixed with main-figure twenty states.'),
('Submission','Current supplement-specific format and limit requirements','JOURNAL_REQUIREMENTS_CHECK.md','UNRESOLVED','ScienceDirect guide returned 403; accessible official page does not establish supplement-specific limits.'),
('Authorship','Exact cover author names, order and correspondence copied','Cover; author review notes','AUTHOR_CONFIRMATION','Joined names and QiLai versus Lai Qi retained pending author decision.'),
]
pd.DataFrame(coverage,columns=['requirement','item','output','status','limitation']).to_csv(O/'audit/CONTENT_COVERAGE_MATRIX.csv',index=False)
# Locate source functions and inspectable source lines associated with central claims.
specs=[('S1 sampling','21_fullv1_compositions.py','ANCHOR|SEED|SOBOL|def generate|round'),('S1 balance','23_fullv1_partitioning.py','def |argmax|presence|1e-4'),('S2 feature reduction','35_step5a_feature_selection.py','PRIORITY|STAT_ORDER|def f4|1e-4|PCC_THRESHOLD|EXACT_TOL'),('S2 tuning','38_step5a_tune.py','N_TRIALS|SEED|def suggest|suggest_|top2'),('S2 production','82_g_train.py','def |Adam|seed|patience|MSE|StandardScaler'),('S4 refit','111_i_refit.py','def |Adam|REFIT_|best_mse'),('S6 geometry','131_k_screen_predict.py','StandardScaler|cKDTree|DEV_TRAIN|600|dist.mean|tree.query'),('S6 rules','k_common.py','OOD_|C15|SEED|RANGE|BOUND'),('S6 constraints','132_k_screen_pareto.py','def |C15|liquid|distance|applicable')]
lines=['# Read-only method source locators','These are inspected historical source-code lines, not code executed for this compilation. Full function ranges are in CODE_FUNCTION_INDEX.csv; data provenance is in SOURCE_MAP.csv.']
for label,base,pattern in specs:
 hits=[Path(p) for p in H if Path(p).name==base]
 for p in hits:
  text=p.read_text(encoding='utf-8-sig').splitlines();extract=[f'{i+1}: {t}' for i,t in enumerate(text) if re.search(pattern,t)]
  lines += [f'## {label}',str(p),'```python\n'+'\n'.join(extract)+'\n```']
(O/'audit/METHOD_SOURCE_DETAILS.md').write_text('\n\n'.join(lines),encoding='utf-8')
