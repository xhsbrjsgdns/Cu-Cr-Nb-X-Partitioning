"""Read frozen tables; write descriptive supplementary extracts only. No model loading."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parents[1];D=O/'review/data';D.mkdir(exist_ok=True)
registry={};outputs=[];checks={}
def protect(p):
 p=Path(p);p=p if p.is_absolute() else R/p
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 registry[str(p)]=h.hexdigest();return p
def read(p,**kw):
 p=protect(p)
 return pd.read_parquet(p,**kw) if p.suffix=='.parquet' else pd.read_csv(p,float_precision='round_trip',**kw)
def js(p):return json.loads(protect(p).read_text(encoding='utf-8-sig'))
def emit(df,title,section,sources,rule):
 name=f'Supplementary_Data_S{len(outputs)+1}.csv';df.to_csv(D/name,index=False,encoding='utf-8',na_rep='NA')
 outputs.append({'file':name,'title':title,'section':section,'rows':len(df),'columns':list(df),'sources':sources,'rule':rule})
 return name
F='03_full_CALPHAD/FULL_V1/';A='04_ML/STEP5A_JMST_ML_FOUNDATION_V2/';C='05_OOD/STEP5C_6X_LOEO_V1/';G='07_FINAL_BLIND/STEP5G_PIPELINE_FREEZE_V1/';I='07_FINAL_BLIND/STEP5I_POSTBLIND_FEWSHOT_RECOVERY_V1/';K='08_FINAL_MANUSCRIPT/STEP5K_ORIGINAL_PLAN_GAP_CLOSURE_V1/';X='08_FINAL_MANUSCRIPT/STEP5K_R3_TARGETED_EXTERNAL_VALIDATION/'
c=read(F+'01_compositions/FULL_V1_COMPOSITIONS.parquet');m=read(F+'03_processed/FULL_V1_ML_TARGET_ALL.parquet')
checks['reference']={'compositions':len(c),'states':len(m),'converged':int(m.state_converged.sum()),'failed':int((~m.state_converged.astype(bool)).sum())}
assert list(checks['reference'].values())==[50000,550000,548432,1568]
q=m.groupby(['family','sampling_domain','Temperature_C']).agg(n_compositions=('composition_id','nunique'),nominal_states=('state_id','size'),converged_states=('state_converged','sum')).reset_index();q['failed_states']=q.nominal_states-q.converged_states
emit(q,'Reference-space counts by identity, sampling domain and temperature','S1',[F+'03_processed/FULL_V1_ML_TARGET_ALL.parquet'],'New descriptive count; all failed rows retained.')
cross=c.groupby(['family','sampling_domain']).size().unstack();cross.to_csv(O/'audit/SAMPLING_CROSS_COUNTS.csv')
phase=read(F+'04_qa/FULL_V1_RAW_PHASE_INVENTORY.csv');emit(phase,'Observed raw phase instances and family mapping','S1',[F+'04_qa/FULL_V1_RAW_PHASE_INVENTORY.csv'],'Frozen phase inventory, including all observed Other solid types.')
sid=m.loc[m.state_converged.astype(bool)&m.family.eq('ZR'),'state_id'].iloc[0]
ph=read(F+'02_raw/FULL_V1_PHASES_LONG.parquet',filters=[('state_id','==',sid)]);pc=read(F+'02_raw/FULL_V1_PHASE_COMPOSITIONS_LONG.parquet',filters=[('state_id','==',sid)])
ex=ph[['state_id','raw_phase_name','aggregated_family','phase_moles']].merge(pc[['state_id','raw_phase_name','element','element_mole_fraction']],on=['state_id','raw_phase_name'],validate='one_to_many');ex['element_amount_mol']=ex.phase_moles*ex.element_mole_fraction;ex['family_element_amount_mol']=ex.groupby(['element','aggregated_family']).element_amount_mol.transform('sum');ex['total_element_amount_mol']=ex.groupby('element').element_amount_mol.transform('sum');ex['family_partition_fraction']=ex.family_element_amount_mol/ex.total_element_amount_mol
emit(ex,'Traceable phase-instance partition calculation','S1',[F+'02_raw/FULL_V1_PHASES_LONG.parquet',F+'02_raw/FULL_V1_PHASE_COMPOSITIONS_LONG.parquet'],'New worked example from one actual Zr state. Family amounts repeat per instance; do not sum the repeated family field.')
fd=read(A+'02_features/FEATURE_DICTIONARY.csv');selected=protect(A+'02_features/PRIMARY_FEATURE_LIST.txt').read_text().splitlines();fd['selection_status']='retained';fd['selection_reason']='retained in fixed production schema';fd['kept_correlate']=pd.NA
for file,key,label in [('NEAR_ZERO_VARIANCE_REMOVAL.csv','feature','near_zero_variance'),('EXACT_REDUNDANCY_LOG.csv','dropped','exact_redundancy'),('PCC_REMOVAL_LOG.csv','dropped','correlation')]:
 z=read(A+'02_features/'+file).set_index(key)
 for _,row in fd.iterrows():
  f=row.feature_name
  if f in z.index:
   mask=fd.feature_name.eq(f);fd.loc[mask,'selection_status']=label;fd.loc[mask,'selection_reason']=z.loc[f,'reason']
   if 'kept' in z:fd.loc[mask,'kept_correlate']=z.loc[f,'kept']
assert set(fd.loc[fd.selection_status.eq('retained'),'feature_name'])==set(selected)
for el in ['ZR','TI','CO','SI','AG','MO']:
 fs=protect(C+f'03_features/FOLD_{el}_SELECTED_FEATURES.txt').read_text().splitlines();fd['LOEO_'+el]=fd.feature_name.isin(fs).astype(int)
fd['fixed_index']=fd.feature_name.map({v:f'F{i+1:02d}' for i,v in enumerate(selected)})
emit(fd,'Complete candidate-feature dictionary, selection decisions and fold membership','S2',[A+'02_features/FEATURE_DICTIONARY.csv',A+'02_features/NEAR_ZERO_VARIANCE_REMOVAL.csv',A+'02_features/EXACT_REDUNDANCY_LOG.csv',A+'02_features/PCC_REMOVAL_LOG.csv',C+'03_features/'],'Frozen decisions joined by exact feature name; no rerun of selection.')
split=read(A+'03_splits/COMPOSITION_SPLIT_MASTER_V1.csv');assert split.composition_id.is_unique
emit(c.merge(split[['composition_id','split_role']],on='composition_id',validate='one_to_one'),'Reference compositions and fixed data partitions','S2',[F+'01_compositions/FULL_V1_COMPOSITIONS.parquet',A+'03_splits/COMPOSITION_SPLIT_MASTER_V1.csv'],'Frozen composition values and split assignments; X-free missing fields remain NA.')
metrics=[]
for name in ['PHASE_PRESENCE_METRICS','PHASE_FRACTION_METRICS','CR_PARTITION_METRICS','NB_PARTITION_METRICS','X_PARTITION_METRICS','TEST_ERROR_BY_TEMPERATURE','TEST_ERROR_BY_FAMILY']:
 z=read(A+'05_results/'+name+'.csv');z.insert(0,'source_table',name);z.insert(0,'protocol','IID');metrics.append(z)
emit(pd.concat(metrics,ignore_index=True),'IID task, family and temperature metrics','S3',[A+'05_results/'],'Union of frozen metric tables; NA denotes fields not applicable to a source row.')
metrics=[]
for name in ['LOEO_ALL_FOLDS_METRICS','LOEO_PER_TARGET_METRICS','LOEO_ERROR_BY_TEMPERATURE','LOEO_TOPOLOGY_METRICS']:
 z=read(C+'06_metrics/'+name+'.csv');z.insert(0,'source_table',name);z.insert(0,'protocol','task_specific_LOEO');metrics.append(z)
emit(pd.concat(metrics,ignore_index=True),'Task-specific LOEO metrics','S3',[C+'06_metrics/'],'Distinct protocol from historical direct MLP and oracle.')
eta=read('JMST_FIGURE_TEXT_CORRECTION/data/ETA2_RECOMPUTED.csv');assert np.allclose(eta.SS_total,eta.SS_between+eta.SS_within,rtol=1e-12);assert np.allclose(eta.eta2_vec,eta.SS_between/eta.SS_total,rtol=1e-12)
emit(eta.drop(columns='frozen_eta2'),'Complete-vector sums of squares and element-identity association','S3',['JMST_FIGURE_TEXT_CORRECTION/data/ETA2_RECOMPUTED.csv'],'Source-backed existing full-precision recomputation, checked again for decomposition identity; no componentwise eta-squared averaging.')
err=read('05_OOD/STEP5D_FAILURE_DIAGNOSIS_V1/05_topology_analysis/X_PARTITION_ERROR_BY_TOPOLOGY_CORRECTNESS.csv');err=err[err.scope.eq('heldout_element+topology_status')].copy();emit(err,'Topology-conditioned counts and TVD error mass','S3',['05_OOD/STEP5D_FAILURE_DIAGNOSIS_V1/05_topology_analysis/X_PARTITION_ERROR_BY_TOPOLOGY_CORRECTNESS.csv'],'Pooled share is sum of wrong-state TVD divided by sum of all-state TVD; macro is mean of six element shares.')
checks['topology_wrong_pooled']=err.loc[err.topology_status.eq('TOPOLOGY_WRONG'),'sum_tvd'].sum()/err.sum_tvd.sum()
oracle=read('05_OOD/STEP5E_XPART_REMEDY_V1/07_ablation/STEP5E_ORACLE_TOPOLOGY_DIAGNOSTIC.csv');emit(oracle,'Reference-topology oracle results','S3',['05_OOD/STEP5E_XPART_REMEDY_V1/07_ablation/STEP5E_ORACLE_TOPOLOGY_DIAGNOSTIC.csv'],'Frozen affinity-augmented conditional regressor; not a routing-only ablation of the historical MLP.')
ni=read(G+'04_ni_composition_prereg/STEP5G_NI_BLIND_COMPOSITIONS.csv');emit(ni,'Ni compositions and frozen partition membership','S4',[G+'04_ni_composition_prereg/STEP5G_NI_BLIND_COMPOSITIONS.csv'],'All 2000 Ni composition rows retained.')
seq=read(G+'06_postblind_fewshot_prereg/STEP5G_NESTED_RANDOM_SEQUENCES.csv');emit(seq,'Three ordered Ni anchor sequences','S4',[G+'06_postblind_fewshot_prereg/STEP5G_NESTED_RANDOM_SEQUENCES.csv'],'Nested prefixes of 1, 2, 4, 8, 16 and 32 compositions; each composition supplies 11 states.')
rec=read(I+'05_metrics/STEP5I_RANDOM_FEWSHOT_PER_REPLICATE.csv');emit(rec,'All frozen random-sequence post-blind recovery results','S4',[I+'05_metrics/STEP5I_RANDOM_FEWSHOT_PER_REPLICATE.csv'],'Original per-sequence metrics; model roles retained.')
summary=rec.groupby(['method','estimator','budget']).agg(n_sequences=('replicate','nunique'),mean_tvd=('mean_tvd','mean'),sd_tvd=('mean_tvd','std'),mean_accuracy=('topology_accuracy','mean'),sd_accuracy=('topology_accuracy','std')).reset_index();emit(summary,'Post-blind recovery descriptive means and standard deviations','S4',[I+'05_metrics/STEP5I_RANDOM_FEWSHOT_PER_REPLICATE.csv'],'New descriptive summary; sample standard deviation ddof=1 over three anchor sequences, not a bootstrap interval.')
for fn,title in [('STEP5I_TRUE_TOPOLOGY_STRATIFIED','Recovery metrics stratified by reference topology'),('STEP5I_TEMPERATURE_REGIME_METRICS','Recovery metrics stratified by temperature regime')]:
 p=I+'06_topology_diagnostics/'+fn+'.csv';emit(read(p),title,'S4',[p],'Frozen descriptive strata; no new experiment.')
for p,title in [(X+'02_source_extraction/STEP5K_R3_CANONICAL_MASTER.csv','All archived literature conditions'),(X+'03_inclusion_freeze/CONDITION_SELECTION.csv','Frozen literature-condition inclusion decisions'),(X+'04_composition_conversion/STEP5K_R3_COMPOSITION_CONVERSIONS.csv','Archived composition conversions'),(X+'09_condition_comparison/STEP5K_R3_CONDITION_COMPARISON.csv','Condition-level phase comparison and grades'),(X+'10_phase_composition/STEP5K_R3_PHASE_COMPOSITION_COMPARISON.csv','Matched phase-instance compositions and errors')]:
 emit(read(p),title,'S5',[p],'Frozen records retained; missing values remain NA; populations must not be conflated.')
candidate=read(K+'04_virtual_screening/STEP5K_VIRTUAL_CANDIDATE_COMPOSITIONS.csv');space=read(K+'04_virtual_screening/STEP5K_VIRTUAL_SCREENING_SPACE.parquet');rules=js(K+'04_virtual_screening/STEP5K_SCREENING_RULES_FREEZE.json')
print('SPACE',list(space));print('NI',list(ni));print('SEQUENCE',list(seq));print('RECOVERY',rec.method.unique())
z=candidate.copy().set_index('candidate_id')
for col,t,out in [('frac_C15',450,'C15_450'),('frac_C15',600,'C15_600'),('Cr_P_FCC',950,'Cr_FCC_950'),('Cr_P_FCC',450,'Cr_FCC_450'),('Cr_P_OTHER_SOLID',450,'Cr_OTHER_450')]:z[out]=space[space.Temperature_C.eq(t)].set_index('candidate_id')[col]
z['Delta_Cr_FCC']=z.Cr_FCC_950-z.Cr_FCC_450;z['LIQUID_max']=space.groupby('candidate_id').frac_LIQUID.max();z['ood_knn5_distance']=space.groupby('candidate_id').ood_knn5_distance.first();b=rules['moderate_c15_band'];z['C15_pass']=z.C15_450.between(b['band_low'],b['band_high'])&z.C15_600.between(b['band_low'],b['band_high']);z['liquid_pass']=z.LIQUID_max.le(.01);z['distance_pass']=z.ood_knn5_distance.le(1.5);z['applicable']=z[['C15_pass','liquid_pass','distance_pass']].all(axis=1)
checks['screening']={k:int(z[k].sum()) for k in ['C15_pass','liquid_pass','distance_pass','applicable']};assert checks['screening']['applicable']==52944
emit(z.reset_index(),'All candidate compositions, source objectives and applicability checks','S6',[K+'04_virtual_screening/STEP5K_VIRTUAL_CANDIDATE_COMPOSITIONS.csv',K+'04_virtual_screening/STEP5K_VIRTUAL_SCREENING_SPACE.parquet',K+'04_virtual_screening/STEP5K_SCREENING_RULES_FREEZE.json'],'New read-only composition-level evaluation of frozen rules; all nonpassing candidates retained; no Pareto recomputation.')
ids=read('JMST_FIG2_9_REFERENCE_STYLE/data/TABLE3_VERIFIED_ID_MAP.csv');print('IDS',list(ids))
emit(ids,'Stable Table 3 finalist identifiers','S6',['JMST_FIG2_9_REFERENCE_STYLE/data/TABLE3_VERIFIED_ID_MAP.csv'],'Existing verified mapping; labels are not assigned by CSV row order.')
for p,title in [(K+'05_screening_confirmation/STEP5K_SCREENING_CALPHAD_CONFIRMATION.csv','Frozen finalist thermodynamic confirmation'),(K+'09_qa/producer_outputs_before_delivery_review/06_figures/Fig_K_07_ML_vs_CALPHAD_confirmation_SOURCE.csv','Finalist confirmation component pairs used in main Fig.9')]:emit(read(p),title,'S6',[p],'Surrogate and thermodynamic reference columns remain distinct; no rank-robustness claim.')
for p in (R/'JMST_FIGURE_TEXT_CORRECTION').glob('*.md'):protect(p)
protect(Path(js(O/'audit/MANUSCRIPT_MANIFEST.json')['path']))
(O/'audit/INPUT_HASHES.json').write_text(json.dumps(registry,indent=2),encoding='utf-8');(O/'audit/DATA_MANIFEST.json').write_text(json.dumps(outputs,indent=2),encoding='utf-8');(O/'audit/NUMERICAL_CONSISTENCY_CHECK.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
pd.DataFrame([{'output':x['file'],'section':x['section'],'source':s,'rule':x['rule']} for x in outputs for s in x['sources']]).to_csv(O/'audit/SOURCE_MAP.csv',index=False)
print('EXPORTED',len(outputs),'data files;',len(registry),'hashed inputs')
