from pathlib import Path
import json,hashlib,ast
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parents[1];D=O/'review/data'
registry=json.loads((O/'audit/INPUT_HASHES.json').read_text());out=json.loads((O/'audit/DATA_MANIFEST.json').read_text());facts={}
def protect(p):
 p=Path(p);p=p if p.is_absolute() else R/p;registry[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def js(p):return json.loads(protect(p).read_text(encoding='utf-8-sig'))
def csv(p):return pd.read_csv(protect(p),float_precision='round_trip')
def emit(df,title,section,sources,rule):
 n=f'Supplementary_Data_S{len(out)+1}.csv';df.to_csv(D/n,index=False,na_rep='NA');out.append({'file':n,'title':title,'section':section,'rows':len(df),'columns':list(df),'sources':sources,'rule':rule});return n
A='04_ML/STEP5A_JMST_ML_FOUNDATION_V2/';C='05_OOD/STEP5C_6X_LOEO_V1/';G='07_FINAL_BLIND/STEP5G_PIPELINE_FREEZE_V1/';B='07_FINAL_BLIND/STEP5H_B_NI_BLIND_REVEAL_V1/';I='07_FINAL_BLIND/STEP5I_POSTBLIND_FEWSHOT_RECOVERY_V1/';K='08_FINAL_MANUSCRIPT/STEP5K_ORIGINAL_PLAN_GAP_CLOSURE_V1/';X='08_FINAL_MANUSCRIPT/STEP5K_R3_TARGETED_EXTERNAL_VALIDATION/'
facts['full_config']=js('03_full_CALPHAD/FULL_V1/06_manifest/FULL_V1_CONFIG.json');facts['iid']=js(A+'09_manifest/DEVELOPMENT_FREEZE_V1.json');facts['ni_design']=js(G+'04_ni_composition_prereg/STEP5G_NI_BLIND_PREREGISTRATION.json');facts['production']=js(G+'01_pipeline_freeze/STEP5G_FINAL_PIPELINE_FREEZE.json');facts['ni_primary']=js(B+'06_primary_score/STEP5H_B_PRIMARY_BLIND_METRICS.json');facts['ni_secondary']=js(B+'07_secondary_score/STEP5H_B_SECONDARY_BLIND_METRICS.json');facts['recovery_baselines']=js(I+'05_metrics/STEP5I_RECOVERY_BASELINES.json');facts['screen_rules']=js(K+'04_virtual_screening/STEP5K_SCREENING_RULES_FREEZE.json');facts['screen_qa']=js(K+'04_virtual_screening/STEP5K_SCREENING_PREDICTION_QA.json')
params=[]
def flatten(obj,prefix=''):
 if isinstance(obj,dict):
  for k,v in obj.items():yield from flatten(v,prefix+'.'+k if prefix else k)
 else:yield prefix,json.dumps(obj,ensure_ascii=False) if isinstance(obj,(list,bool)) or obj is None else str(obj)
def add_param(protocol,fold,task,x,source):
 for k,v in flatten(x):
  if any(s in k.lower() for s in ['path','artifact','sha256','checkpoint','scaler','train_ids','validation_ids','software','elapsed','seconds']):continue
  if ':\\' in v or 'E:/' in v:continue
  params.append({'protocol':protocol,'heldout_element':fold,'task':task,'parameter':k,'value':v,'source_record':Path(source).name})
for task,x in facts['iid']['tasks'].items():add_param('IID','NA',task,x,A+'09_manifest/DEVELOPMENT_FREEZE_V1.json')
facts['loeo']={};facts['historical_mlp']={};facts['oracle']={}
for el in ['ZR','TI','CO','SI','AG','MO']:
 p=C+f'11_manifest/FOLD_{el}_FREEZE.json';x=js(p);facts['loeo'][el]=x
 for task,v in x['tasks'].items():add_param('task_specific_LOEO',el,task,{**v,'random_seed':x['random_seed'],'n_inputs':x['n_selected_features']},p)
 p=f'05_OOD/STEP5D_FAILURE_DIAGNOSIS_V1/07_model_bias_diagnostic/FOLD_{el}/SELECTION_FREEZE.json';x=js(p);facts['historical_mlp'][el]=x;add_param('historical_direct_MLP',el,'X partition',{'mlp':x['mlp'],'seed':x['seed'],'n_features':len(x['features']),'n_train':x['train_rows'],'n_validation':x['validation_rows']},p)
 p=f'05_OOD/STEP5E_XPART_REMEDY_V1/11_manifest/FOLD_{el}_REMEDY_FREEZE.json';x=js(p);facts['oracle'][el]=x;add_param('affinity_conditional_oracle',el,'X partition',{'parameters':x['xgb_hyperparameters'],'seed':x['seed'],'n_original_features':len(x['features_original']),'affinity_features':x['affinity_columns'],'branches':x['branches']},p)
add_param('Ni_production','NA','X partition',facts['production'].get('models',{}),G+'01_pipeline_freeze/STEP5G_FINAL_PIPELINE_FREEZE.json')
emit(pd.DataFrame(params),'Frozen model configurations by protocol and fold','S2',[A+'09_manifest/DEVELOPMENT_FREEZE_V1.json',C+'11_manifest/',G+'01_pipeline_freeze/STEP5G_FINAL_PIPELINE_FREEZE.json','05_OOD/STEP5D_FAILURE_DIAGNOSIS_V1/07_model_bias_diagnostic/','05_OOD/STEP5E_XPART_REMEDY_V1/11_manifest/'],'Explicit saved parameters only; absent implementation defaults are not inferred.')
src=js(X+'02_source_extraction/ALL_SOURCES.json');facts['literature_sources']=src
sources=pd.DataFrame([{k:x.get(k,x.get('DOI') if k=='doi' else '') for k in ['source_id','title','authors','journal','year','doi','source_role']} for x in src]);emit(sources,'Literature source bibliography and source identifiers','S5',[X+'02_source_extraction/ALL_SOURCES.json'],'Original experimental publications; no licensed full texts included.')
master=pd.read_csv(D/'Supplementary_Data_S17.csv',float_precision='round_trip');selection=pd.read_csv(D/'Supplementary_Data_S18.csv');conversion=pd.read_csv(D/'Supplementary_Data_S19.csv',float_precision='round_trip');master=master.merge(selection[['condition_id','selected','selection_reason']],on='condition_id',validate='one_to_one').merge(sources[['source_id','doi','title']],on='source_id',validate='many_to_one')
master['has_composition_coordinates']=master.condition_id.isin(conversion.loc[conversion[['converted_wt_Cu','converted_wt_Cr','converted_wt_Nb','converted_wt_X']].notna().all(axis=1),'condition_id']);master.to_csv(D/'Supplementary_Data_S17.csv',index=False,na_rep='NA');out[16]['columns']=list(master)
comp=pd.read_csv(D/'Supplementary_Data_S21.csv',float_precision='round_trip');rows=[]
for _,r in comp.iterrows():
 if str(r.match_status)!='MATCHED':continue
 a=json.loads(r.experimental_at_pct);b=json.loads(r.calculated_at_pct)
 for el,v in a.items():rows.append({'condition_id':r.condition_id,'source_id':r.source_id,'phase_index':r.phase_index,'literature_phase':r.literature_phase,'matched_calculated_phase':r.matched_calculated_phase,'element':el,'reference_experiment_at_pct':v,'calculated_at_pct':b[el],'absolute_error_at_pct':abs(v-b[el])})
pairs=pd.DataFrame(rows);assert len(pairs)==744;emit(pairs,'All matched experimental and calculated elemental composition pairs','S5',[X+'10_phase_composition/STEP5K_R3_PHASE_COMPOSITION_COMPARISON.csv'],'Expanded full-precision frozen dictionaries: 248 matched phase instances × three elements.')
ids=pd.read_csv(D/'Supplementary_Data_S23.csv');confirm=pd.read_csv(D/'Supplementary_Data_S25.csv',float_precision='round_trip');confirm=confirm.merge(ids[['candidate_id','table3_label']],on='candidate_id',validate='many_to_one');assert len(confirm)==20
# Retain the exact main-figure 20-state confirmation population, not the later 88-state expanded table.
confirm.to_csv(D/'Supplementary_Data_S24.csv',index=False,na_rep='NA');out[23].update(rows=20,columns=list(confirm),title='Twenty global-finalist confirmation states',sources=out[24]['sources'],rule='Main-figure frozen population selected by stable candidate IDs; distinct from the later 88-state expanded confirmation file.')
long=[]
for _,r in confirm.iterrows():
 for task,key in [('phase_fraction','frac_'),('Cr_partition','Cr_P_'),('Nb_partition','Nb_P_'),('X_partition','X_P_')]:
  for fam in ['FCC','BCC','C15','OTHER_SOLID','LIQUID']:long.append({'candidate_id':r.candidate_id,'table3_label':r.table3_label,'state_id':r.state_id,'Temperature_C':r.Temperature_C,'task':task,'phase_family':fam,'surrogate':r['pred_'+key+fam],'thermodynamic_reference':r['calphad_'+key+fam]})
long=pd.DataFrame(long);long.to_csv(D/'Supplementary_Data_S25.csv',index=False,na_rep='NA');out[24].update(rows=len(long),columns=list(long))
screen=pd.read_csv(D/'Supplementary_Data_S22.csv',float_precision='round_trip');final=ids.merge(screen,on=['candidate_id','Cr_wt_pct','Nb_wt_pct','X_wt_pct','Cu_wt_pct'],validate='one_to_one')
for temp,col,key in [(950,'calphad_Cr_P_FCC','confirmed_Cr_FCC_950'),(450,'calphad_Cr_P_FCC','confirmed_Cr_FCC_450'),(450,'calphad_Cr_P_OTHER_SOLID','confirmed_Cr_OTHER_450')]:final[key]=final.candidate_id.map(confirm[confirm.Temperature_C.eq(temp)].set_index('candidate_id')[col])
final['confirmed_Delta_Cr_FCC']=final.confirmed_Cr_FCC_950-final.confirmed_Cr_FCC_450;emit(final,'Finalist screening and direct-confirmation objectives','S6',out[21]['sources']+out[24]['sources'],'New stable-ID join; raw screening and confirmed objective fields kept separately; ranking is not recomputed.')
facts['ni_split_counts']=pd.read_csv(D/'Supplementary_Data_S11.csv').fewshot_role.value_counts().to_dict();facts['external_counts']={'archived':len(master),'eligible':int(selection.direct_eligible.sum()),'selected':int(selection.selected.sum()),'coordinates':int(master.has_composition_coordinates.sum()),'matched_phases':int(comp.match_status.eq('MATCHED').sum()),'component_pairs':len(pairs),'MAE':float(pairs.absolute_error_at_pct.mean())}
facts['sources_checked']={}
for folder in ['03_full_CALPHAD/FULL_V1/code_snapshot',A+'code_snapshot',C+'code_snapshot','05_OOD/STEP5D_FAILURE_DIAGNOSIS_V1/code_snapshot','05_OOD/STEP5E_XPART_REMEDY_V1/code_snapshot',G+'code_snapshot',B+'code_snapshot',I+'code_snapshot',K+'code_snapshot',X+'code_snapshot']:
 for p in (R/folder).glob('*.py'):protect(p)
for p in [A+'01_feature_sources/ELEMENT_DESCRIPTOR_PROVENANCE.csv',A+'01_feature_sources/ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.csv']:
 q=csv(p);emit(q,'Elemental descriptor '+('provenance' if 'PROVENANCE' in p else 'values'),'S2',[p],'Frozen source table; user authorized disclosure of verified software and database identifiers.')
(O/'audit/FACTS.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8');(O/'audit/INPUT_HASHES.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8');(O/'audit/DATA_MANIFEST.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(facts['external_counts']);print('DATA FILES',len(out),'HASHED',len(registry))
