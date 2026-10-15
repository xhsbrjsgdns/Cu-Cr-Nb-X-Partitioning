"""Final descriptive joins and metadata, after assemble_data.py then extend_data.py.

No frozen script is imported. Only constant expressions and the pure feature-name
sort key are extracted from the historical selector's AST.
"""
from pathlib import Path
import ast,json,hashlib
import pandas as pd
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parents[1];D=O/'review/data'
M=json.loads((O/'audit/DATA_MANIFEST.json').read_text(encoding='utf-8'))
assert len(M)==31, 'Run the 25-file foundation and six-file extension first.'
p=R/'04_ML/STEP5A_JMST_ML_FOUNDATION_V2/code_snapshot/35_step5a_feature_selection.py'
ns={}
for node in ast.parse(p.read_text(encoding='utf-8')).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['PRIORITY','STAT_ORDER'] for t in node.targets):
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
 if isinstance(node,ast.FunctionDef) and node.name=='f4_key':
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
df=pd.read_csv(D/'Supplementary_Data_S4.csv',float_precision='round_trip')
cols=df.loc[~df.selection_reason.isin(['constant','near-zero-variance']),'feature_name'].tolist()
priority={c:i for i,c in enumerate(ns['PRIORITY']) if c in cols}
for i,c in enumerate(sorted([c for c in cols if c not in priority],key=ns['f4_key'])):priority[c]=len(ns['PRIORITY'])+i
df['fixed_priority_rank']=df.feature_name.map(priority).astype('Int64');df.to_csv(D/'Supplementary_Data_S4.csv',index=False,na_rep='NA');M[3]['columns']=list(df)
M[3]['rule']='Frozen decisions joined by exact feature name; no rerun of selection. Priority ranks reproduced from constant-list and pure name-key expressions only; no feature filtering rerun.'
pareto='08_FINAL_MANUSCRIPT/STEP5K_ORIGINAL_PLAN_GAP_CLOSURE_V1/04_virtual_screening/STEP5K_VIRTUAL_SCREENING_PARETO.csv'
pa=pd.read_csv(R/pareto,float_precision='round_trip');sc=pd.read_csv(D/'Supplementary_Data_S22.csv',float_precision='round_trip')
rankcols=['pareto_rank','family_pareto_rank','diversity_rank','selected_finalist']
sc=sc.drop(columns=rankcols,errors='ignore').merge(pa[['candidate_id']+rankcols],on='candidate_id',how='left',validate='one_to_one')
assert set(sc.loc[sc.applicable,'candidate_id'])==set(pa.candidate_id)
sc.to_csv(D/'Supplementary_Data_S22.csv',index=False,na_rep='NA');M[21]['columns']=list(sc)
if pareto not in M[21]['sources']:M[21]['sources'].append(pareto)
M[21]['rule']='New read-only composition-level evaluation of frozen rules; all nonpassing candidates retained; no Pareto recomputation. Saved global, within-family and diversity ranks are joined by candidate_id; nonapplicable rows have NA ranks.'
params=pd.read_csv(D/'Supplementary_Data_S26.csv',keep_default_na=False)
tunepath='04_ML/STEP5A_JMST_ML_FOUNDATION_V2/04_models/TUNING_RESULTS.json';tune=json.loads((R/tunepath).read_text(encoding='utf-8'));new=[]
for key,value in [('tuning_seed',tune['seed']),('trials_per_screened_top2_family',tune['n_trials'])]:new.append(dict(protocol='IID_tuning',heldout_element='NA',task='all',parameter=key,value=str(value),source_record='TUNING_RESULTS.json'))
for key,value,source in [('adam.betas','[0.9, 0.999]','torch/optim/adam.py'),('adam.eps','1e-8','torch/optim/adam.py'),('adam.weight_decay','0','torch/optim/adam.py'),('adam.amsgrad','false','torch/optim/adam.py'),('linear.initialization','Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases','torch/nn/modules/linear.py')]:new.append(dict(protocol='MLP_implementation_default',heldout_element='NA',task='X partition',parameter=key,value=value,source_record='PyTorch 2.12.0+cu132 '+source))
params=pd.concat([params,pd.DataFrame(new)],ignore_index=True).drop_duplicates();params.to_csv(D/'Supplementary_Data_S26.csv',index=False,na_rep='NA')
M[25]['rows']=len(params);M[25]['rule']='Saved configurations plus explicitly labeled defaults inspected in matching PyTorch 2.12.0+cu132 source; no absent defaults are guessed.'
if tunepath not in M[25]['sources']:M[25]['sources'].append(tunepath)
M[27]['rule']='New expansion of frozen serialized dictionaries at their stored precision; historical full-precision instance MAE is retained in Data S21.'
M[30]['title']='Development-element descriptor values: Cu, Cr, Nb and six X identities'
(O/'audit/DATA_MANIFEST.json').write_text(json.dumps(M,ensure_ascii=False,indent=2),encoding='utf-8')
print('Finalized 31 frozen-data extracts; no fit, prediction or thermodynamic computation.')
