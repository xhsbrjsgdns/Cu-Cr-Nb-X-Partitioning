"""Bounded literature-only external equilibrium; no training/model imports or upstream writes."""
import sys,json,hashlib,time,os
sys.dont_write_bytecode=True
from pathlib import Path
import pandas as pd
O=Path('E:/Cu_CALPHAD_ML/08_FINAL_MANUSCRIPT/STEP5K_R2_SOURCE_PROVENANCE_REPAIR')
sys.path.insert(0,str(O/'code_snapshot'))
from match_r2 import wt
P=Path('E:/Cu_CALPHAD_ML/02_pilot_2000/code');sys.path.insert(0,str(P))
from dry_run_lib import build_case,run_single_equilibrium,setup_logging
from fivefamily import family_of_5
import tc_python
from tc_python import TCPython
D=O/'09_optional_new_nonNi_calphad'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
ids=['R2_SI_A5','R2_SI_A7','R2_SI_A15']
all_e=json.loads((O/'04_fulltext_evidence/existing_curated.json').read_text(encoding='utf8'))
evidence=[e for e in all_e if e['validation_id'] in ids]
matches=pd.read_csv(O/'06_fullv1_matching/STEP5K_R2_FULLV1_MATCH.csv').set_index('validation_id')
assert len(evidence)==3 and all(e['usable'] and e['comparison_class']=='DIRECTLY_COMPARABLE' for e in evidence)
assert all(matches.loc[e['validation_id'],'match_class']!='EXACT_MATCH' for e in evidence)
inputs=[]
for e in evidence:
 w=wt(e);assert w and 'NI' not in w and set(w)<= {'CU','CR','NB','ZR','TI','CO','SI','AG','MO'}
 inputs.append(dict(validation_id=e['validation_id'],verified_source_id=e['verified_source_id'],X_identity=e['X_identity'],composition_wt_pct=w,temperature_C=e['temperature_C'],source_page=e['page'],source_table=e['table']))
assert len({json.dumps(x['composition_wt_pct'],sort_keys=True) for x in inputs})==2
assert tc_python.__version__=='2025.2.30'
plan=dict(status='FROZEN_BEFORE_EXECUTION',inputs=inputs,reason='No exact FULL_V1 state; long annealed primary experiments with verified Table 4 composition and basis. A5 tests independent Cu/Nb silicides; A7/A15 compare Cu+Nb5Si3+ternary compound at 600/700 C. This is not Cr3Si validation or target-domain calibration.',database='TCCU6 Cu-Alloys v6.2',tc_python=tc_python.__version__,pressure_bar=1,system_size_mol=1,composition_basis='wt.% Cu balance',max_states=3,max_compositions=2,new_Ni=0,source_pdf_sha256=digest(Path('E:/Cu_CALPHAD_ML/08_FINAL_MANUSCRIPT/STEP5K_R1_OA_LITERATURE_MINOR_PATCHES/02_fulltext_extraction/pdf/SI_Zhou2023.pdf')),matching_rule_sha256=digest(O/'06_fullv1_matching/STEP5K_R2_MATCHING_RULE_FREEZE.md'),helper_hashes={n:digest(P/n) for n in ['dry_run_lib.py','fivefamily.py']})
planpath=D/'execution_plan.json'
if (D/'STATES.csv').exists():raise RuntimeError('Existing execution results; refuse automatic rerun')
planpath.write_text(json.dumps(plan,indent=2),encoding='utf8')
logger=setup_logging(D/'run.log',name='R2_external_only')
states=[];phases=[];comps=[]
os.chdir(D)
with TCPython() as session:
 for q in inputs:
  w=q['composition_wt_pct'];xel=q['X_identity'];cid=q['validation_id'];t=q['temperature_C']
  case=build_case(dict(case_id=cid,family='R2_POSTFREEZE_EXTERNAL',Cu_wt_pct=str(w['CU']),Cr_wt_pct=str(w.get('CR',0)),Nb_wt_pct=str(w.get('NB',0)),X_element=xel,X_wt_pct=str(w[xel]),composition_source=q['verified_source_id'],purpose='Source-verified literature equilibrium only'))
  assert case['composition_sum_ok'] and 'NI' not in case['elements']
  z=run_single_equilibrium(session,case,t,t+273.15,cid+'_EQ',logger)
  z['state'].update(validation_id=cid,verified_source_id=q['verified_source_id'],input_composition_wt_pct=json.dumps(w),source_type='POSTFREEZE_R2_NONNI')
  states.append(z['state'])
  for ph in z['phases']:ph['family_5']=family_of_5(ph['phase_model_base_name'])
  phases.extend(z['phases']);comps.extend(z['compositions'])
  pd.DataFrame(states).to_csv(D/'STATES.csv',index=False)
  pd.DataFrame(phases).to_csv(D/'PHASES.csv',index=False)
  pd.DataFrame(comps).to_csv(D/'PHASE_COMPOSITIONS.csv',index=False)
print('Executed',len(states),'states; converged',sum(s['converged'] for s in states))
