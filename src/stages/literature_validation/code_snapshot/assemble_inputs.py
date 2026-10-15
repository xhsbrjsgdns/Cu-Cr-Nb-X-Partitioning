from pathlib import Path
import json,hashlib,datetime,collections
import pandas as pd
O=Path(__file__).resolve().parents[1]; R=O.parents[1]
AW={'CU':63.546,'CR':51.9961,'NB':92.90637,'AG':107.8682,'MO':95.95,'CO':58.933194,'SI':28.085,'TI':47.867}
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def csv(p,rows):
 rows=[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows]
 pd.DataFrame(rows).to_csv(O/p,index=False,encoding='utf-8-sig')
E=[];S=[];N=[]
for folder in ['ag_mo','nb_co_grcop','si_ti']:
 d=O/'02_source_extraction'/folder
 E+=js(d/'conditions.json');S+=js(d/'sources.json');N+=js(d/'phase_structure_notes.json')
assert len(S)==6 and len({e['condition_id'] for e in E})==len(E)
for s in S:
 assert sha(s['pdf_path'])==s['sha256'] and s['identity_verified']
 s['doi']=s.get('doi',s.get('DOI'))
 s['source_role']='QUALITATIVE_CONTEXT' if s['source_id']=='S6_GRCOP' else 'DIRECT_PHASE_EQUILIBRIUM'
 s['experimental_system']='; '.join(s.get('systems',[]))
 if isinstance(s['authors'],list):s['authors']='; '.join(s['authors'])
csv('01_source_pdfs/STEP5K_R3_SOURCE_MANIFEST.csv',S)
conv=[];domains=[];master=[];meas=[]
config=js(R/'03_full_CALPHAD/FULL_V1/06_manifest/FULL_V1_CONFIG.json')['sampling']
for e in E:
 e['system']=e['system'].replace('–','-'); c=e.get('composition') or {}; x=e['X_identity']; valid=bool(c) and set(c)<=set(AW) and all(isinstance(a,(int,float)) and a>=0 for a in c.values()) and abs(sum(c.values())-100)<1e-8
 w={}
 if valid:
  if e['composition_basis']=='at.%':w={k:100*v*AW[k]/sum(a*AW[z] for z,a in c.items()) for k,v in c.items()}
  elif e['composition_basis']=='wt.%':w=c.copy()
  else:valid=False
 if e['direct_eligible']:assert valid and e.get('page') and e.get('phases') and isinstance(e.get('temperature_K'),(int,float)),e['condition_id']
 if valid:assert abs(sum(w.values())-100)<1e-8
 e['composition_wt_pct']=w;e['conversion_valid']=valid
 row=dict(condition_id=e['condition_id'],original_basis=e['composition_basis'],original_composition=e['nominal_composition_original'],atomic_weight_Cu=AW['CU'],atomic_weight_Cr=AW['CR'],atomic_weight_Nb=AW['NB'],atomic_weight_X=AW.get(x),converted_wt_Cu=w.get('CU'),converted_wt_Cr=w.get('CR'),converted_wt_Nb=w.get('NB',0) if valid else None,converted_wt_X=w.get(x,0) if valid else None,sum_wt=sum(w.values()) if valid else None,conversion_formula='w_i=100*x_i*M_i/sum(x_j*M_j)' if e['composition_basis']=='at.%' else 'original wt.% unchanged',conversion_check='PASS' if valid else 'NOT_CONVERTIBLE',converted_composition_wt_pct=w)
 conv.append(row)
 s=w.get('CR',0)+w.get('NB',0); ratio=(w.get('CR',0)/AW['CR'])/(w['NB']/AW['NB']) if w.get('NB',0)>0 else None
 bounds=[]
 for coord in ['S_CrNb','R_CrNb_atomic']:
  bounds.append([min(config[d][coord][0] for d in ['domain_A','domain_B']),max(config[d][coord][1] for d in ['domain_A','domain_B'])])
 vals=[s,ratio]
 if x!='NB_BASE':
  if x in config['domain_A']['X_wt_pct']:
   bounds.append([min(config[d]['X_wt_pct'][x][0] for d in ['domain_A','domain_B']),max(config[d]['X_wt_pct'][x][1] for d in ['domain_A','domain_B'])]);vals.append(w.get(x,0))
 within=valid and all(v is not None and lo<=v<=hi for v,(lo,hi) in zip(vals,bounds))
 near=valid and all(v is not None and lo-.1*(hi-lo)<=v<=hi+.1*(hi-lo) for v,(lo,hi) in zip(vals,bounds))
 relation='QUALITATIVE_PROCESSED_ALLOY' if e['source_id']=='S6_GRCOP' else ('BASELINE_CuCrNb' if x=='NB_BASE' else 'ADJACENT_TERNARY_CuCrX')
 domain='WITHIN_FULLV1_BOUNDS' if within else ('NEAR_FULLV1_BOUNDS' if near else 'OUTSIDE_FULLV1_BOUNDS')
 domains.append(dict(condition_id=e['condition_id'],system=e['system'],TARGET_DOMAIN_RELEVANCE=domain,system_relation_class=relation,S_CrNb_wt_pct=s if valid else None,R_CrNb_atomic=ratio,X_wt_pct=w.get(x,0) if valid else None,bounds=bounds,notes='Undefined ratio and unconvertible composition cannot be within bounds. No nearest neighbor comparison.'))
 e.update(TARGET_DOMAIN_RELEVANCE=domain,system_relation_class=relation)
 m=e.copy()
 for k,el in [('Cu_original','CU'),('Cr_original','CR'),('Nb_original','NB'),('X_original',x)]:m[k]=c.get(el,0) if valid else c.get(el)
 for i in range(4):m['phase_'+str(i+1)]=e['phases'][i] if i<len(e['phases']) else ''
 m['measured_phase_compositions_available']=bool(e['measured_phases']);master.append(m)
 for i,p in enumerate(e['measured_phases']):
  raw=p['composition'];q=raw.copy();balance=False
  if not q or any(not isinstance(a,(int,float)) for a in q.values()):
   meas.append(dict(condition_id=e['condition_id'],source_id=e['source_id'],phase_index=i+1,literature_phase=p['phase'],original_basis=p['basis'],original_values=raw,comparison_at_pct={},page=e['page'],table=e['table'],notes='Unreported/non-numeric phase values retained; no conversion or inferred endpoint.'));continue
  if 'CU' not in q and p['basis']=='wt.%' and 0<=sum(q.values())<=100:q['CU']=100-sum(q.values());balance=True
  if p['basis']=='wt.%':at={k:100*a/AW[k]/sum(b/AW[z] for z,b in q.items()) for k,a in q.items()}
  else:at=q.copy()
  meas.append(dict(condition_id=e['condition_id'],source_id=e['source_id'],phase_index=i+1,literature_phase=p['phase'],original_basis=p['basis'],original_values=raw,Cu_balance_derived=balance,comparison_at_pct=at,original_sum=sum(raw.values()),comparison_sum=sum(at.values()),page=e['page'],table=e['table'],uncertainty=p.get('reported_uncertainty'),notes=p.get('composition_note','')))
csv('04_composition_conversion/STEP5K_R3_COMPOSITION_CONVERSIONS.csv',conv)
csv('04_composition_conversion/STEP5K_R3_DOMAIN_RELEVANCE.csv',domains)
csv('02_source_extraction/STEP5K_R3_EXPERIMENTAL_CONDITIONS_MASTER.csv',master)
csv('10_phase_composition/EXPERIMENTAL_PHASE_COMPOSITIONS.csv',meas)
dump(O/'02_source_extraction/ALL_CONDITIONS.json',E);dump(O/'02_source_extraction/ALL_SOURCES.json',S);dump(O/'05_phase_crosswalk/EXPERIMENTAL_STRUCTURE_NOTES.json',N)
groups=collections.defaultdict(list)
for e in E:
 if e['direct_eligible']:groups[(e['source_id'],e['temperature_K'],';'.join(sorted(e['phases'])))].append(e)
for g in groups.values():g.sort(key=lambda x:x['condition_id'])
selected=[];depth=0
while len(selected)<120:
 batch=[groups[k][depth] for k in sorted(groups) if len(groups[k])>depth]
 if not batch:break
 selected.extend(batch[:120-len(selected)]);depth+=1
req=[];selectedids={e['condition_id'] for e in selected}
for e in selected:
 w=e['composition_wt_pct'];assert 'NI' not in w
 req.append(dict(condition_id=e['condition_id'],state_id=e['condition_id']+'_EQ',source_id=e['source_id'],system=e['system'],X_identity=e['X_identity'],composition_wt_pct=w,temperature_C=e['temperature_C'],temperature_K=e['temperature_K'],pressure_Pa=100000,system_size_mol=1,database='TCCU6',request_status='READY_NOT_EXECUTED',page=e['page'],table=e['table']))
csv('06_calphad_requests/STEP5K_R3_CALPHAD_REQUESTS.csv',req)
csv('03_inclusion_freeze/CONDITION_SELECTION.csv',[dict(condition_id=e['condition_id'],direct_eligible=e['direct_eligible'],selected=e['condition_id'] in selectedids,selection_reason='SELECTED_STRATIFIED' if e['condition_id'] in selectedids else ('BUDGET_STRATIFIED_NOT_SELECTED' if e['direct_eligible'] else e['exclusion_reason'])) for e in E])
seal=O/'03_inclusion_freeze/INPUT_FREEZE.json'
assert not (O/'07_calphad_raw/ATTEMPTS.jsonl').exists(),'Cannot regenerate input after calculations'
dump(seal,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_count=len(S),condition_count=len(E),eligible_count=sum(e['direct_eligible'] for e in E),selected_count=len(req),state_budget=120,request_sha256=sha(O/'06_calphad_requests/STEP5K_R3_CALPHAD_REQUESTS.csv'),rules_sha256=sha(O/'03_inclusion_freeze/STEP5K_R3_CONDITION_INCLUSION_FREEZE.md'),condition_sha256=sha(O/'02_source_extraction/ALL_CONDITIONS.json'),sources={s['source_id']:s['sha256'] for s in S}))
print(json.dumps(dict(sources=len(S),conditions=len(E),eligible=sum(e['direct_eligible'] for e in E),selected=len(req),selection_by_system=dict(collections.Counter(e['system'] for e in selected)))))
