from pathlib import Path
import json,numpy as np,pandas as pd,hashlib
ROOT=Path('E:/Cu_CALPHAD_ML');O=ROOT/'08_FINAL_MANUSCRIPT/STEP5K_R2_SOURCE_PROVENANCE_REPAIR'
AW={'CU':63.546,'CR':51.9961,'NB':92.90637,'ZR':91.224,'TI':47.867,'CO':58.933194,'SI':28.085,'AG':107.8682,'MO':95.95}
def wt(e):
 c=e['nominal_composition']
 if not isinstance(c,dict) or not e.get('composition_valid',True) or e['composition_basis'] not in ['at.%','wt.%']:return None
 if abs(sum(c.values())-100)>1e-6 or set(c)-set(AW):return None
 if e['composition_basis']=='at.%':
  mass={k:v*AW[k] for k,v in c.items()};c={k:100*v/sum(mass.values()) for k,v in mass.items()}
 # Cu is balance after the independent solutes are rounded, as in the frozen input.
 c={k:round(v,4) for k,v in c.items() if k!='CU'};c['CU']=round(100-sum(c.values()),4)
 return c
def run(evidence):
 c=pd.read_csv(ROOT/'03_full_CALPHAD/FULL_V1/01_compositions/FULL_V1_COMPOSITIONS.csv',low_memory=False)
 s=pd.read_csv(ROOT/'03_full_CALPHAD/FULL_V1/02_raw/FULL_V1_EQUILIBRIUM_STATES.csv',usecols=['state_id','case_id','Temperature_C','converged'])
 geo=json.loads((O/'00_config/matching_geometry.json').read_text());rows=[];conv=[]
 for e in evidence:
  w=wt(e);x=e['X_identity'];t=e['temperature_C'];r={k:'NA' for k in ['closest_FULLV1_id','closest_domain','distance_S','distance_R','distance_X','normalized_distance']}
  r.update(validation_id=e['validation_id'],verified_source_id=e['verified_source_id'],literature_composition=json.dumps(e['nominal_composition'],ensure_ascii=False),literature_temperature_C=t,match_class='NO_MATCH',exact_composition=False,exact_temperature=False,eligible_for_direct_validation=False,eligible_for_qualitative_context=False,notes='Unknown/ambiguous chemistry or basis; no finite match attempted.')
  if w and e['composition_basis']=='at.%':
   orig=e['nominal_composition'];den=sum(orig[k]*AW[k] for k in orig)
   conv.append(dict(validation_id=e['validation_id'],verified_source_id=e['verified_source_id'],original_at_pct=json.dumps(orig),atomic_weights_g_mol=json.dumps({k:AW[k] for k in orig}),formula='w_i=100*a_i*M_i/sum_j(a_j*M_j); round solutes to 4 decimals; Cu balance',converted_wt_pct_full_precision=json.dumps({k:100*v*AW[k]/den for k,v in orig.items()}),thermocalc_input_wt_pct=json.dumps(w),source='Fixed standard atomic weights; IUPAC conventional values https://ciaaw.org/atomic-weights.htm',evidence_location=e['page']+' Table '+e['table']))
  if w and x in geo and isinstance(t,(int,float)):
   g=c[c.X_element==x];wc=['Cu_wt_pct','Cr_wt_pct','Nb_wt_pct','X_wt_pct'];a=g[wc].to_numpy(float);v=np.array([w.get(k,0) for k in ['CU','CR','NB',x]])
   exact=np.all(np.abs(a-v)<.00000001,axis=1)
   candidates=g[exact];r['exact_composition']=bool(len(candidates))
   T=s[(s.case_id.isin(candidates.composition_id)) & ((s.Temperature_C-t).abs()<1e-8)]
   temp_exists=bool(((s[s.X_element==x].Temperature_C-t).abs()<1e-8).any()) if 'X_element' in s else bool(((s.Temperature_C-t).abs()<1e-8).any())
   r['exact_temperature']=temp_exists
   if len(T):
    good=T[T.converged.astype(str).str.lower()=='true'];z=(good if len(good) else T).iloc[0];cl=g[g.composition_id==z.case_id].iloc[0]
    r.update(closest_FULLV1_id=z.state_id,closest_domain=cl.sampling_domain,distance_S=0,distance_R=0,distance_X=0,normalized_distance=0,match_class='EXACT_MATCH',eligible_for_direct_validation=bool(len(good) and e.get('usable',False)),notes='Exact composition and temperature; convergence required for eligibility.')
   else:
    # Elemental nearest is a diagnostic only for undefined R.
    idx=np.argmin(np.linalg.norm((a-v)/np.where(np.ptp(a,axis=0)>0,np.ptp(a,axis=0),1),axis=1));cl=g.iloc[idx]
    if w.get('NB',0)>0:
     point=np.array([w.get('CR',0)+w['NB'],(w.get('CR',0)/AW['CR'])/(w['NB']/AW['NB']),w.get(x,0)])
     A=g[['S_CrNb_wt_pct','R_CrNb_atomic_actual','X_wt_pct']].to_numpy(float);span=np.array(geo[x]['range']);dist=np.linalg.norm((A-point)/span,axis=1);ix=dist.argmin();cl=g.iloc[ix];delta=point-A[ix]
     inside=np.all(point>=np.array(geo[x]['min'])) and np.all(point<=np.array(geo[x]['min'])+span)
     tight=bool(inside and dist[ix]<=geo[x]['tight_threshold'] and temp_exists)
     r.update(distance_S=delta[0],distance_R=delta[1],distance_X=delta[2],normalized_distance=dist[ix],match_class='TIGHT_NEIGHBOR' if tight else 'DISTANT_NEIGHBOR',eligible_for_qualitative_context=tight and e.get('usable',False),notes='Finite S/R/X nearest; outside sampling bounds is never tight. No distant neighbor is validation evidence.')
    else:r['notes']='Nb=0: atomic Cr/Nb ratio undefined; no clamping. Closest ID below is elemental-wt.% diagnostic only and is not evidence.'
    r['closest_FULLV1_id']=cl.composition_id;r['closest_domain']=cl.sampling_domain
  rows.append(r)
 pd.DataFrame(rows).to_csv(O/'06_fullv1_matching/STEP5K_R2_FULLV1_MATCH.csv',index=False)
 pd.DataFrame(conv).to_csv(O/'05_composition_conversion/STEP5K_R2_COMPOSITION_CONVERSIONS.csv',index=False)
 return rows
if __name__=='__main__':
 es=json.loads((O/'04_fulltext_evidence/existing_curated.json').read_text(encoding='utf8'))
 combined=O/'04_fulltext_evidence/all_curated.json'
 if combined.exists():es=json.loads(combined.read_text(encoding='utf8'))
 rr=run(es);print(pd.Series([r['match_class'] for r in rr]).value_counts().to_dict())
