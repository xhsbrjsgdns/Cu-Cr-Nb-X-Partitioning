from pathlib import Path
import json,hashlib,re,datetime,sys
sys.dont_write_bytecode=True
import pandas as pd,numpy as np
O=Path('E:/Cu_CALPHAD_ML/08_FINAL_MANUSCRIPT/STEP5K_R2_SOURCE_PROVENANCE_REPAIR');R=O.parent/'STEP5K_R1_OA_LITERATURE_MINOR_PATCHES'
sys.path.insert(0,str(O/'code_snapshot'));from match_r2 import wt,run as match
def js(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def cell(v):return json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else ('NA' if v is None else v)
def csv(path,rows,cols=None):
 df=pd.DataFrame([{k:cell(v) for k,v in r.items()} for r in rows])
 if cols:
  for col in cols:
   if col not in df:df[col]='NA'
  df=df[cols+[x for x in df if x not in cols]]
 df.fillna('NA').to_csv(O/path,index=False,encoding='utf-8-sig');return df
S=js(O/'02_pdf_provenance/all_verified_sources.json');E=js(O/'04_fulltext_evidence/all_curated.json');N=js(O/'03_new_literature_search/candidates_novelty_audited.json');sm={s['verified_source_id']:s for s in S}
sm['R2_ZR_Hughes2022']['pdf_url']='https://pure-oai.bham.ac.uk/ws/files/215681972/HughesJ2022Full-stage.pdf'
sm['R2_Raabe2000_CuCrAg']['verification_level']='C' # Exact title+authors+year in saved Crossref response; paper only prints PII.
dump(O/'02_pdf_provenance/all_verified_sources.json',S)
matches=match(E);mm={m['validation_id']:m for m in matches}
# One bibliographic mapping row per distinct old source id; files and copies counted separately in inventory.
refs=pd.read_csv(R/'13_output_repair/STEP5K_R1_OA_REFERENCES.csv').fillna('NA').set_index('source_id');maps=[]
for s in S[:8]:
 old=refs.loc[s['old_source_id']]
 maps.append(dict(old_source_id=s['old_source_id'],old_filename=Path(s['local_pdf_path']).name,old_claimed_title=old.title,old_claimed_doi=old.doi,verified_title=s['title'],verified_authors=s['authors'],verified_journal=s['journal'],verified_year=s['year'],verified_doi=s['doi'],verification_level=s['verification_level'],bibliographic_match=False,source_mapping_status='MISMATCHED',notes='Actual document verified. R1 summary linked a different paper. Corrected identity below; no cross-paper metadata inheritance.',verified_source_id=s['verified_source_id'],verified_volume=s['volume'],verified_issue=s['issue'],verified_pages=s['pages'],verified_publisher=s['publisher'],visible_license=s['license'],pdf_sha256=sha(Path(s['local_pdf_path'])),verification_evidence=s['identity_evidence']))
csv('02_pdf_provenance/STEP5K_R2_SOURCE_PROVENANCE_MAP.csv',maps)
csv('02_pdf_provenance/VERIFIED_SOURCE_BIBLIOGRAPHY.csv',S)
old20=pd.read_csv(R/'13_output_repair/STEP5K_R1_OA_SOURCE_TRIAGE.csv').fillna('NA');cross=[]
for _,r in old20.iterrows():
 ids=str(r.local_fulltext_ids).split(';');links=[s for s in S[:8] if s['old_source_id'] in ids]
 cross.append(dict(paper_id=r.paper_id,claimed_doi=r.doi,claimed_title=r.title,old_fulltext_ids=r.local_fulltext_ids,corrected_fulltext_ids=[s['verified_source_id'] for s in links],same_document=any(s['doi'].lower()==r.doi.lower() for s in links),status='INVALID_FULLTEXT_LINK' if links else 'NO_LOCAL_FULLTEXT',notes='Bibliographic Crossref record can be correct while attachment association is wrong; abstract not accepted as experimental evidence.'))
csv('02_pdf_provenance/R1_20_RECORD_ATTACHMENT_CROSSWALK.csv',cross)
# Full text claims remain source-located; no numerical phase fraction is invented.
er=[]
for e in E:
 q=e.copy();q.update(experimental_phase_1=e['phases'][0] if len(e['phases'])>0 else 'NA',experimental_phase_2=e['phases'][1] if len(e['phases'])>1 else 'NA',experimental_phase_3=e['phases'][2] if len(e['phases'])>2 else 'NA',experimental_trend=e['evidence_note'])
 er.append(q)
csv('04_fulltext_evidence/STEP5K_R2_FULLTEXT_EVIDENCE.csv',er,['verified_source_id','source_origin','X_identity','system','nominal_composition','composition_basis','processing_state','temperature_C','time','cooling_condition','characterization','experimental_phase_1','experimental_phase_2','experimental_phase_3','measured_phase_compositions','experimental_trend','page','table','figure','section','evidence_note'])
# Inspect original numerical inputs from immutable R1 script, not mismapped bibliography.
oldstates=pd.read_csv(R/'05_optional_nonNi_CALPHAD/02_raw/POSTFREEZE_STATES.csv');oldph=pd.read_csv(R/'05_optional_nonNi_CALPHAD/02_raw/POSTFREEZE_PHASES.csv')
case_map={
 'PF_TI_doi1959':('R2_Doi1959_CuCrTi_1',{'CU':96.4,'CR':.54,'TI':3.06},None,'Basis unverified; both solver states failed.'),
 'PF_CO_vt1955':('R2_DeBoskey1955_CrCoCu',{'CU':3.13,'CR':57.03,'CO':39.84},'R2_CO_specimen29','Input differs from verified at.% conversion at frozen 4-decimal input precision; no exact comparison admitted.'),
 'PF_SI_zhou_nb3si19':('R2_Zhou2023_CuNbSi',{'CU':85.92,'NB':4.83,'SI':9.25},'R2_SI_A5','Rounded old conversion is not exact; 700C composition association conflicts with Table4 and narrative.'),
 'PF_SI_zhou_nb3si25':('R2_Zhou2023_CuNbSi',{'CU':82.34,'NB':5.02,'SI':12.64},'R2_SI_A4','Old conversion incorrect at frozen precision; 700C A12 nominal total90 at.% is ambiguous.'),
 'PF_AG_sun2020':('R2_Sun2020_CuCrAg',{'CU':99.6328,'CR':.1636,'AG':.2036},'R2_AG_Sun_peak','Failed; source time/figure label caveats prohibit simple peak-state equality.'),
 'PF_AG_liu2024':('R2_SI_AG_2024',{'CU':99.54,'CR':.26,'AG':.2},'EXCLUDED_SI_AG_2024','Input drops reported Si0.021 Fe0.018 Ti0.034 wt.%; wrong reduced chemistry and failed.'),
 'PF_AG_raabe2000':('R2_Raabe2000_CuCrAg',{'CU':87.,'CR':10.,'AG':3.},'R2_AG_Raabe_A_context','1323K is1049.85C, not exact1050C; no measured high-temperature phase assemblage.'),
 'PF_MO_cho2021':('R2_Cho2021_AMM_CuMoCr',{'CU':60.,'CR':10.,'MO':30.},'R2_MO_1250_control','Recovered additive-free control only; post-cooled SEM-EDS permits limited phase-chemistry context, not equilibrium fraction/phase-count validation.'),
 'PF_MO_seo2021':('R2_Cho2021_CuCrMo',{'CU':60.,'CR':10.,'MO':30.},'R2_MO_1200','Preparation+Table1 composition agrees; cooled XRD versus hot equilibrium permits only limited refractory-phase chemistry context.')}
em={e['validation_id']:e for e in E};audit=[];retained={}
for _,st in oldstates.iterrows():
 sid,input_c,eid,note=case_map[st.case_id];e=em.get(eid);v=wt(e) if e else None
 if st.case_id.startswith('PF_SI') and st.Temperature_C==700:v=None
 input_ok=bool(v and all(abs(input_c.get(k,0)-v.get(k,0))<1e-8 for k in set(input_c)|set(v)))
 temp_ok=bool(e and abs(e['temperature_C']-st.Temperature_C)<1e-8)
 if st.case_id=='PF_TI_doi1959':temp_ok=st.Temperature_C in [550,700]
 if st.case_id=='PF_AG_liu2024':temp_ok=st.Temperature_C==600 # Actual source Methods p2 includes 600 C/1 h; rejection is chemistry, not temperature.
 retain=bool(st.converged and input_ok and temp_ok and st.case_id.startswith('PF_MO'))
 if retain:retained[eid]=st.state_id
 audit.append(dict(old_state_id=st.state_id,verified_source_id=sid,old_input_composition=input_c,verified_composition=v if v else 'NA',input_correct=input_ok,temperature_correct=temp_ok,source_correct=True,converged=bool(st.converged),retain_as_evidence=retain,reason=note,retention_level='QUALITATIVE_CONTEXT_ONLY' if retain else 'EXCLUDED',old_bibliographic_mapping_correct=False if sid in {s['verified_source_id'] for s in S[:8]} else 'NA',source_correct_note='Actual intended document identified after R2 repair; does not assert old summary mapping was correct.'))
csv('08_old_calphad_audit/STEP5K_R2_OLD_CALPHAD_AUDIT.csv',audit)
newstates=pd.read_csv(O/'09_optional_new_nonNi_calphad/STATES.csv');newby=newstates.set_index('validation_id');validation=[]
for e in E:
 q={k:'NA' for k in ['CALPHAD_source_type','CALPHAD_state_id','CALPHAD_composition','CALPHAD_temperature_C','CALPHAD_phases']}
 q.update(validation_id=e['validation_id'],verified_source_id=e['verified_source_id'],source_origin=e['source_origin'],X_identity=e['X_identity'],alloy_system=e['system'],experimental_composition=e['nominal_composition'],composition_basis=e['composition_basis'],processing_state=e['processing_state'],temperature_C=e['temperature_C'],time=e['time'],experimental_phases=e['phases'],experimental_phase_compositions=e['measured_phase_compositions'],comparison_class=e['comparison_class'],agreement_class='NOT_COMPARABLE',agreement_notes='No admissible successful composition-specific CALPHAD pairing; literature-only context, not a successful validation.',kinetic_caveat=e['evidence_note'],evidence_page=e['page'],evidence_table=e['table'],evidence_figure=e['figure'],usable_experimental_condition=e['usable'],calphad_pair_eligible=False)
 if e['validation_id'] in retained:
  sid=retained[e['validation_id']];st=oldstates[oldstates.state_id==sid].iloc[0];ph=oldph[oldph.state_id==sid]
  q.update(CALPHAD_source_type='POSTFREEZE_R1_NONNI_VERIFIED_INPUT',CALPHAD_state_id=sid,CALPHAD_composition=wt(e),CALPHAD_temperature_C=float(st.Temperature_C),CALPHAD_phases=ph.raw_phase_name.tolist(),agreement_class='PARTIALLY_CONSISTENT',agreement_notes='Cr/Mo-rich refractory phase chemistry agrees qualitatively. CALPHAD predicts BCC+LIQUID at sintering T; experiment characterizes Cu+Mo-Cr after cooling. The comparison does not validate an identical phase assemblage, fractions, or BCC/BCC splitting.',calphad_pair_eligible=True)
 elif e['validation_id'] in newby.index:
  st=newby.loc[e['validation_id']];q.update(CALPHAD_source_type='POSTFREEZE_R2_NONNI_FAILED',CALPHAD_state_id=st.state_id,CALPHAD_composition=wt(e),CALPHAD_temperature_C=float(st.Temperature_C),agreement_notes='New source-verified equilibrium attempt failed: '+str(st.error_message)+'; no phases or agreement inferred.')
 validation.append(q)
csv('07_external_validation/STEP5K_R2_EXTERNAL_VALIDATION.csv',validation)
oa=[]
for s in S:
 es=[e for e in E if e['verified_source_id']==s['verified_source_id']]
 oa.append(dict(verified_source_id=s['verified_source_id'],source_origin=s['source_origin'],title=s['title'],doi=s['doi'],landing_url=s['landing_url'],fulltext_url=s['pdf_url'],pdf_url=s['pdf_url'],local_pdf_path=s['local_pdf_path'],oa_type=('PUBLISHER_CC' if 'CC BY' in s['license'] else ('AUTHOR_REPOSITORY' if 'Raabe' in s['verified_source_id'] or 'ZR_2012' in s['verified_source_id'] else ('INSTITUTIONAL_REPOSITORY' if s['document_type']=='masters_thesis' else 'PUBLISHER_FREE_TO_READ_LICENSE_UNSPECIFIED'))),oa_verified=True,fulltext_verified=True,usable_experimental_source=any(e['usable'] for e in es),reuse_license=s['license'],sha256=sha(Path(s['local_pdf_path']))))
csv('01_source_inventory/STEP5K_R2_OA_SOURCE_STATUS.csv',oa)
search=[]
meta={}
for p in (O/'14_logs/new_search').glob('crossref_final_*.json'):
 try:
  m=js(p);m=m.get('message',m)
  if 'DOI' in m:meta[m['DOI'].lower()]=m
 except Exception:pass
for n in N:
 s=sm.get('R2_'+n['paper_id']);m=meta.get(n['doi'].lower(),{});es=[e for e in E if e['verified_source_id']=='R2_'+n['paper_id']];full=bool(s)
 search.append(dict(new_source_id='R2_'+n['paper_id'],X_identity=n['X_identity'],alloy_system='Cu-Cr-'+n['X_identity'],title=n['title'],authors='; '.join(n['authors']),journal=s['journal'] if s else m.get('container-title',['NA'])[0],year=s['year'] if s else m.get('published',{}).get('date-parts',[['NA']])[0][0],doi=n['doi'],doi_url='https://doi.org/'+n['doi'],landing_url=n['publisher_url'],fulltext_url=s['pdf_url'] if s else 'NA',pdf_url=s['pdf_url'] if s else 'NA',oa_status='LEGAL_FULLTEXT_VERIFIED' if full else 'NOT_RECOVERED',primary_experimental=True if full else 'NA',nominal_composition_available=full,composition_basis_available=full,heat_treatment_available=full,temperature_available=full,phase_identification_available=any(e['phases'] for e in es) if full else 'NA',phase_composition_available='NA',isothermal_section_available=False if full else 'NA',potential_comparison_class='QUALITATIVELY_COMPARABLE' if any(e['usable'] for e in es) else 'NOT_DIRECTLY_COMPARABLE',priority_score=90 if n['paper_id'] in ['TI_Shi2023','SI_Yu2011'] else (75 if 'CO' in n['X_identity'] else 60),needs_user_pdf=n['decision']=='NEEDS_FULLTEXT',notes=n['decision']+'; '+n['metadata_note']+'; '+n['novelty_audit_note'],source_origin=n['source_origin'],genuinely_new_R2=n['new_beyond_R1']))
csv('03_new_literature_search/STEP5K_R2_NEW_LITERATURE_SEARCH.csv',search)
requests=[]
for r in search:
 if r['needs_user_pdf']:
  requests.append(dict(priority=1 if r['priority_score']==90 else 2,X_identity=r['X_identity'],alloy_system=r['alloy_system'],title=r['title'],authors=r['authors'],journal=r['journal'],year=r['year'],doi=r['doi'],landing_url='https://doi.org/'+r['doi'],why_needed='Actual composition/basis/thermal history and phase-identification tables unavailable. No abstract-only validation permitted.',expected_validation_value='Potential direct equilibrium' if r['priority_score']==90 else 'Potential phase-chemistry support',source_origin=r['source_origin']))
for pid in ['LIT_09','LIT_11','LIT_12','LIT_13','LIT_15','LIT_18']:
 r=old20[old20.paper_id==pid].iloc[0];m=js(O/'02_pdf_provenance'/f'{pid}_crossref.json')
 requests.append(dict(priority=1 if pid!='LIT_18' else 2,X_identity=r.X_identity,alloy_system='Cu-Cr-Nb' if pid=='LIT_18' else 'Cu-Cr-'+r.X_identity,title=r.title,authors='; '.join(a.get('given','')+' '+a.get('family','') for a in m.get('author',[])),journal=r.journal,year=r.year,doi=r.doi,landing_url='https://doi.org/'+r.doi,why_needed='R1 did not possess this document; attached PDF links were absent or wrong. Recover actual full text before evidence extraction.',expected_validation_value='Baseline Cr2Nb characterization; processing caveat' if pid=='LIT_18' else 'Primary phase-equilibrium conditions and measured phase chemistry',source_origin='existing_R1'))
csv('03_new_literature_search/STEP5K_R2_HIGH_VALUE_PDF_REQUEST_LIST.csv',requests)
coverage=[]
goals={'ZR':('LIMITED','Source literature: Cu-Zr precipitation and Zr segregation; no admissible matched-state confirmation.','C15/Zr-Laves and exact other-solid topology unverified.'),'TI':('LIMITED','New TEM/EDX demonstrates Cr-Ti enriched precipitates; exact structure unresolved.','Cu4Ti not established; Doi basis missing; no matched-state agreement.'),'CO':('LIMITED','Primary thesis establishes sigma+Cr+Cu at Cr-rich chemistry.','No Co-Nb Laves support; old converted input not exact; no admissible paired state.'),'SI':('LIMITED','Twelve internally unambiguous long-anneal Cu-Nb-Si conditions.','Cr3Si/Cr competition absent; four ambiguous source conditions; three new states failed.'),'AG':('LIMITED','Cr precipitation and processed Cr filaments in Cu-Cr-Ag sources.','Ag-rich/Cu-rich FCC separation not experimentally established in admitted conditions.'),'MO':('LIMITED','Two retained qualitative refractory Cr/Mo phase-chemistry comparisons.','Does not validate two BCC composition sets; no equilibrium fraction comparison.'),'CR_BASE':('NONE','No verified Cu-Cr-Nb full-text condition admitted.','Cu-Cr controls are not Cu-Cr-Nb baseline evidence; recover GRCop/Cr2Nb primary full text.')}
for x in ['ZR','TI','CO','SI','AG','MO','CR_BASE']:
 es=[e for e in E if e['X_identity']==x];vs=[v for v in validation if v['X_identity']==x];strength,support,gap=goals[x]
 coverage.append(dict(X_identity=x,n_verified_sources=len({e['verified_source_id'] for e in es}) if x!='CR_BASE' else 0,n_new_R2_sources=len({e['verified_source_id'] for e in es if e['source_origin']=='new_R2'}) if x!='CR_BASE' else 0,n_direct_conditions=sum(e['comparison_class']=='DIRECTLY_COMPARABLE' for e in es),n_qualitative_conditions=sum(e['comparison_class']=='QUALITATIVELY_COMPARABLE' for e in es),n_consistent=sum(v['agreement_class']=='CONSISTENT' for v in vs),n_partially_consistent=sum(v['agreement_class']=='PARTIALLY_CONSISTENT' for v in vs),n_discrepant=sum(v['agreement_class']=='DISCREPANT_BUT_EXPLAINABLE' for v in vs),experimental_topology_supported=support,remaining_gap=gap,evidence_strength=strength,n_usable_sources=len({e['verified_source_id'] for e in es if e['usable']}),strength_definition='Strength refers to the requested element-specific topology validation, not bibliographic quality.'))
csv('07_external_validation/STEP5K_R2_ELEMENT_COVERAGE.csv',coverage)
# Canonical optional run: failures explicitly retained, no phase/fraction substitution.
newstates.to_parquet(O/'09_optional_new_nonNi_calphad/STEP5K_R2_NEW_EXTERNAL_CALPHAD.parquet',index=False)
freeze=js(O/'09_optional_new_nonNi_calphad/execution_plan.json');freeze.update(status='COMPLETED_WITH_FAILURES',n_states=3,n_converged=int(newstates.converged.sum()),n_failed=int((~newstates.converged).sum()),used_for_training=False,automatic_retry=False,parquet_sha256=sha(O/'09_optional_new_nonNi_calphad/STEP5K_R2_NEW_EXTERNAL_CALPHAD.parquet'))
dump(O/'09_optional_new_nonNi_calphad/STEP5K_R2_NEW_EXTERNAL_CALPHAD_FREEZE.json',freeze)
(O/'09_optional_new_nonNi_calphad/STEP5K_R2_NEW_EXTERNAL_CALPHAD_QA.md').write_text('# New external equilibrium QA\n\nThree source-verified states at two distinct literature compositions were attempted with the unchanged dynamic extraction and five-family mapping. All three failed with QTHISS TOO MANY ITERATIONS. Failed states remain in the canonical parquet. No phase amounts or molar partitions exist for failed states; no empty table is interpreted as phase absence. No additional retries, bridge grids or compositions were launched. NI states=0. The initial wrapper compatibility errors occurred before equilibrium requests; these are not extra thermodynamic states.\n',encoding='utf8')
# Only relevant historical plan items are reassessed; new audit rows do not overwrite history.
pa=pd.read_csv(R/'13_output_repair/STEP5K_R1_UPDATED_PLAN_AUDIT.csv').fillna('NA');plans=[]
for pid,status,note in [('P26','PARTIAL','Source identity repaired; 2 qualitative paired comparisons, 0 direct equilibrium pairs. Broad validation remains open.'),('P32','PARTIAL','External-only replacement insertions ready for review; manuscript not certified ready for final assembly.')]:
 old=pa[pa.plan_item_id==pid].iloc[0].to_dict();old.update(r2_status=status,r2_note=note);plans.append(old)
plans.extend([dict(plan_item_id='R2_PROVENANCE',r2_status='COMPLETED',r2_note='Eight original unique documents verified and eight wrong summary associations replaced; duplicate files not independent sources.'),dict(plan_item_id='R2_EXTERNAL_CALPHAD',r2_status='PARTIAL',r2_note='12 old states audited; 2 retained only qualitatively. Three new source-verified states failed; no direct equilibrium agreement established.')])
csv('11_plan_audit/STEP5K_R2_UPDATED_PLAN_AUDIT.csv',plans)
counts=pd.Series([e['comparison_class'] for e in E]).value_counts().to_dict();ags=pd.Series([v['agreement_class'] for v in validation]).value_counts().to_dict()
summary=dict(stage='STEP5K-R2',status='COMPLETE_SCOPE_WITH_SCIENTIFIC_GAPS',existing_local_unique_documents_verified=8,existing_local_pdf_copies=12,existing_local_journal_papers=7,source_mappings_correct=0,source_mappings_mismatched=8,source_mappings_ambiguous=0,source_mappings_unverified=0,new_candidate_records_reviewed=9,genuinely_new_candidates=6,rediscovered_R1_candidates=3,new_legally_accessible_fulltexts=3,new_usable_experimental_papers=3,recovered_R1_candidate_fulltexts=2,high_value_user_pdf_requests=len(requests),verified_primary_experimental_papers_total=12,verified_primary_experimental_theses_total=1,usable_primary_papers=len({e['verified_source_id'] for e in E if e['usable'] and sm[e['verified_source_id']]['document_type']=='journal_article'}),condition_rows=len(E),condition_comparability=counts,agreement_classes=ags,exact_FULLV1_composition_temperature_matches=0,tight_neighbors=0,distant_neighbors=sum(m['match_class']=='DISTANT_NEIGHBOR' for m in matches),no_matches=sum(m['match_class']=='NO_MATCH' for m in matches),old_nearest_rows_removed=14,old_converged_states=4,old_converged_retained_qualitatively=len(retained),old_converged_retained_directly=0,new_nonNi_calphad_necessary='Justified to test selected verified equilibrium compositions with no exact frozen state; attempted, unsuccessful',new_compositions=2,new_states=3,new_states_converged=0,new_states_failed=3,new_Ni_calphad=0,new_Ni_predictions=0,Ni_composition_changes=0,model_retraining=0,production_scaler_refits=0,feature_reselection=0,hyperparameter_search=0,new_unseen_element=0,element_evidence_strength={x:v[0] for x,v in goals.items()},external_text_safe='YES only for narrow source-grounded qualitative context and explicit gaps; NO for broad experimental validation or numerical accuracy',manuscript_ready_for_final_assembly=False,stop_rule='Stop after R2; await user review; no optimization or further calculations.')
dump(O/'STEP5K_R2_SUMMARY.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
