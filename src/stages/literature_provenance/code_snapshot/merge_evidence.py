import sys,json,re
sys.dont_write_bytecode=True
from pathlib import Path
O=Path('E:/Cu_CALPHAD_ML/08_FINAL_MANUSCRIPT/STEP5K_R2_SOURCE_PROVENANCE_REPAIR')
sys.path.insert(0,str(O/'code_snapshot'))
import curate_existing as c
sources=c.sources;e=c.e
legacy=json.loads((O/'02_pdf_provenance/legacy_review/legacy_findings.json').read_text(encoding='utf8'))['records']
for s in sources:
 if s['old_source_id']=='VT1955_CrCoCu_sigma':
  s.update(authors='Wentzle R. DeBoskey',landing_url='https://hdl.handle.net/10919/40972',pdf_url='https://vtechworks.lib.vt.edu/bitstreams/1b20392c-764f-4b0c-b8d2-7ac58a714bce/download',identity_evidence='PDF title page plus repository DSpace author metadata; remote PDF SHA256 equals local. DOI NA; verified identity does not require DOI.')
 if s['old_source_id']=='JIM1959_CuCrTi_ageing1':s['verification_level']='C'
for k in legacy:
 if k['id'].startswith('JIM1959'):
  vid='R2_Doi1959_CuCrTi_'+('1' if 'ageing1' in k['id'] else '2')
  for j,q in enumerate(k['conditions']):
   c.ev(f'{vid}_{j+1}',vid,'TI','Cu-Cr-Ti',{'CU':96.4,'CR':.54,'TI':3.06},'NA',q['T_C'],str(q['duration_min'])+' min',q.get('phases',[q.get('phase')]),q['location'],'see location','see location','see location','Specimen percent basis unspecified; do not inherit wt.% axes from background binary diagrams. Historic beta-prime Cu3Ti/gamma TiCr2 labels preserved. Source-level precipitation context only; excluded by R2 basis rule.',processing='Solution treated; aged',char='Residue XRD' if vid.endswith('1') else 'Extraction replica electron diffraction',classification='NOT_DIRECTLY_COMPARABLE',usable=False,composition_valid=False)
co=legacy[0]
c.ev('R2_CO_specimen29','R2_DeBoskey1955_CrCoCu','CO','Cu-Cr-Co',{'CU':2.7,'CR':60.2,'CO':37.1},'at.%',1000,'3 or 4 d; general protocol, exact specimen duration NA',co['phases'],'PDF/printed 43-49;51;54','II row29','11','Experimental procedure; results','Revised chemically analyzed at.% composition, not proposed batch composition. Equilibrium judged by successive unchanged microstructures. Thesis is primary experimental work but is NOT a journal paper. No Nb or Co-Nb Laves evidence.',processing='Melted; annealed under helium-hydrogen',cool='Cold tap water quench',char='Metallography; powder XRD',classification='DIRECTLY_COMPARABLE')
c.ev('R2_AG_Raabe_C','R2_Raabe2000_CuCrAg','AG','Cu-Cr-Ag',{'CU':87.,'CR':10.,'AG':3.},'wt.%',999.85,'100 min',['Cr filaments in Cu-based composite'],'PDF 2;4;6-7 (printed187;189;191-192)','1','1C;6b-c','4.1.1','Post-solution-treatment morphology at strain3.95; no complete high-temperature equilibrium assemblage or Ag-rich FCC identification. Nominal Cu10Cr3Ag equals measured Table1 values. This is not the old 1323K processing step.',processing='Induction melt/chill cast; drawn; solution treatment1273K; water quenched',cool='Water quench',char='SEM; selective etching')
c.ev('R2_AG_Raabe_A_context','R2_Raabe2000_CuCrAg','AG','Cu-Cr-Ag',{'CU':87.,'CR':10.,'AG':3.},'wt.%',1049.85,'60 min',[],'PDF 2-3 (printed187-188)','1','1A','Processing','1323 K = 1049.85 C. No measured equilibrium phase assemblage at this processing step; cannot use 1050 C old state to validate Ag-rich FCC separation.',processing='Scheme A solution annealing under Ar',cool='Water quench',classification='NOT_DIRECTLY_COMPARABLE',usable=False)
# R1 candidate not previously locally accessible: source origin remains existing_R1.
mo=dict(old_source_id='MO_Cho2021',verified_source_id='R2_Cho2021_AMM_CuMoCr',X_identity='MO',title='Effects of ZrO2 and Al2O3 Addition on the Physical Properties of Cu-Mo-Cr Alloy by Liquid Phase Sintering',authors='Yeong-Woo Cho; Jae-Jin Sim; Sung-Gue Heo; Hyun-Chul Kim; Yong-Kwan Lee; Jong-Soo Byeon; Yong-Tak Lee; Kee-Ahn Lee; Seok-Jun Seo; Kyoung-Tae Park',journal='Archives of Metallurgy and Materials',year=2021,volume='66',issue='3',pages='683-687',doi='10.24425/amm.2021.136361',verification_level='A',publisher='Polish Academy of Sciences',license='CC BY-NC 4.0 (as printed; license prose also says no modifications)',landing_url='https://doi.org/10.24425/amm.2021.136361',pdf_url='https://journals.pan.pl/Content/119234/PDF/AMM-2021-3-03-Seok-Jun%20Seo.pdf?handler=pdf',local_pdf_path=str(O/'02_pdf_provenance/recovered_R1/MO_Cho2021_recovered.pdf'),source_origin='existing_R1',document_type='journal_article',identity_evidence='Actual publisher PDF p1, DOI printed; recovered R1 failed download candidate')
sources.append(mo)
c.ev('R2_MO_1250_control',mo['verified_source_id'],'MO','Cu-Mo-Cr',{'CU':60.,'CR':10.,'MO':30.},'wt.%',1250,'5 h',['Cu','Mo-Cr solid solution','Cr-rich agglomerates'],'PDF2-3 (printed684-685)','1','3a','2;3','Only the additive-free control in Fig3a is admitted. Ceramic-added samples excluded; never remove oxides from an experimental sample. SEM-EDS chemical contrast does not prove BCC/BCC crystallographic splitting; cooled specimen is not equilibrium liquid-phase measurement.',processing='Powder milled200rpm4h; pressed186MPa3min; heated10C/min; Ar sintered',cool='NA',char='SEM-EDS')
for row in e:
 if row['validation_id']=='R2_AG_Sun_peak':row.update(comparison_class='QUALITATIVELY_COMPARABLE',usable=True)
# New literature full-text review, retain all impurity/trace information.
new=json.loads((O/'03_new_literature_search/NEW_CANDIDATES_FINAL.json').read_text(encoding='utf8'))
ne=json.loads((O/'03_new_literature_search/NEW_EVIDENCE_FINAL.json').read_text(encoding='utf8'))
metadata={}
for p in (O/'14_logs/new_search').glob('crossref_final_*.json'):
 try:
  ob=json.loads(p.read_text(encoding='utf8'));m=ob.get('message',ob)
  if 'DOI' in m:metadata[m['DOI'].lower()]=m
 except Exception:pass
for p in ['CO_2023_crossref.json','TI_2023_crossref.json']:
 ob=json.loads((O/'14_logs/new_search'/p).read_text(encoding='utf8'));m=ob.get('message',ob)
 if 'DOI' in m:metadata[m['DOI'].lower()]=m
urls={'TI_SI_Sumino2019':'https://www.jstage.jst.go.jp/article/jic/58/1/58_64/_pdf','ZR_2012':'https://arxiv.org/pdf/1207.3640','SI_AG_2024':'https://mdpi-res.com/d_attachment/coatings/coatings-14-00909/article_deploy/coatings-14-00909.pdf'}
# Recover resolved full-text URLs from prior retrieval log.
retry=O/'03_new_literature_search/retrieval_candidates_retry.json'
if retry.exists():
 for z in json.loads(retry.read_text(encoding='utf8')):
  if z.get('pdf',{}).get('status')==200:urls[z['id']]=z['pdf']['resolved_url']
for n in new:
 n['new_beyond_R1']=n['paper_id'] not in ['TI_Shi2023','CO_Hui2022','SI_AG_2024']
 n['source_origin']='new_R2' if n['new_beyond_R1'] else 'existing_R1'
 n['novelty_audit_note']='Compared against R1 20-record table AND R1 fetch scripts/matching inputs; Shi/Hui/Liu were already attempted or used in R1.'
 if not n.get('local_fulltext_path'):continue
 m=metadata.get(n['doi'].lower(),{});pid=n['paper_id']
 journal=m.get('container-title',['NA'])[0];year=m.get('published',{}).get('date-parts',[[None]])[0][0]
 if pid=='TI_SI_Sumino2019':journal='Journal of Japan Institute of Copper';year=2019
 if pid=='ZR_Hughes2022':journal='Journal of Materials Research and Technology';year=2022
 sources.append(dict(verified_source_id='R2_'+pid,old_source_id=pid if not n['new_beyond_R1'] else 'NA',X_identity=n['X_identity'],title=n['title'],authors='; '.join(n['authors']),journal=journal,year=year,volume='58' if pid=='TI_SI_Sumino2019' else m.get('volume','20' if pid=='ZR_Hughes2022' else 'NA'),issue='1' if pid=='TI_SI_Sumino2019' else m.get('issue','NA'),pages='64-67' if pid=='TI_SI_Sumino2019' else m.get('page','801-810' if pid=='ZR_Hughes2022' else 'NA'),doi=n['doi'],verification_level='B' if pid=='TI_SI_Sumino2019' else ('C' if pid=='ZR_2012' else 'A'),publisher='Japan Institute of Copper' if pid=='TI_SI_Sumino2019' else m.get('publisher','Elsevier'),license='CC BY' if pid=='ZR_Hughes2022' else ('CC BY 4.0' if pid=='SI_AG_2024' else 'NA'),landing_url=n['publisher_url'],pdf_url=urls.get(pid,'NA'),local_pdf_path=n['local_fulltext_path'],source_origin=n['source_origin'],document_type='journal_article',identity_evidence=n['metadata_note']))
for n in ne:
 comp={k.upper():v for k,v in n['composition'].items()};numeric=all(isinstance(v,(int,float)) for k,v in comp.items() if k!='CU')
 # Do not erase trace elements. Numeric balance permitted only when all solutes quantified.
 if numeric and comp.get('CU')=='balance':comp['CU']=100-sum(v for k,v in comp.items() if k!='CU')
 usable=n['usable_for_qualitative_support'];cl=n['comparability_class']
 if n['X_identity']=='CR_BASE':usable=False;cl='NOT_DIRECTLY_COMPARABLE'
 temp=round(n['temperature_K']-273.15,8)
 if n['paper_id']=='ZR_Hughes2022':temp=480.0 # Article explicitly gives 480C; parenthetical753K rounded. Prefer explicitly stated experimental Celsius.
 q=dict(validation_id=n['evidence_id'],verified_source_id='R2_'+n['paper_id'],source_origin='existing_R1' if n['paper_id']=='SI_AG_2024' else 'new_R2',X_identity=n['X_identity'],system='Cu-Cr-Zr' if n['X_identity']=='ZR' else ('Cu-Cr-'+n['X_identity']),nominal_composition=comp,composition_basis=n['composition_basis'],temperature_C=temp,time=str(n['aging_time_s'])+' s',phases=n['experimental_phases'],page=n['source_location'],table='1',figure='see source location',section='see source location',evidence_note=n['kinetic_and_identity_caveats'],processing_state=n['processing_state'],cooling_condition='See processing_state; absent cooling details NA',characterization='TEM; EDX' if 'SUMINO' in n['evidence_id'] else 'TEM; atom-probe tomography',measured_phase_compositions='NA',comparison_class=cl,usable=usable,composition_valid=numeric)
 if n['X_identity']=='CR_BASE':q['evidence_note']+=' Cu-Cr control contains no Nb; excluded from baseline Cu-Cr-Nb coverage.'
 if not numeric:q['evidence_note']+=' Trace/below-detection concentrations remain literal; no exact reduced-chemistry state is manufactured.'
 if n['paper_id']=='ZR_Hughes2022':q['evidence_note']+=' Explicit 480 C used; 753 K is rounded parenthetical notation.'
 e.append(q)
(O/'04_fulltext_evidence/all_curated.json').write_text(json.dumps(e,indent=2,ensure_ascii=False),encoding='utf8')
(O/'02_pdf_provenance/all_verified_sources.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False),encoding='utf8')
(O/'03_new_literature_search/candidates_novelty_audited.json').write_text(json.dumps(new,indent=2,ensure_ascii=False),encoding='utf8')
print('sources',len(sources),'evidence',len(e),'genuinely new candidates',sum(n['new_beyond_R1'] for n in new))
