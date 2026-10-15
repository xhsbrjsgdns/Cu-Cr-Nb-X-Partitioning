from pathlib import Path
import json,hashlib,re,html,datetime
import pandas as pd
O=Path(__file__).resolve().parents[1]
def js(p):return json.loads((O/p).read_text(encoding='utf8'))
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,s):(O/p).write_text(s,encoding='utf8')
E=js('02_source_extraction/ALL_CONDITIONS.json'); S=js('02_source_extraction/ALL_SOURCES.json')
req=pd.read_csv(O/'06_calphad_requests/STEP5K_R3_CALPHAD_REQUESTS.csv'); chosen=set(req.condition_id)
assert not (O/'07_calphad_raw/ATTEMPTS.jsonl').exists(),'Do not overwrite a calculation run with preflight status'
doc=Path('C:/Program Files/Thermo-Calc/2025b/HTML5/content/databases/tccu/tccu_models_phases.htm')
text=doc.read_text(encoding='utf8')
rows=[[html.unescape(re.sub('<[^>]+>',' ',c)).strip() for c in re.findall(r'<td[^>]*>(.*?)</td>',r,re.S)] for r in re.findall(r'<tr[^>]*>(.*?)</tr>',text,re.S)]
db={r[0]:r for r in rows if len(r)>6}
mapping={'fcc(Cu)':'FCC_L12','fcc (Cu)':'FCC_L12','(Cu)':'FCC_L12','Cu matrix':'FCC_L12','fcc(Ag)':'FCC_L12','(αCo)':'FCC_L12','bcc(Cr)':'BCC_B2','bcc (Cr)':'BCC_B2','(Cr)':'BCC_B2','bcc (Nb)':'BCC_B2','(Cr, Mo)':'BCC_B2','βTi':'BCC_B2','αTi':'HCP_A3','(Si)':'DIAMOND_A4','Liquid':'LIQUID','αTiCr2':'C15_LAVES','Cu4Ti':'CU4TI1','Cu4Ti3':'CU4TI3','Cu3Ti2':'CU3TI2','(Cu, Cr)3Ti2':'CU3TI2','CuTi':'CUTI_B11','CuTi2':'CUZR2_C11B','Cr3Si':'CR3SI','CrSi2':'CRSI2_C40','Cu15Si4':'CU15SI4_EPSILON','Cu33Si7':'CU33SI7_DELTA','Cu56Si11':'CU56SI11_GAMMA','hcp(CuSi)':'HCP_A3'}
notes=js('05_phase_crosswalk/EXPERIMENTAL_STRUCTURE_NOTES.json'); cross=[]
for sid,ph in sorted({(e['source_id'],p) for e in E for p in e['phases']}):
 n=next((n for n in notes if n.get('source_id')==sid and n.get('phase')==ph),{})
 raw=mapping.get(ph,'UNRESOLVED');confidence='CANDIDATE_PENDING_ENGINE_CHECK'
 reason='Source phase/structure and installed TCCU6 phase-description table; calculated chemistry has not been checked.'
 family={'FCC_L12':'FCC','BCC_B2':'BCC','C15_LAVES':'C15','LIQUID':'LIQUID'}.get(raw,'OTHER_SOLID' if raw!='UNRESOLVED' else 'UNRESOLVED')
 if ph=='Cr2Nb':raw='C14_LAVES | C15_LAVES';family='UNRESOLVED_C15_OR_OTHER_SOLID';confidence='AMBIGUOUS_POLYMORPH';reason='Experimental Cr2Nb polymorph unspecified; source model table lists both C14/C15 and cannot establish observed polymorph.'
 if ph=='τ1':reason='Actual source phase is tau1, not sigma1; no supported equivalence to database SIGMA. Do not force mapping.'
 if ph=='(αCo)':confidence='MEDIUM_SOURCE_INTERPRETATION';reason='Source p295 describes FCC miscibility gap; experiment literal alpha-Co label preserved.'
 cross.append(dict(literature_phase=ph,literature_structure=n.get('structure',n.get('pearson_symbol','UNSPECIFIED')),literature_prototype=n.get('prototype'),TCCU_raw_phase_candidate=raw,frozen_family=family,mapping_confidence=confidence,mapping_basis=reason,source_reference=sid+'; '+str(n.get('page','condition source table')),notes=n.get('evidence',n.get('note','')),database_doc_phase_description=db.get(raw,['','','','','','',''])[1:7]))
pd.DataFrame(cross).to_csv(O/'05_phase_crosswalk/STEP5K_R3_PHASE_CROSSWALK.csv',index=False,encoding='utf-8-sig')
pd.DataFrame([dict(raw_phase=k,description=v[1],strukturbericht=v[2],pearson=v[3],space_group=v[4]) for k,v in db.items() if k in set(mapping.values()) or k in ['C14_LAVES','C36_LAVES','SIGMA']]).to_csv(O/'05_phase_crosswalk/DATABASE_DOCUMENTATION_PHASES.csv',index=False)
write('11_system_evidence/STEP5K_R3_DATABASE_INDEPENDENCE_NOTE.md','''# Database independence assessment

Independence is not established. Installed Thermo-Calc 2025b TCCU documentation states in its revision history that Co-Cr-Cu, Cr-Cu-Nb and Cr-Cu-Mo ternaries were assessed in TCCU2. This documents assessment coverage, not a source-by-source provenance trail. The six experimental papers include thermodynamic assessment work; no supplied full text or inspected database documentation proves that their data were withheld from TCCU6. Neither matching publication dates nor the absence of an explicit citation proves independence.

Therefore any eventual result must be called a **source-grounded experimental consistency check**, not statistically independent validation of TCCU6 and not experimental validation of the ML surrogate. Baseline Cu-Cr-Nb, adjacent Cu-Cr-X ternaries, processed GRCop and target quaternary/ML-domain evidence must remain distinct. There are currently no R3 calculated results.

Version-specific sources: [TCCU revision history](https://help.thermocalc.com/2025b/content/databases/tccu/tccu_revision_history.htm), [TCCU phase descriptions](https://help.thermocalc.com/2025b/content/databases/tccu/tccu_models_phases.htm). The installed 2025b copies were inspected. Current database product pages describe newer versions and were not substituted for the frozen TCCU6 implementation.
''')
pending=req.copy();pending['execution_status']='NOT_EXECUTED_INFRASTRUCTURE_BLOCKED';pending['converged']=pd.Series([None]*len(req),dtype='boolean');pending['error_message']='Engine initialization failed before database selection; no equilibrium request submitted.'
pending.to_parquet(O/'08_calphad_processed/STEP5K_R3_EXTERNAL_CALPHAD.parquet',index=False)
pd.DataFrame(columns=['condition_id','state_id','raw_phase_name','phase_model_base_name','phase_mole_fraction','element','element_mole_fraction','element_mass_fraction']).to_csv(O/'07_calphad_raw/STEP5K_R3_EXTERNAL_CALPHAD_RAW_PHASES.csv',index=False)
pd.DataFrame(columns=['condition_id','state_id','family','n_instances','phase_mole_fraction_total']).to_csv(O/'08_calphad_processed/STEP5K_R3_EXTERNAL_CALPHAD_FAMILY.csv',index=False)
comparison=[]
for e in E:
 selected=e['condition_id'] in chosen
 comparison.append(dict(condition_id=e['condition_id'],source_id=e['source_id'],system=e['system'],temperature_K=e['temperature_K'],direct_eligible=e['direct_eligible'],selected=selected,execution_status='NOT_EXECUTED_INFRASTRUCTURE_BLOCKED' if selected else ('NOT_SELECTED_BUDGET' if e['direct_eligible'] else 'EXCLUDED'),verdict=None if selected else 'NOT_COMPARABLE',experimental_raw_phase_set=json.dumps(e['phases'],ensure_ascii=False),calculated_raw_phase_set=None,exact_raw_phase_set_match=None,raw_phase_Jaccard=None,family_exact_set_match=None,family_Jaccard=None,missing_experimental_phases=None,extra_calculated_phases=None,n_FCC_raw_instances=None,n_BCC_raw_instances=None,notes='Pending calculation; no agreement verdict assigned.' if selected else e.get('exclusion_reason') or 'Frozen stratified budget exclusion'))
pd.DataFrame(comparison).to_csv(O/'09_condition_comparison/STEP5K_R3_CONDITION_COMPARISON.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(columns=['condition_id','experimental_phase','calculated_raw_phase','element','experimental_at_pct','calculated_at_pct','absolute_error_at_pct','phase_composition_MAE','matching_status']).to_csv(O/'10_phase_composition/STEP5K_R3_PHASE_COMPOSITION_COMPARISON.csv',index=False)
systems=[]
for name in ['Cu-Cr-Nb','Cu-Cr-Ag','Cu-Cr-Mo','Cu-Cr-Co','Cu-Cr-Si','Cu-Cr-Ti','GRCop qualitative context']:
 es=[e for e in E if (e['source_id']=='S6_GRCOP' if name.startswith('GRCop') else e['system']==name and e['source_id']!='S6_GRCOP')]
 systems.append(dict(system=name,n_source_conditions=len(es),n_directly_admitted=sum(e['direct_eligible'] for e in es),n_selected=sum(e['condition_id'] in chosen for e in es),n_calphad_submitted=0,n_calphad_converged=0,n_calphad_failed=0,n_strongly_consistent=None,n_consistent=None,n_partially_consistent=None,n_discrepant=None,raw_phase_support='EXPERIMENTAL_OBSERVATIONS_ONLY',family_phase_support='NOT_ASSESSED',phase_composition_support='NOT_ASSESSED',topology_support='NOT_ASSESSED',ML_domain_relevance='OUTSIDE_FULLV1_BOUNDS',evidence_strength='LIMITED' if name.startswith('GRCop') else 'NONE',remaining_limitation='R3 calculation blocked before initialization; NONE means no computed consistency evidence, not poor experimental source quality.'))
pd.DataFrame(systems).to_csv(O/'11_system_evidence/STEP5K_R3_SYSTEM_EVIDENCE.csv',index=False)
freeze=js('03_inclusion_freeze/INPUT_FREEZE.json');freeze.update(status='INPUTS_FROZEN_EXECUTION_BLOCKED',n_submitted=0,n_converged=0,n_failed_equilibrium=0,n_pending=120,solver_levels=[dict(level=i,max_iterations=a,global_grid_points=b) for i,(a,b) in enumerate([(500,2000),(2000,10000),(5000,30000),(10000,100000)])],database='TCCU6 Cu-Alloys v6.2',used_for_training=False)
write('08_calphad_processed/STEP5K_R3_EXTERNAL_CALPHAD_FREEZE.json',json.dumps(freeze,indent=2))
write('14_qa/STEP5K_R3_EXTERNAL_CALPHAD_QA.md','''# R3 execution QA — BLOCKED

120 states are frozen requests, not submitted equilibrium calculations. Submitted=0, converged=0, equilibrium failures=0, pending=120. Repeated failures occur during TCPython initialization at setGesVersion(6), before database selection. Native Java crash category is stack overflow (0xc00000fd). A process-local 32 MB Java stack and a pandas-import context check did not restore initialization; no installed files or scientific settings changed.

The canonical parquet contains 120 pending rows with nullable convergence. Raw-phase, family and phase-comparison CSVs contain headers only; they do not establish phase absence. No agreement metrics or phase figures can be generated from these pending data. The R3 stage remains incomplete. The prepared run_r3.py resumes frozen inputs after the runtime is restored; postprocessing and manuscript result updates still remain afterward.
''')
baseline=js('00_config/protected_before.json');checks=[]
for r in baseline:checks.append(dict(path=r['path'],before_sha256=r['sha256'],after_sha256=sha(r['path']),unchanged=sha(r['path'])==r['sha256']))
assert all(r['unchanged'] for r in checks)
pd.DataFrame(checks).to_csv(O/'14_qa/PROTECTED_HASH_VERIFICATION.csv',index=False)
assert all(sha(s['pdf_path'])==s['sha256'] for s in S)
write('14_qa/STEP5K_R3_INTEGRITY_AUDIT.md','''# R3 integrity audit

PASS: all 471 protected baseline SHA-256 hashes unchanged; all six original source PDFs unchanged. No Ni calculation/prediction/composition modification, ML retraining, production scaler refit, feature or hyperparameter change, new prospective chemistry, DFT, screening or finalist change. The source-specific request elements are restricted to Cu, Cr, Nb, Ag, Mo, Co, Si and Ti.

The whole supplied Ag/Ni PDF is retained as a source artifact, but the derivative evidence extract and table image are restricted to Ag Table 2. No Ni conditions enter the evidence or request tables. A Si-table nominal typo is preserved as printed and resolved only using the same specimen's explicit Cu70Cr5Si25 prose; no Ni input is inferred.

Exact file-level before/after hashes are in PROTECTED_HASH_VERIFICATION.csv. The prepared execution script has not submitted any equilibrium calculations. Python compilation validates syntax only, not runtime or scientific results.
''')
write('12_figures/FIGURE_STATUS.md','''# Validation figure status

No experimental-versus-calculated phase matrix, agreement heatmap or phase-composition scatter is generated because no R3 equilibrium has been submitted. Rendering pending states as disagreements or absent phases would be misleading. The system evidence CSV and condition table are ready as source tables; quantitative validation figures remain pending runtime restoration and calculated results.
''')
for section,body in {
 'METHODS':'Experimental conditions were transcribed from six supplied full texts and classified before calculation. Of 149 source records, 135 met the direct inclusion criteria; a source/temperature/phase-region stratified rule selected 120 conditions. Original composition bases and source locations were preserved; atomic-percent compositions were converted without early rounding. Raw phase instances, five-family aggregation and database-independence limitations are assessed separately. These are baseline or adjacent-ternary checks, distinct from ML surrogate and prospective OOD evidence.',
 'RESULTS':'No R3 result paragraph is authorized yet: the thermodynamic engine fails during initialization before equilibrium submission. Do not replace R2 outcome language with a claimed R3 improvement. The 120 selected conditions are pending inputs, not converged or failed equilibrium results.',
 'DISCUSSION':'Source-grounded experimental consistency checks must be distinguished from statistically independent database validation and ML validation. The supplied baseline and adjacent-ternary studies cannot establish target-quaternary OOD transfer. The current runtime failure supplies no evidence about physical agreement or disagreement and no logical contradiction of the frozen prospective Ni/OOD findings.',
 'SUPPLEMENTARY':'The source package comprises six internally verified PDFs with hashes, 149 evidence records, 135 eligible conditions and a frozen 120-condition request list. Unknowns and source contradictions remain explicit. All 471 protected artifact hashes are unchanged. Calculation outputs currently encode pending status with nullable convergence; raw-phase and comparison tables are empty pending execution. No global experimental accuracy is reported.'}.items():
 write(f'13_manuscript_insertions/STEP5K_R3_JMST_{section}_INSERTIONS.md',f'# R3 {section.lower()} insertion — provisional, stage incomplete\n\n'+body+'\n')
summary=dict(stage='STEP5K-R3',status='BLOCKED_ENGINE_INITIALIZATION',complete=False,verified_sources=6,source_records=149,direct_eligible=135,selected_requests=120,submitted_equilibrium_states=0,converged=0,failed_equilibrium_states=0,pending_states=120,protected_files_unchanged=471,new_Ni_calculations=0,new_Ni_predictions=0,Ni_composition_changes=0,model_retraining=0,production_scaler_refit=0,feature_reselection=0,hyperparameter_search=0,new_prospective_element=0,manuscript_ready_for_final_assembly=False,stop_reason='Runtime initialization fails before database selection; complete R3 comparison remains pending.')
write('STEP5K_R3_SUMMARY.json',json.dumps(summary,indent=2))
ans={1:'Yes, all six internally verified and hashed.',2:'149 source records including 6 GRCop processing contexts and 3 Ti diffusion/structure contexts.',3:'135 eligible.',4:'120 requests prepared and frozen; ZERO actually submitted to the engine.',5:'0.',6:'0 equilibrium failures; 120 not executed because engine initialization fails.',7:'29 eligible baseline states selected; no calculated baseline result.',8:'Not established. Source experimental Cr2Nb does not identify its polymorph; database C15 is cubic Laves, but C14 must not be silently excluded.',9:'Pending: 14 exact Ag conditions selected; no result.',10:'Pending: 16 of 31 eligible Mo conditions selected by frozen stratification; no result.',11:'Adjacent Cu-Cr-Co chemistry only, never Co-Nb Laves; 16 states pending.',12:'Not yet assessed; Cr3Si/CrSi/CrSi2/alphaCr5Si3 observations retained.',13:'Not yet assessed at 973 K.',14:'Not yet assessed at 1073 K.',15:'Not assessed; no successful computations.',16:'Not assessed.',17:'Not assessed.',18:'Not assessed; startup failure is not physical discrepancy.',19:'None can be assigned STRONG R3 computed support yet.',20:'None can be assigned MODERATE R3 computed support yet.',21:'GRCop provides LIMITED qualitative materials context; direct computed support remains NONE/pending.',22:'None under the frozen S/Cr:Nb/X chemistry-bound classification.',23:'All 149 records are outside or cannot establish membership; invalid compositions remain explicitly unconvertible.',24:'No observed R3 physical discrepancy exists. A software startup failure has no logical bearing on frozen Ni/OOD outcomes.',25:'No new calculated multi-system consistency claim is justified yet.',26:'Safe now: Six primary full texts provide a source-grounded experimental comparison set; exact-condition calculations remain pending. After execution use source-grounded experimental consistency checks, never independent database or ML validation.',27:'No. Calculations, comparisons, figures and final results insertions remain incomplete.'}
write('STEP5K_R3_REPORT.md','''# STEP5K-R3 — inputs complete, execution blocked

Six supplied PDFs were internally verified. The evidence table contains 149 records, of which 135 satisfy direct inclusion. The frozen 120-state selection contains Ag 14, Mo 16, Co 16, Nb baseline 29, Si 17 and Ti 28. The 15 unselected eligible Mo rows remain explicit budget exclusions; no selection used calculated agreement.

**No equilibrium calculation has been submitted.** Thermo-Calc's native Java engine crashes during TCPython initialization at setGesVersion(6), before database selection. Default startup failed repeatedly; an isolated 32 MB Java-thread-stack test and a pandas-import context test also failed. Error category 0xc00000fd is stack overflow; the originating native routine is unknown. The same signature predates the September 9 OS update, so that update is not established as the cause. Installed binaries, security configuration and thermodynamic settings were not altered.

## Evidence corrections

Mo and Nb/Co compositions are wt.%; the Ag/Si/Ti tables use their explicitly transcribed bases. Si A3 is resolved by the same alloy's explicit Cu70Cr5Si25 prose; two Nb rows and Ti B12 remain unresolved. Ti B15 has an equilibrium caveat and B16 is a binary control. Ti's ternary phase is tau1, not sigma1. Experimental Cr2Nb alone does not establish C15. GRCop remains qualitative materials context.

## Required questions

| No. | Current answer |
|---|---|
'''+ '\n'.join(f'| {i} | {ans[i]} |' for i in range(1,28))+'''

## Recovery and remaining work

The detailed runtime evidence is in 16_logs/RUNTIME_DIAGNOSIS.md and probe logs. A working Thermo-Calc 2025b/TC-Python initialization or vendor-supported runtime repair is required. After restoring it, run code_snapshot/probe_database.py (metadata only), then code_snapshot/run_r3.py with the bundled Thermo-Calc Python. The runner verifies frozen hashes and preserves per-attempt logs and exact conditions; no automatic run should be inferred from this document. Review runtime success before launching the 120 states. Complete raw/family/phase-composition comparisons, QA and figures after calculations. Do not treat pending rows as CALPHAD_FAILED or phase absence.

All 471 protected hashes and six original PDFs are unchanged. The R3 task is **not complete**. No Ni, model, OOD, recovery, screening or finalist change was made.
''')
manifest=[dict(relative_path=p.relative_to(O).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='STEP5K_R3_FILE_HASHES.csv']
pd.DataFrame(manifest).to_csv(O/'15_manifest/STEP5K_R3_FILE_HASHES.csv',index=False)
print(json.dumps(summary))
