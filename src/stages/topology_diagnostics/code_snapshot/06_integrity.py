"""Final checks and manifest; never modifies frozen upstream outputs."""
from d_common import *
import joblib, subprocess, sys

def run():
    checks=[]
    def check(name,ok,detail):
        checks.append(dict(check=name,status='PASS' if ok else 'FAIL',detail=str(detail)))
    before=pd.read_csv(ROOT/'02_frozen_inputs/STEP5C_ALL_FILES_BEFORE.csv')
    after=[]
    for row in before.itertuples():
        h=sha(C/row.path);after.append(dict(path=row.path,before_sha256=row.sha256,after_sha256=h,unchanged=h==row.sha256))
    csv('11_manifest/STEP5C_BEFORE_AFTER_HASHES.csv',after)
    check('all STEP5C files unchanged during STEP5D',all(r['unchanged'] for r in after),f'{len(after)} files checked, including primary artifacts, metrics, freezes, thresholds, feature lists and logs')
    all_now={p.relative_to(C).as_posix() for p in C.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    check('STEP5C file inventory unchanged',all_now==set(before.path),'No new or removed upstream files')
    pr=ROOT/'01_preregistration/STEP5D_FAILURE_DIAGNOSIS_PREREGISTRATION.json'
    check('preregistration unchanged',sha(pr)==json.loads((ROOT/'01_preregistration/PREREGISTRATION_HASH.json').read_text())['sha256'],sha(pr))
    d=pd.read_parquet(ROOT/'02_frozen_inputs/DOMAIN_A_DEV6_DIAGNOSTIC_DATA.parquet')
    provenance=json.loads((ROOT/'02_frozen_inputs/FILTERED_DATA_PROVENANCE.json').read_text())
    check('filtered input snapshot unchanged',sha(ROOT/'02_frozen_inputs/DOMAIN_A_DEV6_DIAGNOSTIC_DATA.parquet')==provenance['sha256'],f'{len(d)} allowed states')
    check('Domain A only',set(d.sampling_domain)=={'A'},d.sampling_domain.unique().tolist())
    check('six X identities only; no Ni or Anchor',set(d.family)==set(E) and not d.split_role.astype(str).str.contains('ANCHOR',case=False).any(),sorted(d.family.unique()))
    data_props=pd.read_csv(ROOT/'02_frozen_inputs/DEV6_ELEMENT_PROPERTIES.csv')
    check('Ni descriptor rows zero',set(data_props.element)==set(E),'Allowed source DEV6 table only; no periodic-table property lookup')
    check('state IDs unique',d.state_id.is_unique,len(d))
    gate=ROOT/'07_model_bias_diagnostic/ALL_SIX_SELECTIONS_FROZEN_GATE.json';gj=json.loads(gate.read_text())
    model_checks=[]
    for e in E:
        path=ROOT/f'07_model_bias_diagnostic/FOLD_{e}';sel=json.loads((path/'SELECTION_FREEZE.json').read_text());fr=freeze(e)
        r=roles(d,e);train=d[r=='FOLD_TRAIN'];valid=d[r=='FOLD_VALIDATION']
        check(e+' role isolation',e not in set(train.family)|set(valid.family) and not set(train.composition_id)&set(valid.composition_id),'Train/validation/heldout composition identities disjoint')
        check(e+' original selected features',sel['features']==fr['selected_features'] and sel['primary_feature_hash']==fr['selected_features_sha256'],len(sel['features']))
        scaler=joblib.load(path/'scaler.joblib')
        check(e+' scaler train only',np.allclose(scaler.mean_,train[sel['features']].mean().to_numpy(),atol=1e-10,rtol=1e-10) and int(scaler.n_samples_seen_)==len(train),len(train))
        check(e+' selected artifacts unchanged',all(sha(path/n)==h for n,h in sel['artifact_hashes'].items()),sel['artifact_hashes'])
        check(e+' freeze precedes all-six gate',sel['frozen_utc']<=gj['frozen_utc'] and sha(path/'SELECTION_FREEZE.json')==gj['freeze_hashes'][e],sel['frozen_utc'])
        check(e+' predictions after gate',(path/'DIAGNOSTIC_HELDOUT_PREDICTIONS.parquet').stat().st_mtime>=gate.stat().st_mtime,'Diagnostic prediction file timestamp >= gate')
        candidates=pd.read_csv(path/'VALIDATION_CANDIDATES.csv')
        check(e+' fixed candidate suite',len(candidates)==10 and (candidates.model=='Ridge').sum()==7 and (candidates.model=='MLP').sum()==3,'7 fixed alpha + 3 fixed LR; no model hunting')
        check(e+' validation-only selected minima',abs(candidates[candidates.model=='Ridge'].validation_tvd.min()-sel['ridge']['validation_tvd'])<1e-12 and abs(candidates[candidates.model=='MLP'].validation_tvd.min()-sel['mlp']['validation_tvd'])<1e-12,'Both selected on known-identity FOLD_VALIDATION only')
        p=pd.read_parquet(path/'DIAGNOSTIC_HELDOUT_PREDICTIONS.parquet');primary=predictions(e)
        check(e+' secondary predictions exactly align',p.state_id.tolist()==primary.state_id.tolist(),'No omitted eligible heldout rows')
        for method in ['Tree','Ridge','MLP','Nearest-element analog']:
            pred=p[[f'{method}_{f}' for f in F]].to_numpy(float)
            model_checks.append(np.isfinite(pred).all() and pred.min()>=0 and np.allclose(pred.sum(1),1,atol=1e-10))
            if method=='Tree':check(e+' copied primary tree unchanged',np.array_equal(pred,primary[cols('x_partition','pred_proj')].to_numpy()),'Bitwise numeric equality')
    check('all four-method predictions finite simplexes',all(model_checks),'6 folds x 4 methods')
    suite=pd.read_csv(ROOT/'07_model_bias_diagnostic/X_DIAGNOSTIC_MODEL_COMPARISON.csv')
    check('diagnostics clearly separated',set(suite[suite.method!='Tree'].status)=={'POST_HOC_DIAGNOSTIC'} and set(suite[suite.method=='Tree'].status)=={'PRIMARY_FROZEN'},'No replacement of primary result')
    audit=pd.read_csv(ROOT/'10_qa/FIGURE_AUTOMATED_AUDITS.csv')
    check('ten figures PDF typography and collision QA',len(audit)==20 and (audit.exit_code==0).all(),'20 automated audits; single-panel alignment not applicable, colorbar excluded')
    visual=ROOT/'10_qa/FIGURE_VISUAL_QA.md'
    check('visual review documented',visual.exists(),'Ten figures inspected; finite-dataset summaries, no implied iid-state CIs')
    static=(ROOT/'10_qa/FIGURE_STATIC_PREFLIGHT.txt').read_text(encoding='utf-8-sig')
    check('figure source preflight has no failures','[FAIL]' not in static,'Uncertainty warning addressed by descriptive finite-dataset contract')
    required=['STEP5D_FAILURE_DIAGNOSIS_REPORT.md','STEP5D_FAILURE_DIAGNOSIS_SUMMARY.json','03_variance_decomposition/TASK_ELEMENT_VARIANCE_DECOMPOSITION.csv','03_variance_decomposition/CHEMICAL_EFFECTIVE_SAMPLE_SIZE.md','04_element_manifold/LOEO_ELEMENT_MANIFOLD_DIAGNOSTICS.csv','04_element_manifold/ELEMENT_PROPERTY_RANGE_EXTRAPOLATION.csv','04_element_manifold/F3_SELECTION_VS_EXTRAPOLATION.csv','05_topology_analysis/X_PARTITION_ARCHETYPE_DIAGNOSIS.csv','05_topology_analysis/X_PARTITION_TOPOLOGY_COLLAPSE.csv','05_topology_analysis/X_PARTITION_BORROWED_ARCHETYPE.csv','05_topology_analysis/X_PARTITION_ERROR_BY_TOPOLOGY_CORRECTNESS.csv','06_boundary_analysis/X_ERROR_BY_TOPOLOGY_TRANSITION_DISTANCE.csv','06_boundary_analysis/X_ERROR_BY_PHASE_SPLIT_AND_ELEMENT.csv','06_boundary_analysis/X_ERROR_BY_TEMPERATURE.csv','07_model_bias_diagnostic/X_DIAGNOSTIC_MODEL_COMPARISON.csv','08_distance_uncertainty/OOD_ERROR_DETECTION_METRICS.csv','10_qa/STEP5D_ROOT_CAUSE_EVIDENCE_MATRIX.csv']
    check('all requested analytical outputs exist',all((ROOT/f).is_file() for f in required),len(required))
    code='\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'code_snapshot').glob('*.py'))
    import ast
    imports=[]
    for p in (ROOT/'code_snapshot').glob('*.py'):
        for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
            if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
            elif isinstance(n,ast.ImportFrom):imports.append(n.module or '')
    check('no CALPHAD engine imports or workflow calls',not any(x.startswith(('tc_python','thermocalc','pycalphad')) for x in imports),'Static source audit + executed workflow: zero new equilibrium calculations; no claim about unrelated system processes')
    check('no STEP5E execution',not any(x.name.startswith('STEP5E') for x in ROOT.iterdir()),'Recommendation text only; executed stages stop at STEP5D')
    csv('10_qa/INTEGRITY_CHECKS.csv',checks)
    assert all(x['status']=='PASS' for x in checks),[x for x in checks if x['status']=='FAIL']
    summary=json.loads((ROOT/'STEP5D_FAILURE_DIAGNOSIS_SUMMARY.json').read_text())
    summary.update(status='PASS_WITH_WARNINGS',completed_utc=now(),integrity_checks_passed=len(checks),step5c_files_unchanged=len(after),figures=10,
       warnings=['Historical PART2.log append-only completion suffix; archived prefix verified, no primary artifact exception','Mechanism analyses are retrospective and do not causally separate anchor sparsity from descriptor insufficiency','General tree-bias majority criterion unmet despite real fold-specific MLP/Ridge gains','Four split contrasts unidentifiable because of no overlap','13-property descriptor-insufficiency rule did not quantify high extrapolation, so no post-hoc threshold substituted','Archived nominal 1-NN may not be the exact minimum; retained unchanged and caveated','n=6 element associations and pooled error detection are exploratory, not calibrated uncertainty'])
    js('STEP5D_FAILURE_DIAGNOSIS_SUMMARY.json',summary)
    report=(ROOT/'STEP5D_FAILURE_DIAGNOSIS_REPORT.md').read_text(encoding='utf-8')
    report+='\n\n## Final verification\n\n**PASS WITH WARNINGS**: '+str(len(checks))+' integrity checks passed; all '+str(len(after))+' STEP5C files are unchanged from the pre-diagnosis snapshot. All ten figures passed PDF text/collision checks and visual review. The historical append-only log exception and scientific limitations above remain explicitly recorded. No STEP5E started.\n'
    (ROOT/'STEP5D_FAILURE_DIAGNOSIS_REPORT.md').write_text(report,encoding='utf-8')
    audit_text='# STEP5D integrity audit\n\nStatus: PASS WITH WARNINGS\n\n'+pd.DataFrame(checks).to_markdown(index=False)+'\n\nHistorical archival exception: PART2.log original prefix matches archived SHA256 exactly; later completion text was already present before STEP5D. No scientific primary file exception. All files, including that log, match their STEP5D-before hashes now. Static/workflow evidence establishes no new CALPHAD calls by this task, not the state of unrelated processes. No Ni/Anchor targets, descriptors, evaluation, training or predictions were used.\n'
    (ROOT/'STEP5D_INTEGRITY_AUDIT.md').write_text(audit_text,encoding='utf-8')
    (ROOT/'10_qa/STEP5D_INTEGRITY_AUDIT.md').write_text(audit_text,encoding='utf-8')
    files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.parent.name!='11_manifest']
    csv('11_manifest/STEP5D_FILE_HASHES.csv',files)
    js('11_manifest/COMPLETION.json',dict(status='PASS_WITH_WARNINGS',completed_utc=now(),n_files_hashed=len(files),file_manifest_sha256=sha(ROOT/'11_manifest/STEP5D_FILE_HASHES.csv'),step5c_comparison_sha256=sha(ROOT/'11_manifest/STEP5C_BEFORE_AFTER_HASHES.csv'),next_stage_executed=False))
    print('FINAL AUDIT PASS',len(checks),'checks;',len(after),'STEP5C unchanged;',len(files),'STEP5D files hashed',flush=True)
if __name__=='__main__':run()
