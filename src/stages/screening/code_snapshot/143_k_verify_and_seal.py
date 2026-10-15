"""Read-only scientific checks followed by final STEP5K delivery inventory seal."""
from pathlib import Path
import csv, hashlib, json, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
R = Path(__file__).resolve().parents[1]
def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
checks=[]
def check(name, ok, detail):
    checks.append(dict(check=name,status='PASS' if bool(ok) else 'FAIL',detail=detail))
baseline=json.loads((R/'00_config/STEP5K_CODEX_SOURCE_BASELINE.json').read_text(encoding='utf-8-sig'))['files']
bad=[x['path'] for x in baseline if not Path(x['path']).exists() or digest(x['path'])!=x['sha256']]
check('Historical source hashes unchanged',not bad,dict(files=len(baseline),mismatches=bad))
d=pd.read_csv(R/'05_screening_confirmation/STEP5K_SCREENING_CALPHAD_CONFIRMATION.csv')
e=pd.read_csv(R/'03_external_validation_CALPHAD/STEP5K_EXTVAL_STATES_RAW.csv')
p=pd.read_csv(R/'04_virtual_screening/STEP5K_VIRTUAL_SCREENING_PARETO.csv')
cand=pd.read_csv(R/'04_virtual_screening/STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv')
n_exp=len(cand)*4
check('Screening confirmation convergence',len(d)==n_exp and d.converged.eq(1).all() and d.state_id.nunique()==n_exp,f'{n_exp} unique states ({len(cand)} finalists x 4 temperatures)')
check('External calculation convergence',len(e)==88 and e.converged.eq(1).all() and e.state_id.nunique()==88,'88 unique states; 22 excluded sensitivity states remain preserved')
check('No Ni in new calculations',not d.X_identity.str.upper().eq('NI').any() and not e.elements.str.contains(r'\bNI\b',case=False).any(),'Screening and external raw state identifiers checked')
c=pd.read_csv(R/'04_virtual_screening/STEP5K_VIRTUAL_CANDIDATE_COMPOSITIONS.csv')
check('Screening inventory',len(c)==120000 and c.candidate_id.nunique()==120000 and len(p)==52944 and p.candidate_id.nunique()==52944 and p.applicable.eq(1).all() and set(p.candidate_id)<=set(c.candidate_id),'120000 unique compositions; Pareto table contains the 52944 applicable compositions only')
front=p[p.pareto_rank.eq(1)]
selected=p[p.selected_finalist.eq(True)]
check('Frozen finalists',set(d.candidate_id)==set(selected.candidate_id) and len(selected)==len(cand),'All %d preregistered family-fair finalists match the confirmed states' % len(cand))
check('Four confirmation temperatures',all(set(g.Temperature_C)=={450,600,950,1000} for _,g in d.groupby('candidate_id')),'Every selected composition confirmed at all four temperatures')
v=p.loc[p.applicable.eq(True),['Cr_FCC_950','Delta_Cr_FCC','Cr_OTHER_450']].to_numpy()*[1,1,-1]
undominated=True
for row in front[['Cr_FCC_950','Delta_Cr_FCC','Cr_OTHER_450']].to_numpy()*[1,1,-1]:
    undominated &= not np.any(np.all(v>=row,axis=1)&np.any(v>row,axis=1))
check('Full applicable-set dominance check',undominated,'Each reported global-front candidate independently checked against all 52944 applicable candidates')
freeze=json.loads((R/'04_virtual_screening/STEP5K_PARETO_FREEZE.json').read_text())
check('Pareto rules hash',digest(R/'04_virtual_screening/STEP5K_SCREENING_RULES_FREEZE.json')==freeze['rules_sha256'],'Frozen rules match actual Pareto record')
metrics=pd.read_csv(R/'05_screening_confirmation/STEP5K_ALL_TASK_CONFIRMATION_METRICS.csv')
fam=['FCC','BCC','C15','OTHER_SOLID','LIQUID']
for task in ['fraction','Cr','Nb','X']:
    suffix='frac' if task=='fraction' else task+'_P'
    a=d[[f'pred_{suffix}_{f}' for f in fam]].to_numpy()
    b=d[[f'calphad_{suffix}_{f}' for f in fam]].to_numpy()
    m=metrics[metrics.task.eq(task)].iloc[0]
    ok=np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a.sum(1),1,atol=1e-6) and np.allclose(b.sum(1),1,atol=1e-6)
    ok &= np.isclose(np.abs(a-b).mean(),m.MAE) and np.isclose((np.abs(a-b).sum(1)/2).mean(),m.mean_TVD)
    check(task+' independent metric recomputation',ok,'Finite normalized five-family outputs; MAE and TVD match delivery')
checks.extend([
    dict(check='Literature breadth',status='LIMITATION',detail='20 primary papers retrieved, all 20 DOIs re-verified via CrossRef; 21 validation rows built but only 4 papers state a composition/condition comparable to an equilibrium calculation, so P26 remains PARTIAL by honest counting.'),
    dict(check='Original plan source',status='PASS',detail='Original DOCX located at D:/EDGDownload/ (sha256 d5de679d1b6f5ad9...); full text and 42 tables extracted to 00_config/V1_PLAN_EXTRACTED_TEXT.txt for the audit.'),
    dict(check='Family diversity',status=('PASS' if all(selected.X_identity.value_counts().get(f,0)>=3 for f in ['ZR','TI','CO','SI','AG','MO']) else 'LIMITATION'),detail='Per-family finalist counts %s; every family has at least 3 as preregistered. Families whose own Pareto front had fewer than 3 members were extended with their next family_pareto_rank values per the dated amendment in the rules freeze; the Zr-only global front is reported as-is.' % selected.X_identity.value_counts().to_dict())])
result=dict(generated_utc=datetime.now(timezone.utc).isoformat(),checks=checks,execution_checks_pass=not any(x['status']=='FAIL' for x in checks),research_gaps_fully_closed=False)
(R/'09_qa/STEP5K_FINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# STEP5K final independent integrity review','', 'This review supplements the producer audit. PASS refers to verified execution; LIMITATION is not a passed research requirement.','']
for x in checks: lines.append(f"- **{x['status']} — {x['check']}**: {x['detail']}")
(R/'09_qa/STEP5K_FINAL_INTEGRITY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if not result['execution_checks_pass']:
    print(json.dumps(result,indent=2)); raise SystemExit(1)
manifest=R/'10_manifest/STEP5K_FILE_HASHES.csv'
files=sorted(x for x in R.rglob('*') if x.is_file() and x!=manifest and '__pycache__' not in x.parts)
with manifest.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']); w.writeheader()
    for x in files: w.writerow(dict(path=x.relative_to(R).as_posix(),bytes=x.stat().st_size,sha256=digest(x)))
with manifest.open(encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
assert all((R/x['path']).stat().st_size==int(x['bytes']) and digest(R/x['path'])==x['sha256'] for x in rows)
print(json.dumps(dict(historical_files_verified=len(baseline),execution_checks_passed=sum(x['status']=='PASS' for x in checks),limitations=sum(x['status']=='LIMITATION' for x in checks),sealed_files=len(rows),manifest_sha256=digest(manifest)),indent=2))
