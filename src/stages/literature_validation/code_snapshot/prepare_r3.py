from pathlib import Path
import json,hashlib,shutil,datetime
R=Path('E:/Cu_CALPHAD_ML'); O=Path(__file__).resolve().parents[1]
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
baseline=json.loads((R/'08_FINAL_MANUSCRIPT/STEP5K_R2_SOURCE_PROVENANCE_REPAIR/00_config/protected_before.json').read_text())
assert all(sha(r['path'])==r['sha256'] for r in baseline)
target=O/'00_config/protected_before.json'
if not target.exists():target.write_text(json.dumps(baseline,indent=2))
sources=[]
for p in sorted((R/'文献').glob('*.pdf')):
 dest=O/'01_source_pdfs'/p.name
 if not dest.exists():shutil.copy2(p,dest)
 sources.append(dict(original_path=str(p),copy_path=str(dest),sha256=sha(p),copy_sha256=sha(dest)))
(O/'00_config/SUPPLIED_FILES.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf8')
helpers={str(p):sha(p) for p in [R/'02_pilot_2000/code/dry_run_lib.py',R/'02_pilot_2000/code/fivefamily.py']}
(O/'00_config/helper_hashes.json').write_text(json.dumps(helpers,indent=2))
print('Protected baseline verified:',len(baseline),'Supplied PDFs:',len(sources))
