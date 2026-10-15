from pathlib import Path
import pandas as pd, numpy as np, json, hashlib, datetime
from scipy.spatial import cKDTree
ROOT=Path('E:/Cu_CALPHAD_ML')
OUT=ROOT/'08_FINAL_MANUSCRIPT/STEP5K_R2_SOURCE_PROVENANCE_REPAIR'
R1=ROOT/'08_FINAL_MANUSCRIPT/STEP5K_R1_OA_LITERATURE_MINOR_PATCHES'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
refs=pd.read_csv(R1/'13_output_repair/STEP5K_R1_OA_REFERENCES.csv').fillna('NA').set_index('source_id')
rows=[]
for p in sorted(R1.rglob('*')):
    if p.is_file() and p.suffix.lower() in ['.pdf','.html','.xml','.txt','.md']:
        old=p.stem if p.stem in refs.index else 'NA'
        r=refs.loc[old] if old!='NA' else {}
        rows.append(dict(local_source_id=f'LOCAL_{len(rows)+1:04d}',original_filename=p.name,full_path=str(p),extension=p.suffix,file_size=p.stat().st_size,sha256=sha(p),old_source_id=old,old_claimed_title=r.get('title','NA'),old_claimed_doi=r.get('doi','NA'),document_role='FULLTEXT_PDF' if p.suffix=='.pdf' else ('DERIVED_FULLTEXT' if p.stem in refs.index else ('LANDING_PAGE' if p.suffix=='.html' else 'WORKFLOW_OR_REPORT_NOT_PRIMARY_FULLTEXT'))))
pd.DataFrame(rows).to_csv(OUT/'01_source_inventory/STEP5K_R2_LOCAL_SOURCE_INVENTORY.csv',index=False)
c=pd.read_csv(ROOT/'03_full_CALPHAD/FULL_V1/01_compositions/FULL_V1_COMPOSITIONS.csv')
geometry={}
for x,g in c.groupby('X_element'):
    cols=['S_CrNb_wt_pct','R_CrNb_atomic_actual','X_wt_pct']
    a=g[cols].to_numpy(float); span=np.ptp(a,axis=0)
    u=np.unique(a,axis=0); ds=cKDTree(u/span).query(u/span,k=2)[0][:,1]
    q=np.quantile(ds,[.05,.1,.5,.9,.95])
    geometry[x]={'n':len(g),'min':a.min(axis=0).tolist(),'range':span.tolist(),'nn_quantiles_05_10_50_90_95':q.tolist(),'tight_threshold':float(q[0]/2)}
(OUT/'00_config/matching_geometry.json').write_text(json.dumps(geometry,indent=2))
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
rule=f'''# STEP5K-R2 matching rule freeze

Frozen at {stamp}, before reading agreement with CALPHAD phases.

Input: all {len(c)} frozen compositions; SHA256 {sha(ROOT/'03_full_CALPHAD/FULL_V1/01_compositions/FULL_V1_COMPOSITIONS.csv')}.
The FULL_V1_CONFIG.json composition_precision_policy fixes 4 decimal wt.% as actual Thermo-Calc input. Exact matching compares all Cu/Cr/Nb/X wt.% rounded to 4 decimal places and exact temperature in degrees C, with a numeric tolerance of 1e-8 C solely for floating arithmetic. No temperature rounding from K is silently applied.

For each identity separately, normalize S=Cr+Nb, R=(Cr/M_Cr)/(Nb/M_Nb), X by observed FULL_V1 full identity ranges (all domains). Euclidean distance in these coordinates is used. The tight cutoff is one half the fifth percentile of the positive nearest-neighbor distance among distinct frozen compositions. This intentionally conservative rule is geometry-only, independent of experimental phase labels or agreement. Tight context also requires the experimental point inside all three observed coordinate bounds and an exact frozen temperature. Exact composition is always checked first, including cases with undefined R. For Nb=0 and Cr>0, R is undefined/infinite; never replace it with a finite cap. Such unmatched chemistry has NO_MATCH in S/R/X space. A diagnostic closest composition in normalized elemental wt.% space is recorded but does not alter this decision. Ambiguous compositions or unspecified basis are NO_MATCH without calculation. Direct comparison requires successful state convergence, not just a composition row.

EXACT_MATCH: exact input chemistry and temperature (convergence recorded separately). TIGHT_NEIGHBOR: qualitative context only. DISTANT_NEIGHBOR: never validation evidence. NO_MATCH: no valid finite descriptor comparison, unknown input, absent identity, or absent temperature; may motivate a separately justified literature-specific run. No threshold may be changed after inspecting phase agreement.

No production scaler is used, fitted, or changed. Geometric normalization is a descriptive audit computation only.

## Identity thresholds
'''
for x,v in geometry.items():rule+=f"\n- {x}: cutoff={v['tight_threshold']:.9g}; NN quantiles={v['nn_quantiles_05_10_50_90_95']}; ranges={v['range']}"
(OUT/'06_fullv1_matching/STEP5K_R2_MATCHING_RULE_FREEZE.md').write_text(rule,encoding='utf8')
print('Inventory',len(rows),'geometry',geometry)
