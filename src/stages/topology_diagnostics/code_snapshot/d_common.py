"""STEP5D: independent read-only input access and reproducible diagnostics."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
C = ROOT.parent / 'STEP5C_6X_LOEO_V1'
A = ROOT.parents[1] / '04_ML/STEP5A_JMST_ML_FOUNDATION_V2'
FULL = ROOT.parents[1] / '03_full_CALPHAD/FULL_V1'
E = ['ZR','TI','CO','SI','AG','MO']
F = ['FCC','BCC','C15','OTHER_SOLID','LIQUID']
TASKS = ['fraction','cr_partition','nb_partition','x_partition']
PROPS = ['AtomicNumber','AtomicWeight','CovalentRadius','Electronegativity','VEC','MeltingT','FirstIonizationEnergy','ElectronAffinity','AtomicVolume_GS','MetallicRadius','Density','MendeleevNumber','CohesiveEnergy']
DIRS = ['00_config','01_preregistration','02_frozen_inputs','03_variance_decomposition','04_element_manifold','05_topology_analysis','06_boundary_analysis','07_model_bias_diagnostic','08_distance_uncertainty','09_figures','10_qa','11_manifest','code_snapshot']
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def js(p,x):
    p=ROOT/p; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else str(o)),encoding='utf-8')
def csv(p,x):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
    (x if isinstance(x,pd.DataFrame) else pd.DataFrame(x)).to_csv(p,index=False)
def freeze(e): return json.loads((C/f'11_manifest/FOLD_{e}_FREEZE.json').read_text())
def cols(task,prefix='true'): return [f'{prefix}_{task}_{f}' for f in F]
def tvd(y,p): return .5*np.abs(np.asarray(y)-np.asarray(p)).sum(axis=1)
def project(p,fallback):
    p=np.clip(np.asarray(p,dtype=float),0,None); s=p.sum(axis=1,keepdims=True)
    return np.where(s>0,p/np.where(s>0,s,1),fallback)
def predictions(e):
    p=pd.read_parquet(C/f'05_predictions/FOLD_{e}_HELDOUT_PREDICTIONS.parquet')
    assert set(p.family)=={e} and p.state_id.is_unique
    return p
def data():
    """Materialize only Domain-A six-element rows with parquet predicate pushdown.
    Baseline excluded for all tasks to make eta-squared comparisons commensurate.
    No source modules imported; no primary source write operations.
    """
    feats=sorted({f for e in E for f in freeze(e)['selected_features']})
    meta=['state_id','composition_id','family','sampling_domain','split_role']
    filt=[('sampling_domain','==','A'),('family','in',E)]
    x=pd.read_parquet(A/'02_features/FULL_V1_CANDIDATE_FEATURES.parquet',columns=list(dict.fromkeys(meta+feats+['Cr_wt_pct','Nb_wt_pct','X_wt_pct'])),filters=filt)
    target={t:([f'{f}_volfrac' for f in F] if t=='fraction' else [f'{dict(cr_partition="CR",nb_partition="NB",x_partition="X")[t]}_P_{f}' for f in F]) for t in TASKS}
    yy=pd.read_parquet(FULL/'03_processed/FULL_V1_ML_READY_CONVERGED.parquet',columns=['state_id']+sum(target.values(),[]),filters=filt)
    yy=yy.rename(columns={s:d for t in TASKS for s,d in zip(target[t],cols(t))})
    d=x.merge(yy,on='state_id',validate='one_to_one')
    assert d.state_id.is_unique and set(d.family)==set(E) and set(d.sampling_domain)=={'A'}
    assert not d.split_role.astype(str).str.contains('ANCHOR',case=False).any()
    return d
def roles(d,e):
    m=pd.read_csv(C/'02_fold_manifests/LOEO_MASTER_FOLD_MANIFEST.csv')
    m=m[m.fold==f'FOLD_{e}'].set_index('composition_id').loeo_role
    return d.composition_id.map(m)
def rho(x,y):
    from scipy.stats import spearmanr
    return float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else np.nan
def metrics(y,p,fallback):
    from sklearn.metrics import f1_score
    v=tvd(y,p); b=tvd(y,fallback).mean()
    return dict(mean_tvd=float(v.mean()),median_tvd=float(np.median(v)),p95_tvd=float(np.quantile(v,.95)),topology_accuracy=float((y.argmax(1)==p.argmax(1)).mean()),topology_macro_f1=float(f1_score(y.argmax(1),p.argmax(1),labels=range(5),average='macro',zero_division=0)),trivial_tvd=float(b),skill_vs_trivial=float(b-v.mean()),relative_skill=float(1-v.mean()/b))

