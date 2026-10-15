"""Shared read-only evaluation helpers; no training entry point is called."""
from pathlib import Path
import hashlib, importlib.util, json, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT=Path('E:/Cu_CALPHAD_ML')
A=ROOT/'04_ML/STEP5A_JMST_ML_FOUNDATION_V2'
O=ROOT/'05_OOD/STEP5B_DOMAIN_B_OOD_V1'
FULL=ROOT/'03_full_CALPHAD/FULL_V1'
FAMILIES=['FCC','BCC','C15','OTHER_SOLID','LIQUID']
TASKS=['presence','fraction','cr_partition','nb_partition','x_partition']
TAGS={'fraction':'frac','cr_partition':'CR_P','nb_partition':'NB_P','x_partition':'X_P'}
TARGETS={'presence':[f'{f}_presence' for f in FAMILIES], 'fraction':[f'{f}_volfrac' for f in FAMILIES],**{t:[f'{tag}_{f}' for f in FAMILIES] for t,tag in TAGS.items() if t!='fraction'}}
RANGES=['S_CrNb_wt_pct','R_CrNb_atomic_actual','X_wt_pct']
THRESHOLDS=np.array([.5,.37,.5,.32,.42])

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''): h.update(chunk)
    return h.hexdigest()
def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(float,np.floating)): return float(v) if np.isfinite(v) else None
    if isinstance(v,(np.bool_,)): return bool(v)
    return v
def write_json(p,v): Path(p).write_text(json.dumps(clean(v),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

# Read exact verified STEP5A functions; importing these files never invokes main().
sys.dont_write_bytecode=True
sys.path.insert(0,str(A/'code_snapshot'))
import step5a_common as C
freeze_mod=load_module('step5a_frozen_prediction',A/'code_snapshot/39_step5a_development_freeze.py')

def validate_matrix(df,features):
    assert len(features)==len(set(features)), 'duplicate requested feature'
    assert not df.columns.duplicated().any(), 'duplicate table column'
    assert all(not any(tok in f for tok in C.FORBIDDEN_FEATURE_TOKENS) for f in features)
    matrix=df.loc[:,features]
    assert matrix.columns.tolist()==features
    x=matrix.to_numpy(np.float64)
    assert np.isfinite(x).all(), 'nonfinite model feature'
    return x

def apply_bins(values,edges):
    edges=np.asarray(edges,float).copy()
    assert len(edges)>=2 and np.all(np.diff(edges)>0)
    return pd.cut(values,edges,labels=[f'Q{i+1}' for i in range(len(edges)-1)],include_lowest=True,right=True)

def feature_shift(train,b,features):
    rows=[]
    for f in features:
        x=train[f].to_numpy(float); y=b[f].to_numpy(float); sd=x.std(ddof=0)
        delta=y.mean()-x.mean()
        rows.append(dict(feature=f,train_mean=x.mean(),train_std=sd,train_min=x.min(),train_max=x.max(),B_mean=y.mean(),B_std=y.std(),standardized_mean_shift=delta/sd if sd>0 else (0. if delta==0 else np.nan),constant_train_feature=sd==0,fraction_outside_train_range=((y<x.min())|(y>x.max())).mean(),ks_statistic=ks_2samp(x,y,method='asymp').statistic))
    return pd.DataFrame(rows)

def task_arrays(df,task):
    d=df[df.family!='BASELINE'] if task=='x_partition' else df
    y=d[[c+'_true' for c in TARGETS[task]]].to_numpy(float)
    assert len(d)>0 and np.isfinite(y).all()
    if task=='presence':
        raw=d[[f'{f}_presence_prob' for f in FAMILIES]].to_numpy(float)
        proj=d[[f'{f}_presence_pred' for f in FAMILIES]].to_numpy(float)
    else:
        raw=d[[f'{f}_{TAGS[task]}_raw' for f in FAMILIES]].to_numpy(float)
        proj=d[[f'{f}_{TAGS[task]}_proj' for f in FAMILIES]].to_numpy(float)
    assert np.isfinite(raw).all() and np.isfinite(proj).all()
    return d,y,raw,proj

def metrics(df,task,model,baseline=None):
    d,y,raw,proj=task_arrays(df,task)
    th=THRESHOLDS
    if baseline is not None:
        raw=np.tile(np.asarray(baseline),(len(d),1));proj=raw.copy();th=np.full(5,.5)
    row=dict(task=task,model=model,n_rows=len(d),n_compositions=d.composition_id.nunique())
    if task=='presence':
        m=C.classification_metrics(y.astype(int),raw,th)
        row.update({k:v for k,v in m.items() if k!='per_label'})
        per=[dict(task=task,model=model,target=f,**v) for f,v in m['per_label'].items()]
    else:
        m=C.regression_metrics(y,raw,proj)
        rename={'mean_tvd_proj':'mean_tvd','median_tvd_proj':'median_tvd','p95_tvd_proj':'p95_tvd','macro_mae_proj':'macro_mae','macro_rmse_proj':'macro_rmse'}
        row.update({rename.get(k,k):v for k,v in m.items() if k!='per_family'})
        row['proj_sum_violation_mean']=float(np.abs(proj.sum(1)-1).mean())
        row['simplex_fallbacks']=int((np.clip(raw,0,None).sum(1)==0).sum()) if baseline is None else 0
        # STEP5A calls sqrt(overall MSE) macro RMSE; retain it for exact comparison,
        # additionally expose mean per-family RMSE to avoid terminology ambiguity.
        row['mean_per_family_rmse']=float(np.sqrt(((proj-y)**2).mean(0)).mean())
        per=[dict(task=task,model=model,target=f,**v) for f,v in m['per_family'].items()]
    return row,per

def composition_errors(df):
    out=df[['composition_id','family']+RANGES+['nn1_distance','nn5_mean_distance','outside_train_range_feature_count']].drop_duplicates('composition_id').set_index('composition_id')
    for t in TASKS:
        d,y,raw,p=task_arrays(df,t)
        e=(y!=p).mean(1) if t=='presence' else .5*np.abs(y-p).sum(1)
        z=pd.DataFrame({'composition_id':d.composition_id.values,'error':e})
        out[t]=z.groupby('composition_id').error.mean()
    return out.reset_index()
