"""Executable STEP5G protocol; no Ni properties or targets are read."""
import hashlib
import os
import platform
import shutil
import sys
import importlib.metadata
import numpy as np
import pandas as pd
import g_common as C

BLIND_ORDER = ['STEP5H-A_INPUTS_AND_PREDICTION_HASHES', 'STOP',
               'STEP5H-B_CALPHAD_REVEAL', 'PRIMARY_SCORE_AND_HASH',
               'SECONDARY_SCORE_AND_HASH', 'POST_BLIND_RECOVERY_ONLY']
BOOTSTRAP = dict(method='composition-level cluster bootstrap', replicates=1000,
                 seed=20260911, resample_unit='composition_id', interval='95% percentile',
                 temperatures_grouped=True, applied_to=['PRIMARY mean TVD'])
CALPHAD = dict(thermo_calc='2025b', tc_python='2025.2.30', database='TCCU6 Cu-Alloys v6.2',
               mode='equilibrium only', pressure_bar=1, amount_mol=1,
               composition_unit='wt.%', balance='Cu',
               phase_mapping={'FCC':'FCC_L12*','BCC':'BCC_B2*','C15':'C15_LAVES*',
                              'LIQUID':'LIQUID*','OTHER_SOLID':'all other non-liquid raw phases'},
               unknown_phase_policy='Preserve exact raw phase name; unmatched phases map to OTHER_SOLID. No post-reveal remapping.',
               partition='n_i,alpha = n_alpha * X_i^alpha; P_i^alpha = n_i,alpha / sum_beta n_i,beta; never volume fraction * mole fraction',
               failure_policy='Retain every formal row; failed target=NaN; no interpolation, imputation, fabrication or silent deletion. Report formal/converged/failed counts and failures by temperature and composition region.')

def split_roles(ids):
    order=sorted(ids, key=lambda x: hashlib.sha256(('NI|'+x+'|20260912').encode()).hexdigest())
    cal=set(order[:600])
    return {x:('FEWSHOT_CALIBRATION' if x in cal else 'RECOVERY_EVALUATION') for x in ids}

def valid_split(frame):
    if len(frame)!=2000 or not frame.composition_id.is_unique: return False
    expected=split_roles(frame.composition_id.tolist())
    return all(expected[i]==r for i,r in zip(frame.composition_id,frame.fewshot_role))

def nested_draws(calibration_ids):
    pool=np.array(sorted(calibration_ids))
    return {str(r):np.random.default_rng(C.child_seed(20260912,'FEWSHOT_SEQ_%d'%r)).permutation(pool)[:32].tolist() for r in range(1,4)}

def symbolic_scan(frame):
    expected={'family':'NI','X_element':'NI','split_role':'NI_BLIND_PROSPECTIVE',
              'composition_source':'STEP5G_prospective_Ni_blind'}
    hits=[]
    for k,v in expected.items():
        if k not in frame or set(frame[k].astype(str))!={v}: hits.append('invalid '+k)
    if 'composition_id' not in frame or not frame.composition_id.str.fullmatch(C.NI_ID_REGEX).all():
        hits.append('invalid composition_id')
    cols=[x for x in list(expected)+['composition_id'] if x in frame]
    return hits+C.ni_scan_frame(frame.drop(columns=cols),'SYMBOLIC_COMPOSITIONS')

def runtime():
    v={'python':sys.version,'OS':platform.platform(),'executable':sys.executable}
    for package in ['numpy','pandas','scikit-learn','torch','pyarrow','scipy','joblib']:
        v[package]=importlib.metadata.version(package)
    import torch
    v['CUDA']=torch.version.cuda
    v['device']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    v['matminer_mendeleev']='not imported in STEP5G'
    return v

def prepare_schema(feats):
    path=os.path.join(C.STEP5A,'02_features','FEATURE_DICTIONARY.csv')
    dictionary=pd.read_csv(path).set_index('feature_name')
    selected=dictionary.loc[feats].reset_index().rename(columns={'feature_name':'feature'})
    assert len(selected)==43 and selected.feature.tolist()==feats
    selected.insert(0,'order',range(1,44))
    selected['transform_rule']='Historical STEP5A feature formula, then role-specific train-only StandardScaler: (x-mean_)/scale_; no new transforms or feature selection'
    selected['historical_feature_list_sha256']=C.sha256_file(C.A_FEATLIST)
    selected.to_csv(os.path.join(C.D_FREEZE,'STEP5G_FEATURE_SCHEMA.csv'),index=False)
    sources=[path, C.A_FEATLIST, C.A_DEVFREEZE,
             os.path.join(C.STEP5A,'code_snapshot','32_step5a_features.py'),
             os.path.join(C.STEP5A,'code_snapshot','31_step5a_element_descriptors.py')]
    out=os.path.join(C.D_FREEZE,'feature_definition_sources'); os.makedirs(out,exist_ok=True)
    for p in sources: shutil.copy2(p,os.path.join(out,os.path.basename(p)))
    C.write_json(os.path.join(C.D_FREEZE,'FEATURE_SOURCE_HASHES.json'),{p:C.sha256_file(p) for p in sources})

def record_ids(frame,path):
    cols=['state_id','composition_id','family','sampling_domain','split_role']
    frame[cols].sort_values('state_id').to_csv(path,index=False)
    return C.sha256_file(path)

def verify_master(frame):
    master=pd.read_csv(C.A_SPLIT).set_index('composition_id')
    for col in ['family','sampling_domain','split_role']:
        assert frame[col].astype(str).tolist()==master.loc[frame.composition_id,col].astype(str).tolist(), 'STEP5A split mismatch '+col

def source_hashes():
    return {p:C.sha256_file(p) for p in [C.D_DIAG_DATA,C.F_NEWFEAT,C.F_NEWDATA,C.A_SPLIT]}

def write_manifest(include_logs=False):
    # Exclude this index, its digest sidecar and active logs; their bytes are mutable.
    out=os.path.join(C.D_MANIFEST,'STEP5G_FILE_HASHES.csv')
    excluded={out,os.path.join(C.D_MANIFEST,'STEP5G_FILE_HASHES.sha256')}
    rows=[]
    for base,dirs,files in os.walk(C.ROOT):
        dirs[:]=[d for d in dirs if d != '__pycache__' and (include_logs or d != '09_logs')]
        for name in files:
            p=os.path.join(base,name)
            if p in excluded or name.endswith('.pyc'): continue
            rows.append(dict(relative_path=os.path.relpath(p,C.ROOT),bytes=os.path.getsize(p),sha256=C.sha256_file(p)))
    pd.DataFrame(rows).sort_values('relative_path').to_csv(out,index=False)
    with open(os.path.join(C.D_MANIFEST,'STEP5G_FILE_HASHES.sha256'),'w') as f: f.write(C.sha256_file(out)+'  STEP5G_FILE_HASHES.csv\n')
    return len(rows)
