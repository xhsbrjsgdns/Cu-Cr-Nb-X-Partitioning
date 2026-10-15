"""Prepare amended protocol and input-only diagnostics, without reading B targets."""
import os, pickle, shutil
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from step5b_core import *

def main():
    assert not (O/'01_preregistration/DOMAIN_B_FIRST_OPEN_RECORD.json').exists()
    freeze=load_json(A/'09_manifest/DEVELOPMENT_FREEZE_V1.json')
    features=freeze['features']; cf=[f for f in features if f!='Temperature_C']
    assert len(features)==43 and len(cf)==42
    # Preserve the original strict-protocol STOP documents, never rewrite their history.
    for name in ['STEP5B_DOMAIN_B_OOD_REPORT.md','STEP5B_DOMAIN_B_OOD_SUMMARY.json','STEP5B_LEAKAGE_AND_SEAL_AUDIT.md']:
        src=O/'07_qa'/name; dst=O/'08_logs'/('ORIGINAL_STRICT_STOP_'+name)
        if src.exists() and not dst.exists(): shutil.copyfile(src,dst)
    checks=[]
    def check(p,expected=None):
        h=sha(p); assert expected is None or h==expected,str(p)
        checks.append(dict(path=str(p),sha256=h,status='PASS',expected_sha256=expected));return h
    fh=check(A/'09_manifest/DEVELOPMENT_FREEZE_V1.json')
    for name,key in [('PRIMARY_FEATURE_LIST.txt','feature_list_sha256'),('PCC_REMOVAL_LOG.csv','pcc_removal_sha256'),('NEAR_ZERO_VARIANCE_REMOVAL.csv','nzv_removal_sha256'),('EXACT_REDUNDANCY_LOG.csv','exact_redundancy_sha256')]: check(A/'02_features'/name,freeze[key])
    for x in load_json(A/'09_manifest/CODE_SNAPSHOT_MANIFEST_STEP5A_FINAL.json')['files']:check(A/'code_snapshot'/x['filename'],x['sha256'])
    assert (A/'02_features/PRIMARY_FEATURE_LIST.txt').read_text().splitlines()==features
    assert list(freeze['tasks']['presence']['thresholds'].values())==THRESHOLDS.tolist()
    splits=pd.read_csv(A/'03_splits/COMPOSITION_SPLIT_MASTER_V1.csv')
    assert splits.composition_id.is_unique
    roles=splits.set_index('composition_id').split_role
    candidate_cols=list(dict.fromkeys(['state_id','composition_id','family','sampling_domain','split_role']+RANGES+features))
    cand=pd.read_parquet(A/'02_features/FULL_V1_CANDIDATE_FEATURES.parquet',columns=candidate_cols)
    assert cand.state_id.is_unique and set(cand.sampling_domain)=={'A'}
    assert (cand.split_role==cand.composition_id.map(roles)).all()
    assert not set(cand.composition_id)&set(splits.loc[splits.sampling_domain!='A','composition_id'])
    cache=A/'02_features/COMPOSITION_FEATURES_ALL50K.parquet'
    check(cache);check(A/'01_feature_sources/ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.parquet')
    cols=list(dict.fromkeys(['composition_id','family','sampling_domain','split_role']+RANGES+cf))
    b=pd.read_parquet(cache,columns=cols,filters=[('sampling_domain','==','B')])
    acomp=pd.read_parquet(cache,columns=cols,filters=[('sampling_domain','==','A')])
    assert b.composition_id.is_unique and len(b)==10000
    assert set(b.composition_id)==set(splits.loc[splits.sampling_domain=='B','composition_id'])
    assert set(b.family)=={'BASELINE','ZR','TI','CO','SI','AG','MO'}
    assert (b.split_role==b.composition_id.map(roles)).all()
    auniq=cand.drop_duplicates('composition_id').set_index('composition_id')
    ac=acomp.set_index('composition_id').loc[auniq.index]
    np.testing.assert_array_equal(auniq[cf].to_numpy(),ac[cf].to_numpy())
    assert not cand.groupby('composition_id')[cf].nunique().gt(1).any().any()
    train=acomp[acomp.split_role=='DEV_TRAIN'].copy(); iid=acomp[acomp.split_role=='IID_TEST'].copy()
    assert len(train)==24500 and len(iid)==5250
    # Metadata-only column projection. No target or phase diagnostic is requested.
    formal=pd.read_parquet(FULL/'03_processed/FULL_V1_ML_TARGET_ALL.parquet',columns=['state_id','composition_id','Temperature_C','sampling_domain'],filters=[('sampling_domain','==','B')])
    assert formal.state_id.is_unique and set(formal.sampling_domain)=={'B'}
    assert formal.composition_id.nunique()==len(b)
    temps=sorted(formal.Temperature_C.unique().tolist())
    assert len(temps)==11 and formal.groupby('composition_id').Temperature_C.nunique().eq(11).all()
    bf=formal.drop(columns='sampling_domain').merge(b,on='composition_id',validate='m:1')
    meta=[c for c in bf if c not in features]
    bf=bf[meta+features];validate_matrix(bf,features)
    assert bf[features].shape[1]==43
    # This scaler is solely a diagnostic distance transform, never a model transform.
    scaler=StandardScaler().fit(validate_matrix(train,cf))
    z=scaler.transform(train[cf].to_numpy(float));tree=cKDTree(z)
    for name,d in [('DOMAIN_B',b),('IID',iid),('DEV_TRAIN',train)]:
        dz=scaler.transform(validate_matrix(d,cf))
        dist,ind=tree.query(dz,k=6 if name=='DEV_TRAIN' else 5,workers=8)
        if name=='DEV_TRAIN':
            # Remove the query's own index explicitly, retaining true duplicate neighbors.
            dist=np.array([row[idx!=j][:5] for j,(row,idx) in enumerate(zip(dist,ind))])
        d['nn1_distance']=dist[:,0];d['nn5_mean_distance']=dist[:,:5].mean(1)
        d['outside_train_range_feature_count']=((d[cf].to_numpy()<train[cf].min().to_numpy())|(d[cf].to_numpy()>train[cf].max().to_numpy())).sum(1)
        d.to_parquet(O/f'02_inputs/{name}_COMPOSITION_CONTEXT.parquet',index=False)
    distance_cols=['composition_id','family']+RANGES+['nn1_distance','nn5_mean_distance','outside_train_range_feature_count']
    b[distance_cols].to_parquet(O/'05_shift_analysis/DOMAIN_B_OOD_DISTANCE.parquet',index=False)
    bf=bf.merge(b[distance_cols[:1]+distance_cols[-3:]],on='composition_id',validate='m:1')
    # Keep the model features together, in the exact frozen order; all metadata outside.
    bf=bf[[c for c in bf if c not in features]+features]
    bf.to_parquet(O/'02_inputs/DOMAIN_B_FEATURES_FROZEN43.parquet',index=False)
    shift=feature_shift(cand[cand.split_role=='DEV_TRAIN'],bf,features)
    shift.to_csv(O/'05_shift_analysis/FEATURE_SHIFT_A_TO_B.csv',index=False)
    bins={c:np.unique(np.quantile(b[c],[0,.25,.5,.75,1])).tolist() for c in RANGES+['nn1_distance']}
    assert all(len(v)>=2 for v in bins.values())
    for c,e in bins.items():assert apply_bins(b[c],e).notna().all()
    # Frozen inference smoke test uses only already-open IID outputs, never B labels.
    iidpred=pd.read_parquet(A/'05_results/IID_TEST_PREDICTIONS.parquet')
    assert set(iidpred.composition_id)<=set(iid.composition_id)
    smoke=cand[cand.split_role=='IID_TEST'].groupby('family',group_keys=False).head(22)
    smoke=smoke.merge(iidpred,on=['state_id','composition_id','family'],suffixes=('','_old'),validate='1:1')
    model_audit={}
    for t in TASKS:
        tf=freeze['tasks'][t]; path=A/f'04_models/final/{t}/model_state.pkl';check(path,tf['artifact_sha256'])
        with open(path,'rb') as f: blob=pickle.load(f)
        trivial=np.asarray(blob['trivial'],float)
        if t!='presence':np.testing.assert_array_equal(trivial,np.asarray(tf['simplex_fallback_row']))
        st=smoke[smoke.family!='BASELINE'] if t=='x_partition' else smoke
        _,predict=freeze_mod.build_final_model(tf['model_family'],t,tf['hyperparameters'])
        pred=predict(blob['state'],validate_matrix(st,features))
        oldcols=[f'{f}_presence_prob' for f in FAMILIES] if t=='presence' else [f'{f}_{TAGS[t]}_raw' for f in FAMILIES]
        err=float(np.max(np.abs(pred-st[oldcols].to_numpy(float))))
        assert err<=1e-12,(t,err)
        dims=[]
        for kind,m in blob['state']:
            if kind!='const':
                # CatBoost pickle retains native feature_names_ but its sklearn
                # n_features_in_ accessor is 0 after deserialization in this runtime.
                n=len(m.feature_names_) if tf['model_family']=='CatBoost' else m.get_booster().num_features()
                assert n==43; dims.append(int(n))
                actual=m.get_params()
                for k,v in tf['hyperparameters'].items():assert actual[k]==v,(t,k,actual[k],v)
        model_audit[t]=dict(sha256=sha(path),dimensions=dims,scaler='None: frozen tree models consume raw frozen43 matrix',trivial_vector=trivial.tolist(),iid_smoke_max_abs_delta=err)
    assert not (O/'03_predictions/DOMAIN_B_OOD_PREDICTIONS.parquet').exists()
    audit=dict(status='PASS_UNDER_AMENDMENT',timestamp_utc=now(),strict_never_read_status='FAIL_RETAINED',artifact_hashes=checks,feature_order=features,feature_count=43,cache_matches_all_A_compositions=True,metadata_only_B_preparation=True,model_audit=model_audit,role_counts=cand.groupby('split_role').size().to_dict(),B_compositions=len(b),formal_feature_rows=len(bf),temperatures=temps,source_scope='Inspected frozen code, metadata and recorded run logs; no claim about unrecorded external executions')
    write_json(O/'07_qa/AMENDED_PREOPEN_AUDIT.json',audit)
    write_json(O/'00_config/DISTANCE_SCALER_DEV_TRAIN_ONLY.json',dict(features=cf,mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,n_compositions=len(train),model_scaler=False))
    write_json(O/'00_config/PREOPEN_CONFIG.json',dict(features=features,bins=bins,thresholds=THRESHOLDS,seed=20260905,n_bootstrap=1000,temperatures=temps,model_audit=model_audit))
    print('PREPARE PASS: features, bins, diagnostic distances, 5 frozen IID inference checks',flush=True)

if __name__=='__main__':main()
