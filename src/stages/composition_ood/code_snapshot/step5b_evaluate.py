"""One prospective Domain-B evaluation under the approved amended seal protocol."""
import pickle
from step5b_core import *

def verify_gates():
    prereg=O/'01_preregistration/STEP5B_PREREGISTRATION_V1.json'
    seal=load_json(O/'09_manifest/PREOPEN_GATE.json')
    assert seal['status']=='PASS_UNDER_AMENDMENT'
    assert sha(prereg)==seal['preregistration_sha256']
    pre=load_json(prereg)
    assert pre['protocol_amendment_approved'] is True
    for entry in pre['frozen_artifacts']+pre['prepared_inputs']:
        assert sha(entry['path'])==entry['sha256'],entry['path']
    cp=O/'09_manifest/CODE_SNAPSHOT_MANIFEST_STEP5B.json'
    assert sha(cp)==pre['code_manifest_sha256']
    for entry in load_json(cp)['files']:
        assert sha(entry['path'])==entry['sha256'],entry['path']
    assert freeze_mod.lib_versions()==pre['package_versions']
    assert not (O/'03_predictions/DOMAIN_B_OOD_PREDICTIONS.parquet').exists()
    resume = (O/'01_preregistration/DOMAIN_B_FIRST_OPEN_RECORD.json').exists()
    if resume:
        # PATCH resume: the first opening already happened physically (targets read once)
        # but the run aborted in support-table binning BEFORE any prediction/metric file
        # was written (STEP5B_PATCH_01). Re-running reproduces the same single evaluation;
        # the original first-open record is preserved, never overwritten.
        assert (O/'09_manifest/STEP5B_PATCH_01.md').exists(), 'resume requires STEP5B_PATCH_01.md'
    return pre, resume

def main():
    pre,resume=verify_gates()
    features=pre['features'];freeze=load_json(A/'09_manifest/DEVELOPMENT_FREEZE_V1.json')
    inputs=pd.read_parquet(O/'02_inputs/DOMAIN_B_FEATURES_FROZEN43.parquet')
    validate_matrix(inputs,features)
    assert set(inputs.sampling_domain)=={'B'} and inputs.state_id.is_unique
    # Exclusive creation: no rerun can overwrite the first evaluation record.
    if resume:
        oldrec=load_json(O/'01_preregistration/DOMAIN_B_FIRST_OPEN_RECORD.json')
        write_json(O/'01_preregistration/FIRST_OPEN_RESUME_RECORD.json',dict(original=oldrec,resume_utc=now(),reason='PATCH_01 resume: aborted before any prediction/metric output; same single evaluation'))
        record=dict(oldrec)
    else:
        record=dict(timestamp_utc=now(),script=str(Path(__file__)),preregistration_sha256=sha(O/'01_preregistration/STEP5B_PREREGISTRATION_V1.json'),development_freeze_sha256=sha(A/'09_manifest/DEVELOPMENT_FREEZE_V1.json'),feature_hash=freeze['feature_list_sha256'],model_hashes={t:v['artifact_sha256'] for t,v in freeze['tasks'].items()},reason='first and only prospective Domain-B OOD model evaluation under approved amended protocol',historical_physical_first_read=False,historical_deviation_record='PROTOCOL_AMENDMENT_V1.md')
    if not resume:
        assert pre['timestamp_utc']<record['timestamp_utc']
        with open(O/'01_preregistration/DOMAIN_B_FIRST_OPEN_RECORD.json','x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    cols=['state_id','composition_id','sampling_domain','state_converged','has_multiple_FCC_instances','has_multiple_BCC_instances']+sum(TARGETS.values(),[])
    # Explicit domain predicate and target column whitelist; no ANCHOR outputs.
    truth=pd.read_parquet(FULL/'03_processed/FULL_V1_ML_TARGET_ALL.parquet',columns=cols,filters=[('sampling_domain','==','B')])
    assert truth.state_id.is_unique and set(truth.sampling_domain)=={'B'}
    assert set(truth.state_id)==set(inputs.state_id)
    assert truth.state_converged.isin([0,1]).all()
    allrows=inputs.merge(truth.drop(columns='sampling_domain'),on=['state_id','composition_id'],validate='1:1')
    support=allrows[['state_id','composition_id','family','Temperature_C']+RANGES+['state_converged']].copy()
    for col in RANGES:support[col+'_bin']=apply_bins(support[col],pre['bins'][col])
    support.to_parquet(O/'07_qa/DOMAIN_B_STATE_SUPPORT.parquet',index=False)
    d=allrows[allrows.state_converged==1].copy().reset_index(drop=True)
    assert d.composition_id.nunique()==inputs.composition_id.nunique()
    fails=allrows[allrows.state_converged==0]
    counts=dict(compositions=inputs.composition_id.nunique(),formal=len(allrows),converged=len(d),failed=len(fails),failure_rate=len(fails)/len(allrows),baseline_compositions=inputs.loc[inputs.family=='BASELINE','composition_id'].nunique(),baseline_formal=int((allrows.family=='BASELINE').sum()),x_usable_rows=int((d.family!='BASELINE').sum()),failures_all_MO_950=bool(((fails.family=='MO')&(fails.Temperature_C==950)).all()))
    write_json(O/'07_qa/DOMAIN_B_COUNTS.json',counts)
    # Presence/fraction/partition target integrity is checked without changing labels.
    for t in TASKS:
        q=d[d.family!='BASELINE'] if t=='x_partition' else d
        assert np.isfinite(q[TARGETS[t]].to_numpy(float)).all(),t
        if t=='presence':assert np.isin(q[TARGETS[t]].to_numpy(),[0,1]).all()
        else:
            v=q[TARGETS[t]].to_numpy(float)
            assert (v>=0).all() and np.max(np.abs(v.sum(1)-1))<1e-7,t
    assert d.loc[d.family=='BASELINE',TARGETS['x_partition']].isna().all().all()
    out=d[[c for c in inputs.columns if c not in features]+['Temperature_C']].copy()
    # Composition metadata may also be a frozen feature: carry it once as metadata.
    for c in RANGES:out[c]=d[c]
    out['has_multiple_FCC_instances']=d.has_multiple_FCC_instances
    out['has_multiple_BCC_instances']=d.has_multiple_BCC_instances
    out['phase_split']=d[['has_multiple_FCC_instances','has_multiple_BCC_instances']].any(axis=1)
    for t in TASKS:
        for c in TARGETS[t]:out[c+'_true']=d[c]
        tf=freeze['tasks'][t]
        path=A/f'04_models/final/{t}/model_state.pkl'
        assert sha(path)==tf['artifact_sha256']
        with open(path,'rb') as f:blob=pickle.load(f)
        eligible=d.family!='BASELINE' if t=='x_partition' else pd.Series(True,index=d.index)
        x=validate_matrix(d.loc[eligible],features)
        _,predict=freeze_mod.build_final_model(tf['model_family'],t,tf['hyperparameters'])
        raw=predict(blob['state'],x)
        assert raw.shape==(int(eligible.sum()),5) and np.isfinite(raw).all()
        if t=='presence':
            assert ((raw>=0)&(raw<=1)).all()
            for j,f in enumerate(FAMILIES):
                out[f'{f}_presence_prob']=raw[:,j]
                out[f'{f}_presence_pred']=(raw[:,j]>=THRESHOLDS[j]).astype(int)
        else:
            proj,nfb=C.project_simplex(raw,np.asarray(blob['trivial']))
            assert np.max(np.abs(proj.sum(1)-1))<1e-12
            for j,f in enumerate(FAMILIES):
                out[f'{f}_{TAGS[t]}_raw']=np.nan;out[f'{f}_{TAGS[t]}_proj']=np.nan
                out.loc[eligible,f'{f}_{TAGS[t]}_raw']=raw[:,j]
                out.loc[eligible,f'{f}_{TAGS[t]}_proj']=proj[:,j]
        assert sha(path)==tf['artifact_sha256']
        print(f'Frozen {t}: {len(x)} rows predicted',flush=True)
    assert not out.columns.duplicated().any()
    out.to_parquet(O/'03_predictions/DOMAIN_B_OOD_PREDICTIONS.parquet',index=False)
    write_json(O/'09_manifest/PREDICTION_COMPLETION.json',dict(timestamp_utc=now(),sha256=sha(O/'03_predictions/DOMAIN_B_OOD_PREDICTIONS.parquet'),counts=counts,model_hashes=record['model_hashes']))
    print('One-time prediction table persisted; no training or tuning',flush=True)

if __name__=='__main__':main()
