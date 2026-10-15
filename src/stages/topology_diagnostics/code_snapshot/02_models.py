"""Stage J: validation-only smooth-model diagnosis, all folds freeze before scoring."""
from d_common import *
import time, copy, random, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import torch
from torch import nn

def network(n): return nn.Sequential(nn.Linear(n,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,5))
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def train_fold(d,e,device,protocol):
    out=ROOT/'07_model_bias_diagnostic'/f'FOLD_{e}';out.mkdir(exist_ok=True)
    if (out/'SELECTION_FREEZE.json').exists():
        frozen=json.loads((out/'SELECTION_FREEZE.json').read_text())
        for n,h in frozen['artifact_hashes'].items():assert sha(out/n)==h
        print(e,'already frozen, verified and reused',flush=True);return
    rr=roles(d,e);tr=d[rr=='FOLD_TRAIN'];va=d[rr=='FOLD_VALIDATION']
    assert e not in set(tr.family)|set(va.family)
    assert set(tr.family)==set(E)-{e} and not set(tr.composition_id)&set(va.composition_id)
    fr=freeze(e);fc=fr['selected_features'];baseline=np.array(fr['tasks']['x_partition']['trivial_baseline'])
    yt=tr[cols('x_partition')].to_numpy(float);yv=va[cols('x_partition')].to_numpy(float)
    assert len(tr)==fr['tasks']['x_partition']['n_train_rows'] and len(va)==fr['tasks']['x_partition']['n_validation_rows']
    assert np.allclose(yt.mean(0)/yt.mean(0).sum(),baseline,atol=1e-12)
    scaler=StandardScaler().fit(tr[fc]);xt=scaler.transform(tr[fc]);xv=scaler.transform(va[fc])
    assert np.isfinite(xt).all() and np.isfinite(xv).all()
    joblib.dump(scaler,out/'scaler.joblib')
    trials=[];best=(np.inf,None,None)
    with threadpool_limits(limits=2):
        for alpha in protocol['models']['ridge']['alpha']:
            model=Ridge(alpha=alpha,solver='cholesky').fit(xt,yt)
            pred=project(model.predict(xv),baseline);score=tvd(yv,pred).mean()
            trials.append(dict(heldout=e,model='Ridge',alpha=alpha,validation_tvd=score))
            if score<best[0]-1e-12:best=(score,alpha,model)
    joblib.dump(best[2],out/'ridge.joblib')
    ridge_info=dict(alpha=best[1],validation_tvd=best[0])
    print(e,'Ridge selected',ridge_info,flush=True)
    xt=torch.as_tensor(xt,dtype=torch.float32,device=device);xv=torch.as_tensor(xv,dtype=torch.float32,device=device)
    yt_t=torch.as_tensor(yt,dtype=torch.float32,device=device);yv_t=torch.as_tensor(yv,dtype=torch.float32,device=device)
    seed=protocol['models']['seeds'][e];best_mlp=(np.inf,None,None);lr_details=[]
    for lr in protocol['models']['mlp']['learning_rates']:
        seed_all(seed);model=network(xt.shape[1]).to(device);opt=torch.optim.Adam(model.parameters(),lr=lr)
        best_mse=np.inf;best_epoch=0;best_state=None;wait=0;history=[];start=time.time()
        for epoch in range(1,201):
            model.train();order=torch.randperm(len(xt),device=device)
            loss_sum=torch.zeros((),device=device)
            for idx in order.split(1024):
                opt.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(model(xt[idx]),yt_t[idx]);loss.backward();opt.step()
                loss_sum+=loss.detach()*len(idx)
            model.eval()
            with torch.no_grad(): raw=model(xv); vmse=float(nn.functional.mse_loss(raw,yv_t).item())
            history.append(dict(epoch=epoch,train_mse=float((loss_sum/len(xt)).item()),validation_mse=vmse))
            if vmse<best_mse-1e-8:
                best_mse=vmse;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0
            else:wait+=1
            if epoch%25==0:print(e,'MLP',lr,'epoch',epoch,'val_mse',round(vmse,8),'elapsed',round(time.time()-start),flush=True)
            if wait>=15:break
        model.load_state_dict(best_state);model.eval()
        with torch.no_grad():raw=model(xv).cpu().numpy()
        score=float(tvd(yv,project(raw,baseline)).mean())
        detail=dict(heldout=e,model='MLP',learning_rate=lr,validation_tvd=score,best_epoch=best_epoch,epochs_run=epoch,best_validation_mse=best_mse,elapsed_seconds=time.time()-start)
        trials.append(detail);lr_details.append(detail)
        csv(f'07_model_bias_diagnostic/FOLD_{e}/MLP_LR_{lr}_HISTORY.csv',history)
        if score<best_mlp[0]-1e-12:best_mlp=(score,detail,best_state)
        print(e,'MLP candidate complete',detail,flush=True)
    torch.save(best_mlp[2],out/'mlp.pt')
    mf=pd.read_csv(ROOT/'04_element_manifold/LOEO_ELEMENT_MANIFOLD_DIAGNOSTICS.csv').set_index('heldout')
    nearest=mf.loc[e,'nearest_element'];analog=tr[tr.family==nearest].groupby('Temperature_C')[cols('x_partition')].mean()
    assert set(va.Temperature_C)<=set(analog.index)
    analog.to_csv(out/'analog_temperature_means.csv')
    csv(f'07_model_bias_diagnostic/FOLD_{e}/VALIDATION_CANDIDATES.csv',trials)
    selection=dict(heldout=e,status='POST_HOC_DIAGNOSTIC',frozen_utc=now(),preregistration_sha256=sha(ROOT/'01_preregistration/STEP5D_FAILURE_DIAGNOSIS_PREREGISTRATION.json'),features=fc,primary_feature_hash=fr['selected_features_sha256'],seed=seed,device=str(device),train_rows=len(tr),validation_rows=len(va),train_compositions=tr.composition_id.nunique(),validation_compositions=va.composition_id.nunique(),training_X_identities=sorted(tr.family.unique()),heldout_targets_used_for_tuning=0,scaler_fit_scope='FOLD_TRAIN X-task only',ridge=ridge_info,mlp=best_mlp[1],analog_nearest_element=nearest,artifact_hashes={n:sha(out/n) for n in ['scaler.joblib','ridge.joblib','mlp.pt','analog_temperature_means.csv']})
    js(f'07_model_bias_diagnostic/FOLD_{e}/SELECTION_FREEZE.json',selection)
    print(e,'SELECTION FROZEN',flush=True)

def score_all(device):
    selections={e:json.loads((ROOT/f'07_model_bias_diagnostic/FOLD_{e}/SELECTION_FREEZE.json').read_text()) for e in E}
    for e,s in selections.items():
        assert s['features']==freeze(e)['selected_features'] and s['heldout_targets_used_for_tuning']==0
        for n,h in s['artifact_hashes'].items():assert sha(ROOT/f'07_model_bias_diagnostic/FOLD_{e}'/n)==h
    js('07_model_bias_diagnostic/ALL_SIX_SELECTIONS_FROZEN_GATE.json',dict(frozen_utc=now(),folds=E,status='PASS',freeze_hashes={e:sha(ROOT/f'07_model_bias_diagnostic/FOLD_{e}/SELECTION_FREEZE.json') for e in E}))
    rows=[];d=pd.read_parquet(ROOT/'02_frozen_inputs/DOMAIN_A_DEV6_DIAGNOSTIC_DATA.parquet')
    for e in E:
        out=ROOT/f'07_model_bias_diagnostic/FOLD_{e}';s=selections[e];p=predictions(e)
        x=d.set_index('state_id').loc[p.state_id,s['features']];scaler=joblib.load(out/'scaler.joblib');x=scaler.transform(x)
        y=p[cols('x_partition')].to_numpy(float);base=np.array(freeze(e)['tasks']['x_partition']['trivial_baseline'])
        tree=p[cols('x_partition','pred_proj')].to_numpy(float);ridge=project(joblib.load(out/'ridge.joblib').predict(x),base)
        model=network(x.shape[1]).to(device);model.load_state_dict(torch.load(out/'mlp.pt',weights_only=True,map_location=device));model.eval()
        with torch.no_grad():mlp=project(model(torch.as_tensor(x,dtype=torch.float32,device=device)).cpu().numpy(),base)
        a=pd.read_csv(out/'analog_temperature_means.csv').set_index('Temperature_C');analog=project(a.loc[p.Temperature_C].to_numpy(float),base)
        predout=p[['state_id','composition_id','Temperature_C']].copy()
        for name,pred in [('Tree',tree),('Ridge',ridge),('MLP',mlp),('Nearest-element analog',analog)]:
            assert np.isfinite(pred).all() and np.allclose(pred.sum(1),1,atol=1e-7)
            rows.append(dict(heldout=e,method=name,status='PRIMARY_FROZEN' if name=='Tree' else 'POST_HOC_DIAGNOSTIC',n_rows=len(y),**metrics(y,pred,base)))
            for j,f in enumerate(F):predout[f'{name}_{f}']=pred[:,j]
        predout.to_parquet(out/'DIAGNOSTIC_HELDOUT_PREDICTIONS.parquet',index=False)
    csv('07_model_bias_diagnostic/X_DIAGNOSTIC_MODEL_COMPARISON.csv',rows)
    print(pd.DataFrame(rows)[['heldout','method','mean_tvd','topology_accuracy','skill_vs_trivial']].to_string(index=False),flush=True)

def run():
    start=time.time();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    protocol=json.loads((ROOT/'01_preregistration/STEP5D_FAILURE_DIAGNOSIS_PREREGISTRATION.json').read_text())
    assert sha(ROOT/'01_preregistration/STEP5D_FAILURE_DIAGNOSIS_PREREGISTRATION.json')==json.loads((ROOT/'01_preregistration/PREREGISTRATION_HASH.json').read_text())['sha256']
    d=pd.read_parquet(ROOT/'02_frozen_inputs/DOMAIN_A_DEV6_DIAGNOSTIC_DATA.parquet')
    js('00_config/DIAGNOSTIC_RUNTIME.json',dict(python=os.sys.version,numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,cuda_available=torch.cuda.is_available(),device=str(device),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,started_utc=now()))
    for e in E:train_fold(d,e,device,protocol)
    score_all(device);print('MODEL DIAGNOSTICS COMPLETE',round(time.time()-start),'seconds',flush=True)
if __name__=='__main__':run()
