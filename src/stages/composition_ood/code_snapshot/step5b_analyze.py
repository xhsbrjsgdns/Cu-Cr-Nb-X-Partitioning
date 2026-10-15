"""All post-open summaries consume the persistent predictions, never re-predict."""
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from step5b_core import *
from step5b_bootstrap import run_bootstrap
from step5b_figures import make_figures

def stratify(df,group):
    rows=[]
    for label,d in df.groupby(group,observed=True,dropna=False):
        for t in TASKS:
            if t=='x_partition' and (d.family=='BASELINE').all():continue
            row,_=metrics(d,t,'Frozen');row['group']=str(label);row['stratifier']=group;rows.append(row)
    return pd.DataFrame(rows)

def main():
    completion=load_json(O/'09_manifest/PREDICTION_COMPLETION.json')
    predpath=O/'03_predictions/DOMAIN_B_OOD_PREDICTIONS.parquet'
    assert sha(predpath)==completion['sha256']
    df=pd.read_parquet(predpath);iidpred=pd.read_parquet(A/'05_results/IID_TEST_PREDICTIONS.parquet')
    assert set(df.sampling_domain)=={'B'}
    pre=load_json(O/'01_preregistration/STEP5B_PREREGISTRATION_V1.json')
    freeze=load_json(A/'09_manifest/DEVELOPMENT_FREEZE_V1.json')
    cfg=load_json(O/'00_config/PREOPEN_CONFIG.json')
    baselines={t:cfg['model_audit'][t]['trivial_vector'] for t in TASKS}
    rows=[];per=[]
    for t in TASKS:
        r,p=metrics(df,t,freeze['tasks'][t]['model_family']);rows.append(r);per+=p
        rb,pb=metrics(df,t,'TrivialPrevalence' if t=='presence' else 'TrivialMeanVector',baseline=baselines[t]);rows.append(rb);per+=pb
        if t=='presence':r['trivial_macro_f1']=rb['macro_f1'];r['performance_above_trivial']=r['macro_f1']>rb['macro_f1']
        else:r['trivial_mean_tvd']=rb['mean_tvd'];r['performance_above_trivial']=r['mean_tvd']<rb['mean_tvd']
    met=pd.DataFrame(rows);perdf=pd.DataFrame(per)
    met.to_csv(O/'04_metrics/DOMAIN_B_OOD_METRICS.csv',index=False)
    perdf.to_csv(O/'04_metrics/DOMAIN_B_PER_TARGET_METRICS.csv',index=False)
    perdf[perdf.target=='LIQUID'].to_csv(O/'04_metrics/LIQUID_DOMAIN_B_OOD_METRICS.csv',index=False)
    iidmet=pd.read_csv(A/'05_results/BASELINE_IID_TEST_METRICS.csv')
    comparison=[]
    for t in TASKS:
        b=met[(met.task==t)&(~met.model.str.startswith('Trivial'))].iloc[0]
        a=iidmet[iidmet.task==t].iloc[0]
        keys=['exact_match','hamming_loss','micro_f1','macro_f1'] if t=='presence' else ['mean_tvd','median_tvd','p95_tvd','macro_mae','macro_rmse','raw_sum_violation_mean','raw_sum_violation_max','proj_sum_violation_max']
        for k in keys:
            av,bv=float(a[k]),float(b[k]);ratio=bv/av if av else np.nan
            comparison.append(dict(task=t,metric=k,IID_value=av,OOD_value=bv,absolute_delta=bv-av,relative_change=(bv-av)/av if av else np.nan,performance_retention=ratio if t=='presence' and k=='macro_f1' else np.nan,error_inflation_factor=ratio if t!='presence' and k=='mean_tvd' else np.nan))
    comparison=pd.DataFrame(comparison);comparison.to_csv(O/'04_metrics/IID_VS_DOMAIN_B_OOD_COMPARISON.csv',index=False)
    for c in RANGES:df[c+'_bin']=apply_bins(df[c],pre['bins'][c])
    df['distance_quartile']=apply_bins(df.nn1_distance,pre['bins']['nn1_distance'])
    strata={}
    for group,filename in [('family','DOMAIN_B_ERROR_BY_FAMILY.csv'),('Temperature_C','DOMAIN_B_ERROR_BY_TEMPERATURE.csv'),('phase_split','DOMAIN_B_ERROR_BY_PHASE_SPLIT.csv')]:
        strata[group]=stratify(df,group);strata[group].to_csv(O/'04_metrics'/filename,index=False)
    ranges=pd.concat([stratify(df,c+'_bin') for c in RANGES],ignore_index=True)
    ranges.to_csv(O/'04_metrics/DOMAIN_B_ERROR_BY_COMPOSITION_RANGE.csv',index=False)
    strata['composition']=ranges
    quart=stratify(df,'distance_quartile');quart.to_csv(O/'05_shift_analysis/OOD_DISTANCE_QUARTILE_METRICS.csv',index=False)
    errs=composition_errors(df);errs.to_parquet(O/'05_shift_analysis/COMPOSITION_MEAN_ERRORS.parquet',index=False)
    corr=[]
    for t in TASKS:
        for distance in ['nn1_distance','nn5_mean_distance','outside_train_range_feature_count']:
            q=errs[[t,distance]].dropna()
            if q[t].nunique()<2 or q[distance].nunique()<2:rho,p=np.nan,np.nan
            else:rho,p=spearmanr(q[t],q[distance])
            corr.append(dict(task=t,distance=distance,n_compositions=len(q),spearman_rho=rho,p_value=p,interpretation='descriptive association; unadjusted p; not causal'))
    pd.DataFrame(corr).to_csv(O/'05_shift_analysis/ERROR_VS_OOD_DISTANCE.csv',index=False)
    support=pd.read_parquet(O/'07_qa/DOMAIN_B_STATE_SUPPORT.parquet')
    failures=[]
    for group in ['family','Temperature_C']+[c+'_bin' for c in RANGES]:
        for label,g in support.groupby(group,observed=True):
            failures.append(dict(stratifier=group,group=str(label),formal=len(g),converged=int(g.state_converged.sum()),failed=int((g.state_converged==0).sum()),failure_rate=float((g.state_converged==0).mean())))
    failtable=pd.DataFrame(failures);failtable.to_csv(O/'07_qa/DOMAIN_B_FAILURE_DISTRIBUTION.csv',index=False)
    counts=load_json(O/'07_qa/DOMAIN_B_COUNTS.json')
    (O/'07_qa/DOMAIN_B_FAILURE_CONTEXT.md').write_text('# Domain B failure context\n\n'+json.dumps(counts,indent=2)+'\n\nNo failed state is interpolated or scored. All formal states remain in the support table.\n\n'+failtable.to_markdown(index=False),encoding='utf-8')
    print('Metrics, strata, failure context and correlations complete; bootstrap starts',flush=True)
    with threadpool_limits(limits=1):
        boot=run_bootstrap(df,iidpred,O/'04_metrics',seed=20260905,n_reps=1000,trivial_vectors=baselines)
    # Verify bootstrap point estimates reproduce frozen IID CSV and direct B metrics.
    for t in TASKS:
        k='macro_f1' if t=='presence' else 'mean_tvd'
        z=boot['comparison'].query('task==@t and metric==@k').iloc[0]
        original=iidmet.loc[iidmet.task==t,k].iloc[0]
        actual=met.loc[(met.task==t)&(~met.model.str.startswith('Trivial')),k].iloc[0]
        assert abs(z.iid_estimate-original)<1e-12,(t,'IID mismatch')
        assert abs(z.ood_estimate-actual)<1e-12,(t,'OOD mismatch')
    print('1000 independent cluster bootstrap replicates complete',flush=True)
    train=pd.read_parquet(O/'02_inputs/DEV_TRAIN_COMPOSITION_CONTEXT.parquet')
    iid=pd.read_parquet(O/'02_inputs/IID_COMPOSITION_CONTEXT.parquet')
    b=pd.read_parquet(O/'02_inputs/DOMAIN_B_COMPOSITION_CONTEXT.parquet')
    shift=pd.read_csv(O/'05_shift_analysis/FEATURE_SHIFT_A_TO_B.csv')
    make_figures(df,iidpred,train,iid,b,met,comparison,shift,strata,O/'06_figures',counts=counts)
    print('Ten figures exported',flush=True)
    write_json(O/'09_manifest/ANALYSIS_COMPLETION.json',dict(timestamp_utc=now(),prediction_sha256=sha(predpath),n_bootstrap=1000))

if __name__=='__main__':main()
