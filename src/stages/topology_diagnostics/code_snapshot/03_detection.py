"""Stage K: OOD error detectability, not calibrated uncertainty."""
from d_common import *
from sklearn.metrics import roc_auc_score,average_precision_score

def run():
    p=pd.read_parquet(ROOT/'02_frozen_inputs/DIAGNOSTIC_STATES.parquet')
    comp=pd.read_parquet(ROOT/'02_frozen_inputs/LOEO_COMPOSITION_DISTANCE.parquet')
    mf=pd.read_csv(ROOT/'04_element_manifold/LOEO_ELEMENT_MANIFOLD_DIAGNOSTICS.csv')
    f3=pd.read_csv(ROOT/'04_element_manifold/F3_SELECTION_VS_EXTRAPOLATION.csv')
    f3=f3[f3.selected_this_fold].groupby('heldout').outside_range.agg(['sum','mean','size']).rename(columns={'sum':'F3_outside_count','mean':'F3_outside_fraction','size':'F3_selected_count'})
    mf=mf.merge(f3,on='heldout',validate='one_to_one')
    csv('04_element_manifold/MANIFOLD_WITH_F3_SUMMARY.csv',mf)
    p=p.merge(comp.drop(columns='family'),on=['heldout_element','composition_id'],validate='many_to_one')
    p=p.merge(mf.drop(columns=['nearest_element','convex_weights']),left_on='heldout_element',right_on='heldout',validate='many_to_one')
    scores=['nn1_distance','nn5_mean_distance','outside_train_range_feature_count','nearest_distance','affine_residual','convex_reconstruction_error','F3_outside_fraction']
    pooled_threshold=float(p.x_tvd.quantile(.9));out=[];quant=[]
    for scope,g in [('POOLED',p)]+[(e,p[p.heldout_element==e]) for e in E]:
        threshold=pooled_threshold if scope=='POOLED' else float(g.x_tvd.quantile(.9))
        y=(g.x_tvd>=threshold).to_numpy(); prev=float(y.mean())
        for score in scores:
            v=g[score].to_numpy(float);valid=np.isfinite(v)
            assert valid.all()
            degenerate=len(np.unique(y))<2;constant=np.ptp(v)==0
            auc=np.nan if degenerate else float(roc_auc_score(y,v))
            ap=np.nan if degenerate else float(average_precision_score(y,v))
            out.append(dict(scope=scope,score=score,n_states=len(g),n_compositions=g.composition_id.nunique(),n_chemical_identities=g.heldout_element.nunique(),high_error_threshold=threshold,high_error_prevalence=prev,AUROC=auc,AUPRC=ap,AP_lift_over_prevalence=ap/prev if prev and not degenerate else np.nan,constant_score=constant,status='DEGENERATE_LABELS' if degenerate else 'NO_WITHIN_FOLD_RESOLUTION' if constant else 'EXPLORATORY',note='archived nominal 1-NN; not guaranteed exact minimum' if score=='nn1_distance' else 'chemical score has only six unique anchor contexts' if score in scores[3:] else 'repeated state rows share composition score; no independent-row inference'))
        # Fixed percentile binning of scores; all states retained, tied bins collapse.
        for score in ['nn5_mean_distance','outside_train_range_feature_count']:
            bins=pd.qcut(g[score],4,duplicates='drop')
            for label,idx in bins.groupby(bins,observed=True).groups.items():
                h=g.loc[idx];quant.append(dict(scope=scope,score=score,distance_bin=str(label),n_rows=len(h),mean_x_tvd=h.x_tvd.mean(),topology_accuracy=h.topology_correct.mean()))
    csv('08_distance_uncertainty/OOD_ERROR_DETECTION_METRICS.csv',out)
    csv('08_distance_uncertainty/ERROR_BY_DISTANCE_QUARTILE.csv',quant)
    correlations=[];met=pd.read_csv(ROOT/'02_frozen_inputs/STEP5C_RECOMPUTED_PRIMARY_METRICS.csv')
    for task in TASKS:
        means=met[met.task==task].set_index('heldout').loc[E,'mean_tvd']
        md=mf.set_index('heldout').loc[E]
        for score in ['nearest_distance','mean_distance','centroid_distance','affine_residual','convex_reconstruction_error','fraction_outside','F3_outside_fraction']:
            correlations.append(dict(task=task,score=score,n_elements=6,spearman_rho=rho(md[score],means),interpretation='n=6 exploratory only; no p-value or significance inference'))
    csv('08_distance_uncertainty/ERROR_VS_ELEMENT_MANIFOLD.csv',correlations)
    cr=[]
    for e in E:
        g=p[p.heldout_element==e]
        for task in TASKS:
            y=g[cols(task)].to_numpy(float);pred=g[cols(task,'pred_proj')].to_numpy(float)
            q=g[['composition_id']+scores[:3]].copy();q['error']=tvd(y,pred)
            q=q.groupby('composition_id').mean(numeric_only=True)
            for score in scores[:3]: cr.append(dict(heldout=e,task=task,score=score,n_compositions=len(q),spearman_rho=rho(q[score],q.error),interpretation='composition aggregated descriptive correlation, no iid-state significance'))
    csv('08_distance_uncertainty/COMPOSITION_DISTANCE_VS_TASK_ERROR.csv',cr)
    # No unpickling or new primary inference: inspect frozen artifact inventory only.
    inv=[str(x.relative_to(C)) for x in (C/'04_models').rglob('*') if x.is_file() and '__pycache__' not in x.parts]
    js('08_distance_uncertainty/EXISTING_DISAGREEMENT_AVAILABILITY.json',dict(status='UNAVAILABLE',reason='Only one selected model_state.pkl per fold/task; no saved paired XGB/CatBoost alternatives or runner-up predictions in primary artifact inventory. No extra ensemble trained.',model_files=inv,new_models_for_disagreement=0))
    presence=pd.read_csv(ROOT/'02_frozen_inputs/LOEO_ERROR_VS_ELEMENT_DISTANCE.csv')
    presence=presence[presence.task=='presence'].drop(columns=['spearman_p'],errors='ignore')
    presence['interpretation_limit']='Archive only; many DEGENERATE_TEST_LABEL outcomes. Negative distance correlation is not evidence that distant chemistry generalizes better.'
    csv('08_distance_uncertainty/PRESENCE_DISTANCE_ARCHIVE_CAVEAT.csv',presence)
    js('08_distance_uncertainty/DETECTION_DESIGN_SUMMARY.json',dict(status='EXPLORATORY_ERROR_DETECTABILITY_NOT_CALIBRATED_UNCERTAINTY',threshold=pooled_threshold,pooled_rows=len(p),pooled_high_error_prevalence=float((p.x_tvd>=pooled_threshold).mean()),chemical_independence_n=6,state_error_wrong_share=float(p.loc[~p.topology_correct,'x_tvd'].sum()/p.x_tvd.sum()),composition_scores_constant_over_temperature=True,element_scores_constant_within_fold=True,confidence_intervals='not estimated; no inferential claims'))
    print('DETECTION COMPLETE',flush=True)
if __name__=='__main__':run()
