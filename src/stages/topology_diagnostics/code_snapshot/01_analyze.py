"""Stages C-I. Descriptive mechanisms, no diagnostic model fitting."""
from d_common import *
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

def variance(y,labels,weights=None):
    y=np.asarray(y,float); labels=np.asarray(labels)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    mu=np.average(y,axis=0,weights=w)
    total=(w[:,None]*(y-mu)**2).sum(axis=0); between=np.zeros(y.shape[1]); within=between.copy()
    for e in np.unique(labels):
        k=labels==e; m=np.average(y[k],axis=0,weights=w[k]); between+=w[k].sum()*(m-mu)**2
        within+=(w[k,None]*(y[k]-m)**2).sum(axis=0)
    assert np.allclose(total,between+within,rtol=1e-8,atol=1e-8)
    return total,between,within

def manifold(tr,test):
    mu=tr.mean(0); sd=tr.std(0); sd=np.where(sd>0,sd,1)
    Z=(tr-mu)/sd; z=(test-mu)/sd
    _,s,vt=np.linalg.svd(Z,full_matrices=False)
    rank=int((s>max(Z.shape)*np.finfo(float).eps*s[0]).sum())
    residual=np.linalg.norm(z-vt[:rank].T@(vt[:rank]@z))
    obj=lambda w:float(np.sum((w@Z-z)**2))
    grad=lambda w:2*Z@(w@Z-z)
    fit=minimize(obj,np.ones(len(tr))/len(tr),jac=grad,bounds=[(0,1)]*len(tr),constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(len(w)))],method='SLSQP',options=dict(ftol=1e-12,maxiter=2000))
    # Projection onto a convex set: residual has nonpositive inner product with
    # every feasible direction from the projection; test all vertices.
    w=fit.x; recon=w@Z; gap=float(np.max((Z-recon)@(z-recon)))
    assert abs(w.sum()-1)<1e-8 and w.min()>-1e-8 and gap<1e-5,(fit.message,gap)
    return Z,z,sd,dict(affine_rank=rank,affine_residual=float(residual),normalized_residual=float(residual/max(np.linalg.norm(z),1e-12)),convex_reconstruction_error=float(np.sqrt(obj(w))),convex_weights=w.tolist(),convex_optimality_gap=gap,solver_success=bool(fit.success))

def transitions(temperatures,topologies,grid):
    ix=np.searchsorted(grid,temperatures)
    valid=(np.diff(ix)==1)&(np.diff(topologies)!=0)
    mids=(ix[:-1][valid]+ix[1:][valid])/2
    distance=np.min(np.abs(ix[:,None]-mids[None,:]),axis=1) if len(mids) else np.full(len(ix),np.inf)
    return distance,int(len(mids)),len(ix)==len(grid)

def grouped_errors(d,keys):
    rows=[]
    iterator=d.groupby(keys,dropna=False,observed=True) if keys else [((),d)]
    for key,g in iterator:
        key=key if isinstance(key,tuple) else (key,)
        rows.append(dict(zip(keys,key))|dict(n_rows=len(g),n_compositions=g.composition_id.nunique(),mean_tvd=g.x_tvd.mean(),median_tvd=g.x_tvd.median(),p95_tvd=g.x_tvd.quantile(.95),topology_accuracy=g.topology_correct.mean(),sum_tvd=g.x_tvd.sum()))
    return pd.DataFrame(rows)

def run():
    assert (ROOT/'01_preregistration/STEP5D_FAILURE_DIAGNOSIS_PREREGISTRATION.json').exists()
    d=data(); print('Allowed data',len(d),'states',d.composition_id.nunique(),'compositions',flush=True)
    assert np.isfinite(d[sum([cols(t) for t in TASKS],[])].to_numpy()).all()
    for t in TASKS:
        yy=d[cols(t)].to_numpy(float)
        assert yy.min()>-1e-9 and np.allclose(yy.sum(1),1,atol=1e-7)
    d.to_parquet(ROOT/'02_frozen_inputs/DOMAIN_A_DEV6_DIAGNOSTIC_DATA.parquet',index=False)
    js('02_frozen_inputs/FILTERED_DATA_PROVENANCE.json',dict(created_utc=now(),n_rows=len(d),n_compositions=d.composition_id.nunique(),elements=sorted(d.family.unique()),domain='A',filter="sampling_domain == A AND family in [ZR,TI,CO,SI,AG,MO] applied by pyarrow before pandas materialization",baseline_rows=0,anchor_rows=0,ni_rows=0,domain_b_rows=0,sha256=sha(ROOT/'02_frozen_inputs/DOMAIN_A_DEV6_DIAGNOSTIC_DATA.parquet')))
    vr=[]; weighted=[]
    for t in TASKS:
        y=d[cols(t)].to_numpy(float)
        for mode,dest,w in [('state_equal',vr,None),('composition_equal',weighted,1/d.groupby('composition_id').state_id.transform('size').to_numpy())]:
            tot,bet,within=variance(y,d.family,w)
            for j,f in enumerate(F+['WHOLE_VECTOR']):
                a,b,c=(float(v[j] if j<5 else v.sum()) for v in [tot,bet,within])
                dest.append(dict(task=t,target_component=f,SS_total=a,SS_between=b,SS_within=c,eta2_element=b/a if a>1e-20 else np.nan,n_rows=len(y),n_elements=6,weighting=mode,identity_residual=a-b-c))
    csv('03_variance_decomposition/TASK_ELEMENT_VARIANCE_DECOMPOSITION.csv',vr)
    csv('03_variance_decomposition/COMPOSITION_EQUAL_VARIANCE_SENSITIVITY.csv',weighted)
    counts=[]
    for e in E:
        rr=roles(d,e)
        for role in ['FOLD_TRAIN','FOLD_VALIDATION','HELDOUT_ELEMENT_TEST']:
            g=d[rr==role]; counts.append(dict(heldout=e,role=role,n_rows=len(g),n_compositions=g.composition_id.nunique(),n_X_identities=g.family.nunique(),identities=';'.join(sorted(g.family.unique()))))
    csv('03_variance_decomposition/CHEMICAL_SAMPLE_COUNTS.csv',counts)
    table=pd.DataFrame(counts)
    (ROOT/'03_variance_decomposition/CHEMICAL_EFFECTIVE_SAMPLE_SIZE.md').write_text('# Chemical versus row-level sample size\n\n'+table.to_markdown(index=False)+'\n\nEvery X-task fold has five training X identities, not 18,375 independent chemistries. Repeated compositions and temperatures increase coverage conditional on identity but do not add chemical anchors. Rows from one composition are dependent. Counts exclude BASELINE (no X target), Anchor, Domain B, and Ni. Missing/nonconverged states remain excluded and no targets are imputed. The five-identity count is an anchor count, not an estimated statistical effective sample size.\n',encoding='utf-8')
    # DEV6 descriptor table is an explicitly allowed prior source. Restrict symbols
    # while streaming CSV records; never load a broad periodic-table database.
    import csv as csvlib
    with open(A/'01_feature_sources/ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.csv',encoding='utf-8-sig',newline='') as fh:
        allowed=[r for r in csvlib.DictReader(fh) if r['element'].upper() in E]
    tab=pd.DataFrame(allowed); tab['element']=tab.element.str.upper(); tab=tab.set_index('element')[PROPS].astype(float)
    assert set(tab.index)==set(E) and tab.shape==(6,13)
    csv('02_frozen_inputs/DEV6_ELEMENT_PROPERTIES.csv',tab.reset_index())
    mf=[]; ranges=[]; f3=[]
    freq=pd.read_csv(C/'06_metrics/LOEO_FEATURE_SELECTION_FREQUENCY.csv')
    f3names=freq.loc[freq.feature_block=='F3','feature'].tolist()
    for e in E:
        others=[o for o in E if o!=e]; tr=tab.loc[others].to_numpy(); test=tab.loc[e].to_numpy()
        Z,z,sd,diag=manifold(tr,test); ds=np.linalg.norm(Z-z,axis=1)
        outside=(test<tr.min(0)-1e-10)|(test>tr.max(0)+1e-10)
        mf.append(dict(heldout=e,nearest_element=others[int(ds.argmin())],nearest_distance=ds.min(),mean_distance=ds.mean(),centroid_distance=np.linalg.norm(z),n_properties_outside=int(outside.sum()),fraction_outside=outside.mean(),**{k:v for k,v in diag.items() if k!='convex_weights'},convex_weights=json.dumps(dict(zip(others,diag['convex_weights'])))))
        for j,p in enumerate(PROPS):
            lo,hi=tr[:,j].min(),tr[:,j].max(); val=test[j]
            ranges.append(dict(heldout=e,property=p,value=val,train_min=lo,train_max=hi,outside_range=outside[j],position='below' if val<lo-1e-10 else 'above' if val>hi+1e-10 else 'inside',standardized_exceedance=max(lo-val,val-hi,0)/sd[j],train_std_raw=tr[:,j].std(),zero_variance=tr[:,j].std()==0))
        rr=roles(d,e); train=d[rr=='FOLD_TRAIN']; held=d[rr=='HELDOUT_ELEMENT_TEST']
        selected=freeze(e)['selected_features']
        for f in f3names:
            a=train[f].to_numpy(float); b=held[f].to_numpy(float); lo,hi=a.min(),a.max(); val=b.mean(); std=a.std() or 1
            assert np.ptp(b)<1e-8,'F3 is expected to be X-identity constant'
            f3.append(dict(feature=f,selected_folds=int(freq.set_index('feature').loc[f,'selected_in_n_folds']),heldout=e,selected_this_fold=f in selected,heldout_value=val,train_min=lo,train_max=hi,outside_range=val<lo-1e-10 or val>hi+1e-10,exceedance=max(lo-val,val-hi,0)/std,training_scope='FOLD_TRAIN X-task rows; BASELINE excluded'))
    csv('04_element_manifold/LOEO_ELEMENT_MANIFOLD_DIAGNOSTICS.csv',mf)
    csv('04_element_manifold/ELEMENT_PROPERTY_RANGE_EXTRAPOLATION.csv',ranges)
    csv('04_element_manifold/F3_SELECTION_VS_EXTRAPOLATION.csv',f3)
    old=pd.read_csv(C/'07_distance_analysis/LOEO_ELEMENT_DISTANCE.csv').set_index('heldout_element')
    for r in mf: assert abs(r['nearest_distance']-old.loc[r['heldout'],'nearest_element_distance'])<1e-10
    ar=[]; borrow=[]; collapse=[]; states=[]; traincent=[]; regions=[]
    grid=np.sort(d.Temperature_C.unique())
    for e in E:
        p=predictions(e); yy=p[cols('x_partition')].to_numpy(float); pp=p[cols('x_partition','pred_proj')].to_numpy(float)
        dd=d[d.family==e].set_index('state_id').loc[p.state_id]
        for t in TASKS: assert np.allclose(dd[cols(t)].to_numpy(),p[cols(t)].to_numpy(),atol=1e-10)
        p['x_tvd']=tvd(yy,pp); p['topology_correct']=yy.argmax(1)==pp.argmax(1)
        p['split']=p.has_multiple_FCC_instances.astype(bool)|p.has_multiple_BCC_instances.astype(bool)
        rr=roles(d,e); tr=d[rr=='FOLD_TRAIN']; assert tr.family.nunique()==5 and e not in set(tr.family)
        cc=tr.groupby('family')[cols('x_partition')].mean().reindex([x for x in E if x!=e])
        for o,row in cc.iterrows(): traincent.append(dict(heldout=e,training_element=o,**dict(zip(F,row))))
        predcen=pp.mean(0); truecen=yy.mean(0); distances=.5*np.abs(cc.to_numpy()-predcen).sum(1); borrowed=cc.index[int(distances.argmin())]
        nearest=next(r['nearest_element'] for r in mf if r['heldout']==e)
        borrow.append(dict(heldout=e,borrowed_element=borrowed,borrowed_archetype_tvd=float(distances.min()),nearest_element_in_descriptor_space=nearest,nearest_equals_borrowed=nearest==borrowed,**{f'tvd_to_{o}':float(v) for o,v in zip(cc.index,distances)}))
        qbins=np.unique(np.quantile(tr.drop_duplicates('composition_id').X_wt_pct,[1/3,2/3]))
        assert len(qbins)==2
        p['composition_region']=np.array(['LOW_X','MID_X','HIGH_X'])[np.searchsorted(qbins,dd.X_wt_pct.to_numpy(),side='right')]
        regions.append(dict(heldout=e,X_low_upper=qbins[0],X_mid_upper=qbins[1],scope='FOLD_TRAIN composition-level terciles'))
        groups=[('overall','ALL',p)]+[('temperature',str(k),g) for k,g in p.groupby('Temperature_C')]+[('composition_region',str(k),g) for k,g in p.groupby('composition_region')]
        for scope,key,g in groups:
            a=g[cols('x_partition')].to_numpy(float).mean(0); b=g[cols('x_partition','pred_proj')].to_numpy(float).mean(0)
            ar.append(dict(heldout=e,scope=scope,group=key,n_rows=len(g),centroid_tvd=.5*np.abs(a-b).sum(),jensen_shannon_divergence=float(jensenshannon(a,b,base=2)**2),cosine_similarity=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),dominant_topology_match=int(a.argmax()==b.argmax()),**{f'true_{f}':v for f,v in zip(F,a)},**{f'pred_{f}':v for f,v in zip(F,b)}))
        td=np.bincount(yy.argmax(1),minlength=5)/len(yy); pdist=np.bincount(pp.argmax(1),minlength=5)/len(pp)
        collapse.append(dict(heldout=e,n_rows=len(p),true_entropy=entropy(td,base=2),pred_entropy=entropy(pdist,base=2),collapse_ratio=pdist.max(),collapsed=pdist.max()>=.95,topology_accuracy=p.topology_correct.mean(),error_share_topology_wrong=p.loc[~p.topology_correct,'x_tvd'].sum()/p.x_tvd.sum(),**{f'true_dominant_{f}':v for f,v in zip(F,td)},**{f'pred_dominant_{f}':v for f,v in zip(F,pdist)}))
        p=p.sort_values(['composition_id','Temperature_C']).reset_index(drop=True)
        p['transition_distance_steps']=np.inf;p['n_observed_transitions']=0;p['complete_temperature_trajectory']=False
        for comp,idx in p.groupby('composition_id',sort=False).indices.items():
            g=p.iloc[idx]; dist,n,complete=transitions(g.Temperature_C.to_numpy(),g[cols('x_partition')].to_numpy().argmax(1),grid)
            p.loc[idx,'transition_distance_steps']=dist;p.loc[idx,'n_observed_transitions']=n;p.loc[idx,'complete_temperature_trajectory']=complete
        p['boundary_group']=np.where(p.transition_distance_steps<=1,'TRANSITION_ADJACENT',np.where(p.n_observed_transitions>0,'STABLE_REGION','NO_OBSERVED_TRANSITION'))
        p['topology_status']=np.where(p.topology_correct,'TOPOLOGY_CORRECT','TOPOLOGY_WRONG')
        states.append(p)
    csv('05_topology_analysis/X_PARTITION_ARCHETYPE_DIAGNOSIS.csv',ar)
    csv('05_topology_analysis/X_PARTITION_BORROWED_ARCHETYPE.csv',borrow)
    csv('05_topology_analysis/X_PARTITION_TOPOLOGY_COLLAPSE.csv',collapse)
    csv('05_topology_analysis/FOLD_TRAIN_ELEMENT_CENTROIDS.csv',traincent)
    csv('05_topology_analysis/COMPOSITION_REGION_DEFINITIONS.csv',regions)
    p=pd.concat(states,ignore_index=True)
    p.to_parquet(ROOT/'02_frozen_inputs/DIAGNOSTIC_STATES.parquet',index=False)
    conditional=[]
    for keys in [['heldout_element','topology_status'],['heldout_element','Temperature_C','topology_status'],['heldout_element','split','topology_status'],['heldout_element','Temperature_C','split','topology_status']]:
        g=grouped_errors(p,keys);g['scope']='+'.join(keys);conditional.append(g)
    csv('05_topology_analysis/X_PARTITION_ERROR_BY_TOPOLOGY_CORRECTNESS.csv',pd.concat(conditional,ignore_index=True))
    csv('06_boundary_analysis/X_ERROR_BY_TOPOLOGY_TRANSITION_DISTANCE.csv',grouped_errors(p,['heldout_element','boundary_group','complete_temperature_trajectory']))
    p['transition_distance_bin']=p.transition_distance_steps.map(lambda x:str(x) if np.isfinite(x) else 'NO_OBSERVED_TRANSITION')
    csv('06_boundary_analysis/TRANSITION_DISTANCE_BIN_DETAIL.csv',grouped_errors(p,['heldout_element','transition_distance_bin']))
    csv('06_boundary_analysis/X_ERROR_BY_PHASE_SPLIT_AND_ELEMENT.csv',grouped_errors(p,['heldout_element','split']))
    csv('06_boundary_analysis/X_ERROR_BY_TEMPERATURE.csv',grouped_errors(p,['heldout_element','Temperature_C']))
    contrasts=[]
    for (e,t,correct),g in p.groupby(['heldout_element','Temperature_C','topology_correct']):
        a=g[g.split];b=g[~g.split]
        contrasts.append(dict(heldout=e,Temperature_C=t,topology_correct=correct,n_split=len(a),n_non_split=len(b),overlap_weight=min(len(a),len(b)),split_mean=a.x_tvd.mean(),non_split_mean=b.x_tvd.mean(),adjusted_stratum_difference=a.x_tvd.mean()-b.x_tvd.mean() if len(a) and len(b) else np.nan))
    csv('06_boundary_analysis/SPLIT_EXACT_MATCHED_STRATA.csv',contrasts)
    ct=pd.DataFrame(contrasts);adjusted=[]
    for e,g in ct.groupby('heldout'):
        both=g[g.overlap_weight>0]
        adjusted.append(dict(heldout=e,n_overlap_strata=len(both),overlap_weight=both.overlap_weight.sum(),standardized_split_minus_non_split=float(np.average(both.adjusted_stratum_difference,weights=both.overlap_weight)) if len(both) else np.nan,interpretation='association conditional on element, temperature, correctness; not causal; overlap-only'))
    csv('06_boundary_analysis/SPLIT_CONDITIONED_CONTRAST.csv',adjusted)
    # Keep primary's archived distance values, noting their implementation limits.
    note='STEP5C nn1_distance used np.partition(dd,5)[:,0], which is not guaranteed the minimum of the first five distances. It is retained unchanged as requested, labelled archived nominal 1-NN. The first-five mean is permutation-invariant and remains usable. No corrected distance is substituted into primary files.\n'
    (ROOT/'08_distance_uncertainty/ARCHIVED_DISTANCE_IMPLEMENTATION_CAVEAT.md').write_text(note,encoding='utf-8')
    print('C-I COMPLETE; eta2',pd.DataFrame(vr).query('target_component == "WHOLE_VECTOR"')[['task','eta2_element']].to_dict('records'),flush=True)
    print('collapse',pd.DataFrame(collapse)[['heldout','collapse_ratio','error_share_topology_wrong']].to_dict('records'),flush=True)

if __name__=='__main__': run()
