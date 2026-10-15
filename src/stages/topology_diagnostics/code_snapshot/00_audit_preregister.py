from d_common import *
import shutil
for d in DIRS: (ROOT/d).mkdir(parents=True,exist_ok=True)
pr=ROOT/'01_preregistration/STEP5D_FAILURE_DIAGNOSIS_PREREGISTRATION.json'
assert not pr.exists(), 'Preregistration immutable: do not rerun initialization'
historical=pd.read_csv(C/'11_manifest/STEP5C_FILE_HASHES.csv')
check=[]
for r in historical.itertuples():
    actual=sha(C/r.path)
    append_only=False
    if r.path=='10_logs/PART2.log' and actual!=r.sha256:
        raw=(C/r.path).read_bytes()
        append_only=len(raw)>r.bytes and hashlib.sha256(raw[:r.bytes]).hexdigest()==r.sha256
        if append_only:
            js('02_frozen_inputs/APPEND_ONLY_LOG_EXCEPTION.json',dict(path=r.path,archived_bytes=r.bytes,current_bytes=len(raw),archived_prefix_hash_matches=True,suffix=raw[r.bytes:].decode('utf-8',errors='replace'),interpretation='Existing run log appended report completion and exit 0 after manifest creation; no primary artifact exception'))
    check.append(dict(path=r.path,expected=r.sha256,actual=actual,pass_hash=actual==r.sha256,verified_append_only_log=append_only))
csv('02_frozen_inputs/STEP5C_ARCHIVED_HASH_VERIFICATION.csv',check)
assert all(r['pass_hash'] or r['verified_append_only_log'] for r in check), 'STOP: unexplained STEP5C archival mismatch'
before=[dict(path=p.relative_to(C).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(C.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
csv('02_frozen_inputs/STEP5C_ALL_FILES_BEFORE.csv',before)
rows=[]
met=pd.read_csv(C/'06_metrics/LOEO_ALL_FOLDS_METRICS.csv')
for e in E:
    p=predictions(e); fr=freeze(e)
    for t in TASKS:
        q=p.dropna(subset=cols(t)); y=q[cols(t)].to_numpy(float); pred=q[cols(t,'pred_proj')].to_numpy(float)
        base=np.array(fr['tasks'][t]['trivial_baseline'])
        z=metrics(y,pred,base); old=met[(met.heldout_element==e)&(met.task==t)].iloc[0]
        assert abs(z['mean_tvd']-old.value)<1e-10 and abs(z['trivial_tvd']-old.trivial_value)<1e-10
        rows.append(dict(heldout=e,task=t,n_rows=len(q),**z))
csv('02_frozen_inputs/STEP5C_RECOMPUTED_PRIMARY_METRICS.csv',rows)
for name in ['LOEO_ELEMENT_DISTANCE.csv','LOEO_ERROR_VS_ELEMENT_DISTANCE.csv','LOEO_COMPOSITION_DISTANCE.parquet']:
    shutil.copy2(C/'07_distance_analysis'/name,ROOT/'02_frozen_inputs'/name)
shutil.copy2(C/'06_metrics/LOEO_FEATURE_SELECTION_FREQUENCY.csv',ROOT/'02_frozen_inputs/LOEO_FEATURE_SELECTION_FREQUENCY.csv')
shutil.copy2(r'C:\Users\admin\.codex\attachments\75a64c6e-f73e-46af-a48e-6f66bf5316c0\pasted-text.txt',ROOT/'00_config/USER_REQUIREMENTS.txt')
protocol=dict(stage='STEP5D_FAILURE_DIAGNOSIS_V1',created_utc=now(),status='FROZEN_BEFORE_DIAGNOSTIC_MODEL_FITTING',designation='POST_HOC_DIAGNOSTIC; retrospective, not confirmatory preregistration before STEP5C results',elements=E,families=F,tasks=TASKS,
    questions=['H1 chemical-anchor sparsity','H2 element-identity dominated target','H3 tree inductive bias','H4 descriptor insufficiency','H5 topology discontinuity versus stable-topology failure','composition distance and high-error detectability'],
    universe='converged Domain A, six X elements only; BASELINE excluded from all four eta2 comparisons; every state counted once; no Ni/Anchor/Domain B materialized',
    primary_integrity=dict(step5c_files=len(before),archived_hashes_verified=len(check),step5c_writes=0,new_calphad_calls=0,ni_rows=0,anchor_rows=0),
    variance=dict(total='sum_i ||y_i-mu||^2',between='sum_e n_e ||mu_e-mu||^2',within='sum_e sum_i_in_e ||y_i-mu_e||^2',eta2='SS_between/SS_total; zero SS_total -> NA',tolerance='abs(total-between-within) <= 1e-8*max(1,total)',interpretation='descriptive association with identity; within-element variance not causal attribution to composition/T',sensitivity='composition-equal weighting, each composition total weight 1 despite missing temperature states'),
    distances=dict(properties=PROPS,scaler='five training X identities; population std; zero std replaced by 1 with flag',range='strict numeric bounds with 1e-10 absolute tolerance',exceedance='max(train_min-value,0,value-train_max)/train_sd',svd='center five standardized anchors; rank tol=max(shape)*eps*smax; residual=norm(z-projection); normalized= residual/max(norm(z),1e-12)',convex='SLSQP nonnegative weights sum=1; ftol=1e-12; maxiter=2000; verify KKT/projection optimality',composition='reuse immutable STEP5C distances, no recomputation or property changes',f3='all F3 ever selected; training ranges exclude heldout and BASELINE; frozen candidate features only'),
    archetypes=dict(centroids='overall, each exact Temperature_C, and X_wt_pct low/mid/high using fold-train composition terciles',borrowing='argmin TVD(predicted centroid, each FOLD_TRAIN element centroid); canonical element order breaks ties',js='base-2 Jensen-Shannon divergence, not square-root distance',topology='argmax five-family vector, canonical family order breaks ties',collapse_ratio='max fraction predicted dominant family',collapse_threshold=.95,entropy='Shannon bits over empirical dominant-family frequencies'),
    boundary=dict(transition='change between consecutive full-grid observed temperatures for same composition; midpoint boundary; never bridge missing states',distance='minimum absolute grid-index distance to transition midpoint; <=1 adjacent, >1 stable; no observed transitions -> no-observed-transition group; incomplete trajectory flagged',split='has_multiple_FCC_instances OR has_multiple_BCC_instances',conditioning='exact heldout x temperature x topology_correctness strata with both split categories; overlap-weight min(n_split,n_nonsplit); descriptive standardized contrast; no causal claim'),
    models=dict(scope='X partition only, all six folds identical, original selected features; no feature reselection; existing tree predictions only',seeds={e:20260907+i for i,e in enumerate(E)},scaler='StandardScaler.fit(FOLD_TRAIN X-task rows only)',projection='clip negative outputs and renormalize, zero sum fallback frozen FOLD_TRAIN mean',ridge=dict(alpha=[1e-4,1e-3,1e-2,1e-1,1,10,100],selection='minimum projected validation mean TVD; ties <=1e-12 choose smaller alpha'),mlp=dict(hidden=[128,64],activation='ReLU',outputs=5,output_activation='linear',loss='raw-component MSE',optimizer='Adam defaults betas=(0.9,0.999), eps=1e-8, weight_decay=0',batch_size=1024,max_epochs=200,patience=15,learning_rates=[1e-4,3e-4,1e-3],early_stopping='minimum raw validation MSE, min_delta=1e-8; restore best epoch',lr_selection='minimum projected validation TVD; ties <=1e-12 choose smaller LR',data='all eligible train/validation rows, no subsampling',device='CUDA if available else CPU; deterministic torch algorithms; fixed seed per fold reset for each LR'),analog='13-property nearest training element, FOLD_TRAIN mean X-partition by exact temperature; no heldout targets',freeze='save all six model selections before diagnostic heldout scoring'),
    metrics=['mean TVD','median TVD','p95 TVD','five-label topology macro F1','topology accuracy','trivial minus model TVD (original STEP5C skill convention)','1-model/trivial TVD (additional relative skill)'],
    detection=dict(high_error='TVD >= pooled empirical 90th percentile, include ties; report achieved prevalence; within-fold 90th percentile sensitivity',scores=['composition nn1','composition nn5','composition outside-range count','nearest-element distance','affine residual','convex reconstruction error','F3 outside fraction'],metrics=['AUROC','AUPRC=average precision'],constant='constant scores -> AUROC .5, AP prevalence; report NO_WITHIN_FOLD_RESOLUTION',disagreement='only if two frozen pre-scoring alternative artifacts exist; do not train or infer missing alternatives'),
    inferential_limits='all n=6 element correlations exploratory, report rho without p-values; rows clustered by composition; no causal identification or proof of descriptor sufficiency/necessity',
    go_no_go=dict(proceed='all source hashes match; no forbidden rows; finite simplex targets; exact split isolation; all six validation selections frozen before heldout scoring',stop='source changes, forbidden rows, missing core input, failed leakage or numeric integrity',tree_bias_support='same smooth model achieves >=0.05 absolute TVD gain AND >=0.10 absolute topology accuracy gain over tree in >=4 folds',rescue='positive skill vs trivial AND topology accuracy >=0.5 in >=4 folds for same smooth model',identity_support='X eta2 >=0.5 AND >=2*max(other task eta2)',topology_support='>=80% aggregate summed X TVD from topology-wrong states',boundary_support='adjacent minus stable mean TVD >=0.05 in >=4 folds with both groups; otherwise no general boundary support',sparsity='five independent identity anchors documented; causal role not experimentally isolated',descriptor='analog fails positive trivial skill or topology accuracy .5 in >=4 folds AND high range extrapolation; compatible evidence, not proof',next_stage='STEP5E recommendations only; STOP after STEP5D'))
js('01_preregistration/STEP5D_FAILURE_DIAGNOSIS_PREREGISTRATION.json',protocol)
js('01_preregistration/PREREGISTRATION_HASH.json',dict(sha256=sha(pr),timestamp=now()))
print('AUDIT PASS; preregistration frozen', len(before),'files; X negative skill folds',sum(r['skill_vs_trivial']<0 for r in rows if r['task']=='x_partition'),flush=True)
