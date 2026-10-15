"""Stage L/O: evidence matrix and complete question-by-question report."""
from d_common import *

def run():
    v=pd.read_csv(ROOT/'03_variance_decomposition/TASK_ELEMENT_VARIANCE_DECOMPOSITION.csv')
    vv=v[v.target_component=='WHOLE_VECTOR'].set_index('task').loc[TASKS]
    m=pd.read_csv(ROOT/'07_model_bias_diagnostic/X_DIAGNOSTIC_MODEL_COMPARISON.csv')
    mf=pd.read_csv(ROOT/'04_element_manifold/MANIFOLD_WITH_F3_SUMMARY.csv').set_index('heldout').loc[E]
    tc=pd.read_csv(ROOT/'05_topology_analysis/X_PARTITION_TOPOLOGY_COLLAPSE.csv').set_index('heldout').loc[E]
    borrow=pd.read_csv(ROOT/'05_topology_analysis/X_PARTITION_BORROWED_ARCHETYPE.csv')
    corr=pd.read_csv(ROOT/'08_distance_uncertainty/ERROR_VS_ELEMENT_MANIFOLD.csv')
    ds=json.loads((ROOT/'08_distance_uncertainty/DETECTION_DESIGN_SUMMARY.json').read_text())
    detection=pd.read_csv(ROOT/'08_distance_uncertainty/OOD_ERROR_DETECTION_METRICS.csv')
    split=pd.read_csv(ROOT/'06_boundary_analysis/SPLIT_CONDITIONED_CONTRAST.csv')
    states=pd.read_parquet(ROOT/'02_frozen_inputs/DIAGNOSTIC_STATES.parquet')
    primary=pd.read_csv(ROOT/'02_frozen_inputs/STEP5C_RECOMPUTED_PRIMARY_METRICS.csv')
    g=states.groupby(['heldout_element','boundary_group']).x_tvd.mean().unstack()
    boundary_difference=g.get('TRANSITION_ADJACENT')-g.get('STABLE_REGION')
    boundary_supported=int((boundary_difference>=.05).sum())>=4
    tree=m[m.method=='Tree'].set_index('heldout').loc[E]
    results={}
    for method in ['Ridge','MLP','Nearest-element analog']:
        z=m[m.method==method].set_index('heldout').loc[E]
        gain=tree.mean_tvd-z.mean_tvd;accgain=z.topology_accuracy-tree.topology_accuracy
        practical=(gain>=.05)&(accgain>=.10);rescue=(z.skill_vs_trivial>0)&(z.topology_accuracy>=.5)
        results[method]=dict(practical_improvement_folds=int(practical.sum()),practical_improvement_elements=list(practical[practical].index),rescue_folds=int(rescue.sum()),rescue_elements=list(rescue[rescue].index),positive_skill_folds=int((z.skill_vs_trivial>0).sum()),mean_tvd=float(z.mean_tvd.mean()),mean_topology_accuracy=float(z.topology_accuracy.mean()))
    tree_support=any(results[x]['practical_improvement_folds']>=4 for x in ['Ridge','MLP'])
    smooth_rescue=any(results[x]['rescue_folds']>=4 for x in ['Ridge','MLP'])
    identity=vv.loc['x_partition','eta2_element']>=.5 and vv.loc['x_partition','eta2_element']>=2*vv.loc[TASKS[:3],'eta2_element'].max()
    evidence=[]
    def add(h,a,metric,result,support,strength,caveat,role='NONE'):
        evidence.append(dict(hypothesis=h,analysis=a,metric=metric,result=str(result),supports=support,strength=strength,caveat=caveat,diagnosed_role=role))
    add('A chemical-anchor sparsity','training identity census','X identities per fold',5,'SUPPORTED','STRONG structural fact','Large state count does not add identities; causal improvement from adding anchors was not tested.','PRIMARY STRUCTURAL CONSTRAINT')
    add('B element-identity-dominated target','SS decomposition','whole-vector eta2 X',vv.loc['x_partition','eta2_element'],'SUPPORTED' if identity else 'NOT SUPPORTED','STRONG descriptive association','Identity also indexes chemistry-specific responses and sampling distributions; not causal ANOVA attribution.','PRIMARY EMPIRICAL ASSOCIATION')
    add('C general/dominant tree limitation under >=4-fold rule','fixed Ridge/MLP comparison','practical improvement counts',results,'SUPPORTED' if tree_support else 'NOT SUPPORTED','MODERATE within fixed suite','Majority rule not met does not mean absence of inductive-bias effects: MLP improves three folds jointly and Ridge rescues AG. Known-identity validation and tree-selected features restrict inference.','SECONDARY' if tree_support else 'NOT ESTABLISHED AS GENERAL PRIMARY')
    add('D descriptor insufficiency','analog plus manifold','analog rescue and extrapolation',dict(analog=results['Nearest-element analog'],F3_counts=mf.F3_outside_count.to_dict()),'INCONCLUSIVE','COMPATIBLE, NOT IDENTIFIED','No enriched-descriptor intervention. Preregistration did not quantify high range extrapolation; no retrospective threshold invented. TI/CO fail despite zero selected F3 marginal violations.','POSSIBLE SECONDARY')
    add('E topology selection failure','error share by argmax correctness','fraction summed TVD from wrong topology',ds['state_error_wrong_share'],'SUPPORTED' if ds['state_error_wrong_share']>=.8 else 'NOT SUPPORTED','STRONG descriptive signature','Argmax correctness and TVD are mathematically related; share alone does not prove a causal topology discontinuity.','DOMINANT FAILURE SIGNATURE')
    add('E topology discontinuity/boundary','within-element transition proximity','adjacent minus stable TVD',boundary_difference.dropna().to_dict(),'SUPPORTED' if boundary_supported else 'NOT SUPPORTED','DESCRIPTIVE within four eligible elements','AG/MO have no observed transitions; discrete grid can miss intervening transitions; no new equilibrium calls.','NOT GENERAL BOUNDARY CAUSE')
    add('F composition-space distance','within-element composition aggregates','distance correlations and high-error AUROC','See composition correlations and detection metrics','INCONCLUSIVE','HETEROGENEOUS exploratory association','Pooled scores confound identity; archived nominal 1-NN implementation not guaranteed exact minimum; prefer archived 5-NN mean.','POSSIBLE SECONDARY')
    add('phase splitting','matched element x T x correctness','overlap-conditioned split difference',split.dropna().to_dict('records'),'INCONCLUSIVE','LIMITED overlap','Only AG and MO have overlap; conditioning on correctness is outcome-conditioned; association is not causal.','NOT ESTABLISHED AS PRIMARY')
    add('C fold-specific model sensitivity','fixed model contrasts','MLP mean TVD reduction and AG Ridge rescue',dict(MLP_mean_tvd_reduction=float(tree.mean_tvd.mean()-results['MLP']['mean_tvd']),MLP_practical_folds=results['MLP']['practical_improvement_elements'],Ridge_rescue_folds=results['Ridge']['rescue_elements']),'SUPPORTED','DESCRIPTIVE fixed-suite comparison','This is an element-specific effect; it does not satisfy or alter the frozen general majority criterion.','SECONDARY EMPIRICAL EFFECT')
    csv('10_qa/STEP5D_ROOT_CAUSE_EVIDENCE_MATRIX.csv',evidence)
    def corrval(score):return float(corr[(corr.task=='x_partition')&(corr.score==score)].spearman_rho.iloc[0])
    def vector(method,col):return {r.heldout:round(float(getattr(r,col)),6) for r in m[m.method==method].itertuples()}
    answers={
      '1 STEP5C primary results unchanged?':'Primary archival hashes matched; PART2.log has a verified identical archived prefix plus completion text. End-of-run hashes are reported in the separate integrity audit.',
      '2 New CALPHAD calls?':'NO. Existing frozen outputs only; no Thermo-Calc or TC-Python imports/calls.',
      '3 Ni touched?':'NO. No Ni properties, compositions, targets or predictions loaded.',
      '4 Fraction eta2_element':float(vv.loc['fraction','eta2_element']),
      '5 Cr eta2_element':float(vv.loc['cr_partition','eta2_element']),
      '6 Nb eta2_element':float(vv.loc['nb_partition','eta2_element']),
      '7 X eta2_element':float(vv.loc['x_partition','eta2_element']),
      '8 Is X much more element-identity dominated?':bool(identity),
      '9 Effective X chemical anchors per fold?':'5 identities; 18,375 train compositions. Identity count is not a formal estimated statistical effective sample size.',
      '10 Farthest heldout from training manifold?':str(mf.affine_residual.idxmax())+' by affine residual; '+str(mf.convex_reconstruction_error.idxmax())+' by convex reconstruction error.',
      '11 Most selected F3 outside-range features?':str(mf.F3_outside_count.idxmax())+'; '+str(int(mf.F3_outside_count.max()))+' selected features.',
      '12 Affine residual versus X error rho':corrval('affine_residual'),
      '13 Convex reconstruction error versus X error rho':corrval('convex_reconstruction_error'),
      '14 Does predicted topology collapse?':dict(strict_95_percent_elements=list(tc[tc.collapsed].index),all_collapse_ratios=tc.collapse_ratio.to_dict(),caveat='MO fails with two wrong archetypes, not low-entropy single-family collapse.'),
      '15 Borrowed archetypes':borrow.set_index('heldout').borrowed_element.to_dict(),
      '16 Descriptor nearest equals borrowed?':borrow.set_index('heldout').nearest_equals_borrowed.to_dict(),
      '17 Fraction of X error from topology-wrong states':ds['state_error_wrong_share'],
      '18 Are topology-transition-adjacent states harder?':dict(adjacent_minus_stable=boundary_difference.dropna().to_dict(),general_boundary_support=boundary_supported,no_observed_transitions=['AG','MO']),
      '19 Are split states harder after conditioning?':dict(AG=float(split.set_index('heldout').loc['AG','standardized_split_minus_non_split']),MO=float(split.set_index('heldout').loc['MO','standardized_split_minus_non_split']),others='INCONCLUSIVE: no overlapping split/non-split strata; not zero effects'),
      '20 Tree X TVD by folds':vector('Tree','mean_tvd'),
      '21 Ridge X TVD by folds':vector('Ridge','mean_tvd'),
      '22 MLP X TVD by folds':vector('MLP','mean_tvd'),
      '23 Nearest-element analog X TVD by folds':vector('Nearest-element analog','mean_tvd'),
      '24 Does smooth modeling rescue X partition?':dict(majority_rescue_rule_met=smooth_rescue,model_results=results),
      '25 Do all model types still fail?':'No. Ridge rescues AG (TVD 0.428753, topology accuracy 0.980017), while MLP improves mean TVD and has positive baseline skill for SI/AG but no fold reaches the joint rescue rule. None of the diagnostic methods rescues a majority of folds.',
      '26 Descriptor insufficiency supported?':'INCONCLUSIVE; compatible with analog failure/manifold separation, but not identified without alternative descriptors or chemical anchors.',
      '27 Chemical-anchor sparsity supported?':'SUPPORTED as a structural limitation (five identities); causal role remains unisolated.',
      '28 Tree extrapolation limitation supported?':dict(frozen_practical_rule_met=tree_support,model_results=results),
      '29 Topology discontinuity supported?':'Topology-selection failure is supported descriptively; a general temperature-transition boundary explanation is not supported.' if not boundary_supported else 'Selection failure and boundary concentration are descriptively supported; causal discontinuity is not identified.',
      '30 Can distance detect high-error states?':detection[detection.scope=='POOLED'][['score','AUROC','AUPRC','high_error_prevalence']].to_dict('records'),
      '31 Primary diagnosed root cause?':'Element-identity-dominated X target combined with only five training chemical anchors; strongest empirical association plus structural limitation, not experimentally isolated causality.',
      '32 Secondary root cause?':'Model inductive bias has clear element-specific effects (MLP gains; AG Ridge rescue), although the general >=4-fold criterion is not met. Descriptor-affinity mismatch remains a possible, unisolated secondary limitation.',
      '33 Recommended STEP5E direction?':'Consider a model-inductive-bias remedy first, with chemistry coverage and descriptor-affinity checks; design separately.' if tree_support else 'Consider a mixed strategy: additional chemical anchors and binary chemical-affinity descriptors, while preserving model-specific gains and testing topology-first prediction. Do not assume model-only changes are useless. Design and preregister separately; no STEP5E execution.',
      '34 Anchor sealed?':'YES: zero training, evaluation or prediction rows.',
      '35 Ni sealed?':'YES: prospective blind test preserved; zero descriptors, compositions, targets, predictions or CALPHAD calls.'}
    js('STEP5D_FAILURE_DIAGNOSIS_SUMMARY.json',dict(stage='STEP5D_FAILURE_DIAGNOSIS_V1',status='ANALYSES_COMPLETE_PENDING_FINAL_QA',generated_utc=now(),primary_results_recomputed=True,preregistration_sha256=sha(ROOT/'01_preregistration/STEP5D_FAILURE_DIAGNOSIS_PREREGISTRATION.json'),eta2_element=vv.eta2_element.to_dict(),model_suite_results=results,tree_bias_rule_met=tree_support,smooth_majority_rescue=smooth_rescue,topology_wrong_error_share=ds['state_error_wrong_share'],root_cause_evidence=evidence,answers_35=answers,ni_sealed=True,anchor_sealed=True,new_calphad_calls=0,step5e_executed=False))
    tasktable=primary.groupby('task').agg(mean_fold_TVD=('mean_tvd','mean'),mean_fold_trivial=('trivial_tvd','mean'),positive_skill_folds=('skill_vs_trivial',lambda x:int((x>0).sum()))).reindex(TASKS)
    tasktable=tasktable.join(vv[['eta2_element']]);tasktable['total_vector_variance']=vv.SS_total/vv.n_rows
    csv('03_variance_decomposition/TASK_TRANSFERABILITY_CONTEXT.csv',tasktable.reset_index())
    (ROOT/'STEP5D_FAILURE_DIAGNOSIS_REPORT.md').write_text(
      '# STEP5D — OOD failure mechanism diagnosis\n\n'
      '**Designation:** post-hoc retrospective mechanism diagnosis. STEP5C remains the primary 6-fold LOEO result. This report is not a new unseen-element prospective validation.\n\n'
      '## Main finding\n\n'
      f'Guest-element partitioning is strongly associated with element identity: whole-vector eta-squared is {vv.loc["x_partition","eta2_element"]:.4f}, versus {vv.loc["fraction","eta2_element"]:.4f} for phase fraction, {vv.loc["cr_partition","eta2_element"]:.4f} for Cr and {vv.loc["nb_partition","eta2_element"]:.4f} for Nb. Each fold has only five training X identities despite 18,375 X-bearing training compositions. These two facts provide the strongest empirical and structural explanation for weak chemical transfer; the analysis does not experimentally isolate causation.\n\n'
      f'{ds["state_error_wrong_share"]:.2%} of summed X TVD comes from states whose dominant topology is wrong. This is the dominant error signature, not proof that proximity to a temperature boundary causes failure. Strict single-family collapse (>=95%) occurs only for ZR and AG. MO fails through two wrong archetypes while retaining substantial predicted entropy.\n\n'
      '## Frozen primary result and scale context\n\n'+tasktable.to_markdown(floatfmt='.6f')+'\n\n'
      'Phase fractions are small-error and improve over fold-train baselines, but the absolute 0.0027 versus 0.85 gap also reflects very different target spread and the dominance of the Cu-rich FCC fraction. A small fraction TVD does not imply that minor-phase relative errors or guest chemistry are accurately inferred. Cr/Nb transfer is partial and target-specific. Within-element variance is a residual category, not a causal estimate of composition or temperature effects. The decomposition uses the same six-element Domain-A universe for all four tasks; BASELINE is excluded for comparability. Composition-equal weighting is supplied as a sensitivity to missing temperatures.\n\n'
      'The X model is worse than the fold-train trivial baseline in **5/6 folds**; only ZR has marginal positive skill.\n\n'+primary[primary.task=='x_partition'][['heldout','mean_tvd','trivial_tvd','skill_vs_trivial','topology_accuracy']].to_markdown(index=False,floatfmt='.6f')+'\n\n'
      '## Chemical manifold and archetype borrowing\n\n'+mf[['nearest_element','nearest_distance','affine_rank','affine_residual','convex_reconstruction_error','F3_outside_count','F3_selected_count']].to_markdown(floatfmt='.4f')+'\n\n'+borrow[['heldout','borrowed_element','nearest_element_in_descriptor_space','nearest_equals_borrowed','borrowed_archetype_tvd']].to_markdown(index=False,floatfmt='.4f')+'\n\n'
      'The five training identities span at most four affine dimensions of 13 standardized properties. Convex-combination reconstruction errors describe distance from mixtures of these anchors; no strict high-dimensional hull membership claim is needed. Marginal range violations are not necessary for failure: TI and CO have zero selected-F3 violations. Archetype borrowing is defined by nearest mean partition vector, not by evidence that the model literally copies a training element. All six element-level correlations are exploratory; no p-values or significance claims are used.\n\n'
      '## Topology, transitions and phase splitting\n\n'+tc[['true_entropy','pred_entropy','collapse_ratio','topology_accuracy','error_share_topology_wrong']].to_markdown(floatfmt='.6f')+'\n\n'
      'Topology correctness is derived from the same vectors used in TVD. Its strong association with error is partly mathematical and cannot by itself identify a discontinuity mechanism. Transition-midpoint distance uses only consecutive observed full-grid temperatures; missing states are never bridged. No observed transition is recorded separately from stable regions of a trajectory that does transition. Unobserved between-grid transitions cannot be excluded.\n\n'+boundary_difference.rename('adjacent_minus_stable_TVD').to_frame().to_markdown(floatfmt='.6f')+'\n\n'
      'AG and MO have no observed X-topology transitions; MO contains incomplete trajectories. Among the four eligible elements, adjacent regions are not harder by the frozen majority rule.\n\n'+split.to_markdown(index=False,floatfmt='.6f')+'\n\n'
      'Split contrasts control exactly for element, temperature and topology correctness and apply only to overlapping strata. ZR/TI/CO/SI have no overlap and their adjusted effects are not identifiable. AG/MO contrasts are small descriptive differences; correctness is outcome-conditioned, so these are not causal split effects.\n\n'
      '## Fixed model-inductive-bias diagnosis\n\n'
      'Ridge: seven fixed alphas; MLP: 128-ReLU/64-ReLU/5 linear outputs, raw-component MSE, Adam, batch 1024, three fixed learning rates, at most 200 epochs and validation-MSE patience 15. Scalers use FOLD_TRAIN only; alpha/LR selection uses projected validation TVD. Each MLP uses one prespecified seed per fold, reset for each LR. All six selections were frozen before diagnostic heldout scoring. No primary tree was refitted. The analog uses only the descriptor-nearest training element and its train-only temperature-specific mean target.\n\n'+m[['heldout','method','mean_tvd','median_tvd','p95_tvd','topology_accuracy','topology_macro_f1','skill_vs_trivial']].to_markdown(index=False,floatfmt='.6f')+'\n\n'
      f'MLP reduces macro-average TVD from {tree.mean_tvd.mean():.6f} to {results["MLP"]["mean_tvd"]:.6f}, an absolute gain of {tree.mean_tvd.mean()-results["MLP"]["mean_tvd"]:.6f} (about 11.5%). TI/SI/AG meet the joint practical-improvement rule (3/6). Ridge rescues AG: TVD 0.914377 to 0.428753 and topology accuracy approximately zero to 0.980017. Model choice therefore has important element-specific effects. The fixed five-label macro F1 includes absent labels with zero F1; it is not directly comparable to overall accuracy on single-topology elements such as AG.\n\n'
      f'Frozen general practical tree-bias rule met: **{tree_support}**. Majority smooth rescue rule met: **{smooth_rescue}**. Failure to meet these rules must not be read as absence of model effects. These are practical thresholds, not statistical significance tests. Validation contains known training identities; features were inherited from tree-oriented STEP5C selection. Consequently, this fixed comparison neither exhausts model remedies nor proves that all smooth extrapolation must fail.\n\n'
      '## Error detectability\n\n'+detection[detection.scope=='POOLED'][['score','AUROC','AUPRC','high_error_prevalence']].to_markdown(index=False,floatfmt='.6f')+'\n\n'
      'High error is at or above the pooled empirical 90th percentile, including ties; within-fold thresholds are supplied as a sensitivity. AUPRC is average precision. Pooled discrimination partly exploits element identity, with only six chemical contexts. Chemical scores are constant within each fold and cannot rank its states; composition scores cannot resolve temperature variation within a composition. This is not calibrated uncertainty. No paired saved XGB/CatBoost alternatives existed, so disagreement is unavailable and no ensemble was trained. The archived nominal 1-NN value used an unsorted partition entry and is not guaranteed to be the exact nearest distance; it is retained unchanged and explicitly caveated. The archived five-neighbor mean is unaffected by ordering. Presence distance correlations are archived only because of degenerate labels.\n\n'
      '## Root-cause evidence matrix\n\n'+pd.DataFrame(evidence).to_markdown(index=False)+'\n\n'
      'The descriptor-insufficiency rule left “high range extrapolation” unquantified in the frozen preregistration. No threshold was invented after inspecting outcomes; H4 remains qualitative and INCONCLUSIVE. Identity dominance and anchor count are supported facts, but distinguishing descriptor insufficiency from sparse chemistry requires a separately designed intervention.\n\n'
      '## JMST scientific message\n\n'
      '> Different thermodynamic targets exhibit different levels of chemical transferability. Phase fractions and host-solute partitioning retain partial unseen-chemistry skill, whereas direct prediction of the guest element’s partition topology is strongly element-specific and fails severely under zero-shot LOEO. Dense composition–temperature sampling does not replace diversity in chemical identity.\n\n'
      '## Answers to the 35 required questions\n\n'+ '\n\n'.join('### '+k+'\n\n'+(json.dumps(val,ensure_ascii=False,default=str) if not isinstance(val,str) else val) for k,val in answers.items())+'\n\n'
      '## Integrity, reproducibility and stopping point\n\n'
      'See STEP5D_INTEGRITY_AUDIT.md, the source hash inventories, immutable preregistration, candidate validation histories, selection freezes and code_snapshot. The only historical archive exception is an append-only completion log with an exact matching prefix; primary predictions, metrics, thresholds, feature lists, models and fold freezes remain authoritative. No new equilibrium calculations, Ni/Anchor access or Domain-B retraining occurred. All figures are descriptive finite-dataset summaries, without population confidence intervals or independent-state inference. Their source CSVs and editable PDF/SVG exports accompany PNG previews. STEP5E is a recommendation only and has not begun.\n',encoding='utf-8')
    print('REPORT generated; model rules',tree_support,smooth_rescue,results,flush=True)
if __name__=='__main__':run()
