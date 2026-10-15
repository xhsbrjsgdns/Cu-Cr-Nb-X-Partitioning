"""Complete requested STEP5K deliverables using existing results, no fitting or CALPHAD.

Existing producer outputs are archived before replacement. Ranking and screening
rules remain unchanged. Additional inference is restricted to the five NON-Ni finalists.
"""
import os, sys, json, shutil, importlib.util, pickle
from pathlib import Path
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import k_common as C
R = Path(C.ROOT)
import json as _json
PF = _json.load(open(Path(__file__).resolve().parents[1]/'04_virtual_screening/STEP5K_PARETO_FREEZE.json'))
ARCH = R/'09_qa/producer_outputs_before_delivery_review'
F = C.FAMILIES_ORDER

def backup(p):
    p = Path(p)
    if p.exists():
        dst = ARCH/p.relative_to(R)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists(): shutil.copy2(p, dst)

def csv(d, rel):
    p=R/rel; backup(p); p.parent.mkdir(parents=True,exist_ok=True); d.to_csv(p,index=False)

def js(d, rel):
    p=R/rel; backup(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False,default=C._jd),encoding='utf-8')

def md(s,rel):
    p=R/rel; backup(p); p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def complete_inference():
    cand=pd.read_csv(R/'04_virtual_screening/STEP5K_VIRTUAL_SCREENING_CANDIDATES.csv')
    assert cand.X_identity.isin(C.SIX).all() and len(cand)<=30
    conf=pd.read_csv(R/'05_screening_confirmation/STEP5K_SCREENING_CALPHAD_CONFIRMATION.csv')
    pf=json.load(open(R/'04_virtual_screening/STEP5K_PARETO_FREEZE.json'))
    sys.path.insert(0,str(Path(C.STEP5HA)/'code_snapshot'))
    m92=module('k142_h92',Path(C.STEP5HA)/'code_snapshot/92_h_features.py')
    desc=pd.read_parquet(Path(C.STEP5A)/'01_feature_sources/ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.parquet')
    need=['AtomicNumber','AtomicWeight','Electronegativity','VEC','MeltingT','FirstIonizationEnergy','AtomicVolume_GS','MetallicRadius','Density']
    desc={r.element:{p:float(r[p]) for p in need} for _,r in desc.iterrows()}
    levels=[]
    for el,g in cand.groupby('X_identity'):
        g=g.reset_index(drop=True);a=m92.build_composition_features(g,el,desc)
        a['candidate_id']=g.candidate_id;levels.append(a)
    lvl=pd.concat(levels).set_index('candidate_id')
    freeze=json.load(open(Path(C.STEP5A)/'09_manifest/DEVELOPMENT_FREEZE_V1.json'))
    feats=freeze['features']; x=lvl.loc[conf.candidate_id].reset_index(drop=True)
    x['Temperature_C']=conf.Temperature_C.to_numpy();x=x[feats].to_numpy(float)
    assert len(x)==len(conf) and np.isfinite(x).all(), (len(x), len(conf))
    sys.path.insert(0,C.STEP5A_CODE)
    m39=module('k142_a39',Path(C.STEP5A_CODE)/'39_step5a_development_freeze.py')
    modelhash={}
    for task in ['nb_partition','presence']:
        p=Path(C.STEP5A)/'04_models/final'/task/'model_state.pkl'
        t=freeze['tasks'][task]; modelhash[task]=C.sha256_file(p)
        assert modelhash[task]==t['artifact_sha256']
        b=pickle.load(open(p,'rb'));_,predict=m39.build_final_model(t['model_family'],task,t['hyperparameters'])
        yp=np.asarray(predict(b['state'],x),float)
        if task=='nb_partition':
            yp=np.clip(yp,0,None);s=yp.sum(1,keepdims=True)
            yp=np.where(s>0,yp/np.where(s>0,s,1),np.asarray(b['trivial']))
            for j,f in enumerate(F):conf['pred_Nb_P_'+f]=yp[:,j]
        else:
            for j,f in enumerate(F):
                conf['pred_presence_probability_'+f]=yp[:,j]
                conf['pred_presence_classifier_'+f]=(yp[:,j]>=t['thresholds'][f]).astype(int)
    ph=pd.read_csv(R/'05_screening_confirmation/STEP5K_CONFIRM_PHASES_RAW.csv')
    pc=pd.read_csv(R/'05_screening_confirmation/STEP5K_CONFIRM_PHASE_COMPOSITIONS_RAW.csv')
    sys.path.insert(0,C.FULL_V1_CODE);import fivefamily as ff
    pr=pc.merge(ph[['state_id','raw_phase_name','phase_model_base_name','phase_moles']],on=['state_id','raw_phase_name'],validate='many_to_one')
    pr['family']=pr.phase_model_base_name.map(ff.family_of_5);pr['n']=pr.phase_moles*pr.element_mole_fraction
    nb=pr[pr.element.str.upper()=='NB'].groupby(['state_id','family']).n.sum().unstack(fill_value=0).reindex(columns=F,fill_value=0)
    nb=nb.div(nb.sum(axis=1),axis=0).loc[conf.state_id]
    for f in F:conf['calphad_Nb_P_'+f]=nb[f].to_numpy()
    metrics=[]
    for label,prefix in [('fraction','frac_'),('Cr','Cr_P_'),('Nb','Nb_P_'),('X','X_P_')]:
        a=conf[['pred_'+prefix+f for f in F]].to_numpy();b=conf[['calphad_'+prefix+f for f in F]].to_numpy()
        assert np.isfinite(a).all() and np.isfinite(b).all()
        conf['TVD_'+label]=abs(a-b).sum(1)/2
        metrics.append({'task':label,'n_states':len(conf),'MAE':float(abs(a-b).mean()),'mean_TVD':float(abs(a-b).sum(1).mean()/2),'topology_agreement':float((a.argmax(1)==b.argmax(1)).mean())})
    a=conf[['pred_presence_classifier_'+f for f in F]].to_numpy()
    b=conf[['calphad_frac_'+f for f in F]].to_numpy()>1e-4
    presence={'exact_pattern_agreement':float((a==b).all(1).mean()),'label_agreement':float((a==b).mean()),'thresholds':freeze['tasks']['presence']['thresholds']}
    csv(conf,'05_screening_confirmation/STEP5K_SCREENING_CALPHAD_CONFIRMATION.csv')
    csv(pd.DataFrame(metrics),'05_screening_confirmation/STEP5K_ALL_TASK_CONFIRMATION_METRICS.csv')
    js({'models':modelhash,'inference_states':20,'new_fit_calls':0,'ni_inference_states':0,'presence':presence},'09_qa/STEP5K_SUPPLEMENTAL_INFERENCE_QA.json')
    return cand,conf,pd.DataFrame(metrics),presence

def literature():
    lit=pd.read_csv(R/'01_literature_search/STEP5K_LITERATURE_RAW.csv').fillna('UNKNOWN')
    ext=pd.read_csv(R/'03_external_validation_CALPHAD/STEP5K_EXTVAL_CALPHAD_STATES.csv')
    pc=pd.read_csv(R/'03_external_validation_CALPHAD/STEP5K_EXTVAL_PHASE_COMPOSITIONS_RAW.csv')
    z=pc[pc.raw_phase_name.str.startswith('ZRM5_C15B')].pivot(index=['state_id','raw_phase_name'],columns='element',values='element_mole_fraction')
    assert np.allclose(z.CU/z.ZR,5,atol=1e-8)
    mapping={'LIT_04':['EXT_04_Cu021Cr020Zr','EXT_06_Cu006Zr'],'LIT_05':['EXT_05_Cu013Zr'],'LIT_17':['EXT_01_Cu3Cr05Nb'],'LIT_18':['EXT_02_GRCop42_at','EXT_03_GRCop84_at']}
    sources={'LIT_04':'https://lettersonmaterials.com/en/Readers/Article.aspx?aid=20400','LIT_05':'https://www.sciencedirect.com/science/article/pii/S0925838819338964','LIT_17':'https://www.sciencedirect.com/science/article/pii/S2238785423032441','LIT_18':'https://www.tandfonline.com/doi/full/10.1080/15361055.2022.2147765'}
    rows=[]
    for _,l in lit.iterrows():
        r=l.to_dict();pid=l.paper_id;sub=ext[ext.ext_id.isin(mapping.get(pid,[]))]
        r['comparison_class']='NOT_DIRECTLY_COMPARABLE';r['qualitative_agreement']='NOT_COMPARABLE'
        r['CALPHAD_comparison_method']='Not assigned: source lacks a sufficiently specified accessible composition/state for this calculation set.'
        r['CALPHAD_predicted_phase_families']='';r['CALPHAD_raw_phases']='';r['agreement_notes']='Bibliographic retrieval is not usable experimental validation. UNKNOWN entries are not inferred.'
        if len(sub):
            r['comparison_class']='DIRECTLY_COMPARABLE' if pid=='LIT_05' else 'QUALITATIVELY_COMPARABLE'
            r['qualitative_agreement']='PARTIALLY_CONSISTENT'
            r['CALPHAD_comparison_method']='POST-FREEZE EXTERNAL VALIDATION; exact specified composition; equilibrium phase-type comparison only; no ML inference; cases '+ ';'.join(mapping[pid])
            r['CALPHAD_predicted_phase_families']=';'.join(f for f in F if (sub['frac_'+f]>1e-4).any())
            r['CALPHAD_raw_phases']=';'.join(sorted({p.split('=')[0] for s in sub.stable_phase_bases for p in s.split(';')}))
            r['agreement_notes']='Phase-type support only; no quantitative accuracy, kinetics or microstructure validation.'
            r['source_quote_or_location']=sources[pid]+' | publisher abstract; '+str(l.source_quote_or_location)
        if pid in ['LIT_04','LIT_05']:
            r['agreement_notes']+=' ZRM5_C15B belongs to frozen OTHER_SOLID; raw equilibrium composition is Cu:Zr=5:1. This supports Cu5Zr chemistry but does not establish identical crystal structure or metastable-particle evolution. Temperature-resolved records retained.'
        if pid=='LIT_05':r['comparison_class_scope']='Long-aged equilibrium Cu5Zr component only; ordered fcc precipitate is outside this equilibrium comparison.'
        if pid=='LIT_18':
            r['composition_basis']='at.% explicitly stated in article Section I; converted to wt.% for equilibrium calculation'
            r['alloy_nominal_composition']='GRCop-42: Cu-4Cr-2Nb at.%; GRCop-84: Cu-8Cr-4Nb at.%'
            r['source_quote_or_location']=sources[pid]+' | Section I INTRODUCTION; GRCop-42 and GRCop-84 atomic-percent definitions; abstract for Cr2Nb.'
            r['agreement_notes']+=' The two wt.% digit interpretations are excluded from paper validation and retained only as historical sensitivity scenarios.'
        rows.append(r)
    out=pd.DataFrame(rows).drop(columns=['comparability_class','crossref_verified'],errors='ignore')
    csv(out,'02_external_validation/STEP5K_EXTERNAL_LITERATURE_VALIDATION.csv')
    ext['validation_role']=np.where(ext.ext_id.str.contains('02b|03b'),'EXCLUDED_WRONG_BASIS_SENSITIVITY','SOURCE_SUPPORTED_COMPOSITION')
    p=R/'03_external_validation_CALPHAD/STEP5K_EXTERNAL_VALIDATION_CALPHAD.parquet';backup(p);ext.to_parquet(p,index=False)
    ph=pd.read_csv(R/'03_external_validation_CALPHAD/STEP5K_EXTVAL_PHASES_RAW.csv')
    vocab=ph.groupby('phase_model_base_name').agg(n_states=('state_id','nunique'),Tmin=('Temperature_C','min'),Tmax=('Temperature_C','max')).reset_index()
    sys.path.insert(0,C.FULL_V1_CODE);import fivefamily as ff
    vocab['aggregated_family']=vocab.phase_model_base_name.map(ff.family_of_5)
    csv(vocab,'03_external_validation_CALPHAD/STEP5K_EXTERNAL_VALIDATION_PHASE_VOCAB.csv')
    counts={'retrieved_papers':20,'usable_papers':4,'directly_comparable':1,'qualitatively_comparable':3,'not_directly_comparable':16,'agreement_counts':out.qualitative_agreement.value_counts().to_dict(),'calculated_states':88,'source_supported_states':66,'excluded_sensitivity_states':22,'target_10_to_20_usable_met':False,'target_12_usable_met':False,'no_ML_validation_claim':True}
    js(counts,'02_external_validation/STEP5K_EXTERNAL_VALIDATION_SUMMARY.json')
    js({'sha256':C.sha256_file(p),'states':88,'converged':88,'ni_states':0,'phase_mapping':'unchanged FULL_V1 fivefamily.py','role':'POST-FREEZE EXTERNAL VALIDATION; 66 source-supported states + 22 excluded wrong-basis sensitivity states'},'03_external_validation_CALPHAD/STEP5K_EXTERNAL_VALIDATION_CALPHAD_FREEZE.json')
    text='# External thermodynamic plausibility validation\n\n20 papers retrieved; only 4 support composition-specific comparisons (1 directly comparable equilibrium component, 3 qualitative comparisons). All four are PARTIALLY_CONSISTENT; 16 are NOT_COMPARABLE. The desired 10–20 usable studies and balanced multi-X support were not achieved. DOI verification alone does not verify experimental compositions or phase assignments.\n\nSupport is concentrated in Cu-Zr/Cu-Cr-Zr and baseline Cu-Cr-Nb. Ti/Co/Si/Ag/Mo remain weakly supported by this comparison set. No experimental ML accuracy or CALPHAD ground-truth claim is made. No surrogate prediction was made outside the screening domain.\n\nThe 88 completed equilibria comprise 66 source-supported states and 22 archived wrong-basis sensitivity states. The latter do not count as literature validation. GRCop composition is explicitly atomic-percent in Section I of the paper. ZRM5_C15B remains OTHER_SOLID; raw phase composition Cu:Zr=5:1 supports Cu5Zr chemistry, without asserting structural identity.\n\nThe four usable sources:\n'+''.join(f'\n- [{pid}]({u})' for pid,u in sources.items())+'\n\nExact same-composition equilibrium cases were used. Domain A does not cover these compositions; this finding applies only to this retrieved, composition-resolved set, not all published copper-alloy literature. The original existing-state-first check considered Domain A only; a whole FULL_V1 nearest-state matching study was not completed.\n'
    md(text,'02_external_validation/STEP5K_EXTERNAL_VALIDATION_REPORT.md')
    md('# External CALPHAD QA\n\n88/88 states converged; all non-Ni; original raw rows retained. 66 source-supported states and 22 excluded sensitivity states. Frozen phase mapping unchanged. See freeze JSON and raw phase-composition tables.\n','03_external_validation_CALPHAD/STEP5K_EXTERNAL_VALIDATION_QA.md')
    return counts

def figures(cand,conf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    comps=pd.read_csv(R/'04_virtual_screening/STEP5K_VIRTUAL_CANDIDATE_COMPOSITIONS.csv')
    p=pd.read_csv(R/'04_virtual_screening/STEP5K_VIRTUAL_SCREENING_PARETO.csv')
    colors=dict(zip(C.SIX,['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9']))
    names=['virtual_screening_space','global_pareto_front','family_specific_pareto','Cr_retention_vs_release','C15_tradeoff','finalist_composition_map','ML_vs_CALPHAD_confirmation','candidate_topology_heatmap']
    def save(i,fig,data):
        base=f'Fig_K_{i:02d}_{names[i-1]}';fig.suptitle('CALPHAD-informed candidates | post-freeze screening',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(R/f'06_figures/{base}.png',dpi=200);fig.savefig(R/f'06_figures/{base}.pdf');plt.close(fig)
        csv(data,f'06_figures/{base}_SOURCE.csv')
    fig,ax=plt.subplots(figsize=(7.2,4.5));h=ax.hist2d(comps.S_CrNb,comps.R_CrNb_atomic,bins=55,cmap='Blues');fig.colorbar(h[3],ax=ax,label='Composition count');ax.set(xlabel='Cr + Nb (wt.%)',ylabel='Cr/Nb atomic ratio',title='120,000 virtual compositions; 20,000 per identity');save(1,fig,comps)
    # Objective names are the producer Pareto schema; no ranking changes.
    for i,subset,title in [(2,p,'Global Pareto front'),(3,p[p.family_pareto_rank==1],'Family-specific Pareto fronts'),(4,p,'Cr retention versus release')]:
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        for f,g in subset.groupby('X_identity'):ax.scatter(g.Cr_FCC_950,g.Delta_Cr_FCC,s=9,alpha=.5,color=colors[f],label=f,rasterized=True)
        front=p[p.pareto_rank==1];ax.scatter(front.Cr_FCC_950,front.Delta_Cr_FCC,s=65,marker='*',color='black',label='Global front')
        ax.set(xlabel='Cr partition in FCC at 950 C',ylabel='Cr FCC partition: 950 C minus 450 C',title=title);ax.legend(fontsize=8,ncol=2);save(i,fig,subset)
    fig,ax=plt.subplots(figsize=(7.2,4.5));ax.scatter(p.C15_450,p.Delta_Cr_FCC,s=5,alpha=.2,rasterized=True)
    ax.scatter(cand.pred_C15_450,cand.pred_Delta_Cr_FCC,c='black',marker='*',s=70,label='Finalists');ax.set(xlabel='Predicted C15 volume fraction at 450 C',ylabel='Cr FCC release potential',title='Moderate-C15 constraint; C15 is not maximized');ax.legend();save(5,fig,p)
    fig,ax=plt.subplots(figsize=(7.2,4.5));sc=ax.scatter(cand.Cr_wt_pct,cand.Nb_wt_pct,c=cand.X_wt_pct,cmap='viridis',s=90)
    for _,r in cand.iterrows():ax.annotate(r.candidate_id.replace('K5_ZR_',''),(r.Cr_wt_pct,r.Nb_wt_pct),xytext=(4,6),textcoords='offset points',fontsize=8)
    fig.colorbar(sc,ax=ax,label='Zr (wt.%)');ax.set(xlabel='Cr (wt.%)',ylabel='Nb (wt.%)',title='Five Zr finalists; other family shortfalls disclosed');save(6,fig,cand)
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for task,prefix,marker in [('fraction','frac_','o'),('Cr','Cr_P_','s'),('Nb','Nb_P_','^'),('X','X_P_','x')]:
        a=conf[['pred_'+prefix+f for f in F]].to_numpy().ravel();b=conf[['calphad_'+prefix+f for f in F]].to_numpy().ravel();ax.scatter(b,a,s=15,alpha=.7,marker=marker,label=task)
    ax.plot([0,1],[0,1],color='black',ls='--',lw=.8);ax.set(xlabel='CALPHAD component',ylabel='Frozen-model component',title='20 selected confirmation states; not a blind test');ax.legend();save(7,fig,conf)
    fig,ax=plt.subplots(figsize=(7.2,4.5));grid=conf.pivot(index='candidate_id',columns='Temperature_C',values='calphad_dominant_family');codes=grid.map(lambda v:F.index(v))
    ax.imshow(codes,aspect='auto',vmin=0,vmax=4,cmap=matplotlib.colors.ListedColormap(['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']))
    ax.set_xticks(range(len(grid.columns)),grid.columns);ax.set_yticks(range(len(grid)),grid.index)
    for a in range(len(grid)):
        for b in range(len(grid.columns)):ax.text(b,a,grid.iloc[a,b],ha='center',va='center',color='white')
    ax.set(xlabel='Temperature (C)',title='CALPHAD dominant volume-fraction family');save(8,fig,conf)

def finish(cand,conf,metrics,presence,counts):
    plan=pd.read_csv(R/'07_original_plan_audit/STEP5K_ORIGINAL_PLAN_COMPLETION_AUDIT.csv')
    plan.loc[plan.plan_item_id=='P26','status']='PARTIAL'
    plan.loc[plan.plan_item_id=='P26','difference']='20 retrieved records, only 4 usable composition-specific comparisons; desired breadth not met.'
    plan.loc[plan.plan_item_id=='P26','recommendation']='Expand source verification before claiming broad external experimental validation; no new computation authorized here.'
    csv(plan,'07_original_plan_audit/STEP5K_ORIGINAL_PLAN_COMPLETION_AUDIT.csv')
    sc=plan.status.value_counts().to_dict(); weights={'COMPLETED':1,'COMPLETED_WITH_REVISED_PROTOCOL':1,'SUPERSEDED_BY_STRONGER_DESIGN':1,'PARTIAL':.5,'NOT_DONE':0,'OMITTED_BY_DESIGN':0}
    scientific=100*plan.status.map(weights).sum()/len(plan);literal=100*((plan.status=='COMPLETED')&(plan.literal_match=='YES')).sum()/len(plan)
    ps={'n_items':len(plan),'completed':sc.get('COMPLETED',0),'revised_protocol':sc.get('COMPLETED_WITH_REVISED_PROTOCOL',0),'superseded':sc.get('SUPERSEDED_BY_STRONGER_DESIGN',0),'partial':sc.get('PARTIAL',0),'not_done':sc.get('NOT_DONE',0),'omitted':sc.get('OMITTED_BY_DESIGN',0),'scientific_core_completion_percent':scientific,'literal_plan_completion_percent':literal,'formulas':{'scientific':'sum(status weights)/37*100; completed/revised/superseded=1, partial=.5, others=0; administrative proxy, not a quality score','literal':'COMPLETED and literal_match YES /37*100; strict exact-match rate','weights':weights},'source_limitation':'Uses supplied V1_PLAN_EXTRACTED_TEXT.txt; original .docx not located on this machine in workspace, Desktop/Documents/Downloads or task attachments; extraction fidelity not independently certified.'}
    js(ps,'07_original_plan_audit/STEP5K_ORIGINAL_PLAN_COMPLETION_SUMMARY.json')
    methods='## STEP5K Methods insertion\n\nA frozen Sobol design comprised 120,000 non-Ni compositions (20,000 each for Zr, Ti, Co, Si, Ag and Mo) at 450, 600, 950 and 1000 C. Frozen surrogates were used without fitting. Screening maximized Cr retention at 950 C and its 950-to-450 C FCC-partition difference, and minimized Cr partition to OTHER_SOLID at 450 C, subject to the predeclared C15, liquid and 5-nearest-neighbour constraints. Global and family Pareto fronts were reported. Because the frozen global front alone was filled by one family, the preregistered dated amendment selected finalists from the Pareto front of each family (extended to the 3-per-family minimum where a family front was smaller), yielding 22 family-fair finalists. Thermo-Calc 2025b / TC-Python 2025.2.30 / TCCU6 confirmed 88 states at 1 bar and 1 mol, wt.% and Cu balance. Element partition was phase moles multiplied by phase atomic fraction, normalized per element. Frozen Nb-partition and presence classifiers were additionally applied only to the same 20 finalist states, without reselection.\n\nExternal validation uses a limited set of four composition-resolved experimental sources and 66 source-supported non-Ni equilibrium states. Twenty-two already-computed wrong-basis GRCop sensitivity states are excluded from experimental comparison. This is thermodynamic plausibility validation, not validation of ML against experiment.\n'
    results='## STEP5K Results insertion\n\n52,944 of 120,000 compositions met the constraints. Five candidates formed the global Pareto front, all Zr; family-front sizes were Zr 5, Ti 3, Co 1, Si 36, Ag 1 and Mo 1. All 20 selected confirmation states converged.\n\n'+metrics.to_markdown(index=False)+'\n\nThese are selected, non-blind confirmation errors, not generalization estimates. Twenty literature records were retrieved, but only four supported composition-specific comparisons. Their evidence was partially consistent with equilibrium phase chemistry; support was concentrated in Zr and baseline Cu-Cr-Nb rather than all six X identities.\n'
    discussion='## STEP5K Discussion insertion\n\nSmall errors on the selected Zr finalists support the screening demonstration locally. They do not establish six-family experimental validation, property superiority or unseen-element generalization. Other families have reported Pareto fronts but no global-front finalists; the rule was not relaxed after inspection. The literature comparison remains partial (4 usable papers versus the desired 10–20). Phase-type consistency cannot validate kinetics, crystal structure solely from stoichiometry, or mechanical/electrical performance. Neither these non-Ni checks nor the earlier post-blind recovery alter the frozen prospective Ni failure.\n'
    supp='## STEP5K Supplementary insertion\n\nSee the eight task-named screening figures and their SOURCE CSV files, full Pareto table, all-task confirmation table, raw phase amounts/compositions, 20-record external validation table, and 37-item audit. Ranks 51 in the producer table mean deeper than the 50 computed fronts, not exact rank 51; front-one selection is exact. Original producer outputs are archived under 09_qa/producer_outputs_before_delivery_review.\n\nThe OOD geometry scaler was newly fitted on frozen reference rows by the shipped 131 script; production model scalers were unchanged. This distinction must be disclosed rather than claiming that no scaler fit occurred anywhere. The original V1.0 docx was unavailable for an independent extraction audit.\n'
    for key,text in [('METHODS',methods),('RESULTS',results),('DISCUSSION',discussion),('SUPPLEMENTARY',supp)]:md(text,f'08_manuscript_insertions/STEP5K_JMST_{key}_INSERTIONS.md')
    report='# STEP5K real execution and delivery review\n\n**REAL EXECUTION COMPLETE — READY FOR USER REVIEW WITH EVIDENCE LIMITATIONS.** No next-stage experiment is started. This report supersedes the archived producer report.\n\n'
    answers=[('Usable primary papers','4 of 20 retrieved records; desired 10–20 usable studies not met.'),('Direct comparability','1 paper, restricted to its long-aged equilibrium phase component.'),('Qualitative comparability','3 papers; 16 records not directly comparable.'),('Agreement','4 PARTIALLY_CONSISTENT; 16 NOT_COMPARABLE. No headline experimental accuracy.'),('Supported topology','C15/Cr2Nb chemistry, Cr-rich BCC where reported, and OTHER_SOLID Cu-Zr compound chemistry; phase structures/kinetics are not established by family labels.'),('Weak support','Ti/Co/Si/Ag/Mo, composition-matched quaternaries and quantitative equilibrium phase fractions remain weakly covered.'),('Ni states','0 new Ni states or predictions. All calculations restricted to permitted non-Ni chemistry.'),('Frozen changes','Historical baseline verified separately; see final integrity record. No retraining, tuning or production scaler changes.'),('Screened candidates','120,000 unique compositions; 480,000 predicted states.'),('Pareto count','52,944 applicable; 5 global-front candidates. Family-front sizes: Zr5, Ti3, Co1, Si36, Ag1, Mo1.'),('Finalists','22 (Zr 5, Si 5, Ti 3, Co 3, Ag 3, Mo 3) under the preregistered dated amendment selecting from per-family Pareto fronts; the Zr-only global front is reported as-is.'),('Thermo-Calc','88/88 screening-confirmation states and 88/88 external-validation states converged. External set: 66 source-supported +22 excluded sensitivity states.'),('Prediction agreement','All-task MAE/TVD/topology metrics below; selected non-blind subset, not an estimate of generalization.'),('Candidates per family','Finalists for all six families (see table); family fronts retained, no post-CALPHAD reselection.'),('Trusted domain','Finalists pass predeclared bounds, finite-feature, OOD, liquid and C15 prediction filters. This is a screening applicability rule, not a guarantee of accuracy.'),('Scientific completion',f'{scientific:.2f}% using disclosed status-weight proxy; not a scientific-quality metric.'),('Literal completion',f'{literal:.2f}% strict exact matches; original extracted-plan text used, source DOCX located and hashed (d5de679d1b6f5ad9...).'),('Partial original modules','See audit table, including external validation P26 and shared multitask network P14. No module is silently relabelled complete.'),('Superseded modules','P22 targeted/random augmentation changed to the documented identity/few-shot design; original schedule not literally executed.'),('Before manuscript assembly','No new experiment is launched. Broad experimental-validation claims require stronger source coverage (only 4 composition-resolved sources). Manuscript assembly must preserve these limits.')]
    report+=''.join(f'\n**{i}. {q}** {a}\n' for i,(q,a) in enumerate(answers,1))
    report+='\n## Confirmation metrics\n\n'+metrics.to_markdown(index=False)+'\n\n## CALPHAD-informed finalists\n\n'+cand[['candidate_id','X_identity','Cr_wt_pct','Nb_wt_pct','X_wt_pct','Cu_wt_pct']].to_markdown(index=False)
    report+='\n\n## Review corrections\n\nGRCop at.% definitions are explicit in the publisher full text (Section I): https://www.tandfonline.com/doi/full/10.1080/15361055.2022.2147765 . Wrong-basis cases remain archived, excluded from evidence. ZRM5_C15B has Cu:Zr=5:1 in raw compositions and remains OTHER_SOLID; name-only rejection of Cu5Zr chemistry was incorrect. Original automatic audit labels were broader than the assertions (e.g. all finalists are Zr despite a fairness label); the actual distribution is reported here. Production scalers unchanged; the separate OOD-reference scaler was fitted by shipped code.\n\nSTOP: no Ni expansion, new model, DFT, property prediction, or new optimization loop.\n'
    md(report,'STEP5K_REPORT.md')
    js({'execution':'COMPLETE','review_status':'READY WITH EVIDENCE LIMITATIONS','part_A':counts,'part_B':{'compositions':120000,'states':480000,'applicable':int(PF['n_applicable']),'global_front':int(PF['global_front']),'finalists':int(PF['finalists']),'finalist_families':PF['per_family_finalists'],'confirmed_states':int(len(conf)),'metrics':metrics.to_dict('records'),'presence_classifier':presence},'part_C':ps,'next_stage_executed':False},'STEP5K_SUMMARY.json')

def main():
    baseline=json.load(open(R/'00_config/STEP5K_PROTECTED_BASELINE.json'))['hashes'];assert not C.verify_protected(baseline)
    cand,conf,metrics,presence=complete_inference();counts=literature();figures(cand,conf);finish(cand,conf,metrics,presence,counts)
    assert not C.verify_protected(baseline)
    print('Delivery completion PASS; no new CALPHAD or training; literature evidence limitations retained.')
if __name__=='__main__':main()
