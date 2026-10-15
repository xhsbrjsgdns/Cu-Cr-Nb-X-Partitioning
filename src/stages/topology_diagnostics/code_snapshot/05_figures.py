"""Ten source-backed scientific figures. No synthetic data or unstated exclusions."""
from d_common import *
import sys,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
SKILL=Path(r'C:\Users\admin\.codex\skills\nature-figure\scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'savefig.facecolor':'white'})
COLORS=['#4B7396','#B77B59','#829D75','#A186AF','#C6B567']
records=[]
def read(p):return pd.read_csv(ROOT/p)
def canvas(height=3.6):
    fig,ax=plt.subplots(figsize=(7.2,height));fig.subplots_adjust(left=.15,right=.97,bottom=.2,top=.83);return fig,ax
def save(fig,stem,source):
    out=ROOT/'09_figures'/stem
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(out)+'.alignment.json',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(out)+'.png',dpi=600)
    fig.savefig(str(out)+'.pdf')
    fig.savefig(str(out)+'.svg')
    plt.close(fig)
    for script,args,kind in [('audit_pdf_text.py',['--json'],'text'),('audit_figure_collisions.py',['--json-out',str(out)+'.collision.json','--overlay-pdf',str(out)+'.collision-overlay.pdf'],'collision')]:
        r=subprocess.run([sys.executable,str(SKILL/script),str(out)+'.pdf']+args,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (ROOT/'10_qa'/f'{stem}_{kind}_audit.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
        records.append(dict(figure=stem,source=source,audit=kind,exit_code=r.returncode))
    print('FIGURE',stem,flush=True)

def run():
    v=read('03_variance_decomposition/TASK_ELEMENT_VARIANCE_DECOMPOSITION.csv');v=v[v.target_component=='WHOLE_VECTOR'].set_index('task').loc[TASKS]
    fig,ax=canvas();xs=np.arange(4);ax.bar(xs,v.eta2_element,color=[COLORS[0]]*3+[COLORS[3]],width=.6)
    ax.set(xticks=xs,xticklabels=['Phase fraction','Cr partition','Nb partition','X partition'],ylim=(0,1),ylabel='Between-element variance / total variance',title='Element identity accounts for most X-partition variance')
    for x,y in zip(xs,v.eta2_element):ax.text(x,y+.035,f'{y:.1%}',ha='center',va='bottom')
    save(fig,'Fig_S5D_01_task_variance_decomposition','TASK_ELEMENT_VARIANCE_DECOMPOSITION.csv')
    a=read('05_topology_analysis/X_PARTITION_ARCHETYPE_DIAGNOSIS.csv');a=a[a.scope=='overall'].set_index('heldout').loc[E]
    mat=np.array([a.loc[e,[f'{mode}_{f}' for f in F]].to_numpy(float) for e in E for mode in ['true','pred']])
    fig,ax=canvas(5.0);fig.subplots_adjust(top=.89,bottom=.12,right=.84)
    im=ax.imshow(mat,aspect='auto',vmin=0,vmax=1,cmap='Blues');ax.set(xticks=np.arange(5),xticklabels=['FCC','BCC','C15','Other','Liquid'],yticks=np.arange(12),yticklabels=[f'{e}  {mode}' for e in E for mode in ['True','Pred']],title='True and predicted mean X-partition archetypes')
    for i in range(12):
        for j in range(5):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]>.6 else '#222222',fontsize=7)
    cb=fig.add_axes([.88,.18,.025,.61]);fig.colorbar(im,cax=cb,label='Mean partition fraction');cb.set_label('<colorbar>')
    save(fig,'Fig_S5D_02_X_partition_true_vs_pred_archetypes','X_PARTITION_ARCHETYPE_DIAGNOSIS.csv')
    t=read('05_topology_analysis/X_PARTITION_TOPOLOGY_COLLAPSE.csv').set_index('heldout').loc[E]
    fig,ax=canvas(4.8);fig.subplots_adjust(top=.8,bottom=.12)
    mat=np.array([t.loc[e,[f'{mode}_dominant_{f}' for f in F]].to_numpy(float) for e in E for mode in ['true','pred']]);left=np.zeros(12)
    for j,f in enumerate(F):ax.barh(np.arange(12),mat[:,j],left=left,color=COLORS[j],label=f.replace('_SOLID',''),height=.72);left+=mat[:,j]
    ax.invert_yaxis();ax.set(yticks=np.arange(12),yticklabels=[f'{e}  {mode}' for e in E for mode in ['True','Pred']],xlim=(0,1),xlabel='Fraction of states with each dominant family',title='Dominant-family failure need not be a single-family collapse')
    ax.legend(ncol=5,loc='lower center',bbox_to_anchor=(.5,1.10),handlelength=1)
    save(fig,'Fig_S5D_03_topology_collapse_by_element','X_PARTITION_TOPOLOGY_COLLAPSE.csv')
    mf=read('04_element_manifold/MANIFOLD_WITH_F3_SUMMARY.csv').set_index('heldout').loc[E]
    fig,ax=canvas();xs=np.arange(6)
    for i,(c,label) in enumerate([('affine_residual','Affine residual'),('convex_reconstruction_error','Convex reconstruction error')]):ax.bar(xs+(i-.5)*.34,mf[c],width=.32,label=label,color=COLORS[i])
    ax.set(xticks=xs,xticklabels=E,ylabel='Standardized descriptor distance',title='Heldout chemistries depart from five-anchor support');ax.legend(loc='upper left')
    save(fig,'Fig_S5D_04_element_manifold_extrapolation','LOEO_ELEMENT_MANIFOLD_DIAGNOSTICS.csv')
    f3=read('04_element_manifold/F3_SELECTION_VS_EXTRAPOLATION.csv')
    names=sorted(f3.feature.unique());codes=np.zeros((len(names),6))
    for i,f in enumerate(names):
        for j,e in enumerate(E):
            r=f3[(f3.feature==f)&(f3.heldout==e)].iloc[0];codes[i,j]=0 if not r.selected_this_fold else 2 if r.outside_range else 1
    fig,ax=canvas(5.1);fig.subplots_adjust(left=.46,right=.96,bottom=.12,top=.81)
    ax.imshow(codes,aspect='auto',vmin=0,vmax=2,cmap=ListedColormap(['#EEEEEE','#9BB4C8','#B97C67']))
    ax.set(xticks=np.arange(6),xticklabels=E,yticks=np.arange(len(names)),yticklabels=[n.replace('FirstIonizationEnergy','IE1').replace('Electronegativity','EN').replace('AtomicNumber','Z').replace('MetallicRadius','Radius') for n in names],title='Selected F3 features versus marginal range violations')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(['#EEEEEE','#9BB4C8','#B97C67'],['Not selected','Selected / in range','Selected / outside'])],loc='upper center',bbox_to_anchor=(.58,.95),ncol=1,frameon=False)
    save(fig,'Fig_S5D_05_F3_range_extrapolation','F3_SELECTION_VS_EXTRAPOLATION.csv')
    er=read('05_topology_analysis/X_PARTITION_ERROR_BY_TOPOLOGY_CORRECTNESS.csv');er=er[er.scope=='heldout_element+topology_status']
    fig,ax=canvas()
    for i,(status,label) in enumerate([('TOPOLOGY_CORRECT','Correct topology'),('TOPOLOGY_WRONG','Wrong topology')]):
        vals=er[er.topology_status==status].set_index('heldout_element').reindex(E).mean_tvd
        ax.bar(xs+(i-.5)*.34,vals,width=.32,label=label,color=[COLORS[0],COLORS[1]][i])
    ax.set(xticks=xs,xticklabels=E,ylim=(0,1.12),ylabel='Mean X-partition TVD',title='Most error is associated with incorrect topology');ax.legend(loc='upper left')
    fig.text(.16,.035,'Missing bar = no states in that category; correctness and TVD are coupled statistics.',fontsize=7)
    save(fig,'Fig_S5D_06_X_error_by_topology_correctness','X_PARTITION_ERROR_BY_TOPOLOGY_CORRECTNESS.csv')
    p=pd.read_parquet(ROOT/'02_frozen_inputs/DIAGNOSTIC_STATES.parquet',columns=['heldout_element','boundary_group','x_tvd'])
    bg=p.groupby(['heldout_element','boundary_group']).x_tvd.mean().unstack().reindex(E)
    fig,ax=canvas()
    for i,(name,label) in enumerate([('TRANSITION_ADJACENT','Adjacent'),('STABLE_REGION','Stable on changing trajectory'),('NO_OBSERVED_TRANSITION','No observed transition')]):ax.bar(xs+(i-1)*.24,bg[name],width=.22,label=label,color=COLORS[i])
    ax.set(xticks=xs,xticklabels=E,ylim=(0,1.25),ylabel='Mean X-partition TVD',title='Observed topology boundaries do not concentrate X error');ax.legend(loc='upper left',fontsize=7)
    fig.text(.16,.035,'Adjacent: within one grid step of an observed transition midpoint; gaps are not bridged.',fontsize=7)
    save(fig,'Fig_S5D_07_X_error_near_topology_transition','X_ERROR_BY_TOPOLOGY_TRANSITION_DISTANCE.csv')
    m=read('07_model_bias_diagnostic/X_DIAGNOSTIC_MODEL_COMPARISON.csv')
    fig,ax=canvas()
    for i,method in enumerate(['Tree','Ridge','MLP','Nearest-element analog']):
        z=m[m.method==method].set_index('heldout').loc[E];ax.bar(xs+(i-1.5)*.19,z.mean_tvd,width=.18,label=method,color=COLORS[i])
    ax.set(xticks=xs,xticklabels=E,ylim=(0,1.2),ylabel='Mean X-partition TVD',title='Fixed post-hoc model comparison across six heldout elements');ax.legend(loc='upper left',ncol=2,fontsize=7)
    fig.text(.16,.035,'Tree: frozen primary. Ridge / MLP / analog: post-hoc diagnostics; one fixed seed per fold.',fontsize=7)
    save(fig,'Fig_S5D_08_tree_vs_ridge_vs_MLP_vs_analog','X_DIAGNOSTIC_MODEL_COMPARISON.csv')
    det=read('08_distance_uncertainty/OOD_ERROR_DETECTION_METRICS.csv')
    score=['nn1_distance','nn5_mean_distance','outside_train_range_feature_count','nearest_distance','affine_residual','convex_reconstruction_error','F3_outside_fraction']
    dm=det.pivot(index='score',columns='scope',values='AUROC').loc[score,['POOLED']+E]
    fig,ax=canvas(4.2);fig.subplots_adjust(left=.31,right=.88,top=.86,bottom=.16)
    im=ax.imshow(dm,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set(xticks=np.arange(7),xticklabels=['Pooled']+E,yticks=np.arange(7),yticklabels=['Archived nominal 1-NN','5-NN mean','Composition range count','Nearest element','Affine residual','Convex error','F3 outside fraction'],title='Distance-based high-error detection (AUROC)')
    for i in range(7):
        for j in range(7):ax.text(j,i,f'{dm.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if dm.iloc[i,j]>.65 else '#222222')
    fig.text(.13,.04,'Chemical distances cannot rank states within a fold; 0.50 denotes no discrimination.',fontsize=7)
    save(fig,'Fig_S5D_09_error_detection_by_distance','OOD_ERROR_DETECTION_METRICS.csv')
    ev=read('10_qa/STEP5D_ROOT_CAUSE_EVIDENCE_MATRIX.csv')
    fig,ax=canvas(4.5);fig.subplots_adjust(left=.46,right=.96,top=.87,bottom=.16)
    labels=['Five chemical anchors','Element-identity variance','General tree limitation (4/6 rule)','Descriptor insufficiency','Topology-selection error','Transition-boundary cause','Composition distance','Phase splitting','Fold-specific model effects']
    states=['SUPPORTED','INCONCLUSIVE','NOT SUPPORTED'];codes=np.array([[1 if s==row.supports else 0 for s in states] for row in ev.itertuples()])
    ax.imshow(codes,cmap=ListedColormap(['#F0F0F0','#647F93']),vmin=0,vmax=1,aspect='auto')
    ax.set(xticks=np.arange(3),xticklabels=['Supported','Inconclusive','Not supported'],yticks=np.arange(len(labels)),yticklabels=labels,title='Evidence strength without overclaiming causality')
    for i,row in enumerate(codes):
        for j,val in enumerate(row):
            if val:ax.text(j,i,'Yes',color='white',ha='center',va='center')
    fig.text(.12,.05,'Evidence classes describe this fixed diagnosis; supported association is not causal identification.',fontsize=7)
    save(fig,'Fig_S5D_10_root_cause_summary','STEP5D_ROOT_CAUSE_EVIDENCE_MATRIX.csv')
    csv('10_qa/FIGURE_AUTOMATED_AUDITS.csv',records)
    assert all(r['exit_code']==0 for r in records),'Inspect PDF audit findings before claiming figure QA'
    print('TEN FIGURES EXPORTED AND AUTOMATED AUDITS PASS',flush=True)
if __name__=='__main__':run()
