"""Two source-backed supplemental views using the existing Python figure workflow."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parents[1];out=O/'review/figures';out.mkdir(exist_ok=True)
sys.path.insert(0,'C:/Users/admin/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def save(fig,n,multi=False):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=str(O/f'audit/Fig_S{n}_alignment.json'),overlay_svg=str(O/f'audit/Fig_S{n}_alignment.svg'),strict=True)
 for ext in ['pdf','svg','png','tif']:
  kw={'dpi':1200 if ext=='tif' else 300}
  if ext=='tif':kw['pil_kwargs']={'compression':'tiff_lzw'}
  fig.savefig(out/f'Fig_S{n}.{ext}',**kw)
 plt.close(fig)
q=pd.read_csv(O/'review/data/Supplementary_Data_S1.csv');q=q.groupby('sampling_domain').failed_states.sum().reindex(['A','B','ANCHOR'])
fig,ax=plt.subplots(figsize=(150/25.4,85/25.4));fig.subplots_adjust(left=.16,right=.97,bottom=.21,top=.92)
ax.bar(range(3),q,color=['#426d89','#81a5b9','#b7ccd7'],width=.55);ax.set_xticks(range(3),['Development','Extended','Perturbed']);ax.set_ylabel('Failed temperature states');ax.set_ylim(0,max(q)*1.18)
for i,v in enumerate(q):ax.text(i,v+max(q)*.025,str(v),ha='center')
save(fig,1)
q=pd.read_csv(O/'review/data/Supplementary_Data_S7.csv');q=q[q.source_table.eq('LOEO_TOPOLOGY_METRICS')]
fig,axes=plt.subplots(2,3,figsize=(150/25.4,145/25.4));fig.subplots_adjust(left=.16,right=.96,bottom=.16,top=.93,wspace=.48,hspace=.50)
names=['FCC','BCC','C15','Other solid','Liquid']
for idx,(el,ax) in enumerate(zip(['ZR','TI','CO','SI','AG','MO'],axes.flat)):
 r=q[q.heldout_element.eq(el)].iloc[0];a=np.array(json.loads(r.confusion_matrix),dtype=float)
 a=np.divide(a,a.sum(axis=1,keepdims=True),out=np.full_like(a,np.nan),where=a.sum(axis=1,keepdims=True)>0)
 ax.imshow(np.ma.masked_invalid(a),vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_title(f'({chr(97+idx)}) {el.title()}',loc='left',fontsize=10)
 ax.set_xticks(range(5),names,rotation=90);ax.set_yticks(range(5),names if idx%3==0 else ['']*5)
 for i in range(5):
  for j in range(5):
   if np.isnan(a[i,j]):t='—'
   elif a[i,j]>=.10:t=f'{a[i,j]:.2f}'
   else:continue
   ax.text(j,i,t,ha='center',va='center',fontsize=8,color='white' if a[i,j]>.55 else '#263b47')
fig.text(.56,.025,'Predicted family',ha='center',fontsize=9);fig.text(.025,.52,'Reference family',rotation=90,va='center',fontsize=9)
save(fig,2,True)
print('Rendered two figures with identical final-size fonts >=8 pt')
