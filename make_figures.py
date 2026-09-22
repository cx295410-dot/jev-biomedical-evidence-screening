import sys,pathlib,json,csv,shutil,zipfile,hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
B=pathlib.Path(__file__).resolve().parent
O=B;F=B/'figures';F.mkdir(exist_ok=True)
def read(n):return list(csv.DictReader((B/('reference_'+n)).open(encoding='utf-8-sig')))
r=read('per_review_metrics.csv');s=read('screening_per_review.csv');summary={x['model']:x for x in read('screening_summary.csv')};c=read('jev_calibration_bins.csv');cs=json.loads((B/'reference_jev_calibration_summary.json').read_text())
ids=[x['dataset_id'] for x in r if x['model']=='Jev'];models=['Jev','BM25','TF-IDF','MiniLM','LR'];colors=['#0072B2','#777777','#CC79A7','#D55E00','#009E73'];markers=['o','s','^','D','P']
plt.rcParams.update({'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,n):
 for ext in ['png','svg','pdf','eps']:fig.savefig(F/f'Figure_{n}.{ext}',dpi=600,bbox_inches='tight',facecolor='white')
 plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(9,5.8),sharey=True)
for a,key,title in zip(ax,['AUROC','AUPRC_AP'],['a','b']):
 for j,m in enumerate(models):
  vals={x['dataset_id']:float(x[key]) for x in r if x['model']==m}
  a.scatter([vals[i] for i in ids],np.arange(10)+(j-2)*.13,s=31,c=colors[j],marker=markers[j],label='Feature-based LORO LR' if m=='LR' else m,zorder=3)
 a.set_title(title,loc='left',fontweight='bold');a.set_xlim(0,1.025);a.set_xticks(np.arange(0,1.01,.2));a.grid(axis='x',color='#e5e5e5');a.set_xlabel('AUROC' if key=='AUROC' else 'Average precision (AP)')
ax[0].set_yticks(range(10),[i.replace('_',' ') for i in ids]);ax[0].invert_yaxis()
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.56,0));fig.subplots_adjust(left=.23,right=.98,top=.93,bottom=.18,wspace=.12);save(fig,1)
fig,ax=plt.subplots(1,2,figsize=(9,5.8),gridspec_kw={'width_ratios':[1.25,1]})
j={x['dataset_id']:x for x in s if x['model']=='Jev'};mi={x['dataset_id']:x for x in s if x['model']=='MiniLM'}
for y,i in enumerate(ids):
 v=[100*float(j[i]['screening_fraction']),100*float(mi[i]['screening_fraction'])];ax[0].plot(v,[y,y],color='#aaaaaa',lw=1.4,zorder=1)
 for z,col in zip(v,[colors[0],colors[3]]):ax[0].scatter(z,y,color=col,s=35,zorder=3)
ax[0].set_yticks(range(10),[i.replace('_',' ')+(' *' if float(j[i]['boundary_score'])==0 else '') for i in ids]);ax[0].invert_yaxis();ax[0].set_xlim(-2,104);ax[0].set_xticks([0,25,50,75,100]);ax[0].set_xlabel('Records screened (%)');ax[0].set_title('a',loc='left',fontweight='bold');ax[0].grid(axis='x',color='#e5e5e5')
for m,col in [('Jev',colors[0]),('MiniLM',colors[3])]:ax[0].scatter([],[],color=col,label=m)
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,.10),ncol=2,frameon=False)
cats=['Record-weighted\nComplete ties','Record-weighted\nRandom-tie expected','Macro\nComplete ties','Macro\nRandom-tie expected'];keys=['WSS95','random_tie_expected_WSS95','macro_WSS95','macro_random_tie_expected_WSS95']
for k,(m,col) in enumerate([('Jev',colors[0]),('MiniLM',colors[3])]):
 vals=[100*float(summary[m][key]) for key in keys];y=np.arange(4)+(k-.5)*.28
 ax[1].barh(y,vals,height=.26,color=col)
 for yy,v in zip(y,vals):ax[1].text(v+.8,yy,f'{v:.1f}',va='center',fontsize=8)
ax[1].set_yticks(range(4),cats,fontsize=8);ax[1].invert_yaxis();ax[1].set_xlim(0,69);ax[1].set_xlabel('WSS@95 (%)');ax[1].set_title('b',loc='left',fontweight='bold');ax[1].grid(axis='x',color='#e5e5e5');ax[1].set_axisbelow(True)
fig.subplots_adjust(left=.21,right=.98,top=.92,bottom=.24,wspace=.68);fig.text(.21,.035,'* Jev boundary score = 0; all records screened under complete ties.\nRetrospective target: at least 95% recall in each review.',fontsize=8);save(fig,2)
fig,ax=plt.subplots(1,2,figsize=(9,5.1),gridspec_kw={'width_ratios':[1.2,1]})
x=[float(t['mean_probability']) for t in c];y=[float(t['observed_fraction']) for t in c];counts=[int(t['n']) for t in c]
ax[0].plot([0,1],[0,1],ls='--',color='#888888',lw=1,label='Perfect calibration');ax[0].plot(x,y,'o-',color=colors[0],lw=1.3,ms=5,label='Frozen bins')
for xx,yy,n in zip(x,y,counts):ax[0].annotate(f'n={n:,}',(xx,yy),xytext=(3,7),textcoords='offset points',fontsize=7)
ax[0].set(xlim=(-.025,1.02),ylim=(-.02,1.02),xlabel='Mean predicted probability',ylabel='Observed final-inclusion fraction');ax[0].set_title('a',loc='left',fontweight='bold');ax[0].legend(loc='upper left',frameon=False,fontsize=8);ax[0].grid(color='#eeeeee');ax[0].set_axisbelow(True)
ax[1].bar(np.arange(10),counts,color=colors[0],width=.75)
for k,n in enumerate(counts):ax[1].text(k,n*1.16,f'{n:,}',ha='center',fontsize=7)
ax[1].set_yscale('log');ax[1].set_ylim(10,60000);ax[1].set_xticks(range(10),[f'{i/10:.1f}' for i in range(10)],rotation=0,fontsize=8);ax[1].set_xlabel('Probability-bin lower bound');ax[1].set_ylabel('Records (log scale)');ax[1].set_title('b',loc='left',fontweight='bold')
fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.34,wspace=.3)
fig.text(.09,.025,f"Exact zero: {cs['zero_probability_n']:,} records, {cs['zero_probability_positive_n']} final inclusions ({cs['zero_probability_positive_n']/cs['zero_probability_n']:.2%}).\nProbability ≥ 0.90: n={cs['high_probability_n']}; mean prediction {cs['high_probability_mean']:.2%}; observed {cs['high_probability_observed']:.2%}.\nBrier score {cs['brier']:.5f}; ECE (10 bins) {cs['ECE10']:.5f}.",fontsize=9,linespacing=1.5);save(fig,3)
assert sum(counts)==17191 and len(ids)==10 and len(r)==50
