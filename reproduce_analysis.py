"""Portable public-data adapter of the archived frozen analysis.
Mathematical analysis is retained; private-input checks are replaced by a public-file
hash check and key/label validation. Text missingness uses published Boolean flags.
Run: python reproduce_analysis.py --out reproduced --bootstrap 5000
"""
import argparse, hashlib, json, math, platform
from pathlib import Path
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--out',default='reproduced');ap.add_argument('--bootstrap',type=int,default=5000)
args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PUBLIC_MANIFEST.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=root/'frozen_records.csv'
assert sha(source)==manifest['files']['frozen_records.csv']
d=pd.read_csv(source,dtype={'dataset_id':str,'record_id':str});key=['dataset_id','record_id'];qc=[]
def check(name,ok,detail=''):
 qc.append(dict(check=name,passed=bool(ok),detail=str(detail)));assert ok,(name,detail)
check('public_rows',len(d)==17191);check('public_positives',d.ground_truth.sum()==392)
check('public_reviews',d.dataset_id.nunique()==10);check('unique_keys',not d.duplicated(key).any())
check('labels_binary',d.ground_truth.isin([0,1]).all())
check('model',d.model.eq('jev-1.13.0').all());check('probability_sum',np.allclose(d.p_include+d.p_exclude,1))
raw={'Jev':'p_include','BM25':'bm25_score','TF-IDF':'tfidf_score','MiniLM':'minilm_score','LR':'lr_loro_p_include'}
primary={'Jev':'p_include','BM25':'bm25_rank_pct','TF-IDF':'tfidf_rank_pct','MiniLM':'minilm_rank_pct','LR':'lr_loro_p_include'}
for col in set(raw.values())|set(primary.values()):check(col+'_finite',np.isfinite(d[col]).all())
for name,col in raw.items():
 dest=name+'_derived_rank_pct';d[dest]=d.groupby('dataset_id')[col].transform(lambda s:(s.rank(method='average')-1)/(len(s)-1))
 if name in ['BM25','TF-IDF','MiniLM']:check(name+'_rank_check',np.allclose(d[dest],d[primary[name]],rtol=0,atol=1e-12))

# Weighted score groups: same-score ties are treated jointly, never broken by labels.
def prep(y,s):
    _,inv=np.unique(-np.asarray(s,float),return_inverse=True)
    return np.asarray(y,float),inv,int(inv.max()+1)
def metrics(p,w=None):
    y,inv,k=p
    if w is None: w=np.ones(len(y))
    pos=np.bincount(inv,weights=w*y,minlength=k); neg=np.bincount(inv,weights=w*(1-y),minlength=k)
    tp=np.cumsum(pos); fp=np.cumsum(neg); P=tp[-1]; N=fp[-1]
    if P==0 or N==0: return np.array([np.nan,np.nan])
    auc=np.sum(pos*(N-fp+0.5*neg))/(P*N)
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
    average_precision=np.sum(pos*precision)/P
    return np.array([auc,average_precision])
def classification(y,pred):
    y=np.asarray(y); pred=np.asarray(pred); tp=int(((y==1)&pred).sum()); fn=int(((y==1)&~pred).sum()); tn=int(((y==0)&~pred).sum()); fp=int(((y==0)&pred).sum())
    denom=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return dict(TP=tp,FN=fn,TN=tn,FP=fp,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/(tp+fp) if tp+fp else 0,F1=2*tp/(2*tp+fp+fn),MCC=(tp*tn-fp*fn)/denom if denom else 0,accuracy=(tp+tn)/len(y))
def screen(y,s):
    y,inv,k=prep(y,s); pos=np.bincount(inv,weights=y,minlength=k); n=np.bincount(inv,minlength=k)
    target=math.ceil(.95*y.sum()); z=int(np.searchsorted(np.cumsum(pos),target)); before=n[:z].sum(); pbefore=pos[:z].sum(); need=target-pbefore
    # Expected position of the need-th positive in random ordering of the boundary tie.
    expected_screened=before+need*(n[z]+1)/(pos[z]+1)
    complete=before+n[z]
    return dict(n=len(y),positive_n=int(y.sum()),target_positive_n=target,boundary_score=float(np.unique(np.asarray(s))[::-1][z]),boundary_tie_n=int(n[z]),boundary_tie_positive_n=int(pos[z]),screened_n=complete,screening_fraction=complete/len(y),work_saved=1-complete/len(y),WSS95=.95-complete/len(y),achieved_recall=(pbefore+pos[z])/y.sum(),random_tie_expected_screened_n=expected_screened,random_tie_expected_fraction=expected_screened/len(y),random_tie_expected_WSS95=.95-expected_screened/len(y),random_tie_best_screened_n=before+need,random_tie_worst_screened_n=before+n[z]-pos[z]+need)
y=d.ground_truth.to_numpy(); names=list(raw); models=[prep(y,d[primary[n]]) for n in names]
point=np.array([metrics(p) for p in models]); overall=[]; per=[]; screening=[]; binary=[]
for i,name in enumerate(names):
    overall.append(dict(model=name,score=primary[name],AUROC=point[i,0],AUPRC_AP=point[i,1]))
    if name in ['Jev','LR']:
        col='jev_decision' if name=='Jev' else 'lr_loro_decision'; binary.append(dict(model=name,**classification(y,d[col].eq('include').values)))
    for rid,g in d.groupby('dataset_id'):
        a=metrics(prep(g.ground_truth,g[raw[name]])); row=dict(dataset_id=rid,model=name,n=len(g),positive_n=int(g.ground_truth.sum()),AUROC=a[0],AUPRC_AP=a[1])
        if name in ['Jev','LR']: row.update(classification(g.ground_truth,g[col].eq('include').values))
        per.append(row); screening.append(dict(dataset_id=rid,model=name,**screen(g.ground_truth,g[raw[name]])))
pd.DataFrame(binary).to_csv(out/'binary_metrics.csv',index=False)
per=pd.DataFrame(per); per.to_csv(out/'per_review_metrics.csv',index=False)
macro=per.groupby('model')[['AUROC','AUPRC_AP','sensitivity']].agg(['mean','min','max']); macro.columns=['_'.join(a) for a in macro.columns]; macro.to_csv(out/'macro_and_ranges.csv')
sc=pd.DataFrame(screening); sc.to_csv(out/'screening_per_review.csv',index=False)
sa=[]
for name,g in sc.groupby('model'):
    f=g.screened_n.sum()/g.n.sum(); ef=g.random_tie_expected_screened_n.sum()/g.n.sum()
    sa.append(dict(model=name,screened_n=g.screened_n.sum(),screening_fraction=f,work_saved=1-f,WSS95=.95-f,macro_screening_fraction=g.screening_fraction.mean(),macro_WSS95=g.WSS95.mean(),min_WSS95=g.WSS95.min(),max_WSS95=g.WSS95.max(),random_tie_expected_screening_fraction=ef,random_tie_expected_WSS95=.95-ef,macro_random_tie_expected_WSS95=g.random_tie_expected_WSS95.mean(),achieved_recall=np.sum(g.achieved_recall*g.positive_n)/g.positive_n.sum()))
pd.DataFrame(sa).to_csv(out/'screening_summary.csv',index=False)

# Independent verification: rank-sum AUROC and direct threshold AP for every primary model.
for i,name in enumerate(names):
    s=d[primary[name]]; P=y.sum(); N=len(y)-P; ranks=s.rank(method='average').to_numpy()
    independent_auc=(ranks[y==1].sum()-P*(P+1)/2)/(P*N)
    check(name+'_independent_auc',abs(independent_auc-point[i,0])<1e-12)
    positive_scores=np.unique(s[y==1]); ap_value=sum(((s[y==1]==v).sum()/P)*y[s>=v].mean() for v in positive_scores)
    check(name+'_independent_ap',abs(ap_value-point[i,1])<1e-12)
print('INTEGRITY AND POINT ESTIMATES COMPLETE',flush=True)
print(pd.DataFrame(overall).to_string(index=False),flush=True)

# Paired ordinary bootstrap within each review. Frozen scores held fixed; models not refit.
rng=np.random.default_rng(20260921); groups=[g.index.to_numpy() for _,g in d.groupby('dataset_id')]; B=args.bootstrap
boots=np.empty((B,len(names),2)); stdmodels=[prep(y,d[n+'_derived_rank_pct']) for n in names]; stdpoint=np.array([metrics(p) for p in stdmodels]); stdboots=np.empty_like(boots)
for z in range(B):
    sampled=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in groups]); w=np.bincount(sampled,minlength=len(d))
    boots[z]=[metrics(p,w) for p in models]
    stdboots[z]=[metrics(p,w) for p in stdmodels]
    if (z+1)%1000==0: print('Bootstrap',z+1,'/',B,flush=True)
np.savez_compressed(out/'bootstrap_draws.npz',primary=boots,all_rank_pct=stdboots,model_names=names)
diffs=[]
for scale,pt,bt in [('primary',point,boots),('all_models_rank_pct',stdpoint,stdboots)]:
    for i,name in enumerate(names):
        for h,metric in enumerate(['AUROC','AUPRC_AP']):
            lo,hi=np.quantile(bt[:,i,h],[.025,.975])
            if scale=='primary': overall[i][metric+'_ci_low']=lo; overall[i][metric+'_ci_high']=hi
            if i:
                dd=bt[:,0,h]-bt[:,i,h]; l,u=np.quantile(dd,[.025,.975]); sl,su=np.quantile(dd,[.003125,.996875])
                diffs.append(dict(score_scale=scale,comparison='Jev - '+name,metric=metric,difference=pt[0,h]-pt[i,h],ci_low=l,ci_high=u,bonferroni_8_comparisons_ci_low=sl,bonferroni_8_comparisons_ci_high=su,bootstrap_n=B))
pd.DataFrame(overall).to_csv(out/'overall_metrics.csv',index=False); pd.DataFrame(diffs).to_csv(out/'paired_differences.csv',index=False)
pd.DataFrame([dict(model=n,AUROC=stdpoint[i,0],AUPRC_AP=stdpoint[i,1]) for i,n in enumerate(names)]).to_csv(out/'all_models_rank_pct_metrics.csv',index=False)
# Sensitivity to review mix: paired resampling of 10 reviews for macro differences.
cluster=[]; rng2=np.random.default_rng(20260922); pick=rng2.integers(0,10,size=(10000,10))
for metric in ['AUROC','AUPRC_AP']:
    tab=per.pivot(index='dataset_id',columns='model',values=metric)
    for name in names[1:]:
        v=(tab.Jev-tab[name]).to_numpy(); draws=v[pick].mean(axis=1); lo,hi=np.quantile(draws,[.025,.975])
        cluster.append(dict(comparison='Jev - '+name,metric=metric,macro_difference=v.mean(),ci_low=lo,ci_high=hi,reviews_jev_better=int((v>0).sum()),reviews_jev_worse=int((v<0).sum())))
pd.DataFrame(cluster).to_csv(out/'review_resampling_macro_differences.csv',index=False)
# Calibration only for Jev. ECE equal-width bins [0,.1), ... [.9,1].
p=d.p_include.to_numpy(); bins=np.minimum((p*10).astype(int),9); calibration=[]
for h in range(10):
    mask=bins==h
    calibration.append(dict(bin=h,lower=h/10,upper=(h+1)/10,n=int(mask.sum()),mean_probability=float(p[mask].mean()) if mask.any() else np.nan,observed_fraction=float(y[mask].mean()) if mask.any() else np.nan))
cal=pd.DataFrame(calibration); cal.to_csv(out/'jev_calibration_bins.csv',index=False)
summary=dict(n=len(d),positive_n=int(y.sum()),prevalence=y.mean(),brier=np.mean((p-y)**2),brier_skill=1-np.mean((p-y)**2)/(y.mean()*(1-y.mean())),ECE10=float(np.nansum(cal.n/len(d)*abs(cal.mean_probability-cal.observed_fraction))),mean_probability=p.mean(),zero_probability_n=int((p==0).sum()),zero_probability_positive_n=int(((p==0)&(y==1)).sum()),high_probability_n=int((p>=.9).sum()),high_probability_mean=float(p[p>=.9].mean()),high_probability_observed=float(y[p>=.9].mean()),jev_decision_differs_from_p_ge_05=int((d.jev_decision.eq('include')!=d.p_include.ge(.5)).sum()),missing_title_n=int((~d.title_present).sum()),missing_abstract_n=int((~d.abstract_present).sum()),cross_review_record_id_repeated_rows=int(d.record_id.duplicated(keep=False).sum()))
(out/'jev_calibration_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
# Fixed-score data-quality sensitivity; rank-percentile columns remain frozen.
complete=d.title_present&d.abstract_present; no_gap=d.abstract_index_gap_n.fillna(0).eq(0)
dedup=~d.duplicated(['dataset_id','duplicate_group']) | d.duplicate_group.isna() | d.duplicate_group.eq('')
conflict=d.groupby(['dataset_id','duplicate_group']).ground_truth.transform('nunique').gt(1)
dedup=dedup & ~conflict
overlap=d.record_id.duplicated(keep=False)
d.loc[overlap,key+['ground_truth']].to_csv(out/'cross_review_overlap.csv',index=False)
sens=[]
for subset,mask in [('primary',np.ones(len(d),bool)),('duplicate_adjusted',dedup),('complete_title_abstract',complete),('no_abstract_index_gap',no_gap),('strict_clean',dedup&complete&no_gap),('no_cross_review_overlap',~overlap)]:
    g=d.loc[mask]
    for name in names:
        a=metrics(prep(g.ground_truth,g[primary[name]])); sens.append(dict(subset=subset,model=name,n=len(g),positive_n=int(g.ground_truth.sum()),AUROC=a[0],AUPRC_AP=a[1]))
pd.DataFrame(sens).to_csv(out/'data_quality_sensitivity.csv',index=False)
# Review-mix uncertainty for high-recall endpoint, conditional on observed per-review curves.
screen_ci=[]
for tie,col in [('complete_tie','screened_n'),('random_tie_expectation','random_tie_expected_screened_n')]:
    counts=sc.pivot(index='dataset_id',columns='model',values=col)
    nn=sc.drop_duplicates('dataset_id').set_index('dataset_id').loc[counts.index,'n'].to_numpy()
    for name in names[1:]:
        delta=(counts[name]-counts.Jev).to_numpy()
        for agg in ['record_weighted','macro']:
            vals=delta[pick].sum(axis=1)/nn[pick].sum(axis=1) if agg=='record_weighted' else (delta/nn)[pick].mean(axis=1)
            pt=delta.sum()/nn.sum() if agg=='record_weighted' else (delta/nn).mean()
            lo,hi=np.quantile(vals,[.025,.975]); screen_ci.append(dict(tie_policy=tie,aggregation=agg,comparison='Jev - '+name,metric='WSS95_difference',difference=pt,ci_low=lo,ci_high=hi))
pd.DataFrame(screen_ci).to_csv(out/'screening_review_resampling.csv',index=False)
check('public_source_unchanged',sha(source)==manifest['files']['frozen_records.csv'])
pd.DataFrame(qc).to_csv(out/'integrity_checks.csv',index=False)
(out/'analysis_settings.json').write_text(json.dumps(dict(seed=20260921,bootstrap_n=B,bootstrap='paired within-review ordinary row bootstrap, fixed predictions',AUPRC='non-interpolated average precision',screening='within review, full boundary tie; secondary exact random-tie expected stopping position',WSS95='0.95 - screening_fraction (target-recall convention)',python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,all_checks=len(qc)),indent=2),encoding='utf-8')
print('COMPLETE',summary,flush=True)
