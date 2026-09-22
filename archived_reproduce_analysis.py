"""Read-only frozen benchmark analysis. Python + numpy + pandas.
Usage: python reproduce_analysis.py --sources sources.json --out analysis_tables
Sources JSON maps original file basenames to local paths. No source is modified.
"""
import argparse, hashlib, json, math, platform
from pathlib import Path
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser(); ap.add_argument('--sources',required=True); ap.add_argument('--out',required=True); ap.add_argument('--bootstrap',type=int,default=5000)
args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
sources=json.loads(Path(args.sources).read_text(encoding='utf-8'))
expected={
'zero_cost_baseline_results.csv':'c663d5d1b3a2f4d56068e1dd68806ee22ebaf6e606fb287ebe1b892036c9e046',
'zero_cost_baseline_protocol.json':'6e7281063ab948bf69c4154f908341c385462191046ba1a18ab271655cae539a',
'zero_cost_baseline_summary.txt':'39164efb19bceca194f475bbed7ceaff2e12627985afee12a096756566035913',
'jev_full_results.csv':'b940a9a8fa9704bf1ea57ac80f1eb02d67d2d9f1c6cdbfc199a650162912ed68',
'jev_full_protocol.json':'4565c8f92835e15285fdc87ee7132a956012cbfb5b5aece3ea32edceba5b909d',
'jev_full_run_summary.txt':'778daa5ed6df7ea04f77df836ef394be89a10bc30089b11c683bb853315a463c'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes=[]
for name,path in sources.items():
    h=sha(path); e=expected.get(name)
    hashes.append(dict(file=name,sha256=h,expected_sha256=e,status='MATCH' if h==e else ('NO_PRIOR_HASH_AVAILABLE' if e is None else 'FAIL')))
    if e: assert h==e,(name,h)
pd.DataFrame(hashes).to_csv(out/'sha256_manifest.csv',index=False)
def read(name): return pd.read_csv(sources[name],dtype={'dataset_id':str,'record_id':str})
m=read('jev_benchmark_master_usable.csv'); j=read('jev_full_results.csv'); b=read('zero_cost_baseline_results.csv')
c=read('review_context.csv'); q=read('data_quality_report.csv'); key=['dataset_id','record_id']
qc=[]
def check(name,ok,detail=''):
    qc.append(dict(check=name,passed=bool(ok),detail=str(detail)))
    assert ok,(name,detail)
for name,d in [('master',m),('jev',j),('baseline',b)]:
    check(name+'_rows',len(d)==17191,len(d)); check(name+'_unique_keys',not d.duplicated(key).any())
    check(name+'_keys_present',not d[key].isna().any().any() and not d[key].eq('').any().any())
    check(name+'_labels_binary',d.ground_truth.isin([0,1]).all())
    check(name+'_review_count',d.dataset_id.nunique()==10)
    check(name+'_key_set',set(map(tuple,d[key].values))==set(map(tuple,m[key].values)))
for name,d in [('context',c),('quality',q)]:
    check(name+'_review_unique',not d.dataset_id.duplicated().any())
    check(name+'_review_set',set(d.dataset_id)==set(m.dataset_id))
check('context_required_text',c[['review_title','eligibility_criteria_verbatim']].fillna('').apply(lambda x:x.str.strip().ne('')).all().all())
d=m.merge(j,on=key,validate='one_to_one',suffixes=('','_jev')).merge(b,on=key,validate='one_to_one',suffixes=('','_baseline'))
check('labels_identical',d.ground_truth.eq(d.ground_truth_jev).all() and d.ground_truth.eq(d.ground_truth_baseline).all())
check('all_jev_success',d.api_status.eq('success').all()); check('jev_model',d.model.eq('jev-1.13.0').all())
check('jev_probability_sum',np.allclose(d.p_include+d.p_exclude,1))
check('jev_prompt_hash',d.prompt_sha256.eq('4f249815241aa09ca3ba01de4d56e0e7ad0754397f15462baee68a3c9a3c9091').all())
raw={'Jev':'p_include','BM25':'bm25_score','TF-IDF':'tfidf_score','MiniLM':'minilm_score','LR':'lr_loro_p_include'}
primary={'Jev':'p_include','BM25':'bm25_rank_pct','TF-IDF':'tfidf_rank_pct','MiniLM':'minilm_rank_pct','LR':'lr_loro_p_include'}
for col in set(raw.values())|set(primary.values()): check(col+'_finite',np.isfinite(d[col]).all())
for col in ['p_include','lr_loro_p_include']: check(col+'_range',d[col].between(0,1).all())
for col in ['jev_decision','lr_loro_decision']: check(col+'_valid',d[col].isin(['include','exclude']).all())
check('lr_decision_matches_frozen_threshold',np.array_equal(d.lr_loro_decision.eq('include'),d.lr_loro_p_include.ge(.5)))
for name,col in raw.items():
    dest=name+'_derived_rank_pct'; d[dest]=d.groupby('dataset_id')[col].transform(lambda s:(s.rank(method='average')-1)/(len(s)-1))
    if name in ['BM25','TF-IDF','MiniLM']:
        check(name+'_rank_pct_reproduced',np.allclose(d[dest],d[primary[name]],rtol=0,atol=1e-12))
for rid,g in d.groupby('dataset_id'):
    qr=q.set_index('dataset_id').loc[rid]
    check(rid+'_quality_n_usable',len(g)==qr.N_usable)
    check(rid+'_quality_positive_n',int(g.ground_truth.sum())==qr.N_include)
    check(rid+'_lr_training_n',g.lr_training_n.eq(len(d)-len(g)).all())
    check(rid+'_lr_training_positive_n',g.lr_training_positive_n.eq(d.ground_truth.sum()-g.ground_truth.sum()).all())
pd.DataFrame(qc).to_csv(out/'integrity_checks.csv',index=False)

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
summary=dict(n=len(d),positive_n=int(y.sum()),prevalence=y.mean(),brier=np.mean((p-y)**2),brier_skill=1-np.mean((p-y)**2)/(y.mean()*(1-y.mean())),ECE10=float(np.nansum(cal.n/len(d)*abs(cal.mean_probability-cal.observed_fraction))),mean_probability=p.mean(),zero_probability_n=int((p==0).sum()),zero_probability_positive_n=int(((p==0)&(y==1)).sum()),high_probability_n=int((p>=.9).sum()),high_probability_mean=float(p[p>=.9].mean()),high_probability_observed=float(y[p>=.9].mean()),jev_decision_differs_from_p_ge_05=int((d.jev_decision.eq('include')!=d.p_include.ge(.5)).sum()),missing_title_n=int(d.title.fillna('').str.strip().eq('').sum()),missing_abstract_n=int(d.abstract.fillna('').str.strip().eq('').sum()),cross_review_record_id_repeated_rows=int(d.record_id.duplicated(keep=False).sum()))
(out/'jev_calibration_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
# Fixed-score data-quality sensitivity; rank-percentile columns remain frozen.
complete=d.title.fillna('').str.strip().ne('')&d.abstract.fillna('').str.strip().ne(''); no_gap=d.abstract_index_gap_n.fillna(0).eq(0)
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
for row in hashes: check(row['file']+'_unchanged_after_analysis',sha(sources[row['file']])==row['sha256'])
pd.DataFrame(qc).to_csv(out/'integrity_checks.csv',index=False)
(out/'analysis_settings.json').write_text(json.dumps(dict(seed=20260921,bootstrap_n=B,bootstrap='paired within-review ordinary row bootstrap, fixed predictions',AUPRC='non-interpolated average precision',screening='within review, full boundary tie; secondary exact random-tie expected stopping position',WSS95='0.95 - screening_fraction (target-recall convention)',python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,all_checks=len(qc)),indent=2),encoding='utf-8')
print('COMPLETE',summary,flush=True)
