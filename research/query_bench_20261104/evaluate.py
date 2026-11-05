"""Complete-seed-only paired reports; no outcome-dependent benchmark selection."""
from common import *
from worker import aggregate
import argparse,pandas as pd
from scipy.special import ndtr
from scipy.stats import wilcoxon
from scipy.optimize import minimize_scalar
from decimal import Decimal,ROUND_CEILING

SEEDS=[271027,272027,273027]

def conformal(cy,cc,cs,tc,ts,alpha=.1):
 n=len(cy);k=int((Decimal(n+1)*(1-Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING));assert k<=n
 threshold=np.sort(np.abs(cy-cc)/np.maximum(cs,1e-6),axis=0)[k-1];half=np.maximum(ts,1e-6)*threshold
 return tc-half,tc+half

def crps_samples(draw,y):
 m=draw.shape[1];v=np.sort(draw,axis=1);coef=2*np.arange(1,m+1)-m-1
 return np.abs(draw-y[:,None,:]).mean(1)-(v*coef[None,:,None]).sum(1)/(m*(m-1))

def crps_gaussian(mu,sd,y):
 sd=np.maximum(sd,1e-6);z=(y-mu)/sd
 return sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))

def collect(seed):
 rows=[];costs=[]
 for fold in range(5):
  cache=read_cache(fold);crops=cache['crops'];T=cache['targets'];folder=RUN/f'fold{fold}'
  paths={k:folder/f'{k}_{seed}' for k in ['dscore','dquery','ligature']}
  for k,p in paths.items():
   if not (p/'complete.json').exists():raise RuntimeError(f'Incomplete fixed comparison: {p}')
   meta=json.loads((p/'complete.json').read_text());assert meta['fold']==fold and meta['seed']==seed and meta['kind']==k
   assert int(meta.get('epochs',15))==15
   costs.append({**meta,'fold':fold,'seed':seed,'kind':k})
  packs={k:{r:np.load(p/(r+'.npz')) for r in ['selection','calibration','test']} for k,p in paths.items()}
  cheap={k:{r:np.load(folder/'cheap'/k/(r+'.npz')) for r in ['selection','calibration','test']} for k in ['ridge','affine']}
  reusefolder=folder/f'reuse_{seed}'
  if not (reusefolder/'complete.json').exists() or not (reusefolder/'analytic_scored/test.npz').exists() or not (reusefolder/'marginal_independent/test.npz').exists():raise RuntimeError(f'Incomplete frozen-reuse comparison: {reusefolder}')
  analytic={r:np.load(reusefolder/'analytic_scored'/(r+'.npz')) for r in ['selection','calibration','test']}
  exact={r:np.load(reusefolder/'marginal_independent'/(r+'.npz')) for r in ['selection','calibration','test']}
  for name in ['direct_moments','joint_moments']:
   cheap[name]={r:np.load(reusefolder/name/(r+'.npz')) for r in ['selection','calibration','test']}
  tr=[c for c in crops if c['role']=='train' and c['eligible']];ix=[c['index'] for c in tr]
  trainS=aggregate(tr,T['scored'][ix])[1][:,424:560];trainQ=aggregate(tr,T['queries'][ix])[1]
  L=packs['ligature'];directS=packs['dscore'];directQ=packs['dquery']
  for role in ['selection','calibration','test']:
   for method in [directS,directQ,*cheap.values(),exact]:
    assert np.array_equal(L[role]['case'],method[role]['case']), 'Patient alignment mismatch'
   np.testing.assert_allclose(L[role]['truth'],directQ[role]['truth'],atol=1e-6)
  for family,ids,kind,training in [('clustering',np.diag_indices(16)[0], 'scored',trainS),('cross',None,'scored',trainS),('Moran',np.arange(16),'queries',trainQ),('Geary',np.arange(16,32),'queries',trainQ),('log_assay_clustering',np.arange(32,48),'queries',trainQ)]:
   if kind=='scored':
    ia,ib=np.triu_indices(16);ids=np.flatnonzero(ia==ib) if family=='clustering' else np.flatnonzero(ia!=ib)
    expected={r:np.load(BASE/f'fold{fold}'/(r+'_patients.npz')) for r in ['selection','calibration','test']}
    cy,ty=expected['calibration']['truth'][:,ids],expected['test']['truth'][:,ids];cp,tp=expected['calibration']['plugin'][:,ids],expected['test']['plugin'][:,ids]
    case=expected['test']['case'];d={r:dict(center=directS[r]['center'][:,424:560][:,ids],sd=directS[r]['sd'][:,424:560][:,ids]) for r in ['calibration','test']}
    ld={r:L[r]['scored_draws'][:,:,424:560][:,:,ids] for r in ['calibration','test']}
    # The reused seed keeps its previously published independent-draw realization.
    if seed==SEEDS[0]:ld={r:expected[r]['draws'][:,:,ids] for r in ['calibration','test']}
    sy,sp=expected['selection']['truth'],expected['selection']['plugin'];slope=float((sy*sp).sum()/max((sp*sp).sum(),1e-12));slopefamily=float((sy[:,ids]*sp[:,ids]).sum()/max((sp[:,ids]**2).sum(),1e-12))
   else:
    cy,ty=L['calibration']['truth'][:,ids],L['test']['truth'][:,ids];cp,tp=L['calibration']['plugin'][:,ids],L['test']['plugin'][:,ids];case=L['test']['case']
    d={r:dict(center=directQ[r]['center'][:,ids],sd=directQ[r]['sd'][:,ids]) for r in ['calibration','test']};ld={r:L[r]['draws'][:,:,ids] for r in ['calibration','test']}
   scale=training[:,ids].std(0,ddof=1).clip(1e-4)
   # Conservative direct-law control: one spread multiplier per reported family,
   # optimized only on the two selection patients, never calibration/test labels.
   ds=directS['selection'] if kind=='scored' else directQ['selection']
   sm=ds['center'][:,424:560][:,ids] if kind=='scored' else ds['center'][:,ids]
   ss=ds['sd'][:,424:560][:,ids] if kind=='scored' else ds['sd'][:,ids]
   sy=ds['truth'][:,424:560][:,ids] if kind=='scored' else ds['truth'][:,ids]
   opt=minimize_scalar(lambda loga:float(np.mean(crps_gaussian(sm,ss*np.exp(loga),sy)/scale)),bounds=(np.log(.1),np.log(10)),method='bounded')
   direct_alpha=float(np.exp(opt.x))
   centers={'plugin':(cp,tp),'direct':(d['calibration']['center'],d['test']['center']),'ligature':(ld['calibration'].mean(1),ld['test'].mean(1))}
   scales={'plugin':(np.ones_like(cp),np.ones_like(tp)),'direct':(d['calibration']['sd'],d['test']['sd']),'ligature':(ld['calibration'].std(1).clip(1e-6),ld['test'].std(1).clip(1e-6))}
   if kind=='scored':
    centers['ligature_analytic']=tuple(analytic[r]['center'][:,424:560][:,ids] for r in ['calibration','test']);scales['ligature_analytic']=scales['ligature']
    for name,s in [('global_rescale',slope),('family_rescale',slopefamily)]:centers[name]=(cp*s,tp*s);scales[name]=(np.ones_like(cp),np.ones_like(tp))
   else:
    for name in ['affine','ridge','direct_moments','joint_moments']:
     centers[name]=tuple(cheap[name][r]['center'][:,ids] for r in ['calibration','test']);scales[name]=(np.ones_like(cp),np.ones_like(tp))
    ind={r:L[r]['independent_draws'][:,:,ids] for r in ['calibration','test']};centers['independent_gaussian']=tuple(ind[r].mean(1) for r in ['calibration','test']);scales['independent_gaussian']=tuple(ind[r].std(1).clip(1e-6) for r in ['calibration','test'])
   if kind=='queries':
    ex={r:exact[r]['draws'][:,:,ids] for r in ['calibration','test']};centers['independent_marginals']=tuple(ex[r].mean(1) for r in ['calibration','test']);scales['independent_marginals']=tuple(ex[r].std(1).clip(1e-6) for r in ['calibration','test'])
   for method,(cc,tc) in centers.items():
    cs,ts=scales[method]
    crps=crps_samples(ld['test'],ty) if method in ['ligature','ligature_analytic'] else crps_gaussian(tc,ts,ty) if method=='direct' else crps_samples(ind['test'],ty) if method=='independent_gaussian' else crps_samples(ex['test'],ty) if method=='independent_marginals' else np.abs(tc-ty)
    crps_scaled=crps_gaussian(tc,ts*direct_alpha,ty) if method=='direct' else crps
    modes=['adaptive','constant','raw'] if method in ['direct','ligature','ligature_analytic','independent_gaussian','independent_marginals'] else ['constant']
    for mode in modes:
     if mode=='raw':
      if method=='direct':lo,hi=tc-1.6448536269514722*ts,tc+1.6448536269514722*ts
      else:lo,hi=np.quantile(ld['test'] if method in ['ligature','ligature_analytic'] else ind['test'] if method=='independent_gaussian' else ex['test'],[.05,.95],axis=1)
     else:lo,hi=conformal(cy,cc,cs if mode=='adaptive' else np.ones_like(cc),tc,ts if mode=='adaptive' else np.ones_like(tc))
     covered=(ty>=lo)&(ty<=hi);width=hi-lo;score=width+20*np.maximum(lo-ty,0)+20*np.maximum(ty-hi,0)
     for i,p in enumerate(case):
      for j,q in enumerate(ids):rows.append(dict(seed=seed,fold=fold,case=str(p),family=family,query=int(q),method=method,interval=mode,truth=float(ty[i,j]),center=float(tc[i,j]),squared_error=float((ty[i,j]-tc[i,j])**2),plugin_squared_error=float((ty[i,j]-tp[i,j])**2),lo=float(lo[i,j]),hi=float(hi[i,j]),normalization_scale=float(scale[j]),n_cal=10 if mode!='raw' else 0,conformal_rank=10 if mode!='raw' else 0,coverage=bool(covered[i,j]),width=float(width[i,j]/scale[j]),interval_score=float(score[i,j]/scale[j]),crps=float(crps[i,j]/scale[j]),crps_selection_scaled=float(crps_scaled[i,j]/scale[j]),direct_selection_spread_multiplier=direct_alpha if method=='direct' else 1.))
 frame=pd.DataFrame(rows)
 pooled=frame[frame.family.isin(['clustering','cross'])].copy()
 pooled['family']='all_scored'
 return pd.concat([frame,pooled],ignore_index=True),costs

def summary(df):
 result=[]
 for (fam,meth,mode),g in df.groupby(['family','method','interval']):
  by=g.groupby(['seed','query'])[['squared_error','plugin_squared_error']].mean();ratio=(by.squared_error/by.plugin_squared_error.clip(lower=1e-12)).mean()
  result.append(dict(family=fam,method=meth,interval=mode,seeds=g.seed.nunique(),patients=g.case.nunique(),queries=g['query'].nunique(),MSE=float(g.squared_error.mean()),plugin_MSE=float(g.plugin_squared_error.mean()),MSE_ratio=float(ratio),coverage=float(g.coverage.mean()),width=float(g.width.mean()),interval_score=float(g.interval_score.mean()),crps=float(g.crps.mean()),crps_selection_scaled=float(g.crps_selection_scaled.mean())))
 return result

def comparisons(df):
 rng=np.random.default_rng(271040);result=[]
 # Fixed family-wise comparison to the full direct adaptation, never a selected weak comparator.
 for fam,g in df[df.interval=='adaptive'].groupby('family'):
  # One equally weighted patient error per seed; normalize marker error with fixed pooled plug-in MSE.
  base=g.groupby('query').plugin_squared_error.mean().clip(lower=1e-12)
  g=g.copy();g['error_normalized']=g.squared_error/g['query'].map(base)
  per=g[g.method.isin(['direct','ligature','ligature_analytic'])].groupby(['case','method'])[['error_normalized','interval_score','width','crps','crps_selection_scaled']].mean().unstack('method')
  for method in ['ligature','ligature_analytic']:
   if method not in per['error_normalized']:continue
   for metric in ['error_normalized','interval_score','width','crps','crps_selection_scaled']:
    a=per[metric][method];b=per[metric]['direct'];delta=a-b;picks=rng.integers(0,len(delta),size=(2000,len(delta)));ci=np.quantile(delta.to_numpy()[picks].mean(1),[.025,.975]);p=float(wilcoxon(delta).pvalue) if np.any(np.abs(delta)>1e-12) else 1.
    result.append(dict(family=fam,method=method,metric=metric,ligature_over_direct=float(a.mean()/b.mean()),mean_difference=float(delta.mean()),difference_ci95=ci.tolist(),ligature_better_patients=int((delta<0).sum()),patients=len(delta),p_exploratory=p))
 primary=[r for r in result if r['method']=='ligature' and r['family'] in ['Moran','Geary'] and r['metric']=='error_normalized'];order=sorted(primary,key=lambda r:r['p_exploratory']);last=0.
 for i,r in enumerate(order):last=max(last,min(1.,(len(order)-i)*r['p_exploratory']));r['p_Holm_two_families']=last
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=SEEDS);a=ap.parse_args();seeds=[a.seed] if a.seed else SEEDS
 frames=[];costs=[]
 for seed in seeds:f,c=collect(seed);frames.append(f);costs+=c
 df=pd.concat(frames,ignore_index=True);out=RUN/('evaluation' if not a.seed else f'evaluation_seed{a.seed}');out.mkdir(exist_ok=True)
 df.to_csv(out/'patient_metrics.csv.gz',index=False);metrics=summary(df);report=dict(status='complete' if not a.seed else 'complete_one_seed_pending_fixed_multiseed',seeds=seeds,metrics=metrics,paired=comparisons(df),costs=costs,scope='Development-cohort comparison; paired bootstrap conditions on fits and does not account for all cross-fold dependence. Coverage is per-statistic marginal, not simultaneous.')
 jsonwrite(out/'summary.json',report);pd.DataFrame(metrics).to_csv(out/'summary.csv',index=False);print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__':main()
