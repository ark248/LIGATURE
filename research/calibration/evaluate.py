#!/usr/bin/env python3
"""Prespecified split-conformal comparison on the independently fitted runs."""
import argparse
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PROJECT/'release/ligature'))).resolve()
RUN=ROOT/'runs/calibration/orion_v1'


def conformal_rank(n, alpha):
    if n<1 or not 0<alpha<1:
        raise ValueError('Need calibration patients and 0 < alpha < 1')
    return int((Decimal(n+1)*(Decimal(1)-Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))


def conformal_threshold(truth, center, scale, alpha=.1):
    if truth.shape != center.shape or truth.shape != scale.shape:
        raise ValueError('Calibration score shapes differ')
    if not np.isfinite(truth).all() or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Nonfinite or nonpositive conformity inputs')
    n=len(truth);k=conformal_rank(n,alpha)
    if k>n:
        return np.full(truth.shape[1],np.inf),k
    r=np.abs(truth-center)/scale
    return np.sort(r,axis=0)[k-1],k


def correction_slopes(selection,diag):
    p,y=selection['plugin'],selection['truth']
    slope=float(np.sum(p*y)/max(float(np.sum(p*p)),1e-12))
    family=np.empty(p.shape[1])
    for mask in [diag,~diag]:
        family[mask]=float(np.sum(p[:,mask]*y[:,mask])/max(float(np.sum(p[:,mask]**2)),1e-12))
    return slope,family


def methods(cal,test,selection,diag,alpha=.1):
    cY,tY=cal['truth'],test['truth']
    cP,tP=cal['plugin'],test['plugin']
    cJ,tJ=cal['draws'].mean(1),test['draws'].mean(1)
    cS,tS=np.maximum(cal['draws'].std(1),1e-6),np.maximum(test['draws'].std(1),1e-6)
    slope,family=correction_slopes(selection,diag)
    cR,tR=slope*cP,slope*tP
    cF,tF=family*cP,family*tP
    outputs={}
    raw_sd=np.maximum(np.sqrt(np.mean((selection['truth']-selection['plugin'])**2,axis=0)),1e-6)
    raw_rs=np.maximum(np.sqrt(np.mean((selection['truth']-slope*selection['plugin'])**2,axis=0)),1e-6)
    z=1.6448536269514722
    outputs['plugin_raw_gaussian']=dict(center=tP,lo=tP-z*raw_sd,hi=tP+z*raw_sd)
    outputs['rescale_raw_gaussian']=dict(center=tR,lo=tR-z*raw_rs,hi=tR+z*raw_rs)
    lo,hi=np.quantile(test['draws'],[alpha/2,1-alpha/2],axis=1)
    outputs['ligature_raw_ensemble']=dict(center=tJ,lo=lo,hi=hi)
    specs={
        'plugin_conformal':(cP,tP,np.ones_like(cP),np.ones_like(tP)),
        'rescale_conformal':(cR,tR,np.ones_like(cP),np.ones_like(tP)),
        'family_rescale_conformal':(cF,tF,np.ones_like(cP),np.ones_like(tP)),
        'ligature_conformal':(cJ,tJ,cS,tS),
        'rescale_ligature_scale_conformal':(cR,tR,cS,tS),
        'ligature_constant_scale_conformal':(cJ,tJ,np.ones_like(cP),np.ones_like(tP))}
    for name,(cc,tc,cs,ts) in specs.items():
        threshold,k=conformal_threshold(cY,cc,cs,alpha)
        half=ts*threshold
        outputs[name]=dict(center=tc,lo=tc-half,hi=tc+half,threshold=threshold,rank=k,n_cal=len(cY))
    return outputs,dict(global_slope=slope,clustering_slope=float(family[diag][0]),cross_slope=float(family[~diag][0]))


def fold_records(folder,alpha=.1):
    cal=np.load(folder/'calibration_patients.npz')
    test=np.load(folder/'test_patients.npz')
    selection=np.load(folder/'selection_patients.npz')
    training=np.load(folder/'training_patients.npz')
    sets=[set(z['case']) for z in [cal,test,selection,training]]
    assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
    markers=json.loads((folder/'markers.json').read_text())
    ia,ib=np.triu_indices(len(markers));diag=ia==ib
    names=[markers[a]+' : '+markers[b] for a,b in zip(ia,ib)]
    scale=np.maximum(training['truth'].std(0,ddof=1),1e-4)
    estimates,slopes=methods(cal,test,selection,diag,alpha)
    rec=[]
    for name,p in estimates.items():
        width=p['hi']-p['lo'];truth=test['truth']
        score=width+2/alpha*np.maximum(p['lo']-truth,0)+2/alpha*np.maximum(truth-p['hi'],0)
        cover=(truth>=p['lo'])&(truth<=p['hi'])
        for i,case in enumerate(test['case']):
            for q,query in enumerate(names):
                rec.append(dict(fold=folder.name,case=str(case),query=query,diag=bool(diag[q]),method=name,
                    truth=float(truth[i,q]),center=float(p['center'][i,q]),lo=float(p['lo'][i,q]),hi=float(p['hi'][i,q]),
                    width=float(width[i,q]),coverage=bool(cover[i,q]),interval_score=float(score[i,q]),
                    width_normalized=float(width[i,q]/scale[q]),interval_score_normalized=float(score[i,q]/scale[q]),
                    squared_error=float((p['center'][i,q]-truth[i,q])**2),
                    plugin_squared_error=float((test['plugin'][i,q]-truth[i,q])**2),
                    n_cal=int(p.get('n_cal',0)),rank=int(p.get('rank',0)),
                    threshold=float(p['threshold'][q]) if 'threshold' in p else np.nan))
    return pd.DataFrame(rec),slopes


def bootstrap_ci(values,seed=17,draws=2000):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(values),size=(draws,len(values)))
    bs=values[picks].mean(1)
    return [float(v) for v in np.quantile(bs,[.025,.975])]


def summarize(df):
    records=[]
    for family,sub in [('all136',df),('clustering16',df[df.diag]),('cross120',df[~df.diag])]:
        for method,g in sub.groupby('method'):
            per=g.groupby('case')[['coverage','width_normalized','interval_score_normalized']].mean()
            mse=g.groupby('query').squared_error.mean()
            base=g.groupby('query').plugin_squared_error.mean()
            records.append(dict(family=family,method=method,patients=g.case.nunique(),queries=g['query'].nunique(),
                coverage=float(g.coverage.mean()),coverage_ci95_descriptive=bootstrap_ci(per.coverage),
                mean_width=float(g.width.mean()),mean_width_normalized=float(g.width_normalized.mean()),
                mean_interval_score_normalized=float(g.interval_score_normalized.mean()),
                mse_relative_to_matched_plugin=float((mse/base.clip(lower=1e-12)).mean()),
                finite_interval_fraction=float(np.isfinite(g.width).mean())))
    return records


def paired(df):
    rows=[]
    for family,sub in [('all136',df),('clustering16',df[df.diag]),('cross120',df[~df.diag])]:
        p=sub.groupby(['case','method'])[['width_normalized','interval_score_normalized']].mean().unstack('method')
        for baseline in ['plugin_conformal','rescale_conformal','family_rescale_conformal','rescale_ligature_scale_conformal','ligature_constant_scale_conformal']:
            for metric in ['width_normalized','interval_score_normalized']:
                a=p[metric]['ligature_conformal'];b=p[metric][baseline]
                delta=a-b
                pvalue=float(wilcoxon(a,b).pvalue) if np.any(np.abs(delta)>1e-12) else 1.
                rows.append(dict(family=family,baseline=baseline,metric=metric,ratio_of_means=float(a.mean()/b.mean()),
                    patients_ligature_better=int((a<b).sum()),patients=len(a),
                    mean_difference=float(delta.mean()),difference_ci95_descriptive=bootstrap_ci(delta),
                    wilcoxon_p_descriptive=pvalue))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    folders=sorted(RUN.glob('smoke*' if args.smoke else 'fold*'))
    folders=[p for p in folders if (p/'complete.json').exists()]
    if not folders:raise RuntimeError('No completed independent-calibration fits')
    if not args.smoke and not args.partial and len(folders)!=5:raise RuntimeError('All five folds are required for the final report')
    all_records=[];slopes={}
    for folder in folders:
        frame,coef=fold_records(folder);all_records.append(frame);slopes[folder.name]=coef
    df=pd.concat(all_records,ignore_index=True)
    destination=RUN/('smoke_evaluation' if args.smoke else 'evaluation');destination.mkdir(exist_ok=True)
    df.to_csv(destination/'patient_intervals.csv',index=False)
    summary=summarize(df);comparisons=paired(df)
    (destination/'summary.json').write_text(json.dumps(dict(alpha=.1,folds=[p.name for p in folders],
        status='complete' if len(folders)==5 and not args.smoke else 'smoke' if args.smoke else 'partial',
        scope='Development evidence; per-statistic marginal split-conformal validity under exchangeability, not simultaneous coverage.',
        uncertainty_note='Bootstrap and paired-test summaries condition on fitted models and do not account for all cross-fold training dependence.',
        metrics=summary,paired_comparisons=comparisons,baseline_slopes=slopes),indent=2)+'\n')
    pd.DataFrame(summary).to_csv(destination/'summary.csv',index=False)
    df.groupby(['fold','method','diag'])[['coverage','width_normalized','interval_score_normalized']].mean().to_csv(destination/'per_fold.csv')
    print(pd.DataFrame(summary)[['family','method','patients','coverage','mean_width_normalized','mean_interval_score_normalized','mse_relative_to_matched_plugin']].to_string(index=False))


if __name__=='__main__':main()
