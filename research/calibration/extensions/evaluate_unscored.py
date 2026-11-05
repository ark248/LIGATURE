#!/usr/bin/env python3
"""Patient-weighted, eligibility-aware conformal evaluation for bounded queries."""
import argparse
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PROJECT/'release/ligature'))).resolve()
RUN = ROOT/'runs/calibration/orion_unscored_v1'


def rank(n, alpha=.1):
    return int((Decimal(n+1)*(1-Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))


def intervals(cal, test, alpha=.1):
    cy = cal['truth']
    cr, tr = cal['samples'].mean(1), test['samples'].mean(1)
    cq, tq = cal['samples_q'].mean(1), test['samples_q'].mean(1)
    cs, ts = np.maximum(cal['samples'].std(1),1e-6), np.maximum(test['samples'].std(1),1e-6)
    cqs, tqs = np.maximum(cal['samples_q'].std(1),1e-6), np.maximum(test['samples_q'].std(1),1e-6)
    onec, onet = np.ones_like(cy), np.ones_like(test['truth'])
    output = {}
    for name, draws, center in [('ligature_raw',test['samples'],tr),('ligature_q_raw',test['samples_q'],tq)]:
        lo,hi = np.quantile(draws,[alpha/2,1-alpha/2],axis=1)
        output[name] = dict(center=center,lo=lo,hi=hi,n_cal=np.zeros(cy.shape[1],int),
                            rank=np.zeros(cy.shape[1],int),threshold=np.ones(cy.shape[1]))
    specs = dict(plugin_conformal=(cal['plugin'],test['plugin'],onec,onet),
                 plugin_q_conformal=(cal['plugin_q'],test['plugin_q'],onec,onet),
                 ligature_conformal=(cr,tr,cs,ts),
                 ligature_q_conformal=(cq,tq,cqs,tqs),
                 ligature_q_constant_scale_conformal=(cq,tq,onec,onet),
                 plugin_q_ligature_scale_conformal=(cal['plugin_q'],test['plugin_q'],cqs,tqs))
    for name,(cc,tc,scale_cal,scale_test) in specs.items():
        threshold = np.empty(cy.shape[1]);counts = np.empty(cy.shape[1],int);ranks = np.empty(cy.shape[1],int)
        for q in range(cy.shape[1]):
            keep = cal['eligible'][:,q]
            counts[q] = n = int(keep.sum())
            ranks[q] = k = rank(n,alpha)
            scores = np.abs(cy[keep,q]-cc[keep,q])/scale_cal[keep,q]
            assert np.isfinite(scores).all()
            threshold[q] = np.sort(scores)[k-1] if k <= n else np.inf
        half = scale_test*threshold
        output[name] = dict(center=tc,lo=np.maximum(0,tc-half),hi=np.minimum(1,tc+half),
                            n_cal=counts,rank=ranks,threshold=threshold)
    return output


def ci(values):
    values = np.asarray(values); rng = np.random.default_rng(17)
    means = values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def records(folder, family):
    cal = dict(np.load(folder/f'{family}_calibration.npz'))
    test = dict(np.load(folder/f'{family}_test.npz'))
    assert not set(cal['case']) & set(test['case'])
    definition = json.loads((folder/'query_definition.json').read_text())
    markers = definition['markers']
    queries = [a+' -> '+b for a in markers for b in markers] if family == 'contact' else [f'local_CN{i:02d}' for i in range(10)]
    estimates = intervals(cal,test)
    rows = []
    for method,p in estimates.items():
        truth = test['truth']
        width = p['hi']-p['lo']
        score = width+20*np.maximum(p['lo']-truth,0)+20*np.maximum(truth-p['hi'],0)
        coverage = (truth>=p['lo']) & (truth<=p['hi'])
        for i,case in enumerate(test['case']):
            for q,query in enumerate(queries):
                if not test['eligible'][i,q]:
                    continue
                rows.append(dict(fold=folder.name,family=family,case=str(case),query=query,method=method,
                    truth=float(truth[i,q]),center=float(p['center'][i,q]),lo=float(p['lo'][i,q]),hi=float(p['hi'][i,q]),
                    coverage=bool(coverage[i,q]),width=float(width[i,q]),interval_score=float(score[i,q]),
                    squared_error=float((p['center'][i,q]-truth[i,q])**2),
                    plugin_q_squared_error=float((test['plugin_q'][i,q]-truth[i,q])**2),
                    n_cal=int(p['n_cal'][q]),rank=int(p['rank'][q]),threshold=float(p['threshold'][q]),
                    finite_threshold=bool(np.isfinite(p['threshold'][q])),
                    full_support=bool(p['lo'][i,q]==0 and p['hi'][i,q]==1),
                    measured_source_positive_cells=int(test['truth_den'][i,q//16]) if family=='contact' else -1))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    folders = sorted(RUN.glob('smoke*' if args.smoke else 'fold*'))
    folders = [f for f in folders if (f/'complete.json').exists()]
    assert folders and (args.smoke or len(folders)==5), 'All five full folds are required'
    frame = pd.concat([records(f,family) for f in folders for family in ['contact','neighborhood']],ignore_index=True)
    destination = RUN/('smoke_evaluation' if args.smoke else 'evaluation');destination.mkdir(exist_ok=True)
    frame.to_csv(destination/'patient_intervals.csv',index=False)
    metrics = [];pairs=[]
    for family,sub in frame.groupby('family'):
        for method,g in sub.groupby('method'):
            p = g.groupby('case')[['coverage','width','interval_score','squared_error','plugin_q_squared_error','finite_threshold','full_support']].mean()
            metrics.append(dict(family=family,method=method,patients=len(p),eligible_patient_queries=len(g),
                coverage=float(p.coverage.mean()),micro_coverage=float(g.coverage.mean()),
                coverage_ci95_descriptive=ci(p.coverage),mean_width=float(p.width.mean()),
                mean_interval_score=float(p.interval_score.mean()),
                mse_ratio_to_matched_plugin_q=float(p.squared_error.mean()/max(p.plugin_q_squared_error.mean(),1e-12)),
                finite_threshold_fraction=float(p.finite_threshold.mean()),full_support_fraction=float(p.full_support.mean()),
                minimum_calibration_patients=int(g.n_cal.min()),maximum_calibration_patients=int(g.n_cal.max())))
        p = sub.groupby(['case','method'])[['width','interval_score']].mean().unstack('method')
        for baseline in ['plugin_q_conformal','plugin_conformal','ligature_q_constant_scale_conformal','plugin_q_ligature_scale_conformal']:
            for metric in ['width','interval_score']:
                a,b = p[metric]['ligature_q_conformal'],p[metric][baseline]
                delta = a-b
                pairs.append(dict(family=family,baseline=baseline,metric=metric,
                    ratio_of_means=float(a.mean()/b.mean()),patients_ligature_q_better=int((a<b).sum()),patients=len(a),
                    mean_difference=float(delta.mean()),difference_ci95_descriptive=ci(delta)))
    result = dict(status='smoke' if args.smoke else 'complete',alpha=.1,folds=[f.name for f in folders],
        weighting='Equal patient weights; statistics averaged within each eligible patient. Micro coverage also reported.',
        scope='Marginal per-query intervals in the stated eligible population. CN labels local to folds; no simultaneous or conditional-patient guarantee.',
        limitations='Development cohorts; outcome-defined contact eligibility; descriptive cross-fit bootstrap conditional on fitted models.',
        metrics=metrics,paired_comparisons=pairs)
    (destination/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(metrics).to_csv(destination/'summary.csv',index=False)
    frame.groupby(['fold','family','method'])[['coverage','width','interval_score','finite_threshold','full_support']].mean().to_csv(destination/'per_fold.csv')
    frame.groupby(['fold','family','query','method'])[['coverage','width','interval_score','n_cal','finite_threshold']].mean().to_csv(destination/'per_query.csv')
    print(pd.DataFrame(metrics)[['family','method','patients','coverage','mean_width','mean_interval_score','finite_threshold_fraction','full_support_fraction']].to_string(index=False))


if __name__ == '__main__':
    main()
