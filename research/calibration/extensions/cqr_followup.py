#!/usr/bin/env python3
"""One explicitly post-primary conservative CQR diagnostic on frozen predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_unscored import RUN, ci, intervals

HERE=Path(__file__).resolve().parent


def bounds(y,clo,chi,tlo,thi,eligible):
    threshold=np.empty(y.shape[1]);counts=eligible.sum(0).astype(int)
    ranks=(9*(counts+1)+9)//10
    for q,(n,k) in enumerate(zip(counts,ranks)):
        scores=np.maximum(clo[eligible[:,q],q]-y[eligible[:,q],q],y[eligible[:,q],q]-chi[eligible[:,q],q])
        threshold[q]=max(0,float(np.sort(scores)[k-1])) if k<=n else np.inf
    return np.maximum(0,tlo-threshold),np.minimum(1,thi+threshold),threshold,counts,ranks


def check_numerics():
    rng=np.random.default_rng(821)
    y=rng.beta(2,5,size=(11,40000))
    scores=np.maximum(.15-y[:10],y[:10]-.55)
    correction=np.maximum(0,scores.max(0))
    coverage=np.mean((y[10]>=np.maximum(0,.15-correction))&(y[10]<=np.minimum(1,.55+correction)))
    assert coverage>=.892,coverage
    for family in ['contact','neighborhood']:
        f=RUN/'smoke0'
        cal=dict(np.load(f/f'{family}_calibration.npz'));test=dict(np.load(f/f'{family}_test.npz'))
        expected=intervals(cal,test)['plugin_q_conformal']
        lo,hi,_,_,_=bounds(cal['truth'],cal['plugin_q'],cal['plugin_q'],test['plugin_q'],test['plugin_q'],cal['eligible'])
        np.testing.assert_allclose(lo,expected['lo'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(hi,expected['hi'],rtol=1e-12,atol=1e-12)
    return float(coverage)


def main():
    simulation=check_numerics()
    primary=RUN/'evaluation/summary.json'
    original_hash=hashlib.sha256(primary.read_bytes()).hexdigest()
    out=RUN/'cqr_followup';out.mkdir(exist_ok=True)
    design=dict(primary_summary_sha256=original_hash,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        design_sha256=hashlib.sha256((HERE/'CQR_FOLLOWUP_DESIGN.md').read_bytes()).hexdigest(),
        status='post-primary development diagnostic; one fixed conservative CQR variant',simulation_coverage=simulation)
    (out/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    records=[]
    for fold in range(5):
        folder=RUN/f'fold{fold}';assert (folder/'complete.json').exists()
        definition=json.loads((folder/'query_definition.json').read_text());markers=definition['markers']
        for family in ['contact','neighborhood']:
            cal=dict(np.load(folder/f'{family}_calibration.npz'));test=dict(np.load(folder/f'{family}_test.npz'))
            clo,chi=np.quantile(cal['samples_q'],[.05,.95],axis=1)
            tlo,thi=np.quantile(test['samples_q'],[.05,.95],axis=1)
            lo,hi,threshold,n,k=bounds(cal['truth'],clo,chi,tlo,thi,cal['eligible'])
            names=[a+' -> '+b for a in markers for b in markers] if family=='contact' else [f'local_CN{i:02d}' for i in range(10)]
            for i,case in enumerate(test['case']):
                for q,name in enumerate(names):
                    if not test['eligible'][i,q]:continue
                    width=hi[i,q]-lo[i,q];truth=test['truth'][i,q]
                    score=width+20*max(lo[i,q]-truth,0)+20*max(truth-hi[i,q],0)
                    records.append(dict(fold=f'fold{fold}',family=family,case=str(case),query=name,method='ligature_q_cqr_followup',
                        truth=truth,lo=lo[i,q],hi=hi[i,q],width=width,interval_score=score,
                        coverage=bool(lo[i,q]<=truth<=hi[i,q]),n_cal=int(n[q]),rank=int(k[q]),
                        finite_threshold=bool(np.isfinite(threshold[q])),full_support=bool(lo[i,q]==0 and hi[i,q]==1)))
    frame=pd.DataFrame(records);frame.to_csv(out/'patient_intervals.csv',index=False)
    original=pd.read_csv(RUN/'evaluation/patient_intervals.csv')
    assert not frame.duplicated(['family','case','query']).any()
    metrics=[];pairs=[]
    for family,sub in frame.groupby('family'):
        p=sub.groupby('case')[['coverage','width','interval_score','finite_threshold','full_support']].mean()
        metrics.append(dict(family=family,method='ligature_q_cqr_followup',patients=len(p),coverage=float(p.coverage.mean()),
            mean_width=float(p.width.mean()),mean_interval_score=float(p.interval_score.mean()),
            finite_threshold_fraction=float(p.finite_threshold.mean()),full_support_fraction=float(p.full_support.mean())))
        for baseline in ['plugin_q_conformal','ligature_q_conformal','ligature_q_constant_scale_conformal']:
            b=original[(original.family==family)&(original.method==baseline)].groupby('case')[['width','interval_score']].mean().loc[p.index]
            for column in ['width','interval_score']:
                delta=p[column]-b[column]
                pairs.append(dict(family=family,baseline=baseline,metric=column,ratio_of_means=float(p[column].mean()/b[column].mean()),
                    patients_cqr_better=int((p[column]<b[column]).sum()),patients=len(p),
                    mean_difference=float(delta.mean()),difference_ci95_descriptive=ci(delta)))
    result=dict(design=design,metrics=metrics,paired_comparisons=pairs)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    assert hashlib.sha256(primary.read_bytes()).hexdigest()==original_hash
    lines=['# Secondary quantile-calibration diagnostic','',
        '**Post-primary development follow-up; not a prespecified primary or confirmatory result.** One fixed conservative CQR variant, no refitting or tuning.',
        'Original eight-method results remain unchanged. Source/design and original-summary hashes are recorded before this calculation.',
        'Method: predictive 5th/95th sample quantiles, exact eligible-patient rank, nonnegative CQR correction, [0,1] support.',
        'Reference: [Romano, Patterson and Candès (2019), equations 14–16](https://arxiv.org/html/1905.03222v1).','',
        '| Family | Coverage | Width | Interval score | Finite threshold | Full support |',
        '|---|---:|---:|---:|---:|---:|']
    for r in metrics:
        lines.append(f"| {r['family']} | {100*r['coverage']:.1f}% | {r['mean_width']:.3f} | {r['mean_interval_score']:.3f} | {100*r['finite_threshold_fraction']:.1f}% | {100*r['full_support_fraction']:.1f}% |")
    lines+=['','| Family | Baseline | Metric | CQR / baseline | Patients lower | Descriptive difference interval |',
            '|---|---|---|---:|---:|---|']
    for r in pairs:
        a,b=r['difference_ci95_descriptive']
        lines.append(f"| {r['family']} | {r['baseline']} | {r['metric']} | {r['ratio_of_means']:.3f} | {r['patients_cqr_better']}/{r['patients']} | [{a:.4f}, {b:.4f}] |")
    lines+=['','Eligibility, local CN identities, data reuse and overlapping-fit uncertainty qualifications are identical to the primary report.',
            'Do not replace the primary analysis selectively, or claim CQR is a new Ligature contribution.','']
    (HERE/'CQR_FOLLOWUP_RESULTS.md').write_text('\n'.join(lines))
    (HERE/'CQR_FOLLOWUP_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__=='__main__':main()
