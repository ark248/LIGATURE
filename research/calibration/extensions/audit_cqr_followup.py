#!/usr/bin/env python3
"""Independently replay the one secondary CQR variant and check original integrity."""
import hashlib
import json

import numpy as np
import pandas as pd

from evaluate_unscored import RUN


def main():
    folder=RUN/'cqr_followup'
    design=json.loads((folder/'design.json').read_text())
    assert hashlib.sha256((RUN/'evaluation/summary.json').read_bytes()).hexdigest()==design['primary_summary_sha256']
    frame=pd.read_csv(folder/'patient_intervals.csv')
    original=pd.read_csv(RUN/'evaluation/patient_intervals.csv')
    keys=['fold','family','case','query']
    expected=original[original.method=='ligature_q_conformal']
    assert set(map(tuple,frame[keys].to_numpy()))==set(map(tuple,expected[keys].to_numpy()))
    np.testing.assert_array_equal(frame.coverage,(frame.truth>=frame.lo)&(frame.truth<=frame.hi))
    np.testing.assert_allclose(frame.width,frame.hi-frame.lo,rtol=1e-12,atol=1e-12)
    direct=frame.width+20*np.maximum(frame.lo-frame.truth,0)+20*np.maximum(frame.truth-frame.hi,0)
    np.testing.assert_allclose(frame.interval_score,direct,rtol=1e-12,atol=1e-12)
    replayed=0
    for fold in range(5):
        f=RUN/f'fold{fold}'
        definition=json.loads((f/'query_definition.json').read_text());markers=definition['markers']
        for family in ['contact','neighborhood']:
            cal=dict(np.load(f/f'{family}_calibration.npz'));test=dict(np.load(f/f'{family}_test.npz'))
            cquant=np.quantile(cal['samples_q'],[.05,.95],axis=1)
            tquant=np.quantile(test['samples_q'],[.05,.95],axis=1)
            names=np.asarray([a+' -> '+b for a in markers for b in markers] if family=='contact' else [f'local_CN{i:02d}' for i in range(10)])
            correction=[]
            for q in range(cal['truth'].shape[1]):
                ok=cal['eligible'][:,q];n=int(ok.sum())
                if n<9:
                    correction.append(np.inf)
                else:
                    # For n in {9,10}, the 90% rank is the maximum; calculate
                    # with a direct generator expression instead of sorted ranks.
                    correction.append(max(0,max(max(cquant[0,i,q]-cal['truth'][i,q],cal['truth'][i,q]-cquant[1,i,q]) for i in np.flatnonzero(ok))))
            correction=np.asarray(correction)
            for i,case in enumerate(test['case']):
                ok=test['eligible'][i]
                rows=frame[(frame.fold==f'fold{fold}')&(frame.family==family)&(frame.case==case)].set_index('query').loc[names[ok]]
                np.testing.assert_allclose(rows.lo,np.maximum(0,tquant[0,i,ok]-correction[ok]),rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(rows.hi,np.minimum(1,tquant[1,i,ok]+correction[ok]),rtol=1e-12,atol=1e-12)
                replayed+=len(rows)
    result=json.loads((folder/'summary.json').read_text())
    for row in result['metrics']:
        per=frame[frame.family==row['family']].groupby('case')[['coverage','width','interval_score','finite_threshold','full_support']].mean()
        for col,key in [('coverage','coverage'),('width','mean_width'),('interval_score','mean_interval_score'),('finite_threshold','finite_threshold_fraction'),('full_support','full_support_fraction')]:
            np.testing.assert_allclose(per[col].mean(),row[key])
    audit=dict(status='passed',intervals_replayed=replayed,original_primary_summary_unchanged=True,
               scope='Secondary development diagnostic only; no fitting or test-selected hyperparameters.')
    (folder/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
