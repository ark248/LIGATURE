#!/usr/bin/env python3
"""Independently reconstruct nonlinear targets and replay calibrated intervals."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PROJECT/'release/ligature'))).resolve()
RUN=ROOT/'runs/calibration/orion_unscored_v1'
TYPE_MARKERS=[['Pan-CK','E-cadherin'],['CD8a'],['FOXP3'],['CD3e','CD4'],['CD20'],['CD68','CD163'],['CD31'],['SMA']]


def main():
    destination=RUN/'evaluation'
    summary=json.loads((destination/'summary.json').read_text())
    assert summary['status']=='complete'
    frame=pd.read_csv(destination/'patient_intervals.csv')
    assert not frame.duplicated(['family','case','query','method']).any()
    assert frame.method.nunique()==8 and frame.case.nunique()==40
    np.testing.assert_array_equal(frame.coverage,(frame.truth>=frame.lo)&(frame.truth<=frame.hi))
    np.testing.assert_allclose(frame.width,frame.hi-frame.lo,rtol=1e-11,atol=1e-12)
    score=frame.width+20*np.maximum(frame.lo-frame.truth,0)+20*np.maximum(frame.truth-frame.hi,0)
    np.testing.assert_allclose(frame.interval_score,score,rtol=1e-11,atol=1e-12)
    assert (frame.lo>=0).all() and (frame.hi<=1).all()
    model_manifest=json.loads((RUN/'source_manifest.json').read_text())
    for name,want in model_manifest['files'].items():
        assert hashlib.sha256((RUN/'src'/name).read_bytes()).hexdigest()==want
    indices=np.load(ROOT/'data/features/cell_idx.npy')
    cells=pd.read_parquet(ROOT/'data/cells/cells.parquet').set_index('cell_idx').loc[indices].reset_index()
    assert cells.groupby('crop_id').case.nunique().max()==1
    xy=cells[['gx','gy']].to_numpy(dtype=np.float64)*.325
    groups=cells.groupby('crop_id',sort=True).indices
    all_cases=[];replayed=0;max_difference=0.
    for fold in range(5):
        folder=RUN/f'fold{fold}'
        roles=json.loads((folder/'roles.json').read_text())
        audit=json.loads((folder/'fit_audit.json').read_text())
        sets={r:{c for c,v in roles.items() if v==r} for r in ['train','selection','calibration','test']}
        assert list(map(len,sets.values()))==[20,2,10,8]
        assert len(set.union(*sets.values()))==40
        assert set(audit['measured_gate_patients'])==sets['train']
        assert set(audit['neighborhood_template_patients'])==sets['train']
        assert set(audit['plugin_quantile_gate_patients'])==sets['selection']
        assert set(audit['sample_quantile_gate_patients'])==sets['selection']
        ref=audit['model_reference']
        for name,want in ref['sha256'].items():
            assert hashlib.sha256((Path(ref['directory'])/name).read_bytes()).hexdigest()==want
        definition=json.loads((folder/'query_definition.json').read_text())
        markers=definition['markers'];threshold=np.asarray(definition['measured_threshold'])
        raw=np.log1p(cells[['y_'+m for m in markers]].to_numpy(dtype=np.float64))
        train=cells.case.map(roles).to_numpy()=='train'
        y=((raw-raw[train].mean(0))/(raw[train].std(0)+1e-6)).astype(np.float32)
        np.testing.assert_allclose(threshold,np.quantile(y[train],.85,axis=0),rtol=0,atol=1e-12)
        centers=np.asarray(definition['centroids']);assert centers.shape==(10,9)
        test_cases=sorted(sets['test']);all_cases.extend(test_cases)
        num={c:np.zeros((16,16),np.int64) for c in test_cases}
        den={c:np.zeros(16,np.int64) for c in test_cases}
        neighborhoods={c:np.zeros(10,np.int64) for c in test_cases}
        n_cells=dict.fromkeys(test_cases,0)
        for ii in groups.values():
            case=cells.case.iat[ii[0]]
            if case not in num or len(ii)<2:continue
            values=y[ii];positions=xy[ii]
            adjacency=cdist(positions,positions).astype(np.float32)<=20
            np.fill_diagonal(adjacency,False)
            positive=(values>threshold.astype(np.float32)).astype(np.int64)
            has_neighbor=(adjacency@positive)>0
            num[case]+=positive.T@has_neighbor.astype(np.int64)
            den[case]+=positive.sum(0)
            if len(ii)>=10:
                nbr=cKDTree(positions).query(positions,k=10)[1]
                positive_cn=values>threshold
                types=np.full(len(ii),8)
                for k in reversed(range(8)):
                    columns=[markers.index(m) for m in TYPE_MARKERS[k]]
                    types=np.where(positive_cn[:,columns].any(1),k,types)
                composition=np.eye(9)[types][nbr].mean(1)
                labels=((composition[:,None,:]-centers[None,:,:])**2).sum(2).argmin(1)
                neighborhoods[case]+=np.bincount(labels,minlength=10)
                n_cells[case]+=len(ii)
        for family in ['contact','neighborhood']:
            cal=dict(np.load(folder/f'{family}_calibration.npz'))
            test=dict(np.load(folder/f'{family}_test.npz'))
            assert set(cal['case'])==sets['calibration'] and set(test['case'])==sets['test']
            assert cal['samples_q'].shape[1]==test['samples_q'].shape[1]==64
            queries=[a+' -> '+b for a in markers for b in markers] if family=='contact' else [f'local_CN{i:02d}' for i in range(10)]
            for i,case in enumerate(test['case']):
                if family=='contact':
                    rebuilt=(num[case]/np.maximum(den[case][:,None],1)).reshape(-1)
                    np.testing.assert_array_equal(test['truth_den'][i],den[case])
                    np.testing.assert_array_equal(test['eligible'][i],np.repeat(den[case]>=20,16))
                else:
                    rebuilt=neighborhoods[case]/n_cells[case]
                np.testing.assert_allclose(test['truth'][i],rebuilt,rtol=0,atol=1e-12)
                max_difference=max(max_difference,float(np.max(np.abs(test['truth'][i]-rebuilt))))
            for method,constant in [('ligature_q_conformal',False),('ligature_q_constant_scale_conformal',True),('plugin_q_conformal',True)]:
                cc=cal['plugin_q'] if method=='plugin_q_conformal' else cal['samples_q'].mean(1)
                tc=test['plugin_q'] if method=='plugin_q_conformal' else test['samples_q'].mean(1)
                cs=np.ones_like(cc) if constant else np.maximum(cal['samples_q'].std(1),1e-6)
                ts=np.ones_like(tc) if constant else np.maximum(test['samples_q'].std(1),1e-6)
                part=frame[(frame.fold==f'fold{fold}')&(frame.family==family)&(frame.method==method)]
                for q,name in enumerate(queries):
                    keep=cal['eligible'][:,q];n=int(keep.sum());k=(9*(n+1)+9)//10
                    threshold_q=np.sort(np.abs(cal['truth'][keep,q]-cc[keep,q])/cs[keep,q])[k-1] if k<=n else np.inf
                    for i,case in enumerate(test['case']):
                        if not test['eligible'][i,q]:continue
                        row=part[(part.case==case)&(part['query']==name)].iloc[0]
                        assert row.n_cal==n and row['rank']==k
                        assert bool(row.finite_threshold)==bool(np.isfinite(threshold_q))
                        np.testing.assert_allclose(row.lo,max(0,tc[i,q]-ts[i,q]*threshold_q),rtol=1e-11,atol=1e-12)
                        np.testing.assert_allclose(row.hi,min(1,tc[i,q]+ts[i,q]*threshold_q),rtol=1e-11,atol=1e-12)
                        replayed+=1
    assert len(all_cases)==len(set(all_cases))==40
    for metric in summary['metrics']:
        sub=frame[(frame.family==metric['family'])&(frame.method==metric['method'])]
        per=sub.groupby('case')[['coverage','width','interval_score','finite_threshold','full_support']].mean()
        for column,key in [('coverage','coverage'),('width','mean_width'),('interval_score','mean_interval_score'),('finite_threshold','finite_threshold_fraction'),('full_support','full_support_fraction')]:
            np.testing.assert_allclose(per[column].mean(),metric[key],rtol=1e-12,atol=1e-12)
    result=dict(status='passed',patients=40,interval_records=len(frame),independently_replayed_intervals=replayed,
        independent_target_max_absolute_difference=max_difference,
        checks=['unchanged model/source references','patient/gate/template isolation','all test targets from raw assay and geometry',
                'eligibility and actual calibration counts','exact ranks with insufficient cases','bounded support and all interval arithmetic','patient-weighted summaries'])
    (destination/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
