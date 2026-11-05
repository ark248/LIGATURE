"""Independently recalculate the complete H&E-geometry control's records."""
from common import *
import pandas as pd

def main():
    out = RUN / 'he_control/evaluation'
    report = json.loads((out / 'summary.json').read_text())
    assert report['status'] == 'complete' and report['folds'] == 5 and report['patients'] == 40
    root = RUN / 'he_control'
    roles = []
    sources = {}
    for fold in range(5):
        fd = root / f'independent/fold{fold}'
        role = json.loads((fd / 'roles.json').read_text())
        assert {v: list(role.values()).count(v) for v in set(role.values())} == dict(train=20, selection=2, calibration=10, test=8)
        assert role == json.loads((BASE / f'fold{fold}/roles.json').read_text())
        roles.extend(p for p, r in role.items() if r == 'test')
        for path in [fd / 'complete.json', fd / 'direct/complete.json', fd / 'mean.pt', fd / 'generator.pt']:
            assert path.exists()
            sources[str(path.relative_to(root))] = sha(path)
        for r in ['selection','calibration','test']:
            a = np.load(fd / (r + '_patients.npz'))
            b = np.load(fd / 'direct' / (r + '.npz'))
            assert np.array_equal(a['case'], b['case'])
            assert all(role[str(p)] == r for p in a['case'])
            np.testing.assert_allclose(a['truth'], b['truth'][:,424:560], atol=3e-6)
    assert len(roles) == len(set(roles)) == 40
    path = out / 'patient_metrics.csv.gz'
    record = out / 'patient_metrics.parquet'
    frame = pd.read_csv(path, float_precision='round_trip') if path.exists() else pd.read_parquet(record)
    assert frame.case.nunique() == 40 and frame.groupby('case').fold.nunique().eq(1).all()
    assert np.isfinite(frame[['truth','center','lo','hi','squared_error','plugin_squared_error','width','interval_score','width_normalized','interval_score_normalized']]).all().all()
    assert ((frame.truth >= frame.lo) & (frame.truth <= frame.hi)).equals(frame.coverage)
    np.testing.assert_allclose(frame.squared_error, (frame.truth-frame.center)**2, rtol=1e-6, atol=1e-9)
    width = frame.hi-frame.lo
    score = width + 20*np.maximum(frame.lo-frame.truth,0) + 20*np.maximum(frame.truth-frame.hi,0)
    np.testing.assert_allclose(frame.width,width,rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(frame.interval_score,score,rtol=1e-6,atol=1e-8)
    ia,ib=np.triu_indices(16)
    for fold in range(5):
        fd=root/f'independent/fold{fold}'
        train=np.load(fd/'training_patients.npz')
        markers=json.loads((fd/'markers.json').read_text())
        names=[markers[a]+' : '+markers[b] for a,b in zip(ia,ib)]
        normalization=dict(zip(names,train['truth'].std(0,ddof=1).clip(1e-4)))
        g=frame[frame.fold==f'fold{fold}'];scale=g['query'].map(normalization)
        assert scale.notna().all()
        np.testing.assert_allclose(g.width_normalized,g.width/scale,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(g.interval_score_normalized,g.interval_score/scale,rtol=1e-6,atol=1e-8)
    for _,g in frame.groupby(['case','query']):
        assert g.truth.max()-g.truth.min() < 1e-6
        assert g.plugin_squared_error.max()-g.plugin_squared_error.min() < 1e-6
    for row in report['metrics']:
        g=frame[frame.method==row['method']]
        if row['family']=='clustering16':g=g[g.diag]
        if row['family']=='cross120':g=g[~g.diag]
        by=g.groupby('query')[['squared_error','plugin_squared_error']].mean()
        assert abs(float((by.squared_error/by.plugin_squared_error.clip(lower=1e-12)).mean())-row['mse_relative_to_matched_plugin']) < 1e-8
        for field, source in [('coverage','coverage'),('mean_width_normalized','width_normalized'),('mean_interval_score_normalized','interval_score_normalized')]:
            assert abs(g[source].mean()-row[field]) < 1e-8
    frame.to_parquet(record,index=False,compression='zstd')
    pd.testing.assert_frame_equal(frame,pd.read_parquet(record))
    dataset=json.loads((root/'dataset_complete.json').read_text())
    features=json.loads((root/'features_complete.json').read_text())
    assert dataset['assay_masks_read'] is False and features['assay_segmentation_features_reused'] is False
    assert dataset['segmentation_model_sha256']=='0c724da169d507d8dd2e206ad3675a0b1566824eee6e20ad9a0ba916311d9dd0'
    assert dataset['plan_sha256']==sha(Path(__file__).with_name('HE_CONTROL_PLAN.md'))
    paired=[];rng=np.random.default_rng(271044)
    for family,g in [('all136',frame),('clustering16',frame[frame.diag]),('cross120',frame[~frame.diag])]:
        gen=g[g.method=='ligature_conformal'].pivot(index='case',columns='query',values='squared_error')
        plugin=g[g.method=='plugin_conformal'].pivot(index='case',columns='query',values='squared_error').loc[gen.index,gen.columns]
        direct=g[g.method=='direct_adaptive'].pivot(index='case',columns='query',values='squared_error').loc[gen.index,gen.columns]
        picks=rng.integers(0,40,(2000,40));G=gen.to_numpy();P=plugin.to_numpy();D=direct.to_numpy()
        ratio=(G[picks].mean(1)/P[picks].mean(1).clip(1e-12)).mean(1)
        delta=((G-D)/P.mean(0).clip(1e-12)).mean(1)
        paired.append(dict(family=family,MSE_ratio_over_plugin=float((G.mean(0)/P.mean(0)).mean()),MSE_ratio_ci95=np.quantile(ratio,[.025,.975]).tolist(),normalized_error_difference_from_direct=float(delta.mean()),difference_from_direct_ci95=np.quantile(delta[picks].mean(1),[.025,.975]).tolist()))
    result=dict(status='passed',patients=40,folds=5,seed=271027,patient_rows=len(frame),summary_rows=len(report['metrics']),paired_point=paired,
                exact_parquet_roundtrip=True,all_MSE_ratios_recalculated=True,intervals_recalculated=True,
                disjoint_roles_and_source_target_alignment=True,dataset=dataset,features=features,
                checkpoint_and_completion_hashes=sources,source_manifest=json.loads((root/'independent/source_manifest.json').read_text()),
                report_sha256=sha(out/'summary.json'),patient_records_sha256=sha(record),
                scope='Single-seed input/target-definition sensitivity on the originally selected crop release. Not whole-slide validation or spillover correction.')
    jsonwrite(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['checkpoint_and_completion_hashes','dataset','features','source_manifest']},indent=2))

if __name__ == '__main__':
    main()
