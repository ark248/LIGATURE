#!/usr/bin/env python3
"""Independently replay published intervals from patient bundles and audit provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PROJECT/'release/ligature'))).resolve()
RUN = ROOT / 'runs/calibration/orion_v1'


def main():
    summary = json.loads((RUN/'evaluation/summary.json').read_text())
    assert summary['status'] == 'complete'
    assert summary['folds'] == [f'fold{i}' for i in range(5)]
    manifest = json.loads((RUN/'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((RUN/'src'/name).read_bytes()).hexdigest() == digest
    frame = pd.read_csv(RUN/'evaluation/patient_intervals.csv')
    assert not frame.duplicated(['case', 'query', 'method']).any()
    assert len(frame) == 40 * 136 * 9
    all_test = []
    replayed = 0
    for fold in summary['folds']:
        folder = RUN/fold
        audit = json.loads((folder/'fit_audit.json').read_text())
        roles = json.loads((folder/'roles.json').read_text())
        bundles = {r: dict(np.load(folder/f'{r}_patients.npz'))
                   for r in ['training', 'selection', 'calibration', 'test']}
        groups = [set(b['case']) for b in bundles.values()]
        assert list(map(len, groups)) == [20, 2, 10, 8]
        assert len(set.union(*groups)) == sum(map(len, groups))
        for role, bundle in bundles.items():
            role_name = 'train' if role == 'training' else role
            assert set(bundle['case']) == set(audit['patient_sets'][role_name])
            assert all(roles[c] == role_name for c in bundle['case'])
            assert np.isfinite(bundle['truth']).all()
        assert set(audit['normalization_and_pca_patients']) == groups[0]
        assert set(audit['mean_early_stopping_patients']) == groups[1]
        assert set(audit['generator_early_stopping_patients']) == groups[1]
        assert set(audit['variance_fitting_patients']) == groups[1]
        cal, test = bundles['calibration'], bundles['test']
        assert cal['draws'].shape == (10, 64, 136)
        assert test['draws'].shape == (8, 64, 136)
        all_test.extend(test['case'])
        clustering = np.equal(*np.triu_indices(16))
        # Derive query ordering from the saved marker list, independently of evaluate.py.
        markers = json.loads((folder/'markers.json').read_text())
        query_names = [a+' : '+b for i, a in enumerate(markers) for b in markers[i:]]
        for method, adaptive in [('ligature_conformal', True),
                                 ('ligature_constant_scale_conformal', False),
                                 ('plugin_conformal', False)]:
            if method.startswith('ligature'):
                cc, tc = cal['draws'].mean(axis=1), test['draws'].mean(axis=1)
            else:
                cc, tc = cal['plugin'], test['plugin']
            cs = np.maximum(cal['draws'].std(axis=1), 1e-6) if adaptive else np.ones_like(cc)
            ts = np.maximum(test['draws'].std(axis=1), 1e-6) if adaptive else np.ones_like(tc)
            # At n=10, alpha=.1 the exact finite-sample threshold is the maximum.
            threshold = np.max(np.abs(cal['truth']-cc)/cs, axis=0)
            part = frame[(frame.fold == fold) & (frame.method == method)]
            for i, case in enumerate(test['case']):
                rows = part[part.case == case].set_index('query').loc[query_names]
                np.testing.assert_allclose(rows.center, tc[i], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(rows.lo, tc[i]-ts[i]*threshold, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(rows.hi, tc[i]+ts[i]*threshold, rtol=1e-10, atol=1e-10)
                np.testing.assert_array_equal(rows.diag.to_numpy(), clustering)
                replayed += len(rows)
    assert len(all_test) == len(set(all_test)) == 40
    np.testing.assert_array_equal(frame.coverage, (frame.truth >= frame.lo) & (frame.truth <= frame.hi))
    for method, rows in frame.groupby('method'):
        # Quantile interpolation and the two-family slope vector use float64;
        # other intervals use the prediction bundles' float32 precision. CSV
        # parsing promotes both to float64, so replay in the recorded precision
        # rather than mistaking one float32 rounding unit for an arithmetic bug.
        precision = np.float64 if method in ['ligature_raw_ensemble', 'family_rescale_conformal'] else np.float32
        lo, hi, truth = [rows[k].to_numpy(dtype=precision) for k in ['lo', 'hi', 'truth']]
        direct_width = hi-lo
        direct_score = direct_width + precision(20)*np.maximum(lo-truth, precision(0)) + precision(20)*np.maximum(truth-hi, precision(0))
        np.testing.assert_allclose(rows.width, direct_width, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(rows.interval_score, direct_score, rtol=1e-10, atol=1e-10)
    for row in summary['metrics']:
        part = frame[frame.method == row['method']]
        if row['family'] == 'clustering16': part = part[part.diag]
        if row['family'] == 'cross120': part = part[~part.diag]
        assert part.case.nunique() == row['patients'] == 40
        assert part['query'].nunique() == row['queries']
        np.testing.assert_allclose(part.coverage.mean(), row['coverage'])
        np.testing.assert_allclose(part.width_normalized.mean(), row['mean_width_normalized'])
        np.testing.assert_allclose(part.interval_score_normalized.mean(), row['mean_interval_score_normalized'])
    result = dict(status='passed', patients=40, methods=9, queries=136,
                  interval_records=len(frame), independently_replayed_intervals=replayed,
                  checks=['source hashes', 'disjoint patient roles and fitting audit',
                          'each patient tested exactly once', '64 draws per held-out patient',
                          'three conformal methods replayed independently',
                          'all coverage, width, and interval-score records', 'all aggregate summaries'])
    (RUN/'evaluation/audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
