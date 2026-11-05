#!/usr/bin/env python3
"""Reconstruct every test patient's targets from raw inputs without model modules."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PROJECT/'release/ligature'))).resolve()
RUN = ROOT/'runs/calibration/orion_v1'


def main():
    indices = np.load(ROOT/'data/features/cell_idx.npy')
    assert len(indices) == len(np.unique(indices))
    cells = pd.read_parquet(ROOT/'data/cells/cells.parquet').set_index('cell_idx').loc[indices].reset_index()
    assert cells.groupby('crop_id').case.nunique().max() == 1
    positions = cells[['gx', 'gy']].to_numpy(dtype=np.float64) * .325
    groups = cells.groupby('crop_id', sort=True).indices
    records = []
    for fold in range(5):
        folder = RUN/f'fold{fold}'
        assert (folder/'complete.json').exists(), 'Every fit must finish first'
        roles = json.loads((folder/'roles.json').read_text())
        markers = json.loads((folder/'markers.json').read_text())
        raw = np.log1p(cells[['y_'+m for m in markers]].to_numpy(dtype=np.float64))
        training = cells.case.map(roles).to_numpy() == 'train'
        y = (raw-raw[training].mean(axis=0))/(raw[training].std(axis=0)+1e-6)
        expected = dict(np.load(folder/'test_patients.npz'))
        saved = {c: v for c, v in zip(expected['case'], expected['truth'])}
        region_counts = dict(zip(expected['case'], expected['n_regions']))
        collected = {c: [] for c in saved}
        iu = np.triu_indices(len(markers))
        for ii in groups.values():
            case = cells.case.iat[ii[0]]
            if case not in collected or len(ii) < 2:
                continue
            distance = cdist(positions[ii], positions[ii])
            near = (distance > 0) & (distance <= 25)
            far = (distance > 50) & (distance <= 75)
            if not near.any() or not far.any():
                continue
            values = y[ii]
            moment_near = values.T @ (near @ values) / near.sum()
            moment_far = values.T @ (far @ values) / far.sum()
            difference = moment_near-moment_far
            difference = (difference+difference.T)/2
            collected[case].append(difference[iu])
        for case, regions in collected.items():
            assert len(regions) == region_counts[case], (fold, case)
            rebuilt = np.mean(regions, axis=0)
            # Independent float64 calculation vs the generator pipeline's
            # float32 tensor reductions; permit the resulting rounding error.
            np.testing.assert_allclose(rebuilt, saved[case], rtol=1e-4, atol=2e-6)
            records.append(dict(fold=fold, case=case, eligible_regions=len(regions),
                                max_absolute_difference=float(np.max(np.abs(rebuilt-saved[case])))))
    assert len(records) == len({r['case'] for r in records}) == 40
    result = dict(status='passed', patients=40, statistics_per_patient=136,
                  micron_per_pixel=.325, near_bin='(0,25]', far_bin='(50,75]',
                  target='log1p whole-cell marker signal; training-cell standardization; equal eligible-region means',
                  maximum_absolute_difference=max(r['max_absolute_difference'] for r in records),
                  per_patient=records)
    destination = RUN/'evaluation/target_audit.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_patient'}, indent=2))


if __name__ == '__main__':
    main()
