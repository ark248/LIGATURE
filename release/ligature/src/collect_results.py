#!/usr/bin/env python
"""Collect patient-level summaries of all runs into one table (paper/results_table.csv).

Every number reported in the paper is read from the summary*.json files written
by analyze_spatial_engression.py; this script only gathers them.
"""
import json
from pathlib import Path

import pandas as pd

RUNS = [
    ('ligature_v3', 'summary.json', 'Ligature (variance scales calibrated)'),
    ('ligature_v3', 'summary_patientcal.json', 'Ligature (patient-level scale calibration)'),
    ('ligature_v3', 'summary_uncal.json', 'Ligature, no scale calibration'),
    ('ligature_v3_cellonly', 'summary.json', 'ablation: cell-level score only'),
    ('ligature_v3_nolatent', 'summary.json', 'ablation: no patient latent'),
    ('ligature_v3_nofield', 'summary.json', 'ablation: no spatial residual fields'),
    ('ligature_v3_stationary', 'summary.json', 'ablation: stationary loadings'),
    ('ligature_v3_marginal', 'summary.json', 'baseline: per-cell residuals (multiple imputation)'),
    ('ligature_v3_seed2', 'summary.json', 'Ligature seed 2'),
    ('ligature_v3_seed3', 'summary.json', 'Ligature seed 3'),
    ('ligature_v3_ridge', 'summary.json', 'Ligature wrapping ridge virtual stain'),
    ('spatial_engression', 'summary.json', 'v1: noise-in-attention, energy scores'),
    ('abl_marginal', 'summary.json', 'v1 ablation: marginal (late) noise'),
    ('abl_cellonly', 'summary.json', 'v1 ablation: cell energy score only'),
]
KEYS = ['nmse_calibrated', 'nmse_direct', 'nmse_joint', 'median_nmse_joint', 'rho_plugin', 'rho_calibrated',
        'rho_direct', 'rho_joint', 'cover_joint', 'cover_indep', 'cover_recal', 'cover_at50', 'cover_at80',
        'cover_at95', 'width_joint', 'crps_plugin', 'crps_calibrated', 'crps_joint', 'median_relbias_plugin',
        'median_relbias_joint', 'median_relbias_calibrated', 'frac_queries_joint_beats_calibrated',
        'frac_queries_joint_rho_gt_plugin']


def main():
    rows = []
    for run, f, label in RUNS:
        p = (Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs') / run / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for subset in ['all136', 'offdiag120', 'diag16', 'predefined9']:
            r = dict(run=run, file=f, label=label, subset=subset, patients=d['patients'], folds=len(d['folds']))
            r.update({k: d[subset].get(k) for k in KEYS})
            rows.append(r)
    df = pd.DataFrame(rows)
    Path('paper').mkdir(exist_ok=True)
    df.to_csv('paper/results_table.csv', index=False)
    pd.set_option('display.width', 250)
    print(df[df.subset == 'all136'][['label', 'patients', 'nmse_joint', 'rho_joint', 'cover_joint', 'cover_recal',
                                      'crps_joint', 'crps_plugin', 'median_relbias_joint']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
