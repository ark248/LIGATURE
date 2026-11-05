#!/usr/bin/env python
"""Regularized anchor conditioning: choose delta on validation patients, report test.

Statistics among measured anchors are excluded. MSE is per statistic relative to the
anchor-substituted plug-in, averaged over statistics; patient level, equal region weights.
"""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_engression import IU

RUN = (Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3')


def patient_table(split, name):
    out = {}
    for fd in sorted(RUN.glob('fold*')):
        z = np.load(fd / f'hybridreg_{name}_{split}.npz')
        obs = set(z['observed'].tolist())
        keep = np.array([not (a in obs and b in obs) for a, b in zip(*IU)])
        keys = [k for k in z.files if k.startswith('q_')]
        for c in np.unique(z['case']):
            m = (z['case'] == c) & z['eligible']
            if not m.any():
                continue
            out[c] = {k: z[k][m].mean(0)[..., keep] for k in keys}
    return out


def scores(tab, key):
    T = np.array([v['q_truth'] for v in tab.values()]); B = np.array([v['q_hybrid_plugin'] for v in tab.values()])
    X = np.array([v[key] for v in tab.values()])
    est = X.mean(1) if X.ndim == 3 else X
    base = ((B - T) ** 2).mean(0)
    res = dict(nmse=float((((est - T) ** 2).mean(0) / base).mean()))
    if X.ndim == 3:
        lo, hi = np.quantile(X, [.05, .95], 1)
        res['cover90'] = float(((T >= lo) & (T <= hi)).mean())
    return res


summary = {}
for name in ['PanCK_CD45', 'PDL1_CD8a', 'CD3e']:
    va, te = patient_table('val', name), patient_table('test', name)
    keys = [k for k in next(iter(va.values())) if k.startswith('q_cond_')]
    val = {k: scores(va, k)['nmse'] for k in keys + ['q_ligature_uncond']}
    best = min(keys, key=lambda k: val[k])
    summary[name] = dict(validation_nmse=val, chosen=best,
                         test={k: scores(te, k) for k in ['q_ridge_impute', 'q_ligature_uncond', best]})
    print(name, 'val', {k: round(v, 3) for k, v in val.items()}, '-> chosen', best)
    print('   test', {k: {a: round(b, 3) for a, b in v.items()} for k, v in summary[name]['test'].items()})
(RUN / 'hybridreg_summary.json').write_text(json.dumps(summary, indent=2))
