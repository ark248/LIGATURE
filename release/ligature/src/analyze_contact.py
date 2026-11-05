#!/usr/bin/env python
"""Patient-level evaluation of gated contact fractions (eval_contact.py outputs).

Patient value: (sum over regions of a+ cells with a b+ neighbor) / (sum of a+ cells).
Pairs are kept for a patient when it has at least 20 measured a+ cells; methods with
no predicted a+ cells in a patient get the cohort-free fallback value 0.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_engression import MARKERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--min-cells', type=int, default=20)
    args = ap.parse_args()
    recs = []
    for fd in sorted(args.run.glob('fold*')):
        f = fd / 'contact_stats.npz'
        if not f.exists():
            continue
        z = np.load(f)
        for c in np.unique(z['case']):
            m = z['case'] == c
            frac = {}
            for k in ['truth', 'plugin', 'plugin_q']:
                num, den = z[k + '_num'][m].sum(0), z[k + '_den'][m].sum(0)
                frac[k] = np.where(den[:, None] > 0, num / np.maximum(den[:, None], 1), 0.)
                if k == 'truth':
                    tden = den
            num, den = z['samples_num'][m].sum(0), z['samples_den'][m].sum(0)  # K,P,P ; K,P
            fs = np.where(den[:, :, None] > 0, num / np.maximum(den[:, :, None], 1), 0.)
            lo, hi = np.quantile(fs, [.05, .95], 0)
            for a in range(len(MARKERS)):
                if tden[a] < args.min_cells:
                    continue
                for b in range(len(MARKERS)):
                    recs.append(dict(fold=fd.name, case=c, pair=f'{MARKERS[a]} -> {MARKERS[b]}', same=a == b,
                                     truth=frac['truth'][a, b], plugin=frac['plugin'][a, b],
                                     plugin_q=frac['plugin_q'][a, b], joint=fs[:, a, b].mean(),
                                     lo=lo[a, b], hi=hi[a, b]))
    df = pd.DataFrame(recs)
    df.to_csv(args.run / 'contact_patient.csv', index=False)
    rows = []
    for q, g in df.groupby('pair'):
        if len(g) < 10 or g.truth.std() == 0:
            continue
        r = dict(pair=q, same=g['same'].iloc[0], n=len(g), truth_mean=g.truth.mean())
        base = ((g.plugin - g.truth) ** 2).mean()
        for m in ['plugin', 'plugin_q', 'joint']:
            r['nmse_' + m] = ((g[m] - g.truth) ** 2).mean() / max(base, 1e-12)
            r['rho_' + m] = spearmanr(g.truth, g[m]).correlation if g[m].std() > 0 else np.nan
            r['relbias_' + m] = (g[m] - g.truth).mean() / g.truth.mean()
        r['cover'] = ((g.truth >= g.lo) & (g.truth <= g.hi)).mean()
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(args.run / 'contact_summary.csv', index=False)
    out = dict(patients=int(df.case.nunique()), pairs=int(len(s)), radius_um=20)
    for m in ['plugin', 'plugin_q', 'joint']:
        out['mean_nmse_' + m] = float(s['nmse_' + m].mean())
        out['median_nmse_' + m] = float(s['nmse_' + m].median())
        out['mean_rho_' + m] = float(s['rho_' + m].mean())
        out['median_relbias_' + m] = float(s['relbias_' + m].median())
    out['frac_joint_beats_plugin_q'] = float((s.nmse_joint < s.nmse_plugin_q).mean())
    out['frac_joint_beats_plugin'] = float((s.nmse_joint < s.nmse_plugin).mean())
    out['coverage_raw_90'] = float(s.cover.mean())
    ex = ['CD8a -> Pan-CK', 'CD3e -> Pan-CK', 'CD68 -> PD-L1', 'PD-L1 -> CD8a', 'FOXP3 -> CD8a', 'CD8a -> CD68']
    out['examples'] = s[s.pair.isin(ex)][['pair', 'truth_mean', 'nmse_plugin_q', 'nmse_joint', 'relbias_plugin',
                                         'relbias_plugin_q', 'relbias_joint', 'rho_plugin', 'rho_joint']].round(3).to_dict('records')
    (args.run / 'contact_summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
