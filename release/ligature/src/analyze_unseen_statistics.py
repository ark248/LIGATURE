#!/usr/bin/env python
"""Patient-level evaluation of statistic families absent from every training loss."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/spatial_engression'))
    ap.add_argument('--alpha', type=float, default=.1)
    args = ap.parse_args()
    recs = []
    for fd in sorted(args.run.glob('fold*')):
        if not (fd / 'unseen_stats.npz').exists():
            continue
        z = np.load(fd / 'unseen_stats.npz')
        for fam, valid in [('frac', None), ('copos', 'valid_copos'), ('fine', 'valid_fine')]:
            ok = np.ones(len(z['case']), bool) if valid is None else z[valid]
            for c in np.unique(z['case']):
                m = (z['case'] == c) & ok
                if not m.any():
                    continue
                t = z[f'{fam}_truth'][m, 0].mean(0)
                p = z[f'{fam}_plugin'][m, 0].mean(0)
                pq = z[f'{fam}_plugin_q'][m, 0].mean(0)
                s = z[f'{fam}_samples'][m].mean(0)  # K, D (patient latent shared within sample)
                lo, hi = np.quantile(s, [args.alpha / 2, 1 - args.alpha / 2], 0)
                for q in range(len(t)):
                    recs.append(dict(family=fam, fold=fd.name, case=c, q=q, truth=t[q], plugin=p[q],
                                     plugin_q=pq[q], joint=s[:, q].mean(), lo=lo[q], hi=hi[q]))
    df = pd.DataFrame(recs)
    df.to_csv(args.run / 'unseen_patient.csv', index=False)
    out = {}
    for fam, g in df.groupby('family'):
        rows = []
        for q, h in g.groupby('q'):
            if h.truth.std() == 0:
                continue
            r = dict(q=q, truth_mean=h.truth.mean())
            base = ((h.plugin - h.truth) ** 2).mean()
            for m in ['plugin', 'plugin_q', 'joint']:
                r['nmse_' + m] = ((h[m] - h.truth) ** 2).mean() / max(base, 1e-12)
                r['rho_' + m] = spearmanr(h.truth, h[m]).correlation if h[m].std() > 0 else np.nan
                r['bias_' + m] = (h[m] - h.truth).mean()
            r['cover'] = ((h.truth >= h.lo) & (h.truth <= h.hi)).mean()
            rows.append(r)
        s = pd.DataFrame(rows)
        out[fam] = {k: float(np.nanmean(s[k])) for k in s.columns if k not in ('q', 'truth_mean') and not k.startswith('bias_')}
        big = s[s.truth_mean.abs() > .02]
        out[fam]['n_statistics_abs_mean_gt_0.02'] = int(len(big))
        for m in ['plugin', 'plugin_q', 'joint']:
            out[fam]['median_relbias_' + m] = float((big['bias_' + m] / big.truth_mean).median()) if len(big) else None
            out[fam]['mean_abs_bias_' + m] = float(s['bias_' + m].abs().mean())
        out[fam]['median_nmse_joint'] = float(s.nmse_joint.median())
        out[fam]['median_nmse_plugin_q'] = float(s.nmse_plugin_q.median())
        out[fam]['frac_joint_beats_plugin_q'] = float((s.nmse_joint < s.nmse_plugin_q).mean())
        out[fam]['n_statistics'] = int(len(s))
    out['patients'] = int(df.case.nunique())
    (args.run / 'unseen_summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
