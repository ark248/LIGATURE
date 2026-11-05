#!/usr/bin/env python
"""Patient-level evaluation of hybrid inference with measured anchor markers.

Only statistics that involve at least one unmeasured marker are scored (statistics
among measured markers are known exactly). MSE is normalized per statistic by the
hybrid plug-in (anchors measured, other markers from the virtual stain).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_spatial_engression import qv, PAIRNAMES, crps_samples
from spatial_engression import IU

SETS = ['PanCK_CD45', 'PDL1_CD8a', 'CD3e']
METHODS = ['hybrid_plugin', 'ridge_impute', 'ligature_uncond', 'ligature_cond']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--alpha', type=float, default=.1)
    args = ap.parse_args()
    summary = {}
    for name in SETS:
        recs = []
        for fd in sorted(args.run.glob('fold*')):
            f = fd / f'hybrid_{name}.npz'
            if not f.exists():
                continue
            z = np.load(f)
            obs = set(z['observed'].tolist())
            keep = np.array([not (a in obs and b in obs) for a, b in zip(*IU)])
            e = z['eligible']
            for c in np.unique(z['case']):
                m = (z['case'] == c) & e
                if not m.any():
                    continue
                t = qv(z['truth'][m]).mean(0)
                vals = {k: qv(z[k][m]).mean(0) for k in ['hybrid_plugin', 'ridge_impute', 'ligature_uncond']}
                s = qv(z['ligature_cond'][m]).mean(0)  # K, Q
                lo, hi = np.quantile(s, [args.alpha / 2, 1 - args.alpha / 2], 0)
                cr = crps_samples(s, t)
                for q in np.flatnonzero(keep):
                    a, b = IU[0][q], IU[1][q]
                    recs.append(dict(fold=fd.name, case=c, query=PAIRNAMES[q], diag=bool(a == b),
                                     involves_anchor=bool(a in obs or b in obs), truth=t[q],
                                     ligature_cond=s[:, q].mean(), lo=lo[q], hi=hi[q], crps_cond=cr[q],
                                     **{k: v[q] for k, v in vals.items()}))
        df = pd.DataFrame(recs)
        df.to_csv(args.run / f'hybrid_{name}_patient.csv', index=False)
        rows = []
        for q, g in df.groupby('query'):
            base = ((g.hybrid_plugin - g.truth) ** 2).mean()
            r = dict(query=q, diag=g['diag'].iloc[0], involves_anchor=g.involves_anchor.iloc[0], truth_mean=g.truth.mean())
            for k in METHODS:
                r['nmse_' + k] = ((g[k] - g.truth) ** 2).mean() / base
                r['rho_' + k] = spearmanr(g.truth, g[k]).correlation if g[k].std() > 0 else np.nan
                r['bias_' + k] = (g[k] - g.truth).mean()
            r['cover'] = ((g.truth >= g.lo) & (g.truth <= g.hi)).mean()
            r['crps_cond'] = g.crps_cond.mean()
            r['crps_hybrid_plugin'] = (g.hybrid_plugin - g.truth).abs().mean()
            rows.append(r)
        s = pd.DataFrame(rows)
        out = {}
        for sub, sel in [('all', s['query'].notna()), ('with_anchor', s.involves_anchor),
                         ('without_anchor', ~s.involves_anchor), ('same_marker', s['diag'])]:
            g = s[sel]
            big = g[g.truth_mean.abs() > .02]
            out[sub] = dict(n=int(len(g)), **{f'nmse_{k}': float(g[f'nmse_{k}'].mean()) for k in METHODS},
                            **{f'rho_{k}': float(g[f'rho_{k}'].mean()) for k in METHODS},
                            **{f'relbias_{k}': float((big[f'bias_{k}'] / big.truth_mean).median()) for k in METHODS},
                            cover=float(g.cover.mean()), crps_cond=float(g.crps_cond.mean()),
                            crps_hybrid_plugin=float(g.crps_hybrid_plugin.mean()),
                            frac_cond_beats_ridge=float((g.nmse_ligature_cond < g.nmse_ridge_impute).mean()),
                            frac_cond_beats_uncond=float((g.nmse_ligature_cond < g.nmse_ligature_uncond).mean()))
        out['patients'] = int(df.case.nunique())
        summary[name] = out
    (args.run / 'hybrid_summary.json').write_text(json.dumps(summary, indent=2))
    for name, out in summary.items():
        print(name, 'patients', out['patients'])
        for sub in ['all', 'with_anchor', 'without_anchor', 'same_marker']:
            o = out[sub]
            print(f"  {sub:15s} n={o['n']:3d} nmse plug {o['nmse_hybrid_plugin']:.3f} ridge {o['nmse_ridge_impute']:.3f} "
                  f"uncond {o['nmse_ligature_uncond']:.3f} cond {o['nmse_ligature_cond']:.3f} | rho plug {o['rho_hybrid_plugin']:.2f} "
                  f"ridge {o['rho_ridge_impute']:.2f} cond {o['rho_ligature_cond']:.2f} | bias plug {o['relbias_hybrid_plugin']:.2f} "
                  f"ridge {o['relbias_ridge_impute']:.2f} cond {o['relbias_ligature_cond']:.2f} | cover {o['cover']:.2f}")


if __name__ == '__main__':
    main()
