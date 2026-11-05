#!/usr/bin/env python
"""Probabilistic comparison of the direct-statistic graph model with Ligature.

The direct model gives a Gaussian for every coordinate of S per region (mean, sd) and a
learned patient-level sd tau. For each test patient and each of the 136 near-minus-far
contrasts, its patient-level forecast is N(mean of region means, sum of region variances / m^2
+ tau^2) over the patient's eligible regions. We score it with the Gaussian CRPS against the
measured patient value and compare with Ligature's sample CRPS (patient_queries.csv) and the
calibrated Gaussian plug-in, on identical patient-statistic pairs; also raw 90% coverage.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_engression import IU, P
from analyze_spatial_engression import PAIRNAMES, qv

K = len(IU[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    rows = []
    for fd in sorted(args.run.glob('fold*')):
        f = fd / f'direct_graph_stain{args.tag}.npz'
        if not f.exists():
            continue
        z, t = np.load(f), np.load(fd / 'test_stats.npz')
        assert (z['test_crop'] == t['crop']).all() and 'test_sd' in z.files
        mu, sd, tau = z['test'][:, P + 3 * K:P + 4 * K], z['test_sd'][:, P + 3 * K:P + 4 * K], z['tau'][P + 3 * K:P + 4 * K]
        truth = qv(t['truth'])
        for c in np.unique(t['case']):
            m = (t['case'] == c) & t['eligible']
            if not m.any():
                continue
            n = m.sum()
            mp, sp = mu[m].mean(0), np.sqrt((sd[m] ** 2).sum(0) / n ** 2 + tau ** 2)
            y = truth[m].mean(0)
            zz = (y - mp) / sp
            crps = sp * (zz * (2 * norm.cdf(zz) - 1) + 2 * norm.pdf(zz) - 1 / np.sqrt(np.pi))
            for q in range(K):
                rows.append(dict(case=c, query=PAIRNAMES[q], crps_direct=crps[q], cover_direct=bool(abs(zz[q]) <= 1.6449), pred_direct=mp[q]))
    d = pd.DataFrame(rows)
    pq = pd.read_csv(args.run / 'patient_queries.csv')[['case', 'query', 'diag', 'truth', 'plugin', 'joint', 'crps_joint', 'crps_gaussplug']]
    m = d.merge(pq, on=['case', 'query'])
    out = {}
    for name, sel in [('all136', m['query'].notna()), ('clustering16', m['diag'].astype(bool))]:
        g = m[sel]
        q = g.groupby('query')[['crps_direct', 'crps_joint', 'crps_gaussplug']].mean()
        pat = g.groupby('case')[['crps_direct', 'crps_joint']].mean()
        e = g.assign(ed=(g.pred_direct - g.truth) ** 2, ej=(g.joint - g.truth) ** 2, ep=(g.plugin - g.truth) ** 2).groupby('query')[['ed', 'ej', 'ep']].mean()
        out[name] = dict(mse_direct=float((e.ed / e.ep).mean()), mse_ligature=float((e.ej / e.ep).mean()), crps_direct=float(g.crps_direct.mean()), crps_ligature=float(g.crps_joint.mean()),
                         crps_gauss_plugin=float(g.crps_gaussplug.mean()),
                         frac_statistics_ligature_better=float((q.crps_joint < q.crps_direct).mean()),
                         patients_ligature_better=int((pat.crps_joint < pat.crps_direct).sum()), patients=int(len(pat)),
                         cover90_direct_raw=float(g.cover_direct.mean()))
    (args.run / f'direct_crps{args.tag}.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
