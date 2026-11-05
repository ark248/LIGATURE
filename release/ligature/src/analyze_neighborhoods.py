#!/usr/bin/env python
"""Patient-level evaluation of cellular-neighborhood frequencies (eval_neighborhoods.py outputs)."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    args = ap.parse_args()
    # k-means is fitted separately in each rotation, so neighborhood indices are only comparable
    # within a rotation. Align every rotation's clusters to the first rotation's by Hungarian
    # matching of centroids before pooling per-neighborhood metrics across patients.
    P, ref, costs = {}, None, []
    for fd in sorted(args.run.glob('fold*')):
        f = fd / 'neighborhoods.json'
        if not f.exists():
            continue
        z = json.loads(f.read_text())
        C = np.asarray(z['centroids'])
        if ref is None:
            ref, perm = C, np.arange(len(C))
        else:
            cost = ((C[:, None] - ref[None]) ** 2).sum(-1)
            rows, cols = linear_sum_assignment(cost)
            perm = np.empty(len(C), int); perm[cols] = rows  # reference cluster j <- this rotation's cluster perm[j]
            costs.append(float(np.sqrt(cost[rows, cols]).mean()))
        for c, d in z['patients'].items():
            d = dict(d)
            for k in ['truth', 'plugin', 'plugin_q']:
                d[k] = list(np.asarray(d[k])[perm])
            for k in ['samples', 'samples_q']:
                if k in d:
                    d[k] = np.asarray(d[k])[:, perm].tolist()
            P[c] = d
    cases = sorted(P)
    norm = lambda v: np.asarray(v, float) / max(np.sum(v), 1)
    T = np.array([norm(P[c]['truth']) for c in cases])
    est = {'plugin': np.array([norm(P[c]['plugin']) for c in cases]),
           'plugin_q': np.array([norm(P[c]['plugin_q']) for c in cases])}
    S = np.array([[norm(s) for s in P[c]['samples']] for c in cases])  # N, K, CN
    est['ligature'] = S.mean(1)
    if 'samples_q' in P[cases[0]]:
        Sq = np.array([[norm(v) for v in P[c]['samples_q']] for c in cases])
        est['ligature_q'] = Sq.mean(1)
    tv = {m: 0.5 * np.abs(v - T).sum(1) for m, v in est.items()}
    lo, hi = np.quantile(S, [.05, .95], 1)
    out = dict(patients=len(cases), n_cn=int(T.shape[1]),
               mean_tv={m: float(v.mean()) for m, v in tv.items()},
               wilcoxon_tv_ligature_vs_plugin_q=float(wilcoxon(tv['ligature'], tv['plugin_q']).pvalue),
               wilcoxon_tv_ligature_q_vs_plugin_q=(float(wilcoxon(tv['ligature_q'], tv['plugin_q']).pvalue) if 'ligature_q' in tv else None),
               patients_ligature_q_better_than_plugin_q=(int((tv['ligature_q'] < tv['plugin_q']).sum()) if 'ligature_q' in tv else None),
               patients_ligature_better_than_plugin_q=int((tv['ligature'] < tv['plugin_q']).sum()),
               per_cn_mse_vs_plugin={m: float(np.mean(((v - T) ** 2).mean(0) / ((est['plugin'] - T) ** 2).mean(0)))
                                     for m, v in est.items()},
               per_cn_rho={m: float(np.nanmean([spearmanr(T[:, j], v[:, j]).correlation for j in range(T.shape[1])]))
                           for m, v in est.items()},
               coverage90=float(((T >= lo) & (T <= hi)).mean()),
               measured_mean_frequency=T.mean(0).round(4).tolist(),
               centroid_alignment_mean_distance=costs)
    (args.run / 'neighborhood_summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
