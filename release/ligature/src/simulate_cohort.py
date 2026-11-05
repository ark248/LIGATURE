#!/usr/bin/env python
"""Controlled simulations on real ORION geometry (cells, regions, patients, image embeddings).

Targets are synthetic with a known law: Y = f_true + R, where f_true is a fixed function of the
image (the fold-0 cell-graph stain's predictions for every cell, treated as the true conditional
mean, so the plug-in is the Bayes per-cell predictor) and R is a known Gaussian residual:
  R_i = sum_l A_l(x_i) z_l(pos_i) + sigma * eps_i + D eta_g          (case 'correct')
with exact squared-exponential fields at 20 and 80 um (rank 2 each), loadings depending on the
cell embedding, cell noise and a patient latent; this lies in Ligature's model class.
Case 'withincell' adds a shared cell-level factor to two marker pairs (CD8a/Pan-CK and
CD68/PD-L1), a within-cell cross-marker covariance that leaves every bin average unchanged
(Proposition 4(ii)). Case 'misspec' leaves the class: exponential (rough) kernels at 30 and 120 um,
outside the model's scales, Student-t (4 df) scale mixtures for each field and the cell noise, and a
random log-normal amplitude per region (all normalized to unit mean square, so the expected residual
covariance is sum_l A_l A_l^T k_l(d) and f_true is still the conditional mean). Writes cohorts/sim_<case>/ (cell table, symlinked features, f_true.npy and
the true within-cell correlations) for runs with SE_RAW_TARGETS=1 and --mean-kind file.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_engression as se

PAIRS = [('CD8a', 'Pan-CK'), ('CD68', 'PD-L1')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', choices=['correct', 'withincell', 'misspec'], required=True)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--sigma', type=float, default=0.6)
    ap.add_argument('--within', type=float, default=0.6, help='loading of the shared within-cell factor')
    ap.add_argument('--tail-df', type=int, default=4, help='misspec: degrees of freedom of the t scale mixtures')
    ap.add_argument('--amp-sd', type=float, default=0.4, help='misspec: sd of the log region amplitude')
    args = ap.parse_args()
    assert not os.environ.get('SE_RAW_TARGETS') and not os.environ.get('SE_MARKERS')
    rng = np.random.default_rng(args.seed)
    cells, x, y, crops, roles = se.load(ROOT, 0, 271027, 128)
    net = se.CellGraphNet(x.shape[1], d=128, layers=3)
    net.load_state_dict(torch.load(ROOT / 'runs/spatial_engression/fold0/mean.pt'))
    with torch.no_grad():
        pm = se.predict_mean(net, crops, x, y)
    n, P = len(cells), se.P
    f = np.zeros((n, P), np.float32)
    for c in crops:
        f[c['ii']] = pm[c['crop']]
    feat = x[:, :8]  # leading embedding components drive the loadings
    mis = args.case == 'misspec'
    scales, rank = ([30., 120.] if mis else [20., 80.]), 2
    tmix = lambda: np.sqrt((args.tail_df - 2) / rng.chisquare(args.tail_df))  # unit mean-square t scale
    W = rng.normal(size=(len(scales), P, rank, 8)) / np.sqrt(8)
    C = rng.normal(size=(len(scales), P, rank))
    D = 0.15 * rng.normal(size=(P, 2))
    cases = sorted(cells.case.unique())
    eta = {g: rng.normal(size=2) for g in cases}
    R = np.zeros((n, P))
    xy = cells[['gx', 'gy']].to_numpy() * .325
    for c in crops:
        ii = c['ii']
        d2 = ((xy[ii][:, None] - xy[ii][None]) ** 2).sum(-1)
        amp = np.exp(args.amp_sd * rng.normal() - args.amp_sd ** 2) if mis else 1.  # E[amp^2] = 1
        for k, ell in enumerate(scales):
            K = (np.exp(-np.sqrt(d2) / ell) if mis else np.exp(-d2 / (2 * ell ** 2))) + 1e-6 * np.eye(len(ii))
            Lc = np.linalg.cholesky(K)
            z = Lc @ rng.normal(size=(len(ii), rank))                       # rank fields at the cells
            if mis:
                z = z * tmix() * amp
            A = 0.35 * np.tanh(np.einsum('prk,nk->npr', W[k], feat[ii])) + 0.15 * C[k][None]
            R[ii] += np.einsum('npr,nr->np', A, z)
        R[ii] += D @ eta[c['case']]
    if mis:  # Student-t cell noise with unit variance per coordinate
        R += args.sigma * rng.normal(size=(n, P)) * np.sqrt((args.tail_df - 2) / rng.chisquare(args.tail_df, size=(n, P)))
    else:
        R += args.sigma * rng.normal(size=(n, P))
    true_corr = {}
    if args.case == 'withincell':
        for a, b in PAIRS:
            u = rng.normal(size=n)
            ia, ib = se.MARKERS.index(a), se.MARKERS.index(b)
            R[:, ia] += args.within * u; R[:, ib] += args.within * u
    # true within-cell residual correlation of the paired markers (empirical, all cells)
    for a, b in PAIRS:
        ia, ib = se.MARKERS.index(a), se.MARKERS.index(b)
        true_corr[f'{a}|{b}'] = float(np.corrcoef(R[:, ia], R[:, ib])[0, 1])
    Y = (f + R).astype(np.float32)
    out = ROOT / f'cohorts/sim_{args.case}'
    (out / 'data/cells').mkdir(parents=True, exist_ok=True)
    (out / 'data/features').mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(ROOT / 'data/cells/cells.parquet').set_index('cell_idx')
    idx = np.load(ROOT / 'data/features/cell_idx.npy')
    sim = base.loc[idx].copy()
    for j, m in enumerate(se.MARKERS):
        sim['y_' + m] = Y[:, j]
    sim.reset_index().to_parquet(out / 'data/cells/cells.parquet', index=False)
    for fn in ['cell_idx.npy', 'X_p64.npy']:
        dst = out / 'data/features' / fn
        if not dst.exists():
            os.symlink(ROOT / 'data/features' / fn, dst)
    np.save(out / 'f_true.npy', f)
    (out / 'truth.json').write_text(json.dumps(dict(case=args.case, seed=args.seed, scales=scales, rank=rank,
                                                    kernel='exponential' if mis else 'se',
                                                    tail_df=args.tail_df if mis else 0, amp_sd=args.amp_sd if mis else 0.,
                                                    sigma=args.sigma, within=args.within if args.case == 'withincell' else 0.,
                                                    within_cell_residual_corr=true_corr), indent=1))
    print(args.case, 'cells', n, 'within-cell corr', true_corr)


if __name__ == '__main__':
    main()
