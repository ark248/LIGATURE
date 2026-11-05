#!/usr/bin/env python
"""Sharp identification of spatial clustering statistics from per-cell predictive laws.

If each cell's marker a has a Gaussian predictive law N(mu_i, s_i^2) and nothing
else is specified, every correlation matrix R (unit diagonal, PSD) is attainable by
a Gaussian joint law with those marginals, and no joint law attains anything else.
So the set of conditional means of Q = sum_{i!=j} w_ij Y_i Y_j consistent with the
per-cell laws is the interval

    Q(mu) + [ min_R <C, R>, max_R <C, R> ],  C_ij = w_ij s_i s_j,  R in elliptope,

two max-cut-type semidefinite programs. We solve them by Riemannian ascent on the
oblique manifold (R = V V^T, unit rows, rank k > sqrt(2n)), certify the maximum with
a dual bound sum(y) + n*max(0, -lambda_min(diag(y) - C)), and round V with random
hyperplanes into explicit rank-one Gaussian witnesses Y_i = mu_i + s_i sigma_i Z.

Per-cell laws are Ligature's calibrated marginals (closed form), i.e. a per-cell
probabilistic virtual stain that is as well calibrated per cell as Ligature itself.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def module_from(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def solve(C, sign, k, iters=600, seed=0):
    """Maximize sign*<C, VV^T> over unit-row V. C: [B, n, n] (zero diagonal). Returns value, V."""
    rng = np.random.default_rng(seed)
    B, n, _ = C.shape
    V = rng.normal(size=(B, n, k))
    V /= np.linalg.norm(V, axis=-1, keepdims=True)
    Cs = sign * C
    step = 1.0 / (np.abs(Cs).sum(-1).max(-1) + 1e-12)  # 1 / max row l1 norm >= 1/||C||
    prev = None
    for it in range(iters):
        G = Cs @ V
        V = V + step[:, None, None] * G
        V /= np.linalg.norm(V, axis=-1, keepdims=True)
        if it % 25 == 0:
            val = np.einsum('bij,bik,bjk->b', Cs, V, V)
            if prev is not None and np.all(np.abs(val - prev) <= 1e-7 * (np.abs(val) + 1e-12)):
                break
            prev = val
    return sign * np.einsum('bij,bik,bjk->b', Cs, V, V), V


def dual_bound(C, V, sign):
    """Certified upper bound on max sign*<C,R> over the elliptope."""
    Cs = sign * C
    y = np.einsum('bij,bik,bjk->bi', Cs, V, V)
    out = []
    for b in range(len(C)):
        lam = np.linalg.eigvalsh(np.diag(y[b]) - Cs[b]).min()
        out.append(y[b].sum() + len(y[b]) * max(0., -lam))
    return sign * np.array(out)


def rounding(C, V, sign, trials, rng):
    B, n, k = V.shape
    best = np.full(B, -np.inf)
    for _ in range(trials):
        g = rng.normal(size=(B, k, 1))
        s = np.sign(V @ g)[..., 0]
        s[s == 0] = 1
        best = np.maximum(best, sign * np.einsum('bi,bij,bj->b', s, C, s))
    return sign * best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--per-patient', type=int, default=10)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    snap = args.run / 'src' / 'ligature.py'  # per-run frozen snapshot if present
    t = module_from(snap if snap.exists() else Path(__file__).resolve().parent / 'ligature.py', 'ligature_frozen')
    se = t.se
    fd = args.run / f'fold{args.fold}'
    a = json.loads((fd / 'args.json').read_text())
    cells, x, y, crops, roles = se.load(args.root, args.fold, int(a['seed']), int(a['pcadim']))
    rng = np.random.default_rng(args.fold)
    te = [c for c in crops if c['role'] == 'test' and c['eligible']]
    pick = []
    for case in sorted({c['case'] for c in te}):
        cs = [c for c in te if c['case'] == case]
        pick += [cs[i] for i in sorted(rng.choice(len(cs), min(args.per_patient, len(cs)), replace=False))]
    mean_net = se.CellGraphNet(x.shape[1], d=int(a['d']), layers=int(a['layers']))
    mean_net.load_state_dict(torch.load(Path(a['mean_from']) / f'fold{args.fold}' / 'mean.pt'))
    scales = tuple(float(v) for v in a['scales'].split(',')) if a['scales'] else ()
    net = t.Ligature(x.shape[1], d=int(a['d']), layers=int(a['layers']), frozen_mean=True,
                    scales=scales, patient_dim=int(a['patient_dim']))
    net.load_state_dict(torch.load(fd / 'ligature.pt'))
    mean_net.eval(); net.eval()
    rows = []
    for c in pick:
        X, Y, D, A, M, NP = se.collate([c], x, y)
        with torch.no_grad():
            f = mean_net(X, D, M)
            _, L = net.encode(X, D, M, f)
        sd = torch.sqrt((L[0] ** 2).sum(-1)).numpy().astype(np.float64)  # n, P
        f = f[0].numpy().astype(np.float64); Yc = Y[0].numpy().astype(np.float64)
        A = c['A'].astype(np.float64); npairs = c['npairs']
        W = A[0] / npairs[0] - A[2] / npairs[2]
        n = W.shape[0]
        C = np.stack([W * np.outer(sd[:, p], sd[:, p]) for p in range(se.P)])
        k = int(np.ceil(np.sqrt(2 * n))) + 2
        hi, Vh = solve(C, 1., k)
        lo, Vl = solve(C, -1., k, seed=1)
        hi_cert = dual_bound(C, Vh, 1.)
        lo_cert = dual_bound(C, Vl, -1.)
        hi_w = rounding(C, Vh, 1., 64, rng)
        lo_w = rounding(C, Vl, -1., 64, rng)
        q = lambda Z: np.array([Z[:, p] @ W @ Z[:, p] for p in range(se.P)])
        # restricted class: stationary, nonnegative, decaying correlation
        # rho(d) = sum_k pi_k exp(-d / l_k) (+ nugget); extremes are single kernels or the nugget
        dist = c['dist'].astype(np.float64)
        dec = {l: np.array([np.sum(C[p] * np.exp(-dist / l)) for p in range(se.P)]) for l in (5, 10, 20, 40, 80, 160)}
        plug, truth = q(f), q(Yc)
        for p in range(se.P):
            rows.append(dict(fold=args.fold, case=c['case'], crop=c['crop'], marker=se.MARKERS[p], ncell=n,
                             plugin=plug[p], truth=truth[p], sdp_lo=plug[p] + lo[p], sdp_hi=plug[p] + hi[p],
                             cert_lo=plug[p] + lo_cert[p], cert_hi=plug[p] + hi_cert[p],
                             witness_lo=plug[p] + lo_w[p], witness_hi=plug[p] + hi_w[p],
                             **{f'dec_{l}': dec[l][p] for l in dec}))
    df = pd.DataFrame(rows)
    df.to_csv(fd / 'identification_sdp.csv', index=False)
    print('fold', args.fold, 'regions', len(pick), 'max relative certificate gap',
          float(((df.cert_hi - df.sdp_hi) / (df.sdp_hi - df.plugin)).max()))


if __name__ == '__main__':
    main()
