#!/usr/bin/env python
"""Statistic families that were never part of any training loss.

For each fold's trained mean/joint models, compute on test crops:
  frac   : fraction of marker-positive cells (16)            -- nonlinear marginal
  copos  : near-minus-far co-positivity of marker pairs (136)  -- nonlinear spatial
  fine   : 0-12.5 um minus 12.5-25 um symmetric products (136)  -- unseen radii
Positivity uses each marker's 85th percentile among training-patient cells.
Plug-in thresholds the mean prediction at the same cut-off; the quantile-matched
plug-in thresholds it at the 85th percentile of validation-patient predictions.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist


def module_from(path):
    spec = importlib.util.spec_from_file_location('se_frozen', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fine_adjacency(cells, c):
    xy = cells[['gx', 'gy']].to_numpy()[c['ii']] * .325
    d = cdist(xy, xy)
    A = np.stack([(d > 0) & (d <= 12.5), (d > 12.5) & (d <= 25.)]).astype(np.float32)
    for k in range(2):
        np.fill_diagonal(A[k], 0)
    return torch.from_numpy(A), torch.from_numpy(A.sum((1, 2)))


def statistics(Y, A, NP, fineA, fineNP, thr, iu):
    """Y: [K, n, P] torch. Returns dict of [K, D] arrays."""
    B = (Y > thr).float()
    frac = B.mean(1)
    def binned(Z, A, NP):
        AZ = torch.einsum('kij,mjp->mkip', A, Z)
        G = torch.einsum('mip,mkiq->mkpq', Z, AZ) / NP.clamp(min=1)[None, :, None, None]
        G = .5 * (G + G.transpose(-1, -2))
        return G[:, :, iu[0], iu[1]]
    cp = binned(B, A[[0, 2]], NP[[0, 2]])
    fn = binned(Y, fineA, fineNP)
    return dict(frac=frac.numpy(), copos=(cp[:, 0] - cp[:, 1]).numpy(), fine=(fn[:, 0] - fn[:, 1]).numpy(),
                valid_copos=bool(NP[0] > 0 and NP[2] > 0), valid_fine=bool(fineNP[0] > 0 and fineNP[1] > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/spatial_engression'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--q', type=float, default=.85)
    args = ap.parse_args()
    torch.set_num_threads(4)
    se = module_from(args.run / 'src' / 'spatial_engression.py')
    fd = args.run / f'fold{args.fold}'
    a = json.loads((fd / 'args.json').read_text())
    cells, x, y, crops, roles = se.load(args.root, args.fold, int(a['seed']), int(a['pcadim']))
    d, L = int(a['d']), int(a['layers'])
    cn, gn = int(a['cell_noise']), int(a['group_noise'])
    late = a.get('late_noise', 'False') == 'True'
    mean = se.CellGraphNet(x.shape[1], d=d, layers=L)
    kw = dict(late=late) if 'late' in se.CellGraphNet.__init__.__code__.co_varnames else {}
    joint = se.CellGraphNet(x.shape[1], d=d, layers=L, cell_noise=cn, group_noise=gn, **kw)
    mean.load_state_dict(torch.load(fd / 'mean.pt'))
    joint.load_state_dict(torch.load(fd / 'joint.pt'))
    mean.eval(); joint.eval()
    train = np.flatnonzero(cells.case.map(roles).to_numpy() == 'train')
    thr = torch.from_numpy(np.quantile(y[train], args.q, axis=0).astype(np.float32))
    va = [c for c in crops if c['role'] == 'val']
    te = [c for c in crops if c['role'] == 'test']
    with torch.no_grad():
        fv = se.predict_mean(mean, va, x, y)
    thr_q = torch.from_numpy(np.quantile(np.concatenate([fv[c['crop']] for c in va]), args.q, axis=0).astype(np.float32))
    iu = se.IU
    gen = torch.Generator().manual_seed(4242)
    eta = {c: torch.randn(args.samples, gn, generator=gen) for c in sorted({c['case'] for c in te})}
    rows = []
    with torch.no_grad():
        fm = se.predict_mean(mean, te, x, y)
        for c in te:
            X, Y, D, A, M, NP = se.collate([c], x, y)
            A, NP = A[0], NP[0]
            fA, fNP = fine_adjacency(cells, c)
            n = X.shape[1]
            eps = torch.randn(args.samples, n, cn, generator=gen)
            smp = joint(X.expand(args.samples, *X.shape[1:]), D, M, eps, eta[c['case']])
            st_t = statistics(Y, A, NP, fA, fNP, thr, iu)
            f = torch.from_numpy(fm[c['crop']])[None]
            st_p = statistics(f, A, NP, fA, fNP, thr, iu)
            st_q = statistics(f, A, NP, fA, fNP, thr_q, iu)
            st_j = statistics(smp, A, NP, fA, fNP, thr, iu)
            rows.append(dict(crop=c['crop'], case=c['case'], ncell=len(c['ii']),
                             valid_copos=st_t['valid_copos'], valid_fine=st_t['valid_fine'],
                             **{f'{fam}_{k}': v[fam] for k, v in
                                [('truth', st_t), ('plugin', st_p), ('plugin_q', st_q)] for fam in ['frac', 'copos', 'fine']},
                             **{f'{fam}_samples': st_j[fam] for fam in ['frac', 'copos', 'fine']}))
    out = {k: np.stack([r[k] for r in rows]) for k in rows[0] if k not in ('crop', 'case')}
    np.savez_compressed(fd / 'unseen_stats.npz', crop=np.array([r['crop'] for r in rows]),
                        case=np.array([r['case'] for r in rows]), **out)
    print('fold', args.fold, 'crops', len(rows))


if __name__ == '__main__':
    main()
