#!/usr/bin/env python
"""Direct crop-level regression baseline for spatial statistics.

Per fold, ridge regression from label-free crop summaries (cell-feature means,
standard deviations, log cell count, and near/mid/far second-order products of
the first 16 feature components) to each crop's measured statistic vector.
Training crops only; the penalty is chosen on validation patients.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_engression import load, crop_stats, P


def summaries(crops, x, y, r=16):
    F, S = [], []
    for c in crops:
        xc = torch.from_numpy(x[c['ii']])
        A = torch.from_numpy(c['A'])
        npairs = torch.from_numpy(c['npairs']).clamp(min=1)
        z = xc[:, :r]
        G = torch.einsum('kij,ip,jq->kpq', A, z, z) / npairs[:, None, None]
        iu = torch.triu_indices(r, r)
        F.append(torch.cat([xc.mean(0), xc.std(0) if len(xc) > 1 else torch.zeros(x.shape[1]),
                            torch.tensor([np.log(len(c['ii']))], dtype=torch.float32),
                            G[:, iu[0], iu[1]].reshape(-1)]).numpy())
        S.append(crop_stats(torch.from_numpy(y[c['ii']])[None], A[None], torch.from_numpy(c['npairs'])[None],
                            torch.ones(1, len(c['ii'])))[0].numpy())
    return np.array(F), np.array(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/spatial_engression'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=271027)
    args = ap.parse_args()
    torch.set_num_threads(4)
    cells, x, y, crops, roles = load(args.root, args.fold, args.seed, 128)
    part = {k: [c for c in crops if c['role'] == k] for k in ['train', 'val', 'test']}
    data = {k: summaries(v, x, y) for k, v in part.items()}
    Ftr, Str = data['train']
    fm, fs = Ftr.mean(0), Ftr.std(0) + 1e-6
    sm = Str.mean(0)
    Z = lambda F: (F - fm) / fs
    best = None
    for lam in [10, 100, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]:
        A = Z(Ftr)
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Str - sm))
        Fv, Sv = data['val']
        err = ((Z(Fv) @ W + sm - Sv) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, lam, W)
    _, lam, W = best
    # refit on train + validation crops with the chosen penalty
    Fa = np.concatenate([Ftr, data['val'][0]]); Sa = np.concatenate([Str, data['val'][1]])
    fm, fs = Fa.mean(0), Fa.std(0) + 1e-6; sm = Sa.mean(0)
    A = (Fa - fm) / fs
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Sa - sm))
    out = {}
    for k in ['val', 'test']:
        out[k] = ((data[k][0] - fm) / fs) @ W + sm
    fd = args.run / f'fold{args.fold}'
    fd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fd / 'direct.npz', lam=lam,
                        val_crop=np.array([c['crop'] for c in part['val']]), val=out['val'],
                        test_crop=np.array([c['crop'] for c in part['test']]), test=out['test'])
    print('fold', args.fold, 'lambda', lam, 'val mse', best[0])


if __name__ == '__main__':
    main()
