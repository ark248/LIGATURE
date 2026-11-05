#!/usr/bin/env python
"""Hybrid inference: condition Ligature on a few measured anchor markers.

Ligature's residual is linear in Gaussian latents theta (random-Fourier-feature field
weights, cell noise, region and patient latents): e = G theta. With features fixed
per region, conditioning on measured anchor residuals e_O = Y_O - f_O is exact
Gaussian conditioning, sampled with Matheron's rule:
    theta | e_O = theta + G_O^T (G_O G_O^T + delta I)^{-1} (e_O - G_O theta),  theta ~ N(0, I).
The patient latent is conditioned within each region (an approximation that ignores
its sharing across regions once anchors are observed).

Methods per region and anchor set O (statistics are compared only where they involve
at least one unmeasured marker):
  hybrid_plugin : anchors measured, other markers from the frozen virtual stain
  ridge_impute  : anchors measured, others from ridge on [image features, anchors]
                  fitted on training-patient cells (conditional-mean imputation)
  ligature_uncond: anchors measured, others sampled from unconditional Ligature
  ligature_cond  : anchors measured, others sampled from Ligature conditioned on anchors
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ANCHORS = {'PanCK_CD45': ['Pan-CK', 'CD45'], 'PDL1_CD8a': ['PD-L1', 'CD8a'], 'CD3e': ['CD3e']}


def module_from(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def design(net, L, pos, gen):
    """G: [n*P, dim] with row (i, a); L: [n, P, K]; pos: [n, 2] (um)."""
    n, P, K = L.shape
    blocks = []
    col = 0
    r, M = net.rank, net.nfeat
    for li, ell in enumerate(net.scales):
        om = torch.randn(M, 2, generator=gen, dtype=torch.float64) / ell
        ph = torch.rand(M, generator=gen, dtype=torch.float64) * 2 * math.pi
        phi = math.sqrt(2 / M) * torch.cos(pos @ om.T + ph)  # n, M
        A = L[:, :, li * r:(li + 1) * r]  # n, P, r
        blocks.append(torch.einsum('nm,npr->npmr', phi, A).reshape(n * P, M * r))
        col += r
    cd, rd, pd = net.dims['cell'], net.dims['region'], net.dims['patient']
    Bm = L[:, :, col:col + cd]
    Gc = torch.zeros(n, P, n, cd, dtype=torch.float64)
    Gc[torch.arange(n), :, torch.arange(n), :] = Bm
    blocks.append(Gc.reshape(n * P, n * cd))
    blocks.append(L[:, :, col + cd:col + cd + rd].reshape(n * P, rd))
    blocks.append(L[:, :, col + cd + rd:col + cd + rd + pd].reshape(n * P, pd))
    return torch.cat(blocks, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--max-regions', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--split', default='test', choices=['test', 'val'])
    ap.add_argument('--deltas', default='', help='comma list: regularized conditioning, near-far contrasts only')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    snap = args.run / 'src' / 'ligature.py'
    t = module_from(snap if snap.exists() else Path(__file__).resolve().parent / 'ligature.py', 'ligature_frozen')
    se = t.se
    fd = args.run / f'fold{args.fold}'
    a = json.loads((fd / 'args.json').read_text())
    cells, x, y, crops, roles = se.load(args.root, args.fold, int(a['seed']), int(a['pcadim']))
    mean_net = se.CellGraphNet(x.shape[1], d=int(a['d']), layers=int(a['layers']))
    mean_net.load_state_dict(torch.load(Path(a['mean_from']) / f'fold{args.fold}' / 'mean.pt'))
    scales = tuple(float(v) for v in a['scales'].split(',')) if a['scales'] else ()
    net = t.Ligature(x.shape[1], d=int(a['d']), layers=int(a['layers']), frozen_mean=True,
                    scales=scales, patient_dim=int(a['patient_dim']))
    net.load_state_dict(torch.load(fd / 'ligature.pt'))
    mean_net.eval(); net.eval()
    role = cells.case.map(roles).to_numpy()
    tr_idx = np.flatnonzero(role == 'train')
    rng = np.random.default_rng(args.fold)
    fit = rng.choice(tr_idx, min(80000, len(tr_idx)), replace=False)
    te = [c for c in crops if c['role'] == args.split]
    deltas = [float(v) for v in args.deltas.split(',')] if args.deltas else []
    k = len(se.IU[0])
    qv = lambda Sx: Sx[..., se.P:se.P + k] - Sx[..., se.P + 2 * k:se.P + 3 * k]
    if args.max_regions:
        te = te[:args.max_regions]
    out = {}
    for name, markers in ANCHORS.items():
        O = [se.MARKERS.index(m) for m in markers]
        U = [p for p in range(se.P) if p not in O]
        Z = np.column_stack([x[fit], y[fit][:, O]]).astype(np.float64)
        zm = Z.mean(0); Yu = y[fit][:, U].astype(np.float64); ymu = Yu.mean(0)
        W = np.linalg.solve((Z - zm).T @ (Z - zm) + 1e2 * np.eye(Z.shape[1]), (Z - zm).T @ (Yu - ymu))
        out[name] = dict(O=O, U=U, W=W, zm=zm, ymu=ymu, rows=[])
    gen = torch.Generator().manual_seed(1234 + args.fold)
    for c in te:
        X, Y, D, A, M, NP = se.collate([c], x, y)
        with torch.no_grad():
            f = mean_net(X, D, M)
            _, L = net.encode(X, D, M, f)
        L = L[0].double(); f0 = f[0].double(); Y0 = Y[0].double()
        n = L.shape[0]
        xy = cells[['gx', 'gy']].to_numpy()[c['ii']] * .325
        pos = torch.from_numpy(xy - xy.mean(0)).double()
        G = design(net, L, pos, gen)  # n*P, dim
        theta = torch.randn(args.samples, G.shape[1], generator=gen, dtype=torch.float64)
        e_prior = (theta @ G.T).reshape(args.samples, n, se.P)
        Mf = M.float()
        S = lambda Ymap: se.crop_stats(Ymap.float()[None] if Ymap.dim() == 2 else Ymap.float()[:, None],
                                       A, NP, Mf)
        truth = S(Y0)[0].numpy()
        for name, d in out.items():
            O, U = d['O'], d['U']
            hyb = f0.clone(); hyb[:, O] = Y0[:, O]
            zx = np.column_stack([x[c['ii']], y[c['ii']][:, O]]).astype(np.float64)
            rid = hyb.clone(); rid[:, U] = torch.from_numpy((zx - d['zm']) @ d['W'] + d['ymu'])
            unc = f0[None] + e_prior; unc[:, :, O] = Y0[None, :, O]
            rows = (torch.arange(n)[:, None] * se.P + torch.tensor(O)[None]).reshape(-1)
            GO = G[rows]
            eO = (Y0[:, O] - f0[:, O]).reshape(-1)
            if deltas:
                row = dict(crop=c['crop'], case=c['case'], eligible=c['eligible'], q_truth=qv(truth),
                           q_hybrid_plugin=qv(S(hyb)[0].numpy()), q_ridge_impute=qv(S(rid)[0].numpy()),
                           q_ligature_uncond=qv(S(unc)[:, 0].numpy()).astype(np.float32))
                GG = GO @ GO.T
                resid = eO[None] - theta @ GO.T  # K, nO
                for dl in deltas:
                    # anchor residual = model residual + independent noise of variance delta
                    Lc = torch.linalg.cholesky(GG + dl * torch.eye(len(rows), dtype=torch.float64))
                    eps_o = torch.randn(resid.shape, generator=gen, dtype=torch.float64) * math.sqrt(dl)
                    sol = torch.cholesky_solve((resid - eps_o).T, Lc).T
                    con = f0[None] + ((theta + sol @ GO) @ G.T).reshape(args.samples, n, se.P)
                    con[:, :, O] = Y0[None, :, O]
                    row[f'q_cond_{dl:g}'] = qv(S(con)[:, 0].numpy()).astype(np.float32)
                d['rows'].append(row)
                continue
            Kmat = GO @ GO.T + 1e-4 * torch.eye(len(rows), dtype=torch.float64)
            Lc = torch.linalg.cholesky(Kmat)
            resid = eO[None] - theta @ GO.T  # K, nO
            sol = torch.cholesky_solve(resid.T, Lc).T  # K, nO
            th_post = theta + sol @ GO
            con = f0[None] + (th_post @ G.T).reshape(args.samples, n, se.P)
            con[:, :, O] = Y0[None, :, O]
            d['rows'].append(dict(crop=c['crop'], case=c['case'], eligible=c['eligible'], truth=truth,
                                  hybrid_plugin=S(hyb)[0].numpy(), ridge_impute=S(rid)[0].numpy(),
                                  ligature_uncond=S(unc)[:, 0].numpy().mean(0), ligature_cond=S(con)[:, 0].numpy(),
                                  ))
    for name, d in out.items():
        R = d['rows']
        if deltas:
            np.savez_compressed(fd / f'hybridreg_{name}_{args.split}.npz', observed=np.array(d['O']),
                                **{key: np.array([r[key] for r in R]) for key in R[0]})
            continue
        np.savez_compressed(fd / f'hybrid_{name}{args.tag}.npz', crop=np.array([r['crop'] for r in R]),
                            case=np.array([r['case'] for r in R]), eligible=np.array([r['eligible'] for r in R]),
                            truth=np.stack([r['truth'] for r in R]), hybrid_plugin=np.stack([r['hybrid_plugin'] for r in R]),
                            ridge_impute=np.stack([r['ridge_impute'] for r in R]),
                            ligature_uncond=np.stack([r['ligature_uncond'] for r in R]),
                            ligature_cond=np.stack([r['ligature_cond'] for r in R]).astype(np.float32),
                            observed=np.array(d['O']))
    print('fold', args.fold, 'regions', len(te))


if __name__ == '__main__':
    main()
