#!/usr/bin/env python
"""Cellular-neighborhood (CN) analysis from trained Ligature models (never scored in training).

Cells are gated into 8 types by marker positivity (training-patient 85th percentile)
with a fixed priority order. Each cell's neighborhood is the type composition of its
10 nearest cells in the region (including itself). k-means (10 CNs) is fitted on
measured compositions of training-patient cells only. For test patients we compare
per-patient CN frequencies computed from the measured assay, the plug-in thresholded at
the same cut-offs, the quantile-matched plug-in, and Ligature joint samples.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

TYPES = [('tumor', ['Pan-CK', 'E-cadherin']), ('CD8 T', ['CD8a']), ('Treg', ['FOXP3']), ('T', ['CD3e', 'CD4']),
         ('B', ['CD20']), ('macrophage', ['CD68', 'CD163']), ('endothelial', ['CD31']), ('stroma', ['SMA'])]


def module_from(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def cell_types(Y, thr, markers):
    """Y: [..., n, P] array -> int types [..., n]; 0..7 by priority, 8 = other."""
    pos = Y > thr
    t = np.full(Y.shape[:-1], len(TYPES), dtype=np.int64)
    for k in reversed(range(len(TYPES))):
        idx = [markers.index(m) for m in TYPES[k][1]]
        t = np.where(pos[..., idx].any(-1), k, t)
    return t


def compositions(t, nbr):
    """t: [..., n] types; nbr: [n, k] neighbor indices -> [..., n, T+1] fractions."""
    oh = np.eye(len(TYPES) + 1)[t]
    return oh[..., nbr, :].mean(-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--ncn', type=int, default=10)
    ap.add_argument('--q', type=float, default=.85)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    snap = args.run / 'src' / 'ligature.py'
    t = module_from(snap if snap.exists() else Path(__file__).resolve().parent / 'ligature.py', 'ligature_frozen')
    se = t.se
    fd = args.run / f'fold{args.fold}'
    a = json.loads((fd / 'args.json').read_text())
    cells, x, y, crops, roles = se.load(args.root, args.fold, int(a['seed']), int(a['pcadim']))
    kind = a.get('mean_kind', 'transformer')
    if kind == 'file':
        F = np.load(a['mean_file'].format(fold=args.fold)).astype(np.float32)
        x = np.column_stack([x, F])
        mean_net = t.ColumnMean(x.shape[1] - se.P)
    elif kind == 'ridge':
        mean_net = t.fit_ridge_mean(x, y, [c for c in crops if c['role'] == 'train'], [c for c in crops if c['role'] == 'val'])
    else:
        mean_net = se.CellGraphNet(x.shape[1], d=int(a['d']), layers=int(a['layers']))
        mean_net.load_state_dict(torch.load(Path(a['mean_from']) / f'fold{args.fold}' / 'mean.pt'))
    scales = tuple(float(v) for v in a['scales'].split(',')) if a['scales'] else ()
    kw = dict(tail_df=int(a['tail_df'])) if 'tail_df' in a else {}
    if a.get('mean_correction') == 'True':
        kw['mean_correction'] = True
    net = t.Ligature(x.shape[1], d=int(a['d']), layers=int(a['layers']), frozen_mean=True, scales=scales,
                    patient_dim=int(a['patient_dim']), **kw)
    net.load_state_dict(torch.load(fd / 'ligature.pt'))
    mean_net.eval(); net.eval()
    M = se.MARKERS
    role = cells.case.map(roles).to_numpy()
    thr = np.quantile(y[role == 'train'], args.q, axis=0)
    xy = cells[['gx', 'gy']].to_numpy() * .325

    def nbrs(c):
        p = xy[c['ii']]
        return cKDTree(p).query(p, k=min(args.k, len(p)))[1].reshape(len(p), -1)

    tr = [c for c in crops if c['role'] == 'train' and len(c['ii']) >= args.k]
    rng = np.random.default_rng(args.fold)
    comp_tr = np.concatenate([compositions(cell_types(y[c['ii']], thr, M), nbrs(c)) for c in tr])
    km = KMeans(args.ncn, n_init=4, random_state=0).fit(comp_tr[rng.choice(len(comp_tr), min(100000, len(comp_tr)), replace=False)])
    va = [c for c in crops if c['role'] == 'val']
    te = [c for c in crops if c['role'] == 'test' and len(c['ii']) >= args.k]
    with torch.no_grad():
        fv = se.predict_mean(mean_net, va, x, y)
        fm = se.predict_mean(mean_net, te, x, y)
    thr_q = np.quantile(np.concatenate([fv[c['crop']] for c in va]), args.q, axis=0)
    gen = torch.Generator().manual_seed(77 + args.fold)
    # quantile-matched thresholds for Ligature samples, from validation patients only
    vsmp = []
    with torch.no_grad():
        for c in [va[i] for i in rng.choice(len(va), min(120, len(va)), replace=False)]:
            Xv, Yv, Dv, Av, Mv, NPv = se.collate([c], x, y)
            mu, L = net.encode(Xv, Dv, Mv, torch.from_numpy(fv[c['crop']])[None])
            z = net.noise(8, t.positions([c], cells), Mv, torch.zeros(1, dtype=torch.long), 1, gen=gen)
            vsmp.append(net.sample(mu, L, z)[:, 0].reshape(-1, se.P).numpy())
    thr_s = np.quantile(np.concatenate(vsmp), args.q, axis=0)
    cases = sorted({c['case'] for c in te})
    eta = {cc: net.latent((args.samples, 1, net.dims['patient']), gen) if hasattr(net, 'latent')
           else torch.randn(args.samples, 1, net.dims['patient'], generator=gen) for cc in cases}
    counts = {cc: dict(truth=np.zeros(args.ncn), plugin=np.zeros(args.ncn), plugin_q=np.zeros(args.ncn),
                       samples=np.zeros((args.samples, args.ncn)), samples_q=np.zeros((args.samples, args.ncn)),
                       cells=0) for cc in cases}
    for c in te:
        nb = nbrs(c)
        X, Y, D, A, Mk, NP = se.collate([c], x, y)
        f = fm[c['crop']]
        with torch.no_grad():
            mu, L = net.encode(X, D, Mk, torch.from_numpy(f)[None])
            z = net.noise(args.samples, t.positions([c], cells), Mk, torch.zeros(1, dtype=torch.long), 1, gen=gen,
                          eta=eta[c['case']])
            smp = net.sample(mu, L, z)[:, 0].numpy()
        k_ = counts[c['case']]
        lab = lambda Yv, th: km.predict(compositions(cell_types(Yv, th, M), nb).reshape(-1, len(TYPES) + 1))
        k_['truth'] += np.bincount(lab(y[c['ii']], thr), minlength=args.ncn)
        k_['plugin'] += np.bincount(lab(f, thr), minlength=args.ncn)
        k_['plugin_q'] += np.bincount(lab(f, thr_q), minlength=args.ncn)
        comp = compositions(cell_types(smp, thr, M), nb)  # S, n, T+1
        labs = km.predict(comp.reshape(-1, len(TYPES) + 1)).reshape(args.samples, -1)
        k_['samples'] += np.stack([np.bincount(l, minlength=args.ncn) for l in labs])
        comp_q = compositions(cell_types(smp, thr_s, M), nb)
        labs_q = km.predict(comp_q.reshape(-1, len(TYPES) + 1)).reshape(args.samples, -1)
        k_['samples_q'] += np.stack([np.bincount(l, minlength=args.ncn) for l in labs_q])
        k_['cells'] += len(c['ii'])
    out = {cc: {kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk, v in d.items()} for cc, d in counts.items()}
    (fd / 'neighborhoods.json').write_text(json.dumps(dict(centroids=km.cluster_centers_.tolist(),
                                                          types=[n for n, _ in TYPES] + ['other'], patients=out)))
    print('fold', args.fold, 'test patients', len(cases))


if __name__ == '__main__':
    main()
