#!/usr/bin/env python
"""Gated contact statistics (never scored in training) from trained Ligature models.

For ordered marker pairs (a, b): the fraction of a-positive cells with at least one
b-positive cell within 20 um (e.g. CD8a+ T cells touching Pan-CK+ tumor cells).
Positivity: value above the training-patient 85th percentile of the marker. Per region
we store numerator and denominator; patient values pool counts across regions.
Methods: measured truth, plug-in thresholded at the same cut-off, quantile-matched
plug-in (cut-off = 85th percentile of validation-patient predictions), Ligature samples.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def module_from(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def contact(Y, thr, adj):
    """Y: [K, n, P]; adj: [n, n] bool (no self). Returns numerators [K,P,P], denominators [K,P]."""
    B = (Y > thr).float()
    has = (torch.einsum('ij,kjp->kip', adj, B) > 0).float()
    num = torch.einsum('kia,kib->kab', B, has)
    return num.numpy(), B.sum(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--radius', type=float, default=20.)
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
    net = t.Ligature(x.shape[1], d=int(a['d']), layers=int(a['layers']), frozen_mean=True,
                    scales=scales, patient_dim=int(a['patient_dim']), **kw)
    net.load_state_dict(torch.load(fd / 'ligature.pt'))
    mean_net.eval(); net.eval()
    role = cells.case.map(roles).to_numpy()
    thr = torch.from_numpy(np.quantile(y[role == 'train'], args.q, axis=0).astype(np.float32))
    va = [c for c in crops if c['role'] == 'val']
    te = [c for c in crops if c['role'] == 'test']
    with torch.no_grad():
        fv = se.predict_mean(mean_net, va, x, y)
        thr_q = torch.from_numpy(np.quantile(np.concatenate([fv[c['crop']] for c in va]), args.q, axis=0).astype(np.float32))
        fm = se.predict_mean(mean_net, te, x, y)
    gen = torch.Generator().manual_seed(4243 + args.fold)
    # quantile-matched thresholds for Ligature samples, from validation patients only
    rngv = np.random.default_rng(args.fold)
    vsmp = []
    with torch.no_grad():
        for c in [va[i] for i in rngv.choice(len(va), min(120, len(va)), replace=False)]:
            Xv, Yv, Dv, Av, Mv, NPv = se.collate([c], x, y)
            mu_v, L_v = net.encode(Xv, Dv, Mv, torch.from_numpy(fv[c['crop']])[None])
            z_v = net.noise(8, t.positions([c], cells), Mv, torch.zeros(1, dtype=torch.long), 1, gen=gen)
            vsmp.append(net.sample(mu_v, L_v, z_v)[:, 0].reshape(-1, se.P))
    thr_s = torch.quantile(torch.cat(vsmp), args.q, dim=0)
    eta = {c: (net.latent((args.samples, 1, net.dims['patient']), gen) if hasattr(net, 'latent')
               else torch.randn(args.samples, 1, net.dims['patient'], generator=gen)) for c in sorted({c['case'] for c in te})}
    out = {k: [] for k in ['truth_num', 'truth_den', 'plugin_num', 'plugin_den', 'plugin_q_num', 'plugin_q_den',
                           'samples_num', 'samples_den', 'samples_q_num', 'samples_q_den']}
    with torch.no_grad():
        for c in te:
            X, Y, D, A, M, NP = se.collate([c], x, y)
            adj = (torch.from_numpy(c['dist']) <= args.radius).float()
            adj.fill_diagonal_(0)
            f = torch.from_numpy(fm[c['crop']])[None]
            mu, L = net.encode(X, D, M, f)
            z = net.noise(args.samples, t.positions([c], cells), M, torch.zeros(1, dtype=torch.long), 1, gen=gen,
                          eta=eta[c['case']])
            smp = net.sample(mu, L, z)[:, 0]
            for key, Yv, th in [('truth', Y, thr), ('plugin', f, thr), ('plugin_q', f, thr_q), ('samples', smp, thr),
                                ('samples_q', smp, thr_s)]:
                n_, d_ = contact(Yv, th, adj)
                out[key + '_num'].append(n_ if key.startswith('samples') else n_[0])
                out[key + '_den'].append(d_ if key.startswith('samples') else d_[0])
    np.savez_compressed(fd / 'contact_stats.npz', crop=np.array([c['crop'] for c in te]),
                        case=np.array([c['case'] for c in te]),
                        **{k: np.stack(v).astype(np.float32) for k, v in out.items()})
    print('fold', args.fold, 'regions', len(te))


if __name__ == '__main__':
    main()
