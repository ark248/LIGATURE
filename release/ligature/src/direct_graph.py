#!/usr/bin/env python
"""Direct-statistic graph baseline: predict region statistics without generating cells.

Same frozen cell embeddings, same cell-graph transformer (width, depth, distance-RBF
attention) and same supervision as Ligature, but the transformer output is pooled over the
region (mean, max, log cell count) and an MLP gives a Gaussian predictive law for every
coordinate of the scored statistic vector S (cell means, binned second moments and their
bin contrasts, standardized as in ligature.py). Trained with Gaussian CRPS on region
statistics plus CRPS on patient averages (mean of region means; variance = mean region
variance / m + a learned per-statistic patient variance), patient-grouped batches, early
stopping on validation patients. Writes fold*/direct_graph.npz (test crops in Ligature's
order) into an existing Ligature run so analyze_spatial_engression.py can score it.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_engression as se
from spatial_engression import P, crop_stats, collate, group_batches, log
from ligature import contrasts, crps_gauss


class DirectGraph(nn.Module):
    def __init__(self, din, D, d=128, layers=3, heads=4, nrbf=16, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.drop = nn.Dropout(dropout)
        self.centers = nn.Parameter(torch.linspace(0, 160, nrbf), requires_grad=False)
        self.blocks = nn.ModuleList([se.Block(d, heads, nrbf) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d + 1, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, 2 * D))
        self.log_tau = nn.Parameter(torch.full((D,), -2.))

    def forward(self, x, dist, mask):
        t = self.drop(self.inp(x))
        rbf = torch.exp(-((dist[..., None] - self.centers) / 10.) ** 2)
        for b in self.blocks:
            t = b(t, rbf, mask, 1)
        t = self.norm(t)
        m = mask.float()[..., None]
        n = m.sum(1).clamp(min=1)
        pooled = torch.cat([(t * m).sum(1) / n, (t + (m - 1) * 1e4).max(1).values, torch.log(n)], -1)
        mu, s = self.head(pooled).chunk(2, -1)
        return mu, F.softplus(s) + 1e-3


def loss_fn(net, batch, gid, ng, x, y, scale, fx=None):
    X, Y, D, A, M, NP = collate(batch, x, y)
    S0 = contrasts(crop_stats(Y, A, NP, M.float())) / scale
    mu, sd = net(X, D, M)
    if fx is not None:  # residual parameterization: plug-in statistic of the stain plus a learned correction
        Fs = collate(batch, fx, y)[0][..., -P:]
        mu = mu + contrasts(crop_stats(Fs, A, NP, M.float())) / scale
    region = crps_gauss(mu, sd, S0).mean()
    w = F.one_hot(gid, ng).float()
    w = w / w.sum(0, keepdim=True)
    mp = torch.einsum('bd,bg->gd', mu, w)
    sp = torch.sqrt(torch.einsum('bd,bg->gd', sd ** 2, w ** 2) + torch.exp(2 * net.log_tau))
    patient = crps_gauss(mp, sp, torch.einsum('bd,bg->gd', S0, w)).mean()
    return region + patient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--groups', type=int, default=4)
    ap.add_argument('--per', type=int, default=4)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--with-stain', action='store_true', help='matched comparator: stain predictions as inputs, predict plug-in statistic + correction')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    fd = args.run / f'fold{args.fold}'  # run directory (relative to the working directory); data from --root
    a = json.loads((fd / 'args.json').read_text())
    seed = int(a['seed'])
    torch.manual_seed(seed + 7)
    rng = np.random.default_rng(seed + 8)
    cells, x, y, crops, roles = se.load(args.root, args.fold, seed, int(a['pcadim']))
    fx = None
    if args.with_stain:
        import ligature as t
        kind = a.get('mean_kind', 'transformer')
        if kind == 'file':
            Fm = np.load(a['mean_file'].format(fold=args.fold)).astype(np.float32)
            xs = np.column_stack([x, Fm]); mean_net = t.ColumnMean(x.shape[1])
        elif kind == 'ridge':
            xs = x; mean_net = t.fit_ridge_mean(x, y, [c for c in crops if c['role'] == 'train'], [c for c in crops if c['role'] == 'val'])
        else:
            xs = x; mean_net = se.CellGraphNet(x.shape[1], d=int(a['d']), layers=int(a['layers']))
            mean_net.load_state_dict(torch.load(Path(a['mean_from']) / f'fold{args.fold}' / 'mean.pt'))
        mean_net.eval()
        f = np.zeros((len(x), P), np.float32)
        with torch.no_grad():
            pm = se.predict_mean(mean_net, crops, xs, y)
        for c in crops:
            f[c['ii']] = pm[c['crop']]
        x = np.column_stack([x, f]).astype(np.float32)  # stain predictions are inputs as well
        fx = x
    tr = [c for c in crops if c['role'] == 'train']
    va = [c for c in crops if c['role'] == 'val']
    te = [c for c in crops if c['role'] == 'test']
    S0 = [contrasts(crop_stats(torch.from_numpy(y[c['ii']])[None], torch.from_numpy(c['A'])[None],
                               torch.from_numpy(c['npairs'])[None], torch.ones(1, len(c['ii'])))) for c in tr]
    scale = torch.cat(S0).std(0).clamp(min=1e-3)
    net = DirectGraph(x.shape[1], scale.numel(), d=int(a['d']), layers=int(a['layers']))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    nb_ep = sum(1 for _ in group_batches(tr, np.random.default_rng(0), args.groups, args.per))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * nb_ep + 10, pct_start=.1)
    best, best_state, hist = math.inf, None, []
    for epoch in range(args.epochs):
        net.train(); tot, nb = 0., 0
        for gb in group_batches(tr, rng, args.groups, args.per):
            flat = [c for g in gb for c in g]
            gid = torch.as_tensor(np.repeat(np.arange(len(gb)), [len(g) for g in gb]))
            loss = loss_fn(net, flat, gid, len(gb), x, y, scale, fx)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item(); nb += 1
        net.eval(); vt, vn = 0., 0
        with torch.no_grad():
            for gb in group_batches(va, np.random.default_rng(11), args.groups, args.per):
                flat = [c for g in gb for c in g]
                gid = torch.as_tensor(np.repeat(np.arange(len(gb)), [len(g) for g in gb]))
                vt += loss_fn(net, flat, gid, len(gb), x, y, scale, fx).item(); vn += 1
        hist.append(dict(epoch=epoch, train=tot / nb, val=vt / vn))
        log(f'direct graph epoch {epoch}: train {tot / nb:.4f} val {vt / vn:.4f}')
        if vt / vn < best:
            best, best_state = vt / vn, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(te), 32):
            b = te[s:s + 32]
            X, Y, D, A, M, NP = collate(b, x, y)
            mu = net(X, D, M)[0]
            if fx is not None:
                mu = mu + contrasts(crop_stats(X[..., -P:], A, NP, M.float())) / scale
            preds.append((mu * scale).numpy())
    np.savez_compressed(fd / ('direct_graph_stain.npz' if args.with_stain else 'direct_graph.npz'), test=np.concatenate(preds),
                        test_crop=np.array([c['crop'] for c in te]))
    (fd / ('direct_graph_stain_history.json' if args.with_stain else 'direct_graph_history.json')).write_text(json.dumps(hist, indent=1))
    log(f'saved direct graph predictions in {fd}')


if __name__ == '__main__':
    main()
