#!/usr/bin/env python
"""Direct contact baseline: a graph model trained on the gated contact statistics themselves.

Ligature never scores gated contacts; this comparator is trained on exactly the query that
eval_contact.py evaluates. Per region the targets are the contact counts of eval_contact.py
(a-positive cells with a b-positive cell within 20 um, and a-positive cells; positivity above the
training-patient 85th percentile), divided by the region's cell count: 16 x 16 + 16 values. The
model is direct_graph.py's DirectGraph (same embeddings, cell-graph transformer, stain predictions
as inputs) predicting a correction of the quantile-matched plug-in's counts (thresholds from
validation-patient predictions, as for the quantile-matched stain), trained with Gaussian CRPS on
regions and on patient averages, early-stopped on validation patients. Writes
fold*/direct_contact.npz (test crops in eval_contact.py's order; predicted counts) into a Ligature run
for analyze_contact.py. With --bounded the counts are parameterized inside their support: a positive-cell
fraction pi_a and a conditional contact probability c_ab, each a logit-scale correction of the
quantile-matched plug-in's, with numerator pi_a c_ab <= denominator pi_a (per cell).
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
from spatial_engression import P, collate, group_batches, log
from ligature import crps_gauss
from direct_graph import DirectGraph
from eval_contact import contact


def counts(Yc, thr, dist, radius):
    """Per-region contact counts divided by the cell count: [P*P + P]."""
    adj = (torch.from_numpy(dist) <= radius).float()
    adj.fill_diagonal_(0)
    num, den = contact(torch.from_numpy(Yc)[None], thr, adj)
    return np.concatenate([num[0].ravel(), den[0]]) / len(Yc)


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
    ap.add_argument('--radius', type=float, default=20.)
    ap.add_argument('--q', type=float, default=.85)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--bounded', action='store_true', help='support-respecting parameterization (pi_a, c_ab)')
    ap.add_argument('--model-seed', type=int, default=0)
    ap.add_argument('--tag', default='', help='output suffix, e.g. _b0')
    ap.add_argument('--ratio-loss', type=float, default=0., help='weight of a patient-level absolute-error loss on the pooled '
                    'contact fraction itself (target-matched); validation then early-stops on that loss')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    fd = args.run / f'fold{args.fold}'
    a = json.loads((fd / 'args.json').read_text())
    seed = int(a['seed'])
    torch.manual_seed(seed + 17 + 1000 * args.model_seed)
    rng = np.random.default_rng(seed + 18 + 1000 * args.model_seed)
    cells, x, y, crops, roles = se.load(args.root, args.fold, seed, int(a['pcadim']))
    kind = a.get('mean_kind', 'transformer')
    assert kind in ('transformer', 'file'), kind
    if kind == 'file':  # external stain predictions for every cell (U-Net, GigaTIME, ...)
        f = np.load(a['mean_file'].format(fold=args.fold)).astype(np.float32)
        assert f.shape == (len(x), P)
    else:
        mean_net = se.CellGraphNet(x.shape[1], d=int(a['d']), layers=int(a['layers']))
        mean_net.load_state_dict(torch.load(Path(a['mean_from']) / f'fold{args.fold}' / 'mean.pt'))
        mean_net.eval()
        with torch.no_grad():
            pm = se.predict_mean(mean_net, crops, x, y)
        f = np.zeros((len(x), P), np.float32)
        for c in crops:
            f[c['ii']] = pm[c['crop']]
    role = cells.case.map(roles).to_numpy()
    thr = torch.from_numpy(np.quantile(y[role == 'train'], args.q, axis=0).astype(np.float32))
    va_ii = np.concatenate([c['ii'] for c in crops if c['role'] == 'val'])
    thr_q = torch.from_numpy(np.quantile(f[va_ii], args.q, axis=0).astype(np.float32))
    T = {c['crop']: counts(y[c['ii']], thr, c['dist'], args.radius) for c in crops}
    O = {c['crop']: counts(f[c['ii']], thr_q, c['dist'], args.radius) for c in crops}
    x = np.column_stack([x, f]).astype(np.float32)  # stain predictions are inputs, as in direct_graph --with-stain
    tr = [c for c in crops if c['role'] == 'train']
    va = [c for c in crops if c['role'] == 'val']
    te = [c for c in crops if c['role'] == 'test']
    D = P * P + P
    scale = torch.from_numpy(np.stack([T[c['crop']] for c in tr]).std(0).astype(np.float32)).clamp(min=1e-3)

    def logit(v):
        v = np.clip(v, 1e-3, 1 - 1e-3)
        return np.log(v / (1 - v))

    def offsets(batch):
        o = np.stack([O[c['crop']] for c in batch]).astype(np.float32)
        if not args.bounded:
            return torch.from_numpy(o)
        pi = o[:, P * P:]
        cab = o[:, :P * P].reshape(-1, P, P) / np.maximum(pi[:, :, None], 1e-6)
        return torch.from_numpy(np.concatenate([logit(cab).reshape(len(o), -1), logit(pi)], 1).astype(np.float32))

    def mean_counts(mu, off):
        """Per-cell contact counts [B, P*P + P] from the network output and the plug-in offsets."""
        if not args.bounded:
            return mu * scale + off
        z = mu + off
        pi = torch.sigmoid(z[:, P * P:])
        cab = torch.sigmoid(z[:, :P * P]).view(-1, P, P)
        return torch.cat([(pi[:, :, None] * cab).flatten(1), pi], 1)

    def loss_fn(net, batch, gid, ng):
        X, _, Dm, _, M, _ = collate(batch, x, y)
        t0 = torch.from_numpy(np.stack([T[c['crop']] for c in batch]).astype(np.float32)) / scale
        mu, sd = net(X, Dm, M)
        mu = mean_counts(mu, offsets(batch)) / scale
        region = crps_gauss(mu, sd, t0).mean()
        w = F.one_hot(gid, ng).float()
        w = w / w.sum(0, keepdim=True)
        mp = torch.einsum('bd,bg->gd', mu, w)
        sp = torch.sqrt(torch.einsum('bd,bg->gd', sd ** 2, w ** 2) + torch.exp(2 * net.log_tau))
        loss = region + crps_gauss(mp, sp, torch.einsum('bd,bg->gd', t0, w)).mean()
        if not args.ratio_loss:
            return loss, loss
        # the evaluated statistic: per patient, summed contact counts over summed positive counts
        n = torch.tensor([float(len(c['ii'])) for c in batch])[:, None]
        oh = F.one_hot(gid, ng).float()
        pc = torch.einsum('bd,bg->gd', mu * scale * n, oh)                 # predicted counts per patient
        tc = torch.einsum('bd,bg->gd', t0 * scale * n, oh)                 # measured counts per patient
        pf = pc[:, :P * P].view(-1, P, P) / pc[:, P * P:].clamp(min=1e-3)[:, :, None]
        tf = tc[:, :P * P].view(-1, P, P) / tc[:, P * P:].clamp(min=1e-3)[:, :, None]
        ok = (tc[:, P * P:] >= 5).float()[:, :, None].expand_as(tf)
        ratio = ((pf - tf).abs() * ok).sum() / ok.sum().clamp(min=1)
        return loss + args.ratio_loss * ratio, ratio

    net = DirectGraph(x.shape[1], D, d=int(a['d']), layers=int(a['layers']))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    nb_ep = sum(1 for _ in group_batches(tr, np.random.default_rng(0), args.groups, args.per))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * nb_ep + 10, pct_start=.1)
    best, best_state, hist = math.inf, None, []
    for epoch in range(args.epochs):
        net.train(); tot, nb = 0., 0
        for gb in group_batches(tr, rng, args.groups, args.per):
            flat = [c for g in gb for c in g]
            gid = torch.as_tensor(np.repeat(np.arange(len(gb)), [len(g) for g in gb]))
            loss = loss_fn(net, flat, gid, len(gb))[0]
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item(); nb += 1
        net.eval(); vt, vn = 0., 0
        with torch.no_grad():
            for gb in group_batches(va, np.random.default_rng(11), args.groups, args.per):
                flat = [c for g in gb for c in g]
                gid = torch.as_tensor(np.repeat(np.arange(len(gb)), [len(g) for g in gb]))
                vt += loss_fn(net, flat, gid, len(gb))[1].item(); vn += 1
        hist.append(dict(epoch=epoch, train=tot / nb, val=vt / vn))
        log(f'direct contact epoch {epoch}: train {tot / nb:.4f} val {vt / vn:.4f}')
        if vt / vn < best:
            best, best_state = vt / vn, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    pred = []
    with torch.no_grad():
        for s in range(0, len(te), 32):
            b = te[s:s + 32]
            X, _, Dm, _, M, _ = collate(b, x, y)
            mu, _ = net(X, Dm, M)
            n = np.array([len(c['ii']) for c in b], np.float32)[:, None]
            pred.append(np.clip(mean_counts(mu, offsets(b)).numpy(), 0, None) * n)  # predicted counts
    pred = np.concatenate(pred)
    np.savez_compressed(fd / f'direct_contact{args.tag}.npz', crop=np.array([c['crop'] for c in te]),
                        case=np.array([c['case'] for c in te]),
                        num=pred[:, :P * P].reshape(-1, P, P).astype(np.float32), den=pred[:, P * P:].astype(np.float32))
    (fd / f'direct_contact{args.tag}_history.json').write_text(json.dumps(hist, indent=1))
    log(f'saved direct contact predictions in {fd}')


if __name__ == '__main__':
    main()
