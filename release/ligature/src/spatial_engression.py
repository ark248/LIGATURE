#!/usr/bin/env python
"""Spatially coherent generative virtual multiplex, trained with proper scoring rules.

One patient-held-out role rotation of ORION-CRC: 24 train / 8 validation / 8 test
patients (same role assignment as pair_transfer_probe.py). Two models share one
cell-graph transformer over frozen Phikon-v2 cell embeddings:

  mean   : deterministic, per-cell squared loss (the standard virtual stain)
  joint  : stochastic; per-cell noise + one patient-level latent are injected and
           mixed by attention, trained with energy scores on cells, on crop-level
           spatial statistics, and on patient-group averages of those statistics.

Evaluation is on the 8 test patients only. Truth is the measured assay.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

MARKERS = ['CD31', 'CD45', 'CD68', 'CD4', 'FOXP3', 'CD8a', 'CD45RO', 'CD20',
           'PD-L1', 'CD3e', 'CD163', 'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA']
# other cohorts (e.g. cohorts/lung, 16 Xenium genes): comma-separated SE_MARKERS, same length
if os.environ.get('SE_MARKERS'):
    MARKERS = os.environ['SE_MARKERS'].split(',')
    assert len(MARKERS) == 16, 'the statistic layout assumes 16 markers'
P = len(MARKERS)
BINS = [(0., 25.), (25., 50.), (50., 75.)]
IU = np.triu_indices(P)  # 136 unordered marker pairs, diagonal included


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


# ----------------------------------------------------------------------------- data

def roles_for(cases, fold, seed):
    rng = np.random.default_rng(seed + 1)
    folds = np.arange(len(cases)) % 5
    rng.shuffle(folds)
    return {c: ('test' if folds[j] == fold else 'val' if folds[j] == (fold + 1) % 5 else 'train')
            for j, c in enumerate(cases)}


def load(root, fold, seed, pcadim):
    cells = pd.read_parquet(root / 'data/cells/cells.parquet').set_index('cell_idx')
    idx = np.load(root / 'data/features/cell_idx.npy')
    cells = cells.loc[idx].reset_index()
    cases = np.sort(cells.case.unique())
    roles = roles_for(cases, fold, seed)
    role = cells.case.map(roles).to_numpy()
    rng = np.random.default_rng(seed + 7 + fold)
    train = np.flatnonzero(role == 'train')
    feats = np.load(root / 'data/features/X_p64.npy', mmap_mode='r')
    fit = np.sort(rng.choice(train, min(60000, len(train)), replace=False))
    a = np.asarray(feats[fit], dtype=np.float32)
    mu = a.mean(0)
    _, _, vt = np.linalg.svd(a - mu, full_matrices=False)
    comp = vt[:pcadim].T.astype(np.float32)
    z = np.empty((len(cells), pcadim), np.float32)
    for s in range(0, len(cells), 20000):
        z[s:s + 20000] = (np.asarray(feats[s:s + 20000], np.float32) - mu) @ comp
    geom = np.column_stack([np.log(cells.cell_area_px), np.log(cells.nuc_area_px),
                            np.log(cells.nuc_perimeter_px), cells.nuc_eccentricity]).astype(np.float32)
    x = np.column_stack([z, geom])
    xm, xs = x[train].mean(0), x[train].std(0) + 1e-6
    x = (x - xm) / xs
    if os.environ.get('SE_RAW_TARGETS'):
        # simulated cohorts: targets are already on the modelling scale (no log, no standardization)
        y = cells[['y_' + m for m in MARKERS]].to_numpy(np.float32)
    else:
        y = np.log1p(cells[['y_' + m for m in MARKERS]].to_numpy(np.float64))
        ym, ys = y[train].mean(0), y[train].std(0) + 1e-6
        y = ((y - ym) / ys).astype(np.float32)
    if os.environ.get('SE_COPULA'):
        # Gaussian-copula scale: per-marker normal scores from the training cells' empirical CDF.
        # Gates are monotone in each marker, so gated statistics are identical on either scale.
        from scipy.stats import norm
        nt = len(train)
        for p in range(y.shape[1]):
            ref = np.sort(y[train, p])
            r = (np.searchsorted(ref, y[:, p], 'left') + np.searchsorted(ref, y[:, p], 'right')) / 2
            y[:, p] = norm.ppf(np.clip((r + .5) / (nt + 1), .5 / (nt + 1), (nt + .5) / (nt + 1)))
    crops = []
    for crop, ii in cells.groupby('crop_id', sort=True).indices.items():
        if len(ii) < 2:
            continue
        xy = cells[['gx', 'gy']].to_numpy()[ii] * .325
        d = cdist(xy, xy)
        A = []
        for lo, hi in BINS:
            m = (d > lo) & (d <= hi)
            np.fill_diagonal(m, False)
            A.append(m)
        A = np.stack(A).astype(np.float32)
        npairs = A.sum((1, 2))
        crops.append(dict(crop=crop, case=cells.case.iat[ii[0]], role=role[ii[0]], ii=ii,
                          dist=d.astype(np.float32), A=A, npairs=npairs,
                          eligible=bool(npairs[0] > 0 and npairs[-1] > 0)))
    log(f'fold {fold}: {len(cells)} cells, {len(crops)} crops, '
        f'roles train/val/test = {[sum(r == k for r in roles.values()) for k in ("train", "val", "test")]}')
    return cells, x, y, crops, roles


# ----------------------------------------------------------------------------- statistics

def crop_stats(Y, A, npairs, mask):
    """Y: [m,B,n,P] (or [B,n,P]); A: [B,3,n,n]; npairs: [B,3]; mask: [B,n].

    Returns [m, B, P + 3*136] (or [B, ...]): means over cells, then mean symmetric
    products over the ordered pairs in each distance bin.
    """
    squeeze = Y.dim() == 3
    if squeeze:
        Y = Y[None]
    Ym = Y * mask[None, ..., None]
    nc = mask.sum(1).clamp(min=1)
    means = Ym.sum(2) / nc[None, :, None]
    AY = torch.einsum('bkij,mbjp->mbkip', A, Ym)
    G = torch.einsum('mbip,mbkiq->mbkpq', Ym, AY) / npairs.clamp(min=1)[None, ..., None, None]
    iu0, iu1 = torch.as_tensor(IU[0]), torch.as_tensor(IU[1])
    G = 0.5 * (G + G.transpose(-1, -2))
    S = torch.cat([means, G[:, :, :, iu0, iu1].reshape(*G.shape[:2], -1)], -1)
    return S[0] if squeeze else S


def energy_score(S, obs):
    """S: [m, ..., D] samples; obs: [..., D]. Unbiased energy score, averaged over leading dims."""
    m = S.shape[0]
    t1 = (S - obs[None]).norm(dim=-1).mean(0)
    diff = (S[:, None] - S[None]).norm(dim=-1)
    t2 = diff.sum((0, 1)) / (m * (m - 1))
    return (t1 - 0.5 * t2).mean()


# ----------------------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, d, h, nrbf):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.bias = nn.Linear(nrbf, h)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, t, rbf, keymask, reps):
        B, n, d = t.shape
        q, k, v = self.qkv(self.n1(t)).view(B, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        bias = self.bias(rbf).permute(0, 3, 1, 2)  # b,h,n,n on the unrepeated batch
        bias = bias.masked_fill(~keymask[:, None, None, :], float('-inf'))
        if reps > 1:
            bias = bias[None].expand(reps, *bias.shape).reshape(B, *bias.shape[1:])
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        t = t + self.o(a.transpose(1, 2).reshape(B, n, d))
        return t + self.ff(self.n2(t))


class CellGraphNet(nn.Module):
    def __init__(self, din, d=128, layers=3, heads=4, cell_noise=0, group_noise=0, nrbf=16, late=False):
        super().__init__()
        self.cell_noise, self.group_noise, self.late = cell_noise, group_noise, late
        # late=True injects noise after attention: samples are conditionally
        # independent across cells given the image (a marginal predictive model)
        self.inp = nn.Linear(din + (0 if late else cell_noise + group_noise), d)
        self.centers = nn.Parameter(torch.linspace(0, 160, nrbf), requires_grad=False)
        self.blocks = nn.ModuleList([Block(d, heads, nrbf) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d + (cell_noise + group_noise if late else 0), d), nn.GELU(), nn.Linear(d, P))

    def forward(self, x, dist, mask, eps=None, eta=None):
        """x: [reps*b, n, din]; dist: [b, n, n]; mask: [b, n]. Sample r of crop i is row r*b + i."""
        reps = x.shape[0] // dist.shape[0]
        noise = []
        if self.cell_noise:
            noise.append(eps)
        if self.group_noise:
            noise.append(eta[:, None, :].expand(-1, x.shape[1], -1))
        t = self.inp(torch.cat([x] + ([] if self.late else noise), -1))
        rbf = torch.exp(-((dist[..., None] - self.centers) / 10.) ** 2)
        for b in self.blocks:
            t = b(t, rbf, mask, reps)
        t = self.out_norm(t)
        if self.late and noise:
            t = torch.cat([t] + noise, -1)
        return self.out(t)


def collate(batch, x, y):
    n = max(len(c['ii']) for c in batch)
    B = len(batch)
    X = np.zeros((B, n, x.shape[1]), np.float32)
    Y = np.zeros((B, n, P), np.float32)
    D = np.full((B, n, n), 500., np.float32)
    A = np.zeros((B, len(BINS), n, n), np.float32)
    M = np.zeros((B, n), bool)
    NP = np.zeros((B, len(BINS)), np.float32)
    for b, c in enumerate(batch):
        k = len(c['ii'])
        X[b, :k] = x[c['ii']]
        Y[b, :k] = y[c['ii']]
        D[b, :k, :k] = c['dist']
        A[b, :, :k, :k] = c['A']
        M[b, :k] = True
        NP[b] = c['npairs']
    return [torch.from_numpy(v) for v in (X, Y, D, A, M, NP)]


# ----------------------------------------------------------------------------- training

def group_batches(crops, rng, groups, per):
    by = {}
    for c in crops:
        by.setdefault(c['case'], []).append(c)
    order = []
    for case, cs in by.items():
        perm = rng.permutation(len(cs))
        for s in range(0, len(cs), per):
            order.append([cs[i] for i in perm[s:s + per]])
    rng.shuffle(order)
    for s in range(0, len(order), groups):
        yield order[s:s + groups]


def stat_scale(crops, y):
    S = []
    for c in crops:
        Yc = torch.from_numpy(y[c['ii']])[None]
        S.append(crop_stats(Yc, torch.from_numpy(c['A'])[None], torch.from_numpy(c['npairs'])[None],
                            torch.ones(1, len(c['ii']))))
    S = torch.cat(S)
    return S.std(0).clamp(min=1e-3)


def train_model(kind, crops_tr, crops_va, x, y, args, scale):
    torch.manual_seed(args.seed + (0 if kind == 'mean' else 1))
    rng = np.random.default_rng(args.seed + 3)
    net = CellGraphNet(x.shape[1], d=args.d, layers=args.layers,
                       cell_noise=args.cell_noise if kind == 'joint' else 0,
                       group_noise=args.group_noise if kind == 'joint' else 0,
                       late=args.late_noise and kind == 'joint')
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    per_case = pd.Series([c['case'] for c in crops_tr]).value_counts()
    chunks = int(np.ceil(per_case.to_numpy() / args.per).sum())
    steps = args.epochs * math.ceil(chunks / args.groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps + 10, pct_start=.1)
    best, best_state, history = float('inf'), None, []
    for epoch in range(args.epochs):
        net.train()
        tot, nb = 0., 0
        for gb in group_batches(crops_tr, rng, args.groups, args.per):
            flat = [c for g in gb for c in g]
            gid = torch.as_tensor(np.repeat(np.arange(len(gb)), [len(g) for g in gb]))
            X, Y, D, A, M, NP = collate(flat, x, y)
            loss = run_loss(net, kind, X, Y, D, A, M, NP, gid, len(gb), args, scale)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        val = evaluate_loss(net, kind, crops_va, x, y, args, scale)
        history.append(dict(epoch=epoch, train=tot / nb, val=val))
        log(f'{kind} epoch {epoch}: train {tot / nb:.4f} val {val:.4f}')
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, history


def contrasts(S):
    """Append bin contrasts (0-2, 0-1, 1-2) of every pair statistic to S [..., P + 3K] (as in ligature.py)."""
    k = len(IU[0])
    g = [S[..., P + b * k:P + (b + 1) * k] for b in range(3)]
    return torch.cat([S, g[0] - g[2], g[0] - g[1], g[1] - g[2]], -1)


def crps_ens(X, y):
    """Unbiased ensemble CRPS per coordinate, X: [m, ..., D], y: [..., D]; returns [..., D]."""
    m = X.shape[0]
    return (X - y[None]).abs().mean(0) - .5 * (X[:, None] - X[None]).abs().sum((0, 1)) / (m * (m - 1))


def run_loss(net, kind, X, Y, D, A, M, NP, gid, ngroups, args, scale):
    if kind == 'mean':
        out = net(X, D, M)
        return ((out - Y) ** 2 * M[..., None]).sum() / (M.sum() * P)
    m = args.samples
    B, n, _ = X.shape
    eps = torch.randn(m, B, n, args.cell_noise)
    eta = torch.randn(m, ngroups, args.group_noise)[:, gid]
    rep = lambda t: t[None].expand(m, *t.shape).reshape(m * B, *t.shape[1:])
    out = net(rep(X), D, M, eps.reshape(m * B, n, -1), eta.reshape(m * B, -1)).view(m, B, n, P)
    if getattr(args, 'objective', 'energy') == 'crps':
        # matched comparator: the statistic-level objective of ligature.py (per-coordinate CRPS on cells, on region
        # statistics with bin contrasts, and on patient averages) applied to this noise-injected generator
        cell = (crps_ens(out, Y) * M[..., None]).sum() / (M.sum() * P)
        S = contrasts(crop_stats(out, A, NP, M.float())) / scale
        S0 = contrasts(crop_stats(Y, A, NP, M.float())) / scale
        onehot = F.one_hot(gid, ngroups).float()
        w = onehot / onehot.sum(0, keepdim=True)
        G, G0 = torch.einsum('mbd,bg->mgd', S, w), torch.einsum('bd,bg->gd', S0, w)
        return cell + args.lam_crop * crps_ens(S, S0).mean() + args.lam_group * crps_ens(G, G0).mean()
    # cell-level energy score over the 16-marker vector
    t1 = (out - Y[None]).norm(dim=-1).mean(0)
    t2 = (out[:, None] - out[None]).norm(dim=-1).sum((0, 1)) / (m * (m - 1))
    cell = ((t1 - .5 * t2) * M).sum() / M.sum()
    # crop-level spatial statistics
    S = crop_stats(out, A, NP, M.float()) / scale
    S0 = crop_stats(Y, A, NP, M.float()) / scale
    crop = energy_score(S, S0)
    # patient-group averages of the statistics (shared patient latent)
    onehot = F.one_hot(gid, ngroups).float()
    w = onehot / onehot.sum(0, keepdim=True)
    G = torch.einsum('mbd,bg->mgd', S, w)
    G0 = torch.einsum('bd,bg->gd', S0, w)
    group = energy_score(G, G0)
    return cell + args.lam_crop * crop / math.sqrt(S.shape[-1]) + args.lam_group * group / math.sqrt(S.shape[-1])


@torch.no_grad()
def evaluate_loss(net, kind, crops, x, y, args, scale):
    net.eval()
    rng = np.random.default_rng(11)
    tot, nb = 0., 0
    state = torch.random.get_rng_state()
    torch.manual_seed(5)
    for gb in group_batches(crops, rng, args.groups, args.per):
        flat = [c for gg in gb for c in gg]
        gid = torch.as_tensor(np.repeat(np.arange(len(gb)), [len(gg) for gg in gb]))
        X, Y, D, A, M, NP = collate(flat, x, y)
        tot += run_loss(net, kind, X, Y, D, A, M, NP, gid, len(gb), args, scale).item()
        nb += 1
    torch.random.set_rng_state(state)
    return tot / nb


# ----------------------------------------------------------------------------- inference

@torch.no_grad()
def predict_mean(net, crops, x, y, bs=32):
    net.eval()
    out = {}
    for s in range(0, len(crops), bs):
        batch = crops[s:s + bs]
        X, Y, D, A, M, NP = collate(batch, x, y)
        f = net(X, D, M)
        for b, c in enumerate(batch):
            out[c['crop']] = f[b, :len(c['ii'])].numpy()
    return out


@torch.no_grad()
def sample_stats(net, crops, x, y, args, K):
    """Returns per-crop statistics for K joint samples: dict crop -> [K, D].

    Sample k of every crop of the same patient shares the patient latent eta[k].
    """
    net.eval()
    gen = torch.Generator().manual_seed(args.seed + 99)
    cases = sorted({c['case'] for c in crops})
    eta = {c: torch.randn(K, args.group_noise, generator=gen) for c in cases}
    out, cellmean = {}, {}
    for c in crops:
        X, Y, D, A, M, NP = collate([c], x, y)
        n = X.shape[1]
        eps = torch.randn(K, n, args.cell_noise, generator=gen)
        smp = net(X.expand(K, *X.shape[1:]), D, M, eps, eta[c['case']])
        out[c['crop']] = crop_stats(smp[:, None], A, NP, M.float())[:, 0].numpy()
        cellmean[c['crop']] = smp.mean(0).numpy()
    return out, cellmean


def stats_of(Ymap, crops):
    S = {}
    for c in crops:
        Yc = torch.from_numpy(np.asarray(Ymap[c['crop']], np.float32))[None]
        S[c['crop']] = crop_stats(Yc, torch.from_numpy(c['A'])[None], torch.from_numpy(c['npairs'])[None],
                                  torch.ones(1, len(c['ii']))).numpy()[0]
    return S


def query_vector(S):
    """near (bin 0) minus far (bin 2) symmetric product for all 136 pairs."""
    k = len(IU[0])
    return S[..., P:P + k] - S[..., P + 2 * k:P + 3 * k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--out', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/spatial_engression'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=271027)
    ap.add_argument('--pcadim', type=int, default=128)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--groups', type=int, default=4)
    ap.add_argument('--per', type=int, default=4)
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--cell-noise', type=int, default=16)
    ap.add_argument('--group-noise', type=int, default=8)
    ap.add_argument('--lam-crop', type=float, default=1.0)
    ap.add_argument('--lam-group', type=float, default=1.0)
    ap.add_argument('--eval-samples', type=int, default=64)
    ap.add_argument('--late-noise', action='store_true')
    ap.add_argument('--skip-mean', action='store_true', help='reuse mean.pt from --mean-from')
    ap.add_argument('--mean-from', type=Path, default=None)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--max-train-crops', type=int, default=0)
    ap.add_argument('--mean-only', action='store_true', help='train and save the per-cell plug-in stain only')
    ap.add_argument('--objective', choices=['energy', 'crps'], default='energy', help='crps: ligature.py statistic-level objective')
    ap.add_argument('--stain-inputs', action='store_true', help='the joint generator also takes the plug-in stain predictions')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out / f'fold{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    cells, x, y, crops, roles = load(args.root, args.fold, args.seed, args.pcadim)
    (out / 'roles.json').write_text(json.dumps(roles, indent=2))
    tr = [c for c in crops if c['role'] == 'train']
    va = [c for c in crops if c['role'] == 'val']
    te = [c for c in crops if c['role'] == 'test']
    if args.max_train_crops:
        r = np.random.default_rng(0)
        tr = [tr[i] for i in r.choice(len(tr), args.max_train_crops, replace=False)]
        va = va[:args.max_train_crops // 4]
        te = te[:args.max_train_crops // 4]
    scale = stat_scale(tr, y)
    if args.objective == 'crps':
        scale = torch.cat([contrasts(crop_stats(torch.from_numpy(y[c['ii']])[None], torch.from_numpy(c['A'])[None],
                                                torch.from_numpy(c['npairs'])[None], torch.ones(1, len(c['ii']))))
                           for c in tr]).std(0).clamp(min=1e-3)
    models = {}
    xj = x
    for kind in ['mean', 'joint']:
        if kind == 'joint' and args.stain_inputs:
            # frozen stain predictions as extra inputs of the generator (the plug-in keeps the original inputs)
            pm = predict_mean(models['mean'], crops, x, y)
            f = np.zeros((len(x), P), np.float32)
            for c in crops:
                f[c['ii']] = pm[c['crop']]
            xj = np.column_stack([x, f]).astype(np.float32)
        if kind == 'mean' and args.skip_mean:
            net = CellGraphNet(x.shape[1], d=args.d, layers=args.layers)
            state = torch.load(args.mean_from / f'fold{args.fold}' / 'mean.pt')
            if 'out.3.weight' in state:  # checkpoints written before out_norm was split out
                ren = {'out.0.': 'out_norm.', 'out.1.': 'out.0.', 'out.3.': 'out.2.'}
                state = {next((v + k[len(o):] for o, v in ren.items() if k.startswith(o)), k): t
                         for k, t in state.items()}
            net.load_state_dict(state)
            models[kind] = net
            continue
        net, hist = train_model(kind, tr, va, xj if kind == 'joint' else x, y, args, scale)
        torch.save(net.state_dict(), out / f'{kind}.pt')
        pd.DataFrame(hist).to_csv(out / f'{kind}_history.csv', index=False)
        models[kind] = net
        if args.mean_only:
            (out / 'complete.json').write_text(json.dumps(dict(mean_only=True)))
            return
    # --- statistics on validation and test crops
    rows = []
    for split, cs in [('val', va), ('test', te)]:
        truth = stats_of({c['crop']: y[c['ii']] for c in cs}, cs)
        fmean = predict_mean(models['mean'], cs, x, y)
        plug = stats_of(fmean, cs)
        smp, cm = sample_stats(models['joint'], cs, xj, y, args, args.eval_samples)
        plug_joint = stats_of(cm, cs)
        np.savez_compressed(out / f'{split}_stats.npz',
                            crop=np.array([c['crop'] for c in cs]), case=np.array([c['case'] for c in cs]),
                            eligible=np.array([c['eligible'] for c in cs]), ncell=np.array([len(c['ii']) for c in cs]),
                            truth=np.stack([truth[c['crop']] for c in cs]),
                            plugin=np.stack([plug[c['crop']] for c in cs]),
                            plugin_joint=np.stack([plug_joint[c['crop']] for c in cs]),
                            samples=np.stack([smp[c['crop']] for c in cs]).astype(np.float32))
        # cell-level accuracy
        for c in cs:
            rows.append(dict(split=split, crop=c['crop'], case=c['case'],
                             mse_mean=float(((fmean[c['crop']] - y[c['ii']]) ** 2).mean()),
                             mse_jointmean=float(((cm[c['crop']] - y[c['ii']]) ** 2).mean()),
                             ncell=len(c['ii'])))
    pd.DataFrame(rows).to_csv(out / 'cell_mse.csv', index=False)
    (out / 'complete.json').write_text(json.dumps(dict(fold=args.fold, test_cases=sorted({c['case'] for c in te}))))
    log('DONE')


if __name__ == '__main__':
    main()
