#!/usr/bin/env python
"""Ligature v2: statistic-scored generator with explicit morphology-dependent residual fields.

Conditional on the image, every cell's 16-marker vector is

    Y_i = mu(H)_i + sum_l A_l(H)_i xi_l(x_i) + B(H)_i eps_i + C(H)_i eta_patient + E(H)_i zeta_region,

where mu and all loading matrices are heads of one cell-graph transformer, xi_l are
independent unit Gaussian random fields with squared-exponential correlation at
length scale l (random Fourier features), eps_i is cell noise, zeta a region
latent and eta a patient latent shared by every region of a slide. This is a
linear model of coregionalization whose loadings depend on morphology.

Training minimizes, with m joint samples per region,
    CRPS of every cell-marker marginal (closed form, Gaussian)
  + mean CRPS over all region statistics and their bin contrasts
  + mean CRPS over the same statistics averaged within a patient group.
Patient roles are identical to spatial_engression.py. Outputs use the same file
format so analyze_spatial_engression.py applies unchanged.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_engression as se
from spatial_engression import P, BINS, IU, crop_stats, collate, group_batches, stat_scale, log
from eval_unseen_statistics import fine_adjacency, statistics as unseen_statistics

SQPI = 1 / math.sqrt(math.pi)


def contrasts(S):
    """Append bin contrasts (0-2, 0-1, 1-2) of every pair statistic to S [..., P + 3K]."""
    k = len(IU[0])
    g = [S[..., P + b * k:P + (b + 1) * k] for b in range(3)]
    return torch.cat([S, g[0] - g[2], g[0] - g[1], g[1] - g[2]], -1)


def crps_samples(X, y):
    """X: [m, ..., D] samples, y: [..., D]. Unbiased sample CRPS, averaged over all entries."""
    m = X.shape[0]
    t1 = (X - y[None]).abs().mean(0)
    t2 = (X[:, None] - X[None]).abs().sum((0, 1)) / (m * (m - 1))
    return (t1 - .5 * t2).mean()


def crps_gauss(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * torch.distributions.Normal(0., 1.).cdf(z) - 1) + 2 * torch.exp(-.5 * z * z) / math.sqrt(2 * math.pi) - SQPI)


class Ligature(nn.Module):
    def __init__(self, din, d=128, layers=3, heads=4, nrbf=16, scales=(10., 20., 40., 80.), rank=4,
                 cell_dim=8, region_dim=4, patient_dim=4, nfeat=64, frozen_mean=False, dropout=0., stationary=False,
                 tail_df=0, mean_correction=False):
        super().__init__()
        # mean_correction=True (with frozen_mean): mean = f + learned correction, initialized at zero, so the
        # wrapper can remove a predictor-residual cross term that a frozen mean leaves in every statistic
        self.mean_correction = mean_correction
        # tail_df > 0: region and patient latents are Student-t (Gaussian scale mixtures)
        self.tail_df = tail_df
        # stationary=True: loadings are free parameters shared by every cell (no morphology dependence)
        self.stationary = stationary
        self.scales, self.rank, self.nfeat, self.frozen_mean = scales, rank, nfeat, frozen_mean
        self.dims = dict(field=len(scales) * rank, cell=cell_dim, region=region_dim, patient=patient_dim)
        self.K = sum(self.dims.values())
        # one log-scale per variance component (each field scale, cell, region, patient);
        # fixed at zero during training, fitted on validation patients afterwards
        groups = np.r_[np.repeat(np.arange(len(scales)), rank), np.full(cell_dim, len(scales)),
                       np.full(region_dim, len(scales) + 1), np.full(patient_dim, len(scales) + 2)]
        self.register_buffer('group', torch.as_tensor(groups, dtype=torch.long))
        self.log_scale = nn.Parameter(torch.zeros(len(scales) + 3), requires_grad=False)
        self.drop = nn.Dropout(dropout)
        self.inp = nn.Linear(din + (P if frozen_mean else 0), d)
        self.centers = nn.Parameter(torch.linspace(0, 160, nrbf), requires_grad=False)
        self.blocks = nn.ModuleList([se.Block(d, heads, nrbf) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.mu = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, P))
        self.load = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, P * self.K))
        self.const_load = nn.Parameter(torch.randn(P, self.K) * .1)
        nn.init.normal_(self.load[-1].weight, std=1e-3)
        nn.init.constant_(self.load[-1].bias, 0.)
        if mean_correction:
            nn.init.zeros_(self.mu[-1].weight); nn.init.zeros_(self.mu[-1].bias)

    def encode(self, x, dist, mask, f=None):
        if getattr(self, 'no_geom', False):  # ablation: no morphology descriptors (last 4 input columns)
            x = torch.cat([x[..., :-4], torch.zeros_like(x[..., -4:])], -1)
        if self.frozen_mean:
            x = torch.cat([x, torch.zeros_like(f) if getattr(self, 'no_stain_input', False) else f], -1)
        t = self.drop(self.inp(x))
        rbf = torch.exp(-((dist[..., None] - self.centers) / 10.) ** 2)
        for b in self.blocks:
            t = b(t, rbf, mask, 1)
        t = self.norm(t)
        mu = (f + self.mu(t) if self.mean_correction else f) if self.frozen_mean else self.mu(t)
        if self.stationary:
            L = self.const_load.expand(*t.shape[:2], P, self.K)
        else:
            L = self.load(t).view(*t.shape[:2], P, self.K)
        L = L * .3 * torch.exp(self.log_scale)[self.group]
        return mu, L

    def latent(self, shape, gen=None):
        kw = dict(generator=gen) if gen is not None else {}
        z = torch.randn(*shape, **kw)
        if self.tail_df:
            chi2 = (torch.randn(self.tail_df, *shape[:-1], **kw) ** 2).sum(0)
            z = z * torch.sqrt(self.tail_df / chi2)[..., None]
        return z

    def noise(self, m, pos, mask, gid, ngroups, gen=None, eta=None):
        """pos: [B, n, 2] in um. Returns z: [m, B, n, K] of unit-variance latent noise."""
        B, n, _ = pos.shape
        kw = dict(generator=gen) if gen is not None else {}
        fields = []
        for ell in self.scales:
            om = torch.randn(m, B, self.nfeat, 2, **kw) / ell
            ph = torch.rand(m, B, self.nfeat, **kw) * 2 * math.pi
            w = torch.randn(m, B, self.nfeat, self.rank, **kw)
            feat = math.sqrt(2 / self.nfeat) * torch.cos(torch.einsum('bnc,mbfc->mbnf', pos, om) + ph[:, :, None])
            fields.append(torch.einsum('mbnf,mbfr->mbnr', feat, w))
        eps = torch.randn(m, B, n, self.dims['cell'], **kw)
        zeta = self.latent((m, B, self.dims['region']), gen)[:, :, None].expand(-1, -1, n, -1)
        if eta is None:
            eta = self.latent((m, ngroups, self.dims['patient']), gen)
        eta = eta[:, gid][:, :, None].expand(-1, -1, n, -1)
        return torch.cat(fields + [eps, zeta, eta], -1)

    def sample(self, mu, L, z):
        return mu[None] + torch.einsum('bnpk,mbnk->mbnp', L, z)


class LinearMean(nn.Module):
    """Per-cell ridge virtual stain without spatial context: f_i = W'x_i + b."""

    def __init__(self, W, b):
        super().__init__()
        self.register_buffer('W', torch.as_tensor(W, dtype=torch.float32))
        self.register_buffer('b', torch.as_tensor(b, dtype=torch.float32))

    def forward(self, X, D, M):
        return X @ self.W + self.b


class ColumnMean(nn.Module):
    """External per-cell virtual stain stored as the last P feature columns."""

    def __init__(self, start):
        super().__init__()
        self.start = start

    def forward(self, X, D, M):
        return X[..., self.start:self.start + P]


def fit_ridge_mean(x, y, tr, va):
    """Closed-form ridge on cell features of training patients; penalty chosen on validation patients."""
    itr = np.concatenate([c['ii'] for c in tr]); iva = np.concatenate([c['ii'] for c in va])
    X, Y = x[itr].astype(np.float64), y[itr].astype(np.float64)
    xm, ym = X.mean(0), Y.mean(0)
    G = (X - xm).T @ (X - xm); R = (X - xm).T @ (Y - ym)
    best = None
    for lam in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
        W = np.linalg.solve(G + lam * np.eye(len(G)), R)
        err = (((x[iva] - xm) @ W + ym - y[iva]) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, lam, W)
    _, lam, W = best
    log(f'ridge virtual stain: lambda {lam:g}, validation MSE {best[0]:.4f}')
    return LinearMean(W, ym - xm @ W)


def positions(batch, cells):
    n = max(len(c['ii']) for c in batch)
    pos = np.zeros((len(batch), n, 2), np.float32)
    xy = cells[['gx', 'gy']].to_numpy() * .325
    for b, c in enumerate(batch):
        p = xy[c['ii']]
        pos[b, :len(p)] = p - p.mean(0)
    return torch.from_numpy(pos)


def gaussian_nll(net, mu, L, Y, pos, M, gen=None, nugget=1e-2):
    """Exact Gaussian negative log-likelihood of each region's cells under the residual model.

    With one random-feature draw per field and region (as in sampling), the region covariance
    is block-diagonal cell noise (plus a small nugget) plus a low-rank part (field features
    times loadings, region and patient loadings), so Woodbury gives the likelihood with one
    (S*r*nfeat + 8)-dimensional Cholesky per region. Returns the mean NLL per cell-marker.
    """
    B, n, Pm, K = L.shape
    S, R, nf = len(net.scales), net.rank, net.nfeat
    kw = dict(generator=gen) if gen is not None else {}
    Mf = M.float()
    Lm = L * Mf[..., None, None]
    r = (Y - mu) * Mf[..., None]
    cols = []
    for k, ell in enumerate(net.scales):
        om = torch.randn(B, nf, 2, **kw) / ell
        ph = torch.rand(B, nf, **kw) * 2 * math.pi
        feat = math.sqrt(2 / nf) * torch.cos(torch.einsum('bnc,bfc->bnf', pos, om) + ph[:, None])
        cols.append(torch.einsum('bnpr,bnf->bnprf', Lm[..., k * R:(k + 1) * R], feat).reshape(B, n, Pm, R * nf))
    c0, cd = S * R, net.dims['cell']
    U = torch.cat(cols + [Lm[..., c0 + cd:]], -1)  # [B, n, P, Kt]
    Bc = Lm[..., c0:c0 + cd]
    eye = torch.eye(Pm)
    D = torch.einsum('bnpc,bnqc->bnpq', Bc, Bc) + nugget * eye
    D = torch.where(M[..., None, None], D, eye.expand_as(D))
    Dc = torch.linalg.cholesky(D)
    logdet = 2 * torch.log(torch.diagonal(Dc, dim1=-2, dim2=-1)).sum((-1, -2))
    DiU = torch.cholesky_solve(U, Dc)
    Dir = torch.cholesky_solve(r[..., None], Dc)[..., 0]
    cap = torch.eye(U.shape[-1]) + torch.einsum('bnpk,bnpl->bkl', U, DiU)
    Cc = torch.linalg.cholesky(cap)
    logdet = logdet + 2 * torch.log(torch.diagonal(Cc, dim1=-2, dim2=-1)).sum(-1)
    v = torch.einsum('bnpk,bnp->bk', U, Dir)
    w = torch.cholesky_solve(v[..., None], Cc)[..., 0]
    quad = (r * Dir).sum((1, 2)) - (v * w).sum(-1)
    npts = Mf.sum(1) * Pm
    nll = .5 * (quad + logdet + npts * math.log(2 * math.pi))
    return (nll / npts).mean()


def within_cell(Y, Mf):
    """Region means of within-cell products Y_ia Y_ib over the 136 marker pairs: [..., 136] (Y: [m,B,n,P] or [B,n,P])."""
    iu0, iu1 = torch.as_tensor(IU[0]), torch.as_tensor(IU[1])
    w = Mf if Y.dim() == 3 else Mf[None]
    return (Y[..., iu0] * Y[..., iu1] * w[..., None]).sum(-2) / w.sum(-1, keepdim=True).clamp(min=1)


def variogram_score(smp, Y, D, M, p=0.5, rmax=75., max_pairs=4000, unbiased=False):
    """Variogram score of order p (Scheuerer & Hamill 2015) over all marker pairs of all cell pairs within
    rmax um (same cell included), averaged over pairs and regions. smp: [m,B,n,P]; Y: [B,n,P]."""
    out = []
    for b in range(Y.shape[0]):
        k = int(M[b].sum())
        i, j = torch.nonzero(D[b, :k, :k] < rmax, as_tuple=True)
        if len(i) > max_pairs:  # unbiased subsample of the averaged score, bounds memory in dense regions
            keep = torch.randperm(len(i))[:max_pairs]
            i, j = i[keep], j[keep]
        obs = (Y[b, i][:, :, None] - Y[b, j][:, None, :]).abs() ** p                      # pairs,P,P
        prd = ((smp[:, b, i][:, :, :, None] - smp[:, b, j][:, :, None, :]).abs() + 1e-8) ** p
        sq = (obs - prd.mean(0)) ** 2
        if unbiased:  # (obs - mean_s)^2 overestimates (obs - E prd)^2 by Var(prd)/m; remove it
            sq = sq - prd.var(0, unbiased=True) / prd.shape[0]
        out.append(sq.mean())
    return torch.stack(out).mean()


def loss_fn(net, batch, gid, ngroups, x, y, cells, args, scale, gen=None, mean_net=None):
    X, Y, D, A, M, NP = collate(batch, x, y)
    pos = positions(batch, cells)
    f = None
    if net.frozen_mean:
        with torch.no_grad():
            f = mean_net(X, D, M)
    mu, L = net.encode(X, D, M, f)
    if getattr(args, 'objective', 'crps') == 'ml':
        # likelihood-trained comparator: same model, exact Gaussian likelihood of the cells
        nll = gaussian_nll(net, mu, L, Y, pos, M, gen=gen)
        return nll, (nll.item(), 0., 0.)
    sd = torch.sqrt((L ** 2).sum(-1) + 1e-4)
    Mf = M.float()
    cell = (crps_gauss(mu, sd, Y) * Mf[..., None]).sum() / (Mf.sum() * P)
    z = net.noise(args.samples, pos, M, gid, ngroups, gen=gen)
    smp = net.sample(mu, L, z)
    if getattr(args, 'objective', 'crps') == 'variogram':
        # conventional dependence-sensitive comparator: per-cell CRPS plus a variogram score on the cells
        vs = variogram_score(smp, Y, D, M, unbiased=getattr(args, 'vs_unbiased', False))
        return cell + getattr(args, 'vs_weight', 1.) * vs, (cell.item(), vs.item(), 0.)
    S = contrasts(crop_stats(smp, A, NP, Mf))
    S0 = contrasts(crop_stats(Y, A, NP, Mf))
    if getattr(args, 'score_within_cell', False):
        # also score within-cell co-expression, the covariance block that bin statistics leave unidentified
        S, S0 = torch.cat([S, within_cell(smp, Mf)], -1), torch.cat([S0, within_cell(Y, Mf)], -1)
    S, S0 = S / scale, S0 / scale
    if getattr(args, 'no_contrasts', False):  # ablation: score means and binned moments only
        S, S0 = S[..., :P + 3 * len(IU[0])], S0[..., :P + 3 * len(IU[0])]
    if getattr(args, 'objective', 'crps') == 'moment':
        # comparator: fit only the expected statistics (squared error of the sample mean), not their distributions
        onehot = F.one_hot(gid, ngroups).float()
        w = onehot / onehot.sum(0, keepdim=True)
        m = S.shape[0]
        mS = S.mean(0)
        region = ((mS - S0) ** 2 - S.var(0, unbiased=True) / m).mean()  # unbiased for (E S - S0)^2
        Sp = torch.einsum('mbd,bg->mgd', S, w)
        patient = ((Sp.mean(0) - torch.einsum('bd,bg->gd', S0, w)) ** 2 - Sp.var(0, unbiased=True) / m).mean()
        mw = getattr(args, 'moment_weight', 1.)
        return (getattr(args, 'lam_cell', 1.) * cell + mw * (args.lam_region * region + args.lam_patient * patient),
                (cell.item(), region.item(), patient.item()))
    region = crps_samples(S, S0)
    onehot = F.one_hot(gid, ngroups).float()
    w = onehot / onehot.sum(0, keepdim=True)
    patient = crps_samples(torch.einsum('mbd,bg->mgd', S, w), torch.einsum('bd,bg->gd', S0, w))
    return getattr(args, 'lam_cell', 1.) * cell + args.lam_region * region + args.lam_patient * patient, (cell.item(), region.item(), patient.item())


def train(crops_tr, crops_va, x, y, cells, args, scale, mean_net=None):
    ms = args.seed if args.model_seed is None else args.model_seed
    torch.manual_seed(ms + 2)
    rng = np.random.default_rng(ms + 3)
    scales = tuple(float(v) for v in args.scales.split(',')) if args.scales else ()
    net = Ligature(x.shape[1], d=args.d, layers=args.layers, frozen_mean=args.frozen_mean, dropout=args.dropout,
                  scales=scales, patient_dim=args.patient_dim, region_dim=args.region_dim, rank=args.rank,
                  stationary=args.stationary, mean_correction=args.mean_correction, tail_df=args.tail_df)
    net.no_stain_input, net.no_geom = args.no_stain_input, args.no_geom
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    per_case = pd.Series([c['case'] for c in crops_tr]).value_counts()
    steps = args.epochs * math.ceil(int(np.ceil(per_case.to_numpy() / args.per).sum()) / args.groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps + 10, pct_start=.1)
    best, best_state, hist = float('inf'), None, []
    for epoch in range(args.epochs):
        net.train()
        tot, parts, nb = 0., np.zeros(3), 0
        for gb in group_batches(crops_tr, rng, args.groups, args.per):
            flat = [c for g in gb for c in g]
            gid = torch.as_tensor(np.repeat(np.arange(len(gb)), [len(g) for g in gb]))
            loss, pr = loss_fn(net, flat, gid, len(gb), x, y, cells, args, scale, mean_net=mean_net)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item(); parts += pr; nb += 1
        net.eval()
        vrng = np.random.default_rng(11)
        gen = torch.Generator().manual_seed(5)
        vt, vn = 0., 0
        with torch.no_grad():
            for gb in group_batches(crops_va, vrng, args.groups, args.per):
                flat = [c for g in gb for c in g]
                gid = torch.as_tensor(np.repeat(np.arange(len(gb)), [len(g) for g in gb]))
                vt += loss_fn(net, flat, gid, len(gb), x, y, cells, args, scale, gen=gen, mean_net=mean_net)[0].item(); vn += 1
        val = vt / vn
        hist.append(dict(epoch=epoch, train=tot / nb, cell=parts[0] / nb, region=parts[1] / nb,
                         patient=parts[2] / nb, val=val))
        log(f'ligature epoch {epoch}: train {tot / nb:.4f} (cell {parts[0] / nb:.4f} region {parts[1] / nb:.4f} '
            f'patient {parts[2] / nb:.4f}) val {val:.4f}')
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, hist


def calibrate_scales(net, crops_va, x, y, cells, args, scale, mean_net=None, epochs=4, weights=None):
    """Fit the variance-component log-scales on validation patients, network frozen.

    weights = (cell, region, patient) multipliers of the three score levels; by
    default the training objective.
    """
    if weights is not None:
        args = argparse.Namespace(**{**vars(args), 'lam_region': weights[1] / max(weights[0], 1e-12),
                                     'lam_patient': weights[2] / max(weights[0], 1e-12)})
        cell_w = weights[0]
    else:
        cell_w = 1.
    for p in net.parameters():
        p.requires_grad_(False)
    net.log_scale.requires_grad_(True)
    net.eval()
    opt = torch.optim.Adam([net.log_scale], lr=.05)
    rng = np.random.default_rng(args.seed + 17)
    for epoch in range(epochs):
        tot, nb = 0., 0
        for gb in group_batches(crops_va, rng, args.groups, args.per):
            flat = [c for g in gb for c in g]
            gid = torch.as_tensor(np.repeat(np.arange(len(gb)), [len(g) for g in gb]))
            loss, _ = loss_fn(net, flat, gid, len(gb), x, y, cells, args, scale, mean_net=mean_net)
            loss = loss * cell_w
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        log(f'scale calibration epoch {epoch}: {tot / nb:.4f} scales {torch.exp(net.log_scale).detach().numpy().round(3)}')
    net.log_scale.requires_grad_(False)
    return net


@torch.no_grad()
def evaluate(net, mean_net, crops, x, y, cells, args, K, unseen=None):
    """Per-region statistics of K joint samples; sample k of a patient shares eta_k."""
    net.eval()
    gen = torch.Generator().manual_seed(args.seed + 99)
    cases = sorted({c['case'] for c in crops})
    eta = {c: net.latent((K, 1, net.dims['patient']), gen) for c in cases}
    fm = se.predict_mean(mean_net, crops, x, y)
    out = dict(truth=[], plugin=[], plugin_joint=[], samples=[])
    extra = []
    for c in crops:
        X, Y, D, A, M, NP = collate([c], x, y)
        pos = positions([c], cells)
        f = torch.from_numpy(fm[c['crop']])[None]
        mu, L = net.encode(X, D, M, f)
        z = net.noise(K, pos, M, torch.zeros(1, dtype=torch.long), 1, gen=gen, eta=eta[c['case']])
        smp = net.sample(mu, L, z)  # K,1,n,P
        Mf = M.float()
        out['samples'].append(crop_stats(smp, A, NP, Mf)[:, 0].numpy())
        out['truth'].append(crop_stats(Y, A, NP, Mf)[0].numpy())
        out['plugin_joint'].append(crop_stats(mu, A, NP, Mf)[0].numpy())
        out['plugin'].append(crop_stats(f, A, NP, Mf)[0].numpy())
        if unseen is not None:
            thr, thr_q = unseen
            fA, fNP = fine_adjacency(cells, c)
            st = {k: unseen_statistics(v, A[0], NP[0], fA, fNP, t, IU) for k, v, t in
                  [('truth', Y, thr), ('plugin', f, thr), ('plugin_q', f, thr_q), ('samples', smp[:, 0], thr)]}
            extra.append(dict(crop=c['crop'], case=c['case'], valid_copos=st['truth']['valid_copos'],
                              valid_fine=st['truth']['valid_fine'],
                              **{f'{fam}_{k}': st[k][fam] for k in ['truth', 'plugin', 'plugin_q', 'samples']
                                 for fam in ['frac', 'copos', 'fine']}))
    return {k: np.stack(v) for k, v in out.items()}, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--out', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature'))
    ap.add_argument('--mean-from', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/spatial_engression'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=271027)
    ap.add_argument('--pcadim', type=int, default=128)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--groups', type=int, default=4)
    ap.add_argument('--per', type=int, default=4)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--lam-region', type=float, default=1.)
    ap.add_argument('--lam-patient', type=float, default=1.)
    ap.add_argument('--eval-samples', type=int, default=64)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--q', type=float, default=.85)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--frozen-mean', action='store_true')
    ap.add_argument('--mean-correction', action='store_true')
    ap.add_argument('--dropout', type=float, default=0.)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--scales', type=str, default='10,20,40,80', help='GRF length scales in um; empty for none')
    ap.add_argument('--patient-dim', type=int, default=4)
    ap.add_argument('--recalibrate', default='', help='cell,region,patient weights: reload ligature_uncalibrated.pt, refit scales only')
    ap.add_argument('--tag', default='')
    ap.add_argument('--mean-kind', choices=['transformer', 'ridge', 'file'], default='transformer')
    ap.add_argument('--mean-file', default='runs/unet_stain/fold{fold}/cell_pred.npy')
    ap.add_argument('--stationary', action='store_true')
    ap.add_argument('--region-dim', type=int, default=4)
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--model-seed', type=int, default=None)
    ap.add_argument('--tail-df', type=int, default=0)
    ap.add_argument('--objective', choices=['crps', 'ml', 'variogram', 'moment'], default='crps',
                    help='ml: exact Gaussian likelihood of cells; variogram: per-cell CRPS + variogram score on cells (no statistic-level scores)')
    ap.add_argument('--calib-split', action='store_true')
    ap.add_argument('--calib-patients', type=int, default=0, help='with --recalibrate: use only this many validation patients')
    ap.add_argument('--lam-cell', type=float, default=1.)
    ap.add_argument('--no-contrasts', action='store_true')
    ap.add_argument('--vs-weight', type=float, default=1., help='weight of the variogram score (objective variogram)')
    ap.add_argument('--moment-weight', type=float, default=1., help='weight of the statistic terms (objective moment)')
    ap.add_argument('--vs-unbiased', action='store_true', help='unbiased finite-ensemble variogram score')
    ap.add_argument('--score-within-cell', action='store_true',
                    help='also score region means of within-cell marker products (identifies within-cell covariance)')
    ap.add_argument('--no-stain-input', action='store_true')
    ap.add_argument('--no-geom', action='store_true')
    ap.add_argument('--train-patients', type=int, default=0)
    args = ap.parse_args()
    # Accept colon-separated scales and 'none' for no spatial fields.
    args.scales = '' if args.scales.lower() == 'none' else args.scales.replace(':', ',')
    torch.set_num_threads(args.threads)
    out = args.out / f'fold{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'args{args.tag}.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    cells, x, y, crops, roles = se.load(args.root, args.fold, args.seed, args.pcadim)
    (out / 'roles.json').write_text(json.dumps(roles, indent=2))
    tr = [c for c in crops if c['role'] == 'train']
    if args.train_patients:  # ablation: residual model trained on fewer patients (the stain is unchanged)
        tc = sorted({c['case'] for c in tr})
        keep = set(np.random.default_rng(args.seed + 5 + args.fold).choice(tc, min(args.train_patients, len(tc)), replace=False))
        tr = [c for c in tr if c['case'] in keep]
    va = [c for c in crops if c['role'] == 'val']
    te = [c for c in crops if c['role'] == 'test']
    cal = None
    if args.calib_split:
        # disjoint calibration patients: half of the validation patients only score
        # conformal intervals; the other half do early stopping and scale fitting
        vc = sorted({c['case'] for c in va})
        keep = set(vc[:len(vc) // 2])
        cal = [c for c in va if c['case'] not in keep]
        va = [c for c in va if c['case'] in keep]
        log(f'calibration split: {len(keep)} validation patients, {len(vc) - len(keep)} conformal patients')
    if args.smoke:
        tr, va, te = tr[:args.smoke], va[:args.smoke // 4], te[:args.smoke // 4]
    S0 = []
    for c in tr:
        S0.append(contrasts(crop_stats(torch.from_numpy(y[c['ii']])[None], torch.from_numpy(c['A'])[None],
                                       torch.from_numpy(c['npairs'])[None], torch.ones(1, len(c['ii'])))))
    scale = torch.cat(S0).std(0).clamp(min=1e-3)
    if args.score_within_cell:
        W0 = torch.cat([within_cell(torch.from_numpy(y[c['ii']])[None], torch.ones(1, len(c['ii']))) for c in tr])
        scale = torch.cat([scale, W0.std(0).clamp(min=1e-3)])
    # deterministic plug-in baseline: reuse the per-cell squared-loss model of the main run
    # the frozen stain keeps its own architecture (from its run), whatever Ligature's width/depth
    ma = args.mean_from / f'fold{args.fold}' / 'args.json'
    sa = json.loads(ma.read_text()) if ma.exists() else {}
    mean_net = se.CellGraphNet(x.shape[1], d=int(sa.get('d', args.d)), layers=int(sa.get('layers', args.layers)))
    mp = args.mean_from / f'fold{args.fold}' / 'mean.pt'
    if args.mean_kind == 'ridge':
        mean_net = fit_ridge_mean(x, y, tr, va)
    elif args.mean_kind == 'file':
        F = np.load(str(args.mean_file).format(fold=args.fold)).astype(np.float32)
        assert F.shape == (len(x), P) and np.isfinite(F).all(), 'external predictions must cover every cell'
        x = np.column_stack([x, F])
        mean_net = ColumnMean(x.shape[1] - P)
        log(f'external virtual stain {args.mean_file}: test-patient cell MSE '
            f'{float(np.mean((F[np.concatenate([c["ii"] for c in te])] - y[np.concatenate([c["ii"] for c in te])]) ** 2)):.4f}')
    elif mp.exists():
        mean_net.load_state_dict(torch.load(mp))
        log(f'loaded plug-in model {mp}')
    else:
        targs = argparse.Namespace(**{**vars(args), 'cell_noise': 0, 'group_noise': 0, 'late_noise': False})
        mean_net, _ = se.train_model('mean', tr, va, x, y, targs, stat_scale(tr, y))
        torch.save(mean_net.state_dict(), out / 'mean.pt')
    mean_net.eval()
    if args.recalibrate:
        w = tuple(float(v) for v in args.recalibrate.replace(':', ',').split(','))  # Colon-separated weights are also accepted.
        scales = tuple(float(v) for v in args.scales.split(',')) if args.scales else ()
        net = Ligature(x.shape[1], d=args.d, layers=args.layers, frozen_mean=args.frozen_mean, dropout=args.dropout,
                      scales=scales, patient_dim=args.patient_dim, region_dim=args.region_dim, rank=args.rank,
                      stationary=args.stationary, mean_correction=args.mean_correction, tail_df=args.tail_df)
        net.no_stain_input, net.no_geom = args.no_stain_input, args.no_geom
        missing, unexpected = net.load_state_dict(torch.load(out / 'ligature_uncalibrated.pt'), strict=False)
        assert set(missing) <= {'const_load'} and not unexpected, (missing, unexpected)  # older checkpoints lack the unused const_load
        va_cal = va
        if args.calib_patients:
            # how many labelled patients does calibration need: use only k validation patients
            vc = sorted({c['case'] for c in va})
            keep = set(np.random.default_rng(args.seed + 99 + args.fold).choice(vc, min(args.calib_patients, len(vc)), replace=False))
            va_cal = [c for c in va if c['case'] in keep]
            log(f'calibrating on {len(keep)} of {len(vc)} validation patients: {sorted(keep)}')
        # the loss scales cell CRPS by 1 and the others by lam; emulate weights with a tiny cell weight
        net = calibrate_scales(net, va_cal, x, y, cells, args, scale, mean_net=mean_net,
                               weights=(max(w[0], 1e-6), w[1], w[2]))
        hist = []
    else:
        net, hist = train(tr, va, x, y, cells, args, scale, mean_net=mean_net)
        torch.save(net.state_dict(), out / 'ligature_uncalibrated.pt')
    if args.calibrate and not args.recalibrate:
        # keep the uncalibrated generator's validation/test outputs for the ablation
        res_u, _ = evaluate(net, mean_net, te, x, y, cells, args, args.eval_samples)
        np.savez_compressed(out / 'test_stats_uncalibrated.npz', crop=np.array([c['crop'] for c in te]),
                            case=np.array([c['case'] for c in te]), eligible=np.array([c['eligible'] for c in te]),
                            **{k: v.astype(np.float32) for k, v in res_u.items()})
        net = calibrate_scales(net, va, x, y, cells, args, scale, mean_net=mean_net)
    torch.save(net.state_dict(), out / f'ligature{args.tag}.pt')
    if hist:
        pd.DataFrame(hist).to_csv(out / 'ligature_history.csv', index=False)
    train_idx = np.flatnonzero(cells.case.map(roles).to_numpy() == 'train')
    thr = torch.from_numpy(np.quantile(y[train_idx], args.q, axis=0).astype(np.float32))
    fv = se.predict_mean(mean_net, va, x, y)
    thr_q = torch.from_numpy(np.quantile(np.concatenate([fv[c['crop']] for c in va]), args.q, axis=0).astype(np.float32))
    rows = []
    for split, cs in [('val', cal if cal is not None else va), ('test', te)]:
        res, extra = evaluate(net, mean_net, cs, x, y, cells, args, args.eval_samples,
                              unseen=(thr, thr_q) if split == 'test' else None)
        np.savez_compressed(out / f'{split}_stats{args.tag}.npz', crop=np.array([c['crop'] for c in cs]),
                            case=np.array([c['case'] for c in cs]), eligible=np.array([c['eligible'] for c in cs]),
                            ncell=np.array([len(c['ii']) for c in cs]),
                            **{k: v.astype(np.float32) for k, v in res.items()})
        if extra:
            ex = {k: np.stack([r[k] for r in extra]) for k in extra[0] if k not in ('crop', 'case')}
            np.savez_compressed(out / f'unseen_stats{args.tag}.npz', crop=np.array([r['crop'] for r in extra]),
                                case=np.array([r['case'] for r in extra]), **ex)
        fm = se.predict_mean(mean_net, cs, x, y)
        with torch.no_grad():
            for c in cs:
                X, Y, D, A, M, NP = collate([c], x, y)
                mu, _ = net.encode(X, D, M, torch.from_numpy(fm[c['crop']])[None])
                rows.append(dict(split=split, crop=c['crop'], case=c['case'], ncell=len(c['ii']),
                                 mse_mean=float(((fm[c['crop']] - y[c['ii']]) ** 2).mean()),
                                 mse_jointmean=float(((mu[0].numpy() - y[c['ii']]) ** 2).mean())))
    if args.recalibrate and args.tag:
        log('DONE (recalibration)')
        return
    pd.DataFrame(rows).to_csv(out / 'cell_mse.csv', index=False)
    (out / 'complete.json').write_text(json.dumps(dict(fold=args.fold, test_cases=sorted({c['case'] for c in te}))))
    log('DONE')


if __name__ == '__main__':
    main()
