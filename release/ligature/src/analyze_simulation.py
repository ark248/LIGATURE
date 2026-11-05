#!/usr/bin/env python
"""Controlled simulations (simulate_cohort.py): what Ligature recovers when the law is known.

For each case (correct, withincell) over the five held-out rotations of runs/sim/<case>:
  * the Bayes statistic predictor (oracle): the plug-in statistic of f_true plus the exact
    expected residual second moment E[G(r) | image] under the simulated law (loadings regenerated
    from the simulation seed, exact squared-exponential kernels); the cross term has mean zero;
  * patient-level MSE relative to the plug-in and median relative bias of the 136 near-minus-far
    contrasts for the plug-in (= Bayes per-cell predictor), Ligature (calibrated, sample mean) and
    the oracle, on identical patient-statistic pairs;
  * Ligature's implied residual correlation (average of L L^T over test cells) against the true
    within-cell residual correlation (empirical, test cells), and the implied against the true
    0-25 um cross-cell correlation, for the two perturbed pairs and the other off-diagonal pairs
    (Proposition 4: the cross-cell quantity is identified by the scores, the within-cell one is not).
Writes runs/sim/simulation_summary.json and paper/appendix_tables/simulation.tex.
Run with SE_RAW_TARGETS=1 (simulated targets are on the modelling scale).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_engression as se
from spatial_engression import BINS, IU, MARKERS, P
from ligature import ColumnMean, Ligature, collate
from simulate_cohort import PAIRS

K = len(IU[0])
OFF = [q for q in range(K) if IU[0][q] != IU[1][q]]


def qidx(a, b):
    i, j = sorted((MARKERS.index(a), MARKERS.index(b)))
    return int(np.flatnonzero((IU[0] == i) & (IU[1] == j))[0])


def loadings_truth(truth, feat):
    """Regenerate the simulation's loadings A[k]: [n, P, rank] (same draw order as simulate_cohort)."""
    rng = np.random.default_rng(truth['seed'])
    S, r = len(truth['scales']), truth['rank']
    W = rng.normal(size=(S, P, r, 8)) / np.sqrt(8)
    C = rng.normal(size=(S, P, r))
    return [0.35 * np.tanh(np.einsum('prk,nk->npr', W[k], feat)) + 0.15 * C[k][None] for k in range(S)]


def expected_contrast(Ak, scales, xy, A, npairs, kernel='se'):
    """E[G_bin0(r) - G_bin2(r)] over the K pair coordinates, given loadings Ak[k]: [n,P,rank]."""
    d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    G = []
    for b in (0, 2):
        g = np.zeros((P, P))
        for k, ell in enumerate(scales):
            w = A[b] * (np.exp(-np.sqrt(d2) / ell) if kernel == 'exponential' else np.exp(-d2 / (2 * ell ** 2)))
            g += np.einsum('ij,ipr,jqr->pq', w, Ak[k], Ak[k], optimize=True)
        g /= max(npairs[b], 1)
        G.append(0.5 * (g + g.T))
    return (G[0] - G[1])[IU]


def model_cov(L, net, xy, A0):
    """Implied within-cell covariance (sum_i L_i L_i^T) and 0-25 um cross-cell covariance (sums)."""
    ns, r = len(net.scales), net.rank
    within = np.einsum('npk,nqk->pq', L, L)
    d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    cross = np.zeros((P, P))
    for k, ell in enumerate(net.scales):
        Lk = L[:, :, k * r:(k + 1) * r]
        cross += np.einsum('ij,ipr,jqr->pq', A0 * np.exp(-d2 / (2 * ell ** 2)), Lk, Lk, optimize=True)
    shared = slice(ns * r + net.dims['cell'], None)  # region and patient latents: shared by every cell
    Ls = L[:, :, shared]
    cross += np.einsum('ij,ipr,jqr->pq', A0, Ls, Ls, optimize=True)
    return within, 0.5 * (cross + cross.T)


def pct(v):
    return f"${round(100 * v):+d}\\%$".replace('+0', '0')


def corr(C):
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def run_case(case):
    run = ROOT / f'runs/sim/{case}'
    croot = ROOT / f'cohorts/sim_{case}'
    truth = json.loads((croot / 'truth.json').read_text())
    F = np.load(croot / 'f_true.npy').astype(np.float32)
    feat = None
    pat = []          # per-patient contrast predictions
    Rw = np.zeros((P, P)); nw = 0                       # true within-cell residual second moment
    Rc = np.zeros((P, P)); nc = 0                       # true 0-25 um cross-cell residual products
    Mw = np.zeros((P, P)); Mc = np.zeros((P, P))        # model-implied counterparts
    for fold in range(5):
        fd = run / f'fold{fold}'
        a = json.loads((fd / 'args.json').read_text())
        cells, x, y, crops, roles = se.load(croot, fold, int(a['seed']), int(a['pcadim']))
        if fold == 0:
            feat = x[:, :8].copy()  # the simulation's loadings used the fold-0 embedding
            Ak_all = loadings_truth(truth, feat)
        x = np.column_stack([x, F])
        mean_net = ColumnMean(x.shape[1] - P)
        scales = tuple(float(v) for v in a['scales'].split(',')) if a['scales'] else ()
        net = Ligature(x.shape[1], d=int(a['d']), layers=int(a['layers']), frozen_mean=True, scales=scales,
                      patient_dim=int(a['patient_dim']), region_dim=int(a['region_dim']), rank=int(a['rank']))
        missing, unexpected = net.load_state_dict(torch.load(fd / 'ligature.pt'), strict=False)
        assert set(missing) <= {'const_load'} and not unexpected, (missing, unexpected)
        net.eval()
        t = np.load(fd / 'test_stats.npz')
        te = {c['crop']: c for c in crops if c['role'] == 'test'}
        assert set(te) == set(t['crop'].tolist())
        xy = cells[['gx', 'gy']].to_numpy() * .325
        orc = np.zeros((len(t['crop']), K))
        with torch.no_grad():
            for j, cr in enumerate(t['crop']):
                c = te[cr]
                ii = c['ii']
                p = xy[ii]
                orc[j] = expected_contrast([A[ii] for A in Ak_all], truth['scales'], p, c['A'], c['npairs'], truth.get('kernel', 'se'))
                X, Y, D, Aa, M, NP = collate([c], x, y)
                mu, L = net.encode(X, D, M, mean_net(X, D, M))
                L = L[0].numpy().astype(np.float64)
                R = (y[ii] - F[ii]).astype(np.float64)
                Rw += R.T @ R; nw += len(ii)
                Rc += 0.5 * (R.T @ c['A'][0] @ R + (R.T @ c['A'][0] @ R).T); nc += c['npairs'][0]
                w_, c_ = model_cov(L, net, p, c['A'][0])
                Mw += w_; Mc += c_
        qv = lambda S: S[..., P:P + K] - S[..., P + 2 * K:P + 3 * K]
        tru, plg, smp = qv(t['truth']), qv(t['plugin']), qv(t['samples']).mean(1)
        orc = plg + orc
        for g in np.unique(t['case']):
            m = (t['case'] == g) & t['eligible']
            if m.any():
                pat.append(dict(case=g, truth=tru[m].mean(0), plugin=plg[m].mean(0), ligature=smp[m].mean(0), oracle=orc[m].mean(0)))
        print(case, 'fold', fold, 'done', flush=True)
    T = np.stack([p['truth'] for p in pat])
    out = dict(case=case, patients=len(pat), within_cell_truth_json=truth['within_cell_residual_corr'])
    base = ((np.stack([p['plugin'] for p in pat]) - T) ** 2).mean(0)
    big = np.abs(T.mean(0)) > .02
    diag = IU[0] == IU[1]
    for k in ['plugin', 'ligature', 'oracle']:
        E = np.stack([p[k] for p in pat])
        r = ((E - T) ** 2).mean(0) / base
        rb = ((E - T).mean(0) / T.mean(0))[big]
        out[k] = dict(mse_all=float(r.mean()), mse_same=float(r[diag].mean()), median_relbias=float(np.median(rb)))
    # cross-cell second moments are normalized by the within-cell variances (neighbour correlations)
    sr, sm = np.sqrt(np.diag(Rw / nw)), np.sqrt(np.diag(Mw / nw))
    cw, cc = corr(Rw / nw), (Rc / nc) / np.outer(sr, sr)
    mw, mc = corr(Mw / nw), (Mc / nc) / np.outer(sm, sm)
    pq = [qidx(a, b) for a, b in PAIRS]
    rest = [q for q in OFF if q not in pq]
    gw, gc = cw[IU], cc[IU]
    hw, hc = mw[IU], mc[IU]
    out['pairs'] = {f'{a}|{b}': dict(within_true=float(gw[q]), within_model=float(hw[q]),
                                     cross25_true=float(gc[q]), cross25_model=float(hc[q]))
                    for (a, b), q in zip(PAIRS, pq)}
    out['other_pairs'] = dict(n=len(rest), within_mae=float(np.abs(gw[rest] - hw[rest]).mean()),
                              cross25_mae=float(np.abs(gc[rest] - hc[rest]).mean()),
                              within_r=float(np.corrcoef(gw[rest], hw[rest])[0, 1]),
                              cross25_r=float(np.corrcoef(gc[rest], hc[rest])[0, 1]))
    return out


def main():
    assert os.environ.get('SE_RAW_TARGETS'), 'run with SE_RAW_TARGETS=1'
    torch.set_num_threads(4)
    res = {c: run_case(c) for c in ['correct', 'withincell', 'misspec'] if (ROOT / f'runs/sim/{c}/fold4/complete.json').exists()}
    (ROOT / 'runs/sim/simulation_summary.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    L = [r'\begin{tabular}{@{}llccc@{}}', r'\toprule',
         r'Simulated truth & Contrast estimator & MSE all & MSE same & bias \\', r'\midrule']
    names = dict(plugin=r'Plug-in of the true mean', ligature=r'\method{}', oracle=r'Bayes statistic predictor')
    cases = dict(correct='Covariance-matched class', withincell='Within-cell dependence added',
                 misspec='Misspecified')
    for ci, (case, lab) in enumerate((c, cases[c]) for c in res):
        if ci:
            L.append(r'\addlinespace')
        for i, k in enumerate(['plugin', 'ligature', 'oracle']):
            a = res[case][k]
            L.append(f"{lab if i == 0 else ''} & {names[k]} & {a['mse_all']:.2f} & {a['mse_same']:.2f} & {pct(a['median_relbias'])} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'', r'\medskip', r'',
          r'\begin{tabular}{@{}lcccccccc@{}}', r'\toprule',
          r' & \multicolumn{4}{c}{Covariance-matched class} & \multicolumn{4}{c}{Within-cell dependence added} \\',
          r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
          r' & \multicolumn{2}{c}{within cell} & \multicolumn{2}{c}{0--25\,\textmu m} & \multicolumn{2}{c}{within cell} & \multicolumn{2}{c}{0--25\,\textmu m} \\',
          r'Residual correlation & true & \method{} & true & \method{} & true & \method{} & true & \method{} \\', r'\midrule']
    for a, b in PAIRS:
        u, v = res['correct']['pairs'][f'{a}|{b}'], res['withincell']['pairs'][f'{a}|{b}']
        L.append(f"{a} / {b} & " + ' & '.join(f'${w[k]:+.2f}$' for w in (u, v) for k in
                                                ['within_true', 'within_model', 'cross25_true', 'cross25_model']) + r' \\')
    u, v = res['correct']['other_pairs'], res['withincell']['other_pairs']
    L.append(f"Other {u['n']} pairs, mean abs.\\ error & \\multicolumn{{2}}{{c}}{{{u['within_mae']:.2f}}} & \\multicolumn{{2}}{{c}}{{{u['cross25_mae']:.2f}}} & "
             f"\\multicolumn{{2}}{{c}}{{{v['within_mae']:.2f}}} & \\multicolumn{{2}}{{c}}{{{v['cross25_mae']:.2f}}} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (Path(__file__).resolve().parents[3] / 'paper/appendix_tables/simulation.tex').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
