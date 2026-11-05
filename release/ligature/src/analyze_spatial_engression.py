#!/usr/bin/env python
"""Patient-level evaluation of spatial-statistic estimates from virtual multiplex.

Reads runs/<name>/fold*/{val,test}_stats.npz. All fitted calibrations use the
validation patients of the same fold only; metrics use test patients only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_engression import MARKERS, IU, P

K = len(IU[0])
PAIRNAMES = [MARKERS[a] + ' : ' + MARKERS[b] for a, b in zip(*IU)]
PREDEF = ['Pan-CK : CD8a', 'CD3e : PD-L1', 'CD8a : PD-L1', 'CD68 : PD-L1', 'FOXP3 : CD3e',
          'CD68 : CD163', 'CD3e : CD3e', 'Pan-CK : Pan-CK', 'PD-L1 : PD-L1']


def qv(S):
    return S[..., P:P + K] - S[..., P + 2 * K:P + 3 * K]


def canonical(name):
    a, b = name.split(' : ')
    for n in PAIRNAMES:
        x, y = n.split(' : ')
        if {x, y} == {a, b} and (a != b or x == y):
            return n
    raise KeyError(name)


def patient_avg(values, case, eligible):
    """values: [crops, ...] -> dict case -> mean over eligible crops (equal crop weight)."""
    out = {}
    for c in np.unique(case):
        m = (case == c) & eligible
        if m.any():
            out[c] = values[m].mean(0)
    return out


def crps_samples(s, y):
    """Sample CRPS per column; s: [K, Q], y: [Q]."""
    t1 = np.abs(s - y).mean(0)
    ss = np.sort(s, 0)
    k = s.shape[0]
    w = (2 * np.arange(1, k + 1) - k - 1)[:, None]
    t2 = (w * ss).sum(0) / (k * k)
    return t1 - t2


LEVELS = (.5, .8, .9, .95)
CONFORMAL_ALPHA = .15


def fold_records(fd, alpha=0.1, iid_draws=64, seed=0, test_file='test_stats.npz', val_file='val_stats.npz'):
    va = np.load(fd / val_file)
    te = np.load(fd / test_file)
    rng = np.random.default_rng(seed)
    ev, et = va['eligible'], te['eligible']
    Tv, Tt = qv(va['truth']), qv(te['truth'])
    Pv, Pt = qv(va['plugin']), qv(te['plugin'])
    Jt = qv(te['samples'])  # crops, Ksamp, Q
    # scalar per-query calibration on validation crops
    calib = np.empty_like(Pt)
    calib_v = np.empty_like(Pv)
    for q in range(K):
        X = np.column_stack([np.ones(ev.sum()), Pv[ev, q]])
        coef = np.linalg.lstsq(X, Tv[ev, q], rcond=None)[0]
        calib[:, q] = coef[0] + coef[1] * Pt[:, q]
        calib_v[:, q] = coef[0] + coef[1] * Pv[:, q]
    # multiplicative-only calibration (ratio of validation patient means)
    tv_p, pv_p = patient_avg(Tv, va['case'], ev), patient_avg(Pv, va['case'], ev)
    ratio = np.mean([tv_p[c] for c in tv_p], 0) / np.where(np.abs(np.mean([pv_p[c] for c in pv_p], 0)) > 1e-8,
                                                           np.mean([pv_p[c] for c in pv_p], 0), 1e-8)
    # global rescale: one slope shared by all statistics, fitted on validation regions
    gslope = float((Tv[ev] * Pv[ev]).sum() / (Pv[ev] ** 2).sum())
    # Gaussian predictive law around the plug-in with per-statistic variance from validation patients
    sig_p = np.sqrt(np.mean([(tv_p[c] - pv_p[c]) ** 2 for c in tv_p], 0))
    # stronger probabilistic baseline: Gaussian around the per-statistic calibrated (corrected) mean,
    # spread from validation-patient residuals of that corrected estimate
    cv_p = patient_avg(calib_v, va['case'], ev)
    sig_c = np.sqrt(np.mean([(tv_p[c] - cv_p[c]) ** 2 for c in tv_p], 0))
    zq = 1.6448536269514722
    # patient-level aggregation
    truth = patient_avg(Tt, te['case'], et)
    plug = patient_avg(Pt, te['case'], et)
    plugj = patient_avg(qv(te['plugin_joint']), te['case'], et)
    cal = patient_avg(calib, te['case'], et)
    direct = None
    if (fd / 'direct.npz').exists():
        dz = np.load(fd / 'direct.npz')
        assert (dz['test_crop'] == te['crop']).all()
        direct = patient_avg(qv(dz['test']), te['case'], et)
    dgraph = dgs = None
    if (fd / 'direct_graph.npz').exists():
        dz = np.load(fd / 'direct_graph.npz')
        assert (dz['test_crop'] == te['crop']).all()
        dgraph = patient_avg(qv(dz['test']), te['case'], et)
    if (fd / 'direct_graph_stain.npz').exists():
        dz = np.load(fd / 'direct_graph_stain.npz')
        assert (dz['test_crop'] == te['crop']).all()
        dgs = patient_avg(qv(dz['test']), te['case'], et)
    jsamp = patient_avg(Jt, te['case'], et)  # case -> [Ksamp, Q] (sample k shares patient latent)
    # one statistic-agnostic spread factor tau from validation patients: the
    # (1 - alpha) quantile of |truth - mean| / interval half-width over all
    # validation patients and all 136 statistics
    vs = patient_avg(qv(va['samples']), va['case'], ev)
    ratios = []
    for c in vs:
        m_ = vs[c].mean(0)
        lo_, hi_ = np.quantile(vs[c], [alpha / 2, 1 - alpha / 2], 0)
        half = np.where(tv_p[c] >= m_, hi_ - m_, m_ - lo_)
        ratios.append(np.abs(tv_p[c] - m_) / np.maximum(half, 1e-12))
    tau = float(np.quantile(np.concatenate(ratios), 1 - alpha))
    # statistic-averaged split conformal with finite-sample guarantee: scores
    # r_iq = |truth - mean| / sd of the patient's samples; tau_c is the smallest t
    # with (1/(n+1)) sum_i mean_q 1[r_iq <= t] >= 1 - alpha_c (infinite if unattainable)
    alpha_c = CONFORMAL_ALPHA
    rs = np.stack([np.abs(tv_p[c] - vs[c].mean(0)) / np.maximum(vs[c].std(0), 1e-12) for c in vs])
    n_val = len(rs)
    grid = np.sort(np.unique(rs))
    frac = np.array([(rs <= t).mean(1).sum() / (n_val + 1) for t in grid])
    ok = np.flatnonzero(frac >= 1 - alpha_c)
    tau_c = float(grid[ok[0]]) if len(ok) else float('inf')
    # same procedure for the plug-in, with a per-statistic scale from validation residuals
    rp = np.stack([np.abs(tv_p[c] - pv_p[c]) / np.maximum(sig_p, 1e-12) for c in tv_p])
    gp = np.sort(np.unique(rp))
    fp = np.array([(rp <= t).mean(1).sum() / (n_val + 1) for t in gp])
    okp = np.flatnonzero(fp >= 1 - alpha_c)
    tau_cp = float(gp[okp[0]]) if len(okp) else float('inf')
    # iid-residual crop-independent sampling is evaluated analytically: under iid
    # per-cell residuals the near-far query mean is unchanged, so its interval is
    # the plug-in plus independent crop noise. We use the joint model's per-crop
    # sample spread with crops permuted independently (destroys patient sharing)
    # as the "crop-independent" ablation instead, below.
    recs = []
    for c in truth:
        js = jsamp[c]
        # crop-independent ablation: for each crop draw sample index independently
        m = (te['case'] == c) & et
        crops_s = Jt[m]  # n_c, Ks, Q
        idx = rng.integers(0, crops_s.shape[1], size=(crops_s.shape[0], iid_draws))
        indep = np.take_along_axis(crops_s, idx[:, :, None], 1).mean(0)  # iid_draws, Q
        lo, hi = np.quantile(js, [alpha / 2, 1 - alpha / 2], 0)
        lo_i, hi_i = np.quantile(indep, [alpha / 2, 1 - alpha / 2], 0)
        mu_ = js.mean(0)
        sd_ = js.std(0)
        lo_r, hi_r = mu_ - tau * (mu_ - lo), mu_ + tau * (hi - mu_)
        lev = {f'cov{int(l * 100)}': ((truth[c] >= np.quantile(js, (1 - l) / 2, 0)) &
                                       (truth[c] <= np.quantile(js, 1 - (1 - l) / 2, 0))) for l in LEVELS}
        cr = crps_samples(js, truth[c])
        zz = (truth[c] - plug[c]) / np.maximum(sig_p, 1e-12)
        from scipy.stats import norm
        crps_g = sig_p * (zz * (2 * norm.cdf(zz) - 1) + 2 * norm.pdf(zz) - 1 / np.sqrt(np.pi))
        cov_g = np.abs(zz) <= zq
        zc = (truth[c] - cal[c]) / np.maximum(sig_c, 1e-12)
        crps_gc = sig_c * (zc * (2 * norm.cdf(zc) - 1) + 2 * norm.pdf(zc) - 1 / np.sqrt(np.pi))
        cr_i = crps_samples(indep, truth[c])
        for q in range(K):
            recs.append(dict(fold=fd.name, case=c, query=PAIRNAMES[q], diag=bool(IU[0][q] == IU[1][q]),
                             truth=truth[c][q], plugin=plug[c][q], plugin_joint=plugj[c][q],
                             calibrated=cal[c][q], ratio=plug[c][q] * ratio[q], joint=js[:, q].mean(),
                             direct=direct[c][q] if direct is not None else np.nan,
                             direct_graph=dgraph[c][q] if dgraph is not None else np.nan,
                             direct_graph_stain=dgs[c][q] if dgs is not None else np.nan,
                             lo=lo[q], hi=hi[q], lo_indep=lo_i[q], hi_indep=hi_i[q],
                             lo_recal=lo_r[q], hi_recal=hi_r[q], tau=tau,
                             cover_conf=bool(abs(truth[c][q] - mu_[q]) <= tau_c * sd_[q]), width_conf=2 * tau_c * sd_[q],
                             tau_conf=tau_c,
                             cover_conf_plugin=bool(abs(truth[c][q] - plug[c][q]) <= tau_cp * sig_p[q]),
                             width_conf_plugin=2 * tau_cp * sig_p[q],
                             **{k: bool(v[q]) for k, v in lev.items()},
                             crps_joint=cr[q], crps_indep=cr_i[q], crps_plugin=abs(plug[c][q] - truth[c][q]),
                             crps_calibrated=abs(cal[c][q] - truth[c][q]),
                             global_rescale=gslope * plug[c][q], crps_gaussplug=crps_g[q], cover_gaussplug=bool(cov_g[q]), crps_gausscal=crps_gc[q], cover_gausscal=bool(abs(zc[q]) <= zq),
                             width_gaussplug=2 * zq * sig_p[q]))
    return recs


def summarize(df, methods):
    rows = []
    for q, g in df.groupby('query'):
        r = dict(query=q, diag=g.diag.iloc[0], truth_mean=g.truth.mean(), truth_sd=g.truth.std())
        base = ((g.plugin - g.truth) ** 2).mean()
        for m in methods:
            r[f'mse_{m}'] = ((g[m] - g.truth) ** 2).mean()
            r[f'nmse_{m}'] = r[f'mse_{m}'] / base
            r[f'bias_{m}'] = (g[m] - g.truth).mean()
            r[f'rho_{m}'] = spearmanr(g.truth, g[m]).correlation if g[m].std() > 0 else np.nan
        r['cover_joint'] = ((g.truth >= g.lo) & (g.truth <= g.hi)).mean()
        r['cover_indep'] = ((g.truth >= g.lo_indep) & (g.truth <= g.hi_indep)).mean()
        r['cover_recal'] = ((g.truth >= g.lo_recal) & (g.truth <= g.hi_recal)).mean()
        r['cover_conf'] = g.cover_conf.mean()
        r['width_conf'] = g.width_conf.mean()
        r['cover_conf_plugin'] = g.cover_conf_plugin.mean()
        r['width_conf_plugin'] = g.width_conf_plugin.mean()
        for l in LEVELS:
            r[f'cover_at{int(l * 100)}'] = g[f'cov{int(l * 100)}'].mean()
        r['width_recal'] = (g.hi_recal - g.lo_recal).mean()
        r['width_joint'] = (g.hi - g.lo).mean()
        r['width_indep'] = (g.hi_indep - g.lo_indep).mean()
        for m in ['joint', 'indep', 'plugin', 'calibrated', 'gaussplug', 'gausscal']:
            r[f'crps_{m}'] = g[f'crps_{m}'].mean()
        r['cover_gaussplug'] = g.cover_gaussplug.mean()
        r['cover_gausscal'] = g.cover_gausscal.mean()
        r['width_gaussplug'] = g.width_gaussplug.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/spatial_engression'))
    ap.add_argument('--test-file', default='test_stats.npz')
    ap.add_argument('--val-file', default='val_stats.npz')
    ap.add_argument('--tag', default='')
    ap.add_argument('--conformal-alpha', type=float, default=None)
    args = ap.parse_args()
    global CONFORMAL_ALPHA
    if args.conformal_alpha is not None:
        CONFORMAL_ALPHA = args.conformal_alpha
    folds = sorted(d for d in args.run.glob('fold*') if (d / 'complete.json').exists() and (d / args.test_file).exists()
                   and (d / args.val_file).exists())
    recs = []
    for fd in folds:
        recs += fold_records(fd, test_file=args.test_file, val_file=args.val_file)
    df = pd.DataFrame(recs)
    df.to_csv(args.run / f'patient_queries{args.tag}.csv', index=False)
    methods = ['plugin', 'plugin_joint', 'ratio', 'calibrated', 'global_rescale', 'direct', 'direct_graph', 'direct_graph_stain', 'joint']
    s = summarize(df, methods)
    s.to_csv(args.run / f'query_summary{args.tag}.csv', index=False)
    pre = []
    for nm in PREDEF:  # predefined ORION biomarkers; absent for other marker panels
        try:
            pre.append(canonical(nm))
        except KeyError:
            pass
    out = {}
    for name, sel in [('all136', s['query'].notna()), ('offdiag120', ~s['diag'].astype(bool)), ('diag16', s['diag'].astype(bool)),
                      ('predefined9', s['query'].isin(pre))]:
        g = s[sel]
        out[name] = {k: float(g[k].mean()) for k in g.columns if k.startswith(('nmse_', 'rho_', 'cover_', 'width_', 'crps_'))}
        big = g[g.truth_mean.abs() > .02]
        out[name]['median_relbias_plugin'] = float((big.bias_plugin / big.truth_mean).median())
        out[name]['median_relbias_joint'] = float((big.bias_joint / big.truth_mean).median())
        out[name]['median_relbias_calibrated'] = float((big.bias_calibrated / big.truth_mean).median())
        out[name]['median_relbias_global_rescale'] = float((big.bias_global_rescale / big.truth_mean).median())
        out[name]['frac_queries_joint_beats_global_rescale'] = float((g.mse_joint < g.mse_global_rescale).mean())
        out[name]['frac_queries_joint_crps_beats_gaussplug'] = float((g.crps_joint < g.crps_gaussplug).mean())
        out[name]['frac_queries_joint_crps_beats_gausscal'] = float((g.crps_joint < g.crps_gausscal).mean())
        out[name]['median_nmse_joint'] = float(g.nmse_joint.median())
        out[name]['frac_queries_joint_beats_calibrated'] = float((g.mse_joint < g.mse_calibrated).mean())
        out[name]['frac_queries_joint_beats_direct'] = float((g.mse_joint < g.mse_direct).mean())
        out[name]['frac_queries_joint_rho_gt_plugin'] = float((g.rho_joint > g.rho_plugin).mean())
        out[name]['n_queries'] = int(len(g))
    out['folds'] = [f.name for f in folds]
    out['tau_by_fold'] = df.groupby('fold').tau.first().to_dict()
    out['conformal_alpha'] = CONFORMAL_ALPHA
    out['tau_conformal_by_fold'] = df.groupby('fold').tau_conf.first().to_dict()
    out['cover_conformal_by_fold'] = df.groupby('fold').cover_conf.mean().to_dict()
    out['cover_conformal_by_patient_min'] = float(df.groupby('case').cover_conf.mean().min())
    out['patients'] = int(df.case.nunique())
    cm = pd.concat([pd.read_csv(f / 'cell_mse.csv') for f in folds])
    cm = cm[cm.split == 'test']
    out['cell_mse'] = dict(mean_model=float(cm.groupby('case').mse_mean.mean().mean()),
                           joint_cellmean=float(cm.groupby('case').mse_jointmean.mean().mean()))
    (args.run / f'summary{args.tag}.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(s[s['query'].isin(pre)][['query', 'truth_mean', 'bias_plugin', 'bias_joint', 'nmse_calibrated', 'nmse_direct', 'nmse_joint',
                                   'rho_plugin', 'rho_calibrated', 'rho_direct', 'rho_joint', 'cover_joint', 'cover_indep']].round(3).to_string())


if __name__ == '__main__':
    main()
