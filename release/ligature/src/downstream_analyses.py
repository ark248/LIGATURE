#!/usr/bin/env python
"""Downstream uses of patient-level spatial statistics, evaluated on test patients.

1. Cohort association study: Spearman correlation across all 40 patients between
   two spatial biomarkers. The plug-in uses virtual-stain values as if measured
   (Fisher-z 90% interval, n = 40). Ligature propagates uncertainty by multiple
   imputation: correlation recomputed on each joint sample (shared patient
   latent), with the 5-95% range of the draws as the interval.
2. Measurement triage: rank patients by Ligature's predictive spread for one
   statistic, measure the top fraction, and compare remaining cohort error
   against measuring a random fraction.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_spatial_engression import qv, PAIRNAMES, PREDEF, canonical


def patient_draws(run):
    """Returns truth [N,Q], plugin [N,Q], draws [N,K,Q] over all test patients of all folds."""
    T, Pp, D, cases = [], [], [], []
    for fd in sorted(run.glob('fold*')):
        if not (fd / 'complete.json').exists():
            continue
        z = np.load(fd / 'test_stats.npz')
        e = z['eligible']
        for c in np.unique(z['case']):
            m = (z['case'] == c) & e
            if not m.any():
                continue
            T.append(qv(z['truth'][m]).mean(0))
            Pp.append(qv(z['plugin'][m]).mean(0))
            D.append(qv(z['samples'][m]).mean(0))
            cases.append(c)
    return np.array(T), np.array(Pp), np.array(D), cases


def fisher_interval(r, n, z=1.645):
    f = np.arctanh(np.clip(r, -.999, .999))
    se = 1.06 / np.sqrt(n - 3)  # Spearman (Fieller et al.) approximation
    return np.tanh(f - z * se), np.tanh(f + z * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/spatial_engression'))
    ap.add_argument('--triage', type=float, default=.2)
    args = ap.parse_args()
    T, Pp, D, cases = patient_draws(args.run)
    n = len(cases)
    idx = [PAIRNAMES.index(canonical(q)) for q in PREDEF]
    # ---- association study over all pairs of the predefined statistics
    rows = []
    for i, j in itertools.combinations(idx, 2):
        rt = spearmanr(T[:, i], T[:, j]).correlation
        rp = spearmanr(Pp[:, i], Pp[:, j]).correlation
        lo_p, hi_p = fisher_interval(rp, n)
        draws = np.array([spearmanr(D[:, k, i], D[:, k, j]).correlation for k in range(D.shape[1])])
        lo_j, hi_j = np.quantile(draws, [.05, .95])
        rows.append(dict(a=PAIRNAMES[i], b=PAIRNAMES[j], r_true=rt, r_plugin=rp, r_ligature=draws.mean(),
                         cover_plugin=bool(lo_p <= rt <= hi_p), cover_ligature=bool(lo_j <= rt <= hi_j),
                         width_plugin=hi_p - lo_p, width_ligature=hi_j - lo_j,
                         sign_plugin=bool(np.sign(rp) == np.sign(rt)), sign_ligature=bool(np.sign(draws.mean()) == np.sign(rt))))
    assoc = pd.DataFrame(rows)
    assoc.to_csv(args.run / 'association_study.csv', index=False)
    # ---- triage over all 136 statistics
    rng = np.random.default_rng(0)
    k = max(1, int(round(args.triage * n)))
    tri = []
    for q in range(T.shape[1]):
        est = D[:, :, q].mean(1)
        sd = D[:, :, q].std(1)
        err = (est - T[:, q]) ** 2
        top = np.argsort(-sd)[:k]
        keep = np.ones(n, bool); keep[top] = False
        rand = np.mean([err[np.setdiff1d(np.arange(n), rng.choice(n, k, replace=False))].sum() for _ in range(200)])
        oracle = np.sort(err)[:n - k].sum()

        def rest(score):
            kp = np.ones(n, bool); kp[np.argsort(-score)[:k]] = False
            return err[kp].sum()
        # mean-conditioned controls: rankings that use only predicted values, and the part of
        # the spread not explained by the predicted magnitude (rank residual of sd on |est|)
        rs, ra = rankdata(sd), rankdata(np.abs(est))
        sd_resid = rs - np.polyval(np.polyfit(ra, rs, 1), ra)
        re_ = rankdata(np.sqrt(err))
        err_resid = re_ - np.polyval(np.polyfit(ra, re_, 1), ra)
        tri.append(dict(query=PAIRNAMES[q], rho_sd_abs_error=spearmanr(sd, np.sqrt(err)).correlation,
                        partial_rho_sd_abs_error=float(np.corrcoef(sd_resid, err_resid)[0, 1]),
                        remaining_sse_triage=err[keep].sum(), remaining_sse_random=rand,
                        remaining_sse_abs_mean=rest(np.abs(est)), remaining_sse_abs_plugin=rest(np.abs(Pp[:, q])),
                        remaining_sse_abs_correction=rest(np.abs(est - Pp[:, q])), remaining_sse_sd_resid=rest(sd_resid),
                        remaining_sse_oracle=oracle, total_sse=err.sum()))
    tri = pd.DataFrame(tri)
    tri.to_csv(args.run / 'triage.csv', index=False)
    out = dict(patients=n, association=dict(
        pairs=len(assoc), mean_abs_err_plugin=float((assoc.r_plugin - assoc.r_true).abs().mean()),
        mean_abs_err_ligature=float((assoc.r_ligature - assoc.r_true).abs().mean()),
        coverage_plugin=float(assoc.cover_plugin.mean()), coverage_ligature=float(assoc.cover_ligature.mean()),
        mean_width_plugin=float(assoc.width_plugin.mean()), mean_width_ligature=float(assoc.width_ligature.mean()),
        sign_agreement_plugin=float(assoc.sign_plugin.mean()),
        sign_agreement_ligature=float(assoc.sign_ligature.mean())),
        triage=dict(fraction=args.triage, measured=k,
                    mean_rho_sd_abs_error=float(tri.rho_sd_abs_error.mean()),
                    sse_reduction_triage=float(1 - (tri.remaining_sse_triage / tri.total_sse).mean()),
                    sse_reduction_random=float(1 - (tri.remaining_sse_random / tri.total_sse).mean()),
                    **{f'sse_reduction_{m}': float(1 - (tri[f'remaining_sse_{m}'] / tri.total_sse).mean())
                       for m in ['abs_mean', 'abs_plugin', 'abs_correction', 'sd_resid']},
                    mean_partial_rho_sd_abs_error_given_abs_mean=float(tri.partial_rho_sd_abs_error.mean()),
                    sse_reduction_oracle=float(1 - (tri.remaining_sse_oracle / tri.total_sse).mean())))
    (args.run / 'downstream_summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
