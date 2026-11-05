#!/usr/bin/env python
"""Main-text figures from analyze_spatial_engression.py outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_export import export_figure

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRAY, DARK, MUTED = '#8a8984', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': DARK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.linewidth': .6, 'legend.frameon': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--ablations', nargs='*', default=['abl_marginal', 'abl_cellonly', 'abl_nolatent'])
    ap.add_argument('--out', type=Path, default=Path('paper/fig_main'))
    args = ap.parse_args()
    s = pd.read_csv(args.run / 'query_summary.csv')
    pq = pd.read_csv(args.run / 'patient_queries.csv')
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.65))
    ax = ax.ravel()
    fig.subplots_adjust(left=.12, right=.97, bottom=.19, top=.94, hspace=.73, wspace=.48)

    # (a) attenuation: cohort-mean estimate vs cohort-mean truth, 136 statistics
    a = ax[0]
    lim = [min(s.truth_mean.min(), 0) - .02, s.truth_mean.max() + .03]
    a.plot(lim, lim, color=GRAY, lw=.8, ls='--', zorder=0)
    a.scatter(s.truth_mean, s.truth_mean + s.bias_plugin, s=9, color=ORANGE, label='plug-in', lw=0)
    a.scatter(s.truth_mean, s.truth_mean + s.bias_joint, s=9, color=BLUE, label='LIGATURE', lw=0)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel('measured cohort mean'); a.set_ylabel('estimated cohort mean')
    a.set_title('a  136 spatial statistics', loc='left', fontsize=10)
    a.legend(loc='upper left', fontsize=10, handletextpad=.2)

    # (b) normalized patient-level MSE by method
    b = ax[1]
    methods = [('plugin', 'plug-in', ORANGE), ('calibrated', 'calib.', AQUA),
               ('direct_graph_stain', 'graph + stain', GRAY), ('joint', 'LIGATURE', BLUE)]
    data = [np.log2(s[f'nmse_{m}'].clip(lower=1e-3)) for m, _, _ in methods]
    bp = b.boxplot(data, widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color=DARK, lw=1))
    for patch, (_, _, c) in zip(bp['boxes'], methods):
        patch.set_facecolor(c); patch.set_alpha(.85); patch.set_edgecolor('none')
    b.axhline(0, color=GRAY, lw=.6, ls='--')
    b.set_xticks(range(1, len(methods) + 1), [l for _, l, _ in methods], rotation=40, ha='right', rotation_mode='anchor', fontsize=10)
    b.set_ylabel('log$_2$ MSE / plug-in MSE')
    b.set_title('b  patient-level error', loc='left', fontsize=10)

    # (c) calibration curve: empirical vs nominal coverage of patient-level intervals
    c = ax[2]
    lv = [50, 80, 90, 95]
    summ = json.loads((args.run / 'summary.json').read_text())['all136']
    c.plot([45, 100], [45, 100], color=GRAY, lw=.6, ls='--')
    curves = [('LIGATURE', summ, BLUE, '-')]
    for name, lab, ls in [('ligature_v3_nolatent', 'no patient latent', '--'), ('ligature_v3_stationary', 'stationary loadings', '-.'),
                          ('ligature_v3_marginal', 'per-cell (mult. imputation)', ':')]:
        f = (Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs') / name / 'summary.json'
        if f.exists():
            curves.append((lab, json.loads(f.read_text())['all136'], GRAY, ls))
    for lab, d, col, ls in curves:
        c.plot(lv, [100 * d[f'cover_at{l}'] for l in lv], color=col, ls=ls, lw=1.4, marker='o', ms=3, label=lab)
    c.scatter([90], [100 * summ['cover_indep']], color=ORANGE, s=14, zorder=3, label='region-independent')
    c.scatter([90], [100 * summ['cover_recal']], facecolor='none', edgecolor=BLUE, s=22, lw=1, zorder=3,
              label='LIGATURE + val. inflation')
    c.set_xlim(45, 100); c.set_ylim(0, 100)
    c.set_xlabel('nominal coverage (%)'); c.set_ylabel('empirical coverage (%)')
    c.set_title('c  patient-level calibration', loc='left', fontsize=10)
    handles, labels = c.get_legend_handles_labels()
    labels = [x.replace('per-cell (mult. imputation)', 'per-cell residuals').replace('LIGATURE + val. inflation', 'LIGATURE + inflation') for x in labels]
    fig.legend(handles, labels, fontsize=10, loc='lower center', ncol=3, handlelength=1.15, columnspacing=.8, borderaxespad=.1)

    # (d) one cross-marker statistic: patients with intervals
    d = ax[3]
    q = 'CD68 : PD-L1' if 'CD68 : PD-L1' in set(pq['query']) else pq['query'].iloc[0]
    g = pq[pq['query'] == q].sort_values('truth')
    d.errorbar(g.truth, g.joint, yerr=[g.joint - g.lo, g.hi - g.joint], fmt='o', ms=2.5, color=BLUE,
               elinewidth=.6, capsize=0, label='LIGATURE')
    d.scatter(g.truth, g.plugin, s=7, color=ORANGE, label='plug-in', zorder=3, lw=0)
    lo, hi = min(g.truth.min(), g.lo.min()), max(g.truth.max(), g.hi.max())
    d.plot([lo, hi], [lo, hi], color=GRAY, lw=.8, ls='--', zorder=0)
    d.set_xlabel('measured, per patient'); d.set_ylabel('estimated')
    d.set_title(f'd  {q}', loc='left', fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_figure(fig, args.out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
