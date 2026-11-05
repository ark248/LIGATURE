#!/usr/bin/env python
"""Hidden spatial organization: fraction of each cohort-mean statistic missing from the plug-in.

measured  h = (T - P) / T      (needs the measured assay)
predicted h_hat = (J - P) / J  (Ligature, from H&E only; J = Ligature cohort-mean estimate)
over statistics with |T| > 0.02.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_export import export_figure
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_spatial_engression import BLUE, GRAY, DARK, MUTED
from spatial_engression import MARKERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--out', type=Path, default=Path('paper/fig_hidden'))
    args = ap.parse_args()
    s = pd.read_csv(args.run / 'query_summary.csv')
    s['P'] = s.truth_mean + s.bias_plugin
    s['J'] = s.truth_mean + s.bias_joint
    s['ok'] = s.truth_mean.abs() > .02
    s['h'] = (s.truth_mean - s.P) / s.truth_mean
    s['hhat'] = (s.J - s.P) / s.J
    g = s[s.ok]
    r, p = pearsonr(g.h, g.hhat)
    rho = spearmanr(g.h, g.hhat).correlation
    order = ['Pan-CK', 'E-cadherin', 'SMA', 'CD31', 'Ki67', 'CD45', 'CD3e', 'CD4', 'CD8a', 'FOXP3', 'CD45RO',
             'PD-1', 'CD20', 'CD68', 'CD163', 'PD-L1']
    idx = {m: k for k, m in enumerate(order)}
    H = np.full((16, 16), np.nan)
    for _, row in s.iterrows():
        a, b = row['query'].split(' : ')
        i, j = sorted([idx[a], idx[b]])
        if row.ok:
            H[j, i] = row.h          # lower triangle and diagonal: measured
            if i != j:
                H[i, j] = row.hhat   # upper triangle: predicted from H&E
    fig = plt.figure(figsize=(6.5, 4.8))
    ax = [fig.add_axes([.14,.34,.40,.58]), fig.add_axes([.69,.34,.28,.58])]
    cmap = plt.get_cmap('Blues').copy(); cmap.set_bad('#f1f0ec')
    im = ax[0].imshow(np.clip(H, 0, .85), cmap=cmap, vmin=0, vmax=.85)
    ax[0].set_xticks(range(16), order, rotation=90, fontsize=10); ax[0].set_yticks(range(16), order, fontsize=10)
    ax[0].plot([-.5, 15.5], [-.5, 15.5], color=DARK, lw=.6)
    ax[0].set_title('a  measured (lower, diagonal)\n    LIGATURE estimate (upper)', loc='left', fontsize=10)
    cax = fig.add_axes([.14,.12,.36,.025])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.ax.tick_params(labelsize=10)
    cb.set_label('hidden fraction', fontsize=10)
    b = ax[1]
    diag = g['query'].str.split(' : ').apply(lambda t: t[0] == t[1])
    b.plot([-.2, 1], [-.2, 1], color=GRAY, lw=.6, ls='--')
    b.scatter(g.h[~diag], g.hhat[~diag], s=9, color=GRAY, lw=0, label='cross-marker')
    b.scatter(g.h[diag], g.hhat[diag], s=14, color=BLUE, lw=0, label='same-marker')
    offs = {'FOXP3': (-22, 4), 'PD-L1': (4, -4), 'CD20': (4, -6), 'Pan-CK': (4, -2), 'E-cadherin': (4, -20)}
    for _, row in g[diag].iterrows():
        m = row['query'].split(' : ')[0]
        if m in offs:
            b.annotate(m, (row.h, row.hhat), fontsize=10, color=MUTED, xytext=offs[m], textcoords='offset points')
    b.set_xlabel('measured hidden fraction'); b.set_ylabel('predicted from H&E (LIGATURE)')
    b.set_title(f'b  {len(g)} statistics, r = {r:.2f}', loc='left', fontsize=10)
    handles, labels = b.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, loc='lower right', bbox_to_anchor=(.97, .04))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_figure(fig, args.out)
    out = dict(n=int(len(g)), pearson=float(r), p=float(p), spearman=float(rho),
               same_marker=g[diag][['query', 'truth_mean', 'P', 'J', 'h', 'hhat']].round(3).to_dict('records'))
    args.out.with_name(args.out.name + '_data.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != 'same_marker'}))


if __name__ == '__main__':
    main()
