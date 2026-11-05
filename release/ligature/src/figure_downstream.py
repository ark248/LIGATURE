#!/usr/bin/env python
"""Figure 3: unseen statistic families, cohort association study, measurement triage."""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downstream_analyses import patient_draws
from figures_spatial_engression import BLUE, ORANGE, AQUA, GRAY, DARK, MUTED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--out', type=Path, default=Path('paper/fig_downstream'))
    args = ap.parse_args()
    un = json.loads((args.run / 'unseen_summary.json').read_text())
    assoc = pd.read_csv(args.run / 'association_study.csv')
    fig = plt.figure(figsize=(6.5, 5.10))
    ax = [fig.add_axes([.12,.64,.49,.23]), fig.add_axes([.72,.64,.25,.23]), fig.add_axes([.12,.15,.84,.30])]

    a = ax[0]
    cs = json.loads((args.run / 'contact_summary.json').read_text())
    un['contact'] = {f'nmse_{m}': cs[f'mean_nmse_{m}'] for m in ['plugin', 'plugin_q', 'joint']}
    fams = [('contact', 'contact\n(20 µm)'), ('copos', 'co-positive\n(near−far)'), ('frac', 'positive\nfractions'),
            ('fine', 'fine radii\n(µm)')]
    w = .26
    for k, (m, lab, col) in enumerate([('plugin', 'plug-in', ORANGE), ('plugin_q', 'quantile-matched', AQUA),
                                         ('joint', 'LIGATURE', BLUE)]):
        a.bar(np.arange(len(fams)) + (k - 1) * w, [un[f][f'nmse_{m}'] for f, _ in fams], width=w - .03, color=col, label=lab)
    a.axhline(1, color=GRAY, lw=.6, ls='--')
    a.set_xticks(range(len(fams)), [l for _, l in fams], fontsize=10)
    a.set_ylabel('patient-level MSE / plug-in')
    a.set_title('a  statistics never scored\nin training', loc='left', fontsize=10)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, loc='lower center', ncol=3, borderaxespad=.1)

    b = ax[1]
    lim = [min(assoc[['r_true', 'r_plugin', 'r_ligature']].min()) - .05, 1.05]
    b.plot(lim, lim, color=GRAY, lw=.6, ls='--')
    b.scatter(assoc.r_true, assoc.r_plugin, s=8, color=ORANGE, lw=0, label='plug-in')
    b.scatter(assoc.r_true, assoc.r_ligature, s=8, color=BLUE, lw=0, label='LIGATURE')
    b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel('measured cohort\ncorrelation'); b.set_ylabel('estimated')
    b.set_title('b  association study\nnot significant', loc='left', fontsize=10)
    b.set_xticks([0, .5, 1])
    # Colors are identified by the shared figure legend; leave data unobstructed.

    c = ax[2]
    T, Pp, D, cases = patient_draws(args.run)
    n = len(cases)
    fr = np.arange(0, n + 1)
    tri, rnd, orc, mag = [], [], [], []
    for q in range(T.shape[1]):
        err = (D[:, :, q].mean(1) - T[:, q]) ** 2
        sd = D[:, :, q].std(1)
        tot = err.sum()
        order = np.argsort(-sd)
        tri.append([1 - err[order[k:]].sum() / tot for k in fr])
        om = np.argsort(-np.abs(Pp[:, q]))  # mean-conditioned control: magnitude of the plug-in value
        mag.append([1 - err[om[k:]].sum() / tot for k in fr])
        orc.append([1 - np.sort(err)[:n - k].sum() / tot for k in fr])
        rnd.append(fr / n)
    for curve, lab, col, ls in [(orc, 'oracle', GRAY, ':'), (tri, 'LIGATURE spread', BLUE, '-'), (mag, '|plug-in value|', GRAY, '-'), (rnd, 'random', ORANGE, '--')]:
        c.plot(100 * fr / n, 100 * np.mean(curve, 0), color=col, ls=ls, lw=1.4, label=lab)
    c.axvline(20, color=GRAY, lw=.5)
    c.set_xlabel('patients measured (%)'); c.set_ylabel('cohort error removed (%)')
    c.set_title('c  which patients to measure', loc='left', fontsize=10)
    c.legend(fontsize=10, loc='lower right')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_figure(fig, args.out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
