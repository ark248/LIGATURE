#!/usr/bin/env python
"""Figure 1: LIGATURE overview schematic (matplotlib, no data)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from figure_export import export_figure

BLUE, ORANGE, GRAY, DARK, MUTED = '#2a78d6', '#eb6834', '#8a8984', '#0b0b0b', '#52514e'
LIGHT = '#eef4fc'


def box(ax, x, y, w, h, text, fc='white', ec=MUTED, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.04', fc=fc, ec=ec, lw=.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=DARK,
            fontweight='bold' if bold else 'normal', linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, col=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=.8, color=col))


def main():
    fig, ax = plt.subplots(figsize=(6.5, 3.10))
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
    rng = np.random.default_rng(3)
    pts = rng.uniform([0.15, 0.55], [1.55, 2.45], size=(28, 2))
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < .38:
                ax.plot(*zip(pts[i], pts[j]), color=GRAY, lw=.4, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color='#c97fb0', edgecolor='white', lw=.4, zorder=2)
    ax.text(.85, 2.7, 'H&E features\n+ supplied\ncell geometry', ha='center', va='center', fontsize=10, color=DARK)
    box(ax, 2.0, 1.85, 1.75, .7, 'any frozen\nvirtual stain\n$f(H)$ (cell means)')
    box(ax, 2.0, .35, 1.75, 1.25, 'cell-graph\ntransformer\n$\\rightarrow$ loadings\n$A_\\ell(H), B(H)$\n$C(H), D(H)$')
    arrow(ax, 1.6, 2.1, 2.0, 2.2); arrow(ax, 1.6, 1.2, 2.0, 1.05)
    box(ax, 4.15, .4, 2.15, 2.25,
        '$Y_i = f_i + \\Sigma_{\\ell} A_{\\ell i}\\xi_{\\ell r}(x_i)$\n$+ B_i\\epsilon_i + C_i\\zeta_r$\n$+ D_i\\eta_g$\n\n'
        '$\\xi_\\ell$: Gaussian fields\n10–80 $\\mu$m\n$\\zeta_r$: region\n$\\eta_g$: patient\n\njoint samples\nof all cells',
        fc=LIGHT, ec=BLUE)
    arrow(ax, 3.75, 2.2, 4.15, 2.0); arrow(ax, 3.75, 1.0, 4.15, 1.1)
    box(ax, 6.65, 1.75, 1.6, .9, 'statistics $S(\\hat Y)$\nmeans, binned\nmoments,\ncontrasts')
    arrow(ax, 6.3, 1.9, 6.7, 2.1)
    box(ax, 8.5, 1.75, 1.4, .9, 'CRPS at\ncells,\nregions,\npatients', fc='#fdf0ea', ec=ORANGE)
    arrow(ax, 8.25, 2.2, 8.5, 2.2)
    ax.text(9.17, 1.55, 'training', ha='center', fontsize=10, color=ORANGE)
    box(ax, 6.7, .3, 3.2, 1.15, 'additional spatial queries\ncomputed from joint samples\nvalidate each query family\ncalibrate intervals separately')
    arrow(ax, 6.3, .85, 6.7, .85)
    ax.text(8.3, .10, 'inference', ha='center', fontsize=10, color=BLUE)
    out = Path('paper/fig_schematic')
    export_figure(fig, out)
    print('wrote', out)


if __name__ == '__main__':
    main()
