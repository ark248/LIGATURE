#!/usr/bin/env python
"""Figure 1b: one held-out region, measured vs plug-in vs Ligature joint samples.

Region choice is automatic and fixed in advance: the test region of fold 0 with
the most cells among those whose measured Pan-CK near-minus-far clustering lies
closest to the cohort median (a typical, not a best-case, region).
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from figure_export import export_figure


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = m
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--marker', default='Pan-CK')
    ap.add_argument('--out', type=Path, default=Path('paper/fig_maps'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    snap = args.run / 'src' / 'ligature.py'  # per-run frozen snapshot if present
    t = module_from(snap if snap.exists() else Path(__file__).resolve().parent / 'ligature.py', 'ligature_frozen')
    se = t.se
    fd = args.run / f'fold{args.fold}'
    a = json.loads((fd / 'args.json').read_text())
    cells, x, y, crops, roles = se.load(args.root, args.fold, int(a['seed']), int(a['pcadim']))
    te = [c for c in crops if c['role'] == 'test' and c['eligible']]
    j = se.MARKERS.index(args.marker)
    k = len(se.IU[0])
    qi = [n for n, (p, q) in enumerate(zip(*se.IU)) if p == j and q == j][0]
    stats = []
    for c in te:
        S = se.crop_stats(torch.from_numpy(y[c['ii']])[None], torch.from_numpy(c['A'])[None],
                          torch.from_numpy(c['npairs'])[None], torch.ones(1, len(c['ii'])))[0].numpy()
        stats.append(S[se.P + qi] - S[se.P + 2 * k + qi])
    stats = np.array(stats)
    med = np.median(stats)
    near = np.argsort(np.abs(stats - med))[:15]
    c = te[max(near, key=lambda i: len(te[i]['ii']))]
    mean_net = se.CellGraphNet(x.shape[1], d=int(a['d']), layers=int(a['layers']))
    mean_net.load_state_dict(torch.load(Path(a['mean_from']) / f'fold{args.fold}' / 'mean.pt'))
    scales = tuple(float(v) for v in a['scales'].split(',')) if a['scales'] else ()
    net = t.Ligature(x.shape[1], d=int(a['d']), layers=int(a['layers']), frozen_mean=True,
                    scales=scales, patient_dim=int(a['patient_dim']))
    net.load_state_dict(torch.load(fd / 'ligature.pt'))
    mean_net.eval(); net.eval()
    X, Y, D, A, M, NP = se.collate([c], x, y)
    with torch.no_grad():
        f = mean_net(X, D, M)
        mu, L = net.encode(X, D, M, f)
        gen = torch.Generator().manual_seed(7)
        z = net.noise(3, t.positions([c], cells), M, torch.zeros(1, dtype=torch.long), 1, gen=gen)
        smp = net.sample(mu, L, z)[:, 0]
    xy = cells[['gx', 'gy']].to_numpy()[c['ii']] * .325
    xy = xy - xy.min(0)
    panels = [('measured', Y[0, :, j].numpy()), ('plug-in (per-cell mean)', f[0, :, j].numpy())] + \
             [(f'LIGATURE sample {s + 1}', smp[s, :, j].numpy()) for s in range(2)]
    vals = np.concatenate([v for _, v in panels])
    lo, hi = np.quantile(vals, [.02, .98])
    fig, ax = plt.subplots(1, 4, figsize=(6.5, 2.35))
    fig.subplots_adjust(left=.025, right=.82, bottom=.09, top=.69, wspace=.12)
    for axi, (name, v) in zip(ax, panels):
        sc = axi.scatter(xy[:, 0], xy[:, 1], c=v, s=9, cmap='Blues', vmin=lo, vmax=hi, lw=0)
        Sv = se.crop_stats(torch.from_numpy(v[None, :, None].repeat(se.P, -1).astype(np.float32)), A, NP, M.float())[0]
        q = float(Sv[se.P + qi] - Sv[se.P + 2 * k + qi])
        lines = name.replace('plug-in (per-cell mean)', 'plug-in\n(per-cell mean)').replace('LIGATURE sample ', 'LIGATURE\nsample ')
        if name == 'measured': lines = 'measured\n'
        axi.set_title(f'{lines}\nclustering {q:.2f}', fontsize=10)
        axi.set_aspect('equal'); axi.invert_yaxis(); axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_visible(False)
    cax = fig.add_axes([.855, .14, .025, .52])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(f'{args.marker}\nstandardized log intensity', fontsize=10)
    cb.ax.tick_params(labelsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_figure(fig, args.out)
    print('region', c['crop'], 'case', c['case'], 'cells', len(c['ii']), 'measured clustering', stats[te.index(c)])


if __name__ == '__main__':
    main()
