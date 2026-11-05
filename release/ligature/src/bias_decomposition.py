#!/usr/bin/env python
"""Empirical form of the bias identity (Proposition 1) for each wrapped virtual stain.

For a region with measured cells y and stain predictions f (residual r = y - f), every
binned second moment is quadratic, so G(y) = G(f) + G(r) + X(f, r) with the cross term
X = G(y) - G(f) - G(r). For the conditional mean, X has expectation zero given the image
and G(r) is the residual cross-covariance that the plug-in omits. We evaluate the 136
near-minus-far contrasts used in the paper on test patients, average over each patient's
eligible regions, and report per statistic the cohort means of G(r) and X relative to the
measured cohort mean (statistics with |measured mean| > 0.02, as for the bias columns).

  per fold:   bias_decomposition.py --fold k --runs ligature_v3 ligature_v3_unet ...
  summarize:  bias_decomposition.py --summarize --runs ...
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def module_from(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def stain(t, se, a, fold, x, y, crops):
    kind = a.get('mean_kind', 'transformer')
    if kind == 'file':
        F = np.load(a['mean_file'].format(fold=fold)).astype(np.float32)
        return np.column_stack([x, F]), t.ColumnMean(x.shape[1])
    if kind == 'ridge':
        return x, t.fit_ridge_mean(x, y, [c for c in crops if c['role'] == 'train'],
                                   [c for c in crops if c['role'] == 'val'])
    net = se.CellGraphNet(x.shape[1], d=int(a['d']), layers=int(a['layers']))
    net.load_state_dict(torch.load(Path(a['mean_from']) / f'fold{fold}' / 'mean.pt'))
    return x, net


def per_fold(args):
    torch.set_num_threads(args.threads)
    loaded = {}
    for run in args.runs:
        rd = args.runs_dir / run
        snap = rd / 'src' / 'ligature.py'
        t = module_from(snap if snap.exists() else Path(__file__).resolve().parent / 'ligature.py', 'ligature_frozen')
        se = t.se
        a = json.loads((rd / f'fold{args.fold}' / 'args.json').read_text())
        key = (int(a['seed']), int(a['pcadim']))
        if key not in loaded:
            loaded = {key: se.load(args.root, args.fold, *key)}
        cells, x0, y, crops, roles = loaded[key]
        x, mean_net = stain(t, se, a, args.fold, x0, y, crops)
        mean_net.eval()
        te = [c for c in crops if c['role'] == 'test']
        with torch.no_grad():
            fm = se.predict_mean(mean_net, te, x, y)
        K = len(se.IU[0])
        out = {k: [] for k in ['truth', 'plugin', 'resid']}
        with torch.no_grad():
            for c in te:
                Y = torch.from_numpy(y[c['ii']]).float()
                Fp = torch.from_numpy(fm[c['crop']]).float()
                A, NP = torch.from_numpy(c['A'])[None].float(), torch.from_numpy(c['npairs'])[None].float()
                M = torch.ones(1, len(c['ii']))
                S = se.crop_stats(torch.stack([Y, Fp, Y - Fp])[:, None], A, NP, M)[:, 0].numpy()
                q = S[:, se.P:se.P + K] - S[:, se.P + 2 * K:se.P + 3 * K]
                for k, v in zip(out, q):
                    out[k].append(v)
        np.savez(rd / f'fold{args.fold}' / 'bias_decomposition.npz', case=np.array([c['case'] for c in te]),
                 eligible=np.array([bool(c['eligible']) for c in te]), **{k: np.stack(v) for k, v in out.items()})
        print(run, 'fold', args.fold, 'regions', len(te), flush=True)


def summarize(args):
    from analyze_spatial_engression import patient_avg
    res = {}
    for run in args.runs:
        rd = args.runs_dir / run
        T, Pl, R = [], [], []
        for fd in sorted(rd.glob('fold*')):
            f = fd / 'bias_decomposition.npz'
            if not f.exists():
                continue
            z = np.load(f)
            for lst, k in [(T, 'truth'), (Pl, 'plugin'), (R, 'resid')]:
                lst.extend(patient_avg(z[k], z['case'], z['eligible']).values())
        T, Pl, R = np.array(T), np.array(Pl), np.array(R)
        tm = T.mean(0)
        big = np.abs(tm) > .02
        X = T - Pl - R
        r_share, x_share = R.mean(0)[big] / tm[big], X.mean(0)[big] / tm[big]
        res[run] = dict(patients=len(T), statistics=int(big.sum()),
                        median_relbias_plugin=float(np.median((Pl.mean(0) - tm)[big] / tm[big])),
                        median_resid_share=float(np.median(r_share)), median_cross_share=float(np.median(x_share)),
                        frac_cross_negative=float(np.mean(x_share < 0)),
                        # per-patient (not cohort-mean) magnitudes relative to the between-patient spread
                        resid_rms_over_sd=float(np.median(np.sqrt((R ** 2).mean(0)) / T.std(0))),
                        cross_rms_over_sd=float(np.median(np.sqrt((X ** 2).mean(0)) / T.std(0))))
    (args.runs_dir / 'bias_decomposition.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--runs', nargs='+', default=['ligature_v3', 'ligature_v3_ridge', 'ligature_v3_unet', 'ligature_v3_rosie',
                                                  'ligature_v3_gigatime', 'ligature_v3_gigatime_ft'])
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--runs-dir', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs'), help='directory holding the runs (data come from --root)')
    args = ap.parse_args()
    summarize(args) if args.summarize else per_fold(args)


if __name__ == '__main__':
    main()
