#!/usr/bin/env python
"""Collect the systematic ablations (runs/abl/*, runs/lung_abl/*) into one record.

For every configuration: patient-level MSE relative to the plug-in (all 136 statistics and
the 16 clustering statistics), mean Spearman correlation, median relative bias, CRPS, and
raw / inflated coverage of 90% intervals, plus a per-patient paired comparison with the
reference configuration (Wilcoxon on per-patient normalized squared error). Seed spread is
the range over the reference and its two reseeded copies; a change is flagged when it
exceeds that range. Writes research/ABLATIONS.md, paper/ablations.csv and
paper/appendix_tables/ablations_full.tex.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

GROUPS = {  # name -> (group, description)
    'full': ('reference', 'reference'),
    'seed2': ('reference', 'reference, seed 2'), 'seed3': ('reference', 'reference, seed 3'),
    'no_region_score': ('score', 'no region-level score'), 'no_patient_score': ('score', 'no patient-level score'),
    'no_cell_score': ('score', 'no cell-level score'), 'cell_only': ('score', 'cell-level score only'),
    'no_contrasts': ('score', 'bin contrasts not scored'),
    'no_fields': ('residual', 'no spatial fields'), 'scale20': ('residual', 'one field scale (20 um)'),
    'scales_fine': ('residual', 'fine scales only (10, 20 um)'), 'scales_coarse': ('residual', 'coarse scales only (40, 80 um)'),
    'no_region_latent': ('residual', 'no region latent'), 'no_patient_latent': ('residual', 'no patient latent'),
    'no_latents': ('residual', 'no region or patient latent'), 'stationary': ('residual', 'stationary loadings'),
    'rank1': ('residual', 'field rank 1'), 'rank8': ('residual', 'field rank 8'),
    'shallow': ('network', '1 transformer layer'), 'narrow': ('network', 'width 64'),
    'no_stain_input': ('network', 'residual network does not see the stain'),
    'no_geom': ('network', 'no morphology descriptors'), 'no_dropout': ('network', 'no dropout'),
    'samples4': ('training', '4 samples per region in the score'), 'train12': ('training', '12 training patients'),
    'train6': ('training', '6 training patients'), 'epochs5': ('training', '5 epochs'),
}


def per_patient(run):
    q = pd.read_csv(run / 'patient_queries.csv')
    base = q.groupby('query').apply(lambda g: ((g.plugin - g.truth) ** 2).mean()).rename('base')
    q = q.join(base, on='query')
    q['e'] = (q.joint - q.truth) ** 2 / q.base.clip(lower=1e-12)
    return q.groupby('case').e.mean(), q[q['diag'].astype(bool)].groupby('case').e.mean()


def collect(root, cohort):
    rows, ref = [], None
    for d in sorted(p for p in root.iterdir() if (p / 'summary.json').exists()):
        s = json.loads((d / 'summary.json').read_text())
        a, g = s['all136'], s['diag16']
        args = json.loads((d / 'fold0' / 'args.json').read_text())
        rows.append(dict(cohort=cohort, name=d.name, group=GROUPS.get(d.name, ('other', ''))[0],
                         description=GROUPS.get(d.name, ('', d.name))[1],
                         mse_all=a['nmse_joint'], mse_same=g['nmse_joint'], rho=a['rho_joint'],
                         bias=a['median_relbias_joint'], crps=a['crps_joint'], crps_gauss=a['crps_gaussplug'],
                         cover_raw=a['cover_joint'], cover_inflated=a['cover_recal'],
                         folds=sum((d / f'fold{k}' / 'complete.json').exists() for k in range(5)),
                         run=str(d)))
    return pd.DataFrame(rows), None


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    out = []
    for root, cohort, refdir in [((Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/abl'), 'ORION', (Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/abl/full')),
                                 ((Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/lung_abl'), 'lung', (Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/lung/ligature'))]:
        if not root.exists():
            continue
        df, _ = collect(root, cohort)
        if df.empty:
            continue
        if cohort == 'lung' and (refdir / 'summary.json').exists():
            s = json.loads((refdir / 'summary.json').read_text()); a, g = s['all136'], s['diag16']
            df = pd.concat([pd.DataFrame([dict(cohort='lung', name='full', group='reference', description=GROUPS['full'][1],
                                               mse_all=a['nmse_joint'], mse_same=g['nmse_joint'], rho=a['rho_joint'],
                                               bias=a['median_relbias_joint'], crps=a['crps_joint'], crps_gauss=a['crps_gaussplug'],
                                               cover_raw=a['cover_joint'], cover_inflated=a['cover_recal'], folds=5, run=str(refdir))]), df])
        if not (refdir / 'patient_queries.csv').exists():
            out.append(df); continue
        ra, rs = per_patient(refdir)
        p_all, p_same = [], []
        for r in df.itertuples():
            if r.run == str(refdir):
                p_all.append(np.nan); p_same.append(np.nan); continue
            ea, es = per_patient(Path(r.run))
            k = ra.index.intersection(ea.index)
            p_all.append(float(wilcoxon(ea[k], ra[k]).pvalue) if len(k) > 5 else np.nan)
            k = rs.index.intersection(es.index)
            p_same.append(float(wilcoxon(es[k], rs[k]).pvalue) if len(k) > 5 else np.nan)
        df['p_vs_ref_all'], df['p_vs_ref_same'] = p_all, p_same
        seeds = df[df.group == 'reference']
        lo, hi = seeds.mse_all.min(), seeds.mse_all.max()
        los, his = seeds.mse_same.min(), seeds.mse_same.max()
        df['outside_seed_range'] = ((df.mse_all < lo) | (df.mse_all > hi) | (df.mse_same < los) | (df.mse_same > his)) & (df.group != 'reference')
        out.append(df)
    d = pd.concat(out, ignore_index=True)
    order = {g: i for i, g in enumerate(['reference', 'score', 'residual', 'network', 'training', 'other'])}
    d['_c'] = d.cohort.map({'ORION': 0, 'lung': 1}); d['_g'] = d.group.map(order).fillna(9)
    d = d.sort_values(['_c', '_g', 'name']).drop(columns=['_c', '_g']).reset_index(drop=True)
    Path('paper').mkdir(exist_ok=True)
    d.to_csv('paper/ablations.csv', index=False)
    fmt = lambda v, f='{:.2f}': '--' if pd.isna(v) else f.format(v)
    L = [r'\begin{tabular}{@{}llcccccc@{}}', r'\toprule',
         r'Cohort & Variant & MSE all & MSE same & $\rho$ & bias & CRPS & cov.\ raw / infl. \\', r'\midrule']
    last = None
    for r in d.itertuples():
        if last is not None and (r.cohort, r.group) != last:
            L.append(r'\addlinespace')
        star = '$^*$' if r.outside_seed_range else ''
        L.append(f"{r.cohort if (r.cohort, r.group) != last else ''} & {r.description}{star} & {fmt(r.mse_all)} & {fmt(r.mse_same)} & "
                 f"{fmt(r.rho)} & {fmt(100 * r.bias, '{:+.0f}')}\\% & {fmt(r.crps, '{:.4f}')} & {fmt(100 * r.cover_raw, '{:.0f}')} / {fmt(100 * r.cover_inflated, '{:.0f}')} \\\\")
        last = (r.cohort, r.group)
    L += [r'\bottomrule', r'\end{tabular}']
    Path('paper/appendix_tables/ablations_full.tex').write_text('\n'.join(L) + '\n')
    md = ['# Ligature ablations', '', 'Generated by `release/ligature/src/collect_ablations.py` from `runs/abl/*` (ORION, 40 patients) and',
          '`runs/lung_abl/*` (lung Xenium, 19 patients). Each variant = reference configuration + one change,',
          'five patient-held-out rotations. MSE is patient-level MSE relative to the plug-in; `p` is a',
          'Wilcoxon test of per-patient normalized squared error against the reference; `*` marks a change',
          'outside the range of the three reference seeds (ORION) or differing from the single lung reference.', '',
          '| cohort | variant | group | MSE all | MSE same | rho | bias | CRPS | cov raw | cov infl | p all | p same | folds |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in d.itertuples():
        md.append(f"| {r.cohort} | {r.description}{' *' if r.outside_seed_range else ''} | {r.group} | {r.mse_all:.3f} | {r.mse_same:.3f} | {r.rho:.3f} | "
                  f"{100 * r.bias:+.1f}% | {r.crps:.4f} | {100 * r.cover_raw:.0f}% | {100 * r.cover_inflated:.0f}% | "
                  f"{fmt(getattr(r, 'p_vs_ref_all', np.nan), '{:.3g}')} | {fmt(getattr(r, 'p_vs_ref_same', np.nan), '{:.3g}')} | {r.folds} |")
    Path('research/ABLATIONS.md').write_text('\n'.join(md) + '\n')
    print(d[['cohort', 'name', 'mse_all', 'mse_same', 'rho', 'bias', 'crps', 'cover_raw', 'cover_inflated', 'folds']].round(3).to_string())


if __name__ == '__main__':
    main()
