#!/usr/bin/env python
"""Ligature across virtual stains: one row per wrapped stain (paper/backbones.csv, .json).

All numbers are recomputed from each run's patient_queries.csv, cell_mse.csv and
contact_summary.json. For GigaTIME, statistics are restricted to the 9 ORION markers
with a directly matched GigaTIME channel (45 statistics); other stains use all 136.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = [('Cell-graph transformer (in-house)', 'ligature_v3', None),
        ('Per-cell ridge, no context (in-house)', 'ligature_v3_ridge', None),
        ('Pixel U-Net (in-house)', 'ligature_v3_unet', None),
        ('ROSIE (published, affine per marker)', 'ligature_v3_rosie', None),
        ('GigaTIME (published, linear head)', 'ligature_v3_gigatime',
         {'PD-1', 'CD4', 'CD68', 'CD20', 'CD3e', 'CD8a', 'PD-L1', 'Pan-CK', 'Ki67'}),
        ('GigaTIME fine-tuned on ORION (published + in-house)', 'ligature_v3_gigatime_ft', None)]


def summarize(df, markers):
    if markers is not None:
        keep = df['query'].str.split(' : ').apply(lambda t: t[0] in markers and t[1] in markers)
        df = df[keep]
    rows = []
    for q, g in df.groupby('query'):
        base = ((g.plugin - g.truth) ** 2).mean()
        tm = g.truth.mean()
        rows.append(dict(diag=g['diag'].iloc[0], truth_mean=tm, nmse=((g.joint - g.truth) ** 2).mean() / base,
                         nmse_rescale=((g.global_rescale - g.truth) ** 2).mean() / base,
                         rb_plug=(g.plugin - g.truth).mean() / tm, rb_joint=(g.joint - g.truth).mean() / tm,
                         crps_joint=g.crps_joint.mean(), crps_gauss=g.crps_gaussplug.mean()))
    s = pd.DataFrame(rows)
    big = s[s.truth_mean.abs() > .02]
    d = s[s['diag']]
    return dict(n_statistics=int(len(s)), nmse_all=float(s.nmse.mean()), nmse_same=float(d.nmse.mean()),
                nmse_rescale_all=float(s.nmse_rescale.mean()), nmse_rescale_same=float(d.nmse_rescale.mean()),
                relbias_plugin=float(big.rb_plug.median()), relbias_ligature=float(big.rb_joint.median()),
                crps_ligature=float(s.crps_joint.mean()), crps_gaussian_plugin=float(s.crps_gauss.mean()))


def main():
    out = []
    for label, run, markers in RUNS:
        p = (Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs') / run
        if not (p / 'patient_queries.csv').exists():
            continue
        r = dict(stain=label, run=run, **summarize(pd.read_csv(p / 'patient_queries.csv'), markers))
        cm = pd.concat([pd.read_csv(f) for f in sorted(p.glob('fold*/cell_mse.csv'))])
        r['cell_mse_test'] = float(cm[cm.split == 'test'].groupby('case').mse_mean.mean().mean())
        if (p / 'contact_summary.json').exists():
            c = json.loads((p / 'contact_summary.json').read_text())
            r.update(contact_nmse_quantile=c['mean_nmse_plugin_q'], contact_nmse_ligature=c['mean_nmse_joint'],
                     contact_rho_quantile=c['mean_rho_plugin_q'], contact_rho_ligature=c['mean_rho_joint'])
        out.append(r)
    df = pd.DataFrame(out)
    df.to_csv('paper/backbones.csv', index=False)
    Path('paper/backbones.json').write_text(json.dumps(out, indent=2))
    pd.set_option('display.width', 250)
    print(df.drop(columns=['run']).round(3).to_string(index=False))



def latex():
    rows = json.loads(Path('paper/backbones.json').read_text())
    pct = lambda v: f'${v * 100:+.0f}\\%$'.replace('+-', '-')
    lines = [r'\begin{tabular}{@{}lcccccccc@{}}', r'\toprule',
             r' & cell & \multicolumn{2}{c}{bias} & \multicolumn{2}{c}{MSE / plug-in} & CRPS & \multicolumn{2}{c}{contact MSE} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}',
             r'Virtual stain & MSE & plug-in & \method{} & all & same & \method{}/Gauss. & quant. & \method{} \\', r'\midrule']
    for r in rows:
        name = r['stain'].replace('(in-house)', '').replace('(published, affine per marker)', '').replace('(published, linear head)', '').strip()
        name = name.replace(' on ORION (published + in-house)', '')
        name = name + ('$^\\dagger$' if r['stain'].startswith('GigaTIME (published') else '')
        c = (f"{r['contact_nmse_quantile']:.2f} & {r['contact_nmse_ligature']:.2f}" if 'contact_nmse_ligature' in r else '-- & --')
        lines.append(f"{name} & {r['cell_mse_test']:.2f} & {pct(r['relbias_plugin'])} & {pct(r['relbias_ligature'])} & "
                     f"{r['nmse_all']:.2f} & {r['nmse_same']:.2f} & {r['crps_ligature'] / r['crps_gaussian_plugin']:.2f} & {c} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    Path('paper/table_backbones.tex').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
    latex()
