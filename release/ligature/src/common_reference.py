#!/usr/bin/env python
"""All estimators on one scale: patient-level MSE relative to the transformer plug-in.

Every run evaluates the same 136 statistics on the same 40 test patients (identical
truth), so per-statistic MSE can be divided by the cell-graph transformer plug-in's MSE
and averaged. Writes paper/appendix_tables/common_reference.tex and paper/common_reference.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = [  # (group, label, run, column)
    ('No stain', 'Direct ridge on region features', 'ligature_v3', 'direct'),
    ('No stain', 'Direct graph (no stain)', 'ligature_v3', 'direct_graph'),
    ('No stain', '\\method{} around a constant stain', 'ligature_v3_const', 'joint'),
    ('No stain', '\\method{} around a constant stain, learned mean', 'ligature_v3_const_mc', 'joint'),
    ('Transformer', 'plug-in', 'ligature_v3', 'plugin'),
    ('Transformer', 'global rescale', 'ligature_v3', 'global_rescale'),
    ('Transformer', 'Direct graph (stain inputs)', 'ligature_v3', 'direct_graph_stain'),
    ('Transformer', 'regularized conditional Gaussian NLL', 'ligature_v3_ml:_uncal', 'joint'),
    ('Transformer', '\\method{}', 'ligature_v3', 'joint'),
    ('Ridge', 'plug-in', 'ligature_v3_ridge', 'plugin'),
    ('Ridge', '\\method{}', 'ligature_v3_ridge', 'joint'),
    ('U-Net', 'plug-in', 'ligature_v3_unet', 'plugin'),
    ('U-Net', '\\method{}', 'ligature_v3_unet', 'joint'),
    ('U-Net', '\\method{}, learned mean correction', 'ligature_v3_unet_mc', 'joint'),
    ('ROSIE', 'plug-in', 'ligature_v3_rosie', 'plugin'),
    ('ROSIE', '\\method{}', 'ligature_v3_rosie', 'joint'),
    ('GigaTIME', 'plug-in', 'ligature_v3_gigatime', 'plugin'),
    ('GigaTIME', '\\method{}', 'ligature_v3_gigatime', 'joint'),
    ('GigaTIME', '\\method{}, learned mean correction', 'ligature_v3_gigatime_mc', 'joint'),
    ('GigaTIME fine-tuned', 'plug-in', 'ligature_v3_gigatime_ft', 'plugin'),
    ('GigaTIME fine-tuned', '\\method{}', 'ligature_v3_gigatime_ft', 'joint'),
    ('GigaTIME fine-tuned', '\\method{}, learned mean correction', 'ligature_v3_gigatime_ft_mc', 'joint'),
]


def main():
    ref = pd.read_csv('runs/ligature_v3/query_summary.csv').set_index('query')
    base = ref['mse_plugin']
    out = []
    for grp, lab, run, col in ROWS:
        run, _, tag = run.partition(':')  # 'run:tag' reads query_summary{tag}.csv
        f = Path(f'runs/{run}/query_summary{tag}.csv')
        if not f.exists():
            continue
        q = pd.read_csv(f).set_index('query').loc[base.index]
        if q[f'mse_{col}'].isna().all():
            continue
        assert np.allclose(q.truth_mean, ref.truth_mean, atol=1e-6), run
        r = q[f'mse_{col}'] / base
        out.append(dict(group=grp, label=lab, run=run, column=col, mse_all=float(r.mean()),
                        mse_same=float(r[q['diag'].astype(bool)].mean()),
                        rho=float(q[f'rho_{col}'].mean())))
    d = pd.DataFrame(out)
    d.to_csv('paper/common_reference.csv', index=False)
    L = [r'\begin{tabular}{@{}llccc@{}}', r'\toprule',
         r'Stain & Estimator & MSE all & MSE same & $\rho$ \\', r'\midrule']
    last = None
    for r in d.itertuples():
        if last is not None and r.group != last:
            L.append(r'\addlinespace')
        L.append(f"{r.group if r.group != last else ''} & {r.label} & {r.mse_all:.2f} & {r.mse_same:.2f} & {r.rho:.2f} \\\\")
        last = r.group
    L += [r'\bottomrule', r'\end{tabular}']
    Path('paper/appendix_tables/common_reference.tex').write_text('\n'.join(L) + '\n')
    print(d.round(3).to_string())


if __name__ == '__main__':
    main()
