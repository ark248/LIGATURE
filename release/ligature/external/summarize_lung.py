#!/usr/bin/env python
"""External lung cohort: headline numbers and a common-scale table (relative to the lung plug-in).

Run after analyze_spatial_engression.py (with SE_MARKERS set) on runs/lung/{ligature,
ligature_const, ligature_const_mc} and analyze_contact.py on runs/lung/ligature. Writes
runs/lung/lung_summary.json and paper/appendix_tables/lung.tex.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('runs/lung')
ROWS = [('Plug-in (cell-graph transformer stain)', 'ligature', 'plugin'),
        ('Global rescale of the plug-in', 'ligature', 'global_rescale'),
        ('Direct graph model (statistics, no cells)', 'ligature', 'direct_graph'),
        ('\\method{} around a constant stain', 'ligature_const', 'joint'),
        ('\\method{} around a constant stain, learned mean', 'ligature_const_mc', 'joint'),
        ('Direct graph with stain inputs (statistics, no cells)', 'ligature', 'direct_graph_stain'),
        ('\\method{} around the transformer stain', 'ligature', 'joint')]


def main():
    ref = pd.read_csv(R / 'ligature/query_summary.csv').set_index('query')
    base = ref['mse_plugin']
    diag = ref['diag'].astype(bool)
    big = ref.truth_mean.abs() > .02
    rows = []
    for lab, run, col in ROWS:
        f = R / run / 'query_summary.csv'
        if not f.exists():
            continue
        q = pd.read_csv(f).set_index('query').loc[base.index]
        if q[f'mse_{col}'].isna().all():
            continue
        r = q[f'mse_{col}'] / base
        rb = (q[f'bias_{col}'] / q.truth_mean)[big]
        rows.append(dict(estimator=lab, run=run, column=col, mse_all=float(r.mean()), mse_same=float(r[diag].mean()),
                         rho=float(q[f'rho_{col}'].mean()), median_relbias=float(rb.median())))
    d = pd.DataFrame(rows)
    s = json.loads((R / 'ligature/summary.json').read_text())
    out = dict(patients=s.get('patients'), table=d.to_dict('records'),
               crps_ligature=float(np.mean([s['all136'].get('crps_joint', np.nan)])),
               crps_gauss_plugin=float(np.mean([s['all136'].get('crps_gaussplug', np.nan)])),
               coverage_raw=s['all136'].get('cover_joint'), coverage_inflated=s['all136'].get('cover_recal'),
               coverage_conformal=s['all136'].get('cover_conf'), cell_mse=s.get('cell_mse'))
    c = R / 'ligature/contact_summary.json'
    if c.exists():
        cs = json.loads(c.read_text())
        out['contact'] = {k: cs[k] for k in cs if k.startswith(('mean_nmse', 'mean_rho', 'median_relbias', 'frac_'))}
    (R / 'lung_summary.json').write_text(json.dumps(out, indent=2))
    L = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule', r'Estimator & MSE all & MSE same & $\rho$ & bias \\', r'\midrule']
    for r in d.itertuples():
        L.append(f"{r.estimator} & {r.mse_all:.2f} & {r.mse_same:.2f} & {r.rho:.2f} & ${100 * r.median_relbias:+.0f}\\%$ \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    Path('paper/appendix_tables/lung.tex').write_text('\n'.join(L) + '\n')
    print(d.round(3).to_string())
    print(json.dumps({k: v for k, v in out.items() if k != 'table'}, indent=1))


if __name__ == '__main__':
    main()
