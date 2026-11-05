#!/usr/bin/env python
"""Patient-level paired tests and patient-bootstrap intervals for headline ratios."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def ratio(df, a, b='plugin'):
    g = df.groupby('query')
    num = g.apply(lambda h: ((h[a] - h.truth) ** 2).mean())
    den = g.apply(lambda h: ((h[b] - h.truth) ** 2).mean())
    return float((num / den).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()
    df = pd.read_csv(args.run / 'patient_queries.csv')
    base = df.groupby('query').apply(lambda h: ((h.plugin - h.truth) ** 2).mean()).rename('base')
    df = df.join(base, on='query')
    out = {}
    for m in ['joint', 'calibrated', 'direct']:
        df['n_' + m] = (df[m] - df.truth) ** 2 / df.base
    df['n_plugin'] = (df.plugin - df.truth) ** 2 / df.base
    per = df.groupby('case')[['n_plugin', 'n_joint', 'n_calibrated', 'n_direct', 'crps_joint', 'crps_plugin']].mean()
    for subset, sel in [('all136', df['query'].notna()), ('diag16', df['diag'].astype(bool))]:
        p = df[sel].groupby('case')[['n_plugin', 'n_joint', 'n_calibrated', 'crps_joint', 'crps_plugin']].mean()
        out[subset] = dict(
            wilcoxon_mse_joint_vs_plugin=float(wilcoxon(p.n_joint, p.n_plugin).pvalue),
            wilcoxon_mse_joint_vs_calibrated=float(wilcoxon(p.n_joint, p.n_calibrated).pvalue),
            wilcoxon_crps_joint_vs_plugin=float(wilcoxon(p.crps_joint, p.crps_plugin).pvalue),
            patients_mse_improved=int((p.n_joint < p.n_plugin).sum()),
            patients_crps_improved=int((p.crps_joint < p.crps_plugin).sum()))
        rng = np.random.default_rng(0)
        cases = df.case.unique()
        sub = df[sel]
        boots = []
        for _ in range(args.boot):
            pick = rng.choice(cases, len(cases), replace=True)
            b = pd.concat([sub[sub.case == c] for c in pick])
            boots.append(ratio(b, 'joint'))
        out[subset]['nmse_joint'] = ratio(sub, 'joint')
        out[subset]['nmse_joint_ci90'] = [float(np.quantile(boots, .05)), float(np.quantile(boots, .95))]
    (args.run / 'significance.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
