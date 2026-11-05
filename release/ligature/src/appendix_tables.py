#!/usr/bin/env python
"""Generate appendix LaTeX tables directly from run outputs (no hand-transcribed numbers)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('paper/appendix_tables')


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{name}.tex').write_text('\n'.join(lines) + '\n')


def identification():
    d = json.loads((Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3/identification_summary.json').read_text())
    rows = sorted(d['per_marker'], key=lambda r: r['ratio_to_cohort_range'])
    L = [r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
         r'Marker & cohort & \method{} & per-cell & ratio & restricted & ratio \\', r'\midrule']
    for r in rows:
        L.append(f"{r['marker']} & {r['cohort_range']:.2f} & {r['ligature_width']:.2f} & {r['identification_width']:.2f} & "
                 f"{r['ratio_to_cohort_range']:.1f} & {r['restricted_width']:.2f} & {r['restricted_ratio_to_cohort_range']:.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    write('identification', L)


def published_markers():
    rows = {}
    for stain in ['rosie', 'gigatime']:
        for k in range(5):
            d = json.loads(Path(f'runs/{stain}_stain/fold{k}/summary.json').read_text())
            for m, v in d['markers'].items():
                if stain == 'gigatime' and not v.get('direct_channel'):
                    continue
                rows.setdefault(m, {}).setdefault(stain, []).append(v['test_r'])
    L = [r'\begin{tabular}{@{}lcc@{}}', r'\toprule', r'Marker & ROSIE & GigaTIME \\', r'\midrule']
    for m in sorted(rows):
        f = lambda s: f"{np.mean(rows[m][s]):.2f}" if s in rows[m] else '--'
        L.append(f"{m} & {f('rosie')} & {f('gigatime')} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    write('published_markers', L)


def per_fold():
    L = [r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
         r'Rotation & raw 90\% & inflated & conformal & $\hat\tau$ infl. & $\hat\tau$ conf. \\', r'\midrule']
    d = json.loads((Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3/summary.json').read_text())
    pq = pd.read_csv('runs/ligature_v3/patient_queries.csv')
    pq['raw'] = (pq.truth >= pq.lo) & (pq.truth <= pq.hi)
    pq['inf'] = (pq.truth >= pq.lo_recal) & (pq.truth <= pq.hi_recal)
    for f, g in pq.groupby('fold'):
        L.append(f"{f.replace('fold', '')} & {g.raw.mean():.2f} & {g.inf.mean():.2f} & {d['cover_conformal_by_fold'][f]:.2f} & "
                 f"{d['tau_by_fold'][f]:.2f} & {d['tau_conformal_by_fold'][f]:.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    write('per_fold', L)


def hybrid():
    d = json.loads((Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3/hybridreg_summary.json').read_text())
    h = json.loads((Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3/hybrid_summary.json').read_text())
    names = {'PanCK_CD45': 'Pan-CK, CD45', 'PDL1_CD8a': 'PD-L1, CD8a', 'CD3e': 'CD3e'}
    L = [r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
         r'Anchors & ridge imput. & \method{}, substituted & exact cond. & regularized cond. & $\delta$ \\', r'\midrule']
    for k, v in d.items():
        t = v['test']; best = v['chosen']
        L.append(f"{names[k]} & {t['q_ridge_impute']['nmse']:.2f} & {t['q_ligature_uncond']['nmse']:.2f} & "
                 f"{h[k]['all']['nmse_ligature_cond']:.2f} & {t[best]['nmse']:.2f} & {best.replace('q_cond_', '')} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    write('hybrid', L)


def contact_examples():
    runs = [('transformer', 'ligature_v3'), ('ridge', 'ligature_v3_ridge'), ('U-Net', 'ligature_v3_unet'),
            ('ROSIE', 'ligature_v3_rosie'), ('GigaTIME', 'ligature_v3_gigatime')]
    pairs = ['CD8a -> Pan-CK', 'CD3e -> Pan-CK', 'CD68 -> PD-L1', 'PD-L1 -> CD8a', 'FOXP3 -> CD8a', 'CD8a -> CD68']
    L = [r'\begin{tabular}{@{}l' + 'c' * len(runs) + r'@{}}', r'\toprule',
         'Pair & ' + ' & '.join(n for n, _ in runs) + r' \\', r'\midrule']
    tabs = {n: pd.read_csv(f'runs/{r}/contact_summary.csv').set_index('pair') for n, r in runs}
    for p in pairs:
        cells = []
        for n, _ in runs:
            t = tabs[n]
            cells.append(f"{t.loc[p, 'nmse_plugin_q']:.2f} / {t.loc[p, 'nmse_joint']:.2f}" if p in t.index else '--')
        L.append(p.replace('->', '$\\to$') + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    write('contact_examples', L)


def neighborhoods():
    runs = [('Cell-graph transformer', 'ligature_v3'), ('Per-cell ridge', 'ligature_v3_ridge'), ('Pixel U-Net', 'ligature_v3_unet'),
            ('ROSIE', 'ligature_v3_rosie'), ('GigaTIME', 'ligature_v3_gigatime'), ('GigaTIME fine-tuned', 'ligature_v3_gigatime_ft')]
    L = [r'\begin{tabular}{@{}lcccccccc@{}}', r'\toprule',
         r' & \multicolumn{4}{c}{total variation to measured} & & & \multicolumn{2}{c}{rank corr.} \\',
         r'\cmidrule(lr){2-5}\cmidrule(lr){8-9}',
         r'Virtual stain & plug-in & quant. & \method{} & \method{}-Q & $p$ & better & quant. & \method{}-Q \\', r'\midrule']
    for n, r in runs:
        f = Path(f'runs/{r}/neighborhood_summary.json')
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        tv, rho = d['mean_tv'], d['per_cn_rho']
        pv = d['wilcoxon_tv_ligature_q_vs_plugin_q']
        e = int(np.floor(np.log10(pv)))
        ps = f'{pv:.3f}' if pv >= 1e-3 else f'${pv / 10 ** e:.1f}{{\\times}}10^{{{e}}}$'
        L.append(f"{n} & {tv['plugin']:.2f} & {tv['plugin_q']:.2f} & {tv['ligature']:.2f} & {tv['ligature_q']:.2f} & {ps} & "
                 f"{d['patients_ligature_q_better_than_plugin_q']}/{d['patients']} & {rho['plugin_q']:.2f} & {rho['ligature_q']:.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    write('neighborhoods', L)


def bias_decomposition():
    names = {'ligature_v3': 'Cell-graph transformer', 'ligature_v3_ridge': 'Per-cell ridge', 'ligature_v3_unet': 'Pixel U-Net',
             'ligature_v3_rosie': 'ROSIE', 'ligature_v3_gigatime': 'GigaTIME', 'ligature_v3_gigatime_ft': 'GigaTIME fine-tuned'}
    d = json.loads((Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/bias_decomposition.json').read_text())
    pct = lambda v: f'${100 * v:+.0f}\\%$'
    L = [r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
         r' & plug-in & \multicolumn{3}{c}{cohort mean / measured} & \multicolumn{2}{c}{per-patient RMS / s.d.} \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}',
         r'Virtual stain & bias & $G(r)$ & $X(f,r)$ & $X<0$ & $G(r)$ & $X(f,r)$ \\', r'\midrule']
    for run, n in names.items():
        if run not in d:
            continue
        v = d[run]
        L.append(f"{n} & {pct(v['median_relbias_plugin'])} & {pct(v['median_resid_share'])} & {pct(v['median_cross_share'])} & "
                 f"{100 * v['frac_cross_negative']:.0f}\\% & {v['resid_rms_over_sd']:.2f} & {v['cross_rms_over_sd']:.2f} \\\\")
    lf = (Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/lung/bias_decomposition.json')
    if lf.exists():  # external lung cohort, cell-graph transformer stain
        v = json.loads(lf.read_text())['ligature']
        L.append(r'\midrule')
        L.append(f"Lung (Xenium), transformer & {pct(v['median_relbias_plugin'])} & {pct(v['median_resid_share'])} & {pct(v['median_cross_share'])} & "
                 f"{100 * v['frac_cross_negative']:.0f}\\% & {v['resid_rms_over_sd']:.2f} & {v['cross_rms_over_sd']:.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    write('bias_decomposition', L)


if __name__ == '__main__':
    identification(); published_markers(); per_fold(); hybrid(); contact_examples(); neighborhoods(); bias_decomposition()
    print(sorted(p.name for p in OUT.glob('*.tex')))
