"""Render the manuscript's exploratory biological case from aggregate JSON."""
import argparse
import json
from pathlib import Path


def render(summary, paper):
    s = json.loads(summary.read_text())
    comparisons = {(r['method'], r['baseline']): r for r in s['contact_comparisons']}
    q, raw = comparisons['ligature_q', 'direct'], comparisons['ligature', 'direct']
    clinical = {(r['feature'], r['method'], r['comparison']): r for r in s['clinical']}
    paired = {(r['feature'], r['comparison'], r['contrast']): r for r in s['clinical_comparisons']}
    controls = {(r['method'], r['comparison']): r for r in s['controls']}
    burden = paired['engaged_burden', 'within_fold', 'ligature_q minus direct']
    vals = dict(BioContactRatio=q['mse_ratio'], BioContactLo=q['ci95'][0], BioContactHi=q['ci95'][1],
                BioRawRatio=raw['mse_ratio'], BioRawLo=raw['ci95'][0], BioRawHi=raw['ci95'][1],
                BioBurdenDelta=burden['difference'], BioBurdenLo=burden['ci95'][0], BioBurdenHi=burden['ci95'][1])
    for name, feature, method in [('BioMeasuredContactC', 'conditional', 'measured'),
            ('BioMeasuredBurdenC', 'engaged_burden', 'measured'),
            ('BioMeasuredAbundanceC', 'abundance', 'measured'),
            ('BioLigBurdenC', 'engaged_burden', 'ligature_q'),
            ('BioDirectBurdenC', 'engaged_burden', 'direct')]:
        vals[name] = clinical[feature, method, 'within_fold']['concordance']
    vals['BioStageC'] = controls['clinical_stage', 'within_fold']['concordance']
    vals['BioIFMC'] = controls['published_IFM1', 'within_fold']['concordance']
    lines = ['% Generated from biological_case.json by src/render_biological_case.py.']
    lines += [f'\\newcommand{{\\{name}}}{{{value:.3f}}}' for name, value in vals.items()]
    lines += [r'\newcommand{\BioContactPercent}{' + f"{100*(1-q['mse_ratio']):.0f}" + '}']
    (paper / 'biological_numbers.tex').write_text('\n'.join(lines) + '\n')
    names = dict(measured='Measured assay', stain='Stain', stain_q='Stain-Q', ligature=r'\method{}',
                 ligature_q=r'\method{}-Q', direct='Contact-trained graph')
    rows = [r'\begin{tabular}{@{}lrcc@{}}', r'\toprule',
            r'Method & Runs & MSE / stain & Spearman $\rho$ \\', r'\midrule']
    for r in s['assay']:
        if r['feature'] == 'conditional' and r['method'] != 'measured':
            rows.append(f"{names[r['method']]} & {r['runs']} & {r['mse_relative_to_stain']:.3f} & {r['spearman']:.3f} " + r'\\')
    rows += [r'\bottomrule', r'\end{tabular}']
    (paper / 'appendix_tables/biological_case.tex').write_text('\n'.join(rows) + '\n')
    rows = [r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
            r'Method & Contact fraction & Engaged burden & CD8a abundance \\', r'\midrule']
    for method in names:
        values = [clinical[f, method, 'within_fold']['concordance'] for f in
                  ('conditional', 'engaged_burden', 'abundance')]
        rows.append(names[method] + ' & ' + ' & '.join(f'{v:.3f}' for v in values) + r' \\')
    rows += [r'\bottomrule', r'\end{tabular}']
    (paper / 'appendix_tables/clinical_case.tex').write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--paper-dir', type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.paper_dir)
