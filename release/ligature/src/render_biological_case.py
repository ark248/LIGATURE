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


def render_feasibility(summary, paper):
    """Write paper/clinical_numbers.tex; stop if the data no longer support the manuscript's wording."""
    s = json.loads(summary.read_text())
    grid, decision, primary = s['grid_summary'], s['decision'], s['decision']['primary']
    power = {r['scope']: r for r in s['power']}
    cox = {(r['feature'], r['added']): r for r in s['nested_cox']}
    assay = {r['feature']: r for r in s['assay_recovery']}
    til = {(r['feature'], r['scope'], r['method']): r for r in s['til']['rows']}
    gap = {(r['feature'], r['scope'], r['contrast']): r for r in s['til']['contrasts']}
    separated = [r for r in s['oracle'] if r['ci95'][0] > 0 or r['ci95'][1] < 0]
    assert not decision['survival_superiority_established'] and not any(decision['gates'].values())
    assert not separated and grid['unadjusted_favor_ligature'] == grid['unadjusted_favor_direct'] == 0
    assert power['within_fold']['minimum_detectable_difference'] > power['within_fold']['oracle_gap']
    assert power['pooled']['minimum_detectable_difference'] > power['pooled']['oracle_gap']
    burden = cox['engaged_burden', 'ligature_q']
    assert burden['p'] > .05 and cox['engaged_burden', 'measured']['p'] > .05
    til_primary = gap['engaged_burden', 'within_fold', 'ligature_q minus direct']
    til_abundance = gap['abundance', 'pooled', 'ligature_q minus direct']
    assert til_primary['ci95'][0] < 0 < til_primary['ci95'][1] and til_abundance['ci95'][1] < 0
    rho, burden_mse, abundance_mse = (assay['engaged_burden']['spearman_difference_ci95'],
        assay['engaged_burden']['mse_ratio_ci95'], assay['abundance']['mse_ratio_ci95'])
    assert rho[0] > 0 and burden_mse[0] < 1 < burden_mse[1] and abundance_mse[0] < 1 < abundance_mse[1]
    oracle = next(r for r in s['oracle'] if (r['feature'], r['scope'], r['aggregate']) ==
                  ('engaged_burden', 'within_fold', 'mean_of_metrics'))
    vals = dict(
        CliEvents=str(s['events']), CliMeasuredLo=f"{decision['measured_ci95'][0]:.3f}",
        CliMeasuredHi=f"{decision['measured_ci95'][1]:.3f}",
        CliPrimaryLo=f"{primary['ci95_familywise'][0]:.3f}", CliPrimaryHi=f"{primary['ci95_familywise'][1]:.3f}",
        CliOracle=f"{oracle['difference']:.3f}", CliOracleLo=f"{oracle['ci95'][0]:.3f}", CliOracleHi=f"{oracle['ci95'][1]:.3f}",
        CliOracleTotal=str(len(s['oracle'])), CliGridTotal=str(grid['contrasts']), CliGridDirect=str(grid['favor_direct_point']),
        CliCoxLig=f"{burden['p']:.2f}", CliCoxMeasured=f"{cox['engaged_burden', 'measured']['p']:.2f}",
        CliMddWithin=f"{power['within_fold']['minimum_detectable_difference']:.2f}",
        CliMddPooled=f"{power['pooled']['minimum_detectable_difference']:.2f}",
        CliGapWithin=f"{power['within_fold']['oracle_gap']:.2f}", CliGapPooled=f"{power['pooled']['oracle_gap']:.2f}",
        CliEventsWithin=str(round(power['within_fold']['events_needed_at_difference']['0.05'], -1)),
        CliEventsPooled=str(round(power['pooled']['events_needed_at_difference']['0.05'], -1)),
        CliTilLig=f"{til['engaged_burden', 'within_fold', 'ligature_q']['concordance']:.3f}",
        CliTilDirect=f"{til['engaged_burden', 'within_fold', 'direct']['concordance']:.3f}",
        CliTilMeasured=f"{til['engaged_burden', 'within_fold', 'measured']['concordance']:.3f}",
        CliTilDiff=f"{til_primary['difference']:.3f}", CliTilLo=f"{til_primary['ci95'][0]:.3f}", CliTilHi=f"{til_primary['ci95'][1]:.3f}",
        CliTilAbund=f"{til_abundance['difference']:.3f}", CliTilAbundLo=f"{til_abundance['ci95'][0]:.3f}",
        CliTilAbundHi=f"{til_abundance['ci95'][1]:.3f}",
        CliRho=f"{assay['engaged_burden']['spearman_difference']:.3f}", CliRhoLo=f"{rho[0]:.3f}", CliRhoHi=f"{rho[1]:.3f}",
        CliBurdenMse=f"{assay['engaged_burden']['mse_ratio']:.2f}", CliBurdenMseLo=f"{burden_mse[0]:.2f}", CliBurdenMseHi=f"{burden_mse[1]:.2f}",
        CliAbundMse=f"{assay['abundance']['mse_ratio']:.2f}", CliAbundMseLo=f"{abundance_mse[0]:.2f}", CliAbundMseHi=f"{abundance_mse[1]:.2f}")
    lines = ['% Generated from clinical_feasibility.json by src/render_biological_case.py.']
    lines += [f'\\newcommand{{\\{name}}}{{{value}}}' for name, value in vals.items()]
    (paper / 'clinical_numbers.tex').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--feasibility', type=Path, help='Optional clinical_feasibility.json')
    parser.add_argument('--paper-dir', type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.paper_dir)
    if args.feasibility:
        render_feasibility(args.feasibility, args.paper_dir)
