"""Publication figures from complete audited reports; no model fitting or new tests."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / 'results'
sys.path.insert(0, str(ROOT / 'release/ligature/src'))
from figure_export import export_figure

BLUE, DARK, GRAY = '#2a78d6', '#263B42', '#727C80'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def forest(ax, rows, limits, ticks, xlabel):
    for i, row in enumerate(rows):
        value, (low, high) = row['estimate'], row['ci95']
        assert low <= value <= high and np.isfinite([value, low, high]).all()
        ax.errorbar(value, i, xerr=[[value-low], [high-value]], fmt='o',
                    color=BLUE, ms=4.5, lw=1.15, capsize=3, capthick=1)
    ax.axvline(0, color=GRAY, ls=(0, (3, 3)), lw=.8, zorder=0)
    ax.set(xlim=limits, xticks=ticks, ylim=(len(rows)-.5, -.5),
           yticks=np.arange(len(rows)), yticklabels=[r['label'] for r in rows], xlabel=xlabel)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRAY)
    ax.tick_params(axis='y', length=0, pad=7)
    ax.tick_params(axis='x', length=3, color=GRAY)
    ax.xaxis.labelpad=5

def canvas(titles):
    fig=plt.figure(figsize=(6.5, 2.65))
    axes=[fig.add_axes([.22, .29, .25, .53]), fig.add_axes([.72, .29, .25, .53])]
    for x, title in zip([.025, .525], titles):
        fig.text(x, .94, title, va='top', fontsize=11, weight='bold', color=DARK)
    fig.text(.5, .035, 'LIGATURE − direct; points and 95% patient-bootstrap intervals',
             ha='center', va='bottom', fontsize=10, color=DARK)
    return fig, axes

def export(fig, name, panels, source_names, scope):
    path=ROOT/'paper'/name
    export_figure(fig, path)
    svg=path.with_suffix('.svg')
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    meta_path=path.with_name(name+'_format.json')
    meta=json.loads(meta_path.read_text())
    meta['note']='New paired contrasts from complete audited reports; all prespecified families retained.'
    meta_path.write_text(json.dumps(meta, indent=2)+'\n')
    manifest=dict(panels=panels, source_hashes={n:sha(RESULTS/n) for n in source_names},
                  renderer_sha256=sha(Path(__file__)), scope=scope,
                  interpretation='Negative differences favor LIGATURE; positive differences favor the matched direct graph. Metrics use distinct axes and units.')
    path.with_name(name+'_data.json').write_text(json.dumps(manifest, indent=2)+'\n')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.family':'STIXGeneral', 'mathtext.fontset':'stix', 'font.size':10,
                         'axes.labelsize':10, 'xtick.labelsize':10, 'ytick.labelsize':10,
                         'pdf.fonttype':42, 'svg.fonttype':'path', 'axes.linewidth':.6})
    q=json.loads((RESULTS/'summary.json').read_text())
    he=json.loads((RESULTS/'he_summary.json').read_text())
    ha=json.loads((RESULTS/'he_audit.json').read_text())
    for name, report in [('audit.json', q), ('he_audit.json', he)]:
        a=json.loads((RESULTS/name).read_text())
        assert a['status']=='passed' and report['status']=='complete'
        assert a['report_sha256']==sha(RESULTS/('he_summary.json' if name.startswith('he_') else 'summary.json'))
    def query(family, metric, label):
        r=next(r for r in q['paired'] if r['family']==family and r['method']=='ligature' and r['metric']==metric)
        return dict(label=label, estimate=r['mean_difference'], ci95=r['difference_ci95'])
    left=[query(f,'interval_score',l) for f,l in [('all_scored','All 136'),('clustering','Clustering (16)'),('cross','Cross-marker (120)')]]
    right=[query(f,'error_normalized',l) for f,l in [('Moran',r'Moran $I$'),('Geary',r'Geary $C$'),('log_assay_clustering','Log contrast')]]
    fig,axes=canvas(['a  Calibrated forecast error','b  New-query point error'])
    forest(axes[0],left,(-1.25,.23),[-1,-.5,0],'Interval-score difference')
    forest(axes[1],right,(-.10,.28),[-.1,0,.1,.2],'Relative MSE difference')
    export(fig,'fig_benchmarks',[left,right],['summary.json','audit.json'],
           'Three fixed residual/direct seeds, 40 patients, independent 20/2/10/8 roles. Seed metrics averaged; conditional patient intervals do not include all cross-fold dependence. Overall forecast comparison is exploratory; primary clustering contrast includes zero. Direct models are adapted for the new queries.')
    left=[];right=[]
    for family,label in [('all136','All 136'),('clustering16','Clustering (16)'),('cross120','Cross-marker (120)')]:
        r=next(r for r in ha['paired_point'] if r['family']==family)
        left.append(dict(label=label,estimate=r['normalized_error_difference_from_direct'],ci95=r['difference_from_direct_ci95']))
        r=next(r for r in he['paired'] if r['family']==family and r['method']=='ligature_conformal' and r['baseline']=='direct_adaptive' and r['metric']=='interval_score_normalized')
        right.append(dict(label=label,estimate=r['difference'],ci95=r['difference_ci95']))
    fig,axes=canvas(['a  H&E-proxy point error','b  Calibrated forecast error'])
    forest(axes[0],left,(-.16,.15),[-.1,0,.1],'Relative MSE difference')
    forest(axes[1],right,(-1.7,.10),[-1.5,-1,-.5,0],'Interval-score difference')
    export(fig,'fig_geometry',[left,right],['he_summary.json','he_audit.json'],
           'One seed, 40 patients, five independently calibrated H&E-derived proxy fits. Targets differ from primary whole-cell targets; point contrasts include zero. No whole-slide validation, isolated segmentation effect or optical-spillover correction.')

if __name__=='__main__':
    main()
