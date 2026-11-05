#!/usr/bin/env python
"""Patient-level summary of per-cell-law identification intervals (identification_sdp.py).

Couplings across regions are unconstrained by per-cell laws, so the identification
interval of a patient average is the average of the region intervals. Regions are a
random subsample (up to 10 per patient), so widths are estimates of the mean width.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/ligature_v3'))
    args = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in sorted(args.run.glob('fold*/identification_sdp.csv'))])
    d['gap'] = (d.cert_hi - d.sdp_hi) / (d.sdp_hi - d.plugin).abs().clip(lower=1e-12)
    d['wit_hi'] = (d.witness_hi - d.plugin) / (d.sdp_hi - d.plugin).clip(lower=1e-12)
    d['wit_lo'] = (d.witness_lo - d.plugin) / (d.sdp_lo - d.plugin).clip(upper=-1e-12)
    deccols = [c for c in d.columns if c.startswith('dec_')]
    p = d.groupby(['case', 'marker'])[['plugin', 'truth', 'sdp_lo', 'sdp_hi'] + deccols].mean().reset_index()
    p['width'] = p.sdp_hi - p.sdp_lo
    if deccols:
        # one stationary decaying kernel shared by all regions of a patient (or the nugget, i.e. zero)
        v = p[deccols].to_numpy()
        p['dec_lo'] = p.plugin + np.minimum(0, v.min(1)); p['dec_hi'] = p.plugin + np.maximum(0, v.max(1))
        p['dec_width'] = p.dec_hi - p.dec_lo
        p['dec_inside'] = (p.truth >= p.dec_lo) & (p.truth <= p.dec_hi)
    p['inside'] = (p.truth >= p.sdp_lo) & (p.truth <= p.sdp_hi)
    pq = pd.read_csv(args.run / 'patient_queries.csv')
    pq = pq[pq['diag'].astype(bool)].copy()
    pq['marker'] = pq['query'].str.split(' : ').str[0]
    spread = pq.groupby('marker').truth.agg(lambda v: np.quantile(v, .95) - np.quantile(v, .05)).rename('cohort_range')
    tw = pq.assign(tw=pq.hi_recal - pq.lo_recal).groupby('marker').tw.mean().rename('ligature_width')
    m = p.groupby('marker').width.mean().rename('identification_width').to_frame().join(spread).join(tw)
    if 'dec_width' in p:
        m = m.join(p.groupby('marker').dec_width.mean().rename('restricted_width'))
        m['restricted_ratio_to_cohort_range'] = m.restricted_width / m.cohort_range
        m['restricted_ratio_to_ligature'] = m.restricted_width / m.ligature_width
    m['ratio_to_cohort_range'] = m.identification_width / m.cohort_range
    m['ratio_to_ligature'] = m.identification_width / m.ligature_width
    out = dict(regions=int(d.crop.nunique()), patients=int(d.case.nunique()),
               max_certificate_gap=float(d.gap.max()), median_witness_fraction_hi=float(d.wit_hi.median()),
               min_witness_fraction_hi=float(d.wit_hi.min()), median_witness_fraction_lo=float(d.wit_lo.median()),
               patient_truth_inside=float(p.inside.mean()),
               median_ratio_width_to_cohort_range=float(m.ratio_to_cohort_range.median()),
               min_ratio_width_to_cohort_range=float(m.ratio_to_cohort_range.min()),
               median_ratio_width_to_ligature_interval=float(m.ratio_to_ligature.median()),
               **({} if 'dec_width' not in p else dict(
                   restricted_truth_inside=float(p.dec_inside.mean()),
                   median_restricted_ratio_to_cohort_range=float(m.restricted_ratio_to_cohort_range.median()),
                   n_markers_restricted_wider_than_cohort_range=int((m.restricted_ratio_to_cohort_range > 1).sum()),
                   median_restricted_ratio_to_ligature=float(m.restricted_ratio_to_ligature.median()))),
               per_marker=m.round(4).reset_index().to_dict('records'))
    (args.run / 'identification_summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_marker'}, indent=2))
    print(m.round(3).to_string())


if __name__ == '__main__':
    main()
