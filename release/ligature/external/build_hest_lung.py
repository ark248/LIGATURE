#!/usr/bin/env python
"""External cohort: HEST-1k Xenium lung samples (Vannan et al. 2025; 20 samples, 19 patients).

Builds the cell table in the format of data/cells/cells.parquet so every model and analysis
script runs unchanged with --root cohorts/lung and SE_MARKERS set to the 16 genes below.

Per sample: transcripts with qv >= 20 assigned to a Xenium cell are counted per gene; each
cell is placed at the centroid of the Xenium cell polygon nearest to its transcript centroid
(within 8 um), and polygon area, perimeter and eccentricity give the morphology inputs (no
transcript-derived quantity enters the inputs). Targets are counts scaled to the cohort
median library size (log1p is applied by the loader). Regions are 166.4 um squares (the
ORION region size) with at least 40 cells, at most 80 per sample, drawn with a fixed seed.
Coordinates are stored in 0.325 um units (ORION pixels), as the loader expects.
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

GENES = ['EPCAM', 'SFTPC', 'AGER', 'SCGB1A1', 'KRT5', 'PTPRC', 'CD3E', 'CD8A',
         'MS4A1', 'CD68', 'MARCO', 'COL1A1', 'ACTA2', 'PECAM1', 'MKI67', 'SPP1']
REGION_UM, MIN_CELLS, MAX_REGIONS, MATCH_UM = 166.4, 40, 80, 8.0


def polygons(path):
    """Outer rings of all cell polygons -> centroid, area, perimeter, eccentricity (pixels)."""
    z = zipfile.ZipFile(path)
    feats = json.load(z.open(z.namelist()[0]))
    rows = []
    for f in feats:
        g = f['geometry']
        polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        for poly in polys:
            r = np.asarray(poly[0], float)
            if len(r) < 4:
                continue
            x, y = r[:, 0], r[:, 1]
            cross = x[:-1] * y[1:] - x[1:] * y[:-1]
            a = cross.sum() / 2
            if abs(a) < 1e-6:
                continue
            cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * a)
            cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * a)
            per = np.hypot(np.diff(x), np.diff(y)).sum()
            ev = np.linalg.eigvalsh(np.cov(np.stack([x[:-1], y[:-1]])) + 1e-9 * np.eye(2))
            ecc = float(np.sqrt(max(0., 1 - ev[0] / ev[1])))
            rows.append((cx, cy, abs(a), per, ecc))
    return np.array(rows)


def sample_cells(root, sid, px_um):
    t = pq.read_table(root / f'transcripts/{sid}_transcripts.parquet', columns=['cell_id', 'feature_name', 'qv', 'he_x', 'he_y'])
    t = t.filter(pc.and_(pc.greater_equal(t['qv'], 20.), pc.not_equal(t['cell_id'], 'UNASSIGNED')))
    df = t.select(['cell_id', 'feature_name', 'he_x', 'he_y']).to_pandas()
    df['cell_id'] = df.cell_id.astype('category')
    g = df.groupby('cell_id', observed=True)
    cen = g[['he_x', 'he_y']].mean()
    total = g.size().rename('total')
    sub = df[df.feature_name.isin(GENES)]
    cnt = sub.groupby(['cell_id', 'feature_name'], observed=True).size().unstack(fill_value=0)
    cnt = cnt.reindex(index=cen.index, columns=GENES, fill_value=0)
    poly = polygons(root / f'xenium_seg/{sid}_xenium_cell_seg.geojson.zip')
    d, j = cKDTree(poly[:, :2]).query(cen.to_numpy(), distance_upper_bound=MATCH_UM / px_um)
    ok = np.isfinite(d)
    # one transcript cell per polygon: keep the closest
    order = np.argsort(d)
    seen, keep = set(), np.zeros(len(d), bool)
    for i in order:
        if ok[i] and j[i] not in seen:
            seen.add(j[i]); keep[i] = True
    out = pd.DataFrame({'he_x': poly[j[keep], 0], 'he_y': poly[j[keep], 1], 'cell_area_px': poly[j[keep], 2],
                        'nuc_area_px': poly[j[keep], 2], 'nuc_perimeter_px': poly[j[keep], 3],
                        'nuc_eccentricity': poly[j[keep], 4], 'total': total.to_numpy()[keep]})
    for gname in GENES:
        out['n_' + gname] = cnt[gname].to_numpy()[keep]
    return out[out.total >= 10].reset_index(drop=True), int(len(cen)), int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--out', type=Path, default=Path('cohorts/lung'))
    ap.add_argument('--seed', type=int, default=271027)
    args = ap.parse_args()
    src = args.root / 'data/external/hest_lung'
    out = args.root / args.out / 'data/cells'
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.root / 'data/external/hest_meta/HEST_v1_3_0.csv').set_index('id')
    ids = sorted(p.stem for p in (src / 'wsis').glob('*.tif'))
    rng = np.random.default_rng(args.seed)
    tables, report = [], []
    for sid in ids:
        m = json.loads((src / f'metadata/{sid}.json').read_text())
        px = float(m['pixel_size_um_estimated'])
        cells, n_tx_cells, n_matched = sample_cells(src, sid, px)
        tile = REGION_UM / px
        tx, ty = np.floor(cells.he_x / tile).astype(int), np.floor(cells.he_y / tile).astype(int)
        key = tx.astype(str) + '_' + ty.astype(str)
        counts = key.value_counts()
        good = counts[counts >= MIN_CELLS].index.to_numpy()
        if len(good) > MAX_REGIONS:
            good = rng.choice(np.sort(good), MAX_REGIONS, replace=False)
        cells = cells[key.isin(set(good))].copy()
        cells['crop_id'] = sid + '_' + key[cells.index]
        cells['sample'] = sid
        cells['case'] = str(m['patient']).replace(' ', '')
        cells['disease'] = m.get('disease_state', '')
        cells['px_um'] = px
        cells['gx'] = cells.he_x * px / .325
        cells['gy'] = cells.he_y * px / .325
        tables.append(cells)
        report.append(dict(sample=sid, case=cells.case.iat[0] if len(cells) else '', px_um=px, xenium_cells=n_tx_cells,
                           matched=n_matched, regions=int(len(good)), cells_in_regions=int(len(cells))))
        print(report[-1], flush=True)
    c = pd.concat(tables, ignore_index=True)
    med = float(np.median(c.total))
    for gname in GENES:
        c['y_' + gname] = c['n_' + gname] / c.total * med
    c['cell_label'] = c.groupby('crop_id').cumcount() + 1
    c['cell_idx'] = np.arange(len(c))
    c.to_parquet(out / 'cells.parquet', index=False)
    (out / 'build_report.json').write_text(json.dumps(dict(genes=GENES, median_library=med, samples=report,
                                                          cells=int(len(c)), regions=int(c.crop_id.nunique()),
                                                          patients=int(c.case.nunique())), indent=1))
    print('cells', len(c), 'regions', c.crop_id.nunique(), 'patients', c.case.nunique())


if __name__ == '__main__':
    main()
