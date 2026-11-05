#!/usr/bin/env python
"""Per-fold virtual stain from ROSIE: one affine map per marker from its matched ROSIE channel.

y_a ~ alpha_a + beta_a * rosie_a, fitted by least squares on training-patient cells
only, so the stain is ROSIE's own prediction on ORION's intensity scale. Writes
runs/rosie_stain/fold{k}/cell_pred.npy in se.load cell order.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_engression as se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--out', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/rosie_stain'))
    args = ap.parse_args()
    cells, x, y, crops, roles = se.load(args.root, args.fold, 271027, 8)
    r = pd.concat([pd.read_parquet(f) for f in sorted((args.root / args.out / 'cells').glob('part*.parquet'))])
    r = cells[['crop_id', 'cell_label']].merge(r, on=['crop_id', 'cell_label'], how='left')
    R = r[se.MARKERS].to_numpy(np.float64)
    assert np.isfinite(R).all(), 'ROSIE outputs missing for some cells'
    role = cells.case.map(roles).to_numpy()
    tr, te = role == 'train', role == 'test'
    F_ = np.zeros_like(y)
    rep = {}
    for j, m in enumerate(se.MARKERS):
        A = np.column_stack([np.ones(tr.sum()), R[tr, j]])
        w = np.linalg.lstsq(A, y[tr, j], rcond=None)[0]
        F_[:, j] = w[0] + w[1] * R[:, j]
        rep[m] = dict(slope=float(w[1]), test_r=float(np.corrcoef(R[te, j], y[te, j])[0, 1]),
                      test_mse=float(np.mean((F_[te, j] - y[te, j]) ** 2)))
    out = args.root / args.out / f'fold{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'cell_pred.npy', F_.astype(np.float32))
    (out / 'summary.json').write_text(json.dumps(dict(fold=args.fold, test_cell_mse=float(np.mean((F_[te] - y[te]) ** 2)),
                                                      markers=rep), indent=2))
    print('fold', args.fold, 'test cell mse', round(float(np.mean((F_[te] - y[te]) ** 2)), 4))


if __name__ == '__main__':
    main()
