#!/usr/bin/env python
"""Per-fold virtual stain derived from GigaTIME outputs.

For each ORION marker, a ridge head maps the logits of GigaTIME's 21 per-cell protein
probabilities to the training-standardized target; fitted on training-patient cells,
penalty chosen on validation patients. Writes runs/gigatime_stain/fold{k}/cell_pred.npy
in se.load cell order, for use with ligature.py --mean-kind file.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_engression as se
from gigatime_infer import PROTEIN

OVERLAP = {'PD-1': 'PD-1', 'CD4': 'CD4', 'CD68': 'CD68', 'CD20': 'CD20', 'CD3e': 'CD3', 'CD8a': 'CD8',
           'PD-L1': 'PD-L1', 'Pan-CK': 'CK', 'Ki67': 'Ki67'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--out', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/gigatime_stain'))
    args = ap.parse_args()
    cells, x, y, crops, roles = se.load(args.root, args.fold, 271027, 8)
    g = pd.concat([pd.read_parquet(f) for f in sorted((args.root / args.out / 'cells').glob('part*.parquet'))])
    g = cells[['crop_id', 'cell_label']].merge(g, on=['crop_id', 'cell_label'], how='left')
    P_ = np.clip(g[PROTEIN].to_numpy(np.float64), 1e-4, 1 - 1e-4)
    Z = np.log(P_ / (1 - P_))
    assert np.isfinite(Z).all(), 'GigaTIME outputs missing for some cells'
    role = cells.case.map(roles).to_numpy()
    tr, va, te = role == 'train', role == 'val', role == 'test'
    zm, zs = Z[tr].mean(0), Z[tr].std(0) + 1e-6
    Zs = (Z - zm) / zs
    F_ = np.zeros_like(y)
    report = {}
    for j, m in enumerate(se.MARKERS):
        best = None
        for lam in [1e-1, 1e1, 1e3, 1e5]:
            A = np.column_stack([np.ones(tr.sum()), Zs[tr]])
            w = np.linalg.solve(A.T @ A + lam * np.diag([0] + [1] * Zs.shape[1]), A.T @ y[tr, j])
            err = np.mean((np.column_stack([np.ones(va.sum()), Zs[va]]) @ w - y[va, j]) ** 2)
            if best is None or err < best[0]:
                best = (err, lam, w)
        F_[:, j] = np.column_stack([np.ones(len(y)), Zs]) @ best[2]
        report[m] = dict(lambda_=best[1], val_mse=float(best[0]), test_mse=float(np.mean((F_[te, j] - y[te, j]) ** 2)),
                         test_r=float(np.corrcoef(F_[te, j], y[te, j])[0, 1]), direct_channel=OVERLAP.get(m))
    out = args.root / args.out / f'fold{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'cell_pred.npy', F_.astype(np.float32))
    (out / 'summary.json').write_text(json.dumps(dict(fold=args.fold, test_cell_mse=float(np.mean((F_[te] - y[te]) ** 2)),
                                                      markers=report), indent=2))
    print('fold', args.fold, 'test cell mse', round(float(np.mean((F_[te] - y[te]) ** 2)), 4))


if __name__ == '__main__':
    main()
