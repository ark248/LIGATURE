#!/usr/bin/env python
"""Run the published GigaTIME virtual mIF model (Valanarasu et al., Cell 2025) on ORION H&E.

Each 512x512 region (0.325 um/pixel) is resized to 336x336 (~0.5 um/pixel), normalized
as in the authors' test script (albumentations Normalize: ImageNet mean/std), passed
through the released U-Net++ (23 output channels, 2 background), and the per-pixel
probabilities are upsampled back to 512x512 and averaged over each retained cell mask.
Writes per-cell mean probabilities for all 21 protein channels, keyed by (crop, label).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prep'))
from build_cells import decode_crop
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gigatime_arch import load_gigatime

CHANNELS = ['DAPI', 'TRITC', 'Cy5', 'PD-1', 'CD14', 'CD4', 'T-bet', 'CD34', 'CD68', 'CD16', 'CD11c', 'CD138',
            'CD20', 'CD3', 'CD8', 'PD-L1', 'CK', 'Ki67', 'Tryptase', 'Actin-D', 'Caspase3-D', 'PHH3-B', 'Transgelin']
PROTEIN = [c for c in CHANNELS if c not in ('TRITC', 'Cy5')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=8)
    ap.add_argument('--size', type=int, default=336)
    ap.add_argument('--out', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/gigatime_stain/cells'))
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_gigatime(args.root / args.weights, num_classes=len(CHANNELS)).to(dev).eval()
    ct = pd.read_csv(args.root / 'data/cells/crops.csv')
    cells = pd.read_parquet(args.root / 'data/cells/cells.parquet', columns=['crop_id', 'cell_label'])
    labels = cells.groupby('crop_id').cell_label.apply(np.array)
    ct = ct[ct.id.isin(labels.index)].sort_values(['file', 'row_group']).reset_index(drop=True)
    mine = ct.iloc[args.part::args.nparts]
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None] * 255
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None] * 255
    rows = []
    files = {}
    with torch.no_grad():
        for r in mine.itertuples():
            path = args.root / 'data/orion' / Path(r.file).name
            if path not in files:
                files[path] = pq.ParquetFile(path)
            row = files[path].read_row_group(int(r.row_group), columns=['image', 'image_if', 'inst', 'inst_cell']).to_pylist()[0]
            he, _, _, mask = decode_crop(row)
            x = torch.from_numpy(he).permute(2, 0, 1).float().to(dev)
            x = F.interpolate(x[None], size=(args.size, args.size), mode='bilinear', align_corners=False)
            x = (x - mean) / std
            p = torch.sigmoid(net(x))
            p = F.interpolate(p, size=(512, 512), mode='bilinear', align_corners=False)[0]
            lab = np.sort(labels[r.id])
            m = torch.from_numpy(mask.astype(np.int64)).to(dev).reshape(-1)
            k = torch.searchsorted(torch.from_numpy(lab).to(dev), m)
            ok = (k < len(lab)) & (torch.from_numpy(lab).to(dev)[k.clamp(max=len(lab) - 1)] == m)
            idx = torch.where(ok, k + 1, torch.zeros_like(k))
            s = torch.zeros(len(CHANNELS), len(lab) + 1, device=dev).index_add_(1, idx, p.reshape(len(CHANNELS), -1))
            cnt = torch.bincount(idx, minlength=len(lab) + 1).clamp(min=1)
            v = (s / cnt)[:, 1:].T.cpu().numpy()
            for j, l in enumerate(lab):
                rows.append((r.id, int(l), *v[j, [CHANNELS.index(c) for c in PROTEIN]]))
    out = args.root / args.out
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['crop_id', 'cell_label'] + PROTEIN).to_parquet(out / f'part{args.part:02d}.parquet')
    print('part', args.part, 'regions', len(mine), 'cells', len(rows))


if __name__ == '__main__':
    main()
