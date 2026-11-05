#!/usr/bin/env python
"""Run the published ROSIE model (Wu et al., Nat Commun 2025) on ORION H&E, one patch per cell.

ROSIE predicts the expression of 50 CODEX markers at the center of a 128x128 H&E patch
(resized to 224, ImageNet normalization; torchvision ConvNeXt-Small). We assume ROSIE's
H&E resolution equals typical CODEX resolution, 0.377 um/pixel, so a patch spans 48 um,
i.e. 148 ORION pixels (0.325 um/pixel), centered on each retained cell's centroid and
zero-padded at region borders as in the authors' code. Weights are loaded with the
restricted (weights_only) unpickler; no downloaded code is executed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prep'))
from build_cells import decode_crop

ROSIE = ['DAPI', 'CD45', 'CD68', 'CD14', 'PD1', 'FoxP3', 'CD8', 'HLA-DR', 'PanCK', 'CD3e', 'CD4', 'aSMA', 'CD31',
         'Vimentin', 'CD45RO', 'Ki67', 'CD20', 'CD11c', 'Podoplanin', 'PDL1', 'GranzymeB', 'CD38', 'CD141', 'CD21',
         'CD163', 'BCL2', 'LAG3', 'EpCAM', 'CD44', 'ICOS', 'GATA3', 'Gal3', 'CD39', 'CD34', 'TIGIT', 'ECad', 'CD40',
         'VISTA', 'HLA-A', 'MPO', 'PCNA', 'ATM', 'TP63', 'IFNg', 'Keratin8/18', 'IDO1', 'CD79a', 'HLA-E', 'CollagenIV',
         'CD66']
TO_ORION = {'CD31': 'CD31', 'CD45': 'CD45', 'CD68': 'CD68', 'CD4': 'CD4', 'FoxP3': 'FOXP3', 'CD8': 'CD8a',
            'CD45RO': 'CD45RO', 'CD20': 'CD20', 'PDL1': 'PD-L1', 'CD3e': 'CD3e', 'CD163': 'CD163', 'ECad': 'E-cadherin',
            'PD1': 'PD-1', 'Ki67': 'Ki67', 'PanCK': 'Pan-CK', 'aSMA': 'SMA'}


def load_rosie(path):
    allowed = [(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype]
    allowed += [type(np.dtype(t)) for t in ('float64', 'float32', 'int64', 'int32', 'bool')]
    with torch.serialization.safe_globals(allowed):
        ck = torch.load(path, map_location='cpu', weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in ck['model_state_dict'].items()}
    m = torchvision.models.convnext_small(weights=None)
    m.classifier[2] = torch.nn.Linear(m.classifier[2].in_features, len(ROSIE))
    m.load_state_dict(sd, strict=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--weights', type=Path, default=Path('checkpoints/external/ROSIE/best_model_single.pth'))
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=4)
    ap.add_argument('--patch', type=int, default=148)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--out', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/rosie_stain/cells'))
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_rosie(args.root / args.weights).to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    ct = pd.read_csv(args.root / 'data/cells/crops.csv')
    cells = pd.read_parquet(args.root / 'data/cells/cells.parquet', columns=['crop_id', 'cell_label', 'cx', 'cy'])
    by = {k: g for k, g in cells.groupby('crop_id')}
    ct = ct[ct.id.isin(by)].sort_values(['file', 'row_group']).reset_index(drop=True)
    mine = ct.iloc[args.part::args.nparts]
    h = args.patch // 2
    keep = [ROSIE.index(k) for k in TO_ORION]
    rows, files = [], {}
    with torch.no_grad():
        for r in mine.itertuples():
            path = args.root / 'data/orion' / Path(r.file).name
            if path not in files:
                files[path] = pq.ParquetFile(path)
            row = files[path].read_row_group(int(r.row_group), columns=['image', 'image_if', 'inst', 'inst_cell']).to_pylist()[0]
            he = decode_crop(row)[0].astype(np.float32) / 255.
            img = torch.from_numpy(np.pad(he, ((h, h), (h, h), (0, 0)))).permute(2, 0, 1).to(dev)
            g = by[r.id]
            cx, cy = np.round(g.cx.to_numpy()).astype(int), np.round(g.cy.to_numpy()).astype(int)
            preds = []
            for s in range(0, len(g), args.batch):
                patches = torch.stack([img[:, y:y + args.patch, x:x + args.patch]
                                       for x, y in zip(cx[s:s + args.batch], cy[s:s + args.batch])])
                patches = F.interpolate(patches, size=(224, 224), mode='bilinear', antialias=True, align_corners=False)
                preds.append(net((patches - mean) / std)[:, keep].float().cpu().numpy())
            v = np.concatenate(preds)
            for j, l in enumerate(g.cell_label.to_numpy()):
                rows.append((r.id, int(l), *v[j]))
    out = args.root / args.out
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['crop_id', 'cell_label'] + list(TO_ORION.values())).to_parquet(out / f'part{args.part:02d}.parquet')
    print('part', args.part, 'regions', len(mine), 'cells', len(rows))


if __name__ == '__main__':
    main()
