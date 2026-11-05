#!/usr/bin/env python
"""Pixel arrays for a pixel-level virtual stain (U-Net) on ORION-CRC.

For every region with retained cells: H&E RGB and the 16 target channels at half
resolution (256x256, 0.65 um/pixel; log1p of raw intensity, area-averaged), and the
full-resolution whole-cell mask restricted to retained cells (uint16), so per-cell
predictions pool over exactly the cells of the cell table. Written in shards.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prep'))
from build_cells import decode_crop
from common import CHANNELS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_engression import MARKERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=8)
    ap.add_argument('--out', type=Path, default=Path('data/pixels'))
    args = ap.parse_args()
    out = args.root / args.out
    out.mkdir(parents=True, exist_ok=True)
    ct = pd.read_csv(args.root / 'data/cells/crops.csv')
    cells = pd.read_parquet(args.root / 'data/cells/cells.parquet', columns=['crop_id', 'cell_label'])
    labels = cells.groupby('crop_id').cell_label.apply(np.array)
    ct = ct[ct.id.isin(labels.index)].sort_values(['file', 'row_group']).reset_index(drop=True)
    mine = ct.iloc[args.part::args.nparts]
    pos = np.array([CHANNELS.index(m) for m in MARKERS])
    he_s, if_s, mk_s, ids, lab_s = [], [], [], [], []
    files = {}
    for r in mine.itertuples():
        path = args.root / 'data/orion' / Path(r.file).name
        if path not in files:
            files[path] = pq.ParquetFile(path)
        row = files[path].read_row_group(int(r.row_group), columns=['image', 'image_if', 'inst', 'inst_cell',
                                                                    'channel_names']).to_pylist()[0]
        assert list(row['channel_names']) == CHANNELS
        he, image, _, mask = decode_crop(row)
        assert image.shape == (19, 512, 512) and he.shape == (512, 512, 3)
        ifl = np.log1p(image[pos].astype(np.float32)).reshape(16, 256, 2, 256, 2).mean((2, 4))
        hes = he.reshape(256, 2, 256, 2, 3).mean((1, 3)).round().astype(np.uint8)
        lab = np.sort(labels[r.id])
        k = np.searchsorted(lab, mask)
        keep = (k < len(lab)) & (lab[np.minimum(k, len(lab) - 1)] == mask)
        # remapped mask: 0 background, j+1 for the j-th retained label in sorted order
        he_s.append(hes.transpose(2, 0, 1)); if_s.append(ifl.astype(np.float16))
        mk_s.append(np.where(keep, k + 1, 0).astype(np.uint16)); ids.append(r.id); lab_s.append(lab)
    np.savez(out / f'part{args.part:02d}.npz', he=np.stack(he_s), ifc=np.stack(if_s), mask=np.stack(mk_s),
             crop=np.array(ids), labels=np.concatenate(lab_s), offsets=np.cumsum([0] + [len(l) for l in lab_s]))
    print('part', args.part, 'regions', len(ids))


if __name__ == '__main__':
    main()
