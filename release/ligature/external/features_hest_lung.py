#!/usr/bin/env python
"""Phikon-v2 cell embeddings for the HEST lung cohort, matching data/features/X_p64.npy.

Same encoder, pinned revision, preprocessing and field of view as ORION: the CLS token of a
20.8 um H&E patch centred on each cell (98 px at 0.2125 um/px; ORION: 64 px at 0.325 um/px),
bicubically resized to 224 px with ImageNet normalization. Border patches are padded white.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PACKAGE)))
sys.path.insert(0, str(PACKAGE / 'prep'))
from extract_features import DEFAULT_MODEL, DEFAULT_REVISION, Encoder  # noqa: E402
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', type=Path, default=ROOT / 'cohorts/lung')
    ap.add_argument('--fov-um', type=float, default=20.8)
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = Encoder(DEFAULT_MODEL, C.HF_HOME + '/hub', dev, DEFAULT_REVISION)
    cells = pd.read_parquet(args.cohort / 'data/cells/cells.parquet')
    X = np.zeros((len(cells), enc.dim), np.float16)
    for sid, g in cells.groupby('sample', sort=True):
        img = tifffile.imread(ROOT / f'data/external/hest_lung/wsis/{sid}.tif', level=0)
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        img = img[..., :3]
        s = int(round(args.fov_um / float(g.px_um.iat[0])))
        h = s // 2
        pad = np.pad(img, ((h + 1, h + 1), (h + 1, h + 1), (0, 0)), constant_values=255)
        cx = np.round(g.he_x.to_numpy()).astype(int) + h + 1
        cy = np.round(g.he_y.to_numpy()).astype(int) + h + 1
        for b in range(0, len(g), 4096):
            pt = np.stack([pad[y - h:y - h + s, x - h:x - h + s] for x, y in zip(cx[b:b + 4096], cy[b:b + 4096])])
            X[g.index[b:b + 4096]] = enc(pt, batch=args.batch).astype(np.float16)
        print(sid, len(g), 'cells, patch', s, 'px', flush=True)
        del img, pad
    out = args.cohort / 'data/features'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'X_p64.npy', X)
    np.save(out / 'cell_idx.npy', cells.cell_idx.to_numpy())
    (out / 'encoder_info.json').write_text(json.dumps({**enc.info, 'fov_um': args.fov_um, 'device': dev}, indent=1))
    print('saved', X.shape)


if __name__ == '__main__':
    main()
