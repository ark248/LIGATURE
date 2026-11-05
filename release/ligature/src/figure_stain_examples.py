#!/usr/bin/env python
"""Raw virtual-stain outputs on held-out regions (inspection figure, not used for any number).

For test-patient regions of one rotation: (a) pixel maps of the measured IF, the in-house
U-Net, the published GigaTIME network (sigmoid of the matched channel, run as in
gigatime_infer.py) and GigaTIME fine-tuned on training patients; (b) per-cell values
(training-standardized units, one shared color scale per marker) of the measured assay and
of each stain, painted on the whole-cell segmentation masks. ROSIE predicts one value per
cell from a patch around its centroid, so it appears only in (b).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prep'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_engression as se
from build_cells import decode_crop
from gigatime_arch import load_gigatime
from gigatime_infer import CHANNELS
from unet_stain import UNet, GigaTIMEFinetune

SHOW = ['Pan-CK', 'CD8a', 'CD68', 'PD-L1']
GT = {'Pan-CK': 'CK', 'CD8a': 'CD8', 'CD68': 'CD68', 'PD-L1': 'PD-L1'}


def paint(local, values):
    """local: [512,512] mask with ids 0 (background), j+1 for cell j; values [n] -> image."""
    img = np.full(local.shape, np.nan, np.float32)
    img[local > 0] = values[local[local > 0] - 1]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--regions', type=int, default=3)
    ap.add_argument('--out', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/stain_examples'))
    args = ap.parse_args()
    R = args.root
    out = R / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    a = json.loads((R / f'runs/ligature_v3/fold{args.fold}/args.json').read_text())
    cells, x, y, crops, roles = se.load(R, args.fold, int(a['seed']), int(a['pcadim']))
    M = se.MARKERS
    idx_of = {(c, int(l)): i for i, (c, l) in enumerate(zip(cells.crop_id, cells.cell_label))}
    # per-cell predictions of every stain (se.load order)
    tnet = se.CellGraphNet(x.shape[1], d=int(a['d']), layers=int(a['layers']))
    tnet.load_state_dict(torch.load(Path(a['mean_from']) / f'fold{args.fold}' / 'mean.pt'))
    te = [c for c in crops if c['role'] == 'test']
    with torch.no_grad():
        ftr = se.predict_mean(tnet, te, x, y)
    pred = {n: np.load(R / f'runs/{d}/fold{args.fold}/cell_pred.npy') for n, d in
            [('U-Net', 'unet_stain'), ('ROSIE', 'rosie_stain'), ('GigaTIME', 'gigatime_stain'),
             ('GigaTIME fine-tuned', 'gigatime_ft_stain')]}
    # candidate regions: test patients, many cells, visible tumor and CD8 structure
    score = {c['crop']: (len(c['ii']), y[c['ii'], M.index('CD8a')].std() * y[c['ii'], M.index('Pan-CK')].std())
             for c in te if len(c['ii']) >= 80}
    by_case = {}
    for c in te:
        if c['crop'] in score:
            by_case.setdefault(c['case'], []).append(c)
    picks = []
    for case in sorted(by_case)[:args.regions]:
        picks.append(max(by_case[case], key=lambda c: score[c['crop']][1]))
    # pixel shards: find picked crops
    want = {c['crop'] for c in picks}
    shard = {}
    for p in sorted((R / 'data/pixels').glob('part*.npz')):
        z = np.load(p)
        names = list(z['crop'])
        hit = [n for n in names if n in want]
        if hit:
            he, ifc, mk, off, lab = z['he'], z['ifc'], z['mask'], z['offsets'], z['labels']
            for n in hit:
                j = names.index(n)
                shard[n] = dict(he=he[j].copy(), ifc=ifc[j].copy(), mask=mk[j].astype(np.int64),
                                labels=lab[off[j]:off[j + 1]].copy())
            del he, ifc, mk
        if len(shard) == len(want):
            break
    unet = UNet(); unet.load_state_dict(torch.load(R / f'runs/unet_stain/fold{args.fold}/unet.pt', map_location='cpu')); unet.eval()
    w = R / 'checkpoints/external/GigaTIME/model.pth'
    gft = GigaTIMEFinetune(w); gft.load_state_dict(torch.load(R / f'runs/gigatime_ft_stain/fold{args.fold}/unet.pt', map_location='cpu')); gft.eval()
    gpub = load_gigatime(w, num_classes=len(CHANNELS)).eval()
    hm = torch.tensor([0.485, 0.456, 0.406])[:, None, None] * 255
    hs = torch.tensor([0.229, 0.224, 0.225])[:, None, None] * 255
    ct = pd.read_csv(R / 'data/cells/crops.csv').set_index('id')
    for c in picks:
        d = shard[c['crop']]
        # shard mask ids j+1 refer to labels[j] of the cell table
        labels_sorted = np.asarray(d['labels'])
        h = (torch.from_numpy(d['he']).float() - hm) / hs
        with torch.no_grad():
            pu = unet(h[None])[0].numpy()
            pf = gft(h[None])[0].numpy()
            r = ct.loc[c['crop']]
            row = pq.ParquetFile(Path(r.file)).read_row_group(int(r.row_group), columns=['image', 'image_if', 'inst', 'inst_cell']).to_pylist()[0]
            he_full = decode_crop(row)[0]
            xg = F.interpolate(torch.from_numpy(he_full).permute(2, 0, 1).float()[None], size=(336, 336), mode='bilinear', align_corners=False)
            pg = torch.sigmoid(gpub((xg - hm) / hs))[0].numpy()
        # per-cell values, in sorted-label order
        ii = np.array([idx_of[(c['crop'], int(l))] for l in labels_sorted])
        local = d['mask']
        tpos = {i: k for k, i in enumerate(c['ii'])}
        tr_vals = np.stack([ftr[c['crop']][tpos[i]] for i in ii])
        percell = {'Measured': y[ii], 'Transformer': tr_vals, **{n: v[ii] for n, v in pred.items()}}
        # sanity: pooled measured pixels vs cell-table values
        cols = [('Measured IF', lambda m: d['ifc'][M.index(m)]), ('U-Net', lambda m: pu[M.index(m)]),
                ('GigaTIME (published)', lambda m: pg[CHANNELS.index(GT[m])]),
                ('GigaTIME fine-tuned', lambda m: pf[M.index(m)])]
        fig, ax = plt.subplots(len(SHOW) + 1, len(cols), figsize=(3.0 * len(cols), 3.0 * (len(SHOW) + 1)))
        ax[0, 0].imshow(d['he'].transpose(1, 2, 0)); ax[0, 0].set_title(f"H&E  {c['case']}  (test patient)", fontsize=9)
        ax[0, 1].imshow(he_full); ax[0, 1].set_title('H&E full resolution (GigaTIME input)', fontsize=9)
        rng = np.random.default_rng(0)
        cm = rng.random((local.max() + 1, 3)); cm[0] = 0
        ax[0, 2].imshow(cm[local]); ax[0, 2].set_title(f'cell masks ({len(labels_sorted)} cells)', fontsize=9)
        ax[0, 3].axis('off')
        for k, m in enumerate(SHOW):
            for j, (name, fn) in enumerate(cols):
                im = fn(m)
                lo, hi = np.percentile(im, [1, 99.5])
                ax[k + 1, j].imshow(im, cmap='magma', vmin=lo, vmax=hi)
                ax[k + 1, j].set_title(f'{name}: {m}' + (f' ({GT[m]})' if 'published' in name else ''), fontsize=9)
        for v in ax.ravel():
            v.set_xticks([]); v.set_yticks([])
        fig.suptitle('Raw pixel outputs (each panel scaled to its own 1-99.5th percentile)', fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"pixels_{c['case']}_{c['crop'].split('_')[-1]}.png", dpi=110)
        plt.close(fig)
        names = list(percell)
        fig, ax = plt.subplots(len(SHOW), len(names), figsize=(2.6 * len(names), 2.7 * len(SHOW)))
        for k, m in enumerate(SHOW):
            j0 = M.index(m)
            lo, hi = np.percentile(percell['Measured'][:, j0], [2, 98])
            for j, n in enumerate(names):
                img = paint(local, percell[n][:, j0])
                ax[k, j].imshow(np.ones((*img.shape, 3)) * .93)
                ax[k, j].imshow(img, cmap='viridis', vmin=lo, vmax=hi)
                r_ = np.corrcoef(percell[n][:, j0], percell['Measured'][:, j0])[0, 1]
                ax[k, j].set_title(f'{n}: {m}' + ('' if n == 'Measured' else f'\nr={r_:.2f}, sd ratio={percell[n][:, j0].std() / percell["Measured"][:, j0].std():.2f}'), fontsize=8)
                ax[k, j].set_xticks([]); ax[k, j].set_yticks([])
        fig.suptitle(f"Per-cell values on segmentation masks, {c['case']} (shared color scale per marker = measured 2-98th pct.)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"cells_{c['case']}_{c['crop'].split('_')[-1]}.png", dpi=110)
        plt.close(fig)
        print(c['crop'], 'cells', len(ii), 'saved', flush=True)


if __name__ == '__main__':
    main()
