#!/usr/bin/env python
"""Pixel-level virtual stain: U-Net from H&E (0.65 um/pixel) to 16 protein channels.

Same patient roles as spatial_engression.py. Loss = per-pixel MSE on standardized
log intensities + per-cell MSE on the cell-table targets after pooling predictions
over each retained cell's mask (with a learned per-channel affine map). Early
stopping on validation-patient cell MSE. Exports per-cell predictions for every
cell in se.load order, in that fold's training-standardized units.
"""
import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_engression as se


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU())


class GigaTIMEFinetune(nn.Module):
    """Published GigaTIME network (tensors-only load) with a new 16-marker output layer.

    Rows of the new layer for markers with a matched GigaTIME channel are initialized
    from that channel; inputs are upsampled to ~0.5 um/pixel and outputs resampled back.
    """
    MATCH = {'PD-1': 'PD-1', 'CD4': 'CD4', 'CD68': 'CD68', 'CD20': 'CD20', 'CD3e': 'CD3', 'CD8a': 'CD8',
             'PD-L1': 'PD-L1', 'Pan-CK': 'CK', 'Ki67': 'Ki67'}

    def __init__(self, weights, out=16):
        super().__init__()
        from gigatime_arch import load_gigatime
        from gigatime_infer import CHANNELS
        self.body = load_gigatime(weights)
        old = self.body.final
        new = nn.Conv2d(old.in_channels, out, 1)
        with torch.no_grad():
            for j, m in enumerate(se.MARKERS):
                if m in self.MATCH:
                    k = CHANNELS.index(self.MATCH[m])
                    new.weight[j] = old.weight[k]; new.bias[j] = old.bias[k]
        self.body.final = new
        self.cell_a = nn.Parameter(torch.ones(out)); self.cell_b = nn.Parameter(torch.zeros(out))

    def forward(self, x):
        n = x.shape[-1]
        y = self.body(F.interpolate(x, size=(336, 336), mode='bilinear', align_corners=False))
        return F.interpolate(y, size=(n, n), mode='bilinear', align_corners=False)


class UNet(nn.Module):
    def __init__(self, ch=(32, 64, 128, 256, 384), out=16):
        super().__init__()
        self.down = nn.ModuleList()
        c = 3
        for o in ch:
            self.down.append(block(c, o)); c = o
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        for o in reversed(ch[:-1]):
            self.up.append(nn.ConvTranspose2d(c, o, 2, stride=2)); self.dec.append(block(2 * o, o)); c = o
        self.head = nn.Conv2d(c, out, 1)
        self.cell_a = nn.Parameter(torch.ones(out)); self.cell_b = nn.Parameter(torch.zeros(out))

    def forward(self, x):
        skips = []
        for k, d in enumerate(self.down):
            x = d(x)
            if k < len(self.down) - 1:
                skips.append(x); x = F.max_pool2d(x, 2)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = d(torch.cat([u(x), s], 1))
        return self.head(x)


def pool_cells(pred, mask, ncell):
    """pred: [B,16,256,256]; mask: [B,512,512] (0 bg, j+1 cell j); returns list of [n_b,16]."""
    up = pred.repeat_interleave(2, -1).repeat_interleave(2, -2)  # 512x512
    out = []
    for b in range(len(pred)):
        m = mask[b].reshape(-1).long()
        p = up[b].reshape(16, -1)
        s = torch.zeros(16, ncell[b] + 1, device=pred.device).index_add_(1, m, p)
        cnt = torch.bincount(m, minlength=ncell[b] + 1).clamp(min=1)
        out.append((s / cnt)[:, 1:].T)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=271027)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', type=Path, default=(Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))) / 'runs/unet_stain'))
    ap.add_argument('--max-parts', type=int, default=0)
    ap.add_argument('--arch', choices=['unet', 'gigatime'], default='unet')
    ap.add_argument('--gigatime-weights', type=Path, default=Path('checkpoints/external/GigaTIME/model.pth'))
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed + args.fold)
    out = args.root / args.out / f'fold{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    cells, x, y, crops, roles = se.load(args.root, args.fold, args.seed, 8)
    key = {(c, int(l)): i for i, (c, l) in enumerate(zip(cells.crop_id, cells.cell_label))}
    parts = sorted((args.root / 'data/pixels').glob('part*.npz'))
    if args.max_parts:
        parts = parts[:args.max_parts]
    HE, IF, MK, IDX, NC, ROLE = [], [], [], [], [], []
    for p in parts:
        z = np.load(p)
        # read each array once; keep copies of slices so shard arrays can be freed
        off, labels_all, crops_p = z['offsets'], z['labels'], z['crop']
        he_all, if_all, mk_all = z['he'], z['ifc'], z['mask']
        for j, crop in enumerate(crops_p):
            lab = labels_all[off[j]:off[j + 1]]
            idx = np.array([key.get((crop, int(l)), -1) for l in lab])
            if (idx < 0).any():
                continue
            HE.append(he_all[j].copy()); IF.append(if_all[j].copy()); MK.append(mk_all[j].astype(np.int16))
            IDX.append(idx); NC.append(len(lab))
            ROLE.append(roles[cells.case.iat[idx[0]]])
        del he_all, if_all, mk_all
    ROLE = np.array(ROLE)
    log(f'regions {len(HE)}: ' + str({r: int((ROLE == r).sum()) for r in ['train', 'val', 'test']}))
    HE = torch.from_numpy(np.stack(HE)); IF = torch.from_numpy(np.stack(IF)); MK = torch.from_numpy(np.stack(MK))
    tr = np.flatnonzero(ROLE == 'train'); va = np.flatnonzero(ROLE == 'val')
    sub = np.random.default_rng(0).choice(tr, min(300, len(tr)), replace=False)
    pm = IF[sub].float().mean((0, 2, 3)); ps = IF[sub].float().std((0, 2, 3)) + 1e-6
    hm = torch.tensor([0.485, 0.456, 0.406]) * 255; hs = torch.tensor([0.229, 0.224, 0.225]) * 255
    Yt = torch.from_numpy(y)
    if args.arch == 'gigatime':
        net = GigaTIMEFinetune(args.root / args.gigatime_weights).to(dev)
    else:
        net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(tr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps + 5, pct_start=.1)

    def batch(ix, aug):
        h = (HE[ix].float() - hm[:, None, None]) / hs[:, None, None]
        t = (IF[ix].float() - pm[:, None, None]) / ps[:, None, None]
        m = MK[ix].long()
        if aug:
            k = np.random.randint(4); fl = np.random.rand() < .5
            h, t, m = [torch.rot90(v, k, (-2, -1)) for v in (h, t, m)]
            if fl:
                h, t, m = [torch.flip(v, (-1,)) for v in (h, t, m)]
        return h.to(dev), t.to(dev), m.to(dev)

    def cell_loss(pred, m, ix):
        pc = pool_cells(pred, m, [NC[i] for i in ix])
        p = torch.cat(pc) * net.cell_a + net.cell_b
        target = torch.cat([Yt[IDX[i]] for i in ix]).to(dev)
        return ((p - target) ** 2).mean()

    best, best_state, hist, start = np.inf, None, [], 0
    # per-epoch checkpoint so a preempted (requeued) job resumes where it stopped
    ck = out / 'ckpt.pt'
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s['net']); opt.load_state_dict(s['opt'])
        best, best_state, hist, start = s['best'], s['best_state'], s['hist'], s['epoch'] + 1
        st = s['np_rng']
        np.random.set_state(('MT19937', st['key'].numpy().astype(np.uint32), int(st['pos']), int(st['has_gauss']),
                             float(st['cached'])))
        torch.set_rng_state(s['torch_rng'])
        for _ in range(start * int(np.ceil(len(tr) / args.batch))):
            sched.step()
        log(f'resumed after epoch {start - 1}')
    for epoch in range(start, args.epochs):
        net.train(); perm = np.random.permutation(tr); tot = 0.
        for s in range(0, len(perm), args.batch):
            ix = perm[s:s + args.batch]
            h, t, m = batch(ix, True)
            pred = net(h)
            loss = ((pred - t) ** 2).mean() + cell_loss(pred, m, ix)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(ix)
        net.eval(); vl = 0.
        with torch.no_grad():
            for s in range(0, len(va), 32):
                ix = va[s:s + 32]; h, t, m = batch(ix, False)
                vl += cell_loss(net(h), m, ix).item() * len(ix)
        vl /= len(va); hist.append(dict(epoch=epoch, train=tot / len(tr), val_cell_mse=vl))
        log(f'epoch {epoch}: train {tot / len(tr):.4f} val cell mse {vl:.4f}')
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}
        st = np.random.get_state()
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), best=best, best_state=best_state, hist=hist,
                        epoch=epoch, torch_rng=torch.get_rng_state(),
                        np_rng=dict(key=torch.from_numpy(st[1].astype(np.int64)), pos=st[2], has_gauss=st[3],
                                    cached=st[4])), str(ck) + '.tmp')
        Path(str(ck) + '.tmp').replace(ck)
    net.load_state_dict(best_state); net.eval()
    pred_cells = np.full((len(cells), 16), np.nan, np.float32)
    with torch.no_grad():
        for s in range(0, len(HE), 32):
            ix = np.arange(s, min(s + 32, len(HE))); h, t, m = batch(ix, False)
            pc = pool_cells(net(h), m, [NC[i] for i in ix])
            for i, p in zip(ix, pc):
                pred_cells[IDX[i]] = (p * net.cell_a + net.cell_b).cpu().numpy()
    np.save(out / 'cell_pred.npy', pred_cells)
    torch.save(net.state_dict(), out / 'unet.pt')
    te = cells.case.map(roles).to_numpy() == 'test'
    ok = te & ~np.isnan(pred_cells[:, 0])
    json.dump(dict(fold=args.fold, best_val_cell_mse=best, history=hist,
                   test_cell_mse=float(np.mean((pred_cells[ok] - y[ok]) ** 2)),
                   missing_cells=int(np.isnan(pred_cells[:, 0]).sum())), open(out / 'summary.json', 'w'), indent=2)
    ck.unlink(missing_ok=True)
    log(f'DONE test cell mse {np.mean((pred_cells[ok] - y[ok]) ** 2):.4f}')


if __name__ == '__main__':
    main()
