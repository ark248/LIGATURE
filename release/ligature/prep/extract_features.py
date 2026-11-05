#!/usr/bin/env python
"""extract_features.py -- frozen H&E encoder -> X (cell-centred patch embedding) and N (neighbour mean).

For every kept cell and each X field-of-view size s in {64, 96} px (20.8 / 31.2 um at 0.325 um/px):
  X_s  = encoder(CLS embedding) of the s x s px H&E patch centred on the cell, resized to 224x224.
  N_s,k = mean of X_s of the k nearest OTHER cells in the SAME crop whose patches do NOT overlap the
          cell's own patch (Chebyshev centroid distance >= s px, ranked by Euclidean distance);
          if fewer than k such cells exist all of them are used (needs >= MIN_NBR, else N is NaN -> cell dropped at fit time).
          k in {4, 8, 16} (8 = primary).
  The within-crop neighbour-SET swap null (N_perm) is generated in nested_fit.py (needs a seed).
Resumable per crop (features/crops/<crop>.npz); GPU fp16 autocast with fp32 fallback on non-finite output.
Stages: embed (GPU) -> assemble (CPU, writes X_p*.npy / N_p*_k*.npy aligned with cells.parquet)
Test-only encoders (NEVER for results): --encoder random | --encoder facebook/dinov2-small on CPU slice.
"""
import argparse, glob, json, os, sys, time, threading, queue
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

DEFAULT_MODEL = "owkin/phikon-v2"
DEFAULT_REVISION = "2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd"     # pinned commit of owkin/phikon-v2 (checked via HF API 2026-10-27)
MIN_NBR = 4


def neighbour_means(emb, cx, cy, s, ks=C.KS, min_nbr=MIN_NBR):
    """emb [n,D] float32, centroids in crop px. Returns {k: (N [n,D] float32 (NaN if too few), count [n])}."""
    n = len(emb)
    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])
    cheb = np.maximum(dx, dy)
    d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
    d2 = np.where(cheb >= s, d2, np.inf)          # eligible: patches do not overlap (also excludes self)
    order = np.argsort(d2, axis=1)
    n_elig = np.isfinite(d2).sum(1)
    out = {}
    for k in ks:
        kk = min(k, n - 1) if n > 1 else 0
        N = np.full_like(emb, np.nan)
        cnt = np.minimum(n_elig, k)
        for i in range(n):
            m = int(cnt[i])
            if m >= min_nbr:
                N[i] = emb[order[i, :m]].mean(0)
        out[k] = (N, cnt.astype(np.int16))
    return out


class Encoder:
    def __init__(self, name, cache_dir, device, revision=None):
        import torch
        self.torch, self.name, self.device = torch, name, device
        self.test_only = False
        if name == "random":
            self.test_only = True
            self.dim = 128
            g = np.random.default_rng(0)
            self.proj = torch.tensor(g.standard_normal((16 * 16 * 3, 128)) / 16.0, dtype=torch.float32)
            self.info = dict(name="random_projection_TEST_ONLY", dim=128, input=16)
            return
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name, cache_dir=cache_dir, revision=revision).to(device).eval()
        if device == "cuda":
            self.model = self.model.half()
        self.dim = int(self.model.config.hidden_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        from huggingface_hub import HfApi
        sha = None
        try:
            sha = HfApi().model_info(name, revision=revision).sha
        except Exception:
            sha = getattr(self.model.config, "_commit_hash", None)
        self.info = dict(name=name, revision=revision or sha, resolved_sha=sha, dim=self.dim, input=224,
                         embedding="CLS token of last_hidden_state", normalisation="ImageNet mean/std",
                         resize="bicubic upsample of s x s px patch to 224 x 224", dtype="fp16 autocast on GPU / fp32 CPU")
        self.test_only = device != "cuda" and name != DEFAULT_MODEL

    def __call__(self, patches_u8, batch=256):
        """patches_u8 [n,s,s,3] uint8 -> [n,D] float32"""
        torch = self.torch
        outs = []
        for i in range(0, len(patches_u8), batch):
            x = torch.from_numpy(patches_u8[i:i + batch]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            if self.name == "random":
                x = torch.nn.functional.interpolate(x, size=(16, 16), mode="area")
                outs.append((x.reshape(len(x), -1) @ self.proj).numpy())
                continue
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
            x = (x - self.mean) / self.std
            with torch.no_grad():
                if self.device == "cuda":
                    f = self.model(pixel_values=x.half()).last_hidden_state[:, 0].float()
                    if not torch.isfinite(f).all():           # fp16 overflow -> recompute in fp32
                        self.model.float()
                        f = self.model(pixel_values=x.float()).last_hidden_state[:, 0].float()
                        self.model.half()
                else:
                    f = self.model(pixel_values=x).last_hidden_state[:, 0].float()
            outs.append(f.cpu().numpy())
        return np.concatenate(outs).astype(np.float32)


def expected_keys(sizes):
    ks = []
    for s in sizes:
        ks.append(f"X_p{s}")
        for k in C.KS:
            ks += [f"N_p{s}_k{k}", f"nn_p{s}_k{k}"]
    return ks


def complete(path, sizes):
    try:
        z = np.load(path)
        return all(k in z.files for k in expected_keys(sizes))
    except Exception:
        return False


def center_crop(p, s):
    o = (p.shape[1] - s) // 2
    return np.ascontiguousarray(p[:, o:o + s, o:o + s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["embed", "assemble", "all"])
    ap.add_argument("--cells", default=C.CELLS)
    ap.add_argument("--out", default=C.FEATS)
    ap.add_argument("--encoder", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None, help="default: pinned revision for the default encoder")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME", C.HF_HOME) + "/hub")
    ap.add_argument("--sizes", nargs="+", type=int, default=list(C.PATCH_SIZES))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="first 40 crops only, writes to <out>_smoke")
    ap.add_argument("--max-crops", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        a.max_crops = a.max_crops or 40
        a.out = a.out.rstrip("/") + "_smoke"
    log = C.setup_logging(f"extract_features_part{a.part}", os.path.join(a.out, "logs"))
    C.log_env(log)
    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
    cells = pd.read_parquet(os.path.join(a.cells, "cells.parquet"), columns=["cell_idx", "crop_id", "cx", "cy"])
    crops = list(dict.fromkeys(cells.crop_id.values))
    if a.max_crops:
        crops = crops[:a.max_crops]
    if a.stage in ("embed", "all"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("device=%s cuda_available=%s", device, torch.cuda.is_available())
        if device == "cuda":
            log.info("gpu=%s mem=%.1fGB", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)
        rev = a.revision or (DEFAULT_REVISION if a.encoder == DEFAULT_MODEL else None)
        enc = Encoder(a.encoder, a.cache_dir, device, rev)
        info = dict(enc.info, patch_sizes_px=a.sizes, patch_sizes_um=[s * C.MPP for s in a.sizes], ks=list(C.KS),
                    min_neighbours=MIN_NBR, neighbour_rule="Chebyshev centroid distance >= patch size (non-overlapping patches), ranked by Euclidean distance, same crop",
                    TEST_ONLY=bool(enc.test_only), torch=torch.__version__)
        json.dump(info, open(os.path.join(a.out, "encoder_info.json"), "w"), indent=1)
        log.info("encoder info: %s", info)
        if enc.test_only:
            log.warning("*** TEST-ONLY ENCODER: outputs must never be used for reported results ***")
        mine = [c for i, c in enumerate(crops) if i % a.nparts == a.part]
        by_crop = {c: g for c, g in cells.groupby("crop_id", sort=False)}
        t0, ncell = time.time(), 0
        q = queue.Queue(maxsize=16)

        def loader():
            for c in mine:
                of = os.path.join(a.out, "crops", c + ".npz")
                if os.path.exists(of) and complete(of, a.sizes):
                    continue
                q.put((c, np.load(os.path.join(a.cells, "shards", c + ".patch.npy"))))
            q.put(None)
        threading.Thread(target=loader, daemon=True).start()
        done = 0
        while True:
            item = q.get()
            if item is None:
                break
            c, patches = item
            g = by_crop[c]
            assert len(g) == len(patches), (c, len(g), len(patches))
            res = {}
            for s in a.sizes:
                emb = enc(center_crop(patches, s), a.batch)
                res[f"X_p{s}"] = emb.astype(np.float16)
                nm = neighbour_means(emb, g.cx.values, g.cy.values, s)
                for k, (N, cnt) in nm.items():
                    res[f"N_p{s}_k{k}"] = N.astype(np.float16)
                    res[f"nn_p{s}_k{k}"] = cnt
            np.savez(os.path.join(a.out, "crops", c + ".npz.tmp.npz"), **res)
            os.replace(os.path.join(a.out, "crops", c + ".npz.tmp.npz"), os.path.join(a.out, "crops", c + ".npz"))
            done += 1
            ncell += len(patches)
            if done % 25 == 0 or done == 1:
                el = time.time() - t0
                log.info("crops done %d/%d cells=%d %.1f cells/s (x%d sizes) elapsed %.0fs", done, len(mine), ncell,
                         ncell / el, len(a.sizes), el)
        log.info("embed finished: %d crops, %d cells, %.0fs", done, ncell, time.time() - t0)
    if a.stage in ("assemble", "all"):
        parts = {}
        n_ok = 0
        want = expected_keys(a.sizes)
        for f in os.listdir(a.out):                       # remove stale assembled files from earlier runs with other sizes
            if f.endswith(".npy") and f != "cell_idx.npy" and f[:-4] not in want:
                os.remove(os.path.join(a.out, f))
                log.warning("removed stale %s", f)
        for c in crops:
            z = np.load(os.path.join(a.out, "crops", c + ".npz"))
            miss = [k for k in want if k not in z.files]
            assert not miss, f"crop {c} lacks {miss}: re-run embed stage with the same --sizes"
            for k in want:
                parts.setdefault(k, []).append(z[k])
        for k, v in parts.items():
            arr = np.concatenate(v)
            np.save(os.path.join(a.out, k + ".npy"), arr)
        sub = cells[cells.crop_id.isin(set(crops))]
        np.save(os.path.join(a.out, "cell_idx.npy"), sub.cell_idx.values)
        log.info("assembled %s for %d cells -> %s", sorted(parts), len(sub), a.out)


if __name__ == "__main__":
    main()
