#!/usr/bin/env python
"""build_cells.py -- turn ORION-CRC crops into a per-cell table + H&E patches (CPU).

Per whole-cell instance (inst_cell label) we record:
  case, sample, crop id, centroid (crop px and source-image px), whole-cell area,
  mean raw fluorescence inside the whole-cell mask for all 19 channels (y_<name>),
  nuclear geometry (area, perimeter, eccentricity) of the nucleus assigned to the cell
  (computed from the `inst` nucleus mask), and a 96x96 px H&E patch centred on the centroid
  (cropped to 64 or 96 px at feature time).
Drop rules (all counted and logged): tiny cell, too close to crop border, duplicate region
(cell lies in the interior of an EARLIER crop of the same sample -> keep one crop per overlapping
region, using recorded crop coordinates), no assigned nucleus, tiny nucleus.

Stages: meta (crop table) -> shards (one npz-pair per crop, resumable, multiprocessing) -> merge.
Usage: python build_cells.py --stage all [--smoke] [--files val] [--limit 20]
"""
import argparse, glob, io, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MIN_CELL_AREA = 40       # px (~4.2 um^2)
MIN_NUC_AREA = 20        # px
MIN_NUC_FRAC = 0.5       # >=50% of the nucleus' pixels must lie inside the cell it is assigned to


def decode_crop(row):
    import tifffile
    from PIL import Image
    he = np.array(Image.open(io.BytesIO(row["image"])).convert("RGB"))
    ifc = tifffile.imread(io.BytesIO(row["image_if"]))
    inst = tifffile.imread(io.BytesIO(row["inst"]))
    cell = tifffile.imread(io.BytesIO(row["inst_cell"]))
    return he, ifc, inst, cell


def cells_from_arrays(he, ifc, inst, cell, meta, earlier_boxes, pairs=None):
    """Pure function: arrays of ONE crop -> (DataFrame of kept cells, uint8 patches [n,96,96,3], counts)."""
    from skimage.measure import regionprops
    counts = {}
    H, W = cell.shape
    nmax = int(cell.max()) + 1
    flat = cell.ravel()
    area = np.bincount(flat, minlength=nmax)
    ids = np.nonzero(area[1:] > 0)[0] + 1
    counts["cells_total"] = int(len(ids))
    # centroid
    yy, xx = np.divmod(np.arange(flat.size), W)
    cx = np.bincount(flat, weights=xx, minlength=nmax)[ids] / area[ids]
    cy = np.bincount(flat, weights=yy, minlength=nmax)[ids] / area[ids]
    a = area[ids]
    keep = a >= MIN_CELL_AREA
    counts["drop_tiny_cell"] = int((~keep).sum())
    m = C.BORDER_MARGIN
    inb = (cx >= m) & (cx < W - m) & (cy >= m) & (cy < H - m)
    counts["drop_border"] = int((keep & ~inb).sum())
    keep &= inb
    gx, gy = meta["x_start"] + cx, meta["y_start"] + cy
    dup = np.zeros(len(ids), bool)
    for (x0, y0, x1, y1) in earlier_boxes:
        dup |= (gx >= x0 + m) & (gx < x1 - m) & (gy >= y0 + m) & (gy < y1 - m)
    counts["drop_duplicate_region"] = int((keep & dup).sum())
    keep &= ~dup
    # nucleus assignment by pixel overlap
    imax = int(inst.max()) + 1
    both = (cell > 0) & (inst > 0)
    key = cell[both].astype(np.int64) * imax + inst[both].astype(np.int64)
    uk, cnt = np.unique(key, return_counts=True)
    ov = pd.DataFrame({"cell": uk // imax, "nuc": uk % imax, "cnt": cnt})
    narea = np.bincount(inst.ravel(), minlength=imax)
    ov["frac"] = ov.cnt / narea[ov.nuc.values]
    ov = ov[ov.frac >= MIN_NUC_FRAC].sort_values("cnt", ascending=False)
    ov = ov.drop_duplicates("cell").drop_duplicates("nuc")
    nuc_of = dict(zip(ov.cell.values, ov.nuc.values))
    nuc = np.array([int(nuc_of.get(i, 0)) for i in ids], dtype=np.int64)
    counts["drop_no_nucleus"] = int((keep & (nuc == 0)).sum())
    keep &= nuc > 0
    tiny_n = np.zeros(len(ids), bool)
    tiny_n[nuc > 0] = narea[nuc[nuc > 0]] < MIN_NUC_AREA
    counts["drop_tiny_nucleus"] = int((keep & tiny_n).sum())
    keep &= ~tiny_n
    sel = np.nonzero(keep)[0]
    counts["cells_kept"] = int(len(sel))
    if len(sel) == 0:
        return None, None, counts
    ids_k, nuc_k = ids[sel], nuc[sel]
    # nuclear geometry from the nucleus mask
    nl = np.where(np.isin(inst, nuc_k), inst, 0)
    geo = {}
    for rp in regionprops(nl):
        geo[rp.label] = (rp.area, rp.perimeter, rp.eccentricity)
    g = np.array([geo[n] for n in nuc_k], float)
    # per-channel mean fluorescence inside whole-cell mask (raw intensities)
    Y = np.stack([np.bincount(flat, weights=ifc[c].ravel().astype(np.float64), minlength=nmax)[ids_k] / area[ids_k]
                  for c in range(ifc.shape[0])], 1)
    pair_set = set()
    if pairs is not None:
        pair_set = {(int(p["nucleus_id"]), int(p["cell_id"])) for p in pairs}
    linked = np.array([(int(n), int(c)) in pair_set for n, c in zip(nuc_k, ids_k)])
    # patches
    h = C.PATCH_STORE // 2
    cxi, cyi = np.rint(cx[sel]).astype(int), np.rint(cy[sel]).astype(int)
    patches = np.stack([he[y - h:y + h, x - h:x + h] for x, y in zip(cxi, cyi)]).astype(np.uint8)
    df = pd.DataFrame({
        "cell_uid": [f"{meta['id']}:{i}" for i in ids_k],
        "crop_id": meta["id"], "case": meta["case"], "sample": meta["sample"],
        "cell_label": ids_k.astype(np.int64), "nuc_label": nuc_k.astype(np.int64),
        "cx": cx[sel], "cy": cy[sel], "gx": gx[sel], "gy": gy[sel],
        "cell_area_px": area[ids_k].astype(np.int64),
        "nuc_area_px": g[:, 0], "nuc_perimeter_px": g[:, 1], "nuc_eccentricity": g[:, 2],
        "nuc_pair_linked": linked,
    })
    for c, name in enumerate(C.CHANNELS):
        df["y_" + name] = Y[:, c]
    return df, patches, counts


_PF = {}


def _worker(task):
    path, rg, meta, earlier, outdir = task
    import pyarrow.parquet as pq
    cid = meta["id"]
    tabf = os.path.join(outdir, cid + ".tab.parquet")
    if os.path.exists(tabf):
        return cid, json.load(open(os.path.join(outdir, cid + ".counts.json")))
    if path not in _PF:
        _PF[path] = pq.ParquetFile(path)
    row = _PF[path].read_row_group(rg, columns=["image", "image_if", "inst", "inst_cell", "pairs", "channel_names"]).to_pylist()[0]
    assert list(row["channel_names"]) == C.CHANNELS, "channel order differs from README"
    he, ifc, inst, cell = decode_crop(row)
    assert he.shape == (C.CROP, C.CROP, 3) and ifc.shape == (19, C.CROP, C.CROP), (he.shape, ifc.shape)
    df, patches, counts = cells_from_arrays(he, ifc, inst, cell, meta, earlier, row["pairs"])
    if df is not None:
        np.save(os.path.join(outdir, cid + ".patch.npy"), patches)
        df.to_parquet(tabf + ".tmp")
    else:
        pd.DataFrame().to_parquet(tabf + ".tmp")
    json.dump(counts, open(os.path.join(outdir, cid + ".counts.json"), "w"))
    os.replace(tabf + ".tmp", tabf)
    return cid, counts


def crop_table(files, log):
    import pyarrow.parquet as pq
    rows = []
    cols = ["id", "case", "sample", "source_index", "source_split", "x_start", "y_start", "x_end", "y_end"]
    for f in files:
        pf = pq.ParquetFile(f)
        t = pf.read(columns=cols).to_pandas()
        t["file"], t["row_group"] = f, np.arange(len(t))
        assert pf.metadata.num_row_groups == len(t), "expected one crop per row group"
        rows.append(t)
    t = pd.concat(rows, ignore_index=True).sort_values(["sample", "source_index"]).reset_index(drop=True)
    log.info("crops=%d cases=%d samples=%d", len(t), t.case.nunique(), t["sample"].nunique())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["meta", "shards", "merge", "all"])
    ap.add_argument("--files", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--out", default=C.CELLS)
    ap.add_argument("--workers", type=int, default=C.ncpu())
    ap.add_argument("--limit", type=int, default=0, help="only first N crops (per sample order) - testing")
    ap.add_argument("--smoke", action="store_true", help="val+test, 5 crops per sample (30 crops, 6 cases) -> <out>_smoke")
    a = ap.parse_args()
    if a.smoke:
        a.files, a.limit = ["val", "test"], 30        # 6 samples x 5 crops
        a.out = a.out.rstrip("/") + "_smoke"            # never touch the real cells table
    log = C.setup_logging("build_cells", os.path.join(a.out, "logs") if a.smoke else None)
    C.log_env(log)
    outdir = os.path.join(a.out, "shards")
    os.makedirs(outdir, exist_ok=True)
    files = [os.path.join(a.data, f + ".parquet") for f in a.files]
    # sanity: dataset channel list equals original-orion-markers.csv
    mk = os.path.join(a.data, "original-orion-markers.csv")
    if os.path.exists(mk):
        names = [l.strip() for l in open(mk) if l.strip()]
        assert names == C.CHANNELS, ("marker csv differs", names)
        log.info("channel mapping verified against original-orion-markers.csv (19 channels, identity mapping)")
    ct = crop_table(files, log)
    if a.limit:
        ct = ct.groupby("sample").head(max(1, a.limit // max(1, ct["sample"].nunique()) + 1)).head(a.limit)
        ct = ct.sort_values(["sample", "source_index"]).reset_index(drop=True)
        # NB: with --limit, the overlap rule only sees the retained crops
    ct.to_csv(os.path.join(a.out, "crops.csv"), index=False)
    if a.stage in ("shards", "all"):
        tasks = []
        for s, g in ct.groupby("sample"):
            boxes = []
            for r in g.itertuples():
                tasks.append((r.file, int(r.row_group), dict(id=r.id, case=r.case, sample=r.sample,
                              x_start=int(r.x_start), y_start=int(r.y_start)), list(boxes), outdir))
                boxes.append((int(r.x_start), int(r.y_start), int(r.x_end), int(r.y_end)))
        log.info("processing %d crops with %d workers", len(tasks), a.workers)
        t0 = time.time()
        if a.workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(a.workers) as pool:
                for i, (cid, c) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1)):
                    if i % 100 == 0:
                        log.info("crop %d/%d %s kept=%s elapsed=%.0fs", i, len(tasks), cid, c.get("cells_kept"), time.time() - t0)
        else:
            for i, t in enumerate(tasks):
                cid, c = _worker(t)
                if i % 20 == 0:
                    log.info("crop %d/%d %s kept=%s elapsed=%.0fs", i, len(tasks), cid, c.get("cells_kept"), time.time() - t0)
    if a.stage in ("merge", "all"):
        tabs, tot = [], {}
        for r in ct.itertuples():
            tf = os.path.join(outdir, r.id + ".tab.parquet")
            c = json.load(open(os.path.join(outdir, r.id + ".counts.json")))
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
            d = pd.read_parquet(tf)
            if len(d):
                d["crop_row"] = np.arange(len(d))
                tabs.append(d)
        cells = pd.concat(tabs, ignore_index=True)
        cells["cell_idx"] = np.arange(len(cells))
        cells.to_parquet(os.path.join(a.out, "cells.parquet"))
        summ = {"crops": int(len(ct)), "cells_per_case": cells.groupby("case").size().to_dict(), **tot,
                "cells_final": int(len(cells)), "cases": int(cells.case.nunique()),
                "params": dict(MIN_CELL_AREA=MIN_CELL_AREA, MIN_NUC_AREA=MIN_NUC_AREA, MIN_NUC_FRAC=MIN_NUC_FRAC,
                               BORDER_MARGIN=C.BORDER_MARGIN, PATCH_STORE=C.PATCH_STORE)}
        json.dump(summ, open(os.path.join(a.out, "cells_summary.json"), "w"), indent=1)
        log.info("DROP/KEEP COUNTS: %s", {k: v for k, v in tot.items()})
        log.info("final cells=%d cases=%d", len(cells), cells.case.nunique())
        log.info("wrote %s", os.path.join(a.out, "cells.parquet"))


if __name__ == "__main__":
    main()
