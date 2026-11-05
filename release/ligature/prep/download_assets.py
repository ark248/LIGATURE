#!/usr/bin/env python
"""download_assets.py -- separate pre-download step (login/CPU node with internet). Compute nodes then run offline.
  --what data   : csmp-hub/orion-crc metadata + train/val/test parquet (~21 GB) into $P/data/orion  (resumable; sizes+checksums reported)
  --what model  : encoder snapshot (default owkin/phikon-v2 at the pinned revision) into $HF_HOME/hub
  --what all
Re-running is safe (hf_hub_download skips completed files).  Needs no token (public/ungated)."""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import extract_features as EF

DATASET = "csmp-hub/orion-crc"
META = ["README.md", "validation.json", "manifest.json", "crc-source-splits.csv", "original-orion-markers.csv",
        "split-audit.json", "source-audit.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=["data", "model", "all"])
    ap.add_argument("--encoder", default=EF.DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    log = C.setup_logging("download_assets")
    C.log_env(log)
    from huggingface_hub import hf_hub_download, snapshot_download, HfApi
    os.environ.setdefault("HF_HOME", C.HF_HOME)
    api = HfApi()
    if a.what in ("data", "all"):
        os.makedirs(C.DATA, exist_ok=True)
        info = api.dataset_info(DATASET, files_metadata=True)
        log.info("dataset %s sha=%s", DATASET, info.sha)
        files = {s.rfilename: s for s in info.siblings}
        log.info("files in repo: %s", sorted(files))
        want = list(META)
        for sp in a.splits:                              # parquet files live under data/ or at root: find by basename
            cand = [f for f in files if os.path.basename(f) == f"{sp}.parquet" or (f.startswith(f"data/{sp}") and f.endswith(".parquet"))]
            if not cand:
                cand = [f for f in files if f.endswith(".parquet") and sp in os.path.basename(f)]
            log.info("split %s -> %s", sp, cand)
            want += cand
        out = {}
        for f in want:
            if f not in files:
                log.warning("not in repo (skipping): %s", f)
                continue
            p = hf_hub_download(DATASET, f, repo_type="dataset", local_dir=C.DATA)
            sz = os.path.getsize(p)
            exp = files[f].size
            log.info("ok %-40s %12d bytes (expected %s)", f, sz, exp)
            if exp is not None and sz != exp:
                raise SystemExit(f"SIZE MISMATCH {f}: {sz} != {exp}")
            out[f] = sz
        # normalise layout: build_cells expects $DATA/{train,val,test}.parquet
        for sp in a.splits:
            tgt = os.path.join(C.DATA, f"{sp}.parquet")
            if not os.path.exists(tgt):
                for f in out:
                    if f.endswith(".parquet") and sp in os.path.basename(f):
                        os.symlink(os.path.abspath(os.path.join(C.DATA, f)), tgt)
                        break
        json.dump(dict(dataset=DATASET, sha=info.sha, files=out), open(os.path.join(C.DATA, "download_manifest.json"), "w"), indent=1)
    if a.what in ("model", "all"):
        rev = a.revision or (EF.DEFAULT_REVISION if a.encoder == EF.DEFAULT_MODEL else None)
        cache = os.path.join(os.environ["HF_HOME"], "hub")
        p = snapshot_download(a.encoder, revision=rev, cache_dir=cache,
                              ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.ot"])
        log.info("model snapshot: %s (revision %s)", p, rev)
        log.info("files: %s", sorted(os.listdir(p)))
        if not os.path.exists(os.path.join(p, "model.safetensors")):
            raise SystemExit("model.safetensors missing in snapshot")
        json.dump(dict(encoder=a.encoder, revision=rev, path=p), open(os.path.join(C.DATA, "encoder_download.json"), "w"), indent=1)
    log.info("download_assets done")


if __name__ == "__main__":
    main()
