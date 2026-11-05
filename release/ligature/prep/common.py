"""Shared constants/helpers for the ORION-CRC conditional-recoverability pipeline."""
import os, sys, json, time, platform, logging

# Data workspace; set ORION_P to choose its location.
P = os.environ.get("ORION_P", os.getcwd())
DATA = os.path.join(P, "data", "orion")          # raw parquet + metadata
CELLS = os.path.join(P, "data", "cells")         # cell table + H&E patch shards
FEATS = os.path.join(P, "data", "features")      # frozen-encoder features
RUNS = os.path.join(P, "runs")                   # results
LOGS = os.path.join(RUNS, "logs")
HF_HOME = os.path.join(P, "data", "hf_cache")

CHANNELS = ["Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3", "CD8a",
            "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67",
            "Pan-CK", "SMA"]

# Marker panel and class. Dataset channel names == names used in the protocol (identity mapping,
# verified against original-orion-markers.csv at build time). Hoechst is NOT a target.
LINEAGE = ["Pan-CK", "E-cadherin", "SMA", "CD31", "CD45", "CD68", "CD163", "CD3e", "CD8a", "CD4", "CD20"]
STATE = ["Ki67", "PD-1", "PD-L1", "FOXP3", "CD45RO"]
# AF1 = POSITIVE ARTIFACT control (autofluorescence of RBC/collagen/elastin is visible in H&E: expected recoverable)
# Argo550 = blank channel = TRUE NULL (expected: insufficient evidence)
POS_CONTROL = ["AF1"]
NULL_CONTROL = ["Argo550"]
MARKERS = LINEAGE + STATE + POS_CONTROL + NULL_CONTROL
MARKER_CLASS = {**{m: "lineage" for m in LINEAGE}, **{m: "state" for m in STATE},
                "AF1": "positive_artifact_control", "Argo550": "blank_null_control"}
MARKER_IDX = [CHANNELS.index(m) for m in MARKERS]

CALL_MORPH, CALL_CTX, CALL_NONE = "morphology", "context", "insufficient_evidence"

# crop / patch geometry (px at 0.325 um/px)
MPP = 0.325
CROP = 512
PATCH_STORE = 96          # stored H&E patch size (centre-cropped to 64 or 96 at feature time)
PATCH_SIZES = (64, 96)    # X field-of-view sizes: 64 px = 20.8 um (primary), 96 px = 31.2 um (sensitivity)
BORDER_MARGIN = 48        # cells whose centroid is closer than this to the crop edge are dropped (= PATCH_STORE/2)
KS = (4, 8, 16)


def setup_logging(name, log_dir=None):
    log_dir = log_dir or LOGS
    os.makedirs(log_dir, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(os.path.join(log_dir, f"{name}.log")))
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def log_env(log):
    """Print paths and package versions."""
    log.info("host=%s python=%s", platform.node(), sys.version.split()[0])
    log.info("P=%s DATA=%s CELLS=%s FEATS=%s RUNS=%s HF_HOME=%s", P, DATA, CELLS, FEATS, RUNS,
             os.environ.get("HF_HOME", HF_HOME))
    for mod in ["numpy", "pandas", "scipy", "sklearn", "pyarrow", "PIL", "skimage", "tifffile",
                "torch", "transformers", "matplotlib", "huggingface_hub"]:
        try:
            m = __import__(mod)
            log.info("pkg %s=%s", mod, getattr(m, "__version__", "?"))
        except Exception:
            log.info("pkg %s=NOT_INSTALLED", mod)
    log.info("env SLURM_JOB_ID=%s SLURM_CPUS_PER_TASK=%s", os.environ.get("SLURM_JOB_ID"),
             os.environ.get("SLURM_CPUS_PER_TASK"))


def ncpu():
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK") or len(os.sched_getaffinity(0)))
    except Exception:
        return os.cpu_count() or 1
