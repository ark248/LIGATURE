#!/usr/bin/env python
"""Download the HEST-1k Xenium lung cohort (Vannan et al., pulmonary fibrosis; 20 samples, 19 patients)."""
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download, hf_hub_download

ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(Path(__file__).resolve().parents[1])))
metadata = ROOT / 'data/external/hest_meta/HEST_v1_3_0.csv'
if not metadata.exists():
    metadata.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download('MahmoodLab/hest', 'HEST_v1_3_0.csv', repo_type='dataset', local_dir=str(metadata.parent))
meta = pd.read_csv(metadata)
ids = sorted(meta[meta.dataset_title.str.startswith('Image-based spatial transcriptomics identifies molecular niche', na=False)].id)
pats = []
for i in ids:
    pats += [f'metadata/{i}.json', f'wsis/{i}.tif', f'transcripts/{i}_transcripts.parquet',
             f'xenium_seg/{i}_xenium_cell_seg.geojson.zip', f'tissue_seg/{i}_contours.geojson']
snapshot_download('MahmoodLab/hest', repo_type='dataset', allow_patterns=pats,
                  local_dir=str(ROOT / 'data/external/hest_lung'), max_workers=4)
print('done', len(ids), 'samples')
