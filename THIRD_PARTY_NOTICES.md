# Third-party data and models

The root MIT license applies to original LIGATURE software and its software
documentation. It does not replace the terms of the following resources.

- **ORION-CRC:** [csmp-hub/orion-crc](https://huggingface.co/datasets/csmp-hub/orion-crc),
  distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Original data: Lin et al., “High-plex immunofluorescence imaging and traditional
  histology of the same tissue section for discovering image-based biomarkers,”
  *Nature Cancer* 4, 1036–1052 (2023). Crop release and supplied segmentations are
  described in the dataset card. ORION-derived paper images/visualizations retain
  this attribution and license; modifications include cropping, cell overlays,
  intensity normalization, color mapping and statistical/model summaries.
- **HEST-1k and the lung Xenium cohort:**
  [MahmoodLab/hest](https://huggingface.co/datasets/MahmoodLab/hest). Obtain access
  and accept the dataset's terms directly. The lung study is attributed to Vannan
  et al., *Nature Genetics* 57, 647–658 (2025), as cited in the manuscript. Raw or
  derived lung cell tables are not distributed here.
- **Phikon-v2:** [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2).
  The model card specifies an Owkin non-commercial model license. Model weights
  are not bundled; consult and comply with the upstream license before use.
- **InstanSeg:** the optional H&E-geometry control downloads the official
  [brightfield nuclei model](https://github.com/instanseg/instanseg/releases/tag/instanseg_models_v0.1.1).
  The downloader records and verifies its SHA-256. Consult the downloaded model
  README for its model and training-data terms.
- **ROSIE and GigaTIME:** optional comparator code loads separately obtained
  published checkpoints. Those weights remain subject to their upstream terms.
  Their papers are cited in the manuscript. The GigaTIME network definition here
  is an author-written implementation compatible with the published architecture.

Python dependencies retain their respective licenses. No third-party checkpoints,
raw cohorts or patient-level result tables are bundled in this repository.
