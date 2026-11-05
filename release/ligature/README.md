# Core LIGATURE implementation

This is the single model source tree for the paper. It includes the frozen stain,
joint residual generator, data preparation, direct comparators, ablation switches,
spatial queries, analytic bilinear means and external-cohort preprocessing.

## Environment and workspace

Install `requirements.txt` using Python 3.11, as described in the root README.
Run the following commands from the **repository root**. By default the data
workspace is this package directory:

```bash
export LIGATURE_DATA_ROOT="$PWD/release/ligature"
export ORION_P="$LIGATURE_DATA_ROOT"
export P="$LIGATURE_DATA_ROOT"
```

`LIGATURE_DATA_ROOT` may point to a separate writable workspace instead. Preserve
the `data/`, `cohorts/` and `runs/` layout beneath it. Downloaded data, embeddings
and fitted outputs are generated locally and are not part of the source release.

## ORION preprocessing and reference evaluation

```bash
python release/ligature/prep/download_assets.py
python release/ligature/prep/build_cells.py --stage all
python release/ligature/prep/extract_features.py

for fold in 0 1 2 3 4; do
  python release/ligature/src/spatial_engression.py --fold "$fold" \
    --root "$LIGATURE_DATA_ROOT" --out "$LIGATURE_DATA_ROOT/runs/spatial_engression"
  python release/ligature/src/direct_query_baseline.py --fold "$fold" \
    --run "$LIGATURE_DATA_ROOT/runs/spatial_engression"
  python release/ligature/src/ligature.py --fold "$fold" \
    --root "$LIGATURE_DATA_ROOT" --out "$LIGATURE_DATA_ROOT/runs/ligature_v3" \
    --mean-from "$LIGATURE_DATA_ROOT/runs/spatial_engression" \
    --frozen-mean --calibrate --dropout 0.1 --wd 1e-3 --epochs 15
  ln -sfn "../../spatial_engression/fold$fold/direct.npz" \
    "$LIGATURE_DATA_ROOT/runs/ligature_v3/fold$fold/direct.npz"
done

python release/ligature/src/analyze_spatial_engression.py --run "$LIGATURE_DATA_ROOT/runs/ligature_v3"
python release/ligature/src/analyze_unseen_statistics.py --run "$LIGATURE_DATA_ROOT/runs/ligature_v3"
python release/ligature/src/significance.py --run "$LIGATURE_DATA_ROOT/runs/ligature_v3"
```

These are the reference 24-training/8-validation/8-test rotations. The reported
independent 20/2/10/8 calibration protocol is implemented separately in
[research/calibration](../../research/calibration/README.md).

For contacts, neighborhoods and identification bounds:

```bash
for fold in 0 1 2 3 4; do
  python release/ligature/src/eval_contact.py --fold "$fold" --run "$LIGATURE_DATA_ROOT/runs/ligature_v3"
  python release/ligature/src/eval_neighborhoods.py --fold "$fold" --run "$LIGATURE_DATA_ROOT/runs/ligature_v3"
  python release/ligature/src/identification_sdp.py --fold "$fold" --run "$LIGATURE_DATA_ROOT/runs/ligature_v3"
done
python release/ligature/src/analyze_contact.py --run "$LIGATURE_DATA_ROOT/runs/ligature_v3"
python release/ligature/src/analyze_neighborhoods.py --run "$LIGATURE_DATA_ROOT/runs/ligature_v3"
python release/ligature/src/summarize_identification.py --run "$LIGATURE_DATA_ROOT/runs/ligature_v3"
```

## Comparators and ablations

- `src/direct_graph.py --with-stain`: matched graph predictor of statistics.
- `src/direct_contact.py`: graph predictor trained on contact statistics.
- `src/unet_stain.py`: pixel-level stain.
- `src/rosie_infer.py`, `src/rosie_mean.py`, `src/gigatime_infer.py`,
  `src/gigatime_mean.py`: optional published stains with separately obtained weights.
- `src/simulate_cohort.py`, `src/analyze_simulation.py`: known-law simulations.
- `src/bias_decomposition.py`, `src/common_reference.py`, `src/collect_backbones.py`:
  dependence decomposition and common-scale/cross-stain summaries.

Use each entry point's `--help` for its arguments. An external cell predictor can
be wrapped with `--mean-kind file --mean-file path/fold{fold}/cell_pred.npy`, with
predictions in the same cell order and training-standardized units as the loader.

The generator exposes `--lam-region 0 --lam-patient 0` for cell-only scoring,
`--scales none` for no fields, `--patient-dim 0` for no shared patient latent,
`--stationary`, `--no-stain-input`, `--no-geom`, `--no-contrasts`,
`--lam-cell 0`, `--model-seed N` and `--tail-df N`. Each variant should write to
its own `--out`. `--objective ml` evaluates the conditional random-feature
Gaussian-likelihood alternative with its numerical nugget; it is not the
likelihood integrated over random-feature draws.

## Lung Xenium cohort

Obtain access to HEST-1k and accept its terms before downloading:

```bash
python release/ligature/external/download_hest_lung.py
python release/ligature/external/build_hest_lung.py
python release/ligature/external/features_hest_lung.py
export SE_MARKERS=EPCAM,SFTPC,AGER,SCGB1A1,KRT5,PTPRC,CD3E,CD8A,MS4A1,CD68,MARCO,COL1A1,ACTA2,PECAM1,MKI67,SPP1
```

Then fit the reference stain/generator commands with
`--root "$LIGATURE_DATA_ROOT/cohorts/lung"` and separate output directories under
`"$LIGATURE_DATA_ROOT/runs/lung"`. Keep the marker variable set for the lung
analysis and unset it before returning to ORION. The independently calibrated
lung protocol is in `research/calibration/extensions/`.

## Analytic bilinear means

`src/analytic_moments.py::expected_statistics` integrates the fitted latent and
random-feature second moments for all 832 scored coordinates. Evaluations also
return `expected_scored`. Nonlinear queries and predictive laws use joint draws;
interval calibration remains query-specific. Fields are independent across
regions, while patient latents couple their draws. Finite Student-t second
moments require degrees of freedom greater than two.

For a data-free execution example, run `python examples/synthetic.py` from the
repository root. Data/model terms are described in the root third-party notices.
