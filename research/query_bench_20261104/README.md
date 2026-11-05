# Matched forecasts and additional spatial queries

This benchmark compares LIGATURE with a matched direct graph, cheap ridge
readouts, plug-in corrections and dependence controls. It evaluates independently
calibrated forecasts, Moran's I, Geary's C, original-log-intensity contrasts and
H&E-derived nuclear-plus-rim proxies. The fixed seeds, endpoints, patient roles
and multiplicity adjustment are in [PROTOCOL.md](PROTOCOL.md).

## Local reproduction

Prepare ORION data/features and all five independent calibration fits using
[these instructions](../calibration/README.md). Set `LIGATURE_DATA_ROOT` to the
same workspace; it defaults to `release/ligature` in this repository.

```bash
python research/query_bench_20261104/run_local.py --stage fixed --threads 8
```

This runs preprocessing, cheap readouts, all 45 fixed fit/reuse units, frozen
moment controls, complete three-seed evaluation and numerical audits. Completed
worker units are restartable. All generated outputs stay in
`$LIGATURE_DATA_ROOT/runs/spatial_queries`.

Individual entry points remain available: `prepare.py`, `cheap_readouts.py`,
`worker.py --fold N --seed N --kind {dscore,dquery,ligature}`, `reuse.py` and
`evaluate.py`. Evaluation refuses missing folds, seeds or controls. The three
seeds are 271027, 272027 and 273027; model-selection rules are fixed.

## H&E-derived geometry control

After the fixed benchmark is complete:

```bash
python research/query_bench_20261104/run_local.py --stage he
```

The control downloads the hash-verified official InstanSeg brightfield model,
constructs H&E-derived nuclear-plus-rim proxies, extracts Phikon-v2 features,
refits five stain/generator/direct comparisons and evaluates them independently.
Feature extraction uses CUDA when available and otherwise CPU. See
[HE_CONTROL_PLAN.md](HE_CONTROL_PLAN.md) for region selection and target definitions.
Changing both geometry and measurement targets precludes comparing absolute
errors with whole-cell targets. This control does not correct optical spillover.

## Results and figures

`results/` contains the aggregate JSON/CSV summaries and compact audit metadata
used by the submitted tables and figures. These retain negative comparisons.
Patient-level records and checkpoints are generated locally; they are not bundled.
`audit_results.py`, `postprocess.py` and `audit_he.py` operate on those local records.

The final exploratory simple baselines are implemented in
[../simple_baselines](../simple_baselines/README.md). They were added after test
inspection and remain distinct from the fixed protocol.

To regenerate the final plots and tables from the included aggregates:

```bash
python research/simple_baselines/render_numbers.py
python research/query_bench_20261104/render_tables.py
python research/query_bench_20261104/render_figures.py
```

To use a fresh evaluation instead of the supplied aggregates, copy its
`summary.json`, `summary.csv` and `audit.json` from the local evaluation directory
into `results/`. For the H&E control, copy the corresponding files with the
`he_` prefix. Keep generated patient files in the local workspace.

The main comparisons average metrics across seeds rather than ensembles of
predictions. Patient-bootstrap intervals condition on fitted models and do not
include all dependence between cross-validation fits. These development-cohort
checks do not constitute a new untouched validation cohort.
