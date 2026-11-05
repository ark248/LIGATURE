# Exploratory simple baselines

These comparisons were added after the fixed benchmark's test results were
examined. They are reported separately as exploratory checks in the manuscript.
They include a no-image training mean, additive per-statistic correction and a
ridge readout from frozen stain/image context. All fitting uses the stated
training/selection patients, with separate conformal calibration patients.

`results.json` is an aggregate report; it contains no patient prediction table.
Regenerate the numerical macros without cohort data:

```bash
python research/simple_baselines/render_numbers.py
```

For a complete numerical replay, first regenerate the reference ORION and lung
outputs and the matched three-seed benchmark. Its `postprocess.py` produces the
local patient-record Parquet required for the comparisons. Then run:

```bash
python research/simple_baselines/simple_baselines.py
python research/simple_baselines/render_numbers.py
python research/query_bench_20261104/render_tables.py
```

The script checks the reproduced plug-in/reference rows and cheap 48-output
ridge before computing the additional comparisons. It writes aggregate results
beside the script. Keep each regenerated run separate from the supplied summaries
when comparing numerical reproducibility.
