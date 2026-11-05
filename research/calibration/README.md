# Independent patient-level calibration

Five rotations assign 20 training, 2 selection, 10 calibration and 8 test patients.
The stain and generator are refitted; calibration labels enter only the final
prediction-interval construction. Every ORION patient is tested once. The
[protocol](PROTOCOL.md) records the estimands and interval definitions.

First prepare ORION cell tables and embeddings using the core instructions.
From the repository root, using the same `LIGATURE_DATA_ROOT`:

```bash
python research/calibration/prepare.py
for fold in 0 1 2 3 4; do
  python research/calibration/fit.py --fold "$fold" --threads 8
done
python research/calibration/evaluate.py
python research/calibration/audit_results.py
python research/calibration/audit_targets.py
```

Preparation derives the fixed roles from the cell-table case IDs and the original
split rule, then snapshots the one public model tree into the local output
workspace. Outputs are in `runs/calibration/orion_v1` below the data workspace.
The source snapshots and fitted checkpoints are generated artifacts.

`python research/calibration/test_numerics.py` checks the finite-sample conformal
rank without patient data. `fit.py --smoke --fold 0` performs a reduced-data
execution check after preprocessing; its output is separate from the full fit.
`test_isolation.py` then checks the smoke fit's role separation.

## Lung and unscored-query extensions

Prepare the lung data/embeddings as well as the ORION fits above. Then:

```bash
python research/calibration/extensions/prepare_extensions.py
for fold in 0 1 2 3 4; do
  python research/calibration/extensions/fit_lung.py --fold "$fold" --threads 8
  python research/calibration/extensions/eval_orion_unscored.py --fold "$fold" --threads 8
done
python research/calibration/extensions/evaluate_lung.py
python research/calibration/extensions/evaluate_unscored.py
```

The [extension protocol](extensions/PROTOCOL.md) defines the lung 80% calibration
target, gates, query eligibility and neighborhood construction. The extension
auditors replay intervals and targets from locally generated outputs. These are
separate study protocols, with separate target definitions and calibration sizes.
