# Non-touching-mask sensitivity

This fixed-model control excludes near cell pairs whose supplied masks have
Chebyshev distance at most two pixels. It compares the restricted statistic
against its own measured target and plug-in on the same retained regions. The
rule, estimand and uncertainty calculation are recorded in `plan.json`.

After regenerating the reference five-fold stain and generator:

```bash
python research/segmentation_control/control.py \
  --root "$LIGATURE_DATA_ROOT" \
  --out "$LIGATURE_DATA_ROOT/runs/segmentation_control"
```

The output directory must differ from the reference generator directory. Cell,
region and patient records are created locally. This is a limited mask-distance
sensitivity, not optical-spillover correction; its reported accuracy interval
includes no improvement. The submitted aggregate values are in the appendix.
