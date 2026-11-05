# Independent patient-level calibration protocol

## Question

Can separately calibrated, patient-adaptive Ligature intervals provide useful
clustering-statistic uncertainty compared with equally calibrated plug-in and
rescaled-mean baselines?

This addresses validity and interval efficiency. It is not an attempt to tune
test coverage to 90%, improve the published score through favorable selection,
or claim a new conformal algorithm.

## Population, roles, and fitting

- ORION's existing five outer test rotations are retained (eight test patients
  each). All 40 patients are evaluated once in the aggregate.
- Per rotation, the original eight validation patients and two seed-selected
  original training patients become the ten calibration patients. Two different
  original training patients become model-selection/variance-fitting patients.
  The remaining twenty patients train the models.
- Split choices use patient IDs and a fixed seed, not molecular outcomes.
- Both the mean stain and the residual generator are retrained. Reusing the
  original mean checkpoint would leak calibration labels through its early
  stopping and is forbidden.
- PCA, feature/target normalization, and statistic scales use only the twenty
  training patients. Early stopping, seven variance-component scales, and
  deterministic baseline corrections use only the two selection patients.
- Calibration labels enter only the final split-conformal score quantiles.
- All patients remain on the original log-intensity target scale. No copula or
  altered target is used in this initial experiment.
- Default architecture, optimizer, field scales, losses, and model seed match
  the current manuscript. Mean stain: 25 epochs, early stopping. Generator: 15
  epochs, early stopping; variance-component fitting: four epochs. Evaluation:
  64 joint draws, preserving the shared patient latent.

## Endpoints and inference

Primary family: sixteen same-marker near-minus-far clustering statistics.
Secondary families: 120 cross-marker statistics and the combined 136.
Statistics average eligible regions equally within each patient, as in the
paper. Report all families; do not select markers after inspecting outcomes.

For each statistic separately, use split-conformal scores
`abs(truth - center) / scale`, with strictly positive image-dependent scale
fixed before calibration. The threshold is the order statistic
`ceil((n_cal + 1) * (1 - alpha))`; if this exceeds `n_cal`, return an infinite
interval rather than an interpolated finite substitute. With ten calibration
patients and alpha=0.10, the tenth score is used. Under exchangeability and
independent fitting this gives marginal coverage for each prespecified
statistic; it is not simultaneous or patient-conditional coverage.

Source: Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511.

This is a new development experiment on previously inspected patients. The
conformal algorithm's assumptions and its empirical coverage are distinct from
an untouched confirmatory evaluation. Pooled cross-fit bootstrap summaries
condition on the fitted models and do not capture all training-set dependence.

## Prespecified methods

1. Raw Ligature empirical 5th–95th percentile interval.
2. Raw Gaussian plug-in interval, spread fitted on selection patients.
3. Raw Gaussian global-rescale interval, correction and spread fitted on selection.
4. Plug-in center with constant-scale split conformal.
5. Global-rescale center with constant-scale split conformal.
6. Two-family rescale center (clustering vs cross-marker) with constant-scale split conformal.
7. Ligature sample-mean center with sample-standard-deviation split conformal.
8. Global-rescale center with Ligature's sample standard deviation and split conformal.
9. Ligature sample-mean center with constant-scale split conformal.

Methods 8 and 9 distinguish the benefit of an adaptive scale from a corrected
center. All conformal methods use identical calibration patients and ranks.
Mean corrections are through-origin least-squares slopes fitted on the two
selection patients' statistic vectors; no test-informed regularization/tuning.

## Reporting and decision criteria

Report patient and per-statistic coverage, width, interval score, point error,
per-fold behavior, and matched paired width/error comparisons. Width and
interval score may also be normalized by training-patient statistic SD, with a
fixed numerical floor, to make heterogeneous statistics comparable.

Utility is assessed by clustering coverage, interval widths and scores against
equally calibrated plug-in baselines. Finite-sample marginal validity alone does
not establish practical efficiency. All outcomes are retained; no architecture
or split is selected after viewing test outcomes.

Run numerical checks before all five rotations. Preserve completed mean and generator checkpoints between stages.
