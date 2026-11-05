# Spatial-query and calibration benchmark protocol

Date: 4 November 2026.
This is development-cohort evidence, not a new confirmatory cohort or preregistration.

## Scientific questions and endpoints

1. Does LIGATURE provide better calibrated spatial forecasts than the matched direct
   graph predictor when both use identical patient-disjoint fitting/calibration roles?
   Primary endpoints: clustering interval score and width at 90% marginal coverage;
   always report coverage alongside efficiency and constant-scale controls.
2. Can frozen molecular samples answer standard ratio-based spatial statistics not
   scored in training? Primary query families: global Moran's I and Geary's C on the
   same symmetric, binary 0–25 micrometer adjacency, all 16 ORION markers.
   Both are averaged equally across eligible regions of each patient. They are
   related spatial-autocorrelation metrics, not independent biological replications.
3. Are conclusions sensitive to fold-specific centering? Secondary diagnostic:
   same-marker near-minus-far products on the original log1p assay scale, obtained
   by inverting each fold's training-only standardization before computing queries.

Moran's I = n/W * z'Az / sum(z^2), z=Y-mean(Y).
Geary's C = (n-1)/(2W) * sum A_ij(Y_i-Y_j)^2 / sum(z^2).
The graph is not row-normalized. Constant-marker regions are assigned zero by
explicit operational convention and counted; no patient is selected by outcome.
Existing eligible-region rules are preserved. No marker/gate/radius is optimized.

## Roles, model selection and comparators

Reuse the existing five 20-training / 2-selection / 10-calibration / 8-test patient
rotations and their trained stains. Training-only PCA, target scaling and features
must reproduce the original independent fits. No calibration/test labels in fitting.
Early stopping, ridge strength and any variance rescaling use selection patients only.

Comparators for all scored statistics: frozen plug-in, global/family rescaling,
LIGATURE, matched stain-input direct graph. All receive the same conformal procedure;
compare raw, adaptive-scale and constant-scale forecasts.

Additional-query comparators: plug-in; independent-cell Gaussian sampling using
LIGATURE's marginal spreads (dependence control); selection-fitted affine correction;
cheap query-specific ridge readout from frozen image/stain region features; and the
same stain-input graph trained end to end on the new queries. The ridge uses a fixed
log-spaced penalty grid, chosen by patient-averaged normalized selection error.
Use all available paired training assays: their new query targets cost computation,
not new molecular labeling. Do not claim fewer assays merely because a query was
absent from the generative objective. Report costs of target computation, fitting,
one-time draw generation, and subsequent query evaluation separately.

The existing independent LIGATURE seed is reused. Two additional generator seeds and
three direct-model seeds are fixed in advance (271027, 272027, 273027). Fit 15 epochs
with the existing width/depth/CRPS/AdamW/batching settings; reuse fixed stains. Do not
select seeds from test scores. If any work units are incomplete, label
incomplete comparisons rather than report favorable completed seeds selectively.

## Reporting and stopping

Report every specified endpoint and comparator, including losses, with paired
patient-bootstrap 95% intervals (2000 replicates) conditional on fitted models.
Average metrics over seeds, not predictions, for the primary multi-seed summary.
Two new spatial-autocorrelation families are the primary query tests; use Holm
adjustment for their paired tests. Per-marker and secondary comparisons are exploratory.

No architecture or hyperparameter is selected using test performance. Comparisons
apply to the specified targets, population and evaluation protocol; uncertainty
and all comparator outcomes are retained.

Outputs are written to `runs/spatial_queries` in the data workspace. Fixed seeds, query families and selection rules are retained.

## Segmentation question

First make the supplied-segmentation deployment assumption explicit in the abstract,
method and schematic. Independently assess feasibility of a bounded H&E-derived
nuclei/rim control using a public pretrained brightfield model. Fix region selection
by source coordinates before evaluating outcomes. Assay masks must not enter model
inputs in an H&E-only control. A partial geometry perturbation is not such validation.
Proceed only after a timing/availability pilot; otherwise report the limitation.

### Frozen-moment reuse controls (added before any new test-performance inspection)

A rich direct statistic predictor can itself be reused algebraically. Include its
near-bin products, predicted regional means/variances and selection-cell residual
variance in a moment plug-in for Moran/Geary; use its directly supervised contrast
for the raw-log contrast. Covariance of a regional mean with a cell is approximated
as constant. Project variance-of-the-mean to [0, cell residual variance] and clamp
ratio predictions to conservative graph-theoretic bounds. No new query labels or
query-specific fitting is used. Also evaluate ratio-of-expected-moments from
LIGATURE's analytically integrated covariance, isolating the value of sampling
versus moment substitution. Both remain approximations to expectations of ratios.
These strengthen the frozen-query comparison; the prespecified full direct-query
head remains the primary paired comparator. They do not replace any comparator.

Before test-performance inspection, also fix a conservative raw-law control for
the direct model: one Gaussian spread multiplier per reported family, fitted by
normalized CRPS on the two selection patients only, bounded to [0.1,10]. Report
raw CRPS and this selection-calibrated CRPS separately. Positive coordinatewise
scale factors cancel from the subsequent split-conformal intervals, so this does
not alter the primary interval comparison. Runtime includes both cached and
recomputed frozen-context ridge readouts; caching context during the common stain
forward pass must not be charged as an extra encoder invocation.

Analytic bilinear means are evaluated as a numerical refinement alongside the
original 64-draw point means, never selected between using test error. Integrate
all latent/RFF second moments to obtain the conditional mean of each scored
bilinear statistic exactly under the fitted model. This removes Monte Carlo
point-estimation noise without retraining or changing the law. Keep sampled
predictive distributions and the original prespecified comparisons; nonlinear
Moran/Geary ratios still require sampling or the labeled moment-ratio approximation.

Exact generator-marginal independence control: preserve the full within-cell marker
law of the finite-RFF sampler, including its Gaussian scale mixture, while drawing
each cell independently. This supplements the moment-matched independent Gaussian;
it isolates dependence without a marginal-shape approximation. Uniform RFF phases
make each cell's field marginal a known scalar-variance Gaussian mixture, shared
across that field's rank coordinates. All region/patient latent marginals are also
preserved within cells but independent between cells. Fixed 64 draws, seed + 777;
no fitting or test-performance-based choice. Original primary tests are unchanged.
