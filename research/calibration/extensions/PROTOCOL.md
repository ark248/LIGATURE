# Lung and unscored-query calibration protocol

Date: 29 October 2026. The completed ORION calibration fits remain fixed.

## A. Lung, 19 independent patients

Keep the original five outer test rotations (4/4/4/4/3 patients). Use four
independent calibration patients per rotation, starting with the original
validation patients and adding one seed-selected original training patient only
when there are three validation patients. Reserve one different original
training patient for model selection and variance fitting. This leaves ten
training patients (eleven when there are three test patients). Role choices use
IDs only, seed `271027 + 1701 + fold`.

Retrain BOTH stain and generator. PCA, intensity normalization, and statistic
scales use only training patients. Early stopping and four epochs of variance
fitting use only the single selection patient. Architecture, optimizer, model
seed, 25/15 mean/generator epochs, 16 training draws and 64 evaluation draws match
the completed ORION calibration design. The single selection patient's limits
must be stated.

Primary nominal coverage is **80%**, not 90%: with four independent calibration
patients the finite-sample rank is four. A 90% threshold is infinite; moving
nine patients into calibration would substantially reduce an already small
fitting cohort. Do not manufacture finite 90% intervals by interpolation or by
pooling genes. Report the 90% infeasibility explicitly.

Primary endpoints: sixteen same-gene clustering statistics; secondary: all136
and cross120. Retain the original near-minus-far definition and equal eligible
region weights. Compare raw Ligature and Gaussian intervals at the same 80%
level, and the six matched conformal center/scale controls from the completed
ORION experiment. Every method uses the same four calibration patients. Report
coverage, width, normalized interval score, matched point error, and fold/patient
variation. Pooled uncertainty summaries are descriptive given overlapping fits.

Compute the mean and generator/evaluation phases separately, preserving completed checkpoints. Run a small-data execution check before full fits.

## B. Previously unscored ORION nonlinear statistics

Reuse the FIVE completed independent 20/2/10/8 model fits from `orion_v1`. No
model is refitted or changed. Re-evaluate joint cell draws, sharing each draw's
patient latent across regions, on calibration and test patients.

Two prespecified families, both absent from every fitting loss:

1. All 256 ordered marker contact fractions at 20 microns. Gates are the 85th
   percentiles of training-patient measured signals. Patient fractions pool
   numerators and denominators across regions, matching the paper. A patient is
   eligible for a query when it has at least 20 measured source-positive cells.
   Eligibility is applied identically to calibration and test patients. Its
   outcome-defined scope and availability only after measurement must be stated;
   it is not an all-patient deployment guarantee. Report effective calibration
   counts and any insufficient-rank intervals, never treating marker pairs as
   additional independent patients.
2. Ten cellular-neighborhood frequencies, with the paper's fixed marker-type
   priority, ten nearest cells including self, and k-means fitted only to
   training-patient measured compositions (up to 100,000 cells, four starts,
   seed zero). Centroids are frozen before calibration/test evaluation. Query
   indices are local to each fold. Primary coverage/width/score and per-patient
   distribution errors are invariant to cluster permutations; do not pool
   arbitrary cluster labels as if they were one globally fixed biological type.

Use 64 joint draws per held-out patient. Quantile-matched thresholds for the
plug-in and Ligature-Q are fitted only on the TWO selection patients, not the
calibration patients. All selection regions contribute plug-in predictions;
Ligature thresholds use eight samples on at most 120 selection regions chosen
using seed `271027 + 2701 + fold`. No threshold or centroid sees a calibration
label. The measured target gates always remain training-derived.

Primary method: Ligature-Q center and sample SD with per-statistic **90%** split
conformal. Controls: raw Ligature-Q, raw unmatched Ligature, uncorrected plug-in
and quantile-matched plug-in with constant-scale conformal; unmatched Ligature
adaptive conformal; Ligature-Q constant-scale conformal; quantile-matched plug-in
center with Ligature-Q scale. All use identical eligible calibration patients.

The conformity rank is `ceil((n_q+1)*(1-alpha))`, or infinity when unattainable.
For fractions, intersect every interval with the known [0,1] support. Keep a
separate indicator of finite pre-intersection thresholds and full-support
intervals; [0,1] resulting from an infinite threshold is valid but uninformative.
Report how often this happens. Report raw fractional width and interval score
as primary efficiency metrics, separately for contacts and neighborhoods;
training-SD normalization is optional descriptive material, not a selection rule.

## Interpretation and checks

Both extensions are development evidence on previously inspected cohorts.
Standard conformal guarantees require independent fitting and exchangeability
in the stated patient population. No simultaneous, conditional-patient,
cell-level joint-law, phenotype-validation, or association-test guarantee is
implied. Do not select gates, markers, nominal levels, model variants, or splits
after looking at test results. Preserve all prespecified methods and families.

Verify role disjointness, original test-set retention, source/input hashes,
test-label isolation, calibration-only threshold dependence, raw target
reconstruction, finite-sample ranks including insufficient-query cases, and
reported interval arithmetic. Keep eligibility and local neighborhood IDs in
the output records. Independent audit precedes interpretation and handoff.
