# Bounded H&E-derived geometry control — fixed before its targets or results are inspected

Use a single fixed seed (271027), existing 20/2/10/8 patient roles and all 40 ORION
patients. This is an exploratory input/target-definition sensitivity, not an
untouched validation cohort, segmentation accuracy benchmark or whole-slide test.

Select the first 20 non-overlapping supplied crops per patient in ascending
(x_start, y_start, source_index) order, retaining all if fewer are available.
Selection uses only source coordinates and the existing assay-selected crop
release; it does not use molecular values, assay masks or prediction performance.

Segment H&E using the official InstanSeg brightfield_nuclei v0.1.1 TorchScript
model, SHA256 0c724da169d507d8dd2e206ad3675a0b1566824eee6e20ad9a0ba916311d9dd0,
with default thresholds and the documented 0.325-to-0.5 micron resizing and channel
percentile normalization. No test-driven threshold or segmentation model selection.
Expand nuclear labels by six pixels (1.95 microns) by nearest-label assignment to
define cell proxies; exclude nuclei below 20 pixels, proxies below 40 pixels,
centroids within 48 pixels of image edges, or zero perimeter. Positions and all
shape descriptors come exclusively from these H&E-derived labels.

Re-extract pinned Phikon-v2 features at the new centroids, using the existing
64-pixel H&E field of view and normalization. Do not reuse features centered on
assay-provided cells. Extract targets as mean paired immunofluorescence within
these H&E-derived cell proxies. Neither supplied nucleus masks nor whole-cell
masks enter the control. Molecular images are used as training/evaluation targets.
The target region and cell definitions therefore differ from the primary study;
no direct numerical comparison of MSE across those different targets is claimed.

Refit the stain (25 epochs), LIGATURE (15 epochs, existing scores/settings) and
matched stain-input direct graph (15 epochs) separately in each of five rotations,
without changing hyperparameters based on this control. Report clustering and all
136 near-minus-far contrasts against that control's own plug-in, all 40 patients,
and equally calibrated interval efficiency/coverage. Single-seed inference only;
patient-bootstrap intervals are descriptive and conditional on these fits.

The control is evaluated over all five folds with its single-seed and
changed-target scope retained in the aggregate report.
