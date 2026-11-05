"""Exploratory clinical sensitivities for frozen ORION forecasts.

Post hoc and exploratory. No model is fitted to images and no outcome enters any generator.
This grid was fixed before any result below was computed, but after the fraction and burden
concordances in biological_case.json had been seen, so it is not a preregistration.

Analyses:
  1. Measured feature minus direct graph (stored under the historical key "oracle").
     Exact reconstruction reproduces this feature's observed association; it is not a general
     upper bound on clinical information or proof that superiority is impossible.
  2. Full grid: LIGATURE or LIGATURE-Q minus direct, over three features, four pair scopes and
     two run aggregations (48 contrasts), with unadjusted and family-wise (max-|t|) intervals.
     No contrast is selected. Fold-rank scope adds the 444 between-fold pairs that the within-fold
     scope discards, and removes between-fold scale differences by ranking within each fold.
  3. Planning illustration: approximate detectable differences and event counts, assuming
     SE scales as 1/sqrt(events). This is not a formal sample-size calculation or a diagnosis
     that low power caused the inconclusive comparison.
  4. Nested Cox: does either predictor add to the other given stage? Ensembled predictions,
     z-scored within fold; likelihood-ratio tests with Holm adjustment.
  5. Assay recovery: paired MSE ratio and Spearman difference on all 40 patients (no outcomes).
  6. A further post hoc endpoint in the same patients: pathologist TIL grade.
     Primary: engaged burden, within-fold concordance, LIGATURE-Q minus direct. Direction fixed
     (higher burden with higher grade). Other features and the pooled scope are secondary.

Decision rule for survival, adding multiplicity to the earlier fixed rule: superiority is
established only if the measured primary interval excludes 0.5 in the expected direction and the
LIGATURE-Q minus direct primary interval (within-fold burden) excludes zero in the favorable
direction after family-wise adjustment. A single unadjusted grid cell does not meet it.
Failure of this exploratory rule does not demonstrate equivalence or impossibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import chi2, rankdata

from biological_case import FEATURES, concordance, interval, read_clinical, read_forecasts

SCOPES = ("within_fold", "pooled", "pooled_fold_ranks", "within_fold_and_stage")
AGGREGATES = ("mean_of_metrics", "ensemble_prediction")
GRADES = dict(Absent=0, Mild=1, Moderate=2, Marked=3)
Z80 = 1.959964 + 0.841621  # Two-sided 5% test at 80% power.


class Panel:
    def __init__(self, time, event, fold, stage, weights):
        self.time, self.event, self.fold, self.weights = time, event, fold, weights
        same_fold = fold[:, None] == fold[None, :]
        everyone = np.ones_like(same_fold)
        self.masks = dict(within_fold=same_fold, pooled=everyone, pooled_fold_ranks=everyone,
                          within_fold_and_stage=same_fold & (stage[:, None] == stage[None, :]))


def fold_ranks(risk, fold):
    out = np.empty_like(risk, dtype=float)
    for f in set(fold):
        ids = fold == f
        out[ids] = rankdata(risk[ids]) / ids.sum()
    return out


def score(values, sign, scope, aggregate, panel):
    """Return the point concordance, bootstrap replicates and comparable pair count."""
    rows = values.mean(0, keepdims=True) if aggregate == "ensemble_prediction" else values
    points, boots = [], []
    for row in rows:
        risk = sign * row
        if scope == "pooled_fold_ranks":
            risk = fold_ranks(risk, panel.fold)
        mask = panel.masks[scope]
        point, pairs = concordance(risk, panel.time, panel.event, mask, np.ones((1, len(risk))))
        points.append(point[0])
        boots.append(concordance(risk, panel.time, panel.event, mask, panel.weights)[0])
    return float(np.mean(points)), np.nanmean(boots, axis=0), pairs


def contrast(a, b, sign, scope, aggregate, panel):
    pa, ba, pairs = score(a, sign, scope, aggregate, panel)
    pb, bb, _ = score(b, sign, scope, aggregate, panel)
    d = ba - bb
    return dict(a=pa, b=pb, difference=pa - pb, se=float(np.nanstd(d)), ci95=interval(d),
                comparable_pairs=pairs), d


def cox_loglik(X, time, event):
    """Cox partial log likelihood at the maximum; survival times must be distinct."""
    assert len(set(time)) == len(time), "Tied survival times need a tie correction"
    order = np.argsort(-time)
    X, event = X[order], event[order]

    def evaluate(beta):
        eta = X @ beta
        w = np.exp(eta)
        s0 = np.cumsum(w)
        s1 = np.cumsum(w[:, None] * X, 0)
        s2 = np.cumsum(w[:, None, None] * X[:, :, None] * X[:, None, :], 0)
        ll = np.sum(event * (eta - np.log(s0)))
        grad = np.sum(event[:, None] * (X - s1 / s0[:, None]), 0)
        hess = -np.sum(event[:, None, None] * (s2 / s0[:, None, None]
                       - s1[:, :, None] * s1[:, None, :] / s0[:, None, None] ** 2), 0)
        return ll, grad, hess

    beta = np.zeros(X.shape[1])
    ll, grad, hess = evaluate(beta)
    for _ in range(100):
        step = np.linalg.solve(hess, grad)
        scale = 1.0
        while True:
            new_ll, new_grad, new_hess = evaluate(beta - scale * step)
            if new_ll >= ll - 1e-10 or scale < 1e-6:
                break
            scale /= 2
        beta = beta - scale * step
        converged = abs(new_ll - ll) < 1e-10
        ll, grad, hess = new_ll, new_grad, new_hess
        if converged:
            break
    return float(ll), beta


def fold_z(values, fold):
    out = np.empty_like(values, dtype=float)
    for f in set(fold):
        ids = fold == f
        out[ids] = (values[ids] - values[ids].mean()) / values[ids].std()
    return out


def nested_cox(predictions, ix, labels, fold):
    time, event = labels.PFSDays.to_numpy(), labels.Recurrence.to_numpy()
    stage = (labels.stage.to_numpy() - labels.stage.mean()).astype(float)
    rows = []
    for j, feature in enumerate(FEATURES):
        z = {m: fold_z(predictions[m][:, ix, j].mean(0), fold) for m in ("measured", "ligature_q", "direct")}
        for added, kept in (("ligature_q", "direct"), ("direct", "ligature_q"), ("measured", "direct")):
            full, beta = cox_loglik(np.c_[z[added], z[kept], stage], time, event)
            reduced, _ = cox_loglik(np.c_[z[kept], stage], time, event)
            lr = max(2 * (full - reduced), 0.0)
            # Higher features are favorable, so a real gain has a negative log hazard ratio.
            rows.append(dict(feature=feature, added=added, given=kept + "+stage",
                             log_hr_per_fold_sd=float(beta[0]), lr=lr, p=float(chi2.sf(lr, 1))))
    ranked = np.argsort([r["p"] for r in rows])
    running = 0.0
    for position, k in enumerate(ranked):
        running = max(running, min(1.0, (len(rows) - position) * rows[k]["p"]))
        rows[k]["p_holm"] = running
    return rows


def assay_recovery(cases, predictions):
    picks = np.random.default_rng(20261105).integers(0, len(cases), (2000, len(cases)))

    def spearman(values, truth):
        x, y = rankdata(values[picks], axis=1), rankdata(truth[picks], axis=1)
        x, y = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
        denom = np.sqrt((x ** 2).sum(1) * (y ** 2).sum(1))
        return np.divide((x * y).sum(1), denom, out=np.full(len(x), np.nan), where=denom > 0)

    rows = []
    for j, feature in enumerate(FEATURES):
        truth = predictions["measured"][0, :, j]
        lig, direct = predictions["ligature_q"][:, :, j], predictions["direct"][:, :, j]
        a, b = ((lig - truth) ** 2).mean(0), ((direct - truth) ** 2).mean(0)
        boot_rho = (np.nanmean([spearman(v, truth) for v in lig], axis=0)
                    - np.nanmean([spearman(v, truth) for v in direct], axis=0))
        point_rho = (np.mean([np.corrcoef(rankdata(v), rankdata(truth))[0, 1] for v in lig])
                     - np.mean([np.corrcoef(rankdata(v), rankdata(truth))[0, 1] for v in direct]))
        rows.append(dict(feature=feature, mse_ratio=float(a.mean() / b.mean()),
                         mse_ratio_ci95=interval(a[picks].mean(1) / b[picks].mean(1)),
                         spearman_difference=float(point_rho), spearman_difference_ci95=interval(boot_rho)))
    return rows


def analyze(cases, meta, predictions, labels):
    clinical_cases = sorted(set(cases) & set(labels.index))
    labels = labels.loc[clinical_cases]
    ix = [cases.index(c) for c in clinical_cases]
    time, event = labels.PFSDays.to_numpy(), labels.Recurrence.to_numpy()
    fold = np.asarray([meta[c][0] for c in clinical_cases])
    stage = labels.stage.to_numpy()
    n = len(labels)
    rng = np.random.default_rng(2026110502)  # Same draws as biological_case.py.
    weights = np.zeros((2000, n))
    for f in sorted(set(fold)):
        ids = np.flatnonzero(fold == f)
        weights += np.eye(n)[rng.choice(ids, (2000, len(ids)), replace=True)].sum(1)
    panel = Panel(time, event, fold, stage, weights)
    values = {name: v[:, ix] for name, v in predictions.items()}
    out = dict(patients=n, events=int(event.sum()), replicates=len(weights),
               scope="Post hoc, fixed-forecast sensitivities in a development cohort. "
               "The historical oracle key denotes measured-feature comparisons, not a general clinical bound. "
               "Inconclusive comparisons do not establish equivalence or impossibility. "
               "Event-count estimates assume inverse-square-root event scaling and are planning illustrations; "
               "Cox tests describe in-sample associations, not out-of-sample prediction or clinical utility.")

    # 1. Oracle.
    out["oracle"] = []
    for j, feature in enumerate(FEATURES):
        for scope in SCOPES:
            for aggregate in AGGREGATES:
                row, _ = contrast(values["measured"][..., j], values["direct"][..., j], -1, scope, aggregate, panel)
                out["oracle"].append(dict(feature=feature, scope=scope, aggregate=aggregate, **row))

    # 2. Full grid with family-wise intervals.
    grid, boots = [], []
    for method in ("ligature_q", "ligature"):
        for j, feature in enumerate(FEATURES):
            for scope in SCOPES:
                for aggregate in AGGREGATES:
                    row, d = contrast(values[method][..., j], values["direct"][..., j], -1, scope, aggregate, panel)
                    grid.append(dict(method=method, feature=feature, scope=scope, aggregate=aggregate, **row))
                    boots.append(d)
    boots = np.asarray(boots)
    diff = np.asarray([r["difference"] for r in grid])
    se = np.asarray([r["se"] for r in grid])
    critical = float(np.quantile(np.nanmax(np.abs(boots - diff[:, None]) / se[:, None], axis=0), .95))
    for r, d, s in zip(grid, diff, se):
        r["ci95_familywise"] = [float(d - critical * s), float(d + critical * s)]
    out["grid"] = grid
    out["grid_summary"] = dict(
        contrasts=len(grid), familywise_critical_t=critical,
        favor_ligature_point=int((diff > 0).sum()), favor_direct_point=int((diff < 0).sum()),
        unadjusted_favor_ligature=int(sum(r["ci95"][0] > 0 for r in grid)),
        unadjusted_favor_direct=int(sum(r["ci95"][1] < 0 for r in grid)),
        familywise_favor_ligature=int(sum(r["ci95_familywise"][0] > 0 for r in grid)),
        familywise_favor_direct=int(sum(r["ci95_familywise"][1] < 0 for r in grid)))

    # 3. Power, by scope, for the primary feature.
    out["power"] = []
    for scope in ("within_fold", "pooled", "pooled_fold_ranks"):
        g = next(r for r in grid if (r["method"], r["feature"], r["scope"], r["aggregate"]) ==
                 ("ligature_q", "engaged_burden", scope, "mean_of_metrics"))
        o = next(r for r in out["oracle"] if (r["feature"], r["scope"], r["aggregate"]) ==
                 ("engaged_burden", scope, "mean_of_metrics"))
        events = {f"{delta:.2f}": int(np.ceil(out["events"] * (Z80 * g["se"] / delta) ** 2)) for delta in (.02, .03, .05, .10)}
        out["power"].append(dict(scope=scope, se_difference=g["se"], minimum_detectable_difference=Z80 * g["se"],
                                 observed_difference=g["difference"], oracle_gap=o["difference"],
                                 oracle_ci95=o["ci95"], events_needed_at_difference=events))

    # 4. Nested Cox.
    out["nested_cox"] = nested_cox(predictions, ix, labels, fold)

    # 5. Assay recovery, all 40 patients and no outcomes.
    out["assay_recovery"] = assay_recovery(cases, predictions)

    # 6. Pathologist TIL grade. A higher grade counts as the "earlier event" in the concordance helper.
    grade = labels.TIL.map(GRADES).to_numpy(float)
    assert np.isfinite(grade).all(), "Unmapped TIL grade"
    til_panel = Panel(-grade, np.ones(n), fold, stage, weights)
    out["til"] = dict(n=n, levels={k: int((grade == v).sum()) for k, v in GRADES.items()}, rows=[], contrasts=[])
    for j, feature in enumerate(FEATURES):
        for scope in ("within_fold", "pooled"):
            for name in ("measured", "stain_q", "ligature", "ligature_q", "direct"):
                point, boot, pairs = score(values[name][..., j], 1, scope, "mean_of_metrics", til_panel)
                out["til"]["rows"].append(dict(feature=feature, scope=scope, method=name, concordance=point,
                                               ci95=interval(boot), comparable_pairs=pairs))
            for a, b in (("ligature_q", "direct"), ("measured", "direct"), ("ligature_q", "stain_q")):
                row, _ = contrast(values[a][..., j], values[b][..., j], 1, scope, "mean_of_metrics", til_panel)
                out["til"]["contrasts"].append(dict(feature=feature, scope=scope, contrast=f"{a} minus {b}", **row))

    # Decision rule, applied to the primary contrast only.
    primary = next(r for r in grid if (r["method"], r["feature"], r["scope"], r["aggregate"]) ==
                   ("ligature_q", "engaged_burden", "within_fold", "mean_of_metrics"))
    measured_point, measured_boot, _ = score(values["measured"][..., 2], -1, "within_fold", "mean_of_metrics", panel)
    measured_ci = interval(measured_boot)
    gates = dict(measured_excludes_half=measured_ci[0] > .5,
                 ligature_beats_direct_familywise=primary["ci95_familywise"][0] > 0)
    out["decision"] = dict(primary=primary, measured_concordance=measured_point, measured_ci95=measured_ci,
                           gates=gates, survival_superiority_established=all(gates.values()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, help="Run containing test_stats.npz; default: first generator")
    parser.add_argument("--direct-dir", type=Path, required=True)
    parser.add_argument("--direct-tags", nargs="+", default=["_b0", "_b1", "_b2"])
    parser.add_argument("--clinical-table", type=Path, required=True, help="Original published Supplementary Tables 1--5 XLSX")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases, meta, predictions = read_forecasts(args.generator_runs,
        args.reference_run or args.generator_runs[0], args.direct_dir, args.direct_tags)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Same-stage replicates can have no comparable pair.
        result = analyze(cases, meta, predictions, read_clinical(args.clinical_table))
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["clinical_source_sha256"] = hashlib.sha256(args.clinical_table.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote aggregate feasibility results to {args.output}")


if __name__ == "__main__":
    main()
