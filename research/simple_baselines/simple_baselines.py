"""Post hoc simple baselines for the RECOMB manuscript.

Added on 4 November 2026 after test results had been seen; these are exploratory and sit
outside the fixed protocol in research/query_bench_20261104/PROTOCOL.md.

Baselines, all constant-scale and calibrated with the published conformal rule:
  no_image            training-patient mean of each statistic.
  constant            plug-in plus one shift per statistic.
  constant_insample   the same shift using in-sample training plug-ins (a stronger variant).
  ridge               the published frozen-feature ridge, retargeted to the 136 statistics.

No test label enters any fit, and no neural model is fitted. Asserts reproduce published
numbers first: the plug-in rows, the reference and lung runs, and the stored 48-output ridge.

Run:  python research/simple_baselines/simple_baselines.py
"""
import json
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PROJECT/'release/ligature'))).resolve()
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/spatial_queries'
BASE = ROOT / 'runs/calibration/orion_v1'
STORED = RUN / 'evaluation/patient_metrics.parquet'
GRID = np.logspace(-4, 4, 9)
REPLICATES = 2000
ia, ib = np.triu_indices(16)
SAME, CROSS = np.flatnonzero(ia == ib), np.flatnonzero(ia != ib)
ALL = np.arange(136)
SCORED = {'all_scored': ALL, 'clustering': SAME, 'cross': CROSS}
QUERIES = {'Moran': np.arange(16), 'Geary': np.arange(16, 32), 'log_assay_clustering': np.arange(32, 48)}
meta = json.loads((RUN / 'geometry.json').read_text())
ELIGIBLE = [j for j, c in enumerate(meta) if c['eligible']]


def rng_for(*key):
    """An independent, order-free random stream for each comparison."""
    return np.random.default_rng([271027, *[int.from_bytes(str(k).encode()[:16], 'little') for k in key]])


def aggregate(idx, values):
    """Equal-weight region average per patient, with patients in sorted order."""
    cases = [meta[j]['case'] for j in idx]
    names = sorted(set(cases))
    return np.array(names), np.stack([values[[i for i, c in enumerate(cases) if c == p]].mean(0) for p in names])


def role_index(roles, role):
    return [j for j in ELIGIBLE if roles[meta[j]['case']] == role]


def conformal_constant(cy, cc, tc):
    """Rank 10 of 10 for nominal 90%, as in research/query_bench_20261104/evaluate.py."""
    k = int((Decimal(len(cy) + 1) * Decimal('0.9')).to_integral_value(rounding=ROUND_CEILING))
    half = np.sort(np.abs(cy - cc), axis=0)[k - 1]
    return tc - half, tc + half


def interval_metrics(ty, lo, hi, scale):
    width = hi - lo
    score = width + 20 * np.maximum(lo - ty, 0) + 20 * np.maximum(ty - hi, 0)
    return dict(score=score / scale, width=width / scale, covered=(ty >= lo) & (ty <= hi))


def ratio_mse(pred, truth, plugin, idx):
    den = ((truth - plugin)[:, idx] ** 2).mean(0).clip(1e-12)
    return float((((truth - pred)[:, idx] ** 2).mean(0) / den).mean())


def rank_correlation(pred, truth, idx):
    values = [spearmanr(pred[:, j], truth[:, j]).statistic for j in idx]
    return float(np.nanmean(values))


def median_bias(pred, truth, idx):
    measured = truth[:, idx].mean(0)
    keep = np.abs(measured) > .02
    return float(100 * np.median(((pred[:, idx].mean(0) - measured) / np.where(keep, measured, np.nan))[keep]))


def point_row(pred, truth, plugin):
    return dict(
        MSE_all=ratio_mse(pred, truth, plugin, ALL), MSE_cross=ratio_mse(pred, truth, plugin, CROSS),
        MSE_same=ratio_mse(pred, truth, plugin, SAME), bias_all=median_bias(pred, truth, ALL),
        bias_same=median_bias(pred, truth, SAME), rho_all=rank_correlation(pred, truth, ALL),
        rho_same=rank_correlation(pred, truth, SAME))


def boot_mse(preds_a, pred_b, truth, plugin, idx, key):
    """Paired patient bootstrap of the MSE-ratio difference A minus B. A may hold several seeds."""
    rng = rng_for('mse', *key)
    se_a = [(truth - p)[:, idx] ** 2 for p in preds_a]
    se_b = (truth - pred_b)[:, idx] ** 2
    se_p = (truth - plugin)[:, idx] ** 2
    n, draws = len(truth), np.empty(REPLICATES)
    for r in range(REPLICATES):
        i = rng.integers(0, n, n)
        d = se_p[i].mean(0).clip(1e-12)
        draws[r] = np.mean([(s[i].mean(0) / d).mean() for s in se_a]) - (se_b[i].mean(0) / d).mean()
    d = se_p.mean(0).clip(1e-12)
    point = np.mean([(s.mean(0) / d).mean() for s in se_a]) - (se_b.mean(0) / d).mean()
    return dict(diff=float(point), lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)))


def boot_score(score_a, score_b, key):
    """Paired patient bootstrap of the interval-score difference A minus B."""
    per = score_a.mean(1) - score_b.mean(1)
    draws = rng_for('score', *key).choice(per, size=(REPLICATES, len(per)), replace=True).mean(1)
    return dict(diff=float(per.mean()), lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)),
                patients_A_better=int((per < 0).sum()), patients=len(per))


def fit_ridge(fold, Y, P, roles):
    """Ridge on the plug-in residual; features, grid and selection follow cheap_readouts.py."""
    X = np.load(RUN / f'fold{fold}/targets.npz')['cheap_features'].astype(np.float64)
    ti, vi = role_index(roles, 'train'), role_index(roles, 'selection')
    xm, xs = X[ti].mean(0), X[ti].std(0).clip(1e-4)
    Z = (X - xm) / xs
    R = Y - P
    ym, ys = R[ti].mean(0), R[ti].std(0).clip(1e-4)
    r = (R - ym) / ys
    eig, V = np.linalg.eigh(Z[ti].T @ Z[ti] / len(ti))
    rhs = V.T @ (Z[ti].T @ r[ti] / len(ti))
    scale = aggregate(ti, Y[ti])[1].std(0, ddof=1).clip(1e-4)
    vtruth = aggregate(vi, Y[vi])[1]
    centers, errors = {}, []
    for penalty in GRID:
        mu = P + ym + (Z @ (V @ (rhs / (eig[:, None] + penalty)))) * ys
        errors.append(float(np.mean(((aggregate(vi, mu[vi])[1] - vtruth) / scale) ** 2)))
        centers[float(penalty)] = {role: aggregate(role_index(roles, role), mu[role_index(roles, role)])
                                   for role in ['calibration', 'test']}
    return centers, float(GRID[int(np.argmin(errors))])


def check_stored_ridge():
    worst = 0.
    for fold in range(5):
        roles = json.loads((BASE / f'fold{fold}/roles.json').read_text())
        T = np.load(RUN / f'fold{fold}/targets.npz')
        centers, penalty = fit_ridge(fold, T['queries'].astype(np.float64), T['queries_plugin'].astype(np.float64), roles)
        stored = np.load(RUN / f'fold{fold}/cheap/ridge/test.npz')
        case, center = centers[penalty]['test']
        assert list(stored['case']) == list(case)
        worst = max(worst, float(np.abs(stored['center'] - center).max()))
    assert worst < 1e-3, worst
    return worst


def contrast(s):
    """Near minus far; coordinates are 16 means followed by three bins of 136."""
    return s[..., 16:152] - s[..., 288:424]


def held_out_constant(folder, label):
    """Reference and lung runs: shift fitted on validation patients, applied to test patients."""
    truth, plugin, joint, shifted = [], [], [], []
    for fold in range(5):
        te, va = np.load(folder / f'fold{fold}/test_stats.npz'), np.load(folder / f'fold{fold}/val_stats.npz')
        def patient(z, values):
            keep = z['eligible'].astype(bool)
            cases = z['case'][keep]
            return np.stack([values[keep][cases == p].mean(0) for p in sorted(set(cases.tolist()))])
        shift = (patient(va, contrast(va['truth'])) - patient(va, contrast(va['plugin']))).mean(0)
        tp = patient(te, contrast(te['plugin']))
        truth.append(patient(te, contrast(te['truth'])))
        plugin.append(tp)
        joint.append(patient(te, contrast(te['samples']).mean(1)))
        shifted.append(tp + shift)
    truth, plugin, joint, shifted = map(np.concatenate, (truth, plugin, joint, shifted))
    return dict(
        patients=len(truth), plugin=point_row(plugin, truth, plugin), ligature=point_row(joint, truth, plugin),
        constant=point_row(shifted, truth, plugin),
        ligature_minus_constant_mse={k: boot_mse([joint], shifted, truth, plugin, idx, (label, k))
                                     for k, idx in SCORED.items()})


def independent():
    """The 20/2/10/8 independent-calibration protocol: all families and all three baselines."""
    parts = {}
    def add(method, family, fold_rows):
        parts.setdefault((method, family), []).append(fold_rows)
    penalties, grid_rows = [], {float(p): [] for p in GRID}
    for fold in range(5):
        roles = json.loads((BASE / f'fold{fold}/roles.json').read_text())
        T = np.load(RUN / f'fold{fold}/targets.npz')
        train = role_index(roles, 'train')
        scored_y = T['scored'][:, 424:560].astype(np.float64)
        scored_p = T['scored_plugin'][:, 424:560].astype(np.float64)
        expected = {r: np.load(BASE / f'fold{fold}/{r}_patients.npz') for r in ['selection', 'calibration', 'test']}
        lig = {r: np.load(RUN / f'fold{fold}/ligature_271027/{r}.npz') for r in ['selection', 'calibration', 'test']}
        ridge_centers, chosen = fit_ridge(fold, scored_y, scored_p, roles)
        penalties.append(chosen)
        queries_y = T['queries'].astype(np.float64)
        queries_p = T['queries_plugin'].astype(np.float64)
        train_y = {'scored': aggregate(train, scored_y[train])[1], 'queries': aggregate(train, queries_y[train])[1]}
        train_p = {'scored': aggregate(train, scored_p[train])[1], 'queries': aggregate(train, queries_p[train])[1]}
        for family, idx, kind, src in [(f, i, 'scored', expected) for f, i in SCORED.items()] + [(f, i, 'queries', lig) for f, i in QUERIES.items()]:
            cy, cp = src['calibration']['truth'][:, idx], src['calibration']['plugin'][:, idx]
            ty, tp = src['test']['truth'][:, idx], src['test']['plugin'][:, idx]
            case = [str(c) for c in src['test']['case']]
            tr_y, tr_p = train_y[kind][:, idx], train_p[kind][:, idx]
            scale = tr_y.std(0, ddof=1).clip(1e-4)
            mean_y = tr_y.mean(0)
            unlabeled = np.concatenate([src['selection']['plugin'][:, idx], cp]).mean(0)
            selected_shift = (src['selection']['truth'][:, idx] - src['selection']['plugin'][:, idx]).mean(0)
            centers = {
                'plugin': (cp, tp),
                'no_image': (np.broadcast_to(mean_y, cy.shape), np.broadcast_to(mean_y, ty.shape)),
                'constant': (cp + mean_y - unlabeled, tp + mean_y - unlabeled),
                'constant_insample': (cp + (tr_y - tr_p).mean(0), tp + (tr_y - tr_p).mean(0)),
                'constant_selection': (cp + selected_shift, tp + selected_shift),
            }
            if kind == 'scored':
                centers['ridge'] = (ridge_centers[chosen]['calibration'][1][:, idx], ridge_centers[chosen]['test'][1][:, idx])
                assert list(ridge_centers[chosen]['test'][0]) == list(src['test']['case'])
            for method, (cc, tc) in centers.items():
                m = interval_metrics(ty, *conformal_constant(cy, cc, tc), scale)
                add(method, family, dict(case=case, truth=ty, plugin=tp, center=tc, **m))
            if family == 'all_scored':
                for penalty, c in ridge_centers.items():
                    m = interval_metrics(ty, *conformal_constant(cy, c['calibration'][1], c['test'][1]), scale)
                    grid_rows[penalty].append(dict(case=case, truth=ty, plugin=tp, center=c['test'][1], **m))
    def stack(rows):
        out = {k: np.concatenate([r[k] for r in rows]) for k in ['truth', 'plugin', 'center', 'score', 'width', 'covered']}
        out['case'] = sum((r['case'] for r in rows), [])
        return out
    frames = {key: stack(rows) for key, rows in parts.items()}
    columns = ['seed', 'case', 'family', 'query', 'method', 'interval', 'center', 'coverage', 'width', 'interval_score', 'squared_error', 'plugin_squared_error']
    stored = pd.read_parquet(STORED, columns=columns)

    # The plug-in path must reproduce the published records. The tolerance is far below the
    # two decimals the manuscript reports; results.json records the measured gap.
    plugin_gap = 0.
    for family in list(SCORED) + list(QUERIES):
        g = stored[(stored.family == family) & (stored.method == 'plugin') & (stored.interval == 'constant')]
        f = frames[('plugin', family)]
        plugin_gap = max(plugin_gap, abs(g.interval_score.mean() - f['score'].mean()), abs(g.width.mean() - f['width'].mean()))
        assert abs(g.coverage.mean() - f['covered'].mean()) < 1e-3, family
    assert plugin_gap < 1e-3, plugin_gap

    metrics = []
    for (method, family), f in frames.items():
        idx = np.arange(f['truth'].shape[1])
        metrics.append(dict(family=family, method=method, interval='constant', patients=len(set(f['case'])),
                            MSE_ratio=ratio_mse(f['center'], f['truth'], f['plugin'], idx), coverage=float(f['covered'].mean()),
                            width=float(f['width'].mean()), interval_score=float(f['score'].mean())))

    order = frames[('no_image', 'all_scored')]['case']
    def stored_scores(method, family, interval='adaptive'):
        g = stored[(stored.method == method) & (stored.family == family) & (stored.interval == interval)]
        return g.pivot_table(index='case', columns='query', values='interval_score', aggfunc='mean').loc[order].to_numpy()
    def stored_centers(method):
        g = stored[stored.method == method]
        g = g[g.family.isin(['clustering', 'cross']) & (g.interval == 'constant')]
        return [s.pivot(index='case', columns='query', values='center').loc[order, list(range(136))].to_numpy() for _, s in g.groupby('seed')]

    comparisons = {}
    for family in list(SCORED) + ['Moran', 'Geary']:
        baselines = ['no_image', 'constant', 'constant_insample'] + (['ridge'] if family in SCORED else [])
        for a in ['ligature', 'direct']:
            for b in baselines:
                comparisons[f'{a}_adaptive_minus_{b}_score_{family}'] = boot_score(stored_scores(a, family), frames[(b, family)]['score'], (a, b, family))

    all_f = frames[('no_image', 'all_scored')]
    truth, plugin = all_f['truth'], all_f['plugin']
    seeds = {m: stored_centers(m) for m in ['ligature', 'direct']}
    points = {}
    for b in ['constant', 'constant_insample', 'ridge']:
        for a in seeds:
            for family, idx in SCORED.items():
                comparisons[f'{a}_minus_{b}_mse_{family}'] = boot_mse(seeds[a], frames[(b, 'all_scored')]['center'], truth, plugin, idx, (a, b, family))
        points[b] = point_row(frames[(b, 'all_scored')]['center'], truth, plugin)
    points['plugin'] = point_row(plugin, truth, plugin)
    for m in seeds:
        rows = [point_row(p, truth, plugin) for p in seeds[m]]
        points[m] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    sensitivity = []
    for penalty, rows in grid_rows.items():
        f = stack(rows)
        sensitivity.append(dict(penalty=penalty, MSE_all=ratio_mse(f['center'], f['truth'], f['plugin'], ALL),
                                MSE_same=ratio_mse(f['center'], f['truth'], f['plugin'], SAME),
                                score_all=float(f['score'].mean()), score_same=float(f['score'][:, SAME].mean())))
    best = min(sensitivity, key=lambda r: r['score_all'])
    f = stack(grid_rows[best['penalty']])
    hardest = boot_score(stored_scores('ligature', 'all_scored'), f['score'], ('ligature', 'best_fixed_ridge'))
    return dict(plugin_reproduction_max_gap=float(plugin_gap), metrics=metrics, comparisons=comparisons, point=points, ridge_selected_penalties=penalties,
                ridge_penalty_sensitivity=sensitivity, ridge_best_fixed_penalty=best,
                ligature_minus_best_fixed_ridge_score=hardest)


def main():
    worst = check_stored_ridge()
    reference = held_out_constant(ROOT / 'runs/ligature_v3', 'reference')
    published = json.loads((ROOT / 'runs/ligature_v3/summary.json').read_text())
    assert abs(reference['ligature']['MSE_all'] - published['all136']['nmse_joint']) < 1e-3
    assert abs(reference['ligature']['MSE_same'] - published['diag16']['nmse_joint']) < 1e-3
    lung = held_out_constant(ROOT / 'runs/lung/ligature', 'lung')
    assert abs(lung['ligature']['MSE_all'] - .84) < .005 and abs(lung['ligature']['MSE_same'] - .48) < .006
    result = dict(
        status='complete',
        note='Post hoc and exploratory: added after test results were seen; outside the fixed protocol.',
        checks=dict(stored_48_output_ridge_max_abs_difference=worst, plugin_rows_reproduced=True,
                    reference_and_lung_reproduced=True),
        reference=reference, lung=lung, independent=independent())
    (OUT / 'results.json').write_text(json.dumps(result, indent=1) + '\n')
    print('status', result['status'], 'ridge reproduction', f'{worst:.2e}')


if __name__ == '__main__':
    main()
