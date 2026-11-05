#!/usr/bin/env python
"""Patient-bootstrap 90% intervals for the downstream results (association, triage)."""
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downstream_analyses import patient_draws
from analyze_spatial_engression import PAIRNAMES, PREDEF, canonical

run = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/ligature_v3')
T, Pp, D, cases = patient_draws(run)
n = len(cases)
idx = [PAIRNAMES.index(canonical(q)) for q in PREDEF]
pairs = list(itertools.combinations(idx, 2))
Dm = D.mean(1)


def assoc_gain(ix):
    ep, et = [], []
    for i, j in pairs:
        rt = spearmanr(T[ix, i], T[ix, j]).correlation
        ep.append(abs(spearmanr(Pp[ix, i], Pp[ix, j]).correlation - rt))
        et.append(abs(np.mean([spearmanr(D[ix, k, i], D[ix, k, j]).correlation for k in range(0, D.shape[1], 4)]) - rt))
    return 1 - np.mean(et) / np.mean(ep)


def triage_gain(ix, frac=.2):
    k = max(1, int(round(frac * len(ix))))
    red = []
    for q in range(T.shape[1]):
        err = (Dm[ix, q] - T[ix, q]) ** 2
        sd = D[ix, :, q].std(1)
        top = np.argsort(-sd)[:k]
        red.append(err[top].sum() / err.sum())
    return np.mean(red) - frac


rng = np.random.default_rng(0)
B = 300
ag, tg = [], []
for _ in range(B):
    ix = rng.choice(n, n, replace=True)
    ag.append(assoc_gain(ix)); tg.append(triage_gain(ix))
full = np.arange(n)
out = dict(patients=n, association_error_reduction=float(assoc_gain(full)),
           association_error_reduction_ci90=[float(np.quantile(ag, .05)), float(np.quantile(ag, .95))],
           triage_excess_over_random=float(triage_gain(full)),
           triage_excess_ci90=[float(np.quantile(tg, .05)), float(np.quantile(tg, .95))])
(run / 'downstream_bootstrap.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
