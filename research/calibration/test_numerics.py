#!/usr/bin/env python3
"""Meaningful checks of conformal ranks, marginal coverage, and scale invariance."""
import numpy as np
from evaluate import conformal_rank, conformal_threshold

assert conformal_rank(10,.1)==10
assert conformal_rank(9,.1)==9
assert conformal_rank(4,.1)==5
assert conformal_rank(19,.05)==19

y=np.arange(1,11,dtype=float)[:,None]
q,k=conformal_threshold(y,np.zeros_like(y),np.ones_like(y),.1)
assert k==10 and q[0]==10
q2,_=conformal_threshold(y,np.zeros_like(y),np.full_like(y,5),.1)
assert np.allclose(q,5*q2)
qsmall,_=conformal_threshold(y[:4],np.zeros_like(y[:4]),np.ones_like(y[:4]),.1)
assert np.isinf(qsmall).all()

# One independent experiment per column: ten calibration and one test patient.
# For continuous iid scores, using the largest of ten covers 10/11 in expectation.
rng=np.random.default_rng(71)
scores=np.abs(rng.normal(size=(11,50000)))
coverage=np.mean(scores[-1]<=scores[:-1].max(0))
assert abs(coverage-10/11)<.006,coverage

# Image-dependent heteroscedastic normalization must retain the same calibration
# rank: this catches accidentally normalizing the final quantile a second time.
scales=np.exp(rng.normal(size=(10,4)))
errors=np.arange(40).reshape(10,4)+1.
q,_=conformal_threshold(errors*scales,np.zeros_like(errors),scales,.1)
assert np.array_equal(q,errors.max(0))
print(f'Conformal rank/scale checks passed; simulation coverage {coverage:.4f} (expected {10/11:.4f}).')
