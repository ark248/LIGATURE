#!/usr/bin/env python3
"""Check finite-sample levels, label isolation, eligibility, and CN invariance."""
import numpy as np
from scipy.stats import norm

import evaluate_lung as lung
from evaluate_unscored import RUN, intervals, rank

assert rank(0,.1)==1 and rank(8,.1)==9 and rank(9,.1)==9 and rank(10,.1)==10
assert lung.conformal_rank(4,.2)==4 and lung.conformal_rank(4,.1)==5
rng=np.random.default_rng(108)
scores=np.abs(rng.normal(size=(5,40000)))
coverage=np.mean(scores[-1]<=scores[:-1].max(0))
assert abs(coverage-.8)<.008,coverage

folder=RUN/'smoke0'
for family in ['contact','neighborhood']:
    cal=dict(np.load(folder/f'{family}_calibration.npz'))
    test=dict(np.load(folder/f'{family}_test.npz'))
    expected=intervals(cal,test)
    poisoned={**test,'truth':test['truth']*0+12345}
    altered=intervals(cal,poisoned)
    for method,p in expected.items():
        for key in ['center','lo','hi']:
            np.testing.assert_array_equal(p[key],altered[method][key])
        assert np.all(p['lo']>=0) and np.all(p['hi']<=1) and np.all(p['lo']<=p['hi'])
    masked={**cal,'eligible':np.zeros_like(cal['eligible'])}
    insufficient=intervals(masked,test)
    for method,p in insufficient.items():
        if method.endswith('conformal'):
            assert np.isinf(p['threshold']).all()
            assert (p['lo']==0).all() and (p['hi']==1).all()
            assert (p['n_cal']==0).all() and (p['rank']==1).all()
    if family=='neighborhood':
        perm=rng.permutation(10)
        def reordered(bundle):
            return {k:v if k=='case' else v[...,perm] for k,v in bundle.items()}
        shuffled=intervals(reordered(cal),reordered(test))
        for method,p in expected.items():
            for key in ['center','lo','hi']:
                np.testing.assert_allclose(p[key][...,perm],shuffled[method][key],rtol=1e-12,atol=1e-12)

# Raw Gaussian baselines must use the lung experiment's 80% level, not 90%.
rng=np.random.default_rng(91)
cal=dict(truth=rng.normal(size=(4,136)),plugin=rng.normal(size=(4,136)),draws=rng.normal(size=(4,32,136)))
test=dict(truth=rng.normal(size=(3,136)),plugin=rng.normal(size=(3,136)),draws=rng.normal(size=(3,32,136)))
selection=dict(truth=rng.normal(size=(1,136)),plugin=rng.normal(size=(1,136)))
ia,ib=np.triu_indices(16)
out,_=lung.methods(cal,test,selection,ia==ib,alpha=.2)
sd=np.maximum(np.sqrt(np.mean((selection['truth']-selection['plugin'])**2,axis=0)),1e-6)
np.testing.assert_allclose(out['plugin_raw_gaussian']['hi']-out['plugin_raw_gaussian']['center'],np.broadcast_to(norm.ppf(.9)*sd,(3,136)))
assert out['ligature_conformal']['rank']==4
print(f'Passed: 80% simulation coverage {coverage:.4f}; exact ranks; support/eligibility edge cases; test-label isolation; CN permutation invariance; matched Gaussian level.')
