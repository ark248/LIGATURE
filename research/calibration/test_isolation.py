#!/usr/bin/env python3
"""Check role disjointness and that evaluation labels cannot change intervals."""
import json
from pathlib import Path

import numpy as np

from evaluate import RUN, methods

test_cases=[]
for fold in range(5):
    roles=json.loads((RUN/f'fold{fold}/roles.json').read_text())
    groups={role:{c for c,v in roles.items() if v==role} for role in ['train','selection','calibration','test']}
    assert list(map(len,groups.values()))==[20,2,10,8]
    assert len(set.union(*groups.values()))==sum(map(len,groups.values()))
    test_cases.extend(groups['test'])
assert len(test_cases)==len(set(test_cases))==40

folder=RUN/'smoke0'
cal=dict(np.load(folder/'calibration_patients.npz'))
test=dict(np.load(folder/'test_patients.npz'))
selection=dict(np.load(folder/'selection_patients.npz'))
ia,ib=np.triu_indices(16);diag=ia==ib
expected,_=methods(cal,test,selection,diag)
poisoned={**test,'truth':test['truth']*0+12345}
actual,_=methods(cal,poisoned,selection,diag)
for name in expected:
    for key in ['center','lo','hi']:
        assert np.array_equal(expected[name][key],actual[name][key]),(name,key)
cal_poisoned={**cal,'truth':cal['truth']*0+12345}
altered,_=methods(cal_poisoned,test,selection,diag)
for name in expected:
    assert np.array_equal(expected[name]['center'],altered[name]['center']),name
    if name.endswith('conformal'):
        assert expected[name]['n_cal']==actual[name]['n_cal']==10
        assert expected[name]['rank']==actual[name]['rank']==10
        assert not np.array_equal(expected[name]['threshold'],altered[name]['threshold']),name
print('All patient splits disjoint; every patient tested once; poisoned test labels do not change intervals; calibration labels change thresholds only.')
