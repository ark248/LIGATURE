#!/usr/bin/env python3
"""Freeze source and patient roles without changing shared research files."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PROJECT/'release/ligature'))).resolve()
OUT = ROOT / 'runs/calibration/orion_v1'
SOURCES = ['ligature.py', 'spatial_engression.py', 'eval_unseen_statistics.py', 'analytic_moments.py']


def main():
    snapshot = OUT / 'src'
    snapshot.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in SOURCES:
        source = PROJECT / 'release/ligature/src' / name
        destination = snapshot / name
        data = source.read_bytes()
        if destination.exists() and destination.read_bytes() != data:
            raise FileExistsError(f'Frozen snapshot already differs: {destination}')
        if not destination.exists():
            destination.write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True, capture_output=True)
    commit = result.stdout.strip() if result.returncode == 0 else 'source-archive'
    provenance = OUT / 'source_manifest.json'
    if provenance.exists():
        original_manifest = json.loads(provenance.read_text())
        if original_manifest['files'] != manifest:
            raise FileExistsError('Existing source manifest differs; preserve the original experiment')
    else:
        provenance.write_text(json.dumps(dict(commit=commit, files=manifest), indent=2)+'\n')
    import sys, pandas as pd
    sys.path.insert(0, str(PROJECT/'release/ligature/src'))
    import spatial_engression as se
    cases = sorted(pd.read_parquet(ROOT/'data/cells/cells.parquet', columns=['case']).case.unique())
    for fold in range(5):
        original = se.roles_for(cases, fold, 271027)
        rng = np.random.default_rng(271027 + 701 + fold)
        previous_train = np.array(sorted(k for k,v in original.items() if v=='train'))
        order = rng.permutation(previous_train)
        calibration = set(order[:2]) | {k for k,v in original.items() if v=='val'}
        selection = set(order[2:4])
        train = set(order[4:])
        test = {k for k,v in original.items() if v=='test'}
        sets = dict(train=train, selection=selection, calibration=calibration, test=test)
        assert [len(sets[k]) for k in ['train','selection','calibration','test']]==[20,2,10,8]
        assert sum(map(len,sets.values()))==len(set.union(*sets.values()))==40
        roles = {patient:next(k for k,v in sets.items() if patient in v) for patient in sorted(original)}
        folder = OUT / f'fold{fold}';folder.mkdir(exist_ok=True)
        destination = folder/'roles.json'
        value = json.dumps(roles,indent=2)+'\n'
        if destination.exists() and destination.read_text()!=value:
            raise FileExistsError(f'Roles already differ: {destination}')
        if not destination.exists():
            destination.write_text(value)
    protocol = (PROJECT/'research/calibration/PROTOCOL.md').read_bytes()
    frozen_protocol = OUT/'PROTOCOL.md'
    if frozen_protocol.exists() and frozen_protocol.read_bytes() != protocol:
        raise FileExistsError('Existing protocol differs; preserve the original experiment')
    if not frozen_protocol.exists():
        frozen_protocol.write_bytes(protocol)
    print(f'Prepared source snapshot and five fixed 20/2/10/8 splits in {OUT}')


if __name__=='__main__':main()
