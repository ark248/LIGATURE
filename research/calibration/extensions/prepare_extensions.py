#!/usr/bin/env python3
"""Prepare immutable source/role records for the two owned calibration extensions."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PROJECT/'release/ligature'))).resolve()
BASE = ROOT/'runs/calibration'
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Frozen artifact differs: {path}')
    else:
        path.write_bytes(data)


def json_once(path, value):
    write_once(path, (json.dumps(value, indent=2)+'\n').encode())


def main():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True, capture_output=True)
    head = result.stdout.strip() if result.returncode == 0 else 'source-archive'
    old = BASE/'orion_v1'
    source_manifest = json.loads((old/'source_manifest.json').read_text())
    for experiment in ['lung_v1', 'orion_unscored_v1']:
        run = BASE/experiment
        files = {}
        for name, want in source_manifest['files'].items():
            path = old/'src'/name
            assert digest(path) == want
            write_once(run/'src'/name, path.read_bytes())
            files[name] = want
        if experiment == 'orion_unscored_v1':
            for name in ['eval_contact.py', 'eval_neighborhoods.py']:
                path = PROJECT/'release/ligature/src'/name
                write_once(run/'src'/name, path.read_bytes())
                files[name] = digest(path)
        provenance = dict(model_source_commit=source_manifest['commit'],
                          evaluator_source_commit=head, files=files)
        manifest = run/'source_manifest.json'
        if manifest.exists():
            assert json.loads(manifest.read_text())['files'] == files
        else:
            json_once(manifest, provenance)
        write_once(run/'PROTOCOL.md', (HERE/'PROTOCOL.md').read_bytes())
    tests = []
    for fold in range(5):
        import sys, pandas as pd
        sys.path.insert(0, str(PROJECT/'release/ligature/src'))
        import spatial_engression as se
        cases = sorted(pd.read_parquet(ROOT/'cohorts/lung/data/cells/cells.parquet', columns=['case']).case.unique())
        old_roles = se.roles_for(cases, fold, 271027)
        train = np.array(sorted(k for k,v in old_roles.items() if v == 'train'))
        order = list(np.random.default_rng(271027+1701+fold).permutation(train))
        cal = {k for k,v in old_roles.items() if v == 'val'}
        while len(cal) < 4:
            cal.add(order.pop(0))
        selection = {order.pop(0)}
        fitting = set(order)
        test = {k for k,v in old_roles.items() if v == 'test'}
        groups = dict(train=fitting, selection=selection, calibration=cal, test=test)
        assert [len(groups[k]) for k in groups] == [10+(len(test)==3), 1, 4, len(test)]
        assert sum(map(len, groups.values())) == len(set.union(*groups.values())) == 19
        roles = {case: next(k for k,g in groups.items() if case in g) for case in sorted(old_roles)}
        json_once(BASE/f'lung_v1/fold{fold}/roles.json', roles)
        tests.extend(test)
        old_fold = old/f'fold{fold}'
        orion_roles = json.loads((old_fold/'roles.json').read_text())
        json_once(BASE/f'orion_unscored_v1/fold{fold}/roles.json', orion_roles)
        model_record = {name: digest(old_fold/name) for name in ['mean.pt', 'generator.pt', 'roles.json']}
        json_once(BASE/f'orion_unscored_v1/fold{fold}/model_reference.json',
                  dict(directory=str(old_fold), sha256=model_record))
    assert len(tests) == len(set(tests)) == 19
    print('Prepared fixed lung roles (19 test patients once) and references to all five independent ORION fits.')


if __name__ == '__main__':
    main()
