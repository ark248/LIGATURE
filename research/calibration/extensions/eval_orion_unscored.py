#!/usr/bin/env python3
"""Evaluate nonlinear targets on frozen, genuinely patient-disjoint ORION fits."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import torch

PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PROJECT/'release/ligature'))).resolve()
BASE = ROOT/'runs/calibration'
RUN = BASE/'orion_unscored_v1'
os.environ.pop('SE_MARKERS', None)
os.environ.pop('SE_COPULA', None)
SOURCE = RUN/'src' if (RUN/'src/ligature.py').exists() else PROJECT/'release/ligature/src'
sys.path.insert(0, str(SOURCE))
import ligature as t
se = t.se


def read_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ct = read_module('contact_statistic_only', SOURCE/'eval_contact.py')
cn = read_module('neighborhood_statistic_only', SOURCE/'eval_neighborhoods.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fractions(num, den):
    return np.divide(num, np.maximum(den[..., None], 1), dtype=np.float64)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    folder = RUN/(f'smoke{args.fold}' if args.smoke else f'fold{args.fold}')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        print('Already complete', folder, flush=True)
        return
    manifest = json.loads((RUN/'source_manifest.json').read_text())
    for name, sha in manifest['files'].items():
        assert digest(RUN/'src'/name) == sha, name
    source_folder = RUN/f'fold{args.fold}'
    ref = json.loads((source_folder/'model_reference.json').read_text())
    original = Path(ref['directory'])
    for name, sha in ref['sha256'].items():
        assert digest(original/name) == sha, name
    roles = json.loads((source_folder/'roles.json').read_text())
    assert roles == json.loads((original/'roles.json').read_text())
    fit_roles = {c: 'train' if r == 'train' else 'val' if r == 'selection' else 'test' for c,r in roles.items()}
    se.roles_for = lambda cases, fold, seed: {str(c): fit_roles[str(c)] for c in cases}
    cells, x, y, crops, loaded = se.load(ROOT, args.fold, 271027, 128)
    assert loaded == fit_roles
    split = {r: [c for c in crops if roles[c['case']] == r] for r in ['train','selection','calibration','test']}
    if args.smoke:
        split = {r: [next(c for c in cs if c['case'] == case and len(c['ii']) >= 10) for case in sorted({c['case'] for c in cs})] for r,cs in split.items()}
    patients = {r: sorted({c['case'] for c in cs}) for r,cs in split.items()}
    assert list(map(len, patients.values())) == [20,2,10,8]
    mean = se.CellGraphNet(x.shape[1], d=128, layers=3)
    mean.load_state_dict(torch.load(original/'mean.pt', map_location='cpu', weights_only=True))
    net = t.Ligature(x.shape[1], d=128, layers=3, frozen_mean=True, dropout=.1,
                    scales=(10.,20.,40.,80.), patient_dim=4, region_dim=4, rank=4)
    net.load_state_dict(torch.load(original/'generator.pt', map_location='cpu', weights_only=True))
    net.no_stain_input = False
    net.no_geom = False
    mean.eval(); net.eval()
    train_cells = cells.case.map(roles).to_numpy() == 'train'
    threshold = np.quantile(y[train_cells], .85, axis=0)
    threshold_tensor = torch.from_numpy(threshold.astype(np.float32))
    xy = cells[['gx','gy']].to_numpy()* .325

    def neighbors(c):
        p = xy[c['ii']]
        return cKDTree(p).query(p, k=10)[1].reshape(len(p), 10)

    train_crops = [c for c in split['train'] if len(c['ii']) >= 10]
    training_compositions = np.concatenate([cn.compositions(cn.cell_types(y[c['ii']], threshold, se.MARKERS), neighbors(c)) for c in train_crops])
    rng = np.random.default_rng(args.fold)
    chosen = rng.choice(len(training_compositions), min(100000, len(training_compositions)), replace=False)
    km = KMeans(10, n_init=4, random_state=0).fit(training_compositions[chosen])
    fv = se.predict_mean(mean, split['selection'], x, y)
    threshold_plugin = np.quantile(np.concatenate(list(fv.values())), .85, axis=0)
    selection_rng = np.random.default_rng(271027+2701+args.fold)
    selection_crops = split['selection']
    chosen = selection_rng.choice(len(selection_crops), min(120, len(selection_crops)), replace=False)
    gen = torch.Generator().manual_seed(271027+2801+args.fold)
    selection_eta = {c: net.latent((8,1,net.dims['patient']),gen) for c in patients['selection']}
    samples = []
    for index in chosen:
        c = selection_crops[index]
        X,Y,D,A,M,NP = se.collate([c], x, y)
        mu,L = net.encode(X,D,M,torch.from_numpy(fv[c['crop']])[None])
        noise = net.noise(8,t.positions([c],cells),M,torch.zeros(1,dtype=torch.long),1,
                          gen=gen,eta=selection_eta[c['case']])
        samples.append(net.sample(mu,L,noise)[:,0].reshape(-1,se.P).numpy())
    threshold_samples = np.quantile(np.concatenate(samples), .85, axis=0)
    (folder/'fit_audit.json').write_text(json.dumps(dict(patient_sets=patients,
        measured_gate_patients=patients['train'],neighborhood_template_patients=patients['train'],
        plugin_quantile_gate_patients=patients['selection'],sample_quantile_gate_patients=patients['selection'],
        final_conformal_only_patients=patients['calibration'],model_reference=ref),indent=2)+'\n')
    (folder/'query_definition.json').write_text(json.dumps(dict(markers=list(se.MARKERS),
        measured_threshold=threshold.tolist(),plugin_threshold=threshold_plugin.tolist(),
        sample_threshold=threshold_samples.tolist(),centroids=km.cluster_centers_.tolist(),
        types=[n for n,_ in cn.TYPES]+['other'],contact_radius_um=20,min_measured_source_cells=20,
        neighborhood_k=10,gate_quantile=.85),indent=2)+'\n')
    held = sorted(split['calibration']+split['test'],key=lambda c:c['crop'])
    fm = se.predict_mean(mean,held,x,y)
    cases = sorted({c['case'] for c in held})
    K = 8 if args.smoke else 64
    gen = torch.Generator().manual_seed(271027+2901+args.fold)
    eta = {c: net.latent((K,1,net.dims['patient']),gen) for c in cases}
    keys = ['truth','plugin','plugin_q','samples','samples_q']
    contacts = {c: {key: [np.zeros((K,se.P,se.P) if key.startswith('samples') else (se.P,se.P)),
                         np.zeros((K,se.P) if key.startswith('samples') else se.P)] for key in keys} for c in cases}
    neighborhoods = {c: {key: np.zeros((K,10) if key.startswith('samples') else 10) for key in keys} for c in cases}
    n_cells = dict.fromkeys(cases,0)
    for index,c in enumerate(held):
        X,Y,D,A,M,NP = se.collate([c],x,y)
        f = torch.from_numpy(fm[c['crop']])[None]
        mu,L = net.encode(X,D,M,f)
        noise = net.noise(K,t.positions([c],cells),M,torch.zeros(1,dtype=torch.long),1,gen=gen,eta=eta[c['case']])
        draws = net.sample(mu,L,noise)[:,0]
        adjacency = (torch.from_numpy(c['dist']) <= 20).float()
        adjacency.fill_diagonal_(0)
        variants = [('truth',Y,threshold_tensor),('plugin',f,threshold_tensor),
                    ('plugin_q',f,torch.from_numpy(threshold_plugin.astype(np.float32))),
                    ('samples',draws,threshold_tensor),
                    ('samples_q',draws,torch.from_numpy(threshold_samples.astype(np.float32)))]
        for key,values,gate in variants:
            num,den = ct.contact(values,gate,adjacency)
            target = contacts[c['case']][key]
            target[0] += num if key.startswith('samples') else num[0]
            target[1] += den if key.startswith('samples') else den[0]
        if len(c['ii']) >= 10:
            nb = neighbors(c)
            for key,values,gate in variants:
                values = values.numpy()
                if not key.startswith('samples'):
                    values = values[0]
                cn_gate = threshold_plugin if key == 'plugin_q' else threshold_samples if key == 'samples_q' else threshold
                compositions = cn.compositions(cn.cell_types(values,cn_gate,se.MARKERS),nb)
                labs = km.predict(compositions.reshape(-1,len(cn.TYPES)+1))
                if key.startswith('samples'):
                    labs = labs.reshape(K,-1)
                    count = np.stack([np.bincount(row,minlength=10) for row in labs])
                else:
                    count = np.bincount(labs,minlength=10)
                neighborhoods[c['case']][key] += count
            n_cells[c['case']] += len(c['ii'])
        if index % 100 == 0:
            se.log(f'nonlinear fold {args.fold}: {index}/{len(held)} held-out regions')
    for role in ['calibration','test']:
        ids = [c for c in cases if roles[c] == role]
        assert all(n_cells[c] > 0 for c in ids)
        output = dict(case=np.asarray(ids))
        for key in keys:
            output[key] = np.stack([fractions(*contacts[c][key]).reshape((K,256) if key.startswith('samples') else 256) for c in ids])
        output['truth_den'] = np.stack([contacts[c]['truth'][1] for c in ids])
        output['eligible'] = np.repeat(output['truth_den'] >= 20, se.P, axis=1)
        np.savez_compressed(folder/f'contact_{role}.npz',**output)
        output = dict(case=np.asarray(ids))
        for key in keys:
            output[key] = np.stack([neighborhoods[c][key]/n_cells[c] for c in ids])
        output['eligible'] = np.ones((len(ids),10),dtype=bool)
        np.savez_compressed(folder/f'neighborhood_{role}.npz',**output)
    (folder/'complete.json').write_text(json.dumps(dict(seconds=time.time()-started,draws=K,
        smoke=args.smoke,contact_queries=256,neighborhood_queries=10),indent=2)+'\n')
    se.log(f'Completed independent nonlinear fold {args.fold} in {time.time()-started:.1f} seconds')


if __name__ == '__main__':
    main()
