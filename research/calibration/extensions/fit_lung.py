#!/usr/bin/env python3
"""Fit a genuinely separate mean/generator/calibration split in an isolated run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__import__('os').environ.get('LIGATURE_DATA_ROOT', str(PROJECT/'release/ligature'))).resolve()
DATA = ROOT / 'cohorts/lung'
RUN = ROOT / 'runs/calibration/lung_v1'
SOURCE = RUN/'src' if (RUN/'src/ligature.py').exists() else PROJECT/'release/ligature/src'
sys.path.insert(0, str(SOURCE))
import os
os.environ['SE_MARKERS'] = 'EPCAM,SFTPC,AGER,SCGB1A1,KRT5,PTPRC,CD3E,CD8A,MS4A1,CD68,MARCO,COL1A1,ACTA2,PECAM1,MKI67,SPP1'
os.environ.pop('SE_COPULA', None)
import ligature as t
se = t.se


def query(S):
    k = len(se.IU[0])
    return S[..., se.P:se.P+k] - S[..., se.P+2*k:se.P+3*k]


def make_args(fold, threads, smoke=False):
    return argparse.Namespace(root=DATA, fold=fold, seed=271027, pcadim=128,
        d=128, layers=3, epochs=1 if smoke else 15, lr=1e-3,
        groups=4, per=4, samples=4 if smoke else 16, eval_samples=8 if smoke else 64,
        threads=threads, q=.85, frozen_mean=True, mean_correction=False,
        dropout=.1, wd=1e-3, scales='10,20,40,80', patient_dim=4,
        region_dim=4, rank=4, model_seed=None, tail_df=0, objective='crps',
        stationary=False, no_stain_input=False, no_geom=False,
        no_contrasts=False, lam_cell=1., lam_region=1., lam_patient=1.,
        cell_noise=0, group_noise=0, late_noise=False, lam_crop=1., lam_group=1.)


def patient_bundle(crops, results):
    truth, plugin, draws = query(results['truth']), query(results['plugin']), query(results['samples'])
    cases = np.array([c['case'] for c in crops])
    eligible = np.array([c['eligible'] for c in crops])
    ids = sorted(set(cases))
    out = dict(case=np.array(ids), truth=[], plugin=[], draws=[], n_regions=[])
    for case in ids:
        mask = (cases == case) & eligible
        if not mask.any():
            raise ValueError(f'No eligible regions for {case}')
        out['truth'].append(truth[mask].mean(0))
        out['plugin'].append(plugin[mask].mean(0))
        out['draws'].append(draws[mask].mean(0))
        out['n_regions'].append(int(mask.sum()))
    return {k:np.asarray(v) for k,v in out.items()}


@torch.no_grad()
def deterministic_bundle(crops, x, y, mean=None):
    predicted = se.predict_mean(mean, crops, x, y) if mean is not None else None
    truth, plugin, cases = [], [], []
    for c in crops:
        if not c['eligible']:
            continue
        Y=torch.from_numpy(y[c['ii']])[None]
        A=torch.from_numpy(c['A'])[None]
        NP=torch.from_numpy(c['npairs'])[None]
        M=torch.ones(1,len(c['ii']))
        truth.append(query(se.crop_stats(Y,A,NP,M))[0].numpy())
        if predicted is not None:
            F=torch.from_numpy(predicted[c['crop']])[None]
            plugin.append(query(se.crop_stats(F,A,NP,M))[0].numpy())
        cases.append(c['case'])
    ids=sorted(set(cases));cases=np.asarray(cases);truth=np.asarray(truth)
    out=dict(case=np.asarray(ids),truth=np.stack([truth[cases==c].mean(0) for c in ids]))
    if predicted is not None:
        plugin=np.asarray(plugin)
        out['plugin']=np.stack([plugin[cases==c].mean(0) for c in ids])
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--phase',choices=['mean','generator','both'],default='both')
    cli=parser.parse_args()
    torch.set_num_threads(cli.threads)
    started=time.time()
    folder=RUN/(f'smoke{cli.fold}' if cli.smoke else f'fold{cli.fold}')
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        print(f'Already complete: {folder}',flush=True);return
    manifest=json.loads((RUN/'source_manifest.json').read_text())
    for name,sha in manifest['files'].items():
        assert hashlib.sha256((RUN/'src'/name).read_bytes()).hexdigest()==sha, name
    roles=json.loads((RUN/f'fold{cli.fold}/roles.json').read_text())
    fit_roles={k:('train' if v=='train' else 'val' if v=='selection' else 'test') for k,v in roles.items()}
    se.roles_for=lambda cases,fold,seed: {str(c):fit_roles[str(c)] for c in cases}
    args=make_args(cli.fold,cli.threads,cli.smoke)
    (folder/'settings.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')
    cells,x,y,crops,loaded_roles=se.load(DATA,args.fold,args.seed,args.pcadim)
    assert loaded_roles==fit_roles
    split={role:[c for c in crops if roles[c['case']]==role] for role in ['train','selection','calibration','test']}
    audit={role:sorted({c['case'] for c in cs}) for role,cs in split.items()}
    assert [len(audit[k]) for k in ['train','selection','calibration','test']]==[10+(len(audit['test'])==3),1,4,len(audit['test'])]
    assert not(set(audit['calibration']) & (set(audit['train'])|set(audit['selection'])|set(audit['test'])))
    (folder/'fit_audit.json').write_text(json.dumps(dict(patient_sets=audit,
        normalization_and_pca_patients=audit['train'],
        mean_training_patients=audit['train'],mean_early_stopping_patients=audit['selection'],
        generator_training_patients=audit['train'],generator_early_stopping_patients=audit['selection'],
        variance_fitting_patients=audit['selection'],final_conformal_only_patients=audit['calibration']),indent=2)+'\n')
    if cli.smoke:
        split={k:[c for case in sorted({z['case'] for z in cs}) for c in [z for z in cs if z['case']==case][:1]] for k,cs in split.items()}
    tr,sel=split['train'],split['selection']
    base_scale=se.stat_scale(tr,y)
    mean_args=argparse.Namespace(**{**vars(args),'epochs':1 if cli.smoke else 25})
    if cli.phase == 'generator' and not (folder/'mean.pt').exists():
        raise RuntimeError('Generator phase requires its independently completed mean fit')
    if (folder/'mean.pt').exists():
        mean=se.CellGraphNet(x.shape[1],d=args.d,layers=args.layers)
        mean.load_state_dict(torch.load(folder/'mean.pt',map_location='cpu',weights_only=True))
        se.log('Loaded completed independent mean fit')
    else:
        mean,hist=se.train_model('mean',tr,sel,x,y,mean_args,base_scale)
        torch.save(mean.state_dict(),folder/'mean.pt')
        pd.DataFrame(hist).to_csv(folder/'mean_history.csv',index=False)
    mean.eval()
    (folder/'mean_stage_complete.json').write_text(json.dumps(dict(seconds=time.time()-started)))
    if cli.phase == 'mean':
        se.log(f'Completed lung mean phase {cli.fold} in {time.time()-started:.1f} s')
        return
    scales=[]
    for c in tr:
        S=se.crop_stats(torch.from_numpy(y[c['ii']])[None],torch.from_numpy(c['A'])[None],
            torch.from_numpy(c['npairs'])[None],torch.ones(1,len(c['ii'])))
        scales.append(t.contrasts(S))
    scale=torch.cat(scales).std(0).clamp(min=1e-3)
    torch.save(scale,folder/'training_stat_scale.pt')
    if (folder/'generator_uncalibrated.pt').exists():
        net=t.Ligature(x.shape[1],d=args.d,layers=args.layers,frozen_mean=True,
            dropout=args.dropout,scales=(10.,20.,40.,80.),patient_dim=4,region_dim=4,rank=4)
        net.load_state_dict(torch.load(folder/'generator_uncalibrated.pt',map_location='cpu',weights_only=True))
        se.log('Loaded completed independent generator fit')
    else:
        net,hist=t.train(tr,sel,x,y,cells,args,scale,mean_net=mean)
        torch.save(net.state_dict(),folder/'generator_uncalibrated.pt')
        pd.DataFrame(hist).to_csv(folder/'generator_history.csv',index=False)
    net.no_stain_input=False;net.no_geom=False
    torch.manual_seed(args.seed+41)
    net=t.calibrate_scales(net,sel,x,y,cells,args,scale,mean_net=mean,epochs=1 if cli.smoke else 4)
    torch.save(net.state_dict(),folder/'generator.pt')
    se.log('All model fitting finished; independent calibration labels have not been fitted')
    held=sorted(split['calibration']+split['test'],key=lambda c:c['crop'])
    res,_=t.evaluate(net,mean,held,x,y,cells,args,args.eval_samples)
    bundle=patient_bundle(held,res)
    for role in ['calibration','test']:
        keep=np.array([roles[str(c)]==role for c in bundle['case']])
        np.savez_compressed(folder/f'{role}_patients.npz',**{k:v[keep] for k,v in bundle.items()})
    np.savez_compressed(folder/'selection_patients.npz',**deterministic_bundle(sel,x,y,mean))
    np.savez_compressed(folder/'training_patients.npz',**deterministic_bundle(tr,x,y))
    (folder/'markers.json').write_text(json.dumps(list(se.MARKERS)))
    (folder/'complete.json').write_text(json.dumps(dict(seconds=time.time()-started,
        split_patient_counts={k:len(v) for k,v in audit.items()},eval_samples=args.eval_samples,
        smoke=cli.smoke,source_commit=manifest['model_source_commit']),indent=2)+'\n')
    se.log(f'Completed isolated fold {cli.fold} in {time.time()-started:.1f} s')


if __name__=='__main__':main()
