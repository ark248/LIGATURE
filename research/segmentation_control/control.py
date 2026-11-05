"""Fixed-model, non-touching-cell sensitivity analysis; separate outputs only."""
from pathlib import Path
import argparse,json,hashlib,io,time,sys
import numpy as np,pandas as pd,pyarrow.parquet as pq,tifffile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mask_pairs(mask):
 pairs=[]
 for dy in range(3):
  for dx in range(-2,3):
   if dy==0 and dx<=0:continue
   y0,y1=max(0,dy),mask.shape[0]+min(0,dy);x0,x1=max(0,dx),mask.shape[1]+min(0,dx)
   a=mask[y0:y1,x0:x1];b=mask[y0-dy:y1-dy,x0-dx:x1-dx]
   keep=(a>0)&(b>0)&(a!=b)
   if keep.any():pairs.append(np.sort(np.column_stack([a[keep],b[keep]]),axis=1))
 return np.unique(np.concatenate(pairs),axis=0) if pairs else np.empty((0,2),dtype=np.int64)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 root=a.root.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 assert not out.is_relative_to(root/'runs/ligature_v3'), 'Use a separate control output directory'
 import shutil
 shutil.copy2(Path(__file__).with_name('plan.json'),out/'segmentation_plan.json')
 start=time.monotonic();package=Path(__file__).resolve().parents[2]/'release/ligature/src';sys.path.insert(0,str(package))
 import torch
 torch.set_num_threads(2)
 import ligature as t
 se=t.se
 # Input-only geometric cache. No outcomes used in the exclusion rule.
 meta=pd.read_csv(root/'data/cells/crops.csv').set_index('id');cache=out/'mask_pairs.npz'
 cropnames=pd.read_parquet(root/'data/cells/cells.parquet',columns=['crop_id']).crop_id.unique()
 if cache.exists():contacts={k:v for k,v in np.load(cache).items()}
 else:
  files={};contacts={}
  for k,name in enumerate(sorted(cropnames)):
   row=meta.loc[name];f=root/'data/orion'/Path(row.file).name
   if f not in files:files[f]=pq.ParquetFile(f)
   v=files[f].read_row_group(int(row.row_group),columns=['inst_cell']).to_pylist()[0]['inst_cell']
   contacts[name]=mask_pairs(tifffile.imread(io.BytesIO(v)))
   if k%100==0:print('masks',k,len(cropnames),'seconds',round(time.monotonic()-start,1),flush=True)
  np.savez_compressed(cache,**contacts)
 print('Mask cache ready',round(time.monotonic()-start,1),flush=True)
 rows=[];checkpoints={};mc_check=None
 for fold in range(5):
  fd=root/f'runs/ligature_v3/fold{fold}';args=json.loads((fd/'args.json').read_text())
  cells,x,y,crops,roles=se.load(root,fold,int(args['seed']),int(args['pcadim']))
  stain=se.CellGraphNet(x.shape[1],d=int(args['d']),layers=int(args['layers']))
  stainfile=root/f'runs/spatial_engression/fold{fold}/mean.pt';model=fd/'ligature.pt'
  stain.load_state_dict(torch.load(stainfile,map_location='cpu',weights_only=True));stain.eval()
  scales=tuple(map(float,args['scales'].split(',')))
  net=t.Ligature(x.shape[1],d=int(args['d']),layers=int(args['layers']),frozen_mean=True,scales=scales,patient_dim=int(args['patient_dim']))
  missing,unexpected=net.load_state_dict(torch.load(model,map_location='cpu',weights_only=True),strict=False)
  assert set(missing)<={'const_load'} and not unexpected
  net.eval()
  checkpoints[str(model.relative_to(root))]=sha(model);checkpoints[str(stainfile.relative_to(root))]=sha(stainfile)
  with torch.no_grad():
   for k,c in enumerate(c for c in crops if c['role']=='test' and c['eligible']):
    X,Y,D,A,M,NP=se.collate([c],x,y);f=stain(X,D,M);mu,L=net.encode(X,D,M,f)
    dist=D[0].numpy();n=len(c['ii'])
    # Preserve the original float64-defined bin membership; D is float32 for the network.
    near=A[0,0].numpy().astype(bool);far=A[0,2].numpy().astype(bool)
    labels=cells.iloc[c['ii']].cell_label.to_numpy();inv={int(v):i for i,v in enumerate(labels)}
    excluded=np.zeros((n,n),bool)
    for u,v in contacts[c['crop']]:
     if int(u) in inv and int(v) in inv:excluded[inv[int(u)],inv[int(v)]]=excluded[inv[int(v)],inv[int(u)]]=True
    restricted=near&~excluded
    if not restricted.any():continue
    weights=[near/near.sum()-far/far.sum(),restricted/restricted.sum()-far/far.sum()]
    truth=Y[0].numpy();mean=mu[0].numpy();load=L[0].numpy()
    # [marker,cell,cell] joint second moments: mean product plus covariance.
    second=mean.T[:,:,None]*mean.T[:,None,:]
    for j,ell in enumerate(scales):
     block=load[...,j*net.rank:(j+1)*net.rank].transpose(1,0,2)
     second+=np.matmul(block,block.transpose(0,2,1))*np.exp(-dist[None]**2/(2*ell**2))
    st=net.dims['field']+net.dims['cell'];shared=load[...,st:].transpose(1,0,2)
    second+=np.matmul(shared,shared.transpose(0,2,1))
    # Cell noise only contributes to diagonal, excluded from both weights.
    if mc_check is None:
     z=net.noise(1024,t.positions([c],cells),M,torch.zeros(1,dtype=torch.long),1,gen=torch.Generator().manual_seed(20261104))
     s=net.sample(mu,L,z)[:,0,:,0].numpy();sampleq=np.einsum('si,ij,sj->s',s,weights[0],s,optimize=True)
     exact=float((second[0]*weights[0]).sum());err=float(sampleq.mean()-exact);se_mc=float(sampleq.std(ddof=1)/np.sqrt(len(sampleq)))
     assert abs(err)<6*se_mc+1e-6,(exact,err,se_mc)
     mc_check={'region':c['crop'],'draws':len(sampleq),'marker':se.MARKERS[0],'exact':exact,'mc_mean':float(sampleq.mean()),'mc_standard_error':se_mc}
    for variant,w in zip(['all_pairs','non_touching'],weights):
     tru=np.einsum('ip,ij,jp->p',truth,w,truth,optimize=True);plug=np.einsum('ip,ij,jp->p',mean,w,mean,optimize=True);joint=(second*w[None]).sum((1,2))
     for j,marker in enumerate(se.MARKERS):rows.append(dict(fold=fold,case=c['case'],crop=c['crop'],marker=marker,variant=variant,truth=float(tru[j]),plugin=float(plug[j]),joint=float(joint[j]),near_pairs=int(near.sum()//2),retained_near_pairs=int(restricted.sum()//2)))
    if k%100==0:print('fold',fold,'region',k,'seconds',round(time.monotonic()-start,1),flush=True)
  print('fold done',fold,round(time.monotonic()-start,1),flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'region_statistics.csv.gz',index=False)
 patient=df.groupby(['variant','case','marker'])[['truth','plugin','joint']].mean().reset_index();patient.to_csv(out/'patient_statistics.csv',index=False)
 summary={};rng=np.random.default_rng(20261104)
 for variant,g in patient.groupby('variant'):
  T=g.pivot(index='case',columns='marker',values='truth');P=g.pivot(index='case',columns='marker',values='plugin').loc[T.index,T.columns];J=g.pivot(index='case',columns='marker',values='joint').loc[T.index,T.columns]
  assert T.shape==(40,16) and np.isfinite(T.to_numpy()).all()
  errors=(J-T).to_numpy()**2;base=(P-T).to_numpy()**2
  metric=lambda idx:float(np.mean(errors[idx].sum(0)/np.maximum(base[idx].sum(0),1e-12)))
  boot=[metric(rng.choice(40,40,replace=True)) for _ in range(2000)]
  means=T.mean(0);valid=means.abs()>.02
  bias=lambda pred:float(np.median(((pred.mean(0)-means)/means)[valid]))
  summary[variant]={'mse_ratio':metric(np.arange(40)),'patient_bootstrap_95_ci':np.percentile(boot,[2.5,97.5]).tolist(),'plugin_median_relative_bias':bias(P),'joint_median_relative_bias':bias(J),'bias_eligible_markers':int(valid.sum()),'patients':40,'markers':16}
 geometry=df[df.variant=='all_pairs'].drop_duplicates('crop');summary['geometry']={'regions':len(geometry),'original_undirected_near_pairs':int(geometry.near_pairs.sum()),'retained_undirected_near_pairs':int(geometry.retained_near_pairs.sum()),'excluded_fraction':float(1-geometry.retained_near_pairs.sum()/geometry.near_pairs.sum())}
 summary['mc_expectation_check']=mc_check;summary['checkpoints']=checkpoints;summary['seconds']=time.monotonic()-start;summary['source_sha256']=sha(Path(__file__));summary['bin_masks']='Original saved/collated bin membership; not recomputed from float32 network distances.';summary['plan_sha256']=sha(out/'segmentation_plan.json')
 (out/'segmentation_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['checkpoints']},indent=2),flush=True)
if __name__=='__main__':main()
