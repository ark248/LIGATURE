from pathlib import Path
import os,sys,json,hashlib
import numpy as np,torch
PROJECT=Path(__file__).resolve().parents[2]
DATA=Path(os.environ.get('LIGATURE_DATA_ROOT',str(PROJECT/'release/ligature'))).resolve()
RUN=DATA/'runs/spatial_queries'
BASE=DATA/'runs/calibration/orion_v1'

def load_modules():
 source=RUN/'src' if (RUN/'src/ligature.py').exists() else PROJECT/'release/ligature/src'
 sys.path.insert(0,str(source))
 import ligature as t
 import direct_graph as dg
 return t,t.se,dg

def query_stats(Y,A,M,target_mean,target_sd):
 """Y [sample,batch,cell,marker]; return Moran, Geary, log-assay contrast."""
 if torch.any(A[:,0].sum((-1,-2))<=0) or torch.any(A[:,2].sum((-1,-2))<=0):
  raise ValueError('Query graphs need nonempty near/far bins; exclude ineligible regions.')
 n=M.sum(-1).clamp_min(1);y=Y*M[None,:,:,None];mean=y.sum(2)/n[None,:,None]
 centered_y=(y-mean[:,:,None])*M[None,:,:,None]
 variance=centered_y.square().sum(2)
 adj=A[:,0];W=adj.sum((-1,-2)).clamp_min(1);deg=adj.sum(-1)
 # Center before quadratic products. Subtracting O(mean^2) terms after summing
 # loses the small numerator for sparse/nearly constant markers in float32.
 ay=torch.einsum('bij,mbjp->mbip',adj,centered_y);cross=(centered_y*ay).sum(2)
 first=(centered_y*deg[None,:,:,None]).sum(2)
 denom=variance.clamp_min(1e-8)
 moran=n[None,:,None]/W[None,:,None]*cross/denom
 square=(centered_y.square()*deg[None,:,:,None]).sum(2)
 geary=(n[None,:,None]-1)/W[None,:,None]*(square-cross)/denom
 moran=torch.where(variance>1e-8,moran,0.);geary=torch.where(variance>1e-8,geary,0.)
 far=A[:,2];WF=far.sum((-1,-2)).clamp_min(1);df=far.sum(-1)
 yf=torch.einsum('bij,mbjp->mbip',far,centered_y);cf=(centered_y*yf).sum(2);ff=(centered_y*df[None,:,:,None]).sum(2)
 sd=torch.as_tensor(target_sd,dtype=y.dtype);mu=torch.as_tensor(target_mean,dtype=y.dtype)
 raw=sd.square()*(cross/W[None,:,None]-cf/WF[None,:,None])+2*sd*(mu+sd*mean)*(first/W[None,:,None]-ff/WF[None,:,None])
 return torch.cat([moran,geary,raw],-1)

def read_cache(fold):
 t,se,dg=load_modules();folder=RUN/f'fold{fold}'
 roles=json.loads((BASE/f'fold{fold}/roles.json').read_text())
 with np.load(RUN/'geometry.npz') as z:geo={k:z[k] for k in z.files}
 meta=json.loads((RUN/'geometry.json').read_text());crops=[]
 for j,c in enumerate(meta):
  lo,hi=geo['offsets'][j:j+2];a,b=geo['matrix_offsets'][j:j+2];n=hi-lo
  crops.append(dict(crop=c['crop'],case=c['case'],role=roles[c['case']],ii=geo['ii'][lo:hi],dist=geo['dist'][a:b].reshape(n,n),A=geo['A'][:,a:b].reshape(3,n,n).astype(np.float32),npairs=geo['npairs'][j],eligible=c['eligible'],index=j))
 return dict(t=t,se=se,dg=dg,folder=folder,roles=roles,crops=crops,x=np.load(folder/'x.npy',mmap_mode='r'),y=np.load(folder/'y.npy',mmap_mode='r'),f=np.load(folder/'stain.npy',mmap_mode='r'),targets=np.load(folder/'targets.npz'),cells_xy=np.load(RUN/'positions.npy',mmap_mode='r'))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonwrite(p,d):Path(p).write_text(json.dumps(d,indent=2,default=str)+'\n')
