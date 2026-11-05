"""Frozen direct-moment reuse and analytic joint moments, no new-query fitting."""
from common import *
from worker import save_predictions
from analytic_moments import expected_statistics
from marginal_control import independent_marginal_samples
import time

def moment_queries(f,near,far,mean,gn,gq,sigma,vbar,degmean_cov,degree_var,centvar,ym,ys):
 n=len(f);W=near.sum().clamp_min(1);deg=near.sum(-1);WF=far.sum().clamp_min(1);df=far.sum(-1);first=(deg[:,None]*f).sum(0);firstfar=(df[:,None]*f).sum(0)
 numerator=W*gn-2*mean*first+mean.square()*W-2*degmean_cov+vbar*W
 moran=n/W*numerator/centvar.clamp_min(1e-8)
 geary=(n-1)/W*((deg[:,None]*f.square()).sum(0)+degree_var-W*gn)/centvar.clamp_min(1e-8)
 bound=n*deg.max()/W;g_bound=2*(n-1)*deg.max()/W
 moran=moran.clamp(-bound,bound);geary=geary.clamp(0,g_bound);moran=torch.where(centvar>1e-8,moran,0.);geary=torch.where(centvar>1e-8,geary,0.)
 sd=torch.as_tensor(ys,dtype=f.dtype);mu=torch.as_tensor(ym,dtype=f.dtype);raw=sd.square()*gq+2*sd*mu*(first/W-firstfar/WF)
 return torch.cat([moran,geary,raw]).numpy()

def unit(fold,seed):
 out=RUN/f'fold{fold}'/f'reuse_{seed}';out.mkdir(exist_ok=True)
 if (out/'complete.json').exists() and (out/'analytic_scored/test.npz').exists() and (out/'marginal_independent/test.npz').exists():return
 C=read_cache(fold);t,se,dg=C['t'],C['se'],C['dg'];T=C['targets'];SP=T['scored_plugin'];ym=T['target_mean'];ys=T['target_sd'];folder=RUN/f'fold{fold}';start=time.monotonic()
 state=torch.load(folder/f'dscore_{seed}/model.pt',weights_only=True);net=dg.DirectGraph(C['x'].shape[1]+16,832,d=128,layers=3,dropout=.1);net.load_state_dict(state['state']);net.eval();scale=state['scale']
 path=BASE/f'fold{fold}/generator.pt' if seed==271027 else folder/f'ligature_{seed}/model.pt'
 joint=t.Ligature(C['x'].shape[1],d=128,layers=3,frozen_mean=True,dropout=.1);joint.load_state_dict(torch.load(path,weights_only=True));joint.eval()
 sigma=[]
 for case in sorted(p for p,v in C['roles'].items() if v=='selection'):
  ii=np.concatenate([c['ii'] for c in C['crops'] if c['case']==case]);sigma.append(np.mean((C['y'][ii]-C['f'][ii])**2,0))
 sigma=torch.from_numpy(np.mean(sigma,0));pred=[c for c in C['crops'] if c['role']!='train' and c['eligible']];ia,ib=np.triu_indices(16);diag=np.flatnonzero(ia==ib);outputs=[];joint_outputs=[];analytic=[];independent=[];igen=torch.Generator().manual_seed(seed+777)
 with torch.no_grad():
  for c in pred:
   X,Y,D,A,M,NP=se.collate([c],C['x'],C['y']);f=torch.from_numpy(np.array(C['f'][c['ii']]))[None];n=f.shape[1];near,far=A[0,0],A[0,2];W=near.sum();deg=near.sum(-1)
   mu,s=net(torch.cat([X,f],-1),D,M);mu=mu[0]*scale+torch.from_numpy(SP[c['index']]);s=s[0]*scale
   fm=mu[:16];shifted=f[0]+(fm-f[0].mean(0));vb=s[:16].square().minimum(sigma).clamp_min(0);cv=((shifted-fm).square()).sum(0)+n*(sigma-vb)
   outputs.append(moment_queries(shifted,near,far,fm,mu[16+diag],mu[424+diag],sigma,vb,W*vb,W*sigma,cv,ym,ys))
   m,L=joint.encode(X,D,M,f);independent.append(query_stats(independent_marginal_samples(joint,m,L,64,igen),A,M,ym,ys)[:,0].numpy());analytic.append(expected_statistics(joint,m,L,D,A,NP,M)[0].numpy());L=L[0];cov=torch.zeros((16,n,n));offset=0
   for ell in joint.scales:
    load=L[:,:,offset:offset+joint.rank].permute(1,0,2);cov+=torch.bmm(load,load.transpose(1,2))*torch.exp(-D[0].square()/(2*ell*ell))[None];offset+=joint.rank
   nc=joint.dims['cell'];cellvar=L[:,:,offset:offset+nc].square().sum(-1).T;cov+=torch.diag_embed(cellvar);offset+=nc
   for size in [joint.dims['region'],joint.dims['patient']]:
    load=L[:,:,offset:offset+size].permute(1,0,2);cov+=torch.bmm(load,load.transpose(1,2));offset+=size
   var=cov.diagonal(dim1=-2,dim2=-1).T;ones=cov.sum(-1);vb=cov.sum((-1,-2))/(n*n);dmc=(ones*deg[None]).sum(-1)/n;cv=(f[0]-f[0].mean(0)).square().sum(0)+var.sum(0)-n*vb
   gn=((f[0]*(near@f[0])).sum(0)+(cov*near[None]).sum((-1,-2)))/W;gf=((f[0]*(far@f[0])).sum(0)+(cov*far[None]).sum((-1,-2)))/far.sum();dv=(var*deg[:,None]).sum(0)
   joint_outputs.append(moment_queries(f[0],near,far,f[0].mean(0),gn,gn-gf,var.mean(0),vb,dmc,dv,cv,ym,ys))
 for name,value in [('direct_moments',outputs),('joint_moments',joint_outputs),('analytic_scored',analytic)]:
  dest=out/name;dest.mkdir(exist_ok=True);save_predictions(dest,C,pred,np.stack(value))
 dest=out/'marginal_independent';dest.mkdir(exist_ok=True);draw=np.stack(independent);save_predictions(dest,C,pred,draw.mean(1),draws=draw)
 jsonwrite(out/'complete.json',dict(fold=fold,seed=seed,seconds=time.monotonic()-start,direct_state_sha256=sha(folder/f'dscore_{seed}/model.pt'),joint_state_sha256=sha(path),cell_variance_selection_only=sigma.tolist(),new_query_labels_used=False,variance_of_mean_projection='[0, selection-cell residual variance]',ratio_projection='Moran +/-n*dmax/W; Geary [0,2(n-1)*dmax/W]',scope='Ratio of conditional moments, not conditional expectation of ratio'))
 print('frozen-moment reuse complete',fold,seed,flush=True)

def main():
 torch.set_num_threads(2)
 units=[(s,f) for s in [271027,272027,273027] for f in range(5)]
 for seed,fold in units:
  while not all((RUN/f'fold{fold}'/f'{k}_{seed}'/'complete.json').exists() for k in ['dscore','ligature']):time.sleep(20)
  unit(fold,seed)
 jsonwrite(RUN/'reuse_complete.json',dict(units=len(units),new_query_training=False))
if __name__=='__main__':main()
