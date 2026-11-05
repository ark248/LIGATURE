"""Conditional means of scored bilinear statistics, integrated over all latents.

The finite random Fourier sampler has the exact squared-exponential second moment
when frequencies/phases are integrated out. This function removes Monte Carlo
noise from bilinear point forecasts; it does not identify the true tissue law.
"""
import torch

def expected_statistics(net,mu,L,dist,A,npairs,mask):
 """Within-region inputs as in collate/encode; output [batch, P+6*P(P+1)/2].

Includes cell-noise contributions if diagonal adjacency weights are supplied.
Fields are independent between regions. Average the result for patient means;
use joint patient-shared samples for predictive laws and nonlinear queries.
"""
 B,N,P=mu.shape;K=A.shape[1]
 if K!=3:raise ValueError('Scored layout uses three distance bins')
 M=mask.to(mu.dtype);mu=mu*M[:,:,None];L=L*M[:,:,None,None];weights=A.to(mu.dtype)/npairs.clamp_min(1)[:,:,None,None]
 G=torch.einsum('bip,bkiq->bkpq',mu,torch.einsum('bkij,bjp->bkip',weights,mu));offset=0
 def add_component(load,W):
  r=load.shape[-1]
  if not r:return torch.zeros_like(G)
  v=torch.einsum('bkij,bjd->bkid',W,load.reshape(B,N,P*r)).reshape(B,K,N,P,r)
  return torch.einsum('bipr,bkiqr->bkpq',load,v)
 for ell in net.scales:
  load=L[:,:,:,offset:offset+net.rank];offset+=net.rank
  kernel=torch.exp(-dist.square()/(2*float(ell)**2))
  G=G+add_component(load,weights*kernel[:,None])
 nc=net.dims['cell'];local=L[:,:,:,offset:offset+nc];offset+=nc
 if nc:
  local_cov=torch.einsum('bipr,biqr->bipq',local,local)
  G=G+torch.einsum('bki,bipq->bkpq',weights.diagonal(dim1=-2,dim2=-1),local_cov)
 tail_df=getattr(net,'tail_df',0)
 if tail_df and tail_df<=2:raise ValueError('Student-t latent second moments require degrees of freedom >2')
 latent_variance=tail_df/(tail_df-2) if tail_df else 1.
 for name in ['region','patient']:
  r=net.dims[name];load=L[:,:,:,offset:offset+r];offset+=r
  G=G+latent_variance*add_component(load,weights)
 assert offset==L.shape[-1]
 G=(G+G.transpose(-1,-2))/2;ia,ib=torch.triu_indices(P,P,device=mu.device)
 means=mu.sum(1)/M.sum(1).clamp_min(1)[:,None];g=G[:,:,ia,ib]
 return torch.cat([means,g.reshape(B,-1),g[:,0]-g[:,2],g[:,0]-g[:,1],g[:,1]-g[:,2]],-1)
