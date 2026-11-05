"""Independent cells with exactly the fitted finite-RFF within-cell marginal law."""
import math,torch

def independent_marginal_samples(net,mu,L,samples,gen):
 """Preserve each cell's complete marker-vector marginal; remove cell dependence.

For a fixed cell, uniform RFF phases yield a scalar variance mixture shared by
that field's rank coordinates. Sampling that mixture independently per cell is
exactly the original within-cell marginal, including finite-feature non-Gaussianity.
Region and patient latent marginals are likewise sampled independently per cell.
"""
 B,N,_=mu.shape;parts=[]
 for ell in net.scales:
  phase=torch.rand(samples,B,N,net.nfeat,generator=gen)*2*math.pi
  variance=(2/net.nfeat)*phase.cos().square().sum(-1)
  parts.append(torch.randn(samples,B,N,net.rank,generator=gen)*variance.sqrt()[...,None])
 parts.append(torch.randn(samples,B,N,net.dims['cell'],generator=gen))
 for name in ['region','patient']:parts.append(net.latent((samples,B,N,net.dims[name]),gen))
 return net.sample(mu,L,torch.cat(parts,-1))
