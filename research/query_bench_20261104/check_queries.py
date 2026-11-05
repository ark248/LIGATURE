from common import *

def main():
 torch.manual_seed(55);n=9;p=16
 y=torch.randn(3,1,n,p,dtype=torch.float64);D=torch.cdist(torch.arange(n,dtype=torch.float64)[:,None],torch.arange(n,dtype=torch.float64)[:,None]);near=((D>0)&(D<=2)).double();far=(D>=5).double();A=torch.stack([near,near,far])[None];M=torch.ones(1,n,dtype=torch.bool);mu=np.arange(p)/7;sd=np.arange(p)/9+1
 out=query_stats(y,A,M,mu,sd).numpy()[:,0]
 for k in range(3):
  for m in range(p):
   z=y[k,0,:,m].numpy();w=near.numpy();v=z-z.mean();mor=n/w.sum()*(w*v[:,None]*v[None,:]).sum()/(v*v).sum();gea=(n-1)/(2*w.sum())*(w*(z[:,None]-z[None,:])**2).sum()/(v*v).sum();raw=z*sd[m]+mu[m];contrast=(w*raw[:,None]*raw[None,:]).sum()/w.sum()-(far.numpy()*raw[:,None]*raw[None,:]).sum()/far.sum().item()
   np.testing.assert_allclose(out[k,[m,p+m,2*p+m]],[mor,gea,contrast],rtol=1e-10,atol=1e-10)
 affine=query_stats(3*y+8,A,M,mu,sd).numpy()[:,:,:32];np.testing.assert_allclose(affine,out[None,:,:32].transpose(1,0,2),atol=1e-10)
 # Padding must not change any query.
 yp=torch.nn.functional.pad(y,(0,0,0,4));ap=torch.nn.functional.pad(A,(0,4,0,4));mp=torch.nn.functional.pad(M,(0,4));np.testing.assert_allclose(query_stats(yp,ap,mp,mu,sd).numpy()[:,0],out,atol=1e-10)
 assert torch.isfinite(query_stats(torch.ones_like(y),A,M,mu,sd)).all()
 # Sparse marker offsets exposed cancellation in the uncentered float32 formula.
 v=(-1+torch.randn(3,1,n,p,dtype=torch.float32)*1e-4)
 q32=query_stats(v,A.float(),M,mu,sd).double().numpy();q64=query_stats(v.double(),A,M,mu,sd).numpy()
 np.testing.assert_allclose(q32[:,:,:32],q64[:,:,:32],atol=.003,rtol=.003)
 try:query_stats(y,torch.zeros_like(A),M,mu,sd)
 except ValueError:pass
 else:raise AssertionError('Empty query graphs must be rejected, not clipped into a fake ratio')
 cache=read_cache(0);T=cache['targets'];crops=cache['crops'];Q=T['scored'][:,424:560];Qp=T['scored_plugin'][:,424:560]
 # Only training and selection used in the normalization/provenance verification.
 for role,oldname in [('train','training'),('selection','selection')]:
  old=np.load(BASE/f'fold0/{oldname}_patients.npz');sub=[c for c in crops if c['eligible'] and c['role']==role]
  for i,case in enumerate(old['case']):
   jj=[c['index'] for c in sub if c['case']==case];np.testing.assert_allclose(Q[jj].mean(0),old['truth'][i],atol=3e-6)
 print('PASS: brute-force Moran/Geary/log-assay contrast; affine invariance; padding; finite constants; original training/selection targets reproduced')
if __name__=='__main__':main()
