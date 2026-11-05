"""Compare the analytic implementation against explicit full cell-marker covariance."""
from common import *
from analytic_moments import expected_statistics
from types import SimpleNamespace

def main():
 torch.set_num_threads(2);torch.manual_seed(71);B,N,P=2,7,16;net=SimpleNamespace(scales=(10.,20.,40.,80.),rank=4,dims=dict(cell=8,region=4,patient=4),tail_df=0)
 mu=torch.randn(B,N,P,dtype=torch.float64);L=torch.randn(B,N,P,32,dtype=torch.float64)*.1;pos=torch.stack([torch.arange(N,dtype=torch.float64)*15,torch.zeros(N)]).T;dist=torch.cdist(pos,pos)[None].repeat(B,1,1);mask=torch.ones(B,N,dtype=torch.bool);mask[1,-2:]=False
 A=torch.stack([((dist>lo)&(dist<=hi)).double() for lo,hi in [(0,25),(25,50),(50,75)]],1);A=A*mask[:,None,:,None]*mask[:,None,None,:];NP=A.sum((-1,-2))
 for df in [0,3]:
  net.tail_df=df;actual=expected_statistics(net,mu,L,dist,A,NP,mask);truth=[];ia,ib=np.triu_indices(P)
  for b in range(B):
   m=int(mask[b].sum());f=mu[b,:m];load=L[b,:m];cov=torch.zeros(m,m,P,P,dtype=torch.float64);off=0
   for ell in net.scales:
    k=torch.exp(-dist[b,:m,:m].square()/(2*ell*ell));cov+=k[:,:,None,None]*torch.einsum('iar,jbr->ijab',load[:,:,off:off+4],load[:,:,off:off+4]);off+=4
   for i in range(m):cov[i,i]+=load[i,:,off:off+8]@load[i,:,off:off+8].T
   off+=8
   for r in [4,4]:cov+=(df/(df-2) if df else 1)*torch.einsum('iar,jbr->ijab',load[:,:,off:off+r],load[:,:,off:off+r]);off+=r
   product=f[:,None,:,None]*f[None,:,None,:]+cov;g=[]
   for k in range(3):
    z=(product*A[b,k,:m,:m,None,None]).sum((0,1))/NP[b,k].clamp_min(1);z=(z+z.T)/2;g.append(z[ia,ib])
   truth.append(torch.cat([f.mean(0),*g,g[0]-g[2],g[0]-g[1],g[1]-g[2]]))
  torch.testing.assert_close(actual,torch.stack(truth),rtol=1e-10,atol=1e-10)
 assert actual.shape==(2,832)
 print('PASS: all 832 analytic coordinates match explicit full covariance; padding and Student-t finite second moments')
if __name__=='__main__':main()
