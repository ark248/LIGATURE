"""Synthetic distributional control check; no study data or performance outputs."""
from common import *
from marginal_control import independent_marginal_samples

def main():
 torch.set_num_threads(2);t,se,_=load_modules();torch.manual_seed(4);net=t.Ligature(1,d=16,layers=1,heads=4,frozen_mean=True);B,N,P,K=1,3,16,32768
 mu=torch.zeros(B,N,P);L=torch.zeros(B,N,P,net.K);L[:,:,:,0]=1
 pos=torch.zeros(B,N,2);mask=torch.ones(B,N,dtype=torch.bool)
 with torch.no_grad():
  joint=net.sample(mu,L,net.noise(K,pos,mask,torch.zeros(B,dtype=torch.long),1,gen=torch.Generator().manual_seed(5)))[:,0,:,0]
  indep=independent_marginal_samples(net,mu,L,K,torch.Generator().manual_seed(6))[:,0,:,0]
 assert torch.equal(joint[:,0],joint[:,1]) # coincident-cell field, same shared draw
 cov=torch.cov(indep.T);assert torch.max(torch.abs(cov-torch.diag(cov.diag())))<.025
 for q in [.01,.1,.5,.9,.99]:assert torch.max(torch.abs(torch.quantile(joint,q,dim=0)-torch.quantile(indep,q,dim=0)))<.14
 assert torch.max(torch.abs(indep.var(0)-1))<.04
 # Markers with identical loadings keep their exact within-cell dependence.
 sample=independent_marginal_samples(net,mu,L,16,torch.Generator().manual_seed(8));assert torch.equal(sample[...,0],sample[...,1])
 print('PASS synthetic exact-marginal control: within-cell marker dependence preserved; cell dependence removed; marginal quantiles/variance match the original finite-RFF sampler')
if __name__=='__main__':main()
