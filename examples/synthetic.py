"""Exercise joint sampling, analytic moments and scoring without downloading data."""
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'release/ligature/src'))
import ligature as model
from analytic_moments import expected_statistics

def main():
    torch.set_num_threads(2)
    torch.manual_seed(33)
    batch,cells,features,markers=4,7,12,16
    net=model.Ligature(features,d=32,layers=1,heads=4,frozen_mean=True)
    image=torch.randn(batch,cells,features)
    mask=torch.ones(batch,cells,dtype=torch.bool)
    positions=torch.stack([torch.arange(cells)*14.,torch.zeros(cells)],1)[None].expand(batch,-1,-1)
    distances=torch.cdist(positions,positions)
    adjacency=torch.stack([(distances>lo)&(distances<=hi) for lo,hi in model.se.BINS],1).float()
    pair_counts=adjacency.sum((-1,-2))
    measurements=torch.randn(batch,cells,markers)
    mean,loadings=net.encode(image,distances,mask,torch.randn_like(measurements))
    noise=net.noise(16,positions,mask,torch.tensor([0,0,1,1]),2,gen=torch.Generator().manual_seed(9))
    assert torch.equal(noise[:,0,:,-4:],noise[:,1,:,-4:])
    assert not torch.equal(noise[:,0,:,-4:],noise[:,2,:,-4:])
    samples=net.sample(mean,loadings,noise)
    statistics=model.contrasts(model.se.crop_stats(samples,adjacency,pair_counts,mask.float()))
    target=model.contrasts(model.se.crop_stats(measurements[None],adjacency,pair_counts,mask.float())[0])
    expected=expected_statistics(net,mean,loadings,distances,adjacency,pair_counts,mask)
    assert statistics.shape==(16,batch,832) and expected.shape==(batch,832)
    loss=model.crps_samples(statistics,target)+.01*expected.square().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    print('PASS: joint draws, shared patient latents, 832 analytic moments and finite CRPS gradients.')
    print('Synthetic execution check; this untrained model is not an empirical research result.')

if __name__=='__main__':main()
