"""Same-allocation cost profile: full 48-query direct/cheap heads versus reuse."""
from common import *
import time,platform,socket

def main():
 torch.set_num_threads(8);C=read_cache(0);t,se,dg=C['t'],C['se'],C['dg'];T=C['targets'];cheap_features=T['cheap_features'];fd=RUN/'fold0';pred=[]
 for p in sorted(p for p,v in C['roles'].items() if v=='test'):
  pred.extend(sorted([c for c in C['crops'] if c['case']==p and c['eligible']],key=lambda c:c['crop'])[:3])
 net=t.Ligature(C['x'].shape[1],d=128,layers=3,frozen_mean=True,dropout=.1);net.load_state_dict(torch.load(BASE/'fold0/generator.pt',weights_only=True));net.eval()
 z=torch.load(fd/'dquery_271027/model.pt',weights_only=True);direct=dg.DirectGraph(C['x'].shape[1]+16,48,d=128,layers=3,dropout=.1);direct.load_state_dict(z['state']);direct.eval()
 mean=se.CellGraphNet(C['x'].shape[1],d=128,layers=3);mean.load_state_dict(torch.load(BASE/'fold0/mean.pt',weights_only=True));mean.eval()
 ridge=np.load(fd/'cheap/ridge/model.npz');gen=torch.Generator().manual_seed(271099);components={k:[] for k in ['joint_cell_draws_64','cached_draw_query_48','direct_48_with_plugin','frozen_feature_ridge_48','cached_feature_ridge_48']}
 with torch.no_grad():
  for c in pred:
   X,Y,D,A,M,NP=se.collate([c],C['x'],C['y']);F=torch.from_numpy(np.array(C['f'][c['ii']]))[None];p=C['cells_xy'][c['ii']];pos=torch.from_numpy(np.array(p-p.mean(0),np.float32))[None]
   start=time.perf_counter();mu,L=net.encode(X,D,M,F);noise=net.noise(64,pos,M,torch.zeros(1,dtype=torch.long),1,gen=gen);draw=net.sample(mu,L,noise);components['joint_cell_draws_64'].append(time.perf_counter()-start)
   times=[]
   for _ in range(5):
    start=time.perf_counter();q=query_stats(draw,A,M,T['target_mean'],T['target_sd']);times.append(time.perf_counter()-start)
   components['cached_draw_query_48'].append(float(np.median(times)))
   start=time.perf_counter();plugin=query_stats(F[None],A,M,T['target_mean'],T['target_sd'])[0,0];dmu,ds=direct(torch.cat([X,F],-1),D,M);dmu=dmu*z['scale']+plugin;components['direct_48_with_plugin'].append(time.perf_counter()-start)
   start=time.perf_counter();h=mean.inp(X);rbf=torch.exp(-((D[...,None]-mean.centers)/10.)**2)
   for block in mean.blocks:h=block(h,rbf,M,1)
   h=mean.out_norm(h)[0];x=np.array(C['x'][c['ii']]);features=np.r_[h.mean(0).numpy(),h.max(0).values.numpy(),x.mean(0),x.std(0),np.log(len(x)),plugin.numpy()];prediction=plugin.numpy()+ridge['ymean']+((features-ridge['xmean'])/ridge['xsd']@ridge['coef'])*ridge['ysd'];components['frozen_feature_ridge_48'].append(time.perf_counter()-start)
   start=time.perf_counter();pq=query_stats(F[None],A,M,T['target_mean'],T['target_sd'])[0,0].numpy();features=np.r_[cheap_features[c['index'],:-48],pq];prediction=pq+ridge['ymean']+((features-ridge['xmean'])/ridge['xsd']@ridge['coef'])*ridge['ysd'];components['cached_feature_ridge_48'].append(time.perf_counter()-start)
 cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
 jsonwrite(RUN/'timing.json',dict(regions=len(pred),patients=len(set(c['case'] for c in pred)),selection='first three eligible test crops per patient by crop ID',cpu_threads=8,cpu=cpu,hostname=socket.gethostname(),seconds_for_all_regions={k:float(sum(v)) for k,v in components.items()},per_region_seconds=components,scope='Frozen Phikon cell features and stain outputs common to all methods are precomputed. Ridge is profiled both with recomputed context and with query-independent context cached during the common stain forward pass; both include new plug-in-query calculation. Direct includes plug-in-query calculation. Both readouts jointly output all 48 queries, not one fit per query. Query timing uses five repeats on cached joint cell draws. Hardware-specific illustrative timings, not a global throughput ranking.'))
 print('same-allocation timing complete',flush=True)
if __name__=='__main__':main()
