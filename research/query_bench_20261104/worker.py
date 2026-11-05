"""Owned, restartable benchmark work units; no writes to source runs."""
from common import *
import argparse,time,math
from types import SimpleNamespace

def aggregate(crops,values):
 cases=sorted({c['case'] for c in crops})
 return np.array(cases),np.stack([values[[i for i,c in enumerate(crops) if c['case']==p]].mean(0) for p in cases])

def save_predictions(out,cache,crops,mu,sd=None,tau=None,draws=None,independent=None,scored=None):
 targets=cache['targets'];idx=np.array([c['index'] for c in crops]);kind='scored' if mu.shape[-1]==832 else 'queries'
 for role in ['selection','calibration','test']:
  jj=[i for i,c in enumerate(crops) if c['role']==role];sub=[crops[j] for j in jj]
  case,Y=aggregate(sub,targets[kind][idx[jj]]);_,P=aggregate(sub,targets[kind+'_plugin'][idx[jj]])
  _,center=aggregate(sub,mu[jj]);result=dict(case=case,truth=Y,plugin=P,center=center,n_regions=np.array([sum(c['case']==p for c in sub) for p in case]))
  if sd is not None:
   _,v=aggregate(sub,sd[jj]**2);result['sd']=np.sqrt(v/result['n_regions'][:,None]+tau**2)
  if draws is not None:result['draws']=aggregate(sub,draws[jj])[1]
  if independent is not None:result['independent_draws']=aggregate(sub,independent[jj])[1]
  if scored is not None:result['scored_draws']=aggregate(sub,scored[jj])[1]
  np.savez_compressed(out/(role+'.npz'),**result)

def direct(cache,out,seed,kind,epochs=15):
 t,se,dg=cache['t'],cache['se'],cache['dg'];crops=cache['crops'];T=cache['targets']
 key='scored' if kind=='dscore' else 'queries';target=T[key];plugin=T[key+'_plugin']
 tr=[c for c in crops if c['role']=='train'];va=[c for c in crops if c['role']=='selection'];pred=[c for c in crops if c['role']!='train' and c['eligible']]
 if kind=='dquery':
  # These query targets require the same valid near/far support in every role.
  tr=[c for c in tr if c['eligible']];va=[c for c in va if c['eligible']]
 scale=torch.from_numpy(target[[c['index'] for c in tr]]).std(0).clamp_min(1e-3)
 x=np.column_stack([cache['x'],cache['f']]).astype(np.float32);y=cache['y']
 torch.manual_seed(seed+7);rng=np.random.default_rng(seed+8)
 net=dg.DirectGraph(x.shape[1],scale.numel(),d=128,layers=3,dropout=.1)
 opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
 nb=sum(1 for _ in se.group_batches(tr,np.random.default_rng(0),4,4))
 sched=torch.optim.lr_scheduler.OneCycleLR(opt,.001,total_steps=epochs*nb+10,pct_start=.1)
 best=float('inf');hist=[];fitstart=time.monotonic()
 def loss(batch,gid,ng):
  X,Y,D,A,M,NP=se.collate(batch,x,y);ix=[c['index'] for c in batch]
  Y0=torch.from_numpy(target[ix])/scale;mu,sd=net(X,D,M);mu=mu+torch.from_numpy(plugin[ix])/scale
  reg=t.crps_gauss(mu,sd,Y0).mean();w=torch.nn.functional.one_hot(gid,ng).float();w=w/w.sum(0,keepdim=True)
  mp=torch.einsum('bd,bg->gd',mu,w);sp=torch.sqrt(torch.einsum('bd,bg->gd',sd**2,w**2)+torch.exp(2*net.log_tau))
  return reg+t.crps_gauss(mp,sp,torch.einsum('bd,bg->gd',Y0,w)).mean()
 for epoch in range(epochs):
  net.train();tot=0.;n=0
  for gb in se.group_batches(tr,rng,4,4):
   b=[c for g in gb for c in g];gid=torch.as_tensor(np.repeat(np.arange(len(gb)),[len(g) for g in gb]))
   L=loss(b,gid,len(gb));opt.zero_grad();L.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();sched.step();tot+=L.item();n+=1
  net.eval();v=[]
  with torch.no_grad():
   for gb in se.group_batches(va,np.random.default_rng(11),4,4):
    b=[c for g in gb for c in g];gid=torch.as_tensor(np.repeat(np.arange(len(gb)),[len(g) for g in gb]));v.append(loss(b,gid,len(gb)).item())
  val=float(np.mean(v));hist.append(dict(epoch=epoch,train=tot/n,selection=val,seconds=time.monotonic()-fitstart));print(kind,seed,'epoch',epoch,'selection',val,'seconds',round(time.monotonic()-fitstart,1),flush=True)
  if val<best:best=val;best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
  jsonwrite(out/'history.json',hist)
 net.load_state_dict(best_state);torch.save(dict(state=best_state,scale=scale,seed=seed,kind=kind),out/'model.pt');net.eval()
 fitseconds=time.monotonic()-fitstart;start=time.monotonic();means=[];sds=[]
 with torch.no_grad():
  for c in pred:
   X,Y,D,A,M,NP=se.collate([c],x,y);mu,sd=net(X,D,M);means.append((mu[0]*scale).numpy()+plugin[c['index']]);sds.append((sd[0]*scale).numpy())
 save_predictions(out,cache,pred,np.stack(means),np.stack(sds),(torch.exp(net.log_tau).detach()*scale).numpy())
 return dict(fit_seconds=fitseconds,inference_seconds=time.monotonic()-start,best_selection=best,parameters=sum(p.numel() for p in net.parameters()))

def generator(cache,out,seed,fold):
 t,se=cache['t'],cache['se'];args=SimpleNamespace(**json.loads((BASE/f'fold{fold}/settings.json').read_text()));args.model_seed=seed
 x,y,crops=cache['x'],cache['y'],cache['crops'];xy=cache['cells_xy']
 def positions(batch,unused):
  a=np.zeros((len(batch),max(len(c['ii']) for c in batch),2),np.float32)
  for j,c in enumerate(batch):p=xy[c['ii']];a[j,:len(p)]=p-p.mean(0)
  return torch.from_numpy(a)
 t.positions=positions
 mean=se.CellGraphNet(x.shape[1],d=128,layers=3);mean.load_state_dict(torch.load(BASE/f'fold{fold}/mean.pt',weights_only=True));mean.eval()
 tr=[c for c in crops if c['role']=='train'];va=[c for c in crops if c['role']=='selection'];fitstart=time.monotonic()
 if seed==271027:statefile=BASE/f'fold{fold}/generator.pt';fitseconds=0.;source='reused independent fit'
 else:
  scale=torch.load(BASE/f'fold{fold}/training_stat_scale.pt',weights_only=True)
  net,hist=t.train(tr,va,x,y,None,args,scale,mean_net=mean);torch.save(net.state_dict(),out/'uncalibrated.pt');jsonwrite(out/'history.json',hist)
  net=t.calibrate_scales(net,va,x,y,None,args,scale,mean_net=mean);torch.save(net.state_dict(),out/'model.pt');statefile=out/'model.pt';fitseconds=time.monotonic()-fitstart;source='new independent seed'
 net=t.Ligature(x.shape[1],d=128,layers=3,frozen_mean=True,dropout=.1);net.load_state_dict(torch.load(statefile,weights_only=True));net.eval()
 pred=[c for c in crops if c['role']!='train' and c['eligible']];K=64;gen=torch.Generator().manual_seed(seed+99);igen=torch.Generator().manual_seed(seed+199)
 eta={case:net.latent((K,1,net.dims['patient']),gen) for case in sorted({c['case'] for c in pred})}
 draws=[];independent=[];scored=[];T=cache['targets'];start=time.monotonic()
 with torch.no_grad():
  for j,c in enumerate(pred):
   X,Y,D,A,M,NP=se.collate([c],x,y);f=torch.from_numpy(np.array(cache['f'][c['ii']]))[None];mu,L=net.encode(X,D,M,f)
   z=net.noise(K,positions([c],None),M,torch.zeros(1,dtype=torch.long),1,gen=gen,eta=eta[c['case']]);smp=net.sample(mu,L,z)
   independent_smp=mu[None]+L.square().sum(-1).sqrt()[None]*torch.randn(smp.shape,generator=igen)
   draws.append(query_stats(smp,A,M,T['target_mean'],T['target_sd'])[:,0].numpy());independent.append(query_stats(independent_smp,A,M,T['target_mean'],T['target_sd'])[:,0].numpy())
   scored.append(t.contrasts(se.crop_stats(smp,A,NP,M.float()))[:,0].numpy())
   if j%250==0:print('generator inference',fold,seed,j,'seconds',round(time.monotonic()-start,1),flush=True)
 draws=np.stack(draws);save_predictions(out,cache,pred,draws.mean(1),draws=draws,independent=np.stack(independent),scored=np.stack(scored))
 return dict(fit_seconds=fitseconds,inference_seconds=time.monotonic()-start,source=source,state_sha256=sha(statefile),samples=K)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--kind',choices=['dscore','dquery','ligature'],required=True);ap.add_argument('--threads',type=int,default=8);ap.add_argument('--epochs',type=int,default=15);a=ap.parse_args()
 torch.set_num_threads(a.threads);out=RUN/f'fold{a.fold}'/f'{a.kind}_{a.seed}';out.mkdir(exist_ok=True)
 if (out/'complete.json').exists():print('already complete',out);return
 start=time.monotonic();cache=read_cache(a.fold)
 result=generator(cache,out,a.seed,a.fold) if a.kind=='ligature' else direct(cache,out,a.seed,a.kind,a.epochs)
 jsonwrite(out/'complete.json',dict(**vars(a),**result,total_seconds=time.monotonic()-start,protocol_sha256=sha(RUN/'PROTOCOL.md')))
 print('complete',out,flush=True)
if __name__=='__main__':main()
