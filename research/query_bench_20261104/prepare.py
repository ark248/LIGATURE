"""Freeze sources and cache training-only preprocessing; original files read-only."""
from common import *
import shutil,time

def main():
 torch.set_num_threads(2);RUN.mkdir(parents=True,exist_ok=True);(RUN/'src').mkdir(exist_ok=True)
 for name,source in [(n,BASE/'src'/n) for n in ['ligature.py','spatial_engression.py','eval_unseen_statistics.py','analytic_moments.py']]+[('direct_graph.py',PROJECT/'release/ligature/src/direct_graph.py')]:
  dst=RUN/'src'/name
  if dst.exists():assert dst.read_bytes()==source.read_bytes()
  else:shutil.copy2(source,dst)
 jsonwrite(RUN/'source_manifest.json',{p.name:sha(p) for p in (RUN/'src').glob('*.py')})
 shutil.copy2(Path(__file__).with_name('PROTOCOL.md'),RUN/'PROTOCOL.md')
 t,se,_=load_modules();original_roles=se.roles_for
 for fold in range(5):
  out=RUN/f'fold{fold}';out.mkdir(exist_ok=True)
  if (out/'cache_complete.json').exists():print('existing cache',fold,flush=True);continue
  start=time.monotonic();roles=json.loads((BASE/f'fold{fold}/roles.json').read_text())
  fit_roles={p:'train' if r=='train' else 'val' if r=='selection' else 'test' for p,r in roles.items()}
  se.roles_for=lambda cases,f,s:fit_roles
  cells,x,y,crops,_=se.load(DATA,fold,271027,128)
  # Arrays are owned by this benchmark and never modify source data.
  np.save(out/'x.npy',x);np.save(out/'y.npy',y)
  if not (RUN/'geometry.npz').exists():
   lengths=np.array([len(c['ii']) for c in crops]);sq=lengths**2
   np.savez_compressed(RUN/'geometry.npz',ii=np.concatenate([c['ii'] for c in crops]),offsets=np.r_[0,np.cumsum(lengths)],matrix_offsets=np.r_[0,np.cumsum(sq)],dist=np.concatenate([c['dist'].ravel() for c in crops]),A=np.concatenate([c['A'].reshape(3,-1) for c in crops],axis=1).astype(np.uint8),npairs=np.stack([c['npairs'] for c in crops]))
   jsonwrite(RUN/'geometry.json',[{k:c[k] for k in ['crop','case','eligible']} for c in crops]);np.save(RUN/'positions.npy',cells[['gx','gy']].to_numpy()*.325)
  else:assert [c['crop'] for c in crops]==[c['crop'] for c in json.loads((RUN/'geometry.json').read_text())]
  for j,c in enumerate(crops):c['role']=roles[c['case']];c['index']=j
  mean=se.CellGraphNet(x.shape[1],d=128,layers=3);mean.load_state_dict(torch.load(BASE/f'fold{fold}/mean.pt',weights_only=True));mean.eval()
  fp=se.predict_mean(mean,crops,x,y);f=np.zeros_like(y)
  tr=np.array([roles[str(p)]=='train' for p in cells.case]);raw=np.log1p(cells[['y_'+m for m in se.MARKERS]].to_numpy(np.float64));ym,ys=raw[tr].mean(0),raw[tr].std(0)+1e-6
  assert np.allclose(y,(raw-ym)/ys,atol=2e-6)
  S=[];SP=[];Q=[];QP=[];cheap=[];constants=0
  with torch.no_grad():
   for k,c in enumerate(crops):
    ii=c['ii'];f[ii]=fp[c['crop']];X,Y,D,A,M,NP=se.collate([c],x,y);F=torch.from_numpy(fp[c['crop']])[None]
    S.append(t.contrasts(se.crop_stats(Y,A,NP,M.float()))[0].numpy());SP.append(t.contrasts(se.crop_stats(F,A,NP,M.float()))[0].numpy())
    if c['eligible']:
     Q.append(query_stats(Y[None],A,M,ym,ys)[0,0].numpy());QP.append(query_stats(F[None],A,M,ym,ys)[0,0].numpy())
    else:Q.append(np.zeros(48,np.float32));QP.append(np.zeros(48,np.float32))
    constants+=int((Y[0].var(0)<1e-8).sum())
    # Frozen stain encoder context plus image moments and plug-in queries.
    h=mean.inp(X);rbf=torch.exp(-((D[...,None]-mean.centers)/10.)**2)
    for block in mean.blocks:h=block(h,rbf,M,1)
    h=mean.out_norm(h)[0]
    cheap.append(np.r_[h.mean(0).numpy(),h.max(0).values.numpy(),x[ii].mean(0),x[ii].std(0),np.log(len(ii)),QP[-1]])
    if k%500==0:print('cache fold',fold,'crop',k,'seconds',round(time.monotonic()-start,1),flush=True)
  np.save(out/'stain.npy',f);np.savez_compressed(out/'targets.npz',scored=np.stack(S),scored_plugin=np.stack(SP),queries=np.stack(Q),queries_plugin=np.stack(QP),cheap_features=np.stack(cheap),target_mean=ym,target_sd=ys)
  jsonwrite(out/'roles.json',roles);jsonwrite(out/'cache_complete.json',dict(seconds=time.monotonic()-start,constant_region_markers=constants,training_patients=sorted(p for p,v in roles.items() if v=='train'),normalization='training patients only',source_mean_sha256=sha(BASE/f'fold{fold}/mean.pt')))
  print('cache complete fold',fold,'seconds',round(time.monotonic()-start,1),flush=True)
 se.roles_for=original_roles
if __name__=='__main__':main()
