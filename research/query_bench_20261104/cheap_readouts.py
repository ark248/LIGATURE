"""Low-cost query adaptation; all tuning uses training/selection roles only."""
from common import *
from worker import aggregate,save_predictions
import time

def main():
 torch.set_num_threads(2)
 for fold in range(5):
  C=read_cache(fold);T=C['targets'];crops=C['crops'];out=RUN/f'fold{fold}'/'cheap';out.mkdir(exist_ok=True)
  if (out/'complete.json').exists():continue
  start=time.monotonic();tr=[c for c in crops if c['role']=='train' and c['eligible']];sel=[c for c in crops if c['role']=='selection' and c['eligible']];pred=[c for c in crops if c['role']!='train' and c['eligible']]
  ti=[c['index'] for c in tr];vi=[c['index'] for c in sel];pi=[c['index'] for c in pred];X=T['cheap_features'].astype(np.float64);Y=T['queries'].astype(np.float64);P=T['queries_plugin'].astype(np.float64)
  xm=X[ti].mean(0);xs=X[ti].std(0).clip(1e-4);Z=(X-xm)/xs;R=Y-P;ym=R[ti].mean(0);ys=R[ti].std(0).clip(1e-4);r=(R-ym)/ys
  # Fixed grid, one penalty selected jointly for all 48 outputs.
  eig,V=np.linalg.eigh(Z[ti].T@Z[ti]/len(ti));rhs=V.T@(Z[ti].T@r[ti]/len(ti));metrics=[];models=[]
  # Query metric normalization uses training-patient target variation only.
  scale=aggregate(tr,Y[ti])[1].std(0,ddof=1).clip(1e-4)
  vtruth=aggregate(sel,Y[vi])[1]
  for penalty in np.logspace(-4,4,9):
   w=V@(rhs/(eig[:,None]+penalty));mu=P[vi]+ym+(Z[vi]@w)*ys;vp=aggregate(sel,mu)[1];score=float(np.mean(((vp-vtruth)/scale)**2));metrics.append(dict(penalty=float(penalty),selection_error=score));models.append(w)
  best=int(np.argmin([m['selection_error'] for m in metrics]));mu=P[pi]+ym+(Z[pi]@models[best])*ys
  dest=out/'ridge';dest.mkdir(exist_ok=True);save_predictions(dest,C,pred,mu.astype(np.float32));np.savez_compressed(dest/'model.npz',xmean=xm,xsd=xs,ymean=ym,ysd=ys,coef=models[best]);jsonwrite(dest/'selection.json',dict(grid=metrics,selected_index=best))
  # Validation-only affine correction, fitted per query over selection regions.
  pm=P[vi].mean(0);slope=((P[vi]-pm)*(Y[vi]-Y[vi].mean(0))).sum(0)/((P[vi]-pm)**2).sum(0).clip(1e-8);intercept=Y[vi].mean(0)-slope*pm
  dest=out/'affine';dest.mkdir(exist_ok=True);save_predictions(dest,C,pred,(P[pi]*slope+intercept).astype(np.float32));jsonwrite(dest/'model.json',dict(slope=slope.tolist(),intercept=intercept.tolist()))
  jsonwrite(out/'complete.json',dict(seconds=time.monotonic()-start,features=X.shape[1],training_regions=len(ti),selection_patients=sorted({c['case'] for c in sel}),chosen_penalty=metrics[best]['penalty'],test_labels_used=False));print('cheap heads complete fold',fold,flush=True)
if __name__=='__main__':main()
