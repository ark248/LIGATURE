"""Matched, independently calibrated H&E-geometry control; complete folds only."""
from common import *
import importlib.util,pandas as pd

def main():
 root=RUN/'he_control';run=root/'independent';spec=importlib.util.spec_from_file_location('cal_eval',PROJECT/'research/calibration/evaluate.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
 frames=[];ia,ib=np.triu_indices(16);diag=ia==ib
 for fold in range(5):
  fd=run/f'fold{fold}'
  if not (fd/'complete.json').exists() or not (fd/'direct/complete.json').exists():raise RuntimeError(f'Incomplete H&E matched comparison: {fd}')
  frame,_=ev.fold_records(fd);frames.append(frame);cal=np.load(fd/'calibration_patients.npz');test=np.load(fd/'test_patients.npz');train=np.load(fd/'training_patients.npz');dc=np.load(fd/'direct/calibration.npz');dt=np.load(fd/'direct/test.npz')
  for old,direct in [(cal,dc),(test,dt)]:
   assert np.array_equal(old['case'],direct['case']);np.testing.assert_allclose(old['truth'],direct['truth'][:,424:560],atol=3e-6)
  cy,ty=cal['truth'],test['truth'];cm,tm=dc['center'][:,424:560],dt['center'][:,424:560];cs,ts=dc['sd'][:,424:560],dt['sd'][:,424:560];scale=train['truth'].std(0,ddof=1).clip(1e-4);rows=[]
  for mode in ['adaptive','constant','raw']:
   if mode=='raw':lo,hi=tm-1.6448536269514722*ts,tm+1.6448536269514722*ts
   else:
    threshold,k=ev.conformal_threshold(cy,cm,cs if mode=='adaptive' else np.ones_like(cm),.1);half=(ts if mode=='adaptive' else np.ones_like(tm))*threshold;lo,hi=tm-half,tm+half
   width=hi-lo;score=width+20*np.maximum(lo-ty,0)+20*np.maximum(ty-hi,0)
   for i,case in enumerate(test['case']):
    for j in range(136):rows.append(dict(fold=fd.name,case=str(case),query=str(j),diag=bool(diag[j]),method='direct_'+mode,truth=float(ty[i,j]),center=float(tm[i,j]),lo=float(lo[i,j]),hi=float(hi[i,j]),width=float(width[i,j]),coverage=bool(lo[i,j]<=ty[i,j]<=hi[i,j]),interval_score=float(score[i,j]),width_normalized=float(width[i,j]/scale[j]),interval_score_normalized=float(score[i,j]/scale[j]),squared_error=float((ty[i,j]-tm[i,j])**2),plugin_squared_error=float((ty[i,j]-test['plugin'][i,j])**2),n_cal=10 if mode!='raw' else 0,rank=10 if mode!='raw' else 0))
  # Match query names to the original evaluator, including marker labels.
  labels=frame[frame.method=='plugin_conformal'].groupby('query').first().index.tolist();markers=json.loads((fd/'markers.json').read_text());names=[markers[a]+' : '+markers[b] for a,b in zip(ia,ib)]
  df=pd.DataFrame(rows);df['query']=df['query'].map(lambda x:names[int(x)]);frames.append(df)
 df=pd.concat(frames,ignore_index=True);df['method']=df.method.str.replace('ligature','ligature',regex=False);metrics=ev.summarize(df);paired=[]
 for family,g in [('all136',df),('clustering16',df[df.diag]),('cross120',df[~df.diag])]:
  pp=g.groupby(['case','method'])[['width_normalized','interval_score_normalized']].mean().unstack('method')
  for gen in ['ligature_conformal','ligature_constant_scale_conformal']:
   for direct in ['direct_adaptive','direct_constant']:
    for metric in ['width_normalized','interval_score_normalized']:
     delta=pp[metric][gen]-pp[metric][direct];paired.append(dict(family=family,method=gen,baseline=direct,metric=metric,ratio=float(pp[metric][gen].mean()/pp[metric][direct].mean()),difference=float(delta.mean()),difference_ci95=ev.bootstrap_ci(delta)))
 out=root/'evaluation';out.mkdir(exist_ok=True);df.to_csv(out/'patient_metrics.csv.gz',index=False);pd.DataFrame(metrics).to_csv(out/'summary.csv',index=False);jsonwrite(out/'summary.json',dict(status='complete',seed=271027,patients=40,folds=5,metrics=metrics,paired=paired,dataset=json.loads((root/'dataset_complete.json').read_text()),scope='Single-seed exploratory geometry/target-definition control. All model inputs use H&E-derived proxies; paired molecular images supply targets. Original selected crop release, not whole-slide validation or optical spillover correction. Different targets preclude comparison of absolute performance with the primary whole-cell study.'))
 print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__':main()
