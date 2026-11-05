"""Exercise all report branches on synthetic forecasts, never real test outcomes."""
from common import *
import evaluate as ev
import tempfile
from types import SimpleNamespace

def main():
 rng=np.random.default_rng(321);root=Path(tempfile.mkdtemp(prefix='ligature-eval-fixture-'));base=root/'base';run=root/'run';cases=np.array([f'SYNTHETIC{i:02}' for i in range(40)]);caches={};seed=271027
 def pack(path,role,ids,**values):
  path.mkdir(parents=True,exist_ok=True);np.savez_compressed(path/(role+'.npz'),case=np.sort(cases[ids]),**{k:v[np.argsort(cases[ids])] for k,v in values.items()})
 for fold in range(5):
  ix=np.roll(np.arange(40),-8*fold);roles={str(cases[j]):r for r,idx in [('test',ix[:8]),('selection',ix[8:10]),('calibration',ix[10:20]),('train',ix[20:])] for j in idx};crops=[dict(index=i,case=str(p),role=roles[str(p)],eligible=True) for i,p in enumerate(cases)]
  S=rng.normal(size=(40,832)).astype(np.float32);P=S+rng.normal(.3,.2,S.shape).astype(np.float32);Q=rng.normal(size=(40,48)).astype(np.float32);QP=Q+rng.normal(.3,.2,Q.shape).astype(np.float32);SJ=S+rng.normal(0,.2,S.shape).astype(np.float32);QJ=Q+rng.normal(0,.2,Q.shape).astype(np.float32)
  caches[fold]=dict(crops=crops,targets=dict(scored=S,scored_plugin=P,queries=Q,queries_plugin=QP))
  fd=run/f'fold{fold}';bf=base/f'fold{fold}';bf.mkdir(parents=True)
  for kind in ['dscore','dquery','ligature']:
   p=fd/f'{kind}_{seed}';p.mkdir(parents=True);jsonwrite(p/'complete.json',dict(fold=fold,seed=seed,kind=kind,fit_seconds=0))
  rp=fd/f'reuse_{seed}';rp.mkdir();jsonwrite(rp/'complete.json',{})
  for role in ['selection','calibration','test']:
   ids=np.array([i for i,c in enumerate(crops) if c['role']==role]);n=len(ids);sd=np.ones((n,832),np.float32)*.2;sdq=np.ones((n,48),np.float32)*.2;drawS=SJ[ids,None]+rng.normal(0,.2,(n,64,832));drawQ=QJ[ids,None]+rng.normal(0,.2,(n,64,48))
   pack(bf,role+'_patients',ids,truth=S[ids,424:560],plugin=P[ids,424:560],draws=drawS[:,:,424:560])
   pack(fd/f'dscore_{seed}',role,ids,truth=S[ids],plugin=P[ids],center=SJ[ids]+.02,sd=sd)
   pack(fd/f'dquery_{seed}',role,ids,truth=Q[ids],plugin=QP[ids],center=QJ[ids]+.02,sd=sdq)
   pack(fd/f'ligature_{seed}',role,ids,truth=Q[ids],plugin=QP[ids],center=drawQ.mean(1),draws=drawQ,independent_draws=drawQ+.1,scored_draws=drawS)
   for name in ['ridge','affine']:pack(fd/'cheap'/name,role,ids,truth=Q[ids],plugin=QP[ids],center=QJ[ids]+.1)
   for name in ['direct_moments','joint_moments']:pack(rp/name,role,ids,truth=Q[ids],plugin=QP[ids],center=QJ[ids]+.05)
   pack(rp/'analytic_scored',role,ids,truth=S[ids],plugin=P[ids],center=SJ[ids])
   pack(rp/'marginal_independent',role,ids,truth=Q[ids],plugin=QP[ids],center=drawQ.mean(1),draws=drawQ+.02)
 ev.RUN=run;ev.BASE=base;ev.read_cache=lambda f:caches[f];frame,cost=ev.collect(seed);summary=ev.summary(frame);paired=ev.comparisons(frame)
 assert frame.case.nunique()==40 and len(cost)==15 and frame.method.eq('ligature_analytic').any();assert all(r['patients']==40 for r in summary)
 assert np.isfinite(frame.select_dtypes(include='number')).all().all();assert len([r for r in paired if 'p_Holm_two_families' in r])==2
 assert ((frame.truth>=frame.lo)&(frame.truth<=frame.hi)).equals(frame.coverage)
 print('PASS synthetic-only complete-fold report: all methods, raw/conformal intervals, analytic means, scale calibration, costs and paired tests; no real benchmark outcomes inspected')
if __name__=='__main__':main()
