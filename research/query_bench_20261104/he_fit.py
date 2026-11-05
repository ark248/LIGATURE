"""Refit original independent protocol and direct comparator on H&E-derived geometry."""
from common import *
from worker import direct
import argparse,importlib.util,shutil,time

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);a=ap.parse_args();torch.set_num_threads(8);load_modules()
 if a.fold==0 and not (RUN/'timing.json').exists():
  from timing import main as profile_cost
  profile_cost()
 root=RUN/'he_control';out=root/'independent';(out/'src').mkdir(parents=True,exist_ok=True)
 for name in ['ligature.py','spatial_engression.py','eval_unseen_statistics.py','analytic_moments.py']:
  target=out/'src'/name
  if target.exists():assert sha(target)==sha(BASE/'src'/name)
  else:shutil.copy2(BASE/'src'/name,target)
 source_manifest=dict(commit='source-snapshot',files={p.name:sha(p) for p in (out/'src').glob('*.py')});jsonwrite(out/'source_manifest.json',source_manifest)
 fd=out/f'fold{a.fold}';fd.mkdir(exist_ok=True);shutil.copy2(BASE/f'fold{a.fold}/roles.json',fd/'roles.json')
 # Use the already audited independent fit unchanged, changing only input/output roots.
 spec=importlib.util.spec_from_file_location('independent_fit',PROJECT/'research/calibration/fit.py');fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit);fit.ROOT=root;fit.RUN=out
 sys.argv=['independent_fit','--fold',str(a.fold),'--threads','8'];fit.main()
 dest=fd/'direct';dest.mkdir(exist_ok=True)
 if (dest/'complete.json').exists():return
 roles=json.loads((fd/'roles.json').read_text());se=fit.se;t=fit.t;se.roles_for=lambda cases,f,s:{str(p):'train' if roles[str(p)]=='train' else 'val' if roles[str(p)]=='selection' else 'test' for p in cases}
 cells,x,y,crops,_=se.load(root,a.fold,271027,128)
 mean=se.CellGraphNet(x.shape[1],d=128,layers=3);mean.load_state_dict(torch.load(fd/'mean.pt',weights_only=True));mean.eval();pm=se.predict_mean(mean,crops,x,y);f=np.zeros_like(y);S=[];SP=[]
 for j,c in enumerate(crops):
  c['role']=roles[c['case']];c['index']=j;f[c['ii']]=pm[c['crop']];X,Y,D,A,M,NP=se.collate([c],x,y);F=torch.from_numpy(pm[c['crop']])[None];S.append(t.contrasts(se.crop_stats(Y,A,NP,M.float()))[0].numpy());SP.append(t.contrasts(se.crop_stats(F,A,NP,M.float()))[0].numpy())
 _,_,dg=load_modules();cache=dict(t=t,se=se,dg=dg,crops=crops,targets=dict(scored=np.stack(S),scored_plugin=np.stack(SP)),x=x,y=y,f=f)
 start=time.monotonic();result=direct(cache,dest,271027,'dscore');jsonwrite(dest/'complete.json',dict(**result,total_seconds=time.monotonic()-start,seed=271027,fold=a.fold,input_geometry='H&E only',targets='IF intensity in H&E-derived nuclear-plus-rim proxies'))
if __name__=='__main__':main()
