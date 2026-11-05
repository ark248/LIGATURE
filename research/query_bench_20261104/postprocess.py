"""Lossless compact patient records and post-fit descriptive contrasts."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from common import RUN
root=RUN/'evaluation'
p=root/'patient_metrics.csv.gz'
f=pd.read_csv(p,float_precision='round_trip')
f.to_parquet(root/'patient_metrics.parquet',index=False,compression='zstd')
g=pd.read_parquet(root/'patient_metrics.parquet');pd.testing.assert_frame_equal(f,g)
rng=np.random.default_rng(271043)
res=[]
for fam in ['all_scored','clustering','cross','Moran','Geary','log_assay_clustering']:
 for method in ['ligature','ligature_analytic','direct']:
  v=f[(f.family==fam)&(f.method==method)&(f.interval.isin(['adaptive','constant']))]
  if v.empty:continue
  v=v.groupby(['case','interval']).interval_score.mean().unstack()
  delta=v.adaptive-v.constant;picks=rng.integers(0,len(v),(10000,len(v)))
  res.append(dict(family=fam,method=method,contrast='adaptive minus same-center constant',ratio=float(v.adaptive.mean()/v.constant.mean()),difference=float(delta.mean()),ci95=np.quantile(delta.to_numpy()[picks].mean(1),[.025,.975]).tolist(),better_patients=int((delta<0).sum())))
seeds=[]
for (seed,fam,method),v in f[f.interval=='adaptive'].groupby(['seed','family','method']):
 by=v.groupby('query')[['squared_error','plugin_squared_error']].mean()
 seeds.append(dict(seed=int(seed),family=fam,method=method,MSE_ratio=float((by.squared_error/by.plugin_squared_error).mean()),interval_score=float(v.interval_score.mean()),coverage=float(v.coverage.mean())))
z=json.loads((root/'summary.json').read_text());costs={}
for kind in ['dscore','dquery','ligature']:
 v=[r['fit_seconds'] for r in z['costs'] if r['kind']==kind and r['fit_seconds']>0]
 costs[kind]=dict(n_new_fits=len(v),mean_fit_seconds=float(np.mean(v)),range_fit_seconds=[min(v),max(v)])
# Cheap heads are selection-only tuned, trained once per fold; do not count them three times.
for fold in range(5):
 from common import RUN
 p=RUN/f'fold{fold}/cheap/complete.json'
 if p.exists():print('cheapcost',fold,p.read_text()[:600])
out=dict(status='passed',parquet_roundtrip_exact=True,patient_rows=len(f),adaptive_constant=res,per_seed=seeds,training_cost=costs,source_csv_sha256=hashlib.sha256((root/'patient_metrics.csv.gz').read_bytes()).hexdigest(),parquet_sha256=hashlib.sha256((root/'patient_metrics.parquet').read_bytes()).hexdigest(),scope='Post-fit descriptive contrasts. Bootstrap conditions on fits; additional comparisons are exploratory. No new endpoint or model fitting.')
(root/'additional_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='per_seed'},indent=2))
