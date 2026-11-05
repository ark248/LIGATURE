"""Recalculate reported summaries from patient records and verify source isolation."""
from common import *
import pandas as pd,subprocess

def main():
 source=Path(__file__).resolve().parent;out=RUN/'evaluation';report=json.loads((out/'summary.json').read_text());assert report['status']=='complete' and report['seeds']==[271027,272027,273027]
 records=out/'patient_metrics.parquet'
 frame=pd.read_parquet(records) if records.exists() else pd.read_csv(out/'patient_metrics.csv.gz',float_precision='round_trip');assert frame.seed.nunique()==3 and frame.case.nunique()==40
 assert frame.groupby('case').fold.nunique().eq(1).all();assert np.isfinite(frame.select_dtypes(include='number')).all().all()
 assert ((frame.truth>=frame.lo)&(frame.truth<=frame.hi)).equals(frame.coverage)
 np.testing.assert_allclose(frame.squared_error,(frame.truth-frame.center)**2,rtol=1e-6,atol=1e-9)
 width=frame.hi-frame.lo;score=width+20*np.maximum(frame.lo-frame.truth,0)+20*np.maximum(frame.truth-frame.hi,0)
 np.testing.assert_allclose(frame.width,width/frame.normalization_scale,rtol=1e-6,atol=1e-8);np.testing.assert_allclose(frame.interval_score,score/frame.normalization_scale,rtol=1e-6,atol=1e-8)
 for row in report['metrics']:
  f=frame[(frame.family==row['family'])&(frame.method==row['method'])&(frame.interval==row['interval'])];assert f.case.nunique()==40 and f.seed.nunique()==3
  means=f.groupby(['seed','query'])[['squared_error','plugin_squared_error']].mean();ratio=(means.squared_error/means.plugin_squared_error.clip(lower=1e-12)).mean();assert abs(ratio-row['MSE_ratio'])<1e-9
  for field in ['coverage','width','interval_score','crps','crps_selection_scaled']:assert abs(f[field].mean()-row[field])<1e-8,(row['method'],field)
 # Same patient/query truth and plug-in are shared by every method and residual seed.
 for _,g in frame.groupby(['family','case','query']):assert g.truth.max()-g.truth.min()<1e-6 and g.plugin_squared_error.max()-g.plugin_squared_error.min()<1e-6
 # Check direct scored coordinates against independently saved assay targets,
 # rather than relying only on the evaluator's common truth column.
 source_alignment=[]
 for fold in range(5):
  for seed in report['seeds']:
   for role in ['selection','calibration','test']:
    expected=np.load(BASE/f'fold{fold}'/(role+'_patients.npz'))
    direct=np.load(RUN/f'fold{fold}'/f'dscore_{seed}'/(role+'.npz'))
    assert np.array_equal(expected['case'],direct['case'])
    np.testing.assert_allclose(expected['truth'],direct['truth'][:,424:560],atol=3e-6)
    source_alignment.append(dict(fold=fold,seed=seed,role=role,passed=True))
 result=dict(status='passed',patients=40,seeds=3,patient_rows=len(frame),metric_rows=len(report['metrics']),same_truth_across_methods=True,intervals_recalculated=True,all_MSE_ratios_recalculated=True,source_arithmetic='float32 forecasts; float64 record recalculation uses 1e-6 relative tolerance',squared_error_recalculation_max_abs_difference=float(np.max(np.abs(frame.squared_error-(frame.truth-frame.center)**2))),protocol_sha256=sha(source/'PROTOCOL.md'),report_sha256=sha(out/'summary.json'),patient_records_file=records.name if records.exists() else 'patient_metrics.csv.gz',patient_records_sha256=sha(records if records.exists() else out/'patient_metrics.csv.gz'),method_scope='Development-cohort evidence; paired intervals condition on fits and do not include all cross-fold dependence.')
 result['direct_source_alignment']=source_alignment
 jsonwrite(out/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['numerical_repair','original_scored_draw_replay','direct_source_alignment']},indent=2))
if __name__=='__main__':main()
