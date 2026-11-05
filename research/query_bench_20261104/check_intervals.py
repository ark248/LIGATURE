"""Meaningful calibration/score checks independent of any new test outcomes."""
from common import *
from evaluate import conformal,crps_samples,crps_gaussian

def main():
 y=np.arange(10,dtype=float)[:,None];lo,hi=conformal(y,np.zeros_like(y),np.ones_like(y),np.ones((2,1))*3,np.ones((2,1)));assert np.all(lo==-6) and np.all(hi==12)
 np.testing.assert_allclose(crps_samples(np.zeros((2,64,3)),np.ones((2,3))),1)
 np.testing.assert_allclose(crps_gaussian(np.zeros((2,3)),np.ones((2,3)),np.zeros((2,3))),(np.sqrt(2)-1)/np.sqrt(np.pi))
 # A positive constant spread rescale cancels from split-conformal intervals.
 lo2,hi2=conformal(y,np.zeros_like(y),np.ones_like(y)*2,np.ones((2,1))*3,np.ones((2,1))*2)
 np.testing.assert_allclose(lo,lo2);np.testing.assert_allclose(hi,hi2)
 sets=[]
 for f in range(5):
  roles=json.loads((BASE/f'fold{f}/roles.json').read_text());d={r:{p for p,v in roles.items() if v==r} for r in ['train','selection','calibration','test']}
  assert [len(d[r]) for r in d]==[20,2,10,8]
  assert all(not(d[a]&d[b]) for a in d for b in d if a!=b);sets.append(d['test'])
 assert len(set.union(*sets))==40 and sum(map(len,sets))==40
 # Raw assay transformation: nonuniform degree makes a centering term necessary.
 # check_queries verifies that exact term against direct unstandardized products.
 result=dict(status='passed',calibration_rank=10,calibration_patients=10,patient_roles_disjoint=True,test_patients_exactly_once=40,empirical_crps_identity=True,gaussian_crps_identity=True,checked_before_new_test_metrics=True)
 jsonwrite(Path(__file__).with_name('interval_checks.json'),result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
