import unittest,ast
from pathlib import Path
import numpy as np
from fit_spme_bidirectional_bol import metric_residuals,stable_rest_endpoint
class BidirectionalTests(unittest.TestCase):
 def fixture(self):
  raw=[dict(t=np.array([0.,600.]),v=np.ones(2)*3.62,i=np.zeros(2),q=np.zeros(2)) for _ in range(8)]
  raw[1].update(v=np.array([3.62,4.2]),q=np.array([0.,-.25]))
  raw[2].update(i=np.array([-1.,-.05]))
  raw[4].update(v=np.array([4.1,2.5]),q=np.array([0.,.5]))
  raw[5]=dict(t=np.array([0.,21000.,21300.,21600.]),v=np.ones(4)*2.9,i=np.zeros(4),q=np.zeros(4))
  raw[7].update(v=np.array([3.,4.2]),q=np.array([0.,-.5]))
  obs=dict(CC_q=np.array([0.,.25]),CC_v=raw[1]['v'].copy(),CV_t=np.array([0.,600.]),CV_i=raw[2]['i'].copy(),C10_q=np.array([0.,.5]),C10_v=raw[4]['v'].copy(),recharge_q=np.array([0.,.5]),recharge_v=raw[7]['v'].copy(),CC_duration_s=600.,CV_duration_s=600.,CC_total_Ah=.25,C10_total_Ah=.5,rest_median_V=2.9,recharge_total_Ah=.5,recharge_duration_s=600.)
  return raw,obs
 def test_identical_curves_and_events_have_zero_residual(self):
  raw,obs=self.fixture();r,m=metric_residuals(raw,obs);np.testing.assert_allclose(r,0,atol=1e-12)
 def test_each_new_voltage_metric_has_mean_scaling(self):
  raw,obs=self.fixture();raw[5]['v']+=.02;raw[7]['v']+=.02;r,m=metric_residuals(raw,obs)
  self.assertAlmostEqual(m['components']['relaxed_voltage'],1);self.assertAlmostEqual(m['components']['C10_recharge_voltage'],1);self.assertAlmostEqual(float(r@r),2)
 def test_insufficient_model_recharge_never_extrapolates(self):
  raw,obs=self.fixture();raw[7]['q']=np.array([0.,-.4])
  with self.assertRaisesRegex(ValueError,'extrapolation'):metric_residuals(raw,obs)
 def test_rest_window_uses_declared_interior(self):
  v,m=stable_rest_endpoint(np.array([0.,21000.,21300.,21600.,21600.1]),np.array([2.7,2.9,2.9,2.9,3.4]))
  self.assertEqual(v,2.9);self.assertEqual(m['sample_count'],3)
 def test_protocol_matches_primary_exact_AST(self):
  here=Path(__file__).parent;a=ast.parse((here/'diagnose_low_rate_recharge.py').read_text());b=ast.parse((here/'fit_spme_bidirectional_bol.py').read_text());primary=next(n.value for n in ast.walk(a) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='steps' for t in n.targets));function=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='protocol_steps');new=next(n.value for n in ast.walk(function) if isinstance(n,ast.Return));self.assertEqual(ast.dump(primary,include_attributes=False),ast.dump(new,include_attributes=False))
if __name__=='__main__':unittest.main()
