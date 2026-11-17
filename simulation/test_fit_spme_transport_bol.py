import unittest,math,ast
from pathlib import Path
import numpy as np
import test_fit_spme_bidirectional_bol as bidirectional_fixture
from fit_spme_transport_bol import BOUNDS,metric_residuals,scale_diffusivity,select_shape_samples
class TransportTests(unittest.TestCase):
 def test_eight_phase_protocol_is_copied_without_change(self):
  here=Path(__file__).parent
  functions=[]
  for name in ['fit_spme_bidirectional_bol.py','fit_spme_transport_bol.py']:
   tree=ast.parse((here/name).read_text());functions.append(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='protocol_steps'))
  self.assertEqual(ast.dump(functions[0],include_attributes=False),ast.dump(functions[1],include_attributes=False))
 def test_bounds_retain_material_and_add_only_log_diffusion(self):
  self.assertEqual(BOUNDS,[[.85,.85,-.4,-1.],[1.,1.,.4,1.]])
 def test_multiplier_preserves_original_sto_and_temperature_dependence(self):
  def original(sto,T):return (1+sto)*math.exp(4000*(1/298.15-1/T))
  scaled=scale_diffusivity(original,.2)
  for sto,T in [(.01,280),(.5,298.15),(.9,320)]:self.assertAlmostEqual(scaled(sto,T),.2*original(sto,T))
  with self.assertRaises(ValueError):scale_diffusivity(original,-1)
 def test_first_reaching_raw_rows_are_selected_without_interpolation(self):
  cutoff=100.;t=cutoff+np.array([.066,60.066,600.066,3600.066,10800.066,21600.001]);v=np.arange(6,dtype=float)
  rows=select_shape_samples(t,v,cutoff);self.assertEqual([x['raw_index'] for x in rows],[1,2,3,4,5]);self.assertEqual([x['raw_voltage_V'] for x in rows],[1.,2.,3.,4.,5.]);self.assertAlmostEqual(rows[0]['raw_t_s'],160.066)
 def test_shape_loss_has_one_group_and_terminal_is_not_extrapolated(self):
  raw,obs=bidirectional_fixture.BidirectionalTests().fixture();obs.update(rest_shape_t=np.array([60.,600.,3600.,10800.]),rest_shape_v=np.array([2.88]*4),rest_shape_terminal_t=21600.001,rest_shape_terminal_v=2.9)
  r,m=metric_residuals(raw,obs);self.assertAlmostEqual(float(r@r),1);self.assertAlmostEqual(m['components']['relaxation_shape'],1);self.assertFalse(m['rest_shape_terminal_diagnostic']['covered']);self.assertIsNone(m['rest_shape_terminal_diagnostic']['model_V']);self.assertFalse(m['rest_shape_terminal_diagnostic']['used_in_loss'])
if __name__=='__main__':unittest.main()
