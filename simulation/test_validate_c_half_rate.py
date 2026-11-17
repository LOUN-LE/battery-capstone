import unittest
import numpy as np
from validate_c_half_rate import specifications,score,validate_native_arrays,programmed_current_A,frozen_material_scales,resolve_diffusivity,apply_diffusivity
class ContinuousHalfRateTests(unittest.TestCase):
 def test_program_current_units_A_and_mA(self):
  self.assertEqual(programmed_current_A({'ctrl1_val':500.,'ctrl1_val_unit':1}),.5)
  self.assertEqual(programmed_current_A({'ctrl1_val':2.5,'ctrl1_val_unit':0}),2.5)
  with self.assertRaises(ValueError):programmed_current_A({'ctrl1_val':500.,'ctrl1_val_unit':2})
 def native(self):
  n=20
  return {'time_s':np.arange(n,dtype=float),'I_mA':np.zeros(n),'V':np.ones(n)*4,'mode':np.ones(n),'Ns':np.repeat(np.arange(10),2),'control_mixed_V_or_mA':np.zeros(n)}
 def test_contract_requires_current_mode_control(self):
  d=self.native();validate_native_arrays(d);del d['control_mixed_V_or_mA']
  with self.assertRaises(ValueError):validate_native_arrays(d)
 def test_native_clock_must_not_run_backwards(self):
  d=self.native();d['time_s'][5]=0
  with self.assertRaises(ValueError):validate_native_arrays(d)
 def test_gap_is_explicit_and_capacity_event_cannot_be_skipped(self):
  s=specifications();self.assertEqual(len(s),17);self.assertEqual(s[10]['file'],'gap');self.assertEqual(s[10]['duration'],.01)
  self.assertTrue(s[-1]['is_capacity']);self.assertEqual(s[-1]['value'],2.5);self.assertEqual(s[-1]['termination'],'2.5 V');self.assertFalse(s[-1].get('allow_initial_event',False));self.assertTrue(s[12]['allow_initial_event'])
 def test_scoring_reports_missing_coverage_without_extrapolation(self):
  r=score(np.array([0.,1.]),np.array([4.,3.]),np.array([0.,1.,2.]),np.array([4.,3.,100.]))
  self.assertEqual(r['RMSE'],0);self.assertAlmostEqual(r['coverage_fraction'],2/3)
 def test_fourth_coordinate_is_log10_not_physical_multiplier(self):
  scales,factor=frozen_material_scales({'parameters':[.93,.92,.1,-1.],'negative_diffusivity_multiplier':.1})
  self.assertEqual(scales,[.93,.92,.1]);self.assertEqual(factor,.1)
  with self.assertRaises(ValueError):frozen_material_scales({'parameters':[.93,.92,.1,-1.],'negative_diffusivity_multiplier':-1.})
  with self.assertRaises(ValueError):frozen_material_scales({'parameters':[.93,.92,.1,-1.]})
  with self.assertRaises(ValueError):frozen_material_scales({'parameters':[.93,.92,.1,-1.],'negative_diffusivity_multiplier':10.})
 def test_frozen_diffusivity_cannot_be_silently_overridden(self):
  for override in [{'source':'ORegan2022'},{'multiplier':.1}]:
   with self.assertRaises(ValueError):resolve_diffusivity(frozen_multiplier=.5,**override)
  self.assertEqual(resolve_diffusivity(frozen_multiplier=.5)['multiplier'],.5)
  with self.assertRaises(ValueError):resolve_diffusivity(multiplier=.1,source='ORegan2022')
  for value in [0,-1,float('nan'),float('inf'),True]:
   with self.assertRaises(ValueError):resolve_diffusivity(multiplier=value)
 def test_published_hybrid_replaces_only_Dn_function(self):
  import pybamm
  p=pybamm.ParameterValues('OKane2022');before={key:p[key] for key in p.keys()};key='Negative particle diffusivity [m2.s-1]'
  receipt=apply_diffusivity(p,resolve_diffusivity(source='ORegan2022'))
  self.assertEqual(receipt['source'],'ORegan2022');self.assertFalse(receipt['whole_parameter_set_replaced'])
  self.assertNotEqual(p[key],before[key])
  for name,value in before.items():
   if name!=key:self.assertEqual(p[name],value)
  self.assertAlmostEqual(receipt['sample_Dn_m2_s'][1]/4.4241003077280405e-15,1.,places=12)
if __name__=='__main__':unittest.main()
