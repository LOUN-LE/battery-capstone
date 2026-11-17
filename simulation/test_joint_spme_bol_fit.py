import unittest
import numpy as np
from joint_spme_bol_fit import derive_updates,homogeneous_state,interpolate_required,select_rest_anchor

class JointTests(unittest.TestCase):
 def parameters(self):
  return {'Electrode width [m]':.1,'Electrode height [m]':1.,'Negative electrode thickness [m]':1e-4,'Positive electrode thickness [m]':1e-4,'Negative electrode active material volume fraction':.7,'Positive electrode active material volume fraction':.7,'Negative electrode porosity':.3,'Positive electrode porosity':.3,'Initial concentration in negative electrode [mol.m-3]':12000.,'Initial concentration in positive electrode [mol.m-3]':30000.,'Maximum concentration in negative electrode [mol.m-3]':30000.,'Maximum concentration in positive electrode [mol.m-3]':50000.,'Negative electrode OCP [V]':lambda x:.5-.6*x,'Positive electrode OCP [V]':lambda y:5-2.5*y}
 def test_independent_loading_changes_ratio_and_preserves_bounds(self):
  p=self.parameters();u,a=derive_updates(p,.9,.95,.1)
  self.assertAlmostEqual(u['Negative electrode active material volume fraction'],.63)
  self.assertAlmostEqual(u['Positive electrode active material volume fraction'],.665)
  self.assertLess(a['Q_n_Ah']/a['Q_p_Ah'],30000/50000)
 def test_unchanged_domain_rejects_outside(self):
  for x in [(1.01,.9,0),(.9,.84,0),(.9,.9,.401)]:
   with self.assertRaises(ValueError):derive_updates(self.parameters(),*x)
 def test_equilibrium_anchor_conserves_inventory_without_SOC_free_parameter(self):
  p=self.parameters();u,a=derive_updates(p,.9,.9,.1);p.update(u)
  state,m=homogeneous_state(p,a,3.662)
  self.assertAlmostEqual(m['initial_equilibrium_V'],3.662,places=8)
  self.assertAlmostEqual(m['lithium_inventory_residual_Ah'],0,places=12)
  self.assertGreater(m['equilibrium_SOC'],0);self.assertLess(m['equilibrium_SOC'],1)
 def test_fixed_grid_never_extrapolates_or_stretches(self):
  with self.assertRaisesRegex(ValueError,'extrapolation'):interpolate_required(np.array([0.,1.,2.]),np.array([4.,3.,2.]),np.array([0.,1.,3.]))
  np.testing.assert_allclose(interpolate_required(np.array([0.,1.,2.]),np.array([4.,3.,2.]),np.array([.5,1.5])),[3.5,2.5])
 def test_rest_anchor_excludes_only_declared_control_boundary(self):
  phase={'t':np.array([0.,90.,100.,110.,120.044]),'v':np.array([3.62,3.62,3.621,3.62,3.662]),'i':np.zeros(5)}
  target,meta=select_rest_anchor(phase)
  self.assertAlmostEqual(target,(3.62+3.621+3.62)/3)
  self.assertEqual(meta['selected_indices'],[1,2,3]);self.assertEqual(meta['boundary_candidates'][0]['index'],4)
  self.assertEqual(len(phase['v']),5)
 def test_rest_anchor_requires_sufficient_inside_samples(self):
  with self.assertRaises(ValueError):select_rest_anchor({'t':np.array([0.,130.,140.]),'v':np.array([3.6,3.7,3.8]),'i':np.zeros(3)})
 def test_nonmonotone_axis_refused(self):
  with self.assertRaises(ValueError):interpolate_required(np.array([0.,0.,2.]),np.array([4.,3.,2.]),np.array([.5]))

if __name__=='__main__':unittest.main()
