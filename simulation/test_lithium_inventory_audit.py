import unittest
import numpy as np
from lithium_inventory_audit import audit_plating_capacity


class PlatingCapacityAuditTests(unittest.TestCase):
    def run_audit(self,plated,dead,total,**override):
        kwargs=dict(electrode_area_m2=.1,electrode_thickness_m=1e-4,faraday_C_per_mol=96485.33212)
        kwargs.update(override)
        return audit_plating_capacity(plated,dead,total,**kwargs)

    def test_total_can_fall_when_plated_li_strips_without_reversing_dead_li(self):
        conversion=.1*1e-4*96485.33212/3600
        r=self.run_audit([10.,2.],[1.,1.5],np.array([11.,3.5])*conversion)
        self.assertTrue(r['capacity_identity_pass'])
        self.assertGreater(r['plated_Ah'][0],r['plated_Ah'][1])
        self.assertLess(r['dead_Ah'][0],r['dead_Ah'][1])

    def test_active_material_fraction_would_fail_the_identity(self):
        total=np.array([11.])*.1*1e-4*96485.33212/3600
        r=self.run_audit([10.],[1.],total*.75)
        self.assertFalse(r['capacity_identity_pass'])

    def test_concentration_cannot_be_treated_as_capacity(self):
        self.assertFalse(self.run_audit([10.],[1.],[11.])['capacity_identity_pass'])

    def test_nonfinite_or_unpaired_samples_rejected(self):
        for p,d,t in [([np.nan],[1.],[1.]),([1.,2.],[1.],[1.]),([],[],[])]:
            with self.subTest(p=p),self.assertRaises(ValueError):self.run_audit(p,d,t)

    def test_geometry_must_have_physical_units_and_positive_values(self):
        for v in [True,0.,-1.,float('nan')]:
            with self.subTest(v=v),self.assertRaises(ValueError):
                self.run_audit([0.],[0.],[0.],electrode_thickness_m=v)

    def test_zero_plating_is_well_defined(self):
        self.assertTrue(self.run_audit([0.],[0.],[0.])['capacity_identity_pass'])


if __name__=='__main__':unittest.main()
