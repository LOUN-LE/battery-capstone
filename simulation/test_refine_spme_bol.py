import unittest
import numpy as np
from refine_spme_bol import build_config,score_absolute_curve


class RefinementTests(unittest.TestCase):
    def config(self):
        return dict(family='SPMe',parameter_set='OKane2022',initial_soc_assumption=.2,initial_rest_s=120,
            pre_discharge_rest_s=7230,bounds=[[.85,-.4],[1.,.4]],limitations=[])

    def test_only_voltage_window_expanded_not_SOC_or_bounds(self):
        base=self.config();p=build_config(base,dict(chosen_C_candidate=dict(s_Q=.88,delta_Q_Li_Ah=.399)))
        self.assertEqual(p['voltage_capacity_fraction_grid'],[.02,.98,97])
        self.assertEqual(p['bounds'],base['bounds']);self.assertEqual(p['initial_soc_assumption'],.2)
        self.assertLessEqual(p['max_model_calls']+1,60);self.assertLessEqual(p['max_wall_s'],600)

    def test_protocol_mismatch_refused(self):
        c=self.config();c['pre_discharge_rest_s']=3600
        with self.assertRaises(ValueError):build_config(c,{})

    def test_absolute_Ah_axis_does_not_hide_capacity_difference(self):
        q=np.array([0.,1.,2.]);v=np.array([4.,3.5,3.])
        r=score_absolute_curve(q,v,np.array([0.,1.,2.,3.]),np.array([4.,3.5,3.,2.5]))
        self.assertAlmostEqual(r['capacity_error_percent'],-100/3)
        self.assertEqual(r['covered_fraction'],.75);self.assertEqual(r['full_covered_voltage_RMSE_V'],0)

    def test_malformed_curve_rejected(self):
        with self.assertRaises(ValueError):score_absolute_curve(np.array([0.,0.]),np.array([4.,3.]),np.array([0.,1.]),np.array([4.,3.]))


if __name__=='__main__':unittest.main()
