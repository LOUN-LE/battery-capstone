"""Clock alignment and prefix tests; no PDE or external sources required."""
import unittest
import numpy as np
from compare_dfn_initial_rpt import select_initial_route, strictly_increasing_trace, score_observed_window, check_local_bounds


class InitialRPTComparisonTests(unittest.TestCase):
    def test_prefix_never_includes_later_current(self):
        first = dict(kind="native_RPT_control",measurement=dict(rpt=0),spec=dict(operation="current"))
        gap = dict(kind="assumed_exposure",source_step=dict(assumed_operation="rest"))
        recovery = dict(kind="native_RPT_control",measurement=None,spec=dict(operation="rest"))
        later = dict(kind="native_RPT_control",measurement=None,spec=dict(operation="current"))
        self.assertEqual(select_initial_route([first,gap,recovery,later]), [first,gap,recovery])
        self.assertEqual(select_initial_route([first,gap,later]), [first])
        self.assertEqual(select_initial_route([first,gap,recovery],False), [first])

    def test_missing_or_duplicate_capacity_fails(self):
        with self.assertRaises(ValueError):
            select_initial_route([])

    def test_boundary_duplicate_chooses_right_sample(self):
        t,v,i = strictly_increasing_trace([0,1,1,2],[3,3,4,4],[0,0,1,1])
        np.testing.assert_array_equal(t,[0,1,2])
        np.testing.assert_array_equal(v,[3,4,4])
        with self.assertRaises(ValueError):
            strictly_increasing_trace([1,0],[3,4])

    def test_original_clock_drift_is_not_hidden_or_extrapolated(self):
        observed=dict(t_s=np.array([0.,1.,2.,3.]),V=np.array([3.,4.,4.,4.]),
                      I_A=np.zeros(4),valid=np.ones(4,dtype=bool))
        score=score_observed_window(np.array([0.,2.]),np.array([3.,3.]),np.zeros(2),observed)
        self.assertEqual(score["scored_raw_rows"],3)
        self.assertEqual(score["uncovered_raw_rows"],1)
        self.assertGreater(score["voltage_V"]["rmse"],.8)

    def test_local_peak_cannot_hide_behind_valid_average(self):
        parameters={"Primary: Maximum concentration in negative electrode [mol.m-3]":10.,
                    "Secondary: Maximum concentration in negative electrode [mol.m-3]":20.,
                    "Maximum concentration in positive electrode [mol.m-3]":30.}
        extrema={key:dict(min=.1,max=.9) for key in ("electrolyte","negative_primary","negative_secondary",
                 "positive","negative_porosity","positive_porosity")}
        self.assertEqual(check_local_bounds(extrema,parameters),[])
        extrema["negative_primary"]={"min":.1,"max":10.2}
        extrema["electrolyte"]["min"]=0.
        failures=check_local_bounds(extrema,parameters)
        self.assertEqual(len(failures),2)


if __name__ == "__main__":
    unittest.main()
