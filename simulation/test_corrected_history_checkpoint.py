import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from corrected_history_checkpoint import check_contract,measurement_evidence


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(candidate_metadata={'initial_state_updates':{'primary':123.},'source_sha256':{'candidate':'abc'}},
                      options={'SEI':['solvent-diffusion limited','none']},mesh={'r_n':20},rtol=1e-7,atol=1e-9,
                      period_s=10,temperature_C=25,SEI_parameter_status='example coefficient; not calibrated; one diagnostic trial')
        self.plan=[{'step':0,'current':-.5},{'step':1,'rest_s':86400},{'step':2,'current':.5}]

    def test_identical_prefix(self):
        check_contract(self.cfg,copy.deepcopy(self.cfg),self.plan,copy.deepcopy(self.plan),2)

    def test_each_physical_configuration_change_refused(self):
        for key,new in [('candidate_metadata',{'multiplier':30}),('options',{'SEI':'none'}),('mesh',{'r_n':40}),
                        ('rtol',1e-6),('atol',1e-8),('temperature_C',10),('period_s',1),('SEI_parameter_status','fitted30')]:
            with self.subTest(key=key):
                other=copy.deepcopy(self.cfg);other[key]=new
                with self.assertRaisesRegex(ValueError,'contract changed'):check_contract(self.cfg,other,self.plan,self.plan,2)

    def test_prefix_rest_or_current_tamper_refused(self):
        for k,v in [('rest_s',3600),('current',0.)]:
            other=copy.deepcopy(self.plan);other[1][k]=v
            with self.assertRaisesRegex(ValueError,'prefix changed'):check_contract(self.cfg,self.cfg,self.plan,other,2)

    def test_zero_and_complete_prefix_refused(self):
        for n in [0,-1,3,4]:
            with self.assertRaises(ValueError):check_contract(self.cfg,self.cfg,self.plan,self.plan,n)

    def test_RPT_Q0_is_recomputed_from_raw_PDE_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);plan=[{'measurement':{'rpt':0}},{}];values=[dict(rpt_id='RPT0',Qref_Ah=4.8)]
            np.savez(folder/'step_000000.npz',Q_Ah=[1.,5.8],V=[4.2,2.5])
            evidence=measurement_evidence(folder,plan,values,1)
            self.assertEqual(evidence[0]['Qref_Ah'],4.8);self.assertEqual(len(evidence[0]['sha256']),64)
            # Even matching report/measurement JSON tampering cannot change Q0
            # without contradicting the independent raw solver artifact.
            values[0]['Qref_Ah']=4.9
            with self.assertRaisesRegex(ValueError,'saved raw PDE'):measurement_evidence(folder,plan,values,1)

    def test_RPT_cutoff_artifact_tamper_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);np.savez(folder/'step_000000.npz',Q_Ah=[0.,4.8],V=[4.2,2.6])
            with self.assertRaisesRegex(ValueError,'2.5V'):measurement_evidence(folder,[{'measurement':{'rpt':0}}],[dict(rpt_id='RPT0',Qref_Ah=4.8)],1)


if __name__=='__main__':unittest.main()
