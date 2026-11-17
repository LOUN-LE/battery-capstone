import copy,tempfile,unittest
from pathlib import Path
from runtime_physics_identity import make_identity,validate_identical


class IdentityTests(unittest.TestCase):
    def test_unchanged_identity(self):
        old=make_identity({'SEI':2.5e-22,'OCP':{'expression':'tanh(x)'}},{'python':'3.11'},[])
        validate_identical(old,copy.deepcopy(old))

    def test_physical_parameter_and_runtime_changes_refused(self):
        old=make_identity({'SEI':2.5e-22,'OCP':{'expression':'tanh(x)'}},{'python':'3.11'},[])
        for params,versions in [({'SEI':7.5e-21,'OCP':{'expression':'tanh(x)'}},{'python':'3.11'}),
                                ({'SEI':2.5e-22,'OCP':{'expression':'tanh(2*x)'}},{'python':'3.11'}),
                                (old['actual_parameter_values'],{'python':'3.12'})]:
            with self.assertRaises(ValueError):validate_identical(old,make_identity(params,versions,[]))

    def test_callable_source_changes_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'ocp.py';source.write_text('def ocp(x): return x')
            old=make_identity({'function':'ocp'},{},[source]);source.write_text('def ocp(x): return x*x')
            with self.assertRaises(ValueError):validate_identical(old,make_identity({'function':'ocp'},{},[source]))

    def test_tampered_identity_digest_refused(self):
        old=make_identity({'SEI':1.},{},[]);new=copy.deepcopy(old);new['actual_parameter_values']['SEI']=30.
        with self.assertRaises(ValueError):validate_identical(old,new)


if __name__=='__main__':unittest.main()
