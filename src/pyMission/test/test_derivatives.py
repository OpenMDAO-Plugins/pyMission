
import unittest
import numpy as np
import random
import warnings

from openmdao.main.api import Assembly, set_as_top
from openmdao.util.testutil import assert_rel_error

from pyMission.aerodynamics import SysAeroSurrogate, SysCM
from pyMission.atmospherics import SysSFC, SysTemp, SysRho, SysSpeed
from pyMission.bsplines import SysXBspline, SysHBspline, SysMVBspline, \
                               SysGammaBspline
from pyMission.coupled_analysis import SysCLTar, SysCTTar, SysFuelWeight


# Ignore the numerical warnings from performing the rel error calc.
warnings.simplefilter("ignore")

NUM_ELEM = 3
NUM_PT = 4

arg_dict = {'num_elem': NUM_ELEM}

class Testcase_pyMission_derivs(unittest.TestCase):

    """ Test that analytica derivs match fd ones.. """

    def setUp(self):
        """ Called before each test. """
        self.model = set_as_top(Assembly())
        arg_dict = {'num_elem': NUM_ELEM}

    def tearDown(self):
        """ Called after each test. """
        self.model = None
        self.inputs = None
        self.outputs = None

    def setup(self, compname, kwargs):

        self.model.add('comp', eval('%s(**kwargs)' % compname))

        self.model.driver.workflow.add('comp')
        self.inputs, self.outputs = self.model.comp.list_deriv_vars()

        for item in self.inputs:
            val = self.model.comp.get(item)
            if hasattr(val, 'shape'):
                shape1 = val.shape
                self.model.comp.set(item, np.random.random(shape1))
            else:
                self.model.comp.set(item, random.random())

    def run_model(self):

        self.model.run()

    def compare_derivatives(self, rel_error=False):

        wflow = self.model.driver.workflow
        inputs = ['comp.%s' % v for v in self.inputs]
        outputs = ['comp.%s' % v for v in self.outputs]

        # Uncomment for manual testing.
        wflow.check_gradient(inputs=inputs, outputs=outputs)
        wflow.check_gradient(inputs=inputs, outputs=outputs, mode='adjoint')

        # Numeric
        wflow.config_changed()
        Jn = wflow.calc_gradient(inputs=inputs,
                                 outputs=outputs,
                                 mode="fd")
        #print Jn

        # Analytic forward
        wflow.config_changed()
        Jf = wflow.calc_gradient(inputs=inputs,
                                 outputs=outputs,
                                 mode='forward')

        #print Jf

        if rel_error:
            diff = np.nan_to_num(abs(Jf - Jn) / Jn)
        else:
            diff = abs(Jf - Jn)

        assert_rel_error(self, diff.max(), 0.0, 1e-3)

        # Analytic adjoint
        wflow.config_changed()
        Ja = wflow.calc_gradient(inputs=inputs,
                                 outputs=outputs,
                                 mode='adjoint')

        #print Ja

        if rel_error:
            diff = np.nan_to_num(abs(Ja - Jn) / Jn)
        else:
            diff = abs(Ja - Jn)

        assert_rel_error(self, diff.max(), 0.0, 1e-3)

    #def test_SysAeroSurrogate(self):

        #compname = 'SysAeroSurrogate'
        #self.setup(compname, arg_dict)
        #self.run_model()
        #self.compare_derivatives()

    #def test_SysCM(self):

        #compname = 'SysCM'
        #self.setup(compname, arg_dict)
        #self.model.comp.eval_only = True
        #self.run_model()
        #self.compare_derivatives()

    #def test_SysCLTar(self):

        #compname = 'SysCLTar'
        #self.setup(compname, arg_dict)
        #self.run_model()
        #self.compare_derivatives(rel_error=True)

    #def test_SysCTTar(self):

        #compname = 'SysCTTar'
        #self.setup(compname, arg_dict)
        #self.run_model()
        #self.compare_derivatives(rel_error=True)

    #def test_SysFuelWeight(self):

        #compname = 'SysFuelWeight'
        #self.setup(compname, arg_dict)
        #self.run_model()
        #self.compare_derivatives()

    #def test_SysSFC(self):

        #compname = 'SysSFC'
        #self.setup(compname, arg_dict)
        #self.run_model()
        #self.compare_derivatives()

    #def test_SysTemp(self):

        #compname = 'SysTemp'
        #self.setup(compname, arg_dict)
        #self.run_model()
        #self.compare_derivatives()

    #def test_SysRho(self):

        #compname = 'SysRho'
        #self.setup(compname, arg_dict)
        #self.run_model()
        #self.compare_derivatives()

    #def test_SysSpeed(self):

        #compname = 'SysSpeed'
        #self.setup(compname, arg_dict)
        #self.run_model()
        #self.compare_derivatives()

        #self.model.comp.v_specified = True
        #self.run_model()
        #self.compare_derivatives()

    #def test_SysXBspline(self):

        #compname = 'SysXBspline'
        #arg_dict['num_pt'] = NUM_PT
        #arg_dict['num_elem'] = 12
        #self.setup(compname, arg_dict)
        #self.model.comp.x_init = 100.0*(1 - np.cos(np.linspace(0, 1, NUM_PT)*np.pi))/2/1e6
        #self.run_model()
        #self.compare_derivatives()

    #def test_SysHBspline(self):

        #compname = 'SysHBspline'
        #arg_dict['num_pt'] = NUM_PT
        #arg_dict['num_elem'] = 12
        #self.setup(compname, arg_dict)
        #self.model.comp.x_init = 100.0*(1 - np.cos(np.linspace(0, 1, NUM_PT)*np.pi))/2/1e6
        #self.run_model()
        #self.compare_derivatives()

    #def test_SysMVBspline(self):

        #compname = 'SysMVBspline'
        #arg_dict['num_pt'] = NUM_PT
        #arg_dict['num_elem'] = 12
        #self.setup(compname, arg_dict)
        #self.model.comp.x_init = 100.0*(1 - np.cos(np.linspace(0, 1, NUM_PT)*np.pi))/2/1e6
        #self.run_model()
        #self.compare_derivatives()

    def test_SysGammaBspline(self):

        compname = 'SysGammaBspline'
        arg_dict['num_pt'] = NUM_PT
        arg_dict['num_elem'] = 12
        self.setup(compname, arg_dict)
        self.model.comp.x_init = 100.0*(1 - np.cos(np.linspace(0, 1, NUM_PT)*np.pi))/2/1e6
        self.run_model()
        self.compare_derivatives()

if __name__ == "__main__":

    unittest.main()
