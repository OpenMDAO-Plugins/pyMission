
import os
import pickle
import unittest
import warnings

import numpy as np

from openmdao.main.api import set_as_top, Driver
from openmdao.util.testutil import assert_rel_error

from pyMission.segment import MissionSegment

# Ignore the numerical warnings from performing the rel error calc.
warnings.simplefilter("ignore")

class Testcase_pyMissionSegment(unittest.TestCase):
    """ Test that the segment output matches the original pyMission code. """

    def test_MissionSegment(self):

        num_elem = 100
        num_cp = 30
        x_range = 150.0

        x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
        v_init = np.ones(num_cp)*2.3
        h_init = 1 * np.sin(np.pi * x_init / (x_range/1e3))

        model = set_as_top(MissionSegment(num_elem, num_cp, x_init))

        model.h_pt = h_init
        model.v_pt = v_init

        # Pull velocity from BSpline instead of calculating it.
        model.SysSpeed.v_specified = True

        # Initial parameters
        model.S = 427.8/1e2
        model.ac_w = 210000*9.81/1e6
        model.thrust_sl = 1020000.0/1e6/3
        model.SFCSL = 8.951
        model.AR = 8.68
        model.oswald = 0.8

        # Change some scaling parameters so that we match what they were when
        # the pickle was created.
        model.SysTau.thrust_scale = 0.072
        model.SysCLTar.fuel_scale = 1e6
        model.SysCTTar.fuel_scale = 1e6
        model.SysFuelWeight.fuel_scale = 1e6
        model.run()

        # Load in original data from pickle
        dirname = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(dirname, 'analysis.p')
        old_data = pickle.load(open(filename, 'rb'))

        # Some names changed
        old_data['Gamma'] = old_data['gamma']
        old_data['temp'] = old_data['Temp']

        # Don't compare the extra constraint/objective stuff, because we
        # don't create comps for them.
        old_keys = old_data.keys()
        old_keys.remove('gamma')
        old_keys.remove('gamma_max')
        old_keys.remove('gamma_min')
        #old_keys.remove('Tmin')
        #old_keys.remove('Tmax')
        #old_keys.remove('h_i')
        #old_keys.remove('h_f')
        #old_keys.remove('wf_obj')
        old_keys.remove('CL_tar')
        old_keys.remove('thrust_sl')
        old_keys.remove('e')
        old_keys.remove('Temp')
        old_keys.remove('M_pt')


        # Find data in model
        new_data = {}
        comps = [comp for comp in model.list_components() if comp not in ['coupled_solver', 'drag_solver']]
        for name in comps:
            comp = model.get(name)
            for key in old_keys:
                if key in comp.list_vars():
                    new_data[key] = comp.get(key)
                    old_keys.remove(key)

        self.assertEqual(len(old_keys), 0)

        for key in new_data.keys():
            old = old_data[key]
            new = new_data[key]

            #diff = np.nan_to_num(abs(new - old) / old)
            diff = new-old
            #print key
            #print old
            #print new
            assert_rel_error(self, diff.max(), 0.0, 1e-11)

if __name__ == "__main__":

    unittest.main()
