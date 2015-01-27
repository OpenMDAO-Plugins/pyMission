
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
        x_range = 9000.0

        altitude = np.zeros(num_elem+1)
        altitude = 10 * np.sin(np.pi * np.linspace(0,1,num_elem+1))        

        x_range *= 1.852
        x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
        M_init = np.ones(num_cp)*0.8
        h_init = 10 * np.sin(np.pi * x_init / (x_range/1e3))        

        model = set_as_top(MissionSegment(num_elem=num_elem, num_cp=num_cp,
                                          x_pts=x_init, surr_file='../crm_surr'))

        model.h_pt = h_init
        model.M_pt = M_init
        model.set_init_h_pt(altitude)

        # Initial parameters
        model.S = 427.8/1e2
        model.ac_w = 210000*9.81/1e6
        model.thrust_sl = 1020000.0/1e6
        model.SFCSL = 8.951*9.81
        model.AR = 8.68
        model.oswald = 0.8
    
        # Some adjustments to match the case ran for the pickle
        #model.SysFuelWeight.fuel_scale = 1e6
        #model.SysTau.thrust_scale = 0.072

        model.run()

        # Load in original data from pickle
        dirname = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(dirname, 'analysis2.p')
        old_data = pickle.load(open(filename, 'rb'))

        # Some names changed
        old_data['Gamma'] = old_data['gamma']
        old_data['temp'] = old_data['Temp']

        # Don't compare the extra constraint/objective stuff, because we
        # don't create comps for them.
        old_keys = old_data.keys()
        old_keys.remove('gamma')
        old_keys.remove('CL_tar')
        old_keys.remove('Temp')
        old_keys.remove('M_i')
        old_keys.remove('M_f')
        old_keys.remove('M_spline')
        old_keys.remove('jason')
        old_keys.remove('time')
        

        # Find data in model
        new_data = {}
        comps = [comp for comp in model.list_components() if comp not in ['coupled_solver']]
        for name in comps:
            comp = model.get(name)
            s1 = set(comp.list_vars())
            s2 = set(old_keys)
            s_int = s1.intersection(s2)
            for key in s_int:
                new_data[key] = comp.get(key)
                old_keys.remove(key)

        print old_keys
        self.assertEqual(len(old_keys), 0)

        for key in new_data.keys():
            old = old_data[key]
            new = new_data[key]

            #diff = np.nan_to_num(abs(new - old) / old)
            diff = new-old
            #print key
            #print old
            #print new
            assert_rel_error(self, diff.max(), 0.0, 1e-9)

if __name__ == "__main__":

    unittest.main()
