
import os
import pickle
import unittest
import warnings

import numpy as np

from openmdao.main.api import set_as_top, Driver
from openmdao.main.test.test_derivatives import SimpleDriver
from openmdao.util.testutil import assert_rel_error

from pyMission.segment import MissionSegment

# Ignore the numerical warnings from performing the rel error calc.
warnings.simplefilter("ignore")

class Testcase_pyMissionSegment(unittest.TestCase):
    """ Test that the segment output matches the original pyMission code. """

    def test_MissionSegment(self):

        # define bounds for the flight path angle
        gamma_lb = np.tan(-10.0 * (np.pi/180.0))/1e-1
        gamma_ub = np.tan(10.0 * (np.pi/180.0))/1e-1

        num_elem = 10
        num_cp = 5
        x_range = 150.0

        x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
        v_init = np.ones(num_cp)*2.3
        h_init = 1 * np.sin(np.pi * x_init / (x_range/1e3))

        model = set_as_top(MissionSegment(num_elem, num_cp, x_init))
        model.replace('driver', SimpleDriver())

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

        model.driver.add_parameter('h_pt', low=0.0, high=20.0)
        model.driver.add_objective('SysFuelObj.wf_obj')
        model.driver.add_constraint('SysHi.h_i = 0.0')
        model.driver.add_constraint('SysHf.h_f = 0.0')
        model.driver.add_constraint('SysTmin.Tmin < 0.0')
        model.driver.add_constraint('SysTmax.Tmax < 0.0')
        model.driver.add_constraint('%.15f < SysGammaBspline.Gamma < %.15f' % \
                                    (gamma_lb, gamma_ub), linear=True)

        model.run()

        new_derivs = model.driver.calc_gradient()

        # Load in original data from pickle
        dirname = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(dirname, 'derivs.p')
        old_derivs_dict = pickle.load(open(filename, 'rb'))

        for i in range(0, 4):

            old_derivs = old_derivs_dict['h_pt'+str(i)]
            print 'h_pt' + str(i)

            for j, key in enumerate(['wf_obj', 'h_i', 'h_f', 'Tmin', 'Tmax']):
                old = old_derivs[key]
                new = new_derivs[j, i]

                #diff = np.nan_to_num(abs(new - old) / old)
                diff = new-old
                print key
                print old
                print new
                assert_rel_error(self, diff.max(), 0.0, 1e-5)

if __name__ == "__main__":

    unittest.main()
