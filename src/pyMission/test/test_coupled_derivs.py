
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
        model.replace('driver', SimpleDriver())
    
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
        
        model.driver.add_parameter('h_pt', low=0.0, high=20.0)
        model.driver.add_objective('SysFuelObj.fuelburn')
        model.driver.add_constraint('SysHi.h_i = 0.0')
        model.driver.add_constraint('SysHf.h_f = 0.0')
        model.driver.add_constraint('SysTmin.Tmin < 0.0')
        model.driver.add_constraint('SysTmax.Tmax < 0.0')
        model.driver.add_constraint('%.15f < SysGammaBspline.Gamma < %.15f' % \
                                    (gamma_lb, gamma_ub), linear=True)
        
        model.run()
    
        new_derivs = model.driver.calc_gradient(return_format='dict')
    
        # Load in original data from pickle
        dirname = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(dirname, 'derivs2.p')
        old_derivs_dict = pickle.load(open(filename, 'rb'))
    
        translate_dict = {'fuelburn': '_pseudo_7.out0',
                          'h_i':      '_pseudo_8.out0',
                          'h_f':      '_pseudo_9.out0',
                          'Tmin':     '_pseudo_10.out0',
                          'Tmax':     '_pseudo_11.out0'}
                          #'gamma':    '_pseudo_12.out0'}
                          
        for j, key in enumerate(translate_dict.keys()):
            
            old = old_derivs_dict[key]['h_pt']
            print 'h_pt', key
    
            openmdao_key = translate_dict[key]
            new = new_derivs[openmdao_key]['h_pt']
    
            diff = new-old
            print 'old', old
            print 'new', new
            print old.shape, new.shape
            #assert_rel_error(self, diff.max(), 0.0, 1e-5)
        
        for i in xrange(0, num_elem):
            old = old_derivs_dict['gamma'][i]['h_pt']
            #print 'h_pt', 'gamma'+str(i)
    
            new = new_derivs['_pseudo_12.out0']['h_pt'][i, :]
    
            diff = new-old
            #print 'old', old
            #print 'new', new
            #print old.shape, new.shape
            assert_rel_error(self, diff.max(), 0.0, 1e-5)
                 
if __name__ == "__main__":

    unittest.main()
