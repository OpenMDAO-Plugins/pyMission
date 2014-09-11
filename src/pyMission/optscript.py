"""
MISSION ANALYSIS/TRAJECTORY OPTIMIZATION
This is the runscript used for the trajectory optimization problem.
For details regarding the setup of the analysis problem, see mission.py
The mission analysis and trajectory optimization tool was developed by:
    Jason Kao*
    John Hwang*

* University of Michigan Department of Aerospace Engineering,
  Multidisciplinary Design Optimization lab
  mdolab.engin.umich.edu

copyright July 2014
"""

import time

import numpy as np

from openmdao.main.api import set_as_top
from openmdao.lib.casehandlers.api import BSONCaseRecorder

from pyoptsparse_driver.pyoptsparse_driver import pyOptSparseDriver
from pyMission.segment import MissionSegment


num_elem = 3000
num_cp_init = 10
num_cp_max = 10        # set to 200 for the sweep
num_cp_step = 10
x_range = 15000.0

# define bounds for the flight path angle
gamma_lb = np.tan(-10.0 * (np.pi/180.0))/1e-1
gamma_ub = np.tan(10.0 * (np.pi/180.0))/1e-1

start = time.time()
num_cp = num_cp_init
while num_cp <= num_cp_max:

    x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
    v_init = np.ones(num_cp)*2.3
    h_init = 1 * np.sin(np.pi * x_init / (x_range/1e3))

    model = set_as_top(MissionSegment(num_elem, num_cp, x_init))
    model.replace('driver', pyOptSparseDriver())
    model.driver.optimizer = 'SNOPT'
    #opt_dict = {'Iterations limit': 1000000,
    #            'Major iterations limit': 1000000,
    #            'Minor iterations limit': 1000000 }
    #model.driver.options = opt_dict

    # Add parameters, objectives, constraints
    model.driver.add_parameter('h_pt', low=0.0, high=20.0)
    model.driver.add_objective('SysFuelObj.wf_obj')
    model.driver.add_constraint('SysHi.h_i = 0.0')
    model.driver.add_constraint('SysHf.h_f = 0.0')
    model.driver.add_constraint('SysTmin.Tmin < 0.0')
    model.driver.add_constraint('SysTmax.Tmax < 0.0')
    model.driver.add_constraint('%.15f < SysGammaBspline.Gamma < %.15f' % \
                                (gamma_lb, gamma_ub), linear=True)

    # Initial value of the parameter
    model.h_pt = h_init
    model.v_pt = v_init

    # Pull velocity from BSpline instead of calculating it.
    model.SysSpeed.v_specified = True

    # Initial design parameters
    model.S = 427.8/1e2
    model.ac_w = 210000*9.81/1e6
    model.thrust_sl = 1020000.0/1e6
    model.SFCSL = 8.951*9.81
    model.AR = 8.68
    model.oswald = 0.8

    # Recording the results - This records just the parameters, objective,
    # and constraints to mission_history_cp_#.bson
    filename = 'mission_history_cp_%d.bson' % num_cp
    model.recorders = [BSONCaseRecorder(filename)]
    model.includes = model.driver.list_param_targets()
    model.includes.extend(model.driver.list_constraint_targets())
    model.includes.append('SysFuelObj.wf_obj')

    # Optimize
    model.run()
    print 'h_pt', model.h_pt

    # Save final optimization results. This records the final value of every
    # variable in the model, and saves them in mission_final_cp_#.bson
    from openmdao.main.test.test_derivatives import SimpleDriver
    model.replace('driver', SimpleDriver())
    filename = 'mission_final_cp_%d.bson' % num_cp
    model.recorders = [BSONCaseRecorder(filename)]
    model.includes = ['*']
    model.run()

    num_cp += num_cp_step