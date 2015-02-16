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

from openmdao.main.api import set_as_top, Driver
from openmdao.main.test.simpledriver import SimpleDriver
from openmdao.lib.casehandlers.api import BSONCaseRecorder

from pyoptsparse_driver.pyoptsparse_driver import pyOptSparseDriver
from pyMission.segment import MissionSegment


num_elem = 250
num_cp_init = 50
num_cp_max = 50

#num_elem = 500
#num_cp_init = 100
#num_cp_max = 100

#num_elem = 750
#num_cp_init = 150
#num_cp_max = 150

#num_elem = 1000
#num_cp_init = 200
#num_cp_max = 200

#num_elem = 1250
#num_cp_init = 250
#num_cp_max = 250

#num_elem = 1500
#num_cp_init = 300
#num_cp_max = 300

#num_elem = 1750
#num_cp_init = 350
#num_cp_max = 350

num_cp_step = 100
x_range = 9000.0  # nautical miles

#num_elem = 6
#num_cp_init = 3
#num_cp_max = 3
#num_cp_step = 33

# define bounds for the flight path angle
gamma_lb = np.tan(-35.0 * (np.pi/180.0))/1e-1
gamma_ub = np.tan(35.0 * (np.pi/180.0))/1e-1
takeoff_speed = 83.3
landing_speed = 72.2

altitude = np.zeros(num_elem+1)
altitude = 10 * np.sin(np.pi * np.linspace(0,1,num_elem+1))

start = time.time()
num_cp = num_cp_init
while num_cp <= num_cp_max:

    x_range *= 1.852
    x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
    M_init = np.ones(num_cp)*0.82
    h_init = 10 * np.sin(np.pi * x_init / (x_range/1e3))

    model = set_as_top(MissionSegment(num_elem=num_elem, num_cp=num_cp,
                                      x_pts=x_init, surr_file='crm_surr'))

    model.replace('driver', pyOptSparseDriver())
    #model.replace('driver', SimpleDriver())
    model.driver.optimizer = 'SNOPT'
    model.driver.options = {'Iterations limit': 5000000}

    # Add parameters, objectives, constraints
    model.driver.add_parameter('h_pt', low=0.0, high=14.1)
    model.driver.add_objective('SysFuelObj.fuelburn')
    model.driver.add_constraint('SysHBspline.h[0] = 0.0')
    model.driver.add_constraint('SysHBspline.h[-1] = 0.0')
    model.driver.add_constraint('SysTmin.Tmin < 0.0')
    model.driver.add_constraint('SysTmax.Tmax < 0.0')
    model.driver.add_constraint('%.15f < SysGammaBspline.Gamma < %.15f' % \
                                (gamma_lb, gamma_ub), linear=True)

    # Initial value of the parameter
    model.h_pt = h_init
    #model.v_pt = v_init
    model.M_pt = M_init
    model.set_init_h_pt(altitude)

    # Calculate velocity from the Mach we have specified.
    model.SysSpeed.v_specified = False

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
    #model.recorders = [BSONCaseRecorder(filename)]
    #model.recorders.save_problem_formulation = True
    # model.recording_options.includes = model.driver.list_param_targets()
    # model.recording_options.includes.extend(model.driver.list_constraint_targets())
    # model.recording_options.includes.append('SysFuelObj.fuelburn')

    # Flag for making sure we run serial if we do an mpirun
    model.driver.system_type = 'serial'
    model.coupled_solver.system_type = 'serial'
    #model.coupled_solver.gradient_options.lin_solver = 'petsc_ksp'

    PROFILE = False

    # Optimize
    if PROFILE==True:
        import cProfile
        import pstats
        import sys
        cProfile.run('model.run()', 'profout')
        p = pstats.Stats('profout')
        p.strip_dirs()
        p.sort_stats('time')
        p.print_stats()
        print '\n\n---------------------\n\n'
        p.print_callers()
        print '\n\n---------------------\n\n'
        p.print_callees()
    else:
        start = time.time()
        # from openmdao.util.dotgraph import plot_graphs, plot_system_tree
        # model._setup()
        # plot_system_tree(model._system, fmt='pdf',
        #                    outfile='segment_sys_tree.pdf')
        # exit()
        model.run()
        print "Objective", model.driver.eval_objectives()
        print 'OPTIMIZATION TIME:', time.time() - start

    # Save final optimization results. This records the final value of every
    # variable in the model, and saves them in mission_final_cp_#.bson
    #model.replace('driver', SimpleDriver())
    #filename = 'mission_final_cp_%d.bson' % num_cp
    #model.recorders = [BSONCaseRecorder(filename)]
    #model.includes = ['*']
    #model.run()

    num_cp += num_cp_step
