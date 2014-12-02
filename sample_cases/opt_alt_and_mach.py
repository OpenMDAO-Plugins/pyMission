# Simultaneous optimization of Altitude and Mach to reproduce figure 10 in the paper.

import time
import os.path
import numpy as np

from openmdao.main.api import set_as_top
from openmdao.main.test.simpledriver import SimpleDriver
from openmdao.lib.casehandlers.api import BSONCaseRecorder

from pyoptsparse_driver.pyoptsparse_driver import pyOptSparseDriver
from pyMission.segment import MissionSegment

num_elem = 150
num_cp = 30
x_range = 1000.0  # nautical miles

# define bounds for the flight path angle
gamma_lb = np.tan(-35.0 * (np.pi/180.0))/1e-1
gamma_ub = np.tan(35.0 * (np.pi/180.0))/1e-1
takeoff_speed = 83.3
landing_speed = 72.2
takeoff_M = takeoff_speed / np.sqrt(1.4*287*288)
landing_M = landing_speed / np.sqrt(1.4*287*288)

# Convert from NMi to km
x_range *= 1.852

x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
M_init = 0.5 * np.sin(np.pi * x_init / (x_range/1e3)) + 0.3
h_init = 10 * np.sin(np.pi * x_init / (x_range/1e3))

model = set_as_top(MissionSegment(num_elem=num_elem, num_cp=num_cp,
                                  x_pts=x_init, surr_file='../crm_surr'))

model.replace('driver', pyOptSparseDriver())
#model.replace('driver', SimpleDriver())
model.driver.optimizer = 'SNOPT'

# Initial design parameters
model.S = 427.8/1e2
model.ac_w = 210000*9.81/1e6
model.thrust_sl = 1020000.0/1e6
model.SFCSL = 8.951*9.81
model.AR = 8.68
model.oswald = 0.8

# Add parameters, objectives, constraints
model.driver.add_parameter('h_pt', low=0.0, high=14.1)
model.driver.add_parameter('M_pt', low=0.001, high=0.949)
model.driver.add_objective('SysFuelObj.fuelburn')
model.driver.add_constraint('SysHi.h_i = 0.0')
model.driver.add_constraint('SysHf.h_f = 0.0')
#model.driver.add_constraint('SysMVBspline.M[0] = %f' % takeoff_M)
#model.driver.add_constraint('SysMVBspline.M[-1] = %f' % landing_M)
model.driver.add_constraint('SysMi.M_i = %f' % takeoff_M)
model.driver.add_constraint('SysMf.M_f = %f' % landing_M)
model.driver.add_constraint('SysTmin.Tmin < 0.0')
model.driver.add_constraint('SysTmax.Tmax < 0.0')
model.driver.add_constraint('SysBlockTime.time > 0.0')
model.driver.add_constraint('SysBlockTime.time < 11*3600.0')
model.driver.add_constraint('%.15f < SysGammaBspline.Gamma < %.15f' % \
                            (gamma_lb, gamma_ub), linear=True)

# Initial value of the parameter
model.h_pt = h_init
model.M_pt = M_init

# Calculate velocity from the Mach we have specified.
model.SysSpeed.v_specified = False

# Recording the results - This records just the parameters, objective,
# and constraints to mission_history_cp_#.bson
filename = os.path.join('plotting','opt_alt_and_mach','opt_alt_and_mach_history.bson')
model.recorders = [BSONCaseRecorder(filename)]
model.recorders.save_problem_formulation = True
model.recording_options.includes = ['*']

# Flag for making sure we run serial if we do an mpirun
model.driver.system_type = 'serial'
model.coupled_solver.system_type = 'serial'


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
    model.run()
    print 'OPTIMIZATION TIME:', time.time() - start


