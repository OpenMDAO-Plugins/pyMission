''' Analysis with multiple mission segments in parallel '''

import time

import numpy as np

from openmdao.main.api import set_as_top, Driver, Assembly
from openmdao.main.mpiwrap import MPI
from openmdao.main.test.simpledriver import SimpleDriver
from openmdao.lib.casehandlers.api import BSONCaseRecorder

from pyoptsparse_driver.pyoptsparse_driver import pyOptSparseDriver
from pyMission.segment import MissionSegment
from pyMission.bsplines import setup_MBI
from pyMission.aeroTripan import setup_surrogate

aero_surr = {}
aero_surr['CL'], aero_surr['CD'], aero_surr['CM'], aero_surr['nums'] = setup_surrogate('crm_surr')

# Same discretization for each segment for now.
num_elem = 50#250
num_cp = 10#50

model = set_as_top(Assembly())

#------------------------
# Mission Segment 1
#------------------------

x_range = 9000.0  # nautical miles

# define bounds for the flight path angle
gamma_lb = np.tan(-35.0 * (np.pi/180.0))/1e-1
gamma_ub = np.tan(35.0 * (np.pi/180.0))/1e-1
takeoff_speed = 83.3
landing_speed = 72.2

altitude = 10 * np.sin(np.pi * np.linspace(0,1,num_elem+1))

start = time.time()

x_range *= 1.852
x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
M_init = np.ones(num_cp)*0.82
h_init = 10 * np.sin(np.pi * x_init / (x_range/1e3))
jac_h, jac_gamma = setup_MBI(num_elem+1, num_cp, x_init)

model.add('seg1', MissionSegment(num_elem=num_elem, num_cp=num_cp,
                                 x_pts=x_init, params_file='CRM_params.py',
                                 aero_surr=aero_surr,
                                 jac_h=jac_h, jac_gamma=jac_gamma))

# Initial value of the parameter
model.seg1.h_pt = h_init
model.seg1.M_pt = M_init
model.seg1.set_init_h_pt(altitude)

# Calculate velocity from the Mach we have specified.
model.seg1.SysSpeed.v_specified = False

# Initial design parameters
model.seg1.S = 427.8/1e2
model.seg1.ac_w = 210000*9.81/1e6
model.seg1.thrust_sl = 1020000.0/1e6
model.seg1.SFCSL = 8.951*9.81
model.seg1.AR = 8.68
model.seg1.oswald = 0.8

# Flag for making sure we run serial if we do an mpirun
model.seg1.driver.system_type = 'serial'
model.seg1.coupled_solver.system_type = 'serial'

#------------------------
# Mission Segment 2
#------------------------

x_range = 7000.0  # nautical miles

# define bounds for the flight path angle
gamma_lb = np.tan(-35.0 * (np.pi/180.0))/1e-1
gamma_ub = np.tan(35.0 * (np.pi/180.0))/1e-1
takeoff_speed = 83.3
landing_speed = 72.2

altitude = 10 * np.sin(np.pi * np.linspace(0,1,num_elem+1))

start = time.time()

x_range *= 1.852
x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
M_init = np.ones(num_cp)*0.82
h_init = 10 * np.sin(np.pi * x_init / (x_range/1e3))
jac_h, jac_gamma = setup_MBI(num_elem+1, num_cp, x_init)

model.add('seg2', MissionSegment(num_elem=num_elem, num_cp=num_cp,
                                 x_pts=x_init, params_file='CRM_params.py',
                                 aero_surr=aero_surr,
                                 jac_h=jac_h, jac_gamma=jac_gamma))

# Initial value of the parameter
model.seg2.h_pt = h_init
model.seg2.M_pt = M_init
model.seg2.set_init_h_pt(altitude)

# Calculate velocity from the Mach we have specified.
model.seg2.SysSpeed.v_specified = False

# Initial design parameters
model.seg2.S = 427.8/1e2
model.seg2.ac_w = 210000*9.81/1e6
model.seg2.thrust_sl = 1020000.0/1e6
model.seg2.SFCSL = 8.951*9.81
model.seg2.AR = 8.68
model.seg2.oswald = 0.8

# Flag for making sure we run serial if we do an mpirun
model.seg2.driver.system_type = 'serial'
model.seg2.coupled_solver.system_type = 'serial'

#------------------------
# Mission Segment 3
#------------------------

x_range = 5000.0  # nautical miles

# define bounds for the flight path angle
gamma_lb = np.tan(-35.0 * (np.pi/180.0))/1e-1
gamma_ub = np.tan(35.0 * (np.pi/180.0))/1e-1
takeoff_speed = 83.3
landing_speed = 72.2

altitude = 10 * np.sin(np.pi * np.linspace(0,1,num_elem+1))

start = time.time()

x_range *= 1.852
x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
M_init = np.ones(num_cp)*0.82
h_init = 10 * np.sin(np.pi * x_init / (x_range/1e3))
jac_h, jac_gamma = setup_MBI(num_elem+1, num_cp, x_init)

model.add('seg3', MissionSegment(num_elem=num_elem, num_cp=num_cp,
                                 x_pts=x_init, params_file='CRM_params.py',
                                 aero_surr=aero_surr,
                                 jac_h=jac_h, jac_gamma=jac_gamma))

# Initial value of the parameter
model.seg3.h_pt = h_init
model.seg3.M_pt = M_init
model.seg3.set_init_h_pt(altitude)

# Calculate velocity from the Mach we have specified.
model.seg3.SysSpeed.v_specified = False

# Initial design parameters
model.seg3.S = 427.8/1e2
model.seg3.ac_w = 210000*9.81/1e6
model.seg3.thrust_sl = 1020000.0/1e6
model.seg3.SFCSL = 8.951*9.81
model.seg3.AR = 8.68
model.seg3.oswald = 0.8

# Flag for making sure we run serial if we do an mpirun
model.seg3.driver.system_type = 'serial'
model.seg3.coupled_solver.system_type = 'serial'

#----------------------
# Prepare to Run
#----------------------

model.driver.workflow.add(['seg1', 'seg2', 'seg3'])

#model._setup()
#from openmdao.util.dotgraph import plot_system_tree
#plot_system_tree(model._system)

start = time.time()
model.run()

print "."
if MPI:
    comm = model._system.mpi.comm
    fuelburn = (model.seg1.SysFuelObj.fuelburn, model.seg2.SysFuelObj.fuelburn, 
                model.seg3.SysFuelObj.fuelburn)
    dist_seg = comm.gather(fuelburn, root=0)
    if MPI.COMM_WORLD.rank == 0:
        
        print "seg1 fuel burn", max(dist_seg[0][0], dist_seg[1][0], dist_seg[2][0])
        print "seg2 fuel burn", max(dist_seg[0][1], dist_seg[1][1], dist_seg[2][1])
        print "seg3 fuel burn", max(dist_seg[0][2], dist_seg[1][2], dist_seg[2][2])
else:
    print "seg1 fuel burn", model.seg1.SysFuelObj.fuelburn
    print "seg2 fuel burn", model.seg2.SysFuelObj.fuelburn
    print "seg3 fuel burn", model.seg3.SysFuelObj.fuelburn
print 'Simulation TIME:', time.time() - start

model.seg1.check_comp_derivatives()
model.seg1.list_unconnected_inputs()
