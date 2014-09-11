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


from mission import *
from history import *
import time
from subprocess import call

##########################
# USER SPECIFIED DATA

params = {
    'S': 427.8/1e2,
    'ac_w': 210000*9.81/1e6,
    'thrust_sl': 1020000.0/1e6,
    'SFCSL': 8.951*9.81,
    'AR': 8.68,
    'e': 0.8,
    }

num_elem = 1000
num_cp_init = 10
num_cp_max = 110
num_cp_step = 50
x_range = 15000.0
folder_path = '/home/jason/Documents/Results/PlotTest_'

# END USER SPECIFIED DATA
##########################

num_cp = num_cp_init
if ((num_cp_max - num_cp_init)%num_cp_step) != 0:
    raise Exception('Specified max control pts and step do not agree!')

# determine folder name
name = '%ikm_i%i_d%i_f%i_p%i' % (int(x_range),
                                 num_cp_init,
                                 num_cp_step,
                                 num_cp_max,
                                 num_elem)


# define bounds for the flight path angle
gamma_lb = numpy.tan(-35.0 * (numpy.pi/180.0))/1e-1
gamma_ub = numpy.tan(35.0 * (numpy.pi/180.0))/1e-1

# define initial altitude profile, as well as fixed profile for
# x-distance and airspeed
x_init = x_range * 1e3 * (1-numpy.cos(numpy.linspace(0, 1, num_cp)*numpy.pi))/2/1e6
v_init = numpy.ones(num_cp)*2.3
h_init = 1 * numpy.sin(numpy.pi * x_init / (x_range/1e3))

altitude = numpy.zeros(num_elem+1)

first = True
start = time.time()
while num_cp <= num_cp_max:

    # define initial altitude profile, as well as fixed profile for
    # x-distance and airspeed
    v_init = numpy.ones(num_cp)*2.3
    x_init = x_range * 1e3 * (1-numpy.cos(numpy.linspace(0, 1, num_cp)*numpy.pi))/2/1e6

    # initialize the mission analysis problem with the framework
    traj = OptTrajectory(num_elem, num_cp, first)
    traj.set_init_h(h_init)
    traj.set_init_v(v_init)
    traj.set_init_x(x_init)
    traj.set_params(params)
    traj.set_folder(folder_path)
    traj.set_name(name)
    traj.setup_MBI()
    traj.set_init_h_pt(altitude)
    main = traj.initialize_framework()

    main.compute(output=True)

    # initialize the trajectory optimization problem using the framework
    # instance initialized before with Optimization.py
    traj.set_gamma_bound(gamma_lb, gamma_ub)
    opt = traj.initialize_opt(main)

    # start timing, and perform optimization
    opt('SNOPT')

    run_case, last_itr = traj.history.get_index()
    folder_name = folder_path + name + '_%03i/' % (run_case)
    call (["mv", "./SNOPT_print.out", folder_name + 'SNOPT_%04i_print.out' %(num_cp)])
    call (["mv", "./hist.hst", folder_name + 'hist_%04i.hst' %(num_cp)])
    altitude = main.vec['u']('h')
    num_cp += num_cp_step
    first = False
    
print 'OPTIMIZATION TIME', time.time() - start
traj.history.print_max_min(main.vec['u'])



