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
    'SFCSL': 40,#8.951,
    'AR': 8.68,
    'e': 0.8,
    }

num_elem = 100
num_cp_init = 10
num_cp_max = 200
num_cp_step = 10
x_range = 1000.0
folder_name = '/home/jason/Documents/Results/MGtest-'

# END USER SPECIFIED DATA
##########################

num_cp = num_cp_init
if ((num_cp_max - num_cp_init)%num_cp_step) != 0:
    raise Exception('Specified max control pts and step do not agree!')

# define bounds for the flight path angle
gamma_lb = numpy.tan(-10.0 * (numpy.pi/180.0))/1e-1
gamma_ub = numpy.tan(10.0 * (numpy.pi/180.0))/1e-1

# define initial altitude profile, as well as fixed profile for
# x-distance and airspeed
v_init = numpy.ones(num_cp)*2.3
x_init = x_range * 1e3 * (1-numpy.cos(numpy.linspace(0, 1, num_cp)*numpy.pi))/2/1e6
h_init = 1 * numpy.sin(numpy.pi * x_init / (x_range/1e3))

altitude = numpy.zeros(num_elem+1)

start = time.time()
while num_cp <= num_cp_max:

    # define initial altitude profile, as well as fixed profile for
    # x-distance and airspeed
    v_init = numpy.ones(num_cp)*2.3
    x_init = x_range * 1e3 * (1-numpy.cos(numpy.linspace(0, 1, num_cp)*numpy.pi))/2/1e6

    # initialize the mission analysis problem with the framework
    traj = OptTrajectory(num_elem, num_cp)
    traj.set_init_h(h_init)
    traj.set_init_v(v_init)
    traj.set_init_x(x_init)
    traj.set_params(params)
    traj.set_folder_name(folder_name)
    traj.setup_MBI()
    traj.set_init_h_pt(altitude)
    main = traj.initialize_framework()

    main.compute(True)

    # initialize the trajectory optimization problem using the framework
    # instance initialized before with Optimization.py
    traj.set_gamma_bound(gamma_lb, gamma_ub)
    opt = traj.initialize_opt(main)

    # start timing, and perform optimization
    opt('SNOPT')

    altitude = main.vec['u']('h')
    num_cp += num_cp_step
    
print 'OPTIMIZATION TIME', time.time() - start
main.history.print_max_min(main.vec['u'])
run_case, last_itr = main.history.get_index()

# move SNOPT output file to specified folder
folder_name = folder_name + 'dist'+str(int(x_range))\
    +'km-'+str(num_cp)+'-'+str(num_elem)+'-'+str(run_case)+'/.'
call (["mv", "./SNOPT_print.out", folder_name])

