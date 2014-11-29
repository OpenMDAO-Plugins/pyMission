"""
MISSION ANALYSIS/TRAJECTORY OPTIMIZATION
This is the runscript used for the trajectory optimization problem.
For details regarding the setup of the analysis problem, see mission.py
The mission analysis and trajectory optimization tool was developed by:
    Jason Kao*
    John Hwang*

* University of Michigan Department of Aerospace Engineering,
  Multidisciplinary Design Optimization Lab
  mdolab.engin.umich.edu

copyright July 2014
"""


from mission import *
from history import *
import time
from subprocess import call
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab

##########################
# USER SPECIFIED DATA

params = {
    'S': 124.58/1e2,
    'ac_w': 57719.3*9.81/1e6,
    'thrust_sl': 2*121436/1e6,
    'SFCSL': (2*0.00001076/1e-6)*9.81,
    #'SFCSL': 8.951 * 9.81,
    'AR': 8.93,
    'e': 0.8,
    't_c': 0.11,
    'sweep': 25.33 * numpy.pi/180,
    }

num_elem = 500
num_cp_init = 100
num_cp_max = 100
num_cp_step = 100
x_range = 2950.0      # range in nautical miles!
folder_path = './Results/738Case_'

# END USER SPECIFIED DATA
##########################

num_cp = num_cp_init
if ((num_cp_max - num_cp_init)%num_cp_step) != 0:
    raise Exception('Specified max control pts and step do not agree!')

# determine folder name
name = '%inm_i%i_d%i_f%i_p%i' % (int(x_range),
                                 num_cp_init,
                                 num_cp_step,
                                 num_cp_max,
                                 num_elem)

# define bounds for the flight path angle
gamma_lb = numpy.tan(-35.0 * (numpy.pi/180.0))/1e-1
gamma_ub = numpy.tan(35.0 * (numpy.pi/180.0))/1e-1
takeoff_speed = 83.3
landing_speed = 72.2

# define initial altitude profile, as well as fixed profile for
# x-distance and airspeed
x_range *= 1.852
#x_init = x_range * 1e3 * (1-numpy.cos(numpy.linspace(0, 1, num_cp)*numpy.pi))/2/1e6
x_init = x_range * 1e3 * numpy.linspace(0, 1, num_cp) / 1e6
M_init = numpy.ones(num_cp)*0.78
M_init[:2] = numpy.linspace(0.5, 0.78, 2)
M_init[95:] = numpy.linspace(0.78, 0.5, 5)
#M_init[:1] = numpy.linspace(0.2, 0.78, 1)
#M_init[8:] = numpy.linspace(0.78, 0.2, 2)
#h_init = 10 * numpy.sin(numpy.pi * x_init / (x_range/1e3))
h_init = (35000 / 3.28) * numpy.ones(num_cp) / 1e3
h_init[:2] = numpy.linspace(0, 35000/3.28, 2) / 1e3
h_init[95:] = numpy.linspace(35000/3.28, 0, 5) / 1e3
#h_init[:1] = numpy.linspace(0, 35000/3.28, 1) / 1e3
#h_init[8:] = numpy.linspace(35000/3.28, 0, 2) / 1e3

altitude = numpy.zeros(num_elem+1)
#altitude = 10 * numpy.sin(numpy.pi * numpy.linspace(0,1,num_elem+1))
#altitude = (35000 / 3.28) * numpy.ones(num_elem+1) / 1e3
#altitude[:30] = numpy.linspace(0, 35000/3.28, 30) / 1e3
#altitude[1101:] = numpy.linspace(35000/3.28, 0, 100) / 1e3

first = True
start = time.time()
while num_cp <= num_cp_max:

    # initialize the mission analysis problem with the framework
    traj = OptTrajectory(num_elem, num_cp, first)
    traj.set_init_h(h_init)
    traj.set_init_M(M_init)
    traj.set_init_x(x_init)
    traj.set_params(params)
    traj.set_folder(folder_path)
    traj.set_name(name)
    traj.setup_MBI()
    #traj.set_init_h_pt(altitude)
    main = traj.initialize_framework()

    #start_comp = time.time()
    main.compute(output=True)
    #print 'FINISHED COMPUTING:', time.time() - start_comp
    #exit()
    traj.history.save_history(main.vec['u'])

    '''
    traj.set_gamma_bound(gamma_lb, gamma_ub)
    traj.set_takeoff_speed(takeoff_speed)
    traj.set_landing_speed(landing_speed)
    opt = traj.initialize_opt(main)

    opt('SNOPT')
    '''

    run_case, last_itr = traj.history.get_index()
    folder_name = folder_path + name + '_%03i/' % (run_case)
    altitude = main.vec['u']('h')
    num_cp += num_cp_step
    first = False
    
print 'OPTIMIZATION TIME:', time.time() - start
seconds = main.vec['u']('time') * 1e4
mnt, sec = divmod(seconds, 60)
hrs, mnt = divmod(mnt, 60)
print 'FLIGHT TIME:', '%d:%02d:%02d' % (hrs, mnt, sec)
print 'FINAL FUEL BURN:', main.vec['u']('fuel_w')[0] * 1e5 * 0.225, 'lb'
traj.history.print_max_min(main.vec['u'])



