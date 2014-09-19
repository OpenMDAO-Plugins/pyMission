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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab

##########################
# USER SPECIFIED DATA

params = {
    'S': 427.8/1e2,
    'ac_w': 210000*9.81/1e6,
    'thrust_sl': 1020000.0/1e6,
    'SFCSL': 8.951*9.81,
    'AR': 8.68,
    'e': 0.8,
    't_c': 0.09,
    'sweep': 31.6 * numpy.pi/180,
    }

num_elem = 3000
num_cp_init = 10
num_cp_max = 10
num_cp_step = 100
x_range = 8100.0      # range in nautical miles!
folder_path = '/home/kenmoore/Work/John/pyMission'

num_elem = 6
num_cp_init = 3
num_cp_max = 3
num_cp_step = 33

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
x_init = x_range * 1e3 * (1-numpy.cos(numpy.linspace(0, 1, num_cp)*numpy.pi))/2/1e6
v_init = numpy.ones(num_cp)*2.5
h_init = 10 * numpy.sin(numpy.pi * x_init / (x_range/1e3))

altitude = numpy.zeros(num_elem+1)
altitude = 10 * numpy.sin(numpy.pi * numpy.linspace(0,1,num_elem+1))

first = True
start = time.time()
while num_cp <= num_cp_max:

    # define initial altitude profile, as well as fixed profile for
    # x-distance and airspeed
    v_init = numpy.ones(num_cp)*2.5
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
    #traj.set_init_h_pt(altitude)
    main = traj.initialize_framework()

    #start_comp = time.time()
    main.compute(output=True)
    #print 'FINISHED COMPUTING:', time.time() - start_comp
    #exit()

    #dist = main.vec['u']('x')*1e3
    #alt = main.vec['u']('h')*1e3
    #speed = main.vec['u']('v')*1e2
    #alpha = main.vec['u']('alpha')*1e-1
    #throttle = main.vec['u']('tau')
    #eta = main.vec['u']('eta')*1e-1
    #fuel = main.vec['u']('fuel_w')*1e6
    #rho = main.vec['u']('rho')
    #lift_c = main.vec['u']('CL')
    #drag_c = main.vec['u']('CD')*1e-1
    #thrust = main.vec['u']('CT_tar')*1e-1
    #gamma = main.vec['u']('gamma')*1e-1
    #Mach = main.vec['u']('M')
    #thrust *= (0.5 * rho * speed**2 * 427.8)

    #values = [alt*3.28, speed*1.94, eta*180/numpy.pi,
    #          gamma*180/numpy.pi, Mach, alpha*180/numpy.pi,
    #          rho, throttle, lift_c,
    #          fuel*0.225, thrust*0.225, drag_c]
    #labels = ['altitude', 'TAS', 'trim',
    #          'path angle', 'Mach', 'AoA',
    #          'density', 'throttle', 'C_L',
    #          'fuel', 'thrust', 'C_D']

    #fig = matplotlib.pylab.figure(figsize=(18.0, 8.0))
    #fplot = fig.add_subplot
    #for i in xrange(12):
    #    fplot(4, 3, i+1).plot(dist/1.852, values[i])
    #    fplot(4, 3, i+1).set_ylabel(labels[i])
    #fig.savefig('surrogate_test_results.png')

    #print 'U NORM:', numpy.linalg.norm(main.vec['u'].array)
    #print 'F NORM:', numpy.linalg.norm(main.vec['f'].array)
    #main.check_derivatives_all()
    #exit()

    # initialize the trajectory optimization problem using the framework
    # instance initialized before with Optimization.py
    traj.set_gamma_bound(gamma_lb, gamma_ub)
    traj.set_takeoff_speed(takeoff_speed)
    traj.set_landing_speed(landing_speed)
    opt = traj.initialize_opt(main)

    # start timing, and perform optimization

    PROFILE = False

    # Optimize
    if PROFILE==True:
        import cProfile
        import pstats
        import sys
        cProfile.run('opt("SNOPT")', 'profout')
        p = pstats.Stats('profout')
        p.strip_dirs()
        p.sort_stats('time')
        p.print_stats()
        print '\n\n---------------------\n\n'
        p.print_callers()
        print '\n\n---------------------\n\n'
        p.print_callees()
    else:
        opt('SNOPT')

    run_case, last_itr = traj.history.get_index()
    folder_name = folder_path + name + '_%03i/' % (run_case)
    call (["mv", "./SNOPT_print.out", folder_name + 'SNOPT_%04i_print.out' %(num_cp)])
    call (["mv", "./hist.hst", folder_name + 'hist_%04i.hst' %(num_cp)])
    altitude = main.vec['u']('h')
    num_cp += num_cp_step
    first = False

print 'OPTIMIZATION TIME:', time.time() - start
seconds = main.vec['u']('time') * 1e4
mnt, sec = divmod(seconds, 60)
hrs, mnt = divmod(mnt, 60)
print 'FLIGHT TIME:', '%d:%02d:%02d' % (hrs, mnt, sec)
traj.history.print_max_min(main.vec['u'])



