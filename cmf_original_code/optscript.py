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

execfile('./crm_params.py')

num_elem = 250
num_cp_init = 50
num_cp_max = 50
#num_elem = 1500#50
#num_cp_init = 300#5
#num_cp_max = 300#5
num_cp_step = 100
x_range = 9000.0      # range in nautical miles!
#fileloc = open('./path.txt', 'r')
#folder_path = fileloc.readlines()[0][:-1]
#fileloc.close()
folder_path = '.'

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
h_init = 10 * numpy.sin(numpy.pi * x_init / (x_range/1e3))

altitude = numpy.zeros(num_elem+1)
altitude = 10 * numpy.sin(numpy.pi * numpy.linspace(0,1,num_elem+1))

first = True
start = time.time()
while num_cp <= num_cp_max:

    # define initial altitude profile, as well as fixed profile for
    # x-distance and airspeed
    M_init = numpy.ones(num_cp)*0.82
    x_init = x_range * 1e3 * (1-numpy.cos(numpy.linspace(0, 1, num_cp)*numpy.pi))/2/1e6

    # initialize the mission analysis problem with the framework
    traj = OptTrajectory(num_elem, num_cp, first)
    traj.set_init_h(h_init)
    traj.set_init_M(M_init)
    traj.set_init_x(x_init)
    traj.set_params(params)
    traj.set_folder(folder_path)
    traj.set_name(name)
    traj.setup_MBI()
    traj.set_init_h_pt(altitude)
    main = traj.initialize_framework()

    print main.vec['u'].array.shape[0]

    #start_comp = time.time()
    main.compute(output=True)
    #print 'FINISHED COMPUTING:', time.time() - start_comp
    #exit()

    #print 'computing derivatives'
    #start_comp = time.time()
    #main.compute_derivatives('rev', 'wf_obj', output=True)
    #print 'FINISHED COMPUTING:', time.time() - start_comp
    #exit()

    #main.check_derivatives_all()
    #exit()

    # initialize the trajectory optimization problem using the framework
    # instance initialized before with Optimization.py
    traj.set_gamma_bound(gamma_lb, gamma_ub)
    traj.set_takeoff_speed(takeoff_speed)
    traj.set_landing_speed(landing_speed)
    opt = traj.initialize_opt(main)

    # start timing, and perform optimization
    PROFILE = True

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
        exit()
    else:
        opt('SNOPT')

    run_case, last_itr = traj.history.get_index()
    folder_name = folder_path + name + '_%03i/' % (run_case)
    call (["cp", "./SNOPT_print.out", folder_name + 'SNOPT_%04i_print.out' %(num_cp)])
    call (["mv", "./hist.hst", folder_name + 'hist_%04i.hst' %(num_cp)])
    altitude = main.vec['u']('h')
    num_cp += num_cp_step
    first = False
    
opt_time = time.time() - start
print 'OPTIMIZATION TIME:', opt_time
numpy.savetxt('%inm_%i_time.dat' %(x_range, num_cp), [opt_time])
seconds = main.vec['u']('time') * 1e4
mnt, sec = divmod(seconds, 60)
hrs, mnt = divmod(mnt, 60)
print 'FLIGHT TIME:', '%d:%02d:%02d' % (hrs, mnt, sec)
traj.history.print_max_min(main.vec['u'], params)



