from mission import *
from history import *
import time
from subprocess import call

# USER SPECIFIED DATA

params = {
    'S': 427.8/1e2,
    'ac_w': 210000*9.81/1e6,
    'thrust_sl': 1020000.0/1e6/3,
    'SFCSL': 40,#8.951,
    'AR': 8.68,
    'e': 0.8,
    }

num_elem = 75
num_cp = 15
x_range = 1000.0e3
folder_name = '/home/jason/Documents/Results/'

# END USER SPECIFIED DATA

v_init = numpy.ones(num_cp)*2.3
#x_init = numpy.linspace(0.0, x_range, num_cp)/1e6
x_init = x_range * (1-numpy.cos(numpy.linspace(0, 1, num_cp)*numpy.pi))/2/1e6

h_init = 1 * numpy.sin(numpy.pi * x_init / (x_range/1e6))

gamma_lb = numpy.tan(-20.0 * (numpy.pi/180.0))/1e-1
gamma_ub = numpy.tan(20.0 * (numpy.pi/180.0))/1e-1

traj = OptTrajectory(num_elem, num_cp)
traj.set_init_h(h_init)
traj.set_init_v(v_init)
traj.set_init_x(x_init)
traj.set_params(params)
traj.set_folder_name(folder_name)
main = traj.initialize()

main.compute(True)

if 0:
    v = main.vec['u']
    FD = numpy.zeros(num_elem)
    for i in xrange(num_elem):
        FD[i] = (v('h')[i+1] - v('h')[i])*1e3 / ((v('x')[i+1] - v('x')[i])*1e6)
        print FD[i] - v('gamma')[i] * 1e-1

    fig = matplotlib.pylab.figure()
    fig.add_subplot(3,1,1).plot(v('x')*1000.0, v('h'))
    fig.add_subplot(3,1,1).set_ylabel('Altitude (km)')
    fig.add_subplot(3,1,2).plot(v('x')*1000.0, v('gamma')*1e-1)
    fig.add_subplot(3,1,2).set_ylabel('Flight Path Angle')
    fig.add_subplot(3,1,3).plot(v('x')[0:-1]*1000.0, FD)
    fig.add_subplot(3,1,3).set_ylabel('Flight Path Angle')
    fig.savefig("test.png")
    exit()

if 0:
    # derivatives check #
    main.check_derivatives_all2()
    exit()

opt = Optimization(main)
opt.add_design_variable('h_pt', value=h_init, lower=0.0, upper=20.0)
opt.add_objective('wf_obj')
opt.add_constraint('h_i', lower=0.0, upper=0.0)
opt.add_constraint('h_f', lower=0.0, upper=0.0)
opt.add_constraint('Tmin', upper=0.0)
opt.add_constraint('Tmax', upper=0.0)
#opt.add_constraint('gamma_min', upper=0.0)
#opt.add_constraint('gamma_max', upper=0.0)
opt.add_constraint('gamma', lower=gamma_lb, upper=gamma_ub)
start = time.time()
opt('SNOPT')
print 'OPTIMIZATION TIME', time.time() - start
main.history.print_max_min(main.vec['u'])
run_case, last_itr = main.history.get_index()


v = main.vec['u']
fig = matplotlib.pylab.figure()
fig.add_subplot(1,1,1).plot(v('x')*1e3, v('h'), 'ob')
fig.add_subplot(1,1,1).plot(v('x_pt')*1e3, v('h_pt'), '+r')
fig.savefig('cp_test.png')

folder_name = '/home/jason/Documents/Results/dist'+str(int(x_range/1e3))\
    +'km-'+str(num_cp)+'-'+str(num_elem)+'-'+str(run_case)+'/.'
call (["mv", "./SNOPT_print.out", folder_name])
call (["mv", "./cp_test.png", folder_name])
exit()
print 'PLOTTING---------'

output_plt = Plotting(num_elem, num_cp, x_range, run_case)
output_plt.plot_history([last_itr])
exit()


