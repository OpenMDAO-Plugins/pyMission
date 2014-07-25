""" This is used to generate the pickle for testing the OpenMDAO version vs
the CMF version.
"""

from mission import *
from history import *
import time
from subprocess import call

params = {
    'S': 427.8/1e2,
    'ac_w': 210000*9.81/1e6,
    'thrust_sl': 1020000.0/1e6/3,
    'SFCSL': 8.951,
    'AR': 8.68,
    'e': 0.8,
    }

num_elem = 100
num_cp = 30
x_range = 150.0

#h_init = numpy.ones(num_cp)*0.000005
#h_init[0] = 0.0
#h_init[-1] = 0.0

v_init = numpy.ones(num_cp)*2.3
x_init = x_range * 1e3 * (1-numpy.cos(numpy.linspace(0, 1, num_cp)*numpy.pi))/2/1e6
h_init = 1 * numpy.sin(numpy.pi * x_init / (x_range/1e3))

gamma_lb = numpy.tan(-20.0 * (numpy.pi/180.0))/1e-1
gamma_ub = numpy.tan(20.0 * (numpy.pi/180.0))/1e-1

traj = OptTrajectory(num_elem, num_cp)
traj.set_init_h(h_init)
traj.set_init_v(v_init)
traj.set_init_x(x_init)
traj.set_params(params)
traj.set_folder_name('.')
main = traj.initialize_framework()

from time import time
t1 = time()
main.compute(True)
print "Elapsed time:", time()-t1

print 'done'

#exit()

keys = main.vec['u'].keys()
data = {}
for key in keys:
    data[key[0]] = main.vec['u'][key]

import pickle
pickle.dump( data, open( "analysis.p", "wb" ) )

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

