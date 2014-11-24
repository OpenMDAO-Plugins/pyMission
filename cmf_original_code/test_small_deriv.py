""" This is used to generate the pickle for testing the OpenMDAO version vs
the CMF version.
"""
import numpy as np

from mission import *
from history import *
import time
from subprocess import call

params = {
    'S': 427.8/1e2,
    'ac_w': 210000*9.81/1e6,
    'thrust_sl': 1020000.0/1e6,
    'SFCSL': 8.951*9.81,
    'AR': 8.68,
    'e': 0.8,
    't_c': 0.09,
    'sweep': 31.6 * np.pi/180,
    'surr': './crm_surr',
}
folder_path = '.'
name = 'junk'

num_elem = 2
num_cp = 2
x_range = 9000.0

# for debugging only
#num_elem = 6
#num_cp = 3

x_range *= 1.852
x_init = x_range * 1e3 * (1-numpy.cos(numpy.linspace(0, 1, num_cp)*numpy.pi))/2/1e6
h_init = 10 * numpy.sin(numpy.pi * x_init / (x_range/1e3))
M_init = numpy.ones(num_cp)*0.8

gamma_lb = numpy.tan(-20.0 * (numpy.pi/180.0))/1e-1
gamma_ub = numpy.tan(20.0 * (numpy.pi/180.0))/1e-1

altitude = numpy.zeros(num_elem+1)
altitude = 10 * numpy.sin(numpy.pi * numpy.linspace(0,1,num_elem+1))

traj = OptTrajectory(num_elem, num_cp, first=True)
traj.set_init_h(h_init)
traj.set_init_M(M_init)
traj.set_init_x(x_init)
traj.set_params(params)
traj.set_folder(folder_path)
traj.set_name(name)
traj.setup_MBI()
traj.set_init_h_pt(altitude)
main = traj.initialize_framework()

from time import time
t1 = time()
main.compute(True)
print "Elapsed time:", time()-t1

print 'done'

# Derivative checking stuff.
# ------------------------------
#main.check_derivatives_all()
#print main.compute_derivatives('fwd', 'h_pt', output=False)
#print main.compute_derivatives('rev', 'h_pt', output=False)
#print 'fwd', main.compute_derivatives('fwd', 'h_pt', output=False)[0][('CL_tar', 0)][0]
#print 'rev', main.compute_derivatives('rev', 'h_pt', output=False)[0][('CL_tar', 0)][0]
import pickle
data = {}
for key in ['fuelburn', 'h_i', 'h_f', 'Tmin', 'Tmax']:
    data[key] = {}
    grad, _ = main.compute_derivatives('rev', key, output=False)
    print grad[('h_pt', 0)]
    for item in grad.keys():
        key2 = item[0]
        data[key][key2] = grad[item].copy()
        
data['gamma'] = {}
for index in range(0, num_elem):
    data['gamma'][index] = {}
    grad, _ = main.compute_derivatives('rev', 'gamma', ind=index, output=False)
    for item in grad.keys():
        key2 = item[0]
        data['gamma'][index][key2] = grad[item].copy()

pickle.dump( data, open( "derivs2.p", "wb" ) )

exit()

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

