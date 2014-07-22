import numpy
import os
import time
from subprocess import call
from history import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab

# USER SPECIFIED INPUTS:

num_elem = 100
num_cp = 30
x_range = 5000.0
step = 1
initial_ind = 0
file_index = 0
video = True

# END USER SPECIFIED INPUTS

fig = matplotlib.pylab.figure(figsize=(18.0,8.0))
index = initial_ind
folder_name = '/home/jason/Documents/Results/dist'+str(int(x_range))+'km-'\
    +str(num_cp)+'-'+str(num_elem)+'-'+str(file_index)+'/'
if not os.path.exists(folder_name):
    raise('ERROR: SPECIFIED CASE DOES NOT EXIST')

max_name = str(int(x_range))+'km-'+str(num_cp)+\
    '-'+str(num_elem)+'-maxmin.dat'
rnd = numpy.around
fplot = fig.add_subplot
file_name = '%ikm-%i-%i-%04i' % (int(x_range),
                                 num_cp,
                                 num_elem,
                                 index)

while ((not os.path.isfile(folder_name+max_name))
       or (os.path.isfile(folder_name+file_name+'.dat'))):
    if os.path.isfile(folder_name+'fig-'+file_name+'.png'):
        index += step
        file_name = '%ikm-%i-%i-%04i' % (int(x_range),
                                         num_cp,
                                         num_elem,
                                         index)
    else:
        if os.path.isfile(folder_name+file_name+'.dat'):

            [dist, altitude, speed, alpha, throttle, eta, fuel,
             rho, lift_c, drag_c, thrust, gamma, weight, temp,
             SFC] = numpy.loadtxt(folder_name+file_name+'.dat')
            dist = dist/1e3
            mach = speed / numpy.sqrt(1.4*288.0*temp)
            altitude *= 3.28
            speed *= 1.94
            fuel *= 0.225
            thrust *= 0.225
            weight *= 0.225

            print 'Printing fig: ', folder_name+file_name+'...'
            fig.clf()
            nr, nc = 4, 3
            fplot = fig.add_subplot
            fplot(nr, nc, 1).plot(dist, altitude/1e3)
            fplot(nr, nc, 1).set_ylabel('Altitude (*10^3 ft)')
            fplot(nr, nc, 1).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 1).set_ylim([-1, 51])
            fplot(nr, nc, 2).plot(dist, speed)
            fplot(nr, nc, 2).set_ylabel('Airspeed (knots)')
            fplot(nr, nc, 2).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 2).set_ylim([100, 600])
            fplot(nr, nc, 6).plot(dist, alpha)
            fplot(nr, nc, 6).set_ylabel('AoA (deg)')
            fplot(nr, nc, 6).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 6).set_ylim([-5, 10])
            fplot(nr, nc, 5).plot(dist, mach)
            fplot(nr, nc, 5).set_ylabel('Mach Number')
            fplot(nr, nc, 5).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 5).set_ylim([0.05, 1.2])
            fplot(nr, nc, 8).plot(dist, throttle)
            fplot(nr, nc, 8).set_ylabel('Throttle')
            fplot(nr, nc, 8).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 8).set_ylim([-0.1, 1.1])
            fplot(nr, nc, 3).plot(dist, eta)
            fplot(nr, nc, 3).set_ylabel('Trim Angle (deg)')
            fplot(nr, nc, 3).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 3).set_ylim([-10, 10])
            fplot(nr, nc, 10).plot(dist, fuel)
            fplot(nr, nc, 10).set_ylabel('Fuel Weight (lb)')
            fplot(nr, nc, 10).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 10).set_ylim([-100.0, 800000.0])
            fplot(nr, nc, 7).plot(dist, rho)
            fplot(nr, nc, 7).set_ylabel('Density (kg/m^3)')
            fplot(nr, nc, 7).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 7).set_ylim([0.0, 1.3])
            fplot(nr, nc, 9).plot(dist, lift_c)
            fplot(nr, nc, 9).set_ylabel('Lift Coef')
            fplot(nr, nc, 9).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 9).set_ylim([0.0, 0.8])
            fplot(nr, nc, 12).plot(dist, drag_c)
            fplot(nr, nc, 12).set_ylabel('Drag Coef')
            fplot(nr, nc, 12).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 12).set_ylim([0.01, 0.05])
            fplot(nr, nc, 10).set_xlabel('Distance (km)')
            fplot(nr, nc, 11).plot(dist, thrust/1e3)
            fplot(nr, nc, 11).set_ylabel('Thrust (10^3 lb)')
            fplot(nr, nc, 11).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 11).set_ylim([0.0, 100.0])
            fplot(nr, nc, 11).set_xlabel('Distance (km)')
            fplot(nr, nc, 4).plot(dist, gamma)
            fplot(nr, nc, 4).set_ylabel('Path Angle (deg)')
            fplot(nr, nc, 4).set_xlim([-100.0, rnd(x_range, -3)+100.0])
            fplot(nr, nc, 4).set_ylim([-10.0, 10.0])
            fplot(nr, nc, 12).set_xlabel('Distance (km)')
            fig.savefig(folder_name+'fig-'+file_name+'.png')

            index += 1
            file_name = '%ikm-%i-%i-%04i' % (int(x_range),
                                             num_cp,
                                             num_elem,
                                             index)

        else:
            time.sleep(0.1)

if video == True:
    file_name = '%ikm-%i-%i' % (int(x_range),
                                     num_cp,
                                     num_elem)
    call(["mencoder", "mf://"+folder_name+'fig-*.png', "-mf", 
          "fps=5:type=png", "-ovc", "x264", "-x264encopts", 
          "bitrate=15000", "-o", folder_name+file_name+".avi"])


