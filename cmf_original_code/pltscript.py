"""
MISSION ANALYSIS/TRAJECTORY OPTIMIZATION
This is the runscript used for plotting the history of the trajectory 
optimization problem. The history plotting can be done simultaneously as
the trajectory optimization runscript is being ran. At the end of the
figure generation, a video of the history is also produced.
The mission analysis and trajectory optimization tool was developed by:
    Jason Kao*
    John Hwang*

* University of Michigan Department of Aerospace Engineering,
  Multidisciplinary Design Optimization lab
  mdolab.engin.umich.edu

copyright July 2014
"""

import numpy
import os
import time
from subprocess import call
from history import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab

###########################
# USER SPECIFIED INPUTS:

num_elem = 3000
num_cp_init = 10
num_cp_max = 110
num_cp_step = 100
x_range = 5500.0
step = 1
initial_ind = 0
file_index = 1
video = True
folder_path = '/home/jason/Documents/Results/TIME-OptTest_'
fuel_guess = 200000.0

# END USER SPECIFIED INPUTS
###########################

# initialize figure, set up folder-paths
fig = matplotlib.pylab.figure(figsize=(18.0,8.0))
index = initial_ind

# determine folder path
name = '%inm_i%i_d%i_f%i_p%i' % (int(x_range),
                                 num_cp_init,
                                 num_cp_step,
                                 num_cp_max,
                                 num_elem)

folder_name = folder_path + name + '_%03i/' % (file_index)

if not os.path.exists(folder_name):
    print folder_name
    raise Exception('ERROR: SPECIFIED CASE DOES NOT EXIST')

num_cp = num_cp_init
rnd = numpy.around
fplot = fig.add_subplot
max_name = name + '_maxmin.dat'
file_name = name + '_%04i_%04i' % (num_cp, index)
next_file_name = name + '_%04i_%04i' % (num_cp+num_cp_step, 0)
sleep = False
nr, nc = 4, 3

# continues loop for figure generation until BOTH end file (*-maxmin.dat)
# has been found AND the next .dat file doesn't exist
while ((not os.path.isfile(folder_name+max_name))
       or (os.path.isfile(folder_name+file_name+'.dat'))
       or (os.path.isfile(folder_name+next_file_name+'.dat'))):
        
    # skip figure generation if a corresponding figure exists already
    if os.path.isfile(folder_name+'fig-'+file_name+'.png'):
        index += step
        file_name = name + '_%04i_%04i' % (num_cp, index)

    # reads data file and save figure if the next file is found
    else:
        if os.path.isfile(folder_name+file_name+'.dat'):

            # this delay is necesesary to prevent the script from reading
            # the data file before history.py is done writing it
            if sleep == True:
                time.sleep(0.2)
                sleep = False

            [dist, altitude, speed, alpha, throttle, eta, fuel,
             rho, lift_c, drag_c, thrust, gamma, weight, temp,
             SFC] = numpy.loadtxt(folder_name+file_name+'.dat')
            dist = dist/(1e3 * 1.852)
            mach = speed / numpy.sqrt(1.4*288.0*temp)
            altitude *= 3.28
            speed *= 1.94
            fuel *= 0.225
            thrust *= 0.225
            weight *= 0.225

            print 'Printing fig: ', folder_name+file_name+'...'
            fig.clf()

            values = [altitude/1e3, speed, eta, 
                      gamma, mach, alpha,
                      rho, throttle, lift_c,
                      fuel/1e3, thrust/1e3, drag_c]
            labels = ['Altitude (*10^3 ft)', 'TAS (knots)', 'Trim (deg)',
                      'Path Angle (deg)', 'Mach Number', 'AoA (deg)',
                      'Density (kg/m^3)', 'Throttle', 'C_L',
                      'Fuel wt. (10^3 lb)', 'Thrust (10^3 lb)', 'C_D']
            limits = [[-1, 75], [100, 1000], [-10, 10],
                      [-32.0, 32.0], [0.05, 1.7], [-5, 10],
                      [0.0, 1.3], [-0.1, 1.1], [0.0, 0.8],
                      [-100.0/1e3, fuel_guess/1e3], [0.0, 250.0], [0.01, 0.05]]

            fplot = fig.add_subplot
            for i in xrange(12):
                fplot(nr, nc, i+1).plot(dist, values[i])
                fplot(nr, nc, i+1).set_ylabel(labels[i])
                fplot(nr, nc, i+1).set_xlim([-100.0, rnd(x_range, -2)+100.0])
                fplot(nr, nc, i+1).set_ylim(limits[i])

            fplot(nr, nc, 10).set_xlabel('Distance (nm)')
            fplot(nr, nc, 11).set_xlabel('Distance (nm)')
            fplot(nr, nc, 12).set_xlabel('Distance (nm)')
            fig.savefig(folder_name+'fig-'+file_name+'.png')

            index += 1
            file_name = name + '_%04i_%04i' % (num_cp, index)

        elif os.path.isfile(folder_name+next_file_name+'.dat'):
            num_cp += num_cp_step
            index = 0
            file_name = name + '_%04i_%04i' % (num_cp, index)
            next_file_name = name + '_%04i_%04i' % (num_cp+num_cp_step, 0)

        # The next data file hasn't been written yet, so wait until it
        # exists
        else:
            sleep = True
            time.sleep(0.1)
index -= 1
[v_min, v_max] = numpy.loadtxt(folder_name+name+'_maxmin.dat')
v_min[15] *= 3.28
v_max[15] *= 3.28
v_min[15] /= 1e3
v_max[15] /= 1e3
fplot(nr, nc, 1).plot(dist, numpy.ones(num_elem+1)*v_min[15], ':r')
fplot(nr, nc, 1).plot(dist, numpy.ones(num_elem+1)*v_max[15], ':r')
file_name = name + '_%04i_%04i' % (num_cp, index)
fig.savefig(folder_name+'fig-'+file_name+'.pdf')

# generate video of history from the figures
if video == True:
    file_name = name
    call(["mencoder", "mf://"+folder_name+'fig-*.png', "-mf", 
          "fps=10:type=png", "-ovc", "x264", "-x264encopts", 
          "bitrate=15000", "-o", folder_name+file_name+".avi"])


