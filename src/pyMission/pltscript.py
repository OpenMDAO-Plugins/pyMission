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

import numpy as np
import os
import time
from subprocess import call

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab

from openmdao.lib.casehandlers.api import CaseDataset

###########################
# USER SPECIFIED INPUTS:

#num_elem = 1000
#num_cp_init = 10
#num_cp_max = 110
#num_cp_step = 50
#x_range = 15000.0
#step = 1
#initial_ind = 0
#file_index = 0
#video = True
#folder_path = '/home/jason/Documents/Results/PlotTest_'
fuel_guess = 200000.0

num_elem = 3000
num_cp_init = 10
num_cp_max = 10        # set to 200 for the sweep
num_cp_step = 10
x_range = 15000.0

# END USER SPECIFIED INPUTS
###########################

# initialize figure, set up folder-paths
fig = matplotlib.pylab.figure(figsize=(18.0,8.0))
nr, nc = 4, 3

# Read in the openmdao final dataset.
cds1 = CaseDataset('mission_final_cp_10.bson', 'bson')
final_data = cds1.data.fetch() # results
final_data = final_data[-1] # Get last case. Earlier cases are sub-iterations.

# Constants are stored here
x_init = np.array(cds1.simulation_info['constants']['SysXBspline.x_init'])

# Variables are stored in the cases.
dist = np.array(final_data['SysXBspline.x'])
altitude = np.array(final_data['SysHBspline.h'])
speed = np.array(final_data['SysSpeed.v'])
eta = np.array(final_data['SysAeroSurrogate.eta'])
gamma = np.array(final_data['SysGammaBspline.Gamma'])
temp = np.array(final_data['SysTemp.temp'])
alpha = np.array(final_data['SysCLTar.alpha'])
rho = np.array(final_data['SysRho.rho'])
throttle = np.array(final_data['SysTau.tau'])
lift_c = np.array(final_data['SysAeroSurrogate.CL'])
fuel = np.array(final_data['SysCLTar.fuel_w'])
thrust = np.array(final_data['SysCTTar.CT_tar'])
drag_c = np.array(final_data['SysAeroSurrogate.CD'])

# various scaling params
dist = dist/1e3
mach = speed / np.sqrt(1.4*288.0*temp)
altitude *= 3.28
speed *= 1.94
fuel *= 0.225
thrust *= 0.225

values = [altitude/1e3, speed, eta,
          gamma, mach, alpha,
          rho, throttle, lift_c,
          fuel/1e3, thrust/1e3, drag_c]
labels = ['Altitude (*10^3 ft)', 'TAS (knots)', 'Trim (deg)',
          'Path Angle (deg)', 'Mach Number', 'AoA (deg)',
          'Density (kg/m^3)', 'Throttle', 'C_L',
          'Fuel wt. (10^3 lb)', 'Thrust (10^3 lb)', 'C_D']
limits = [[-1, 51], [100, 600], [-10, 10],
          [-32.0, 32.0], [0.05, 1.2], [-5, 10],
          [0.0, 1.3], [-0.1, 1.1], [0.0, 0.8],
          [-100.0/1e3, fuel_guess/1e3], [0.0, 250.0], [0.01, 0.05]]


fplot = fig.add_subplot
rnd = np.around
fig.clf()

for i in xrange(12):
    fplot(nr, nc, i+1).plot(dist, values[i])
    fplot(nr, nc, i+1).set_ylabel(labels[i])
    #fplot(nr, nc, i+1).set_xlim([-100.0, rnd(x_range, -2)+100.0])
    #fplot(nr, nc, i+1).set_ylim(limits[i])

fplot(nr, nc, 10).set_xlabel('Distance (km)')
fplot(nr, nc, 11).set_xlabel('Distance (km)')
fplot(nr, nc, 12).set_xlabel('Distance (km)')

matplotlib.pylab.show()

print 'done'
