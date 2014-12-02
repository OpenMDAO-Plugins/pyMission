from __future__ import division

import os.path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from openmdao.lib.casehandlers.api import CaseDataset

NCURVES = 1
num_lines = range(NCURVES)
folder_path = 'opt_alt_and_mach'
dists = []
alts = []
machs = []
speeds = []
alphas = []
fuels = []
thrusts = []
weights = []
throttles = []
etas = []
rhos = []
lift_cs = []
drag_cs = []
gammas = []
temps = []

num_plts = 23


name = 'opt_alt_and_mach_history.bson'

cds = CaseDataset(os.path.join(folder_path, name), 'bson')

ac_w = cds.simulation_info['constants']['ac_w']
S = cds.simulation_info['constants']['S']
# print [n for n in cds.simulation_info['constants'].keys() if 'alpha' in n]
# print cds.simulation_info['constants']['coupled_solver.alpha']
# exit() # ['SysAeroSurrogate.alpha']

data = cds.data.driver('driver').by_variable().fetch()

n_cases = len(data['SysXBspline.x'])

for i in xrange(n_cases): 
    print "case %d" % i
    index = i
    dist = np.array(data['SysXBspline.x'][index]) * 1e6
    altitude = np.array(data['SysHBspline.h'][index]) * 1e3
    rho = np.array(data['SysRho.rho'][index]) * 1e2

    temp = np.array(data['SysTemp.temp'][index]) * 1e2
    speed = np.array(data['SysSpeed.v'][index]) * 1e2

    alpha = np.array(data['SysTripanCLSurrogate.alpha'][index]) * 1e-1 * 180/np.pi
    throttle = np.array(data['SysTau.tau'][index])

    eta = np.array(data['SysTripanCMSurrogate.eta'][index]) * 1e-1 * 180/np.pi
    fuel = np.array(data['SysFuelWeight.fuel_w'][index]) * 1e5

    thrust = np.array(data['SysCTTar.CT_tar'][index])*0.5*rho*speed**2*S*1e2 * 1e-1
    weight = ac_w*1e6 + fuel*1e5

    lift_c = np.array(data['SysCLTar.CL'][index])
    drag_c = np.array(data['SysTripanCDSurrogate.CD'][index]) * 1e-1

    gamma = np.array(data['SysGammaBspline.Gamma'][index]) * 1e-1 * 180/np.pi

    # file_name = os.path.join(folder_path, '737')
    # file_array = [dist, altitude, speed, alpha, throttle, eta, fuel,
    #               rho, lift_c, drag_c, thrust, gamma, weight,
    #               temp]

    # np.save(file_name, file_array)

    # file_name = os.path.join(folder_path, '737.npy')
    # (dist, altitude, speed, alpha, throttle, eta, fuel,
    # rho, lift_c, drag_c, thrust, gamma, weight,
    # temp) = np.load(file_name)


    dist = dist/(1e3 * 1.852)
    mach = speed / np.sqrt(1.4*288.0*temp)
    altitude *= 3.28
    speed *= 1.94
    fuel *= 0.225
    thrust *= 0.225
    weight *= 0.225


    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    #fig = plt.figure(figsize=(8.0,8.0))
    subfig, subax = plt.subplots(8, 1, sharex=True)


    values = [altitude/1e3, # speeds[index], 
              gamma, mach, alpha,
              lift_c, lift_c/drag_c,
              thrust/1e5, fuel/1e3, throttle]
    labels = [' Alt ($10^3$ ft)', # 'TAS (knots)',
              'Path Angle (deg)', 'Mach', 'AoA (deg)',
              '$C_L$', '$L/D$',
              'Thrust ($10^3$ lb)', 'Fuel ($10^3$ lb)', 'Throttle']
    limits = [[-5, 28], # [386, 586],
              [-4, 23], [.1, 0.88], [-3, 5],
              [0.1, 0.7], [3, 32],
              [-1, 228], [-2, 22], [-0.1, 1.1]]
    ticks = [[0, 25], [-3, 22], [.2, .85], [-2, 4],
             [0.15, 0.65], [5, 30], [0, 225], [0, 20], [0, 1]]


    for i, a in enumerate(subax):

        a.plot(dist, values[i], color='b', linewidth=1)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        if i < 7:
            a.spines['bottom'].set_visible(False)
            a.get_yaxis().tick_left()
            #a.set_xticks([])
            a.get_xaxis().set_visible(False)
        else:
            a.get_yaxis().tick_left()
            a.set_xticks([0.0, 1000])
        a.set_ylabel(r'%s' % (labels[i]), rotation='horizontal', fontsize=10, ha='right')
        a.set_xlim(-50, 1025)
        a.set_yticks(ticks[i])
        a.set_ylim(limits[i])
        for tick in a.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        subfig.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.08,
                               wspace=0.05, hspace=0.1)

    a.set_xlabel(r'Range', fontsize=10)
    subfig.savefig('./plots/SciTech_mach_alt_opt_%04d.png' % index)

# subfig.savefig('./plots/SciTech_mach_alt_opt.pdf')
