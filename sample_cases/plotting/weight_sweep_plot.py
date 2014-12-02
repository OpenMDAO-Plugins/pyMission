from __future__ import division

import os.path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from openmdao.lib.casehandlers.api import CaseDataset

ac_weights = np.arange(150000, 260000, 10000)*9.81/1e6

folder_path = 'weight_sweep_data'
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

NCURVES = len(ac_weights)
num_lines = range(NCURVES)

for index in xrange(NCURVES):
    print "reading data for %i case" % index
    name = 'mission_history_weight_%i.bson' % index

    # cds = CaseDataset(os.path.join(folder_path, name), 'bson')
    # ac_w = cds.simulation_info['constants']['ac_w']
    # S = cds.simulation_info['constants']['S']

    # data = cds.data.driver('driver').by_variable().fetch()

    # dist = np.array(data['SysXBspline.x'][-1]) * 1e6
    # altitude = np.array(data['SysHBspline.h'][-1]) * 1e3
    # rho = np.array(data['SysRho.rho'][-1]) * 1e2

    # temp = np.array(data['SysTemp.temp'][-1]) * 1e2
    # speed = np.array(data['SysSpeed.v'][-1]) * 1e2

    # alpha = np.array(data['SysTripanCLSurrogate.alpha'][-1]) * 1e-1 * 180/np.pi
    # throttle = np.array(data['SysTau.tau'][-1])
    # eta = np.array(data['SysTripanCMSurrogate.eta'][-1]) * 1e-1 * 180/np.pi
    # fuel = np.array(data['SysFuelWeight.fuel_w'][-1]) * 1e5

    # thrust = np.array(data['SysCTTar.CT_tar'][-1])*0.5*rho*speed**2*S*1e2 * 1e-1
    # weight = ac_weights[index]

    # lift_c = np.array(data['SysCLTar.CL'][-1])
    # drag_c = np.array(data['SysTripanCDSurrogate.CD'][-1]) * 1e-1
    # gamma = np.array(data['SysGammaBspline.Gamma'][-1]) * 1e-1 * 180/np.pi

    # file_name = os.path.join(folder_path, 'weight_%04i' % index)
    # file_array = [dist, altitude, speed, alpha, throttle, eta, fuel,
    #               rho, lift_c, drag_c, thrust, gamma, weight,
    #               temp]

    # np.save(file_name, file_array)

    file_name = os.path.join(folder_path, 'weight_%04i.npy' % index)
    (dist, altitude, speed, alpha, throttle, eta, fuel,
    rho, lift_c, drag_c, thrust, gamma, weight,
    temp) = np.load(file_name)

    dist = dist/(1e3 * 1.852)
    mach = speed / np.sqrt(1.4*288.0*temp)
    altitude *= 3.28
    speed *= 1.94
    fuel *= 0.225
    thrust *= 0.225
    weight *= 0.225
    dists.append(dist)
    alts.append(altitude)
    machs.append(mach)
    speeds.append(speed)
    alphas.append(alpha)
    fuels.append(fuel)
    thrusts.append(thrust)
    weights.append(weight)
    throttles.append(throttle)
    etas.append(eta)
    rhos.append(rho)
    lift_cs.append(lift_c)
    drag_cs.append(drag_c)
    gammas.append(gamma)
    temps.append(temp)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#fig = plt.figure(figsize=(8.0,8.0))
subfig, subax = plt.subplots(8, 1, sharex=True)
winter = cm = plt.get_cmap('winter')
cNorm = colors.Normalize(vmin=0, vmax=NCURVES-1+2)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=winter)

lines = []

for index in xrange(NCURVES):
    dists[index] /= (100.)  # * 1.853)
    values = [alts[index]/1e3,  # speeds[index], 
              gammas[index], machs[index], alphas[index],
              lift_cs[index], lift_cs[index]/drag_cs[index],
              thrusts[index]/1e5, fuels[index]/1e1, throttles[index]]
    labels = [' Alt ($10^3$ ft)',  # 'TAS (knots)',
              'Path Angle (deg)', 'Mach', 'AoA (deg)',
              '$C_L$', '$L/D$',
              'Thrust ($10^3$ lb)', 'Fuel ($10^3$ lb)', 'Throttle']
    limits = [[-5, 45],  # [386, 586],
              [-10, 25], [0.72, 0.87], [-2.5, 2.5],
              [0.05, 0.6], [0, 20],
              [-10, 240], [-20, 300], [-0.1, 1.1]]
    ticks = [[0, 40], [-5, 20], [0.82], [-2.0, 2.0],
             [0.08, 0.5], [5, 15], [0, 220], [0, 270], [0, 1]]
    colorVal = scalarMap.to_rgba(num_lines[index])

    for i, a in enumerate(subax):
        retLine, = a.plot(dists[index], values[i], color=colorVal, linewidth=0.1)
        lines.append(retLine)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        if i < 7:
            a.spines['bottom'].set_visible(False)
            a.get_yaxis().tick_left()
            a.set_xticks([])
        else:
            a.get_yaxis().tick_left()
            #a.set_xticks([0.0, 1.0])
        a.set_ylabel(r'%s' % (labels[i]), rotation='horizontal', fontsize=10, ha='right')
        a.set_xlim(-0.05, 1.05)
        a.set_yticks(ticks[i])
        a.set_ylim(limits[i])
        for tick in a.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        subfig.subplots_adjust(left=0.25, right=0.98, top=0.98, bottom=0.05,
                               wspace=0.05, hspace=0.05)
a.set_xlabel(r'Normalized Range', fontsize=10)
# subfig.savefig('./plots/SciTech_CarpetRange.png')
subfig.savefig('./plots/SciTech_CarpetWeight.pdf')
