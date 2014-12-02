from __future__ import division

import os.path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from openmdao.lib.casehandlers.api import CaseDataset

from SNOPThistoryReader import SNOPThistoryReader

if __name__ == '__main__':

    length = 9000
    cps = [50, 100, 150, 200, 250, 300, 350]
    times = [60, 175, 484, 1058,
             2179, 9814, 54096]

    NCURVES = 7
    niteration = 1400
    ptmax = 400

    #length = 100
    #cps = [10, 50, 100, 150]
    #times = [32.51, 1223.01, 11967.74, 13463.54]
    #NCURVES = 4
    #niteration = 1600
    #ptmax = 200
    num_lines = range(NCURVES)
    folder_path = 'cp_sweep_data'
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
    lines = []
    opts = []
    feas = []
    sbs = []
    mfs = []

    for i in xrange(NCURVES): 
        cp = cps[i]
        print "loading data from %d cp run" % cp
        name = 'mission_history_cp_%i.bson' % cp 
        
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
        # fuel = np.array(data['SysFuelWeight.fuel_w']) * 1e5

        # thrust = np.array(data['SysCTTar.CT_tar'][-1])*0.5*rho*speed**2*S*1e2 * 1e-1
        # weight = ac_w*1e6 + fuel*1e5

        # lift_c = np.array(data['SysCLTar.CL'][-1])
        # drag_c = np.array(data['SysTripanCDSurrogate.CD'][-1]) * 1e-1
        # gamma = np.array(data['SysGammaBspline.Gamma'][-1]) * 1e-1 * 180/np.pi

        # file_name = os.path.join(folder_path, 'cp_%04i.dat' % cp)
        # file_array = [dist, altitude, speed, alpha, throttle, eta, fuel,
        #               rho, lift_c, drag_c, thrust, gamma, weight,
        #               temp]

        # np.save(file_name, file_array)

        file_name = os.path.join(folder_path, 'cp_%04i.dat.npy' % cp)
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

        name = 'SNOPT_%i_print.out' % cp
        hist = SNOPThistoryReader(os.path.join(folder_path,name)).getDict()
        opts.append(hist['Optimality'])
        feas.append(hist['Feasibility'])
        sbs.append(hist['Superbasics'])
        mfs.append(hist['MeritFunction'])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
        
    #subfig, subax = plt.subplots(2, 1, sharex=True)
    subfig, a = plt.subplots(1, 1, sharex=True, figsize=(8.0, 2.0))
    winter = cm = plt.get_cmap('winter')
    cNorm = colors.Normalize(vmin=0, vmax=NCURVES-1+1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=winter)
        
    line = []
    
    for index in xrange(NCURVES):
        colorVal = scalarMap.to_rgba(num_lines[index])
        
        retLine, = a.plot(dists[index], alts[index]/1e3, color=colorVal, linewidth=0.2)
        lines.append(retLine)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.get_xaxis().tick_bottom()
        a.get_yaxis().tick_left()
        a.set_xticks([0, 9000])
        a.set_xlim([-100, 9100])
        a.set_ylabel(r'Altitude ($10^3$ ft)', rotation='horizontal', fontsize=10, ha="right")
        #a.text(0, 0.5, r'Altitude ($10^3$ ft)', rotation='horizontal', fontsize=10)
        a.set_yticks([25, 37])
        a.set_ylim([23, 40])
        a.set_xlabel(r'Distance (nmi)', fontsize=10)
        for tick in a.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        for tick in a.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        subfig.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.10,
                              wspace=0.05, hspace=0.05)

    plt.tight_layout()
    #subfig.savefig('./plots/SciTech_Refine%i_alt.png' %(length))
    subfig.savefig('./plots/SciTech_Refine%i_alt.pdf' %(length))

    subfig = plt.figure(figsize=(8.0, 4.0))
    
    for index in xrange(NCURVES):
        a = subfig.add_subplot(2, 1, 1)
        colorVal = scalarMap.to_rgba(num_lines[index])
        retLine, = a.semilogy(opts[index], color=colorVal, linewidth=0.2)
        lines.append(retLine)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.get_xaxis().set_ticks_position('none')
        a.get_yaxis().tick_left()
        a.spines['bottom'].set_visible(False)
        a.set_ylabel(r'Optimality', rotation='horizontal', fontsize=10)
        a.set_yticks([1, 1e-3, 1e-6])
        for tick in a.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
    
        a = subfig.add_subplot(2, 1, 2)
        retLine, = a.semilogy(feas[index], color=colorVal, linewidth=0.2)
        lines.append(retLine)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.get_yaxis().tick_left()
        a.set_ylabel(r'Feasibility', rotation='horizontal', fontsize=10)
        a.set_yticks([1, 1e-9, 1e-18])
        a.set_xlabel(r'Iterations', fontsize=10)
        for tick in a.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        for tick in a.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        subfig.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.1,
                               wspace=0.05, hspace=0.05)

    limit = np.ones(niteration) * 2e-6
    a = subfig.add_subplot(2, 1, 1)
    a.semilogy(limit, color='r', linewidth=0.1, alpha=0.2)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.get_xaxis().set_ticks_position('none')
    a.get_yaxis().tick_left()
    a.set_xticks([])
    a.spines['bottom'].set_visible(False)
    a.set_ylabel(r'Optimality', rotation='horizontal', fontsize=10, ha='right')
    a.set_yticks([1, 1e-3, 1e-6])
    for tick in a.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    
    a = subfig.add_subplot(2, 1, 2)
    a.semilogy(limit, color='r', linewidth=0.1, alpha=0.2)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.get_yaxis().tick_left()
    a.get_xaxis().tick_bottom()
    a.set_ylabel(r'Feasibility', rotation='horizontal', fontsize=10, ha='right')
    a.set_yticks([1, 1e-9, 1e-18])
    a.set_xticks([0, niteration/4, niteration/2, 3*niteration/4, niteration])
    a.set_xlabel(r'Iterations', fontsize=10)
    for tick in a.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    for tick in a.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)

    #subfig.savefig('./plots/SciTech_Refine%i_opt.png' %(length))
    plt.tight_layout()
    subfig.savefig('./plots/SciTech_Refine%i_opt.pdf' %(length))

    subfig, a = plt.subplots(1, 1, sharex=True, figsize=(8.0, 2.0))

    for index in xrange(NCURVES):
        colorVal = scalarMap.to_rgba(num_lines[index])
        retLine, = a.semilogy(cps[index], times[index], color=colorVal, marker='x',
                                      linewidth=3)
        line.append(retLine)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.get_xaxis().tick_bottom()
        a.get_yaxis().tick_left()
        a.set_xlim([10, ptmax])
        a.set_yticks([1e2, 1e5])
        a.set_ylim([10, 1e5])
        a.set_ylabel(r'Time (s)', rotation='horizontal', fontsize=10, ha='right')
        a.set_xlabel(r'Number of Control Points', fontsize=10)
        for tick in a.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        for tick in a.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        subfig.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.1,
                               wspace=0.05, hspace=0.05)

    #subfig.savefig('./plots/SciTech_Refine%i_time.png' %(length))
    plt.tight_layout()
    subfig.savefig('./plots/SciTech_Refine%i_time.pdf' %(length))
