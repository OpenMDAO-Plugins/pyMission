from __future__ import division
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from SNOPThistoryReader import SNOPThistoryReader

if __name__=="__main__":

    lengths = [100, 200, 300, 400, 500, 750, 1000, 1500, 2000,
               2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000,
               6500, 7000, 7500, 8000, 8500, 9000]
    num_missions = 23
    num_lines = range(num_missions)
    opts = []
    feas = []
    sbs = []
    mfs = []

    for length in lengths:
        filename = 'range_sweep_data/SNOPT_%i_print.out' %(length)

        hist = SNOPThistoryReader(filename).getDict()
        opts.append(hist['Optimality'])
        feas.append(hist['Feasibility'])
        sbs.append(hist['Superbasics'])
        mfs.append(hist['MeritFunction'])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    subfig, subax = plt.subplots(2, 1, sharex=True, figsize=(10,5))
    winter = cm = plt.get_cmap('winter')
    cNorm = colors.Normalize(vmin=0, vmax=num_missions-1+2)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=winter)

    lines = []

    for index in xrange(num_missions):
        colorVal = scalarMap.to_rgba(num_lines[index])
        values = [opts[index], feas[index], sbs[index], mfs[index]]
        labels = ['Optimality', 'Feasibility',
                  'Num of Superbasic', 'Merit Function']
        ticks = [[1, 1e-3, 1e-6], [1, 1e-5, 1e-10]]

        for i, a in enumerate(subax):
            retLine, = a.semilogy(values[i], color=colorVal, linewidth=0.1)
            lines.append(retLine)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.get_yaxis().tick_left()
            if i < 1:
                a.spines['bottom'].set_visible(False)
                a.get_xaxis().set_ticks_position('none')
            a.set_ylabel(r'%s'%(labels[i]), rotation='horizontal', fontsize=10, position=(0,.4))
            a.set_yticks(ticks[i])
            for tick in a.yaxis.get_major_ticks():
                tick.label.set_fontsize(10)
            subfig.subplots_adjust(left=0.15, right=0.98, bottom=0.10,
                                   wspace=0.05, hspace=0.05)

    limit = numpy.ones(150) * 2e-6
    for i, a in enumerate(subax):
        a.semilogy(limit, color='r', linewidth=0.1, alpha=0.2)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.get_yaxis().tick_left()
        if i < 1:
            a.spines['bottom'].set_visible(False)
            a.get_xaxis().set_ticks_position('none')
        a.set_ylabel(r'%s'%(labels[i]), rotation='horizontal', fontsize=10, position=(0,.4))
        a.set_yticks(ticks[i])
        for tick in a.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        subfig.subplots_adjust(left=0.15, right=0.98, bottom=0.10,
                               wspace=0.05, hspace=0.05)
    a.set_xlabel(r'Iterations', fontsize=10)
    a.get_xaxis().tick_bottom()
    # subfig.savefig('./plots/SciTech_Optimization_history.png')
    subfig.savefig('./plots/SciTech_Optimization_history.pdf')
