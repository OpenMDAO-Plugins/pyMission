from __future__ import division
import numpy
import copy
import os
from framework import *
from optimization import *
from bsplines import *
from atmospherics import *
from coupled_analysis import *
from functionals import *
from aerodynamics import *
from propulsion import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab
import MBI, scipy.sparse

class GlobalizedSystem(SerialSystem):
    ''' doc string '''
    
    def solve_F(self):
        """ Solve f for u, p |-> u """

        kwargs = self.kwargs
        self.solvers['NL']['NLN_GS'](ilimit=kwargs['GL_GS_ilimit'],
                                     atol=kwargs['GL_GS_atol'],
                                     rtol=kwargs['GL_GS_rtol'])
        return self.solvers['NL']['NEWTON'](ilimit=kwargs['GL_NT_ilimit'],
                                            atol=kwargs['GL_NT_atol'],
                                            rtol=kwargs['GL_NT_rtol'])

class Top(SerialSystem):

    def initialize_plotting(self):
        #self.fig = matplotlib.pylab.figure(figsize=(12.0,12.0))
        self.counter = 0

    def initialize_history(self, num_elem, num_cp, x_range):
        self.history = history(num_elem, num_cp, x_range)
        self.hist_counter = 0

    def compute0(self, output=False):
        fig = self.fig
        fig.clf()
        temp, success = super(Top, self).compute(output)
        v = self.vec['u']
        nr, nc = 6, 2
        fig.add_subplot(nr,nc,1).plot(v('x')*1000.0, v('h'))
        fig.add_subplot(nr,nc,1).set_ylabel('Altitude (km)')
        fig.add_subplot(nr,nc,2).plot(v('x')*1000.0, v('v')*1e2)
        fig.add_subplot(nr,nc,2).set_ylabel('Velocity (m/s)')
        fig.add_subplot(nr,nc,3).plot(v('x')*1000.0, v('alpha')*1e-1*180.0/numpy.pi)
        fig.add_subplot(nr,nc,3).set_ylabel('AoA (deg)')
        fig.add_subplot(nr,nc,4).plot(v('x')*1000.0, v('tau'))
        fig.add_subplot(nr,nc,4).set_ylabel('Throttle')
        fig.add_subplot(nr,nc,5).plot(v('x')*1000.0, v('eta')*1e-1*180.0/numpy.pi)
        fig.add_subplot(nr,nc,5).set_ylabel('Trim Angle (deg)')
        fig.add_subplot(nr,nc,6).plot(v('x')*1000.0, v('fuel_w')*1e6/(9.81*0.804))
        fig.add_subplot(nr,nc,6).set_ylabel('Fuel (L)')
        fig.add_subplot(nr,nc,7).plot(v('x')*1000.0, v('rho'))
        fig.add_subplot(nr,nc,7).set_ylabel('rho')
        fig.add_subplot(nr,nc,8).plot(v('x')*1000.0, v('CL_tar'))
        fig.add_subplot(nr,nc,8).set_ylabel('CL_tar')
        fig.add_subplot(nr,nc,9).plot(v('x')*1000.0, v('CD')*0.1)
        fig.add_subplot(nr,nc,9).set_ylabel('CD')
        fig.add_subplot(nr,nc,10).plot(v('x')*1000.0, v('CT_tar')*0.1)
        fig.add_subplot(nr,nc,10).set_ylabel('CT_tar')
        fig.add_subplot(nr,nc,11).plot(v('x')*1000.0, v('gamma')*0.1*180/numpy.pi)
        fig.add_subplot(nr,nc,11).set_ylabel('gamma')
        fig.add_subplot(nr,nc,12).plot(v('x')*1000.0, (v('fuel_w')+v('ac_w'))*1e6/9.81*2.2)
        fig.add_subplot(nr,nc,12).set_ylabel('W (lb)')
        fig.add_subplot(nr,nc,6).set_xlabel('Distance (km)')
        fig.add_subplot(nr,nc,12).set_xlabel('Distance (km)')
        fig.savefig("plots/OptFig_%i.pdf"%(self.counter))
        fig.savefig("plots/OptFig_%i.png"%(self.counter))
        self.counter += 1

        return temp, success

    def compute(self, output=False):
        temp, success = super(Top, self).compute(output)
        self.history.save_history(self.vec['u'])

        return temp, success

class history(object):

    def __init__(self, num_elem, num_cp, x_range):

        self.num_elem = num_elem
        self.num_cp = num_cp
        self.x_range = x_range
        self.folder_name = './dist'+str(int(self.x_range*1e3))+'km-'\
            +str(self.num_cp)+'-'+str(self.num_elem)
        index = 0
        while os.path.exists(self.folder_name+'-'+str(index)):
            index += 1
        self.folder_name = self.folder_name+'-'+str(index)+'/'
        os.makedirs(self.folder_name)

        self.hist_counter = 0

    def save_history(self, vecu):

        dist = vecu('x') * 1e6
        altitude = vecu('h') * 1e3
        speed = vecu('v') * 1e2
        alpha = vecu('alpha') * 1e-1 * 180/numpy.pi
        throttle = vecu('tau')
        eta = vecu('eta') * 1e-1 * 180/numpy.pi
        fuel = vecu('fuel_w') * 1e6
        rho = vecu('rho')
        thrust_c = vecu('CT_tar') * 1e-1
        drag_c = vecu('CD') * 1e-1
        lift_c = vecu('CL')
        gamma = vecu('gamma') * 1e-1 * 180/numpy.pi
        weight = (vecu('ac_w') + vecu('fuel_w')) * 1e6
        temp = vecu('Temp') * 1e2
        SFC = vecu('SFC') * 1e-6

        file_name = str(int(self.x_range*1e3))+'km-'+str(self.num_cp)+'-'\
            +str(self.num_elem)+'-'+str(self.hist_counter)

        output_file = open(self.folder_name+file_name, 'w')

        file_array = [dist, altitude, speed, alpha, throttle, eta, fuel,
                      rho, lift_c, drag_c, thrust_c, gamma, weight,
                      temp, SFC]
        numpy.savetxt(output_file, file_array)

        self.hist_counter += 1

class OptTrajectory(object):
    """ class used to define and setup trajectory optimization problem """

    def __init__(self, num_elem, num_cp):
        self.num_elem = num_elem
        self.num_pt = num_cp
        self.h_pts = numpy.zeros(num_elem+1)
        self.M_pts = numpy.zeros(num_elem+1)
        self.v_pts = numpy.zeros(num_elem+1)
        self.x_pts = numpy.zeros(num_elem+1)
        self.wing_area = 0.0
        self.ac_w = 0.0
        self.thrust_sl = 0.0
        self.sfc_sl = 0.0
        self.aspect_ratio = 0.0
        self.oswald = 0.0
        self.v_specified = 0

    def set_init_h(self, h_init):
        self.h_pts = h_init

    def set_init_M(self, M_init):
        self.M_pts = M_init

    def set_init_x(self, x_init):
        self.x_pts = x_init

    def set_init_v(self, v_init):
        self.v_pts = v_init
        self.v_specified = 1

    def set_params(self, kw):
        self.wing_area = kw['S']
        self.ac_w = kw['ac_w']
        self.thrust_sl = kw['thrust_sl']
        self.sfc_sl = kw['SFCSL']
        self.aspect_ratio = kw['AR']
        self.oswald = kw['e']

    def initialize(self):

        self.main = Top('mission',
                        NL='NLN_GS',
                        LN='LIN_GS',
                        LN_ilimit=1,
                        NL_ilimit=1,
                        NL_rtol=1e-6,
                        NL_atol=1e-10,
                        LN_rtol=1e-6,
                        LN_atol=1e-10,
                        output=True,
                        subsystems=[
                SerialSystem('mission_param',
                             NL='NLN_GS',
                             LN='LIN_GS',
                             LN_ilimit=1,
                             NL_ilimit=1,
                             output=True,
                             subsystems=[
                        IndVar('S', val=self.wing_area, size=1),
                        IndVar('ac_w', val=self.ac_w, size=1),
                        IndVar('thrust_sl', val=self.thrust_sl, size=1),
                        IndVar('SFCSL', val=self.sfc_sl, size=1),
                        IndVar('AR', val=self.aspect_ratio, size=1),
                        IndVar('e', val=self.oswald, size=1),
                        ]),
                SerialSystem('segment',
                             NL='NLN_GS',
                             LN='LIN_GS',
                             LN_ilimit=1,
                             NL_ilimit=1,
                             output=True,
                             subsystems=[
                        SerialSystem('bsplines',
                                     NL='NLN_GS',
                                     LN='LIN_GS',
                                     LN_ilimit=1,
                                     NL_ilimit=1,
                                     output=True,
                                     subsystems=[
                                IndVar('x_pt', val=self.x_pts, lower=0),
                                IndVar('h_pt', val=self.h_pts, lower=0),
                                IndVar('M_pt', val=self.M_pts, lower=0),
                                IndVar('v_pt', val=self.v_pts, lower=0),
                                SysXBspline('x', num_elem=self.num_elem,
                                            num_pt=self.num_pt,
                                            x_init=self.x_pts,
                                            x_0=numpy.linspace(0.0, self.x_pts[-1], self.num_elem+1)),
                                SysHBspline('h', num_elem=self.num_elem,
                                            num_pt=self.num_pt,
                                            x_init=self.x_pts),
                                SysMVBspline('M', num_elem=self.num_elem,
                                             num_pt=self.num_pt,
                                             x_init=self.x_pts),
                                SysGammaBspline('gamma',
                                                num_elem=self.num_elem,
                                                num_pt=self.num_pt,
                                                x_init=self.x_pts),
                                ]),
                        SerialSystem('atmospherics',
                                     NL='NLN_GS',
                                     LN='LIN_GS',
                                     LN_ilimit=1,
                                     NL_ilimit=1,
                                     output=True,
                                     subsystems=[
                                SysSFC('SFC', num_elem=self.num_elem),
                                SysTemp('Temp', num_elem=self.num_elem),
                                SysRho('rho', num_elem=self.num_elem),
                                SysSpeed('v', num_elem=self.num_elem,
                                         v_specified=self.v_specified),
                                ]),
                        GlobalizedSystem('coupled_analysis',
                                         LN='KSP_PC',
                                         PC='LIN_GS',
                                         LN_ilimit=8,
                                         GL_GS_ilimit=5,
                                         GL_NT_ilimit=8,
                                         PC_ilimit=3,
                                         GL_GS_rtol=1e-6,
                                         GL_GS_atol=1e-10,
                                         GL_NT_rtol=1e-14,
                                         GL_NT_atol=1e-14,
                                         LN_rtol=1e-14,
                                         LN_atol=1e-14,
                                         PC_rtol=1e-6,
                                         PC_atol=1e-10,
                                         output=True,
                                         subsystems=[
                                SerialSystem('vert_eqlm',
                                             NL='NLN_GS',
                                             LN='KSP_PC',
                                             LN_ilimit=1,
                                             NL_ilimit=1,
                                             NL_rtol=1e-10,
                                             NL_atol=1e-10,
                                             LN_rtol=1e-10,
                                             LN_atol=1e-10,
                                             subsystems=[
                                        SysCLTar('CL_tar',
                                                 num_elem=self.num_elem),
                                        ]),
                                SerialSystem('drag',
                                             NL='NEWTON',
                                             LN='KSP_PC',
                                             LN_ilimit=15,
                                             NL_ilimit=15,
                                             PC_ilimit=2,
                                             NL_rtol=1e-10,
                                             NL_atol=1e-10,
                                             LN_rtol=1e-10,
                                             LN_atol=1e-10,
                                             subsystems=[
                                        SysAeroSurrogate('CL', num_elem=self.num_elem),
                                        SysAlpha('alpha',
                                                 num_elem=self.num_elem),
                                        ]),
                                SerialSystem('hor_eqlm',
                                             NL='NLN_GS',
                                             LN='KSP_PC',
                                             LN_ilimit=1,
                                             NL_ilimit=1,
                                             NL_rtol=1e-10,
                                             NL_atol=1e-10,
                                             LN_rtol=1e-10,
                                             LN_atol=1e-10,
                                             subsystems=[
                                        SysCTTar('CT_tar',
                                                 num_elem=self.num_elem),
                                        ]),
                                SerialSystem('mmt_eqlm',
                                             NL='NLN_GS',
                                             LN='KSP_PC',
                                             LN_ilimit=1,
                                             NL_ilimit=1,
                                             NL_rtol=1e-10,
                                             NL_atol=1e-10,
                                             LN_rtol=1e-10,
                                             LN_atol=1e-10,
                                             subsystems=[
                                        SysCM('eta', num_elem=self.num_elem),
                                        ]),
                                SerialSystem('weight',
                                             NL='NLN_GS',
                                             LN='KSP_PC',
                                             LN_ilimit=1,
                                             NL_ilimit=1,
                                             NL_rtol=1e-10,
                                             NL_atol=1e-10,
                                             LN_rtol=1e-10,
                                             LN_atol=1e-10,
                                             subsystems=[
                                        SysFuelWeight('fuel_w',
                                                      num_elem=self.num_elem,
                                                      fuel_w_0=numpy.linspace(1.0, 0.0, self.num_elem+1)),
                                        ]),
                                ]),
                        SerialSystem('functionals',
                                     NL='NLN_GS',
                                     LN='LIN_GS',
                                     NL_ilimit=1,
                                     LN_ilimit=1,
                                     NL_rtol=1e-10,
                                     NL_atol=1e-10,
                                     LN_rtol=1e-10,
                                     LN_ato1=1e-10,
                                     output=True,
                                     subsystems=[
                                SysTau('tau', num_elem=self.num_elem),
                                SysFuelObj('wf_obj'),
                                SysHi('h_i'),
                                SysHf('h_f', num_elem=self.num_elem),
                                SysTmin('Tmin', num_elem=self.num_elem),
                                SysTmax('Tmax', num_elem=self.num_elem),
                                SysSlopeMin('gamma_min',
                                            num_elem=self.num_elem),
                                SysSlopeMax('gamma_max',
                                            num_elem=self.num_elem),
                                ]),
                        ]),
                 ]).setup()
        #self.main.initialize_plotting()
        self.main.initialize_history(self.num_elem, self.num_pt,
                                     self.x_pts[-1])

        return self.main
