"""
INTENDED FOR MISSION ANALYSIS USE
This file contains the segment assembly.

The mission analysis and trajectory optimization tool was developed by:
    Jason Kao*
    John Hwang*

* University of Michigan Department of Aerospace Engineering,
  Multidisciplinary Design Optimization lab
  mdolab.engin.umich.edu

copyright July 2014
"""

# pylint: disable=E1101
import numpy as np
import scipy.sparse.linalg

from openmdao.lib.drivers.api import NewtonSolver, FixedPointIterator, BroydenSolver
from openmdao.main.api import Assembly, set_as_top, Driver
from openmdao.main.datatypes.api import Array, Float

from pyMission.aeroTripan import SysTripanCDSurrogate, SysTripanCLSurrogate, \
                                 SysTripanCMSurrogate, setup_surrogate
from pyMission.atmospherics import SysTemp, SysRho, SysSpeed
from pyMission.bsplines import SysXBspline, SysHBspline, SysMVBspline, \
                               SysGammaBspline, setup_MBI
from pyMission.coupled_analysis import SysCLTar, SysCTTar, SysFuelWeight
from pyMission.functionals import SysTmin, SysTmax, SysSlopeMin, SysSlopeMax, \
                                  SysFuelObj, SysBlockTime 
from pyMission.propulsion import SysSFC, SysTau

import check_deriv_patch
import unconn_patch


def is_differentiable(self): 
    return True
Driver.is_differentiable = is_differentiable


class MissionSegment(Assembly):
    """ Defines a single segment for the Mission Analysis. """


    def __init__(self, num_elem=10, num_cp=5, x_pts=None, params_file=None,
                 aero_surr=None, jac_h=None, jac_gamma=None):
        """Initialize this segment trajectory problem.

        num_elem: int
            number of computations points in the mission profile

        num_cp: int
            number of control points for the splines

        x_pts: 1d array
            array containing the x locations of the spline control points.

        surr_file: Name of file for generating the Tripan surrogate models.
        """

        super(MissionSegment, self).__init__()

        self.num_elem = num_elem
        self.num_pt = num_cp
        self.x_pts = x_pts

        data = {}
        execfile(params_file, data)
        params = data['params']
        self.S = params['S']
        self.ac_w = params['ac_w']
        self.thrust_sl = params['thrust_sl']
        self.SFCSL = params['SFCSL']
        self.AR = params['AR']
        self.oswald = params['e']
        self.wt_pax = 84 * 9.81

        # Generate jacobians for b-splines using MBI package
        self.jac_h, self.jac_gamma = jac_h, jac_gamma
        #self.jac_h, self.jac_gamma = setup_MBI(num_elem+1, num_cp, x_pts)

        # Setup the surrogate models
        self.CL_arr = aero_surr['CL']
        self.CD_arr = aero_surr['CD']
        self.CM_arr = aero_surr['CM']
        self.num = aero_surr['nums']

#    def configure(self):
#        """ Set it all up. """

        self.add('pax_flt', Float(0.0, iotype='in'))

        # Splines
        self.add('SysXBspline', SysXBspline(num_elem=self.num_elem,
                                            num_pt=self.num_pt,
                                            x_init=self.x_pts,
                                            jac_h=self.jac_h))
        self.SysXBspline.x_pt = self.x_pts

        self.add('SysHBspline', SysHBspline(num_elem=self.num_elem,
                                            num_pt=self.num_pt,
                                            x_init=self.x_pts,
                                            jac_h=self.jac_h))

        self.add('SysMVBspline', SysMVBspline(num_elem=self.num_elem,
                                            num_pt=self.num_pt,
                                            x_init=self.x_pts,
                                            jac_h=self.jac_h))

        self.add('SysGammaBspline', SysGammaBspline(num_elem=self.num_elem,
                                            num_pt=self.num_pt,
                                            x_init=self.x_pts,
                                            jac_gamma=self.jac_gamma))



        # Atmospherics
        self.add('SysSFC', SysSFC(num_elem=self.num_elem, SFCSL=self.SFCSL))
        self.add('SysTemp', SysTemp(num_elem=self.num_elem))
        self.add('SysRho', SysRho(num_elem=self.num_elem))
        self.add('SysSpeed', SysSpeed(num_elem=self.num_elem))
        self.SysSpeed.v_specified = False

        self.connect('SysHBspline.h', 'SysSFC.h')
        self.connect('SysHBspline.h', 'SysTemp.h')
        self.connect('SysHBspline.h', 'SysRho.h')
        self.connect('SysTemp.temp', 'SysRho.temp')
        self.connect('SysTemp.temp', 'SysSpeed.temp')
        self.connect('SysMVBspline.M', 'SysSpeed.M')
        self.connect('SysMVBspline.v_spline', 'SysSpeed.v_spline')


        # -----------------------------------
        # Comps for Coupled System begin here
        # -----------------------------------

        # Vertical Equilibrium
        self.add('SysCLTar', SysCLTar(num_elem=self.num_elem, wt_pax=self.wt_pax,
                                      S=self.S, ac_w=self.ac_w))

        self.connect('SysRho.rho', 'SysCLTar.rho')
        self.connect('SysGammaBspline.Gamma', 'SysCLTar.Gamma')
        self.connect('SysSpeed.v', 'SysCLTar.v')

        # Tripan Alpha
        self.add('SysTripanCLSurrogate', SysTripanCLSurrogate(num_elem=self.num_elem,
                                                              num=self.num,
                                                              CL=self.CL_arr))
        self.connect('SysMVBspline.M', 'SysTripanCLSurrogate.M')
        self.connect('SysHBspline.h', 'SysTripanCLSurrogate.h')
        self.connect('SysCLTar.CL', 'SysTripanCLSurrogate.CL_tar')

        # Tripan Eta
        self.add('SysTripanCMSurrogate', SysTripanCMSurrogate(num_elem=self.num_elem,
                                                              num=self.num,
                                                              CM=self.CM_arr))
        self.connect('SysMVBspline.M', 'SysTripanCMSurrogate.M')
        self.connect('SysHBspline.h', 'SysTripanCMSurrogate.h')
        self.connect('SysTripanCLSurrogate.alpha', 'SysTripanCMSurrogate.alpha')

        # Tripan Drag
        self.add('SysTripanCDSurrogate', SysTripanCDSurrogate(num_elem=self.num_elem,
                                                              num=self.num,
                                                              CD=self.CD_arr))
        self.connect('SysMVBspline.M', 'SysTripanCDSurrogate.M')
        self.connect('SysHBspline.h', 'SysTripanCDSurrogate.h')
        self.connect('SysTripanCMSurrogate.eta', 'SysTripanCDSurrogate.eta')
        self.connect('SysTripanCLSurrogate.alpha', 'SysTripanCDSurrogate.alpha')

        # Horizontal Equilibrium
        self.add('SysCTTar', SysCTTar(num_elem=self.num_elem, wt_pax=self.wt_pax,
                                      S=self.S, ac_w=self.ac_w))

        self.connect('SysGammaBspline.Gamma', 'SysCTTar.Gamma')
        self.connect('SysTripanCDSurrogate.CD', 'SysCTTar.CD')
        self.connect('SysTripanCLSurrogate.alpha', 'SysCTTar.alpha')
        self.connect('SysRho.rho', 'SysCTTar.rho')
        self.connect('SysSpeed.v', 'SysCTTar.v')

        # Weight
        self.add('SysFuelWeight', SysFuelWeight(num_elem=self.num_elem, S=self.S))
        self.SysFuelWeight.fuel_w = np.linspace(1.0, 0.0, self.num_elem+1)

        self.connect('SysSpeed.v', 'SysFuelWeight.v')
        self.connect('SysGammaBspline.Gamma', 'SysFuelWeight.Gamma')
        self.connect('SysCTTar.CT_tar', 'SysFuelWeight.CT_tar')
        self.connect('SysXBspline.x', 'SysFuelWeight.x')
        self.connect('SysSFC.SFC', 'SysFuelWeight.SFC')
        self.connect('SysRho.rho', 'SysFuelWeight.rho')

        # ------------------------------------------------
        # Coupled Analysis - Newton for outer loop
        # TODO: replace with GS/Newton cascaded solvers when working
        # -----------------------------------------------

        self.add('coupled_solver', NewtonSolver())

        # Direct connections (cycles) are faster.
        self.connect('SysFuelWeight.fuel_w', 'SysCLTar.fuel_w')
        self.connect('SysCTTar.CT_tar', 'SysCLTar.CT_tar')
        self.connect('SysTripanCLSurrogate.alpha', 'SysCLTar.alpha')
        self.connect('SysTripanCMSurrogate.eta', 'SysTripanCLSurrogate.eta')
        self.connect('SysFuelWeight.fuel_w', 'SysCTTar.fuel_w')

        #self.coupled_solver.add_parameter('SysCLTar.fuel_w')
        #self.coupled_solver.add_constraint('SysFuelWeight.fuel_w = SysCLTar.fuel_w')
        #self.coupled_solver.add_parameter('SysCLTar.CT_tar')
        #self.coupled_solver.add_constraint('SysCTTar.CT_tar = SysCLTar.CT_tar')
        #self.coupled_solver.add_parameter('SysCLTar.alpha')
        #self.coupled_solver.add_constraint('SysTripanCLSurrogate.alpha = SysCLTar.alpha')
        #self.coupled_solver.add_parameter('SysTripanCLSurrogate.eta')
        #self.coupled_solver.add_constraint('SysTripanCMSurrogate.eta = SysTripanCLSurrogate.eta')
        #self.coupled_solver.add_parameter('SysCTTar.fuel_w')
        #self.coupled_solver.add_constraint('SysFuelWeight.fuel_w = SysCTTar.fuel_w')

        # (Implicit comps)
        self.coupled_solver.add_parameter('SysTripanCLSurrogate.alpha')
        self.coupled_solver.add_constraint('SysTripanCLSurrogate.alpha_res = 0')
        self.coupled_solver.add_parameter('SysTripanCMSurrogate.eta')
        self.coupled_solver.add_constraint('SysTripanCMSurrogate.CM = 0')

        # --------------------
        # Downstream of solver
        # --------------------

        # Functionals (i.e., components downstream of the coupled system.)
        self.add('SysTau', SysTau(num_elem=self.num_elem,
                                  S=self.S, thrust_sl=self.thrust_sl))
        self.add('SysTmin', SysTmin(num_elem=self.num_elem))
        self.add('SysTmax', SysTmax(num_elem=self.num_elem))
        #self.add('SysSlopeMin', SysSlopeMin(num_elem=self.num_elem))
        #self.add('SysSlopeMax', SysSlopeMax(num_elem=self.num_elem))
        self.add('SysFuelObj', SysFuelObj(num_elem=self.num_elem))
        self.add('SysBlockTime', SysBlockTime(num_elem=self.num_elem))

        self.connect('SysRho.rho', 'SysTau.rho')
        self.connect('SysCTTar.CT_tar', 'SysTau.CT_tar')
        self.connect('SysHBspline.h', 'SysTau.h')
        self.connect('SysSpeed.v', 'SysTau.v')
        self.connect('SysTau.tau', 'SysTmin.tau')
        self.connect('SysTau.tau', 'SysTmax.tau')
        #self.connect('SysGammaBspline.Gamma', 'SysSlopeMin.Gamma')
        #self.connect('SysGammaBspline.Gamma', 'SysSlopeMax.Gamma')
        self.connect('SysFuelWeight.fuel_w', 'SysFuelObj.fuel_w')
        #self.connect('SysHBspline.h', 'SysHi.h')
        #self.connect('SysHBspline.h', 'SysHf.h')
        self.connect('SysXBspline.x', 'SysBlockTime.x')
        self.connect('SysSpeed.v', 'SysBlockTime.v')
        self.connect('SysGammaBspline.Gamma', 'SysBlockTime.Gamma')


        # Promote useful variables to the boundary.
        self.create_passthrough('SysHBspline.h_pt')
        self.connect('h_pt', 'SysGammaBspline.h_pt')
        self.create_passthrough('SysMVBspline.v_pt')
        self.create_passthrough('SysMVBspline.M_pt')
        self.create_passthrough('SysTmin.Tmin')
        self.create_passthrough('SysTmax.Tmax')
        self.create_passthrough('SysFuelObj.fuelburn')
        self.create_passthrough('SysBlockTime.time')
        self.create_passthrough('SysHBspline.h')
        self.create_passthrough('SysGammaBspline.Gamma')

        self.connect('pax_flt', 'SysCLTar.pax_flt')
        self.connect('pax_flt', 'SysCTTar.pax_flt')

        #-------------------------
        # Iteration Hierarchy
        #-------------------------
        self.driver.workflow.add(['SysXBspline', 'SysHBspline',
                                  'SysMVBspline', 'SysGammaBspline',
                                  'SysSFC', 'SysTemp', 'SysRho', 'SysSpeed',
                                  'coupled_solver',
                                  'SysTau', 'SysTmin', 'SysTmax',
                                  'SysFuelObj', 'SysBlockTime'])

        self.coupled_solver.workflow.add(['SysCLTar', 'SysTripanCLSurrogate',
                                          'SysTripanCMSurrogate', 'SysTripanCDSurrogate',
                                          'SysCTTar', 'SysFuelWeight'])

        #-------------------------
        # Driver Settings
        #-------------------------

        self.driver.gradient_options.lin_solver = "linear_gs"
        self.driver.gradient_options.maxiter = 1
        self.driver.gradient_options.derivative_direction = 'adjoint'
        self.driver.gradient_options.iprint = 1

        self.coupled_solver.atol = 1e-9
        self.coupled_solver.rtol = 1e-9
        self.coupled_solver.max_iteration = 15
        self.coupled_solver.gradient_options.atol = 1e-14
        self.coupled_solver.gradient_options.rtol = 1e-20
        self.coupled_solver.gradient_options.maxiter = 50
        self.coupled_solver.iprint = 2
        self.coupled_solver.gradient_options.iprint = 1
        self.coupled_solver.gradient_options.lin_solver = 'petsc_ksp'

    def set_init_h_pt(self, h_init_pt):
        ''' Solve for a good initial altitude profile.'''
        A = self.jac_h
        b = h_init_pt
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        self.h_pt = scipy.sparse.linalg.gmres(ATA, ATb)[0]

if __name__ == "__main__":

    #num_elem = 100
    #num_cp = 30
    x_range = 9000.0

    # for debugging only
    num_elem = 6
    num_cp = 3

    altitude = np.zeros(num_elem+1)
    altitude = 10 * np.sin(np.pi * np.linspace(0,1,num_elem+1))

    x_range *= 1.852
    x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
    M_init = np.ones(num_cp)*0.82
    h_init = 10 * np.sin(np.pi * x_init / (x_range/1e3))

    model = set_as_top(MissionSegment(num_elem=num_elem, num_cp=num_cp,
                                      x_pts=x_init, surr_file='crm_surr'))

    model.h_pt = h_init
    model.M_pt = M_init
    #model.set_init_h_pt(altitude)

    # Calculate velocity from the Mach we have specified.
    model.SysSpeed.v_specified = False
    
    # Initial parameters
    model.S = 427.8/1e2
    model.ac_w = 210000*9.81/1e6
    model.thrust_sl = 1020000.0/1e6
    model.SFCSL = 8.951*9.81
    model.AR = 8.68
    model.oswald = 0.8

    profile = False

    if profile is False:
        from time import time
        t1 = time()
        model.run()
        print "Elapsed time:", time()-t1
    else:
        import cProfile
        import pstats
        import sys
        cProfile.run('model.run()', 'profout')
        p = pstats.Stats('profout')
        p.strip_dirs()
        p.sort_stats('time')
        p.print_stats()
        print '\n\n---------------------\n\n'
        p.print_callers()
        print '\n\n---------------------\n\n'
        p.print_callees()
