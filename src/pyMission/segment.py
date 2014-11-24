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
                                  SysFuelObj, SysHi, SysHf, SysMf, SysMi
from pyMission.propulsion import SysSFC, SysTau


def is_differentiable(self): 
        return True
Driver.is_differentiable = is_differentiable


class MissionSegment(Assembly):
    """ Defines a single segment for the Mission Analysis. """


    def __init__(self, num_elem=10, num_cp=5, x_pts=None, surr_file=None):
        """Initialize this segment trajectory problem.

        num_elem: int
            number of computations points in the mission profile

        num_cp: int
            number of control points for the splines

        x_pts: 1d array
            array containing the x locations of the spline control points.

        surr_file: Name of file for generating the Tripan surrogate models.
        """

        self.num_elem = num_elem
        self.num_pt = num_cp
        self.x_pts = x_pts

        # Generate jacobians for b-splines using MBI package
        self.jac_h, self.jac_gamma = setup_MBI(num_elem+1, num_cp, x_pts)

        # Setup the surrogate models
        self.CL_arr, self.CD_arr, self.CM_arr, self.num = \
            setup_surrogate(surr_file)

        super(MissionSegment, self).__init__()

    def configure(self):
        """ Set it all up. """

        # Place all design variables on the Assembly boundary.
        self.add('S', Float(0.0, iotype='in', desc = 'Wing Area'))
        self.add('ac_w', Float(0.0, iotype='in',
                               desc = 'Weight of aircraft + payload'))
        self.add('SFCSL', Float(0.0, iotype='in',
                                desc = 'sea-level SFC value'))
        self.add('thrust_sl', Float(0.0, iotype='in',
                                    desc = 'Maximum sea-level thrust'))
        self.add('AR', Float(0.0, iotype='in',
                             desc = 'Aspect Ratio'))
        self.add('oswald', Float(0.0, iotype='in',
                                 desc = "Oswald's efficiency"))


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
        self.add('SysSFC', SysSFC(num_elem=self.num_elem))
        self.add('SysTemp', SysTemp(num_elem=self.num_elem))
        self.add('SysRho', SysRho(num_elem=self.num_elem))
        self.add('SysSpeed', SysSpeed(num_elem=self.num_elem))

        self.connect('SFCSL', 'SysSFC.SFCSL')
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
        self.add('SysCLTar', SysCLTar(num_elem=self.num_elem))

        self.connect('S', 'SysCLTar.S')
        self.connect('ac_w', 'SysCLTar.ac_w')
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
        self.add('SysCTTar', SysCTTar(num_elem=self.num_elem))

        self.connect('SysGammaBspline.Gamma', 'SysCTTar.Gamma')
        self.connect('SysTripanCDSurrogate.CD', 'SysCTTar.CD')
        self.connect('SysTripanCLSurrogate.alpha', 'SysCTTar.alpha')
        self.connect('SysRho.rho', 'SysCTTar.rho')
        self.connect('SysSpeed.v', 'SysCTTar.v')
        self.connect('S', 'SysCTTar.S')
        self.connect('ac_w', 'SysCTTar.ac_w')

        # Weight
        self.add('SysFuelWeight', SysFuelWeight(num_elem=self.num_elem))
        self.SysFuelWeight.fuel_w = np.linspace(1.0, 0.0, self.num_elem+1)

        self.connect('SysSpeed.v', 'SysFuelWeight.v')
        self.connect('SysGammaBspline.Gamma', 'SysFuelWeight.Gamma')
        self.connect('SysCTTar.CT_tar', 'SysFuelWeight.CT_tar')
        self.connect('SysXBspline.x', 'SysFuelWeight.x')
        self.connect('SysSFC.SFC', 'SysFuelWeight.SFC')
        self.connect('SysRho.rho', 'SysFuelWeight.rho')
        self.connect('S', 'SysFuelWeight.S')

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

        # (Implicit comps)
        self.coupled_solver.add_parameter('SysTripanCLSurrogate.alpha')
        self.coupled_solver.add_constraint('SysTripanCLSurrogate.alpha_res = 0')
        self.coupled_solver.add_parameter('SysTripanCMSurrogate.eta')
        self.coupled_solver.add_constraint('SysTripanCMSurrogate.CM = 0')

        self.coupled_solver.atol = 1e-9
        self.coupled_solver.rtol = 1e-9
        self.coupled_solver.max_iteration = 15
        self.coupled_solver.gradient_options.atol = 1e-14
        self.coupled_solver.gradient_options.rtol = 1e-20
        self.coupled_solver.gradient_options.maxiter = 50

        self.coupled_solver.iprint = 1


        # --------------------
        # Downstream of solver
        # --------------------

        # Functionals (i.e., components downstream of the coupled system.)
        self.add('SysTau', SysTau(num_elem=self.num_elem))
        self.add('SysTmin', SysTmin(num_elem=self.num_elem))
        self.add('SysTmax', SysTmax(num_elem=self.num_elem))
        #self.add('SysSlopeMin', SysSlopeMin(num_elem=self.num_elem))
        #self.add('SysSlopeMax', SysSlopeMax(num_elem=self.num_elem))
        self.add('SysFuelObj', SysFuelObj(num_elem=self.num_elem))
        self.add('SysHi', SysHi(num_elem=self.num_elem))
        self.add('SysHf', SysHf(num_elem=self.num_elem))

        self.connect('S', 'SysTau.S')
        self.connect('thrust_sl', 'SysTau.thrust_sl')
        self.connect('SysRho.rho', 'SysTau.rho')
        self.connect('SysCTTar.CT_tar', 'SysTau.CT_tar')
        self.connect('SysHBspline.h', 'SysTau.h')
        self.connect('SysSpeed.v', 'SysTau.v')
        self.connect('SysTau.tau', 'SysTmin.tau')
        self.connect('SysTau.tau', 'SysTmax.tau')
        #self.connect('SysGammaBspline.Gamma', 'SysSlopeMin.Gamma')
        #self.connect('SysGammaBspline.Gamma', 'SysSlopeMax.Gamma')
        self.connect('SysFuelWeight.fuel_w', 'SysFuelObj.fuel_w')
        self.connect('SysHBspline.h', 'SysHi.h')
        self.connect('SysHBspline.h', 'SysHf.h')


        # Promote useful variables to the boundary.
        self.create_passthrough('SysHBspline.h_pt')
        self.connect('h_pt', 'SysGammaBspline.h_pt')
        self.create_passthrough('SysMVBspline.v_pt')
        self.create_passthrough('SysMVBspline.M_pt')
        self.create_passthrough('SysTmin.Tmin')
        self.create_passthrough('SysTmax.Tmax')
        self.create_passthrough('SysFuelObj.fuelburn')
        self.create_passthrough('SysHi.h_i')
        self.create_passthrough('SysHf.h_f')

        #-------------------------
        # Iteration Hieararchy
        #-------------------------
        # self.driver.gradient_options.lin_solver = "linear_gs"
        # self.driver.workflow.add(['SysXBspline', 'SysHBspline',
        #                           'SysMVBspline', 'SysGammaBspline',
        #                           'SysSFC', 'SysTemp', 'SysRho', 'SysSpeed',
        #                           'coupled_solver',
        #                           'SysTau', 'SysTmin', 'SysTmax',
        #                           'SysFuelObj', 'SysHi', 'SysHf'])

        self.coupled_solver.workflow.add(['SysCLTar', 'SysTripanCLSurrogate',
                                          'SysTripanCMSurrogate', 'SysTripanCDSurrogate',
                                          'SysCTTar', 'SysFuelWeight'])

        self.driver.gradient_options.lin_solver = "linear_gs"
        self.driver.gradient_options.maxiter = 1
        self.driver.workflow.add(['bsplines','atmospherics',  
                                  'coupled_solver', 
                                  'SysTau', 'SysTmin', 'SysTmax',
                                  'SysFuelObj', 'SysHi', 'SysHf'])

        bsplines = self.add('bsplines', Driver())
        # bsplines.gradient_options.lin_solver = 'linear_gs'
        # bsplines.gradient_options.maxiter = 1
        # bsplines.gradient_options.rtol = 1e-10
        # bsplines.gradient_options.atol = 1e-12
        bsplines.workflow.add(['SysXBspline', 'SysHBspline', 'SysMVBspline', 'SysGammaBspline'])

        atmospherics = self.add('atmospherics', Driver())
        # atmospherics.gradient_options.lin_solver = 'linear_gs'
        # atmospherics.gradient_options.maxiter = 1
        # atmospherics.gradient_options.rtol = 1e-6
        # atmospherics.gradient_options.atol = 1e-10
        atmospherics.workflow.add(['SysSFC', 'SysTemp', 'SysRho', 'SysSpeed',])

        # self.coupled_solver.workflow.add(['vert_eqlm', 'tripan_alpha',
        #                                   'SysTripanCMSurrogate', 'SysTripanCDSurrogate',
        #                                   'SysCTTar', 'SysFuelWeight'])
        
        # vert_eqlm = self.add('vert_eqlm', Driver())
        # vert_eqlm.gradient_options.lin_solver = 'scipy_gmres'
        # # vert_eqlm.gradient_options.maxiter = 1
        # # vert_eqlm.gradient_options.rtol = 1e-20
        # # vert_eqlm.gradient_options.atol = 1e-14
        # vert_eqlm.workflow.add('SysCLTar')

        # tripan_alpha = self.add('tripan_alpha', Driver())
        # tripan_alpha.gradient_options.lin_solver = 'linear_gs'
        # tripan_alpha.gradient_options.maxiter = 18
        # # tripan_alpha.gradient_options.rtol = 1e-6
        # # tripan_alpha.gradient_options.atol = 1e-6
        # tripan_alpha.workflow.add('SysTripanCLSurrogate')


    def set_init_h_pt(self, h_init_pt):
        ''' Solve for a good initial altitude profile.'''
        A = self.jac_h
        b = h_init_pt
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        self.h_pt = scipy.sparse.linalg.gmres(ATA, ATb)[0]

if __name__ == "__main__":

    num_elem = 250
    num_cp = 50
    x_range = 9000.0

    # for debugging only
    #num_elem = 6
    #num_cp = 3

    x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
    v_init = np.ones(num_cp)*2.3
    h_init = 1 * np.sin(np.pi * x_init / (x_range/1e3))

    model = set_as_top(MissionSegment(num_elem=num_elem, num_cp=num_cp,
                                      x_pts=x_init, surr_file='crm_surr'))

    model.h_pt = h_init
    model.v_pt = v_init

    # Pull velocity from BSpline instead of calculating it.
    model.SysSpeed.v_specified = True

    # Initial parameters
    model.S = 427.8/1e2
    model.ac_w = 210000*9.81/1e6
    model.thrust_sl = 1020000.0/1e6/3
    model.SFCSL = 8.951
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
