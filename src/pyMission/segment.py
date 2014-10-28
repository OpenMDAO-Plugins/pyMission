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

from openmdao.lib.drivers.api import NewtonSolver, FixedPointIterator, BroydenSolver
from openmdao.main.api import Assembly, set_as_top
from openmdao.main.datatypes.api import Array, Float

from pyMission.aerodynamics import SysAeroSurrogate, SysCM
from pyMission.atmospherics import SysTemp, SysRho, SysSpeed
from pyMission.bsplines import SysXBspline, SysHBspline, SysMVBspline, \
                               SysGammaBspline
from pyMission.coupled_analysis import SysCLTar, SysCTTar, SysFuelWeight
from pyMission.functionals import SysTmin, SysTmax, SysSlopeMin, SysSlopeMax, \
                                  SysFuelObj, SysHi, SysHf
from pyMission.propulsion import SysSFC, SysTau


class MissionSegment(Assembly):
    """ Defines a single segment for the Mission Analysis. """


    def __init__(self, num_elem=10, num_cp=5, x_pts=None):
        """Initialize this segment trajectory problem.

        num_elem: int
            number of computations points in the mission profile

        num_cp: int
            number of control points for the splines

        x_pts: 1d array
            array containing the x locations of the spline control points.
        """

        self.num_elem = num_elem
        self.num_pt = num_cp
        self.x_pts = x_pts

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
                                            num_pt=self.num_pt))
        self.SysXBspline.x_init = self.x_pts
        self.SysXBspline.x_pt = self.x_pts

        self.add('SysHBspline', SysHBspline(num_elem=self.num_elem,
                                            num_pt=self.num_pt))
        self.SysHBspline.x_init = self.x_pts

        self.add('SysMVBspline', SysMVBspline(num_elem=self.num_elem,
                                            num_pt=self.num_pt))
        self.SysMVBspline.x_init = self.x_pts

        self.add('SysGammaBspline', SysGammaBspline(num_elem=self.num_elem,
                                            num_pt=self.num_pt))
        self.SysGammaBspline.x_init = self.x_pts



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


        # Drag
        self.add('SysAeroSurrogate', SysAeroSurrogate(num_elem=self.num_elem))

        self.connect('AR', 'SysAeroSurrogate.AR')
        self.connect('oswald', 'SysAeroSurrogate.oswald')


        # Horizontal Equilibrium
        self.add('SysCTTar', SysCTTar(num_elem=self.num_elem))

        self.connect('S', 'SysCTTar.S')
        self.connect('ac_w', 'SysCTTar.ac_w')
        self.connect('SysRho.rho', 'SysCTTar.rho')
        self.connect('SysGammaBspline.Gamma', 'SysCTTar.Gamma')
        self.connect('SysSpeed.v', 'SysCTTar.v')
        self.connect('SysAeroSurrogate.CD', 'SysCTTar.CD')
        self.connect('SysAeroSurrogate.alpha', 'SysCTTar.alpha')


        # Moment Equilibrium
        self.add('SysCM', SysCM(num_elem=self.num_elem))
        self.connect('SysAeroSurrogate.alpha', 'SysCM.alpha')
        self.SysCM.eval_only = True


        # Weight
        self.add('SysFuelWeight', SysFuelWeight(num_elem=self.num_elem))
        self.SysFuelWeight.fuel_w = np.linspace(1.0, 0.0, self.num_elem+1)

        self.connect('S', 'SysFuelWeight.S')
        self.connect('SysRho.rho', 'SysFuelWeight.rho')
        self.connect('SysXBspline.x', 'SysFuelWeight.x')
        self.connect('SysGammaBspline.Gamma', 'SysFuelWeight.Gamma')
        self.connect('SysSpeed.v', 'SysFuelWeight.v')
        self.connect('SysSFC.SFC', 'SysFuelWeight.SFC')
        self.connect('SysCTTar.CT_tar', 'SysFuelWeight.CT_tar')


        # ----------------------------------------
        # Drag subsystem - Newton for inner loop
        # ----------------------------------------

        self.add('drag_solver', NewtonSolver())
        self.drag_solver.add_parameter(('SysAeroSurrogate.alpha'))
        self.drag_solver.add_constraint('SysAeroSurrogate.CL = SysCLTar.CL')

        self.drag_solver.iprint = 1
        self.drag_solver.atol = 1e-9
        self.drag_solver.rtol = 1e-9
        self.drag_solver.max_iteration = 15
        self.drag_solver.gradient_options.atol = 1e-10
        self.drag_solver.gradient_options.rtol = 1e-10
        self.drag_solver.gradient_options.maxiter = 15


        # ------------------------------------------------
        # Coupled Analysis - Newton for outer loop
        # TODO: replace with GS/Newton cascaded solvers when working
        # -----------------------------------------------

        self.add('coupled_solver', NewtonSolver())

        # Old way, using params and eq-constraints
        #self.coupled_solver.add_parameter('SysCLTar.CT_tar')
        #self.coupled_solver.add_parameter('SysCLTar.fuel_w')
        #self.coupled_solver.add_parameter('SysCLTar.alpha')
        #self.coupled_solver.add_parameter('SysAeroSurrogate.eta')
        #self.coupled_solver.add_parameter('SysCTTar.fuel_w')
        #self.coupled_solver.add_constraint('SysCLTar.CT_tar = SysCTTar.CT_tar')
        #self.coupled_solver.add_constraint('SysCLTar.fuel_w = SysFuelWeight.fuel_w')
        #self.coupled_solver.add_constraint('SysCLTar.alpha = SysAeroSurrogate.alpha')
        #self.coupled_solver.add_constraint('SysAeroSurrogate.eta = SysCM.eta')
        #self.coupled_solver.add_constraint('SysCTTar.fuel_w = SysFuelWeight.fuel_w')

        # Direct connections (cycles) are faster.
        self.connect('SysCTTar.CT_tar', 'SysCLTar.CT_tar')
        self.connect('SysFuelWeight.fuel_w', 'SysCLTar.fuel_w')
        self.connect('SysAeroSurrogate.alpha', 'SysCLTar.alpha')
        self.connect('SysCM.eta', 'SysAeroSurrogate.eta')
        self.connect('SysFuelWeight.fuel_w', 'SysCTTar.fuel_w')

        # (Only non-GS pair)
        self.coupled_solver.add_parameter('SysCM.eta')
        self.coupled_solver.add_constraint('SysCM.eta_res = 0')

        self.coupled_solver.atol = 1e-9
        self.coupled_solver.rtol = 1e-9
        self.coupled_solver.max_iteration = 15
        self.coupled_solver.gradient_options.atol = 1e-14
        self.coupled_solver.gradient_options.rtol = 1e-14
        self.coupled_solver.gradient_options.maxiter = 18

        self.coupled_solver.iprint = 1


        # --------------------
        # Downstream of solver
        # --------------------

        # Functionals (i.e., components downstream of the coupled system.)
        self.add('SysTau', SysTau(num_elem=self.num_elem))
        self.add('SysTmin', SysTmin(num_elem=self.num_elem))
        self.add('SysTmax', SysTmax(num_elem=self.num_elem))
        self.add('SysSlopeMin', SysSlopeMin(num_elem=self.num_elem))
        self.add('SysSlopeMax', SysSlopeMax(num_elem=self.num_elem))
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
        self.connect('SysGammaBspline.Gamma', 'SysSlopeMin.Gamma')
        self.connect('SysGammaBspline.Gamma', 'SysSlopeMax.Gamma')
        self.connect('SysFuelWeight.fuel_w', 'SysFuelObj.fuel_w')
        self.connect('SysHBspline.h', 'SysHi.h')
        self.connect('SysHBspline.h', 'SysHf.h')


        # Promote useful variables to the boundary.
        self.create_passthrough('SysHBspline.h_pt')
        self.connect('h_pt', 'SysGammaBspline.h_pt')
        self.create_passthrough('SysMVBspline.v_pt')
        self.create_passthrough('SysTmin.Tmin')
        self.create_passthrough('SysTmax.Tmax')
        self.create_passthrough('SysFuelObj.wf_obj')
        self.create_passthrough('SysHi.h_i')
        self.create_passthrough('SysHf.h_f')


        #-------------------------
        # Iteration Hieararchy
        #-------------------------
        self.driver.workflow.add(['SysXBspline', 'SysHBspline',
                                  'SysMVBspline', 'SysGammaBspline',
                                  'SysSFC', 'SysTemp', 'SysRho', 'SysSpeed',
                                  'coupled_solver',
                                  'SysTau', 'SysTmin', 'SysTmax', 'SysSlopeMin', 'SysSlopeMax',
                                  'SysFuelObj', 'SysHi', 'SysHf'])
        self.coupled_solver.workflow.add(['SysCLTar', 'drag_solver', 'SysCTTar', 'SysCM', 'SysFuelWeight'])
        self.drag_solver.workflow.add(['SysAeroSurrogate'])



if __name__ == "__main__":

    num_elem = 3000
    num_cp = 30
    x_range = 15000.0

    # for debugging only
    #num_elem = 6
    #num_cp = 3

    x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
    v_init = np.ones(num_cp)*2.3
    h_init = 1 * np.sin(np.pi * x_init / (x_range/1e3))

    model = set_as_top(MissionSegment(num_elem, num_cp, x_init))

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
