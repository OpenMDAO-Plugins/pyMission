"""
Trajectory optimization.
"""

# pylint: disable=E1101
import numpy as np

from openmdao.lib.casehandlers.api import BSONCaseRecorder
from openmdao.lib.drivers.api import NewtonSolver, BroydenSolver
from openmdao.main.api import Assembly, set_as_top
from openmdao.main.datatypes.api import Array, Float

from pyMission.aerodynamics import SysAeroSurrogate, SysCM
from pyMission.atmospherics import SysSFC, SysTemp, SysRho, SysSpeed
from pyMission.bsplines import SysXBspline, SysHBspline, SysMVBspline, \
                               SysGammaBspline
from pyMission.coupled_analysis import SysCLTar, SysCTTar, SysFuelWeight
from pyMission.propulsion import SysTau


class OptTrajectory(Assembly):
    """ class used to define and setup trajectory optimization problem """


    def __init__(self, num_elem=10, num_cp=5, x_pts=None):
        """Initialize the trajectory optimization problem.

        num_elem: int
            number of computations points in the mission profile

        num_cp: int
            number of control points for the splines
        """

        self.num_elem = num_elem
        self.num_pt = num_cp
        self.x_pts = x_pts

        super(OptTrajectory, self).__init__()

    def configure(self):
        """ set it all up. """

        # Make some boundary variables for ease of use.
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
        #self.SysXBspline.x_pt = np.linspace(0.0, self.x_pts[-1], self.num_elem)

        self.add('SysHBspline', SysHBspline(num_elem=self.num_elem,
                                            num_pt=self.num_pt))
        self.SysHBspline.x_init = self.x_pts

        self.add('SysMVBspline', SysMVBspline(num_elem=self.num_elem,
                                            num_pt=self.num_pt))
        self.SysMVBspline.x_init = self.x_pts

        self.add('SysGammaBspline', SysGammaBspline(num_elem=self.num_elem,
                                            num_pt=self.num_pt))
        self.SysGammaBspline.x_init = self.x_pts

        self.driver.workflow.add(['SysXBspline', 'SysHBspline',
                                  'SysMVBspline', 'SysGammaBspline'])


        # Atmospherics
        self.add('SysSFC', SysSFC(num_elem=self.num_elem))
        self.add('SysTemp', SysTemp(num_elem=self.num_elem))
        self.add('SysRho', SysRho(num_elem=self.num_elem))
        self.add('SysSpeed', SysSpeed(num_elem=self.num_elem))

        self.connect('SFCSL', 'SysSFC.SFCSL')
        self.connect('SysHBspline.h', 'SysSFC.h')
        self.connect('SysHBspline.h', 'SysTemp.h')
        self.connect('SysTemp.temp', 'SysRho.temp')
        self.connect('SysTemp.temp', 'SysSpeed.temp')
        self.connect('SysMVBspline.M', 'SysSpeed.M')
        self.connect('SysMVBspline.v_spline', 'SysSpeed.v_spline')

        self.driver.workflow.add(['SysSFC', 'SysTemp', 'SysRho', 'SysSpeed'])


        # Weight
        self.add('SysFuelWeight', SysFuelWeight(num_elem=self.num_elem))
        self.SysFuelWeight.fuel_w = np.linspace(1.0, 0.0, self.num_elem)

        self.connect('S', 'SysFuelWeight.S')
        self.connect('SysRho.rho', 'SysFuelWeight.rho')
        self.connect('SysXBspline.x', 'SysFuelWeight.x')
        self.connect('SysGammaBspline.Gamma', 'SysFuelWeight.Gamma')
        self.connect('SysSpeed.v', 'SysFuelWeight.v')
        self.connect('SysSFC.SFC', 'SysFuelWeight.SFC')


        # Moment Equilibrium
        self.add('SysCM', SysCM(num_elem=self.num_elem))


        # Drag
        self.add('SysAeroSurrogate', SysAeroSurrogate(num_elem=self.num_elem))

        self.connect('AR', 'SysAeroSurrogate.AR')
        self.connect('oswald', 'SysAeroSurrogate.oswald')
        self.connect('SysCM.eta', 'SysAeroSurrogate.eta')


        # Horizontal Equilibrium
        self.add('SysCTTar', SysCTTar(num_elem=self.num_elem))

        self.connect('S', 'SysCTTar.S')
        self.connect('ac_w', 'SysCTTar.ac_w')
        self.connect('SysRho.rho', 'SysCTTar.rho')
        self.connect('SysGammaBspline.Gamma', 'SysCTTar.Gamma')
        self.connect('SysSpeed.v', 'SysCTTar.v')
        self.connect('SysAeroSurrogate.CD', 'SysCTTar.CD')
        self.connect('SysFuelWeight.fuel_w', 'SysCTTar.fuel_w')


        # Vertical Equilibrium
        self.add('SysCLTar', SysCLTar(num_elem=self.num_elem))

        self.connect('S', 'SysCLTar.S')
        self.connect('ac_w', 'SysCLTar.ac_w')
        self.connect('SysRho.rho', 'SysCLTar.rho')
        self.connect('SysGammaBspline.Gamma', 'SysCLTar.Gamma')
        self.connect('SysSpeed.v', 'SysCLTar.v')
        self.connect('SysCTTar.CT_tar', 'SysCLTar.CT_tar')
        self.connect('SysFuelWeight.fuel_w', 'SysCLTar.fuel_w')


          # Coupled Analysis
        self.add('coupled_solver', NewtonSolver())

        self.coupled_solver.add_parameter(('SysCLTar.alpha', 'SysCM.alpha',
                                           'SysAeroSurrogate.alpha', 'SysCTTar.alpha'))
        self.coupled_solver.add_parameter('SysFuelWeight.CT_tar')

        self.coupled_solver.add_constraint('SysAeroSurrogate.CL = SysCLTar.CL')
        self.coupled_solver.add_constraint('SysFuelWeight.CT_tar = SysCTTar.CT_tar')

        self.coupled_solver.iprint = 1
        self.driver.workflow.add(['coupled_solver'])


        # Functionals (i.e., components downstream of the coupled system.)
        self.add('SysTau', SysTau(num_elem=self.num_elem))

        self.connect('S', 'SysTau.S')
        self.connect('thrust_sl', 'SysTau.thrust_sl')
        self.connect('SysRho.rho', 'SysTau.rho')
        self.connect('SysCTTar.CT_tar', 'SysTau.CT_tar')
        self.connect('SysHBspline.h', 'SysTau.h')
        self.connect('SysSpeed.v', 'SysTau.v')
        self.driver.workflow.add(['SysTau'])


        # Optimization

if __name__ == "__main__":

    num_elem = 100
    num_cp = 30
    x_range = 5000.0e3

    x_init = x_range * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
    v_init = np.ones(num_cp)*2.3
    h_init = 1 * np.sin(np.pi * x_init / (x_range/1e6))

    model = OptTrajectory(num_elem, num_cp, x_init)
    model.recorders = [BSONCaseRecorder('mission.bson')]

    model.SysHBspline.h_pt = h_init
    model.SysMVBspline.v_pt = v_init

    # Pull velocity from BSpline instead of calculating it.
    model.SysSpeed.v_specified = True

    # Initial parameters
    model.S = 427.8/1e2
    model.ac_w = 210000*9.81/1e6
    model.thrust_sl = 1020000.0/1e6/3
    model.SFCSL = 8.951
    model.AR = 8.68
    model.oswald = 0.8

    model.run()
