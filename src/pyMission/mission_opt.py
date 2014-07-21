"""
Trajectory optimization.
"""

# pylint: disable=E1101
import numpy as np

from openmdao.main.api import Assembly, set_as_top

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

        self.connect('SysHBspline.h', 'SysSFC.h')
        self.connect('SysHBspline.h', 'SysTemp.h')
        self.connect('SysTemp.temp', 'SysRho.temp')
        self.connect('SysTemp.temp', 'SysSpeed.temp')
        self.connect('SysMVBspline.M', 'SysSpeed.M')
        self.connect('SysMVBspline.v_spline', 'SysSpeed.v_spline')

        self.driver.workflow.add(['SysSFC', 'SysTemp', 'SysRho', 'SysSpeed'])


        # Vertical Equilibrium
        self.add('SysCLTar', SysCLTar(num_elem=self.num_elem))

        self.connect('SysRho.rho', 'SysCLTar.rho')
        # Need fuel_w, Gamma, CT_tar, v, alpha


        # Drag
        self.add('SysAeroSurrogate', SysAeroSurrogate(num_elem=self.num_elem))

        # need alpha, eta
        # see coupled_analysis.SysAlpha for a solver constraint


        # Horizontal Equilibrium
        self.add('SysCTTar', SysCTTar(num_elem=self.num_elem))

        self.connect('SysRho.rho', 'SysCTTar.rho')
        # need feul_w, Gamma, CD, alpha, v


        # Moment Equilibrium
        self.add('SysCM', SysCM(num_elem=self.num_elem))

        # need alpha (state=eta)


        # Coupled Analysis

        # Optimization

if __name__ == "__main__":

    num_elem = 1000
    num_cp = 50
    x_range = 1000.0e3

    x_init = x_range * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6

    model = OptTrajectory(num_elem, num_cp, x_init)

    model.run()