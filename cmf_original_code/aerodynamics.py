"""
INTENDED FOR MISSION ANALYSIS USE
This file contains the aerodynamic models used by the mission analysis
code. The present implementation uses linear aerodynamics.
The mission analysis and trajectory optimization tool was developed by:
    Jason Kao*
    John Hwang*

* University of Michigan Department of Aerospace Engineering,
  Multidisciplinary Design Optimization lab
  mdolab.engin.umich.edu

copyright July 2014
"""

# pylint: disable=E1101
from __future__ import division
import sys
from framework import *
import numpy

class SysCM(ImplicitSystem):
    """ compute the tail rotation angle necessary to maintain pitch moment
        equilibrium
    """

    def _declare(self):
        """ owned variable: eta (tail rotation angle)
            dependencies: alpha (angle of attack)
        """

        self.num_elem = self.kwargs['num_elem']
        ind_pts = range(self.num_elem+1)

        self._declare_variable('eta', size=self.num_elem+1)#, lower=-20*numpy.pi/180/1e-1, upper=20*numpy.pi/180/1e-1)
        self._declare_argument('Cm', indices=ind_pts)

    def apply_F(self):
        """ compute CM value using alpha and eta, and use the CM value as
            residual for eta
        """

        res = self.vec['f']('eta')
        Cm = self.vec['p']('Cm')

        res[:] = Cm

    def apply_dFdpu(self, args):
        """ compute the derivatives of tail rotation angle wrt angle of attack
        """

        dres = self.vec['df']('eta')
        dCm = self.vec['dp']('Cm')

        if self.mode == 'fwd':
            dres[:] = 0.0
            if self.get_id('Cm') in args:
                dres[:] += dCm
        elif self.mode == 'rev':
            dCm[:] = 0.0
            if self.get_id('Cm') in args:
                dCm[:] += dres
