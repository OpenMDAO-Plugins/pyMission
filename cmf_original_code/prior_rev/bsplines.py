"""
INTENDED FOR MISSION ANALYSIS USE
b-splines parameterization of altitude, x-distance, and Mach number.
These provide altitude and Mach number profile wrt x-distance.
Gamma (flight path angle) is also computed using the altitude
parameterization
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
import MBI, scipy.sparse

class SysXBspline(ExplicitSystem):
    """ a b-spline parameterization of distance """

    def _declare(self):
        """ owned variable: x (with b-spline parameterization)
            dependencies: x_pt (control points)
        """

        self.num_pts = self.kwargs['num_elem']+1
        self.num_cp = self.kwargs['num_cp']
        self.x_init = self.kwargs['x_init']
        x_0 = self.kwargs['x_0']

        self._declare_variable('x', size=self.num_pts, val=x_0)
        self._declare_argument('x_pt', indices=range(self.num_cp))

    def apply_G(self):
        """ compute x b-spline values with x control point values
            using pre-calculated MBI jacobian
        """

        jac_h = self.kwargs['jac_h']
        x_dist = self.vec['u']('x')
        x_pt = self.vec['p']('x_pt')
        x_dist[:] = jac_h.dot(x_pt[:])

    def apply_dGdp(self, args):
        """ compute derivative of x b-spline values wrt x control points
            using pre-calculated MBI jacobian
        """

        jac_h = self.kwargs['jac_h']
        dx_dist = self.vec['dg']('x')
        dx_pt = self.vec['dp']('x_pt')

        if self.mode == 'fwd':
            dx_dist[:] = 0.0
            if self.get_id('x_pt') in args:
                dx_dist[:] += jac_h.dot(dx_pt[:])
        if self.mode == 'rev':
            dx_pt[:] = 0.0
            if self.get_id('x_pt') in args:
                dx_pt[:] += jac_h.T.dot(dx_dist[:])

class SysHBspline(ExplicitSystem):
    """ a b-spline parameterization of altitude """

    def _declare(self):
        """ owned variable: h (altitude with b-spline parameterization)
            dependencies: h_pt (control points)
        """

        self.num_pts = self.kwargs['num_elem']+1
        self.num_cp = self.kwargs['num_cp']
        self.x_init = self.kwargs['x_init']

        self._declare_variable('h', size=self.num_pts)
        self._declare_argument('h_pt', indices=range(self.num_cp))

    def apply_G(self):
        """ compute h b-splines values using h control point values
            using pre-calculated MBI jacobian
        """

        jac_h = self.kwargs['jac_h']
        alt = self.vec['u']('h')
        alt_pt = self.vec['p']('h_pt')
        alt[:] = jac_h.dot(alt_pt[:])

    def apply_dGdp(self, args):
        """ compute h b-spline derivatives wrt h control points
            using pre-calculated MBI jacobian
        """

        jac_h = self.kwargs['jac_h']
        dalt = self.vec['dg']('h')
        dalt_pt = self.vec['dp']('h_pt')

        if self.mode == 'fwd':
            dalt[:] = 0.0
            if self.get_id('h_pt') in args:
                dalt[:] += jac_h.dot(dalt_pt[:])
        if self.mode == 'rev':
            dalt_pt[:] = 0.0
            if self.get_id('h_pt') in args:
                dalt_pt[:] += jac_h.T.dot(dalt[:])

class SysMVBspline(ExplicitSystem):
    """ a b-spline parameterization of Mach number """

    def _declare(self):
        """ owned variable: M (Mach # with b-spline parameterization)
            dependencies: M_pt (control points)
        """

        self.num_pts = self.kwargs['num_elem']+1
        self.num_cp = self.kwargs['num_cp']
        self.x_init = self.kwargs['x_init']

        self._declare_variable('M_spline', size=self.num_pts)
        self._declare_variable('v_spline', size=self.num_pts)
        self._declare_argument('M_pt', indices=range(self.num_cp))
        self._declare_argument('v_pt', indices=range(self.num_cp))

    def apply_G(self):
        """ compute M b-spline values using M control point values
            using pre-calculated MBI jacobian
        """

        jac_h = self.kwargs['jac_h']
        mach = self.vec['u']('M_spline')
        mach_pt = self.vec['p']('M_pt')
        speed = self.vec['u']('v_spline')
        speed_pt = self.vec['p']('v_pt')
        mach[:] = jac_h.dot(mach_pt[:])
        speed[:] = jac_h.dot(speed_pt[:])

    def apply_dGdp(self, args):
        """ compute M b-spline derivatives wrt M control points
            using pre-calculated MBI jacobian
        """

        jac_h = self.kwargs['jac_h']
        dmach = self.vec['dg']('M_spline')
        dmach_pt = self.vec['dp']('M_pt')
        dspeed = self.vec['dg']('v_spline')
        dspeed_pt = self.vec['dp']('v_pt')

        if self.mode == 'fwd':
            dmach[:] = 0.0
            dspeed[:] = 0.0
            if self.get_id('M_pt') in args:
                dmach[:] += jac_h.dot(dmach_pt[:])
            if self.get_id('v_pt') in args:
                dspeed[:] += jac_h.dot(dspeed_pt[:])
        if self.mode == 'rev':
            dmach_pt[:] = 0.0
            dspeed_pt[:] = 0.0
            if self.get_id('M_pt') in args:
                dmach_pt[:] += jac_h.T.dot(dmach[:])
            if self.get_id('v_pt') in args:
                dspeed_pt[:] += jac_h.T.dot(dspeed[:])

class SysGammaBspline(ExplicitSystem):
    """ dh/dx obtained from b-spline parameterization of altitude """

    def _declare(self):
        """ owned variable: gamma (flight path angle w/ b-spline 
            parameterization
            dependencies: h_pt (altitude control points)
        """

        self.num_pts = self.kwargs['num_elem']+1
        self.num_cp = self.kwargs['num_cp']
        self.x_init = self.kwargs['x_init']

        self._declare_variable('gamma', size=self.num_pts)
        self._declare_argument('h_pt', indices=range(self.num_cp))

    def apply_G(self):
        """ compute gamma b-spline values using gamma control point values
            using pre-calculated MBI jacobian
        """

        jac_gamma = self.kwargs['jac_gamma']
        gamma = self.vec['u']('gamma')
        h_pt = self.vec['p']('h_pt')
        gamma[:] = jac_gamma.dot(h_pt[:]) * 1e3/1e-1

    def apply_dGdp(self, args):
        """ compute gamma b-spline derivatives wrt gamma control points
            using pre-calculated MBI jacobian
        """

        jac_gamma = self.kwargs['jac_gamma']
        dgamma = self.vec['dg']('gamma')
        dh_pt = self.vec['dp']('h_pt')

        if self.mode == 'fwd':
            dgamma[:] = 0.0
            if self.get_id('h_pt') in args:
                dgamma[:] += jac_gamma.dot(dh_pt[:]) * 1e3/1e-1
        if self.mode == 'rev':
            dh_pt[:] = 0.0
            if self.get_id('h_pt') in args:
                dh_pt[:] += jac_gamma.T.dot(dgamma[:]) * 1e3/1e-1

    def get_jacs(self):
        jac_gamma = self.kwargs['jac_gamma']
        return {'h_pt': jac_gamma * 1e3/1e-1}
