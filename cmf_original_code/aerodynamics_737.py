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

class SysAeroSurrogate(ExplicitSystem):
    """ simulates the presence of an aero surrogate mode using
        linear aerodynamic model
    """

    def _declare(self):
        """ owned variables: CL (lift coefficient),
                             CD (drag coefficient),
            dependencies: alpha (angle of attack),
                          eta (tail rotation angle),
                          AR (aspect ratio),
                          e (Oswald's efficiency)
        """

        self.num_elem = self.kwargs['num_elem']
        ind_pts = range(self.num_elem+1)

        self._declare_variable('CL', size=self.num_elem+1)
        self._declare_variable('CD', size=self.num_elem+1)
        self._declare_argument('alpha', indices=ind_pts)
        self._declare_argument('eta', indices=ind_pts)
        self._declare_argument('AR', indices=[0])
        self._declare_argument('e', indices=[0])

    def apply_G(self):
        """ compute lift and drag coefficient using angle of attack
            and tail rotation angles. linear aerodynamics assumed
        """

        pvec = self.vec['p']
        uvec = self.vec['u']

        alpha = pvec('alpha') * 1e-1
        eta = pvec('eta') * 1e-1
        aspect_ratio = pvec(['AR', 0])
        oswald = pvec(['e', 0])
        lift_c = uvec('CL')
        drag_c = uvec('CD')

        lift_c0 = 0.30
        lift_ca = 6.00
        lift_ce = 0.27
        drag_c0 = 0.015

        lift_c[:] = lift_c0 + lift_ca * alpha + lift_ce * eta
        drag_c[:] = (drag_c0 + lift_c[:]**2 / 
                     (numpy.pi * aspect_ratio * oswald)) / 1e-1

    def apply_dGdp(self, args):
        """ compute the derivatives of lift and drag coefficient wrt
            alpha, eta, aspect ratio, and osawlds efficiency
        """

        dpvec = self.vec['dp']
        dgvec = self.vec['dg']
        pvec = self.vec['p']
        uvec = self.vec['u']

        dalpha = dpvec('alpha')
        deta = dpvec('eta')
        daspect_ratio = dpvec('AR')
        doswald = dpvec('e')
        dlift_c = dgvec('CL')
        ddrag_c = dgvec('CD')

        aspect_ratio = pvec('AR')
        oswald = pvec('e')
        lift_c = uvec('CL')

        lift_ca = 3.00
        lift_ce = 0.27

        if self.mode == 'fwd':
            dlift_c[:] = 0.0
            ddrag_c[:] = 0.0
            if self.get_id('alpha') in args:
                dlift_c[:] += lift_ca * dalpha * 1e-1
                ddrag_c[:] += 2 * lift_c * lift_ca / (numpy.pi * oswald *
                                                     aspect_ratio) * dalpha
            if self.get_id('eta') in args:
                dlift_c[:] += lift_ce * deta * 1e-1
                ddrag_c[:] += 2 * lift_c * lift_ce / (numpy.pi * oswald *
                                                      aspect_ratio) * deta
            if self.get_id('AR') in args:
                ddrag_c[:] -= lift_c**2 / (numpy.pi * aspect_ratio**2 *
                                           oswald) * daspect_ratio / 1e-1
            if self.get_id('e') in args:
                ddrag_c[:] -= lift_c**2 / (numpy.pi * aspect_ratio *
                                           oswald**2) * doswald / 1e-1

        elif self.mode == 'rev':
            dalpha[:] = 0.0
            deta[:] = 0.0
            daspect_ratio[:] = 0.0
            doswald[:] = 0.0
            if self.get_id('alpha') in args:
                dalpha[:] += lift_ca * dlift_c * 1e-1
                dalpha[:] += 2 * lift_c * lift_ca / (numpy.pi * oswald *
                                                     aspect_ratio) * ddrag_c
            if self.get_id('eta') in args:
                deta[:] += lift_ce * dlift_c * 1e-1
                deta[:] += 2 * lift_c * lift_ce / (numpy.pi * oswald *
                                                   aspect_ratio) * ddrag_c
            if self.get_id('AR') in args:
                daspect_ratio[:] -= numpy.sum((lift_c**2 / 
                                               (numpy.pi * oswald *
                                                aspect_ratio**2) *
                                               ddrag_c)) / 1e-1
            if self.get_id('e') in args:
                doswald[:] -= numpy.sum((lift_c**2 /
                                         (numpy.pi * oswald**2 *
                                          aspect_ratio) *
                                         ddrag_c)) / 1e-1

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

        self._declare_variable('eta', size=self.num_elem+1)
        self._declare_argument('alpha', indices=ind_pts)

    def apply_F(self):
        """ compute CM value using alpha and eta, and use the CM value as
            residual for eta
        """

        pvec = self.vec['p']
        uvec = self.vec['u']
        fvec = self.vec['f']

        alpha = pvec('alpha') * 1e-1
        eta = uvec('eta') * 1e-1
        eta_res = fvec('eta')

        mmt_ca = 0.63
        mmt_ce = 1.06

        eta_res[:] = (mmt_ca * alpha + mmt_ce * eta) / 1e-1

    def apply_dFdpu(self, args):
        """ compute the derivatives of tail rotation angle wrt angle of attack
        """

        dpvec = self.vec['dp']
        duvec = self.vec['du']
        dfvec = self.vec['df']

        dalpha = dpvec('alpha')
        deta = duvec('eta')
        deta_res = dfvec('eta')

        mmt_ca = 0.63
        mmt_ce = 1.06

        if self.mode == 'fwd':
            deta_res[:] = 0.0
            if self.get_id('alpha') in args:
                deta_res[:] += mmt_ca * dalpha
            if self.get_id('eta') in args:
                deta_res[:] += mmt_ce * deta

        elif self.mode == 'rev':
            dalpha[:] = 0.0
            deta[:] = 0.0
            if self.get_id('alpha') in args:
                dalpha[:] += mmt_ca * deta_res
            if self.get_id('eta') in args:
                deta[:] += mmt_ce * deta_res
