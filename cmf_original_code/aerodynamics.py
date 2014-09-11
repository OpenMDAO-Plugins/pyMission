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

class SysCLSurrogate(ExplicitSystem):

    def _declare(self):
        self.num_elem = self.kwargs['num_elem']
        ind_pts = range(self.num_elem+1)

        self._declare_variable('CL', size=self.num_elem+1)
        self._declare_argument('alpha', indices=ind_pts)
        self._declare_argument('eta', indices=ind_pts)

    def apply_G(self):
        pvec = self.vec['p']
        uvec = self.vec['u']

        lift_c = uvec('CL')
        alpha = pvec('alpha') * 1e-1
        eta = pvec('eta') * 1e-1

        lift_c0 = 0.26
        lift_ca = 4.24
        lift_ce = 0.27

        lift_c[:] = lift_c0 + lift_ca * alpha + lift_ce * eta

    def apply_dGdp(self, args):
        dpvec = self.vec['dp']
        dgvec = self.vec['dg']

        dalpha = dpvec('alpha')
        deta = dpvec('eta')
        dlift_c = dgvec('CL')

        lift_ca = 4.24
        lift_ce = 0.27

        if self.mode == 'fwd':
            dlift_c[:] = 0.0
            if self.get_id('alpha') in args:
                dlift_c[:] += lift_ca * dalpha * 1e-1
            if self.get_id('eta') in args:
                dlift_c[:] += lift_ce * deta * 1e-1
        if self.mode == 'rev':
            dalpha[:] = 0.0
            deta[:] = 0.0
            if self.get_id('alpha') in args:
                dalpha[:] += lift_ca * dlift_c * 1e-1
            if self.get_id('eta') in args:
                deta[:] += lift_ce * dlift_c * 1e-1

class SysCDSurrogate(ExplicitSystem):
    """ simulates the presence of an aero surrogate mode using
        linear aerodynamic model
    """

    def _declare(self):
        self.num_elem = self.kwargs['num_elem']
        ind_pts = range(self.num_elem+1)

        self._declare_variable('CD', size=self.num_elem+1)
        self._declare_argument('CL', indices=ind_pts)
        self._declare_argument('M', indices=ind_pts)
        self._declare_argument('AR', indices=[0])
        self._declare_argument('e', indices=[0])
        self._declare_argument('t_c', indices=[0])
        self._declare_argument('sweep', indices=[0])

    def apply_G(self):
        pvec = self.vec['p']
        uvec = self.vec['u']

        Mach = pvec('M')
        aspect_ratio = pvec(['AR', 0])
        oswald = pvec(['e', 0])
        t_c = pvec(['t_c', 0])
        sweep = pvec(['sweep', 0])
        lift_c = pvec('CL')
        drag_c = uvec('CD')

        drag_c0 = 0.018

        M_dd = (0.95/numpy.cos(sweep) - t_c/(10*(numpy.cos(sweep))**2)) *\
            numpy.ones(self.num_elem+1)
        M_dd -= lift_c/(numpy.cos(sweep))**3
        M_crit = M_dd - (0.1/80)**(0.333)*numpy.ones(self.num_elem+1)

        drag_c[:] = (drag_c0 - lift_c[:]**2 /
                     (numpy.pi * aspect_ratio * oswald)) / 1e-1

        for index in xrange(self.num_elem+1):
            if Mach[index] >= M_crit[index]:
                drag_c[index] -= 20*(Mach[index]-M_crit[index])**4 / 1e-1

    def apply_dGdp(self, args):
        dpvec = self.vec['dp']
        dgvec = self.vec['dg']
        pvec = self.vec['p']

        dMach = dpvec('M')
        daspect_ratio = dpvec('AR')
        doswald = dpvec('e')
        dt_c = dpvec('t_c')
        dsweep = dpvec('sweep')
        dlift_c = dpvec('CL')
        ddrag_c = dgvec('CD')

        Mach = pvec('M')
        aspect_ratio = pvec('AR')
        oswald = pvec('e')
        t_c = pvec('t_c')
        sweep = pvec('sweep')
        lift_c = pvec('CL')

        if self.mode == 'fwd':
            ddrag_c[:] = 0.0
            if self.get_id('AR') in args:
                ddrag_c[:] -= lift_c**2 / (numpy.pi * aspect_ratio**2 *
                                           oswald) * daspect_ratio / 1e-1
            if self.get_id('e') in args:
                ddrag_c[:] -= lift_c**2 / (numpy.pi * aspect_ratio *
                                           oswald**2) * doswald / 1e-1
            if self.get_id('CL') in args:
                ddrag_c[:] += (2 * lift_c * dlift_c /
                               (numpy.pi * aspect_ratio * oswald)) /1e-1

            M_dd = (0.95/numpy.cos(sweep) - t_c/(10*(numpy.cos(sweep))**2)) *\
                numpy.ones(self.num_elem+1)
            M_dd -= lift_c/(numpy.cos(sweep))**3
            M_crit = M_dd - (0.1/80)**(0.333)*numpy.ones(self.num_elem+1)

            for index in xrange(self.num_elem+1):
                if Mach[index] >= M_crit[index]:
                    if self.get_id('CL') in args:
                        dM_dd = -1/(numpy.cos(sweep))**3
                        ddrag_c[index] += 80 * (Mach[index] - M_crit[index])**3 *\
                            dM_dd * dlift_c[index] / 1e-1
                    if self.get_id('M') in args:
                        ddrag_c[index] += 80 * (Mach[index] - M_crit[index])**3 *\
                            dMach[index] / 1e-1
                    if self.get_id('t_c') in args:
                        dM_dd = -dt_c[0]/(10*(numpy.cos(sweep))**2)
                        ddrag_c[index] -= 80 * (Mach[index] - M_crit[index])**3 *\
                            dM_dd / 1e-1
                    if self.get_id('sweep') in args:
                        dM_dd = (0.95/(numpy.cos(sweep))**2 -
                                 t_c/(5*(numpy.cos(sweep))**3) -
                                 (3*lift_c[index])/(numpy.cos(sweep))**4)*numpy.sin(sweep)
                        ddrag_c[index] -= 80 * (Mach[index] - M_crit[index])**3 *\
                            dM_dd * dsweep[0] / 1e-1

        elif self.mode == 'rev':
            dlift_c[:] = 0.0
            dMach[:] = 0.0
            daspect_ratio[:] = 0.0
            doswald[:] = 0.0
            dt_c[:] = 0.0
            dsweep[:] = 0.0
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
            if self.get_id('CL') in args:
                dlift_c[:] += (2 * lift_c * ddrag_c /
                               (numpy.pi * aspect_ratio * oswald)) /1e-1

            M_dd = (0.95/numpy.cos(sweep) - t_c/(10*(numpy.cos(sweep))**2)) *\
                numpy.ones(self.num_elem+1)
            M_dd -= lift_c/(numpy.cos(sweep))**3
            M_crit = M_dd - (0.1/80)**(1/3)

            for index in xrange(self.num_elem+1):
                if Mach[index] >= M_crit[index]:
                    if self.get_id('M') in args:
                        dMach[index] += 80 * (Mach[index] - M_crit[index])**3 *\
                            ddrag_c[index] / 1e-1
                    if self.get_id('CL') in args:
                        dM_dd = -1/(numpy.cos(sweep))**3
                        dlift_c[index] += 80 * (Mach[index] - M_crit[index])**3 *\
                            dM_dd * ddrag_c[index] / 1e-1
                    if self.get_id('t_c') in args:
                        dM_dd = -1/(10*(numpy.cos(sweep))**2)
                        dt_c[0] -= 80 * (Mach[index] - M_crit[index])**3 *\
                            dM_dd * ddrag_c[index] / 1e-1
                    if self.get_id('sweep') in args:
                        dM_dd = (0.95/(numpy.cos(sweep))**2 -
                                 t_c/(5*(numpy.cos(sweep))**3) -
                                 (3*lift_c[index])/(numpy.cos(sweep))**4)*numpy.sin(sweep)
                        dsweep[0] -= 80 * (Mach[index] - M_crit[index])**3 *\
                            dM_dd * ddrag_c[index] / 1e-1

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
