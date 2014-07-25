"""
INTENDED FOR MISSION ANALYSIS USE
The horizontal and vertical flight equilibrium equations are presented
and enforced. The fuel weight is also computed. The user-defined relation
between alpha, CL, and CD are not here!
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

class SysCLTar(ExplicitSystem):
    """ compute CL values using collocation at the control points
        the vertical flight equilibrium equation is enforced
    """

    def _declare(self):
        """ owned variable: CL (lift coefficient)
            dependencies: fuel_w (fuel weight)
                          gamma (flight path angle)
                          CT (thrust coefficient)
                          alpha (angle of attack)
                          rho (density)
                          v (speed)
                          S (wing area)
                          ac_w (weight of aircraft + payload)
        """

        self.num_elem = self.kwargs['num_elem']
        ind_pts = range(self.num_elem+1)

        self._declare_variable('CL_tar', val=1, size=self.num_elem+1)
        self._declare_argument('fuel_w', indices=ind_pts)
        self._declare_argument('gamma', indices=ind_pts)
        self._declare_argument('CT_tar', indices=ind_pts)
        self._declare_argument('alpha', indices=ind_pts)
        self._declare_argument('rho', indices=ind_pts)
        self._declare_argument('v', indices=ind_pts)
        self._declare_argument(['S', 0], indices=[0])
        self._declare_argument(['ac_w', 0], indices=[0])

    def apply_G(self):
        """ Compute lift coefficient based on other variables at the
            control points. This is done by solving for CL with the
            vertical flight equilibrium equation.
        """

        pvec = self.vec['p']
        uvec = self.vec['u']

        fuel_w = pvec('fuel_w') * 1e6
        gamma = pvec('gamma') * 1e-1
        thrust_c = pvec('CT_tar') * 1e-1
        alpha = pvec('alpha') * 1e-1
        rho = pvec('rho')
        speed = pvec('v') * 1e2
        wing_area = pvec(['S', 0]) * 1e2
        ac_w = pvec(['ac_w', 0]) * 1e6
        lift_c = uvec('CL_tar')

        lift_c[:] = (ac_w + fuel_w[:])*numpy.cos(gamma[:]) /\
                     (0.5*rho[:]*speed[:]**2*wing_area) - \
                     thrust_c[:]*numpy.sin(alpha[:])

    def apply_dGdp(self, args):
        """ compute lift coefficient derivatives wrt weight of aircraft,
            wing area, velocity, density, fuel weight, flight path angle,
            thrust coefficient, and angle of attack
        """

        pvec = self.vec['p']
        dpvec = self.vec['dp']
        dgvec = self.vec['dg']

        ac_w = pvec(['ac_w', 0]) * 1e6
        wing_area = pvec(['S', 0]) * 1e2
        speed = pvec('v') * 1e2
        rho = pvec('rho')
        fuel_w = pvec('fuel_w') * 1e6
        gamma = pvec('gamma') * 1e-1
        thrust_c = pvec('CT_tar') * 1e-1
        alpha = pvec('alpha') * 1e-1

        dac_w = dpvec(['ac_w', 0])
        dwing_area = dpvec(['S', 0])
        dspeed = dpvec('v')
        drho = dpvec('rho')
        dfuel_w = dpvec('fuel_w')
        dgamma = dpvec('gamma')
        dthrust_c = dpvec('CT_tar')
        dalpha = dpvec('alpha')
        dlift_c = dgvec('CL_tar')

        if self.mode == 'fwd':
            dlift_c[:] = 0.0
            if self.get_id('ac_w') in args:
                dlift_c[:] += numpy.cos(gamma[:]) /\
                              (0.5*rho[:]*speed[:]**2*wing_area) * dac_w[:]* \
                              1e6
            if self.get_id('S') in args:
                dlift_c[:] += -(ac_w + fuel_w[:])*numpy.cos(gamma[:]) /\
                              (0.5*rho[:]*speed[:]**2*wing_area**2) *\
                              dwing_area[:] * 1e2
            if self.get_id('v') in args:
                dlift_c[:] += -2*(ac_w + fuel_w[:])*numpy.cos(gamma[:]) /\
                              (0.5*rho[:]*speed[:]**3*wing_area) * dspeed[:] *\
                              1e2
            if self.get_id('rho') in args:
                dlift_c[:] += -(ac_w + fuel_w[:])*numpy.cos(gamma[:]) /\
                              (0.5*rho[:]**2*speed[:]**2*wing_area) * drho[:]
            if self.get_id('fuel_w') in args:
                dlift_c[:] += numpy.cos(gamma[:]) /\
                              (0.5*rho[:]*speed[:]**2*wing_area) *\
                              dfuel_w[:] * 1e6
            if self.get_id('gamma') in args:
                dlift_c[:] += -(ac_w + fuel_w[:])*numpy.sin(gamma[:]) /\
                              (0.5*rho[:]*speed[:]**2*wing_area) * dgamma[:] *\
                              1e-1
            if self.get_id('CT_tar') in args:
                dlift_c[:] += -numpy.sin(alpha[:]) * dthrust_c[:] * 1e-1
            if self.get_id('alpha') in args:
                dlift_c[:] += -thrust_c[:]*numpy.cos(alpha[:]) * dalpha[:] *\
                              1e-1

        if self.mode == 'rev':
            dac_w[:] = 0.0
            dwing_area[:] = 0.0
            dspeed[:] = 0.0
            drho[:] = 0.0
            dfuel_w[:] = 0.0
            dgamma[:] = 0.0
            dthrust_c[:] = 0.0
            dalpha[:] = 0.0
            if self.get_id('ac_w') in args:
                dac_w[:] += numpy.dot(numpy.cos(gamma[:]) /\
                    (0.5*rho[:]*speed[:]**2*wing_area), dlift_c[:]) * 1e6
            if self.get_id('S') in args:
                dwing_area[:] += -numpy.dot((ac_w + fuel_w[:])*\
                                            numpy.cos(gamma[:]) /\
                    (0.5*rho[:]*speed[:]**2*wing_area**2), dlift_c[:]) *\
                    1e2
            if self.get_id('v') in args:
                dspeed[:] += -2*(ac_w + fuel_w[:])*numpy.cos(gamma[:]) /\
                    (0.5*rho[:]*speed[:]**3*wing_area) * dlift_c[:] * 1e2
            if self.get_id('rho') in args:
                drho[:] += -(ac_w + fuel_w[:])*numpy.cos(gamma[:]) /\
                    (0.5*rho[:]**2*speed[:]**2*wing_area) * dlift_c[:]
            if self.get_id('fuel_w') in args:
                dfuel_w[:] += numpy.cos(gamma[:]) /\
                    (0.5*rho[:]*speed[:]**2*wing_area) * dlift_c[:] * 1e6
            if self.get_id('gamma') in args:
                dgamma[:] += -(ac_w + fuel_w[:])*numpy.sin(gamma[:]) /\
                    (0.5*rho[:]*speed[:]**2*wing_area) * dlift_c[:] * 1e-1
            if self.get_id('CT_tar') in args:
                dthrust_c[:] += -numpy.sin(alpha[:]) * dlift_c[:] * 1e-1
            if self.get_id('alpha') in args:
                dalpha[:] += -thrust_c[:]*numpy.cos(alpha[:]) * dlift_c[:] *\
                             1e-1


class SysCTTar(ExplicitSystem):
    """ compute CT values using collocation at the control points
        the horizontal flight equilibrium equation is enforced
    """

    def _declare(self):
        """ owned variable: CT (thrust coefficient)
            dependencies: fuel_w (weight of fuel)
                          gamma (flight path angle)
                          CD (drag coefficient)
                          alpha (angle of attack)
                          rho (density)
                          v (speed)
                          S (wing area)
                          ac_w (weight of aircraft + payload)
        """

        self.num_elem = self.kwargs['num_elem']
        ind_pts = range(self.num_elem+1)

        self._declare_variable('CT_tar', val=1, size=self.num_elem+1)
        self._declare_argument('fuel_w', indices=ind_pts)
        self._declare_argument('gamma', indices=ind_pts)
        self._declare_argument('CD', indices=ind_pts)
        self._declare_argument('alpha', indices=ind_pts)
        self._declare_argument('rho', indices=ind_pts)
        self._declare_argument('v', indices=ind_pts)
        self._declare_argument(['S', 0], indices=[0])
        self._declare_argument(['ac_w', 0], indices=[0])

    def apply_G(self):
        """ Compute thrust coefficient using variables at the control
            points. The thrust coefficients are solved from the horizontal
            flight equilibrium equation.
        """

        pvec = self.vec['p']
        uvec = self.vec['u']

        fuel_w = pvec('fuel_w') * 1e6
        gamma = pvec('gamma') * 1e-1
        drag_c = pvec('CD') * 1e-1
        alpha = pvec('alpha') * 1e-1
        rho = pvec('rho')
        speed = pvec('v') * 1e2
        wing_area = pvec(['S', 0]) * 1e2
        ac_w = pvec(['ac_w', 0]) * 1e6
        thrust_c = uvec('CT_tar')

        thrust_c[:] = (drag_c[:]/numpy.cos(alpha[:]) +
                       (ac_w + fuel_w[:])*numpy.sin(gamma[:]) /
                       (0.5*rho[:]*speed[:]**2*wing_area*numpy.cos(alpha[:]))) /\
                       1e-1

    def apply_dGdp(self, args):
        """ compute thrust coefficient derivatives wrt weight of aircraft,
            wing area, velocity, density, fuel weight, flight path angle,
            drag coefficient, and angle of attack
        """

        pvec = self.vec['p']
        dpvec = self.vec['dp']
        dgvec = self.vec['dg']

        ac_w = pvec(['ac_w', 0]) * 1e6
        wing_area = pvec(['S', 0]) * 1e2
        speed = pvec('v') * 1e2
        rho = pvec('rho')
        fuel_w = pvec('fuel_w') * 1e6
        gamma = pvec('gamma') * 1e-1
        drag_c = pvec('CD') * 1e-1
        alpha = pvec('alpha') * 1e-1

        dac_w = dpvec(['ac_w', 0])
        dwing_area = dpvec(['S', 0])
        dspeed = dpvec('v')
        drho = dpvec('rho')
        dfuel_w = dpvec('fuel_w')
        dgamma = dpvec('gamma')
        ddrag_c = dpvec('CD')
        dalpha = dpvec('alpha')
        dthrust_c = dgvec('CT_tar')

        if self.mode == 'fwd':
            dthrust_c[:] = 0.0
            if self.get_id('ac_w') in args:
                dthrust_c[:] += (numpy.sin(gamma) / (0.5*rho*speed**2*wing_area
                                                     *numpy.cos(alpha))
                                 * dac_w * 1e6/1e-1)
            if self.get_id('S') in args:
                dthrust_c[:] += -((ac_w + fuel_w)*numpy.sin(gamma)/
                                  (0.5*rho*speed**2*wing_area**2*
                                   numpy.cos(alpha)) * dwing_area * 1e2/1e-1)
            if self.get_id('v') in args:
                dthrust_c[:] += -2*(ac_w + fuel_w[:]) * numpy.sin(gamma[:]) /\
                                (0.5*rho[:]*speed[:]**3*wing_area* \
                                 numpy.cos(alpha[:])) * dspeed * 1e2/1e-1
            if self.get_id('rho') in args:
                dthrust_c[:] += -(ac_w + fuel_w[:]) * numpy.sin(gamma[:]) /\
                                (0.5*rho[:]**2*speed[:]**2*wing_area* \
                                 numpy.cos(alpha[:])) * drho / 1e-1
            if self.get_id('fuel_w') in args:
                dthrust_c[:] += numpy.sin(gamma[:]) /\
                                (0.5*rho[:]*speed[:]**2*wing_area* \
                                 numpy.cos(alpha[:])) * dfuel_w * 1e6/1e-1
            if self.get_id('gamma') in args:
                dthrust_c[:] += (ac_w + fuel_w[:])*numpy.cos(gamma[:]) /\
                                (0.5*rho[:]*speed[:]**2*wing_area* \
                                 numpy.cos(alpha[:])) * dgamma * 1e-1/1e-1
            if self.get_id('CD') in args:
                dthrust_c[:] += 1/numpy.cos(alpha[:]) * ddrag_c * 1e-1/1e-1
            if self.get_id('alpha') in args:
                dthrust_c[:] += (drag_c[:]*numpy.sin(alpha[:])/\
                                 (numpy.cos(alpha[:]))**2 \
                                 + (ac_w + fuel_w[:])*numpy.sin(gamma[:])* \
                                 numpy.sin(alpha[:])/(0.5*rho[:]*speed[:]**2* \
                                                      wing_area* \
                                                      (numpy.cos(
                                                          alpha[:]))**2)) \
                    * dalpha * 1e-1/1e-1

        if self.mode == 'rev':
            dac_w[:] = 0.0
            dwing_area[:] = 0.0
            dspeed[:] = 0.0
            drho[:] = 0.0
            dfuel_w[:] = 0.0
            dgamma[:] = 0.0
            ddrag_c[:] = 0.0
            dalpha[:] = 0.0
            if self.get_id('ac_w') in args:
                dac_w[:] += numpy.sum(numpy.sin(gamma)/
                                      (0.5*rho*speed**2*wing_area*
                                       numpy.cos(alpha))*
                                      dthrust_c) * 1e6/1e-1
            if self.get_id('S') in args:
                dwing_area[:] -= numpy.sum((ac_w + fuel_w)*numpy.sin(gamma)/
                                           (0.5*rho*speed**2*wing_area**2*
                                            numpy.cos(alpha))
                                           *dthrust_c) * 1e2/1e-1
            if self.get_id('v') in args:
                dspeed[:] += -2*(ac_w + fuel_w[:]) * numpy.sin(gamma[:]) /\
                         (0.5*rho[:]*speed[:]**3*wing_area* \
                          numpy.cos(alpha[:])) * dthrust_c * 1e2/1e-1
            if self.get_id('rho') in args:
                drho[:] += -(ac_w + fuel_w[:]) * numpy.sin(gamma[:]) /\
                           (0.5*rho[:]**2*speed[:]**2*wing_area* \
                            numpy.cos(alpha[:])) * dthrust_c  /1e-1
            if self.get_id('fuel_w') in args:
                dfuel_w[:] += numpy.sin(gamma[:]) /\
                          (0.5*rho[:]*speed[:]**2*wing_area* \
                           numpy.cos(alpha[:])) * dthrust_c * 1e6/1e-1
            if self.get_id('gamma') in args:
                dgamma[:] += (ac_w + fuel_w[:])*numpy.cos(gamma[:]) /\
                             (0.5*rho[:]*speed[:]**2*wing_area* \
                              numpy.cos(alpha[:])) * dthrust_c * 1e-1/1e-1
            if self.get_id('CD') in args:
                ddrag_c[:] += 1/numpy.cos(alpha[:]) * dthrust_c * 1e-1/1e-1
            if self.get_id('alpha') in args:
                dalpha[:] += ((drag_c[:]*numpy.sin(alpha[:])/
                               (numpy.cos(alpha[:]))**2
                               + (ac_w + fuel_w[:])*numpy.sin(gamma[:])*
                               numpy.sin(alpha[:])/(0.5*rho[:]*speed[:]**2* \
                                                    wing_area*
                                                    (numpy.cos(alpha[:]))**2)) \
                                  * dthrust_c) * 1e-1/1e-1


class SysFuelWeight(ExplicitSystem):
    """ computes the fuel consumption for each element, and compute
        the fuel weight carried at each element control point
    """

    def _declare(self):
        """ the owned variable is fuel weight: fuel_w
            the dependencies are: speed (v)
                                  pitch angle (gamma)
                                  thrust coefficient (CT)
                                  x distance (x)
                                  SFC (SFC)
                                  density (rho)
                                  wing area (S)
        """

        self.num_elem = self.kwargs['num_elem']
        fuel_w_0 = self.kwargs['fuel_w_0']
        num_pts = self.num_elem+1
        ind_pts = range(num_pts)

        self._declare_variable('fuel_w', size=num_pts, val=fuel_w_0)
        self._declare_argument('v', indices=ind_pts)
        self._declare_argument('gamma', indices=ind_pts)
        self._declare_argument('CT_tar', indices=ind_pts)
        self._declare_argument('x', indices=ind_pts)
        self._declare_argument('SFC', indices=ind_pts)
        self._declare_argument('rho', indices=ind_pts)
        self._declare_argument(['S', 0], indices=[0])

    def apply_G(self):
        """ the fuel burnt over each section is computed using trapezoidal
            rule and the neighboring control points
        """

        pvec = self.vec['p']
        uvec = self.vec['u']

        x_dist = pvec('x') * 1e6
        speed = pvec('v') * 1e2
        gamma = pvec('gamma') * 1e-1
        thrust_c = pvec('CT_tar') * 1e-1
        SFC = pvec('SFC') * 1e-6
        rho = pvec('rho')
        fuel_w_end = 0.0
        wing_area = pvec(['S', 0]) * 1e2
        fuel_w = uvec('fuel_w')

        fuel_delta = numpy.zeros(self.num_elem)
        x_int = numpy.zeros(self.num_elem)
        x_int = x_dist[1:] - x_dist[0:-1]
        q_int = 0.5*rho*speed**2*wing_area
        cos_gamma = numpy.cos(gamma)

        fuel_delta = ((SFC[0:-1] * thrust_c[0:-1] * q_int[0:-1] /
                       (speed[0:-1] * cos_gamma[0:-1]) + SFC[1:] *
                       thrust_c[1:] * q_int[1:] / (speed[1:] * cos_gamma[1:]))
                      * x_int/2)

        fuel_cumul = numpy.cumsum(fuel_delta[::-1])[::-1]
        fuel_w[0:-1] = (fuel_cumul + fuel_w_end) / 1e6
        fuel_w[-1] = fuel_w_end / 1e6

    def linearize(self):
        """ pre-compute the derivatives of fuel weight wrt speed (v),
            flight path angle (gamma), thrust_c (thrust coefficient),
            specific fuel consumption (SFC), and density (rho)
        """

        pvec = self.vec['p']

        x_dist = pvec('x') * 1e6
        speed = pvec('v') * 1e2
        gamma = pvec('gamma') * 1e-1
        thrust_c = pvec('CT_tar') * 1e-1
        SFC = pvec('SFC') * 1e-6
        rho = pvec('rho')
        wing_area = pvec(['S', 0]) * 1e2

        x_int = x_dist[1:] - x_dist[0:-1]
        q_int = 0.5*rho*speed**2*wing_area
        cos_gamma = numpy.cos(gamma)
        sin_gamma = numpy.sin(gamma)
        dq_dS = 0.5*rho*speed**2

        self.dfuel_dS = ((SFC[0:-1] * thrust_c[0:-1] * dq_dS[0:-1] /
                          (speed[0:-1] * cos_gamma[0:-1]) + SFC[1:] *
                          thrust_c[1:] * dq_dS[1:]/(speed[1:] * cos_gamma[1:]))
                         * x_int/2)
        self.dfuel_dx1 = ((SFC[0:-1] * thrust_c[0:-1] * q_int[0:-1] /
                           (speed[0:-1] * cos_gamma[0:-1]) + SFC[1:] *
                           thrust_c[1:] * q_int[1:]/(speed[1:] * cos_gamma[1:]))
                          * (-1)) / 2
        self.dfuel_dx2 = ((SFC[0:-1] * thrust_c[0:-1] * q_int[0:-1] /
                           (speed[0:-1] * cos_gamma[0:-1]) + SFC[1:] *
                           thrust_c[1:] * q_int[1:]/(speed[1:] * cos_gamma[1:]))
                          / 2)
        self.dfuel_dSFC1 = (thrust_c[0:-1] * q_int[0:-1] /
                            (speed[0:-1] * cos_gamma[0:-1]) * x_int/2)
        self.dfuel_dSFC2 = (thrust_c[1:] * q_int[1:] /
                            (speed[1:] * cos_gamma[1:])) * x_int/2
        self.dfuel_dthrust1 = (SFC[0:-1] * q_int[0:-1] /
                               (speed[0:-1] * cos_gamma[0:-1]) * x_int/2)
        self.dfuel_dthrust2 = (SFC[1:] * q_int[1:] /
                               (speed[1:] * cos_gamma[1:]) * x_int/2)
        self.dfuel_drho1 = (SFC[0:-1] * thrust_c[0:-1] *
                            0.5 * speed[0:-1] * wing_area /
                            cos_gamma[0:-1]) * x_int/2
        self.dfuel_drho2 = (SFC[1:] * thrust_c[1:] *
                            0.5 * speed[1:] * wing_area /
                            cos_gamma[1:])* x_int/2
        self.dfuel_dv1 = (SFC[0:-1] * thrust_c[0:-1] *
                          0.5 * rho[0:-1] * wing_area /
                          cos_gamma[0:-1]) * x_int/2
        self.dfuel_dv2 = (SFC[1:] * thrust_c[1:] *
                          0.5 * rho[1:] * wing_area /
                          cos_gamma[1:]) * x_int/2
        self.dfuel_dgamma1 = (SFC[0:-1] * thrust_c[0:-1] * q_int[0:-1] /
                              (speed[0:-1] * cos_gamma[0:-1]**2) *
                              (sin_gamma[0:-1]) * x_int/2)
        self.dfuel_dgamma2 = (SFC[1:] * thrust_c[1:] * q_int[1:] /
                              (speed[1:] * cos_gamma[1:]**2) *
                              (sin_gamma[1:]) * x_int/2)

    def apply_dGdp(self, args):
        """ apply the pre-computed derivatives to the directional
            derivatives required
        """

        dpvec = self.vec['dp']
        dgvec = self.vec['dg']

        dwing_area = dpvec('S')
        dx_dist = dpvec('x')
        dspeed = dpvec('v')
        dgamma = dpvec('gamma')
        dthrust_c = dpvec('CT_tar')
        dSFC = dpvec('SFC')
        drho = dpvec('rho')
        dfuel_w = dgvec('fuel_w')

        dfuel_dS = self.dfuel_dS

        dfuel_dx1 = self.dfuel_dx1
        dfuel_dSFC1 = self.dfuel_dSFC1
        dfuel_dthrust1 = self.dfuel_dthrust1
        dfuel_drho1 = self.dfuel_drho1
        dfuel_dv1 = self.dfuel_dv1
        dfuel_dgamma1 = self.dfuel_dgamma1

        dfuel_dx2 = self.dfuel_dx2
        dfuel_dSFC2 = self.dfuel_dSFC2
        dfuel_dthrust2 = self.dfuel_dthrust2
        dfuel_drho2 = self.dfuel_drho2
        dfuel_dv2 = self.dfuel_dv2
        dfuel_dgamma2 = self.dfuel_dgamma2

        dfuel_temp = numpy.zeros(self.num_elem+1)

        if self.mode == 'fwd':
            dfuel_w[:] = 0.0
            if self.get_id('S') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] = dfuel_dS * dwing_area
                dfuel_temp = numpy.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp * 1e2 / 1e6
            if self.get_id('x') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_dx1 * dx_dist[0:-1]
                dfuel_temp[0:-1] += dfuel_dx2 * dx_dist[1:]
                dfuel_temp = numpy.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp
            if self.get_id('v') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_dv1 * dspeed[0:-1]
                dfuel_temp[0:-1] += dfuel_dv2 * dspeed[1:]
                dfuel_temp = numpy.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp * 1e2/1e6
            if self.get_id('gamma') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_dgamma1 * dgamma[0:-1]
                dfuel_temp[0:-1] += dfuel_dgamma2 * dgamma[1:]
                dfuel_temp = numpy.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp * 1e-1/1e6
            if self.get_id('CT_tar') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_dthrust1 * dthrust_c[0:-1]
                dfuel_temp[0:-1] += dfuel_dthrust2 * dthrust_c[1:]
                dfuel_temp = numpy.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp * 1e-1/1e6
            if self.get_id('SFC') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_dSFC1 * dSFC[0:-1]
                dfuel_temp[0:-1] += dfuel_dSFC2 * dSFC[1:]
                dfuel_temp = numpy.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp * 1e-6/1e6
            if self.get_id('rho') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_drho1 * drho[0:-1]
                dfuel_temp[0:-1] += dfuel_drho2 * drho[1:]
                dfuel_temp = numpy.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp / 1e6

        elif self.mode == 'rev':
            dwing_area[:] = 0.0
            dx_dist[:] = 0.0
            dspeed[:] = 0.0
            dgamma[:] = 0.0
            dthrust_c[:] = 0.0
            dSFC[:] = 0.0
            drho[:] = 0.0
            if self.get_id('S') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = numpy.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dS * fuel_cumul
                dwing_area[:] += numpy.sum(dfuel_temp) * 1e2/1e6
            if self.get_id('x') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = numpy.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dx1 * fuel_cumul
                dfuel_temp[1:] += dfuel_dx2 * fuel_cumul
                dx_dist[:] += dfuel_temp
            if self.get_id('v') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = numpy.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dv1 * fuel_cumul
                dfuel_temp[1:] += dfuel_dv2 * fuel_cumul
                dspeed[:] += dfuel_temp * 1e2/1e6
            if self.get_id('gamma') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = numpy.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dgamma1 * fuel_cumul
                dfuel_temp[1:] += dfuel_dgamma2 * fuel_cumul
                dgamma[:] += dfuel_temp * 1e-1/1e6
            if self.get_id('CT_tar') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = numpy.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dthrust1 * fuel_cumul
                dfuel_temp[1:] += dfuel_dthrust2 * fuel_cumul
                dthrust_c[:] += dfuel_temp * 1e-1/1e6
            if self.get_id('SFC') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = numpy.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dSFC1 * fuel_cumul
                dfuel_temp[1:] += dfuel_dSFC2 * fuel_cumul
                dSFC[:] += dfuel_temp * 1e-6/1e6
            if self.get_id('rho') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = numpy.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_drho1 * fuel_cumul
                dfuel_temp[1:] += dfuel_drho2 * fuel_cumul
                drho[:] += dfuel_temp / 1e6

class SysAlpha(ImplicitSystem):
    """ system used to make user provided CL match with CL target """
    
    def _declare(self):
        """ owned variable: alpha (angle of attack)
            dependencies: CL (user provided coefficient of lift)
                          CL_tar (target coefficient of lift)
        """

        self.num_elem = self.kwargs['num_elem']
        num_pts = self.num_elem+1
        ind_pts = range(num_pts)

        self._declare_variable('alpha', size=num_pts)
        self._declare_argument('CL', indices=ind_pts)
        self._declare_argument('CL_tar', indices=ind_pts)

    def apply_F(self):
        """ the residual of the system is simply the difference between
            the two CL values
        """

        pvec = self.vec['p']
        fvec = self.vec['f']

        lift_c = pvec('CL')
        lift_c_tar = pvec('CL_tar')
        alpha_res = fvec('alpha')

        alpha_res[:] = lift_c - lift_c_tar

    def apply_dFdpu(self, args):
        """ compute the trivial derivatives of the system """

        dpvec = self.vec['dp']
        duvec = self.vec['du']
        dfvec = self.vec['df']

        dlift_c = dpvec('CL')
        dlift_c_tar = dpvec('CL_tar')
        dalpha_res = dfvec('alpha')
        dalpha = duvec('alpha')

        if self.mode == 'fwd':
            dalpha_res[:] = 0.0
            if self.get_id('CL') in args:
                dalpha_res[:] += dlift_c
            if self.get_id('CL_tar') in args:
                dalpha_res[:] -= dlift_c_tar

        elif self.mode == 'rev':
            dlift_c[:] = 0.0
            dlift_c_tar[:] = 0.0
            dalpha[:] = 0.0
            if self.get_id('CL') in args:
                dlift_c[:] += dalpha_res
            if self.get_id('CL_tar') in args:
                dlift_c_tar[:] -= dalpha_res


