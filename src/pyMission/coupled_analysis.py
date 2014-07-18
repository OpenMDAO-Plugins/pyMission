"""
INTENDED FOR MISSION ANALYSIS USE
The horizontal and vertical flight equilibrium equations are presented
and enforced. The fuel weight is also computed. The user-defined relation
between alpha, CL, and CD are not here!
"""

# pylint: disable=E1101
from __future__ import division
import sys

import numpy as np

from openmdao.main.api import Component, ImplicitComponent
from openmdao.main.datatypes.api import Array, Float

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103

class SysCLTar(Component):
    """ Compute CL values using collocation at the control points the
    vertical flight equilibrium equation is enforced.
    """

    def __init__(self, num_elem):
        super(SysCLTar, self).__init__()

        # Inputs
        self.add('fuel_w', Array(np.zeros((num_elem, )), iotype='in',
                                 desc = 'Fuel Weight'))
        self.add('Gamma', Array(np.zeros((num_elem, )), iotype='in',
                                 desc = 'Flight path angle'))
        self.add('CT_tar', Array(np.zeros((num_elem, )), iotype='in',
                                 desc = 'Thrust Coefficient'))
        self.add('rho', Array(np.zeros((num_elem, )), iotype='in',
                              desc = 'Density'))
        self.add('v', Array(np.zeros((num_elem, )), iotype='in',
                            desc = 'Speed'))
        self.add('alpha', Array(np.zeros((num_elem, )), iotype='in',
                                desc = 'Angle of attack'))
        self.add('S', Float(0.0, iotype='in', desc = 'Wing Area'))
        self.add('ac_w', Float(0.0, iotype='in',
                               desc = 'Weight of aircraft + payload'))

        # Outputs
        self.add('CL', Array(np.zeros((num_elem, )), iotype='out',
                             desc = 'Lift Coefficient'))

    def execute(self):
        """ Compute lift coefficient based on other variables at the control
        points. This is done by solving for CL with the vertical flight
        equilibrium equation.
        """

        fuel_w = self.fuel_w * 1e6
        Gamma = self.Gamma * 1e-1
        thrust_c = self.CT_tar * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6

        self.CL = (ac_w + fuel_w)*np.cos(Gamma) /\
                  (0.5*rho*speed**2*wing_area) - \
                  thrust_c*np.sin(alpha)

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['fuel_w', 'Gamma', 'CT_tar', 'alpha', 'rho',
                      'v', 'S', 'ac_w']
        output_keys = ['CL']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute lift coefficient derivatives wrt weight of aircraft, wing
        area, velocity, density, fuel weight, flight path angle, thrust
        coefficient, and angle of attack.
        Forward Mode
        """

        fuel_w = self.fuel_w * 1e6
        Gamma = self.Gamma * 1e-1
        thrust_c = self.CT_tar * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6

        if 'ac_w' in arg:
            result['CL'] += np.cos(Gamma) /(0.5*rho*speed**2*wing_area) *\
                           arg['ac_w']*1e6
        if 'S' in arg:
            result['CL'] += -(ac_w + fuel_w)*np.cos(Gamma) /\
                          (0.5*rho*speed**2*wing_area**2) *\
                          arg['S'] * 1e2
        if 'v' in arg:
            result['CL'] += -2*(ac_w + fuel_w)*np.cos(Gamma) /\
                          (0.5*rho*speed**3*wing_area) * arg['v'] * 1e2
        if 'rho' in arg:
            result['CL'] += -(ac_w + fuel_w)*np.cos(Gamma) /\
                          (0.5*rho**2*speed**2*wing_area) * arg['rho']
        if 'fuel_w' in arg:
            result['CL'] += np.cos(Gamma) /\
                          (0.5*rho*speed**2*wing_area) *\
                          arg['fuel_w'] * 1e6
        if 'Gamma' in arg:
            result['CL'] += -(ac_w + fuel_w)*np.sin(Gamma) /\
                          (0.5*rho*speed**2*wing_area) * arg['Gamma'] * 1e-1
        if 'CT_tar' in arg:
            result['CL'] += -np.sin(alpha) * arg['CT_tar'] * 1e-1
        if 'alpha' in arg:
            result['CL'] += -thrust_c*np.cos(alpha) * arg['alpha'] * 1e-1

    def apply_derivT(self, arg, result):
        """ Compute lift coefficient derivatives wrt weight of aircraft, wing
        area, velocity, density, fuel weight, flight path angle, thrust
        coefficient, and angle of attack.
        Adjoint Mode
        """

        fuel_w = self.fuel_w * 1e6
        Gamma = self.Gamma * 1e-1
        thrust_c = self.CT_tar * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6

        d_CL = arg['CL']

        if 'ac_w' in result:
            result['ac_w'] += np.dot(np.cos(Gamma) /\
                               (0.5*rho*speed**2*wing_area), d_CL) * 1e6
        if 'S' in result:
            result['S'] += -np.dot((ac_w + fuel_w)*np.cos(Gamma) /\
                            (0.5*rho*speed**2*wing_area**2), d_CL) * 1e2
        if 'v' in result:
            result['v'] += -2*(ac_w + fuel_w)*np.cos(Gamma) /\
                            (0.5*rho*speed**3*wing_area) * d_CL * 1e2
        if 'rho' in result:
            result['rho'] += -(ac_w + fuel_w)*np.cos(Gamma) /\
                              (0.5*rho**2*speed**2*wing_area) * d_CL
        if 'fuel_w' in result:
            result['fuel_w'] += np.cos(Gamma) /\
                                (0.5*rho*speed**2*wing_area) * d_CL * 1e6
        if 'Gamma' in result:
            result['Gamma'] += -(ac_w + fuel_w)*np.sin(Gamma) /\
                                (0.5*rho*speed**2*wing_area) * d_CL * 1e-1
        if 'CT_tar' in result:
            result['CT_tar'] += -np.sin(alpha) * d_CL * 1e-1
        if 'alpha' in result:
            result['alpha'] += -thrust_c*np.cos(alpha) * d_CL *1e-1


class SysCTTar(Component):
    """ Compute CT values using collocation at the control points the
    horizontal flight equilibrium equation is enforced.
    """

    def __init__(self, num_elem):
        super(SysCTTar, self).__init__()

        # Inputs

        self.add('fuel_w', Array(np.zeros((num_elem, )), iotype='in',
                                 desc = 'Fuel Weight'))
        self.add('Gamma', Array(np.zeros((num_elem, )), iotype='in',
                                 desc = 'Flight path angle'))
        self.add('CD', Array(np.zeros((num_elem, )), iotype='in',
                             desc = 'Drag Coefficient'))
        self.add('alpha', Array(np.zeros((num_elem, )), iotype='in',
                                desc = 'Angle of attack'))
        self.add('rho', Array(np.zeros((num_elem, )), iotype='in',
                              desc = 'Density'))
        self.add('v', Array(np.zeros((num_elem, )), iotype='in',
                            desc = 'Speed'))
        self.add('S', Float(0.0, iotype='in', desc = 'Wing Area'))
        self.add('ac_w', Float(0.0, iotype='in',
                               desc = 'Weight of aircraft + payload'))

        # Outputs
        self.add('CT_tar', Array(np.zeros((num_elem, )), iotype='out',
                                 desc = 'Thrust Coefficient'))

    def execute(self):
        """ Compute thrust coefficient using variables at the control points.
        The thrust coefficients are solved from the horizontal flight
        equilibrium equation.
        """

        fuel_w = self.fuel_w * 1e6
        Gamma = self.Gamma * 1e-1
        drag_c = self.CD * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6

        self.CT_tar = (drag_c/np.cos(alpha) +
                       (ac_w + fuel_w)*np.sin(Gamma) /
                       (0.5*rho*speed**2*wing_area*np.cos(alpha))) / 1e-1

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['fuel_w', 'Gamma', 'CD', 'alpha', 'rho',
                      'v', 'S', 'ac_w']
        output_keys = ['CT_tar']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute thrust coefficient derivatives wrt weight of aircraft,
        wing area, velocity, density, fuel weight, flight path angle, drag
        coefficient, and angle of attack.
        Forward Mode
        """

        fuel_w = self.fuel_w * 1e6
        Gamma = self.Gamma * 1e-1
        drag_c = self.CD * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6

        dthrust_c = result['CT_tar']

        if 'ac_w' in arg:
            dthrust_c += (np.sin(Gamma) / (0.5*rho*speed**2*wing_area
                           *np.cos(alpha)) * arg['ac_w'] * 1e6/1e-1)
        if 'S' in arg:
            dthrust_c += -((ac_w + fuel_w)*np.sin(Gamma)/
                              (0.5*rho*speed**2*wing_area**2*
                               np.cos(alpha)) * arg['S'] * 1e2/1e-1)
        if 'v' in arg:
            dthrust_c += -2*(ac_w + fuel_w) * np.sin(Gamma) /\
                            (0.5*rho*speed**3*wing_area* \
                             np.cos(alpha)) * arg['v'] * 1e2/1e-1
        if 'rho' in arg:
            dthrust_c += -(ac_w + fuel_w) * np.sin(Gamma) /\
                            (0.5*rho**2*speed**2*wing_area* \
                             np.cos(alpha)) * arg['rho'] / 1e-1
        if 'fuel_w' in arg:
            dthrust_c += np.sin(Gamma) /\
                            (0.5*rho*speed**2*wing_area* \
                             np.cos(alpha)) * arg['fuel_w'] * 1e6/1e-1
        if 'Gamma' in arg:
            dthrust_c += (ac_w + fuel_w)*np.cos(Gamma) /\
                            (0.5*rho*speed**2*wing_area* \
                             np.cos(alpha)) * arg['Gamma'] * 1e-1/1e-1
        if 'CD' in arg:
            dthrust_c += 1/np.cos(alpha) * arg['CD'] * 1e-1/1e-1
        if 'alpha' in arg:
            dthrust_c += (drag_c*np.sin(alpha)/\
                             (np.cos(alpha))**2 \
                             + (ac_w + fuel_w)*np.sin(Gamma)* \
                             np.sin(alpha)/(0.5*rho*speed**2* \
                             wing_area* (np.cos(alpha))**2)) \
                             * arg['alpha'] * 1e-1/1e-1

    def apply_derivT(self, arg, result):
        """ Compute thrust coefficient derivatives wrt weight of aircraft,
        wing area, velocity, density, fuel weight, flight path angle, drag
        coefficient, and angle of attack.
        Adjoint Mode
        """

        fuel_w = self.fuel_w * 1e6
        Gamma = self.Gamma * 1e-1
        drag_c = self.CD * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6

        dthrust_c = arg['CT_tar']

        if 'ac_w' in result:
            result['ac_w'] += np.sum(np.sin(Gamma)/
                                  (0.5*rho*speed**2*wing_area*
                                   np.cos(alpha))*
                                  dthrust_c) * 1e6/1e-1
        if 'S' in result:
            result['S'] -= np.sum((ac_w + fuel_w)*np.sin(Gamma)/
                           (0.5*rho*speed**2*wing_area**2*
                           np.cos(alpha))*dthrust_c) * 1e2/1e-1
        if 'v' in result:
            result['v'] += -2*(ac_w + fuel_w) * np.sin(Gamma) /\
                            (0.5*rho*speed**3*wing_area* \
                            np.cos(alpha)) * dthrust_c * 1e2/1e-1
        if 'rho' in result:
            result['rho'] += -(ac_w + fuel_w) * np.sin(Gamma) /\
                              (0.5*rho**2*speed**2*wing_area* \
                              np.cos(alpha)) * dthrust_c  /1e-1
        if 'fuel_w' in result:
            result['fuel_w'] += np.sin(Gamma) /\
                                (0.5*rho*speed**2*wing_area* \
                                np.cos(alpha)) * dthrust_c * 1e6/1e-1
        if 'Gamma' in result:
            result['Gamma'] += (ac_w + fuel_w)*np.cos(Gamma) /\
                               (0.5*rho*speed**2*wing_area* \
                               np.cos(alpha)) * dthrust_c * 1e-1/1e-1
        if 'CD' in result:
            result['CD'] += 1.0/np.cos(alpha) * dthrust_c * 1e-1/1e-1
        if 'alpha' in result:
            result['alpha'] += ((drag_c*np.sin(alpha)/(np.cos(alpha))**2
                               + (ac_w + fuel_w)*np.sin(Gamma)*
                                 np.sin(alpha)/(0.5*rho*speed**2* \
                                 wing_area * (np.cos(alpha))**2)) \
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

        fuel_delta = np.zeros(self.num_elem)
        x_int = np.zeros(self.num_elem)
        x_int = x_dist[1:] - x_dist[0:-1]
        q_int = 0.5*rho*speed**2*wing_area
        cos_gamma = np.cos(gamma)

        fuel_delta = ((SFC[0:-1] * thrust_c[0:-1] * q_int[0:-1] /
                       (speed[0:-1] * cos_gamma[0:-1]) + SFC[1:] *
                       thrust_c[1:] * q_int[1:] / (speed[1:] * cos_gamma[1:]))
                      * x_int/2)

        fuel_cumul = np.cumsum(fuel_delta[::-1])[::-1]
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
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
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

        dfuel_temp = np.zeros(self.num_elem+1)

        if self.mode == 'fwd':
            dfuel_w[:] = 0.0
            if 'S') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] = dfuel_dS * dwing_area
                dfuel_temp = np.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp * 1e2 / 1e6
            if 'x') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_dx1 * dx_dist[0:-1]
                dfuel_temp[0:-1] += dfuel_dx2 * dx_dist[1:]
                dfuel_temp = np.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp
            if 'v') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_dv1 * dspeed[0:-1]
                dfuel_temp[0:-1] += dfuel_dv2 * dspeed[1:]
                dfuel_temp = np.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp * 1e2/1e6
            if 'gamma') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_dgamma1 * dgamma[0:-1]
                dfuel_temp[0:-1] += dfuel_dgamma2 * dgamma[1:]
                dfuel_temp = np.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp * 1e-1/1e6
            if 'CT_tar') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_dthrust1 * dthrust_c[0:-1]
                dfuel_temp[0:-1] += dfuel_dthrust2 * dthrust_c[1:]
                dfuel_temp = np.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp * 1e-1/1e6
            if 'SFC') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_dSFC1 * dSFC[0:-1]
                dfuel_temp[0:-1] += dfuel_dSFC2 * dSFC[1:]
                dfuel_temp = np.cumsum(dfuel_temp[::-1])
                dfuel_temp = dfuel_temp[::-1]
                dfuel_w[:] += dfuel_temp * 1e-6/1e6
            if 'rho') in args:
                dfuel_temp[:] = 0.0
                dfuel_temp[0:-1] += dfuel_drho1 * drho[0:-1]
                dfuel_temp[0:-1] += dfuel_drho2 * drho[1:]
                dfuel_temp = np.cumsum(dfuel_temp[::-1])
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
            if 'S') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = np.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dS * fuel_cumul
                dwing_area[:] += np.sum(dfuel_temp) * 1e2/1e6
            if 'x') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = np.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dx1 * fuel_cumul
                dfuel_temp[1:] += dfuel_dx2 * fuel_cumul
                dx_dist[:] += dfuel_temp
            if 'v') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = np.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dv1 * fuel_cumul
                dfuel_temp[1:] += dfuel_dv2 * fuel_cumul
                dspeed[:] += dfuel_temp * 1e2/1e6
            if 'gamma') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = np.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dgamma1 * fuel_cumul
                dfuel_temp[1:] += dfuel_dgamma2 * fuel_cumul
                dgamma[:] += dfuel_temp * 1e-1/1e6
            if 'CT_tar') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = np.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dthrust1 * fuel_cumul
                dfuel_temp[1:] += dfuel_dthrust2 * fuel_cumul
                dthrust_c[:] += dfuel_temp * 1e-1/1e6
            if 'SFC') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = np.cumsum(dfuel_w[0:-1])
                dfuel_temp[0:-1] += dfuel_dSFC1 * fuel_cumul
                dfuel_temp[1:] += dfuel_dSFC2 * fuel_cumul
                dSFC[:] += dfuel_temp * 1e-6/1e6
            if 'rho') in args:
                dfuel_temp[:] = 0.0
                fuel_cumul = np.cumsum(dfuel_w[0:-1])
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
            if 'CL') in args:
                dalpha_res[:] += dlift_c
            if 'CL_tar') in args:
                dalpha_res[:] -= dlift_c_tar

        elif self.mode == 'rev':
            dlift_c[:] = 0.0
            dlift_c_tar[:] = 0.0
            dalpha[:] = 0.0
            if 'CL') in args:
                dlift_c[:] += dalpha_res
            if 'CL_tar') in args:
                dlift_c_tar[:] -= dalpha_res


