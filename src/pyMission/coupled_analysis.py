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

import numpy as np

from openmdao.main.api import Component, ImplicitComponent
from openmdao.main.datatypes.api import Array, Float

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103

class SysCLTar(Component):
    """ Compute CL values using collocation at the control points the
    vertical flight equilibrium equation is enforced.
    """

    def __init__(self, num_elem=10, wt_pax=0.0, S=0, ac_w=0):
        super(SysCLTar, self).__init__()

        self.wt_pax = wt_pax
        self.S = S
        self.ac_w = ac_w

        # Inputs
        self.add('fuel_w', Array(np.zeros((num_elem+1, )), iotype='in',
                                 desc = 'Fuel Weight'))
        self.add('Gamma', Array(np.zeros((num_elem+1, )), iotype='in',
                                 desc = 'Flight path angle'))
        self.add('CT_tar', Array(np.zeros((num_elem+1, )), iotype='in',
                                 desc = 'Thrust Coefficient'))
        self.add('rho', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Density'))
        self.add('v', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Speed'))
        self.add('alpha', Array(np.ones((num_elem+1, )), iotype='in',
                                desc = 'Angle of attack'))
        self.add('pax_flt', Float(0.0, iotype='in',
                                  desc = 'Passengers per flight'))

        # Outputs
        self.add('CL', Array(np.zeros((num_elem+1, )), iotype='out',
                             desc = 'Lift Coefficient'))

        self.fuel_scale = 1e5
        #self.fuel_scale = 1e6

    def execute(self):
        """ Compute lift coefficient based on other variables at the control
        points. This is done by solving for CL with the vertical flight
        equilibrium equation.
        """

        fuel_w = self.fuel_w * self.fuel_scale
        Gamma = self.Gamma * 1e-1
        thrust_c = self.CT_tar * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6
        pax_flt = self.pax_flt
        wt_pax = self.wt_pax

        self.CL = (ac_w + fuel_w + wt_pax * pax_flt)*np.cos(Gamma) /\
                  (0.5*rho*speed**2*wing_area) - \
                  thrust_c*np.sin(alpha)

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['fuel_w', 'Gamma', 'CT_tar', 'alpha', 'rho',
                      'v', 'pax_flt']
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

        fuel_w = self.fuel_w * self.fuel_scale
        Gamma = self.Gamma * 1e-1
        thrust_c = self.CT_tar * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6
        pax_flt = self.pax_flt
        wt_pax = self.wt_pax

        if 'ac_w' in arg:
            result['CL'] += np.cos(Gamma) /(0.5*rho*speed**2*wing_area) *\
                           arg['ac_w']*1e6
        if 'S' in arg:
            result['CL'] += -(ac_w + fuel_w + wt_pax * pax_flt)*np.cos(Gamma) /\
                          (0.5*rho*speed**2*wing_area**2) *\
                          arg['S'] * 1e2
        if 'v' in arg:
            result['CL'] += -2*(ac_w + fuel_w + wt_pax * pax_flt)*np.cos(Gamma) /\
                          (0.5*rho*speed**3*wing_area) * arg['v'] * 1e2
        if 'rho' in arg:
            result['CL'] += -(ac_w + fuel_w + wt_pax * pax_flt)*np.cos(Gamma) /\
                          (0.5*rho**2*speed**2*wing_area) * arg['rho']
        if 'fuel_w' in arg:
            result['CL'] += np.cos(Gamma) /\
                          (0.5*rho*speed**2*wing_area) *\
                          arg['fuel_w'] * self.fuel_scale
        if 'Gamma' in arg:
            result['CL'] += -(ac_w + fuel_w + wt_pax * pax_flt)*np.sin(Gamma) /\
                          (0.5*rho*speed**2*wing_area) * arg['Gamma'] * 1e-1
        if 'CT_tar' in arg:
            result['CL'] += -np.sin(alpha) * arg['CT_tar'] * 1e-1
        if 'alpha' in arg:
            result['CL'] += -thrust_c*np.cos(alpha) * arg['alpha'] * 1e-1
        if 'pax_flt' in arg:
            result['CL'] += wt_pax * np.cos(Gamma) / (0.5*rho*speed**2*wing_area) * arg['pax_flt']

    def apply_derivT(self, arg, result):
        """ Compute lift coefficient derivatives wrt weight of aircraft, wing
        area, velocity, density, fuel weight, flight path angle, thrust
        coefficient, and angle of attack.
        Adjoint Mode
        """

        fuel_w = self.fuel_w * self.fuel_scale
        Gamma = self.Gamma * 1e-1
        thrust_c = self.CT_tar * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6
        wt_pax = self.wt_pax
        pax_flt = self.pax_flt

        d_CL = arg['CL']

        if 'ac_w' in result:
            result['ac_w'] += np.dot(np.cos(Gamma) /\
                               (0.5*rho*speed**2*wing_area), d_CL) * 1e6
        if 'S' in result:
            result['S'] += -np.dot((ac_w + fuel_w + wt_pax * pax_flt)*np.cos(Gamma) /\
                            (0.5*rho*speed**2*wing_area**2), d_CL) * 1e2
        if 'v' in result:
            result['v'] += -2*(ac_w + fuel_w + wt_pax * pax_flt)*np.cos(Gamma) /\
                            (0.5*rho*speed**3*wing_area) * d_CL * 1e2
        if 'rho' in result:
            result['rho'] += -(ac_w + fuel_w + wt_pax * pax_flt)*np.cos(Gamma) /\
                              (0.5*rho**2*speed**2*wing_area) * d_CL
        if 'fuel_w' in result:
            result['fuel_w'] += np.cos(Gamma) /\
                                (0.5*rho*speed**2*wing_area) * d_CL * self.fuel_scale
        if 'Gamma' in result:
            result['Gamma'] += -(ac_w + fuel_w + wt_pax * pax_flt)*np.sin(Gamma) /\
                                (0.5*rho*speed**2*wing_area) * d_CL * 1e-1
        if 'CT_tar' in result:
            result['CT_tar'] += -np.sin(alpha) * d_CL * 1e-1
        if 'alpha' in result:
            result['alpha'] += -thrust_c*np.cos(alpha) * d_CL *1e-1
        if 'pax_flt' in result:
            result['pax_flt'] += np.sum(wt_pax * np.cos(Gamma) / (0.5*rho*speed**2*wing_area) * d_CL)


class SysCTTar(Component):
    """ Compute CT values using collocation at the control points the
    horizontal flight equilibrium equation is enforced.
    """

    def __init__(self, num_elem=10, wt_pax=0.0, S=0, ac_w=0):
        super(SysCTTar, self).__init__()

        self.wt_pax = wt_pax
        self.S = S
        self.ac_w = ac_w

        # Inputs
        self.add('fuel_w', Array(np.zeros((num_elem+1, )), iotype='in',
                                 desc = 'Fuel Weight'))
        self.add('Gamma', Array(np.zeros((num_elem+1, )), iotype='in',
                                 desc = 'Flight path angle'))
        self.add('CD', Array(np.zeros((num_elem+1, )), iotype='in',
                             desc = 'Drag Coefficient'))
        self.add('alpha', Array(np.ones((num_elem+1, )), iotype='in',
                                desc = 'Angle of attack'))
        self.add('rho', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Density'))
        self.add('v', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Speed'))
        self.add('pax_flt', Float(0.0, iotype='in',
                                  desc = 'Passengers per flight'))

        # Outputs
        self.add('CT_tar', Array(np.zeros((num_elem+1, )), iotype='out',
                                 desc = 'Thrust Coefficient'))

        #self.fuel_scale = 1e6
        self.fuel_scale = 1e5

    def execute(self):
        """ Compute thrust coefficient using variables at the control points.
        The thrust coefficients are solved from the horizontal flight
        equilibrium equation.
        """

        fuel_w = self.fuel_w * self.fuel_scale
        Gamma = self.Gamma * 1e-1
        drag_c = self.CD * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6
        pax_flt = self.pax_flt
        wt_pax = self.wt_pax

        self.CT_tar = (drag_c/np.cos(alpha) +
                       (ac_w + fuel_w + wt_pax * pax_flt)*np.sin(Gamma) /
                       (0.5*rho*speed**2*wing_area*np.cos(alpha))) / 1e-1

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['fuel_w', 'Gamma', 'CD', 'alpha', 'rho',
                      'v', 'pax_flt']
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

        fuel_w = self.fuel_w * self.fuel_scale
        Gamma = self.Gamma * 1e-1
        drag_c = self.CD * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6
        pax_flt = self.pax_flt
        wt_pax = self.wt_pax

        dthrust_c = result['CT_tar']
        cos_alpha = np.cos(alpha)
        sin_gamma = np.sin(Gamma)
        fact = 0.5*rho*speed**2*wing_area*cos_alpha

        if 'ac_w' in arg:
            dthrust_c += (sin_gamma / (fact) * \
                          arg['ac_w'] * 1e6/1e-1)
        if 'S' in arg:
            dthrust_c += -((ac_w + fuel_w + wt_pax * pax_flt)*sin_gamma / \
                              (fact*wing_area) * \
                              arg['S'] * 1e2/1e-1)
        if 'v' in arg:
            dthrust_c += -2*(ac_w + fuel_w + wt_pax * pax_flt) * sin_gamma /\
                            (fact*speed) * \
                            arg['v'] * 1e2/1e-1
        if 'rho' in arg:
            dthrust_c += -(ac_w + fuel_w + wt_pax * pax_flt) * sin_gamma /\
                            (fact*rho) * arg['rho'] / 1e-1
        if 'fuel_w' in arg:
            dthrust_c += sin_gamma /\
                            (fact) * arg['fuel_w'] * self.fuel_scale/1e-1
        if 'Gamma' in arg:
            dthrust_c += (ac_w + fuel_w + wt_pax * pax_flt)*np.cos(Gamma) /\
                            (fact) * arg['Gamma'] * 1e-1/1e-1
        if 'CD' in arg:
            dthrust_c += 1.0/cos_alpha * arg['CD'] * 1e-1/1e-1
        if 'alpha' in arg:
            dthrust_c += (drag_c*np.sin(alpha)/\
                             (cos_alpha)**2 \
                             + (ac_w + fuel_w + wt_pax * pax_flt)*sin_gamma* \
                             np.sin(alpha)/(fact*cos_alpha)) \
                             * arg['alpha'] * 1e-1/1e-1
        if 'pax_flt' in arg:
            dthrust_c += wt_pax * np.sin(Gamma) /\
                       (0.5*rho*speed**2*wing_area*np.cos(alpha)) * arg['pax_flt'] / 1e-1

    def apply_derivT(self, arg, result):
        """ Compute thrust coefficient derivatives wrt weight of aircraft,
        wing area, velocity, density, fuel weight, flight path angle, drag
        coefficient, and angle of attack.
        Adjoint Mode
        """

        fuel_w = self.fuel_w * self.fuel_scale
        Gamma = self.Gamma * 1e-1
        drag_c = self.CD * 1e-1
        alpha = self.alpha * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        wing_area = self.S * 1e2
        ac_w = self.ac_w * 1e6
        pax_flt = self.pax_flt
        wt_pax = self.wt_pax

        dthrust_c = arg['CT_tar']
        cos_alpha = np.cos(alpha)
        sin_gamma = np.sin(Gamma)
        fact = 0.5*rho*speed**2*wing_area

        if 'ac_w' in result:
            result['ac_w'] += np.sum(sin_gamma/(fact*cos_alpha) * \
                                  dthrust_c) * 1e6/1e-1
        if 'S' in result:
            result['S'] -= np.sum((ac_w + fuel_w + wt_pax * pax_flt)*sin_gamma/
                           (fact*wing_area*cos_alpha) * \
                           dthrust_c) * 1e2/1e-1
        if 'v' in result:
            result['v'] += -2*(ac_w + fuel_w + wt_pax * pax_flt) * sin_gamma /\
                            (fact*speed*cos_alpha) * \
                            dthrust_c * 1e2/1e-1
        if 'rho' in result:
            result['rho'] += -(ac_w + fuel_w + wt_pax * pax_flt) * sin_gamma /\
                              (fact*rho*cos_alpha) * dthrust_c  /1e-1
        if 'fuel_w' in result:
            result['fuel_w'] += sin_gamma /\
                                (fact*cos_alpha) * dthrust_c * self.fuel_scale/1e-1
        if 'Gamma' in result:
            result['Gamma'] += (ac_w + fuel_w + wt_pax * pax_flt)*np.cos(Gamma) /\
                               (fact*cos_alpha) * dthrust_c * 1e-1/1e-1
        if 'CD' in result:
            result['CD'] += 1.0/cos_alpha * dthrust_c * 1e-1/1e-1
        if 'alpha' in result:
            result['alpha'] += ((drag_c*np.sin(alpha)/(cos_alpha)**2
                               + (ac_w + fuel_w + wt_pax * pax_flt)*sin_gamma*
                                 np.sin(alpha)/(fact*(cos_alpha)**2)) \
                                 * dthrust_c) * 1e-1/1e-1
        if 'pax_flt' in result:
            result['pax_flt'] += np.sum(wt_pax * np.sin(Gamma) /\
                       (0.5*rho*speed**2*wing_area*np.cos(alpha)) * dthrust_c / 1e-1)


class SysFuelWeight(Component):
    """ Computes the fuel consumption for each element, and compute the fuel
    weight carried at each element control point
    """

    def __init__(self, num_elem=10, S=0):
        super(SysFuelWeight, self).__init__()

        self.S = S

        # Inputs
        self.add('v', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Speed'))
        self.add('Gamma', Array(np.zeros((num_elem+1, )), iotype='in',
                                 desc = 'Pitch angle'))
        self.add('CT_tar', Array(np.zeros((num_elem+1, )), iotype='in',
                                 desc = 'Thrust Coefficient'))
        self.add('x', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Distance'))
        self.add('SFC', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Specific Fuel Consumption'))
        self.add('rho', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Density'))

        # Note, this isn't used in the calculation, but it may have been used
        # to help convergence due to lack of graph.
        #self.add('fuel_w_0', Float(0.0, iotype='in',
        #                           desc = 'Initial Fuel Weight'))

        # Outputs
        self.add('fuel_w', Array(np.zeros((num_elem+1, )), iotype='out',
                                 desc = 'Fuel Weight'))

        #self.fuel_scale = 1e6
        self.fuel_scale = 1e5

    def execute(self):
        """ The fuel burnt over each section is computed using trapezoidal
        rule and the neighboring control points.
        """

        x_dist = self.x * 1e6
        speed = self.v * 1e2
        gamma = self.Gamma * 1e-1
        thrust_c = self.CT_tar * 1e-1
        SFC = self.SFC * 1e-6
        rho = self.rho

        fuel_w_end = 0.0
        wing_area = self.S * 1e2

        num_elem = len(x_dist)

        fuel_delta = np.zeros(num_elem)
        x_int = np.zeros(num_elem)
        x_int = x_dist[1:] - x_dist[0:-1]
        q_int = 0.5*rho*speed**2*wing_area
        cos_gamma = np.cos(gamma)

        fuel_delta = ((SFC[0:-1] * thrust_c[0:-1] * q_int[0:-1] /
                       (speed[0:-1] * cos_gamma[0:-1]) + SFC[1:] *
                       thrust_c[1:] * q_int[1:] / (speed[1:] * cos_gamma[1:]))
                      * x_int/2)

        fuel_cumul = np.cumsum(fuel_delta[::-1])[::-1]
        self.fuel_w[0:-1] = (fuel_cumul + fuel_w_end) / self.fuel_scale
        self.fuel_w[-1] = fuel_w_end / self.fuel_scale

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['v', 'Gamma', 'CT_tar', 'x', 'SFC', 'rho']
        output_keys = ['fuel_w']
        return input_keys, output_keys

    def provideJ(self):
        """ Pre-compute the derivatives of fuel weight wrt speed (v), flight
        path angle (gamma), thrust_c (thrust coefficient), specific fuel
        consumption (SFC), and density (rho).
        """

        x_dist = self.x * 1e6
        speed = self.v * 1e2
        gamma = self.Gamma * 1e-1
        thrust_c = self.CT_tar * 1e-1
        SFC = self.SFC * 1e-6
        rho = self.rho
        wing_area = self.S * 1e2

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

    def apply_deriv(self, arg, result):
        """ Apply the pre-computed derivatives to the directional derivatives
        required.
        Forward mode.
        """

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

        dfuel_temp = np.zeros(len(self.x))

        if 'S' in arg:
            dfuel_temp[:] = 0.0
            dfuel_temp[0:-1] = dfuel_dS * arg['S']
            dfuel_temp = np.cumsum(dfuel_temp[::-1])
            dfuel_temp = dfuel_temp[::-1]
            result['fuel_w'] += dfuel_temp * 1e2 / self.fuel_scale
        if 'x' in arg:
            dfuel_temp[:] = 0.0
            dfuel_temp[0:-1] += dfuel_dx1 * arg['x'][0:-1]
            dfuel_temp[0:-1] += dfuel_dx2 * arg['x'][1:]
            dfuel_temp = np.cumsum(dfuel_temp[::-1])
            dfuel_temp = dfuel_temp[::-1]
            result['fuel_w'] += dfuel_temp * 1e6/self.fuel_scale
        if 'v' in arg:
            dfuel_temp[:] = 0.0
            dfuel_temp[0:-1] += dfuel_dv1 * arg['v'][0:-1]
            dfuel_temp[0:-1] += dfuel_dv2 * arg['v'][1:]
            dfuel_temp = np.cumsum(dfuel_temp[::-1])
            dfuel_temp = dfuel_temp[::-1]
            result['fuel_w'] += dfuel_temp * 1e2/self.fuel_scale
        if 'Gamma' in arg:
            dfuel_temp[:] = 0.0
            dfuel_temp[0:-1] += dfuel_dgamma1 * arg['Gamma'][0:-1]
            dfuel_temp[0:-1] += dfuel_dgamma2 * arg['Gamma'][1:]
            dfuel_temp = np.cumsum(dfuel_temp[::-1])
            dfuel_temp = dfuel_temp[::-1]
            result['fuel_w'] += dfuel_temp * 1e-1/self.fuel_scale
        if 'CT_tar' in arg:
            dfuel_temp[:] = 0.0
            dfuel_temp[0:-1] += dfuel_dthrust1 * arg['CT_tar'][0:-1]
            dfuel_temp[0:-1] += dfuel_dthrust2 * arg['CT_tar'][1:]
            dfuel_temp = np.cumsum(dfuel_temp[::-1])
            dfuel_temp = dfuel_temp[::-1]
            result['fuel_w'] += dfuel_temp * 1e-1/self.fuel_scale
        if 'SFC' in arg:
            dfuel_temp[:] = 0.0
            dfuel_temp[0:-1] += dfuel_dSFC1 * arg['SFC'][0:-1]
            dfuel_temp[0:-1] += dfuel_dSFC2 * arg['SFC'][1:]
            dfuel_temp = np.cumsum(dfuel_temp[::-1])
            dfuel_temp = dfuel_temp[::-1]
            result['fuel_w'] += dfuel_temp * 1e-6/self.fuel_scale
        if 'rho' in arg:
            dfuel_temp[:] = 0.0
            dfuel_temp[0:-1] += dfuel_drho1 * arg['rho'][0:-1]
            dfuel_temp[0:-1] += dfuel_drho2 * arg['rho'][1:]
            dfuel_temp = np.cumsum(dfuel_temp[::-1])
            dfuel_temp = dfuel_temp[::-1]
            result['fuel_w'] += dfuel_temp / self.fuel_scale

    def apply_derivT(self, arg, result):
        """ Apply the pre-computed derivatives to the directional derivatives
        required.
        Adjoint mode.
        """

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

        dfuel_temp = np.zeros(len(self.x))

        if 'S' in result:
            dfuel_temp[:] = 0.0
            fuel_cumul = np.cumsum(arg['fuel_w'][0:-1])
            dfuel_temp[0:-1] += dfuel_dS * fuel_cumul
            result['S'] += np.sum(dfuel_temp) * 1e2/self.fuel_scale
        if 'x' in result:
            dfuel_temp[:] = 0.0
            fuel_cumul = np.cumsum(arg['fuel_w'][0:-1])
            dfuel_temp[0:-1] += dfuel_dx1 * fuel_cumul
            dfuel_temp[1:] += dfuel_dx2 * fuel_cumul
            result['x'] += dfuel_temp * 1e6/self.fuel_scale
        if 'v' in result:
            dfuel_temp[:] = 0.0
            fuel_cumul = np.cumsum(arg['fuel_w'][0:-1])
            dfuel_temp[0:-1] += dfuel_dv1 * fuel_cumul
            dfuel_temp[1:] += dfuel_dv2 * fuel_cumul
            result['v'] += dfuel_temp * 1e2/self.fuel_scale
        if 'Gamma' in result:
            dfuel_temp[:] = 0.0
            fuel_cumul = np.cumsum(arg['fuel_w'][0:-1])
            dfuel_temp[0:-1] += dfuel_dgamma1 * fuel_cumul
            dfuel_temp[1:] += dfuel_dgamma2 * fuel_cumul
            result['Gamma'] += dfuel_temp * 1e-1/self.fuel_scale
        if 'CT_tar' in result:
            dfuel_temp[:] = 0.0
            fuel_cumul = np.cumsum(arg['fuel_w'][0:-1])
            dfuel_temp[0:-1] += dfuel_dthrust1 * fuel_cumul
            dfuel_temp[1:] += dfuel_dthrust2 * fuel_cumul
            result['CT_tar'] += dfuel_temp * 1e-1/self.fuel_scale
        if 'SFC' in result:
            dfuel_temp[:] = 0.0
            fuel_cumul = np.cumsum(arg['fuel_w'][0:-1])
            dfuel_temp[0:-1] += dfuel_dSFC1 * fuel_cumul
            dfuel_temp[1:] += dfuel_dSFC2 * fuel_cumul
            result['SFC'] += dfuel_temp * 1e-6/self.fuel_scale
        if 'rho' in result:
            dfuel_temp[:] = 0.0
            fuel_cumul = np.cumsum(arg['fuel_w'][0:-1])
            dfuel_temp[0:-1] += dfuel_drho1 * fuel_cumul
            dfuel_temp[1:] += dfuel_drho2 * fuel_cumul
            result['rho'] += dfuel_temp / self.fuel_scale
