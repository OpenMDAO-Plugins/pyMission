"""
INTENDED FOR MISSION ANALYSIS USE
Atmospheric models for specific fuel consumption (SFC), temperature, and
density. All models extracted from the linear portion of the standard
atmosphere

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

from openmdao.main.api import Component
from openmdao.main.datatypes.api import Array, Float, Bool

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103

class SysTemp(Component):
    """ Linear temperature model using standard atmosphere with smoothing
    at the temperature discontinuity.
    """

    def __init__(self, num_elem=10):
        super(SysTemp, self).__init__()

        # Inputs
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Altitude'))

        # Outputs
        self.add('temp', Array(np.zeros((num_elem+1, )), iotype='out', low=0.001,
                            desc = 'Temperature'))

        self.epsilon = 500
        h_lower = 11000 - self.epsilon
        h_upper = 11000 + self.epsilon

        matrix = np.array([[h_lower**3, h_lower**2, h_lower, 1],
                           [h_upper**3, h_upper**2, h_upper, 1],
                           [3*h_lower**2, 2*h_lower, 1, 0],
                           [3*h_upper**2, 2*h_upper, 1, 0]])
        rhs = np.array([288.16-(6.5e-3)*h_lower, 216.65,
                           -6.5e-3, 0])
        self.coefs = np.linalg.solve(matrix, rhs)

    def execute(self):
        """ Temperature model extracted from linear portion and constant
        portion of the standard atmosphere.
        """

        alt = self.h * 1e3
        alt_boundary = 11000
        n_elem = len(self.h)
        temp = self.temp

        a = self.coefs[0]
        b = self.coefs[1]
        c = self.coefs[2]
        d = self.coefs[3]

        for index in xrange(n_elem):
            if alt[index] <= (alt_boundary - self.epsilon):
                temp[index] = (288.16 - (6.5e-3) * alt[index]) / 1e2
            elif alt[index] >= (alt_boundary + self.epsilon):
                temp[index] = 216.65 / 1e2
            else:
                h_star = alt[index]
                temp[index] = (a*h_star**3 + b*h_star**2 + c*h_star + d) / 1e2

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['h']
        output_keys = ['temp']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute temperature derivative wrt altitude
        Forward mode.
        """

        alt = self.h * 1e3
        alt_boundary = 11000
        n_elem = len(self.h)

        a = self.coefs[0]
        b = self.coefs[1]
        c = self.coefs[2]

        for index in xrange(n_elem):
            if alt[index] <= (alt_boundary - self.epsilon):
                result['temp'][index] += -6.5e-3 *  arg['h'][index] * 1e3/1e2
            elif alt[index] >= (alt_boundary + self.epsilon):
                result['temp'][index] += 0.0
            else:
                h_star = alt[index]
                result['temp'][index] += (3*a*h_star**2 + 2*b*h_star
                                 + c) *  arg['h'][index] * 1e3/1e2

    def apply_derivT(self, arg, result):
        """ Compute temperature derivative wrt altitude
        Adjoint mode.
        """

        alt = self.h * 1e3
        alt_boundary = 11000
        n_elem = len(self.h)

        a = self.coefs[0]
        b = self.coefs[1]
        c = self.coefs[2]

        for index in xrange(n_elem):
            if alt[index] <= (alt_boundary - self.epsilon):
                result['h'][index] += -6.5e-3 * arg['temp'][index] * 1e3/1e2
            elif alt[index] >= (alt_boundary + self.epsilon):
                result['h'][index] += 0.0
            else:
                h_star = alt[index]
                result['h'][index] += (3*a*h_star**2 + 2*b*h_star
                                + c) * arg['temp'][index] * 1e3/1e2


class SysTempOld(Component):
    """ Linear temperature model using the standard atmosphere """

    def __init__(self, num_elem=10):
        super(SysTempOld, self).__init__()

        # Inputs
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Altitude'))

        # Outputs
        self.add('temp', Array(np.zeros((num_elem+1, )), iotype='out', low=0.001,
                            desc = 'Temperature'))

    def execute(self):
        """ Temperature model extracted from linear portion of the standard
        atmosphere.
        """

        alt = self.h * 1e3

        self.temp = (288.16 - (6.5e-3) * alt) / 1e2

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['h']
        output_keys = ['temp']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute temperature derivative wrt altitude.
        Forward mode.
        """

        dtemp_dalt = -6.5e-3

        result['temp'] += (dtemp_dalt * arg['h']) * 1e3/1e2

    def apply_derivT(self, arg, result):
        """ Compute temperature derivative wrt altitude.
        Adjoint mode.
        """

        dtemp_dalt = -6.5e-3

        result['h'] += dtemp_dalt * arg['temp'] * 1e3/1e2


class SysRho(Component):
    """ Density model using standard atmosphere model with troposphere,
    stratosphere.
    """

    def __init__(self, num_elem=10):
        super(SysRho, self).__init__()

        # Inputs
        self.add('temp', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Temperature'))
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Altitude'))

        # Outputs
        self.add('rho', Array(np.zeros((num_elem+1, )), iotype='out', low=0.001,
                            desc = 'Density'))

        self.epsilon = 500
        h_lower = 11000 - self.epsilon
        h_upper = 11000 + self.epsilon
        matrix = np.array([[h_lower**3, h_lower**2, h_lower, 1],
                           [h_upper**3, h_upper**2, h_upper, 1],
                           [3*h_lower**2, 2*h_lower, 1, 0],
                           [3*h_upper**2, 2*h_upper, 1, 0]])
        rhs = np.array([101325*(1-0.0065*h_lower/288.16)**5.2561,
                        22632*np.exp(-9.81*self.epsilon/(288*216.65)),
                        (-101325*5.2561*(0.0065/288.16)*
                         (1-0.0065*h_lower/288.15)**4.2561),
                        (22632*(-9.81/(288*216.65))*
                         np.exp(-9.81*self.epsilon/(288*216.65)))])
        self.coefs = np.linalg.solve(matrix, rhs)

    def execute(self):
        """ Density model extracted from the standard atmosphere. Depends on
        the temperature and the altitude. Model is valid for troposphere and
        stratosphere, and accounts for the linear decreasing temperature
        segment (troposphere), and the constant temperature segment.
        (stratosphere)
        """

        temp = self.temp * 1e2
        alt = self.h * 1e3
        n_elem = len(self.h)
        rho = self.rho

        alt_boundary = 11000
        a = self.coefs[0]
        b = self.coefs[1]
        c = self.coefs[2]
        d = self.coefs[3]

        for index in xrange(n_elem):
            if alt[index] <= (alt_boundary - self.epsilon):
                pressure = 101325*(1-0.0065*alt[index]/288.16)**5.2561
                rho[index] = pressure / (288 * temp[index])
            elif alt[index] >= (alt_boundary + self.epsilon):
                pressure = 22632*np.exp(-9.81*(alt[index]-alt_boundary)/
                                             (288*216.65))
                rho[index] = pressure / (288 * temp[index])
            else:
                h_star = alt[index]
                pressure = a*h_star**3 + b*h_star**2 + c*h_star + d
                rho[index] = pressure / (288 * temp[index])

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['temp', 'h']
        output_keys = ['rho']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute density derivative wrt altitude and temperature.
        Forward mode.
        """

        alt = self.h * 1e3
        n_elem = len(alt)
        temp = self.temp * 1e2
        alt_boundary = 11000

        a = self.coefs[0]
        b = self.coefs[1]
        c = self.coefs[2]
        d = self.coefs[3]

        drho = result['rho']

        if 'h' in arg:
            dalt = arg['h']
            for index in xrange(n_elem):
                if alt[index] <= (alt_boundary - self.epsilon):
                    dpressure = 101325*5.2561*(-0.0065/288.16)*\
                        (1-0.0065*alt[index]/288.16)**4.2561
                    drho[index] += dpressure * dalt[index] /\
                        (288*temp[index])*1e3
                elif alt[index] >= (alt_boundary + self.epsilon):
                    dpressure = (22632*(-9.81/(288*216.65))*
                                 np.exp(9.81*11000/(288*216.65))*
                                 np.exp(-9.81*alt[index]/
                                            (288*216.65)))
                    drho[index] += dpressure * dalt[index] /\
                        (288*temp[index])*1e3
                else:
                    h_star = alt[index]
                    dpressure = 3*a*h_star**2 + 2*b*h_star + c
                    drho[index] += dpressure * dalt[index] /\
                        (288*temp[index])*1e3
        if 'temp' in arg:
            dtemp = arg['temp']
            for index in xrange(n_elem):
                if alt[index] <= (alt_boundary - self.epsilon):
                    pressure = 101325*(1-0.0065*alt[index]/
                                       288.16)**5.2561
                    drho[index] += -pressure / (288*temp[index]**2) *\
                        dtemp[index] * 1e2
                elif alt[index] >= (alt_boundary + self.epsilon):
                    pressure = 22632*np.exp(-9.81*(alt[index]-
                                                      alt_boundary)/
                                                (288*216.65))
                    drho[index] += -pressure / (288*temp[index]**2) *\
                        dtemp[index] * 1e2
                else:
                    h_star = alt[index]
                    pressure = (a*h_star**3 + b*h_star**2 + c*h_star +
                                d)
                    drho[index] += -pressure / (288*temp[index]**2) *\
                        dtemp[index] * 1e2

    def apply_derivT(self, arg, result):
        """ Compute density derivative wrt altitude and temperature.
        Adjoint mode.
        """

        alt = self.h * 1e3
        n_elem = len(alt)
        temp = self.temp * 1e2
        alt_boundary = 11000

        a = self.coefs[0]
        b = self.coefs[1]
        c = self.coefs[2]
        d = self.coefs[3]

        drho = arg['rho']

        if 'h' in result:
            dalt = result['h']
            for index in xrange(n_elem):
                if alt[index] <= (alt_boundary - self.epsilon):
                    dpressure = 101325*5.2561*(-0.0065/288.16)*\
                        (1-0.0065*alt[index]/288.16)**4.2561
                    dalt[index] += dpressure * drho[index] /\
                        (288*temp[index])*1e3
                elif alt[index] >= (alt_boundary + self.epsilon):
                    dpressure = (22632*(-9.81/(288*216.65))*
                                 np.exp(9.81*11000/(288*216.65))*
                                 np.exp(-9.81*alt[index]/
                                            (288*216.65)))
                    dalt[index] += dpressure * drho[index] /\
                        (288*temp[index])*1e3
                else:
                    h_star = alt[index]
                    dpressure = 3*a*h_star**2 + 2*b*h_star + c
                    dalt[index] += dpressure * drho[index] /\
                        (288*temp[index])*1e3
        if 'temp' in result:
            dtemp = result['temp']
            for index in xrange(n_elem):
                if alt[index] <= (alt_boundary - self.epsilon):
                    pressure = 101325*(1-0.0065*alt[index]/
                                       288.16)**5.2561
                    dtemp[index] += -pressure / (288*temp[index]**2) *\
                        drho[index] * 1e2
                elif alt[index] >= (alt_boundary + self.epsilon):
                    pressure = 22632*np.exp(-9.81*(alt[index]-
                                                      alt_boundary)/
                                                (288*216.65))
                    dtemp[index] += -pressure / (288*temp[index]**2) *\
                        drho[index] * 1e2
                else:
                    h_star = alt[index]
                    pressure = (a*h_star**3 + b*h_star**2 + c*h_star +
                                d)
                    dtemp[index] += -pressure / (288*temp[index]**2) *\
                        drho[index] * 1e2


class SysRhoOld(Component):
    """ Density model using the linear temperature std atm model """

    def __init__(self, num_elem=10):
        super(SysRhoOld, self).__init__()

        # Inputs
        self.add('temp', Array(np.zeros((num_elem+1, )), iotype='in', low=0.001,
                            desc = 'Temperature'))

        # Outputs
        self.add('rho', Array(np.zeros((num_elem+1, )), iotype='out', low=0.001,
                            desc = 'Density'))

    def execute(self):
        """ Density model extracted from the standard atmosphere. Only
        dependence on temperature, with indirect dependence on altitude.
        Temperature model extracted from linear portion of the standard
        atmosphere.
        """

        temp = self.temp * 1e2
        self.rho = 1.225*(temp/288.16)**(-((9.81/((-6.5e-3)*287))+1))

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['temp']
        output_keys = ['rho']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute density derivative wrt temperature.
        Forward mode.
        """

        temp = self.temp * 1e2

        drho_dtemp = 1.225*(temp/288.16)**(-((9.81/((-6.5e-3)*287))+2)) * \
                     (-9.81/((-6.5e-3)*287)-1)*(1/288.16)

        result['rho'] += (drho_dtemp * arg['temp']) * 1e2

    def apply_derivT(self, arg, result):
        """ Compute density derivative wrt temperature.
        Adjoint mode.
        """

        temp = self.temp * 1e2

        drho_dtemp = 1.225*(temp/288.16)**(-((9.81/((-6.5e-3)*287))+2)) * \
                     (-9.81/((-6.5e-3)*287)-1)*(1/288.16)

        result['temp']  += drho_dtemp * arg['rho'] * 1e2

class SysSpeed(Component):
    """ Compute airspeed using specified Mach number. """

    def __init__(self, num_elem=10):
        super(SysSpeed, self).__init__()

        # Inputs
        self.add('temp', Array(np.zeros((num_elem+1, )), iotype='in', low=0.001,
                            desc = 'Temperature'))
        self.add('M', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Mach Number'))
        self.add('v_spline', Array(np.zeros((num_elem+1, )), iotype='in',
                                   desc = 'Speed'))
        self.add('v_specified', Bool(False, iotype='in',
                 desc='Set to True to disable calculation and use v_spline '
                 'instead.'))

        # Outputs
        self.add('v', Array(np.zeros((num_elem+1, )), iotype='out',
                            desc = 'Speed'))

    def execute(self):
        """ Airspeed is computed by first calculating the speed of sound
        given the temperature, and then multiplying by the Mach number.
        """

        temp = self.temp * 1e2
        mach = self.M
        speed_spline = self.v_spline

        gamma = 1.4
        gas_c = 287

        if self.v_specified:
            self.v = speed_spline
        else:
            self.v = mach * np.sqrt(gamma*gas_c*temp) / 1e2

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['temp', 'M', 'v_spline']
        output_keys = ['v']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute speed derivatives wrt temperature and Mach number.
        Forward mode
        """

        temp = self.temp * 1e2
        mach = self.M

        gamma = 1.4
        gas_c = 287

        ds_dM = np.sqrt(gamma*gas_c*temp)
        ds_dT = 0.5 * mach * gamma * gas_c / np.sqrt(gamma*gas_c*temp)

        if self.v_specified:
            if 'v_spline' in arg:
                result['v'] += arg['v_spline']
        else:
            if 'temp' in arg:
                result['v'] += ds_dT * arg['temp']
            if 'M' in arg:
                result['v'] += ds_dM * arg['M'] / 1e2

    def apply_derivT(self, arg, result):
        """ Compute speed derivatives wrt temperature and Mach number.
        Adjoint mode
        """

        temp = self.temp * 1e2
        mach = self.M

        gamma = 1.4
        gas_c = 287

        ds_dM = np.sqrt(gamma*gas_c*temp)
        ds_dT = 0.5 * mach * gamma * gas_c / np.sqrt(gamma*gas_c*temp)
        dspeed = arg['v']

        if self.v_specified:
            if 'v_spline' in result:
                result['v_spline'] += dspeed
        else:
            if 'temp' in result:
                result['temp']  += ds_dT * dspeed
            if 'M' in result:
                result['M']  += ds_dM * dspeed / 1e2
