"""
INTENDED FOR MISSION ANALYSIS USE
Atmospheric models for specific fuel consumption (SFC), temperature, and
density. All models extracted from the linear portion of the standard
atmosphere
"""

# pylint: disable=E1101
from __future__ import division
import sys

import numpy as np

from openmdao.main.api import Component
from openmdao.main.datatypes.api import Array, Float, Bool

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103

class SysSFC(Component):
    """ Linear SFC model wrt altitude """

    def __init__(self, num_elem=10):
        super(SysSFC, self).__init__()

        # Inputs
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Altitude'))
        self.add('SFCSL', Float(0.0, iotype='in',
                                desc = 'sea-level SFC value'))

        # Outputs
        self.add('SFC', Array(np.zeros((num_elem+1, )), iotype='out',
                              desc = 'Specific Fuel Consumption'))

    def execute(self):
        """ Compute SFC value using sea level SFC and altitude the model is a
        linear correction for altitude changes.
        """

        alt = self.h * 1e3
        sfcsl = self.SFCSL * 1e-6

        sfc_temp = sfcsl + (6.39e-13) * alt
        self.SFC = sfc_temp / 1e-6

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['h', 'SFCSL']
        output_keys = ['SFC']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute SFC derivatives wrt sea level SFC and altitude.
        Forward mode
        """

        dsfc_dalt = 6.39e-13

        if 'h' in arg:
            result['SFC'] += (dsfc_dalt * arg['h']) * 1e3/1e-6
        if 'SFCSL' in arg:
            result['SFC'] += arg['SFCSL']

    def apply_derivT(self, arg, result):
        """ Compute SFC derivatives wrt sea level SFC and altitude.
        Adjoint mode
        """

        dsfc_dalt = 6.39e-13
        dsfc = arg['SFC']

        if 'h' in result:
            result['h'] += dsfc_dalt * dsfc * 1e3/1e-6
        if 'SFCSL' in result:
            result['SFCSL'] += np.sum(dsfc)


class SysTemp(Component):
    """ Linear temperature model using the standard atmosphere """

    def __init__(self, num_elem=10):
        super(SysTemp, self).__init__()

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

        result['h'] = dtemp_dalt * arg['temp'] * 1e3/1e2


class SysRho(Component):
    """ Density model using the linear temperature std atm model """

    def __init__(self, num_elem=10):
        super(SysRho, self).__init__()

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
