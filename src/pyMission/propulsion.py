"""
INTENDED FOR MISSION ANALYSIS USE
provides propulsion models for the use of mission analysis.

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
from openmdao.main.datatypes.api import Array, Float

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103

class SysSFC(Component):
    """ Linear SFC model wrt altitude """

    def __init__(self, num_elem=10, SFCSL=0):
        super(SysSFC, self).__init__()

        self.SFCSL = SFCSL

        # Inputs
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Altitude'))

        # Outputs
        self.add('SFC', Array(np.zeros((num_elem+1, )), iotype='out',
                              desc = 'Specific Fuel Consumption'))

    def execute(self):
        """ Compute SFC value using sea level SFC and altitude the model is a
        linear correction for altitude changes.
        """

        alt = self.h * 1e3
        sfcsl = self.SFCSL * 1e-6

        sfc_temp = sfcsl + (6.39e-13*9.81) * alt
        self.SFC = sfc_temp / 1e-6

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['h']
        output_keys = ['SFC']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute SFC derivatives wrt sea level SFC and altitude.
        Forward mode
        """

        dsfc_dalt = 6.39e-13*9.81

        if 'h' in arg:
            result['SFC'] += (dsfc_dalt * arg['h']) * 1e3/1e-6
        if 'SFCSL' in arg:
            result['SFC'] += arg['SFCSL']

    def apply_derivT(self, arg, result):
        """ Compute SFC derivatives wrt sea level SFC and altitude.
        Adjoint mode
        """

        dsfc_dalt = 6.39e-13*9.81
        dsfc = arg['SFC']

        if 'h' in result:
            result['h'] += dsfc_dalt * dsfc * 1e3/1e-6
        if 'SFCSL' in result:
            result['SFCSL'] += np.sum(dsfc)


class SysTau(Component):
    """ Throttle setting determined primarily by thrust coefficient A simple
    linear relationship using the sea-level max thrust and a linear
    dependence on altitude is used
    """

    def __init__(self, num_elem=10, S=0, thrust_sl=0):
        super(SysTau, self).__init__()

        self.S = S
        self.thrust_sl = thrust_sl

        # Inputs
        self.add('CT_tar', Array(np.zeros((num_elem+1, )), iotype='in',
                                 desc = 'Thrust Coefficient'))
        self.add('rho', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Density'))
        self.add('v', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Speed'))
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Altitude'))

        # Outputs
        self.add('tau', Array(np.zeros((num_elem+1, )), iotype='out',
                                 desc = 'Throttle setting'))

        #self.thrust_scale = 0.072
        self.thrust_scale = 72

    def execute(self):
        """ Compute throttle setting primarily using thrust coefficient """

        thrust_c = self.CT_tar * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        alt = self.h * 1e3
        thrust_sl = self.thrust_sl * 1e6
        wing_area = self.S * 1e2

        cThrust = thrust_sl - self.thrust_scale * alt
        Thrust = 0.5*rho*speed**2*wing_area*thrust_c
        self.tau = Thrust / cThrust

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['CT_tar', 'rho', 'v', 'h']
        output_keys = ['tau']
        return input_keys, output_keys

    def provideJ(self):
        """ Pre-compute the throttle derivatives wrt density, velocity wing
        area, thrust coefficient, sea level thrust, and altitude.
        """

        thrust_c = self.CT_tar * 1e-1
        rho = self.rho
        speed = self.v * 1e2
        alt = self.h * 1e3
        thrust_sl = self.thrust_sl * 1e6
        wing_area = self.S * 1e2

        fact = 1.0/(thrust_sl - self.thrust_scale*alt)

        self.dt_drho = (0.5*speed**2*wing_area*thrust_c) * fact
        self.dt_dspeed = (rho*speed*wing_area*thrust_c) * fact
        self.dt_dS = (0.5*rho*speed**2*thrust_c) * fact
        self.dt_dthrust_c = (0.5*rho*speed**2*wing_area) * fact
        self.dt_dthrust_sl = -(0.5*rho*speed**2*wing_area*thrust_c) * fact**2
        self.dt_dalt = self.thrust_scale * (0.5*rho*speed**2*wing_area*thrust_c) * fact**2

    def apply_deriv(self, arg, result):
        """ Assign throttle directional derivatives.
        Forward mode.
        """

        dt_dthrust_sl = self.dt_dthrust_sl
        dt_dalt = self.dt_dalt
        dt_dthrust_c = self.dt_dthrust_c
        dt_drho = self.dt_drho
        dt_dspeed = self.dt_dspeed
        dt_dS = self.dt_dS

        dtau = result['tau']

        if 'S' in arg:
            dtau += (dt_dS * arg['S']) * 1e2
        if 'thrust_sl' in arg:
            dtau += (dt_dthrust_sl * arg['thrust_sl']) * 1e6
        if 'h' in arg:
            dtau += (dt_dalt * arg['h']) * 1e3
        if 'CT_tar' in arg:
            dtau += (dt_dthrust_c * arg['CT_tar']) * 1e-1
        if 'rho' in arg:
            dtau += dt_drho * arg['rho']
        if 'v' in arg:
            dtau += (dt_dspeed * arg['v']) * 1e2

    def apply_derivT(self, arg, result):
        """ Assign throttle directional derivatives.
        Adjoint mode.
        """

        dt_dthrust_sl = self.dt_dthrust_sl
        dt_dalt = self.dt_dalt
        dt_dthrust_c = self.dt_dthrust_c
        dt_drho = self.dt_drho
        dt_dspeed = self.dt_dspeed
        dt_dS = self.dt_dS

        dtau = arg['tau']

        if 'S' in result:
            result['S'] += np.sum(dt_dS * dtau) * 1e2
        if 'thrust_sl' in result:
            result['thrust_sl'] += np.sum(dt_dthrust_sl * dtau) * 1e6
        if 'h' in result:
            result['h'] += dt_dalt * dtau * 1e3
        if 'CT_tar' in result:
            result['CT_tar'] += dt_dthrust_c * dtau * 1e-1
        if 'rho' in result:
            result['rho'] += dt_drho * dtau
        if 'v' in result:
            result['v'] += dt_dspeed * dtau * 1e2


# TODO: The following surrogate is never used, and is seemingly incomplete..

class SysTauSurrogate(Component):
    """ Compute the throttle setting from target CT by using existing engine
    data.
    """

    def __init__(self, num_elem=10):
        super(SysTauSurrogate, self).__init__()

        # Inputs
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Altitude'))
        self.add('temp', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Temperature'))
        self.add('v', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Speed'))
        self.add('CT_tar', Array(np.zeros((num_elem+1, )), iotype='in',
                                 desc = 'Thrust Coefficient'))
        self.add('rho', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Density'))
        self.add('S', Float(0.0, iotype='in', desc = 'Wing Area'))

        # Outputs
        self.add('tau', Array(np.zeros((num_elem+1, )), iotype='out',
                                 desc = 'Throttle setting'))

        self.build_surrogate('UHB.outputFLOPS')

    def build_surrogate(self, file_name):
        """ builds the surrogate model from the data stored in the file name
            given in the input arguments
        """

        data_file = open(file_name, 'r')

        for i, l in enumerate(data_file):
            pass

        file_len = i+1
        mach = np.zeros(file_len)
        altitude = np.zeros(file_len)
        power_code = np.zeros(file_len)
        thrust = np.zeros(file_len)
        drag = np.zeros(file_len)
        TSFC = np.zeros(file_len)
        i = 0

        data_file = open(file_name, 'r')

        for line in data_file:
            [mach[i], altitude[i], power_code[i], thrust[i], drag[i], fuel_burn,
             TSFC[i], Nox, area] = line.split()
            i += 1

        mach = [float(i) for i in mach]
        altitude = [float(i) for i in altitude]
        power_code = [float(i) for i in power_code]
        thrust = [float(i) for i in thrust]
        drag = [float(i) for i in drag]
        TSFC = [float(i) for i in TSFC]


