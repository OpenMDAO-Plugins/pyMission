"""
INTENDED FOR MISSION ANALYSIS USE
This file contains the functional systems used for the optimization
problem. These include objective and constraint functions defined for
the trajectory optimization case
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


class SysHi(Component):
    """ Initial altitude point used for constraints """

    def __init__(self, num_elem=10):
        super(SysHi, self).__init__()

        # Inputs
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Altitude points'))

        # Outputs
        self.add('h_i', Float(0.0, iotype='out',
                               desc = 'Initial altitude point'))

    def execute(self):
        """ Assign system to the initial altitude point """

        self.h_i = self.h[0]

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['h']
        output_keys = ['h_i']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Derivative of this is same as intial altitude point.
        Forward mode.
        """

        result['h_i'] += arg['h'][0]

    def apply_derivT(self, arg, result):
        """ Derivative of this is same as intial altitude point.
        Adjoint mode.
        """

        result['h'][0] += arg['h_i']


class SysHf(Component):
    """ Final altitude point used for constraints. """

    def __init__(self, num_elem=10):
        super(SysHf, self).__init__()

        # Inputs
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Altitude points'))

        # Outputs
        self.add('h_f', Float(0.0, iotype='out',
                               desc = 'Final altitude point'))

    def execute(self):
        """ Assign system to the final altitude point. """

        self.h_f = self.h[0]

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['h']
        output_keys = ['h_f']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ derivative of this is same as final altitude point """

        result['h_f'] += arg['h'][0]

    def apply_derivT(self, arg, result):
        """ derivative of this is same as final altitude point """

        result['h'][0] += arg['h_f']


class SysTmin(Component):
    """ KS constraint function for minimum throttle. """

    def __init__(self, num_elem=10):
        super(SysTmin, self).__init__()

        # Inputs
        self.add('tau', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Throttle setting'))

        # Outputs
        self.add('Tmin', Float(0.0, iotype='out',
                               desc = 'Minimum Thrust Constraint'))

        self.min = 0.01
        self.rho = 100

    def execute(self):
        """ Compute the KS function of minimum throttle """

        tau = self.tau

        fmax = np.max(self.min - tau)
        self.Tmin = fmax + 1.0/self.rho * \
            np.log(np.sum(np.exp(self.rho*(self.min - tau - fmax))))

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['tau']
        output_keys = ['Tmin']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute min throttle KS function derivatives wrt throttle.
        Forward mode.
        """
        tau = self.tau

        ind = np.argmax(self.min - tau)
        fmax = self.min - tau[ind]
        dfmax_dtau = np.zeros(tau.shape[0])
        dfmax_dtau[ind] = -1.0

        deriv = dfmax_dtau + 1/self.rho * \
            1/np.sum(np.exp(self.rho*(self.min - tau - fmax))) * \
            np.exp(self.rho*(self.min - tau - fmax)) * (-self.rho)
        deriv[ind] -= 1/self.rho * \
            (-self.rho)
        #    1/np.sum(np.exp(self.rho*(self.min - tau - fmax))) * \
        #    np.sum(np.exp(self.rho*(self.min - tau - fmax))) * (-self.rho)

        result['Tmin'] += np.sum(deriv * arg['tau'])

    def apply_derivT(self, arg, result):
        """ Compute min throttle KS function derivatives wrt throttle.
        Adjoint mode.
        """
        tau = self.tau

        ind = np.argmax(self.min - tau)
        fmax = self.min - tau[ind]
        dfmax_dtau = np.zeros(tau.shape[0])
        dfmax_dtau[ind] = -1.0

        deriv = dfmax_dtau + 1/self.rho * \
            1/np.sum(np.exp(self.rho*(self.min - tau - fmax))) * \
            np.exp(self.rho*(self.min - tau - fmax)) * (-self.rho)
        deriv[ind] -= 1/self.rho * \
            (-self.rho)
        #    1/np.sum(np.exp(self.rho*(self.min - tau - fmax))) * \
        #    np.sum(np.exp(self.rho*(self.min - tau - fmax))) * (-self.rho)

        result['tau'] += deriv * arg['Tmin'][0]


class SysTmax(Component):
    """ KS constraint function for maximum throttle """

    def __init__(self, num_elem=10):
        super(SysTmax, self).__init__()

        # Inputs
        self.add('tau', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Throttle setting'))

        # Outputs
        self.add('Tmax', Float(0.0, iotype='out',
                               desc = 'Maximum Thrust Constraint'))

        self.max = 1.0
        self.rho = 100

    def execute(self):
        """ Compute KS function for max throttle setting """

        tau = self.tau

        fmax = np.max(tau - self.max)
        self.Tmax = fmax + 1/self.rho * \
            np.log(np.sum(np.exp(self.rho*(tau - self.max - fmax))))

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['tau']
        output_keys = ['Tmax']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute min throttle KS function derivatives wrt throttle.
        Forward mode.
        """
        tau = self.tau

        ind = np.argmax(tau - self.max)
        fmax = tau[ind] - self.max
        dfmax_dtau = np.zeros(tau.shape[0])
        dfmax_dtau[ind] = 1.0

        deriv = dfmax_dtau + 1/self.rho * \
            1/np.sum(np.exp(self.rho*(tau - self.max - fmax))) * \
            np.exp(self.rho*(tau - self.max - fmax)) * (self.rho)
        deriv[ind] -= 1/self.rho * \
            1/np.sum(np.exp(self.rho*(tau - self.max - fmax))) * \
            np.sum(np.exp(self.rho*(tau - self.max - fmax))) * (self.rho)

        result['Tmax'] += np.sum(deriv * arg['tau'])

    def apply_derivT(self, arg, result):
        """ Compute min throttle KS function derivatives wrt throttle.
        Adjoint mode.
        """
        tau = self.tau

        ind = np.argmax(tau - self.max)
        fmax = tau[ind] - self.max
        dfmax_dtau = np.zeros(tau.shape[0])
        dfmax_dtau[ind] = 1.0

        deriv = dfmax_dtau + 1/self.rho * \
            1/np.sum(np.exp(self.rho*(tau - self.max - fmax))) * \
            np.exp(self.rho*(tau - self.max - fmax)) * (self.rho)
        deriv[ind] -= 1/self.rho * \
            1/np.sum(np.exp(self.rho*(tau - self.max - fmax))) * \
            np.sum(np.exp(self.rho*(tau - self.max - fmax))) * (self.rho)

        result['tau'] += deriv * arg['Tmax'][0]


class SysSlopeMin(Component):
    """ KS-constraint used to limit min slope to prevent unrealistic
    trajectories stalling optimization.
    """

    def __init__(self, num_elem=10):
        super(SysSlopeMin, self).__init__()

        # Inputs
        self.add('Gamma', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Flight path angle'))

        # Outputs
        self.add('gamma_min', Float(0.0, iotype='out',
                               desc = 'Flight path angle Constraint'))

        # FIX HARD CODED MIN SLOPE!!!
        self.min = np.tan(-20.0*(np.pi/180.0))
        self.rho = 30

    def execute(self):
        """ Compute the KS function of minimum slope """

        gamma = self.Gamma * 1e-1

        fmax = np.max(self.min - gamma)
        self.gamma_min = (fmax + 1/self.rho *\
                          np.log(np.sum(np.exp(self.rho*(self.min-gamma-fmax)))))\
                          *1e-6

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['Gamma']
        output_keys = ['gamma_min']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute min slope KS function derivatives wrt flight path angle.
        Forward mode.
        """

        gamma = self.Gamma * 1e-1

        ind = np.argmax(self.min-gamma)
        fmax = self.min - gamma[ind]
        dfmax_dgamma = np.zeros(gamma.shape[0])
        dfmax_dgamma[ind] = -1.0

        deriv = dfmax_dgamma + 1/self.rho *\
            1/np.sum(np.exp(self.rho*(self.min-gamma-fmax))) *\
            np.exp(self.rho*(self.min-gamma-fmax))*(-self.rho)
        deriv[ind] -= 1/self.rho *\
            1/np.sum(np.exp(self.rho*(self.min-gamma-fmax))) *\
            np.sum(np.exp(self.rho*(self.min-gamma-fmax)))*(-self.rho)

        result['gamma_min'] += np.sum(deriv * arg['Gamma'][:]) * 1e-6 * 1e-1

    def apply_derivT(self, arg, result):
        """ Compute min slope KS function derivatives wrt flight path angle.
        Adjoint mode.
        """

        gamma = self.Gamma * 1e-1

        ind = np.argmax(self.min-gamma)
        fmax = self.min - gamma[ind]
        dfmax_dgamma = np.zeros(gamma.shape[0])
        dfmax_dgamma[ind] = -1.0

        deriv = dfmax_dgamma + 1/self.rho *\
            1/np.sum(np.exp(self.rho*(self.min-gamma-fmax))) *\
            np.exp(self.rho*(self.min-gamma-fmax))*(-self.rho)
        deriv[ind] -= 1/self.rho *\
            1/np.sum(np.exp(self.rho*(self.min-gamma-fmax))) *\
            np.sum(np.exp(self.rho*(self.min-gamma-fmax)))*(-self.rho)

        result['Gamma'] += deriv * arg['gamma_min'] * 1e-6 * 1e-1

class SysSlopeMax(Component):
    """ KS-constraint used to limit max slope to prevent unrealistic
    trajectories stalling optimization.
    """

    def __init__(self, num_elem=10):
        super(SysSlopeMax, self).__init__()

        # Inputs
        self.add('Gamma', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'Flight path angle'))

        # Outputs
        self.add('gamma_max', Float(0.0, iotype='out',
                               desc = 'Flight path angle Constraint'))

        # FIX HARDCODING OF MAX GAMMA!!!!!!
        self.max = np.tan(20.0*(np.pi/180.0))
        self.rho = 30

    def execute(self):
        """ Compute KS function for max slope. """

        gamma = self.Gamma * 1e-1

        fmax = np.max(gamma - self.max)
        self.gamma_max = (fmax + 1/self.rho * \
                          np.log(np.sum(np.exp(self.rho*(gamma-self.max-fmax)))))\
                         * 1e-6

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['Gamma']
        output_keys = ['gamma_max']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute max slope KS function derivatives wrt flight path angle.
        Forward mode.
        """

        gamma = self.Gamma * 1e-1

        ind = np.argmax(gamma - self.max)
        fmax = gamma[ind] - self.max
        dfmax_dgamma = np.zeros(gamma.shape[0])
        dfmax_dgamma[ind] = 1.0

        deriv = dfmax_dgamma + 1/self.rho *\
            1/np.sum(np.exp(self.rho*(gamma-self.max-fmax))) *\
            np.exp(self.rho*(gamma-self.max-fmax))*self.rho
        deriv[ind] -= 1/self.rho *\
            1/np.sum(np.exp(self.rho*(gamma-self.max-fmax))) *\
            np.sum(np.exp(self.rho*(gamma-self.max-fmax)))*self.rho

        result['gamma_max'] += np.sum(deriv * arg['Gamma']) * 1e-6 * 1e-1

    def apply_derivT(self, arg, result):
        """ Compute max slope KS function derivatives wrt flight path angle.
        Adjoint mode.
        """

        gamma = self.Gamma * 1e-1

        ind = np.argmax(gamma - self.max)
        fmax = gamma[ind] - self.max
        dfmax_dgamma = np.zeros(gamma.shape[0])
        dfmax_dgamma[ind] = 1.0

        deriv = dfmax_dgamma + 1/self.rho *\
            1/np.sum(np.exp(self.rho*(gamma-self.max-fmax))) *\
            np.exp(self.rho*(gamma-self.max-fmax))*self.rho
        deriv[ind] -= 1/self.rho *\
            1/np.sum(np.exp(self.rho*(gamma-self.max-fmax))) *\
            np.sum(np.exp(self.rho*(gamma-self.max-fmax)))*self.rho

        result['Gamma'] += deriv * arg['gamma_max'] * 1e-6 * 1e-1


class SysFuelObj(Component):
    """ Objective function used for the optimization problem """

    def __init__(self, num_elem=10):
        super(SysFuelObj, self).__init__()

        # Inputs
        self.add('fuel_w', Array(np.zeros((num_elem+1, )), iotype='in',
                                 desc = 'Fuel Weight'))

        # Outputs
        self.add('wf_obj', Float(0.0, iotype='out',
                               desc = 'Objective fual burn (initial fuel carried)'))

    def execute(self):
        """ Set objective fuel weight to initial fuel carried (required for
        mission.
        """

        self.wf_obj = self.fuel_w[0]

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['fuel_w']
        output_keys = ['wf_obj']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute objective derivatives (equal to initial fuel weight
        derivative.
        Forward mode.
        """

        result['wf_obj'] += arg['fuel_w'][0]

    def apply_derivT(self, arg, result):
        """ Compute objective derivatives (equal to initial fuel weight
        derivative.
        Adjoint mode.
        """

        result['fuel_w'][0] += arg['wf_obj']
