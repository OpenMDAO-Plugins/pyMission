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
        self.add('fuelburn', Float(0.0, iotype='out',
                               desc = 'Objective fuel burn (initial fuel carried)'))

    def execute(self):
        """ Set objective fuel weight to initial fuel carried (required for
        mission.
        """

        self.fuelburn = self.fuel_w[0]

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['fuel_w']
        output_keys = ['fuelburn']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute objective derivatives (equal to initial fuel weight
        derivative.
        Forward mode.
        """

        result['fuelburn'] += arg['fuel_w'][0]

    def apply_derivT(self, arg, result):
        """ Compute objective derivatives (equal to initial fuel weight
        derivative.
        Adjoint mode.
        """

        result['fuel_w'][0] += arg['fuelburn']


class SysBlockTime(Component):
    """ Used to compute block time of a particular flight """

    def __init__(self, num_elem=10):
        """ scaling: 1e4
        """
        super(SysBlockTime, self).__init__()

        # Inputs
        self.add('v', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'airspeed'))
        self.add('x', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'distance'))
        self.add('Gamma', Array(np.zeros((num_elem+1, )), iotype='in',
                              desc = 'flight path angle'))

        # Outputs
        self.add('time', Float(0.0, iotype='out',
                               desc = 'Initial Mach number point'))

    def execute(self):
        """ Compute the block time required by numerically integrating (using
        the mid-point rule) the velocity values. this assumes that the
        airspeed varies linearly between data points.
        """

        speed = self.v * 1e2
        dist = self.x * 1e6
        gamma = self.Gamma * 1e-1

        time_temp = ((dist[1:] - dist[0:-1]) /
                     (((speed[1:] + speed[0:-1])/2) *
                      np.cos((gamma[1:] + gamma[0:-1])/2)))
        self.time = np.sum(time_temp)/1e4

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['v', 'x', 'Gamma']
        output_keys = ['time']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute the derivatives of blocktime wrt the velocity and
        distance points
        """

        speed = self.v * 1e2
        dist = self.x * 1e6
        gamma = self.Gamma * 1e-1
        dtime = result['time']

        if 'x' in arg:
            ddist = arg['x']
            dtime[0] += np.sum((ddist[1:] - ddist[0:-1]) /
                                  ((speed[1:] + speed[0:-1])/2 *
                                   np.cos((gamma[1:] + gamma[0:-1])/2))) \
                * 1e6/1e4
        if 'v' in arg:
            dspeed = arg['v']
            dtime[0] += np.sum(-2*(dist[1:] - dist[0:-1]) *
                                  (dspeed[1:] + dspeed[0:-1]) /
                                  ((speed[1:] + speed[0:-1])**2 *
                                   np.cos((gamma[1:] + gamma[0:-1])/2))) \
                * 1e2/1e4
        if 'Gamma' in arg:
            dgamma = arg['Gamma']
            dtime[0] += np.sum((np.sin((gamma[1:] + gamma[0:-1])/2)/
                                   (np.cos((gamma[1:] +
                                               gamma[0:-1])/2))**2) *
                                  (dgamma[1:] + dgamma[0:-1]) *
                                  ((dist[1:] - dist[0:-1])/
                                   (speed[1:] + speed[0:-1]))) * 1e-1/1e4

    def apply_derivT(self, arg, result):
        """ Compute the derivatives of blocktime wrt the velocity and
        distance points. Adjoint
        """

        speed = self.v * 1e2
        dist = self.x * 1e6
        gamma = self.Gamma * 1e-1
        dtime = arg['time']

        if 'x' in result:
            ddist = result['x']
            ddist[0:-1] += (-2/((speed[0:-1] + speed[1:]) *
                                np.cos((gamma[0:-1] + gamma[1:])/2)) *
                            dtime[0]) * 1e6/1e4
            ddist[1:] += (2/((speed[0:-1] + speed[1:]) *
                             np.cos((gamma[0:-1] + gamma[1:])/2)) *
                          dtime[0]) * 1e6/1e4
        if 'v' in result:
            dspeed = result['v']
            dspeed[0:-1] -= 2*((dist[1:] - dist[0:-1]) * dtime[0] /
                               ((speed[1:] + speed[0:-1])**2 *
                                np.cos((gamma[1:] + gamma[0:-1])/2))
                               * 1e2/1e4)
            dspeed[1:] -= 2*((dist[1:] - dist[0:-1]) * dtime[0] /
                             ((speed[1:] + speed[0:-1])**2 *
                              np.cos((gamma[1:] + gamma[0:-1])/2))
                             * 1e2/1e4)
        if 'Gamma' in result:
            dgamma = result['Gamma']
            dgamma[0:-1] += (((dist[1:] - dist[0:-1]) /
                              (speed[1:] + speed[0:-1])) *
                             ((np.sin((gamma[1:] + gamma[0:-1])/2)) /
                              (np.cos((gamma[1:] + gamma[0:-1])/2))**2) *
                             dtime[0]) * 1e-1/1e4
            dgamma[1:] += (((dist[1:] - dist[0:-1]) /
                            (speed[1:] + speed[0:-1])) *
                           ((np.sin((gamma[1:] + gamma[0:-1])/2)) /
                            (np.cos((gamma[1:] + gamma[0:-1])/2))**2) *
                           dtime[0]) * 1e-1/1e4