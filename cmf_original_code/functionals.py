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
from framework import *
import numpy


class SysHi(ExplicitSystem):
    """ initial altitude point used for constraints """

    def _declare(self):
        """ owned variable: h_i (initial altitude point)
            dependencies: h (altitude points)
        """

        self._declare_variable('h_i')
        self._declare_argument('h', indices=[0])

    def apply_G(self):
        """ assign system to the initial altitude point """

        alt_i = self.vec['u']('h_i')
        alt = self.vec['p']('h')

        alt_i[0] = alt[0]

    def apply_dGdp(self, args):
        """ derivative of this is same as intial altitude point """

        dalt_i = self.vec['dg']('h_i')
        dalt = self.vec['dp']('h')

        if self.mode == 'fwd':
            dalt_i[0] = 0.0
            if self.get_id('h') in args:
                dalt_i[0] += dalt[0]
        if self.mode == 'rev':
            dalt[0] = 0.0
            if self.get_id('h') in args:
                dalt[0] += dalt_i[0]

class SysHf(ExplicitSystem):
    """ final altitude point used for constraints """

    def _declare(self):
        """ owned variable: h_f (final altitude point)
            dependencies: h (altitude points)
        """

        num_elem = self.kwargs['num_elem']
        self._declare_variable('h_f')
        self._declare_argument('h', indices=[num_elem])

    def apply_G(self):
        """ assign system to the final altitude point """

        alt_f = self.vec['u']('h_f')
        alt = self.vec['p']('h')

        alt_f[0] = alt[0]

    def apply_dGdp(self, args):
        """ derivative of this is same as final altitude point """

        dalt_f = self.vec['dg']('h_f')
        dalt = self.vec['dp']('h')

        if self.mode == 'fwd':
            dalt_f[0] = 0.0
            if self.get_id('h') in args:
                dalt_f[0] += dalt[0]
        if self.mode == 'rev':
            dalt[0] = 0.0
            if self.get_id('h') in args:
                dalt[0] += dalt_f[0]

class SysTmin(ExplicitSystem):
    """ KS constraint function for minimum throttle """

    def _declare(self):
        """ owned variable: Tmin (minimum thrust constraint)
            dependencies: tau (throttle setting)
        """

        num_elem = self.kwargs['num_elem']

        self._declare_variable('Tmin')
        self._declare_argument('tau', indices=range(0, num_elem+1))
        self.min = 0.01
        self.rho = 100

    def apply_G(self):
        """ compute the KS function of minimum throttle """

        tmin = self.vec['u']('Tmin')
        tau = self.vec['p']('tau')

        fmax = numpy.max(self.min - tau)
        tmin[0] = fmax + 1/self.rho * \
            numpy.log(numpy.sum(numpy.exp(self.rho*(self.min - tau - fmax))))

    def apply_dGdp(self, args):
        """ compute min throttle KS function derivatives wrt throttle """

        tau = self.vec['p']('tau')

        dtmin = self.vec['dg']('Tmin')
        dtau = self.vec['dp']('tau')

        ind = numpy.argmax(self.min - tau)
        fmax = self.min - tau[ind]
        dfmax_dtau = numpy.zeros(tau.shape[0])
        dfmax_dtau[ind] = -1.0

        deriv = dfmax_dtau + 1/self.rho * \
            1/numpy.sum(numpy.exp(self.rho*(self.min - tau - fmax))) * \
            numpy.exp(self.rho*(self.min - tau - fmax)) * (-self.rho)
        deriv[ind] -= 1/self.rho * \
            (-self.rho)
        #    1/numpy.sum(numpy.exp(self.rho*(self.min - tau - fmax))) * \
        #    numpy.sum(numpy.exp(self.rho*(self.min - tau - fmax))) * (-self.rho)

        if self.mode == 'fwd':
            dtmin[0] = 0.0
            if self.get_id('tau') in args:
                dtmin[0] += numpy.sum(deriv * dtau[:])
        if self.mode == 'rev':
            dtau[:] = 0.0
            if self.get_id('tau') in args:
                dtau[:] += deriv * dtmin[0]

class SysTmax(ExplicitSystem):
    """ KS constraint function for maximum throttle """

    def _declare(self):
        """ owned variable: Tmax (maximum thrust constraint)
            dependencies: tau (throttle setting)
        """

        num_elem = self.kwargs['num_elem']
        self._declare_variable('Tmax')
        self._declare_argument('tau', indices=range(0, num_elem+1))
        self.max = 1.0
        self.rho = 100

    def apply_G(self):
        """ compute KS function for max throttle setting """

        tmax = self.vec['u']('Tmax')
        tau = self.vec['p']('tau')

        fmax = numpy.max(tau - self.max)
        tmax[0] = fmax + 1/self.rho * \
            numpy.log(numpy.sum(numpy.exp(self.rho*(tau - self.max - fmax))))

    def apply_dGdp(self, args):
        """ compute max throttle KS function derivatives wrt throttle """

        tau = self.vec['p']('tau')

        dtmax = self.vec['dg']('Tmax')
        dtau = self.vec['dp']('tau')

        ind = numpy.argmax(tau - self.max)
        fmax = tau[ind] - self.max
        dfmax_dtau = numpy.zeros(tau.shape[0])
        dfmax_dtau[ind] = 1.0

        deriv = dfmax_dtau + 1/self.rho * \
            1/numpy.sum(numpy.exp(self.rho*(tau - self.max - fmax))) * \
            numpy.exp(self.rho*(tau - self.max - fmax)) * (self.rho)
        deriv[ind] -= 1/self.rho * \
            1/numpy.sum(numpy.exp(self.rho*(tau - self.max - fmax))) * \
            numpy.sum(numpy.exp(self.rho*(tau - self.max - fmax))) * (self.rho)

        if self.mode == 'fwd':
            dtmax[0] = 0.0
            if self.get_id('tau') in args:
                dtmax[0] += numpy.sum(deriv * dtau[:])
        if self.mode == 'rev':
            dtau[:] = 0.0
            if self.get_id('tau') in args:
                dtau[:] += deriv * dtmax[0]

class SysSlopeMin(ExplicitSystem):
    """ KS-constraint used to limit min slope to prevent
        unrealistic trajectories stalling optimization
    """

    def _declare(self):
        """ owned variable: gamma_min (flight path angle constraint)
            dependencies: gamma (flight path angle)
        """

        num_elem = self.kwargs['num_elem']

        self._declare_variable('gamma_min')
        self._declare_argument('gamma', indices=range(num_elem+1))
        # FIX HARD CODED MIN SLOPE!!!
        self.min = numpy.tan(-20.0*(numpy.pi/180.0))
        self.rho = 30

    def apply_G(self):
        """ compute the KS function of minimum slope """

        gmin = self.vec['u']('gamma_min')
        gamma = self.vec['p']('gamma') * 1e-1

        fmax = numpy.max(self.min - gamma)
        gmin[0] = (fmax + 1/self.rho *\
                       numpy.log(numpy.sum(numpy.exp(self.rho*(self.min-gamma-fmax)))))\
                       *1e-6

    def apply_dGdp(self, args):
        """ compute min slope KS function derivatives wrt flight
            path angle
        """

        gamma = self.vec['p']('gamma')*1e-1

        dgmin = self.vec['dg']('gamma_min')
        dgamma = self.vec['dp']('gamma')

        ind = numpy.argmax(self.min-gamma)
        fmax = self.min - gamma[ind]
        dfmax_dgamma = numpy.zeros(gamma.shape[0])
        dfmax_dgamma[ind] = -1.0

        deriv = dfmax_dgamma + 1/self.rho *\
            1/numpy.sum(numpy.exp(self.rho*(self.min-gamma-fmax))) *\
            numpy.exp(self.rho*(self.min-gamma-fmax))*(-self.rho)
        deriv[ind] -= 1/self.rho *\
            1/numpy.sum(numpy.exp(self.rho*(self.min-gamma-fmax))) *\
            numpy.sum(numpy.exp(self.rho*(self.min-gamma-fmax)))*(-self.rho)

        if self.mode == 'fwd':
            dgmin[0] = 0.0
            if self.get_id('gamma') in args:
                dgmin[0] += numpy.sum(deriv * dgamma[:]) * 1e-6 * 1e-1
        if self.mode == 'rev':
            dgamma[:] = 0.0
            if self.get_id('gamma') in args:
                dgamma[:] += deriv * dgmin[0] * 1e-6 * 1e-1

class SysSlopeMax(ExplicitSystem):
    """ KS-constraint used to limit max slope to prevent
        unrealistic trajectories stalling optimization
    """

    def _declare(self):
        """ owned variable: gamma_max (flight path angle constraint)
            dependencies: gamma (flight path angle)
        """

        num_elem = self.kwargs['num_elem']
        self._declare_variable('gamma_max')
        self._declare_argument('gamma', indices=range(num_elem+1))
        # FIX HARDCODING OF MAX GAMMA!!!!!!
        self.max = numpy.tan(20.0*(numpy.pi/180.0))
        self.rho = 30

    def apply_G(self):
        """ compute KS function for max slope """

        gmax = self.vec['u']('gamma_max')
        gamma = self.vec['p']('gamma') * 1e-1

        fmax = numpy.max(gamma - self.max)
        gmax[0] = (fmax + 1/self.rho * \
                       numpy.log(numpy.sum(numpy.exp(self.rho*(gamma-self.max-fmax)))))\
                       * 1e-6

    def apply_dGdp(self, args):
        """ compute max slope KS function derivatives wrt flight
            path angle
        """

        gamma = self.vec['p']('gamma') * 1e-1

        dgmax = self.vec['dg']('gamma_max')
        dgamma = self.vec['dp']('gamma')

        ind = numpy.argmax(gamma - self.max)
        fmax = gamma[ind] - self.max
        dfmax_dgamma = numpy.zeros(gamma.shape[0])
        dfmax_dgamma[ind] = 1.0

        deriv = dfmax_dgamma + 1/self.rho *\
            1/numpy.sum(numpy.exp(self.rho*(gamma-self.max-fmax))) *\
            numpy.exp(self.rho*(gamma-self.max-fmax))*self.rho
        deriv[ind] -= 1/self.rho *\
            1/numpy.sum(numpy.exp(self.rho*(gamma-self.max-fmax))) *\
            numpy.sum(numpy.exp(self.rho*(gamma-self.max-fmax)))*self.rho

        if self.mode == 'fwd':
            dgmax[0] = 0.0
            if self.get_id('gamma') in args:
                dgmax[0] += numpy.sum(deriv * dgamma[:]) * 1e-6 * 1e-1
        if self.mode == 'rev':
            dgamma[:] = 0.0
            if self.get_id('gamma') in args:
                dgamma[:] += deriv * dgmax[0] * 1e-6 * 1e-1

class SysFuelObj(ExplicitSystem):
    """ objective function used for the optimization problem """

    def _declare(self):
        """ owned variable: objective fual burn (initial fuel carried)
            dependencies: weight of fuel (fuel_w)
        """

        self._declare_variable('wf_obj', size=1)
        self._declare_argument('fuel_w', indices=[0])

    def apply_G(self):
        """ set objective fuel weight to initial fuel carried (required for
            mission
        """

        p = self.vec['p']
        u = self.vec['u']
        Wf = p('fuel_w')

        u('wf_obj')[0] = Wf[0]

    def apply_dGdp(self, arguments):
        """ compute objective derivatives (equal to initial fuel weight
            derivative
        """

        dp = self.vec['dp']
        dg = self.vec['dg']
        
        if self.mode == 'fwd':
            dg('wf_obj')[0] = 0.0
            if self.get_id('fuel_w') in arguments:
                dg('wf_obj')[0] += dp('fuel_w')[0]
        if self.mode == 'rev':
            dp('fuel_w')[0] = 0.0
            if self.get_id('fuel_w') in arguments:
                dp('fuel_w')[0] += dg('wf_obj')[0]

class SysVi(ExplicitSystem):
    """ initial airspeed point used for constraints """

    def _declare(self):
        """ owned variable: v_i (initial airspeed point)
            dependencies: v (airspeed points)
        """

        self._declare_variable('v_i')
        self._declare_argument('v', indices=[0])

    def apply_G(self):
        """ assign system to the initial airspeed point """

        speed_i = self.vec['u']('v_i')
        speed = self.vec['p']('v')

        speed_i[0] = speed[0]

    def apply_dGdp(self, args):
        """ derivative of this is same as initial airspeed point """

        dspeed_i = self.vec['dg']('v_i')
        dspeed = self.vec['dp']('v')

        if self.mode == 'fwd':
            dspeed_i[0] = 0.0
            if self.get_id('v') in args:
                dspeed_i[0] += dspeed[0]
        if self.mode == 'rev':
            dspeed[0] = 0.0
            if self.get_id('v') in args:
                dspeed[0] += dspeed_i[0]

class SysVf(ExplicitSystem):
    """ final airspeed point used for constraints """

    def _declare(self):
        """ owned variable: v_f (final airspeed point)
            dependencies: v (airspeed points)
        """

        num_elem = self.kwargs['num_elem']
        self._declare_variable('v_f')
        self._declare_argument('v', indices=[num_elem])

    def apply_G(self):
        """ assign system to the final airspeed point """

        speed_f = self.vec['u']('v_f')
        speed = self.vec['p']('v')

        speed_f[0] = speed[0]

    def apply_dGdp(self, args):
        """ derivative of this is same as final airspeed point """

        dspeed_f = self.vec['dg']('v_f')
        dspeed = self.vec['dp']('v')

        if self.mode == 'fwd':
            dspeed_f[0] = 0.0
            if self.get_id('v') in args:
                dspeed_f[0] += dspeed[0]
        if self.mode == 'rev':
            dspeed[0] = 0.0
            if self.get_id('v') in args:
                dspeed[0] += dspeed_f[0]

class SysBlockTime(ExplicitSystem):
    """ used to compute block time of a particular flight """

    def _declare(self):
        """ owned variable: blocktime
            dependencies: airspeed (v),
                          dist (x)
            scaling: 1e4
        """
        num_elem = self.kwargs['num_elem']
        ind = range(num_elem+1)

        self._declare_variable('time', size=1)
        self._declare_argument('v', indices=ind)
        self._declare_argument('x', indices=ind)
        self._declare_argument('gamma', indices=ind)

    def apply_G(self):
        """ compute the block time required by numerically integrating
            (using the mid-point rule) the velocity values. this assumes
            that the airspeed varies linearly between data points.
        """

        pvec = self.vec['p']
        uvec = self.vec['u']

        speed = pvec('v') * 1e2
        dist = pvec('x') * 1e6
        gamma = pvec('gamma') * 1e-1
        time = uvec('time')

        time_temp = ((dist[1:] - dist[0:-1]) /
                     (((speed[1:] + speed[0:-1])/2) *
                      numpy.cos((gamma[1:] + gamma[0:-1])/2)))
        time[0] = numpy.sum(time_temp)/1e4

    def apply_dGdp(self, args):
        """ compute the derivatives of blocktime wrt the velocity
            and distance points
        """

        pvec = self.vec['p']
        dpvec = self.vec['dp']
        dgvec = self.vec['dg']

        speed = pvec('v') * 1e2
        dist = pvec('x') * 1e6
        gamma = pvec('gamma') * 1e-1
        dspeed = dpvec('v')
        ddist = dpvec('x')
        dgamma = dpvec('gamma')
        dtime = dgvec('time')

        if self.mode == 'fwd':
            dtime[:] = 0.0
            if self.get_id('x') in args:
                dtime[0] += numpy.sum((ddist[1:] - ddist[0:-1]) /
                                      ((speed[1:] + speed[0:-1])/2 *
                                       numpy.cos((gamma[1:] + gamma[0:-1])/2))) \
                    * 1e6/1e4
            if self.get_id('v') in args:
                dtime[0] += numpy.sum(-2*(dist[1:] - dist[0:-1]) *
                                      (dspeed[1:] + dspeed[0:-1]) /
                                      ((speed[1:] + speed[0:-1])**2 *
                                       numpy.cos((gamma[1:] + gamma[0:-1])/2))) \
                    * 1e2/1e4
            if self.get_id('gamma') in args:
                dtime[0] += numpy.sum((numpy.sin((gamma[1:] + gamma[0:-1])/2)/
                                       (numpy.cos((gamma[1:] +
                                                   gamma[0:-1])/2))**2) *
                                      (dgamma[1:] + dgamma[0:-1]) *
                                      ((dist[1:] - dist[0:-1])/
                                       (speed[1:] + speed[0:-1]))) * 1e-1/1e4
        if self.mode == 'rev':
            dspeed[:] = 0.0
            ddist[:] = 0.0
            dgamma[:] = 0.0
            if self.get_id('x') in args:
                ddist[0:-1] += (-2/((speed[0:-1] + speed[1:]) *
                                    numpy.cos((gamma[0:-1] + gamma[1:])/2)) *
                                dtime[0]) * 1e6/1e4
                ddist[1:] += (2/((speed[0:-1] + speed[1:]) *
                                 numpy.cos((gamma[0:-1] + gamma[1:])/2)) *
                              dtime[0]) * 1e6/1e4
            if self.get_id('v') in args:
                dspeed[0:-1] -= 2*((dist[1:] - dist[0:-1]) * dtime[0] /
                                   ((speed[1:] + speed[0:-1])**2 *
                                    numpy.cos((gamma[1:] + gamma[0:-1])/2))
                                   * 1e2/1e4)
                dspeed[1:] -= 2*((dist[1:] - dist[0:-1]) * dtime[0] /
                                 ((speed[1:] + speed[0:-1])**2 *
                                  numpy.cos((gamma[1:] + gamma[0:-1])/2))
                                 * 1e2/1e4)
            if self.get_id('gamma') in args:
                dgamma[0:-1] += (((dist[1:] - dist[0:-1]) /
                                  (speed[1:] + speed[0:-1])) *
                                 ((numpy.sin((gamma[1:] + gamma[0:-1])/2)) /
                                  (numpy.cos((gamma[1:] + gamma[0:-1])/2))**2) *
                                 dtime[0]) * 1e-1/1e4
                dgamma[1:] += (((dist[1:] - dist[0:-1]) /
                                (speed[1:] + speed[0:-1])) *
                               ((numpy.sin((gamma[1:] + gamma[0:-1])/2)) /
                                (numpy.cos((gamma[1:] + gamma[0:-1])/2))**2) *
                               dtime[0]) * 1e-1/1e4
