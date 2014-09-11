"""
INTENDED FOR MISSION ANALYSIS USE
Atmospheric models for specific fuel consumption (SFC), temperature, and
density. All models extracted from the linear portion of the standard
atmosphere.
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

class SysTemp(ExplicitSystem):
    """ linear temperature model using standard atmosphere with smoothing
        at the temperature discontinuity
    """

    def _declare(self):
        """ owned variable: Temp (temperature)
            dependencies: h (altitude)
        """

        self.num_elem = self.kwargs['num_elem']
        num_pts = self.num_elem+1
        ind_pts = range(num_pts)
        self.epsilon = 500

        self._declare_variable('Temp', size=num_pts, lower=0.001)
        self._declare_argument('h', indices=ind_pts)

        h_lower = 11000 - self.epsilon
        h_upper = 11000 + self.epsilon
        matrix = numpy.array([[h_lower**3, h_lower**2, h_lower, 1],
                              [h_upper**3, h_upper**2, h_upper, 1],
                              [3*h_lower**2, 2*h_lower, 1, 0],
                              [3*h_upper**2, 2*h_upper, 1, 0]])
        rhs = numpy.array([288.16-(6.5e-3)*h_lower, 216.65,
                           -6.5e-3, 0])
        self.coefs = numpy.linalg.solve(matrix, rhs)

    def apply_G(self):
        """ temperature model extracted from linear portion and constant
            portion of the standard atmosphere
        """

        pvec = self.vec['p']
        uvec = self.vec['u']
        alt = pvec('h') * 1e3
        temp = uvec('Temp')
        alt_boundary = 11000

        a = self.coefs[0]
        b = self.coefs[1]
        c = self.coefs[2]
        d = self.coefs[3]

        for index in xrange(self.num_elem+1):
            if alt[index] <= (alt_boundary - self.epsilon):
                temp[index] = (288.16 - (6.5e-3) * alt[index]) / 1e2
            elif alt[index] >= (alt_boundary + self.epsilon):
                temp[index] = 216.65 / 1e2
            else:
                h_star = alt[index]
                temp[index] = (a*h_star**3 + b*h_star**2 + c*h_star + d) / 1e2

    def apply_dGdp(self, args):
        """ compute temperature derivative wrt altitude """

        dpvec = self.vec['dp']
        dgvec = self.vec['dg']
        pvec = self.vec['p']

        dalt = dpvec('h')
        dtemp = dgvec('Temp')
        alt = pvec('h') * 1e3
        alt_boundary = 11000

        a = self.coefs[0]
        b = self.coefs[1]
        c = self.coefs[2]

        if self.mode == 'fwd':
            dtemp[:] = 0.0
            if self.get_id('h') in args:
                for index in xrange(self.num_elem+1):
                    if alt[index] <= (alt_boundary - self.epsilon):
                        dtemp[index] += -6.5e-3 * dalt[index] * 1e3/1e2
                    elif alt[index] >= (alt_boundary + self.epsilon):
                        dtemp[index] += 0.0
                    else:
                        h_star = alt[index]
                        dtemp[index] += (3*a*h_star**2 + 2*b*h_star
                                         + c) * dalt[index] * 1e3/1e2
        if self.mode == 'rev':
            dalt[:] = 0.0
            if self.get_id('h') in args:
                for index in xrange(self.num_elem+1):
                    if alt[index] <= (alt_boundary - self.epsilon):
                        dalt[index] += -6.5e-3 * dtemp[index] * 1e3/1e2
                    elif alt[index] >= (alt_boundary + self.epsilon):
                        dalt[index] += 0.0
                    else:
                        h_star = alt[index]
                        dalt[index] += (3*a*h_star**2 + 2*b*h_star
                                        + c) * dtemp[index] * 1e3/1e2

class SysTempOld(ExplicitSystem):
    """ linear temperature model using the standard atmosphere """

    def _declare(self):
        """ owned variable: Temp (temperature)
            dependencies: h (altitude)
        """

        self.num_elem = self.kwargs['num_elem']
        num_pts = self.num_elem+1
        ind_pts = range(num_pts)

        self._declare_variable('Temp', size=num_pts, lower=0.001)
        self._declare_argument('h', indices=ind_pts)

    def apply_G(self):
        """ temperature model extracted from linear portion of the
            standard atmosphere
        """

        pvec = self.vec['p']
        uvec = self.vec['u']
        alt = pvec('h') * 1e3
        temp = uvec('Temp')

        temp[:] = (288.16 - (6.5e-3) * alt) / 1e2

    def apply_dGdp(self, args):
        """ compute temperature derivative wrt altitude """

        dpvec = self.vec['dp']
        dgvec = self.vec['dg']

        dalt = dpvec('h')
        dtemp = dgvec('Temp')

        dtemp_dalt = -6.5e-3

        if self.mode == 'fwd':
            dtemp[:] = 0.0
            if self.get_id('h') in args:
                dtemp[:] += (dtemp_dalt * dalt) * \
                    1e3/1e2
        if self.mode == 'rev':
            dalt[:] = 0.0
            if self.get_id('h') in args:
                dalt[:] = dtemp_dalt * dtemp * 1e3/1e2

class SysRho(ExplicitSystem):
    """ density model using standard atmosphere model with 
        troposphere, stratosphere
    """

    def _declare(self):
        """ owned variable: rho (density)
            dependencies: temp (temperature)
                          h (altitude)
        """

        self.num_elem = self.kwargs['num_elem']
        num_pts = self.num_elem+1
        ind_pts = range(num_pts)

        self._declare_variable('rho', size=num_pts, lower=0.001)
        self._declare_argument('Temp', indices=ind_pts)
        self._declare_argument('h', indices=ind_pts)

        self.epsilon = 500
        h_lower = 11000 - self.epsilon
        h_upper = 11000 + self.epsilon
        matrix = numpy.array([[h_lower**3, h_lower**2, h_lower, 1],
                              [h_upper**3, h_upper**2, h_upper, 1],
                              [3*h_lower**2, 2*h_lower, 1, 0],
                              [3*h_upper**2, 2*h_upper, 1, 0]])
        rhs = numpy.array([101325*(1-0.0065*h_lower/288.16)**5.2561,
                           22632*numpy.exp(-9.81*self.epsilon/(288*216.65)),
                           (-101325*5.2561*(0.0065/288.16)*
                             (1-0.0065*h_lower/288.15)**4.2561),
                           (22632*(-9.81/(288*216.65))*
                            numpy.exp(-9.81*self.epsilon/(288*216.65)))])
        self.coefs = numpy.linalg.solve(matrix, rhs)

    def apply_G(self):
        """ Density model extracted from the standard atmosphere.
            Depends on the temperature and the altitude. Model is
            valid for troposphere and stratosphere, and accounts for
            the linear decreasing temperature segment (troposphere),
            and the constant temperature segment (stratosphere)
        """

        pvec = self.vec['p']
        uvec = self.vec['u']
        temp = pvec('Temp') * 1e2
        alt = pvec('h') * 1e3
        rho = uvec('rho')

        alt_boundary = 11000
        a = self.coefs[0]
        b = self.coefs[1]
        c = self.coefs[2]
        d = self.coefs[3]

        for index in xrange(self.num_elem+1):
            if alt[index] <= (alt_boundary - self.epsilon):
                pressure = 101325*(1-0.0065*alt[index]/288.16)**5.2561
                rho[index] = pressure / (288 * temp[index])
            elif alt[index] >= (alt_boundary + self.epsilon):
                pressure = 22632*numpy.exp(-9.81*(alt[index]-alt_boundary)/
                                             (288*216.65))
                rho[index] = pressure / (288 * temp[index])
            else:
                h_star = alt[index]
                pressure = a*h_star**3 + b*h_star**2 + c*h_star + d
                rho[index] = pressure / (288 * temp[index])

    def apply_dGdp(self, args):
        """ compute density derivative wrt altitude and temperature """

        dpvec = self.vec['dp']
        dgvec = self.vec['dg']
        pvec = self.vec['p']

        dalt = dpvec('h')
        dtemp = dpvec('Temp')
        drho = dgvec('rho')
        alt = pvec('h') * 1e3
        temp = pvec('Temp') * 1e2
        alt_boundary = 11000

        a = self.coefs[0]
        b = self.coefs[1]
        c = self.coefs[2]
        d = self.coefs[3]

        if self.mode == 'fwd':
            drho[:] = 0.0
            if self.get_id('h') in args:
                for index in xrange(self.num_elem+1):
                    if alt[index] <= (alt_boundary - self.epsilon):
                        dpressure = 101325*5.2561*(-0.0065/288.16)*\
                            (1-0.0065*alt[index]/288.16)**4.2561
                        drho[index] += dpressure * dalt[index] /\
                            (288*temp[index])*1e3
                    elif alt[index] >= (alt_boundary + self.epsilon):
                        dpressure = (22632*(-9.81/(288*216.65))*
                                     numpy.exp(9.81*11000/(288*216.65))*
                                     numpy.exp(-9.81*alt[index]/
                                                (288*216.65)))
                        drho[index] += dpressure * dalt[index] /\
                            (288*temp[index])*1e3
                    else:
                        h_star = alt[index]
                        dpressure = 3*a*h_star**2 + 2*b*h_star + c
                        drho[index] += dpressure * dalt[index] /\
                            (288*temp[index])*1e3
            if self.get_id('Temp') in args:
                for index in xrange(self.num_elem+1):
                    if alt[index] <= (alt_boundary - self.epsilon):
                        pressure = 101325*(1-0.0065*alt[index]/
                                           288.16)**5.2561
                        drho[index] += -pressure / (288*temp[index]**2) *\
                            dtemp[index] * 1e2
                    elif alt[index] >= (alt_boundary + self.epsilon):
                        pressure = 22632*numpy.exp(-9.81*(alt[index]-
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

        if self.mode == 'rev':
            dalt[:] = 0.0
            dtemp[:] = 0.0
            if self.get_id('h') in args:
                for index in xrange(self.num_elem+1):
                    if alt[index] <= (alt_boundary - self.epsilon):
                        dpressure = 101325*5.2561*(-0.0065/288.16)*\
                            (1-0.0065*alt[index]/288.16)**4.2561
                        dalt[index] += dpressure * drho[index] /\
                            (288*temp[index])*1e3
                    elif alt[index] >= (alt_boundary + self.epsilon):
                        dpressure = (22632*(-9.81/(288*216.65))*
                                     numpy.exp(9.81*11000/(288*216.65))*
                                     numpy.exp(-9.81*alt[index]/
                                                (288*216.65)))
                        dalt[index] += dpressure * drho[index] /\
                            (288*temp[index])*1e3
                    else:
                        h_star = alt[index]
                        dpressure = 3*a*h_star**2 + 2*b*h_star + c
                        dalt[index] += dpressure * drho[index] /\
                            (288*temp[index])*1e3
            if self.get_id('Temp') in args:
                for index in xrange(self.num_elem+1):
                    if alt[index] <= (alt_boundary - self.epsilon):
                        pressure = 101325*(1-0.0065*alt[index]/
                                           288.16)**5.2561
                        dtemp[index] += -pressure / (288*temp[index]**2) *\
                            drho[index] * 1e2
                    elif alt[index] >= (alt_boundary + self.epsilon):
                        pressure = 22632*numpy.exp(-9.81*(alt[index]-
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

class SysRhoOld(ExplicitSystem):
    """ density model using the linear temperature std atm model """

    def _declare(self):
        """ owned variable: rho (density)
            dependencies: temp (temperature)
        """

        self.num_elem = self.kwargs['num_elem']
        num_pts = self.num_elem+1
        ind_pts = range(num_pts)

        self._declare_variable('rho', size=num_pts, lower=0.001)
        self._declare_argument('Temp', indices=ind_pts)

    def apply_G(self):
        """ Density model extracted from the standard atmosphere.
            Only dependence on temperature, with indirect dependence on
            altitude. Temperature model extracted from linear portion of
            the standard atmosphere
        """

        pvec = self.vec['p']
        uvec = self.vec['u']
        temp = pvec('Temp') * 1e2
        rho = uvec('rho')

        rho[:] = 1.225*(temp/288.16)**(-((9.81/((-6.5e-3)*287))+1))

    def apply_dGdp(self, args):
        """ compute density derivative wrt temperature """

        pvec = self.vec['p']
        dpvec = self.vec['dp']
        dgvec = self.vec['dg']

        temp = pvec('Temp') * 1e2

        dtemp = dpvec('Temp')
        drho = dgvec('rho')

        drho_dtemp = 1.225*(temp/288.16)**(-((9.81/((-6.5e-3)*287))+2)) * \
                     (-9.81/((-6.5e-3)*287)-1)*(1/288.16)

        if self.mode == 'fwd':
            drho[:] = 0.0
            if self.get_id('Temp') in args:
                drho[:] += (drho_dtemp * dtemp) * 1e2
        if self.mode == 'rev':
            dtemp[:] = 0.0
            if self.get_id('Temp') in args:
                dtemp[:] += drho_dtemp * drho * 1e2

class SysSpeed(ExplicitSystem):
    """ compute airspeed using specified Mach number """

    def _declare(self):
        """ owned variable: v (speed)
            dependencies: M (Mach number)
                          temp (temperature)
        """

        self.num_elem = self.kwargs['num_elem']
        self.v_specified = self.kwargs['v_specified']
        num_pts = self.num_elem+1
        ind_pts = range(num_pts)

        self._declare_variable('v', size=num_pts)
        self._declare_argument('v_spline', indices=ind_pts)
        self._declare_argument('M', indices=ind_pts)
        self._declare_argument('Temp', indices=ind_pts)

    def apply_G(self):
        """ Airspeed is computed by first calculating the speed of sound
            given the temperature, and then multiplying by the Mach number
        """

        pvec = self.vec['p']
        uvec = self.vec['u']
        temp = pvec('Temp') * 1e2
        mach = pvec('M')
        speed_spline = pvec('v_spline')
        speed = uvec('v')

        gamma = 1.4
        gas_c = 287

        if self.v_specified:
            speed[:] = speed_spline
        else:
            speed[:] = mach * numpy.sqrt(gamma*gas_c*temp) / 1e2

    def apply_dGdp(self, args):
        """ compute speed derivatives wrt temperature and Mach number """

        pvec = self.vec['p']
        dpvec = self.vec['dp']
        dgvec = self.vec['dg']
        
        temp = pvec('Temp') * 1e2
        mach = pvec('M')

        dtemp = dpvec('Temp')
        dmach = dpvec('M')
        dspeed_spline = dpvec('v_spline')
        dspeed = dgvec('v')

        gamma = 1.4
        gas_c = 287

        ds_dM = numpy.sqrt(gamma*gas_c*temp)
        ds_dT = 0.5 * mach * gamma * gas_c / numpy.sqrt(gamma*gas_c*temp)

        if self.mode == 'fwd':
            dspeed[:] = 0.0
            if self.v_specified:
                if self.get_id('v_spline') in args:
                    dspeed[:] += dspeed_spline
            else:
                if self.get_id('Temp') in args:
                    dspeed[:] += ds_dT * dtemp
                if self.get_id('M') in args:
                    dspeed[:] += ds_dM * dmach / 1e2
        
        elif self.mode == 'rev':
            dtemp[:] = 0.0
            dmach[:] = 0.0
            dspeed_spline[:] = 0.0
            if self.v_specified:
                if self.get_id('v_spline') in args:
                    dspeed_spline[:] += dspeed
            else:
                if self.get_id('Temp') in args:
                    dtemp[:] += ds_dT * dspeed
                if self.get_id('M') in args:
                    dmach[:] += ds_dM * dspeed / 1e2
