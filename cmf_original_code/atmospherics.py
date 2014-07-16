"""
INTENDED FOR MISSION ANALYSIS USE
Atmospheric models for specific fuel consumption (SFC), temperature, and
density. All models extracted from the linear portion of the standard
atmosphere
"""

# pylint: disable=E1101
from __future__ import division
import sys
from framework import *
import numpy

class SysSFC(ExplicitSystem):
    """ linear SFC model wrt altitude """

    def _declare(self):
        """ owned variable: SFC (specific fuel consumption)
            dependencies: h (altitude)
                          SFCSL (sea-level SFC value)
        """

        self.num_elem = self.kwargs['num_elem']
        num_pts = self.num_elem+1
        ind_pts = range(num_pts)

        self._declare_variable('SFC', size=num_pts)
        self._declare_argument('h', indices=ind_pts)
        self._declare_argument(['SFCSL', 0], indices=[0])

    def apply_G(self):
        """ compute SFC value using sea level SFC and altitude
            the model is a linear correction for altitude changes
        """

        pvec = self.vec['p']
        uvec = self.vec['u']
        alt = pvec('h') * 1e3
        sfcsl = pvec(['SFCSL', 0]) * 1e-6
        sfc = uvec('SFC')

        sfc_temp = sfcsl + (6.39e-13) * alt
        sfc[:] = sfc_temp / 1e-6

    def apply_dGdp(self, args):
        """ compute SFC derivatives wrt sea level SFC and altitude """

        dpvec = self.vec['dp']
        dgvec = self.vec['dg']

        dalt = dpvec('h')
        dsfcsl = dpvec('SFCSL')
        dsfc = dgvec('SFC')

        dsfc_dalt = 6.39e-13

        if self.mode == 'fwd':
            dsfc[:] = 0.0
            if self.get_id('h') in args:
                dsfc[:] += (dsfc_dalt * dalt) * 1e3/1e-6
            if self.get_id('SFCSL') in args:
                dsfc[:] += dsfcsl

        if self.mode == 'rev':
            dalt[:] = 0.0
            dsfcsl[:] = 0.0

            if self.get_id('h') in args:
                dalt[:] += dsfc_dalt * dsfc * 1e3/1e-6
            if self.get_id('SFCSL') in args:
                dsfcsl[:] += numpy.sum(dsfc)


class SysTemp(ExplicitSystem):
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
