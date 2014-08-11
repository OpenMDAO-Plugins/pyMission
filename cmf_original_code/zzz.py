from __future__ import division
import sys
from framework import *

class Comp1(ImplicitSystem):
    """ Comp1
    """

    def _declare(self):
        """ owned variable:
            dependencies:
        """

        self._declare_variable('x')
        self._declare_argument('a')

    def apply_F(self):
        """ compute CM value using alpha and eta, and use the CM value as
            residual for eta
        """

        pvec = self.vec['p']
        uvec = self.vec['u']
        fvec = self.vec['f']

        a = pvec('a')
        x = uvec('x')
        res = fvec('x')

        res[:] = 3.0*a + 5.0*x - 2

    def apply_dFdpu(self, args):
        """ compute the derivatives of tail rotation angle wrt angle of attack
        """

        dpvec = self.vec['dp']
        duvec = self.vec['du']
        dfvec = self.vec['df']

        da = dpvec('a')
        dx = duvec('x')
        dres = dfvec('x')

        if self.mode == 'fwd':
            dres[:] = 0.0
            if self.get_id('a') in args:
                dres[:] += 3 * da
            if self.get_id('x') in args:
                dres[:] += 5 * dx

        elif self.mode == 'rev':
            da[:] = 0.0
            dx[:] = 0.0
            if self.get_id('a') in args:
                da[:] += 3 * dres
            if self.get_id('x') in args:
                dx[:] += 5 * dres

class Comp2(ExplicitSystem):
    """ Comp1
    """

    def _declare(self):
        """ owned variable:
            dependencies:
        """

        self._declare_variable('y')
        self._declare_argument('x')
        self._declare_argument('a')

    def apply_G(self):
        """ compute CM value using alpha and eta, and use the CM value as
            residual for eta
        """

        pvec = self.vec['p']
        uvec = self.vec['u']
        fvec = self.vec['f']

        a = pvec('a')
        x = pvec('x')
        y = uvec('y')

        y[:] = 7.0*a - 4.0*x - 1

    def apply_dGdpu(self, args):
        """ compute the derivatives of tail rotation angle wrt angle of attack
        """

        dpvec = self.vec['dp']
        duvec = self.vec['du']
        dfvec = self.vec['df']

        da = dpvec('a')
        dx = dpvec('x')
        dy = dfvec('y')

        if self.mode == 'fwd':
            dy[:] = 0.0
            if self.get_id('a') in args:
                dy[:] += 7 * da
            if self.get_id('x') in args:
                dy[:] += -3 * dx

        elif self.mode == 'rev':
            da[:] = 0.0
            dx[:] = 0.0
            if self.get_id('a') in args:
                da[:] += 7 * dy
            if self.get_id('x') in args:
                dx[:] += -4 * dy

class Comp3(ExplicitSystem):
    """ Comp1
    """

    def _declare(self):
        """ owned variable:
            dependencies:
        """

        self._declare_variable('yy')
        self._declare_argument('y')

    def apply_G(self):
        """ compute CM value using alpha and eta, and use the CM value as
            residual for eta
        """

        pvec = self.vec['p']
        uvec = self.vec['u']
        fvec = self.vec['f']

        y = pvec('y')
        yy = uvec('yy')

        yy[:] = 13.5*y

    def apply_dGdpu(self, args):
        """ compute the derivatives of tail rotation angle wrt angle of attack
        """

        dpvec = self.vec['dp']
        duvec = self.vec['du']
        dfvec = self.vec['df']

        dy = dpvec('y')
        dyy = dfvec('yy')

        if self.mode == 'fwd':
            dyy[:] = 0.0
            if self.get_id('y') in args:
                dyy[:] += 13.5 * dy

        elif self.mode == 'rev':
            dy[:] = 0.0
            if self.get_id('y') in args:
                dy[:] += 13.5 * dyy


top = SerialSystem('mission',
                   subsystems=[IndVar('a', val=1.0),
                               SerialSystem('loop',
                                            NL='NEWTON',
                                            LN='KSP_PC',
                                            LN_ilimit=150,
                                            NL_ilimit=150,
                                            PC_ilimit=2,
                                            NL_rtol=1e-10,
                                            NL_atol=1e-10,
                                            LN_rtol=1e-10,
                                            LN_atol=1e-10,
                                            subsystems=[
                                                Comp1('x'),
                                                Comp2('y'),
                                                # (Un)comment to toggle this explicit system
                                                Comp3('yy')
                                            ]),
                               ]).setup()

print top.vec['u'], top.vec['f']
print top.compute()
print top.vec['u'], top.vec['f']
