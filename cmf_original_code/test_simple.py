from __future__ import division
from framework import *

class Discipline1(ExplicitSystem):
    """Evaluates Paraboloid"""

    def _declare(self):
        self._declare_variable('f_xy')
        self._declare_argument('x')
        self._declare_argument('y')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        x, y = p('x')[0], p('y')[0]
        f_xy = u('f_xy')
        f_xy[0]= (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        x, y = p('x')[0], p('y')[0]
        f_xy = u('f_xy')[0]
        dx, dy = dp('x'), dp('y')
        df_xy = dg('f_xy')

        df_dx = 2.0*x - 6.0 + y
        df_dy = 2.0*y + 8.0 + x

        if self.mode == 'fwd':
            df_xy[0] = 0
            if self.get_id('x') in args:
                df_xy[0] += df_dx*dx[0]
            if self.get_id('y') in args:
                df_xy[0] += df_dy*dy[0]
        else:
            if self.get_id('x') in args:
                dx[0] = df_dx*df_xy[0]
            if self.get_id('y') in args:
                dy[0] = df_dy*df_xy[0]


class Dummy2(ExplicitSystem):
    """Evaluates Paraboloid"""

    def _declare(self):
        self._declare_variable('dum2')
        self._declare_argument('f_xy')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        x = p('f_xy')[0]
        y = u('dum2')
        y[0]= x

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        x = p('f_xy')[0]
        y = u('dum2')
        dx = dp('f_xy')
        dy_dx = dg('dum2')

        if self.mode == 'fwd':
            dy_dx[0] = dx[0]
        else:
            dx[0] = dy_dx[0]

class Dummy3(ExplicitSystem):
    """Evaluates Paraboloid"""

    def _declare(self):
        self._declare_variable('dum3')
        self._declare_argument('dum2')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        x = p('dum2')[0]
        y = u('dum3')
        y[0]= x

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        x = p('dum2')[0]
        y = u('dum3')
        dx = dp('dum2')
        dy_dx = dg('dum3')

        if self.mode == 'fwd':
            dy_dx[0] = dx[0]
        else:
            dx[0] = dy_dx[0]

class Pcomp1(ExplicitSystem):
    """Evaluates something"""

    def _declare(self):
        self._declare_variable('p1')
        self._declare_argument('f_xy')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        y2 =  p('f_xy')[0]
        p1 = u('p1')
        p1[0] = y2

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        y2 =  p('f_xy')[0]
        p1 = u('p1')
        dy2 = dp('f_xy')
        dp1 = dg('p1')

        if self.mode == 'fwd':
            dp1[0] = 0
            if self.get_id('f_xy') in args:
                dp1[0] += dy2[0]
        else:
            if self.get_id('f_xy') in args:
                dy2[0] += dp1[0]


main = SerialSystem('main',
                    NL='NLN_GS',
                    LN='KSP_PC',
                    LN_ilimit=1,
                    NL_ilimit=1,
                    NL_rtol=1e-6,
                    NL_atol=1e-9,
                    LN_rtol=1e-10,
                    LN_atol=1e-12,
                    output=True,
                    subsystems=[
                        ParallelSystem('par',
                                       NL='NLN_GS',
                                       LN='KSP_PC',
                                       LN_ilimit=1,
                                       NL_ilimit=1,
                                       NL_rtol=1e-6,
                                       NL_atol=1e-9,
                                       LN_rtol=1e-10,
                                       LN_atol=1e-12,
                                       subsystems = [
                                           IndVar('x', val=3.0),
                                           IndVar('y', val=5.0),]),
                        Discipline1('f_xy'),
                        Pcomp1('p1'),
                        ]).setup()


print main.compute()
print 'fwd'
print main.compute_derivatives('fwd', 'x', output=False)
#print main.compute_derivatives('fwd', 'y', output=False)
print 'rev'
#print main.compute_derivatives('fwd', 'p1', output=False)
#print main.compute_derivatives('rev', 'p1', output=False)

main.check_derivatives_all()