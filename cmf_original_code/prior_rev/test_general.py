from __future__ import division
from framework import *

class Discipline1(ExplicitSystem):
    """Evaluates Simple EQ from test_newton:test_general_solver
        comp = a.add('comp', ExecComp(exprs=["f=a * x**n + b * x - c"]))
        comp.n = 77.0/27.0
        comp.a = 1.0
        comp.b = 1.0
        comp.c = 10.0
        comp.x = 0.0"""

    def _declare(self):
        self._declare_variable('f')
        self._declare_argument('x')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        x = p('x')[0]
        f = u('f')
        f[0]= 1.0*x**(77.0/27.0) + 1.0*x - 10.0

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        x = p('x')[0]
        f = u('f')[0]
        dx = dp('x')
        df = dg('f')

        df_dx = (77.0/27.0)*x**(50.0/27.0) + 1.0

        if self.mode == 'fwd':
            df[0] = 0
            if self.get_id('x') in args:
                df[0] += df_dx*dx[0]
        else:
            if self.get_id('x') in args:
                dx[0] = df_dx*df[0]

class Pcomp(ImplicitSystem):
    """Evaluates Simple EQ from test_newton:test_general_solver
    driver.add_constraint('comp.f=0')
    """

    def _declare(self):
        self._declare_variable('out')
        self._declare_argument('f')

    def apply_F(self):

        pvec = self.vec['p']
        fvec = self.vec['f']

        f = pvec('f')
        out_res = fvec('out')

        out_res[:] = f

    def apply_dFdpu(self, args):
        """ compute the trivial derivatives of the system """

        dpvec = self.vec['dp']
        duvec = self.vec['du']
        dfvec = self.vec['df']

        df = dpvec('f')
        dout_res = dfvec('out')
        dout = duvec('out')

        if self.mode == 'fwd':
            dout_res[:] = 0.0
            if self.get_id('f') in args:
                dout_res[:] += df

        elif self.mode == 'rev':
            df[:] = 0.0
            dout[:] = 0.0
            if self.get_id('f') in args:
                df[:] += dout_res

main = SerialSystem('main',
                    NL='NEWTON',
                    LN='KSP_PC',
                    LN_ilimit=10,
                    NL_ilimit=10,
                    NL_rtol=1e-6,
                    NL_atol=1e-9,
                    LN_rtol=1e-6,
                    LN_atol=1e-10,
                    output=True,
                    subsystems=[
                        IndVar('x', val=3.0),
                        Discipline1('f'),
                        Pcomp('out'),
                        ]).setup()


print main.compute()
print 'fwd'
#print main.compute_derivatives('fwd', 'x', output=False)
#print main.compute_derivatives('fwd', 'y', output=False)
print 'rev'
#print main.compute_derivatives('fwd', 'p1', output=False)
#print main.compute_derivatives('rev', 'p1', output=False)

main.check_derivatives_all()