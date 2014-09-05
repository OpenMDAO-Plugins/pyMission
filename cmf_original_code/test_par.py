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
                dx[0] += df_dx*df_xy[0]
            if self.get_id('y') in args:
                dy[0] += df_dy*df_xy[0]

class Discipline2(ExplicitSystem):
    """Evaluates Paraboloid"""

    def _declare(self):
        self._declare_variable('f_xy2')
        self._declare_argument('x2')
        self._declare_argument('y2')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        x, y = p('x2')[0], p('y2')[0]
        f_xy = u('f_xy2')
        f_xy[0]= (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        x, y = p('x2')[0], p('y2')[0]
        f_xy = u('f_xy2')[0]
        dx, dy = dp('x2'), dp('y2')
        df_xy = dg('f_xy2')

        df_dx = 2.0*x - 6.0 + y
        df_dy = 2.0*y + 8.0 + x

        if self.mode == 'fwd':
            df_xy[0] = 0
            if self.get_id('x2') in args:
                df_xy[0] += df_dx*dx[0]
            if self.get_id('y2') in args:
                df_xy[0] += df_dy*dy[0]
        else:
            if self.get_id('x2') in args:
                dx[0] += df_dx*df_xy[0]
            if self.get_id('y2') in args:
                dy[0] += df_dy*df_xy[0]

class Discipline3(ExplicitSystem):
    """Evaluates Paraboloid"""

    def _declare(self):
        self._declare_variable('zz')
        self._declare_argument('f_xy')
        self._declare_argument('f_xy2')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        x, y = p('f_xy')[0], p('f_xy2')[0]
        f_xy = u('zz')
        f_xy[0]= x + y

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        x, y = p('f_xy')[0], p('f_xy2')[0]
        f_xy = u('zz')[0]
        dx, dy = dp('f_xy'), dp('f_xy2')
        df_xy = dg('zz')

        df_dx = 1.0
        df_dy = 1.0

        if self.mode == 'fwd':
            df_xy[0] = 0
            if self.get_id('f_xy') in args:
                df_xy[0] += df_dx*dx[0]
            if self.get_id('f_xy2') in args:
                df_xy[0] += df_dy*dy[0]
        else:
            if self.get_id('f_xy') in args:
                dx[0] += df_dx*df_xy[0]
            if self.get_id('f_xy2') in args:
                dy[0] += df_dy*df_xy[0]


main = SerialSystem('main', subsystems=[
    IndVar('x', val=3.0),
    IndVar('y', val=5.0),
    IndVar('x2', val=2.0),
    IndVar('y2', val=4.0),
    Discipline1('f_xy'),
    Discipline2('f_xy2'),
    Discipline3('zz'),
    ]).setup()


print main.compute()
#print 'fwd'
#print main.compute_derivatives('fwd', 'x', output=False)
#print main.compute_derivatives('fwd', 'y', output=False)
#print main.compute_derivatives('fwd', 'x2', output=False)
#print main.compute_derivatives('fwd', 'y2', output=False)
#print 'rev'
#print main.compute_derivatives('rev', 'x', output=False)
#print main.compute_derivatives('rev', 'y', output=False)
#print main.compute_derivatives('rev', 'x2', output=False)
#print main.compute_derivatives('rev', 'y2', output=False)

main.check_derivatives_all(fwd=True)
main.check_derivatives_all(fwd=False)

print "Calculating full derivatives"
ikey = ('zz', 0)
for item in ['x', 'y', 'x2', 'y2']:
    der = main.compute_derivatives('rev', item, output=False)
    if ikey in der[0].keys():
        print item, der[0][ikey][0]
    else:
        print item, 'key not found in', der



