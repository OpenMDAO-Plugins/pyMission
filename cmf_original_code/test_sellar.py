from __future__ import division
from framework import *

class Discipline1(ExplicitSystem):
    """Evaluates the equation
    y1 = z1**2 + z2 + x1 - 0.2*y2"""

    def _declare(self):
        self._declare_variable('y1')
        self._declare_argument('x1')
        self._declare_argument('z1')
        self._declare_argument('z2')
        self._declare_argument('y2')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        x1, z1, z2, y2 = p('x1')[0], p('z1')[0], p('z2')[0], p('y2')[0]
        y1 = u('y1')
        y1[0] = z1**2 + z2 + x1 - 0.2*y2
        print 'D1', y1, y2

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        x1, z1, z2, y2 = p('x1')[0], p('z1')[0], p('z2')[0], p('y2')[0]
        y1 = u('y1')[0]
        dx1, dz1, dz2, dy2 = dp('x1'), dp('z1'), dp('z2'), dp('y2')
        dy1 = dg('y1')

        dy1_dx1 = 1.0
        dy1_dz1 = 2.0*dz1
        dy1_dz2 = 1.0
        dy1_dy2 = -0.2

        if self.mode == 'fwd':
            dy1[0] = 0
            if self.get_id('x1') in args:
                dy1[0] += dy1_dx1*dx1[0]
            if self.get_id('z1') in args:
                dy1[0] += dy1_dz1*dz1[0]
            if self.get_id('z2') in args:
                dy1[0] += dy1_dz2*dz2[0]
            if self.get_id('y2') in args:
                dy1[0] += dy1_dy2*dy2[0]
        else:
            if self.get_id('x1') in args:
                dx1[0] = dy1_dx1*dy1[0]
            if self.get_id('z1') in args:
                dz1[0] = dy1_dz1*dy1[0]
            if self.get_id('z2') in args:
                dz2[0] = dy1_dz2*dy1[0]
            if self.get_id('y2') in args:
                dy2[0] = dy1_dy2*dy1[0]


class Discipline2(ExplicitSystem):
    """Evaluates the equation
    y2 = y1**(.5) + z1 + z2"""

    def _declare(self):
        self._declare_variable('y2')
        self._declare_argument('z1')
        self._declare_argument('z2')
        self._declare_argument('y1')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        z1, z2, y1 =  p('z1')[0], p('z2')[0], p('y1')[0]
        y2 = u('y2')
        y2[0] = y1**(.5) + z1 + z2
        print 'D2', y2, y1

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        z1, z2, y1 = p('z1')[0], p('z2')[0], p('y1')[0]
        y2 = u('y2')[0]
        dz1, dz2, dy1 = dp('z1'), dp('z2'), dp('y1')
        dy2 = dg('y2')

        dy2_dz1 = 1.0
        dy2_dz2 = 1.0
        dy2_dy1 = 0.5*y1**(-0.5)

        if self.mode == 'fwd':
            dy2[0] = 0
            if self.get_id('z1') in args:
                dy2[0] += dy2_dz1*dz1[0]
            if self.get_id('z2') in args:
                dy2[0] += dy2_dz2*dz2[0]
            if self.get_id('y1') in args:
                dy2[0] += dy2_dy1*dy1[0]
        else:
            if self.get_id('z1') in args:
                dz1[0] = dy2_dz1*dy2[0]
            if self.get_id('z2') in args:
                dz2[0] = dy2_dz2*dy2[0]
            if self.get_id('y1') in args:
                dy1[0] = dy2_dy1*dy2[0]

class Pcomp1(ExplicitSystem):
    """Evaluates something"""

    def _declare(self):
        self._declare_variable('p1')
        self._declare_argument('y2')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        y2 =  p('y2')[0]
        p1 = u('p1')
        p1[0] = y2

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        y2 =  p('y2')[0]
        p1 = u('p1')
        dy2 = dp('y2')
        dp1 = dg('p1')

        if self.mode == 'fwd':
            dp1[0] = 0
            if self.get_id('y2') in args:
                dp1[0] += dy2[0]
        else:
            if self.get_id('y2') in args:
                dy2[0] += dp1[0]

class Pcomp2(ExplicitSystem):
    """Evaluates something"""

    def _declare(self):
        self._declare_variable('p2')
        self._declare_argument('y1')

    def apply_G(self):
        vec = self.vec
        p, u = vec['p'], vec['u']
        y1 =  p('y1')[0]
        p2 = u('p2')
        p2[0] = y1

    def apply_dGdp(self, args):
        vec = self.vec
        p, u, dp, du, dg = vec['p'], vec['u'], vec['dp'], vec['du'], vec['dg']
        y1 =  p('y1')[0]
        p2 = u('p2')
        dy1 = dp('y1')
        dp2 = dg('p2')

        if self.mode == 'fwd':
            dp2[0] = 0
            if self.get_id('y1') in args:
                dp2[0] += dy1[0]
        else:
            if self.get_id('y1') in args:
                dy1[0] += dp2[0]

main = SerialSystem('main',
                    #LN='KSP_PC',
                    LN='LIN_GS',
                    NL='NLN_GS',
                    LN_ilimit=1,
                    output=True,
                    subsystems=[IndVar('x1', val=1.0),
                                SerialSystem('subdriver',
                                             NL='NEWTON',
                                             output=True,
                                             subsystems=[
                                                 IndVar('z2', val=2.0),
                                                 IndVar('z1', val=5.0),
                                                 Discipline1('Disy1'),
                                                 Discipline2('Disy2')
                                                 ]),
                                #Pcomp1('p1'),
                                #Pcomp2('p2')
    ]).setup()


print main.compute()
print 'done'
#print main.vec['u']
#print 'fwd'
#print main.compute_derivatives('fwd', 'x1', output=False)
#print main.compute_derivatives('fwd', 'z2', output=False)
#print main.compute_derivatives('fwd', 'x1', output=False)
#print 'rev'
#print main.compute_derivatives('rev', 'p1', output=False)
#print main.compute_derivatives('rev', 'z2', output=False)
#print main.compute_derivatives('rev', 'x1', output=False)

#main.check_derivatives_all()
