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

import numpy as np
import MBI, scipy.sparse

from openmdao.main.api import ImplicitComponent, Component
from openmdao.main.datatypes.api import Array, Float


def setup_surrogate(surr_file):

    raw = np.loadtxt(surr_file+'_inputs.dat')
    [CL, CD, CM] = np.loadtxt(surr_file+'_outputs.dat')

    M_num, a_num, h_num, e_num = raw[:4].astype(int)
    M_surr = raw[4:4 + M_num]
    a_surr = raw[4 + M_num:4 + M_num + a_num]
    h_surr = raw[4 + M_num + a_num:4 + M_num + a_num + h_num]
    e_surr = raw[4 + M_num + a_num + h_num:]

    mbi_CL = np.zeros((M_num, a_num, h_num, e_num))
    mbi_CD = np.zeros((M_num, a_num, h_num, e_num))
    mbi_CM = np.zeros((M_num, a_num, h_num, e_num))

    count = 0
    for i in xrange(M_num):
        for j in xrange(a_num):
            for k in xrange(h_num):
                for l in xrange(e_num):
                    mbi_CL[i][j][k][l] = CL[count]
                    mbi_CD[i][j][k][l] = CD[count]
                    mbi_CM[i][j][k][l] = CM[count]
                    count += 1

    CL_arr = MBI.MBI(mbi_CL, [M_surr, a_surr, h_surr, e_surr],
                     [M_num, a_num, h_num, e_num], [4, 4, 4, 4])
    CD_arr = MBI.MBI(mbi_CD, [M_surr, a_surr, h_surr, e_surr],
                     [M_num, a_num, h_num, e_num], [4, 4, 4, 4])
    CM_arr = MBI.MBI(mbi_CM, [M_surr, a_surr, h_surr, e_surr],
                     [M_num, a_num, h_num, e_num], [4, 4, 4, 4])

    nums = {
        'M': M_num,
        'a': a_num,
        'h': h_num,
        'e': e_num,
        }

    return [CL_arr, CD_arr, CM_arr, nums]


class SysTripanCLSurrogate(ImplicitComponent):
    """ Tripan CL Surrogate Model"""

    def __init__(self, num_elem=10, num=None, CL=None):
        super(SysTripanCLSurrogate, self).__init__()

        # Inputs
        self.add('M', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Mach Number'))
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Altitude points'))
        self.add('eta', Array(np.ones((num_elem+1, )), iotype='in',
                              desc = 'Tail rotation angle'))
        self.add('CL_tar', Array(np.zeros((num_elem+1, )), iotype='in',
                                  desc = 'Target Coefficient of Lift'))

        # States
        self.add('alpha', Array(np.ones((num_elem+1, )), iotype='state',
                                desc = 'Angle of attack'))

        # Residuals
        self.add('alpha_res', Array(np.zeros((num_elem+1, )), iotype='residual',
                                desc = 'residual for Angle of attack'))

        self.num = num
        self.CL_arr = CL
        self.J_CL = [None for i in range(4)]

    def evaluate(self):
        """ Calculate residual for surrogate. """

        n_elem = len(self.alpha)

        Mach = self.M
        alpha = self.alpha * 180 / np.pi * 1e-1
        alt = self.h * 3.28 * 1e3
        eta = self.eta * 180 / np.pi * 1e-1
        CL_tar = self.CL_tar

        inputs = np.zeros((n_elem, 4))
        inputs[:, 0] = Mach
        inputs[:, 1] = alpha
        inputs[:, 2] = alt
        inputs[:, 3] = eta

        CL_temp = self.CL_arr.evaluate(inputs)

        CL = np.zeros(n_elem)
        for index in xrange(n_elem):
            CL[index] = CL_temp[index, 0]

        self.alpha_res = CL - CL_tar

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['M', 'h', 'eta', 'CL_tar', 'alpha']
        output_keys = ['alpha_res']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        n_elem = len(self.alpha)

        Mach = self.M
        alpha = self.alpha * 180 / np.pi * 1e-1
        alt = self.h * 3.28 * 1e3
        eta = self.eta * 180 / np.pi * 1e-1

        inputs = np.zeros((n_elem, 4))
        inputs[:, 0] = Mach
        inputs[:, 1] = alpha
        inputs[:, 2] = alt
        inputs[:, 3] = eta

        for index in xrange(4):
            self.J_CL[index] = self.CL_arr.evaluate(inputs,
                                                    1+index, 0)[:, 0]

    def apply_deriv(self, arg, result):
        """ Compute the derivatives of lift and drag coefficient wrt alpha,
        eta, aspect ratio, and Osawld's efficiency.
        Forward Mode
        """

        dres = result['alpha_res']

        if 'M' in arg:
            dMach = arg['M']
            dres[:] += self.J_CL[0] * dMach
        if 'alpha' in arg:
            dalpha = arg['alpha']
            dres[:] += self.J_CL[1] * dalpha * 180 / np.pi * 1e-1
        if 'h' in arg:
            dalt = arg['h']
            dres[:] += self.J_CL[2] * dalt * 3.28 * 1e3
        if 'eta' in arg:
            deta = arg['eta']
            dres[:] += self.J_CL[3] * deta * 180 / np.pi * 1e-1
        if 'CL_tar' in arg:
            dCL = arg['CL_tar']
            dres[:] -= dCL

    def apply_derivT(self, arg, result):
        """ Compute the derivatives of lift and drag coefficient wrt alpha,
        eta, aspect ratio, and Osawld's efficiency.
        Adjoint Mode
        """

        dres = arg['alpha_res']

        if 'M' in result:
            dMach = result['M']
            dMach[:] += self.J_CL[0] * dres
        if 'alpha' in result:
            dalpha = result['alpha']
            dalpha[:] += self.J_CL[1] * dres * 180 / np.pi * 1e-1
        if 'h' in result:
            dalt = result['h']
            dalt[:] += self.J_CL[2] * dres * 3.28 * 1e3
        if 'eta' in result:
            deta = result['eta']
            deta[:] += self.J_CL[3] * dres * 180 / np.pi * 1e-1
        if 'CL_tar' in result:
            dCL = result['CL_tar']
            dCL[:] -= dres

class SysTripanCDSurrogate(Component):
    """ Tripan CD Surrogate Model"""

    def __init__(self, num_elem=10, num=None, CD=None):
        super(SysTripanCDSurrogate, self).__init__()

        # Inputs
        self.add('M', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Mach Number'))
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Altitude points'))
        self.add('eta', Array(np.ones((num_elem+1, )), iotype='in',
                              desc = 'Tail rotation angle'))
        self.add('alpha', Array(np.ones((num_elem+1, )), iotype='in',
                                desc = 'Angle of attack'))

        # Outputs
        self.add('CD', Array(np.zeros((num_elem+1, )), iotype='out',
                                desc = 'Drag coefficient'))

        self.num = num
        self.CD_arr = CD
        self.J_CD = [None for i in range(4)]

    def execute(self):
        """ Calculate residual for surrogate. """

        n_elem = len(self.alpha)

        Mach = self.M
        alpha = self.alpha * 180 / np.pi * 1e-1
        alt = self.h * 3.28 * 1e3
        eta = self.eta * 180 / np.pi * 1e-1

        inputs = np.zeros((n_elem, 4))
        inputs[:, 0] = Mach
        inputs[:, 1] = alpha
        inputs[:, 2] = alt
        inputs[:, 3] = eta

        CD_temp = self.CD_arr.evaluate(inputs)

        CD = np.zeros(n_elem)
        for index in xrange(n_elem):
            CD[index] = CD_temp[index, 0] / 1e-1 + 0.015/1e-1

        self.CD = CD

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['M', 'h', 'eta', 'alpha']
        output_keys = ['CD']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        n_elem = len(self.alpha)

        Mach = self.M
        alpha = self.alpha * 180 / np.pi * 1e-1
        alt = self.h * 3.28 * 1e3
        eta = self.eta * 180 / np.pi * 1e-1

        inputs = np.zeros((n_elem, 4))
        inputs[:, 0] = Mach
        inputs[:, 1] = alpha
        inputs[:, 2] = alt
        inputs[:, 3] = eta

        for index in xrange(4):
            self.J_CD[index] = self.CD_arr.evaluate(inputs,
                                                    1+index, 0)[:, 0]

    def apply_deriv(self, arg, result):
        """ Compute the derivatives of lift and drag coefficient wrt alpha,
        eta, aspect ratio, and Osawld's efficiency.
        Forward Mode
        """

        dCD = result['CD']

        if 'M' in arg:
            dMach = arg['M']
            dCD[:] += self.J_CD[0] * dMach / 1e-1
        if 'alpha' in arg:
            dalpha = arg['alpha']
            dCD[:] += self.J_CD[1] * dalpha * 180 / np.pi
        if 'h' in arg:
            dalt = arg['h']
            dCD[:] += self.J_CD[2] * dalt * 3.28 * 1e3 / 1e-1
        if 'eta' in arg:
            deta = arg['eta']
            dCD[:] += self.J_CD[3] * deta * 180 / np.pi

    def apply_derivT(self, arg, result):
        """ Compute the derivatives of lift and drag coefficient wrt alpha,
        eta, aspect ratio, and Osawld's efficiency.
        Adjoint Mode
        """

        dCD = arg['CD']

        if 'M' in result:
            dMach = result['M']
            dMach[:] += self.J_CD[0] * dCD / 1e-1
        if 'alpha' in result:
            dalpha = result['alpha']
            dalpha[:] += self.J_CD[1] * dCD * 180 / np.pi
        if 'h' in result:
            dalt = result['h']
            dalt[:] += self.J_CD[2] * dCD * 3.28 * 1e3 / 1e-1
        if 'eta' in result:
            deta = result['eta']
            deta[:] += self.J_CD[3] * dCD * 180 / np.pi

class SysTripanCMSurrogate(ImplicitComponent):
    """ Tripan CM Surrogate Model"""

    def __init__(self, num_elem=10, num=None, CM=None):
        super(SysTripanCMSurrogate, self).__init__()

        # Inputs
        self.add('M', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Mach Number'))
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='in',
                            desc = 'Altitude points'))
        self.add('alpha', Array(np.ones((num_elem+1, )), iotype='in',
                                desc = 'Angle of attack'))

        # States
        self.add('eta', Array(np.ones((num_elem+1, )), iotype='state',
                              desc = 'Tail rotation angle'))

        # Residuals
        self.add('CM', Array(np.zeros((num_elem+1, )), iotype='residual',
                                 desc = 'residual of Moment coefficient'))

        self.num = num
        self.CM_arr = CM
        self.J_CM = [None for i in range(4)]

    def execute(self):
        """ Calculate residual for surrogate. """

        n_elem = len(self.alpha)

        Mach = self.M
        alpha = self.alpha * 180 / np.pi * 1e-1
        alt = self.h * 3.28 * 1e3
        eta = self.eta * 180 / np.pi * 1e-1
        res = self.CM

        inputs = np.zeros((n_elem, 4))
        inputs[:, 0] = Mach
        inputs[:, 1] = alpha
        inputs[:, 2] = alt
        inputs[:, 3] = eta

        CM_temp = self.CM_arr.evaluate(inputs)

        CD = np.zeros(n_elem)
        for index in xrange(n_elem):
            res[index] = CM_temp[index, 0]

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['M', 'h', 'eta', 'alpha']
        output_keys = ['CM']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        n_elem = len(self.alpha)

        Mach = self.M
        alpha = self.alpha * 180 / np.pi * 1e-1
        alt = self.h * 3.28 * 1e3
        eta = self.eta * 180 / np.pi * 1e-1

        inputs = np.zeros((n_elem, 4))
        inputs[:, 0] = Mach
        inputs[:, 1] = alpha
        inputs[:, 2] = alt
        inputs[:, 3] = eta

        for index in xrange(4):
            self.J_CM[index] = self.CM_arr.evaluate(inputs,
                                                    1+index, 0)[:, 0]

    def apply_deriv(self, arg, result):
        """ Compute the derivatives of lift and drag coefficient wrt alpha,
        eta, aspect ratio, and Osawld's efficiency.
        Forward Mode
        """

        dres = result['CM']

        if 'M' in arg:
            dMach = arg['M']
            dres[:] += self.J_CM[0] * dMach
        if 'alpha' in arg:
            dalpha = arg['alpha']
            dres[:] += self.J_CM[1] * dalpha * 180 / np.pi * 1e-1
        if 'h' in arg:
            dalt = arg['h']
            dres[:] += self.J_CM[2] * dalt * 3.28 * 1e3
        if 'eta' in arg:
            deta = arg['eta']
            dres[:] += self.J_CM[3] * deta * 180 / np.pi * 1e-1

    def apply_derivT(self, arg, result):
        """ Compute the derivatives of lift and drag coefficient wrt alpha,
        eta, aspect ratio, and Osawld's efficiency.
        Adjoint Mode
        """

        dres = arg['CM']

        if 'M' in result:
            dMach = result['M']
            dMach[:] += self.J_CM[0] * dres
        if 'alpha' in result:
            dalpha = result['alpha']
            dalpha[:] += self.J_CM[1] * dres * 180 / np.pi * 1e-1
        if 'h' in result:
            dalt = result['h']
            dalt[:] += self.J_CM[2] * dres * 3.28 * 1e3
        if 'eta' in result:
            deta = result['eta']
            deta[:] += self.J_CM[3] * dres * 180 / np.pi * 1e-1

