"""
INTENDED FOR MISSION ANALYSIS USE
b-splines parameterization of altitude, x-distance, and Mach number.
These provide altitude and Mach number profile wrt x-distance.
Gamma (flight path angle) is also computed using the altitude
parameterization

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

import MBI, scipy.sparse
import numpy as np

from openmdao.main.api import Component
from openmdao.main.datatypes.api import Array, Float, Bool

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103

def setup_MBI(num_pts, num_cp, x_init):
    """ generate jacobians for b-splines using MBI package """

    alt = np.linspace(0, 16, num_pts)
    x_dist = np.linspace(0, x_init[-1], num_pts)/1e6

    arr = MBI.MBI(alt, [x_dist], [num_cp], [4])
    jac = arr.getJacobian(0, 0)
    jacd = arr.getJacobian(1, 0)

    c_arryx = x_init
    d_arryx = jacd.dot(c_arryx)*1e6

    lins = np.linspace(0, num_pts-1, num_pts).astype(int)
    diag = scipy.sparse.csc_matrix((1.0/d_arryx,
                                    (lins,lins)))
    jace = diag.dot(jacd)

    return jac, jace


class BSplineSystem(Component):
    """ Class used to allow the setup of b-splines """

    def __init__(self, num_elem=10, num_pt=5, x_init=None):
        super(BSplineSystem, self).__init__()

        # Inputs
        #self.add('x_init', Array(x_init, iotype='in',
        #         desc = 'Initial control point positions.',
        #         deriv_ignore=True))

        self.num_elem = num_elem+1
        self.num_pt = num_pt


class SysXBspline(BSplineSystem):
    """ A b-spline parameterization of distance """

    def __init__(self, num_elem=10, num_pt=5, x_init=None, jac_h=None):
        super(SysXBspline, self).__init__(num_elem, num_pt, x_init)

        # Inputs
        self.add('x_pt', Array(np.zeros((num_pt, )), iotype='in',
                            desc = 'Control points for distance'))

        # Outputs
        self.add('x', Array(np.zeros((num_elem+1, )), iotype='out',
                            desc = 'b-spline parameterization for distance'))

        self.jac_h = jac_h

    def execute(self):
        """ Compute x b-spline values with x control point values using
        pre-calculated MBI jacobian.
        """

        self.x = self.jac_h.dot(self.x_pt)

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['x_pt']
        output_keys = ['x']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute derivative of x b-spline values wrt x control points
        using pre-calculated MBI jacobian.
        Forward mode.
        """

        result['x'] += self.jac_h.dot(arg['x_pt'])

    def apply_derivT(self, arg, result):
        """ Compute derivative of x b-spline values wrt x control points
        using pre-calculated MBI jacobian.
        Adjoint mode.
        """

        result['x_pt'] += self.jac_h.T.dot(arg['x'])


class SysHBspline(BSplineSystem):
    """ A b-spline parameterization of altitude """

    def __init__(self, num_elem=10, num_pt=5, x_init=None, jac_h=None):
        super(SysHBspline, self).__init__(num_elem, num_pt, x_init)

        # Inputs
        self.add('h_pt', Array(np.zeros((num_pt, )), iotype='in',
                            desc = 'Control points for altitude'))

        # Outputs
        self.add('h', Array(np.zeros((num_elem+1, )), iotype='out',
                            desc = 'b-spline parameterization for altitude'))

        self.jac_h = jac_h

    def execute(self):
        """ Compute h b-splines values using h control point values using
        pre-calculated MBI jacobian.
        """

        self.h = self.jac_h.dot(self.h_pt)

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['h_pt']
        output_keys = ['h']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute h b-spline derivatives wrt h control points using
        pre-calculated MBI jacobian.
        Forward mode.
        """

        result['h'] += self.jac_h.dot(arg['h_pt'])

    def apply_derivT(self, arg, result):
        """ Compute h b-spline derivatives wrt h control points using
        pre-calculated MBI jacobian.
        Adjoint mode.
        """

        result['h_pt'] += self.jac_h.T.dot(arg['h'])


class SysMVBspline(BSplineSystem):
    """ A b-spline parameterization of Mach number """

    def __init__(self, num_elem=10, num_pt=5, x_init=None, jac_h=None):
        super(SysMVBspline, self).__init__(num_elem, num_pt, x_init)

        # Inputs
        self.add('M_pt', Array(np.zeros((num_pt, )), iotype='in',
                            desc = 'Control points for Mach number'))
        self.add('v_pt', Array(np.zeros((num_pt, )), iotype='in',
                            desc = 'Control points for velocity'))

        # Outputs
        self.add('M', Array(np.zeros((num_elem+1, )), iotype='out',
                            desc = 'b-spline parameterization for Mach number'))
        self.add('v_spline', Array(np.zeros((num_elem+1, )), iotype='out',
                            desc = 'b-spline parameterization for velocity'))

        self.jac_h = jac_h

    def execute(self):
        """ Compute M b-spline values using M control point values using
        pre-calculated MBI jacobian.
        """

        self.M = self.jac_h.dot(self.M_pt)
        self.v_spline = self.jac_h.dot(self.v_pt)

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['M_pt', 'v_pt']
        output_keys = ['M', 'v_spline']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute M b-spline derivatives wrt M control points using
        pre-calculated MBI jacobian.
        Forward mode.
        """

        if 'M_pt' in arg and 'M' in result:
            result['M'] += self.jac_h.dot(arg['M_pt'])
        if 'v_pt' in arg and 'v_spline' in result:
            result['v_spline'] += self.jac_h.dot(arg['v_pt'])

    def apply_derivT(self, arg, result):
        """ Compute M b-spline derivatives wrt M control points using
        pre-calculated MBI jacobian.
        Adjoint mode.
        """

        if 'M_pt' in result and 'M' in arg:
            result['M_pt'] += self.jac_h.T.dot(arg['M'])
        if 'v_pt' in result and 'v_spline' in arg:
            result['v_pt'] += self.jac_h.T.dot(arg['v_spline'])


class SysGammaBspline(BSplineSystem):
    """ dh/dx obtained from b-spline parameterization of altitude """

    def __init__(self, num_elem=10, num_pt=5, x_init=None, jac_gamma=None):
        super(SysGammaBspline, self).__init__(num_elem, num_pt, x_init)

        # Inputs
        self.add('h_pt', Array(np.zeros((num_pt, )), iotype='in',
                            desc = 'Control points for altitude'))

        # Outputs
        self.add('Gamma', Array(np.zeros((num_elem+1, )), iotype='out',
                            desc = 'Flight path angle w/ b-spline '
                            'parameterization'))

        self.jac_gamma = jac_gamma

    def execute(self):
        """ Compute gamma b-spline values using gamma control point values
        using pre-calculated MBI jacobian.
        """

        self.Gamma = self.jac_gamma.dot(self.h_pt) * 1e3/1e-1

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['h_pt']
        output_keys = ['Gamma']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute gamma b-spline derivatives wrt gamma control points using
        pre-calculated MBI jacobian.
        Forward mode.
        """

        result['Gamma'] += self.jac_gamma.dot(arg['h_pt']) * 1e3/1e-1

    def apply_derivT(self, arg, result):
        """ Compute gamma b-spline derivatives wrt gamma control points using
        pre-calculated MBI jacobian.
        Adjoint mode.
        """

        result['h_pt'] += self.jac_gamma.T.dot(arg['Gamma']) * 1e3/1e-1
