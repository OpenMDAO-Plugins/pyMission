"""
INTENDED FOR MISSION ANALYSIS USE
This file contains the aerodynamic models used by the mission analysis
code. The present implementation uses linear aerodynamics.
"""

# pylint: disable=E1101
from __future__ import division
import sys

import numpy as np

from openmdao.main.api import Component, ImplicitComponent
from openmdao.main.datatypes.api import Array, Float

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103

class SysAeroSurrogate(Component):
    """ Simulates the presence of an aero surrogate mode using linear
    aerodynamic model
    """

    def __init__(self, num_elem):
        super(SysAeroSurrogate, self).__init__()

        # Inputs
        self.add('alpha', Array(np.zeros((num_elem, )), iotype='in',
                                desc = 'Angle of attack'))
        self.add('eta', Array(np.zeros((num_elem, )), iotype='in',
                              desc = 'Tail rotation angle'))
        self.add('AR', Float(0.0, iotype='in',
                             desc = 'Aspect Ratio'))
        self.add('oswald', Float(0.0, iotype='in',
                                 desc = "Oswald's efficiency"))

        # Outputs
        self.add('CL', Array(np.zeros((num_elem, )), iotype='out',
                             desc = 'Lift Coefficient'))
        self.add('CD', Array(np.zeros((num_elem, )), iotype='out',
                             desc = 'Drag Coefficient'))

    def execute(self):
        """ Compute lift and drag coefficient using angle of attack and tail
        rotation angles. Linear aerodynamics is assumed.
        """

        alpha = self.alpha * 1e-1
        eta = self.eta * 1e-1
        aspect_ratio = self.AR
        oswald = self.oswald

        lift_c0 = 0.26
        lift_ca = 4.24
        lift_ce = 0.27
        drag_c0 = 0.018

        self.CL = lift_c0 + lift_ca*alpha + lift_ce*eta
        self.CD = (drag_c0 + self.CL**2 /
                   (np.pi * aspect_ratio * oswald))/1e-1

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['alpha', 'eta', 'AR', 'oswald']
        output_keys = ['CL', 'CD']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute the derivatives of lift and drag coefficient wrt alpha,
        eta, aspect ratio, and Osawld's efficiency.
        Forward Mode
        """

        aspect_ratio = self.AR
        oswald = self.oswald
        lift_c = self.CL

        lift_ca = 4.24
        lift_ce = 0.27

        if 'alpha' in arg:
            dalpha = arg['alpha']
            if 'CL' in result:
                result['CL'] += lift_ca * dalpha * 1e-1
            if 'CD' in result:
                result['CD'] += 2 * lift_c * lift_ca / (np.pi * oswald *
                                                        aspect_ratio) * dalpha
        if 'eta' in arg:
            deta = arg['eta']
            if 'CL' in result:
                result['CL'] += lift_ce * deta * 1e-1
            if 'CD' in result:
                result['CD'] += 2 * lift_c * lift_ce / (np.pi * oswald *
                                                        aspect_ratio) * deta
        if 'AR' in arg:
            daspect_ratio = arg['AR']
            if 'CD' in result:
                result['CD'] -= lift_c**2 / (np.pi * aspect_ratio**2 *
                                             oswald) * daspect_ratio / 1e-1
        if 'oswald' in arg:
            doswald = arg['oswald']
            if 'CD' in result:
                result['CD'] -= lift_c**2 / (np.pi * aspect_ratio *
                                             oswald**2) * doswald / 1e-1

    def apply_derivT(self, arg, result):
        """ Compute the derivatives of lift and drag coefficient wrt alpha,
        eta, aspect ratio, and Osawld's efficiency.
        Adjoint Mode
        """

        aspect_ratio = self.AR
        oswald = self.oswald
        lift_c = self.CL

        lift_ca = 4.24
        lift_ce = 0.27

        if 'alpha' in result:
            if 'CL' in arg:
                result['alpha'] += lift_ca * arg['CL'] * 1e-1
            if 'CD' in arg:
                result['alpha'] += 2 * lift_c * lift_ca / (np.pi * oswald *
                                                           aspect_ratio) * arg['CD']
        if 'eta' in result:
            if 'CL' in arg:
                result['eta'] += lift_ce * arg['CL'] * 1e-1
            if 'CD' in arg:
                result['eta'] += 2 * lift_c * lift_ce / (np.pi * oswald *
                                                         aspect_ratio) * arg['CD']
        if 'AR' in result:
            if 'CD' in arg:
                result['AR'] -= np.sum((lift_c**2 /
                                        (np.pi * oswald *
                                         aspect_ratio**2) *
                                        arg['CD'])) / 1e-1
        if 'oswald' in result:
            if 'CD' in arg:
                result['oswald'] -= np.sum((lift_c**2 /
                                            (np.pi * oswald**2 *
                                             aspect_ratio) *
                                            arg['CD'])) / 1e-1

class SysCM(ImplicitComponent):
    """ Compute the tail rotation angle necessary to maintain pitch moment
    equilibrium.
    """

    def __init__(self, num_elem):
        super(SysCM, self).__init__()

        # Inputs
        self.add('alpha', Array(np.zeros((num_elem, )), iotype='in',
                                desc = 'Angle of attack'))

        # States
        self.add('eta', Array(np.zeros((num_elem, )), iotype='state',
                              desc = 'Tail rotation angle'))

        # Residuals
        self.add('eta_res', Array(np.zeros((num_elem, )), iotype='residual',
                              desc = 'Tail rotation angle'))


    def evaluate(self):
        """ compute CM value using alpha and eta, and use the CM value as
            residual for eta
        """

        alpha = self.alpha * 1e-1
        eta = self.eta * 1e-1

        mmt_ca = 0.63
        mmt_ce = 1.06

        self.eta_res = (mmt_ca*alpha + mmt_ce*eta) / 1e-1

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['alpha', 'eta']
        output_keys = ['eta_res']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        """ Compute the derivatives of tail rotation angle wrt angle of
        attack.
        Forward Mode
        """

        mmt_ca = 0.63
        mmt_ce = 1.06

        if 'alpha' in arg:
            result['eta_res'] += mmt_ca * arg['alpha']
        if 'eta' in arg:
            result['eta_res'] += mmt_ce * arg['eta']

    def apply_derivT(self, arg, result):
        """ Compute the derivatives of tail rotation angle wrt angle of
        attack.
        Adjoint Mode
        """

        mmt_ca = 0.63
        mmt_ce = 1.06

        if 'alpha' in result:
            result['alpha'] += mmt_ca * arg['eta_res']
        if 'eta' in result:
            result['eta'] += mmt_ce * arg['eta_res']
