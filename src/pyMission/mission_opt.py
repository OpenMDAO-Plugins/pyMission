"""
Trajectory optimization.
"""

# pylint: disable=E1101
import numpy as np

from openmdao.lib.casehandlers.api import BSONCaseRecorder
from openmdao.lib.drivers.api import NewtonSolver, BroydenSolver
from openmdao.main.api import Assembly, set_as_top
from openmdao.main.datatypes.api import Array, Float

from pyMission.segment import MissionSegment


num_elem = 100
num_cp = 30
x_range = 5000.0e3

x_init = x_range * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
v_init = np.ones(num_cp)*2.3
h_init = 1 * np.sin(np.pi * x_init / (x_range/1e6))

model = set_as_top(MissionSegment(num_elem, num_cp, x_init))

model.h_pt = h_init
model.v_pt = v_init

# Pull velocity from BSpline instead of calculating it.
model.SysSpeed.v_specified = True

# Initial parameters
model.S = 427.8/1e2
model.ac_w = 210000*9.81/1e6
model.thrust_sl = 1020000.0/1e6/3
model.SFCSL = 8.951
model.AR = 8.68
model.oswald = 0.8

model.run()