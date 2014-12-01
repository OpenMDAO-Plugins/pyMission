import time
import numpy as np

from openmdao.lib.casehandlers.api import BSONCaseRecorder

from pyMission.segment_737 import MissionSegment


num_elem = 500
num_cp = 100
x_range = 2950.0

gamma_lb = np.tan(-35.0 * (np.pi/180.0))/1e-1
gamma_ub = np.tan(35.0 * (np.pi/180.0))/1e-1
takeoff_speed = 83.3
landing_speed = 72.2


# define initial altitude profile, as well as fixed profile for
# x-distance and airspeed
x_range *= 1.852
x_init = x_range * 1e3 * np.linspace(0, 1, num_cp) / 1e6

M_init = np.ones(num_cp)*0.78
M_init[:2] = np.linspace(0.5, 0.78, 2)
M_init[95:] = np.linspace(0.78, 0.5, 5)

h_init = (35000 / 3.28) * np.ones(num_cp) / 1e3
h_init[:2] = np.linspace(0, 35000/3.28, 2) / 1e3
h_init[95:] = np.linspace(35000/3.28, 0, 5) / 1e3

model = MissionSegment(num_elem, num_cp, x_init)
model.h_pt = h_init
model.M_pt = M_init

# Calculate velocity from the Mach we have specified.
model.SysSpeed.v_specified = False

model.S = 124.58/1e2
model.ac_w = 57719.3*9.81/1e6
model.thrust_sl = 2*121436/1e6
model.SFCSL = (2*0.00001076/1e-6)*9.81
model.AR = 8.93
model.oswald = 0.8

filename = 'mission_history_737.bson'
model.recorders = [BSONCaseRecorder(filename)]
model.recorders.save_problem_formulation = True

start = time.time()
model.run()

print repr(model.SysAeroSurrogate.CD)
print 'OPTIMIZATION TIME:', time.time() - start