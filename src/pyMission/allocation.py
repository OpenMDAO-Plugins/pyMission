import numpy as np

from openmdao.lib.drivers.api import NewtonSolver, FixedPointIterator, BroydenSolver
from openmdao.main.api import Assembly, set_as_top, Driver
from openmdao.main.datatypes.api import Array, Float

from pyMission.segment import MissionSegment
from pyMission.aeroTripan import setup_surrogate
from pyMission.alloc_func import Profit, PaxCon, AircraftCon
from pyMission.bsplines import setup_MBI

from openmdao.main.container import dump
from pyoptsparse_driver.pyoptsparse_driver import pyOptSparseDriver
import check_deriv_patch
import unconn_patch


class AllocationProblem(Assembly):
    """ Allocation problem containing a set of mission analyses """

    def __init__(self, problem_file):
        super(AllocationProblem, self).__init__()

        data = {}
        execfile(problem_file, data)
        misc_data = data['misc_data']
        ac_data = data['ac_data']
        rt_data = data['rt_data']

        num_routes = rt_data['number']
        num_existing_ac = len(ac_data['existing_ac'])
        num_new_ac = len(ac_data['new_ac'])
        num_ac = num_existing_ac + num_new_ac

        self.num_routes = num_routes
        self.num_existing_ac = num_existing_ac
        self.num_new_ac = num_new_ac
        self.num_ac = num_ac

        ac_aerosurr = {}
        for ac_name in ac_data['new_ac']:
            data = {}
            execfile(ac_name+'_params.py', data)
            params = data['params']

            surr = {}
            surr['CL'], surr['CD'], surr['CM'], surr['nums'] = setup_surrogate(params['surr'])
            ac_aerosurr[ac_name] = surr

        self.gamma_lb = np.tan(-35.0 * (np.pi/180.0))/1e-1
        self.gamma_ub = np.tan(35.0 * (np.pi/180.0))/1e-1

        for irt in xrange(num_routes):
            num_elem = rt_data['num_elem'][irt]
            num_cp = rt_data['num_cp'][irt]
            x_range = rt_data['range'][irt] * 1.852
            altitude = 10 * np.sin(np.pi * np.linspace(0,1,num_elem+1))

            x_init = x_range * 1e3 * (1-np.cos(np.linspace(0, 1, num_cp)*np.pi))/2/1e6
            M_init = np.ones(num_cp)*0.82
            h_init = 10 * np.sin(np.pi * x_init / (x_range/1e3))

            jac_h, jac_gamma = setup_MBI(num_elem+1, num_cp, x_init)

            #altitude = 10 * np.sin(np.pi * np.linspace(0,1,num_elem+1))

            for iac in xrange(num_new_ac):
                ac_name = ac_data['new_ac'][iac]
                seg_name = 'Seg_%03i_%03i'%(irt,iac)

                seg = self.add(seg_name, MissionSegment(num_elem=num_elem, num_cp=num_cp,
                                                        x_pts=x_init, params_file=ac_name+'_params.py',
                                                        aero_surr=ac_aerosurr[ac_name],
                                                        jac_h=jac_h, jac_gamma=jac_gamma))

                seg.h_pt = h_init
                seg.M_pt = M_init
                seg.set_init_h_pt(altitude)
                seg.driver.system_type = 'serial'
                seg.coupled_solver.system_type = 'serial'
                seg.driver.gradient_options.iprint = 0
                seg.coupled_solver.gradient_options.iprint = 0
                seg.coupled_solver.iprint = 0
#                seg.coupled_solver.gradient_options.iprint = 0
                #seg.coupled_solver.pre_setup()

        self.add('pax_flt', Array(10*np.ones((num_routes, num_ac)), iotype='in'))
        self.add('flt_day', Array(np.ones((num_routes, num_ac)), iotype='in'))

        cost_fuel = np.zeros((num_routes, num_ac))
        prc_pax = np.zeros((num_routes, num_ac))
        cost_nf = np.zeros((num_routes, num_ac))
        maintenance = np.zeros((num_routes, num_ac))
        turnaround = np.zeros((num_routes, num_ac))
        for irt in xrange(num_routes):
            for iac in xrange(num_ac):
                if iac < num_existing_ac:
                    ac_name = ac_data['existing_ac'][iac]
                else:
                    inac = iac - num_existing_ac
                    ac_name = ac_data['new_ac'][inac]
                cost_fuel[irt, iac] = misc_data['cost/fuel'] * 2.2 / 9.81
                prc_pax[irt, iac] = ac_data['ticket price', ac_name][irt]
                cost_nf[irt, iac] = ac_data['flight cost no fuel', ac_name][irt]
                maintenance[irt, iac] = ac_data['MH', ac_name]
                turnaround[irt, iac] = misc_data['turnaround']

        self.add('SysProfit', Profit(num_routes, num_ac,
                                     prc_pax, cost_fuel, cost_nf))
        self.add('SysPaxCon', PaxCon(num_routes, num_ac))
        self.add('SysAircraftCon', AircraftCon(num_routes, num_ac,
                                               maintenance, turnaround))

        for iac in xrange(num_existing_ac):
            ac_name = ac_data['existing_ac'][iac]
            self.SysProfit.fuelburn[:, iac] = ac_data['fuel', ac_name] / 1e5
            self.SysAircraftCon.time[:, iac] = ac_data['block time', ac_name]

        self.connect('pax_flt', 'SysProfit.pax_flt')
        self.connect('flt_day', 'SysProfit.flt_day')

        self.connect('pax_flt', 'SysPaxCon.pax_flt')
        self.connect('flt_day', 'SysPaxCon.flt_day')

        self.connect('flt_day', 'SysAircraftCon.flt_day')

        for irt in xrange(num_routes):
            for inac in xrange(num_new_ac):
                iac = inac + num_existing_ac
                seg_name = 'Seg_%03i_%03i'%(irt,inac)

                self.connect(seg_name+'.fuelburn',
                             'SysProfit.fuelburn[%i, %i]'%(irt,iac))
                self.connect(seg_name+'.time',
                             'SysAircraftCon.time[%i, %i]'%(irt,iac))

        self.create_passthrough('SysProfit.profit')
        self.create_passthrough('SysPaxCon.pax_con')
        self.create_passthrough('SysAircraftCon.ac_con')

        pax_upper = np.zeros((num_routes, num_ac))
        demand = np.zeros(num_routes)
        ac_avail = np.zeros(num_ac)
        for irt in xrange(num_routes):
            for iac in xrange(num_ac):
                if iac < num_existing_ac:
                    ac_name = ac_data['existing_ac'][iac]
                else:
                    inac = iac - num_existing_ac
                    ac_name = ac_data['new_ac'][inac]
                pax_upper[irt, iac] = ac_data['capacity', ac_name]
        for irt in xrange(num_routes):
            demand[irt] = rt_data['demand'][irt]
        for iac in xrange(num_ac):
            if iac < num_existing_ac:
                ac_name = ac_data['existing_ac'][iac]
            else:
                inac = iac - num_existing_ac
                ac_name = ac_data['new_ac'][inac]
            ac_avail[iac] = 12 * ac_data['number', ac_name]

        self.add('pax_upper', Array(np.zeros((num_routes, num_ac)), iotype='out'))
        self.add('demand', Array(np.zeros(num_routes), iotype='out'))
        self.add('ac_avail', Array(np.zeros(num_ac), iotype='out'))

        self.pax_upper = pax_upper
        self.demand = demand
        self.ac_avail = ac_avail

        self.add('missions', Driver())
        for irt in xrange(self.num_routes):
            for inac in xrange(self.num_new_ac):
                seg_name = 'Seg_%03i_%03i'%(irt,inac)
                self.missions.workflow.add(seg_name)
#        self.missions.system_type = 'parallel'
        self.missions.gradient_options.lin_solver = "linear_gs"
        self.missions.gradient_options.iprint = 0

        self.driver.workflow.add(['missions', 'SysProfit', 'SysPaxCon', 'SysAircraftCon'])


if __name__ == '__main__':
    from subprocess import call

    def setup_opt(seg):
        asm = set_as_top(Assembly())
        asm.add('driver', pyOptSparseDriver())
        asm.driver.optimizer = 'SNOPT'
        asm.driver.options = {'Iterations limit': 5000000}#, 'Verify level':3}
        asm.driver.gradient_options.lin_solver = "linear_gs"
        asm.driver.gradient_options.maxiter = 1
        asm.driver.gradient_options.derivative_direction = 'adjoint'
        asm.driver.gradient_options.iprint = 0
        asm.driver.system_type = 'serial'

        asm.add('segment', seg)
        asm.driver.add_objective('segment.fuelburn')
        asm.driver.add_parameter('segment.h_pt', low=0.0, high=14.1)
        asm.driver.add_constraint('segment.h[0] = 0.0')
        asm.driver.add_constraint('segment.h[-1] = 0.0')
        asm.driver.add_constraint('segment.Tmin < 0.0')
        asm.driver.add_constraint('segment.Tmax < 0.0')
        asm.driver.add_constraint('%.15f < segment.Gamma < %.15f' % \
                                  (alloc.gamma_lb,alloc.gamma_ub), linear=True)

        return asm

    alloc = AllocationProblem('problem_3rt_2ac.py')
    if 0:
        alloc.run()
        dump(alloc.SysPaxCon, recurse=True)
        alloc.check_comp_derivatives()
        exit()

    for irt in xrange(alloc.num_routes):
        for inac in xrange(alloc.num_new_ac):
            seg_name = 'Seg_%03i_%03i' % (irt,inac)
            seg = alloc.get(seg_name)
            sub_opt = setup_opt(seg)

            sub_opt.run()

            call(['mv', 'SNOPT_print.out', 'SNOPT_%03i_%03i_print.out' % (irt,inac)])
            call(['rm', 'SNOPT_summary.out'])

    alloc.replace('driver', pyOptSparseDriver())
    alloc.driver.optimizer = 'SNOPT'
    alloc.driver.options = {'Iterations limit': 5000000}#, 'Verify level':3}
    alloc.driver.gradient_options.lin_solver = "linear_gs"
    alloc.driver.gradient_options.maxiter = 1
    alloc.driver.gradient_options.derivative_direction = 'adjoint'
    alloc.driver.gradient_options.iprint = 0
    alloc.driver.add_objective('profit')
    alloc.driver.add_parameter('pax_flt', low=0, high=alloc.pax_upper)
    alloc.driver.add_parameter('flt_day', low=0, high=10)
    alloc.driver.add_constraint('0.1*demand < pax_con < demand')
    alloc.driver.add_constraint('0.0 < ac_con < ac_avail')
    for irt in xrange(alloc.num_routes):
        for inac in xrange(alloc.num_new_ac):
            alloc.driver.add_parameter(seg_name+'.h_pt_%03i_%03i'%(irt,inac), low=0.0, high=14.1)
            alloc.driver.add_constraint(seg_name+'.h_%03i_%03i[0] = 0.0'%(irt,inac))
            alloc.driver.add_constraint(seg_name+'.h_%03i_%03i[-1] = 0.0'%(irt,inac))
            alloc.driver.add_constraint(seg_name+'.Tmin_%03i_%03i < 0.0'%(irt,inac))
            alloc.driver.add_constraint(seg_name+'.Tmax_%03i_%03i < 0.0'%(irt,inac))
            alloc.driver.add_constraint('%.15f < '+seg_name+'.Gamma_%03i_%03i < %.15f'%
                                        (alloc.gamma_lb,irt,inac,alloc.gamma_ub), linear=True)

    alloc.run()
