"""
Framework interface to pyOptSparse
John Hwang, March 2014
"""

from __future__ import division
from pyoptsparse import Optimization as OptProblem
from pyoptsparse import OPT as Optimizer
import numpy


class Optimization(object):
    """ Automatically sets up and runs an optimization """

    def __init__(self, system):
        """ Takes system containing all DVs and outputs """
        self._system = system
        self._variables = {'dv': {}, 'func': {}}

    def _get_name(self, var_id):
        """ Returns unique string for the variable """
        return var_id[0] + '_' + str(var_id[1])

    def _add_var(self, typ, var, value=0.0, lower=None, upper=None):
        """ Wrapped by next three methods """
        var_id = self._system.get_id(var)
        var_name = self._get_name(var_id)
        self._variables[typ][var_name] = {'ID': var_id,
                                          'value': value,
                                          'lower': lower,
                                          'upper': upper}

    def add_design_variable(self, var, value=0.0, lower=None, upper=None):
        """ Self-explanatory; part of API """
        self._add_var('dv', var, value=value, lower=lower, upper=upper)

    def add_objective(self, var):
        """ Self-explanatory; part of API """
        self._add_var('func', var)

    def add_constraint(self, var, lower=None, upper=None):
        """ Self-explanatory; part of API """
        self._add_var('func', var, lower=lower, upper=upper)

    def obj_func(self, dv_dict):
        """ Objective function passed to pyOptSparse """
        system = self._system
        variables = self._variables

        for dv_name in variables['dv'].keys():
            dv_id = variables['dv'][dv_name]['ID']
            system(dv_id).value = dv_dict[dv_name]

        print '********************'
        print 'Evaluating functions'
        print '********************'
        print

        temp, success = system.compute(True)
        fail = not success

        print 'DVs:'
        print dv_dict
        print 'Failure:', fail
        print
        print '-------------------------'
        print 'Done evaluating functions'
        print '-------------------------'
        print

        func_dict = {}
        for func_name in variables['func'].keys():
            func_id = variables['func'][func_name]['ID']
            func_dict[func_name] = system.vec['u'][func_id]

        if fail:
            system.vec['u'].array[:] = 1.0
            system.vec['du'].array[:] = 0.0
            for var in system.variables:
                system.vec['u'][var][:] = \
                    system.variables[var]['u'] /\
                    system.variables[var]['u0']

        return func_dict, fail

    def sens_func(self, dv_dict, func_dict):
        """ Derivatives function passed to pyOptSparse """
        system = self._system
        variables = self._variables

        print '**********************'
        print 'Evaluating derivatives'
        print '**********************'
        print 

        fail = False
        sens_dict = {}
        for func_name in variables['func'].keys():
            func_id = variables['func'][func_name]['ID']
            nfunc = system.vec['u'][func_id].shape[0]

            sens_dict[func_name] = {}
            for dv_name in variables['dv'].keys():
                dv_id = variables['dv'][dv_name]['ID']
                ndv = system.vec['u'][dv_id].shape[0]

                sens_dict[func_name][dv_name] \
                    = numpy.zeros((nfunc, ndv))

            for ind in xrange(nfunc):
                temp, success = system.compute_derivatives('rev', func_id, ind, False)#True)#False)
                fail = fail or not success

                for dv_name in variables['dv'].keys():
                    dv_id = variables['dv'][dv_name]['ID']

                    sens_dict[func_name][dv_name][ind, :] \
                        = system.vec['df'][dv_id]

        print 'DVs:'
        print dv_dict
        print 'Functions:'
        print func_dict
        print 'Derivatives:'
        print sens_dict
        print 'Failure:', fail
        print
        print '---------------------------'
        print 'Done evaluating derivatives'
        print '---------------------------'
        print

        if fail:
            system.vec['du'].array[:] = 0.0

        return sens_dict, fail

    def __call__(self, optimizer, options=None):
        """ Run optimization """
        system = self._system
        variables = self._variables

        opt_prob = OptProblem('Optimization', self.obj_func)
        for dv_name in variables['dv'].keys():
            dv_id = variables['dv'][dv_name]['ID']
            value = variables['dv'][dv_name]['value']
            lower = variables['dv'][dv_name]['lower']
            upper = variables['dv'][dv_name]['upper']
            size = system.vec['u'](dv_id).shape[0]
            opt_prob.addVarGroup(dv_name, size, value=value,
                                 lower=lower, upper=upper)
        opt_prob.finalizeDesignVariables()
        for func_name in variables['func'].keys():
            func_id = variables['func'][func_name]['ID']
            lower = variables['func'][func_name]['lower']
            upper = variables['func'][func_name]['upper']
            size = system.vec['u'](func_id).shape[0]
            if lower is None and upper is None:
                opt_prob.addObj(func_name)
            else:
                opt_prob.addConGroup(func_name, size,
                                     lower=lower, upper=upper)

        if options is None:
            options = {}

        opt = Optimizer(optimizer, options=options)
        sol = opt(opt_prob, sens=self.sens_func)
        print sol
