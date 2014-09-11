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
        self.sens_callback = None

    def _get_name(self, var_id):
        """ Returns unique string for the variable """
        return var_id[0] + '_' + str(var_id[1])

    def _add_var(self, typ, var, value=0.0,
                 lower=None, upper=None,
                 get_jacs=None, linear=False):
        """ Wrapped by next three methods """
        var_id = self._system.get_id(var)
        var_name = self._get_name(var_id)
        self._variables[typ][var_name] = {'ID': var_id,
                                          'value': value,
                                          'lower': lower,
                                          'upper': upper,
                                          'get_jacs': get_jacs,
                                          'linear': linear}

    def add_design_variable(self, var, value=0.0, 
                            lower=None, upper=None):
        """ Self-explanatory; part of API """
        self._add_var('dv', var, value=value,
                      lower=lower, upper=upper)

    def add_objective(self, var):
        """ Self-explanatory; part of API """
        self._add_var('func', var)

    def add_constraint(self, var, lower=None, upper=None,
                       get_jacs=None, linear=False):
        """ Self-explanatory; part of API """
        self._add_var('func', var, lower=lower, upper=upper,
                      get_jacs=get_jacs, linear=linear)

    def add_sens_callback(self, callback):
        self.sens_callback = callback

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
            func = variables['func'][func_name]
            func_id = func['ID']
            get_jacs = func['get_jacs']
            nfunc = system.vec['u'][func_id].shape[0]

            sens_dict[func_name] = {}
            if func['get_jacs'] is None:
                for dv_name in variables['dv'].keys():
                    dv_id = variables['dv'][dv_name]['ID']
                    ndv = system.vec['u'][dv_id].shape[0]

                    sens_dict[func_name][dv_name] \
                        = numpy.zeros((nfunc, ndv))

                for ind in xrange(nfunc):
                    temp, success = system.compute_derivatives('rev', func_id, ind, False)
                    fail = fail or not success

                    for dv_name in variables['dv'].keys():
                        dv_id = variables['dv'][dv_name]['ID']

                        sens_dict[func_name][dv_name][ind, :] \
                            = system.vec['df'][dv_id]
            else:
                jacs = get_jacs()
                for dv_var in jacs:
                    dv_id = self._system.get_id(dv_var)
                    dv_name = self._get_name(dv_id)
                    sens_dict[func_name][dv_name] \
                        = jacs[dv_var]

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

        if self.sens_callback is not None:
            self.sens_callback()

        return sens_dict, fail

    def __call__(self, optimizer, options=None):
        """ Run optimization """
        system = self._system
        variables = self._variables

        opt_prob = OptProblem('Optimization', self.obj_func)
        for dv_name in variables['dv'].keys():
            dv = variables['dv'][dv_name]
            dv_id = dv['ID']
            value = dv['value']
            lower = dv['lower']
            upper = dv['upper']
            size = system.vec['u'](dv_id).shape[0]
            opt_prob.addVarGroup(dv_name, size, value=value,
                                 lower=lower, upper=upper)
        opt_prob.finalizeDesignVariables()
        for func_name in variables['func'].keys():
            func = variables['func'][func_name]
            func_id = func['ID']
            lower = func['lower']
            upper = func['upper']
            linear = func['linear']
            get_jacs = func['get_jacs']
            size = system.vec['u'](func_id).shape[0]
            if lower is None and upper is None:
                opt_prob.addObj(func_name)
            else:
                if func['get_jacs'] is None:
                    opt_prob.addConGroup(func_name, size,
                                         lower=lower, upper=upper)
                else:
                    jacs_var = get_jacs()

                    dv_names = []
                    jacs = {}
                    for dv_var in jacs_var:
                        dv_id = self._system.get_id(dv_var)
                        dv_name = self._get_name(dv_id)
                        dv_names.append(dv_name)
                        jacs[dv_name] = jacs_var[dv_var]

                    opt_prob.addConGroup(func_name, size,
                                         wrt=dv_names,
                                         jac=jacs, linear=linear,
                                         lower=lower, upper=upper)

        if options is None:
            options = {}

        opt = Optimizer(optimizer, options=options)
        opt.setOption('Iterations limit', int(1e6))
        #opt.setOption('Verify level', 3)
        sol = opt(opt_prob, sens=self.sens_func, storeHistory='hist.hst')
        print sol
