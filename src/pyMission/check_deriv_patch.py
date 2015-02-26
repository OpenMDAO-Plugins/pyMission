import numpy as np

from openmdao.main.api import Driver, Assembly, set_as_top


# monkey patch for assembly
def check_comp_derivatives(self): 

    for comp_name in self.list_components(): 

        comp = self.get(comp_name)
        comp.eval_only = True # needed so we can get derivs of resids for implicit comps

        if isinstance(comp, Driver): 
            continue

        pound_line = 123*'-'
        #print pound_line
        #print comp_name
        #print pound_line

        invars, outvars = comp.list_deriv_vars()

        # for i in invars: 
        #     print comp.get_metadata(i)['iotype']

        ins = ['%s.%s' % (comp_name, invar) for invar in invars]    

        # for invar_name in ins: 
        #     print invar_name, self.get(invar_name)    

        outs = ['%s.%s' % (comp_name, outvar) for outvar in outvars]

        # create a new assembly, which I don't think I should have to do... 
        clean_asmb = set_as_top(Assembly())
        clean_asmb.add(comp_name, comp)
        clean_asmb.driver.workflow.add(comp_name)
        clean_asmb._setup()

        J_fwd = clean_asmb.driver.calc_gradient(ins, outs, mode="forward", return_format="dict")
        J_rev = clean_asmb.driver.calc_gradient(ins, outs, mode="adjoint", return_format="dict")

        step_sizes = [-1,-3,-5,-7,-9]
        J_fd = []
        for ss in step_sizes: 
            step = 10**ss

            clean_asmb.driver.gradient_options.fd_step = step
            j = clean_asmb.driver.calc_gradient(ins, outs, mode="fd", return_format="dict")
            J_fd.append(j)

        # clean_asmb.driver.gradient_options.fd_form = "forward"
        # J_fd = clean_asmb.driver.calc_gradient(ins, outs, mode="fd", return_format="dict")

        res_line = "%-24s  | %-24s | %-18s | %-18s | %-18s |" 

#        print res_line % ('output', 'input', 'fwd-fd', 'rev-fd', 'fwd-rev')
#        print pound_line

        for outvar_name in J_fwd:
            for invar_name in J_fwd[outvar_name]:
                fwd_fd_lowest = 1e90
                rev_fd_lowest = 1e90

                for j in J_fd: 
                    fwd_fd = np.linalg.norm(J_fwd[outvar_name][invar_name] - j[outvar_name][invar_name])
                    rev_fd = np.linalg.norm(J_rev[outvar_name][invar_name] - j[outvar_name][invar_name])

                    fwd_fd_lowest = min(fwd_fd, fwd_fd_lowest)
                    rev_fd_lowest = min(rev_fd, rev_fd_lowest)
                
                fwd_rev = np.linalg.norm(J_fwd[outvar_name][invar_name] - J_rev[outvar_name][invar_name])

                print res_line % (outvar_name, invar_name, fwd_fd_lowest, rev_fd_lowest, fwd_rev)
        
#        print '\n'
#        print pound_line, "\n\n\n"


Assembly.check_comp_derivatives = check_comp_derivatives
