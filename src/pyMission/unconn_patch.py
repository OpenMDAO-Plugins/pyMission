import numpy as np

from openmdao.main.api import Driver, Assembly, set_as_top

def list_unconnected_inputs(self): 

    dg = self._depgraph

    for comp_name in self.list_components(): 

        comp = self.get(comp_name)
        
        if isinstance(comp, Driver): 
            continue

        unconns = dg.list_inputs(comp_name, connected=False)

        vars = []
        for vname in unconns: 
            is_fv = self.get_metadata(vname).get('framework_var', False)
            if not is_fv: 
                vars.append(vname)

        print comp_name, vars


Assembly.list_unconnected_inputs = list_unconnected_inputs