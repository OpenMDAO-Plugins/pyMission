# Just plotting a few things for a single optimization case

from matplotlib import pyplot as plt
from openmdao.lib.casehandlers.api import CaseDataset

filename = 'mission_cp_10.bson'

case_dataset = CaseDataset(filename, 'bson')
data = case_dataset.data.by_case().fetch()

obj = [case['SysFuelObj.wf_obj'] for case in data]
X = range(0, len(obj))

plt.figure()

plt.plot(X, obj)
plt.xlabel("Iteration number")
plt.ylabel("Objective")

plt.show()