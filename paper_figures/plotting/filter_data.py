from openmdao.lib.casehandlers.api import CaseDataset

cds = CaseDataset('cp_sweep_data/mission_history_cp_350.bson','bson')

var_names = ['SysXBspline.x','SysHBspline.h','SysRho.rho','SysTemp.temp','SysSpeed.v','SysTripanCLSurrogate.alpha','SysTau.tau','SysTripanCMSurrogate.eta','SysFuelWeight.fuel_w','SysCTTar.CT_tar','SysCLTar.CL','SysTripanCDSurrogate.CD','SysGammaBspline.Gamma']

cds.data.driver('driver').vars(var_names).write('cp_sweep_data/mission_history_cp_350_smaller.bson')
