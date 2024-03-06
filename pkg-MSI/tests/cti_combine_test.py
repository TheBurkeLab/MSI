import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.optimization.matrix_loader as ml
import MSI.optimization.opt_runner as opt
import MSI.simulations.absorbance.curve_superimpose as csp
import MSI.simulations.yaml_parser as yp
import MSI.cti_core.cti_combine as ctic
import cantera as ct

ctiFile = 'MSI/data/test_data/FFCM1.cti'
test_p = pr.Processor('MSI/data/test_data/FFCM1.cti')
ctiFile_Name = 'FFCM1'
new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(original_cti=ctiFile,working_directory='MSI/data/test_data',file_name='FFCM1_updated_file')
