import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import MSI.cti_core.soln2cti as cti_pp

import cantera as ct
gas = ct.Solution("../data/test_data/lam.cti")

cti_pp.write(gas, "../data/test_data/cti_test.cti")
