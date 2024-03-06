import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.utilities.uncertainty_merge as um
import cantera as ct


test_p = pr.Processor('MSI/data/test_data/glarborg_custom.cti')
reaction_equations = test_p.solution.reaction_equations()
#df = um.get_uncertainties(reaction_equations,uncert_file='MSI/data/test_data/uncertainty.csv')
df = um.get_uncertainties(reaction_equations,uncert_file='MSI/data/test_data/FFCM1_reaction_uncertainty_FFCM_values.csv')
#test_tube = st.shockTube(pressure=1.355,
#                         temperature=1590,
#                         observables=['OH'],
#                         kineticSens=0,
#                         physicalSens=0,
#                         conditions={'H2O':.001234,'H2O2':.00254,'O2':.00062,'Ar':.995606},
#                         initialTime=0,
#                         finalTime=0.0025,
#                         thermalBoundary='Adiabatic',
#                         mechanicalBoundary='constant pressure',
#                         processor=test_p,
#                         save_timeHistories=1,
#                         save_physSensHistories=1)
