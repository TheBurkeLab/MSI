import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import cantera as ct

# test_p = pr.Processor('MSI/data/test_data/FFCM1.cti')
# test_tube = st.shockTube(pressure=1.74,
#                          temperature=1880,
#                          observables=['OH','H2O'],
#                          kineticSens=1,
#                          physicalSens=0,
#                          conditions={'H2O':.013,'O2':.0099,'H':.0000007,'Ar':0.9770993},
#                          initialTime=0,
#                          finalTime=0.1,
#                          thermalBoundary='Adiabatic',
#                          mechanicalBoundary='constant pressure',
#                          processor=test_p)




test_p = pr.Processor('/Users/carlylagrotta/Dropbox/Columbia/MSI/data/Burke_atmospheric_chemistry/burke_atmospheric_chemistry_model.cti')
#test_p2 = pr.Processor('/Users/carlylagrotta/Dropbox/Columbia/MSI/data/branching_reaction_study/ffcm1_branching_reactions_theory.cti')


test_tube = st.shockTube(pressure=1 ,
                         temperature=298, 
                         observables=['H2CO'],
                         kineticSens=1,
                         physicalSens=1,
                         conditions={'CH2O': .00131,
                                     'Pin': 0.01,
                                     'O2':.209,
                                     'N2':.791},
                         initialTime=0,
                         finalTime=300,
                         thermalBoundary='isothermal',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1
                         )
test_tube.run()
test_tube.printVars()
print(test_tube.timeHistory)
print(test_tube.kineticSensitivities)
data = test_tube.sensitivity_adjustment(temp_del = .01)
print("TEMPERATURE TEST:\n",data)
print(test_tube.temperature)
data = test_tube.species_adjustment(spec_del = .01)
print("SPECIES TEST:\n",data)
print(test_tube.conditions)
