import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import cantera as ct
import pandas
test_p = pr.Processor('MSI/data/test_data/FFCM1.cti')
test_tube = st.shockTube(pressure=1.74,
                         temperature=1880,
                         observables=['OH','H2O'],
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'H2O':.013,'O2':.0099,'H':.0000007,'Ar':0.9770993},
                         initialTime=0,
                         finalTime=0.5,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1)

csv_paths = ['MSI/data/test_data/hong_oh_4.csv','MSI/data/test_data/hong_h2o_4.csv',]
exp_data = test_tube.importExperimentalData(csv_paths)

test_tube.run() #set up original time history
int_ksens_exp_mapped= test_tube.map_and_interp_ksens()
test_tube.sensitivity_adjustment(temp_del = .01)
test_tube.sensitivity_adjustment(pres_del = .01)
test_tube.species_adjustment(.01) #do some sensitivity adjustments

int_tp_psen_against_experimental = test_tube.interpolate_experimental([test_tube.interpolate_physical_sensitivities(index=1),
                                                                    test_tube.interpolate_physical_sensitivities(index=2)])
int_spec_psen_against_experimental = test_tube.interpolate_experimental(pre_interpolated=test_tube.interpolate_species_sensitivities())



#print(int_tp_psen_against_experimental)
#print(int_spec_psen_against_experimental)
