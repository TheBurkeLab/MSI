import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

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

csv_paths = ['MSI/data/test_data/hong_oh_4.csv','MSI/data/test_data/hong_h2o_4.csv']
exp_data = test_tube.importExperimentalData(csv_paths)

test_tube.run()
test_tube.species_adjustment(.01)
spec_data = test_tube.interpolate_species_adjustment()
interpolated_time_history= test_tube.interpolate_experimental(pre_interpolated=spec_data)

#print(interpolated_time_history)
single_data = test_tube.interpolate_experimental(single=test_tube.timeHistories[0])
#print(single_data)

