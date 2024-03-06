import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built
import numpy as np
import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import cantera as ct


test_p2 = pr.Processor('MSI/data/chebyshev_data/FFCM1_custom_cheb_test.cti')

test_tube = st.shockTube(pressure=1.909,
                         temperature=1398,
                         observables=['OH','H2O'],
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'O2':0.00062 ,
                                     'H2O':0.001234,
                                     'H2O2':0.00254 ,
                                     'Ar': 0.9974968858350951},
                         initialTime=0,
                         finalTime=.0002,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant volume',
                         processor=test_p2,
                         save_timeHistories=1,
                         save_physSensHistories=1)



csv_paths = ['MSI/data/test_data/hong_oh_0_time_dependent.csv', 'MSI/data/test_data/hong_h2o_0_time_dependent.csv']
exp_data = test_tube.importExperimentalData(csv_paths)
test_tube.run()
data = test_tube.interpolate_experimental_kinetic()
int_ksens_exp_mapped= test_tube.map_and_interp_ksens()
#print(data)


