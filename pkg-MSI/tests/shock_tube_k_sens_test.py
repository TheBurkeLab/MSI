import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')
import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import cantera as ct

#test_p = pr.Processor('MSI/data/test_data/FFCM1_custom.cti')
#test_p = pr.Processor('MSI/data/chebyshev_data/FFCM1_custom_cheb.cti')
#test_p = pr.Processor('MSI/data/chebyshev_data/FFCM1_custom_cheb_test.cti')
test_p = pr.Processor('MSI/data/flow_reactor/FFCM1_customA1.cti')
reaction_equations = test_p.solution.reaction_equations()

test_tube = st.shockTube(pressure=1.909,
                         temperature=1398,
                         observables=['H2O'],
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'O2':0.00062 ,
                                     'H2O':0.001234,
                                     'H2O2':0.00254 ,
                                     'Ar': 0.9974968858350951},
                         initialTime=0,
                         finalTime=.001,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1)

test_tube.run()

ksens = test_tube.kineticSensitivities
timeHist=test_tube.timeHistories[0]