import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.jsr_steadystate as jsr
import MSI.cti_core.cti_processor as pr
import cantera as ct
import matplotlib.pyplot as plt

test_p = pr.Processor('C:\\Users\\HP USER\\Google Drive\\Burke Group\\Codes\\Mechanisms\\grimech3\\gri30.cti')
jsr = jsr.JSR_steadystate(pressure=1.0,
                         temperature=1283,
                         observables=['OH','H2O'],
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'H2O2':0.003094,'H2O':0.001113,'O2':0.000556,'Ar':0.995237},
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_physSensHistories=1)
jsr.set_geometry(volume=9.19523225755e-5)
a,b=jsr.run()
print(a)
#test_tube.printVars()
#time_History = test_tube.timeHistory


#plt.plot(time_History['time']*1e3,time_History['OH']*1e6)
#time_OH = time_History['time']
#OH_ppm = time_History['OH']*1e6
#plt.figure()
#plt.plot(time_History['time']*1e3,time_History['H2O']*1e6)
#time_H2O = time_History['time']
#H2O_ppm = time_History['H2O']*1e6
