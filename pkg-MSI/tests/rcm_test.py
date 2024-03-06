import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import MSI.simulations.instruments.RCM as rcm
import MSI.cti_core.cti_processor as pr
import cantera as ct
import matplotlib.pyplot as plt

test_p = pr.Processor('MSI/data/RCM/mech.cti')
#test_p = pr.Processor('MSI/data/chebyshev_data/FFCM1_custom_cheb.cti')
#test_p = pr.Processor('MSI/data/chebyshev_data/FFCM1_custom_cheb_test.cti')

test_tube = rcm.RCM(pressure=1.4939057488280285,
                         temperature=323, 
                         observables=['H2O'],
                         kineticSens=0,
                         physicalSens=0,
                         conditions={'C3H8': 0.024950099800399205,
                                     'CH3OCH3': 0.024950099800399205,
                                     'N2': 0.7504990019960079,
                                     'O2': 0.1996007984031936},
                         initialTime=0,
                         finalTime=.1,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant volume',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1,
                         volumeTrace='MSI/data/RCM/volume_trace_file.csv',
                         exactDerivFlag=True)

test_tube.run()
#test_tube.printVars()
time_History = test_tube.timeHistory
plt.figure()
plt.plot(time_History['time'],time_History['pressure']/100000)
plt.xlabel('Time')
plt.ylabel('Presure')
plt.ylim(0,140)
plt.figure()
plt.plot(time_History['time'],time_History['temperature'])
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.figure()
plt.plot(time_History['time'],time_History['volume'])
plt.xlabel('Time')
plt.ylabel('Volume')