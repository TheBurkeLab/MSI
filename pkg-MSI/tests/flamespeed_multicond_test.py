import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built
import os
os.chdir('C:\\Users\\Skoron\\Desktop')
import MSI.simulations.instruments.flames as f
import MSI.cti_core.cti_processor as pr
import cantera as ct
import matplotlib.pyplot as plt
import numpy as np



test_p=pr.Processor('C:\\Users\\Skoron\\Google Drive\\Burke Group\\Codes\\Mechanisms\\FFCM-1\\FFCM1.cti')
f1 = f.flamespeed_multi_condition(pressures=[1.00],
                         temperatures=[298.0],
                         kineticSens=1,
						 physicalSens=0,
                         conditions=[{'H2':0.5,'O2':0.5,'He':4.0},
                                     {'H2':0.75,'O2':0.5,'He':4.0},
                                     {'H2':1.0,'O2':0.5,'He':4.0},
                                     {'H2':1.25,'O2':0.5,'He':4.0},
                                     {'H2':1.5,'O2':0.5,'He':4.0},
                                     {'H2':1.75,'O2':0.5,'He':4.0},
                                     {'H2':2.0,'O2':0.5,'He':4.0},
                                     {'H2':2.25,'O2':0.5,'He':4.0},
                                     {'H2':2.5,'O2':0.5,'He':4.0}],
                         thermalBoundary='Adiabatic'                         ,
                         processor=test_p,
                         save_physSensHistories=0,save_timeHistories=1,loglevel=0)						 
solution,ksens=f1.run()
methane_profile=[]
#for i in jsr1.JSR_objects:
#	print(i.pressure,i.temperature,i.conditions)
#	print(i.solution['ch4'],i.reactorPressure)
#for i in range(len(jsr1.JSR_objects)):
#	methane_profile.append(jsr1.JSR_objects[i].solution['ch4'])

	
#plt.plot(np.linspace(858,1258,25),methane_profile)
#plt.savefig('C:\\Users\\HP USER\\Google Drive\\Burke Group\\Mark\\MSI\\data\\jsr_test\\methane.pdf',
#				dpi=1200, bbox_inches='tight')
#print(solution['ch4'])
#print(ksen
plt.figure()
plt.plot(solution['H2'],solution['u0'],'k-')
plt.xlabel('Equivalence Ratio')
plt.ylabel('Flame Speed (m/s)')
plt.title('H2/O2/He Flame speed, O2/(O2+He)=0.125')


