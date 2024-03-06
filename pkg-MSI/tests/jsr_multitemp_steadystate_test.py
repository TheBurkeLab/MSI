import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.jsr_steadystate as jsr
import MSI.cti_core.cti_processor as pr
import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

#test_p = pr.Processor('C:\\Users\\HP USER\\Google Drive\\Burke #Group\\Codes\\Mechanisms\\heptane_lowT\\Mech.cti')
data='C:\\Users\\HP USER\\Google Drive\\Burke Group\\JSR Experiments\\LeClercMethane\\oxygen_profile.csv'
data2='C:\\Users\\HP USER\\Google Drive\\Burke Group\\JSR Experiments\\LeClercMethane\\data_08082019.csv'


test_p=pr.Processor('C:\\Users\\HP USER\\Google Drive\\Burke Group\\Codes\\Mechanisms\\CH4_DME\\chem_760_762_763.cti')
jsr1 = jsr.JSR_multiTemp_steadystate(volume=8.5e-5,pressure=1.0520602,
                         temperatures=np.linspace(850,1250,100),
                         observables=['OH','H2O'],
                         kineticSens=1,
						 physicalSens=0,
                         conditions={'CH4':0.063,'O2':0.063,'He':0.874},
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_physSensHistories=0,save_timeHistories=1,residence_time=1.0,
						 rtol=1e-8,atol=1e-7)
						 
solution,ksens=jsr1.run()
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
#print(ksens)


with open(data,'r') as f:
	temp=f.readlines()
	for i in range(len(temp)):
		temp[i]=temp[i].rstrip(',\n')+'\n'
with open(data,'w') as f:
	f.writelines(temp)
import pandas as pd

m=pd.read_csv(data,delimiter=',',header=3)
measT=m['T']

#measT=np.array(measT)+273.15

m2=pd.read_csv(data2,delimiter=',')
species_to_plot=['o2','co','co2','c2h4','c2h6']
species_to_plot=['o2']
for i in species_to_plot:
	plt.figure()
	plt.plot(solution['temperature'],solution[i],'r')
	measured=m['X']
	plt.plot(measT,measured,'ko')
	plt.plot(m2['Temp'],m2[i]/100.,'bo')
	plt.savefig('C:\\Users\\HP USER\\Google Drive\\Burke Group\\JSR Experiments\\LeClercMethane\\'+i+'_leclerc_15p2_08212019_1p0s_He.pdf',dpi=1200,bbox_inches='tight')
	#plt.savefig('C:\\Users\\HP USER\\Google Drive\\Burke Group\\JSR Experiments\\nheptane\\'+i+'_zhang.pdf',dpi=1200,bbox_inches='tight')
#jsr1.sensitivity_adjustment(temp_del=0.01)
#jsr1.sensitivity_adjustment(pres_del=0.01)
#jsr1.species_adjustment(spec_del=0.01)
#print(jsr1.timeHistories)
#print(len(jsr1.timeHistories))	
#print(measured)			 
#jsr1.set_geometry(volume=9.19523225755e-5)
#a,b=jsr1.run()
#print(a)
#test_tube.printVars()
#time_History = test_tube.timeHistory


#plt.plot(time_History['time']*1e3,time_History['OH']*1e6)
#time_OH = time_History['time']
#OH_ppm = time_History['OH']*1e6
#plt.figure()
#plt.plot(time_History['time']*1e3,time_History['H2O']*1e6)
#time_H2O = time_History['time']
#H2O_ppm = time_History['H2O']*1e6
