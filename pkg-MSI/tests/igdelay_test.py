import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built
import os
os.chdir('C:\\Users\\HP USER\\Desktop')
#import MSI.simulations.instruments.flames as f
import MSI.cti_core.cti_processor as pr
import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import MSI.simulations.instruments.ignition_delay as ig

#test_p = pr.Processor('C:\\Users\\HP USER\\Google Drive\\Burke #Group\\Codes\\Mechanisms\\heptane_lowT\\Mech.cti')


test_p=pr.Processor('C:\\Users\\HP USER\\Desktop\\MSI\\data\\rodger_plotting_error_testing\\Glarborg_HNO_updated.cti')
s = ig.ignition_delay_wrapper(pressures=[1.422800621],
                         temperatures=[1970.89,1982.38,2060.95,2078.74,2162.57,2224.36,2262.05,2327.96,2353.06,2391.77],
                         observables=['OH'],
                         kineticSens=0,
						 physicalSens=0,
                         conditions=[{'NH3':0.007273181,'O2':0.002727294,'Ar':0.987550463}],
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         finalTime=0.1,
                         target='OH*',
                         target_type='max derivative tangent intercept',
                         save_physSensHistories=0,save_timeHistories=1,n_processors=2,
                         fullParsedYamlFile={'simulationType':'Shock Tube'})		

			 
solution,sens=s.run()
print(solution,sens)
print(np.shape(sens))
plt.figure()
plt.semilogy(1000/np.array(solution['temperature']),solution['delay'],'kx')
plt.savefig('C:\\Users\\HP USER\\Desktop\\MSI\\data\\rodger_plotting_error_testing\\plot_result.jpg',bbox_inches='tight',dpi=1200)
#methane_profile=[]
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



