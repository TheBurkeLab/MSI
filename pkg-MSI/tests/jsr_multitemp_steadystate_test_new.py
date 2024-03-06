import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import MSI.simulations.instruments.jsr_steadystate as jsr
import MSI.cti_core.cti_processor as pr
import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
#test_p = pr.Processor('C:\\Users\\HP USER\\Google Drive\\Burke #Group\\Codes\\Mechanisms\\heptane_lowT\\Mech.cti')
#data='C:\\Users\\HP USER\\Google Drive\\Burke Group\\JSR Experiments\\RodgerSymposium\\exp58.csv'
#data2='C:\\Users\\HP USER\\Google Drive\\Burke Group\\JSR Experiments\\LeClercMethane\\data_08082019.csv'
#_760_762_763

#outfile='C:\\Users\\HP USER\\Google Drive\\Burke Group\\JSR Experiments\\N2O_AR\\glar_comp_lam_2027.csv'

#outfile2='C:\\Users\\HP USER\\Google Drive\\Burke Group\\JSR Experiments\\N2O_AR\\glar_comp_lam_2027_ksens.csv'
test_p=pr.Processor('C:\\Users\\HP USER\\Google Drive\\Burke Group\\Codes\\Mechanisms\\Lamoreaux\\lam.cti')
jsr1 = jsr.JSR_multiTemp_steadystate(volume=8.2e-5,pressure=1.02069,
                         temperatures=[1050],
                         observables=['N2'],
                         kineticSens=1,
						 physicalSens=0,
                         conditions={'N2O':0.045,'Ar':0.955},
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_physSensHistories=0,save_timeHistories=1,residence_time=1.2,
						 rtol=1e-14,atol=1e-15)

tic=time.time()						 
solution,ksens=jsr1.run()
abs_ksens=np.abs(ksens)

sens_data=pd.DataFrame(columns=['Reactions','ksens','abs'])
sens_data['ksens']=ksens.flatten()
sens_data['abs']=abs_ksens.flatten()
sens_data['Reactions']=test_p.solution.reaction_equations()
sens_data.sort_values(by='abs',ascending=False,inplace=True)
# tempsens=ksens[10,:,:].flatten()
# sens=pd.DataFrame(columns=['Rxn','NO'])
# sens['NO']=tempsens
# sens['Rxn']=test_p.solution.reaction_equations()
# sens['abs']=np.abs(sens['NO'])
# sens=sens.sort_values(by=['abs'],ascending=False)
toc=time.time()
print('Total Simulation time was {:3.2f}s to compute'.format(toc-tic))
methane_profile=[]
#print(ksens, np.shape(ksens))
#ksens_data=pd.DataFrame(columns=['Rxn','Sens','Abs'])
#ksens_data['Sens']=ksens.flatten()
#ksens_data['Abs']=np.abs(ksens.flatten())
#ksens_data['Rxn']=test_p.solution.reaction_equations()
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


#with open(data,'r') as f:
#	temp=f.readlines()
#	for i in range(len(temp)):
#		temp[i]=temp[i].rstrip(',\n')+'\n'
#with open(data,'w') as f:
#	f.writelines(temp)
import pandas as pd
#solution.to_csv('C:\\Users\\Skoron\\Google Drive\\Burke Group\\JSR Experiments\\N2ODecomp\\glarborg_updated.csv')

#m=pd.read_csv(data,delimiter=',')
#measT=m['T']

#measT=np.array(measT)+273.15

#m2=pd.read_csv(data)
#species_to_plot=['o2','co','co2','c2h4','c2h6']
species_to_plot=['N2O','N2','NO']
#solution.to_csv(outfile,index=False)
#ksens_data.to_csv(outfile2,index=False)
axis_font = {'fontname':'Arial', 'size':'14'}

#for i in species_to_plot:
#	plt.figure()
#	labels=['Glarborg','RDX_Caltech_GlarborgTRall','RDX_Caltech_GlarborgTR','RDX_Caltech']
#	plt.plot(solution['temperature'],solution[i],'r',label=labels[0])
#	plt.plot(solution2['temperature'],solution2[i],'b',label=labels[1])
#	plt.plot(solution3['temperature'],solution3[i],'g',label=labels[2])
#	plt.plot(solution4['temperature'],solution4[i],'k',label=labels[3])
#	plt.legend(loc=3)
#	measured=np.array(m[i])/100.0
#	plt.plot(measT,measured,'ko')
#	plt.xlabel('Temperature (K)',**axis_font)
#	plt.ylabel('Mole Fraction',**axis_font)
#	plt.title(i+' temperature profile')
#	#plt.plot(m2['Temp'],m2[i]/100.,'bo')
#	plt.savefig('C:\\Users\\HP USER\\Google Drive\\Burke Group\\JSR Experiments\\RodgerSymposium\\'+i+'_exp58.pdf',dpi=1200,bbox_inches='tight')
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
