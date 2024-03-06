import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')#get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.simulations.absorbance.curve_superimpose as csp  
import MSI.simulations.yaml_parser as yp
import cantera as ct
import pandas as pd 
import matplotlib.pyplot as plt

value = 15

#MSI YAML FILES AND CONDITIONS
parser = yp.Parser()      
            
exp = parser.load_to_obj('MSI/data/klip_optimization_with_raw_data/Hong_HO2_fake_data_'+str(value)+'_updated.yaml')
absp = parser.load_to_obj('MSI/data/klip_optimization_with_raw_data/Hong_HO2_fake_data_'+str(value)+'_abs_updated.yaml')
loaded_tube = parser.parse_shock_tube_obj(loaded_exp=exp, loaded_absorption=absp)

pressure=loaded_tube['pressure']
temperature=loaded_tube['temperature']
conditions = loaded_tube['conditions']
print("THESE ARE THE MSI CONDITIONS")
print(pressure)
print(temperature)
print(conditions)
#MSI YAML FILES AND CONDITIONS


#HONG YAML FILES AND CONDITIONS
parser2 = yp.Parser()                  
exp2 = parser2.load_to_obj('MSI/data/klip_optimization_with_raw_data/Hong_HO2_fake_data_'+str(value)+'.yaml')
absp2 = parser2.load_to_obj('MSI/data/klip_optimization_with_raw_data/Hong_HO2_fake_data_'+str(value)+'_abs.yaml')
loaded_tube2 = parser2.parse_shock_tube_obj(loaded_exp=exp2, loaded_absorption=absp2)

pressure2=loaded_tube2['pressure']
temperature2=loaded_tube2['temperature']
conditions2 = loaded_tube2['conditions']
print("THESE ARE THE HONG CONDITIONS")
print(pressure2)
print(temperature2)
print(conditions2)


test_p = pr.Processor('MSI/data/klip_optimization_with_raw_data/FFCM1_custom_extra_reaction_updated.cti')


#HONG YAML FILES AND CONDITIONS


test_tube = st.shockTube(pressure=pressure,
                         temperature=temperature,
                         observables=['OH','H2O','H2O2'],
                         kineticSens=0,
                         physicalSens=0,
                         conditions=conditions,
                         initialTime=0,
                         finalTime=0.0035,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1)

test_p2 = pr.Processor('MSI/data/klip_optimization_comparison/Hong_new_temp_dependent_reactions.cti')
#test_p2 = pr.Processor('MSI/data/hong_H2O2_fake_data/Hong_new_original_reactions.cti')

test_tube2 = st.shockTube(pressure=pressure2,
                         temperature=temperature2,
                         observables=['OH','H2O'],
                         kineticSens=0,
                         physicalSens=0,
                         conditions=conditions2,
                         initialTime=0,
                         finalTime=0.0035,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p2,
                         save_timeHistories=1,
                         save_physSensHistories=1)




OH = pd.read_csv('MSI/data/klip_optimization_with_raw_data/hong_oh_fake_data_'+str(value)+'.csv')
OH_time = OH['Time']
OH_ppm = OH['OH_ppm']



H2O = pd.read_csv('MSI/data/klip_optimization_with_raw_data/hong_h2o_fake_data_'+str(value)+'.csv')
H2O_time = H2O['Time']
H2O_ppm = H2O['H2O_ppm']


abs_csv = pd.read_csv('MSI/data/klip_optimization_with_raw_data/hong_abs_fake_data_'+str(value)+'.csv')
abs_time = abs_csv['time']
abs_values = abs_csv['Absorbance_227']

test_tube.run()
test_tube2.run()



time_history = test_tube.timeHistory
time_history2 = test_tube2.timeHistory

parser = yp.Parser()
parser2 = yp.Parser()


abs_instance = csp.Absorb()
abs_instance2 = csp.Absorb()


exp_loaded = parser.load_to_obj('MSI/data/klip_optimization_with_raw_data/Hong_HO2_fake_data_'+str(value)+'.yaml')
abs_loaded = parser.load_to_obj('MSI/data/hong_H2O2_fake_data/Hong_HO2_fake_data_'+str(value)+'_abs_updated.yaml')
abs_data = abs_instance.superimpose_shock_tube(test_tube,abs_loaded,15.2,kinetic_sens=0)
#
##
exp_loaded2 = parser2.load_to_obj('MSI/data/klip_optimization_with_raw_data/Hong_HO2_fake_data_'+str(value)+'.yaml')
abs_loaded2 = parser2.load_to_obj('MSI/data/klip_optimization_with_raw_data/Hong_HO2_fake_data_'+str(value)+'_abs.yaml')
abs_data2 = abs_instance2.superimpose_shock_tube(test_tube2,abs_loaded2,15.2,kinetic_sens=0)



plt.xlabel('Time (ms)')
plt.ylabel('Absorbance 227nm')
plt.plot(test_tube.timeHistories[0]['time']*1e3,abs_data[227],label = 'MSI')
plt.plot(test_tube2.timeHistories[0]['time']*1e3,abs_data2[227],label= 'Hong temperature dependent rate constants',color='g')
plt.plot(abs_time*1e3,abs_values,color='k',label = 'Experimental Data')
plt.legend()
plt.savefig('MSI/data/klip_optimization_comparison/Abs_'+str(value)+'_temp_dep'+'.pdf',dpi=1000,bbox_inches='tight')    



plt.figure()
plt.xlabel('Time (ms)')
plt.ylabel(r'H$_2$O ppm')
plt.plot(time_history['time']*1e3,time_history['H2O']*1e6,label='MSI')
plt.plot(time_history2['time']*1e3,time_history2['H2O']*1e6,label='Hong temperature dependent rate constants',color='g')
plt.plot(H2O_time*1e3,H2O_ppm,color='k',label = 'Experimental Data')
plt.legend()
plt.savefig('MSI/data/klip_optimization_comparison/H2O_'+str(value)+'_temp_dep'+'.pdf',dpi=1000,bbox_inches='tight')    


plt.figure()
plt.xlabel('Time (ms)')
plt.ylabel('OH ppm')
plt.plot(time_history['time']*1e3,time_history['OH']*1e6,label='MSI')
plt.plot(time_history2['time']*1e3,time_history2['OH']*1e6,label='Hong temperature dependent rate constants',color='g')

plt.plot(OH_time*1e3,OH_ppm,color='k',label = 'Experimental Data')
plt.legend()
plt.savefig('MSI/data/klip_optimization_comparison/OH_'+str(value)+'_temp_dep'+'.pdf',dpi=1000,bbox_inches='tight')    
import os

