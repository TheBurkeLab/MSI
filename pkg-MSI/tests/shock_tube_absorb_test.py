
import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')
import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.simulations.absorbance.curve_superimpose as csp  
import MSI.simulations.yaml_parser as yp
import cantera as ct
import pandas as pd 
import matplotlib.pyplot as plt

test_p = pr.Processor('MSI/data/branching_reaction_study/FFCM1_custom_cheb_extra_zeros_new_extra_reactions_precursor.cti')



test_tube = st.shockTube(pressure=1.17051635825,
                         temperature=1972,
                         observables=['OH'],
                         kineticSens=0,
                         physicalSens=0,
                         conditions={ 'CH3OH':5.694252633828817e-06,
                                     'Ar':0.9999943057473661},
                         initialTime=0,
                         finalTime=0.002,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1)

#test_p2 = pr.Processor('MSI/data/hong_H2O2_fake_data/Hong_new_half_temperature_dep.cti')
#test_tube2 = st.shockTube(pressure=1.676,
#                         temperature=1182,
#                         observables=['OH','H2O'],
#                         kineticSens=0,
#                         physicalSens=0,
#                         conditions={'H2O2':0.002046 ,'H2O':0.001113,
#                                     'O2':0.000556,'Ar':0.995212},
#                         initialTime=0,
#                         finalTime=0.001,
#                         thermalBoundary='Adiabatic',
#                         mechanicalBoundary='constant pressure',
#                         processor=test_p2,
#                         save_timeHistories=1,
#                         save_physSensHistories=1)
#
#test_p3 = pr.Processor('MSI/data/hong_H2O2_fake_data/Hong_new_full_temperature_dep.cti')
#test_tube3 = st.shockTube(pressure=1.676,
#                         temperature=1182,
#                         observables=['OH','H2O'],
#                         kineticSens=0,
#                         physicalSens=0,
#                         conditions={'H2O2':0.002046 ,'H2O':0.001113,
#                                     'O2':0.000556,'Ar':0.995212},
#                         initialTime=0,
#                         finalTime=0.001,
#                         thermalBoundary='Adiabatic',
#                         mechanicalBoundary='constant pressure',
#                         processor=test_p3,
#                         save_timeHistories=1,
#                         save_physSensHistories=1)


# OH = pd.read_csv('MSI/data/test_data/hong_oh_4.csv')
# OH_time = OH['Time']
# OH_ppm = OH['OH_ppm']



# H2O = pd.read_csv('MSI/data/test_data/hong_h2o_4.csv')
# H2O_time = H2O['Time']
# H2O_ppm = H2O['H2O_ppm']


# abs_csv = pd.read_csv('MSI/data/test_data/hong_abs_4.csv')
# abs_time = abs_csv['time']
# abs_values = abs_csv['Absorbance_227']

test_tube.run()
#test_tube2.run()
#test_tube3.run()

time_history = test_tube.timeHistory
#time_history2 = test_tube2.timeHistory
#time_history3 = test_tube3.timeHistory


parser = yp.Parser()
#parser2 = yp.Parser()
#parser3 = yp.Parser()

abs_instance = csp.Absorb()
#abs_instance2 = csp.Absorb()
#abs_instance3 = csp.Absorb()


#exp_loaded = parser.load_to_obj('MSI/data/test_data/Hong_4.yaml')
abs_loaded = parser.load_to_obj('MSI/data/branching_reaction_study/Krasnoperov_abs.yaml')
abs_data = abs_instance.superimpose_shock_tube(test_tube,abs_loaded,8.745,kinetic_sens=0)

df = pd.read_csv('MSI/data/branching_reaction_study/Krasnoperov_abs_1.csv')


#abs_loaded2 = parser2.load_to_obj('MSI/data/hong_H2O2_fake_data/Hong_HO2_fake_data_3_abs.yaml')
#abs_data2 = abs_instance2.superimpose_shock_tube(test_tube2,abs_loaded2,15.2,kinetic_sens=0)
#
#abs_loaded3 = parser.load_to_obj('MSI/data/hong_H2O2_fake_data/Hong_HO2_fake_data_3_abs.yaml')
#abs_data3 = abs_instance.superimpose_shock_tube(test_tube3,abs_loaded3,15.2,kinetic_sens=0)

plt.figure()
plt.xlabel('Time (ms)')
plt.ylabel('Absorbance 210nm')
plt.plot(test_tube.timeHistories[0]['time']*1e3,abs_data[308])
plt.plot(df['time']*1e3,df['Absorbance_308'])
#plt.plot(test_tube2.timeHistories[0]['time']*1e3,abs_data2[227],label='k1,k2=constant value  k3,k4=expression from paper')
#plt.plot(test_tube3.timeHistories[0]['time']*1e3,abs_data3[227],label='k1,k2=derived A value  k3,k4=expression from paper')
#plt.plot(abs_time*1e3,abs_values,color='r',label = 'Experimental Data')
plt.savefig('/Users/carlylagrotta/Desktop/abs.pdf')


plt.figure()
plt.xlabel('Time (ms)')
plt.ylabel('H2O2 ppm')
plt.plot(time_history['time']*1e3,time_history['H2O2']*1e6)
#plt.plot(time_history2['time']*1e3,time_history2['OH']*1e6,label='k1,k2=constant value  k3,k4=expression from paper')
#plt.plot(time_history3['time']*1e3,time_history3['OH']* 1e6,label='k1,k2=derived A value  k3,k4=expression from paper')
plt.savefig('/Users/carlylagrotta/Desktop/h2o2.pdf')

#plt.plot(OH_time*1e3,OH_ppm,color='r',label = 'Experimental Data')
plt.figure()
plt.xlabel('Time (ms)')
plt.ylabel(r'HO$_2$ ppm')
plt.plot(time_history['time']*1e3,time_history['HO2']*1e6)
#plt.plot(time_history2['time']*1e3,time_history2['H2O']*1e6,label='k1,k2=constant value  k3,k4=expression from paper')
#plt.plot(time_history3['time']*1e3,time_history3['H2O']*1e6,label='k1,k2=derived A value  k3,k4=expression from paper')
plt.savefig('/Users/carlylagrotta/Desktop/ho2.pdf')


# plt.figure()
#plt.xlabel('Time (ms)')
#plt.ylabel('Cl ')
#plt.plot(test_tube.timeHistories[0]['time']*1e3,time_history['Cl']*1e6)
#plt.plot(test_tube2.timeHistories[0]['time']*1e3,abs_data2[227],label='k1,k2=constant value  k3,k4=expression from paper')
#plt.plot(test_tube3.timeHistories[0]['time']*1e3,abs_data3[227],label='k1,k2=derived A value  k3,k4=expression from paper')
#plt.plot(abs_time*1e3,abs_values,color='r',label = 'Experimental Data')
#plt.savefig('/Users/carlylagrotta/Desktop/cl.pdf')


