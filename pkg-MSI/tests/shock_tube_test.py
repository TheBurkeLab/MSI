import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# test_p = pr.Processor('MSI/data/test_data/FFCM1_custom.cti')
# #test_p = pr.Processor('MSI/data/chebyshev_data/FFCM1_custom_cheb.cti')
# #test_p = pr.Processor('MSI/data/chebyshev_data/FFCM1_custom_cheb_test.cti')

# test_tube = st.shockTube(pressure=1.909,
#                          temperature=1398,
#                          observables=['H2O'],
#                          kineticSens=0,
#                          physicalSens=0,
#                          conditions={'O2':0.00062 ,
#                                      'H2O':0.001234,
#                                      'H2O2':0.00254 ,
#                                      'Ar': 0.9974968858350951},
#                          initialTime=0,
#                          finalTime=.07,
#                          thermalBoundary='Adiabatic',
#                          mechanicalBoundary='constant volume',
#                          processor=test_p,
#                          save_timeHistories=1,
#                          save_physSensHistories=1,
#                          volumeTrace='MSI/data/RCM/volume_trace_file.csv')

# test_tube.run()
# #test_tube.printVars()
# time_History = test_tube.timeHistory

test_p = pr.Processor('/Users/carlylagrotta/Dropbox/Columbia/MSI/data/Burke_atmospheric_chemistry/burke_atmospheric_chemistry_model.cti')
#test_p2 = pr.Processor('/Users/carlylagrotta/Dropbox/Columbia/MSI/data/branching_reaction_study/ffcm1_branching_reactions_theory.cti')


test_tube = st.shockTube(pressure=1 ,
                         temperature=298, 
                         observables=['H2CO'],
                         kineticSens=0,
                         physicalSens=0,
                         conditions={'CH2O': .00131,
                                     'Pin': 0.01,
                                     'O2':.209,
                                     'N2':.791},
                         initialTime=0,
                         finalTime=300,
                         thermalBoundary='isothermal',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1
                         )

test_tube.run()
#test_tube.printVars()
time_History = test_tube.timeHistory






#print(time_History['time'])

#plt.figure() 

#plt.plot(time_History['time'],time_History['H2CO']*1e6)
#plt.plot(df_experiment['Time'],df_experiment['H2O_ppm'])

#plt.legend( )

#plt.savefig('/Users/carlylagrotta/Dropbox/Columbia/MSI/data/Burke_atmospheric_chemistry/test.pdf')