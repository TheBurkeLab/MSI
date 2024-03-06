import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')


import cantera as ct
import pandas as pd
import numpy as np
gas = ct.Solution('MSI/data/test_data/FFCM1.cti')
gas.TPX = 1880, 1.74*101325, {'H2O':.013,'O2':.0099,'H':.0000007,'Ar':0.9770993}
shockTube = ct.IdealGasConstPressureReactor(gas,name = 'R1',energy= 'on')

sim=ct.ReactorNet([shockTube])

columnNames = [shockTube.component_name(item) for item in range(shockTube.n_vars)]  
columnNames = ['time']+['pressure'] + columnNames
timeHistory = pd.DataFrame(columns = columnNames)

counter = 0
t=0
while t < .1:
     t = sim.step() 
     state = np.hstack([t, shockTube.thermo.P, shockTube.mass, 
     shockTube.T, shockTube.thermo.X])
     timeHistory.loc[counter] = state
     counter +=1

print(timeHistory)
