import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')


import itertools
import cantera as ct
import pandas as pd
import numpy as np
gas = ct.Solution('MSI/data/test_data/FFCM1.cti')
gas.TPX = 1880, 1.74*101325, {'H2O':.013,'O2':.0099,'H':.0000007,'Ar':0.9770993}
observables = ['OH','H2O']
shockTube = ct.IdealGasConstPressureReactor(gas,name = 'R1',energy= 'on')
dfs = [pd.DataFrame() for x in range(len(observables))]
tempArray = [np.zeros(gas.n_reactions) for x in range(len(observables))]
kineticSens =1

sim=ct.ReactorNet([shockTube]) 
for i in range(gas.n_reactions):
    shockTube.add_sensitivity_reaction(i)           
t = 0
counter = 0
while t < .1:
    t = sim.step()
    if kineticSens == 1:
        counter_1 = 0
        for observable,reaction in itertools.product(observables, range(gas.n_reactions)):
            tempArray[observables.index(observable)][reaction] = sim.sensitivity(observable, reaction)                    
            counter_1 +=1
            if(counter_1 % gas.n_reactions == 0):
                dfs[observables.index(observable)] = dfs[observables.index(observable)].append(((pd.DataFrame(tempArray[observables.index(observable)])).transpose()),ignore_index=True)
                
    counter+=1
if kineticSens == 1:
    numpyMatrixsksens = [dfs[dataframe].as_matrix() for dataframe in range(len(dfs))]
    S = np.dstack(numpyMatrixsksens) 
    print(S)

