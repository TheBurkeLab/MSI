
import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import cantera as ct
import time
### testing FFCM
start = time.time()
gas= ct.Solution('MSI/data/test_data/FFCM1.cti')
all_species = gas.species_names


test_p = pr.Processor('MSI/data/test_data/FFCM1.cti')
test_tube = st.shockTube(pressure=1.74,
                         temperature=1880,
                         observables=all_species,
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'H2O':.013,'O2':.0099,'H':.0000007,'Ar':0.9770993},
                         initialTime=0,
                         finalTime=0.1,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1)
test_tube.run()

ksens = test_tube.kineticSensitivities

end = time.time()
print(end-start,'Looping over all speices FFCM')


gas= ct.Solution('MSI/data/test_data/FFCM1.cti')
loop_length = len(gas.reaction_equations())
test_p = pr.Processor('MSI/data/test_data/FFCM1.cti')
start = time.time()
for x in range(len(gas.reaction_equations())):
    
    test_tube = st.shockTube(pressure=1.74,
                             temperature=1880,
                             observables=['OH','H2O'],
                             kineticSens=1,
                             physicalSens=0,
                             conditions={'H2O':.013,'O2':.0099,'H':.0000007,'Ar':0.9770993},
                             initialTime=0,
                             finalTime=0.1,
                             thermalBoundary='Adiabatic',
                             mechanicalBoundary='constant pressure',
                             processor=test_p,
                             save_timeHistories=1)
    test_tube.run()
    
    ksens = test_tube.kineticSensitivities
    
    end = time.time()
    
print(end-start,'looping over all reactions FFCM1')


# Aramco

### testing Aramco
start = time.time()
gas= ct.Solution('MSI/data/test_data/Aramco.cti')
all_species = gas.species_names


test_p = pr.Processor('MSI/data/test_data/Aramco.cti')
test_tube = st.shockTube(pressure=1.74,
                         temperature=1880,
                         observables=all_species,
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'H2O':.013,'O2':.0099,'H':.0000007,'Ar':0.9770993},
                         initialTime=0,
                         finalTime=0.1,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1)
test_tube.run()

ksens = test_tube.kineticSensitivities

end = time.time()
print(end-start,'Calculating for all species Aramco')


gas= ct.Solution('MSI/data/test_data/Aramco.cti')
loop_length = len(gas.reaction_equations())
test_p = pr.Processor('MSI/data/test_data/Aramco.cti')
start = time.time()
for x in range(len(gas.reaction_equations())):
    
    test_tube = st.shockTube(pressure=1.74,
                             temperature=1880,
                             observables=['OH','H2O'],
                             kineticSens=1,
                             physicalSens=0,
                             conditions={'H2O':.013,'O2':.0099,'H':.0000007,'Ar':0.9770993},
                             initialTime=0,
                             finalTime=0.1,
                             thermalBoundary='Adiabatic',
                             mechanicalBoundary='constant pressure',
                             processor=test_p,
                             save_timeHistories=1)
    test_tube.run()
    
    ksens = test_tube.kineticSensitivities
    
    end = time.time()
    
print(end-start,'looping over all reactions Aramco')

### testing heptane

start = time.time()
gas= ct.Solution('MSI/data/test_data/heptane.cti')
all_species = gas.species_names


test_p = pr.Processor('MSI/data/test_data/heptane.cti')
test_tube = st.shockTube(pressure=1.74,
                         temperature=1880,
                         observables=all_species,
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'H2O':.013,'O2':.0099,'H':.0000007,'Ar':0.9770993},
                         initialTime=0,
                         finalTime=0.1,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1)
test_tube.run()

ksens = test_tube.kineticSensitivities

end = time.time()
print(end-start,'Calculating for all species heptane')


gas= ct.Solution('MSI/data/test_data/heptane.cti')
loop_length = len(gas.reaction_equations())
test_p = pr.Processor('MSI/data/test_data/heptane.cti')
start = time.time()
for x in range(len(gas.reaction_equations())):
    
    test_tube = st.shockTube(pressure=1.74,
                             temperature=1880,
                             observables=['OH','H2O'],
                             kineticSens=1,
                             physicalSens=0,
                             conditions={'H2O':.013,'O2':.0099,'H':.0000007,'Ar':0.9770993},
                             initialTime=0,
                             finalTime=0.1,
                             thermalBoundary='Adiabatic',
                             mechanicalBoundary='constant pressure',
                             processor=test_p,
                             save_timeHistories=1)
    test_tube.run()
    
    ksens = test_tube.kineticSensitivities
    
    end = time.time()
    
print(end-start,'looping over all reactions heptane')