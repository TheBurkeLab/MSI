#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:53:15 2018

@author: carly
"""

import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import cantera as ct
import time
start = time.time()
gas= ct.Solution('MSI/data/test_data/FFCM1.cti')
all_species = gas.species_names


test_p = pr.Processor('MSI/data/test_data/FFCM1.cti')
test_tube = st.shockTube(pressure=1.74,
                         temperature=1880,
                         observables=['OH'],
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
print(end-start,'FFCM1 2 species')


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
print(end-start,'FFCM1 all species')



#Aramco

start = time.time()
gas= ct.Solution('MSI/data/test_data/Aramco.cti')
all_species = gas.species_names


test_p = pr.Processor('MSI/data/test_data/Aramco.cti')
test_tube = st.shockTube(pressure=1.74,
                         temperature=1880,
                         observables=['OH'],
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
print(end-start,'Aramco 2 species')


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
print(end-start,'Aramco all species')

#Heptane



start = time.time()
gas= ct.Solution('MSI/data/test_data/heptane.cti')
all_species = gas.species_names


test_p = pr.Processor('MSI/data/test_data/heptane.cti')
test_tube = st.shockTube(pressure=1.74,
                         temperature=1880,
                         observables=['OH'],
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
print(end-start,'heptane 2 species')


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
print(end-start,'heptane all species')