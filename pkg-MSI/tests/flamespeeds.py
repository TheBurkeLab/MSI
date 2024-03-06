# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:53:46 2017

@author: Mark Barbet
"""

#Free Flame simulator for use in TMRP project
import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import numpy as np
import cantera as ct
import pandas as pd
import os
import simulations.free_flame as ff
import simulations.simulations as sim
#import simulations as sim

fuels=['CH3OH']
#fuels=[{'H2':1,'CO':1}]
phi=[]
phi=[1]
phi=[0.6,0.9,1.0,1.2,1.4,1.6,1.8]
phi=np.arange(0.6,2.0,0.2)
#phi=[0.4]
#phi=[0.6]
#results=[]
if 'results' not in locals().keys():
    results=[]

pressures=[1]
temps=[298]
mechanisms = ['../data/test_data/FFCM1.cti']
width=0.3

for i in np.arange(len(phi)):
    for filename in np.arange(len(mechanisms)):
        
        oxidizer={'O2':0.5, 'N2':0.5*3.76}
        
        gas=ct.Solution(mechanisms[filename])
        gas.TP=temps[0],pressures[0]*ct.one_atm
        results.append(ff.free_flame(phi[i],fuels[0],oxidizer,gas,width,kinetic_sens=0,energycon=True,flamespeed_sens=1,soret=False))
        results[-1].add_mechanism(mechanisms[filename])
#results[0].plot_flamespeed_sens(1,'FFCM1',0.01)
        


