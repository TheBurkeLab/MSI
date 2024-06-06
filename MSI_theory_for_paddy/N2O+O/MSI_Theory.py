#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import soln2cti as s2c
import Sensitivity_Parse as sp
import Extract_Rate_Constants as erc
import Vibe_Checker as vc

calculation_directory = '/home/jl/MSI_Theory/PAPR-MESS_calculation/NO+NO/calculation_9/'

p_list = ['P1->P2']

states_dict = {'P1':'N2O + O', 'P2':'2 NO'}

erc.
