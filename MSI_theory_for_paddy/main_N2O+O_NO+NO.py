#%%
# -*- coding: utf-8 -*-
"""
Pipeline for PAPR-MESS MSI code.
@author: Lei Lei
"""

import postprocessor as MSI
import os
import numpy as np

# import sensitivity_parse as sp
# import extract_rate_constants as erc
# import vibe_checker as vc

import carly_code as cc

# Initialize MSI.PAPR_MESS:
# -- the first arguement is the file path for the nominal PAPR-MESS input files
# -- the second arguement is the name for the unperturbed (nominal) input file
# -- the third arguement is a dictionary specifying the nominal condition (to allow optimization starting from nonzero perturbations),
#    the value specifies temperature list, pressure list, and nominal perturbation
# -- the fourth arguement is a dictionary with elements specifying the perturbations (sensitivity analysis are based on the nominal conditions defined above),
#    the keys specify names of perturbation runs and values specify temperature, pressure, and perturbation
# -  the fifth arguement is a list indicating the channels of interest

# Temperature_list = np.arange(1000,5000,5).tolist() # K
Temperature_list = np.arange(200,700,10).tolist() + np.arange(700,4100,100).tolist()
Pressure_list = [[1], ['[atm]']] # Torr will be converted into atm internally
channels = ['P1->P2']



# channels = ['P1->W3', 'P1->P2', 'P1->P3', 'P1->P4', 'P1->P5',
#             'P2->W3', 'P2->P3', 'P2->P4', 'P2->P5',
#             'P3->W3', 'P3->P4', 'P3->P5',
#             'P4->W3', 'P4->P5',
#             'P5->W3']# channel-specific rate constants of interest

# channels = ['P1->W3','W3->P1', 'P1->P3', 'P1->P4', 'P1->P5',
#             'P3->W3','W3->P3', 'P3->P4', 'P3->P5',
#             'P4->W3','W3->P4', 'P4->P5',
#             'P5->W3','W3->P5'] # channel-specific rate constants of interest

pertubation_percent = 0.1 # percentage of perturbation
nominal_MESS_input_path = os.getcwd() + '/N2O+O/'
nominal_MESS_input = 'NO+NO.inp'
#nominal_MESS_input = 'combined.inp'

#nominal_MESS_input = 'ho2+ho2_Lei.inp'


model = MSI.PAPR_MESS(nominal_MESS_input_path, nominal_MESS_input,
                                    {'P1_Energy_1': [Temperature_list, Pressure_list, 0.00]},
                                    {
                                     'P1_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'P1_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'P1_Energy_1': [Temperature_list, Pressure_list, pertubation_percent * 349.759],
                                     
                                     'P2_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'P2_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'P2_Energy_1': [Temperature_list, Pressure_list, pertubation_percent * 349.759],                                         
                                                                             
                                     'W1_Energy_1': [Temperature_list, Pressure_list, pertubation_percent * 349.759],
                                     'W1_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'W1_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    
                                     'W2_Energy_1': [Temperature_list, Pressure_list, pertubation_percent * 349.759],
                                     'W2_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'W2_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],                                    
                                     
                                    #  'B1_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    #  'B1_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    #  'B1_Energy_1' : [Temperature_list, Pressure_list, pertubation_percent * 349.759],
                                     
                                    #  'B2_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    #  'B2_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    #  'B2_Energy_1' : [Temperature_list, Pressure_list, pertubation_percent * 349.759],
                                    
                                    #  'B3_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    #  'B3_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    #  'B3_Energy_1' : [Temperature_list, Pressure_list, pertubation_percent * 349.759],

                                     'B4_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'B4_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'B4_Energy_1' : [Temperature_list, Pressure_list, pertubation_percent * 349.759],
                                     'B4_ImaginaryFrequency_1': [Temperature_list, Pressure_list, pertubation_percent],

                                     'B5_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'B5_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'B5_Energy_1' : [Temperature_list, Pressure_list, pertubation_percent * 349.759],                                                                       
                                     'B5_ImaginaryFrequency_1': [Temperature_list, Pressure_list, pertubation_percent],

                                    },
                                    channels)

# fit rate constants
n_T = 7 # number of temperature polynominals
n_P = 1  # number of pressure polynominals

TP_dict = {'P_min':0.0001, 'P_max':100.0, 'T_min':200.0, 'T_max':4000.0}

same_line_result = True
model.Run() # execute PAPR-MESS perturbations
model.fit_Cheb_rates(n_P, n_T, P_min=TP_dict['P_min'], P_max=TP_dict['P_max'], T_min=TP_dict['T_min'], T_max=TP_dict['T_max'], same_line_result=same_line_result) # fit rate constants into Chebyshev expressions
model.Cheb_sens_coeff(same_line_result=same_line_result) # calculate sensitivity coefficients

# for debugging, fitting rate constants for a given trail
# target_dir = '/home/leil/MSI/PAPR-MESS_calculation/h+ho2=hooh/calculation_5/'
# model.fit_Cheb_rates(n_P, n_T, target_dir=target_dir)
# model.Cheb_sens_coeff()

#%%

calculation_directory = '/home/jl/MSI_Theory/PAPR-MESS_calculation/NO+NO/calculation_9/'
states_dict = {'P1':'N2O + O', 'P2':'2 NO'}

MSI_Theory = cc.MSI_Theory(calculation_directory, channels, states_dict, [n_T, n_P], TP_dict)

MSI_Theory.extract_rate_constants()

reaction = 'N2O + O <=> 2 NO'
channel = 'P1->P2'
original_model = '/home/jl/MSI_Theory/N2O+O/gonzalez.cti'
chebyshev_model = '/home/jl/MSI_Theory/N2O+O/comparision_rate_constants.cti'

MSI_Theory.vibe_checker(reaction, channel, original_model, chebyshev_model, temperature_list = np.arange(200,4000), pressure_list = [1], conversion = 1000)

key_words, sensitivities = MSI_Theory.sensitivity_parse()
