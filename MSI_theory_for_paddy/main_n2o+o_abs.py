# -*- coding: utf-8 -*-
"""

Pipeline for PAPR-MESS MSI code.

@author: Lei Lei
"""

import postprocessor as MSI
import os
import numpy as np

# Initialize MSI.PAPR_MESS:
# -- the first arguement is the file path for the nominal PAPR-MESS input files
# -- the second arguement is the name for the unperturbed (nominal) input file
# -- the third arguement is a dictionary specifying the nominal condition (to allow optimization starting from nonzero perturbations),
#    the value specifies temperature list, pressure list, and nominal perturbation
# -- the fourth arguement is a dictionary with elements specifying the perturbations (sensitivity analysis are based on the nominal conditions defined above),
#    the keys specify names of perturbation runs and values specify temperature, pressure, and perturbation
# -  the fifth arguement is a list indicating the channels of interest

Temperature_list = np.arange(1000,5000,10).tolist() # K
Pressure_list = [[10], ['[atm]']] # Torr will be converted into atm internally
channels = ['P1->P2']

pertubation_percent = 0.1 # percentage of perturbation
nominal_MESS_input_path = os.getcwd() + '/n2o+o_abs/'
nominal_MESS_input = 'N2+O2_A2_tnl.inp'



model = MSI.PAPR_MESS(nominal_MESS_input_path, nominal_MESS_input,
                                    {'B1_Energy_1': [Temperature_list, Pressure_list, 0.00]},
                                    {#'colrel_FactorOne': [Temperature_list, Pressure_list, pertubation_percent],
                                    # 'colrel_PowerOne': [Temperature_list, Pressure_list, pertubation_percent],
                                    # 'colrel_Epsilons': [Temperature_list, Pressure_list, pertubation_percent],
                                    # 'colrel_Sigmas': [Temperature_list, Pressure_list, pertubation_percent],
                                    # 'W2a_Energy_1': [Temperature_list, Pressure_list, pertubation_percent * 349.759],
                                    # 'W2a_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    # 'W2a_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    # 'W2b_Energy_1': [Temperature_list, Pressure_list, pertubation_percent * 349.759],
                                    # 'W2b_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    # 'W2b_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    #  'P2_Energy_1': [Temperature_list, Pressure_list, pertubation_percent * 349.759],
                                    #  'P2_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    #  'P2_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    #  'P2_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'B1_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                     'B1_Energy_1' : [Temperature_list, Pressure_list, pertubation_percent * 349.759],
                                     'B1_ImaginaryFrequency_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    # 'P1_SymmetryFactor_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    # 'P1_Frequencies_1': [Temperature_list, Pressure_list, pertubation_percent],
                                    # 'P1_Energy_1': [Temperature_list, Pressure_list, pertubation_percent * 349.759],                           
                                    },
                                    channels, abstraction=True)

# fit rate constants
n_P = 1  # number of pressure polynominals
n_T = 15 # number of temperature polynominals
same_line_result = True
aggregated_sens = False
# model.Run()
model.fit_Cheb_rates(n_P=n_P, n_T=n_T, P_min=0.01, P_max=100, T_min=200, T_max=5400, same_line_result=same_line_result)
model.Cheb_sens_coeff(same_line_result=same_line_result, aggregated_sens=aggregated_sens)

# for debugging, fitting rate constants for a given trail

