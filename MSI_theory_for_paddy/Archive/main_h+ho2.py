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

temperature_list = np.arange(200,700,10).tolist() + np.arange(700,2500,100).tolist() # K
pressure_list = [[1.], ['[torr]']] # Torr, will be converted into atm internally
channels = ['R->P1', 'R->P2', 'R->P3', 'P1->P2', 'P1->P3', 'P2->P3'] # channel-specific rate constants of interest
pertubation_percent = 0.1 # percentage of perturbation
nominal_MESS_input_path = os.getcwd() + '/h+ho2/'
nominal_MESS_input = 'h+ho2=hooh.inp'

model = MSI.PAPR_MESS(nominal_MESS_input_path, nominal_MESS_input,
                                        {'W1_Energy_1':[temperature_list, pressure_list, 0.00]},
                                        {# 'W1_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                         # 'W1_Frequencies_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'W1_SymmetryFactor_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'W2_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                         # 'W2_Frequencies_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'W2_SymmetryFactor_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'W3_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                         # 'W3_Frequencies_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'W3_SymmetryFactor_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B1_NejScale_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B1_Energy_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B2_NejScale_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B2_Energy_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B3_NejScale_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B3_Energy_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B4_NejScale_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B4_Energy_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B5_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                         # 'B5_Frequencies_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B5_SymmetryFactor_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B6_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                         # 'B6_Frequencies_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B6_SymmetryFactor_1':[temperature_list, pressure_list, pertubation_percent],
                                         'B7_NejScale_1':[temperature_list, pressure_list, pertubation_percent],
                                         'B7_Energy_1':[temperature_list, pressure_list, pertubation_percent],
                                         },
                                        channels)

# fit rate constants
n_P = 1  # number of pressure polynominals
n_T = 15 # number of temperature polynominals
same_line_result = True
model.Run() # execute PAPR-MESS perturbations
model.fit_Cheb_rates(n_P=n_P, n_T=n_T, P_min=0.01, P_max=100, T_min=200, T_max=2400, same_line_result=same_line_result) # fit rate constants into Chebyshev expressions
model.Cheb_sens_coeff(same_line_result=same_line_result) # calculate sensitivity coefficients


# for debugging, fitting rate constants for a given trail
# target_dir = '/home/leil/MSI/PAPR-MESS_calculation/h+ho2=hooh/calculation_14/'
# model.fit_Cheb_rates(n_P=n_P, n_T=n_T, target_dir=target_dir)
# model.Cheb_sens_coeff(debug=True)
