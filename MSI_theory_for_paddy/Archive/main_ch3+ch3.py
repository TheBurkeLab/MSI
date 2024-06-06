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

# energies in cm-1

temperature_list = np.arange(50,700,10).tolist() + np.arange(700,3100,100).tolist() # K
pressure_list = [[0.001, 0.01, 0.1, 1., 10.], ['[atm]']] # Torr will be converted into atm internally

channels = [#'R->W1', 'R->P1', 'R->P2', 'R->P3', 'R->P4', 'R->P5', 'R->P6',
            #'W1->P1', 'W1->P2', 'W1->P3', 'W1->P4', 'W1->P5', 'W1->P6',
            #'P1->P2', 'P1->P3', 'P1->P4', 'P1->P5', 'P1->P6',
            #'P2->P3', 'P2->P4', 'P2->P5', 'P2->P6',
            #'P3->P4', 'P3->P5', 'P3->P6',
            #'P4->P5', 'P4->P6',
            #'P5->P6',] # channel-specific rate constants of interest
            'W1->P1', 'W1->P2', 'W1->P3',
            'P1->P2', 'P1->P3',
            'P2->P3',]
            

pertubation_percent = 0.05 # percentage of perturbation
nominal_MESS_input_path = os.getcwd() + '/ch3+ch3/'
nominal_MESS_input = 'c2h6_rad.inp'

model = MSI.PAPR_MESS(nominal_MESS_input_path, nominal_MESS_input,
                                        {'W1_Energy_1':[temperature_list, pressure_list, 0.00]},
                                        { 'W1_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                          # 'W1_Frequencies_1':[temperature_list, pressure_list, pertubation_percent],
                                          # 'W1_SymmetryFactor_1':[temperature_list, pressure_list, pertubation_percent],
                                          #'W1_HinderRotor_1':[temperature_list, pressure_list, pertubation_percent],
                                          # 'B1_NejScale_1':[temperature_list, pressure_list, pertubation_percent],
                                          # 'B1_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                          # 'B2_NejScale_1':[temperature_list, pressure_list, pertubation_percent],
                                          # 'B2_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                          # 'B3_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                          # 'B3_Frequencies_1':[temperature_list, pressure_list, pertubation_percent],
                                          # 'B3_SymmetryFactor_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B5_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                         # 'B5_Frequencies_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B5_SymmetryFactor_1':[temperature_list, pressure_list, pertubation_percent],
                                         # # 'B6_NejScale_1':[temperature_list, pressure_list, pertubation_percent],
                                         # # 'B6_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                         #'B7_NejScale_1':[temperature_list, pressure_list, pertubation_percent],
                                         #'B7_Energy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                         # 'B3_VariationalEnergy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                         # 'B3_VariationalFrequencies_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B3_ImaginaryFrequency_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B4_VariationalEnergy_1':[temperature_list, pressure_list, pertubation_percent * 349.759],
                                         # 'B4_VariationalFrequencies_1':[temperature_list, pressure_list, pertubation_percent],
                                         # 'B4_ImaginaryFrequency_1':[temperature_list, pressure_list, pertubation_percent],
                                         },
                                        channels)

# fit rate constants
n_P = 5  # number of pressure polynominals
n_T = 15 # number of temperature polynominals
model.Run() # execute PAPR-MESS perturbations
model.fit_Cheb_rates(n_P, n_T, P_min=0.0001, P_max=100, T_min=200, T_max=3000) # fit rate constants into Chebyshev expressions
model.Cheb_sens_coeff() # calculate sensitivity coefficients


# for debugging, fitting rate constants for a given trail
# target_dir = '/home/leil/MSI/PAPR-MESS_calculation/h+ho2=hooh/calculation_5/'
# model.fit_Cheb_rates(n_P, n_T, target_dir=target_dir)
# model.Cheb_sens_coeff()
