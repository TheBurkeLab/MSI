import sys, os

#os.chdir('../../') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.optimization.matrix_loader as ml
import MSI.optimization.opt_runner as opt
import MSI.simulations.absorbance.curve_superimpose as csp
import MSI.simulations.yaml_parser as yp
import MSI.optimization.shock_tube_optimization_shell as stMSI
import cantera as ct
import MSI.utilities.plotting_script as plotter
import numpy as np
import MSI.master_equation.master_equation as meq
files_to_include = [['Masten_0.yaml'],
                    ['Masten_1.yaml'],
                    ['Masten_2.yaml'],
                    ['Masten_3.yaml']] 
#files_to_include = [['Masten_0.yaml'],
#                    ['Masten_1.yaml'],
#                    ['Masten_2.yaml'],
#                    ['Masten_3.yaml'],
#                    ['Masten_4.yaml'],
#                    ['Masten_5.yaml'],
#                    ['Masten_6.yaml'],
#                    ['Masten_7.yaml'],
#                    ['Masten_8.yaml'],
#                    ['Masten_9.yaml']]
numer_of_iterations = 2                                        
cti_file = 'masten_paper_gri_thermo.cti'
working_directory = 'MSI/data/automating_HO2_masten'
reaction_uncertainty_csv = 'reaction_uncertainty_masten.csv'

#rate_constant_target_value_data = 'burke_target_value_single_reactions.csv'
#rate_constant_target_value_data = 'target_reactions_test.csv'
rate_constant_target_value_data = ''
#this would be an empty string '' if you do not want to include it 
run_with_k_target_values = 'Off'
#this could be 'On'

rate_constant_target_value_data_for_plotting = 'target_reactions_test.csv'



MSI_st_instance_one = stMSI.MSI_shocktube_optimization(cti_file,
                                                   .01,
                                                   1,
                                                   1,
                                                   working_directory,
                                                   files_to_include,                 
                                                   reaction_uncertainty_csv,rate_constant_target_value_data )
MSI_st_instance_one.one_run_shock_tube_optimization()

S_matrix_original = MSI_st_instance_one.S_matrix
exp_dict_list_original = MSI_st_instance_one.experiment_dictonaries
original_covariance = MSI_st_instance_one.covarience
X_one_itteration = MSI_st_instance_one.X
MSI_st_instance_one.deltaXAsNsEas
list_of_parsed_yamls = MSI_st_instance_one.list_of_parsed_yamls
    


#just use this array as a test 
a1 = np.array([-4.71844785e-14,  -1.11022302e-14,   4.16333634e-15,   5.55111512e-15,
  5.55111512e-15,  -1.64246079e-02,   5.42951118e-03,   1.81168588e-03,
 -2.88407459e-03,   4.68262347e-03,  -8.32667268e-15,   2.22044605e-14,
 -1.11022302e-14,  -0.00000000e+00,  -1.11022302e-14,  -1.64246079e-02,
  5.42951118e-03,   1.81168588e-03,  -2.88407459e-03,   4.68262347e-03,
 -1.94289029e-14,  -2.49800181e-14,   2.77555756e-15,   2.77555756e-15,
 -2.77555756e-15,  -1.64246079e-02,   5.42951118e-03,   1.81168588e-03,
 -2.88407459e-03,   4.68262347e-03,  -4.16333634e-14,  -2.49800181e-14,
  2.77555756e-15,   2.77555756e-15,  -2.77555756e-15,  -1.64246079e-02,
  5.42951118e-03,   1.81168588e-03,  -2.88407459e-03,   4.68262347e-03,
 -4.16333634e-14,  -2.49800181e-14,   2.77555756e-15,   2.77555756e-15,
 -2.77555756e-15,  -1.64246079e-02,   5.42951118e-03,   1.81168588e-03,
 -2.88407459e-03,   4.68262347e-03,  -4.16333634e-14,  -2.49800181e-14,
  2.77555756e-15,   2.77555756e-15,  -2.77555756e-15,  -1.64246079e-02,
  5.42951118e-03,   1.81168588e-03,  -2.88407459e-03,   4.68262347e-03,
 -4.16333634e-14,  -2.49800181e-14,   2.77555756e-15,   2.77555756e-15,
 -2.77555756e-15,  -1.64246079e-02,   5.42951118e-03,   1.81168588e-03,
 -2.88407459e-03,   4.68262347e-03,  -4.16333634e-14,  -2.49800181e-14,
  2.77555756e-15,   2.77555756e-15,  -2.77555756e-15,  -1.64246079e-02,
  5.42951118e-03,   1.81168588e-03,  -2.88407459e-03,   4.68262347e-03,
 -4.16333634e-14,  -2.49800181e-14,   2.77555756e-15,   2.77555756e-15,
 -2.77555756e-15,  -1.64246079e-02,   5.42951118e-03,   1.81168588e-03,
 -2.88407459e-03,   4.68262347e-03,  -4.16333634e-14,  -2.49800181e-14,
  2.77555756e-15,   2.77555756e-15,  -2.77555756e-15,  -1.64246079e-02,
  5.42951118e-03,   1.81168588e-03,  -2.88407459e-03,   4.68262347e-03])
a1 = a1.reshape((25,4))
  
sensitivity_dict = {'H2O2 + OH <=> H2O + HO2':[a1,a1,a1], 'HO2 + OH <=> H2O + O2':[a1,a1,a1]}


master_equation_instance = meq.Master_Equation()
mapped_to_alpha_full_simulation,nested_list = master_equation_instance.map_to_alpha(sensitivity_dict,
                                      exp_dict_list_original,
                                      list_of_parsed_yamls,
                                      ['H2O2 + OH <=> H2O + HO2','HO2 + OH <=> H2O + O2'])

    
S_matrix = master_equation_instance.map_to_S(nested_list,sensitivity_dict, ['H2O2 + OH <=> H2O + HO2','HO2 + OH <=> H2O + O2'])
#
#matrix_instance = ml.Adding_Target_Values(MSI_st_instance_one.S_matrix,MSI_st_instance_one.Y_matrix,MSI_st_instance_one.sigma)
#matrix_instance.target_values_for_S(MSI_st_instance_one.data_directory+'/'+MSI_st_instance_one.k_target_values_csv,MSI_st_instance_one.experiment_dictonaries,
#                                   ['H + HCO <=> CO + H2'],sensitivity_dict)
