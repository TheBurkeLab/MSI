import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

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


#files_to_include = [['Hong_HO2_fake_data_15.yaml','Hong_HO2_fake_data_15_abs.yaml']] 
#files_to_include = [['Hong_HO2_fake_data_15.yaml','Hong_HO2_fake_data_15_abs.yaml']] 
#files_to_include = [['Hong_4.yaml','Hong_4_abs.yaml']] 
files_to_include = [['Hong_0.yaml'],
                    ['Hong_2.yaml'],
                    ['Hong_3.yaml'],
                    ['Hong_1.yaml'],
                    ['Troe_4.yaml','Troe_4_abs.yaml'],
                    ['Troe_5.yaml','Troe_5_abs.yaml'],
                    ['Troe_6.yaml','Troe_6_abs.yaml'],
                    ['Troe_7.yaml','Troe_7_abs.yaml'],
                    ['Troe_8.yaml','Troe_8_abs.yaml'],
                    ['Hong_HO2_fake_data_0.yaml','Hong_HO2_fake_data_0_abs.yaml'],
                    ['Hong_HO2_fake_data_1.yaml','Hong_HO2_fake_data_1_abs.yaml'],
                    ['Hong_HO2_fake_data_2.yaml','Hong_HO2_fake_data_2_abs.yaml'],
                    ['Hong_HO2_fake_data_3.yaml','Hong_HO2_fake_data_3_abs.yaml'],
                    ['Hong_HO2_fake_data_4.yaml','Hong_HO2_fake_data_4_abs.yaml'],
                    ['Hong_HO2_fake_data_5.yaml','Hong_HO2_fake_data_5_abs.yaml'],
                    ['Hong_HO2_fake_data_6.yaml','Hong_HO2_fake_data_6_abs.yaml'],
                    ['Hong_HO2_fake_data_7.yaml','Hong_HO2_fake_data_7_abs.yaml'],
                    ['Hong_HO2_fake_data_8.yaml','Hong_HO2_fake_data_8_abs.yaml'],
                    ['Hong_HO2_fake_data_9.yaml','Hong_HO2_fake_data_9_abs.yaml'],
                    ['Hong_HO2_fake_data_10.yaml','Hong_HO2_fake_data_10_abs.yaml'],
                    ['Hong_HO2_fake_data_11.yaml','Hong_HO2_fake_data_11_abs.yaml'],
                    ['Hong_HO2_fake_data_12.yaml','Hong_HO2_fake_data_12_abs.yaml'],
                    ['Hong_HO2_fake_data_13.yaml','Hong_HO2_fake_data_13_abs.yaml'],
                    ['Hong_HO2_fake_data_14.yaml','Hong_HO2_fake_data_14_abs.yaml'],
                    ['Hong_HO2_fake_data_15.yaml','Hong_HO2_fake_data_15_abs.yaml']]
                    # ['Farooq_0_large_uncertainty.yaml'],
                    # ['Farooq_1_large_uncertainty_2.yaml'],
                    # ['Farooq_2_large_uncertainty.yaml'],
                    # ['Farooq_3_large_uncertainty.yaml']] 
#lock in physical model paramters 
#update cti file to have the hong paramters for that temperature in them 
numer_of_iterations = 2                                     
cti_file = 'Hong_new_original_reactions_2.cti'
working_directory = 'MSI/data/klip_optimization_with_raw_data'
reaction_uncertainty_csv = 'Hong_new_reaction_uncertainty_2.csv'

#rate_constant_target_value_data = 'burke_target_value_single_reactions.csv'
#rate_constant_target_value_data = 'target_reactions_test.csv'
rate_constant_target_value_data = ''
#this would be an empty string '' if you do not want to include it 
run_with_k_target_values = 'Off'
#this could be 'On'

#rate_constant_target_value_data_for_plotting = 'target_reactions_test.csv'



MSI_st_instance_one_hong = stMSI.MSI_shocktube_optimization(cti_file,
                                                   .01,
                                                   1,
                                                   1,
                                                   working_directory,
                                                   files_to_include,                 
                                                   reaction_uncertainty_csv,rate_constant_target_value_data )
MSI_st_instance_one_hong.one_run_shock_tube_optimization()

S_matrix_original = MSI_st_instance_one_hong.S_matrix
exp_dict_list_original = MSI_st_instance_one_hong.experiment_dictonaries
original_covariance = MSI_st_instance_one_hong.covarience
X_one_itteration = MSI_st_instance_one_hong.X
MSI_st_instance_one_hong.deltaXAsNsEas
    



#need to fix this and return _s_matrix and y_matrix



MSI_st_instance_two_hong = stMSI.MSI_shocktube_optimization(cti_file,
                                                   .01,
                                                   1,
                                                   1,
                                                   working_directory,
                                                   files_to_include,                 
                                                   reaction_uncertainty_csv,rate_constant_target_value_data )
#
#
#
#
#
#
X_list = MSI_st_instance_two_hong.multiple_shock_tube_runs(numer_of_iterations)


deltaXAsNsEas = MSI_st_instance_two_hong.deltaXAsNsEas
physical_obervable_updates_list = MSI_st_instance_two_hong.physical_obervable_updates_list
absorbance_observables_updates_list = MSI_st_instance_two_hong.absorbance_coef_update_dict
Ydf = MSI_st_instance_two_hong.Y_data_frame
Zdf = MSI_st_instance_two_hong.z_data_frame
experimental_dicts = MSI_st_instance_two_hong.experiment_dictonaries
z_matrix = MSI_st_instance_two_hong.z_matrix
s_matrix = MSI_st_instance_two_hong.s_matrix
y = MSI_st_instance_two_hong.y_matrix
Y_matrix = MSI_st_instance_two_hong.Y_matrix
S_matrix = MSI_st_instance_two_hong.S_matrix

X = MSI_st_instance_two_hong.X
Xdf = MSI_st_instance_two_hong.X_data_frame
covarience = MSI_st_instance_two_hong.covarience
exp_dict_list_optimized = MSI_st_instance_two_hong.experiment_dictonaries
parsed_yaml_list = MSI_st_instance_two_hong.list_of_parsed_yamls
sigma = MSI_st_instance_two_hong.sigma
X = MSI_st_instance_two_hong.X
delta_X = MSI_st_instance_two_hong.delta_X
#target_value_rate_constant_csv = 'MSI/data/test_data/FFCM1_custom_target_value_test.csv'
original_cti_file = MSI_st_instance_two_hong.data_directory +'/'+ MSI_st_instance_two_hong.cti_file_name

experiment_dict_uncertainty = MSI_st_instance_two_hong.experiment_dict_uncertainty_original
target_value_csv = MSI_st_instance_two_hong.data_directory +'/'+ MSI_st_instance_two_hong.k_target_values_csv

if run_with_k_target_values == 'On' or run_with_k_target_values == 'on':
    k_target_value_S_matrix = MSI_st_instance_two_hong.k_target_values_for_s
else:
    k_target_value_S_matrix = None


##########################################################################################################################
#PLOTTING##
##########################################################################################################################


plotting_instance = plotter.Plotting(S_matrix,
                                     s_matrix,
                                     Y_matrix,
                                     Y_matrix,
                                     z_matrix,
                                     X,
                                     sigma,
                                     covarience,
                                     original_covariance,
                                     S_matrix_original,
                                     exp_dict_list_optimized,
                                     exp_dict_list_original,
                                     parsed_yaml_list,
                                     Ydf,
                                     target_value_rate_constant_csv= MSI_st_instance_two_hong.data_directory +'/'+'target_reactions_test.csv' ,
                                     k_target_value_S_matrix =k_target_value_S_matrix,
                                     k_target_values=run_with_k_target_values,
                                     working_directory=working_directory,
                                     shock_tube_instance = MSI_st_instance_two_hong)

observable_counter_and_absorbance_wl,length_of_experimental_data = plotting_instance.lengths_of_experimental_data()
sigmas_optimized,test = plotting_instance.calculating_sigmas(S_matrix,covarience)
sigmas_original,test2 = plotting_instance.calculating_sigmas(S_matrix_original,original_covariance)
plotting_instance.plotting_observables(sigmas_original = sigmas_original,sigmas_optimized= sigmas_optimized)
diag = plotting_instance.getting_matrix_diag(covarience)







