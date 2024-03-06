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


#files_to_include = [['Hong_HO2_fake_data_15.yaml','Hong_HO2_fake_data_15_abs.yaml']] 
#files_to_include = [['Hong_HO2_fake_data_15.yaml','Hong_HO2_fake_data_15_abs.yaml']] 
files_to_include = [['Hong_4.yaml','Hong_4_abs.yaml']] 
#lock in physical model paramters 
#update cti file to have the hong paramters for that temperature in them 
numer_of_iterations = 30                                     
cti_file = 'Hong_new.cti'
working_directory = 'MSI/data/hong_fake_data_high_temp_optimization'
reaction_uncertainty_csv = 'Hong_new_reaction_uncertainty.csv'

#rate_constant_target_value_data = 'burke_target_value_single_reactions.csv'
#rate_constant_target_value_data = 'target_reactions_test.csv'
rate_constant_target_value_data = ''
#this would be an empty string '' if you do not want to include it 
run_with_k_target_values = 'Off'
#this could be 'On'

#rate_constant_target_value_data_for_plotting = 'target_reactions_test.csv'



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
    



#need to fix this and return _s_matrix and y_matrix



MSI_st_instance_two = stMSI.MSI_shocktube_optimization(cti_file,
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
X_list = MSI_st_instance_two.multiple_shock_tube_runs(numer_of_iterations)


deltaXAsNsEas = MSI_st_instance_two.deltaXAsNsEas
physical_obervable_updates_list = MSI_st_instance_two.physical_obervable_updates_list
absorbance_observables_updates_list = MSI_st_instance_two.absorbance_coef_update_dict
Ydf = MSI_st_instance_two.Y_data_frame
Zdf = MSI_st_instance_two.z_data_frame
experimental_dicts = MSI_st_instance_two.experiment_dictonaries
z_matrix = MSI_st_instance_two.z_matrix
s_matrix = MSI_st_instance_two.s_matrix
y = MSI_st_instance_two.y_matrix
Y_matrix = MSI_st_instance_two.Y_matrix
S_matrix = MSI_st_instance_two.S_matrix

X = MSI_st_instance_two.X
Xdf = MSI_st_instance_two.X_data_frame
covarience = MSI_st_instance_two.covarience
exp_dict_list_optimized = MSI_st_instance_two.experiment_dictonaries
parsed_yaml_list = MSI_st_instance_two.list_of_parsed_yamls
sigma = MSI_st_instance_two.sigma
X = MSI_st_instance_two.X
delta_X = MSI_st_instance_two.delta_X
#target_value_rate_constant_csv = 'MSI/data/test_data/FFCM1_custom_target_value_test.csv'
original_cti_file = MSI_st_instance_two.data_directory +'/'+ MSI_st_instance_two.cti_file_name

experiment_dict_uncertainty = MSI_st_instance_two.experiment_dict_uncertainty_original
target_value_csv = MSI_st_instance_two.data_directory +'/'+ MSI_st_instance_two.k_target_values_csv

if run_with_k_target_values == 'On' or run_with_k_target_values == 'on':
    k_target_value_S_matrix = MSI_st_instance_two.k_target_values_for_s
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
                                     target_value_rate_constant_csv= MSI_st_instance_two.data_directory +'/'+'target_reactions_test.csv' ,
                                     k_target_value_S_matrix =k_target_value_S_matrix,
                                     k_target_values=run_with_k_target_values,
                                     working_directory=working_directory,
                                     shock_tube_instance = MSI_st_instance_two)

observable_counter_and_absorbance_wl,length_of_experimental_data = plotting_instance.lengths_of_experimental_data()
sigmas_optimized,test = plotting_instance.calculating_sigmas(S_matrix,covarience)
sigmas_original,test2 = plotting_instance.calculating_sigmas(S_matrix_original,original_covariance)
plotting_instance.plotting_observables(sigmas_original = sigmas_original,sigmas_optimized= sigmas_optimized)
diag = plotting_instance.getting_matrix_diag(covarience)

#plotting_instance.Y_matrix_plotter(Y_matrix,exp_dict_list_optimized,y,sigma)



#plotting_instance.plotting_rate_constants(optimized_cti_file=MSI_st_instance_two.new_cti_file,
#                                original_cti_file=original_cti_file,
#                                initial_temperature=250,
#                                final_temperature=2500)
                                


#sensitivity, top_sensitivity = plotting_instance.sort_top_uncertainty_weighted_sens()
#obs = plotting_instance.plotting_uncertainty_weighted_sens()
#plotting_normal_distributions(self,

        
#plotting_instance.plotting_normal_distributions(['A_5','A_6','A_7','A_8','Sigma_1','Sigma_2'],
#                                                optimized_cti_file=MSI_st_instance_two.new_cti_file,
#                                                pdf_distribution_file='',
#                                                shock_tube_instance=MSI_st_instance_two)
#
#
#
#plotting_instance.plotting_joint_normal_distributions([('A_5','A_6'),('A_5','A_7'),('A_5','A_8'),('A_6','A_7'),
#                                                       ('A_6','A_8'),('A_7','A_8'),('A_8','A_11'),('A_5','A_11'),('Sigma_1','Sigma_2'),
#                                                       ('A_5','Sigma_1'),('A_6','Sigma_1'),('A_7','Sigma_1'),
#                                                       ('A_8','Sigma_1'),('A_5','Sigma_2'),('A_6','Sigma_2'),
#                                                       ('A_7','Sigma_2'),('A_8','Sigma_2')],
#                                                       optimized_cti_file=MSI_st_instance_two.new_cti_file,
#                                                       joint_data_csv='')

plotting_instance.plotting_joint_normal_distributions([('A_5','A_8'),
                                                       ('A_6','A_8'),('A_7','A_8'),('A_11','A_8'),                                                       
                                                       ('A_8','Sigma_1'),
                                                       ('A_8','Sigma_2')],
                                                       optimized_cti_file=MSI_st_instance_two.new_cti_file,
                                                       joint_data_csv='')

#plotting_instance.difference_plotter(['A_1','n_1','Ea_1','A_0'],
#                                                optimized_cti_file=MSI_st_instance_two.new_cti_file,
#                                                pdf_distribution_file='MSI/data/automating_HO2_masten/graph_read_pdf.csv')
#

gas_optimized = ct.Solution(MSI_st_instance_two.new_cti_file)
gas_optimized.TPX = 1283,101325*1.635,{'H2O2':0.002046 ,'H2O':0.001113,
                                     'O2':0.000556,'Ar':0.995212}

k5 = gas_optimized.forward_rate_constants[5]*1000

k6 =  gas_optimized.forward_rate_constants[6]*1000
k7 = gas_optimized.forward_rate_constants[7]*1000
k8 = gas_optimized.forward_rate_constants[8]*1000
k11 = gas_optimized.forward_rate_constants[11]*1000


# make shit plotting 
#import matplotlib.pyplot as plt 
#import pandas as pd
#df_MSI = pd.read_csv(working_directory+'/new_channel_MSI_results.csv')
#
#test_tube = exp_dict_list_optimized[0]['simulation']
#time_history = test_tube.timeHistories[0] 
#plt.xlabel('Time (ms)')
#plt.ylabel('Absorbance 227nm')
#abs_data = exp_dict_list_optimized[0]['absorbance_calculated_from_model'][227]
#plt.figure()
#plt.plot(test_tube.timeHistories[0]['time']*1e3,abs_data,label='Hong Simulation')
#plt.plot(df_MSI['Absorbance_time']*1e3, df_MSI['Absorbance'],label='MSI New Channel Results')
#plt.legend()
#
#
#
#
#plt.figure()
#plt.xlabel('Time (ms)')
#plt.ylabel('OH ppm')
#plt.plot(time_history['time']*1e3,time_history['OH']*1e6,label='Hong Simulation')
#plt.plot(df_MSI['OH_time'], df_MSI['OH_ppm'],label='MSI New Channel Results')
#plt.legend()
#
#
#plt.figure()
#plt.xlabel('Time (ms)')
#plt.ylabel(r'H$_2$O ppm')
#plt.plot(time_history['time']*1e3,time_history['H2O']*1e6,label='Hong Simulation')
#plt.plot(df_MSI['H2O_time']*1e3, df_MSI['H2O_ppm'],label='MSI New Channel Results')
#plt.legend()





