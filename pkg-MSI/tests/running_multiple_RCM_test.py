import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')
import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.optimization.matrix_loader as ml
import MSI.optimization.opt_runner as opt
import MSI.simulations.absorbance.curve_superimpose as csp
import MSI.simulations.yaml_parser as yp
import cantera as ct
import pandas as pd
#

#test_p = pr.Processor('C:\\Users\\Skoron\\Google Drive\\Burke Group\\Codes\\Mechanisms\\FFCM-1\\FFCM1.cti')
test_p = pr.Processor('MSI/data/DME-Methanol_Blends_RCM/DME-MeOH_combined_mech_Hongfu(Cantera).cti')



yaml_file_list = [('MSI/data/DME-Methanol_Blends_RCM_2/Wang_0_short.yaml',)]
yaml_instance = yp.Parser()





list_of_yaml_objects = yaml_instance.load_yaml_list(yaml_list=yaml_file_list)


list_of_experiment_dicts = yaml_instance.parsing_multiple_dictonaries(list_of_yaml_objects = list_of_yaml_objects)
# # #
optimization_instance = opt.Optimization_Utility()

test = optimization_instance.looping_over_parsed_yaml_files(list_of_experiment_dicts,
                                          yaml_file_list,
                                          processor=test_p, 
                                          kineticSens=1,
                                          physicalSens=1,
                                          dk=.01)
#matrix_instance = ml.OptMatrix()
# master_equation_uncertainty_df=pd.read_csv('MSI/data/flow_reactor/six_parameter_fit_uncertainty_df.csv')
#Y_matrix,Y1 = matrix_instance.load_Y(test,list_of_experiment_dicts,loop_counter=0,master_equation_flag=False)
# # Y_matrix,Y1 = matrix_instance.load_Y(test,list_of_experiment_dicts,loop_counter=0)
# ru = '/Users/carlylagrotta/Dropbox/Columbia/MSI/data/flow_reactor/FFCM1_reaction_uncertainty.csv'

# Z,Z_data_Frame,sigma,active_parameters = matrix_instance.build_Z(test,list_of_experiment_dicts,loop_counter=0f)
# # #
#Z,Z_data_Frame,sigma,active_parameters = matrix_instance.build_Z(test,list_of_experiment_dicts,loop_counter=0,reaction_uncertainty='MSI/data/DME-Methanol_Blends_RCM/DME-MeOH_combined_mech_Hongfu(Cantera)_reaction_uncertanties.csv')
# # #
# # #x1,x2,x3,x4 = matrix_instance.breakup_delta_x(z_matrix[257:],test,loop_counter=0)
# # #
#S_matrix = matrix_instance.load_S(test,list_of_experiment_dicts,dk=.01)
# X,c,s_matrix,y_matrix,delta_X,z_matrix,X_data_frame,prior_diag,prior_diag_df,sorted_prior_diag,covariance_prior_df,prior_sigmas_df=matrix_instance.matrix_manipulation(0,S_matrix,Y_matrix,Z,active_parameters=active_parameters)
# ##adding target values
# deltaXAsNsEas,physical_observables,absorbance_coef_update_dict,X_to_subtract_from_Y,kinetic_paramter_dict=matrix_instance.breakup_X(X,test,test,loop_counter = 0)

# #target_value_instance = ml.Adding_Target_Values(S_matrix,Y_matrix,z_matrix,sigma)
# #k_targets_for_y = target_value_instance.target_values_Y('MSI/data/test_data/FFCM1_target_values.csv',test)
# #k_targets_for_z,sigma = target_value_instance.target_values_for_Z('MSI/data/test_data/FFCM1_target_values.csv')
# #s_target_values = target_value_instance.target_values_for_S('MSI/data/test_data/FFCM1_target_values.csv',test)


