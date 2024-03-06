import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')
import MSI_2.simulations.instruments.shock_tube as st
import MSI_2.cti_core.cti_processor as pr
import MSI_2.optimization.matrix_loader_temp as ml
import MSI_2.optimization.opt_runner as opt
import MSI_2.simulations.absorbance.curve_superimpose as csp
import MSI_2.simulations.yaml_parser as yp
import cantera as ct
import pandas as pd
#

#test_p = pr.Processor('C:\\Users\\Skoron\\Google Drive\\Burke Group\\Codes\\Mechanisms\\FFCM-1\\FFCM1.cti')
test_p = pr.Processor('C:\\Users\\Skoron\\Google Drive\\Burke Group\\Codes\\Mechanisms\\FFCM-1\\FFCM1.cti')

yaml_file_list = [('C:\\Users\\Skoron\\Google Drive\\Burke Group\\Mark\\MSI_2\\data\\yaml_files\\JSR_template_new.yaml',)]
yaml_instance = yp.Parser()
list_of_yaml_objects = yaml_instance.load_yaml_list(yaml_list=yaml_file_list)


list_of_experiment_dicts = yaml_instance.parsing_multiple_dictonaries(list_of_yaml_objects = list_of_yaml_objects)
#
optimization_instance = opt.Optimization_Utility()

test = optimization_instance.looping_over_parsed_yaml_files(list_of_experiment_dicts,
                                          yaml_file_list,
                                          processor=test_p, 
                                          kineticSens=1,
                                          physicalSens=1,
                                          dk=.01)
matrix_instance = ml.OptMatrix()
# #
# #
master_equation_uncertainty_df=pd.read_csv('C:\\Users\\Skoron\\Google Drive\\Burke Group\\Mark\\MSI_2\\data\\test_data\\six_parameter_fit_uncertainty_df.csv')
# Y_matrix,Y1 = matrix_instance.load_Y(test,list_of_experiment_dicts,loop_counter=0,master_equation_reactions = ['H2O2 + OH <=> H2O + HO2',
#                               '2 HO2 <=> H2O2 + O2',
#                               'HO2 + OH <=> H2O + O2',
#                               '2 OH <=> H2O + O',
#                               'CH3 + HO2 <=> CH4 + O2',
#                               'CH3 + HO2 <=> CH3O + OH'],master_equation_flag=True, 
#                                 master_equation_uncertainty_df=master_equation_uncertainty_df)
Y_matrix,Y1 = matrix_instance.load_Y(test,list_of_experiment_dicts,loop_counter=0)
# #

# Z,Z_data_Frame,sigma,active_parameters = matrix_instance.build_Z(test,list_of_experiment_dicts,loop_counter=0,reaction_uncertainty='MSI_2\\data\\test_data\\FFCM1_reaction_uncertainty.csv',
#                                             master_equation_reaction_list = ['H2O2 + OH <=> H2O + HO2',
#                               '2 HO2 <=> H2O2 + O2',
#                               'HO2 + OH <=> H2O + O2',
#                               '2 OH <=> H2O + O',
#                               'CH3 + HO2 <=> CH4 + O2',
#                               'CH3 + HO2 <=> CH3O + OH'],master_equation_flag=True, 
#                                 master_equation_uncertainty_df=master_equation_uncertainty_df)
#
Z,Z_data_Frame,sigma,active_parameters = matrix_instance.build_Z(test,list_of_experiment_dicts,loop_counter=0,reaction_uncertainty='MSI_2\\data\\test_data\\FFCM1_reaction_uncertainty.csv')
#
#x1,x2,x3,x4 = matrix_instance.breakup_delta_x(z_matrix[257:],test,loop_counter=0)
#
S_matrix = matrix_instance.load_S(test,list_of_experiment_dicts,dk=.01)
X,c,s_matrix,y_matrix,delta_X,z_matrix,X_data_frame,prior_diag,prior_diag_df,sorted_prior_diag,covariance_prior_df,prior_sigmas_df=matrix_instance.matrix_manipulation(0,S_matrix,Y_matrix,Z,active_parameters=active_parameters)
##adding target values
deltaXAsNsEas,physical_observables,absorbance_coef_update_dict,X_to_subtract_from_Y,kinetic_paramter_dict=matrix_instance.breakup_X(X,test,test,loop_counter = 0)

#target_value_instance = ml.Adding_Target_Values(S_matrix,Y_matrix,z_matrix,sigma)
#k_targets_for_y = target_value_instance.target_values_Y('MSI/data/test_data/FFCM1_target_values.csv',test)
#k_targets_for_z,sigma = target_value_instance.target_values_for_Z('MSI/data/test_data/FFCM1_target_values.csv')
#s_target_values = target_value_instance.target_values_for_S('MSI/data/test_data/FFCM1_target_values.csv',test)


