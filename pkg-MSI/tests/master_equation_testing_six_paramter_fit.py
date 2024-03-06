
import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.optimization.matrix_loader as ml
import MSI.optimization.opt_runner as opt
import MSI.simulations.absorbance.curve_superimpose as csp
import MSI.simulations.yaml_parser as yp
import MSI.optimization.shock_tube_optimization_shell as stMSI
import MSI.master_equation.master_equation_six_parameter_fit as spf
import cantera as ct
import numpy as np 



files_to_include = [['Hong_1.yaml'],['Hong_2.yaml']]
numer_of_iterations = 10
cti_file = 'chem_original_burke.cti'
working_directory = 'MSI/data/test_data'
reaction_uncertainty_csv = 'burke_uncertainty_test.csv'

#rate_constant_target_value_data = 'burke_target_value_single_reactions.csv'
rate_constant_target_value_data = 'burke_target_value_test.csv'
rate_constant_target_value_data = ''
#this would be an empty string '' if you do not want to include it 
run_with_k_target_values = 'On'
#this could be 'On'

rate_constant_target_value_data_for_plotting = 'burke_target_value_test.csv'



MSI_st_instance_one = stMSI.MSI_shocktube_optimization(cti_file,
                                                   .01,
                                                   1,
                                                   1,
                                                   working_directory,
                                                   files_to_include,                 
                                                   reaction_uncertainty_csv,rate_constant_target_value_data )
MSI_st_instance_one.one_run_shock_tube_optimization()
#just use this array as a test 
experiment_dictonaries = MSI_st_instance_one.experiment_dictonaries
list_of_parsed_yamls = MSI_st_instance_one.list_of_parsed_yamls
S_matrix = MSI_st_instance_one.S_matrix
r1a = np.array([-0.373074255,	-5.658058364,-2.203911028,1.69333527,-7.110529947,-0.272049596,1.373125254,-0.644666166])
r1n = np.array([0.043611058,	0.15417925,	-0.208413633,	-0.306031876,	0.81053055,	0.031772359	,-0.136901806,	0.073807424])
r1Ea = np.array([0.419762882,	-1.301125209,	-0.681648059,	-0.091866582,	-2.353326781,	-0.064230907,	0.047721593	,0.147941186])
  
sensitivity_dict = {'H2O2 + OH <=> H2O + HO2':{'A':r1a,'N':r1n,'Ea':r1Ea}}


master_equation_instance = spf.Master_Equation_Six_Parameter_Fit()
k_mapping = master_equation_instance.master_equation_handling(experiment_dictonaries, 
                                     list_of_parsed_yamls,
                                     sensitivity_dict,
                                     ['H2O2 + OH <=> H2O + HO2'])


fake_delta_X_updates = {'R_0':[.1,.1,.1,.1,.1,.1,.1,.1]}
MP = master_equation_instance.surrogate_model_molecular_parameters(sensitivity_dict,
                                                  ['H2O2 + OH <=> H2O + HO2'],
                                                  fake_delta_X_updates,
                                                  experiment_dictonaries)
    
#S_matrix = master_equation_instance.map_to_S(nested_list,sensitivity_dict, ['H + HCO <=> CO + H2'])
#
#matrix_instance = ml.Adding_Target_Values(MSI_st_instance.S_matrix,MSI_st_instance.Y_matrix,MSI_st_instance.sigma)
#matrix_instance.target_values_for_S(MSI_st_instance.data_directory+'/'+MSI_st_instance.k_target_values_csv,MSI_st_instance.experiment_dictonaries,
#                                   ['H + HCO <=> CO + H2'],sensitivity_dict)
