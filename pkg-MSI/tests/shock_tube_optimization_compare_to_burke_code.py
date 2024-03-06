#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:00:09 2019

@author: carly
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:05:29 2019

@author: carly
"""


import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.optimization.matrix_loader as ml
import MSI.optimization.opt_runner as opt
import MSI.simulations.absorbance.curve_superimpose as csp
import MSI.simulations.yaml_parser as yp
import MSI.optimization.shock_tube_optimization_shell_six_param_fit as stMSIspf
import cantera as ct
import pandas as pd
import numpy as np
import MSI.utilities.plotting_script as plotter
import MSI.utilities.post_processor as post_processor



files_to_include = [['Hong_0_secondary_reactions.yaml'],
                    ['Hong_2_burke_weights.yaml'],
                    ['Hong_3.yaml'],
                    ['Hong_1.yaml'],
                    ['Troe_4_cv.yaml','Troe_4_abs_burke_weights.yaml'],
                    ['Troe_5_cv.yaml','Troe_5_abs_burke_weights.yaml'],
                    ['Troe_6_cv.yaml','Troe_6_abs_burke_weights.yaml'],
                    ['Troe_7_secondary_reactions.yaml','Troe_7_abs_burke_weights.yaml'],
                    ['Troe_8_secondary_reactions.yaml','Troe_8_abs_burke_weights.yaml']]

numer_of_iterations = 3
cti_file = 'burke_og_model.cti'
working_directory = 'MSI/data/test_data'
reaction_uncertainty_csv = 'burke_reaction_uncertainty.csv'
master_reaction_equation_cti_name = 'master_reactions_FFCM1_burke_code_reproduction.cti'
#rate_constant_target_value_data = 'burke_target_value_single_reactions.csv'

#this would be an empty string '' if you do not want to include it 
run_with_k_target_values = 'On'
master_equation_reactions = ['H2O2 + OH <=> H2O + HO2',
                             '2 HO2 <=> H2O2 + O2',
                             'HO2 + OH <=> H2O + O2',
                             '2 OH <=> H2O + O',
                             '2 OH (+M) <=> H2O2 (+M)']

#master_index = [2,3,4,5,6,7]
master_index = [1,2,3,4,5]

master_equation_uncertainty_df = pd.read_csv('MSI/data/test_data/six_parameter_fit_uncertainty_df_burke.csv')
#this could be 'On'

rate_constant_target_value_data_for_plotting = 'FFCM1_target_reactions_1_plotting_burke_code_reproduction.csv'
rate_constant_target_value_data = 'FFCM1_target_reactions_1_burke_code_reproduction.csv'
rate_constant_target_value_data_extra = 'FFCM1_target_reactions_extra_data_burke_code_reproduction.csv'

#start here 


six_parameter_fit_sensitivities = {'H2O2 + OH <=> H2O + HO2':{'A':np.array([-13.37032086,	32.42060027,	19.23022032,	6.843287462	, 36.62853824	,-0.220309785	,-0.099366346,	-4.134352081]),
                                                              'n':np.array([1.948532282,	-5.341557065,	-3.337497841,	-1.025292166,	-5.813524857,	0.011862923	,0.061801326,	0.581628835]),
                                                              'Ea':np.array([-0.463042822,	1.529151218,	0.808025472	,0.359889935,	-0.021309254,	-0.098013004,	-0.102022118,	-0.097024727]),
                                                              'c':np.array([0.00163576,	-0.008645666,	-0.003111179,	-0.002541995,	0.014228149	,0.001263134,	0.001236963,	-0.000390567]),
                                                              'd':np.array([1.071992802,	-2.780550365,	-1.71391034	,-0.274481751,	-4.491132406,	-0.054960894,	0.049553379,	0.270885383]),
                                                              'f':np.array([-0.027060156,	0.056903076,	0.041102936	,0.001361221,	0.144385439,	0.003136796	,0.001374015,	-0.006089248])},
                                    '2 HO2 <=> H2O2 + O2': {'A':np.array([-12.93733217,	24.39245077	,17.73177606,	4.37803475,	33.44985889,	0.381601192	,3.748890308]),
                                                            'n':np.array([1.872602872,	-4.096806067,	-3.09439453	,-0.63226683,	-5.125008418,	-0.061610462,	-0.677953862]),
                                                            'Ea':np.array([-0.463903763	,1.259537237,	0.826684258	,0.257400116,	0.803882706	,2.20E-05,	0.181336266]),
                                                            'c':np.array([0.002069572,	-0.008314769,	-0.00424128	,-0.002016113,	0.000134642	,0.000122049	,-0.001026567]),
                                                            'd':np.array([0.981856324,	-1.847383095,	-1.493544053,	0.016222685,	-3.428753345,	-0.050708107,	-0.526284003]),
                                                            'f':np.array([-0.022628436,	0.023558844,	0.031573523	,-0.00732987,	0.096573278	,0.001668073,	0.01033547])},
                                    'HO2 + OH <=> H2O + O2': {'A':np.array([-4.795727446,	6.426354909	,4.878258417,	2.472791017,	7.856296474,	1.328033302	,-3.457932692,	-0.349839371,	2.331070924	,2.403555921,	-0.165397001,	0.246540172	,0.722946077]),
                                                              'n':np.array([0.624241134,	-1.321082842,	-1.032242319,	-0.36532386,	-1.112545721,	-0.188622956,	0.421083939	,0.038859478	,-0.360855106,	-0.38989218,	0.029669899	,-0.04371581,	-0.130487515]),
                                                              'Ea':np.array([-0.259799111,	0.205620792	,0.130799794,	0.137023666	,0.379232542,	6.19E-02,	-0.198196699,	-0.023548432,	0.118069394	,0.104383314	,-0.003830947,	0.011566499	,-0.073557828]),
                                                              'c':np.array([0.00161312,	-0.001906694,	-0.000863021,	-0.00105112	,-0.002185605,	-0.000334461,	0.001817049	,0.000170761,	-0.000859313,	-0.000653029,	-3.11E-06	,-6.37E-05,	0.00047058]),
                                                              'd':np.array([0.124499363,	-0.645652135,	-0.535188558,	0.052734001	,-0.45181066,	-0.082250635,	0.034779283,	-0.011522821,	0.017057742,	-0.165960963,	0.057288687,	-0.012776017,	-0.192422381]),
                                                              'f':np.array([0.002033109,	-0.011099716,	0.005351213	,-0.007623667,	0.005327017	,0.001259485,0.00245957,	0.000976725	,-0.004879845,	0.001903886	,-0.001838669	,0.000252269,	0.004691829])},
                                    '2 OH <=> H2O + O':      {'A': np.array([-5.40485067,	18.96061659	,8.089301961,	6.953940096	,-12.54280438,	-3.264972401,	2.106487623	,-1.657943467,	1.614935	,-1.536463599]),
                                                              'n': np.array([0.803274875,	-3.167851673,	-1.607661056,	-1.041258197,	1.679914849,	0.466415264	,-0.326136934,	0.355297684	,-0.16618967,	0.253903734]),
                                                              'Ea': np.array([0.147285831,	0.605814544,	-0.062253282,	0.372322712,	-1.884116555,	-0.281992263,	0.099465537	,0.030650483,	0.176069015	,-0.056967886]),
                                                              'c': np.array([-0.003001658,	-0.001870536,	0.003820535	,-0.002753277,	0.014224162,	0.00032969	,-0.000627241,	-0.001081979,	-0.002009835,	0.000255318]),
                                                              'd':np.array([0.446957978,	-1.467039994,	-1.298391635,	-0.402720385,	0.568106728	,0.229877892,	-0.194395052,	1.033858025	,0.527183366,	0.308743056]),
                                                              'f':np.array([-0.010053913,	0.025128322,	0.035579811	,0.00515753	,-0.0083511,	-0.00512885,	0.003954,	-0.029711993	,-0.01986861,	-0.007691647])},
                                    '2 OH (+M) <=> H2O2 (+M)':{'A':np.array([1,0,0]),
                                                               'n':np.array([0,1,0]),
                                                               'Ea':np.array([0,0,1]),
                                                               'c':np.array([0,0,0]),
                                                               'd':np.array([0,0,0]),
                                                               'f':np.array([0,0,0])}}



molecular_parameter_sensitivities = {'H2O2 + OH <=> H2O + HO2':{'A':np.array([-0.373074255,	-5.658058364,-2.203911028,1.69333527,-7.110529947,-0.272049596,1.373125254,-0.644666166]),
                                                              'n':np.array([0.043611058,	0.15417925,	-0.208413633,	-0.306031876,	0.81053055,	0.031772359	,-0.136901806,	0.073807424]),
                                                              'Ea':np.array([0.419762882,	-1.301125209,	-0.681648059,	-0.091866582,	-2.353326781,	-0.064230907,	0.047721593	,0.147941186])},
                                    '2 HO2 <=> H2O2 + O2': {'A':np.array([-0.166005487,	-6.797175212,	-2.798300682,	1.973896891	,-4.354910767,	-0.082067357,	-3.839749825]),
                                                            'n':np.array([0.018748596,	0.294710827	,-0.135488286,	-0.332967052,	0.4930396,	0.009470627	,0.409095255]),
                                                            'Ea':np.array([0.459015825,	-1.401810899,	-0.722040616,	-0.066133729,	-1.52807633	,-0.021832631,	-0.411667639])},
                                    'HO2 + OH <=> H2O + O2': {'A':np.array([-1.30109642,	-11.63457509,	-4.680271526,	0.782373804	, -0.016083278,	0.005513255	,-1.738426278,	-0.232013539,	0.884067816	,-0.500473791,	0.399272687	,0.062255923	,-1.667253993]),
                                                              'n':np.array([0.152797314,	1.1181845,	0.306250902	,-0.164846884,	-0.008229148,	-0.001531881,	0.195875814	,0.026844834,	-0.18238354	,0.017363927,	-0.055634983	,-0.017324495,	0.218771679]),
                                                              'Ea':np.array([0.101558432,	-1.638858106,	-0.704325409,	-0.119041648,	-0.307281167,	-0.04872945,	0.001603412	,0.000324159,	-0.08089174,	-0.148811902,	0.027266121	,-0.002907638,	-0.237949453])},
                                    '2 OH <=> H2O + O':      {'A': np.array([0.299144373,	-2.662684629,	-6.643003014,	0.370230493	,-3.354253502,	-0.271981922,	-0.581195748,	9.774024441	, 5.90328859,	2.272800133]),
                                                              'n': np.array([-0.028599275,	-0.071787028,	0.572722706	,-0.109709456,	0.381272207	,0.03153973	,0.061282516,	-1.341475144,	-0.835422411,	-0.302994441]),
                                                              'Ea': np.array([0.535103651,	-1.054606857,	-0.989721261,	-0.169631331,	-1.099840578,	-0.069647609,	-0.101285313,	0.74522721,	0.352517552	,0.205464658])},
                                   '2 OH (+M) <=> H2O2 (+M)': {'A':np.array([1,0,0]),
                                                               'n':np.array([0,1,0]),
                                                               'Ea':np.array([0,0,1])}} 






six_parameter_fit_nominal_parameters_dict = {'H2O2 + OH <=> H2O + HO2':{'A':4.64E-06,'n':5.605491008,'Ea':-5440.266692,'c':126875776.1,'d':0.000441194,'f':-5.35E-13},
                                '2 HO2 <=> H2O2 + O2':{'A':1.30E+04,'n':1.997152351,'Ea':-3628.04407,'c':93390973.44,'d':-0.000732521,'f':8.20E-12} ,
                                 'HO2 + OH <=> H2O + O2':{'A':1.41E+18,'n':-2.05344973,'Ea':-232.0064051,'c':15243859.12,'d':-0.001187694,'f':8.01E-12},
                                 '2 OH <=> H2O + O':{'A':354.5770856,'n':2.938741717,'Ea':-1836.492972,'c':12010735.18,'d':-4.87E-05,'f':1.22E-12},
                                 '2 OH (+M) <=> H2O2 (+M)':{'A':5.030000e+12,'n':0.058,'Ea':-634.33,'c':0.0,'d':0.0,'f':0.0}}





MSI_st_instance_one = stMSIspf.MSI_shocktube_optimization_six_parameter_fit(cti_file,
                                                   .01,
                                                   1,
                                                   1,
                                                   working_directory,
                                                   files_to_include,                 
                                                   reaction_uncertainty_csv,rate_constant_target_value_data,
                                                   master_equation_reactions = master_equation_reactions,
                                                   molecular_parameter_sensitivities = molecular_parameter_sensitivities,
                                                   six_parameter_fit_sensitivities = six_parameter_fit_sensitivities,
                                                   master_reaction_equation_cti_name = master_reaction_equation_cti_name,
                                                   master_index = master_index,
                                                   master_equation_uncertainty_df = master_equation_uncertainty_df,
                                                   six_paramter_fit_nominal_parameters_dict = six_parameter_fit_nominal_parameters_dict)
MSI_st_instance_one.one_run_shock_tube_optimization()

S_matrix_original = MSI_st_instance_one.S_matrix
exp_dict_list_original = MSI_st_instance_one.experiment_dictonaries
original_covariance = MSI_st_instance_one.covarience
X_one_itteration = MSI_st_instance_one.X
MSI_st_instance_one.deltaXAsNsEas




#need to fix this and return _s_matrix and y_matrix



MSI_st_instance_two = stMSIspf.MSI_shocktube_optimization_six_parameter_fit(cti_file,
                                                   .01,
                                                   1,
                                                   1,
                                                   working_directory,
                                                   files_to_include,                 
                                                   reaction_uncertainty_csv,rate_constant_target_value_data,
                                                   master_equation_reactions = master_equation_reactions,
                                                   molecular_parameter_sensitivities = molecular_parameter_sensitivities,
                                                   six_parameter_fit_sensitivities = six_parameter_fit_sensitivities,
                                                   master_reaction_equation_cti_name = master_reaction_equation_cti_name,
                                                   master_index = master_index,
                                                   master_equation_uncertainty_df = master_equation_uncertainty_df,
                                                   six_paramter_fit_nominal_parameters_dict = six_parameter_fit_nominal_parameters_dict)
                                                   
                                                 
#
#
#
#
#
#ALL OF THIS STUFF CAN PROBABLY GO INTO SOME SORT OF CLASS
delta_X_list = MSI_st_instance_two.multiple_shock_tube_runs(numer_of_iterations)


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
molecular_parameter_updates = MSI_st_instance_two.delta_x_molecular_params_by_reaction_dict
nominal_dict_six_p_fit  = MSI_st_instance_two.six_paramter_fit_nominal_parameters_dict
original_diag = np.diag(original_covariance)




#target_value_rate_constant_csv = 'MSI/data/test_data/FFCM1_custom_target_value_test.csv'
original_cti_file = MSI_st_instance_two.data_directory +'/'+ MSI_st_instance_two.cti_file_name

experiment_dict_uncertainty = MSI_st_instance_two.experiment_dict_uncertainty_original
target_value_csv = MSI_st_instance_two.data_directory +'/'+ MSI_st_instance_two.k_target_values_csv
six_parameter_fit_dict_optimized = MSI_st_instance_two.updated_six_parameter_fits_dict
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
                                     target_value_rate_constant_csv= MSI_st_instance_two.data_directory +'/'+ rate_constant_target_value_data_for_plotting ,
                                     target_value_rate_constant_csv_extra_values = MSI_st_instance_two.data_directory +'/'+rate_constant_target_value_data_extra,
                                     k_target_value_S_matrix =k_target_value_S_matrix,
                                     k_target_values=run_with_k_target_values,
                                     working_directory = working_directory,
                                     sigma_uncertainty_weighted_sensitivity_csv='')
##MSI_st_instance_two.data_directory +'/'+'sigma_for_uncertainty_weighted_sensitivity_updated.csv'
#
observable_counter_and_absorbance_wl,length_of_experimental_data = plotting_instance.lengths_of_experimental_data()
sigmas_optimized,test = plotting_instance.calculating_sigmas(S_matrix,covarience)
sigmas_original,test2 = plotting_instance.calculating_sigmas(S_matrix_original,original_covariance)
plotting_instance.plotting_observables(sigmas_original = sigmas_original,sigmas_optimized= sigmas_optimized)
diag = plotting_instance.getting_matrix_diag(covarience)


#plotting_instance.Y_matrix_plotter(Y_matrix,exp_dict_list_optimized,y,sigma)

#
#
#plotting_instance.plotting_rate_constants(optimized_cti_file=MSI_st_instance_two.new_cti_file,
#                                original_cti_file=original_cti_file,
#                                initial_temperature=250,
#                                final_temperature=2500)
                                


sensitivity, top_sensitivity = plotting_instance.sort_top_uncertainty_weighted_sens()
obs = plotting_instance.plotting_uncertainty_weighted_sens()

plotting_instance.plotting_rate_constants_six_paramter_fit(optimized_cti_file=MSI_st_instance_two.new_cti_file,
                                original_cti_file=original_cti_file,
                               initial_temperature=250,
                                final_temperature=2500,
                                master_equation_reactions = master_equation_reactions,
                                six_parameter_fit_dict_optimized = six_parameter_fit_dict_optimized,
                                six_parameter_fit_dict_nominal = six_parameter_fit_nominal_parameters_dict,
                                six_parameter_fit_sensitivity_dict =six_parameter_fit_sensitivities )

#plotting_instance.plotting_X_itterations(list_of_X_values_to_plot = [0,1,2,3,4,5,50],list_of_X_array=X_list,number_of_iterations=numer_of_iterations)
post_processor_instance = post_processor.post_processing(optimized_cti_file = MSI_st_instance_two.new_cti_file,
                                                    original_cti_file = original_cti_file,
                                                    kinetic_paramter_dictonary = MSI_st_instance_two.kinetic_paramter_dict,
                                                    master_equation_reactions=master_equation_reactions,
                                                    six_parameter_fit_nominal_parameters_dict = six_parameter_fit_nominal_parameters_dict,
                                                    six_parameter_fit_optimized_paramter_dict = six_parameter_fit_dict_optimized,
                                                    exp_dict_list_optimized = exp_dict_list_optimized,
                                                    exp_dict_list_original = exp_dict_list_original,
                                                    parsed_yaml_list = parsed_yaml_list)

kinetic_paramters_dict = post_processor_instance.create_active_kinetic_paramter_dictonary()
physical_params_dict = post_processor_instance.create_active_physical_paramter_dictonary()

