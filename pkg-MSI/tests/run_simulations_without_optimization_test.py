

import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.utilities.run_simulations_without_optimization as rswo
import pandas as pd
import numpy as np


#start here 


files_to_include = [['Hong_0_updated.yaml'],
                    ['Hong_2_updated.yaml'],
                    ['Hong_3_updated.yaml'],
                    ['Hong_1_updated.yaml'],
                    ['Troe_4_updated.yaml','Troe_4_abs_updated.yaml'],
                    ['Troe_5_updated.yaml','Troe_5_abs_updated.yaml'],
                    ['Troe_6_updated.yaml','Troe_6_abs_updated.yaml'],
                    ['Troe_7_updated.yaml','Troe_7_abs_updated.yaml'],
                    ['Troe_8_updated.yaml','Troe_8_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_0_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_1_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_2_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_3_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_4_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_5_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_6_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_7_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_8_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_9_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_10_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_11_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_12_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_13_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_14_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],
                    ['Hong_HO2_fake_data_15_updated.yaml','Hong_fake_data_fitted_abs_updated.yaml'],    
                    ['Farooq_0.yaml'],
                    ['Farooq_1.yaml'],
                    ['Farooq_2.yaml'],
                    ['Farooq_3.yaml']]   
                                                      
                                                      
numer_of_iterations = 1
cti_file = 'FFCM1_custom_extra_reaction_updated.cti'
working_directory = 'MSI/data/hong_H2O2_fake_data'
reaction_uncertainty_csv = 'FFCM1_reaction_uncertainty_extra_reaction.csv'
master_reaction_equation_cti_name = 'master_reactions_FFCM1_optimized_extra_reaction.cti'
#rate_constant_target_value_data = 'burke_target_value_single_reactions.csv'

#this would be an empty string '' if you do not want to include it 
run_with_k_target_values = 'On'
master_equation_reactions = ['H2O2 + OH <=> H2O + HO2',
                             '2 HO2 <=> H2O2 + O2',
                             'HO2 + OH <=> H2O + O2',
                             '2 OH <=> H2O + O',
                             'CH3 + HO2 <=> CH4 + O2',
                             'CH3 + HO2 <=> CH3O + OH']

#master_index = [2,3,4,5,6]
#master_index = [2,3,4,5,6,7]

#master_equation_uncertainty_df = pd.read_csv('MSI/data/test_data/six_parameter_fit_uncertainty_df.csv')
#this could be 'On'

#rate_constant_target_value_data = 'FFCM1_target_reactions_1_extra_reaction.csv'

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
                                    'CH3 + HO2 <=> CH4 + O2': {'A':np.array([.007845,-.89278,-.94908]),
                                                               'n':np.array([-0.00104,-.36888,.154462]),
                                                               'Ea':np.array([.504278,-.44379,-0.03181]),
                                                               'c':np.array([0,0,0]),
                                                               'd':np.array([0,0,0]),
                                                               'f':np.array([0,0,0])},
                                    'CH3 + HO2 <=> CH3O + OH': {'A':np.array([1.319108,-.92151]),
                                                                'n':np.array([-.04282,.150846]),
                                                                'Ea':np.array([0.024285,-0.02956]),
                                                                'c':np.array([0,0]),
                                                                'd':np.array([0,0]),
                                                                'f':np.array([0,0])}}



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
                                    'CH3 + HO2 <=> CH4 + O2': {'A':np.array([.007845,-.89278,-.94908]),
                                                               'n':np.array([-0.00104,-.36888,.154462]),
                                                               'Ea':np.array([.504278,-.44379,-0.03181])},
                                    'CH3 + HO2 <=> CH3O + OH': {'A':np.array([1.319108,-.92151]),
                                                                'n':np.array([-.04282,.150846]),
                                                                'Ea':np.array([0.024285,-0.02956])}} 






six_parameter_fit_nominal_parameters_dict = {'H2O2 + OH <=> H2O + HO2':{'A':4.64E-06,'n':5.605491008,'Ea':-5440.266692,'c':126875776.1,'d':0.000441194,'f':-5.35E-13},
                                '2 HO2 <=> H2O2 + O2':{'A':1.30E+04,'n':1.997152351,'Ea':-3628.04407,'c':93390973.44,'d':-0.000732521,'f':8.20E-12} ,
                                 'HO2 + OH <=> H2O + O2':{'A':1.41E+18,'n':-2.05344973,'Ea':-232.0064051,'c':15243859.12,'d':-0.001187694,'f':8.01E-12},
                                 '2 OH <=> H2O + O':{'A':354.5770856,'n':2.938741717,'Ea':-1836.492972,'c':12010735.18,'d':-4.87E-05,'f':1.22E-12},
                                 'CH3 + HO2 <=> CH4 + O2':{'A':3.19e3,'n':2.670857,'Ea':-4080.73,'c':0.0,'d':0.0,'f':0.0},
                                 'CH3 + HO2 <=> CH3O + OH':{'A':8.38e11,'n':.29,'Ea':-785.45,'c':0.0,'d':0.0,'f':0.0}}





MSI_st_instance_one = rswo.running_simulations_without_optimization(cti_file,
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
MSI_st_instance_one.multiple_shock_tube_runs(1)
experimental_dict_two_klip_reactions = MSI_st_instance_one.experiment_dictonaries