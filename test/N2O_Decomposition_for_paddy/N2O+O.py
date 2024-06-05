import sys, os
sys.path.append('C:\\Users\\pjsin\\Documents')
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import yaml
import glob
import os
from pypdf import PdfMerger
import natsort
import shutil
import MSI.optimization.optimization_shell_chebyshev as stMSIcheb
import MSI.utilities.plotting_script as plotter

plot_only = False
# mark = False

number_of_iterations = 2
step_size = 1

experiments=[ 
  
# ['allen_1_spe_1.yaml'],              
# ['allen_1_spe_2.yaml'],

# ['barbet_experiment_id1_0.yaml'],
# ['barbet_experiment_id1_1.yaml'],
# ['barbet_experiment_id2_0.yaml'],
# ['barbet_experiment_id2_1.yaml'],
# ['barbet_experiment_id3_0.yaml'],
# ['barbet_experiment_id3_1.yaml'],
# ['barbet_experiment_id4_0.yaml'],
# ['barbet_experiment_id4_1.yaml'],
# ['barbet_experiment_id5_0.yaml'],
# ['barbet_experiment_id5_1.yaml'],
# ['barbet_experiment_id6_0.yaml'],
# ['barbet_experiment_id6_1.yaml'],
# ['barbet_experiment_id7_0.yaml'],
# ['barbet_experiment_id7_1.yaml'],
# ['barbet_experiment_id8_0.yaml'],
# ['barbet_experiment_id8_1.yaml'],
# ['barbet_experiment_id9_0.yaml'],
# ['barbet_experiment_id9_1.yaml'],
# ['barbet_experiment_id10_0.yaml'],
# ['barbet_experiment_id10_1.yaml'],
# ['barbet_experiment_id11_0.yaml'],
# ['barbet_experiment_id11_1.yaml'],
# ['barbet_experiment_id12_0.yaml'],
# ['barbet_experiment_id12_1.yaml'],
# ['barbet_experiment_id13_0.yaml'],
# ['barbet_experiment_id13_1.yaml'],
# ['barbet_experiment_id14_0.yaml'],
# ['barbet_experiment_id14_1.yaml'],
# ['barbet_experiment_id15_0.yaml'],
# ['barbet_experiment_id15_1.yaml'],
# ['barbet_experiment_id16_0.yaml'],
# ['barbet_experiment_id16_1.yaml'],
# ['barbet_experiment_id17_0.yaml'],
# ['barbet_experiment_id17_1.yaml'],
# ['barbet_experiment_id18_0.yaml'],
# ['barbet_experiment_id18_1.yaml'],
# ['barbet_experiment_id19_0.yaml'],
# ['barbet_experiment_id19_1.yaml'],
# ['barbet_experiment_id20_0.yaml'],
# ['barbet_experiment_id20_1.yaml'],
# ['barbet_experiment_id21_0.yaml'],
# ['barbet_experiment_id21_1.yaml'],
# ['barbet_experiment_id22_0.yaml'],
# ['barbet_experiment_id22_1.yaml'],
# ['barbet_experiment_id23_0.yaml'],
# ['barbet_experiment_id23_1.yaml'],
# ['barbet_experiment_id24_0.yaml'],
# ['barbet_experiment_id24_1.yaml'],
# ['barbet_experiment_id25_0.yaml'],
# ['barbet_experiment_id25_1.yaml'],
# ['barbet_experiment_id26_0.yaml'],
# ['barbet_experiment_id26_1.yaml'],
# ['barbet_experiment_id27_0.yaml'],
# ['barbet_experiment_id27_1.yaml'],
# ['barbet_experiment_id28_0.yaml'],
# ['barbet_experiment_id28_1.yaml'],
# ['barbet_experiment_id29_0.yaml'],
# ['barbet_experiment_id29_1.yaml'],
# ['barbet_experiment_id30_0.yaml'],
# ['barbet_experiment_id30_1.yaml'],
# ['barbet_experiment_id31_0.yaml'],
# ['barbet_experiment_id31_1.yaml'],
# ['barbet_experiment_id32_0.yaml'],
# ['barbet_experiment_id32_1.yaml'],
# ['barbet_experiment_id33_0.yaml'],
# ['barbet_experiment_id33_1.yaml'],
# ['barbet_experiment_id34_0.yaml'],
# ['barbet_experiment_id34_1.yaml'],
# ['barbet_experiment_id35_0.yaml'],
# ['barbet_experiment_id35_1.yaml'],
# ['barbet_experiment_id36_0.yaml'],
# ['barbet_experiment_id36_1.yaml'],
# ['barbet_experiment_id37_0.yaml'],
# ['barbet_experiment_id37_1.yaml'],
# ['barbet_experiment_id38_0.yaml'],
# ['barbet_experiment_id38_1.yaml'],
# ['barbet_experiment_id39_0.yaml'],
# ['barbet_experiment_id39_1.yaml'],
# ['barbet_experiment_id40_0.yaml'],
# ['barbet_experiment_id40_1.yaml'],
# ['barbet_experiment_id41_0.yaml'],
# ['barbet_experiment_id41_1.yaml'],
# ['barbet_experiment_id42_0.yaml'],
# ['barbet_experiment_id42_1.yaml'],
# ['barbet_experiment_id43_0.yaml'],
# ['barbet_experiment_id43_1.yaml'],
# ['barbet_experiment_id44_0.yaml'],
# ['barbet_experiment_id44_1.yaml'],

['davidson_2_spe_1.yaml'],
# ['davidson_2_spe_2.yaml'],     

# ['glarborg_1_H2O_set1_unc01_res005.yaml'],
# ['glarborg_1_H2O_set2_unc01_res005.yaml'],
# ['glarborg_1_H2O_set3_unc01_res005.yaml'],
# ['glarborg_1_O2_set4_unc01_res005.yaml'],

# ['haas_1_spe.yaml'],

# ['johnsson_1_spe_Ar_unc03_res005.yaml'],
# ['johnsson_1_spe_He_unc03_res005.yaml'],
# ['johnsson_1_spe_N2_unc03_res005.yaml'],

# # ['li2021_spe1.yaml'],
# ['li2021_spe2.yaml'],
# ['li2021_spe3.yaml'],
# # ['li2021_spe4.yaml'],
# # ['li2021_spe5.yaml'],
# ['li2021_spe6.yaml'],
# ['li2021_spe7.yaml'],
# ['li2021_spe8.yaml'],

# ['mulvihill_2_spe_1.yaml'],
# ['mulvihill_2_spe_2.yaml'],
# ['mulvihill_2_spe_3.yaml'],
# ['mulvihill_2_spe_4.yaml'],
# ['mulvihill_2_spe_5.yaml'],
# ['mulvihill_2_spe_6.yaml'],
# ['mulvihill_2_spe_7.yaml'],
# ['mulvihill_2_spe_8.yaml'],
# ['mulvihill_2_spe_9.yaml'],
# ['mulvihill_2_spe_10.yaml'],
# ['mulvihill_2_spe_11.yaml'],
# ['mulvihill_2_spe_12.yaml'],
# ['mulvihill_2_spe_13.yaml'],
# ['mulvihill_2_spe_14.yaml'],
# ['mulvihill_2_spe_15.yaml'],

# ['pham_1_spe_N2O_1.yaml'],
# ['pham_1_spe_NO_2.yaml'],

# ['ross_1_exp_1.yaml','ross_1_abs.yaml'],
# ['ross_1_exp_2.yaml','ross_1_abs.yaml'],
# ['ross_1_exp_3.yaml','ross_1_abs.yaml'],

]

data_files_folder = 'MSI_N2O_Decomposition_mar_25_2024'
yaml_files_folder = 'MSI_N2O_Decomposition_mar_25_2024'

cti_file = 'reduced_glarborg_cheby_v31_Li_fit_H2O_v6.cti'
model_uncertainty_csv = 'uncertainties_reduced_glarborg_cheby_v31_Li_fit_v9.csv'
real_uncertainty_csv = 'real_uncertainties_reduced_glarborg.csv'
#everything above is required

X_prior_csv=''#Xdf_test711_Li.csv'#can be an empty string (opt arg) - usually is empty

rate_constant_plots_csv = 'rate_constant_plots.csv' #can be an empty string - usually we put it in. A list of all reactions that you want plotted. You can have targets wout plotting them (rarely the case)
rate_constant_target_csv = 'rate_constant_targets_final_shortened.csv' #can be an empty string (opt arg, but usually used)

master_equation_flag = True #if false, then everything until END OF MASTER EQUATION INPUTS is optional
master_reaction_equation_cti_name = 'master_equation_reduced_glarborg_cheby_v31_Li_fit_H2O_v6.cti' #only ME reactions in here
master_equation_uncertainty_csv = 'master_equation_parameter_uncertainties_final_no_W.csv' #only uncertainty of ME reactions in here
master_equation_reactions = ['N2O + O <=> 2 NO', 'N2O + O <=> N2 + O2']
master_index = [1, 2] # can be improved, where maybe do without indexing. But searching by name can cause errors.
T_P_min_max_dict = { #direct outputs from MSI_theory. Could be streamlined if this code were integrated with MSI_theory
                    'N2O + O <=> 2 NO':{'T_min':200,'T_max':4000,'P_min':101325e-4,'P_max':101325e2},
                    'N2O + O <=> N2 + O2':{'T_min':200,'T_max':4000,'P_min':101325e-4,'P_max':101325e2}       
                   }
cheb_sensitivity_dict = { #direct outputs from MSI_theory. Could be streamlined if this code were integrated with MSI_theory
         'N2O + O <=> 2 NO': [np.array([[ 9.99997911e-01, 0],[-1.01323276e-07, 0],[-2.43448265e-06, 0],[-3.21178513e-06, 0],[ 4.01331348e-06, 0],[ 3.68635879e-07, 0],[ 7.74801663e-07, 0]]),
                              np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                              np.array([[-1.30715736, 0],[ 1.21798971, 0],[ 0.01299371, 0],[ 0.00319519, 0],[-0.00221091, 0],[-0.00290965, 0],[-0.00308971, 0]]), 
                              np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                              np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                              np.array([[ 0.00308724, 0],[ 0.00413944, 0],[ 0.000795  , 0],[-0.00155552, 0],[-0.0018208 , 0],[-0.00114674, 0],[-0.00041149, 0]]), 
                              np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                              np.array([[ 3.61558421, 0],[-5.56611341, 0],[ 2.73284833, 0],[-1.08262976, 0],[ 0.39631728, 0],[-0.1083588 , 0],[ 0.01747353, 0]]), 
                              np.array([[-0.19123555, 0],[-0.35820002, 0],[-0.28718553, 0],[-0.20150148, 0],[-0.12114503, 0],[-0.05879841, 0],[-0.02918919, 0]]), 
                              np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                              np.array([[-1.39336293, 0],[-1.24590988, 0],[-0.24544511, 0],[ 0.07429575, 0],[ 0.11034933, 0],[ 0.06856676, 0],[ 0.03781361, 0]]),
                              np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                              np.array([[-0.9394612 , 0],[ 0.11067446, 0],[ 0.0841814 , 0],[ 0.05336743, 0],[ 0.0276621 , 0],[ 0.011345  , 0],[ 0.00313567, 0]]), 
                              np.array([[1.01926497, 0],[1.3243695 , 0],[0.55491603, 0],[0.19588425, 0],[0.06565789, 0],[0.01966359, 0],[0.00419789, 0]]), 
                              np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                              np.array([[-0.05628151, 0],[-0.10304893, 0],[-0.07865591, 0],[-0.05026884, 0],[-0.0264133 , 0],[-0.01104112, 0],[-0.00338527, 0]]),
                              np.array([[-0.01344154, 0],[-0.0223235 , 0],[-0.0129225 , 0],[-0.00341346, 0],[ 0.0019651 , 0],[ 0.00274956, 0],[ 0.00304885, 0]]),
                              np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                              np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                              np.array([[ 1.32094890e+00, 0],[-1.19514332e+00, 0],[-2.11446739e-06, 0],[-4.00423597e-06, 0],[-9.65727234e-07, 0],[-3.09878781e-07, 0],[ 8.39936271e-06, 0]])],


         'N2O + O <=> N2 + O2': [np.array([[ 9.99997879e-01, 0],[-3.61286169e-07, 0],[-3.03642267e-06, 0],[ 6.55656176e-06, 0],[-3.81333850e-06, 0],[ 3.48158493e-06, 0],[-2.84335608e-06, 0]]),
                                 np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                                 np.array([[-1.31876293, 0],[ 1.19886558, 0],[ 0.003977, 0],[ 0.00303719, 0],[ 0.00272921, 0],[ 0.00148647, 0],[ 0.00137597, 0]]), 
                                 np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                                 np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                                 np.array([[ 0.54643617, 0],[-1.02637883, 0],[ 0.88588513, 0],[-0.68204939, 0],[ 0.46004548, 0],[-0.26068762, 0],[ 0.11299838, 0]]), 
                                 np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                                 np.array([[ 1.73086384e+01, 0],[-2.63837374e+01, 0],[ 1.08396730e+01, 0],[-4.76832610e-01, 0],[-2.37901520e+00, 0],[ 1.31391918e+00, 0],[-6.00821865e-04, 0]]), 
                                 np.array([[-0.04209322, 0],[-0.09017349, 0],[-0.08556649, 0],[-0.08341831, 0],[-0.07345397, 0],[-0.04552558, 0],[-0.05022567, 0]]), 
                                 np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                                 np.array([[-1.96597117, 0],[-1.53943933, 0],[-0.34477812, 0],[-0.00710602, 0],[ 0.06694799, 0],[ 0.05455289, 0],[ 0.06927936, 0]]), 
                                 np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                                 np.array([[-0.9876543 , 0],[ 0.02605909, 0],[ 0.02475294, 0],[ 0.02372434, 0],[ 0.02073346, 0],[ 0.01279018, 0],[ 0.01377738, 0]]), 
                                 np.array([[1.01926856, 0],[1.3243601 , 0],[0.55491831, 0],[0.19589042, 0],[0.06566142, 0],[0.01966427, 0],[0.00419393, 0]]), 
                                 np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                                 np.array([[-0.01132161, 0],[-0.02391362, 0],[-0.02272851, 0],[-0.02180434, 0],[-0.01909096, 0],[-0.01176643, 0],[-0.01273426, 0]]), 
                                 np.array([[-0.00211581, 0],[-0.00373376, 0],[-0.00385928, 0],[-0.00303589, 0],[-0.00267619, 0],[-0.0014784 , 0],[-0.00136935, 0]]), 
                                 np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                                 np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]), 
                                 np.array([[ 1.32095123e+00, 0],[-1.19515092e+00, 0],[ 2.10758453e-06, 0],[ 2.78652801e-06, 0],[ 1.56078723e-06, 0],[ 2.03998112e-06, 0],[-1.67535313e-06, 0]])]
                    }

#END OF MASTER EQUATION INPUTS

if plot_only == False:

    #start structuring directory
    main_directory = os.path.dirname(os.path.abspath(__file__)) 
    test_directory = os.path.join(main_directory,'test')
    test_directory_list = [f for f in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory,f))]
    test_number_list = [eval(f.split('test')[1]) for f in test_directory_list if f.split('test')[1].isnumeric()]
    current_test_directory =  'test' + str(max(test_number_list)+1)
    working_directory =  os.path.join(test_directory, current_test_directory) # /home/jl/main/test/test#
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    os.chdir(working_directory)
    matrix_path = os.path.join(working_directory,'matrix')
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path)
    out_path = os.path.join(working_directory,'out')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    data_path = os.path.join(working_directory,'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    yaml_path = os.path.join(working_directory,'yaml')
    if not os.path.exists(yaml_path):
        os.makedirs(yaml_path)
    
    data_files = os.listdir(os.path.join(os.path.join(main_directory,'data'),data_files_folder))
    for fname in data_files:
        shutil.copy2(os.path.join(os.path.join(os.path.join(main_directory,'data'),data_files_folder), fname), data_path)

    yaml_files = os.listdir(os.path.join(os.path.join(main_directory,'yaml'),yaml_files_folder))
    for fname in yaml_files:
        shutil.copy2(os.path.join(os.path.join(os.path.join(main_directory,'yaml'),yaml_files_folder), fname), yaml_path)

    shutil.copy2(os.path.join(main_directory, os.path.basename(__file__)), working_directory)
    shutil.move(os.path.join(working_directory,os.path.basename(__file__)), os.path.join(working_directory,os.path.splitext(os.path.basename(__file__))[0]+'_'+current_test_directory+'.txt'))
    shutil.copy2(os.path.join(os.path.join(main_directory,'model'), cti_file), working_directory)
    shutil.copy2(os.path.join(os.path.join(main_directory,'model'), model_uncertainty_csv), working_directory)

    if bool(real_uncertainty_csv):
        shutil.copy2(os.path.join(os.path.join(main_directory,'model'), real_uncertainty_csv), working_directory)
    if bool(X_prior_csv):
        shutil.copy2(os.path.join(os.path.join(main_directory,'model'), X_prior_csv), working_directory)
    if bool(master_reaction_equation_cti_name):
        shutil.copy2(os.path.join(os.path.join(main_directory,'me'), master_reaction_equation_cti_name), working_directory)
    if bool(master_equation_uncertainty_csv):
        shutil.copy2(os.path.join(os.path.join(main_directory,'me'), master_equation_uncertainty_csv), working_directory)
    if bool(rate_constant_target_csv):
        shutil.copy2(os.path.join(os.path.join(main_directory,'rc'), rate_constant_target_csv), working_directory)
    if bool(rate_constant_plots_csv):
        shutil.copy2(os.path.join(os.path.join(main_directory,'rc'), rate_constant_plots_csv), working_directory)

    files_to_include = [[os.path.join('yaml',fti) for fti in fti_list] for fti_list in experiments]

    if bool(master_equation_flag):
        master_equation_df = pd.read_csv(os.path.join(working_directory,master_equation_uncertainty_csv))
        ncolumns = len(master_equation_df.columns)
        theory_parameters_df = pd.read_csv(os.path.join(working_directory,master_equation_uncertainty_csv), usecols=[x for x in range(0, ncolumns, 2)])
        master_equation_uncertainty_df = pd.read_csv(os.path.join(working_directory,master_equation_uncertainty_csv), usecols=[x for x in range(1, ncolumns, 2)])
        master_equation_uncertainty_df.columns = list(theory_parameters_df.columns)
    else:
        theory_parameters_df = None
        master_equation_uncertainty_df = None
    #stop structuring directory


    MSI_st_instance = stMSIcheb.MSI_optimization_chebyshev( #initialize MSI
                                                        working_directory,
                                                        cti_file,
                                                        files_to_include,
                                                        model_uncertainty_csv,

                                                        k_target_values_csv = rate_constant_target_csv,
                                                        
                                                        master_equation_flag = master_equation_flag,
                                                        master_equation_reactions = master_equation_reactions,
                                                        master_index = master_index,
                                                        T_P_min_max_dict = T_P_min_max_dict,
                                                        chebyshev_sensitivities = cheb_sensitivity_dict,
                                                        master_reaction_equation_cti_name = master_reaction_equation_cti_name,
                                                        master_equation_uncertainty_df = master_equation_uncertainty_df,
                                                        theory_parameters_df = theory_parameters_df,

                                                        step_size=step_size,
                                                        X_prior_csv=X_prior_csv
                                                        )
        
    MSI_st_instance.multiple_runs(number_of_iterations) #run MSI

    #start saving MSI outputs
    exp_dict_list_original = MSI_st_instance.experiment_dictonaries_original # multi-level dictionaries, so these aren't easily saved into files. Currently we're not doing this, so the plot_only==True option skips over this.
    parsed_yaml_list_original = MSI_st_instance.list_of_parsed_yamls_original # multi-level dictionaries, so these aren't easily saved into files. Currently we're not doing this, so the plot_only==True option skips over this.

    Ydf_prior = MSI_st_instance.Ydf_prior
    Sdf_prior = MSI_st_instance.Sdf_prior
    covdf_prior = MSI_st_instance.covdf_prior
    sigdf_prior = MSI_st_instance.sigdf_prior

    exp_dict_list_optimized = MSI_st_instance.experiment_dictonaries # multi-level dictionaries, so these aren't easily saved into files. Currently we're not doing this, so the plot_only==True option skips over this.
    parsed_yaml_list_optimized = MSI_st_instance.list_of_parsed_yamls # multi-level dictionaries, so these aren't easily saved into files. Currently we're not doing this, so the plot_only==True option skips over this.

    Xdf = MSI_st_instance.Xdf
    Ydf = MSI_st_instance.Ydf
    ydf = MSI_st_instance.ydf
    Zdf = MSI_st_instance.Zdf
    Sdf = MSI_st_instance.Sdf
    sdf = MSI_st_instance.sdf
    covdf = MSI_st_instance.covdf
    sigdf = MSI_st_instance.sigdf
    
    convergence_sorted = MSI_st_instance.convergence_sorted

    Xdf_list = MSI_st_instance.Xdf_list
    active_parameters = MSI_st_instance.active_parameters
    target_parameters = MSI_st_instance.target_parameters
    #stop saving MSI outputs

else: #if plot_only==True, then optimization is skipped and only the outputs of (a previous, saved) optimization are being plotted
    # find where previous optimization results are saved
    main_directory = os.path.dirname(__file__) # /home/jl/main
    test_directory = os.path.join(main_directory,'test') # /home/jl/main/test
    current_test_directory =  'test' + str(test) #specify the test number
    working_directory =  os.path.join(test_directory, current_test_directory) # /home/jl/main/test/test#
    matrix_path = os.path.join(working_directory,'matrix')
    out_path = os.path.join(working_directory,'out')
    data_path = os.path.join(working_directory,'data')
    yaml_path = os.path.join(working_directory,'yaml')

    Ydf_prior = pd.read_csv(os.path.join(matrix_path,'Ydf_prior.csv'), index_col=0)
    Sdf_prior = pd.read_csv(os.path.join(matrix_path,'Sdf_prior.csv'), index_col=0)
    covdf_prior = pd.read_csv(os.path.join(matrix_path,'covdf_prior.csv'), index_col=0)
    sigdf_prior = pd.read_csv(os.path.join(matrix_path,'sigdf_prior.csv'), index_col=0)

    Xdf = pd.read_csv(os.path.join(matrix_path,'Xdf.csv'), index_col=0)
    Ydf = pd.read_csv(os.path.join(matrix_path,'Ydf.csv'), index_col=0)
    ydf = pd.read_csv(os.path.join(matrix_path,'ydf.csv'), index_col=0)
    Zdf = pd.read_csv(os.path.join(matrix_path,'Zdf.csv'), index_col=0) 
    Sdf = pd.read_csv(os.path.join(matrix_path,'Sdf.csv'), index_col=0)
    sdf = pd.read_csv(os.path.join(matrix_path,'sdf.csv'), index_col=0)
    covdf = pd.read_csv(os.path.join(matrix_path,'covdf.csv'), index_col=0)
    sigdf = pd.read_csv(os.path.join(matrix_path,'sigdf.csv'), index_col=0)

    X_over_sig_sorted = pd.read_csv(os.path.join(matrix_path,'X_over_sigma.csv'), index_col=0)
    convergence_sorted = pd.read_csv(os.path.join(matrix_path,'convergence.csv'), index_col=0)

    Xdf_list = [pd.read_csv(Xdf_csv, index_col=0) for i, Xdf_csv in enumerate(natsort.natsorted(set(glob.glob(os.path.join(matrix_path, "Xdf_*.csv")))))]

    active_parameters = Xdf.index.to_list()
    target_parameters = Ydf.index.to_list()

#the variables in the "saving MSI outputs" section are now plotted here
plotting_instance = plotter.Plotting(
                                     Ydf_prior, Sdf_prior, covdf_prior, sigdf_prior,
                                                                               
                                     Xdf, Ydf, ydf, Zdf, Sdf, sdf, covdf, sigdf,
                                     
                                     Xdf_list, active_parameters, target_parameters,
                                     
                                     parsed_yaml_list_original,
                                     exp_dict_list_original,
                                     parsed_yaml_list_optimized,
                                     exp_dict_list_optimized,
                                    
                                     working_directory = working_directory,
                                     number_of_iterations = number_of_iterations,
                                     files_to_include = files_to_include,
                                     T_P_min_max_dict = T_P_min_max_dict,
                                     master_equation_reactions = master_equation_reactions,
                                     cheby_sensitivity_dict = cheb_sensitivity_dict,
                                    
                                     target_value_rate_constant_csv= rate_constant_target_csv,
                                     rate_constant_plots_csv = rate_constant_plots_csv,
                                     master_equation_flag = master_equation_flag,
                                     theory_parameters_df = theory_parameters_df,
                                     real_uncertainty_csv = real_uncertainty_csv,
                                    
                                     k_target_value_S_matrix = MSI_st_instance.k_target_values_for_S,
                                    
                                     original_cti_file = os.path.join(working_directory, cti_file),
                                     optimized_cti_file = MSI_st_instance.new_cti_file,
                                    
                                     sigma_ones = False,     
                                     simulation_run=None,
                                     shock_tube_instance = None,                                                     
                                    
                                     pdf = True, png = False, svg = False, dpi = 10) #add these all as inputs to yaml file

plotting_instance.plotting_convergence(convergence_sorted, 50) #plots the deltaX for the top 50 parameters. Can make "50" an adjustable input, to be added in yaml

plotting_instance.plotting_observables()
plotting_instance.plotting_uncertainty_weighted_sens()
plotting_instance.plotting_Sdx()
plotting_instance.merge_observable_pdfs()

if bool(rate_constant_plots_csv): #this is skipped if rate constant plots is an empty string
    plotting_instance.plotting_rate_constants_combined_channels()    
    plotting_instance.plotting_uncertainty_weighted_sens_rate_constant()
    plotting_instance.plotting_Sdx_rate_constant()
    plotting_instance.merge_rate_constant_pdfs()