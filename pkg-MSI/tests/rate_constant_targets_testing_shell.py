
import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built
import pandas as pd 
import MSI.optimization.shock_tube_optimization_shell as stMSI
import MSI.utilities.testing_class as testing_class 
import MSI.utilities.add_target_values_testing_shock_tube_shell as tvtst
import MSI.utilities.plotting_script as plotter


X_list_new = []
run_with_k_target_values = 'On'


macroscopic_test_data_directory = 'MSI/data/test_data/rate_constant_test_data'
 
testing_class_instance = testing_class.testing_code(macroscopic_test_data_directory=macroscopic_test_data_directory)
file_list,folder_list = testing_class_instance.get_files_in_working_directory()

Y_difference_list = []
Reaction = []
Sigma_optimized_list = []
Sigma_original_list = []
for i,files in enumerate(file_list):

    
    yaml_file,cti_file,reaction_uncertainty_file,absorption_yaml_file,k_target_value_file = testing_class_instance.sorting_files(files)
    if absorption_yaml_file == None:
        MSI_st_instance_one = stMSI.MSI_shocktube_optimization(cti_file,
                                                           .01,
                                                           1,
                                                           1,
                                                           macroscopic_test_data_directory+'/'+folder_list[i],
                                                           [[yaml_file]],                 
                                                           reaction_uncertainty_file,k_target_value_file )
        MSI_st_instance_one.one_run_shock_tube_optimization()
    else:
        MSI_st_instance_one = stMSI.MSI_shocktube_optimization(cti_file,
                                                           .01,
                                                           1,
                                                           1,
                                                           macroscopic_test_data_directory+'/'+folder_list[i],
                                                           [[yaml_file,absorption_yaml_file]],                 
                                                           reaction_uncertainty_file,k_target_value_file )
        MSI_st_instance_one.one_run_shock_tube_optimization()        
        
    
    S_matrix_original = MSI_st_instance_one.S_matrix
    exp_dict_list_original = MSI_st_instance_one.experiment_dictonaries
    original_covariance = MSI_st_instance_one.covarience
    X_one_itteration = MSI_st_instance_one.X
    
# might not     
    
    if absorption_yaml_file == None:
        MSI_st_instance_two = stMSI.MSI_shocktube_optimization(cti_file,
                                                               .01,
                                                               1,
                                                               1,
                                                               macroscopic_test_data_directory+'/'+folder_list[i],
                                                               [[yaml_file]],                 
                                                               reaction_uncertainty_file,k_target_value_file )
        
        X_list = MSI_st_instance_two.multiple_shock_tube_runs(100)
        #add multiple optimization runs 
        
    else:
        MSI_st_instance_two = stMSI.MSI_shocktube_optimization(cti_file,
                                                           .01,
                                                           1,
                                                           1,
                                                           macroscopic_test_data_directory+'/'+folder_list[i],
                                                           [[yaml_file,absorption_yaml_file]],                 
                                                           reaction_uncertainty_file,k_target_value_file )
        
        
        X_list = MSI_st_instance_two.multiple_shock_tube_runs(100)
        
        
        
    
    
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
        k_target_value_S_matrix =None
    
    
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
                                         target_value_rate_constant_csv= MSI_st_instance_two.data_directory +'/'+k_target_value_file ,
                                         k_target_value_S_matrix =k_target_value_S_matrix,
                                         k_target_values=run_with_k_target_values)
    
    observable_counter_and_absorbance_wl,length_of_experimental_data = plotting_instance.lengths_of_experimental_data()
    sigmas_optimized,test = plotting_instance.calculating_sigmas(S_matrix,covarience)
    sigmas_original,test2 = plotting_instance.calculating_sigmas(S_matrix_original,original_covariance)
    
    
    
    plotting_instance.plotting_rate_constants(optimized_cti_file=MSI_st_instance_two.new_cti_file,
                                    original_cti_file=original_cti_file,
                                    initial_temperature=250,
                                    final_temperature=2500)
    sigma_optimized = plotting_instance.sigma_list_for_target_ks_optimized
    sigma_original = plotting_instance.sigma_list_for_target_ks_original
    Y_difference_list.append(Ydf.iloc[Ydf.shape[0]-1]['ln_difference'])
    Reaction.append(Ydf.iloc[Ydf.shape[0]-1]['value']+'_'+str(i))
    Sigma_optimized_list.append(sigma_optimized[0][0])
    Sigma_original_list.append(sigma_original[0][0])
df = pd.DataFrame({'Reaction':Reaction,'Y_difference':Y_difference_list,'Sigma_optimized':Sigma_optimized_list,'Sigma_original_list':Sigma_original_list})