
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
import pandas as pd
    

files_to_include = [['Masten_0.yaml'],
                    ['Masten_1.yaml'],
                    ['Masten_2.yaml'],
                    ['Masten_3.yaml'],
                    ['Masten_4.yaml'],
                    ['Masten_5.yaml'],
                    ['Masten_6.yaml'],
                    ['Masten_7.yaml'],
                    ['Masten_8.yaml'],
                    ['Masten_9.yaml'],
                    ['Masten_10.yaml'],
                    ['Masten_11.yaml'],
                    ['Masten_12.yaml'],
                    ['Masten_13.yaml'],
                    ['Masten_14.yaml'],
                    ['Masten_15.yaml'],
                    ['Masten_16.yaml'],
                    ['Masten_17.yaml'],
                    ['Masten_18.yaml'],
                    ['Masten_19.yaml'],
                    ['Masten_20.yaml'],
                    ['Masten_21.yaml'],
                    ['Masten_22.yaml'],
                    ['Masten_23.yaml'],
                    ['Masten_24.yaml'],
                    ['Masten_25.yaml'],
                    ['Masten_26.yaml'],
                    ['Masten_27.yaml'],
                    ['Masten_28.yaml']] 


#files_to_include = [['Masten_9.yaml']] 

for i,file in enumerate(files_to_include):
    percent = 100
    original_sigma_value = .9
    counter = 0
    while percent > 10.0:
        files_to_include = [file]                                                       
                                     
        numer_of_iterations = 2                                         
        cti_file = 'masten_paper_gri_thermo.cti'
        working_directory = 'MSI/data/automating_HO2_masten'
        reaction_uncertainty_csv = 'reaction_uncertainty_masten_differnt_sigmas_for_A1.csv'
        
        #rate_constant_target_value_data = 'burke_target_value_single_reactions.csv'
        rate_constant_target_value_data = 'target_reactions_test.csv'
        #this would be an empty string '' if you do not want to include it 
        run_with_k_target_values = 'On'
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
        covarience = MSI_st_instance_two.covarience
        exp_dict_list_optimized = MSI_st_instance_two.experiment_dictonaries
        parsed_yaml_list = MSI_st_instance_two.list_of_parsed_yamls
        sigma = MSI_st_instance_two.sigma
        X = MSI_st_instance_two.X
        delta_X = MSI_st_instance_two.delta_X
        #target_value_rate_constant_csv = 'MSI/data/test_data/FFCM1_custom_target_value_test.csv'
        original_cti_file = MSI_st_instance_two.data_directory +'/'+ MSI_st_instance_two.cti_file_name
        covarience = MSI_st_instance_two.covarience
        experiment_dict_uncertainty = MSI_st_instance_two.experiment_dict_uncertainty_original
        target_value_csv = MSI_st_instance_two.data_directory +'/'+ MSI_st_instance_two.k_target_values_csv
        
        if run_with_k_target_values == 'On' or run_with_k_target_values == 'on':
            k_target_value_S_matrix = MSI_st_instance_two.k_target_values_for_s
        else:
            k_target_value_S_matrix = None
    
    
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
        
        
        posterior_A1 = diag[1]
        optimized_cti_file=MSI_st_instance_two.new_cti_file
        gas_optimized = ct.Solution(optimized_cti_file)
        A1=gas_optimized.reaction(1).rate.pre_exponential_factor
        gas_original = ct.Solution(MSI_st_instance_two.data_directory +'/'+ MSI_st_instance_two.cti_file_name)
        A1=gas_original.reaction(1).rate.pre_exponential_factor

        percent = np.sqrt(posterior_A1)/np.log((A1))
        percent = percent *100
        print('THIS IS THE PERCENT:',percent)
        def sigma_adjuster(sigma,file_range):
            for file in range(file_range):
                df = pd.read_csv(working_directory+'/'+'masten_oh_'+str(file)+'.csv')
                shape = df.shape[0]
                new_sigma = np.array([sigma]*shape)
                new_sigma = new_sigma.reshape((new_sigma.shape[0],1))
                new_sigma = pd.DataFrame(new_sigma,columns=['Relative_Uncertainty'])
                df['Relative_Uncertainty'] = new_sigma['Relative_Uncertainty']
                df.to_csv(working_directory+'/'+'masten_oh_'+str(file)+'.csv', index = False)
        counter +=1
        sigma_adjuster(original_sigma_value-(.01*counter),len(files_to_include))
