
import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.batch_reactor as st
import MSI.cti_core.cti_processor as pr
import MSI.optimization.matrix_loader as ml
import MSI.optimization.opt_runner as opt
import MSI.simulations.absorbance.curve_superimpose as csp
import MSI.simulations.yaml_parser as yp
import MSI.master_equation.master_equation_six_parameter_fit as mespf
import MSI.cti_core.cti_combine as ctic
import copy
import cantera as ct
import numpy as np
import pandas as pd 
import os


class MSI_optimization_six_parameter_fit(object):
        
    def __init__(self, cti_file:str,perturbment:int,
                 kineticSens:int,physicalSens:int,
                 data_directory:str,yaml_file_list:list,
                 reaction_uncertainty_csv:str,
                 k_target_values_csv:str,
                 master_equation_reactions:list=[],
                 molecular_parameter_sensitivities:dict={},
                 six_parameter_fit_sensitivities:dict={},
                 master_reaction_equation_cti_name:str = '',
                 master_index = [],
                 master_equation_uncertainty_df = None,
                 six_paramter_fit_nominal_parameters_dict = None):
        
        self.cti_file_name = cti_file
        copy.deepcopy(self.cti_file_name)
        self.perturbment = perturbment
        self.kineticSens = kineticSens
        self.physicalSens = physicalSens
        self.data_directory = data_directory
        self.yaml_file_list = yaml_file_list
        self.yaml_file_list_with_working_directory = None
        self.processor = None
        self.list_of_yaml_objects = None
        self.list_of_parsed_yamls = None
        self.experiment_dictonaries = None
        self.reaction_uncertainty_csv = reaction_uncertainty_csv
        self.k_target_values_csv = k_target_values_csv
        self.master_equation_reactions = master_equation_reactions
        self.MP_for_S_matrix = np.array(())
        if bool(self.master_equation_reactions):
            self.master_equation_flag = True
            self.master_reaction_equation_cti_name = master_reaction_equation_cti_name
            self.master_index = master_index
            self.master_equation_uncertainty_df = master_equation_uncertainty_df
            self.six_paramter_fit_nominal_parameters_dict = six_paramter_fit_nominal_parameters_dict
            self.six_parameter_fit_sensitivities = six_parameter_fit_sensitivities
        else:
            self.master_equation_flag=False
            self.master_equation_uncertainty_df=None
        self.molecular_parameter_sensitivities = molecular_parameter_sensitivities
       
        
    # call all of leis functions where we do the molecular paramter stuff and turn a flag on 
    
    def append_working_directory(self):
        yaml_file_list_with_working_directory = []
        for i, file_set in enumerate(self.yaml_file_list):
            temp = []
            for j,file in enumerate(self.yaml_file_list[i]):
                ##############################################################################################
                #Fix for OS compatibility later
                ##############################################################################################
                temp.append(os.path.join(self.data_directory,file))
            temp = tuple(temp)
            yaml_file_list_with_working_directory.append(temp)
        self.yaml_file_list_with_working_directory = yaml_file_list_with_working_directory
        return
# pre process the cti file to remove reactions and rename it,also save it as the first run of the file
        
    # run the cti writer to establish processor and make cti file 
    def establish_processor(self,loop_counter=0):
        if loop_counter==0 and self.master_equation_flag==False:
            new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(original_cti=os.path.join(self.data_directory,self.cti_file_name),
                                                                      working_directory=self.data_directory,
                                                                      file_name= self.cti_file_name.replace('.cti','')+'_updated')
            self.new_cti_file = new_file
             
        if loop_counter==0 and self.master_equation_flag==True:
            new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(original_cti=os.path.join(self.data_directory,self.cti_file_name),
                                                                      master_rxns = os.path.join(self.data_directory,self.master_reaction_equation_cti_name),
                                                                      master_index = self.master_index,
                                                                      working_directory=self.data_directory,
                                                                      file_name= self.cti_file_name.replace('.cti','')+'_updated')
            self.new_cti_file = new_file
            
        processor = pr.Processor(self.new_cti_file)
        #processor = pr.Processor(self.data_directory +'/'+ self.cti_file_name)
        self.processor = processor
        return 
    
    def parsing_yaml_files(self,loop_counter=0,list_of_updated_yamls=[]):
        if loop_counter==0:
            yaml_instance = yp.Parser()
        else:
            yaml_instance = yp.Parser(original_experimental_conditions=self.original_experimental_conditions_local)
            #print(self.original_experimental_conditions_local[0]['coupledCoefficients'],'other copy')
            
        self.yaml_instance = yaml_instance
        if loop_counter ==0:
            list_of_yaml_objects = yaml_instance.load_yaml_list(yaml_list=self.yaml_file_list_with_working_directory)
            self.list_of_yaml_objects = list_of_yaml_objects
            list_of_parsed_yamls = yaml_instance.parsing_multiple_dictonaries(list_of_yaml_objects = list_of_yaml_objects,loop_counter=loop_counter)
            list_of_parsed_yamls_original = copy.deepcopy(list_of_parsed_yamls)
            self.list_of_parsed_yamls_original = list_of_parsed_yamls_original
            self.list_of_parsed_yamls = list_of_parsed_yamls_original
            
        else:
            list_of_yaml_objects = yaml_instance.load_yaml_list(yaml_list=self.updated_yaml_file_name_list)            
            self.list_of_yaml_objects = list_of_yaml_objects
            list_of_parsed_yamls = yaml_instance.parsing_multiple_dictonaries(list_of_yaml_objects = list_of_yaml_objects,loop_counter=loop_counter)
            self.list_of_parsed_yamls = list_of_parsed_yamls
            
            
        
        return
    
    def running_simulations(self,loop_counter=0):
        optimization_instance = opt.Optimization_Utility()
        if loop_counter == 0:
            experiment_dictonaries = optimization_instance.looping_over_parsed_yaml_files(self.list_of_parsed_yamls,
                                              self.yaml_file_list_with_working_directory ,
                                              processor=self.processor, 
                                              kineticSens=self.kineticSens,
                                              physicalSens=self.physicalSens,
                                              dk=self.perturbment)
            
            experiment_dict_uncertainty_original = optimization_instance.saving_experimental_dict(experiment_dictonaries)
            
            
            self.experiment_dict_uncertainty_original = copy.deepcopy(experiment_dict_uncertainty_original)
            
            #call function taht loops opver experient dicts og and saves them
            
        else:
            
            experiment_dictonaries = optimization_instance.looping_over_parsed_yaml_files(self.list_of_parsed_yamls,
                                              self.updated_yaml_file_name_list ,
                                              processor=self.processor, 
                                              kineticSens=self.kineticSens,
                                              physicalSens=self.physicalSens,
                                              dk=self.perturbment)
    
        
           
        self.experiment_dictonaries = experiment_dictonaries
        #maybe save this and just pass it in 
        return
    def master_equation_s_matrix_building(self,loop_counter=0):
        master_equation_six_param_fit_instance = mespf.Master_Equation_Six_Parameter_Fit()
        self.master_equation_six_param_fit_instance = master_equation_six_param_fit_instance
        
        MP_for_S_matrix = master_equation_six_param_fit_instance.master_equation_handling(self.experiment_dictonaries,
                                                                                          self.list_of_parsed_yamls,
                                                                                          self.molecular_parameter_sensitivities,
                                                                                          self.master_equation_reactions)

        self.MP_for_S_matrix = MP_for_S_matrix
        return
        
    def building_matrices(self,loop_counter=0):
        matrix_builder_instance = ml.OptMatrix()
        self.matrix_builder_instance = matrix_builder_instance
        S_matrix = matrix_builder_instance.load_S(self.experiment_dictonaries,
                                                  self.list_of_parsed_yamls,
                                                  dk=self.perturbment,
                                                  master_equation_reactions = self.master_equation_reactions,
                                                  mapped_master_equation_sensitivities=self.MP_for_S_matrix,
                                                  master_equation_flag = self.master_equation_flag)
        self.S_matrix = S_matrix

        
        
        if loop_counter == 0:
            Y_matrix,Y_data_frame = matrix_builder_instance.load_Y(self.experiment_dictonaries,
                                                                   self.list_of_parsed_yamls,
                                                                   loop_counter=loop_counter,
                                                                   master_equation_flag = self.master_equation_flag,
                                                                   master_equation_uncertainty_df = self.master_equation_uncertainty_df,
                                                                   master_equation_reactions = self.master_equation_reactions)
        else:
            Y_matrix,Y_data_frame = matrix_builder_instance.load_Y(self.experiment_dictonaries,
                                                                   self.list_of_parsed_yamls,
                                                                   loop_counter=loop_counter,
                                                                   X=self.X_to_subtract_from_Y,
                                                                   master_equation_flag = self.master_equation_flag,
                                                                   master_equation_uncertainty_df = self.master_equation_uncertainty_df,
                                                                   master_equation_reactions = self.master_equation_reactions)    
            
        self.Y_matrix = Y_matrix
        self.Y_data_frame = Y_data_frame
        
        z_matrix,z_data_frame,sigma,active_parameters = matrix_builder_instance.build_Z(self.experiment_dictonaries,
                                                                      self.list_of_parsed_yamls,
                                                                       loop_counter=loop_counter,
                                                                       reaction_uncertainty = self.data_directory +'/'+self.reaction_uncertainty_csv,
                                                                       master_equation_uncertainty_df=self.master_equation_uncertainty_df,
                                                                       master_equation_flag = self.master_equation_flag,
                                                                       master_equation_reaction_list = self.master_equation_reactions)
        self.z_matrix = z_matrix
        self.z_data_frame = z_data_frame
        self.sigma = sigma
        self.active_parameters = active_parameters
        return
    
    
    #test up to here first 
    
    def defining_six_parameter_fit_dictonary(self):
        
        
        return 
    def adding_k_target_values(self,loop_counter=0):
        
        ### add dataframe  start hete  
        k_target_values_for_z,sigma_target_values,z_data_frame = self.master_equation_six_param_fit_instance.target_values_for_Z_six_paramter_fit(os.path.join(self.data_directory,self.k_target_values_csv),
                                                                                                                            self.z_data_frame)
        if loop_counter == 0:
            k_target_values_for_Y,Y_data_frame = self.master_equation_six_param_fit_instance.target_values_Y_six_parameter_fit(os.path.join(self.data_directory,self.k_target_values_csv),
                                                                                                                self.experiment_dictonaries,
                                                                                                                self.Y_data_frame,
                                                                                                                master_equation_reaction_list=self.master_equation_reactions,
                                                                                                                updated_six_paramter_fits_dict=self.six_paramter_fit_nominal_parameters_dict)
        else:
            k_target_values_for_Y,Y_data_frame = self.master_equation_six_param_fit_instance.target_values_Y_six_parameter_fit(os.path.join(self.data_directory,self.k_target_values_csv),
                                                                                                                self.experiment_dictonaries,
                                                                                                                self.Y_data_frame,
                                                                                                                master_equation_reaction_list=self.master_equation_reactions,
                                                                                                                updated_six_paramter_fits_dict=self.updated_six_parameter_fits_dict)        
        

        
        
        k_target_values_for_S = self.master_equation_six_param_fit_instance.target_values_for_S_six_parameter_fit(os.path.join(self.data_directory, self.k_target_values_csv),
                                                                                                                  self.experiment_dictonaries,
                                                                                                                  self.S_matrix,
                                                                                                                  master_equation_reaction_list=self.master_equation_reactions,
                                                                                                                  six_parameter_fit_sensitivity_dict=self.six_parameter_fit_sensitivities)
        

        S_matrix,Y_matrix,z_matrix,sigma = self.master_equation_six_param_fit_instance.appending_target_values(k_target_values_for_z,
                                                                                                               k_target_values_for_Y,
                                                                                                               k_target_values_for_S,
                                                                                                               sigma_target_values,
                                                                                                               self.S_matrix,
                                                                                                               self.Y_matrix,
                                                                                                               self.z_matrix,
                                                                                                               self.sigma)
        
        
        
        self.S_matrix = S_matrix
        self.Y_matrix = Y_matrix
        self.z_matrix = z_matrix
        self.sigma = sigma
        self.Y_data_frame = Y_data_frame
        self.z_data_frame = z_data_frame
        self.k_target_values_for_s = k_target_values_for_S
        return
    
    def matrix_math(self,loop_counter = 0):
        if loop_counter ==0:
            X,covarience,s_matrix,y_matrix,delta_X,z_matrix,X_data_frame,prior_diag,prior_diag_df,sorted_prior_diag,covariance_prior_df,prior_sigmas_df = self.matrix_builder_instance.matrix_manipulation(loop_counter,self.S_matrix,self.Y_matrix,self.z_matrix,XLastItteration = np.array(()),active_parameters=self.active_parameters)            
            self.X = X
            self.covarience = covarience
            self.s_matrix = s_matrix
            self.y_matrix = y_matrix
            self.delta_X = delta_X
            self.z_matrix = z_matrix
            self.prior_diag = prior_diag
            self.prior_diag_df = prior_diag_df
            self.sorted_prior_diag = sorted_prior_diag
            self.covariance_prior_df = covariance_prior_df
            self.prior_sigmas_df = prior_sigmas_df
            self.X_data_frame = X_data_frame
            
        else:
            X,covarience,s_matrix,y_matrix,delta_X,z_matrix,X_data_frame,posterior_diag,posterior_diag_df,sorted_posterior_diag,covariance_posterior_df,posterior_sigmas_df = self.matrix_builder_instance.matrix_manipulation(loop_counter,self.S_matrix,self.Y_matrix,self.z_matrix,XLastItteration = self.X,active_parameters=self.active_parameters)
            self.X = X
            self.covarience = covarience
            self.s_matrix = s_matrix
            self.y_matrix = y_matrix
            self.delta_X = delta_X
            self.z_matrix = z_matrix
            self.X_data_frame = X_data_frame
            self.posterior_diag = posterior_diag
            self.posterior_diag_df = posterior_diag_df
            self.sorted_posterior_diag = sorted_posterior_diag
            self.covariance_posterior_df = covariance_posterior_df
            #self.posterior_over_prior = pd.concat([self.prior_diag_df, self.posterior_diag_df], axis=1, join_axes=[self.prior_diag_df.index])
            self.posterior_over_prior = pd.concat([self.prior_diag_df, self.posterior_diag_df], axis=1, join='outer')
            self.posterior_over_prior['posterior/prior'] = (self.posterior_diag_df['value'] / self.prior_diag_df['value'])
            self.posterior_over_prior = self.posterior_over_prior.sort_values(by=['posterior/prior'])
            self.posterior_sigmas_df = posterior_sigmas_df

        

        if self.master_equation_flag == True:
            deltaXAsNsEas,physical_observables,absorbance_coef_update_dict, X_to_subtract_from_Y,delta_x_molecular_params_by_reaction_dict,kinetic_paramter_dict = self.matrix_builder_instance.breakup_X(self.X,
                                                                                                                                          self.experiment_dictonaries,
                                                                                                                                          self.experiment_dict_uncertainty_original,
                                                                                                                                            loop_counter=loop_counter,
                                                                                                                                            master_equation_flag = self.master_equation_flag,
                                                                                                                                            master_equation_uncertainty_df=self.master_equation_uncertainty_df,
                                                                                                                                            master_equation_reactions = self.master_equation_reactions)
            self.delta_x_molecular_params_by_reaction_dict = delta_x_molecular_params_by_reaction_dict
        else:
            deltaXAsNsEas,physical_observables,absorbance_coef_update_dict, X_to_subtract_from_Y,kinetic_paramter_dict = self.matrix_builder_instance.breakup_X(self.X,
                                                                                                                                          self.experiment_dictonaries,
                                                                                                                                          self.experiment_dict_uncertainty_original,
                                                                                                                                            loop_counter=loop_counter)
        
        self.physical_obervable_updates_list = physical_observables 
        self.absorbance_coef_update_dict = absorbance_coef_update_dict
        self.deltaXAsNsEas = deltaXAsNsEas
        self.X_to_subtract_from_Y = X_to_subtract_from_Y
        self.kinetic_paramter_dict = kinetic_paramter_dict
        return
    
    
    def saving_first_itteration_matrices(self,loop_counter=0):
        if loop_counter==0:


            
            original_S_matrix = copy.deepcopy(self.S_matrix)
            self.original_S_matrix = original_S_matrix
            
            original_Y_matrix = copy.deepcopy(self.Y_matrix)
            self.original_Y_matrix = original_Y_matrix
            
            original_z_matrix = copy.deepcopy(self.z_matrix)
            self.original_z_matrix = original_z_matrix
            
            original_covarience = copy.deepcopy(self.covarience)
            self.original_covarience = original_covarience
            
            six_paramter_fit_nominal_parameters_dict = copy.deepcopy(self.six_paramter_fit_nominal_parameters_dict)
            self.six_paramter_fit_nominal_parameters_dict = six_paramter_fit_nominal_parameters_dict
            
            #original_experiment_dictonaries  = copy.deepcopy(self.experiment_dictonaries[0]['ksens'])

        return
    
    
    
    def updating_files(self,loop_counter=0):
        if loop_counter==0:
            updated_file_name_list = self.yaml_instance.yaml_file_updates(self.yaml_file_list_with_working_directory,
                                                 self.list_of_parsed_yamls,self.experiment_dictonaries,
                                                 self.physical_obervable_updates_list,
                                                 loop_counter = loop_counter)
            self.updated_file_name_list = updated_file_name_list
            
            updated_absorption_file_name_list = self.yaml_instance.absorption_file_updates(self.updated_file_name_list,
                                                                                       self.list_of_parsed_yamls,
                                                                                       self.experiment_dictonaries,
                                                                                       self.absorbance_coef_update_dict,
                                                                                       loop_counter = loop_counter)
            
            
            
        else:
            
            updated_file_name_list = self.yaml_instance.yaml_file_updates(self.updated_yaml_file_name_list,
                                                 self.list_of_parsed_yamls,self.experiment_dictonaries,
                                                 self.physical_obervable_updates_list,
                                                 loop_counter = loop_counter)
            
            
            
            
            
            updated_absorption_file_name_list = self.yaml_instance.absorption_file_updates(self.updated_yaml_file_name_list,
                                                                                       self.list_of_parsed_yamls,
                                                                                       self.experiment_dictonaries,
                                                                                       self.absorbance_coef_update_dict,
                                                                                       loop_counter = loop_counter)
            #print(self.original_experimental_conditions_local[0]['coupledCoefficients'],' ',loop_counter,'post simulation')
            
            
        self.updated_absorption_file_name_list = updated_absorption_file_name_list
        self.updated_yaml_file_name_list = self.updated_absorption_file_name_list
        
       
        
        if self.master_equation_flag == True:
            updated_six_parameter_fits_dict = self.master_equation_six_param_fit_instance.update_six_paramter_fits_dict(self.six_parameter_fit_sensitivities,
                                                                                      self.delta_x_molecular_params_by_reaction_dict, 
                                                                                      self.master_equation_reactions,
                                                                                      self.six_paramter_fit_nominal_parameters_dict)
        
            self.updated_six_parameter_fits_dict = updated_six_parameter_fits_dict
            
            
        if self.master_equation_flag == True:
            master_equation_surrogate_model_update_dictonary = self.master_equation_six_param_fit_instance.surrogate_model_molecular_parameters(self.molecular_parameter_sensitivities,
                                                                                                                                                self.master_equation_reactions,
                                                                                                                                                self.delta_x_molecular_params_by_reaction_dict,
                                                                                                                                                self.experiment_dictonaries)
                                                                                                                                                
            
            

        #this may not be the best way to do this 

            
            self.master_equation_surrogate_model_update_dictonary = master_equation_surrogate_model_update_dictonary
            
        if self.master_equation_flag == False:
            self.master_equation_surrogate_model_update_dictonary = {}

        
        #update the cti files pass in the renamed file 

        # is this how this function works 
        if self.master_equation_flag == True:
            new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(x = self.deltaXAsNsEas,
                                                                      original_cti=os.path.join(self.data_directory,self.cti_file_name),
                                                                      master_rxns = os.path.join(self.data_directory,self.master_reaction_equation_cti_name),
                                                                      master_index = self.master_index,
                                                                      MP = self.master_equation_surrogate_model_update_dictonary,
                                                                      working_directory=self.data_directory,
                                                                      file_name= self.cti_file_name.replace('.cti','')+'_updated')
            
        if self.master_equation_flag == False:
            new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(x = self.deltaXAsNsEas,
                                                                      original_cti=os.path.join(self.data_directory,self.cti_file_name),
                                                                      MP = self.master_equation_surrogate_model_update_dictonary,
                                                                      working_directory=self.data_directory,
                                                                      file_name= self.cti_file_name.replace('.cti','')+'_updated')
        self.new_cti_file = new_file 
        
        
       
        
        return
     
    def one_run_optimization(self,loop_counter=0):
        self.append_working_directory()
        #every loop run this, probably not?
        self.establish_processor(loop_counter=loop_counter)
        self.parsing_yaml_files(loop_counter = loop_counter)

        
        if loop_counter == 0:
            original_experimental_conditions_local = copy.deepcopy(self.yaml_instance.original_experimental_conditions)
            self.original_experimental_conditions_local = original_experimental_conditions_local
        
        
        self.running_simulations(loop_counter=loop_counter)
        
        if self.master_equation_flag == True:
            self.master_equation_s_matrix_building(loop_counter=loop_counter)
            #need to add functionality to update with the surgate model or drop out of loop
        self.building_matrices(loop_counter=loop_counter)
        
        if bool(self.k_target_values_csv):
            self.adding_k_target_values(loop_counter=loop_counter)
            
        self.matrix_math(loop_counter=loop_counter)
        if loop_counter==0:
            self.saving_first_itteration_matrices(loop_counter=loop_counter)
            

        self.updating_files(loop_counter=loop_counter)

        
        
    def multiple_simulation_runs(self,loops):
        delta_X_list = []
        for loop in range(loops):            
            self.one_run_optimization(loop_counter=loop)
            delta_x_df = pd.DataFrame(self.delta_X)
            delta_x_df = pd.concat([self.X_data_frame,delta_x_df],axis=1)
            delta_x_df.columns = ['parameter', 'X_values','delta_X_values']
            delta_x_df = delta_x_df.sort_values(by=['delta_X_values'])

            if loop==0:
                delta_x_df_norm_sig_prior = pd.DataFrame(delta_x_df['delta_X_values']/self.prior_sigmas_df['value'])
                delta_x_df_norm_sig_prior = pd.concat([self.X_data_frame,delta_x_df_norm_sig_prior],axis=1)
                delta_x_df_norm_sig_prior.columns = ['parameter', 'X_values','delta_X_values']
                delta_x_df_norm_sig_prior = delta_x_df_norm_sig_prior.sort_values(by=['delta_X_values'])
                self.delta_x_df_norm_sig_prior = delta_x_df_norm_sig_prior
            
            else:
                delta_x_df_norm_sig_posterior = delta_x_df['delta_X_values']/self.posterior_sigmas_df['value']
                delta_x_df_norm_sig_posterior = pd.concat([self.X_data_frame,delta_x_df_norm_sig_posterior],axis=1)
                delta_x_df_norm_sig_posterior.columns = ['parameter', 'X_values','delta_X_values']
                delta_x_df_norm_sig_posterior = delta_x_df_norm_sig_posterior.sort_values(by=['delta_X_values'])
                self.delta_x_df_norm_sig_posterior = delta_x_df_norm_sig_posterior
            
            
            self.delta_x_df = delta_x_df
            delta_X_list.append(self.delta_x_df)
            
        return delta_X_list
            
    
                                                                  
        
                                                       
            
                



