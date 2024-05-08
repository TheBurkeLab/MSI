import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.optimization.matrix_loader as ml
import MSI.optimization.opt_runner as opt
import MSI.simulations.absorbance.curve_superimpose as csp
import MSI.simulations.yaml_parser as yp
# import MSI.master_equation.master_equation_six_parameter_fit as mespf
import MSI.master_equation.master_equation as mecheb
import MSI.cti_core.cti_combine as ctic
import copy
import cantera as ct
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from art import *
import enlighten

class MSI_optimization_chebyshev(object):
        
    def __init__(self, 
                 working_directory:str,
                 cti_file:str,
                 yaml_file_list:list,
                 reaction_uncertainty_csv:str,
                 perturbment:int=0.01,
                 kineticSens:int=1,
                 physicalSens:int=1,                 
                 k_target_values_csv:str='',
                 master_equation_flag:bool=False,
                 master_equation_reactions:list=[],
                 chebyshev_sensitivities:dict={},
                 master_reaction_equation_cti_name:str = '',
                 master_index = [],
                 master_equation_uncertainty_df = None,
                 theory_parameters_df = None,
                 chebyshev_fit_nominal_parameters_dict = None,
                 step_size:int = 1,
                 T_P_min_max_dict={},
                 X_prior_csv:str=''):
        
        
        
        self.cti_file_name = cti_file
        copy.deepcopy(self.cti_file_name)
        self.perturbment = perturbment
        self.kineticSens = kineticSens
        self.physicalSens = physicalSens
        self.working_directory = working_directory
        self.matrix_path = os.path.join(self.working_directory, 'matrix')
        self.yaml_file_list = yaml_file_list
        self.yaml_file_list_with_working_directory = None
        self.processor = None
        self.list_of_yaml_objects = None
        self.list_of_parsed_yamls = None
        self.experiment_dictonaries = None
        self.reaction_uncertainty_csv = reaction_uncertainty_csv
        self.k_target_values_csv = k_target_values_csv
        self.MP_for_S_matrix = np.array(())
        self.step_size = step_size,
        
        # self.list_of_csv_names = None
        self.X_prior_csv = X_prior_csv
        if self.X_prior_csv == '':
            self.X_prior = pd.DataFrame().to_numpy()
        else:
            self.X_prior = np.array([list(pd.read_csv(X_prior_csv)['value'])]).T
        
        self.master_equation_flag = master_equation_flag
        if bool(self.master_equation_flag):
            # self.master_equation_flag = True
            self.master_equation_reactions = master_equation_reactions
            self.master_reaction_equation_cti_name = master_reaction_equation_cti_name
            self.master_index = master_index
            self.master_equation_uncertainty_df = master_equation_uncertainty_df
            self.theory_parameters_df = theory_parameters_df
            self.chebyshev_fit_nominal_parameters_dict = chebyshev_fit_nominal_parameters_dict
            self.chebyshev_sensitivities = chebyshev_sensitivities
            self.T_P_min_max_dict=T_P_min_max_dict
            
            Art=text2art("MSI",font='varsity')
            print('\n')
            print(Art)
            print('\n')
            print('--------------------------------------------------------------------------')
            print('Initializing Optimization Shell with Theory')
            print('--------------------------------------------------------------------------')
                    
        else:
            self.master_equation_reactions = []
            self.chebyshev_sensitivities = {}
            self.master_reaction_equation_cti_name = ''
            self.master_index = []    
            # self.master_equation_flag=False
            self.theory_parameters_df = None
            self.master_equation_uncertainty_df=None
            self.T_P_min_max_dict={}
            
            Art=text2art("MSI",font='varsity')
            print('\n')
            print(Art)
            print('\n')
            print('--------------------------------------------------------------------------')
            print('Initializing Optimization Shell')
            print('--------------------------------------------------------------------------')            

                
    # call all of leis functions where we do the molecular paramter stuff and turn a flag on 
    
    def append_working_directory(self):
        yaml_file_list_with_working_directory = []
        for i, file_set in enumerate(self.yaml_file_list):
            temp = []
            for j,file in enumerate(self.yaml_file_list[i]):
                temp.append(os.path.join(self.working_directory,file))
            temp = tuple(temp)
            yaml_file_list_with_working_directory.append(temp)
        self.yaml_file_list_with_working_directory = yaml_file_list_with_working_directory
        return
# pre process the cti file to remove reactions and rename it,also save it as the first run of the file
        
    # run the cti writer to establish processor and make cti file 
    def establish_processor(self,loop_counter=0):
        if loop_counter==0 and self.master_equation_flag==False:
            new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(original_cti=os.path.join(self.working_directory,self.cti_file_name),
                                                                      working_directory=self.working_directory,
                                                                      file_name= self.cti_file_name.replace('.cti','')+'_updated')
            self.new_cti_file = new_file
             
        if loop_counter==0 and self.master_equation_flag==True:
            new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(original_cti=os.path.join(self.working_directory, self.cti_file_name),
                                                                      master_rxns = os.path.join(self.working_directory, self.master_reaction_equation_cti_name),
                                                                      master_index = self.master_index,
                                                                      working_directory=self.working_directory,
                                                                      file_name= self.cti_file_name.replace('.cti','')+'_updated')
            self.new_cti_file = new_file
            
        
        processor = pr.Processor(self.new_cti_file)
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
            
        # print(list_of_parsed_yamls)  
        
        # py_list = []
        # for i, py in enumerate(self.list_of_parsed_yamls_original):
        #     for j, file in enumerate(py['csvFiles']):
        #         py_list.append(os.path.basename(file)[:-4])
                
        # self.list_of_csv_names = py_list
        
        return
    
    def running_simulations(self,loop_counter=0):
        optimization_instance = opt.Optimization_Utility()
        if loop_counter == 0:
            experiment_dictonaries = optimization_instance.looping_over_parsed_yaml_files(self.list_of_parsed_yamls,
                                              self.yaml_file_list_with_working_directory ,
                                              self.manager,
                                              processor=self.processor, 
                                              kineticSens=self.kineticSens,
                                              physicalSens=self.physicalSens,
                                              dk=self.perturbment,loop_counter=loop_counter)
            
            # print(experiment_dictonaries[0]['simulation'].timeHistories)
            
            experiment_dict_uncertainty_original = optimization_instance.saving_experimental_dict(experiment_dictonaries)
            
            
            self.experiment_dict_uncertainty_original = copy.deepcopy(experiment_dict_uncertainty_original)
            
            #call function taht loops opver experient dicts og and saves them
            
        else:
            
            experiment_dictonaries = optimization_instance.looping_over_parsed_yaml_files(self.list_of_parsed_yamls,
                                              self.updated_yaml_file_name_list ,
                                              self.manager,
                                              processor=self.processor, 
                                              kineticSens=self.kineticSens,
                                              physicalSens=self.physicalSens,
                                              dk=self.perturbment,loop_counter=loop_counter)
    
        
           
        self.experiment_dictonaries = experiment_dictonaries
       

        #maybe save this and just pass it in 
        return
    def master_equation_s_matrix_building(self,loop_counter=0):
        #stub
        master_equation_cheby_instance = mecheb.Master_Equation(T_P_min_max_dict=self.T_P_min_max_dict)
        self.master_equation_cheby_instance = master_equation_cheby_instance
        
        
        mapped_to_alpha_full_simulation,nested_list = master_equation_cheby_instance.map_to_alpha(self.chebyshev_sensitivities,
                                                                                                  self.experiment_dictonaries,
                                                                                                  self.list_of_parsed_yamls,
                                                                                                  self.master_equation_reactions)   
        self.mapped_to_alpha_full_simulation = mapped_to_alpha_full_simulation
        MP_for_S_matrix,new_sens_dict,broken_up_by_reaction,tottal_dict,tester = master_equation_cheby_instance.map_parameters_to_s_matrix(self.mapped_to_alpha_full_simulation,
                                                                                    self.chebyshev_sensitivities,
                                                                                    self.master_equation_reactions)
        
        
        new_S_matrix_for_MP  = master_equation_cheby_instance.combine_multiple_channels(MP_for_S_matrix,
                                  self.chebyshev_sensitivities,
                                 self.master_equation_reactions)
        
        

        self.MP_for_S_matrix = new_S_matrix_for_MP
        self.new_sens_dict = new_sens_dict
        self.broken_up_by_reaction = broken_up_by_reaction
        self.tottal_dict = tottal_dict
        self.tester=tester
        return
        
    def building_matrices(self,loop_counter=0):
        
        
        matrix_builder_instance = ml.OptMatrix()
        self.matrix_builder_instance = matrix_builder_instance
        S_matrix = matrix_builder_instance.load_S(self.experiment_dictonaries,
                                                  self.list_of_parsed_yamls,
                                                  dk=self.perturbment,
                                                  master_equation_reactions = self.master_equation_reactions,
                                                  mapped_master_equation_sensitivites=self.MP_for_S_matrix,
                                                  master_equation_flag = self.master_equation_flag)
        self.S_matrix = S_matrix

        
        
        if loop_counter == 0:
            Y_matrix,Ydf,active_parameters = matrix_builder_instance.load_Y(self.experiment_dictonaries,
                                                                   self.list_of_parsed_yamls,
                                                                   loop_counter=loop_counter,
                                                                   master_equation_flag = self.master_equation_flag,
                                                                   master_equation_uncertainty_df = self.master_equation_uncertainty_df,
                                                                   theory_parameters_df = self.theory_parameters_df,
                                                                   master_equation_reactions = self.master_equation_reactions)
                
            # nominal_physical_parameters,optimized_physical_parameters,physical_parameters_df = matrix_builder_instance.get_physical_parameters_df(self.experiment_dictonaries,
            #                                                                                                     self.list_of_parsed_yamls_original,
            #                                                                                                     loop_counter=loop_counter,
            #                                                                                                     master_equation_flag = self.master_equation_flag,
            #                                                                                                     master_equation_uncertainty_df = self.master_equation_uncertainty_df,
            #                                                                                                     master_equation_reactions = self.master_equation_reactions)            
                        
        else:
            Y_matrix,Ydf,active_parameters = matrix_builder_instance.load_Y(self.experiment_dictonaries,
                                                                   self.list_of_parsed_yamls,
                                                                   loop_counter=loop_counter,
                                                                   X=self.X_to_subtract_from_Y,
                                                                   master_equation_flag = self.master_equation_flag,
                                                                   master_equation_uncertainty_df = self.master_equation_uncertainty_df,
                                                                   theory_parameters_df = self.theory_parameters_df,
                                                                   master_equation_reactions = self.master_equation_reactions)    
            
            # nominal_physical_parameters,optimized_physical_parameters,physical_parameters_df = matrix_builder_instance.get_physical_parameters_df(self.experiment_dictonaries,
            #                                                                                                     self.list_of_parsed_yamls_original,
            #                                                                                                     loop_counter=loop_counter,
            #                                                                                                     X=self.X_to_subtract_from_Y,
            #                                                                                                     master_equation_flag = self.master_equation_flag,
            #                                                                                                     master_equation_uncertainty_df = self.master_equation_uncertainty_df,
            #                                                                                                     master_equation_reactions = self.master_equation_reactions)            
            
        self.Y_matrix = Y_matrix
        self.Ydf = Ydf
        
        Z_matrix,zdf,sigma = matrix_builder_instance.build_Z(self.experiment_dictonaries,
                                                                      self.list_of_parsed_yamls,
                                                                       loop_counter=loop_counter,
                                                                       reaction_uncertainty = os.path.join(self.working_directory, self.reaction_uncertainty_csv),
                                                                       master_equation_uncertainty_df=self.master_equation_uncertainty_df,
                                                                       master_equation_flag = self.master_equation_flag,
                                                                       master_equation_reaction_list = self.master_equation_reactions)
        

        # self.physical_parameters_df = physical_parameters_df 
        self.Z_matrix = Z_matrix
        self.zdf = zdf
        self.sigma = sigma
        
        yaml_names = [yaml_file[0][5:-5] for yaml_file in self.yaml_file_list]
        new_active_parameters = []
        for parameter in active_parameters:
            if 'experiment' in parameter:
                param_split = parameter.split('experiment')
                new_active_parameters.append(param_split[0] + yaml_names[eval(param_split[1])])
            else:
                new_active_parameters.append(parameter)
        self.active_parameters = new_active_parameters
        
        return
    
    

    def adding_k_target_values(self,loop_counter=0):
        
        ### This needs to be editied to accomidate chebychev 
        
        adding_target_values_instance = ml.Adding_Target_Values(self.S_matrix,self.Y_matrix,self.Z_matrix,self.sigma,
                                                                self.Ydf,self.zdf,T_P_min_max_dict = self.T_P_min_max_dict)
        
        self.adding_target_values_instance = adding_target_values_instance
        

        k_target_values_for_z,sigma_target_values,zdf = self.adding_target_values_instance.target_values_for_Z(os.path.join(self.working_directory, self.k_target_values_csv),
                                                                                                                            self.zdf)
        
        
        if loop_counter == 0:
    
            
            k_target_values_for_Y,Ydf = self.adding_target_values_instance.target_values_Y(os.path.join(self.working_directory, self.k_target_values_csv),
                                                                                              self.experiment_dictonaries,self.Ydf,self.master_equation_reactions)
        else:
            k_target_values_for_Y,Ydf = self.adding_target_values_instance.target_values_Y(os.path.join(self.working_directory, self.k_target_values_csv),
                                                                                              self.experiment_dictonaries,self.Ydf,self.master_equation_reactions)       
        

        
        
       

        k_target_values_for_S = self.adding_target_values_instance.target_values_for_S(os.path.join(self.working_directory, self.k_target_values_csv),
                                                                                 self.experiment_dictonaries,
                                                                                 self.S_matrix,
                                                                                 master_equation_reaction_list = self.master_equation_reactions,
                                                                                 master_equation_sensitivites = self.chebyshev_sensitivities)    

        

                                
        S_matrix,Y_matrix,Z_matrix,sigma = self.adding_target_values_instance.appending_target_values(k_target_values_for_z,
                                                                                                               k_target_values_for_Y,
                                                                                                               k_target_values_for_S,
                                                                                                               sigma_target_values,
                                                                                                               self.S_matrix,
                                                                                                               self.Y_matrix,
                                                                                                               self.Z_matrix,
                                                                                                               self.sigma)                        

        
        # make S dataframe here
        
        self.S_matrix = S_matrix
        self.Y_matrix = Y_matrix
        self.Z_matrix = Z_matrix
        self.sigma = sigma
        self.Ydf = Ydf
        self.zdf = zdf
        self.k_target_values_for_S = k_target_values_for_S
        
        return
    
    def matrix_math(self,loop_counter = 0):
        if loop_counter ==0:
            X,Xdf_prior,covariance,s_matrix,y_matrix,delta_X,Z_matrix,Xdf,prior_diag,prior_diag_df,sorted_prior_diag,covariance_prior_df,prior_sigmas_df = self.matrix_builder_instance.matrix_manipulation(loop_counter,self.S_matrix,self.Y_matrix,self.Z_matrix,XLastItteration = self.X_prior, active_parameters=self.active_parameters,step_size=self.step_size)            
            self.X = X
            self.covariance = covariance
            self.s_matrix = s_matrix
            self.y_matrix = y_matrix
            self.delta_X = delta_X
            self.Z_matrix = Z_matrix
            self.prior_diag = prior_diag
            self.prior_diag_df = prior_diag_df
            self.sorted_prior_diag = sorted_prior_diag
            self.covariance_prior_df = covariance_prior_df
            self.prior_sigmas_df = prior_sigmas_df
            self.Xdf = Xdf
            self.Xdf_prior = Xdf_prior
            
        else:
            X,covariance,s_matrix,y_matrix,delta_X,Z_matrix,Xdf,posterior_diag,posterior_diag_df,sorted_posterior_diag,covariance_posterior_df,posterior_sigmas_df = self.matrix_builder_instance.matrix_manipulation(loop_counter,self.S_matrix,self.Y_matrix,self.Z_matrix,XLastItteration = self.X,active_parameters=self.active_parameters,step_size=self.step_size)
            self.X = X
            self.covariance = covariance
            self.s_matrix = s_matrix
            self.y_matrix = y_matrix
            self.delta_X = delta_X
            self.Z_matrix = Z_matrix
            self.Xdf = Xdf
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
                                                                                                                                          self.experiment_dict_uncertainty_original,                                                                                                                         loop_counter=loop_counter)
        # self.target_parameters = list(self.Ydf['parameter'])
        yaml_names = [yaml_file[0][5:-5] for yaml_file in self.yaml_file_list]
        new_target_parameters = []
        for parameter in list(self.Ydf['parameter']):
            if 'experiment' in parameter:
                param_split = parameter.split('experiment')
                new_target_parameters.append(param_split[0] + yaml_names[eval(param_split[1])])
            else:
                new_target_parameters.append(parameter)
        self.target_parameters = new_target_parameters
        self.physical_obervable_updates_list = physical_observables 
        self.absorbance_coef_update_dict = absorbance_coef_update_dict
        self.deltaXAsNsEas = deltaXAsNsEas
        self.X_to_subtract_from_Y = X_to_subtract_from_Y
        self.kinetic_paramter_dict = kinetic_paramter_dict
        return
    
    
    def saving_first_itteration_matrices(self,loop_counter=0):
  
        self.Ydf_prior = copy.deepcopy(self.Ydf)
        self.Sdf_prior = copy.deepcopy(self.Sdf)
        self.covdf_prior = copy.deepcopy(self.covdf)
        self.sigdf_prior = copy.deepcopy(self.sigdf)
        self.experiment_dictonaries_original = self.experiment_dictonaries     
        
        self.Xdf_prior.to_csv(os.path.join(self.matrix_path,'Xdf_prior.csv'))
        self.Ydf_prior.to_csv(os.path.join(self.matrix_path,'Ydf_prior.csv'))
        self.Sdf_prior.to_csv(os.path.join(self.matrix_path,'Sdf_prior.csv'))
        self.covdf_prior.to_csv(os.path.join(self.matrix_path,'covdf_prior.csv'))
        self.sigdf_prior.to_csv(os.path.join(self.matrix_path,'sigdf_prior.csv')) 
                         
        return
    
    
    def updating_files(self,loop_counter=0):
        if loop_counter==0:
            updated_file_name_list = self.yaml_instance.yaml_file_updates(self.yaml_file_list_with_working_directory,
                                                 self.list_of_parsed_yamls,self.experiment_dictonaries,
                                                 self.physical_obervable_updates_list,
                                                 loop_counter = loop_counter)
            self.updated_file_name_list = updated_file_name_list
            
            self.optimized_physical_parameters = self.yaml_instance.optimized_physical_parameters
            
            updated_absorption_file_name_list = self.yaml_instance.absorption_file_updates(self.updated_file_name_list,
                                                                                       self.list_of_parsed_yamls,
                                                                                       self.experiment_dictonaries,
                                                                                       self.absorbance_coef_update_dict,
                                                                                       loop_counter = loop_counter)
            
            self.optimized_absorption_parameters = self.yaml_instance.optimized_absorption_parameters
            
        else:
            
            updated_file_name_list = self.yaml_instance.yaml_file_updates(self.updated_yaml_file_name_list,
                                                 self.list_of_parsed_yamls,self.experiment_dictonaries,
                                                 self.physical_obervable_updates_list,
                                                 loop_counter = loop_counter)
            
            self.optimized_physical_parameters = self.yaml_instance.optimized_physical_parameters
            
            updated_absorption_file_name_list = self.yaml_instance.absorption_file_updates(self.updated_yaml_file_name_list,
                                                                                       self.list_of_parsed_yamls,
                                                                                       self.experiment_dictonaries,
                                                                                       self.absorbance_coef_update_dict,
                                                                                       loop_counter = loop_counter)
            #print(self.original_experimental_conditions_local[0]['coupledCoefficients'],' ',loop_counter,'post simulation')
            
            self.optimized_absorption_parameters = self.yaml_instance.optimized_absorption_parameters
            
            
        self.updated_absorption_file_name_list = updated_absorption_file_name_list
        self.updated_yaml_file_name_list = self.updated_absorption_file_name_list
        
       
        
        if self.master_equation_flag == True:

                                                   
            master_equation_surrogate_model_update_dictonary = self.master_equation_cheby_instance.surrogate_model_molecular_parameters_chevy(self.chebyshev_sensitivities,
                                                                                                                                              self.new_sens_dict,
                                                                                                                                              self.master_equation_reactions,
                                                                                                                                              self.delta_x_molecular_params_by_reaction_dict,
                                                                                                                                              self.experiment_dictonaries)     
                                                                                                                                                

        #this may not be the best way to do this 

            
            self.master_equation_surrogate_model_update_dictonary = master_equation_surrogate_model_update_dictonary
            
        lei=False
        if lei==True:
            print('This is where lei would run his stuff')
            
        if self.master_equation_flag == False:
            self.master_equation_surrogate_model_update_dictonary = {}

        
        #update the cti files pass in the renamed file 

        # is this how this function works 
        if self.master_equation_flag == True:
            new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(x = self.deltaXAsNsEas,
                                                                      original_cti= os.path.join(self.working_directory, self.cti_file_name),
                                                                      master_rxns = os.path.join(self.working_directory, self.master_reaction_equation_cti_name),
                                                                      master_index = self.master_index,
                                                                      MP = self.master_equation_surrogate_model_update_dictonary,
                                                                      working_directory=self.working_directory,
                                                                      file_name= self.cti_file_name.replace('.cti','')+'_updated')
            
        if self.master_equation_flag == False:
            new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(x = self.deltaXAsNsEas,
                                                                      original_cti=os.path.join(self.working_directory, self.cti_file_name),
                                                                      MP = self.master_equation_surrogate_model_update_dictonary,
                                                                      working_directory=self.working_directory,
                                                                      file_name= self.cti_file_name.replace('.cti','')+'_updated')
        self.new_cti_file = new_file 
        return
     
    def one_run_optimization(self,loop_counter=0):
        
        print('\n')
        print('--------------------------------------------------------------------------')
        print('Iteration ' + str(loop_counter+1))    
        print('--------------------------------------------------------------------------')    
        
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
        # if not self.k_target_values_csv.empty:
            self.adding_k_target_values(loop_counter=loop_counter)
        else:
            self.k_target_values_for_S = np.array([])
        
        self.matrix_math(loop_counter=loop_counter)
        
        self.Xdf = pd.DataFrame({'value': self.X.T[0]}, index=self.active_parameters)   
        self.Ydf = pd.DataFrame({'value': self.Y_matrix.T[0]}, index=self.target_parameters)          
        self.ydf = pd.DataFrame({'value': self.y_matrix.T[0]}, index=self.target_parameters)          
        self.Zdf = pd.DataFrame({'value': self.Z_matrix.T[0]}, index=self.target_parameters)          
        self.Sdf = pd.DataFrame(self.S_matrix, columns=self.active_parameters, index=self.target_parameters)
        self.sdf = pd.DataFrame(self.s_matrix, columns=self.active_parameters, index=self.target_parameters)
        self.covdf = pd.DataFrame(self.covariance, columns=self.active_parameters, index=self.active_parameters)
        self.sigdf = pd.DataFrame({'value': list(np.sqrt(np.diag(self.covariance)))}, index=self.active_parameters)   
        
        self.Xdf.to_csv(os.path.join(self.matrix_path,'Xdf.csv'))
        self.Xdf.to_csv(os.path.join(self.matrix_path,'Xdf_'+str(loop_counter+1)+'.csv'))
        self.Ydf.to_csv(os.path.join(self.matrix_path,'Ydf.csv'))
        self.ydf.to_csv(os.path.join(self.matrix_path,'ydf.csv'))
        self.Zdf.to_csv(os.path.join(self.matrix_path,'Zdf.csv'))
        self.Sdf.to_csv(os.path.join(self.matrix_path,'Sdf.csv'))
        self.sdf.to_csv(os.path.join(self.matrix_path,'sdf.csv'))
        self.covdf.to_csv(os.path.join(self.matrix_path,'covdf.csv'))
        self.sigdf.to_csv(os.path.join(self.matrix_path,'sigdf.csv'))
                
        if loop_counter==0:
            self.saving_first_itteration_matrices(loop_counter=loop_counter)
            
        self.updating_files(loop_counter=loop_counter)
        
        
    def multiple_runs(self,loops):
        
        self.manager = enlighten.get_manager()
        self.mainloop=self.manager.counter(total=loops,desc='MSI Optimization:',unit='iterations',color='red')            
                
        Xdf_list = []
        for loop in range(loops):       
                
            
            # if self.X_prior.size == 0:
            #     Xdf_list.append(np.zeros(np.shape(active_parameters)))
            # elif len(self.X_prior) != len(active_parameters):
            #     print('User defined X_prior is not the correct length. Using zeros as X_prior instead.')
            #     Xdf_list.append(np.zeros(np.shape(active_parameters)))
            # else:
            #     Xdf_list.append(self.X_prior)
                            
            self.one_run_optimization(loop_counter=loop)
            
            # delta_Xdf = pd.DataFrame({'X':list(self.Xdf['value']), 'dX':list(self.delta_X.T[0])}, index=self.Xdf.index.values.tolist())
            # delta_Xdf = pd.concat([self.Xdf,delta_Xdf],axis=1)
            # delta_Xdf.columns = ['parameter', 'X_values','delta_X_values']
            # delta_x_df = delta_x_df.sort_values(by=['delta_X_values'])

            # if loop==0:
            #     delta_x_df_norm_sig_prior = pd.DataFrame(delta_x_df['delta_X_values']/self.prior_sigmas_df['value'])
            #     delta_x_df_norm_sig_prior = pd.concat([self.Xdf,delta_x_df_norm_sig_prior],axis=1)
            #     delta_x_df_norm_sig_prior.columns = ['parameter', 'X_values','delta_X_values']
            #     delta_x_df_norm_sig_prior = delta_x_df_norm_sig_prior.sort_values(by=['delta_X_values'])
            #     self.delta_x_df_norm_sig_prior = delta_x_df_norm_sig_prior
            
            # else:
            #     delta_x_df_norm_sig_posterior = delta_x_df['delta_X_values']/self.posterior_sigmas_df['value']
            #     delta_x_df_norm_sig_posterior = pd.concat([self.Xdf,delta_x_df_norm_sig_posterior],axis=1)
            #     delta_x_df_norm_sig_posterior.columns = ['parameter', 'X_values','delta_X_values']
            #     delta_x_df_norm_sig_posterior = delta_x_df_norm_sig_posterior.sort_values(by=['delta_X_values'])
            #     self.delta_x_df_norm_sig_posterior = delta_x_df_norm_sig_posterior
                
            # self.delta_x_df = delta_x_df
            
            Xdf_list.append(self.Xdf)
            
            if loop != 0:
                X_over_sig = pd.DataFrame({'X':list(self.Xdf.value),'sigma_prior':list(self.sigdf_prior.value),'sigma_posterior':list(self.sigdf.value),'X/sigma_prior':np.divide(list(self.Xdf.value),list(self.sigdf_prior.value))},index=self.active_parameters)
                self.X_over_sig_sorted = X_over_sig.loc[X_over_sig['X/sigma_prior'].abs().sort_values(ascending=False).index]
                self.X_over_sig_sorted.to_csv(os.path.join(self.matrix_path,'X_over_sigma.csv'))
                
                convergence=pd.DataFrame(np.array([list(Xdf_list[i].value) for i in range(len(Xdf_list))]).T, index=self.active_parameters, columns=np.arange(1,len(Xdf_list)+1))     
                self.convergence_sorted = convergence.loc[convergence.abs().sum(axis=1).sort_values(ascending=False).index]
                self.convergence_sorted.to_csv(os.path.join(self.matrix_path,'convergence.csv'))
                    
            # self.physical_parameters_df = self.matrix_builder_instance.get_physical_parameters_df(self.experiment_dictonaries,
            #                                                                                  self.optimized_physical_parameters,
            #                                                                                  self.optimized_absorption_parameters,
            #                                                                                  self.list_of_parsed_yamls_original)       
            
            self.mainloop.update()
                     
        self.Xdf_list = Xdf_list
        


        # for i,Xdf_iteration in enumerate(self.Xdf_list):
        #     Xdf_iteration.to_csv(os.path.join(self.matrix_path,'Xdf_'+str(i)+'.csv'))     
        self.manager.stop()         
        return
    
                                                                  
        
                                                       
            
                



