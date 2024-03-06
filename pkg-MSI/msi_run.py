# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:33:39 2020

@author: Mark Barbet
"""

import sys, os
import fire
import MSI.optimization.optimization_shell as shell
import MSI.utilities.plotting_script as plotter
import MSI.optimization.optimization_shell_chebyshev as MSIcheb
import re
import pandas as pd
import numpy as np
import ast

class multiscale_informatics:
    '''
    Multi-scale Informatics (MSI) container class.  Contains all the attributes of an
    MSI simulation and determines which sub-functions to run based on information provided in an input file.
    
    MSI may be run with or without master-equation parameters, and may be run on multiple experimental yaml
    files in series, or all at once.    

    '''
    
    
    def __init__(self,input_options):
        '''
        Initializes the 'multiscale_informatics' class, and calls 'self.get_parameters'
        to seperate and sort parameters from the parsed input file.

        Parameters
        ----------
        input_options : List
            List of strings of the parsed lines of the MSI input file.

        Returns
        -------
        None.

        '''
        self.input_options=input_options
        self.get_parameters()
        
        
        
    def run_msi(self):
        '''
        Call 'self.run_msi' to run the optimization after class has been initialized. 
        Function determines how to run optimizations on the basis of whether master-eqution
        information was included in the input file, and whether input file indicates to run
        experimental conditions serially or all at once.

        Returns
        -------
        None.

        '''
        print(self.wdir)
        os.chdir(self.wdir)
        for m,mech in enumerate(self.models):
            if self.run_individual_yaml:
                #
                for y,yamlfile in enumerate(self.yaml_files):
                    if not self.master_equation_models:
                        self.msi_no_master(m,yaml_input=[self.yaml_files[y]])
                        self.plotting_no_master(m,file_identifier=self.yaml_files[y][0].strip('.yaml'))
                    elif self.master_equation_models:
                        self.msi_master(m,yaml_input=[self.yaml_files[y]])
                        self.plotting_master(m, file_identifier=self.yaml_files[y][0].strip('.yaml'))
                
            elif not self.run_individual_yaml:
                
                if not self.master_equation_models:
                    self.msi_no_master(m,yaml_input=self.yaml_files)
                    self.plotting_no_master(m,file_identifier='all_yamls')
                    
                elif self.master_equation_models:
                    self.msi_master(m,yaml_input=self.yaml_files)
                    self.plotting_master(m,file_identifier='all_yamls')
                
                
    def write_convergence(self):
        '''
        Function can be used to write delta X data from each MSI iteration to a
        csv file for post-processing.  Runs by default, saves file as convergence.csv
        in the MSI working directory.

        Returns
        -------
        None.

        '''
        #print(type(self.X_list))
        #print(len(self.X_list))
        convergence=pd.DataFrame(columns=['Parameter'])
        for i in range(len(self.X_list)):
            #print(np.shape(self.X_list[i]))
            convergence[str(i)]=np.array(self.X_list[i]).flatten()
        print(self.Xdf['value'])
        convergence['Parameter']=self.Xdf['value']
        #print(self.Xdf.columns)
        convergence.to_csv(self.wdir+'\\convergence_data.csv',index=False)
        
    def msi_master(self,iteration,yaml_input=[]):
        '''
        Runs the MSI simulation for cases where master-equation data is included. 
        Saves results internally to attributes of 'self.multiscale_informatics'

        Parameters
        ----------
        iteration : int
            Integer indicating which kinetic model MSI is currently optimizing..
        yaml_input : List, optional
           List of the experiment yaml files to be run for this optimization. Contains 
           onlt one item if experiments are being run serially, otherwise usually contains
           all yaml files.

        Returns
        -------
        None.

        '''
        current_master_uncertainties=pd.read_csv(self.master_uncertainties[iteration])
        self.get_chebyshev_coefficients(iteration)
        #print(self.cheb_coeffs)
        MSI_instance_one=MSIcheb.MSI_optimization_chebyshev(self.models[iteration],
                                                            0.01,
                                                            1,
                                                            1,
                                                            self.wdir,
                                                            yaml_input,
                                                            self.reaction_uncertainties[iteration],
                                                            k_target_values_csv=self.targets[iteration],
                                                            master_equation_reactions=self.master_equation_reactions[iteration],
                                                            chebyshev_sensitivities=self.cheb_coeffs[iteration],
                                                            master_reaction_equation_cti_name=self.master_equation_models[iteration],
                                                            master_index=self.indices[iteration],
                                                            master_equation_uncertainty_df=current_master_uncertainties,
                                                            chebyshev_fit_nominal_parameters_dict= None)
        MSI_instance_one.one_run_optimization()
        self.S_matrix_original=MSI_instance_one.S_matrix
        '''Contains the S matrix of sensitivities for the nominal (0th iteration) model'''
        self.exp_dict_list_original=MSI_instance_one.experiment_dictonaries
        '''Contains the original dictionary of experiment data constructed from yaml files'''
        self.original_covariance=MSI_instance_one.covarience
        '''Covariance matrix after running the nominal model'''
        self.X_one_iteration=MSI_instance_one.X
        '''The change in active parameters estimated after one iteration of MSI'''
        self.z_df_original=MSI_instance_one.z_data_frame
        '''Dataframe of uncertainties in model parameters after one iteration of MSI'''
        self.y_df_original=MSI_instance_one.Y_data_frame
        '''Values calculated by the nominal kinetic model at the conditions of the 
        experiments'''
        
        
        self.MSI_instance_two=MSIcheb.MSI_optimization_chebyshev(self.models[iteration],
                                                            0.01,
                                                            1,
                                                            1,
                                                            self.wdir,
                                                            yaml_input,
                                                            self.reaction_uncertainties[iteration],
                                                            k_target_values_csv=self.targets[iteration],
                                                            master_equation_reactions=self.master_equation_reactions[iteration],
                                                            chebyshev_sensitivities=self.cheb_coeffs[iteration],
                                                            master_reaction_equation_cti_name=self.master_equation_models[iteration],
                                                            master_index=self.indices[iteration],
                                                            master_equation_uncertainty_df=current_master_uncertainties,
                                                            chebyshev_fit_nominal_parameters_dict= None)
        
        self.X_list=self.MSI_instance_two.multiple_runs(self.iterations)
        '''List of the updates to the X vector (active parameters) for each iteration.  
        This is usually written to convergence.csv'''
        self.deltaXAsNsEas=self.MSI_instance_two.deltaXAsNsEas
        
        self.physical_obervable_updates_list = self.MSI_instance_two.physical_obervable_updates_list
        '''List of updates to the physical observables from experiments'''
        self.absorbance_observables_updates_list = self.MSI_instance_two.absorbance_coef_update_dict
        '''List of updates to the absorbances from experiments'''
        self.Ydf = self.MSI_instance_two.Y_data_frame
        '''Stores the Dataframe of the simulation results after each iteration'''
        self.Zdf = self.MSI_instance_two.z_data_frame
        '''Stores Dataframe of updated parameter uncertainties after each iteration'''
        self.experimental_dicts = self.MSI_instance_two.experiment_dictonaries
        '''Stores dictionaries of experimental data after each iteration'''
        self.z_matrix = self.MSI_instance_two.z_matrix
        '''Numpy vector of parameter uncertainties after each MSI iteration'''
        self.s_matrix = self.MSI_instance_two.s_matrix
        '''Numpy matrix of the sensitivities after each MSI iteration'''
        self.y = self.MSI_instance_two.y_matrix
        '''Numpy vector of the simulation results after each MSI iteration'''
        self.Y_matrix = self.MSI_instance_two.Y_matrix
        self.S_matrix = self.MSI_instance_two.S_matrix
        
        self.X = self.MSI_instance_two.X
        self.Xdf = self.MSI_instance_two.X_data_frame
        self.covarience = self.MSI_instance_two.covarience
        self.exp_dict_list_optimized = self.MSI_instance_two.experiment_dictonaries
        self.parsed_yaml_list = self.MSI_instance_two.list_of_parsed_yamls
        self.sigma = self.MSI_instance_two.sigma
        self.X = self.MSI_instance_two.X
        self.delta_X = self.MSI_instance_two.delta_X
        self.molecular_parameter_updates=self.MSI_instance_two.delta_x_molecular_params_by_reaction_dict
        self.original_diag=np.diag(self.original_covariance)
        self.original_cti_file=self.MSI_instance_two.data_directory+'/'+self.MSI_instance_two.cti_file_name
        self.experiment_dict_uncertainty=self.MSI_instance_two.experiment_dict_uncertainty_original
        self.target_value_csv=self.MSI_instance_two.data_directory+'/'+self.MSI_instance_two.k_target_values_csv
        if self.targets[iteration]:
            self.k_target_value_S_matrix = self.MSI_instance_two.k_target_values_for_S
            self.run_with_k_target_values='On'
        else:
            self.k_target_value_S_matrix = None
            self.run_with_k_target_values='Off'
        self.csv_file_sigma=''
    def plotting_master(self,iteration,file_identifier=''):
        '''
        This function generates plots of the MSI optimization results.

        Parameters
        ----------
        iteration : int
            Integer indicating which kinetic model is being plotted.
        file_identifier : string, optional
            String which can be used to provide a unique identifier to a saved plot.  
            Here it is used to indicate what yaml file experimental conditions the 
            plot depicts.  The default is ''.

        Returns
        -------
        None.

        '''
        plotting_instance = plotter.Plotting(self.S_matrix,
                                     self.s_matrix,
                                     self.Y_matrix,
                                     self.y,
                                     self.z_matrix,
                                     self.X,
                                     self.sigma,
                                     self.covarience,
                                     self.original_covariance,
                                     self.S_matrix_original,
                                     self.exp_dict_list_optimized,
                                     self.exp_dict_list_original,
                                     self.parsed_yaml_list,
                                     self.Ydf,
                                     target_value_rate_constant_csv= self.optional_targets[iteration],
                                     target_value_rate_constant_csv_extra_values = self.optional_targets[iteration],
                                     k_target_value_S_matrix =self.k_target_value_S_matrix,
                                     k_target_values=self.run_with_k_target_values,
                                     working_directory = self.wdir,
                                     sigma_uncertainty_weighted_sensitivity_csv=self.csv_file_sigma,
                                     cheby_sensitivity_dict = self.cheb_coeffs[iteration],
                                     mapped_to_alpha_full_simulation=self.MSI_instance_two.mapped_to_alpha_full_simulation)

        #csv_file_sigma = MSI_st_instance_two.data_directory +'/'+'sigma_for_uncertainty_weighted_sensitivity_updated.csv'
        self.observable_counter_and_absorbance_wl,self.length_of_experimental_data = plotting_instance.lengths_of_experimental_data()
        self.sigmas_optimized,self.test = plotting_instance.calculating_sigmas(self.S_matrix,self.covarience)
        self.sigmas_original,self.test2 = plotting_instance.calculating_sigmas(self.S_matrix_original,self.original_covariance)
        plotting_instance.plotting_observables(sigmas_original = self.sigmas_original,sigmas_optimized= self.sigmas_optimized)
        self.diag = plotting_instance.getting_matrix_diag(self.covarience)
        
        plotting_instance.plotting_rate_constants(optimized_cti_file=self.MSI_instance_two.new_cti_file,
                                original_cti_file=self.original_cti_file,
                                initial_temperature=250,
                                final_temperature=2500,
                                master_equation_reactions = self.master_equation_reactions[iteration])



        self.sensitivity, self.top_sensitivity = plotting_instance.sort_top_uncertainty_weighted_sens()
        self.obs = plotting_instance.plotting_uncertainty_weighted_sens()
        
        
        
    def get_chebyshev_coefficients(self,iteration):
        '''
        Loads chebyshev coefficients from a raw python file that containes them in a numpy array.  
        Only used when running with master-equation in the MSI simulation.

        Parameters
        ----------
        iteration : int
            Integer indicating which kinetic model is being plotted.

        Returns
        -------
        None.

        '''
        self.cheb_coeffs=[]
        self.cheb_sensitivity_dict=None
        import importlib.util
        cheb=importlib.util.spec_from_file_location(self.master_sens[iteration].strip('.py'),self.wdir+'\\'+self.master_sens[iteration])
        #print(self.master_sens[iteration].strip('.py'))
        module=importlib.util.module_from_spec(cheb)
        cheb.loader.exec_module(module)
        #print(module.cheb_sensitivity_dict)
        #import self.master_sens[iteration].strip('.py') as c
        # file=open(self.master_sens[iteration],'r')
        
        # with open(self.master_sens[iteration],'r') as f:
        #     contents=f.read()
        #     dictionary=ast.literal_eval(contents)
        
        # print(dictionary)
        self.cheb_coeffs.append(module.cheb_sensitivity_dict)
        
        
        
        
        
    def msi_no_master(self,iteration,yaml_input=[]):
        '''
        Runs the MSI simulation for cases where master-equation data is not included. 
        Saves results internally to attributes of 'self.multiscale_informatics'

        Parameters
        ----------
        iteration : int
            Integer indicating which kinetic model MSI is currently optimizing..
        yaml_input : List, optional
           List of the experiment yaml files to be run for this optimization. Contains 
           onlt one item if experiments are being run serially, otherwise usually contains
           all yaml files.

        Returns
        -------
        None.

        '''
        MSI_instance_one=shell.MSI_optimization(self.models[iteration],
                                                0.01,
                                                1,
                                                1,
                                                self.wdir,
                                                yaml_input,
                                                self.reaction_uncertainties[iteration],
                                                k_target_values_csv=self.targets[iteration])
        MSI_instance_one.one_run_optimization()
        self.S_matrix_original=MSI_instance_one.S_matrix
        self.exp_dict_list_original=MSI_instance_one.experiment_dictonaries
        self.original_covariance=MSI_instance_one.covarience
        self.X_one_iteration=MSI_instance_one.X
        self.z_df_original=MSI_instance_one.z_data_frame
        
        self.MSI_instance_two=shell.MSI_optimization(self.models[iteration],
                                                0.01,
                                                1,
                                                1,
                                                self.wdir,
                                                yaml_input,
                                                self.reaction_uncertainties[iteration],
                                                k_target_values_csv=self.targets[iteration])
        self.X_list=self.MSI_instance_two.multiple_runs(self.iterations)
        
        self.deltaXAsNsEas=self.MSI_instance_two.deltaXAsNsEas
        self.physical_obervable_updates_list = self.MSI_instance_two.physical_obervable_updates_list
        self.absorbance_observables_updates_list = self.MSI_instance_two.absorbance_coef_update_dict
        self.Ydf = self.MSI_instance_two.Y_data_frame
        self.Zdf = self.MSI_instance_two.z_data_frame
        self.experimental_dicts = self.MSI_instance_two.experiment_dictonaries
        self.z_matrix = self.MSI_instance_two.z_matrix
        self.s_matrix = self.MSI_instance_two.s_matrix
        self.y = self.MSI_instance_two.y_matrix
        self.Y_matrix = self.MSI_instance_two.Y_matrix
        self.S_matrix = self.MSI_instance_two.S_matrix
        
        self.X = self.MSI_instance_two.X
        self.Xdf = self.MSI_instance_two.X_data_frame
        self.covarience = self.MSI_instance_two.covarience
        self.exp_dict_list_optimized = self.MSI_instance_two.experiment_dictonaries
        self.parsed_yaml_list = self.MSI_instance_two.list_of_parsed_yamls
        self.sigma = self.MSI_instance_two.sigma
        self.X = self.MSI_instance_two.X
        self.delta_X = self.MSI_instance_two.delta_X
        #target_value_rate_constant_csv = 'MSI/data/test_data/FFCM1_custom_target_value_test.csv'
        self.original_cti_file = self.MSI_instance_two.data_directory +'/'+ self.MSI_instance_two.cti_file_name
        self.experiment_dict_uncertainty = self.MSI_instance_two.experiment_dict_uncertainty_original
        self.target_value_csv = self.MSI_instance_two.data_directory +'/'+ self.MSI_instance_two.k_target_values_csv
        
        if self.targets[iteration]:
            self.k_target_value_S_matrix = self.MSI_instance_two.k_target_values_for_s
            self.run_with_k_target_values='On'
        else:
            self.k_target_value_S_matrix = None
            self.run_with_k_target_values='Off'
            
    def plotting_no_master(self, iteration,file_identifier=''):
        '''
        This function generates plots of the MSI optimization results when no master-equation material
        is contained in the MSI simulations.

        Parameters
        ----------
        iteration : int
            Integer indicating which kinetic model is being plotted.
        file_identifier : string, optional
            String which can be used to provide a unique identifier to a saved plot.  
            Here it is used to indicate what yaml file experimental conditions the 
            plot depicts.  The default is ''.

        Returns
        -------
        None.

        '''
        plotting_instance = plotter.Plotting(self.S_matrix,
                                          self.s_matrix,
                                          self.Y_matrix,
                                          self.Y_matrix,
                                          self.z_matrix,
                                          self.X,
                                          self.sigma,
                                          self.covarience,
                                          self.original_covariance,
                                          self.S_matrix_original,
                                          self.exp_dict_list_optimized,
                                          self.exp_dict_list_original,
                                          self.parsed_yaml_list,
                                          self.Ydf,
                                          target_value_rate_constant_csv= self.optional_targets[iteration],
                                          k_target_value_S_matrix =self.k_target_value_S_matrix,
                                          k_target_values=self.run_with_k_target_values,
                                          working_directory=self.wdir,
                                          batch_reactor_instance = self.MSI_instance_two,
                                          optimized_cti_file=self.MSI_instance_two.new_cti_file,
                                          original_cti_file=self.original_cti_file)
    
        self.observable_counter_and_absorbance_wl,self.length_of_experimental_data = plotting_instance.lengths_of_experimental_data()
        self.sigmas_optimized,test = plotting_instance.calculating_sigmas(self.S_matrix,self.covarience)
        self.sigmas_original,self.test2 = plotting_instance.calculating_sigmas(self.S_matrix_original,self.original_covariance)
        plotting_instance.plotting_observables(sigmas_original = self.sigmas_original,sigmas_optimized= self.sigmas_optimized,
                                               file_identifier=self.models[iteration].rstrip('.cti')+'_'+file_identifier,
                                               filetype='.pdf')
        self.diag = plotting_instance.getting_matrix_diag(self.covarience)
        
        #plotting_instance.Y_matrix_plotter(Y_matrix,exp_dict_list_optimized,y,sigma)
        
        
        
        plotting_instance.plotting_rate_constants(optimized_cti_file=self.MSI_instance_two.new_cti_file,
                                        original_cti_file=self.original_cti_file,
                                        initial_temperature=250,
                                        final_temperature=2500)
                                        
        
        
        self.sensitivity, self.top_sensitivity = plotting_instance.sort_top_uncertainty_weighted_sens()
        self.obs = plotting_instance.plotting_uncertainty_weighted_sens()
        
        
        
    def get_parameters(self):
        '''
        Container function for other functions to parse information from the input file.

        Returns
        -------
        None.

        '''
        self.get_working_directory()
        self.get_iterations()
        self.get_yaml_option()
        self.get_yaml_files()
        self.get_kinetic_models()
        self.get_reaction_uncertainties_list()
        self.get_master_equation_models()
        self.get_master_equation_uncertainties()
        self.get_targets()
        self.get_optional_plotting_targets()
        self.get_master_equation_sens()
        self.get_master_reactions()
        self.get_master_index()
        
    def get_working_directory(self):
        '''
        Reads 'self.input_options' and sets the working directory for MSI simulations.

        Returns
        -------
        None.

        '''
        self.wdir=self.input_options[0].split('=')[1].rstrip('\'').lstrip('\'')
        '''Path to the working directory for the MSI simulation'''
    
    def get_iterations(self):
        '''
        Reads 'self.input_options' and sets the number of iterations to run MSI.

        Returns
        -------
        None.

        '''
        self.iterations=int(self.input_options[1].split('=')[1])
        '''The number of iterations MSI will run'''
        
    def get_yaml_option(self):
        '''
        Reads 'self.input_options' and determines if MSI will run on experiments 
        sequentially or all at once.

        Raises
        ------
        Exception
            Exception is raised if the input option is set to neither True or False.

        Returns
        -------
        None.

        '''
        if re.match('[Ff][Aa][Ll][Ss][Ee]',self.input_options[2].split('=')[1]):
            self.run_individual_yaml=False
            '''Boolean determines whether or not yamls are run sequentially or 
            all at once.  True to run sequentially, False to run MSI for all experiments 
            simultaneously.'''
        elif re.match('[Tt][Rr][Uu][Ee]',self.input_options[2].split('=')[1]):
            self.run_individual_yaml=True
        else:
            raise Exception('Please set run_individual_yaml=True/False in the input file to run.')
        
    def get_yaml_files(self):
        '''
        Reads 'self.input_options' and returns a list of yaml files which contain
        experimental data.

        Returns
        -------
        None.

        '''
        yaml_bool=False
        self.yaml_files=[]
        '''Contains a list of yaml file lists.  Each item in the list has either 
        one or two strings - the second string is an optional string for the name
        of a yaml file containing absorbance data.'''
        for i,string in enumerate(self.input_options):
            if 'end_yaml' in string:
                break
            if yaml_bool:
                temp_entry=string.lstrip('[')
                temp_entry=temp_entry.rstrip(']')
                entry=temp_entry.split(',')
                #print(entry)
                if entry[1]=='':
                    entry=[entry[0]]
                self.yaml_files.append(entry)               
                
                
            if 'begin_yaml_list' in string:
                yaml_bool=True
            
            
    def get_kinetic_models(self):
        '''
        Reads 'self.input_options' and returns a list of kinetic models to run MSI 
        optimization.

        Returns
        -------
        None.

        '''
        model_bool=False
        self.models=[]
        '''List of kinetic models by file name'''
        for i,string in enumerate(self.input_options):
            if 'end_model_list' in string:
                break
            if model_bool:
                self.models.append(string)
            if 'begin_model_list' in string:
                model_bool=True
            
    def get_master_index(self):
        '''
        Reads 'self.input_options' and returns a list of lists of indices for each kinetic model.
        These indices correspond to reactions in the model that will be removed and replaced with 
        equivalent reactions treated with master-equation and given in the format
        of a Chebyshev reaction.

        Returns
        -------
        None.

        '''
        index_bool=False
        self.indices=[]
        '''List of lists of indices'''
        for i,string in enumerate(self.input_options):
            if 'end_master_equation_index' in string:
                break
            if index_bool:
                self.indices.append(list(map(int, string.lstrip('[').rstrip(']').split(','))))
            if 'begin_master_equation_index' in string:
                index_bool=True
                
        
    def get_reaction_uncertainties_list(self):
        '''
        Reads 'self.input_options' and returns a list of files containing 
        reaction parametric uncertainties.

        Returns
        -------
        None.

        '''
        uncertainties_bool=False
        self.reaction_uncertainties=[]
        '''List of strings containing file names of uncertainty csv files.'''
        for i,string in enumerate(self.input_options):
            if 'end_reaction_uncertainty_list' in string:
                break
            if uncertainties_bool:
                self.reaction_uncertainties.append(string)
            if 'begin_reaction_uncertainty_list' in string:
                uncertainties_bool=True
    def get_master_equation_models(self):
        '''
        Reads 'self.input_options' and returns the list of Cantera cti files containing the 
        reactions treated with master-equation, in Chenyshev format.

        Returns
        -------
        None.

        '''
        master_bool=False
        self.master_equation_models=[]
        for i,string in enumerate(self.input_options):
            if 'end_master_equation_model_list' in string:
                break
            if master_bool:
                self.master_equation_models.append(string)
            if 'begin_master_equation_model_list' in string:
                master_bool=True
                
                
    def get_master_equation_uncertainties(self):
        '''
        Reads 'self.input_options' and returns a list of files containing 
        master equation parametric uncertainties.

        Returns
        -------
        None.

        '''
        
        master_bool=False
        self.master_uncertainties=[]
        '''List of strings containing file names of uncertainty csv files.'''
        for i,string in enumerate(self.input_options):
            if 'end_master_equation_uncertainties' in string:
                break
            if master_bool:
                self.master_uncertainties.append(string)
            if 'begin_master_equation_uncertainties' in string:
                master_bool=True
                
    def get_targets(self):
        '''
        Reads 'self.input_options' and gets a list of rate constant targets for MSI.
        These are optional and ignored if not present in the input file.

        Returns
        -------
        None.

        '''
        targets_bool=False
        self.targets=[]
        for i,string in enumerate(self.input_options):  
            if 'end_rate_constant_targets' in string:
                break
            if targets_bool:
                self.targets.append(string)
            if 'begin_rate_constant_targets' in string:
                targets_bool=True
        if not self.targets:
            self.targets=['']*len(self.models)
        
    def get_optional_plotting_targets(self):
        '''
        Reads 'self.input_options' and gets a list of rate constant targets for MSI.
        This one is exclusively for use in generating higher fidelity plots.

        Returns
        -------
        None.

        '''
        targets_bool=False
        self.optional_targets=[]
        for i,string in enumerate(self.input_options):  
            if 'end_optional_plotting_targets' in string:
                break
            if targets_bool:
                self.optional_targets.append(string)
            if 'begin_optional_plotting_targets' in string:
                targets_bool=True
        if not self.optional_targets and all('' == s or s.isspace() for s in self.targets):
    
            self.optional_targets=['']*len(self.models)
        elif not self.optional_targets and not all('' == s or s.isspace() for s in self.targets):
            self.optional_targets=self.targets
                
                
                
    
    def get_master_equation_sens(self):
        '''
        This reads 'self.input_options' and returns a list of raw Python files 
        containing Chebyshev sensitivities.  

        Returns
        -------
        None.

        '''
        sens_bool=False
        self.master_sens=[]
        for i,string in enumerate(self.input_options):  
            if 'end_master_equation_sensitivities' in string:
                break
            if sens_bool:
                self.master_sens.append(string)
            if 'begin_master_equation_sensitivities' in string:
                sens_bool=True
            
            
    def get_master_reactions(self):
        '''
        Reads 'self.input_options' and returns a list of lists of reaction 
        equations corresponding to those reactions analyzed by the master equation.

        Returns
        -------
        None.

        '''
        rxn_list_len=len(self.models)
        set_names=[]
        reactions_bool=False
        set_bools=[False]*rxn_list_len
        self.master_equation_reactions=[]
        for i in range(rxn_list_len):
            set_names.append('set'+str(i+1))
        for i,string in enumerate(self.input_options):
            if 'end_master_equation_reactions' in string:
                break
            if reactions_bool:
                for j,string1 in enumerate(set_bools):
                    #print(j)
                    index_top=self.input_options.index('set'+str(j+1))
                    index_bottom=self.input_options.index('end set'+str(j+1))
                    self.master_equation_reactions.append(self.input_options[index_top+1:index_bottom])
                break
                
                self.master_sens.append(string)
            if 'begin_master_equation_reactions' in string:
                reactions_bool=True
        


def parser(input_file):
    '''
    Function to parse the MSI input file.  Reads the information in and removes
    extraneous whitespace and comments (declared with "#" at the beginning of a line).

    Parameters
    ----------
    input_file : String
        Path to the MSI input file.

    Returns
    -------
    input_lines : List
        List of strings enumerating the lines of the input file, without extra whitespace
        and comments.

    '''
    
    
    #print(#input_file)
    
    
    with open(input_file,'r') as f:
        input_lines=f.readlines()
        
    for i in range(len(input_lines)):
        #rint('char='+str(input_lines[i]))
        input_lines[i]=input_lines[i].lstrip()
        input_lines[i]=input_lines[i].rstrip('\n')
        input_lines[i]=input_lines[i].rstrip()
        if input_lines[i]:
            if input_lines[i][0]=='#':
               input_lines[i]=''
        
        
    while '' in input_lines:
        #print('a')
        input_lines.remove('')
    
      
    return input_lines
    
        



def main(input_file=''):
    '''
    An optimization code to run Multi-Scale-Informatics on 
    combustion experiments and first principles calculations, to better constrain 
    parameters within kinetic models.

    Parameters
    ----------
    input_file : String
        The default is ''.  Enter the path to the directory where the MSI simulation input file is located.

    Returns
    -------
    simulation : TYPE
        DESCRIPTION.

    '''
    if input_file=='':
        print('Please run program with defined input file using --input_file=FILEPATH')

    elif input_file !='':
        
        input_options=parser(input_file)
        simulation=multiscale_informatics(input_options)
        simulation.run_msi()
        simulation.write_convergence()
        
        return simulation

if __name__ == '__main__':
    a=fire.Fire(main)