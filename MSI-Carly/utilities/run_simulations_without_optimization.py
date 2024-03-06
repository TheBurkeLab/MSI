import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.shock_tube as st
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


class running_simulations_without_optimization(object):
        
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
                temp.append(self.data_directory+'/'+file)
            temp = tuple(temp)
            yaml_file_list_with_working_directory.append(temp)
        self.yaml_file_list_with_working_directory = yaml_file_list_with_working_directory
        return
# pre process the cti file to remove reactions and rename it,also save it as the first run of the file
        
    # run the cti writer to establish processor and make cti file 
    def establish_processor(self,loop_counter=0):
        if loop_counter==0 and self.master_equation_flag==False:
            new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(original_cti=self.data_directory +'/'+ self.cti_file_name,
                                                                      working_directory=self.data_directory,
                                                                      file_name= self.cti_file_name.replace('.cti','')+'_updated')
            self.new_cti_file = new_file
             
        if loop_counter==0 and self.master_equation_flag==True:
            new_file,original_rxn_eqs,master_rxn_eqs =ctic.cti_write2(original_cti=self.data_directory +'/'+ self.cti_file_name,
                                                                      master_rxns = self.data_directory+'/'+self.master_reaction_equation_cti_name,
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
    
    def running_shock_tube_simulations(self,loop_counter=0):
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
            self.experiment_dictonaries = experiment_dictonaries
            #call function taht loops opver experient dicts og and saves them
            

        #maybe save this and just pass it in 
        return
    def one_run_shock_tube_optimization(self,loop_counter=0):
        self.append_working_directory()
        #every loop run this, probably not?
        self.establish_processor(loop_counter=loop_counter)
        self.parsing_yaml_files(loop_counter = loop_counter)

        
        if loop_counter == 0:
            original_experimental_conditions_local = copy.deepcopy(self.yaml_instance.original_experimental_conditions)
            self.original_experimental_conditions_local = original_experimental_conditions_local
            #self.coupled_coefficients_original = copy.deepcopy(original_experimental_conditions_local[0]['coupledCoefficients'])
        
        
        self.running_shock_tube_simulations(loop_counter=loop_counter)

    
        
    def multiple_shock_tube_runs(self,loops):
        for loop in range(loops):            
            self.one_run_shock_tube_optimization(loop_counter=loop)
            
        return 
            
    
                                                                  
        
                                                       
            
                



