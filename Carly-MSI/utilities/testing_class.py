import numpy as np
import cantera as ct
import os 
import pandas as pd
class testing_code(object):
    def __init__(self,shape_of_X=None,
                 shape_of_X_counter=None,value=.01,
                 Y_matrix = None,experimental_dict_list=None,
                 S_matrix_original=None,Y_matrix_original=None,
                 number_of_reactions_in_cti_file = 288,macroscopic_test_data_directory = None):
        self.S_matrix = S_matrix_original
        self.s_matrix = None
        self.Y_matrix = Y_matrix
        self.y_matrix = None
        self.shape_of_X = shape_of_X
        self.value_to_perturb_by = value
        self.experimental_dict_list = experimental_dict_list
        self.shape_of_X_counter = shape_of_X_counter
        self.Y_matrix_original = Y_matrix_original
        self.number_of_reactions_in_cti_file = number_of_reactions_in_cti_file
        self.macroscopic_test_data_directory = macroscopic_test_data_directory
        
        
        
        
    def adjust_macroscopic_target_uncertainty(self,Z_matrix,simulation=0,experiment=0):
        data_length = self.simulation_lengths_of_experimental_data[simulation][experiment]
        Z_matrix[data_length,:] = 1e-6
        # if we waant to change physical observabLE uncertainty 
        #do that here  also 
        
                
        return Z_matrix
    
    def change_uncertainty_of_targets(self):
        return 
    def perturb_X(self,counter):
        
       
        zeroed_out = np.zeros((self.shape_of_X))
        zeroed_out[counter,:] = self.value_to_perturb_by
        
        return zeroed_out
    
    
    
    def calculate_Sij(self):
        start=0 
        S_percent_difference = np.zeros(np.shape(self.S_matrix))
        S_residuals = np.zeros((np.shape(self.S_matrix)))
        S_new = np.zeros((np.shape(self.S_matrix)))
        
        for x in range(len(self.simulation_lengths_of_experimental_data)):
            for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                stop = self.simulation_lengths_of_experimental_data[x][y] + start
                original = self.S_matrix[start:stop,self.shape_of_X_counter]
                original = original.reshape(original.shape[0],1)
                #numerator =   self.Y_matrix[start:stop,:] -self.Y_matrix_original[start:stop,:]
                numerator =   self.Y_matrix_original[start:stop,:] - self.Y_matrix[start:stop,:]
                
                denominator = self.value_to_perturb_by

                #if (self.shape_of_X_counter > self.number_of_reactions_in_cti_file*2)-6 and self.shape_of_X_counter < (self.number_of_reactions_in_cti_file*3)-6:
                    #denominator = self.value_to_perturb_by * ct.gas_constant
                
                Sij = np.divide(numerator,denominator)
                S_temp = Sij
                S_temp = S_temp.reshape(S_temp.shape[0],)
                S_new[start:stop,self.shape_of_X_counter] = S_temp
                residuals = Sij-original
                percent_difference = np.divide(residuals,Sij)
                percent_difference = percent_difference*100
                #residuals = np.multiply(residuals,100)
                residuals = residuals.reshape(residuals.shape[0],)
                S_residuals[start:stop,self.shape_of_X_counter] = residuals
                
                percent_difference = percent_difference.reshape(percent_difference.shape[0],)
                S_percent_difference[start:stop,self.shape_of_X_counter] = percent_difference
                start = start + self.simulation_lengths_of_experimental_data[x][y]
        
        return Sij, numerator,S_new,S_percent_difference
                
    def lengths_of_experimental_data(self):
        simulation_lengths_of_experimental_data = []
        for i,exp in enumerate(self.experimental_dict_list):
            length_of_experimental_data=[]
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables'] + exp['ignition_delay_observables']):
                if observable == None:
                    continue
                if observable in exp['mole_fraction_observables']:
                    length_of_experimental_data.append(exp['experimental_data'][observable_counter].shape[0])
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:
                    length_of_experimental_data.append(exp['experimental_data'][observable_counter].shape[0])
                    observable_counter+=1
                if observable in exp['ignition_delay_observables']:
                    length_of_experimental_data.append(exp['experimental_data'][observable_counter].shape[0])
                    observable_counter+=1                    
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list[i]['absorbanceCsvWavelengths']
                absorbance_wl=0
                for k,wl in enumerate(wavelengths):
                    length_of_experimental_data.append(exp['absorbance_experimental_data'][k].shape[0])
                    absorbance_wl+=1
            else:
                absorbance_wl=0
                    
            simulation_lengths_of_experimental_data.append(length_of_experimental_data)
            
                    
        self.simulation_lengths_of_experimental_data=simulation_lengths_of_experimental_data
        
        return 
    
    def get_files_in_working_directory(self):
        folder_list = os.listdir(self.macroscopic_test_data_directory)
        file_list = []
        for folder in folder_list:
            
            file_list.append(os.listdir(self.macroscopic_test_data_directory+'/'+folder))
            
            
        return file_list,folder_list
    
    
    

    def sorting_files(self,file_list):
        absorption_yaml_file = None
        reaction_uncertainty_targets = None
        for file in file_list:
            if '.yaml' in file and 'updated' not in file and 'abs' not in file:
                yaml_file = file
            if '.cti' in file and 'updated' not in file:
                cti_file = file
            if '.csv' in file and 'uncertainty'  in file:
                reaction_uncertainty_file = file
            if '.yaml' in file and 'abs' in file and 'updated' not in file:
                absorption_yaml_file = file
            if '.csv' in file and 'target' in file:
                reaction_uncertainty_targets = file
            
        if reaction_uncertainty_targets == None:
            return yaml_file,cti_file,reaction_uncertainty_file,absorption_yaml_file
        else:
            return yaml_file,cti_file,reaction_uncertainty_file,absorption_yaml_file,reaction_uncertainty_targets
    
    
    def making_target_value_rate_constant_csv(self,cti_file):
        T = 1500
        P = 101325
        X = {'H2O2':0.00195373,'Ar':0.99804627}
        factor = 2
        gas = ct.Solution(cti_file)
        gas.TPX = T,P,X
        reaction_list = gas.reaction_equations() 
        data_frame_list  = []
        for i,reaction in enumerate(reaction_list):
            k = gas.forward_rate_constants[i] * factor
            df = pd.DataFrame({'Reaction':[reaction],'temperature':[T],'pressure':[P],'M':[0],'k':[k],'ln_unc_k':[.693],'W':[1]})
            data_frame_list.append(df)
                        
        return data_frame_list
    
    def write_target_value_rate_constant_csv_to_directory(self,working_directory,data_frame_list):
        folder_list = os.listdir(working_directory)
        for i,folder in enumerate(folder_list):
            data_frame_list[i].to_csv(working_directory+'/'+'reaction_'+str(i)+'/'+'reaction_target_reaction_'+str(i)+'.csv',index=False)            
        return 
    
    def make_reaction_uncertainty_csv(self,cti_file):
        data_frame_list = []
        gas = ct.Solution(cti_file)
        reaction_list = gas.reaction_equations()
        A_uncertainty = [1e-9]*len(reaction_list)
        N_uncertainty = [1e-9]*len(reaction_list)
        Ea_uncertainty = [1e-9]*len(reaction_list)
        for i,reaction in enumerate(reaction_list):
            A_uncertainty[i] = 1e6
            df = pd.DataFrame({'Reaction':reaction_list,'Uncertainty A (unit)':A_uncertainty,'Uncertainty N (unit)':N_uncertainty,'Uncertainty Ea (unit)':Ea_uncertainty})
            data_frame_list.append(df)
            A_uncertainty[i] = 1e-9
            
        return data_frame_list
    
    def write_reaction_uncertainty_csv_to_directory(self,working_directory,data_frame_list):
        folder_list = os.listdir(working_directory)
        for i,folder in enumerate(folder_list):
            data_frame_list[i].to_csv(working_directory+'/'+'reaction_'+str(i)+'/'+'reaction_uncertainty_reaction_'+str(i)+'.csv',index=False)
        return
#    def write_reaction_unertainty_files(self,working_directory,reaction_list):
#        folder_list = os.listdir(working_directory)
    
        