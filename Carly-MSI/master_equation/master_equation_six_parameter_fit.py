
import numpy as np

import numpy as np
import pandas as pd
import cantera as ct

class Master_Equation_Six_Parameter_Fit(object):
    def __init__(self):
        self.matrix = None
        
        
    def master_equation_handling(self, exp_dict_list:list, 
                                     parsed_yaml_file_list:list,
                                     master_equation_sensitivites:dict,
                                     master_equation_reactions:list): 
            
           
            def slicing_out_reactions(reaction_string,array):
                reactions_in_cti_file = exp_dict_list[0]['simulation'].processor.solution.reaction_equations()
                index_of_reaction_in_cti = reactions_in_cti_file.index(reaction_string)
                column_of_array = array[:,index_of_reaction_in_cti]
                column_of_array = column_of_array.reshape((column_of_array.shape[0],
                                                              1))                  
                return column_of_array
            
            def assemble_slices_into_array(list_of_columns):
                array = np.hstack((list_of_columns))
                return array
            
            def multiply_by_sensitivites(array,list_of_sensitivites):
                sensitivity_multiplied_list = []                
                for sensitivity in list_of_sensitivites:
                    temp = np.multiply(array,sensitivity)
                    temp =  temp.reshape((temp.shape[0],1))
                    sensitivity_multiplied_list.append(temp)
                sensitivity_mapped_array = np.hstack((sensitivity_multiplied_list))
                return sensitivity_mapped_array
            

            
            k_mapping = []   
            for i,exp in enumerate(exp_dict_list):
                
                vertically_stacked_single_experiment_A = []
                vertically_stacked_single_experiment_N = []
                vertically_stacked_single_experiment_Ea = []
                if parsed_yaml_file_list[i]['moleFractionObservables'][0] != None or parsed_yaml_file_list[i]['concentrationObservables'][0] !=None or parsed_yaml_file_list[i]['ignitionDelayObservables'][0]!=None:
                    As = exp['ksens']['A']
                    A_observable_arrays = []
                    for observable in As:
                        temp = []
                        for reaction in master_equation_reactions:
                            column = slicing_out_reactions(reaction,observable)
                            array_mapped_to_sensitivity = multiply_by_sensitivites(column,master_equation_sensitivites[reaction]['A'])    
                            temp.append(array_mapped_to_sensitivity)
                        temp = np.hstack((temp))
    
                        A_observable_arrays.append(temp)
                        
                    Ns = exp['ksens']['N']
                    N_observable_arrays = []    
                    for p,observable in enumerate(Ns):
                        temp=[]
                        for reaction in master_equation_reactions:
                            column = slicing_out_reactions(reaction,observable)
                            array_mapped_to_sensitivity = multiply_by_sensitivites(column,master_equation_sensitivites[reaction]['n'])
                            
                            #reverse mapping array
                            #array_mapped_to_N = mapping_N(array_mapped_to_sensitivity,exp['simulation'].pressureAndTemperatureToExperiment[p]['temperature'])
                            temp.append(array_mapped_to_sensitivity)
                        temp = np.hstack((temp))
                        
                        N_observable_arrays.append(temp)
                     
                    Eas = exp['ksens']['Ea']  
                    Ea_observable_arrays = [] 
                    for observable in Eas:
                        temp = []
                        for reaction in master_equation_reactions:
                            column = slicing_out_reactions(reaction,observable)
                            array_mapped_to_sensitivity = multiply_by_sensitivites(column*1000,master_equation_sensitivites[reaction]['Ea'])
                            #multiply by 1000 to match burke's coef

                            temp.append(array_mapped_to_sensitivity)
                            
                        temp = np.hstack((temp))
                        Ea_observable_arrays.append(temp)
                        
                    #vertically stacked arrays
                    vertically_stacked_A_arrays = np.vstack((A_observable_arrays))
                    vertically_stacked_n_arrays = np.vstack((N_observable_arrays))
                    vertically_stacked_Ea_arrays = np.vstack((Ea_observable_arrays))
                    
                    
                    #appending things to overall experiment list which will include absorbance and 
                    #mole fractions and concentration observables if they apply
                    vertically_stacked_single_experiment_A.append(vertically_stacked_A_arrays)
                    vertically_stacked_single_experiment_N.append(vertically_stacked_n_arrays)
                    vertically_stacked_single_experiment_Ea.append(vertically_stacked_Ea_arrays)
                    
                
                if 'absorbance_observables' in list(exp.keys()):
                    
                    wavelengths = parsed_yaml_file_list[i]['absorbanceCsvWavelengths']
                    absorbance_k_sens = exp['absorbance_ksens']
                    #absorbance k sens is a dict with wavelenghts as keys and the values are lists of 3 things A,N,Ea sensitivity arrays
                    A_absorbance_observable_arrays = []
                    N_absorbance_observable_arrays = []
                    Ea_absorbance_observable_arrays = []
                        
                    for k,wl in enumerate(wavelengths):
                        temp = []
                        for reaction in master_equation_reactions:
                            column = slicing_out_reactions(reaction,absorbance_k_sens[wl][0])
                            array_mapped_to_sensitivity = multiply_by_sensitivites(column,master_equation_sensitivites[reaction]['A']) 
                            temp.append(array_mapped_to_sensitivity)
                        temp = np.hstack((temp))
                        A_absorbance_observable_arrays.append(temp)
                        
                    for k,wl in enumerate(wavelengths):
                        temp=[]
                        for reaction in master_equation_reactions:
                            column = slicing_out_reactions(reaction,absorbance_k_sens[wl][1])
                            array_mapped_to_sensitivity = multiply_by_sensitivites(column,master_equation_sensitivites[reaction]['n']) 
                            
                            #reverse mapping of a single column 
                            #array_mapped_to_N = mapping_N(array_mapped_to_sensitivity,exp['time_history_interpolated_against_abs'][wl]['temperature'])
                            temp.append(array_mapped_to_sensitivity)
                        temp = np.hstack((temp))
                        N_absorbance_observable_arrays.append(temp)
                        
                    for k,wl in enumerate(wavelengths):
                        temp = []
                        for reaction in master_equation_reactions:
                            column = slicing_out_reactions(reaction,absorbance_k_sens[wl][2])
                            array_mapped_to_sensitivity = multiply_by_sensitivites(column*1000,master_equation_sensitivites[reaction]['Ea'])
                            #units of K so that's fine but burke is multiplying by 1000?
                            #SO first check if we multiply the column by 1000 if we still get the same results 
                             
                            temp.append(array_mapped_to_sensitivity)
                            
                        temp = np.hstack(temp)
                        Ea_absorbance_observable_arrays.append(temp)         
                        
                    vertically_stacked_A_absorbance_array = np.vstack((A_absorbance_observable_arrays))
                    vertically_stacked_n_absorbance_array = np.vstack((N_absorbance_observable_arrays))
                    vertically_stacked_Ea_absorbance_array = np.vstack((Ea_absorbance_observable_arrays))
                    
                    vertically_stacked_single_experiment_A.append(vertically_stacked_A_absorbance_array)
                    vertically_stacked_single_experiment_N.append(vertically_stacked_n_absorbance_array)
                    vertically_stacked_single_experiment_Ea.append(vertically_stacked_Ea_absorbance_array)
                
                vertically_stacked_single_experiment_A = np.vstack((vertically_stacked_single_experiment_A))
                vertically_stacked_single_experiment_N = np.vstack((vertically_stacked_single_experiment_N))
                vertically_stacked_single_experiment_Ea = np.vstack((vertically_stacked_single_experiment_Ea))
                
                #mapping the single experiment back to K
                k_mapping_single_experiment = vertically_stacked_single_experiment_A + vertically_stacked_single_experiment_N + vertically_stacked_single_experiment_Ea
                k_mapping.append(k_mapping_single_experiment)
                
                
            k_mapping = np.vstack((k_mapping))
            
            return k_mapping
        
    def surrogate_model_molecular_parameters(self,master_equation_sensitivites:dict,
                                                  master_equation_reactions:list,
                                                  delta_x_molecular_params_by_reaction_dict,
                                                  exp_dict_list):
            
            reactions_in_cti_file = exp_dict_list[0]['simulation'].processor.solution.reaction_equations()
            number_of_reactions = len(reactions_in_cti_file)
            As = []
            Ns = []
            Eas = []
            for numb,reaction in enumerate(master_equation_reactions):
                tempA=[]
                for i,sensitivty in enumerate(master_equation_sensitivites[reaction]['A']):
                    tempA.append(sensitivty*delta_x_molecular_params_by_reaction_dict[reaction][i])
                    #tempA.append(sensitivty*delta_x_molecular_params_by_reaction_dict['R_'+str(numb)][i])
                
                sum_A = sum(tempA)
                As.append(sum_A)
                 
                tempN = []
                for i,sensitivty in enumerate(master_equation_sensitivites[reaction]['n']):
                    tempN.append(sensitivty*delta_x_molecular_params_by_reaction_dict[reaction][i])
                    #tempN.append(sensitivty*delta_x_molecular_params_by_reaction_dict['R_'+str(numb)][i])
                sum_N = sum(tempN)
                Ns.append(sum_N)
    
                tempEa = []
                for i,sensitivty in enumerate(master_equation_sensitivites[reaction]['Ea']):
                    tempEa.append(sensitivty*delta_x_molecular_params_by_reaction_dict[reaction][i])
                    #tempEa.append(sensitivty*delta_x_molecular_params_by_reaction_dict['R_'+str(numb)][i])
                sum_Ea = sum(tempEa)*1000.0*1.987*4184
                #First multiply by 1.987 (R) to get Ea in units of kcal/mol
                #Then covert to J/kmol 1000*4.186e3
                Eas.append(sum_Ea)
                
            
           
            
           
            AsNsEas =[[] for x in range(len(master_equation_reactions))]
            for x in range(len(As)):
                AsNsEas[x].append(As[x])
                AsNsEas[x].append(Ns[x])
                AsNsEas[x].append(Eas[x])
            innerDict = ['A','n','Ea']   
            l = [dict(zip(innerDict,AsNsEas[x])) for x in range(len(AsNsEas))]
            Keys = []
    
            for x in np.arange(number_of_reactions-len(master_equation_reactions),number_of_reactions):
                Keys.append('r'+str(x))
    
            MP = dict(zip(Keys,l))
            #print(MP,'this is MP')
            return MP
        
        
        
    # add dataframe 
    def target_values_for_Z_six_paramter_fit(self,target_value_csv,z_data_Frame):
        z_over_w = []
        sigma = []
        target_value_csv = pd.read_csv(target_value_csv)
        target_ln_uncertainty = target_value_csv['ln_unc_k']
        target_W = target_value_csv['W']
        target_reactions = target_value_csv['Reaction']
        z_df_list=[]
        z_values = []
        for i,value in enumerate(target_ln_uncertainty):
            temp = np.divide(value,target_W[i])
            sigma.append(value)
            z_over_w.append(temp)
            z_values.append(temp)
            z_df_list.append(target_reactions[i])
            
        k_targets_for_z = np.array(z_over_w)
        sigma = np.array(sigma)
        sigma = sigma.reshape((sigma.shape[0],1))
        z_values = np.array(z_values)
        k_targets_for_z = k_targets_for_z.reshape((k_targets_for_z.shape[0],1))
        Z_data_Frame_temp = pd.DataFrame({'value': z_df_list,'Uncertainty': z_values.reshape((z_values.shape[0],))})
        z_data_Frame = z_data_Frame.append(Z_data_Frame_temp, ignore_index=True) 
        return k_targets_for_z,sigma,z_data_Frame
            
        

    def update_six_paramter_fits_dict(self,six_parameter_fit_sensitivity_dict:dict, 
        molecular_parameter_updates_dict:dict,
        master_equation_reactions:list,
        nominal_parameter_dictonary:dict):
        fit_variables = ['A','n','Ea','c','d','f']
        six_paramter_fit_update_dict = {}
        
        def multiply_sensitivty_MP_updates(reaction,variable):
                array = np.array(())
                #print(molecular_parameter_updates_dict)
                temp = np.array(six_parameter_fit_sensitivity_dict[reaction][variable])*np.array(molecular_parameter_updates_dict[reaction])

                
                array = sum(temp)
                #if reaction == 'H2O2 + OH <=> H2O + HO2' and variable =='A' : 
                    #print(temp)
                    #print(array)
                return array

        for reaction in master_equation_reactions:
            temp_dict = {}
            for variable in fit_variables:
                summation = multiply_sensitivty_MP_updates(reaction,variable)
               # print(summation)
                #need to append here
                
                temp_dict[variable] = summation
            six_paramter_fit_update_dict.update({reaction:temp_dict})



        updated_six_paramter_fits_dict = {}
        for reaction in  master_equation_reactions:
            temp_dict = {}

            
            temp_dict = {'A':np.exp(six_paramter_fit_update_dict[reaction]['A'])*nominal_parameter_dictonary[reaction]['A'],
                         'n':six_paramter_fit_update_dict[reaction]['n']+nominal_parameter_dictonary[reaction]['n'],
                         'Ea': six_paramter_fit_update_dict[reaction]['Ea']*1.987*1000 + nominal_parameter_dictonary[reaction]['Ea'],
                         'c':six_paramter_fit_update_dict[reaction]['c']*((1.987*1000)**3) + nominal_parameter_dictonary[reaction]['c'],
                         'd':six_paramter_fit_update_dict[reaction]['d']*((1.987*1000)**-1) + nominal_parameter_dictonary[reaction]['d'],
                         'f':six_paramter_fit_update_dict[reaction]['f']*((1.987*1000)**-3) + nominal_parameter_dictonary[reaction]['f']}          
            
            
            updated_six_paramter_fits_dict[reaction] = temp_dict


        return updated_six_paramter_fits_dict

    def calculate_six_parameter_fit(self,reaction,dictonary,temperature):
        #finish editing this 
        #calc Ea,c,d,F seprately 
            A = dictonary[reaction]['A']
            n = dictonary[reaction]['n']
            Ea_temp = dictonary[reaction]['Ea']/(1.987*temperature)
            c_temp = dictonary[reaction]['c']/((1.987*temperature)**3)
            d_temp = dictonary[reaction]['d']*(1.987*temperature)
            f_temp = dictonary[reaction]['f']* ((1.987*temperature)**3)
            

            k = A*(temperature**n)*np.exp(-Ea_temp-c_temp-d_temp-f_temp)
            return k 
    # add dataframe 
    def target_values_Y_six_parameter_fit(self,target_value_csv,
                                          exp_dict_list:list,
                                          Y_data_frame,
                                          master_equation_reaction_list=[],
                                          updated_six_paramter_fits_dict={}):

        Y_df_list = []
        Y_values = []
        #make sure we put the reactions into the file in the units cantera uses
        target_value_csv = pd.read_csv(target_value_csv)
        target_reactions = target_value_csv['Reaction']
        target_temp = target_value_csv['temperature']
        target_press = target_value_csv['pressure']
        target_k = target_value_csv['k']
        bath_gas = target_value_csv['M']
        reactions_in_cti_file = exp_dict_list[0]['simulation'].processor.solution.reaction_equations()
        gas = ct.Solution(exp_dict_list[0]['simulation'].processor.cti_path)
        diff_in_ks_for_Y = []

        #for reaction in master_equation_reactions:
            

        for i,reaction in enumerate(target_reactions): 
            if reaction in master_equation_reaction_list:
                #check untis might needto change these units  either multiply or divide by 1000
                k = self.calculate_six_parameter_fit(reaction,updated_six_paramter_fits_dict,target_temp[i])
                #convert units
                k = k
                #print(k,target_k[i])
            else:
                #ask about the mixture composition
                if target_press[i] == 0:
                    pressure = 1e-9
                else:
                    pressure = target_press[i]
                if bath_gas[i] !=0:
                    gas.TPX = target_temp[i],pressure*101325,{'H2O':.013,'O2':.0099,'H':.0000007,'Ar':.9770993}
    
                else:
                    gas.TPX = target_temp[i],pressure*101325,{'Ar':.99}
                reaction_number_in_cti = reactions_in_cti_file.index(reaction)
                k = gas.forward_rate_constants[reaction_number_in_cti]
                #check units on original stuff
                k=k*1000
                
                

                    
            difference = np.log(target_k[i]) - np.log(k) 
            


            diff_in_ks_for_Y.append(difference)
            Y_df_list.append(reaction)
            Y_values.append(difference)
            
        k_targets_for_y = np.array(diff_in_ks_for_Y)
        k_targets_for_y = k_targets_for_y.reshape((k_targets_for_y.shape[0],1))
        Y_values = np.array(Y_values)
        
        Y_df_temp = pd.DataFrame({'value': Y_df_list,'ln_difference': Y_values.reshape((Y_values.shape[0],))}) 
        Y_data_frame = Y_data_frame.append(Y_df_temp, ignore_index=True)
        
        return k_targets_for_y,Y_data_frame








#########################################################################################################################

    def target_values_for_S_six_parameter_fit(self,target_value_csv,
                                exp_dict_list,
                                S_matrix,
                                master_equation_reaction_list = [],
                                six_parameter_fit_sensitivity_dict = {}):
                
                
                
                
            target_value_csv = pd.read_csv(target_value_csv)
            target_reactions = target_value_csv['Reaction']
            target_temp = target_value_csv['temperature']
            target_press = target_value_csv['pressure']
            target_k = target_value_csv['k']
            reactions_in_cti_file = exp_dict_list[0]['simulation'].processor.solution.reaction_equations()
            number_of_reactions_in_cti = len(reactions_in_cti_file)
            As = []
            Ns =  []
            Eas = []
                
            Number_of_MP = []
            #nested_reaction_list = [[] for x in range(len(master_equation_reaction_list))]
            #print(six_parameter_fit_sensitivity_dict.keys())
            
            def create_empty_nested_reaction_list():
                nested_reaction_list = [[] for x in range(len(master_equation_reaction_list))]
                for reaction in master_equation_reaction_list:
                    for i,MP in enumerate(six_parameter_fit_sensitivity_dict[reaction]['A']):
                        nested_reaction_list[master_equation_reaction_list.index(reaction)].append(0)
                #copy.deepcopy(nested_reaction_list) 
                #don't think i need this 
                return nested_reaction_list          
                  
            MP_stack = []
            target_values_to_stack =  []
            for i,reaction in enumerate(target_reactions):
                #temp_array = np.zeros((1,Number_of_MP))
                if reaction in master_equation_reaction_list:
                    nested_reaction_list = create_empty_nested_reaction_list()
                    for s,sensitivity in enumerate(six_parameter_fit_sensitivity_dict[reaction]['A']):
                        #stub
                        # stub addition 4.186
                        nested_reaction_list[master_equation_reaction_list.index(reaction)][s] = 1*six_parameter_fit_sensitivity_dict[reaction]['A'][s] + np.log(target_temp[i])*six_parameter_fit_sensitivity_dict[reaction]['n'][s] + (-1000/target_temp[i])*six_parameter_fit_sensitivity_dict[reaction]['Ea'][s] + (-(1000/target_temp[i])**3)*six_parameter_fit_sensitivity_dict[reaction]['c'][s]+ (-(1000/target_temp[i])**-1)*six_parameter_fit_sensitivity_dict[reaction]['d'][s] + (-(1000/target_temp[i])**-3)*six_parameter_fit_sensitivity_dict[reaction]['f'][s]
                    temp  = nested_reaction_list
                    flat_list = [item for sublist in temp for item in sublist]
                    MP_stack.append(nested_reaction_list)
                    flat_list = np.array(flat_list)
                    flat_list = flat_list.reshape((1,flat_list.shape[0])) 
                    target_values_to_stack.append(flat_list)
                    
                            
                                      
                        
                else:
                    A_temp = np.zeros((1,number_of_reactions_in_cti-len(master_equation_reaction_list)))
    
                    N_temp = np.zeros((1,number_of_reactions_in_cti-len(master_equation_reaction_list)))
                    Ea_temp = np.zeros((1,number_of_reactions_in_cti-len(master_equation_reaction_list)))
                        #decide if this mapping is correct             
                    A_temp[0,reactions_in_cti_file.index(reaction)] = 1
                    N_temp [0,reactions_in_cti_file.index(reaction)] = np.log(target_temp[i])
                    Ea_temp[0,reactions_in_cti_file.index(reaction)] = (-1/target_temp[i])
                    
                    As.append(A_temp)
                    Ns.append(N_temp)
                    Eas.append(Ea_temp)
                    A_temp = A_temp.reshape((1,A_temp.shape[1]))
                    N_temp = N_temp.reshape((1,N_temp.shape[1]))
                    Ea_temp = Ea_temp.reshape((1,Ea_temp.shape[1]))
                    target_values_to_stack.append(np.hstack((A_temp,N_temp,Ea_temp)))
                    
                    
               # might need to edit this to pass in s? and  
            S_matrix = S_matrix
            shape_s = S_matrix.shape
            S_target_values = []
            for i,row in enumerate(target_values_to_stack):
                if target_reactions[i] in master_equation_reaction_list:
                    zero_to_append_infront = np.zeros((1,((number_of_reactions_in_cti-len(master_equation_reaction_list))*3)))
                    
                    zero_to_append_behind = np.zeros((1, shape_s[1] - ((number_of_reactions_in_cti-len(master_equation_reaction_list))*3) - np.shape(row)[1] ))                
                    temp_array = np.hstack((zero_to_append_infront,row,zero_to_append_behind))
                    S_target_values.append(temp_array)
                else:
                    zero_to_append_behind = np.zeros((1,shape_s[1]-np.shape(row)[1]))
                    temp_array = np.hstack((row,zero_to_append_behind))
                    S_target_values.append(temp_array)


            S_target_values = np.vstack((S_target_values))
            return S_target_values




    def appending_target_values(self,target_values_for_z,
                                target_values_for_y,
                                target_values_for_s,
                                sigma_target_values,
                                S_matrix,
                                Y_matrix,
                                z_matrix,
                                sigma):
        z_matrix = np.vstack((z_matrix ,target_values_for_z))
        Y_matrix = np.vstack((Y_matrix,target_values_for_y))
        
        S_matrix = np.vstack((S_matrix,target_values_for_s))
        sigma = np.vstack((sigma,sigma_target_values))
        
        self.S_matrix = S_matrix
        self.Y_matrix = Y_matrix
        self.z_matrix = z_matrix
        self.sigma = sigma
        
        return S_matrix,Y_matrix,z_matrix,sigma













