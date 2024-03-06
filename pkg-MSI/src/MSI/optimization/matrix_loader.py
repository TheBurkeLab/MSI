import numpy as np
import pandas as pd
from ..master_equation import master_equation as meq
#import MSI.master_equation.master_equation as meq 
import copy
import re
import cantera as ct

class OptMatrix(object):
    def __init__(self):
        self.S_matrix = None
        self.s_matrix = None
        self.Y_matrix = None
        self.y_matrix = None
        self.z_matrix = None
        self.delta_X  = None
        self.X = None
        self.sigma = None
 
#    #loads one experiment into self.matrix. Decides padding based on previous matrix or handle based on total exp num?

    def build_Z(self, exp_dict_list:list,
                parsed_yaml_file_list:list,
                loop_counter:int = 0,
                reaction_uncertainty=None,
                master_equation_uncertainty_df=None,
                master_equation_reaction_list=[],
                master_equation_flag = False):
        '''
        Builds the Z vector. 
        
        Arguments:
            exp_dic_list -- the dictionary that is built after a simulation
            that contains things like sensitivity coefficients
            parsed_yaml_file_list -- a list of dictonaries that contain the 
            information stored in the yaml files. 
        Keyword Arguments:
            loop_counter -- keeps track of the iteration number for the optimization (default 0)
            reaction_uncertainty -- a csv file that contains all the reactions
            in the cti file being used for optimization and their corresponding
            A,n and Ea uncertainty values (default None)
            master_equation_uncertainty_df -- a pandas dataframe that contains
            the reactions being treated with theory paramters along with the 
            associated uncertainty values of those paramters (default None)
            master_equation_reaction_list -- a list of the reactions being treated
            with theory paramters (default [])
            master_equation_flag -- a boolean that indicates if reactions being
            represented by theory parameters are being used in the optimization (default False)
            
        '''
        Z = []
        Z_data_Frame = [] 
        sigma = []
        def jsr_temp_uncertainties(experiment_dict):
            if 'Relative_Uncertainty' in list(experiment_dict['experimental_data'][0].columns):
                temp_uncertainties=experiment_dict['experimental_data'][0]['Relative_Uncertainty'].values
                temp_uncertainties = list(temp_uncertainties)
            else:
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']*np.ones(np.shape(experiment_dict['experimental_data'][0]['Temperature'].values))
                temp_uncertainties = list(temp_uncertainties)
            return temp_uncertainties
        def flow_reactor_time_shift_uncertainties(parsed_yaml_file_list,experiment_dict):
            if len(parsed_yaml_file_list['timeShiftOriginal']) ==1:
                time_shift_uncertainties = [experiment_dict['uncertainty']['time_shift_uncertainty']]
            elif len(parsed_yaml_file_list['timeShiftOriginal']) >1:
                time_shift_uncertainties = [experiment_dict['uncertainty']['time_shift_uncertainty']]*len(parsed_yaml_file_list['timeShiftOriginal'])
            return time_shift_uncertainties
        def flow_reactor_temp_uncertainties(experiment_dict):
            if 'Temperature_Uncertainty' in list(experiment_dict['experimental_data'][0].columns):
                temp_uncertainties=experiment_dict['experimental_data'][0]['Temperature_Uncertainty'].values
                temp_uncertainties = list(temp_uncertainties)
            else:
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']*np.ones(np.shape(experiment_dict['experimental_data'][0]['Temperature'].values))
                temp_uncertainties = list(temp_uncertainties)
            return temp_uncertainties           
            
        def flame_speed_temp_uncertainties(experiment_dict):
            if 'Relative_Uncertainty' in list(experiment_dict['experimental_data'][0].columns) and 'Temperature' in list(experiment_dict['experimental_data'][0].columns):
                temp_uncertainties=experiment_dict['experimental_data'][0]['Relative_Uncertainty'].values
                temp_uncertainties = list(temp_uncertainties)
            elif 'Temperature' in list(experiment_dict['experimental_data'][0].columns):
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']*np.ones(np.shape(experiment_dict['experimental_data'][0]['Temperature'].values))
                temp_uncertainties = list(temp_uncertainties) 
            elif 'Pressure' in list(experiment_dict['experimental_data'][0].columns) or 'Phi' in list(experiment_dict['experimental_data'][0].columns):
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']
                temp_uncertainties = list(temp_uncertainties)
            return temp_uncertainties
        def flame_speed_press_uncertainties(experiment_dict):
            if 'Relative_Uncertainty' in list(experiment_dict['experimental_data'][0].columns) and 'Pressure' in list(experiment_dict['experimental_data'][0].columns):
                press_uncertainties=experiment_dict['experimental_data'][0]['Relative_Uncertainty'].values
                press_uncertainties = list(press_uncertainties)
            elif 'Pressure' in list(experiment_dict['experimental_data'][0].columns):
                press_uncertainties=experiment_dict['uncertainty']['pressure_relative_uncertainty']*np.ones(np.shape(experiment_dict['experimental_data'][0]['Pressure'].values))
                press_uncertainties = list(temp_uncertainties) 
            elif 'Temperature' in list(experiment_dict['experimental_data'][0].columns) or 'Phi' in list(experiment_dict['experimental_data'][0].columns):
                press_uncertainties=experiment_dict['uncertainty']['pressure_relative_uncertainty']
                press_uncertainties = list(temp_uncertainties)
            return press_uncertainties            

        def igdelay_temp_uncertainties(experiment_dict):

            if 'Relative_Uncertainty' in list(experiment_dict['experimental_data'][0].columns) and 'pressure' in list(experiment_dict['experimental_data'][0].columns) and len(experiment_dict['simulation'].temperatures) == len(experiment_dict['simulation'].pressures) and len(experiment_dict['simulation'].temperatures)>1 and len(experiment_dict['simulation'].pressures)>1:
                temp_uncertainties=experiment_dict['experimental_data'][0]['Relative_Uncertainty'].values
                temp_uncertainties = list(temp_uncertainties)*len(experiment_dict['simulation'].temperatures) 

            elif 'Relative_Uncertainty' in list(experiment_dict['experimental_data'][0].columns) and 'temperature' in list(experiment_dict['experimental_data'][0].columns):
                temp_uncertainties=experiment_dict['experimental_data'][0]['Relative_Uncertainty'].values
                temp_uncertainties = list(temp_uncertainties)

            elif 'temperature' in list(experiment_dict['experimental_data'][0].columns):
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']*np.ones(np.shape(experiment_dict['experimental_data'][0]['temperature'].values))
                temp_uncertainties = list(temp_uncertainties) 
                #stub this is where we are editing
            elif 'pressure' in list(experiment_dict['experimental_data'][0].columns) and len(experiment_dict['simulation'].temperatures) == len(experiment_dict['simulation'].pressures) and len(experiment_dict['simulation'].temperatures)>1 and len(experiment_dict['simulation'].pressures)>1 :
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']*np.ones(np.shape(experiment_dict['experimental_data'][0]['pressure'].values))
                temp_uncertainties = list(temp_uncertainties)* len(experiment_dict['simulation'].temperatures)
                
            elif 'pressure' in list(experiment_dict['experimental_data'][0].columns) and len(experiment_dict['simulation'].temperatures) != len(experiment_dict['simulation'].pressures):
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']
                temp_uncertainties = list(temp_uncertainties) 

            elif len(experiment_dict['conditions_to_run'])>1 and len(experiment_dict['simulation'].temperatures)>1 and len(experiment_dict['simulation'].pressures)>1 and len(experiment_dict['simulation'].temperatures) == len(experiment_dict['simulation'].pressures):
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']
                temp_uncertainties = list(temp_uncertainties)  * len(experiment_dict['simulation'].temperatures)    


            elif len(experiment_dict['conditions_to_run'])>1:
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']
                temp_uncertainties = list(temp_uncertainties)
            return temp_uncertainties
        def igdelay_press_uncertainties(experiment_dict):
            if 'Relative_Uncertainty' in list(experiment_dict['experimental_data'][0].columns) and 'temperature' in list(experiment_dict['experimental_data'][0].columns) and len(experiment_dict['simulation'].temperatures) == len(experiment_dict['simulation'].pressures) and len(experiment_dict['simulation'].temperatures)>1 and len(experiment_dict['simulation'].pressures)>1:
                press_uncertainties=experiment_dict['experimental_data'][0]['Relative_Uncertainty'].values
                press_uncertainties = list(press_uncertainties)*len(experiment_dict['simulation'].temperatures) 

            elif 'Relative_Uncertainty' in list(experiment_dict['experimental_data'][0].columns) and 'pressure' in list(experiment_dict['experimental_data'][0].columns):
                press_uncertainties=experiment_dict['experimental_data'][0]['Relative_Uncertainty'].values
                press_uncertainties = list(press_uncertainties)


            elif 'pressure' in list(experiment_dict['experimental_data'][0].columns):
                press_uncertainties=experiment_dict['uncertainty']['pressure_relative_uncertainty']*np.ones(np.shape(experiment_dict['experimental_data'][0]['pressure'].values))
                press_uncertainties = list(press_uncertainties) 
            elif 'temperature' in list(experiment_dict['experimental_data'][0].columns) and len(experiment_dict['simulation'].temperatures) == len(experiment_dict['simulation'].pressures) and len(experiment_dict['simulation'].temperatures)>1 and len(experiment_dict['simulation'].pressures)>1:
                press_uncertainties=experiment_dict['uncertainty']['pressure_relative_uncertainty']
                press_uncertainties = list(press_uncertainties) * len(experiment_dict['simulation'].temperatures)               
             #stub this is where editing is happening                   
            elif 'temperature' in list(experiment_dict['experimental_data'][0].columns) and len(experiment_dict['simulation'].temperatures) != len(experiment_dict['simulation'].pressures):
                press_uncertainties=experiment_dict['uncertainty']['pressure_relative_uncertainty']
                press_uncertainties = list(press_uncertainties)
            elif len(experiment_dict['conditions_to_run'])>1 and len(experiment_dict['simulation'].temperatures)>1 and len(experiment_dict['simulation'].pressures)>1 and len(experiment_dict['simulation'].temperatures) == len(experiment_dict['simulation'].pressures):
                press_uncertainties=experiment_dict['uncertainty']['pressure_relative_uncertainty']
                press_uncertainties = list(press_uncertainties)* len(experiment_dict['simulation'].temperatures)


            elif len(experiment_dict['conditions_to_run'])>1:
                press_uncertainties=experiment_dict['uncertainty']['pressure_relative_uncertainty']
                press_uncertainties = list(press_uncertainties)
            return press_uncertainties     
        
        def rcm_temp_uncertainties(experiment_dict):


            if 'Relative_Uncertainty' in list(experiment_dict['experimental_data'][0].columns) and 'temperature' in list(experiment_dict['experimental_data'][0].columns):
                temp_uncertainties=experiment_dict['experimental_data'][0]['Relative_Uncertainty'].values
                temp_uncertainties = list(temp_uncertainties)                
            elif 'temperature' in list(experiment_dict['experimental_data'][0].columns) and len(experiment_dict['simulation'].fullParsedYamlFile['temperatures'])==len(experiment_dict['simulation'].fullParsedYamlFile['pressures']):
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']*np.ones(np.shape(experiment_dict['experimental_data'][0]['temperature'].values))
                temp_uncertainties = list(temp_uncertainties)               
            elif 'pressure' in list(experiment_dict['experimental_data'][0].columns):
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']
                temp_uncertainties = list(temp_uncertainties)
            elif len(experiment_dict['conditions_to_run'])>1:
                temp_uncertainties=experiment_dict['uncertainty']['temperature_relative_uncertainty']
                temp_uncertainties = list(temp_uncertainties)
            return temp_uncertainties            
            
        def rcm_press_uncertainties(experiment_dict):
            if 'Relative_Uncertainty' in list(experiment_dict['experimental_data'][0].columns) and 'pressure' in list(experiment_dict['experimental_data'][0].columns):
                press_uncertainties=experiment_dict['experimental_data'][0]['Relative_Uncertainty'].values
                press_uncertainties = list(press_uncertainties)
            elif 'pressure' in list(experiment_dict['experimental_data'][0].columns) and len(experiment_dict['simulation'].fullParsedYamlFile['temperatures'])==len(experiment_dict['simulation'].fullParsedYamlFile['pressures']):
                press_uncertainties=experiment_dict['uncertainty']['pressure_relative_uncertainty']*np.ones(np.shape(experiment_dict['experimental_data'][0]['pressure'].values))
                press_uncertainties = list(press_uncertainties) 
            elif 'temperature' in list(experiment_dict['experimental_data'][0].columns) and len(experiment_dict['simulation'].fullParsedYamlFile['temperatures'])==len(experiment_dict['simulation'].fullParsedYamlFile['pressures']):
                press_uncertainties=experiment_dict['uncertainty']['pressure_relative_uncertainty']*np.ones(np.shape(experiment_dict['experimental_data'][0]['temperature'].values))
                press_uncertainties = list(press_uncertainties)
            elif len(experiment_dict['conditions_to_run'])>1:
                press_uncertainties=experiment_dict['uncertainty']['pressure_relative_uncertainty']
                press_uncertainties = list(press_uncertainties)
            return press_uncertainties                 
            
        
        
        
        #need to append to sigma
        def uncertainty_calc(relative_uncertainty,absolute_uncertainty,data,experimental_data):
            absolute_uncertainty=float(absolute_uncertainty)
                       
            length_of_data = data.shape[0]
            if 'Relative_Uncertainty' in list(experimental_data.columns):  
                
                x_dependent_uncertainty = experimental_data['Relative_Uncertainty'].values
                
                relative_uncertainty_array = copy.deepcopy(x_dependent_uncertainty)
                relative_uncertainty_array = relative_uncertainty_array.reshape((relative_uncertainty_array.shape[0],1))


                
            elif 'Relative_Uncertainty' not in list(experimental_data.columns):
                
                relative_uncertainty_array = np.full((length_of_data,1),relative_uncertainty)
                relative_uncertainty_array = relative_uncertainty_array.reshape((relative_uncertainty_array.shape[0],1))

            if 'Absolute_Uncertainty' in list(experimental_data.columns):
                x_dependent_a_uncertainty = experimental_data['Absolute_Uncertainty'].values
                
                
                absolute_uncertainty_array = copy.deepcopy(x_dependent_a_uncertainty)
                #Fix this to deal with 0 data.
                absolute_uncertainty_array = np.divide(absolute_uncertainty_array,data)
                absolute_uncertainty_array = absolute_uncertainty_array.reshape((absolute_uncertainty_array.shape[0],1))
                
            elif 'Absolute_Uncertainty' not in list(experimental_data.columns):
                
                absolute_uncertainty_array = np.divide(absolute_uncertainty,data)
                absolute_uncertainty_array = absolute_uncertainty_array.reshape((absolute_uncertainty_array.shape[0],1))
                
                
            total_uncertainty = np.sqrt(np.square(relative_uncertainty_array) + np.square(absolute_uncertainty_array))
            un_weighted_uncertainty = copy.deepcopy(total_uncertainty)                
                
            if 'W' not in list(experimental_data.columns):         
                weighting_factor = (1/length_of_data**.5)
                
                total_uncertainty = np.divide(total_uncertainty,weighting_factor)
                
                total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],1))


            
            elif 'W' in list(experimental_data.columns):
                weighting_factor = experimental_data['W'].values
                weighting_factor = weighting_factor.reshape((weighting_factor.shape[0],1))

                total_uncertainty = np.divide(total_uncertainty,weighting_factor)
                #total_uncertainty = total_uncertainty/weighting_factor

                
                
                
            return total_uncertainty,un_weighted_uncertainty
        #tab, start working here tomorrow with how we want to read in csv file     
        for i,exp_dic in enumerate(exp_dict_list):
            counter = 0
            #print(exp_dic)
            for j,observable in enumerate(exp_dic['mole_fraction_observables']+
                                           exp_dic['concentration_observables']+
                                           exp_dic['flame_speed_observables']+
                                           exp_dic['ignition_delay_observables']):

                if observable == None:
                    pass
                else:
                    if observable in exp_dic['mole_fraction_observables']:
                        ## add ppm statment here ? check if it exists? and add concentration statment below just for parcing 
                        
                        total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp_dic['uncertainty']['mole_fraction_relative_uncertainty'][counter],
                            exp_dic['uncertainty']['mole_fraction_absolute_uncertainty'][counter],
                            exp_dic['experimental_data'][counter][observable].values,exp_dic['experimental_data'][counter])
                        total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],1))
                        un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0], 1))
                    elif observable in exp_dic['concentration_observables'] and '_ppm' in exp_dic['experimental_data'][counter].columns[1]:
                        total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp_dic['uncertainty']['concentration_relative_uncertainty'][counter],
                             exp_dic['uncertainty']['concentration_absolute_uncertainty'][counter],
                             exp_dic['experimental_data'][counter][observable+'_ppm'].values,exp_dic['experimental_data'][counter])
                       
                        total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0], 1))
                        un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0], 1))
                    
                    elif observable in exp_dic['concentration_observables'] and '_mol/cm^3' in exp_dic['experimental_data'][counter].columns[1]:
                         
                        total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp_dic['uncertainty']['concentration_relative_uncertainty'][counter],
                              exp_dic['uncertainty']['concentration_absolute_uncertainty'][counter],
                              exp_dic['experimental_data'][counter][observable+'_mol/cm^3'].values,exp_dic['experimental_data'][counter])
                        total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],1))
                        un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0], 1))               

                    elif observable in exp_dic['flame_speed_observables'] and '_cm/s' in exp_dic['experimental_data'][counter].columns[1]:
                        total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp_dic['uncertainty']['flame_speed_relative_uncertainty'][counter], 
                                                                                     exp_dic['uncertainty']['flame_speed_absolute_uncertainty'][counter], 
                                                                                     exp_dic['experimental_data'][counter][observable+'_cm/s'].values,exp_dic['experimental_data'][counter])
                       
                        total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],1))
                        un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0], 1))        
                    elif observable in exp_dic['ignition_delay_observables'] and '_s'in exp_dic['experimental_data'][counter].columns[1]:
                        
                        total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp_dic['uncertainty']['ignition_delay_relative_uncertainty'][counter], 
                                                                                     exp_dic['uncertainty']['ignition_delay_absolute_uncertainty'][counter], 
                                                                                     exp_dic['experimental_data'][counter][observable+'_s'].values,exp_dic['experimental_data'][counter])
                        total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],1))

                        un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0], 1))               

                    else: 
                        raise Exception('We Do Not Have This Unit Installed, Please Use Mole Fraction, ppm, mol/cm^3 or cm/s')               

                    
                    
                    
                    
                    
                    
                    Z.append(total_uncertainty)
                    sigma.append(un_weighted_uncertainty)
                    tempList = [observable+'_'+'experiment'+str(i)]*np.shape(total_uncertainty)[0]
                    Z_data_Frame.extend(tempList)
                    #print(Z_data_Frame)
                    counter+=1
            if 'absorbance_observables' in list(exp_dic.keys()):
                wavelengths = parsed_yaml_file_list[i]['absorbanceCsvWavelengths']
                    
                for k,wl in enumerate(wavelengths):
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp_dic['uncertainty']['absorbance_relative_uncertainty'][k],
                                                             exp_dic['uncertainty']['absorbance_absolute_uncertainty'][k],
                                                             exp_dic['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values,exp_dic['absorbance_experimental_data'][k])
                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0], 1))
                    
                    un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0], 1))
                    
                    tempList = [str(wl)+'_'+'experiment'+'_'+str(i)]*np.shape(total_uncertainty)[0]
                    Z_data_Frame.extend(tempList)                   
                    Z.append(total_uncertainty)
                    sigma.append(un_weighted_uncertainty)
                        
        Z = np.vstack((Z))
        
        sigma = np.vstack((sigma))
        
        #Here we are adding A,n,and Ea uncertainty
        #we go do not through an additional step to make sure that the A,N and Ea 
        #values are paired with the correct reactions as in the old code,
        #because we wrote a function to make the excel sheet which will arrange things in the correct order 
        #We also need to decide if we want to put this in as ln values or not in the spreadsheet 
        active_parameters = []
        reaction_uncertainty = pd.read_csv(reaction_uncertainty)
        
        #Flatten master equation reaction list 
        flatten = lambda *n: (e for a in n
            for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))        
        
        flattened_master_equation_reaction_list = list(flatten(master_equation_reaction_list))  
        
        
        if master_equation_flag:
            for reaction in flattened_master_equation_reaction_list:
                index = reaction_uncertainty.loc[reaction_uncertainty['Reaction'] == reaction].index[0]
                reaction_uncertainty = reaction_uncertainty.drop([index])
                
                
                
                

        #tab fix this correctly, this unit needs to be fixed when we make a decision what the spreadsheet looks like

        uncertainty_As = reaction_uncertainty['Uncertainty A (unit)'].values 
        uncertainty_As = uncertainty_As.reshape((uncertainty_As.shape[0],
                                                  1))
        
        
        
        #uncertainty_As = np.log(uncertainty_As)
        Z = np.vstack((Z,uncertainty_As))
        sigma = np.vstack((sigma,uncertainty_As))
        for variable in range(uncertainty_As.shape[0]):
            Z_data_Frame.append('A'+'_'+str(variable))
            active_parameters.append('A'+'_'+str(variable))
        
       
        
        uncertainty_ns = reaction_uncertainty['Uncertainty N (unit)'].values
        uncertainty_ns = uncertainty_ns.reshape((uncertainty_ns.shape[0],
                                                  1))
        Z = np.vstack((Z,uncertainty_ns))
        sigma = np.vstack((sigma,uncertainty_ns))
        for variable in range(uncertainty_ns.shape[0]):
            Z_data_Frame.append('n'+'_'+str(variable))
            active_parameters.append('n'+'_'+str(variable))
        
        
        uncertainty_Eas = reaction_uncertainty['Uncertainty Ea (unit)'].values
        uncertainty_Eas = uncertainty_Eas.reshape((uncertainty_Eas.shape[0],
                                                  1))
     
        
        Z = np.vstack((Z,uncertainty_Eas))
        
        sigma = np.vstack((sigma,uncertainty_Eas))
        for variable in range(uncertainty_Eas.shape[0]):
            Z_data_Frame.append('Ea'+'_'+str(variable))
            active_parameters.append('Ea'+'_'+str(variable))
        

        
        
        
        if master_equation_flag == True:
            master_equation_uncertainty = []
            for i,reaction in enumerate(master_equation_reaction_list):
                if type(reaction)==str:
                    master_equation_uncertainty.append(list(master_equation_uncertainty_df[reaction].dropna().values))

                elif type(reaction)==tuple:
                    column_headers = master_equation_uncertainty_df.columns.to_list()
                    for sub_reaction in reaction:
                        if sub_reaction in column_headers:
                            master_equation_uncertainty.append(list(master_equation_uncertainty_df[sub_reaction].dropna().values))        
        
        
        
        
        # if master_equation_flag ==True:
        #     master_equation_uncertainty = []
        #     for col in master_equation_uncertainty_df:
        #         master_equation_uncertainty.append(list(master_equation_uncertainty_df[col].dropna().values))
                
            
        if master_equation_flag == True:
            
            for i,reaction in enumerate(master_equation_reaction_list):
                if type(reaction)==str:
                    for j,paramter in enumerate(master_equation_uncertainty_df[reaction].dropna()):
                        Z_data_Frame.append(str(reaction)+'_'+'P'+'_'+str(j))
                        active_parameters.append(master_equation_reaction_list[i]+'_P_'+str(j))
                
                elif type(reaction)==tuple:
                    column_headers = master_equation_uncertainty_df.columns.to_list()
                    for sub_reaction in reaction:
                        if sub_reaction in column_headers:
                            for j,paramter in enumerate(master_equation_uncertainty_df[sub_reaction].dropna()):
                                Z_data_Frame.append(str(reaction)+'_'+'P'+'_'+str(j))    
                                active_parameters.append(str(master_equation_reaction_list[i])+'_P_'+str(j))
            
            
            
            # for i,reaction in enumerate(master_equation_uncertainty):
            #     for j,uncer in enumerate(reaction):
            #         Z_data_Frame.append('R'+'_'+str(i)+'_'+'P'+str(j))
            #         #This might not look right in the data frame but we can try
            #         #stub
            #         active_parameters.append(master_equation_reaction_list[i]+'_P_'+str(j))
                    
           
            ##check this 
            
            master_equation_uncertainty = [item for sublist in master_equation_uncertainty for item in sublist]
            master_equation_uncertainty = np.array(master_equation_uncertainty)
            master_equation_uncertainty = master_equation_uncertainty.reshape((master_equation_uncertainty.shape[0],
                                                  1))
            Z = np.vstack((Z,master_equation_uncertainty))
            sigma = np.vstack((sigma,master_equation_uncertainty))
            
            
        #This is going to have to be simulation specific 
        if exp_dict_list[0]['simulation'].physicalSens ==1:
           for i, exp_dic in enumerate(exp_dict_list):
               if re.match('[Bb]atch[ _-][Rr]eactor',exp_dict_list[i]['simulation_type']) and re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']):
               #for i,exp_dic in enumerate(exp_dict_list):
                    experiment_physical_uncertainty = []
                    #Temperature Uncertainty 
                    experiment_physical_uncertainty.append(exp_dic['uncertainty']['temperature_relative_uncertainty'])
                    Z_data_Frame.append('T'+'_'+'experiment'+'_'+str(i))
                    active_parameters.append('T'+'_'+'experiment'+'_'+str(i))
                    #Pressure Uncertainty
                    experiment_physical_uncertainty.append(exp_dic['uncertainty']['pressure_relative_uncertainty'])
                    Z_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))
                    active_parameters.append('P'+'_'+'experiment'+'_'+str(i))
                    #Species Uncertainty
                    species_uncertainties = exp_dic['uncertainty']['species_relative_uncertainty']['dictionary_of_values']
                    species_to_loop =  exp_dic['uncertainty']['species_relative_uncertainty']['species']
                    dilluant = ['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']
                    

                    for specie in species_to_loop:
                        if specie in dilluant:
                            continue
                        experiment_physical_uncertainty.append(species_uncertainties[specie])
                        Z_data_Frame.append('X'+'_'+str(specie)+'_'+'experiment'+'_'+str(i))
                        active_parameters.append('X'+'_'+str(specie)+'_'+'experiment'+'_'+str(i))
                    
                    
                    experiment_physical_uncertainty.append(exp_dic['uncertainty']['time_shift_absolute_uncertainty'])
                    Z_data_Frame.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                    active_parameters.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                    
                    experiment_physical_uncertainty = np.array(experiment_physical_uncertainty)
                    experiment_physical_uncertainty =  experiment_physical_uncertainty.reshape((experiment_physical_uncertainty.shape[0],
                                                      1))
                    Z = np.vstack((Z,experiment_physical_uncertainty))
                    sigma = np.vstack((sigma,experiment_physical_uncertainty))
                    
               elif re.match('[Jj][Ss][Rr]',exp_dict_list[i]['simulation_type']):
                   #ASK MARK WHAT TO ADD HERE
                   
                #for i,exp_dic in enumerate(exp_dict_list):
                    experiment_physical_uncertainty = []
                    #Temperature Uncertainty 
                    temp_uncertainties=jsr_temp_uncertainties(exp_dic)
                    experiment_physical_uncertainty=experiment_physical_uncertainty+temp_uncertainties
                    
                    #experiment_physical_uncertainty.append(exp_dic['uncertainty']['temperature_relative_uncertainty'])
                    Z_data_Frame=Z_data_Frame+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                    
                    active_parameters=active_parameters+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                    #Pressure Uncertainty
                    experiment_physical_uncertainty.append(exp_dic['uncertainty']['pressure_relative_uncertainty'])
                    Z_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))
                    active_parameters.append('P'+'_'+'experiment'+'_'+str(i))
                    #Species Uncertainty
                    species_uncertainties = exp_dic['uncertainty']['species_relative_uncertainty']['dictionary_of_values']
                    species_to_loop =  exp_dic['uncertainty']['species_relative_uncertainty']['species']
                    dilluant = ['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']
                    
                    for specie in species_to_loop:
                        if specie in dilluant:
                            continue
                        experiment_physical_uncertainty.append(species_uncertainties[specie])
                        Z_data_Frame.append('X'+'_'+str(specie)+'_'+'experiment'+'_'+str(i))
                        active_parameters.append('X'+'_'+str(specie)+'_'+'experiment'+'_'+str(i))
                    
                    experiment_physical_uncertainty.append(exp_dic['uncertainty']['restime_relative_uncertainty'])
                    Z_data_Frame.append('R_experiment_'+str(i))
                   
                    active_parameters.append('R_experiment_'+str(i))
                    experiment_physical_uncertainty = np.array(experiment_physical_uncertainty)
                    experiment_physical_uncertainty =  experiment_physical_uncertainty.reshape((experiment_physical_uncertainty.shape[0],1))
                    Z = np.vstack((Z,experiment_physical_uncertainty))
                    sigma = np.vstack((sigma,experiment_physical_uncertainty))
                    #print(Z_data_Frame)
               elif re.match('[Ff]lame[- ][Ss]peed',exp_dict_list[i]['simulation_type']) and re.match('[Oo][Nn][Ee]|[1][ -][dD][ -][Ff]lame',exp_dict_list[i][['experiment_type']]):
                #for i,exp_dic in enumerate(exp_dict_list):
                    experiment_physical_uncertainty = []
                    #Temperature Uncertainty 
                    temp_uncertainties=flame_speed_temp_uncertainties(exp_dic)
                    experiment_physical_uncertainty=experiment_physical_uncertainty+temp_uncertainties
                    Z_data_Frame=Z_data_Frame+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                    active_parameters=active_parameters+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                    #Pressure Uncertainty
                    
                    press_uncertainties = flame_speed_press_uncertainties(exp_dic)
                    Z_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))*len(press_uncertainties)
                    active_parameters.append('P'+'_'+'experiment'+'_'+str(i))*len(press_uncertainties)
                    #Species Uncertainty
                    conditions = exp_dic['conditions']
                    species_uncertainties = exp_dic['uncertainty']['species_relative_uncertainty']['dictionary_of_values']
                    species_to_loop =  exp_dic['uncertainty']['species_relative_uncertainty']['species']
                    
                    list_with_most_species_in_them = []
                    for specie in species_to_loop:
                        list_with_most_species_in_them.append(len(conditions[specie]))
                    max_species = max(list_with_most_species_in_them)
                        
                    if 'Diluant' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluant' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                        diluant = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluant']
                             
                    for nmbr_of_species_sets in range(max_species):
                         for specie in species_to_loop:
                             if specie in dilluant:
                                 continue
                             experiment_physical_uncertainty.append(species_uncertainties[specie])
                             Z_data_Frame.append('X'+'_'+str(specie)+'_'+'experiment'+'_'+str(i))
                             active_parameters.append('X'+'_'+str(specie)+'_'+'experiment'+'_'+str(i))

                    experiment_physical_uncertainty = np.array(experiment_physical_uncertainty)
                    experiment_physical_uncertainty =  experiment_physical_uncertainty.reshape((experiment_physical_uncertainty.shape[0],
                                                       1))
                    Z = np.vstack((Z,experiment_physical_uncertainty))
                    sigma = np.vstack((sigma,experiment_physical_uncertainty))                    

               elif re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp_dict_list[i]['simulation_type']):
                   #ASK MARK WHAT TO ADD HERE
                   
                #for i,exp_dic in enumerate(exp_dict_list):
                    experiment_physical_uncertainty = []
                    #Temperature Uncertainty 
                    temp_uncertainties=flow_reactor_temp_uncertainties(exp_dic)
                    experiment_physical_uncertainty=experiment_physical_uncertainty+temp_uncertainties
                    
                    #experiment_physical_uncertainty.append(exp_dic['uncertainty']['temperature_relative_uncertainty'])
                    Z_data_Frame=Z_data_Frame+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                    
                    active_parameters=active_parameters+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                    #Pressure Uncertainty
                    experiment_physical_uncertainty.append(exp_dic['uncertainty']['pressure_relative_uncertainty'])
                    Z_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))
                    active_parameters.append('P'+'_'+'experiment'+'_'+str(i))
                    #Species Uncertainty
                    species_uncertainties = exp_dic['uncertainty']['species_relative_uncertainty']['dictionary_of_values']
                    species_to_loop =  exp_dic['uncertainty']['species_relative_uncertainty']['species']
                    dilluant = ['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']
                    
                    for specie in species_to_loop:
                        if specie in dilluant:
                            continue
                        experiment_physical_uncertainty.append(species_uncertainties[specie])
                        Z_data_Frame.append('X'+'_'+str(specie)+'_'+'experiment'+'_'+str(i))
                        active_parameters.append('X'+'_'+str(specie)+'_'+'experiment'+'_'+str(i))
                    
                    
                    time_shift_uncertainties = flow_reactor_time_shift_uncertainties(parsed_yaml_file_list[i],exp_dic)
                    experiment_physical_uncertainty=experiment_physical_uncertainty+time_shift_uncertainties
                    Z_data_Frame=Z_data_Frame+['Time_Shift'+'_'+'experiment'+'_'+str(i)]*len(time_shift_uncertainties)                    
                    active_parameters=active_parameters+['Time_Shift'+'_'+'experiment'+'_'+str(i)]*len(time_shift_uncertainties)



                    experiment_physical_uncertainty = np.array(experiment_physical_uncertainty)
                    experiment_physical_uncertainty =  experiment_physical_uncertainty.reshape((experiment_physical_uncertainty.shape[0],1))
                    Z = np.vstack((Z,experiment_physical_uncertainty))
                    sigma = np.vstack((sigma,experiment_physical_uncertainty))                   
               
               
               elif re.match('[Bb]atch[- ][Rr]eactor',exp_dict_list[i]['simulation_type']) and re.match('[Ii]gnition[- ][Dd]elay',exp_dict_list[i]['experiment_type']):
                #for i,exp_dic in enumerate(exp_dict_list):

                    if len(exp_dic['simulation'].temperatures) == len(exp_dic['simulation'].pressures) and len(exp_dic['simulation'].temperatures) >1 and len(exp_dic['simulation'].pressures) >1:
                       # print('inside z matrix')
                        experiment_physical_uncertainty = []
                        #Temperature Uncertainty 
                        temp_uncertainties=igdelay_temp_uncertainties(exp_dic)
                        experiment_physical_uncertainty=experiment_physical_uncertainty+temp_uncertainties
                        
                        for index in range(len(temp_uncertainties)):
                            Z_data_Frame.append('T'+str(index+1)+'_'+'experiment'+'_'+str(i))
                            active_parameters.append('T'+str(index+1)+'_'+'experiment'+'_'+str(i))
                        #Z_data_Frame=Z_data_Frame+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                            
                        #active_parameters=active_parameters+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                        #Pressure Uncertainty
                        
                        press_uncertainties = igdelay_press_uncertainties(exp_dic)
                        for index in range(len(press_uncertainties)):
                            Z_data_Frame.append('P'+str(index+1)+'_'+'experiment'+'_'+str(i))
                            active_parameters.append('P'+str(index+1)+'_'+'experiment'+'_'+str(i))
                        
                        #Z_data_Frame=Z_data_Frame+['P'+'_'+'experiment'+'_'+str(i)]*len(press_uncertainties)
                        #active_parameters=active_parameters+['P'+'_'+'experiment'+'_'+str(i)]*len(press_uncertainties)
                        experiment_physical_uncertainty=experiment_physical_uncertainty+press_uncertainties
                        #print(len(press_uncertainties))
                        #Species Uncertainty
                        conditions = exp_dic['conditions_dict_list']
                        species_uncertainties = exp_dic['uncertainty']['species_relative_uncertainty']['dictionary_of_values']
                        species_to_loop =  list(exp_dic['conditions_dict_list'].keys())
                        
                        list_with_most_species_in_them = []
                        for specie in species_to_loop:
                            list_with_most_species_in_them.append(len(conditions[specie]))
                        max_species = max(list_with_most_species_in_them)
                        diluent=[]   
                        if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                        
                        singular_species=[]
                        for species in list(exp_dic['simulation'].fullParsedYamlFile['conditions'].keys()):
                            
                            if len(exp_dic['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                                singular_species.append(species)
                        
                        for x,species in enumerate(exp_dic['simulation'].fullParsedYamlFile['speciesNames']):
                            if species in singular_species and species not in diluent:
                                Z_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                                experiment_physical_uncertainty.append(species_uncertainties[specie])
                                active_parameters.append('X'+str(x+1)+'_'+species+'_'+'experiment'+'_'+str(i))
                            elif species not in singular_species and species not in diluent:
                                for j in range(len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])):
                                    Z_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                                    experiment_physical_uncertainty.append(species_uncertainties[specie])
                                    active_parameters.append('X'+str(x+1)+'_'+species+'_'+'experiment'+'_'+str(i))                    
                           
                        
    
                        experiment_physical_uncertainty.append(exp_dic['uncertainty']['time_shift_absolute_uncertainty'])
                        Z_data_Frame.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                        active_parameters.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                        experiment_physical_uncertainty = np.array(experiment_physical_uncertainty)
                        experiment_physical_uncertainty =  experiment_physical_uncertainty.reshape((experiment_physical_uncertainty.shape[0],
                                                           1))
                        Z = np.vstack((Z,experiment_physical_uncertainty))
                        sigma = np.vstack((sigma,experiment_physical_uncertainty))                       
                    else:
                        experiment_physical_uncertainty = []
                        #Temperature Uncertainty 
                        temp_uncertainties=igdelay_temp_uncertainties(exp_dic)
                        experiment_physical_uncertainty=experiment_physical_uncertainty+temp_uncertainties
                        
                        for index in range(len(temp_uncertainties)):
                            Z_data_Frame.append('T'+str(index+1)+'_'+'experiment'+'_'+str(i))
                            active_parameters.append('T'+str(index+1)+'_'+'experiment'+'_'+str(i))
                        #Z_data_Frame=Z_data_Frame+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                            
                        #active_parameters=active_parameters+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                        #Pressure Uncertainty
                        
                        press_uncertainties = igdelay_press_uncertainties(exp_dic)
                        for index in range(len(press_uncertainties)):
                            Z_data_Frame.append('P'+str(index+1)+'_'+'experiment'+'_'+str(i))
                            active_parameters.append('P'+str(index+1)+'_'+'experiment'+'_'+str(i))
                        
                        #Z_data_Frame=Z_data_Frame+['P'+'_'+'experiment'+'_'+str(i)]*len(press_uncertainties)
                        #active_parameters=active_parameters+['P'+'_'+'experiment'+'_'+str(i)]*len(press_uncertainties)
                        experiment_physical_uncertainty=experiment_physical_uncertainty+press_uncertainties
                        #print(len(press_uncertainties))
                        #Species Uncertainty
                        conditions = exp_dic['conditions_dict_list']
                        species_uncertainties = exp_dic['uncertainty']['species_relative_uncertainty']['dictionary_of_values']
                        species_to_loop =  list(exp_dic['conditions_dict_list'].keys())
                        
                        list_with_most_species_in_them = []
                        for specie in species_to_loop:
                            list_with_most_species_in_them.append(len(conditions[specie]))
                        max_species = max(list_with_most_species_in_them)
                        
                            
                            
                            
                            
                        diluent=[]   
                        if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                        
                        singular_species=[]
                        for species in list(exp_dic['simulation'].fullParsedYamlFile['conditions'].keys()):
                            
                            if len(exp_dic['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                                singular_species.append(species)
                        
                        for x,species in enumerate(exp_dic['simulation'].fullParsedYamlFile['speciesNames']):
                            if species in singular_species and species not in diluent:
                                Z_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                                experiment_physical_uncertainty.append(species_uncertainties[specie])
                                active_parameters.append('X'+str(x+1)+'_'+species+'_'+'experiment'+'_'+str(i))
                            elif species not in singular_species and species not in diluent:
                                for j in range(len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])):
                                    Z_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                                    experiment_physical_uncertainty.append(species_uncertainties[specie])
                                    active_parameters.append('X'+str(x+1)+'_'+species+'_'+'experiment'+'_'+str(i))                    
                           
                        
    
                        experiment_physical_uncertainty.append(exp_dic['uncertainty']['time_shift_absolute_uncertainty'])
                        Z_data_Frame.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                        active_parameters.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                        experiment_physical_uncertainty = np.array(experiment_physical_uncertainty)
                        experiment_physical_uncertainty =  experiment_physical_uncertainty.reshape((experiment_physical_uncertainty.shape[0],
                                                           1))
                        Z = np.vstack((Z,experiment_physical_uncertainty))
                        sigma = np.vstack((sigma,experiment_physical_uncertainty))   
                    
               
               elif re.match('[Rr][Cc][Mm]',exp_dict_list[i]['simulation_type']) and re.match('[Ii]gnition[- ][Dd]elay',exp_dict_list[i]['experiment_type']):
               
                   #for i,exp_dic in enumerate(exp_dict_list):
                    
                    experiment_physical_uncertainty = []
                    #Temperature Uncertainty 
                    temp_uncertainties=rcm_temp_uncertainties(exp_dic)
                    experiment_physical_uncertainty=experiment_physical_uncertainty+temp_uncertainties
                    
                    for index in range(len(temp_uncertainties)):
                        Z_data_Frame.append('T'+str(index+1)+'_'+'experiment'+'_'+str(i))
                        active_parameters.append('T'+str(index+1)+'_'+'experiment'+'_'+str(i))
                    #Z_data_Frame=Z_data_Frame+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                        
                    #active_parameters=active_parameters+['T'+'_'+'experiment'+'_'+str(i)]*len(temp_uncertainties)
                    #Pressure Uncertainty
                    
                    press_uncertainties = rcm_press_uncertainties(exp_dic)
                    for index in range(len(press_uncertainties)):
                        Z_data_Frame.append('P'+str(index+1)+'_'+'experiment'+'_'+str(i))
                        active_parameters.append('P'+str(index+1)+'_'+'experiment'+'_'+str(i))
                    
                    #Z_data_Frame=Z_data_Frame+['P'+'_'+'experiment'+'_'+str(i)]*len(press_uncertainties)
                    #active_parameters=active_parameters+['P'+'_'+'experiment'+'_'+str(i)]*len(press_uncertainties)
                    experiment_physical_uncertainty=experiment_physical_uncertainty+press_uncertainties
                    #print(len(press_uncertainties))
                    #Species Uncertainty
                    conditions = exp_dic['conditions_dict_list']
                    species_uncertainties = exp_dic['uncertainty']['species_relative_uncertainty']['dictionary_of_values']
                    species_to_loop =  list(exp_dic['conditions_dict_list'].keys())
                    
                    list_with_most_species_in_them = []
                    for specie in species_to_loop:
                        list_with_most_species_in_them.append(len(conditions[specie]))
                    max_species = max(list_with_most_species_in_them)
                    
                        
                        
                        
                        
                    diluent=[]   
                    if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                        diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                    
                    singular_species=[]
                    for species in list(exp_dic['simulation'].fullParsedYamlFile['conditions'].keys()):
                        
                        if len(exp_dic['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                            singular_species.append(species)
                    
                    for x,species in enumerate(exp_dic['simulation'].fullParsedYamlFile['speciesNames']):
                        if species in singular_species and species not in diluent:
                            Z_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                            experiment_physical_uncertainty.append(species_uncertainties[specie])
                            active_parameters.append('X'+str(x+1)+'_'+species+'_'+'experiment'+'_'+str(i))
                        elif species not in singular_species and species not in diluent:
                            for j in range(len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])):
                                Z_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                                experiment_physical_uncertainty.append(species_uncertainties[specie])
                                active_parameters.append('X'+str(x+1)+'_'+species+'_'+'experiment'+'_'+str(i))                    
                       
                    

                    experiment_physical_uncertainty.append(exp_dic['uncertainty']['time_shift_absolute_uncertainty'])
                    Z_data_Frame.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                    active_parameters.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                    experiment_physical_uncertainty = np.array(experiment_physical_uncertainty)
                    experiment_physical_uncertainty =  experiment_physical_uncertainty.reshape((experiment_physical_uncertainty.shape[0],
                                                       1))
                    Z = np.vstack((Z,experiment_physical_uncertainty))
                    sigma = np.vstack((sigma,experiment_physical_uncertainty))                    
                    
                    
                    
                    
               #print(exp_dict_list[i]['simulation_type'],exp_dict_list[i]['experiment_type']) 
        #building dictionary to keep track of independtend coupled coefficients 
        count = 0
        coef_dict = {} 
        
        uncertainties_of_coefficents = []
        for i,exp_dic in enumerate(exp_dict_list):
            if 'perturbed_coef' not in exp_dic.keys():
                continue
            dictionary_of_coef_and_uncertainty = exp_dic['uncertainty']['coupled_coef_and_uncertainty']
            
            for x in dictionary_of_coef_and_uncertainty:
                if x not in coef_dict.keys():
                    coef_dict[x] = dictionary_of_coef_and_uncertainty[x]
                    
        for x in coef_dict:       
            for y in coef_dict[x]:
                if y[0]!=0:                            #this might cause a problem in the future
                        count+=1
                        uncertainties_of_coefficents.append(y)
                        Z_data_Frame.append('Sigma'+'_'+str(count))
                        active_parameters.append('Sigma'+'_'+str(count))
                        
        uncertainties_of_coefficents = np.array(uncertainties_of_coefficents)
        
       
        if uncertainties_of_coefficents.any() == True:
            uncertainties_of_coefficents =  uncertainties_of_coefficents.reshape((uncertainties_of_coefficents.shape[0],
                                                  1))
            Z = np.vstack((Z,uncertainties_of_coefficents))            
            sigma = np.vstack((sigma,uncertainties_of_coefficents))
        #return(Z,Z_data_Frame)
        #print('THIS IS Z',Z_data_Frame)
        Z_data_Frame = pd.DataFrame({'value': Z_data_Frame,'Uncertainty': Z.reshape((Z.shape[0],))})       
        self.z_matrix = Z
        self.sigma = sigma
        #print(Z.shape)

        return Z,Z_data_Frame,sigma,active_parameters


    
    def load_Y(self, exp_dict_list:list,parsed_yaml_file_list:list,
               loop_counter:int = 0,
               X:dict={},
               master_equation_reactions = [],
               master_equation_uncertainty_df = None,
               master_equation_flag = False):


        '''
        Builds the Y vector. 
        
        Arguments:
            exp_dic_list -- the dictionary that is built after a simulation
            that contains things like sensitivity coefficients
            parsed_yaml_file_list -- a list of dictonaries that contain the 
            information stored in the yaml files. 
        Keyword Arguments:
            loop_counter -- keeps track of the iteration number for the optimization (default 0)
             X -- a dict that holds the updates for optimization variables (default empty)
            master_equation_uncertainty_df -- a pandas dataframe that contains 
            the reactions being treated with theory paramters along with the 
            associated uncertainty values of those paramters (default None)
            master_equation_reaction_list -- a list of the reactions being treated
            with theory paramters (default [])
            master_equation_flag -- a boolean that indicates if reactions being
            represented by theory parameters are being used in the optimization (default False)
            
        '''
       
        
        def natural_log_difference(experiment,model):
            natural_log_diff = np.log(np.array(experiment)) - np.log(np.array(model))
            
            return natural_log_diff
        
        Y = []
        Y_data_Frame = []
        for i,exp_dic in enumerate(exp_dict_list):
            counter = 0
            for j,observable in enumerate((exp_dic['mole_fraction_observables']+
                                           exp_dic['concentration_observables'] + 
                                           exp_dic['flame_speed_observables']+  
                                           exp_dic['ignition_delay_observables'])):
                if observable == None:
                    pass
                else:
                    #if you need to add something with concentration add it here

                    if 'ppm' in exp_dic['experimental_data'][counter].columns.tolist()[1]:
                        if re.match('[Bb]atch[ _-][Rr]eactor',exp_dict_list[i]['simulation_type']):
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable+'_ppm'].values,
                                                                  (exp_dic['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6)
                        
                        
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0],1))
                            
                        if re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp_dict_list[i]['simulation_type']):
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable+'_ppm'].values,
                                                                  (exp_dic['simulation'].timeHistories[0][observable].dropna().values)*1e6)
                        
                        
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0],1))                            
                            
                        if re.match('[Jj][Ss][Rr]',exp_dict_list[i]['simulation_type']):
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable+'_ppm'].values,
                                                                  (exp_dic['simulation'].timeHistories[0][observable].dropna().values)*1e6)
                        
                        
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0],1))

                        
                    elif 'mol/cm^3' in exp_dic['experimental_data'][counter].columns.tolist()[1]:
                        if re.match('[Bb]atch[ _-][Rr]eactor',exp_dict_list[i]['simulation_type']) and re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']):
                            
                            concentration = np.true_divide(1,exp_dic['simulation'].pressureAndTemperatureToExperiment[counter]['temperature'].to_numpy())*exp_dic['simulation'].pressureAndTemperatureToExperiment[counter]['pressure'].to_numpy()
                           
                            concentration *= (1/(8.314e6))*exp_dic['simulation'].timeHistoryInterpToExperiment[observable].dropna().to_numpy()
                            
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable+'_mol/cm^3'].to_numpy(),concentration)
                            
                            
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0], 1))
                        if re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp_dict_list[i]['simulation_type']):
                                                        
                            concentration = np.true_divide(1,exp_dic['simulation'].timeHistories[0]['temperature'].to_numpy())*exp_dic['simulation'].timeHistories[0]['pressure'].to_numpy()
                           
                            concentration *= (1/(8.314e6))*exp_dic['simulation'].timeHistories[0][observable].dropna().to_numpy()
                            
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable+'_mol/cm^3'].to_numpy(),concentration)
                            
                            
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0], 1))

                        if re.match('[Jj][Ss][Rr]',exp_dict_list[i]['simulation_type']):
                            concentration = np.true_divide(1.0,exp_dic['simulation'].pressure*ct.one_atm)*np.array(exp_dic['simulation'].temperatures)
                           
                            concentration *= (1/(8.314e6))*exp_dic['simulation'].timeHistories[0][observable].dropna().to_numpy()
                            
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable+'_mol/cm^3'].to_numpy(),concentration)
                            
                            
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0], 1))
                    
                    elif 'cm/s'  in exp_dic['experimental_data'][counter].columns.tolist()[1]:
                        if re.match('[Ff]lame [Ss]peed',exp_dict_list[i]['simulation_type']) and re.match('[Oo][Nn][Ee]|[1][ -][dD][ -][Ff]lame',exp_dict_list[i]['experiment_type']):
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable+'_cm/s'].to_numpy(),
                                                                      exp_dic['simulation'].timeHistories[0][observable])
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0], 1))
                    
                    elif 's' in exp_dic['experimental_data'][counter].columns.tolist()[1]:
                        if re.match('[Ii]gnition[- ][Dd]elay',exp_dict_list[i]['experiment_type']):
                            #check these units would be in seconds of ms?
                            
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable+'_s'].to_numpy(),
                                                                      exp_dic['simulation'].timeHistories[0]['delay'])
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0], 1))
                    else:
                        if re.match('[Bb]atch[ _-][Rr]eactor',exp_dict_list[i]['simulation_type']) and re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']):
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable].values,
                                                                  exp_dic['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0], 1))
                        if re.match('[Jj][Ss][Rr]',exp_dict_list[i]['simulation_type']):
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable].values,
                                                                  exp_dic['simulation'].timeHistories[0][observable].values)
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0], 1))
                            
                        if re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp_dict_list[i]['simulation_type']):
                            natural_log_diff = natural_log_difference(exp_dic['experimental_data'][counter][observable].values,
                                                                  exp_dic['simulation'].timeHistories[0][observable].values)
                            natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0], 1))
                
                    tempList = [observable+'_'+'experiment'+str(i)]*np.shape(natural_log_diff)[0]
                    Y_data_Frame.extend(tempList)
                
                    Y.append(natural_log_diff)
                    counter+=1
            if 'absorbance_observables' in list(exp_dic.keys()):
                wavelengths = parsed_yaml_file_list[i]['absorbanceCsvWavelengths']
                
                for k,wl in enumerate(wavelengths):
                    natural_log_diff = natural_log_difference(exp_dic['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values,exp_dic['absorbance_model_data'][wl])
                    natural_log_diff =  natural_log_diff.reshape((natural_log_diff.shape[0],
                                                  1))
                    
                    tempList = [str(wl)+'_'+'experiment'+'_'+str(i)]*np.shape(natural_log_diff)[0]
                    Y_data_Frame.extend(tempList)
                    
                    
                    Y.append(natural_log_diff)
        
        Y = np.vstack((Y))
       
        
       
        #YdataFrame = pd.DataFrame({'value': YdataFrame,'ln_difference': Y})
        
        reactions_in_cti_file = exp_dict_list[0]['simulation'].processor.solution.reaction_equations()
       
            #assembling the target values portion of the Y matrix 
            #getting the size of the cti file from the first simulation because 
            #they all use the same cti file and it shouldn't matter 
            
            
            # add in a conditional statment for if there is master equation data
            #which is getting included in the simulation
            
            
            
        #Flatten master equation reaction list 
        flatten = lambda *n: (e for a in n
            for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))        
        
        flattened_master_equation_reaction_list = list(flatten(master_equation_reactions))              

        if master_equation_flag ==True:
            A_n_Ea_length = int((len(reactions_in_cti_file) - len(flattened_master_equation_reaction_list))*3)            
            number_of_molecular_parameters_list = []
            for col in master_equation_uncertainty_df:
                number_of_molecular_parameters_list.append(len(master_equation_uncertainty_df[col].dropna().values))
                
            number_of_molecular_parameters = sum(number_of_molecular_parameters_list) 
            #print('we do not have master equation installed yet')
            #subtract out the necessary target values and add the other ones in 
        else:
            A_n_Ea_length = len(reactions_in_cti_file)*3
            
            #addint the zeros to the Y array 
            
            #adding the strings to the dictionary 
            ## making a,n and Ea zero list 
        A_n_Ea_zeros = np.zeros((A_n_Ea_length,1))  
        
        if master_equation_flag ==True:
            molecular_paramter_zeros = np.zeros((number_of_molecular_parameters,1))
        
        
        for variable in range(A_n_Ea_length//3):
            Y_data_Frame.append('A'+'_'+str(variable))
        for variable in range(A_n_Ea_length//3):
            Y_data_Frame.append('n'+'_'+str(variable))
        for variable in range(A_n_Ea_length//3):
            Y_data_Frame.append('Ea'+'_'+str(variable))
            
        
        #make this the order of master equation list
        if master_equation_flag == True:
            for i,reaction in enumerate(master_equation_reactions):
                if type(reaction)==str:
                    for j,paramter in enumerate(master_equation_uncertainty_df[reaction].dropna()):
                        Y_data_Frame.append(str(reaction)+'_P'+'_'+str(j))
                elif type(reaction)==tuple:
                    column_headers = master_equation_uncertainty_df.columns.to_list()
                    
                    for sub_reaction in reaction:
                        if sub_reaction in column_headers:
                            
                            for j,paramter in enumerate(master_equation_uncertainty_df[sub_reaction].dropna()):
                                Y_data_Frame.append(str(reaction)+'_P'+'_'+str(j))
                        
            
        
        # if master_equation_flag == True:
        #     for i,value in enumerate(number_of_molecular_parameters_list):
        #         for j,parameter in enumerate(range(value)):
        #             Y_data_Frame.append('R'+'_'+str(i)+'P'+'_'+str(j))
               
            
        
        
        if loop_counter == 0:
            Y = np.vstack((Y,A_n_Ea_zeros))
            if master_equation_flag ==True:
                Y = np.vstack((Y,molecular_paramter_zeros))
        
             
        else:
            #print('we do not have loop counter installed yet')
            #need to check what we would need to do here 
            #should be tottal X ?
            
            #clean this part of the code up here
            temp_array = np.array(X['As_ns_Eas'])*-1
            temp_array = temp_array.reshape((temp_array.shape[0],
                                                      1))
            
            Y = np.vstack((Y, temp_array))
            #clean this part of the code up here
            #tab
            if master_equation_flag == True:
                temp_array = np.array(X['molecular_parameters'])*-1
                temp_array = temp_array.reshape((temp_array.shape[0],
                                                      1))
                Y = np.vstack((Y,temp_array))
    

     #Assembling the phsycial portion of the Y matrix 
        if exp_dict_list[0]['simulation'].physicalSens ==1:
            #print(exp_dict_list)
            for i,exp_dic in enumerate(exp_dict_list):
                
                if loop_counter ==0:
                    if re.match('[Bb]atch[ _-][Rr]eactor',exp_dict_list[i]['simulation_type']) and re.match('[Ss]pecies[ -][Pp]rofile',exp_dict_list[i]['experiment_type']):
                        dic_of_conditions = exp_dic['simulation'].conditions
                        #subtract out the dilluant 
                        species_in_simulation = len(set(dic_of_conditions.keys()).difference(['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']))
                        
                        #add two for Temperature and Pressure
                        len_of_phsycial_observables_in_simulation = species_in_simulation + 2 + 1
                        temp_zeros = np.zeros((len_of_phsycial_observables_in_simulation,1))
                        #stacking the zeros onto the Y array 
                        Y = np.vstack((Y,temp_zeros))
                        Y_data_Frame.append('T'+'_'+'experiment'+'_'+str(i))
                        Y_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))                        
                        for variable in range(species_in_simulation):
                            Y_data_Frame.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                        Y_data_Frame.append('Time_shift'+'_'+'experiment'+'_'+str(i)) 
                        
                    elif re.match('[Jj][Ss][Rr]',exp_dict_list[i]['simulation_type']) and re.match('[Ss]pecies[ -][Pp]rofile',exp_dict_list[i]['experiment_type']):
                        dict_of_conditions = exp_dic['simulation'].conditions
                        species_in_simulation = len(set(dict_of_conditions.keys()).difference(['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']))
                        temperatures_in_simulation = len(exp_dic['simulation'].temperatures)
                        pressure_in_simulation = 1
                        restime_in_simulation = 1
                        len_of_phsycial_observables_in_simulation = species_in_simulation+temperatures_in_simulation+pressure_in_simulation+restime_in_simulation
                        temp_zeros = np.zeros((len_of_phsycial_observables_in_simulation,1))
                        
                        Y = np.vstack((Y,temp_zeros))
                        for value in range(temperatures_in_simulation):
                            Y_data_Frame.append('T'+'_'+'experiment'+'_'+str(i))
                        Y_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))
                        for variable in range(species_in_simulation):
                            Y_data_Frame.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                        Y_data_Frame.append('R_experiment_'+str(i))

                            
                    elif re.match('[Ff]lame [Ss]peed',exp_dict_list[i]['simulation_type']) and re.match('[Oo][Nn][Ee]|[1][ -][dD][ -][Ff]lame',exp_dict_list[i]['experimentType']):
                        conditions = exp_dic['conditions_dict_list']
                        species_to_loop =  exp_dic['uncertainty']['species_relative_uncertainty']['species']
                        pressures_in_simulation = len(exp_dic['simulation'].pressures)
                        list_with_most_species_in_them = []
                        for specie in species_to_loop:
                            list_with_most_species_in_them.append(len(conditions[specie]))
                        max_species = max(list_with_most_species_in_them)
                        
                        if 'Diluant' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluant' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluant = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluant']
                                                     
                             
                       # species_in_simulation = list(exp_dic['conditions_to_run'][0].keys())
                        species_in_simulation = len(set(species_in_simulation).difference(diluant)) * max_species
                        temperatures_in_simulation = len(exp_dic['simulation'].temperatures)
                        pressure_in_simulation = len(exp_dic['simulation'].pressures)
                        len_of_phsycial_observables_in_simulation = species_in_simulation+temperatures_in_simulation+pressure_in_simulation
                        temp_zeros = np.zeros((len_of_phsycial_observables_in_simulation,1))
                        
                        Y = np.vstack((Y,temp_zeros))
                        for value in range(temperatures_in_simulation):
                            Y_data_Frame.append('T'+'_'+'experiment'+'_'+str(i))
                        for value in range(pressures_in_simulation):
                            Y_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))
                        for variable in range(species_in_simulation):
                            Y_data_Frame.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))  
                            
                    elif re.match('[Bb]atch[ _-][Rr]eactor',exp_dict_list[i]['simulation_type']) and re.match('[Ii]gnition[- ][Dd]elay',exp_dict_list[i]['experiment_type']):

                        conditions = exp_dic['conditions_dict_list']
                        species_uncertainties = exp_dic['uncertainty']['species_relative_uncertainty']['dictionary_of_values']
                        species_to_loop =  list(exp_dic['conditions_dict_list'].keys())
                        
                        list_with_most_species_in_them = []
                        for specie in species_to_loop:
                            list_with_most_species_in_them.append(len(conditions[specie]))
                        max_species = max(list_with_most_species_in_them)
                        diluent=[]   
                        if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                             
                        singular_species=[]
                        for species in list(exp_dic['simulation'].fullParsedYamlFile['conditions'].keys()):
                            
                            if len(exp_dic['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                                singular_species.append(species)                            
                             
    
                        #species_in_simulation = len(set(dict_of_conditions.keys()).difference(diluant)) * max_species
                        species = copy.deepcopy(species_to_loop)
                        species_in_simulation = int(len(singular_species)+((len(set(exp_dic['simulation'].fullParsedYamlFile['speciesNames']).difference(diluent))-len(singular_species))*len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])))

                        temperatures_in_simulation = len(exp_dic['simulation'].temperatures)
                        
                        
                        pressures_in_simulation = len(exp_dic['simulation'].pressures)
                        time_shift_length = 1
                        #print(species_in_simulation,temperatures_in_simulation,pressures_in_simulation)
                        len_of_phsycial_observables_in_simulation = species_in_simulation+temperatures_in_simulation+pressures_in_simulation + time_shift_length
                        temp_zeros = np.zeros((len_of_phsycial_observables_in_simulation,1))
                        
                        Y = np.vstack((Y,temp_zeros))
                        for value in range(temperatures_in_simulation):
                            Y_data_Frame.append('T'+str(value+1)+'_'+'experiment'+'_'+str(i))
                        for value in range(pressures_in_simulation):
                            Y_data_Frame.append('P'+str(value+1)+'_'+'experiment'+'_'+str(i))
                            
                            
                        diluent=[]   
                        if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                        
                        
                        
                        for x,species in enumerate(exp_dic['simulation'].fullParsedYamlFile['speciesNames']):
                            if species in singular_species and species not in diluent:
                                Y_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                            elif species not in singular_species and species not in diluent:
                                for j in range(len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])):
                                    Y_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))    
                            
                            
                        
                        Y_data_Frame.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                    elif re.match('[Rr][Cc][Mm]',exp_dict_list[i]['simulation_type']) and re.match('[Ii]gnition[- ][Dd]elay',exp_dict_list[i]['experiment_type']):

                        conditions = exp_dic['conditions_dict_list']
                        species_uncertainties = exp_dic['uncertainty']['species_relative_uncertainty']['dictionary_of_values']
                        species_to_loop =  list(exp_dic['conditions_dict_list'].keys())
                        
                        list_with_most_species_in_them = []
                        for specie in species_to_loop:
                            list_with_most_species_in_them.append(len(conditions[specie]))
                        max_species = max(list_with_most_species_in_them)
                        diluent=[]   
                        if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                             
                        singular_species=[]
                        for species in list(exp_dic['simulation'].fullParsedYamlFile['conditions'].keys()):
                            
                            if len(exp_dic['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                                singular_species.append(species)                            
                             
    
                        #species_in_simulation = len(set(dict_of_conditions.keys()).difference(diluant)) * max_species
                        species = copy.deepcopy(species_to_loop)
                        species_in_simulation = int(len(singular_species)+((len(set(exp_dic['simulation'].fullParsedYamlFile['speciesNames']).difference(diluent))-len(singular_species))*len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])))

                        temperatures_in_simulation = len(exp_dic['simulation'].temperatures)
                        pressures_in_simulation = len(exp_dic['simulation'].pressures)
                        time_shift_length = 1
                        #print(species_in_simulation,temperatures_in_simulation,pressures_in_simulation)
                        len_of_phsycial_observables_in_simulation = species_in_simulation+temperatures_in_simulation+pressures_in_simulation + time_shift_length
                        temp_zeros = np.zeros((len_of_phsycial_observables_in_simulation,1))
                        
                        Y = np.vstack((Y,temp_zeros))
                        for value in range(temperatures_in_simulation):
                            Y_data_Frame.append('T'+str(value+1)+'_'+'experiment'+'_'+str(i))
                        for value in range(pressures_in_simulation):
                            Y_data_Frame.append('P'+str(value+1)+'_'+'experiment'+'_'+str(i))
                            
                            
                        diluent=[]   
                        if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                        
                        
                        
                        for x,species in enumerate(exp_dic['simulation'].fullParsedYamlFile['speciesNames']):
                            if species in singular_species and species not in diluent:
                                Y_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                            elif species not in singular_species and species not in diluent:
                                for j in range(len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])):
                                    Y_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))    
                            
                            
                        
                        Y_data_Frame.append('Time_shift'+'_'+'experiment'+'_'+str(i))                        
                        
                    elif re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp_dict_list[i]['simulation_type']):
                        
                        
                        dict_of_conditions = exp_dic['simulation'].conditions
                        species_in_simulation = len(set(dict_of_conditions.keys()).difference(['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']))
                        temperatures_in_simulation = len(exp_dic['simulation'].temperatures)
                        time_shift_in_simulation = len(parsed_yaml_file_list[i]['timeShiftOriginal'])
                        pressure_in_simulation = 1
                        len_of_phsycial_observables_in_simulation = species_in_simulation+temperatures_in_simulation+pressure_in_simulation+time_shift_in_simulation
                        temp_zeros = np.zeros((len_of_phsycial_observables_in_simulation,1))
                        
                        Y = np.vstack((Y,temp_zeros))
                        for value in range(temperatures_in_simulation):
                            Y_data_Frame.append('T'+'_'+'experiment'+'_'+str(i))
                        Y_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))
                        for variable in range(species_in_simulation):
                            Y_data_Frame.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                        
                        for variable in range(time_shift_in_simulation):
                            Y_data_Frame.append('Time_shift'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                        
                        
                        
                        
                else:
                    if re.match('[Bb]atch[ _-][Rr]eactor',exp_dict_list[i]['simulation_type']) and re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']):
                        dic_of_conditions = exp_dic['simulation'].conditions
                        #subtract out the dilluant 
                        species_in_simulation = len(set(dic_of_conditions.keys()).difference(['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']))
                        
                        Y_data_Frame.append('T'+'_'+'experiment'+'_'+str(i))
                        Y_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))
                        for variable in range(species_in_simulation):
                            Y_data_Frame.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                        Y_data_Frame.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                        

                    elif re.match('[Jj][Ss][Rr]',exp_dict_list[i]['simulation_type']):
                        dict_of_conditions = exp_dic['simulation'].conditions
                        species_in_simulation = len(set(dict_of_conditions.keys()).difference(['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']))
                        temperatures_in_simulation = len(exp_dic['simulation'].temperatures)
                        pressure_in_simulation = 1
                        restime_in_simulation = 1
                        len_of_phsycial_observables_in_simulation = species_in_simulation+temperatures_in_simulation+pressure_in_simulation+restime_in_simulation
                        for value in range(temperatures_in_simulation):
                            Y_data_Frame.append('T'+'_'+'experiment'+'_'+str(i))
                        Y_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))
                        for variable in range(species_in_simulation):
                            Y_data_Frame.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                        Y_data_Frame.append('R_experiment_'+str(i))

                    
                    elif re.match('[Ff]lame [Ss]peed',exp_dict_list[i]['simulation_type']) and re.match('[Oo][Nn][Ee]|[1][ -][dD][ -][Ff]lame',exp_dict_list[i]['experimentType']):
                        species_to_loop =  exp_dic['uncertainty']['species_relative_uncertainty']['species']
                        list_with_most_species_in_them = []
                        for specie in species_to_loop:
                            list_with_most_species_in_them.append(len(conditions[specie]))
                        max_species = max(list_with_most_species_in_them)
                        
                        if 'Diluant' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluant' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluant = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluant']
   
                        species_in_simulation = len(set(dict_of_conditions.keys()).difference(diluant)) * max_species
                        temperatures_in_simulation = len(exp_dic['simulation'].temperatures)
                        pressure_in_simulation = len(exp_dic['simulation'].pressures)
                        len_of_phsycial_observables_in_simulation = species_in_simulation+temperatures_in_simulation+pressure_in_simulation
                        
                        for value in range(temperatures_in_simulation):
                            Y_data_Frame.append('T'+'_'+'experiment'+'_'+str(i))
                        for value in range(pressures_in_simulation):
                            Y_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))
                        for variable in range(species_in_simulation):
                            Y_data_Frame.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))  
                    
                    
                    
                    elif re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp_dict_list[i]['simulation_type']):
                        
                        
                        dict_of_conditions = exp_dic['simulation'].conditions
                        species_in_simulation = len(set(dict_of_conditions.keys()).difference(['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']))
                        temperatures_in_simulation = len(exp_dic['simulation'].temperatures)
                        time_shift_in_simulation = len(parsed_yaml_file_list[i]['timeShiftOriginal'])
                        pressure_in_simulation = 1
                        
                
                        for value in range(temperatures_in_simulation):
                            Y_data_Frame.append('T'+'_'+'experiment'+'_'+str(i))
                        Y_data_Frame.append('P'+'_'+'experiment'+'_'+str(i))
                        for variable in range(species_in_simulation):
                            Y_data_Frame.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                        
                        for variable in range(time_shift_in_simulation):
                            Y_data_Frame.append('Time_shift'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                    
                    
                    
                    elif re.match('[Bb]atch[ _-][Rr]eactor',exp_dict_list[i]['simulation_type']) and re.match('[Ii]gnition[- ][Dd]elay',exp_dict_list[i]['experiment_type']):
                        conditions = exp_dic['conditions_dict_list']
                        species_to_loop =  list(exp_dic['conditions_dict_list'].keys())
                        temperatures_in_simulation = len(exp_dic['simulation'].temperatures)
                        pressures_in_simulation = len(exp_dic['simulation'].pressures)
                        list_with_most_species_in_them = []
                        for specie in species_to_loop:
                            list_with_most_species_in_them.append(len(conditions[specie]))
                        max_species = max(list_with_most_species_in_them)
                        diluant=[]   
                        if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluant = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                             
                        
                        for value in range(temperatures_in_simulation):
                            Y_data_Frame.append('T'+str(value+1)+'_'+'experiment'+'_'+str(i))
                        for value in range(pressures_in_simulation):
                            Y_data_Frame.append('P'+str(value+1)+'_'+'experiment'+'_'+str(i))
                            
                        diluent=[]   
                        if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                        
                        singular_species=[]
                        for species in list(exp_dic['simulation'].fullParsedYamlFile['conditions'].keys()):
                            
                            if len(exp_dic['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                                singular_species.append(species)
                        
                        for x,species in enumerate(exp_dic['simulation'].fullParsedYamlFile['speciesNames']):
                            if species in singular_species and species not in diluent:
                                Y_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                            elif species not in singular_species and species not in diluent:
                                for j in range(len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])):
                                    Y_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                        Y_data_Frame.append('Time_shift'+'_'+'experiment'+'_'+str(i)) 

                    elif re.match('[Rr][Cc][Mm]',exp_dict_list[i]['simulation_type']) and re.match('[Ii]gnition[- ][Dd]elay',exp_dict_list[i]['experiment_type']):
                        conditions = exp_dic['conditions_dict_list']
                        species_to_loop =  list(exp_dic['conditions_dict_list'].keys())
                        temperatures_in_simulation = len(exp_dic['simulation'].temperatures)
                        pressures_in_simulation = len(exp_dic['simulation'].pressures)
                        list_with_most_species_in_them = []
                        for specie in species_to_loop:
                            list_with_most_species_in_them.append(len(conditions[specie]))
                        max_species = max(list_with_most_species_in_them)
                        diluant=[]   
                        if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluant = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                             
                        
                        for value in range(temperatures_in_simulation):
                            Y_data_Frame.append('T'+str(value+1)+'_'+'experiment'+'_'+str(i))
                        for value in range(pressures_in_simulation):
                            Y_data_Frame.append('P'+str(value+1)+'_'+'experiment'+'_'+str(i))
                            
                        diluent=[]   
                        if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                            diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                        
                        singular_species=[]
                        for species in list(exp_dic['simulation'].fullParsedYamlFile['conditions'].keys()):
                            
                            if len(exp_dic['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                                singular_species.append(species)
                        
                        for x,species in enumerate(exp_dic['simulation'].fullParsedYamlFile['speciesNames']):
                            if species in singular_species and species not in diluent:
                                Y_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                            elif species not in singular_species and species not in diluent:
                                for j in range(len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])):
                                    Y_data_Frame.append('X'+str(x+1)+'_'+species+'_experiment_'+str(i))
                        Y_data_Frame.append('Time_shift'+'_'+'experiment'+'_'+str(i)) 
                        
                    if i==len(exp_dict_list)-1:
                        temp_array = np.array(X['physical_observables'])*-1

                        temp_array = temp_array.reshape((temp_array.shape[0],
                                                      1))
                        Y = np.vstack((Y,temp_array))

            
        #Assembling the portion of the Y matrix for the absorbance coefficient sensitiviteis 
        pert_coef = {} #build a dict matching pert_coef to their experiment and wavelength.             
               #length of the dict gives padding information
        for exp in exp_dict_list:
            if 'perturbed_coef' not in exp.keys():
                continue
            perturbed_for_exp = exp['perturbed_coef']
            for x in perturbed_for_exp:
                if x[0][2] not in pert_coef.keys():
                    pert_coef[x[0][2]] = [x[1]]
                else:
                    pert_coef[x[0][2]].append(x[1])
                    
        num_ind_pert_coef = len(pert_coef)         
        temp_zeros = np.zeros((num_ind_pert_coef,1))
        if loop_counter == 0:
            Y = np.vstack((Y,temp_zeros))
        else:  
            if 'absorbance_coefficent_observables' in X.keys():
                
                #temp_array = np.array(X['absorbance_coefficent_observables'])
                temp_array = X['absorbance_coefficent_observables'] 
                temp_array = [a for a in temp_array if a != 'null']
               
                #temp_array = temp_array[temp_array!=0]
                #temp_array = temp_array[temp_array!=0]
                temp_array = np.array(temp_array)
                temp_array = np.array(temp_array)*-1
                
                temp_array = temp_array.reshape((temp_array.shape[0],
                                                      1))
                
                Y = np.vstack((Y,temp_array))
                
        for x in range(num_ind_pert_coef):
            Y_data_Frame.append('Sigma'+'_'+str(x))
        
        
        Y_data_Frame = pd.DataFrame({'value': Y_data_Frame,'ln_difference': Y.reshape((Y.shape[0],))})  
        
        self.Y_matrix = Y
        #print(Y.shape,'Y matrix without k targets')

        return Y, Y_data_Frame       

    def load_S(self, exp_dict_list:list,parsed_yaml_list:list,
               dk=.01,
               master_equation_reactions = [],
               mapped_master_equation_sensitivities=np.array(()),
               master_equation_uncertainty_df = None,
               master_equation_flag = False):
        
        '''
        Builds the S vector. 
        
        Arguments:
            exp_dic_list -- the dictionary that is built after a simulation
            that contains things like sensitivity coefficients
            parsed_yaml_file_list -- a list of dictonaries that contain the 
            information stored in the yaml files. 
        Keyword Arguments:
            dk -- pertabation percent (defalut 0.01)
            mapped_master_equation_sensitivities -- numpy matrix of theory parameter sensitivity values (default empty array)
            master_equation_uncertainty_df -- a pandas dataframe that contains 
            the reactions being treated with theory paramters along with the 
            associated uncertainty values of those paramters (default None)
            master_equation_reaction_list -- a list of the reactions being treated
            with theory paramters (default [])
            master_equation_flag -- a boolean that indicates if reactions being
            represented by theory parameters are being used in the optimization (default False)
            
        '''
        
        #preprocessing for padding
        num_exp = len(exp_dict_list)
        pert_coef = {} #build a dict matching pert_coef to their experiment and wavelength.
                       #length of the dict gives padding information
        list_to_keep_order_of_coef = []
        for exp in exp_dict_list:
            if 'perturbed_coef' not in exp.keys():
                continue
            perturbed_for_exp = exp['perturbed_coef']
            for x in perturbed_for_exp:
                if x[0][2] not in pert_coef.keys():
                    pert_coef[x[0][2]] = [x[1]]
                else:

                    pert_coef[x[0][2]].append(x[1])

                    
                if x[0][2] not in list_to_keep_order_of_coef:
                    list_to_keep_order_of_coef.append(x[0][2])
            
        num_ind_pert_coef = len(pert_coef)
        #print(pert_coef.keys())
        
        #print(num_ind_pert_coef," sigmas")
        #establish # of independent pert before hand, to proper pad the observables, put in list, make a dict of cc,
        # values will be a list of tabs data?
        # use the list to get the padding size
        k_sens_for_whole_simulation = []
        p_sens_for_whole_simulation = []
        abs_coef_sens_for_whole_simulation = []
        
        temps = []
        for i,exp in enumerate(exp_dict_list):
            ttl_kinetic_observables_for_exp = []
            obs_counter =0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']+ exp['ignition_delay_observables']):
                if observable == None:
                    continue


                single_obs_matrix = np.hstack((exp['ksens']['A'][obs_counter],
                                        exp['ksens']['N'][obs_counter],
                                        exp['ksens']['Ea'][obs_counter]))
                
                #print(single_obs_matrix)
                ttl_kinetic_observables_for_exp.append(single_obs_matrix)
                obs_counter +=1
                
            if 'perturbed_coef' in exp.keys():
                wavelengths = parsed_yaml_list[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    single_obs_matrix = np.hstack((exp['absorbance_ksens'][wl][0],
                                                   exp['absorbance_ksens'][wl][1],
                                                   exp['absorbance_ksens'][wl][2]))
                    
                    ttl_kinetic_observables_for_exp.append(single_obs_matrix)
                                       
            ttl_kinetic_observables_for_exp = np.vstack((ttl_kinetic_observables_for_exp))              
            k_sens_for_whole_simulation.append(ttl_kinetic_observables_for_exp)
            #print(np.shape(k_sens_for_whole_simulation))
            ####vstack  ttl_kinetic_observables_for_exp   and append somwehre else
            
            if exp['simulation'].physicalSens ==1:
                ttl_phsycal_obs_for_exp = []
                for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables'] + exp['ignition_delay_observables']):
                    obs_counter = 0
                    if observable == None:
                        continue
                    
                    if re.match('[Bb]atch[ _-][Rr]eactor',exp['simulation_type']) and re.match('[Ss]pecies[- ][Pp]rofile',exp['experiment_type']):
                        temperature_sensitivity = exp['temperature'][observable].dropna().values
                        temperature_sensitivity = temperature_sensitivity.reshape((temperature_sensitivity.shape[0], 1))
                        
                        time_shift_sensitivity = exp['time_shift'][observable].dropna().values
                        time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))
                        pressure_sensitivity = exp['pressure'][observable].dropna().values
                        pressure_sensitivity = pressure_sensitivity.reshape((pressure_sensitivity.shape[0], 1))                        
                        species_sensitivty = []
                        for df in exp['species']:
                            single_species_sensitivty = df[observable].dropna().values
                            single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0]
                                                               ,1))
                            species_sensitivty.append(single_species_sensitivty)
                            
                        species_sensitivty = np.hstack((species_sensitivty))
                        
                    elif re.match('[Jj][Ss][Rr]',exp['simulation_type']):
                        temperature_sensitivity=np.array(exp['temperature'][observable])*np.identity(len(exp['simulation'].temperatures))                    
                        pressure_sensitivity = exp['pressure'][observable].dropna().values
                        pressure_sensitivity = pressure_sensitivity.reshape((pressure_sensitivity.shape[0], 1))
                        restime_sensitivity=exp['restime_sens'][observable].dropna().values
                        restime_sensitivity = restime_sensitivity.reshape((restime_sensitivity.shape[0],1))
                        
                        
                        species_sensitivty = []
                        for df in exp['species']:
                            single_species_sensitivty = df[observable].dropna().values
                            single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                            species_sensitivty.append(single_species_sensitivty)
                            
                        species_sensitivty = np.hstack((species_sensitivty))
                        restime_sensitivity=exp['restime_sens'][observable].dropna().values
                        restime_sensitivity = restime_sensitivity.reshape((restime_sensitivity.shape[0],1))

                    elif re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp_dict_list[i]['simulation_type']):
                        temperature_sensitivity=np.array(exp['temperature'][observable])*np.identity(len(exp['simulation'].temperatures))                    
                        pressure_sensitivity = exp['pressure'][observable].dropna().values
                        pressure_sensitivity = pressure_sensitivity.reshape((pressure_sensitivity.shape[0], 1))

                        
                        
                        species_sensitivty = []
                        for df in exp['species']:
                            single_species_sensitivty = df[observable].dropna().values
                            single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                            species_sensitivty.append(single_species_sensitivty)
                            
                        species_sensitivty = np.hstack((species_sensitivty))
                        if len(parsed_yaml_list[i]['timeShiftOriginal'])>1:
                            time_shift_sensitivity = np.array(exp['time_shift'][observable])*np.identity(len(exp['simulation'].temperatures))  
                        else:
                            time_shift_sensitivity = np.array(exp['time_shift'][observable])
                            time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))

                        
                    elif re.match('[Ii]gnition[- ][Dd]elay',exp['experiment_type']) and re.match('[Bb]atch[- ][Rr]eactor',exp['simulation_type']):



                    	#CHECK HOW MANY SPECIES THERE ARE.
                        conditions = exp['conditions_dict_list']
                        species_to_loop =  list(exp['conditions_dict_list'].keys())
                        list_with_most_species_in_them = []
                            
                        for specie in species_to_loop:
                        	list_with_most_species_in_them.append(len(conditions[specie]))

                        if len(exp['simulation'].temperatures)>1 and len(exp['simulation'].pressures)==1:
                            temperature_sensitivity=np.array(exp['temperature']['delay'])*np.identity(len(exp['simulation'].temperatures))
                            pressure_sensitivity = exp['pressure']['delay'].dropna().values
                            pressure_sensitivity = pressure_sensitivity.reshape((pressure_sensitivity.shape[0], 1))
                            species_sensitivty = []
                            for df in exp['species']:
                                single_species_sensitivty = df['delay'].dropna().values
                                single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                                species_sensitivty.append(single_species_sensitivty)
                            
                            species_sensitivty = np.hstack((species_sensitivty))
                            time_shift_sensitivity = exp['time_shift']['delay'].dropna().values
                            time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))
                            #print("INSIDE HERE")

                        elif len(exp['simulation'].pressures)>1 and len(exp['simulation'].temperatures)==1:
                            pressure_sensitivity = np.array(exp['pressure']['delay'])*np.identity(len(exp['simulation'].pressures))
                            temperature_sensitivity = exp['temperature']['delay'].dropna().values
                            temperature_sensitivity = temperature_sensitivity.reshape((temperature_sensitivity.shape[0], 1))
                            species_sensitivty = []
                            for df in exp['species']:
                                single_species_sensitivty = df['delay'].dropna().values
                                single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                                species_sensitivty.append(single_species_sensitivty)
                            
                            species_sensitivty = np.hstack((species_sensitivty))
                            time_shift_sensitivity = exp['time_shift']['delay'].dropna().values
                            time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))
                            
                        elif len(exp['simulation'].pressures)==1 and len(exp['simulation'].temperatures)==1 and len(list_with_most_species_in_them)>1:
                            pressure_sensitivity = exp['pressure']['delay'].dropna().values
                            pressure_sensitivity = pressure_sensitivity.reshape((pressure_sensitivity.shape[0], 1))
                            temperature_sensitivity = exp['temperature']['delay'].dropna().values
                            temperature_sensitivity = temperature_sensitivity.reshape((temperature_sensitivity.shape[0], 1))
                            species_sensitivty=[]
                            conditions = exp['conditions_dict_list']
                            
                            species_to_loop =  list(exp['conditions_dict_list'].keys())
                            list_with_most_species_in_them = []
                            
                            for specie in species_to_loop:
                                list_with_most_species_in_them.append(len(conditions[specie]))
                            
                            diluent=[]   
                            if 'Diluent' in exp['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                                diluent = exp['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                            
                            singular_species=[]
                            for species in list(exp['simulation'].fullParsedYamlFile['conditions'].keys()):
                                
                                if len(exp['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                                    singular_species.append(species)
                            
                            for x,species in enumerate(exp['simulation'].fullParsedYamlFile['speciesNames']):
                                if species in singular_species and species not in diluent:
                                    single_species_sensitivty = exp['species'][x]['delay'].dropna().values
                                    single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                                    #print(single_species_sensitivty)
                                    species_sensitivty.append(single_species_sensitivty)
                                elif species not in singular_species and species not in diluent:
                                    single_species_sensitivty = np.array(exp['species'][x]['delay'])*np.identity(len(exp['species'][x]['delay'])) 
                                    species_sensitivty.append(single_species_sensitivty)
                            species_sensitivty=np.hstack((species_sensitivty))
                            time_shift_sensitivity = exp['time_shift']['delay'].dropna().values
                            time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))

                        elif len(exp['simulation'].pressures)>1 and len(exp['simulation'].temperatures)>1 and len(list_with_most_species_in_them)>1 and len(exp['simulation'].pressures)==len(exp['simulation'].temperatures):
                            
                            temperature_sensitivity=np.array(exp['temperature']['delay'])*np.identity(len(exp['simulation'].temperatures))
                            pressure_sensitivity = np.array(exp['pressure']['delay'])*np.identity(len(exp['simulation'].pressures))

                            species_sensitivty=[]
                            conditions = exp['conditions_dict_list']
                            
                            species_to_loop =  list(exp['conditions_dict_list'].keys())
                            list_with_most_species_in_them = []
                            
                            for specie in species_to_loop:
                                list_with_most_species_in_them.append(len(conditions[specie]))
                            
                            diluent=[]   
                            if 'Diluent' in exp['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                                diluent = exp['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                            
                            singular_species=[]
                            for species in list(exp['simulation'].fullParsedYamlFile['conditions'].keys()):
                                
                                if len(exp['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                                    singular_species.append(species)
                            
                            for x,species in enumerate(exp['simulation'].fullParsedYamlFile['speciesNames']):
                                if species in singular_species and species not in diluent:
                                    single_species_sensitivty = exp['species'][x]['delay'].dropna().values
                                    single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                                    #print(single_species_sensitivty)
                                    species_sensitivty.append(single_species_sensitivty)
                                elif species not in singular_species and species not in diluent:
                                    single_species_sensitivty = np.array(exp['species'][x]['delay'])*np.identity(len(exp['species'][x]['delay'])) 
                                    species_sensitivty.append(single_species_sensitivty)
                            species_sensitivty=np.hstack((species_sensitivty))
                            time_shift_sensitivity = exp['time_shift']['delay'].dropna().values
                            time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))                        


                        elif len(exp['simulation'].pressures)>1 and len(exp['simulation'].temperatures)>1 and len(exp['simulation'].pressures) == len(exp['simulation'].temperatures):
                            temperature_sensitivity=np.array(exp['temperature']['delay'])*np.identity(len(exp['simulation'].temperatures))
                            pressure_sensitivity = np.array(exp['pressure']['delay'])*np.identity(len(exp['simulation'].pressures))

                            species_sensitivty = []
                            for df in exp['species']:
                                single_species_sensitivty = df['delay'].dropna().values
                                single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                                species_sensitivty.append(single_species_sensitivty)
                            
                            species_sensitivty = np.hstack((species_sensitivty))
                            time_shift_sensitivity = exp['time_shift']['delay'].dropna().values
                            time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))


                    elif re.match('[Ii]gnition[- ][Dd]elay',exp['experiment_type']) and re.match('[Rr][Cc][Mm]',exp['simulation_type']):
                        if len(exp['simulation'].temperatures)>1 and len(exp['simulation'].pressures)>1:
                            temperature_sensitivity=np.array(exp['temperature']['delay'])*np.identity(len(exp['simulation'].temperatures))
                            pressure_sensitivity = np.array(exp['pressure']['delay'])*np.identity(len(exp['simulation'].pressures))

                            species_sensitivty = []
                            for df in exp['species']:
                                single_species_sensitivty = df['delay'].dropna().values
                                single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                                species_sensitivty.append(single_species_sensitivty)
                            
                            species_sensitivty = np.hstack((species_sensitivty))
                            time_shift_sensitivity = exp['time_shift']['delay'].dropna().values
                            time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))



                        elif len(exp['simulation'].temperatures)>1:
                            temperature_sensitivity=np.array(exp['temperature']['delay'])*np.identity(len(exp['simulation'].temperatures))
                            pressure_sensitivity = exp['pressure']['delay'].dropna().values
                            pressure_sensitivity = pressure_sensitivity.reshape((pressure_sensitivity.shape[0], 1))
                            species_sensitivty = []
                            for df in exp['species']:
                                single_species_sensitivty = df['delay'].dropna().values
                                single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                                species_sensitivty.append(single_species_sensitivty)
                            
                            species_sensitivty = np.hstack((species_sensitivty))
                            time_shift_sensitivity = exp['time_shift']['delay'].dropna().values
                            time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))
                            #print("INSIDE HERE")
                        elif len(exp['simulation'].pressures)>1:
                            pressure_sensitivity = np.array(exp['pressure']['delay'])*np.identity(len(exp['simulation'].pressures))
                            temperature_sensitivity = exp['temperature']['delay'].dropna().values
                            temperature_sensitivity = temperature_sensitivity.reshape((temperature_sensitivity.shape[0], 1))
                            species_sensitivty = []
                            for df in exp['species']:
                                single_species_sensitivty = df['delay'].dropna().values
                                single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                                species_sensitivty.append(single_species_sensitivty)
                            
                            species_sensitivty = np.hstack((species_sensitivty))
                            time_shift_sensitivity = exp['time_shift']['delay'].dropna().values
                            time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))
                            
                        elif len(exp['simulation'].pressures)==1 and len(exp['simulation'].temperatures)==1:
                            pressure_sensitivity = exp['pressure']['delay'].dropna().values
                            pressure_sensitivity = pressure_sensitivity.reshape((pressure_sensitivity.shape[0], 1))
                            temperature_sensitivity = exp['temperature']['delay'].dropna().values
                            temperature_sensitivity = temperature_sensitivity.reshape((temperature_sensitivity.shape[0], 1))
                            species_sensitivty=[]
                            conditions = exp['conditions_dict_list']
                            
                            species_to_loop =  list(exp['conditions_dict_list'].keys())
                            list_with_most_species_in_them = []
                            
                            for specie in species_to_loop:
                                list_with_most_species_in_them.append(len(conditions[specie]))
                            
                            diluent=[]   
                            if 'Diluent' in exp['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                                diluent = exp['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                            
                            singular_species=[]
                            for species in list(exp['simulation'].fullParsedYamlFile['conditions'].keys()):
                                
                                if len(exp['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                                    singular_species.append(species)
                            
                            for x,species in enumerate(exp['simulation'].fullParsedYamlFile['speciesNames']):
                                if species in singular_species and species not in diluent:
                                    single_species_sensitivty = exp['species'][x]['delay'].dropna().values
                                    single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0],1))
                                    #print(single_species_sensitivty)
                                    species_sensitivty.append(single_species_sensitivty)
                                elif species not in singular_species and species not in diluent:
                                    single_species_sensitivty = np.array(exp['species'][x]['delay'])*np.identity(len(exp['species'][x]['delay'])) 
                                    species_sensitivty.append(single_species_sensitivty)
                            species_sensitivty=np.hstack((species_sensitivty))
                            time_shift_sensitivity = exp['time_shift']['delay'].dropna().values
                            time_shift_sensitivity = time_shift_sensitivity.reshape((time_shift_sensitivity.shape[0], 1))




                            
                            
                    elif re.match('[Ff]lame[- ][Ss]peed',exp['simulation_type']) and re.match('[Oo][Nn][Ee]|[1][ -][dD][ -][Ff]lame',exp['experiment_type']):
                        len_of_temperature_list = len(exp['simulation'].temperatures)
                        if len_of_temperature_list > 1:
                            temperature_sensitivity=np.array(exp['temperature'][observable])*np.identity(len(exp['simulation'].temperatures))
                        else:
                            temperature_sensitivity = np.array(exp['temperature'][observable])
                            temperature_sensitivity = temperature_sensitivity.reshape((temperature_sensitivity.shape[0], 1))
                        len_of_pressure_list = len(exp['simulation'].pressures)
                        if len_of_pressure_list >1:
                            pressure_sensitivity=np.array(exp['pressure'][observable])*np.identity(len(exp['simulation'].pressures))
                        else:
                            pressure_sensitivity=np.array(exp['pressure'][observable])
                            pressure_sensitivity = pressure_sensitivity.reshape((pressure_sensitivity.shape[0], 1))

                        #FIX THIS 
                        #print('FIXXXX')
                        species_sensitivty = []
                        for df in exp['species']:
                            single_species_sensitivty = df[observable].dropna().values
                            single_species_sensitivty = single_species_sensitivty.reshape((single_species_sensitivty.shape[0]
                                                               ,1))
                            species_sensitivty.append(single_species_sensitivty)
                            
                        species_sensitivty = np.hstack((species_sensitivty))
                                            
                    
                    if re.match('[Jj][Ss][Rr]',exp['simulation_type']):
                        single_obs_physical = np.hstack((temperature_sensitivity,pressure_sensitivity,species_sensitivty,restime_sensitivity))

                       
                    elif re.match('[Bb]atch[ _-][Rr]eactor',exp['simulation_type']) and re.match('[Ss]pecies[- ][Pp]rofile',exp['experiment_type']):
                        single_obs_physical = np.hstack((temperature_sensitivity,pressure_sensitivity,species_sensitivty,time_shift_sensitivity)) 
                        
                        
                        
                    elif re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp_dict_list[i]['simulation_type']):
                        single_obs_physical = np.hstack((temperature_sensitivity,pressure_sensitivity,species_sensitivty,time_shift_sensitivity)) 

    
                    elif re.match('[Ii]gnition[- ][Dd]elay',exp['experiment_type']) and re.match('[Bb]atch[- ][Rr]eactor',exp['simulation_type']):
                        single_obs_physical = np.hstack((temperature_sensitivity,pressure_sensitivity,species_sensitivty,time_shift_sensitivity))
                        #print("INSIDE HERE")
                    
                    elif re.match('[Ii]gnition[- ][Dd]elay',exp['experiment_type']) and re.match('[Rr][Cc][Mm]',exp['simulation_type']):
                        single_obs_physical = np.hstack((temperature_sensitivity,pressure_sensitivity,species_sensitivty,time_shift_sensitivity))                    
                    
                    
                    ttl_phsycal_obs_for_exp.append(single_obs_physical)
                    obs_counter +=1
                    
                    
                if 'perturbed_coef' in exp.keys():
                    wavelengths = parsed_yaml_list[i]['absorbanceCsvWavelengths']                    
                    for k,wl in enumerate(wavelengths):
                        physical_sens = []
                        for p_sens in exp['absorbance_psens']:                            
                            array = p_sens[wl]
                            array = array.reshape((array.shape[0],1))
                            physical_sens.append(array)
                        for time_sens in exp['absorbance_time_shift']:
                            array2 = p_sens[wl]
                            array2 = array2.reshape((array2.shape[0],1))
                            physical_sens.append(array2)
                            
                        physical_sens = np.hstack((physical_sens))
                        ttl_phsycal_obs_for_exp.append(physical_sens)
                    
                ttl_phsycal_obs_for_exp = np.vstack((ttl_phsycal_obs_for_exp))
                p_sens_for_whole_simulation.append(ttl_phsycal_obs_for_exp)
#######################################################################################################################################################               

            
            
            if 'perturbed_coef' in exp.keys():
                ttl_absorbance_obs_for_exp = []
                wavelengths = parsed_yaml_list[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    perturbed_coefficeints = []
                    index_list = []
                    for xx in range(len(parsed_yaml_list[i]['coupledCoefficients'])):
                        for yy in range(len(parsed_yaml_list[i]['coupledCoefficients'][xx])):
                            ff = parsed_yaml_list[i]['functionalForm'][xx][yy]
                            #temp = list(parsed_yaml_list[i]['coupledCoefficients'][xx][yy])
                            for zz in range(len(parsed_yaml_list[i]['coupledCoefficients'][xx][yy])):
                                temp = list(parsed_yaml_list[i]['coupledCoefficients'][xx][yy])
                                coefficent = parsed_yaml_list[i]['coupledCoefficients'][xx][yy][zz]    
                                if coefficent!=0:
                                    perturbed_coefficent=coefficent+coefficent*dk
                                    if zz==1 and ff =='F':
                                        #change back tab
                                        perturbed_coefficent = coefficent + .01*coefficent
                                    temp[zz] = perturbed_coefficent

                                    key = tuple(temp)

                                    indx = list_to_keep_order_of_coef.index(key)
                                    
                                    index_list.append(indx)
                                    
                                    exp_index_sigma = temps.count(key)
                                    temps.append(key)

                                    array = pert_coef[key][exp_index_sigma][wl]
                                    
                                    array = array.reshape((array.shape[0],1))
                                    perturbed_coefficeints.append(array)

        
                    missing_sigmas = []  
                    for indp_sigma in range(len(list_to_keep_order_of_coef)):
                        if indp_sigma not in index_list:
                            missing_sigmas.append(indp_sigma)
                            
                    perturbed_coefficents_padded_with_zeros = []
                    count_sigma=0
                    for indp_sigma in range(len(list_to_keep_order_of_coef)):
                        if indp_sigma in missing_sigmas:
                            zero_array = np.zeros((perturbed_coefficeints[0].shape[0],1))
                            perturbed_coefficents_padded_with_zeros.append(zero_array)                            
                        else:
                            perturbed_coefficents_padded_with_zeros.append(perturbed_coefficeints[count_sigma])
                            count_sigma +=1
                    
                    
                        
                        
                        
                        
                    
                    perturbed_coefficents_padded_with_zeros = np.hstack((perturbed_coefficents_padded_with_zeros))
                    ttl_absorbance_obs_for_exp.append(perturbed_coefficents_padded_with_zeros) 
                    
                ttl_absorbance_obs_for_exp = np.vstack((ttl_absorbance_obs_for_exp))
                abs_coef_sens_for_whole_simulation.append(ttl_absorbance_obs_for_exp)
            
                #vstack ttl_absorbance_obs_for_exp and append somehwere else 
            else:
                 abs_coef_sens_for_whole_simulation.append(0)
                
               
######################################################################################################################################################
        flatten = lambda *n: (e for a in n
            for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))        
        
        flattened_master_equation_reaction_list = list(flatten(master_equation_reactions)) 
                
        #assembling the S matrix from the individual experiments 
        #master_equation = False
        if master_equation_flag == True:
            S_ksens = np.vstack((k_sens_for_whole_simulation))           
            A_k = np.hsplit(S_ksens,3)[0]
            N_k = np.hsplit(S_ksens,3)[1]
            Ea_k  = np.hsplit(S_ksens,3)[2]
            
            number_of_master_equation_reactions = len(flattened_master_equation_reaction_list)
            
            A_k = A_k[:,:-number_of_master_equation_reactions]
            N_k = N_k[:,:-number_of_master_equation_reactions]
            Ea_k = Ea_k[:,:-number_of_master_equation_reactions]
            
           
            S_ksens = np.hstack((A_k,N_k,Ea_k))
            #print(np.shape(S_ksens),'this is the shape of the S matrix before MP')
            S_ksens = np.hstack((S_ksens,mapped_master_equation_sensitivities))
            

        else:
            S_ksens = np.vstack((k_sens_for_whole_simulation))
        
        
        def sum_of_zeros(idx,array,column_list):
            rows_behind = array.shape[0]
            rows_infront = array.shape[0]
            columns_behind = sum(column_list[:idx])
            columns_infront = sum(column_list[idx+1:])  
            behind_tuple = (rows_behind,columns_behind)
            infront_tuple = (rows_infront,columns_infront)
            
            return (behind_tuple,infront_tuple)
        
        
        if exp_dict_list[0]['simulation'].physicalSens ==1:
            number_of_columns_in_psens_arrays = []
            number_of_rows_in_psens_arrays=[]
            
            for i,array in enumerate(p_sens_for_whole_simulation):                
                number_of_rows_in_psens_arrays.append(array.shape[0])
                number_of_columns_in_psens_arrays.append(array.shape[1])
                
            p_sens_whole_simulation_with_padding = []
            for i,array in enumerate(p_sens_for_whole_simulation):
                zero_array_behind = np.zeros(sum_of_zeros(i,array,number_of_columns_in_psens_arrays)[0])
                if zero_array_behind.shape[1] != 0:
                    array = np.hstack((zero_array_behind,array))
                zero_array_infront = np.zeros(sum_of_zeros(i,array,number_of_columns_in_psens_arrays)[1])
                if zero_array_infront.shape[1] != 0:
                    array = np.hstack((array,zero_array_infront))
                
                p_sens_whole_simulation_with_padding.append(array)
            
            S_psens = np.vstack((p_sens_whole_simulation_with_padding))
            
            
            
            
##############################################################################################         
        absorb_coef_whole_simulation_with_padding = []
        
        for i,exp in enumerate(exp_dict_list):
            single_experiment_absorption = []
            if exp['mole_fraction_observables'][0] != None or exp['concentration_observables'][0] != None or exp['ignition_delay_observables'][0] != None:                
                if 'perturbed_coef' not in exp.keys():
                    zero_array_for_observables_padding = np.zeros((number_of_rows_in_psens_arrays[i],
                                           num_ind_pert_coef))    
                    
                    single_experiment_absorption.append(zero_array_for_observables_padding)
                    
                    
            if 'perturbed_coef' in exp.keys():  
                zero_padded_aborption_coef_array = abs_coef_sens_for_whole_simulation[i]   
                combined = abs_coef_sens_for_whole_simulation[i] 

                if exp['mole_fraction_observables'][0] != None or exp['concentration_observables'][0] != None or exp['ignition_delay_observables'][0] != None:
                    zero_array_for_observables_padding = np.zeros((number_of_rows_in_psens_arrays[i]-zero_padded_aborption_coef_array.shape[0],
                                       num_ind_pert_coef))  
                    combined = np.vstack((zero_array_for_observables_padding,zero_padded_aborption_coef_array))
                
                
                
                single_experiment_absorption.append(combined)     

            single_experiment_absorption = np.vstack((single_experiment_absorption))
            absorb_coef_whole_simulation_with_padding.append(single_experiment_absorption) 
            
        
        
        absorb_coef_whole_simulation_with_padding = np.vstack((absorb_coef_whole_simulation_with_padding))  
        S_abs_coef  = absorb_coef_whole_simulation_with_padding

        
 
        S_matrix = np.hstack((S_ksens,S_psens,S_abs_coef))
        shape = np.shape(S_matrix)[1]
        #append identy matrix
        identity_matrix = np.identity(shape)
               

        
        
        
        S_matrix = np.vstack((S_matrix,identity_matrix))
        self.S_matrix = S_matrix
        S_matrix_wo_k_targets = copy.deepcopy(self.S_matrix)
        self.S_matrix_wo_k_targets = S_matrix_wo_k_targets
        #print(S_matrix_wo_k_targets.shape,'S matrix without k targets')
        S_matrix_df = pd.DataFrame(S_matrix)
        return S_matrix
    def grouping_physical_model_parameters(self,exp:list):
        final_groups=[]
        for i in exp['simulation'].fullParsedYamlFile['overallDict'].keys():
            if not re.match('[dD]iluent',i['type']):
                final_groups.append(i)
        
        
        
    def breakup_X(self, X, 
                        exp_dict_list:list, 
                        exp_uncertainty_dict_list_original:list,
                        loop_counter:int = 0,
                        master_equation_uncertainty_df=None,
                        master_equation_reactions = [],
                        master_equation_flag = False):


        '''
        Breaks up X vector into various  
        
        Arguments:
            exp_dic_list -- the dictionary that is built after a simulation
            that contains things like sensitivity coefficients
            exp_uncertainty_dict_list_original -- the dictonary that is built after the first
            set of simulations that contains all relevent information

        Keyword Arguments:
            loop_counter -- the iteration numbe rthe code is on
            mapped_master_equation_sensitivities -- numpy matrix of theory parameter sensitivity values (default empty array)
            master_equation_uncertainty_df -- a pandas dataframe that contains 
            the reactions being treated with theory paramters along with the 
            associated uncertainty values of those paramters (default None)
            master_equation_reactions -- a list of the reactions being treated
            with theory paramters (default [])
            master_equation_flag -- a boolean that indicates if reactions being
            represented by theory parameters are being used in the optimization (default False)
            
        '''


        
        
        X_to_subtract_from_Y = {}
        reactions_in_cti_file = exp_dict_list[0]['simulation'].processor.solution.reaction_equations()
        number_of_reactions = len(reactions_in_cti_file)
        

        if loop_counter !=0:
            X_new = X 
        
            
        else:
            X_new = X
      
        flatten = lambda *n: (e for a in n
            for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))        
        
        flattened_master_equation_reaction_list = list(flatten(master_equation_reactions))         
        
        X_new = list(X_new.flatten())            
        if exp_dict_list[0]['simulation'].kineticSens ==1:
            
            value1 = 3*(number_of_reactions - len(flattened_master_equation_reaction_list))
            
               
            AsNsEas = X_new[:value1]
            X_to_subtract_from_Y['As_ns_Eas'] = AsNsEas
            
            #### pickup here
            dividedBy = int(len(AsNsEas) / 3)
            
            def list_slice(S,step):
                return [S[i::step] for i in range(step)]
            
            resortedList = list_slice(AsNsEas,dividedBy)
            
            innerDict = ['A','n','Ea']
            l = [dict(zip(innerDict,resortedList[x])) for x in range(len(resortedList))]    
            Keys= []
            for xx in range(int(value1/3)):
                Keys.append('r'+str(xx))
                
                
            deltaXAsNsEas = dict(zip(Keys,l))
            
            
            innerDictNew = ['A_update','n_update','Ea_update']
            ll = [dict(zip(innerDictNew,resortedList[x])) for x in range(len(resortedList))]
            kinetic_paramter_dict = dict(zip(reactions_in_cti_file,ll))
        #molecularParams = np.array([.1,.2,.3,.4,.2,.3,.4]).flatten().tolist()
        # might need to fix this based on how lei is passing me information, check in notebook
        
        if master_equation_flag == True:
            # number_of_molecular_parameters_list = []
            # for col in master_equation_uncertainty_df:
            #     number_of_molecular_parameters_list.append(len(master_equation_uncertainty_df[col].dropna().values))
                
            
            number_of_molecular_parameters_list = []

            for i,reaction in enumerate(master_equation_reactions):
                if type(reaction)==str:
                    number_of_molecular_parameters_list.append(len(list(master_equation_uncertainty_df[reaction].dropna().values)))

                elif type(reaction)==tuple:
                    column_headers = master_equation_uncertainty_df.columns.to_list()
                    for sub_reaction in reaction:
                        if sub_reaction in column_headers:
                            number_of_molecular_parameters_list.append(len(list(master_equation_uncertainty_df[sub_reaction].dropna().values)))     



                
                
            
            sum_of_moleular_paramters = sum(number_of_molecular_parameters_list)
            value2 = sum_of_moleular_paramters     
            deltaXmolecularParams = X_new[value1:(value1+value2)]
            
            X_to_subtract_from_Y['molecular_parameters'] = deltaXmolecularParams
            molecular_paramters_by_reaction = []
            reaction_numbers = []
            start_mp = 0
            for r,number in enumerate(number_of_molecular_parameters_list):
                stop_mp = start_mp + number
                molecular_paramters_by_reaction.append(deltaXmolecularParams[start_mp:stop_mp])
                start_mp = stop_mp
                reaction_numbers.append('R_'+str(r))

            delta_x_molecular_params_by_reaction_dict = dict(zip(master_equation_reactions,molecular_paramters_by_reaction))
            list_of_mp = []
            for i,reaction in enumerate(molecular_paramters_by_reaction):
                temp=[]
                for j,value in enumerate(reaction):
                    temp.append('Paramter_'+str(j)+'_Update')
                list_of_mp.append(temp)
                
                
                
            inner_dict_temp = [dict(zip(list_of_mp[x],molecular_paramters_by_reaction[x])) for x in range(len(molecular_paramters_by_reaction))]
            inner_dict_temp_2 = dict(zip(master_equation_reactions,inner_dict_temp))
                    
            kinetic_paramter_dict.update(inner_dict_temp_2)
            #its possible this kinetic paramters dict might break
            
           
       
        else:
            value2 = 0
        
        
        physical_observables = []
        previous_value = 0
        physical_observables_for_Y = []
        if exp_dict_list[0]['simulation'].physicalSens ==1:
            for i,exp_dic in enumerate(exp_dict_list):
                if re.match('[Bb]atch[ _-][Rr]eactor',exp_dic['simulation_type']) and re.match('[Ss]pecies[- ][Pp]rofile',exp_dic['experiment_type']):
                    dic_of_conditions = exp_dic['simulation'].conditions
                        #subtract out the dilluant 
                    species_in_simulation = len(set(dic_of_conditions.keys()).difference(['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']))
                        #add two for Temperature and Pressure
                    len_of_phsycial_observables_in_simulation = species_in_simulation + 2 +1 
                    new_value = previous_value + len_of_phsycial_observables_in_simulation
                    single_experiment_physical_observables = X_new[(value1+value2+previous_value):(value1+value2+new_value)]
                    physical_observables_for_Y.append(single_experiment_physical_observables)
                    temp_keys = []
                        #stacking the zeros onto the Y array 
                    temp_keys.append('T'+'_'+'experiment'+'_'+str(i))
                    temp_keys.append('P'+'_'+'experiment'+'_'+str(i))
                    for variable in range(species_in_simulation):
                        temp_keys.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                    
                    temp_keys.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                    
                    temp_dict = dict(zip(temp_keys,single_experiment_physical_observables))
                    physical_observables.append(temp_dict)
                    ##come back to this and do a test on paper
                    previous_value = new_value
                    
                elif re.match('[Bb]atch[ _-][Rr]eactor',exp_dic['simulation_type']) and re.match('[Ii]gnition[- ][Dd]elay',exp_dic['experiment_type']):

                    diluent=[]   
                    if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                        diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                    
                    singular_species=[]
                    for species in list(exp_dic['simulation'].fullParsedYamlFile['conditions'].keys()):
                        
                        if len(exp_dic['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                            singular_species.append(species)
                    
                    
                    species_in_simulation = int(len(singular_species)+((len(exp_dic['simulation'].fullParsedYamlFile['speciesNames'])-len(singular_species))*len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])))
                    len_of_phsycial_observables_in_simulation = species_in_simulation + len(exp_dic['simulation'].pressures)+len(exp_dic['simulation'].temperatures)+1 
                    new_value = previous_value + len_of_phsycial_observables_in_simulation
                    single_experiment_physical_observables = X_new[(value1+value2+previous_value):(value1+value2+new_value)]
                    physical_observables_for_Y.append(single_experiment_physical_observables)
                    
                    
                    temp_keys = []
                        #stacking the zeros onto the Y array 
                    for j in range(len(exp_dic['simulation'].temperatures)):
                        temp_keys.append('T'+str(j+1)+'_'+'experiment'+'_'+str(i))
                        #stacking the zeros onto the Y array 
                    for j in range(len(exp_dic['simulation'].pressures)):
                        temp_keys.append('P'+str(j+1)+'_'+'experiment'+'_'+str(i))
                    for x,species in enumerate(exp_dic['simulation'].fullParsedYamlFile['speciesNames']):
                        if species in singular_species and species not in diluent:
                            temp_keys.append('X'+str(x+1)+'_cond'+str(0)+'_'+species+'_experiment_'+str(i))
                            
                        elif species not in singular_species and species not in diluent:
                            for j in range(len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])):
                                temp_keys.append('X'+str(x+1)+'_cond'+str(j)+'_'+species+'_experiment_'+str(i))
                    temp_keys.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                    temp_dict = dict(zip(temp_keys,single_experiment_physical_observables))
                    physical_observables.append(temp_dict)
                    ##come back to this and do a test on paper
                    previous_value = new_value
                    #print(temp_dict)
                    
                
                
                
                elif re.match('[Rc][Cc][Mm]',exp_dic['simulation_type']) and re.match('[Ii]gnition[- ][Dd]elay',exp_dic['experiment_type']):
                    
                    diluent=[]   
                    if 'Diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in exp_dic['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                        diluent = exp_dic['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                    
                    singular_species=[]
                    for species in list(exp_dic['simulation'].fullParsedYamlFile['conditions'].keys()):
                        
                        if len(exp_dic['simulation'].fullParsedYamlFile['conditions'][species])==1 and species not in diluent:
                            singular_species.append(species)
                    
                    
                    species_in_simulation = int(len(singular_species)+((len(exp_dic['simulation'].fullParsedYamlFile['speciesNames'])-len(singular_species))*len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])))
                    len_of_phsycial_observables_in_simulation = species_in_simulation + len(exp_dic['simulation'].pressures)+len(exp_dic['simulation'].temperatures)+1 
                    new_value = previous_value + len_of_phsycial_observables_in_simulation
                    single_experiment_physical_observables = X_new[(value1+value2+previous_value):(value1+value2+new_value)]
                    physical_observables_for_Y.append(single_experiment_physical_observables)
                    
                    
                    temp_keys = []
                        #stacking the zeros onto the Y array 
                    for j in range(len(exp_dic['simulation'].temperatures)):
                        temp_keys.append('T'+str(j+1)+'_'+'experiment'+'_'+str(i))
                        #stacking the zeros onto the Y array 
                    for j in range(len(exp_dic['simulation'].pressures)):
                        temp_keys.append('P'+str(j+1)+'_'+'experiment'+'_'+str(i))
                    for x,species in enumerate(exp_dic['simulation'].fullParsedYamlFile['speciesNames']):
                        if species in singular_species and species not in diluent:
                            temp_keys.append('X'+str(x+1)+'_cond'+str(0)+'_'+species+'_experiment_'+str(i))
                            
                        elif species not in singular_species and species not in diluent:
                            for j in range(len(exp_dic['simulation'].fullParsedYamlFile['conditions_to_run'])):
                                temp_keys.append('X'+str(x+1)+'_cond'+str(j)+'_'+species+'_experiment_'+str(i))
                    temp_keys.append('Time_shift'+'_'+'experiment'+'_'+str(i))
                    temp_dict = dict(zip(temp_keys,single_experiment_physical_observables))
                    physical_observables.append(temp_dict)
                    ##come back to this and do a test on paper
                    previous_value = new_value                    
                    
                    
                    
                    
                elif re.match('[Jj][Ss][Rr]',exp_dic['simulation_type']):
                    
                    dic_of_conditions = exp_dic['simulation'].conditions
                        #subtract out the dilluant 
                    species_in_simulation = len(set(dic_of_conditions.keys()).difference(['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']))
                        #add two for Temperature and Pressure
                    len_of_phsycial_observables_in_simulation = species_in_simulation + 1+len(exp_dic['simulation'].temperatures)+1 
                    #print(len_of_phsycial_observables_in_simulation)
                    new_value = previous_value + len_of_phsycial_observables_in_simulation
                    single_experiment_physical_observables = X_new[(value1+value2+previous_value):(value1+value2+new_value)]
                    #print(len(single_experiment_physical_observables))
                    physical_observables_for_Y.append(single_experiment_physical_observables)
                    temp_keys = []
                        #stacking the zeros onto the Y array 
                    for j in range(len(exp_dic['simulation'].temperatures)):
                        temp_keys.append('T'+str(j+1)+'_'+'experiment'+'_'+str(i))
                    temp_keys.append('P'+'_'+'experiment'+'_'+str(i))
                    for variable in range(species_in_simulation):
                        temp_keys.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                    temp_keys.append('R'+'_'+'experiment'+'_'+str(i))
                    temp_dict = dict(zip(temp_keys,single_experiment_physical_observables))
                    physical_observables.append(temp_dict)
                    ##come back to this and do a test on paper
                    previous_value = new_value
                    
                    
                    
                elif re.match('[Ss]pecies[- ][Pp]rofile',exp_dict_list[i]['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp_dict_list[i]['simulation_type']):
                    dic_of_conditions = exp_dic['simulation'].conditions
                        #subtract out the dilluant 
                    species_in_simulation = len(set(dic_of_conditions.keys()).difference(['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']))
                        #add two for Temperature and Pressure
                    time_shift_length = len(exp_dic['simulation'].fullParsedYamlFile['timeShiftOriginal'])
                    
                    len_of_phsycial_observables_in_simulation = species_in_simulation + 1+len(exp_dic['simulation'].temperatures)+time_shift_length 
                    #print(len_of_phsycial_observables_in_simulation)
                    new_value = previous_value + len_of_phsycial_observables_in_simulation
                    single_experiment_physical_observables = X_new[(value1+value2+previous_value):(value1+value2+new_value)]
                    #print(len(single_experiment_physical_observables))
                    physical_observables_for_Y.append(single_experiment_physical_observables)
                    temp_keys = []
                        #stacking the zeros onto the Y array 
                    for j in range(len(exp_dic['simulation'].temperatures)):
                        temp_keys.append('T'+str(j+1)+'_'+'experiment'+'_'+str(i))
                    temp_keys.append('P'+'_'+'experiment'+'_'+str(i))
                    
                    for variable in range(species_in_simulation):
                        temp_keys.append('X'+'_'+str(variable)+'_'+'experiment'+'_'+str(i))
                        
                    for j in range(time_shift_length):
                        temp_keys.append('Time_Shift'+str(j+1)+'_'+'experiment'+'_'+str(i))
                    
                    temp_dict = dict(zip(temp_keys,single_experiment_physical_observables))
                    physical_observables.append(temp_dict)
                    ##come back to this and do a test on paper
                    previous_value = new_value
        physical_observables_for_Y = [item for sublist in physical_observables_for_Y for item in sublist]   
        X_to_subtract_from_Y['physical_observables'] = physical_observables_for_Y
        
        
        test_abs = []
        absorbance_coefficients_for_Y = []
        coef_dict = {}  
        coef_dict_list = []
        absorbance_coef_update_dict = {}
        for i,exp_dic in enumerate(exp_uncertainty_dict_list_original):
            if 'coupled_coef_and_uncertainty' not in exp_dic.keys():
                continue
            dictionary_of_coef_and_uncertainty = exp_dic['coupled_coef_and_uncertainty']
            #tab start working here tomorrow, need to pass in the original version of this dict 
            #dictionary_of_coef_and_uncertainty = {(140000, 0.0): ([0.7], [0.0]), (1270000, 0.0): ([0.7], [0.0])}

            for x in dictionary_of_coef_and_uncertainty:
                if x not in coef_dict.keys():
                    coef_dict[x] = dictionary_of_coef_and_uncertainty[x]
                if x not in coef_dict_list:
                    coef_dict_list.append(x)
                    
                    
                    
        start_abs = 0
        stop_abs = 1         
        for i,cof in enumerate(coef_dict_list):
            temp=[]
            temp2=[]
         #   counter=1
            for value in cof:
                if value==0:
                    temp.append([0])
                    temp2.append(['null'])
                else:
                    temp.append(X_new[(value1+value2+new_value+start_abs):(value1+value2+new_value+stop_abs)])
                    temp2.append(X_new[(value1+value2+new_value+start_abs):(value1+value2+new_value+stop_abs)])
                    start_abs = stop_abs
                    stop_abs +=1
                    
            temp = [item for sublist in temp for item in sublist]     
            temp2 = [item for sublist in temp2 for item in sublist]     
            absorbance_coef_update_dict[cof] = temp
            absorbance_coefficients_for_Y.append(temp2)
            test_abs.append(temp2)
            
            

        # return everything in a dictionary??   
        absorbance_coefficients_for_Y = [item for sublist in absorbance_coefficients_for_Y for item in sublist] 
        X_to_subtract_from_Y['absorbance_coefficent_observables'] = absorbance_coefficients_for_Y
#       
        if master_equation_flag == False:
            return deltaXAsNsEas,physical_observables,absorbance_coef_update_dict,X_to_subtract_from_Y,kinetic_paramter_dict
        else:
            return deltaXAsNsEas,physical_observables,absorbance_coef_update_dict,X_to_subtract_from_Y,delta_x_molecular_params_by_reaction_dict,kinetic_paramter_dict
    
    
    def matrix_manipulation(self,runCounter,S_matrix,Y_matrix,z_matrix,XLastItteration = np.array(()),active_parameters=[]):

        '''
        Solve for the X and covariance matrix.   
        
        Arguments:
            runCounter -- the iteration number the code is on
            S_matrix -- S matrix
            Y_matrix -- Y matrix
            z_matrix -- z_matrix
            XLastItteration -- numpy array that contains the last x values
            loop_counter -- the iteration numbe rthe code is on
            active_parameters -- list of active parameters
            
        '''

        one_over_z = np.true_divide(1,z_matrix)
        #print(Y_matrix)
        y_matrix = Y_matrix * one_over_z
        
        s_matrix = S_matrix * (one_over_z.flatten()[:,np.newaxis])
        self.y_matrix = y_matrix
        
        sTimesZ = S_matrix * (z_matrix.flatten())[:,np.newaxis]
        #calculate covariance matrix 
        shape = np.shape(self.S_matrix_wo_k_targets)

        s_wo_k_targets = s_matrix[:shape[0],:shape[1]]
        identity_matrix = s_wo_k_targets[shape[0]-len(active_parameters):,:]

        
        
        #try:
        if runCounter==0:
              
                c = np.dot(np.transpose(identity_matrix),identity_matrix)
                c = np.linalg.inv(c)
                prior_diag = np.diag(c)
                prior_sigmas = np.sqrt(prior_diag)
                covariance_prior_df = pd.DataFrame(c)
                if active_parameters:
                    covariance_prior_df.columns = active_parameters
                    covariance_prior_df.reindex(labels = active_parameters)
                    prior_diag_df = pd.DataFrame({'parameter': active_parameters,'value': prior_diag.reshape((prior_diag.shape[0],))})
                    sorted_prior_diag = prior_diag_df.sort_values(by=['value'])
                    prior_sigmas_df = pd.DataFrame({'parameter': active_parameters,'value': prior_sigmas.reshape((prior_sigmas.shape[0],))})
        else:
                c = np.dot(np.transpose(s_matrix),s_matrix)
                c = np.linalg.inv(c)
                
                covariance_posterior_df = pd.DataFrame(c)
                if active_parameters:
                    covariance_posterior_df.columns = active_parameters
                    covariance_posterior_df.reindex(labels = active_parameters)
                    posterior_diag = np.diag(c)
                    posterior_sigmas = np.sqrt(posterior_diag)
                    posterior_sigmas_df = pd.DataFrame({'parameter': active_parameters,'value': posterior_sigmas.reshape((posterior_sigmas.shape[0],))})
                    posterior_diag_df =  pd.DataFrame({'parameter': active_parameters,'value': posterior_diag.reshape((posterior_diag.shape[0],))})
                    sorted_posterior_diag  = posterior_diag_df.sort_values(by=['value'])

        
        self.covariance = c


        
    
        self.s_matrix = s_matrix
    
        psudoInverse = np.linalg.pinv(s_matrix)
        delta_X = np.dot(psudoInverse,y_matrix)
        self.delta_X = delta_X
        
    

        if runCounter == 0:
            XlastItteration = np.zeros(np.shape(delta_X))

        else:     
            XlastItteration = XLastItteration

        X = XlastItteration + delta_X

        self.X = X
       
        #STUB THIS
        try:
            X_data_frame = pd.DataFrame({'value': active_parameters,'Parameter': X.reshape((X.shape[0],))})
        except:
            X_data_frame = -1
        if runCounter==0:
            return X,c,s_matrix,y_matrix,delta_X,z_matrix,X_data_frame,prior_diag,prior_diag_df,sorted_prior_diag,covariance_prior_df,prior_sigmas_df
        else:
            return X,c,s_matrix,y_matrix,delta_X,z_matrix,X_data_frame,posterior_diag,posterior_diag_df,sorted_posterior_diag,covariance_posterior_df,posterior_sigmas_df

class Adding_Target_Values(meq.Master_Equation):
    def __init__(self,S_matrix,Y_matrix,z_matrix,sigma,Y_data_Frame,z_data_Frame,T_P_min_max_dict={}):
        self.S_matrix = S_matrix
        self.Y_matrix = Y_matrix
        self.z_matrix = z_matrix
        self.sigma = sigma
        self.Y_data_Frame = Y_data_Frame
        self.z_data_Frame = z_data_Frame
        meq.Master_Equation.__init__(self)
        self.T_P_min_max_dict = T_P_min_max_dict
        
        '''
        Class to calculate and add k target values.   
        
        Arguments:
            S_matrix -- S matrix
            Y_matrix -- Y matrix
            z_matrix -- z_matrix
            sigma -- sigma
            Y_data_Frame -- Y_data_Frame
            z_data_Frame -- z_data_Frame
            XLastItteration -- numpy array that contains the last x values
            T_P_min_max_dict -- Dict that contains reactions and the max and min values for 
            chebyshev poly
            active_parameters -- list of active parameters
            
        '''        
        
         
        
    def target_values_Y(self,target_value_csv,
                        exp_dict_list:list,
                        Y_data_Frame,
                        master_equation_reactions):

        '''
        Function to add k target values to y matrix.   
        
        Arguments:
            target_value_csv -- csv that contains list of k target values 
            exp_dict_list -- list of experiment dic built after simulation run
            Y_matrix -- Y matrix
            master_equation_reactions -- list of master equation reactions
            
        '''   




        import cantera as ct
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
        
        
        def check_if_M_in_reactants(list_to_append_to,
                                    gas,
                                    reactants_in_target_reactions,
                                    reverse_reactants_in_target_reaction):
            
            if reverse_reactants_in_target_reaction !=None:
                for reaction_number_in_cti_file in range(gas.n_reactions):
                    if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'  or 
                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M') : 
                            list_to_append_to.append(reactions_in_cti_file[reaction_number_in_cti_file])
                            
            elif reverse_reactants_in_target_reaction==None:
                for reaction_number_in_cti_file in range(gas.n_reactions):
                    if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'): 
                            list_to_append_to.append(reactions_in_cti_file[reaction_number_in_cti_file])  
  
            return list_to_append_to
        
        
        for i,reaction in enumerate(target_reactions): 
                #ask about the mixture composition
            #if reaction not in flattened_linked_channel_reactions:
                
            if '*' not in reaction and reaction != 'More Complex Combination Rule' and '(+)' not in reaction:
                index_in_cti_file = gas.reaction_equations().index(reaction)
                units_reaction_types=['ElementaryReaction',
                                  'PlogReaction',
                                  'ChebyshevReaction',
                                  'ThreeBodyReaction',
                                  'FalloffReaction']
                coeff_sum = sum(gas.reaction(index_in_cti_file).reactants.values())
                
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
                
                if coeff_sum==1:
                    k = k
                elif coeff_sum==2:
                    k=k*1000
                elif coeff_sum==3:
                    k=k*1000000
                
                    #check and make sure we are subtracting in the correct order 
    
                difference = np.log(target_k[i]) - np.log(k)
    
                diff_in_ks_for_Y.append(difference)
                Y_df_list.append(reaction)
                Y_values.append(difference)
                
            #elif reaction in flattened_linked_channel_reactions:
            elif '*' in reaction and reaction != 'More Complex Combination Rule' and '/' not in reaction:
                
                reactions_in_cti_file_with_these_reactants = []
                #might be a more comprehensive way to do this 
                            
                                
                
                reactants_in_target_reactions = reaction.split('<=>')[0].rstrip()
                reverse_reactants_in_target_reaction=None
                if len(reactants_in_target_reactions.split('+'))>1:
                    reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                    temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                    temp = temp.lstrip()
                    temp = temp.rstrip()
                    reverse_reactants_in_target_reaction = temp
                
                for reaction_number_in_cti_file in range(gas.n_reactions):
                    if gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction:                        
                        reactions_in_cti_file_with_these_reactants.append(reactions_in_cti_file[reaction_number_in_cti_file])
                

                reactions_in_cti_file_with_these_reactants =  check_if_M_in_reactants(reactions_in_cti_file_with_these_reactants,
                                        gas,
                                        reactants_in_target_reactions,
                                        reactants_in_target_reactions)

                if target_press[i] == 0:
                    pressure = 1e-9
                else:
                    pressure = target_press[i]
                if bath_gas[i] !=0:
                    gas.TPX = target_temp[i],pressure*101325,{'H2O':.013,'O2':.0099,'H':.0000007,'Ar':.9770993}
    
                else:
                    gas.TPX = target_temp[i],pressure*101325,{'Ar':.99}
                    
                tottal_k = []  
                
                for secondary_reaction in reactions_in_cti_file_with_these_reactants:
                    reaction_number_in_cti = reactions_in_cti_file.index(secondary_reaction)
                    coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())
                    k = gas.forward_rate_constants[reaction_number_in_cti]
                    if coeff_sum==1:
                        k=k
                    elif coeff_sum==2:
                        k = k*1000
                    elif coeff_sum==3:
                        k= k*1000000
                    tottal_k.append(k)
                
                    #check and make sure we are subtracting in the correct order 
                k=sum(tottal_k)    
                difference = np.log(target_k[i]) - np.log(k)
    
                diff_in_ks_for_Y.append(difference)
                #I guess i could append the tuple 
                Y_df_list.append(reaction)
                Y_values.append(difference)                  

            elif  '/' in reaction:
                reactants_in_numerator = reaction.split('/')[0].rstrip()
                reactants_in_numerator = reactants_in_numerator.lstrip()
                
                reactants_in_denominator = reaction.split('/')[1].rstrip()
                reactants_in_denominator = reactants_in_denominator.lstrip()
                
                reactions_in_cti_file_with_these_reactants_numerator = []
                reactions_in_cti_file_with_these_reactants_denominator = []
                #take back here
                if '*' in reactants_in_numerator:
                    reactants_in_target_reactions_numerator = reactants_in_numerator.split('<=>')[0].rstrip()
                    reverse_reactants_in_target_reaction_in_numerator=None
                    if len(reactants_in_target_reactions_numerator.split('+'))>1:
                        reverse_reactants_in_target_reaction_in_numerator = reactants_in_target_reactions_numerator.split('+')
                        temp = reverse_reactants_in_target_reaction_in_numerator[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction_in_numerator[0]
                        temp = temp.lstrip()
                        temp = temp.rstrip()
                        reverse_reactants_in_target_reaction_in_numerator = temp
                    
                    for reaction_number_in_cti_file in range(gas.n_reactions):
                        if gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions_numerator or gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction_in_numerator:                        
                            reactions_in_cti_file_with_these_reactants_numerator.append(reactions_in_cti_file[reaction_number_in_cti_file])
                            
                            
                    reactions_in_cti_file_with_these_reactants_numerator =  check_if_M_in_reactants(reactions_in_cti_file_with_these_reactants_numerator,
                                        gas,
                                        reactants_in_target_reactions_numerator,
                                        reverse_reactants_in_target_reaction_in_numerator)         
                else:
                    #need to figure out how to split addition of reactions 
                    if '(+)' not in reactants_in_numerator:
                        reactions_in_cti_file_with_these_reactants_numerator.append(reactants_in_numerator)
                    else:
                        list_of_reactions_in_numerator = reactants_in_numerator.split('(+)')
                        list_of_reactions_in_numerator_cleaned=[]
                        for reaction in list_of_reactions_in_numerator:
                            reaction = reaction.rstrip()
                            reaction = reaction.lstrip()
                            list_of_reactions_in_numerator_cleaned.append(reaction)
                        
                        reactions_in_cti_file_with_these_reactants_numerator  =    list_of_reactions_in_numerator_cleaned                    
                    
                    

                if '*' in reactants_in_denominator:

                    reactants_in_target_reactions_denominator = reactants_in_denominator.split('<=>')[0].rstrip()
                    reverse_reactants_in_target_reaction_in_denominator=None
                    if len(reactants_in_target_reactions_denominator.split('+'))>1:
                        reverse_reactants_in_target_reaction_in_denominator = reactants_in_target_reactions_denominator.split('+')
                        temp = reverse_reactants_in_target_reaction_in_denominator[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction_in_denominator[0]
                        temp = temp.lstrip()
                        temp = temp.rstrip()
                        reverse_reactants_in_target_reaction_in_denominator = temp
                    
                    for reaction_number_in_cti_file in range(gas.n_reactions):
                        if gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions_denominator or gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction_in_denominator:                        
                            reactions_in_cti_file_with_these_reactants_denominator.append(reactions_in_cti_file[reaction_number_in_cti_file])
                    
                
                
                    reactions_in_cti_file_with_these_reactants_denominator =  check_if_M_in_reactants(reactions_in_cti_file_with_these_reactants_denominator,
                                        gas,
                                        reactants_in_target_reactions_denominator,
                                        reverse_reactants_in_target_reaction_in_denominator)
                
                
                
                else:
                    #need to figure out how to split addition of reactions 
                    if '(+)' not in reactants_in_denominator:
                        reactions_in_cti_file_with_these_reactants_denominator.append(reactants_in_denominator)
                    
                    else:
                        list_of_reactions_in_denominator = reactants_in_denominator.split('(+)')
                        list_of_reactions_in_denominator_cleaned=[]
                        for reaction in list_of_reactions_in_denominator:
                            reaction = reaction.rstrip()
                            reaction = reaction.lstrip()
                            list_of_reactions_in_denominator_cleaned.append(reaction)
                        
                        reactions_in_cti_file_with_these_reactants_denominator  =    list_of_reactions_in_denominator_cleaned
                            
                            

                
                if target_press[i] == 0:
                    pressure = 1e-9
                else:
                    pressure = target_press[i]
                if bath_gas[i] !=0:
                    gas.TPX = target_temp[i],pressure*101325,{'H2O':.013,'O2':.0099,'H':.0000007,'Ar':.9770993}
    
                else:
                    gas.TPX = target_temp[i],pressure*101325,{'Ar':.99}
                    
                tottal_k_numerator = []    
                for secondary_reaction in reactions_in_cti_file_with_these_reactants_numerator:
                    reaction_number_in_cti = reactions_in_cti_file.index(secondary_reaction)
                    coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())
                    k = gas.forward_rate_constants[reaction_number_in_cti]
                    
                    if coeff_sum==1:
                        k=k
                    elif coeff_sum==2:
                        k = k*1000
                    elif coeff_sum==3:
                        k = k*1000000
                    tottal_k_numerator.append(k)
                
                    #check and make sure we are subtracting in the correct order 
                k_numerator=sum(tottal_k_numerator)
                
                tottal_k_denominator = []    
                for secondary_reaction in reactions_in_cti_file_with_these_reactants_denominator:
                    reaction_number_in_cti = reactions_in_cti_file.index(secondary_reaction)
                    coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())
                    k = gas.forward_rate_constants[reaction_number_in_cti]
                    if coeff_sum==1:
                        k=k
                    elif coeff_sum==2:
                        k = k*1000
                    elif coeff_sum==3:
                        k = k*1000000
                    tottal_k_denominator.append(k)                
                
                k_denominator=sum(tottal_k_denominator)
                
                k = k_numerator/k_denominator
                difference = np.log(target_k[i]) - np.log(k)
                #print(k_numerator,k_denominator)
                ##print(target_k[i],k)
    
                diff_in_ks_for_Y.append(difference)
                #I guess i could append the tuple 
                Y_df_list.append(reaction)
                Y_values.append(difference)                 
                
                
                    
            elif  '(+)' in reaction and '/' not in reaction and '*' not in reaction: 
                list_of_reactions = reaction.split('(+)')
                list_of_reactions_cleaned=[]
                for reaction in list_of_reactions:
                    reaction = reaction.rstrip()
                    reaction = reaction.lstrip()
                    list_of_reactions_cleaned.append(reaction)
                        
                reactions_in_cti_file_with_these_reactants  =    list_of_reactions_cleaned                
                if target_press[i] == 0:
                    pressure = 1e-9
                else:
                    pressure = target_press[i]
                if bath_gas[i] !=0:
                    gas.TPX = target_temp[i],pressure*101325,{'H2O':.013,'O2':.0099,'H':.0000007,'Ar':.9770993}
    
                else:
                    gas.TPX = target_temp[i],pressure*101325,{'Ar':.99}
                    
                tottal_k = []  
                
                for secondary_reaction in reactions_in_cti_file_with_these_reactants:
                    reaction_number_in_cti = reactions_in_cti_file.index(secondary_reaction)
                    coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())
                    k = gas.forward_rate_constants[reaction_number_in_cti]
                    if coeff_sum==1:
                        k=k
                    elif coeff_sum==2:
                        k = k*1000
                    elif coeff_sum==3:
                        k= k*1000000
                    tottal_k.append(k)
                
                    #check and make sure we are subtracting in the correct order 
                k=sum(tottal_k)    
                difference = np.log(target_k[i]) - np.log(k)
    
                diff_in_ks_for_Y.append(difference)
                #I guess i could append the tuple 
                Y_df_list.append(reaction)
                Y_values.append(difference)  


                
                
            elif reaction == 'More Complex Combination Rule': 
                print('do someting else ')                   
                
                
                
                
                
            
        k_targets_for_y = np.array(diff_in_ks_for_Y)
        k_targets_for_y = k_targets_for_y.reshape((k_targets_for_y.shape[0],1))
        Y_values = np.array(Y_values)
        
        Y_df_temp = pd.DataFrame({'value': Y_df_list,'ln_difference': Y_values.reshape((Y_values.shape[0],))}) 
        Y_data_Frame = Y_data_Frame.append(Y_df_temp, ignore_index=True)
        #print(k_targets_for_y.shape,'k targets for y')
        

        
        return k_targets_for_y,Y_data_Frame
    
    def target_values_for_Z(self,target_value_csv,z_data_Frame):

        '''
        Function to add k target values to Z matrix.   
        
        Arguments:
            target_value_csv -- csv that contains list of k target values
            z_data_Frame -- Y z_data_Frame
            
        '''


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
    
    def target_values_for_S(self,target_value_csv,
                            exp_dict_list,
                            S_matrix,
                            master_equation_reaction_list = [],
                            master_equation_sensitivities = {}):
            '''
            Function to add k target values to S matrix.   
            
            Arguments:
            target_value_csv -- csv that contains list of k target values
            exp_dict_list -- list of dicts created after cantera simulations are run
            S_matrix -- S matrix
            master_equation_reaction_list -- list of master equation reactions
            master_equation_sensitivities -- dict of master equation sensitivities
            '''
            target_value_csv = pd.read_csv(target_value_csv)
            target_reactions = target_value_csv['Reaction']
            target_temp = target_value_csv['temperature']
            target_press = target_value_csv['pressure']
            target_k = target_value_csv['k']
            bath_gas = target_value_csv['M']
            reactions_in_cti_file = exp_dict_list[0]['simulation'].processor.solution.reaction_equations()
            number_of_reactions_in_cti = len(reactions_in_cti_file)
            gas = ct.Solution(exp_dict_list[0]['simulation'].processor.cti_path)
            As = []
            Ns =  []
            Eas = []
                
            flatten = lambda *n: (e for a in n
                for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))  
            flattened_master_equation_reaction_list = list(flatten(master_equation_reaction_list))
            
            coupled_reaction_list = []
            list_of_reaction_tuples = []
            for reaction in master_equation_reaction_list:
                if type(reaction)==tuple:
                    list_of_reaction_tuples.append(reaction)
                    for secondary_reaction in reaction:
                        coupled_reaction_list.append(secondary_reaction)
                        
                        
                        
            def reactants_in_master_equation_reactions(flattened_master_equation_reaction_list):
                reactants = []
                for me_reaction in flattened_master_equation_reaction_list:
                    reactants_in_master_equation_reaction = me_reaction.split('<=>')[0].rstrip()
                    reactants.append(reactants_in_master_equation_reaction)
    
                    if len(reactants_in_master_equation_reaction.split('+')) >1:
                        reverse_reactants_in_target_reaction = reactants_in_master_equation_reaction.split('+')
                        
                        temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                        temp = temp.lstrip()
                        temp = temp.rstrip()
                        reverse_reactants_in_target_reaction = temp
                        reactants.append(reverse_reactants_in_target_reaction)
                return reactants            
            
            master_equation_reactants_and_reverse_reactants = reactants_in_master_equation_reactions(flattened_master_equation_reaction_list)
            #print(master_equation_reactants_and_reverse_reactants)
            
            
            
            
            def calculate_weighting_factor_summation(rate_constant_list,gas,temperature,Press,bath_gas):
                if Press == 0:
                    pressure = 1e-9
                else:
                    pressure = Press
                    
                if bath_gas !=0:
                    gas.TPX = temperature,pressure*101325,{'H2O':.013,'O2':.0099,'H':.0000007,'Ar':.9770993}   
                else:
                    gas.TPX = temperature,pressure*101325,{'Ar':.99}
                
                tottal_k = []    
                original_rc_dict = {}
                for reaction in rate_constant_list:
                    reaction_number_in_cti = reactions_in_cti_file.index(reaction)
                    coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())

                    k = gas.forward_rate_constants[reaction_number_in_cti]
                    if coeff_sum==1:
                        k=k
                    elif coeff_sum==2:
                        k = k*1000
                    elif coeff_sum==3:
                        k = k*1000000
                    original_rc_dict[reaction] = k
                    tottal_k.append(k)
                    
                            
                                #check and make sure we are subtracting in the correct order 
                k_summation=sum(tottal_k)    
                
                weighting_factor_dict = {}
                for reaction in rate_constant_list:
                    weighting_factor_dict[reaction] = original_rc_dict[reaction] / k_summation
                    
                return weighting_factor_dict
            
            def calculate_weighting_factor_summation_with_denominator(numerator_rate_constant_list,denominator_rate_constant_list,gas,temperature,Press,bath_gas):
                if Press == 0:
                    pressure = 1e-9
                else:
                    pressure = Press
                    
                if bath_gas !=0:
                    gas.TPX = temperature,pressure*101325,{'H2O':.013,'O2':.0099,'H':.0000007,'Ar':.9770993}   
                else:
                    gas.TPX = temperature,pressure*101325,{'Ar':.99}
                
                tottal_k_numerator = []    
                original_rc_dict = {}
                for reaction in numerator_rate_constant_list:
                    reaction_number_in_cti = reactions_in_cti_file.index(reaction)
                    coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())

                    k = gas.forward_rate_constants[reaction_number_in_cti]
                    if coeff_sum==1:
                        k=k
                    elif coeff_sum==2:
                        k = k*1000
                    elif coeff_sum==3:
                        k = k*1000000
                    original_rc_dict[reaction] = k
                    tottal_k_numerator.append(k)
                    
                            
                                #check and make sure we are subtracting in the correct order 
                k_summation_numerator=sum(tottal_k_numerator)    
                
                weighting_factor_dict_numerator = {}
                for reaction in numerator_rate_constant_list:
                    weighting_factor_dict_numerator[reaction] = original_rc_dict[reaction] / k_summation_numerator

                tottal_k_denominator = []    
                original_rc_dict = {}
                for reaction in denominator_rate_constant_list:
                    reaction_number_in_cti = reactions_in_cti_file.index(reaction)
                    coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())

                    k = gas.forward_rate_constants[reaction_number_in_cti]
                    if coeff_sum==1:
                        k=k
                    elif coeff_sum==2:
                        k = k*1000
                    elif coeff_sum==3:
                        k = k*1000000
                    original_rc_dict[reaction] = k
                    tottal_k_denominator.append(k)
                    
                            
                                #check and make sure we are subtracting in the correct order 
                k_summation_denominator=sum(tottal_k_denominator)    
                
                weighting_factor_dict_denominator = {}
                for reaction in denominator_rate_constant_list:
                    weighting_factor_dict_denominator[reaction] = -(original_rc_dict[reaction] / k_summation_denominator)

                
                reactions_in_common = weighting_factor_dict_numerator.keys() &  weighting_factor_dict_denominator.keys()
                
                weighting_factor_dict = {}
                for reaction in reactions_in_common:
                    weighting_factor_dict[reaction] = weighting_factor_dict_numerator[reaction] + weighting_factor_dict_denominator[reaction]
                
                for reaction in weighting_factor_dict_numerator.keys():
                    if reaction in reactions_in_common:
                        pass
                    else:
                        weighting_factor_dict[reaction] = weighting_factor_dict_numerator[reaction]
                        
                for reaction in weighting_factor_dict_denominator.keys():
                    if reaction in reactions_in_common:
                        pass
                    else:
                        weighting_factor_dict[reaction] = weighting_factor_dict_denominator[reaction]


                    
                return weighting_factor_dict            
            
            
            def add_tuple_lists(nested_list,master_euqation_reactions_list):
                if any(isinstance(x, tuple) for x in master_euqation_reactions_list) == False:
                
                    return nested_list
                else:
                    all_tuple_summations = []
                    indexes_that_need_to_be_removed = []
                    indexes_to_replace_with = []
                    counter = 0
                    for i,reaction in enumerate(master_euqation_reactions_list):
                        if type(reaction) == str:
                            counter+=1
                            
                        elif type(reaction) == tuple:
                            tuple_sublist=[]
                            indexes_to_replace_with.append(counter)
                            for j,secondary_reaction in enumerate(reaction):
                                tuple_sublist.append(np.array(nested_list[counter])) 
                                if j!= 0:
                                    indexes_that_need_to_be_removed.append(counter)
                                counter+=1
                            sum_of_tupe_sublist = list(sum(tuple_sublist))
                            all_tuple_summations.append(sum_of_tupe_sublist)
                    
                    new_nested_list = copy.deepcopy(nested_list)
                    for i,replacment in enumerate(indexes_to_replace_with):  
                        new_nested_list[replacment] = all_tuple_summations[i]
                    
                    new_nested_list = [x for i,x in enumerate(new_nested_list) if i not in indexes_that_need_to_be_removed]
            
            
                    return new_nested_list            
            
            
            
            
            
            
            
            def create_empty_nested_reaction_list():
                
                
                nested_reaction_list = [[] for x in range(len(flattened_master_equation_reaction_list))]
                
                for reaction in flattened_master_equation_reaction_list:
                    for i,MP in enumerate(master_equation_sensitivities[reaction]):
                        nested_reaction_list[flattened_master_equation_reaction_list.index(reaction)].append(0)
                        
                return nested_reaction_list   
            
            

            
            
            def create_tuple_list(array_of_sensitivities):
                tuple_list = []
                for ix,iy in np.ndindex(array_of_sensitivities.shape):
                    tuple_list.append((ix,iy))
                return tuple_list
            
            def check_if_M_in_reactants(list_to_append_to,
                                        gas,
                                        reactants_in_target_reactions,
                                        reverse_reactants_in_target_reaction):
                if reverse_reactants_in_target_reaction !=None:
                    for reaction_number_in_cti_file in range(gas.n_reactions):
                        if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                            gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                            gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'  or 
                            gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M') : 
                                list_to_append_to.append(reactions_in_cti_file[reaction_number_in_cti_file])
                                
                elif reverse_reactants_in_target_reaction==None:
                    for reaction_number_in_cti_file in range(gas.n_reactions):
                        if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                            gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'): 
                                list_to_append_to.append(reactions_in_cti_file[reaction_number_in_cti_file])  
      
                return list_to_append_to
            
            
            
            
            def check_if_reaction_is_theory_or_not(reaction):
                
                is_reaction_in_master_equation_list = False
                is_reacton_in_normal_reaction_list = False
                if '/' in reaction:
                    #check numerator and denominator
                    reactants_in_numerator = reaction.split('/')[0].rstrip()
                    reactants_in_numerator = reactants_in_numerator.lstrip()
                    
                    reactants_in_denominator = reaction.split('/')[1].rstrip()
                    reactants_in_denominator = reactants_in_denominator.lstrip()

                    if '*' in reactants_in_numerator and '(+)' not in reactants_in_numerator:

                        reactions_in_numerator_with_these_reactants = []
                        #might be a more comprehensive way to do this 
                        reactants_in_target_reactions = reaction.split('<=>')[0].rstrip()
                        reverse_reactants_in_target_reaction=None
                        if len(reactants_in_target_reactions.split('+'))>1:
                            reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                            temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                            temp = temp.lstrip()
                            temp = temp.rstrip()
                            reverse_reactants_in_target_reaction = temp
                        
                        for reaction_number_in_cti_file in range(gas.n_reactions):
                            if gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction:                        
                                reactions_in_numerator_with_these_reactants.append(reactions_in_cti_file[reaction_number_in_cti_file]) 
                        
                        reactions_in_numerator_with_these_reactants =  check_if_M_in_reactants(reactions_in_numerator_with_these_reactants,
                                        gas,
                                        reactants_in_target_reactions,
                                        reverse_reactants_in_target_reaction)
                        
                        
                        
                        
                        



                    elif '(+)' in reactants_in_numerator and '*' not in reactants_in_numerator:
                        list_of_reactions_in_numerator = reactants_in_numerator.split('(+)')
                        list_of_reactions_in_numerator_cleaned=[]
                        for reaction in list_of_reactions_in_numerator:
                            reaction = reaction.rstrip()
                            reaction = reaction.lstrip()
                            list_of_reactions_in_numerator_cleaned.append(reaction)
                                
                        reactions_in_numerator_with_these_reactants  =    list_of_reactions_in_numerator_cleaned 




                    elif '(+)' in reactants_in_numerator and '*' in reactants_in_numerator:
                        print('need to make rule')
                    else:
                        reactions_in_numerator_with_these_reactants = []
                        reactions_in_numerator_with_these_reactants.append(reactants_in_numerator)

                    




                    #check reactants in numerator 
                    if '*' in reactants_in_denominator and '(+)' not in reactants_in_denominator:
                        reactions_in_denominator_with_these_reactants = []
                        #might be a more comprehensive way to do this 
                        reactants_in_target_reactions = reaction.split('<=>')[0].rstrip()
                        reverse_reactants_in_target_reaction=None
                        if len(reactants_in_target_reactions.split('+'))>1:
                            reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                            temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                            temp = temp.lstrip()
                            temp = temp.rstrip()
                            reverse_reactants_in_target_reaction = temp
                        
                        for reaction_number_in_cti_file in range(gas.n_reactions):
                            if gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction:                        
                                reactions_in_denominator_with_these_reactants.append(reactions_in_cti_file[reaction_number_in_cti_file]) 

                        reactions_in_denominator_with_these_reactants =  check_if_M_in_reactants(reactions_in_denominator_with_these_reactants,
                                                                gas,
                                                                reactants_in_target_reactions,
                                                                reverse_reactants_in_target_reaction)




                    elif '(+)' in reactants_in_denominator and '*' not in reactants_in_denominator:
                        list_of_reactions_in_denominator = reactants_in_numerator.split('(+)')
                        list_of_reactions_in_denominator_cleaned=[]
                        for reaction in list_of_reactions_in_denominator:
                            reaction = reaction.rstrip()
                            reaction = reaction.lstrip()
                            list_of_reactions_in_denominator_cleaned.append(reaction)
                                
                        reactions_in_denominator_with_these_reactants  =    list_of_reactions_in_numerator_cleaned 
                    

                    elif '(+)' in reactants_in_denominator and '*' in reactants_in_denominator:
                        print('need to make rule')
                    else:
                        reactions_in_denominator_with_these_reactants=[]
                        reactions_in_denominator_with_these_reactants.append(reactants_in_denominator)

                    
                    reactions_in_numerator_and_denominator = reactions_in_numerator_with_these_reactants+reactions_in_denominator_with_these_reactants
                    for reaction_check in reactions_in_numerator_and_denominator:
                        if reaction_check in flattened_master_equation_reaction_list:
                            is_reaction_in_master_equation_list = True
                        else:
                            is_reacton_in_normal_reaction_list = True

                    if is_reaction_in_master_equation_list == True and is_reacton_in_normal_reaction_list==False:
                        return 'master_equations_only', (reactions_in_numerator_with_these_reactants,reactions_in_denominator_with_these_reactants)
                    elif is_reaction_in_master_equation_list == False and is_reacton_in_normal_reaction_list==True:
                        return 'not_master_equations_only', (reactions_in_numerator_with_these_reactants,reactions_in_denominator_with_these_reactants)
                    elif is_reaction_in_master_equation_list == True and is_reacton_in_normal_reaction_list==True:
                        return 'mixed', (reactions_in_numerator_with_these_reactants,reactions_in_denominator_with_these_reactants)



                elif '(+)' in reaction and '/' not in reaction and '*' not in reaction:
                    list_of_reactions = reaction.split('(+)')
                    list_of_reactions_cleaned=[]
                    for reaction in list_of_reactions:
                        reaction = reaction.rstrip()
                        reaction = reaction.lstrip()
                        list_of_reactions_cleaned.append(reaction)
                                
                    reactions_in_cti_file_with_these_reactants  =    list_of_reactions_cleaned

                    for reaction_check in reactions_in_cti_file_with_these_reactants:
                        if reaction_check in flattened_master_equation_reaction_list:
                            is_reaction_in_master_equation_list = True
                        else:
                            is_reacton_in_normal_reaction_list = True


                    if is_reaction_in_master_equation_list == True and is_reacton_in_normal_reaction_list==False:
                        return 'master_equations_only', (reactions_in_cti_file_with_these_reactants,)
                    elif is_reaction_in_master_equation_list == False and is_reacton_in_normal_reaction_list==True:
                        return 'not_master_equations_only', (reactions_in_cti_file_with_these_reactants,)
                    elif is_reaction_in_master_equation_list == True and is_reacton_in_normal_reaction_list==True:
                        return 'mixed', (reactions_in_cti_file_with_these_reactants,)



                elif '*' in reaction and '/' not in reaction and '(+)' not in reaction:

                    reactions_in_cti_file_with_these_reactants = []
                        #might be a more comprehensive way to do this 
                    reactants_in_target_reactions = reaction.split('<=>')[0].rstrip()
                    reverse_reactants_in_target_reaction=None
                    if len(reactants_in_target_reactions.split('+'))>1:
                        reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                        temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                        temp = temp.lstrip()
                        temp = temp.rstrip()
                        reverse_reactants_in_target_reaction = temp
                    
                    for reaction_number_in_cti_file in range(gas.n_reactions):
                        if gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction:                        
                            reactions_in_cti_file_with_these_reactants.append(reactions_in_cti_file[reaction_number_in_cti_file]) 
                            
                    reactions_in_cti_file_with_these_reactants =  check_if_M_in_reactants(reactions_in_cti_file_with_these_reactants,
                                                                gas,
                                                                reactants_in_target_reactions,
                                                                reverse_reactants_in_target_reaction)
                    

                    for reaction_check in reactions_in_cti_file_with_these_reactants:
                        if reaction_check in flattened_master_equation_reaction_list:
                            is_reaction_in_master_equation_list = True
                        else:
                            is_reacton_in_normal_reaction_list = True


                    if is_reaction_in_master_equation_list == True and is_reacton_in_normal_reaction_list==False:
                        return 'master_equations_only', (reactions_in_cti_file_with_these_reactants,)
                    elif is_reaction_in_master_equation_list == False and is_reacton_in_normal_reaction_list==True:
                        return 'not_master_equations_only', (reactions_in_cti_file_with_these_reactants,)
                    elif is_reaction_in_master_equation_list == True and is_reacton_in_normal_reaction_list==True:
                        return 'mixed', (reactions_in_cti_file_with_these_reactants,)



                else:
                    #normal reaction 
                    reactions_in_cti_file_with_these_reactants=[]
                    for reaction_check in [reaction]:
                        if reaction_check in flattened_master_equation_reaction_list:
                            is_reaction_in_master_equation_list = True
                        else:
                            is_reacton_in_normal_reaction_list = True

                    reactions_in_cti_file_with_these_reactants.append(reaction)
                    if is_reaction_in_master_equation_list == True and is_reacton_in_normal_reaction_list==False:
                        return 'master_equations_only', (reactions_in_cti_file_with_these_reactants,)
                    elif is_reaction_in_master_equation_list == False and is_reacton_in_normal_reaction_list==True:
                        return 'not_master_equations_only', (reactions_in_cti_file_with_these_reactants,)
                    elif is_reaction_in_master_equation_list == True and is_reacton_in_normal_reaction_list==True:
                        return 'mixed', (reactions_in_cti_file_with_these_reactants,)




            MP_stack = []
            target_values_to_stack =  []
            for i,reaction in enumerate(target_reactions):
                type_of_reaction, reaction_tuple = check_if_reaction_is_theory_or_not(reaction)


                if type_of_reaction== 'master_equations_only':
                    
                    if len(reaction_tuple)==1:
                        if len(reaction_tuple[0])==1:
                            
                            

                            nested_reaction_list = create_empty_nested_reaction_list()
                            for j, MP_array in enumerate(master_equation_sensitivities[reaction]):
                                tuple_list = create_tuple_list(MP_array)
                                temp = []
                                counter = 0    
                                for sensitivity in np.nditer(MP_array,order='C'):
                                    k = tuple_list[counter][0]
                                    l= tuple_list[counter][1]
                                    counter +=1
                                       #need to add reduced p and t, and check these units were using to map
                                        
                                    #these might not work
                                    
                                    t_alpha= meq.Master_Equation.chebyshev_specific_poly(self,k,meq.Master_Equation.calc_reduced_T(self,target_temp[i],reaction,self.T_P_min_max_dict))
                                    
                                    if target_press[i] ==0:
                                        target_press_new = 1e-9
                                    else:
                                        target_press_new=target_press[i]
                                    p_alpha = meq.Master_Equation.chebyshev_specific_poly(self,l,meq.Master_Equation.calc_reduced_P(self,target_press_new*101325,reaction,self.T_P_min_max_dict))
                                    #these might nowt work 
                                    single_alpha_map = t_alpha*p_alpha*sensitivity
                                    temp.append(single_alpha_map)
                                temp =sum(temp)
                                #should there be an = temp here 
                                #nested_reaction_list[master_equation_reaction_list.index(reaction)][j]=temp
                                nested_reaction_list[flattened_master_equation_reaction_list.index(reaction)][j]=temp
                                
                            
                            temp2  = nested_reaction_list
                            
                            temp2_summed = add_tuple_lists(temp2,master_equation_reaction_list)
                            
                            flat_list = [item for sublist in temp2_summed for item in sublist]
                            #print(flat_list)
                            MP_stack.append(nested_reaction_list)
                            flat_list = np.array(flat_list)
                            flat_list = flat_list.reshape((1,flat_list.shape[0])) 
                            target_values_to_stack.append(flat_list)

                        elif len(reaction_tuple[0])>1:
                            reactions_in_cti_file_with_these_reactants = reaction_tuple[0]
                            weighting_factor_dictionary = calculate_weighting_factor_summation(reactions_in_cti_file_with_these_reactants,
                                                                                          gas,
                                                                                          target_temp[i],
                                                                                          target_press[i],
                                                                                          bath_gas[i])
                            nested_reaction_list = create_empty_nested_reaction_list()
                        
                            for secondary_reaction in reactions_in_cti_file_with_these_reactants:
                                for j, MP_array in enumerate(master_equation_sensitivities[secondary_reaction]):
                                    tuple_list = create_tuple_list(MP_array)
                                    temp = []
                                    counter = 0    
                                    for sensitivity in np.nditer(MP_array,order='C'):
                                        k = tuple_list[counter][0]
                                        l= tuple_list[counter][1]
                                        counter +=1
                                           #need to add reduced p and t, and check these units were using to map
                                            
                                        #these might not work
                                        
                                        t_alpha= meq.Master_Equation.chebyshev_specific_poly(self,k,meq.Master_Equation.calc_reduced_T(self,target_temp[i],secondary_reaction,self.T_P_min_max_dict))
                                        
                                        if target_press[i] ==0:
                                            target_press_new = 1e-9
                                        else:
                                            target_press_new=target_press[i]
                                        p_alpha = meq.Master_Equation.chebyshev_specific_poly(self,l,meq.Master_Equation.calc_reduced_P(self,target_press_new*101325,secondary_reaction,self.T_P_min_max_dict))
                                        #these might nowt work 
                                        single_alpha_map = t_alpha*p_alpha*sensitivity
                                        temp.append(single_alpha_map)
                                    temp =sum(temp)
                                    nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)][j]=temp
                                    
                                sub_array_to_apply_weighting_factor_to = list(np.array(nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)])*weighting_factor_dictionary[secondary_reaction])
                                nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)] = sub_array_to_apply_weighting_factor_to
                                
                                
                            temp2  = nested_reaction_list     
                            #print('THIS IS TEMP:',temp2)
                            temp2_summed = add_tuple_lists(temp2,master_equation_reaction_list)
                            #print('THIS IS TEMP SUMMED:',temp2_summed)
                            flat_list = [item for sublist in temp2_summed for item in sublist]
                            #print(flat_list)
                            MP_stack.append(nested_reaction_list)
                            flat_list = np.array(flat_list)
                            flat_list = flat_list.reshape((1,flat_list.shape[0])) 
                            target_values_to_stack.append(flat_list)


                    elif len(reaction_tuple)==2:
                        reactions_in_cti_file_with_these_reactants_numerator = reaction_tuple[0]
                        reactions_in_cti_file_with_these_reactants_denominator= reaction_tuple[1]

                        weighting_factor_dictionary = calculate_weighting_factor_summation_with_denominator(reactions_in_cti_file_with_these_reactants_numerator,
                                                                                          reactions_in_cti_file_with_these_reactants_denominator,                 
                                                                                          gas,
                                                                                          target_temp[i],
                                                                                          target_press[i],
                                                                                          bath_gas[i])

                        #now need to add to S matrix 
                        for secondary_reaction in (reactions_in_cti_file_with_these_reactants_numerator+reactions_in_cti_file_with_these_reactants_denominator):
                            for j, MP_array in enumerate(master_equation_sensitivities[secondary_reaction]):
                                tuple_list = create_tuple_list(MP_array)
                                temp = []
                                counter = 0    
                                for sensitivity in np.nditer(MP_array,order='C'):
                                    k = tuple_list[counter][0]
                                    l= tuple_list[counter][1]
                                    counter +=1
                                       #need to add reduced p and t, and check these units were using to map
                                        
                                    #these might not work
                                    
                                    t_alpha= meq.Master_Equation.chebyshev_specific_poly(self,k,meq.Master_Equation.calc_reduced_T(self,target_temp[i],secondary_reaction,self.T_P_min_max_dict))
                                    
                                    if target_press[i] ==0:
                                        target_press_new = 1e-9
                                    else:
                                        target_press_new=target_press[i]
                                    p_alpha = meq.Master_Equation.chebyshev_specific_poly(self,l,meq.Master_Equation.calc_reduced_P(self,target_press_new*101325,secondary_reaction,self.T_P_min_max_dict))
                                    #these might nowt work 
                                    single_alpha_map = t_alpha*p_alpha*sensitivity
                                    temp.append(single_alpha_map)
                                temp =sum(temp)
                                nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)][j]=temp
                                
                            sub_array_to_apply_weighting_factor_to = list(np.array(nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)])*weighting_factor_dictionary[secondary_reaction])
                            nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)] = sub_array_to_apply_weighting_factor_to
                            
                            
                        temp2  = nested_reaction_list     
                        #print('THIS IS TEMP:',temp2)
                        temp2_summed = add_tuple_lists(temp2,master_equation_reaction_list)
                        #print('THIS IS TEMP SUMMED:',temp2_summed)
                        flat_list = [item for sublist in temp2_summed for item in sublist]
                        #print(flat_list)
                        MP_stack.append(nested_reaction_list)
                        flat_list = np.array(flat_list)
                        flat_list = flat_list.reshape((1,flat_list.shape[0])) 
                        target_values_to_stack.append(flat_list)                        




       
                elif type_of_reaction== 'not_master_equations_only':
                    if len(reaction_tuple)==1:
                        if len(reaction_tuple[0])==1:        

                            A_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
            
                            N_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
                            Ea_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
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
                        
                    


                        elif len(reaction_tuple[0])>1:
                            reactions_in_cti_file_with_these_reactants = reaction_tuple[0]


                            weighting_factor_dictionary = calculate_weighting_factor_summation(reactions_in_cti_file_with_these_reactants,
                                                                                              gas,
                                                                                              target_temp[i],
                                                                                              target_press[i],
                                                                                              bath_gas[i])
                            
                            A_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))        
                            N_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
                            Ea_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
                            
                            for secondary_reaction in reactions_in_cti_file_with_these_reactants:
                                #need to multiply by the weighting factor for the reaction
                                
                                A_temp[0,reactions_in_cti_file.index(secondary_reaction)] = 1 * weighting_factor_dictionary[secondary_reaction]
                                N_temp [0,reactions_in_cti_file.index(secondary_reaction)] = np.log(target_temp[i]) * weighting_factor_dictionary[secondary_reaction]
                                Ea_temp[0,reactions_in_cti_file.index(secondary_reaction)] = (-1/target_temp[i]) * weighting_factor_dictionary[secondary_reaction]
                            As.append(A_temp)
                            Ns.append(N_temp)
                            Eas.append(Ea_temp)
                            A_temp = A_temp.reshape((1,A_temp.shape[1]))
                            N_temp = N_temp.reshape((1,N_temp.shape[1]))
                            Ea_temp = Ea_temp.reshape((1,Ea_temp.shape[1]))
                            target_values_to_stack.append(np.hstack((A_temp,N_temp,Ea_temp)))   

                    elif len(reaction_tuple)==2:

                        reactions_in_cti_file_with_these_reactants_numerator = reaction_tuple[0]
                        reactions_in_cti_file_with_these_reactants_denominator= reaction_tuple[1]
                        weighting_factor_dictionary = calculate_weighting_factor_summation_with_denominator(reactions_in_cti_file_with_these_reactants_numerator,
                                                                                          reactions_in_cti_file_with_these_reactants_denominator,                 
                                                                                          gas,
                                                                                          target_temp[i],
                                                                                          target_press[i],
                                                                                          bath_gas[i])
                        
                        A_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))        
                        N_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
                        Ea_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
                        
                        for secondary_reaction in (reactions_in_cti_file_with_these_reactants_numerator+reactions_in_cti_file_with_these_reactants_denominator):
                            
                            if reaction not in flattened_master_equation_reaction_list:
                                A_temp[0,reactions_in_cti_file.index(secondary_reaction)] = 1 * weighting_factor_dictionary[secondary_reaction]
                                N_temp [0,reactions_in_cti_file.index(secondary_reaction)] = np.log(target_temp[i]) * weighting_factor_dictionary[secondary_reaction]
                                Ea_temp[0,reactions_in_cti_file.index(secondary_reaction)] = (-1/target_temp[i]) * weighting_factor_dictionary[secondary_reaction]
                                
                        
                        As.append(A_temp)
                        Ns.append(N_temp)
                        Eas.append(Ea_temp)
                        A_temp = A_temp.reshape((1,A_temp.shape[1]))
                        N_temp = N_temp.reshape((1,N_temp.shape[1]))
                        Ea_temp = Ea_temp.reshape((1,Ea_temp.shape[1]))
                        target_values_to_stack.append(np.hstack((A_temp,N_temp,Ea_temp))) 



                elif type_of_reaction== 'mixed':
                    #need to figure out what is going in here
                    
                    
                    if len(reaction_tuple) == 1:
                        
                        reactions_in_cti_file_with_these_reactants = reaction_tuple[0]
                        weighting_factor_dictionary = calculate_weighting_factor_summation(reactions_in_cti_file_with_these_reactants,
                                                                                            gas,
                                                                                            target_temp[i],
                                                                                            target_press[i],
                                                                                            bath_gas[i])
                        #fill in respective lists and figure out what to do with them?
                        

                        A_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))        
                        N_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
                        Ea_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))

                        nested_reaction_list = create_empty_nested_reaction_list()

                        for secondary_reaction in reactions_in_cti_file_with_these_reactants:

                            if secondary_reaction not in flattened_master_equation_reaction_list:
                                A_temp[0,reactions_in_cti_file.index(secondary_reaction)] = 1 * weighting_factor_dictionary[secondary_reaction]
                                N_temp [0,reactions_in_cti_file.index(secondary_reaction)] = np.log(target_temp[i]) * weighting_factor_dictionary[secondary_reaction]
                                Ea_temp[0,reactions_in_cti_file.index(secondary_reaction)] = (-1/target_temp[i]) * weighting_factor_dictionary[secondary_reaction]
                  
                            
                            elif secondary_reaction in flattened_master_equation_reaction_list:
                                for j, MP_array in enumerate(master_equation_sensitivities[secondary_reaction]):
                                    tuple_list = create_tuple_list(MP_array)
                                    temp = []
                                    counter = 0    
                                    for sensitivity in np.nditer(MP_array,order='C'):
                                        k = tuple_list[counter][0]
                                        l= tuple_list[counter][1]
                                        counter +=1
                                           #need to add reduced p and t, and check these units were using to map
                                            
                                        #these might not work
                                        
                                        t_alpha= meq.Master_Equation.chebyshev_specific_poly(self,k,meq.Master_Equation.calc_reduced_T(self,target_temp[i],secondary_reaction,self.T_P_min_max_dict))
                                        
                                        if target_press[i] ==0:
                                            target_press_new = 1e-9
                                        else:
                                            target_press_new=target_press[i]
                                        p_alpha = meq.Master_Equation.chebyshev_specific_poly(self,l,meq.Master_Equation.calc_reduced_P(self,target_press_new*101325,secondary_reaction,self.T_P_min_max_dict))
                                        #these might nowt work 
                                        single_alpha_map = t_alpha*p_alpha*sensitivity
                                        temp.append(single_alpha_map)
                                    temp =sum(temp)
                                    nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)][j]=temp
                                    
                                sub_array_to_apply_weighting_factor_to = list(np.array(nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)])*weighting_factor_dictionary[secondary_reaction])
                                nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)] = sub_array_to_apply_weighting_factor_to
                                
                                
                        temp2  = nested_reaction_list     
                        temp2_summed = add_tuple_lists(temp2,master_equation_reaction_list)
                        flat_list = [item for sublist in temp2_summed for item in sublist]
                        MP_stack.append(nested_reaction_list)
                        flat_list = np.array(flat_list)
                        flat_list = flat_list.reshape((1,flat_list.shape[0])) 
                            
                        master_equation_stacked = flat_list


                        As.append(A_temp)
                        Ns.append(N_temp)
                        Eas.append(Ea_temp)
                        A_temp = A_temp.reshape((1,A_temp.shape[1]))
                        N_temp = N_temp.reshape((1,N_temp.shape[1]))
                        Ea_temp = Ea_temp.reshape((1,Ea_temp.shape[1]))
                        A_n_Ea_stacked = (np.hstack((A_temp,N_temp,Ea_temp))) 

                        combined_master_and_A_n_Ea= np.hstack((A_n_Ea_stacked,master_equation_stacked))
                        target_values_to_stack.append(combined_master_and_A_n_Ea)




                    elif len(reaction_tuple) == 2:
                        reactions_in_cti_file_with_these_reactants_numerator = reaction_tuple[0]
                        reactions_in_cti_file_with_these_reactants_denominator = reaction_tuple[1]

                        weighting_factor_dictionary = calculate_weighting_factor_summation_with_denominator(reactions_in_cti_file_with_these_reactants_numerator,
                                                                                          reactions_in_cti_file_with_these_reactants_denominator,                 
                                                                                          gas,
                                                                                          target_temp[i],
                                                                                          target_press[i],
                                                                                          bath_gas[i])
                        #fill in respective lists and figure out what to do with them?

                        A_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))        
                        N_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
                        Ea_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))

                        nested_reaction_list = create_empty_nested_reaction_list()

                        for secondary_reaction in (reactions_in_cti_file_with_these_reactants_numerator+reactions_in_cti_file_with_these_reactants_denominator):

                            if secondary_reaction not in flattened_master_equation_reaction_list:
                                A_temp[0,reactions_in_cti_file.index(secondary_reaction)] = 1 * weighting_factor_dictionary[secondary_reaction]
                                N_temp [0,reactions_in_cti_file.index(secondary_reaction)] = np.log(target_temp[i]) * weighting_factor_dictionary[secondary_reaction]
                                Ea_temp[0,reactions_in_cti_file.index(secondary_reaction)] = (-1/target_temp[i]) * weighting_factor_dictionary[secondary_reaction]
                  
                            
                            elif secondary_reaction in flattened_master_equation_reaction_list:
                                for j, MP_array in enumerate(master_equation_sensitivities[secondary_reaction]):
                                    tuple_list = create_tuple_list(MP_array)
                                    temp = []
                                    counter = 0    
                                    for sensitivity in np.nditer(MP_array,order='C'):
                                        k = tuple_list[counter][0]
                                        l= tuple_list[counter][1]
                                        counter +=1
                                           #need to add reduced p and t, and check these units were using to map
                                            
                                        #these might not work
                                        
                                        t_alpha= meq.Master_Equation.chebyshev_specific_poly(self,k,meq.Master_Equation.calc_reduced_T(self,target_temp[i],secondary_reaction,self.T_P_min_max_dict))
                                        
                                        if target_press[i] ==0:
                                            target_press_new = 1e-9
                                        else:
                                            target_press_new=target_press[i]
                                        p_alpha = meq.Master_Equation.chebyshev_specific_poly(self,l,meq.Master_Equation.calc_reduced_P(self,target_press_new*101325,secondary_reaction,self.T_P_min_max_dict))
                                        #these might nowt work 
                                        single_alpha_map = t_alpha*p_alpha*sensitivity
                                        temp.append(single_alpha_map)
                                    temp =sum(temp)
                                    nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)][j]=temp
                                    
                                sub_array_to_apply_weighting_factor_to = list(np.array(nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)])*weighting_factor_dictionary[secondary_reaction])
                                nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)] = sub_array_to_apply_weighting_factor_to
                                
                                
                        temp2  = nested_reaction_list     
                        temp2_summed = add_tuple_lists(temp2,master_equation_reaction_list)
                        flat_list = [item for sublist in temp2_summed for item in sublist]
                        MP_stack.append(nested_reaction_list)
                        flat_list = np.array(flat_list)
                        flat_list = flat_list.reshape((1,flat_list.shape[0])) 
                            
                        master_equation_stacked = flat_list


                        As.append(A_temp)
                        Ns.append(N_temp)
                        Eas.append(Ea_temp)
                        A_temp = A_temp.reshape((1,A_temp.shape[1]))
                        N_temp = N_temp.reshape((1,N_temp.shape[1]))
                        Ea_temp = Ea_temp.reshape((1,Ea_temp.shape[1]))
                        A_n_Ea_stacked = (np.hstack((A_temp,N_temp,Ea_temp))) 

                        combined_master_and_A_n_Ea= np.hstack((A_n_Ea_stacked,master_equation_stacked))
                        target_values_to_stack.append(combined_master_and_A_n_Ea)





                        
                    
                    
            S_matrix = S_matrix
            shape_s = S_matrix.shape
            S_target_values = []
            #print(target_values_to_stack,target_values_to_stack[0].shape)
            #this whole part needs to be edited
            for i,row in enumerate(target_values_to_stack):
                type_of_reaction, reaction_tuple = check_if_reaction_is_theory_or_not(target_reactions[i])

                if type_of_reaction=='master_equations_only':
                    #zero_to_append_infront = np.zeros((1,((number_of_reactions_in_cti-len(master_equation_reaction_list))*3)))
                    zero_to_append_infront = np.zeros((1,((number_of_reactions_in_cti-len(flattened_master_equation_reaction_list))*3)))

                    zero_to_append_behind = np.zeros((1, shape_s[1] - ((number_of_reactions_in_cti-len(flattened_master_equation_reaction_list))*3) - np.shape(row)[1] ))                
                    temp_array = np.hstack((zero_to_append_infront,row,zero_to_append_behind))
                    S_target_values.append(temp_array)
                elif type_of_reaction=='not_master_equations_only':
                    zero_to_append_behind = np.zeros((1,shape_s[1]-np.shape(row)[1]))
                    temp_array = np.hstack((row,zero_to_append_behind))
                    S_target_values.append(temp_array)
                elif type_of_reaction=='mixed':
                    zero_to_append_behind = np.zeros((1,shape_s[1]-np.shape(row)[1]))
                    temp_array = np.hstack((row,zero_to_append_behind))
                    S_target_values.append(temp_array)                   


            S_target_values = np.vstack((S_target_values))
            
            return S_target_values
        


    def preprocessing_rate_constant_target_csv(self,target_value_csv,
                                               master_equation_reactions):

            #split up the master equations into multiple data frames 
            
            master_equation_df_list = []
            master_equation_df_sorted_list = []
            df_summation_list = []
            for reaction in master_equation_reactions:
                if type(reaction) == tuple:
                    master_equation_df_list.append([])
                    master_equation_df_sorted_list.append([])            
                    
            df_ttl = pd.read_csv(target_value_csv)
            counter = 0 
            for reaction in master_equation_reactions:
                if type(reaction) == tuple:
                    for secondary_reaction in reaction:                
                        temp = df_ttl.loc[df_ttl['Reaction'] == secondary_reaction]
                        if not temp.empty:
                            master_equation_df_list[counter].append(temp)
                    counter +=1
            
            for i,lst in enumerate(master_equation_df_list):
                for j,df in enumerate(lst):
                    df = df.sort_values(["temperature", "pressure"], ascending = (True, True))
                    master_equation_df_sorted_list[i].append(df)
                    
            for i,lst in enumerate(master_equation_df_sorted_list):
                df_summation = pd.DataFrame()
                df_summation['Reaction'] = lst[0]['Reaction']
                df_summation['temperature'] = lst[0]['temperature']
                df_summation['pressure'] = lst[0]['pressure']
                df_summation['M'] = lst[0]['M']
                df_summation['ln_unc_k'] = lst[0]['ln_unc_k']
                df_summation['W'] = lst[0]['W']  
                k_summation_list=[]
                for j,df in enumerate(lst):
                    k_summation_list.append(df['k'].to_numpy())
                df_summation['k'] = sum(k_summation_list)
                df_summation_list.append(df_summation)
        
            reactions_to_remove = []
            for reaction in master_equation_reactions:
                if type(reaction) == tuple:
                    for secondary_reaction in reaction:
                        reactions_to_remove.append(secondary_reaction)
            
            df_cleaned =   df_ttl[~df_ttl['Reaction'].isin(reactions_to_remove)]
            
            df_concat_list = [df_cleaned]+ df_summation_list
            df_new_tottal = pd.concat(df_concat_list)
            new_file_name = target_value_csv[:-4]
            new_file_path = new_file_name+'_combined_channels.csv'
            df_new_tottal.to_csv(new_file_path,
                                 index=False)
                 
        
            return df_new_tottal,new_file_path
    
    
    
    def appending_target_values(self,target_values_for_z,
                                target_values_for_Y,
                                target_values_for_S,
                                sigma_target_values,
                                S_matrix,
                                Y_matrix,
                                z_matrix,
                                sigma):
                                
        z_matrix = np.vstack((z_matrix ,target_values_for_z))
        Y_matrix = np.vstack((Y_matrix,target_values_for_Y))
        
        S_matrix = np.vstack((S_matrix,target_values_for_S))
        sigma = np.vstack((sigma,sigma_target_values))
        
        self.S_matrix = S_matrix
        self.Y_matrix = Y_matrix
        self.z_matrix = z_matrix
        self.sigma = sigma
        
        return S_matrix,Y_matrix,z_matrix,sigma
    

                
       
            
            
                        
                        
                        
               
            
                
                
            
            
        
        
        
              
