import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cantera as ct
import copy
from textwrap import wrap

class post_processing(object):
    def __init__(self,optimized_cti_file='',
                 original_cti_file='',
                 kinetic_paramter_dictonary={},
                 master_equation_reactions=[],
                 six_parameter_fit_nominal_parameters_dict={},
                 six_parameter_fit_optimized_paramter_dict={},
                 exp_dict_list_optimized = [],
                 exp_dict_list_original = [],
                 parsed_yaml_list = []):
        
        
        self.optimized_cti_file = optimized_cti_file
        self.original_cti_file = original_cti_file
        self.kinetic_paramter_dictonary = kinetic_paramter_dictonary
        self.master_equation_reactions = master_equation_reactions
        self.six_parameter_fit_nominal_parameters_dict=six_parameter_fit_nominal_parameters_dict
        self.six_parameter_fit_optimized_paramter_dict=six_parameter_fit_optimized_paramter_dict
        self.exp_dict_list_optimized = exp_dict_list_optimized
        self.exp_dict_list_original = exp_dict_list_original
        self.parsed_yaml_list = parsed_yaml_list
        
    def create_active_kinetic_paramter_dictonary(self):

        OriginalModel = ct.Solution(self.original_cti_file)
        reaction_equations_original = OriginalModel.reaction_equations()
        NewModel = ct.Solution(self.optimized_cti_file)
        reaction_equations_new = NewModel.reaction_equations()
        six_parameter_fit_nominal_parameters_dict = copy.deepcopy(self.six_parameter_fit_nominal_parameters_dict)   
        six_parameter_fit_optimized_paramter_dict = copy.deepcopy(self.six_parameter_fit_optimized_paramter_dict)
        for j, reaction in enumerate(reaction_equations_original):
           if reaction not in self.master_equation_reactions:
               if 'ThreeBodyReaction' in str(type(OriginalModel.reaction(reaction_equations_original.index(reaction)))):

                   
                   A_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).rate.pre_exponential_factor
                   n_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).rate.temperature_exponent
                   Ea_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_original':A_original, 'n_original':n_original,'Ea_original':Ea_original})
               elif 'ElementaryReaction' in str(type(OriginalModel.reaction(reaction_equations_original.index(reaction)))):
                   
                   A_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).rate.pre_exponential_factor
                   n_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).rate.temperature_exponent
                   Ea_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_original':A_original, 'n_original':n_original,'Ea_original':Ea_original})
               elif 'FalloffReaction' in str(type(OriginalModel.reaction(reaction_equations_original.index(reaction)))):
                  
                   A_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).high_rate.pre_exponential_factor
                   n_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).high_rate.temperature_exponent
                   Ea_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).high_rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_original_high_rate':A_original, 'n_original_high_rate':n_original,'Ea_original_high_rate':Ea_original})
                   
                   A_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).low_rate.pre_exponential_factor
                   n_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).low_rate.temperature_exponent
                   Ea_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).low_rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_original_low_rate':A_original, 'n_original_low_rate':n_original,'Ea_original_low_rate':Ea_original})

                   if OriginalModel.reaction(reaction_equations_original.index(reaction)).falloff.type=='Troe':
                       Troe = OriginalModel.reaction(reaction_equations_original.index(reaction)).falloff
                       self.kinetic_paramter_dictonary[reaction].update({'Troe_original':Troe})
                   if OriginalModel.reaction(reaction_equations_original.index(reaction)).falloff.type=='Sri':
                       Sri = OriginalModel.reaction(reaction_equations_original.index(reaction)).falloff
                       self.kinetic_paramter_dictonary[reaction].update({'Sri_original':Sri})
               elif 'ChemicallyActivatedReaction' in str(type(OriginalModel.reaction(reaction_equations_original.index(reaction)))):
                   A_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).high_rate.pre_exponential_factor
                   n_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).high_rate.temperature_exponent
                   Ea_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).high_rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_original_high_rate':A_original, 'n_original_high_rate':n_original,'Ea_original_high_rate':Ea_original})

                   A_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).low_rate.pre_exponential_factor
                   n_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).low_rate.temperature_exponent
                   Ea_original=OriginalModel.reaction(reaction_equations_original.index(reaction)).low_rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_original_low_rate':A_original, 'n_original_low_rate':n_original,'Ea_original_low_rate':Ea_original})

                   if OriginalModel.reaction(reaction_equations_original.index(reaction)).falloff.type=='Troe':
                       Troe = OriginalModel.reaction(reaction_equations_original.index(reaction)).falloff
                       self.kinetic_paramter_dictonary[reaction].update({'Troe_original':Troe})

                   if OriginalModel.OriginalModel(reaction_equations_original.index(reaction)).falloff.type=='Sri':
                       Sri = OriginalModel.reaction(reaction_equations_original.index(reaction)).falloff
                       self.kinetic_paramter_dictonary[reaction].update({'Sri_original':Sri})

               elif 'PlogReaction' in str(type(OriginalModel.reaction(reaction_equations_original.index(reaction)))):
                   for number, reactions in enumerate(OriginalModel.reaction(reaction_equations_original.index(reaction)).rates):
                       A_original = OriginalModel.reaction(reaction_equations_original.index(reaction))[number][1].pre_exponential_factor
                       n_original = OriginalModel.reaction(reaction_equations_original.index(reaction))[number][1].temperature_exponent
                       Ea_original = OriginalModel.reaction(reaction_equations_original.index(reaction))[number][1].activation_energy
                       self.kinetic_paramter_dictonary[reaction].update({'A_original_'+str(number):A_original, 'n_original_'+str(number):n_original,'Ea_original_'+str(number):Ea_original})

               elif 'ChebyshevReaction' in str(type(OriginalModel.reaction(reaction_equations_original.index(reaction)))):
                   T_min = OriginalModel.reaction(reaction_equations_original.index(reaction)).Tmin
                   T_max = OriginalModel.reaction(reaction_equations_original.index(reaction)).Tmax
                   P_min = OriginalModel.reaction(reaction_equations_original.index(reaction)).Pmin
                   P_max = OriginalModel.reaction(reaction_equations_original.index(reaction)).Pmax
                   coeffs = OriginalModel.reaction(reaction_equations_original.index(reaction)).coeffs
                   self.kinetic_paramter_dictonary[reaction].update({'T_min_original':T_min,'T_max_original':T_max,'P_min_original':P_min,'P_max_original':P_max,'Coeffs_original':coeffs})
                   
           else:
                              
               six_parameter_fit_nominal_parameters_dict[reaction] = {k+'_original': v for k, v in six_parameter_fit_nominal_parameters_dict[reaction].items()}
               
               self.kinetic_paramter_dictonary[reaction].update(six_parameter_fit_nominal_parameters_dict[reaction])

#start here tomorrow and do the updated model paramters! 
        
        
        for j, reaction in enumerate(reaction_equations_new):
           if reaction not in self.master_equation_reactions:
               if 'ThreeBodyReaction' in str(type(NewModel.reaction(reaction_equations_new.index(reaction)))):

                   
                   A_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).rate.pre_exponential_factor
                   n_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).rate.temperature_exponent
                   Ea_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_optimized':A_optimized, 'n_optimized':n_optimized,'Ea_optimized':Ea_optimized})
               elif 'ElementaryReaction' in str(type(NewModel.reaction(reaction_equations_new.index(reaction)))):
                   
                   A_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).rate.pre_exponential_factor
                   n_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).rate.temperature_exponent
                   Ea_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_optimized':A_optimized, 'n_optimized':n_optimized,'Ea_optimized':Ea_optimized})
                   
               elif 'FalloffReaction' in str(type(NewModel.reaction(reaction_equations_new.index(reaction)))):
                  
                   A_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).high_rate.pre_exponential_factor
                   n_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).high_rate.temperature_exponent
                   Ea_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).high_rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_optimized_high_rate':A_optimized, 'n_optimized_high_rate':n_optimized,'Ea_optimized_high_rate':Ea_optimized})
                   
                   A_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).low_rate.pre_exponential_factor
                   n_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).low_rate.temperature_exponent
                   Ea_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).low_rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_optimized_low_rate':A_optimized, 'n_optimized_low_rate':n_optimized,'Ea_optimized_low_rate':Ea_optimized})

                   if NewModel.reaction(reaction_equations_new.index(reaction)).falloff.type=='Troe':
                       Troe = NewModel.reaction(reaction_equations_new.index(reaction)).falloff
                       self.kinetic_paramter_dictonary[reaction].update({'Troe_optimized':Troe})
                   if NewModel.reaction(reaction_equations_new.index(reaction)).falloff.type=='Sri':
                       Sri = OriginalModel.reaction(reaction_equations_new.index(reaction)).falloff
                       self.kinetic_paramter_dictonary[reaction].update({'Sri_optimized':Sri})
                       
                       
               elif 'ChemicallyActivatedReaction' in str(type(NewModel.reaction(reaction_equations_new.index(reaction)))):
                   A_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).high_rate.pre_exponential_factor
                   n_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).high_rate.temperature_exponent
                   Ea_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).high_rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_optimized_high_rate':A_optimized, 'n_optimized_high_rate':n_optimized,'Ea_optimized_high_rate':Ea_optimized})

                   A_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).low_rate.pre_exponential_factor
                   n_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).low_rate.temperature_exponent
                   Ea_optimized=NewModel.reaction(reaction_equations_new.index(reaction)).low_rate.activation_energy
                   self.kinetic_paramter_dictonary[reaction].update({'A_optimized_low_rate':A_optimized, 'n_optimized_low_rate':n_optimized,'Ea_optimized_low_rate':Ea_optimized})

                   if NewModel.reaction(reaction_equations_new.index(reaction)).falloff.type=='Troe':
                       Troe = NewModel.reaction(reaction_equations_new.index(reaction)).falloff
                       self.kinetic_paramter_dictonary[reaction].update({'Troe_optimized':Troe})

                   if NewModel.reaction(reaction_equations_new.index(reaction)).falloff.type=='Sri':
                       Sri = NewModel.reaction(reaction_equations_new.index(reaction)).falloff
                       self.kinetic_paramter_dictonary[reaction].update({'Sri_optimized':Sri})

               elif 'PlogReaction' in str(type(NewModel.reaction(reaction_equations_new.index(reaction)))):
                   for number, reactions in enumerate(NewModel.reaction(reaction_equations_new.index(reaction)).rates):
                       A_optimized = NewModel.reaction(reaction_equations_new.index(reaction))[number][1].pre_exponential_factor
                       n_optimized = NewModel.reaction(reaction_equations_new.index(reaction))[number][1].temperature_exponent
                       Ea_optimized = NewModel.reaction(reaction_equations_new.index(reaction))[number][1].activation_energy
                       self.kinetic_paramter_dictonary[reaction].update({'A_optimized_'+str(number):A_optimized, 'n_optimized_'+str(number):n_optimized,'Ea_optimized_'+str(number):Ea_optimized})

               elif 'ChebyshevReaction' in str(type(NewModel.reaction(reaction_equations_new.index(reaction)))):
                   T_min = NewModel.reaction(reaction_equations_new.index(reaction)).Tmin
                   T_max = NewModel.reaction(reaction_equations_new.index(reaction)).Tmax
                   P_min = NewModel.reaction(reaction_equations_new.index(reaction)).Pmin
                   P_max = NewModel.reaction(reaction_equations_new.index(reaction)).Pmax
                   coeffs = NewModel.reaction(reaction_equations_new.index(reaction)).coeffs
                   self.kinetic_paramter_dictonary[reaction].update({'T_min_optimized':T_min,'T_max_optimized':T_max,'P_min_optimized':P_min,'P_max_optimized':P_max,'Coeffs_optimized':coeffs})
                   
           else:
               
               six_parameter_fit_optimized_paramter_dict[reaction] = {k+'_optimized': v for k, v in six_parameter_fit_optimized_paramter_dict[reaction].items()}
               
               self.kinetic_paramter_dictonary[reaction].update(six_parameter_fit_optimized_paramter_dict[reaction])
#               
#               for key in self.six_parameter_fit_optimized_paramter_dict[reaction]:
#                   self.six_parameter_fit_optimized_paramter_dict[reaction][key[:-10]] = self.six_parameter_fit_optimized_paramter_dict[reaction].pop(key)

        return self.kinetic_paramter_dictonary
    def create_active_physical_paramter_dictonary(self):
        
        active_physical_param_dict = {}
        for i,exp in enumerate(self.exp_dict_list_optimized):
            temp_exp_dict= {}
            temp_exp_dict.update({'Temp_optimized':exp['simulation'].temperature,'Temp_original':self.exp_dict_list_original[i]['simulation'].temperature})
            temp_exp_dict.update({'Pres_optimized':exp['simulation'].pressure,'Pres_original':self.exp_dict_list_original[i]['simulation'].pressure})
            temp_exp_dict.update({'Spec_optimized':exp['simulation'].conditions,'Spec_original':self.exp_dict_list_original[i]['simulation'].conditions})
            temp_exp_dict.update({'Exp_data':exp['experimental_data']})
            temp_exp_dict.update({'Observables':exp['simulation'].observables})

         
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list[i]['absorbanceCsvWavelengths']
                plt.figure()
                for k,wl in enumerate(wavelengths):
                    temp_exp_dict.update({'Temp_optimized':exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)]})
                    temp_exp_dict.update({'Absob_coeff_optimized':exp['simulation'].observables,'Absob_coeff_original':self.exp_dict_list_original[i]['simulation'].observables})
                    
                    
        active_physical_param_dict['Experiment_'+str(i+1)+'_T:'+ str(self.exp_dict_list_original[i]['simulation'].temperature)+'P_:'+ str(self.exp_dict_list_original[i]['simulation'].pressure)] = temp_exp_dict
        self.active_physical_param_dict = active_physical_param_dict
        
        return self.active_physical_param_dict

