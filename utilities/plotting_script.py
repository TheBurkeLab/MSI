import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cantera as ct
import copy
from textwrap import wrap
import scipy.stats as stats
import math
from scipy.stats import multivariate_normal
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import MSI.master_equation.master_equation as meq 
import re
import os
import MSI.simulations.instruments.ignition_delay as ig
import MSI.cti_core.cti_processor as pr
import MSI.simulations.instruments.jsr_steadystate as jsr
import glob
from pypdf import PdfMerger
import natsort
import shutil
import enlighten

class Plotting(object):
    def __init__(self,
                 
                  Ydf_prior,                                          
                  Sdf_prior,
                  covdf_prior,
                  sigdf_prior,

                  Xdf,      
                  Ydf,
                  ydf,
                  Zdf,                                      
                  Sdf,
                  sdf,
                  covdf,
                  sigdf,

                  Xdf_list,
                  active_parameters,
                  target_parameters,  
                                    
                  parsed_yaml_list_original,
                  exp_dict_list_original,
                  parsed_yaml_list_optimized,
                  exp_dict_list_optimized,
                  
                  working_directory='',
                  number_of_iterations = int,
                  files_to_include = [],

                  real_uncertainty_csv = '',
                  
                  target_value_rate_constant_csv='',
                  rate_constant_plots_csv = '',
                  k_target_value_S_matrix = np.array([]),

                  master_equation_flag = False,
                  theory_parameters_df = pd.DataFrame(),  
                  T_P_min_max_dict = {},
                  master_equation_reactions = [],
                  cheby_sensitivity_dict = None,                  
                  
                  
                  original_cti_file='',
                  optimized_cti_file='',
                  
                #   mapped_to_alpha_full_simulation=[],
                  sigma_ones=False,
                  simulation_run=None,
                  shock_tube_instance = None,                  

                  pdf = True,
                  png = True,
                  svg = True,
                  dpi = 1000
                ):
                  

        self.Ydf_prior = Ydf_prior
        self.Sdf_prior = Sdf_prior
        self.S_matrix_original = self.Sdf_prior.to_numpy() 
        self.covdf_prior = covdf_prior
        self.covariance_original = self.covdf_prior.to_numpy() 
        self.sigdf_prior = sigdf_prior
        self.sigma_original = self.sigdf_prior.to_numpy()         

        self.Xdf = Xdf    
        self.X = self.Xdf.to_numpy()   
        self.Ydf = Ydf
        self.Y_matrix = self.Ydf.to_numpy() 
        self.ydf= ydf
        self.y = self.ydf.to_numpy() 
        self.Zdf = Zdf   
        self.Z_matrix = self.Zdf.to_numpy()  
        self.Sdf = Sdf
        self.S_matrix = self.Sdf.to_numpy()
        self.sdf = sdf
        self.s_matrix = self.sdf.to_numpy()
        self.covdf = covdf  
        self.covariance = self.covdf.to_numpy()       
        self.sigdf = sigdf  
        self.sigma = self.sigdf.to_numpy()            

        self.Xdf_list = Xdf_list
        self.active_parameters = active_parameters
        self.target_parameters = target_parameters
    
        self.parsed_yaml_list_original = parsed_yaml_list_original
        self.exp_dict_list_original = exp_dict_list_original
        self.parsed_yaml_list_optimized = parsed_yaml_list_optimized
        self.exp_dict_list_optimized = exp_dict_list_optimized
        
        self.working_directory = working_directory
        self.out_path = os.path.join(working_directory, 'out')
        self.matrix_path = os.path.join(working_directory, 'matrix')        
        self.number_of_iterations = number_of_iterations
        self.files_to_include = files_to_include,
        # self.T_P_min_max_dict = T_P_min_max_dict
        # self.master_equation_reactions = master_equation_reactions
        # self.cheby_sensitivity_dict=cheby_sensitivity_dict
        
        # if bool(target_value_rate_constant_csv):
        self.target_value_rate_constant_csv = target_value_rate_constant_csv
        # if bool(target_value_rate_constant_csv_extra_values):
        self.rate_constant_plots_csv = rate_constant_plots_csv

        self.master_equation_flag = master_equation_flag
        if bool(self.master_equation_flag):
            self.master_equation_reactions = master_equation_reactions
            self.cheby_sensitivity_dict = cheby_sensitivity_dict
            self.T_P_min_max_dict=T_P_min_max_dict
        else:
            self.master_equation_reactions = [],
            self.cheby_sensitivity_dict = {},
            self.T_P_min_max_dict={},


        self.real_uncertainty_csv = real_uncertainty_csv
        
        self.k_target_value_S_matrix = k_target_value_S_matrix

        self.new_cti=optimized_cti_file
        self.nominal_cti=original_cti_file
        
        # self.mapped_to_alpha_full_simulation = mapped_to_alpha_full_simulation        
        self.sigma_ones = sigma_ones
        self.simulation_run = simulation_run
        self.shock_tube_instance = shock_tube_instance        

        self.pdf = pdf
        self.png = png
        self.svg = svg
        self.dpi = dpi
        
        self.lengths_of_experimental_data()
        
        self.manager = enlighten.get_manager()


    def lengths_of_experimental_data(self):
        simulation_lengths_of_experimental_data = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
           
            length_of_experimental_data=[]
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables'] + exp['ignition_delay_observables']):
                if observable == None:
                    continue
                if observable in exp['mole_fraction_observables']:
                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']):
                        length_of_experimental_data.append(exp['experimental_data'][observable_counter]['Time'].shape[0])
                        observable_counter+=1
                    elif re.match('[Jj][Ss][Rr]',exp['simulation_type']):
                        length_of_experimental_data.append(exp['experimental_data'][observable_counter]['Temperature'].shape[0])
                        observable_counter+=1
                    elif re.match('[Ss]pecies[- ][Pp]rofile',exp['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp['simulation_type']):
                        length_of_experimental_data.append(exp['experimental_data'][observable_counter]['Temperature'].shape[0])
                        observable_counter+=1                        
                if observable in exp['concentration_observables']:
                    
                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']):
                        length_of_experimental_data.append(exp['experimental_data'][observable_counter]['Time'].shape[0])
                        observable_counter+=1
                    elif re.match('[Jj][Ss][Rr]',exp['simulation_type']):
                        length_of_experimental_data.append(exp['experimental_data'][observable_counter]['Temperature'].shape[0])
                        observable_counter+=1
                    elif re.match('[Ss]pecies[- ][Pp]rofile',exp['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp['simulation_type']):
                        
                        length_of_experimental_data.append(exp['experimental_data'][observable_counter]['Temperature'].shape[0])
                        observable_counter+=1                         
                if observable in exp['ignition_delay_observables']:
                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']) and re.match('[iI]gnition[- ][Dd]elay',exp['experiment_type']):
                        if 'temperature' in list(exp['experimental_data'][observable_counter].columns):
                            length_of_experimental_data.append(exp['experimental_data'][observable_counter]['temperature'].shape[0])
                            observable_counter+=1
                        elif 'pressure' in list(exp['experimental_data'][observable_counter].columns):
                            length_of_experimental_data.append(exp['experimental_data'][observable_counter]['pressure'].shape[0])
                            observable_counter+=1
                        else:
                            length_of_experimental_data.append(exp['experimental_data'][observable_counter].shape[0])
                            observable_counter+=1
                    elif re.match('[Rr][Cc][Mm]',exp['simulation_type']) and re.match('[iI]gnition[- ][Dd]elay',exp['experiment_type']):
                        if 'temperature' in list(exp['experimental_data'][observable_counter].columns):
                            length_of_experimental_data.append(exp['experimental_data'][observable_counter]['temperature'].shape[0])
                            observable_counter+=1
                        elif 'pressure' in list(exp['experimental_data'][observable_counter].columns):
                            length_of_experimental_data.append(exp['experimental_data'][observable_counter]['pressure'].shape[0])
                            observable_counter+=1
                        else:
                            length_of_experimental_data.append(exp['experimental_data'][observable_counter].shape[0])
                            observable_counter+=1
            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                absorbance_wl=0
                for k,wl in enumerate(wavelengths):
                    length_of_experimental_data.append(exp['absorbance_experimental_data'][k]['time'].shape[0])
                    absorbance_wl+=1
            else:
                absorbance_wl=0
                    
            simulation_lengths_of_experimental_data.append(length_of_experimental_data)
            
                    
        self.simulation_lengths_of_experimental_data=simulation_lengths_of_experimental_data
        
        return observable_counter+absorbance_wl,length_of_experimental_data
                    

        
    def calculating_sigmas(self,S_matrix,covariance):  
        sigmas =[[] for x in range(len(self.simulation_lengths_of_experimental_data))]
        
                 
        counter=0
        for x in range(len(self.simulation_lengths_of_experimental_data)):
            for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                temp=[]
                for z in np.arange(counter,(self.simulation_lengths_of_experimental_data[x][y]+counter)):       
                    SC = np.dot(S_matrix[z,:],covariance)
                    sigma = np.dot(SC,np.transpose(S_matrix[z,:]))
                    test = sigma
                    sigma = np.sqrt(sigma)
                    temp.append(sigma)

                temp = np.array(temp)            
                sigmas[x].append(temp)
                    
                
                counter = counter + self.simulation_lengths_of_experimental_data[x][y]
                
        
        return sigmas, test
    
    def run_ignition_delay(self,exp,cti,n_of_data_points=10):
        
        p=pr.Processor(cti)
        if 'volumeTraceCsv' not in exp['simulation'].fullParsedYamlFile.keys():
            if len(exp['simulation'].fullParsedYamlFile['temperatures'])>1:
                tempmin=np.min(exp['simulation'].fullParsedYamlFile['temperatures'])
                print(tempmin , 'THis is the min temp')
                tempmax=np.max(exp['simulation'].fullParsedYamlFile['temperatures'])
                print(tempmax,'This is the max temp')
                total_range=tempmax-tempmin
                tempmax=tempmax+0.1*total_range
                tempmin=tempmin-0.1*total_range
                temprange=np.linspace(tempmin,tempmax,n_of_data_points)
                pressures=exp['simulation'].fullParsedYamlFile['pressures']
                print(pressures,'These are the pressures')
                conds=exp['simulation'].fullParsedYamlFile['conditions_to_run']
                print(conds,'These are the conditions')
                
            elif len(exp['simulation'].fullParsedYamlFile['pressures'])>1:
                pmin = exp['simulation'].fullParsedYamlFile['pressures']*0.9
                pmax = exp['simulation'].fullParsedYamlFile['pressures']*1.1
                total_range=pmax-pmin
                pmax=pmax+0.1*total_range
                pmin=pmin-0.1*total_range
                pressures = np.linspace(pmin,pmax,n_of_data_points)
                temprange = exp['simulation'].fullParsedYamlFile['temperatures']
                conds = exp['simulation'].fullParsedYamlFile['conditions_to_run']
                
            elif len(exp['simulation'].fullParsedYamlFile['conditions_to_run'])>1:
                print('Plotting for conditions depedendent ignition delay not yet installed')
                
                
                
            
            ig_delay=ig.ignition_delay_wrapper(pressures=pressures,
                                               temperatures=temprange,
                                               observables=exp['simulation'].fullParsedYamlFile['observables'],
                                               kineticSens=0,
                                               physicalSens=0,
                                               conditions=conds,
                                               thermalBoundary=exp['simulation'].fullParsedYamlFile['thermalBoundary'],
                                               mechanicalBoundary=exp['simulation'].fullParsedYamlFile['mechanicalBoundary'],
                                               processor=p,
                                               cti_path="", 
                                               save_physSensHistories=0,
                                               fullParsedYamlFile=exp['simulation'].fullParsedYamlFile, 
                                               save_timeHistories=0,
                                               log_file=True,
                                               log_name='log.txt',
                                               timeshift=exp['simulation'].fullParsedYamlFile['time_shift'],
                                               initialTime=exp['simulation'].fullParsedYamlFile['initialTime'],
                                               finalTime=exp['simulation'].fullParsedYamlFile['finalTime'],
                                               target=exp['simulation'].fullParsedYamlFile['target'],
                                               target_type=exp['simulation'].fullParsedYamlFile['target_type'],
                                               n_processors=2)
            soln,temp=ig_delay.run()
        elif 'volumeTraceCsv' in exp['simulation'].fullParsedYamlFile.keys():
            if len(exp['simulation'].fullParsedYamlFile['temperatures'])>1:
                tempmin=np.min(exp['simulation'].fullParsedYamlFile['temperatures'])
                tempmax=np.max(exp['simulation'].fullParsedYamlFile['temperatures'])
                total_range=tempmax-tempmin
                tempmax=tempmax+0.1*total_range
                tempmin=tempmin-0.1*total_range
                temprange=np.linspace(tempmin,tempmax,n_of_data_points)
                pressures=exp['simulation'].fullParsedYamlFile['pressures']
                conds=exp['simulation'].fullParsedYamlFile['conditions_to_run']
                volumeTrace = exp['simulation'].fullParsedYamlFile['volumeTraceCsv']
                
            elif len(exp['simulation'].fullParsedYamlFile['pressures'])>1:
                pmin = exp['simulation'].fullParsedYamlFile['pressures']*0.9
                pmax = exp['simulation'].fullParsedYamlFile['pressures']*1.1
                total_range=pmax-pmin
                pmax=pmax+0.1*total_range
                pmin=pmin-0.1*total_range
                pressures = np.linspace(pmin,pmax,n_of_data_points)
                temprange = exp['simulation'].fullParsedYamlFile['temperatures']
                conds = exp['simulation'].fullParsedYamlFile['conditions_to_run']
                volumeTrace = exp['simulation'].fullParsedYamlFile['volumeTraceCsv']
                
            elif len(exp['simulation'].fullParsedYamlFile['conditions_to_run'])>1:
                print('Plotting for conditions depedendent ignition delay not yet installed')
                
                
                
            
            ig_delay=ig.ignition_delay_wrapper(pressures=pressures,
                                               temperatures=temprange,
                                               observables=exp['simulation'].fullParsedYamlFile['observables'],
                                               kineticSens=0,
                                               physicalSens=0,
                                               conditions=conds,
                                               thermalBoundary=exp['simulation'].fullParsedYamlFile['thermalBoundary'],
                                               mechanicalBoundary=exp['simulation'].fullParsedYamlFile['mechanicalBoundary'],
                                               processor=p,
                                               cti_path="", 
                                               save_physSensHistories=0,
                                               fullParsedYamlFile=exp['simulation'].fullParsedYamlFile, 
                                               save_timeHistories=0,
                                               log_file=True,
                                               log_name='log.txt',
                                               timeshift=exp['simulation'].fullParsedYamlFile['time_shift'],
                                               initialTime=exp['simulation'].fullParsedYamlFile['initialTime'],
                                               finalTime=exp['simulation'].fullParsedYamlFile['finalTime'],
                                               target=exp['simulation'].fullParsedYamlFile['target'],
                                               target_type=exp['simulation'].fullParsedYamlFile['target_type'],
                                               n_processors=2,
                                               volumeTrace=volumeTrace)
            soln,temp=ig_delay.run()        
        
        #print(soln)
        return soln
    def run_jsr(self,exp,cti,n_of_data_points=100):
        
        p=pr.Processor(cti)
        
        tempmin=np.min(exp['simulation'].fullParsedYamlFile['temperatures'])
        # print('Tempmin: '+str(tempmin))
        tempmax=np.max(exp['simulation'].fullParsedYamlFile['temperatures'])
        # print('Tempmax: '+str(tempmax))
        if tempmax!=tempmin:
            total_range=tempmax-tempmin
            tempmax=tempmax+0.1*total_range
            tempmin=tempmin-0.1*total_range
        elif tempmax==tempmin:
            tempmax=tempmax*1.1
            tempmin=tempmin*0.9
        temprange=np.linspace(tempmin,tempmax,n_of_data_points)
        # print(temprange)
        pressures=exp['simulation'].fullParsedYamlFile['pressure']
        conds=exp['simulation'].fullParsedYamlFile['conditions']
            
            
            
        
        jsr1=jsr.JSR_multiTemp_steadystate(volume=exp['simulation'].fullParsedYamlFile['volume'],
                    pressure=pressures,
                    temperatures=temprange,
                    observables=exp['simulation'].fullParsedYamlFile['observables'],
                    kineticSens=0,
                    physicalSens=0,
                    conditions=conds,
                    thermalBoundary=exp['simulation'].fullParsedYamlFile['thermalBoundary'],
                    mechanicalBoundary=exp['simulation'].fullParsedYamlFile['mechanicalBoundary'],
                    processor=p,
                    save_physSensHistories=0,
                    save_timeHistories=0,
                    residence_time=exp['simulation'].fullParsedYamlFile['residence_time'],
                    moleFractionObservables = exp['simulation'].fullParsedYamlFile['moleFractionObservables'],
                    fullParsedYamlFile = exp['simulation'].fullParsedYamlFile)
        soln,temp=jsr1.run()
        
        
        #print(soln)
        return soln
    
    def plotting_observables(self,file_identifier='',filetype='.jpg'):
            
        print('\n')
        print('--------------------------------------------------------------------------')
        print('Observable Plots')
        print('--------------------------------------------------------------------------')
         
        obs_list = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            for observable in exp['mole_fraction_observables'] + exp['concentration_observables'] + exp['ignition_delay_observables']:
                if observable == None:
                    continue 
                obs_list.append(observable)
            if 'perturbed_coef' in exp.keys():
                for wl in self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']:    
                    obs_list.append(wl)    
        self.obs_list = obs_list     
                                      
        self.obs_loop = self.manager.counter(total=len(self.obs_list), desc='Observable Plots:', unit='plots', color='blue') 
                    
        sigmas_original,test1 = self.calculating_sigmas(self.S_matrix_original, self.covariance_original)
        sigmas_optimized,test2 = self.calculating_sigmas(self.S_matrix, self.covariance)
        
        
        for i,exp in enumerate(self.exp_dict_list_optimized):
            
            observable_counter=0     
                   
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables'] + exp['ignition_delay_observables']):
                
                plt.figure()
                temp_optimized_df_for_species = pd.DataFrame()
                temp_optimized_error_df_for_species = pd.DataFrame()
                temp_original_df_for_species = pd.DataFrame()
                temp_original_error_df_for_species = pd.DataFrame()   
            
                            
                # print(observable, observable_counter)
                if observable == None:
                    continue
                
                print(os.path.basename(self.files_to_include[0][i][0])[:-5] + ' ' + observable)
                
                plt.title('Experiment_'+str(i+1) + ' (' + os.path.basename(self.files_to_include[0][i][0])[:-5] + ')')
                
                observable_ylabel_string = list(observable)
                for k, oys in enumerate(observable_ylabel_string):

                    if oys.isdigit() == True:
                        if k == 0:
                            pass
                        elif observable_ylabel_string[k-1] == ' ':
                            pass
                        else:
                            observable_ylabel_string[k] = '$_{' + str(oys) + '}$'

                observable_ylabel_transformed = r'' + "".join(observable_ylabel_string)                     

                if observable in exp['mole_fraction_observables']:

                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']) and re.match('[Ss]pecies[ -][Pp]rofile',exp['experiment_type']):
                        
                        data_df = pd.DataFrame(exp['experimental_data'][observable_counter])
                        
                        if 'W' in list(data_df.columns):
                            weighted_df = data_df[data_df['W'] > 1.00e-06]
                            unweighted_df = data_df[data_df['W'] <= 1.00e-06]   
                            # print('hey')
                        else:
                            weighted_df = data_df
                            unweighted_df = pd.DataFrame(columns=weighted_df.columns)
                        
                        # weighted_df = data_df[data_df['W'] != 1.00e-09]
                        # unweighted_df = data_df[data_df['W'] == 1.00e-09]      
                                          
                        # weighted_df = data_df[data_df['W'] != 1.00e-06 or < 1.00e-06]
                        # unweighted_df = data_df[data_df['W'] <= 1.00e-06]                        
                        
                        plt.xlabel('Time [ms]')
                        plt.ylabel(observable_ylabel_transformed + ' Mole Fraction')
                        # plt.title('Experiment_'+str(i+1) + ' ' + self.files_to_include[0][i][0][:-5])
                        
                        temp_optimized_df_for_species['time [s]'] = pd.Series(exp['simulation'].timeHistories[0]['time'])
                        temp_optimized_df_for_species[observable] = pd.Series(exp['simulation'].timeHistories[0][observable])
                        
                        temp_original_df_for_species['time [s]'] = pd.Series(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time'])
                        temp_original_df_for_species[observable] = pd.Series(self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable])
                        
                        if bool(sigmas_optimized) == True:
                            
                            high_error_original = np.exp(sigmas_original[i][observable_counter])
                            high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                            low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            

                            # print(pd.Series(exp['experimental_data'][observable_counter]['Time']))
                            
                            temp_original_error_df_for_species['low_error_bar_time [s]'] = pd.Series(exp['experimental_data'][observable_counter]['Time'])
                            temp_original_error_df_for_species['high_error_bar_time [s]'] = pd.Series(exp['experimental_data'][observable_counter]['Time'])
                            
                            temp_original_error_df_for_species['low_error_bar'] = pd.Series(low_error_original)
                            temp_original_error_df_for_species['high_error_bar'] =  pd.Series(high_error_original)                                
                            
                            high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                            high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            
                            if len(data_df) != len(unweighted_df):
                                
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3, high_error_optimized,'b--')
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3, low_error_optimized,'b--')
                                
                            temp_optimized_error_df_for_species['low_error_bar_time [s]'] = pd.Series(exp['experimental_data'][observable_counter]['Time'])
                            temp_optimized_error_df_for_species['high_error_bar_time [s]'] = pd.Series(exp['experimental_data'][observable_counter]['Time'])
                            
                            temp_optimized_error_df_for_species['low_error_bar'] = pd.Series(low_error_optimized)
                            temp_optimized_error_df_for_species['high_error_bar'] = pd.Series(high_error_optimized)
                        
                        if len(data_df) == len(weighted_df):
                            plt.scatter(weighted_df['Time']*1e3,weighted_df[observable],marker='o',color='black',label='Experimental Data')    
                        else:
                            plt.scatter(weighted_df['Time']*1e3,weighted_df[observable],marker='o',color='black',label='Experimental Data')  
                            plt.scatter(unweighted_df['Time']*1e3,unweighted_df[observable],marker='o',color='black', facecolors='none',label='Unweighted Data')     
                        
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable],'r',label= r"$\it{A}$ $\it{priori}$ model")     
                        
                        plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable],'b',label='MSI')                                      
                            
                        #stub
                        plt.plot([],'w' ,label= 'T:'+ str(self.exp_dict_list_original[i]['simulation'].temperature))
                        plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
                        key_list = []
                        for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():
                            
                            plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                            key_list.append(key)
                       
                        #plt.legend(handlelength=3)
                        plt.legend(ncol=2)
                        sp = '_'.join(key_list)
                        #print(sp)

                        # plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K'+'_'+str(self.exp_dict_list_original[i]['simulation'].pressure)+'_'+sp+'_'+'.pdf', bbox_inches='tight')
                        # plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K'+'_'+str(self.exp_dict_list_original[i]['simulation'].pressure)+'_'+sp+'_'+'.svg', bbox_inches='tight',transparent=True)
                        # temp_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K'+'_'+str(self.exp_dict_list_original[i]['simulation'].pressure)+'_'+sp+'.csv',index=False)  

                        
                        if self.pdf == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=self.dpi)
                        if self.png == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.png', bbox_inches='tight',dpi=self.dpi)
                        if self.svg == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)
                                
                        temp_optimized_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_optimized.csv',index=False)  
                        temp_optimized_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_optimized.csv',index=False)  
                        temp_original_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_original.csv',index=False)  
                        temp_original_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_original.csv',index=False)  
                        
                        #stub
                        # plt.savefig(self.out_path+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        # plt.savefig(self.out_path+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)                      
                          


                        observable_counter+=1

                    elif re.match('[Jj][Ss][Rr]',exp['simulation_type']):
                        
                        # nominal=self.run_jsr(self.exp_dict_list_original[i],self.nominal_cti)
                        # MSI_model=self.run_jsr(exp,self.new_cti)
                        
                        # print('HERE')
                        # # print(exp['simulation'].timeHistories[0]['temperature'])
                        # print(self.exp_dict_list_original[i])
                        
                        
                        
                        # plt.plot(MSI_model['temperature'],MSI_model[observable],'b',label='MSI')
                        # plt.plot(nominal['temperature'],nominal[observable],'r',label= "$\it{A priori}$ model")
                        # plt.plot(exp['experimental_data'][observable_counter]['Temperature'],exp['experimental_data'][observable_counter][observable],'o',color='black',label='Experimental Data')
                        
                        data_df = pd.DataFrame(exp['experimental_data'][observable_counter])
                        
                        if 'W' in list(data_df.columns):
                            weighted_df = data_df[data_df['W'] > 1.00e-06]
                            unweighted_df = data_df[data_df['W'] <= 1.00e-06]   
                        else:
                            weighted_df = data_df
                            unweighted_df = pd.DataFrame(columns=weighted_df.columns)
                                                    
                        # weighted_df = data_df[data_df['W'] != 1.00e-09]
                        # unweighted_df = data_df[data_df['W'] == 1.00e-09]
                        
                        # weighted_df = data_df[data_df['W'] != 1.00e-06 or < 1.00e-06]
                        # unweighted_df = data_df[data_df['W'] <= 1.00e-06]                              

                        plt.xlabel('Temperature [K]')
                        plt.ylabel(observable_ylabel_transformed + ' Mole Fraction')
                        # plt.title('Experiment_'+str(i+1) + ' ' + self.files_to_include[0][i][0][:-5])
                        
                        temp_optimized_df_for_species['temperature [K]'] = pd.Series(exp['simulation'].timeHistories[0]['temperature'])
                        temp_optimized_df_for_species[observable] = pd.Series(exp['simulation'].timeHistories[0][observable])         
                        
                        temp_original_df_for_species['temperature [K]'] = pd.Series(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'])
                        temp_original_df_for_species[observable] =  pd.Series(self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable])                    
                                                
                        if bool(sigmas_optimized) == True:
                            
                            high_error_original = np.exp(sigmas_original[i][observable_counter])
                            high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable].dropna().values)
                            low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                            low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable].dropna().values)   
                                                                                
                            temp_original_error_df_for_species['low_error_bar_temperature [K]'] = pd.Series(exp['experimental_data'][observable_counter]['Temperature'])
                            temp_original_error_df_for_species['high_error_bar_temperature [K]'] = pd.Series(exp['experimental_data'][observable_counter]['Temperature'])
                            temp_original_error_df_for_species['low_error_bar'] = pd.Series(low_error_original)
                            temp_original_error_df_for_species['high_error_bar'] = pd.Series(high_error_original)                               
                            
                            high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                            high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values)
                            low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values)
                            
                            if len(data_df) != len(unweighted_df):
                                if len(high_error_optimized)>1 and len(low_error_optimized) > 1:
                                    plt.plot(exp['experimental_data'][observable_counter]['Temperature'], high_error_optimized,'b--')
                                    plt.plot(exp['experimental_data'][observable_counter]['Temperature'], low_error_optimized,'b--')
                                    
                                else:
                                    plt.plot(exp['experimental_data'][observable_counter]['Temperature'], high_error_optimized,'bx')
                                    plt.plot(exp['experimental_data'][observable_counter]['Temperature'], low_error_optimized,'bx')
                            
                            temp_optimized_error_df_for_species['low_error_bar_temperature [K]'] = pd.Series(exp['experimental_data'][observable_counter]['Temperature'])
                            temp_optimized_error_df_for_species['high_error_bar_temperature [K]'] = pd.Series(exp['experimental_data'][observable_counter]['Temperature'])
                            temp_optimized_error_df_for_species['low_error_bar'] = pd.Series(low_error_optimized)
                            temp_optimized_error_df_for_species['high_error_bar'] = pd.Series(high_error_optimized)            

                        # plt.scatter(weighted_df['Temperature'],weighted_df[observable],marker='o',color='black',label='Experimental Data')    
                        # plt.scatter(unweighted_df['Temperature'],unweighted_df[observable],marker='o',color='black', facecolors='none',label='Unweighted Data')      
                        
                        if len(data_df) == len(weighted_df):
                            plt.scatter(weighted_df['Temperature'],weighted_df[observable],marker='o',color='black',label='Experimental Data')   
                        else:
                            plt.scatter(weighted_df['Temperature'],weighted_df[observable],marker='o',color='black',label='Experimental Data')    
                            plt.scatter(unweighted_df['Temperature'],unweighted_df[observable],marker='o',color='black', facecolors='none',label='Unweighted Data')                          
                        
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'],self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable],'ro',label= r"$\it{A}$ $\it{priori}$ model")    
                        plt.plot(exp['simulation'].timeHistories[0]['temperature'],exp['simulation'].timeHistories[0][observable],'bo',label='MSI')
             
                        
                        plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
                        key_list = []
                        for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():
                            
                            plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                            key_list.append(key)
                       
                        #plt.legend(handlelength=3)
                        plt.legend(ncol=2)
                        sp = '_'.join(key_list)
                        
                        if self.pdf == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=self.dpi)
                        if self.png == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.png', bbox_inches='tight',dpi=self.dpi)
                        if self.svg == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)                        
                        
                                   

                        temp_optimized_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_optimized.csv',index=False)  
                        temp_optimized_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_optimized.csv',index=False)   
                        temp_original_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_original.csv',index=False)  
                        temp_original_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_original.csv',index=False)   
                        
                        # temp_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.csv',index=False)    

                        observable_counter+=1    

                    elif re.match('[Ss]pecies[- ][Pp]rofile',exp['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp['simulation_type']):
                        plt.plot(exp['simulation'].timeHistories[0]['temperature'],exp['simulation'].timeHistories[0][observable],'b',label='MSI')
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'],self.exp_dict_list_original['simulation'].timeHistories[0][observable],'r',label= r"$\it{A}$ $\it{priori}$ model")
                        plt.plot(exp['experimental_data'][observable_counter]['Temperature'],exp['experimental_data'][observable_counter][observable],'o',color='black',label='Experimental Data')
                        plt.xlabel('Temperature [K]')
                        plt.ylabel(observable_ylabel_transformed + ' Mole Fraction')
                        # plt.title('Experiment_'+str(i+1) + ' ' + self.files_to_include[0][i][0][:-5])
                        
                        temp_optimized_df_for_species['temperature'] = pd.Series(exp['simulation'].timeHistories[0]['temperature'])
                        temp_optimized_df_for_species[observable] = pd.Series(exp['simulation'].timeHistories[0][observable])               
                        
                        temp_original_df_for_species['temperature'] = pd.Series(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'])
                        temp_original_df_for_species[observable] =  pd.Series(self.exp_dict_list_original['simulation'].timeHistories[0][observable])              
                                            
                        if bool(sigmas_optimized) == True:
                            
                            high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                            high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values)
                            low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values)
                            
                            if len(high_error_optimized)>1 and len(low_error_optimized) > 1:
                                plt.plot(exp['experimental_data'][observable_counter]['Temperature'], high_error_optimized,'b--')
                                plt.plot(exp['experimental_data'][observable_counter]['Temperature'], low_error_optimized,'b--')
                                
                            else:
                                plt.plot(exp['experimental_data'][observable_counter]['Temperature'], high_error_optimized,'bX')
                                plt.plot(exp['experimental_data'][observable_counter]['Temperature'], low_error_optimized,'bX')
                            
                            temp_optimized_error_df_for_species['low_error_bar_temperature'] = pd.Series(exp['experimental_data'][observable_counter]['Temperature'])
                            temp_optimized_error_df_for_species['high_error_bar_temperature'] = pd.Series(exp['experimental_data'][observable_counter]['Temperature'])
                            
                            temp_optimized_error_df_for_species['low_error_bar'] = pd.Series(low_error_optimized)
                            temp_optimized_error_df_for_species['high_error_bar'] = pd.Series(high_error_optimized)            
                            
                            
                            high_error_original = np.exp(sigmas_original[i][observable_counter])
                            # high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable].dropna().values)
                            high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                            low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            
                            temp_original_error_df_for_species['low_error_bar_temperature'] = pd.Series(exp['experimental_data'][observable_counter]['Temperature'])
                            temp_original_error_df_for_species['high_error_bar_temperature'] = pd.Series(exp['experimental_data'][observable_counter]['Temperature'])
                                                    
                            temp_original_error_df_for_species['low_error_bar'] = pd.Series(low_error_original)
                            temp_original_error_df_for_species['high_error_bar'] = pd.Series(high_error_original)   
                        
                        plt.plot([],'w' ,label= 'T:'+ str(self.exp_dict_list_original[i]['simulation'].temperature))
                        plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
                        key_list = []
                        for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():
                            
                            plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                            key_list.append(key)
                       
                        #plt.legend(handlelength=3)
                        plt.legend(ncol=2)
                        sp = '_'.join(key_list)
                        
                        if self.pdf == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=self.dpi)
                        if self.png == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.png', bbox_inches='tight',dpi=self.dpi)
                        if self.svg == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)          

                        temp_optimized_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_optimized.csv',index=False)  
                        temp_optimized_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_optimized.csv',index=False)   
                        temp_original_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_original.csv',index=False)   
                        temp_original_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_original.csv',index=False)   
                        
                        # temp_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.csv',index=False)    

                        observable_counter+=1    
                        
                if observable in exp['concentration_observables']:
                    #print(observable_counter,'THIS IS OBSERVABLE COUNTER')
                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']) and re.match('[Ss]pecies[ -][Pp]rofile',exp['experiment_type']):
                        #print(observable_counter)
                        if observable+'_ppm' in exp['experimental_data'][observable_counter].columns:
                            plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable]*1e6,'b',label='MSI')
                            plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable]*1e6,'r',label= r"$\it{A}$ $\it{priori}$ model")
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable+'_ppm'],'o',color='black',label='Experimental Data') 
                            plt.xlabel('Time [ms]')
                            plt.ylabel(observable_ylabel_transformed+ ' Mole Fraction [ppm]')
                            # plt.title('Experiment_'+str(i+1) + ' ' + self.files_to_include[0][i][0][:-5])
                            
                            temp_optimized_df_for_species['time'] = pd.Series(exp['simulation'].timeHistories[0]['time']*1e3)
                            temp_optimized_df_for_species[observable] = pd.Series(exp['simulation'].timeHistories[0][observable]*1e6)
                            
                            temp_original_df_for_species['time'] = pd.Series(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3)
                            temp_original_df_for_species[observable] = pd.Series(self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable]*1e6)                            
                            
                            if bool(sigmas_optimized)==True:
                                
                                high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                                high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                                # low_error_optimized = np.exp(np.array(sigmas_optimized[i][observable_counter])*-1)
                                low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)                                
                                low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                                
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3, high_error_optimized,'b--')
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3, low_error_optimized,'b--')       
                                
                                temp_optimized_error_df_for_species['low_error_bar_time'] = pd.Series(exp['experimental_data'][observable_counter]['Time']*1e3)
                                temp_optimized_error_df_for_species['high_error_bar_time'] = pd.Series(exp['experimental_data'][observable_counter]['Time']*1e3)
                                
                                temp_optimized_error_df_for_species['low_error_bar'] = pd.Series(low_error_optimized)
                                temp_optimized_error_df_for_species['high_error_bar'] = pd.Series(high_error_optimized)
                                
                                high_error_original = np.exp(sigmas_original[i][observable_counter])
                                high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                                low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                                low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                                                            
                                temp_original_error_df_for_species['low_error_bar_time'] = pd.Series(exp['experimental_data'][observable_counter]['Time']*1e3)
                                temp_original_error_df_for_species['high_error_bar_time'] = pd.Series(exp['experimental_data'][observable_counter]['Time']*1e3)
                                
                                temp_original_error_df_for_species['low_error_bar'] = pd.Series(low_error_original)
                                temp_original_error_df_for_species['high_error_bar'] =  pd.Series(high_error_original)                                       

                                #high_error_original = np.exp(sigmas_original[i][observable_counter])
                                #high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                                #low_error_original = np.exp(np.array(sigmas_original[i][observable_counter])*-1)
                                #low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                                
                                #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_original,'r--')
                                #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_original,'r--')
                        elif observable+'_mol/cm^3' in exp['experimental_data'][observable_counter].columns:
                            concentration_optimized = np.true_divide(1,exp['simulation'].timeHistories[0]['temperature'].to_numpy())*exp['simulation'].timeHistories[0]['pressure'].to_numpy()
                           
                            concentration_optimized *= (1/(8.314e6))*exp['simulation'].timeHistories[0][observable].dropna().to_numpy()
                            concentration_original = np.true_divide(1,self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'].to_numpy())*self.exp_dict_list_original[i]['simulation'].timeHistories[0]['pressure'].to_numpy()
                           
                            concentration_original *= (1/(8.314e6))*self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable].dropna().to_numpy()
                            
                            plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,concentration_optimized,'b',label='MSI')
                            plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,concentration_original,'r',label= r"$\it{A}$ $\it{priori}$ model")
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable+'_mol/cm^3'],'o',color='black',label='Experimental Data') 
                            plt.xlabel('Time [ms]')
                            plt.ylabel(r'$\frac{mol}{cm^3}$'+''+observable_ylabel_transformed)
                            # plt.title('Experiment_'+str(i+1) + ' ' + self.files_to_include[0][i][0][:-5])
                            
                            temp_optimized_df_for_species['time'] = pd.Series(exp['simulation'].timeHistories[0]['time']*1e3)
                            temp_optimized_df_for_species[observable] = pd.Series(concentration_optimized)
                            
                            temp_original_df_for_species['time'] = pd.Series(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3)
                            temp_original_df_for_species[observable] = pd.Series(concentration_original)                                    
                            
                            if bool(sigmas_optimized)==True:
                                concentration_sig = np.true_divide(1,exp['simulation'].pressureAndTemperatureToExperiment[observable_counter]['temperature'].to_numpy())*exp['simulation'].pressureAndTemperatureToExperiment[observable_counter]['pressure'].to_numpy()
                        
                                concentration_sig *= (1/(8.314e6))*exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().to_numpy()
                                high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                                high_error_optimized = np.multiply(high_error_optimized,concentration_sig)
                                low_error_optimized = np.exp(np.array(sigmas_optimized[i][observable_counter])*-1)
                                low_error_optimized = np.multiply(low_error_optimized,concentration_sig)
                                
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3, high_error_optimized,'b--')
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3, low_error_optimized,'b--') 
                                
                                temp_optimized_error_df_for_species['low_error_bar_time'] = pd.Series(exp['experimental_data'][observable_counter]['Time']*1e3)
                                temp_optimized_error_df_for_species['high_error_bar_time'] = pd.Series(exp['experimental_data'][observable_counter]['Time']*1e3)
                                
                                temp_optimized_error_df_for_species['low_error_bar'] = pd.Series(low_error_optimized)
                                temp_optimized_error_df_for_species['high_error_bar'] = pd.Series(high_error_optimized)
                                
                                # concentration_sig_original = np.true_divide(1,high_error_original,self.exp_dict_list_original[i]['simulation'].pressureAndTemperatureToExperiment[observable_counter]['temperature'].to_numpy())*high_error_original,self.exp_dict_list_original[i]['simulation'].pressureAndTemperatureToExperiment[observable_counter]['pressure'].to_numpy()
                                # concentration_sig_original *= (1/(8.314e6))*high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().to_numpy()                                
                                
                                # high_error_original = np.exp(sigmas_original[i][observable_counter])
                                # high_error_original = np.multiply(high_error_original,concentration_sig_original)
                                # low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                                # low_error_original = np.multiply(low_error_original,concentration_sig_original)
                                                            

                                temp_original_error_df_for_species['low_error_bar_time'] = pd.Series(exp['experimental_data'][observable_counter]['Time']*1e3)
                                temp_original_error_df_for_species['high_error_bar_time'] = pd.Series(exp['experimental_data'][observable_counter]['Time']*1e3)
                                
                                # temp_original_error_df_for_species['low_error_bar'] = pd.Series(low_error_original)
                                # temp_original_error_df_for_species['high_error_bar'] =  pd.Series(high_error_original)                                                                              
                        
                        plt.plot([],'w' ,label= 'T:'+ str(self.exp_dict_list_original[i]['simulation'].temperature))
                        plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
                        key_list = []
                        for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():
                            
                            plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                            key_list.append(key)
                       
                        #plt.legend(handlelength=3)
                        plt.legend(ncol=2)
                        sp = '_'.join(key_list)
                        
                        #print(sp)
                        #plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K'+'_'+str(self.exp_dict_list_original[i]['simulation'].pressure)+'_'+sp+'_'+'.pdf', bbox_inches='tight')
                        
                        #stub
                        # plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        # plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)
                    
                        if self.pdf == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=self.dpi)
                        if self.png == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.png', bbox_inches='tight',dpi=self.dpi)
                        if self.svg == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True) 

                        temp_optimized_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_optimized.csv',index=False)  
                        temp_optimized_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_optimized.csv',index=False)  
                        temp_original_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_original.csv',index=False)  
                        temp_original_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_original.csv',index=False)  

                        observable_counter+=1
                    
                    elif re.match('[Ff]low [Rr]eactor',exp['simulation_type']) and re.match('[Ss]pecies[ -][Pp]rofile',exp['experiment_type']):

                        data_df = pd.DataFrame(exp['experimental_data'][observable_counter])
                        
                        if 'W' in list(data_df.columns):
                            weighted_df = data_df[data_df['W'] > 1.00e-06]
                            unweighted_df = data_df[data_df['W'] <= 1.00e-06]   
                        else:
                            weighted_df = data_df
                            unweighted_df = pd.DataFrame(columns=weighted_df.columns)
                                                    
                        # weighted_df = data_df[data_df['W'] != 1.00e-09]
                        # unweighted_df = data_df[data_df['W'] == 1.00e-09]
                        
                        # weighted_df = data_df[data_df['W'] != 1.00e-06 or < 1.00e-06]
                        # unweighted_df = data_df[data_df['W'] <= 1.00e-06]                              

                        plt.xlabel('Temperature [K]')
                        plt.ylabel(observable_ylabel_transformed + ' Mole Fraction [ppm]')
                        # plt.title('Experiment_'+str(i+1) + ' ' + self.files_to_include[0][i][0][:-5])
                        
                        temp_optimized_df_for_species['temperature [K]'] = exp['simulation'].timeHistories[0]['initial_temperature']
                        temp_optimized_df_for_species[observable + ' [ppm]'] = exp['simulation'].timeHistories[0][observable]*1e6             
                        
                        temp_original_df_for_species['temperature [K]'] = self.exp_dict_list_original[i]['simulation'].timeHistories[0]['initial_temperature']
                        temp_original_df_for_species[observable + ' [ppm]'] =  self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable]*1e6   
    
                        
                        if bool(sigmas_optimized) == True:
                            #stub
                            high_error_original = np.exp(sigmas_original[i][observable_counter])
                            high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable].dropna().values*1e6)
                            # high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                            low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                            low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable].dropna().values*1e6)
                            # low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                            
                            temp_original_error_df_for_species['low_error_bar_temperature [K]'] = list(exp['experimental_data'][observable_counter]['Temperature'])
                            temp_original_error_df_for_species['high_error_bar_temperature [K]'] = list(exp['experimental_data'][observable_counter]['Temperature'])
                                                    
                            temp_original_error_df_for_species['low_error_bar' + ' [ppm]'] = low_error_original
                            temp_original_error_df_for_species['high_error_bar' + ' [ppm]'] = high_error_original   
                            
                                                        
                            high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                            high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values*1e6)
                            low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values*1e6)
                            
                            if len(data_df) != len(unweighted_df):
                            
                                if len(high_error_optimized)>1 and len(low_error_optimized) > 1:
                                    plt.plot(exp['experimental_data'][observable_counter]['Temperature'], high_error_optimized,'b--')
                                    plt.plot(exp['experimental_data'][observable_counter]['Temperature'], low_error_optimized,'b--')
                                    
                                else:
                                    plt.plot(exp['experimental_data'][observable_counter]['Temperature'], high_error_optimized,'bX')
                                    plt.plot(exp['experimental_data'][observable_counter]['Temperature'], low_error_optimized,'bX')                            

                            temp_optimized_error_df_for_species['low_error_bar_temperature [K]'] = list(exp['experimental_data'][observable_counter]['Temperature'])
                            temp_optimized_error_df_for_species['high_error_bar_temperature [K]'] = list(exp['experimental_data'][observable_counter]['Temperature'])
                            
                            temp_optimized_error_df_for_species['low_error_bar' + ' [ppm]'] = low_error_optimized
                            temp_optimized_error_df_for_species['high_error_bar' + ' [ppm]'] = high_error_optimized   
                            
                            # high_error_original = np.exp(sigmas_original[i][observable_counter])
                            # high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable].dropna().values)
                            # low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                            # low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            
                            #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_original,'r--')
                            #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_original,'r--')
                        
                        #plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=1000)        
                        # 
                        # 
                        # plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight',dpi=1000) 
                        # plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',dpi=1000,transparent=True)             
                        # temp_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.csv',index=False)                            
                        #              
                        
                        # plt.scatter(weighted_df['Temperature'],weighted_df[observable+'_ppm'],marker='o',color='black',label='Experimental Data')    
                        # plt.scatter(unweighted_df['Temperature'],unweighted_df[observable+'_ppm'],marker='o',color='black', facecolors='none',label='Unweighted Data')         
                        
                        if len(data_df) == len(weighted_df):
                            plt.scatter(weighted_df['Temperature'],weighted_df[observable+'_ppm'],marker='o',color='black',label='Experimental Data')    
                        else:
                            plt.scatter(weighted_df['Temperature'],weighted_df[observable+'_ppm'],marker='o',color='black',label='Experimental Data')    
                            plt.scatter(unweighted_df['Temperature'],unweighted_df[observable+'_ppm'],marker='o',color='black', facecolors='none',label='Unweighted Data')                       
                        
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['initial_temperature'],self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable]*1e6,'r',label= r"$\it{A}$ $\it{priori}$ model")
                        plt.plot(exp['simulation'].timeHistories[0]['initial_temperature'],exp['simulation'].timeHistories[0][observable]*1e6,'b',label='MSI')
                        
                        # plt.plot(exp['experimental_data'][observable_counter]['Temperature'],exp['experimental_data'][observable_counter][observable+'_ppm'],'o',color='black',label='Experimental Data')                                        

                        # plt.plot([],'w' ,label= 'T:'+ str(self.exp_dict_list_original[i]['simulation'].emperature))
                        plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
                        key_list = []
                        for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():
                            
                            plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                            key_list.append(key)
                       
                        #plt.legend(handlelength=3)
                        plt.legend(ncol=2)
                        sp = '_'.join(key_list)

                        if self.pdf == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=self.dpi)
                        if self.png == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.png', bbox_inches='tight',dpi=self.dpi)
                        if self.svg == True:
                            plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True) 
                                   
                        temp_optimized_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_optimized.csv',index=False)  
                        temp_optimized_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_optimized.csv',index=False)  
                        temp_original_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_original.csv',index=False) 
                        temp_original_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_original.csv',index=False)   

                        observable_counter+=1       


                if observable in exp['ignition_delay_observables']:
                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']):
                        if len(exp['simulation'].temperatures)>1:
                            nominal=self.run_ignition_delay(self.exp_dict_list_original[i], self.nominal_cti)
                            MSI_model=self.run_ignition_delay(exp, self.new_cti)
                            #plt.semilogy(1000/MSI_model['temperature'],MSI_model['delay'],'b',label='MSI')
                            #changed to plotting at nominal temperature
                            
                            #plt.semilogy(1000/MSI_model['temperature'],MSI_model['delay'],'b',label='MSI')
                        
                            #a, b = zip(*sorted(zip(1000/exp['experimental_data'][observable_counter]['temperature'],exp['simulation'].timeHistories[0]['delay'].dropna().values)))
                            
                            a, b = zip(*sorted(zip(1000/np.array(exp['simulation'].temperatures),exp['simulation'].timeHistories[0]['delay'].dropna().values)))

                            plt.semilogy(a,b,'b',label='MSI')

                            plt.semilogy(1000/nominal['temperature'],nominal['delay'],'r',label= r"$\it{A}$ $\it{priori}$ model")
                            
                            #plt.semilogy(1000/exp['simulation'].timeHistories[0]['temperature'],exp['simulation'].timeHistories[0]['delay'],'b',label='MSI')
                            #plt.semilogy(1000/self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'],self.exp_dict_list_original[i]['simulation'].timeHistories[0]['delay'],'r',label= "$\it{A priori}$ model")
                            plt.semilogy(1000/exp['experimental_data'][observable_counter]['temperature'],exp['experimental_data'][observable_counter][observable+'_s'],'o',color='black',label='Experimental Data')
                            plt.xlabel('1000/T [1000/K]')
                            plt.ylabel('Time [s]')
                            # plt.title('Experiment_'+str(i+1) + ' ' + self.files_to_include[0][i][0][:-5])
                            
                            if bool(sigmas_optimized) == True:

                                
                                high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                                high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistories[0]['delay'].dropna().values)
                                
                                low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                                low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistories[0]['delay'].dropna().values)
                                #plt.figure()
                                #print(exp['simulation'].timeHistories[0]['delay'].dropna().values,'THIS IS IN THE PLOTTER')
                                
                                
                                #a, b = zip(*sorted(zip(1000/exp['experimental_data'][observable_counter]['temperature'],high_error_optimized)))
                                a, b = zip(*sorted(zip(1000/np.array(exp['simulation'].temperatures),high_error_optimized)))
                                
                                plt.semilogy(a,b,'b--')

                                #a, b = zip(*sorted(zip(1000/exp['experimental_data'][observable_counter]['temperature'],low_error_optimized)))
                                a, b = zip(*sorted(zip(1000/np.array(exp['simulation'].temperatures),low_error_optimized)))
                                
                                plt.semilogy(a,b,'b--')                           
                                
                                #plt.plot(1000/exp['experimental_data'][observable_counter]['temperature'],exp['simulation'].timeHistories[0]['delay'].dropna().values,'x')
                                #plt.plot(1000/np.array(exp['simulation'].temperatures),exp['simulation'].timeHistories[0]['delay'].dropna().values,'o')
                                
                                
                                #high_error_original = np.exp(sigmas_original[i][observable_counter])
                               # high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                                #low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                                #low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                                #plt.figure()
                               # plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_original,'r--')
                                #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_original,'r--')

                            plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressures))
                            key_list = []
                            for key in self.exp_dict_list_original[i]['simulation'].fullParsedYamlFile['conditions_to_run'][0].keys():
                               # ['simulation'].fullParsedYamlFile['conditions_to_run']
                                plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].fullParsedYamlFile['conditions_to_run'][0][key]))
                                key_list.append(key)
                       
                            #plt.legend(handlelength=3)
                            plt.legend(ncol=2)
                            sp = '_'.join(key_list)
                            
                            # plt.savefig(os.path.join(self.out_path,'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf'), bbox_inches='tight',dpi=1000)
                            # plt.savefig(os.path.join(self.out_path,'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg'), bbox_inches='tight',dpi=1000,transparent=True)
                            # temp_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.csv',index=False) 

                            if self.pdf == True:
                                plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=self.dpi)
                            if self.png == True:
                                plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.png', bbox_inches='tight',dpi=self.dpi)
                            if self.svg == True:
                                plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True) 
                            
                            temp_optimized_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_optimized.csv',index=False) 
                            temp_optimized_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_optimized.csv',index=False) 
                            temp_original_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_original.csv',index=False) 
                            temp_original_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_original.csv',index=False) 


                            observable_counter+=1
                    elif re.match('[Rr][Cc][Mm]',exp['simulation_type']):
                        if len(exp['simulation'].temperatures)>1:
                            
                            plt.semilogy(1000/exp['simulation'].timeHistories[0]['ignition_temperature'],exp['simulation'].timeHistories[0]['delay']-exp['simulation'].timeHistories[0]['end_of_compression_time'],'b',label='MSI')
                            plt.semilogy(1000/self.exp_dict_list_original[i]['simulation'].timeHistories[0]['ignition_temperature'],self.exp_dict_list_original[i]['simulation'].timeHistories[0]['delay']-self.exp_dict_list_original[i]['simulation'].timeHistories[0]['end_of_compression_time'],'r',label= r"$\it{A}$ $\it{priori}$ model")
    
                            #plt.semilogy(1000/exp['simulation'].timeHistories[0]['temperature'],exp['simulation'].timeHistories[0]['delay'],'b',label='MSI')
                            #plt.semilogy(1000/self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'],self.exp_dict_list_original[i]['simulation'].timeHistories[0]['delay'],'r',label= "$\it{A priori}$ model")
                            plt.semilogy(1000/exp['experimental_data'][observable_counter]['temperature'],exp['experimental_data'][observable_counter][observable+'_s'],'o',color='black',label='Experimental Data')
                            plt.xlabel('1000/T [1000/K]')
                            plt.ylabel('Time [ms]')
                            # plt.title('Experiment_'+str(i+1) + ' ' + self.files_to_include[0][i][0][:-5])
                            
                            if bool(sigmas_optimized) == True:
                                
                                high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                                high_error_optimized = np.multiply(high_error_optimized,(exp['simulation'].timeHistories[0]['delay']-exp['simulation'].timeHistories[0]['end_of_compression_time']).dropna().values)
                                low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                                low_error_optimized = np.multiply(low_error_optimized,(exp['simulation'].timeHistories[0]['delay']-exp['simulation'].timeHistories[0]['end_of_compression_time']).dropna().values)
                                #plt.figure()
                                a, b = zip(*sorted(zip(1000/exp['experimental_data'][observable_counter]['ignition_temperature'],high_error_optimized)))
                                plt.semilogy(a,b,'b--')
                                #plt.plot(1000/exp['experimental_data'][observable_counter]['temperature'],low_error_optimized,'b--')
                                a, b = zip(*sorted(zip(1000/exp['experimental_data'][observable_counter]['ignition_temperature'],low_error_optimized)))
                                plt.semilogy(a,b,'b--')                           
                                
                                
                                #high_error_original = np.exp(sigmas_original[i][observable_counter])
                               # high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                                #low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                                #low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                                #plt.figure()
                               # plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_original,'r--')
                                #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_original,'r--')
                            
                            if self.pdf == True:
                                plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=self.dpi)
                            if self.png == True:
                                plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.png', bbox_inches='tight',dpi=self.dpi)
                            if self.svg == True:
                                plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True) 

                            temp_optimized_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_optimized.csv',index=False) 
                            temp_optimized_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_optimized.csv',index=False) 
                            temp_original_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_original.csv',index=False) 
                            temp_original_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_error_original.csv',index=False) 
                            
                            observable_counter+=1    
                            
                self.obs_loop.update()                  
                        
            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                
                temp_optimized_df_for_species = pd.DataFrame()
                temp_optimized_error_df_for_species = pd.DataFrame()
                temp_original_df_for_species = pd.DataFrame()
                temp_original_error_df_for_species = pd.DataFrame()   
                                
                for k,wl in enumerate(wavelengths):
                    
                    print(os.path.basename(self.files_to_include[0][i][0])[:-5] + ' ' + str(wl))

                    data_df = pd.DataFrame(exp['absorbance_experimental_data'][k])

                    if 'W' in list(data_df.columns):
                        weighted_df = data_df[data_df['W'] > 1.00e-06]
                        unweighted_df = data_df[data_df['W'] <= 1.00e-06]   
                    else:
                        weighted_df = data_df
                        unweighted_df = pd.DataFrame(columns=weighted_df.columns)
                                                
                    # weighted_df = data_df[data_df['W'] != 1.00e-09]
                    # unweighted_df = data_df[data_df['W'] == 1.00e-09]

                    # weighted_df = data_df[data_df['W'] != 1.00e-06 or < 1.00e-06]
                    # unweighted_df = data_df[data_df['W'] <= 1.00e-06]                          
                    
                    #plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Experimental Data')
                    plt.title('Experiment_'+str(i+1) + ' (' + os.path.basename(self.files_to_include[0][i][0])[:-5] + ')')
                    plt.xlabel('Time [ms]')
                    plt.ylabel('Absorbance'+''+str(wl))
                    # plt.title('Experiment_'+str(i+1) + ' ' + self.files_to_include[0][i][0][:-5])
                    
                    temp_optimized_df_for_species['time [s]'] = pd.Series(exp['simulation'].timeHistories[0]['time'])
                    temp_optimized_df_for_species[observable] = pd.Series(exp['absorbance_calculated_from_model'][wl])
                    
                    temp_original_df_for_species['time [s]'] = pd.Series(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time'])
                    temp_original_df_for_species[observable] = pd.Series(self.exp_dict_list_original[i]['absorbance_calculated_from_model'][wl])                    
                    
                    if bool(sigmas_optimized)==True:
                        
                        high_error_original = np.exp(sigmas_original[i][observable_counter])
                        high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['absorbance_model_data'][wl])
                        low_error_original =  np.exp(sigmas_original[i][observable_counter]*-1)
                        low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['absorbance_model_data'][wl])
                        
                        temp_original_error_df_for_species['low_error_bar_time [s]'] = pd.Series(exp['absorbance_experimental_data'][k]['time'])
                        temp_original_error_df_for_species['high_error_bar_time [s]'] = pd.Series(exp['absorbance_experimental_data'][k]['time'])
                        
                        temp_original_error_df_for_species['low_error_bar'] = pd.Series(low_error_original)
                        temp_original_error_df_for_species['high_error_bar'] = pd.Series(high_error_original)    
                        
                                                
                        high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])
                        high_error_optimized = np.multiply(high_error_optimized,exp['absorbance_model_data'][wl])
                        low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                        low_error_optimized = np.multiply(low_error_optimized,exp['absorbance_model_data'][wl])
                        
                        if len(data_df) != len(unweighted_df):
                       
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,high_error_optimized,'b--')
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,low_error_optimized,'b--')
            
                        temp_optimized_error_df_for_species['low_error_bar_time [s]'] = pd.Series(exp['absorbance_experimental_data'][k]['time'])
                        temp_optimized_error_df_for_species['high_error_bar_time [s]'] = pd.Series(exp['absorbance_experimental_data'][k]['time'])
                        
                        temp_optimized_error_df_for_species['low_error_bar'] = pd.Series(low_error_optimized)
                        temp_optimized_error_df_for_species['high_error_bar'] = pd.Series(high_error_optimized)                                                                                            
                    
                    if len(data_df) == len(weighted_df):
                        plt.scatter(weighted_df['time']*1e3,weighted_df['Absorbance_'+str(wl)],marker='o',color='black',label='Experimental Data')     
                    else:
                        plt.scatter(weighted_df['time']*1e3,weighted_df['Absorbance_'+str(wl)],marker='o',color='black',label='Experimental Data')    
                        plt.scatter(unweighted_df['time']*1e3,unweighted_df['Absorbance_'+str(wl)],marker='o',color='black', facecolors='none',label='Unweighted Data')                      
                    
                    plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['absorbance_calculated_from_model'][wl],'r',label= r"$\it{A}$ $\it{priori}$ model")
                    plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['absorbance_calculated_from_model'][wl],'b',label='MSI')
                    
                    # plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Experimental Data')                    


                    key_list=[]
                    plt.plot([],'w' ,label= 'T:'+ str(self.exp_dict_list_original[i]['simulation'].temperature))
                    plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
                    for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():                        
                        plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                        key_list.append(key)

                    #plt.legend(handlelength=3)
                    plt.legend(ncol=2)
                    #plt.savefig(self.out_path+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                    sp = '_'.join(key_list)
                    
                    # plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                    # plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)


                    if self.pdf == True:
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_Absorb_at'+'_'+str(wl)+'.pdf', bbox_inches='tight',dpi=self.dpi)
                    if self.png == True:
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_Absorb_at'+'_'+str(wl)+'.png', bbox_inches='tight',dpi=self.dpi)
                    if self.svg == True:
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_Absorb_at'+'_'+str(wl)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True) 
                                


                    temp_optimized_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_Absorb_optimized.csv',index=False) 
                    temp_optimized_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_Absorb_error_optimized.csv',index=False) 
                    temp_original_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_Absorb_original.csv',index=False)   
                    temp_original_error_df_for_species.to_csv(self.out_path+'/'+'Experiment_'+str(i+1)+'_Absorb_error_original.csv',index=False)                     
                    
                    observable_counter+=1 
                    self.obs_loop.update()
            

                
    def plotting_X_itterations(self,list_of_X_values_to_plot = [], list_of_X_array=[],number_of_iterations=None):
        for value in list_of_X_values_to_plot:
            temp = []
            for array in list_of_X_array:
                temp.append(array[value][0])
            plt.figure()
            plt.plot(np.arange(0,number_of_iterations,1),temp)
        return
        
    def getting_matrix_diag(self,cov_matrix):
        diag = cov_matrix.diagonal()
        return diag
               
    def Y_matrix_plotter(self,Y_matrix,exp_dict_list_optimized,y_matrix,sigma):  
        #sigmas =[[] for x in range(len(self.simulation_lengths_of_experimental_data))]
             
        counter=0
        for x in range(len(self.simulation_lengths_of_experimental_data)):
            observable_counter = 0 
            for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                #for z in np.arange(counter,(self.simulation_lengths_of_experimental_data[x][y]+counter)):       
                
                    
                plt.figure()   
                Y_values_to_plot = list(Y_matrix[counter:self.simulation_lengths_of_experimental_data[x][y]+counter,:])
                y_values_to_plot = list(y_matrix[counter:self.simulation_lengths_of_experimental_data[x][y]+counter,:])  
                sigmas_to_plot = list(sigma[counter:self.simulation_lengths_of_experimental_data[x][y]+counter,:])  
                if 'perturbed_coef' in exp_dict_list_optimized[x].keys():
                    wavelengths = self.parsed_yaml_list_optimized[x]['absorbanceCsvWavelengths'][0]
                    time = exp_dict_list_optimized[x]['absorbance_experimental_data'][0]['time']                     
                    plt.subplot(4, 1, 1)    
                    plt.title('Experiment_'+str(x+1)+'_Wavelength_'+str(wavelengths))
                    plt.plot(time*1e3,Y_values_to_plot)
                    plt.tick_params(labelbottom=False)
                    plt.ylabel('Y_matrix')
                    plt.subplot(plt.subplot(4, 1, 2))
                    plt.plot(time*1e3,y_values_to_plot)
                    plt.tick_params(labelbottom=False)
                    plt.ylabel('y_matrix')
                    plt.subplot(plt.subplot(4, 1, 3))
                    plt.plot(time*1e3,sigmas_to_plot)
                    plt.tick_params(labelbottom=False)
                    plt.ylabel('sigma')
                    plt.subplot(plt.subplot(4, 1, 4))
                    plt.plot(time*1e3,np.array(Y_values_to_plot)/np.array(sigmas_to_plot))
                    plt.ylabel('Y/sigma')
                    plt.xlabel('time')
                    
                    
                    if self.pdf == True:
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(x+1)+' '+'Absorbance at'+'_'+str(wavelengths)+'.pdf', bbox_inches='tight',dpi=self.dpi)
                    if self.png == True:
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(x+1)+' '+'Absorbance at'+'_'+str(wavelengths)+'.png', bbox_inches='tight',dpi=self.dpi)
                    if self.svg == True:
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(x+1)+' '+'Absorbance at'+'_'+str(wavelengths)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)                          
                    
                else:
                      
                    time = exp_dict_list_optimized[x]['experimental_data'][y]['Time']
                    plt.subplot(4, 1, 1)                  
                    plt.plot(time*1e3,Y_values_to_plot)
                    plt.tick_params(labelbottom=False)
                    plt.title('Experiment_'+str(x+1)+'_observable_'+exp_dict_list_optimized[0]['observables'][observable_counter])
                    plt.ylabel('Y_matrix')
                    plt.subplot(plt.subplot(4, 1, 2))
                    plt.plot(time*1e3,y_values_to_plot)
                    plt.tick_params(labelbottom=False)
                    plt.ylabel('y_matrix')
                    plt.subplot(plt.subplot(4, 1, 3))
                    plt.plot(time*1e3,sigmas_to_plot)
                    plt.tick_params(labelbottom=False)
                    plt.ylabel('sigma')
                    plt.subplot(plt.subplot(4, 1, 4))
                    plt.plot(time*1e3,np.array(Y_values_to_plot)/np.array(sigmas_to_plot))
                    plt.ylabel('Y/sigma')
                    plt.xlabel('time')
                    
                    if self.pdf == True:
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(x+1)+'_observable_'+exp_dict_list_optimized[0]['observables'][observable_counter]+'.pdf', bbox_inches='tight',dpi=self.dpi)
                    if self.png == True:
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(x+1)+'_observable_'+exp_dict_list_optimized[0]['observables'][observable_counter]+'.png', bbox_inches='tight',dpi=self.dpi)
                    if self.svg == True:
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(x+1)+'_observable_'+exp_dict_list_optimized[0]['observables'][observable_counter]+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)                       
                    
                observable_counter+=1
        
                
                counter = counter + self.simulation_lengths_of_experimental_data[x][y]
        
        return   


    def shorten_sigma(self):
        flat_list = [item for sublist in self.simulation_lengths_of_experimental_data for item in sublist]
        length = sum(flat_list)
        observables_list = self.target_parameters[length:]
        short_sigma = list(self.Z_matrix)[length:]
        short_sigma = np.array(short_sigma)
        #print(flat_list)
        if bool(self.target_value_rate_constant_csv):
           
           k_target_value_csv = pd.read_csv(os.path.join(self.working_directory,self.target_value_rate_constant_csv)) 
           shape = k_target_value_csv.shape[0]
           slc = len(observables_list) - shape
           observables_list = observables_list[:slc]
           short_sigma = short_sigma[:slc]
           short_sigma = np.array(short_sigma)
        self.short_sigma =  short_sigma
           
        
        return 
            
    def sort_top_uncertainty_weighted_sens(self,top_sensitivity=10):
        S_matrix_copy = copy.deepcopy(self.S_matrix)
        self.shorten_sigma()
        # sigma_csv = self.sigma_uncertainty_weighted_sensitivity_csv
        # real_reaction_uncertainty_csv = self.real_reaction_uncertainty_csv

        if bool(self.real_uncertainty_csv):
            df = pd.read_csv(self.real_uncertainty_csv)
            if bool(self.master_equation_flag):
                for i, m_reaction in enumerate(self.master_equation_reactions):
                    if type(m_reaction) == tuple:
                        for j, t_reaction in enumerate(m_reaction):
                            df = df.drop(df[df['Reaction'] == t_reaction].index) 
                    else:
                        df = df.drop(df[df['Reaction'] == m_reaction].index)   

            A_unc = list(df.iloc[:,1])
            n_unc = list(df.iloc[:,2])
            Ea_unc = list(df.iloc[:,3])
            Sig_1D = list(self.short_sigma.T[0])
            Sig_1D[:len(df)*3] = A_unc + n_unc + Ea_unc
            Sig = np.array(Sig_1D)
            Sig = Sig.reshape((Sig.shape[0],1))
        elif self.sigma_ones==True:
            shape = len(self.short_sigma)
            Sig = np.ones((shape,1))
        else:
            Sig = self.short_sigma
        for pp  in range(np.shape(S_matrix_copy)[1]):
            S_matrix_copy[:,pp] *=Sig[pp]
        sensitivitys =[[] for x in range(len(self.simulation_lengths_of_experimental_data))]
        topSensitivities = [[] for x in range(len(self.simulation_lengths_of_experimental_data))]   
        start=0
        stop = 0
        for x in range(len(self.simulation_lengths_of_experimental_data)):
            for y in range(len(self.simulation_lengths_of_experimental_data[x])):           
                stop = self.simulation_lengths_of_experimental_data[x][y] + start
                temp = S_matrix_copy[start:stop,:]
                sort_s= pd.DataFrame(temp).reindex(pd.DataFrame(temp).abs().max().sort_values(ascending=False).index, axis=1)
                cc=pd.DataFrame(sort_s).iloc[:,:]
                top_five_reactions=cc.columns.values.tolist()
                topSensitivities[x].append(top_five_reactions)
                ccn=pd.DataFrame(cc).to_numpy()
                sensitivitys[x].append(ccn)           
                start = start + self.simulation_lengths_of_experimental_data[x][y]
        return sensitivitys,topSensitivities
    
    def sort_top_Sdx(self,top_sensitivity=10):
        S_matrix_copy = copy.deepcopy(self.S_matrix)
        flat_list = [item for sublist in self.simulation_lengths_of_experimental_data for item in sublist]
        length = sum(flat_list)
        observables_list = self.target_parameters[length:]
        # observables_list = self.sum_arrhenius_observables[length:]

        reactions_in_cti_file = self.exp_dict_list_original[0]['simulation'].processor.solution.reaction_equations()
        flatten = lambda *n: (e for a in n
            for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))        
        flattened_master_equation_reaction_list = list(flatten(self.master_equation_reactions))     
        A_n_Ea_length = int((len(reactions_in_cti_file) - len(flattened_master_equation_reaction_list))*3)
        
        X = self.X
        for pp  in range(np.shape(S_matrix_copy)[1]):
            S_matrix_copy[:,pp] *=X[pp]
        sensitivitys =[[] for x in range(len(self.simulation_lengths_of_experimental_data))]
        topSensitivities = [[] for x in range(len(self.simulation_lengths_of_experimental_data))]   
        start=0
        stop = 0
        for x in range(len(self.simulation_lengths_of_experimental_data)):
            for y in range(len(self.simulation_lengths_of_experimental_data[x])):           
                stop = self.simulation_lengths_of_experimental_data[x][y] + start
                temp = S_matrix_copy[start:stop,:]
                temp_df = pd.DataFrame(temp, columns=observables_list[:len(temp.T)])
                
                if self.k_target_value_S_matrix.any():
                    k_target_length = len(self.k_target_value_S_matrix)
                    num_rxns = len(observables_list[:-k_target_length])/3
                
                A_temp_df = temp_df.iloc[:,:int(A_n_Ea_length/3)]
                n_temp_df = temp_df.iloc[:,int(A_n_Ea_length/3):int(2*A_n_Ea_length/3)]
                Ea_temp_df = temp_df.iloc[:,int(2*A_n_Ea_length/3):int(3*A_n_Ea_length/3)]


                if len(A_temp_df.T) != len(n_temp_df.T):
                    print('Length of A_temp_df and n_temp_df not equal')
                if len(A_temp_df.T) != len(Ea_temp_df.T):
                    print('Length of A_temp_df and Ea_temp_df not equal')
                if len(n_temp_df.T) != len(Ea_temp_df.T):
                    print('Length of n_temp_df and Ea_temp_df not equal')                                        

                sum_temp_list = []
                col_reactions = []
                for i, col in enumerate(A_temp_df.columns):
                    col_reactions.append(col[2:])
                    sum_temp_list.append(list(A_temp_df.iloc[:, i] + n_temp_df.iloc[:, i]+ Ea_temp_df.iloc[:, i]))
                    
                new_temp_array = np.concatenate((np.array(sum_temp_list),temp_df.iloc[:,A_n_Ea_length:].to_numpy().T))
                new_temp_df = pd.DataFrame(new_temp_array.T)
                
                new_temp_df_columns = col_reactions + list(temp_df.iloc[:,A_n_Ea_length:].columns)
                # new_temp_df = pd.concat([sum_temp_df, temp_df.iloc[:,A_n_Ea_length:]], axis=1)
                # self.summed_observable_list = new_temp_df.columns
                sort_s= new_temp_df.reindex(new_temp_df.abs().max().sort_values(ascending=False).index, axis=1)
                sort_s.columns = [new_temp_df_columns[i] for i in list(sort_s.columns)]
                # sort_s= new_temp_df.loc[new_temp_df.abs().max().sort_values(ascending=False).index]
                # print(new_temp_df)
                # sort_s= pd.DataFrame(temp).reindex(pd.DataFrame(temp).abs().max().sort_values(ascending=False).index, axis=1)
                cc=pd.DataFrame(sort_s).iloc[:,:]
                top_five_reactions=cc.columns.values.tolist()
                topSensitivities[x].append(top_five_reactions)
                ccn=pd.DataFrame(cc).to_numpy()
                sensitivitys[x].append(ccn)           
                start = start + self.simulation_lengths_of_experimental_data[x][y]
                
                
                
        return sensitivitys,topSensitivities,new_temp_df_columns,len(A_temp_df.T)
    
    def getting_time_profiles_for_experiments(self, exp_dict_list_optimized):
        time_profiles =[[] for x in range(len(self.simulation_lengths_of_experimental_data))]
        observables = [[] for x in range(len(self.simulation_lengths_of_experimental_data))]
        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + 
                                          exp['concentration_observables'] +
                                          exp['ignition_delay_observables']):
                if observable == None:
                    continue                                
                if observable in exp['mole_fraction_observables']:
                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']):
                        time_profiles[i].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        observables[i].append(observable)
                        observable_counter+=1
                    elif re.match('[Jj][Ss][Rr]',exp['simulation_type']):
                        time_profiles[i].append(exp['experimental_data'][observable_counter]['Temperature'])
                        observables[i].append(observable)
                        observable_counter+=1
                    elif re.match('[Ff]low[ -][Rr][eactor]',exp['simulation_type']):
                        time_profiles[i].append(exp['experimental_data'][observable_counter]['Temperature'])
                        observables[i].append(observable)
                        observable_counter+=1                        
                elif observable in exp['concentration_observables']:
                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']):
                        time_profiles[i].append(exp['experimental_data'][observable_counter]['Time']*1e3)        
                        observables[i].append(observable)                                
                        observable_counter+=1
                    elif re.match('[Jj][Ss][Rr]',exp['simulation_type']):
                        time_profiles[i].append(exp['experimental_data'][observable_counter]['Temperature'])        
                        observables[i].append(observable)                                
                        observable_counter+=1
                    elif re.match('[Ff]low[ -][Rr][eactor]',exp['simulation_type']):
                        time_profiles[i].append(exp['experimental_data'][observable_counter]['Temperature'])
                        observables[i].append(observable)
                        observable_counter+=1    
                elif observable in exp['ignition_delay_observables']:
                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']):
                        time_profiles[i].append(exp['experimental_data'][observable_counter]['temperature'])        
                        observables[i].append(observable)                                
                        observable_counter+=1
                      
                                            
                        
            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    time_profiles[i].append(exp['absorbance_experimental_data'][k]['time']*1e3)       
                    observables[i].append('Absorbance_'+str(wl))                                 
        self.time_profiles = time_profiles
        self.observable_list = observables
        return time_profiles
    
    
    def get_sum_arrhenius_observables(self):

        gas = ct.Solution(self.new_cti)
        reaction_equations = gas.reaction_equations()
        
        if bool(self.real_uncertainty_csv):
            
            df = pd.read_csv(self.real_uncertainty_csv)
            for i, m_reaction in enumerate(self.master_equation_reactions):
                if type(m_reaction) == tuple:
                    for j, t_reaction in enumerate(m_reaction):
                        df = df.drop(df[df['Reaction'] == t_reaction].index) 
                else:
                    df = df.drop(df[df['Reaction'] == m_reaction].index)            
            A_unc = list(df.iloc[:,1])
            n_unc = list(df.iloc[:,2])
            Ea_unc = list(df.iloc[:,3])
            Sig_1D = list(self.short_sigma.T[0])
            Sig_1D[:len(df)*3] = A_unc + n_unc + Ea_unc
            Sig = np.array(Sig_1D)
            Sig = Sig.reshape((Sig.shape[0],1))            
            self.sigma_list = Sig
            
        flat_list = [item for sublist in self.simulation_lengths_of_experimental_data for item in sublist]
        length = sum(flat_list)
        observables_list = self.target_parameters[length:]
        
        # observables_list = self.active_parameters
        
        if bool(self.target_value_rate_constant_csv):
            
            k_target_value_csv = pd.read_csv(os.path.join(self.working_directory,self.target_value_rate_constant_csv)) 
            shape = k_target_value_csv.shape[0]
            slc = len(observables_list) - shape
            observables_list = observables_list[:slc]

        sum_arrhenius_observables = []

        for obs in observables_list:
            lst = obs.split(' ')
            
            if lst[0] =='A':
                sum_arrhenius_observables.append('k '+ lst[1])
            elif lst[0] =='n':
                pass               
            elif lst[0] =='Ea':
                pass
            else:
                sum_arrhenius_observables.append(obs)
        self.sum_arrhenius_observables = sum_arrhenius_observables          
                                     
        return self.sum_arrhenius_observables
    
    
    def plotting_uncertainty_weighted_sens(self):
          
        print('\n')
        print('--------------------------------------------------------------------------')
        print('Observable UWSA Plots')
        print('--------------------------------------------------------------------------')
        
           
        
        sensitivities,top_sensitivities = self.sort_top_uncertainty_weighted_sens()
        # self.get_observables_list()
        # sum_arrhenius_observables = self.get_sum_arrhenius_observables()
        observables_list_for_legend = self.active_parameters
        if bool(self.real_uncertainty_csv):
            df = pd.read_csv(self.real_uncertainty_csv)
            if bool(self.master_equation_flag):
                for i, m_reaction in enumerate(self.master_equation_reactions):
                    if type(m_reaction) == tuple:
                        for j, t_reaction in enumerate(m_reaction):
                            df = df.drop(df[df['Reaction'] == t_reaction].index) 
                    else:
                        df = df.drop(df[df['Reaction'] == m_reaction].index)            
            A_unc = list(df.iloc[:,1])
            n_unc = list(df.iloc[:,2])
            Ea_unc = list(df.iloc[:,3])
            Sig_1D = list(self.short_sigma.T[0])
            Sig_1D[:len(df)*3] = A_unc + n_unc + Ea_unc
            Sig = np.array(Sig_1D)
            Sig = Sig.reshape((Sig.shape[0],1))            
            self.sigma_list = Sig            
            sigma_list = self.sigma_list
        else:
            sigma_list = list(self.short_sigma)
        time_profiles = self.getting_time_profiles_for_experiments(self.exp_dict_list_optimized)
        list_of_experiment_observables = self.observable_list

        def subplot_function(number_of_observables_in_simulation,time_profiles,sensitivities,top_sensitivity_single_exp,observables_list_for_legend,list_of_experiment_observables,experiment_number):
                             
            for plot_number in range(number_of_observables_in_simulation):
                
                print(os.path.basename(self.files_to_include[0][experiment_number][0])[:-5] + ' ' + list_of_experiment_observables[plot_number])
                
                plt.figure()
                UWSA_df = pd.DataFrame()
                UWSA_df['x'] = pd.Series(time_profiles[plot_number])
                for c,top_columns in enumerate(top_sensitivity_single_exp[plot_number]):
                    plt.title('Experiment_'+str(experiment_number+1) + ' (' + os.path.basename(self.files_to_include[0][experiment_number][0])[:-5] + ')')
                    if c < 10:
                        plt.plot(time_profiles[plot_number],sensitivities[plot_number][:,c],label = observables_list_for_legend[top_columns] +' '+str(sigma_list[top_columns])) 
                    observable_ylabel = list_of_experiment_observables[plot_number]
                    observable_ylabel_string = list(observable_ylabel)
                    for i, oys in enumerate(observable_ylabel_string):
                        if oys.isdigit() == True:
                            if i == 0:
                                pass
                            elif observable_ylabel_string[i-1] == ' ':
                                pass
                            else:
                                observable_ylabel_string[i] = '_{' + str(oys) + '}'
                    observable_ylabel_transformed = "".join(observable_ylabel_string)                             
                    if 'Absorbance' in observable_ylabel:
                        plt.ylabel(observable_ylabel)
                    else:   
                        plt.ylabel(r'$\frac{\partial( \rm'+observable_ylabel_transformed+r')}{\partial(\rm x_j)} \rm \sigma_j$')
                    plt.legend(ncol=1, loc='upper left',bbox_to_anchor=(1,1))
                    UWSA_df[observables_list_for_legend[top_columns] +' '+str(sigma_list[top_columns])] = pd.Series(sensitivities[plot_number][:,c])   
                UWSA_df.to_csv(self.out_path+'/'+'Experiment'+ '_' +str(experiment_number+1)+'_UWSA'+'_'+str(list_of_experiment_observables[plot_number])+'.csv',index=False)   
                if self.simulation_run==None:
                    if self.pdf == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(experiment_number+1)+'_UWSA'+'_'+str(list_of_experiment_observables[plot_number])+'.pdf', bbox_inches='tight',dpi=self.dpi)
                    if self.png == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(experiment_number+1)+'_UWSA'+'_'+str(list_of_experiment_observables[plot_number])+'.png', bbox_inches='tight',dpi=self.dpi)
                    if self.svg == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(experiment_number+1)+'_UWSA'+'_'+str(list_of_experiment_observables[plot_number])+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)   
                else:
                    plt.title('Experiment_'+str(self.simulation_run) + ' (' + os.path.basename(self.files_to_include[0][experiment_number][0])[:-5] + ')')
                    if self.pdf == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(self.simulation_run)+'_UWSA'+'_'+str(list_of_experiment_observables[plot_number])+'.pdf', bbox_inches='tight',dpi=self.dpi)
                    if self.png == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(self.simulation_run)+'_UWSA'+'_'+str(list_of_experiment_observables[plot_number])+'.png', bbox_inches='tight',dpi=self.dpi)
                    if self.svg == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(self.simulation_run)+'_UWSA'+'_'+str(list_of_experiment_observables[plot_number])+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)    
                
                self.obs_UWSA_loop.update()     
                
                   
        self.obs_UWSA_loop = self.manager.counter(total=len(self.obs_list), desc='Observable UWSA Plots:', unit='plots', color='blue')                        
        for x in range(len(sensitivities)):            
            number_of_observables_in_simulation = len(sensitivities[x])
            subplot_function(number_of_observables_in_simulation,time_profiles[x],sensitivities[x],top_sensitivities[x],observables_list_for_legend,list_of_experiment_observables[x],x)
        return 
            
    def plotting_Sdx(self):
         
        print('\n')
        print('--------------------------------------------------------------------------')
        print('Observable Sdx Plots')
        print('--------------------------------------------------------------------------')
        
        sensitivities,top_sensitivities,summed_observable_list,num_rxn = self.sort_top_Sdx()
        observables_list_for_legend = self.active_parameters
        # sum_arrhenius_observables = self.get_sum_arrhenius_observables()
        if bool(self.real_uncertainty_csv):
            df = pd.read_csv(self.real_uncertainty_csv)
            if bool(self.master_equation_flag):
                for i, m_reaction in enumerate(self.master_equation_reactions):
                    if type(m_reaction) == tuple:
                        for j, t_reaction in enumerate(m_reaction):
                            df = df.drop(df[df['Reaction'] == t_reaction].index) 
                    else:
                        df = df.drop(df[df['Reaction'] == m_reaction].index)            
            A_unc = list(df.iloc[:,1])
            n_unc = list(df.iloc[:,2])
            Ea_unc = list(df.iloc[:,3])
            Sig_1D = list(self.short_sigma.T[0])
            Sig_1D[:len(df)*3] = A_unc + n_unc + Ea_unc
            Sig = np.array(Sig_1D)
            Sig = Sig.reshape((Sig.shape[0],1))            
            self.sigma_list = Sig            
            sigma_list = self.sigma_list
        else:
            sigma_list = list(self.short_sigma)
        time_profiles = self.getting_time_profiles_for_experiments(self.exp_dict_list_optimized)
        list_of_experiment_observables = self.observable_list
        X = self.X

        def subplot_function(number_of_observables_in_simulation,time_profiles,sensitivities,top_sensitivity_single_exp,observables_list_for_legend,list_of_experiment_observables,experiment_number):
                        
            for plot_number in range(number_of_observables_in_simulation):
                print(os.path.basename(self.files_to_include[0][experiment_number][0])[:-5] + ' ' + list_of_experiment_observables[plot_number])
                plt.figure()
                Sdx_df = pd.DataFrame()
                Sdx_df['x'] = pd.Series(time_profiles[plot_number])
                # num_rxn = top_sensitivity_single_exp[plot_number]
                for c,top_columns in enumerate(top_sensitivity_single_exp[plot_number]):

                    top_columns_index = summed_observable_list.index(top_columns)

                    plt.title('Experiment_'+str(experiment_number+1) + ' (' + os.path.basename(self.files_to_include[0][experiment_number][0])[:-5] + ')')
                    if c < 10:
                        # if top_columns.split('_')[0] == 'k':
                        if top_columns_index < num_rxn:
                            X_A = X[top_columns_index][0]
                            X_n = X[top_columns_index+num_rxn][0]
                            X_Ea = X[top_columns_index+2*num_rxn][0]
                            summed_X = X_A + X_n + X_Ea
                            sigma_A = sigma_list[top_columns_index][0]
                            sigma_n = sigma_list[top_columns_index+num_rxn][0]
                            sigma_Ea = sigma_list[top_columns_index+2*num_rxn][0]

                            plt.plot(time_profiles[plot_number],sensitivities[plot_number][:,c],
                            label = summed_observable_list[top_columns_index] +' [A:'+str(sigma_A)+', n:'+str(sigma_n)+', Ea:'+str(sigma_Ea)+', '+str(summed_X)+']') 
                        else:
                            plt.plot(time_profiles[plot_number],sensitivities[plot_number][:,c],
                            label = summed_observable_list[top_columns_index] +' ['+str(sigma_list[top_columns_index+2*num_rxn][0])+', '+str(X[top_columns_index+2*num_rxn][0])+']') 
                    observable_ylabel = list_of_experiment_observables[plot_number]
                    observable_ylabel_string = list(observable_ylabel)
                    for i, oys in enumerate(observable_ylabel_string):
                        if oys.isdigit() == True:
                            if i == 0:
                                pass
                            elif observable_ylabel_string[i-1] == ' ':
                                pass
                            else:
                                observable_ylabel_string[i] = '_{' + str(oys) + '}'
                    observable_ylabel_transformed = "".join(observable_ylabel_string)                             
                    if 'Absorbance' in observable_ylabel:
                        plt.ylabel(observable_ylabel)
                    else:   
                        plt.ylabel(r'$\frac{\partial( \rm'+observable_ylabel_transformed+r')}{\partial(\rm x_j)} \rm \Delta x_j$')
                    plt.legend(ncol=1, loc='upper left',bbox_to_anchor=(1,1))
                    if top_columns.split('_')[0] == 'k':
                        X_A = X[top_columns_index][0]
                        X_n = X[top_columns_index+num_rxn][0]
                        X_Ea = X[top_columns_index+2*num_rxn][0]
                        summed_X = X_A + X_n + X_Ea
                        sigma_A = sigma_list[top_columns_index][0]
                        sigma_n = sigma_list[top_columns_index+num_rxn][0]
                        sigma_Ea = sigma_list[top_columns_index+2*num_rxn][0]
                        Sdx_df[summed_observable_list[top_columns_index] +' [A:'+str(sigma_A)+', n:'+str(sigma_n)+', Ea:'+str(sigma_Ea)+', '+str(summed_X)+']'] = pd.Series(sensitivities[plot_number][:,c])  
                    else:
                        Sdx_df[summed_observable_list[top_columns_index] +' ['+str(sigma_list[top_columns_index+2*num_rxn][0])+', '+str(X[top_columns_index+2*num_rxn][0])+']'] = pd.Series(sensitivities[plot_number][:,c])                     
                    # Sdx_df[sum_arrhenius_observables[top_columns_index] +' '+str(sigma_list[top_columns])] = pd.Series(sensitivities[plot_number][:,c])   
                Sdx_df.to_csv(self.out_path+'/'+'Experiment'+ '_' +str(experiment_number+1)+'_Sdx'+'_'+str(list_of_experiment_observables[plot_number])+'.csv',index=False)   
                if self.simulation_run==None:
                    if self.pdf == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(experiment_number+1)+'_Sdx'+'_'+str(list_of_experiment_observables[plot_number])+'.pdf', bbox_inches='tight',dpi=self.dpi)
                    if self.png == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(experiment_number+1)+'_Sdx'+'_'+str(list_of_experiment_observables[plot_number])+'.png', bbox_inches='tight',dpi=self.dpi)
                    if self.svg == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(experiment_number+1)+'_Sdx'+'_'+str(list_of_experiment_observables[plot_number])+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)   
                else:
                    plt.title('Experiment_'+str(self.simulation_run) + ' (' + os.path.basename(self.files_to_include[0][experiment_number][0])[:-5] + ')')
                    if self.pdf == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(self.simulation_run)+'_Sdx'+'_'+str(list_of_experiment_observables[plot_number])+'.pdf', bbox_inches='tight',dpi=self.dpi)
                    if self.png == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(self.simulation_run)+'_Sdx'+'_'+str(list_of_experiment_observables[plot_number])+'.png', bbox_inches='tight',dpi=self.dpi)
                    if self.svg == True:
                        plt.savefig(self.out_path+'/'+'Experiment'+ '_' +str(self.simulation_run)+'_Sdx'+'_'+str(list_of_experiment_observables[plot_number])+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)           
                
                self.obs_Sdx_loop.update() 
                  
        self.obs_Sdx_loop = self.manager.counter(total=len(self.obs_list), desc='Observable Sdx Plots:', unit='plots', color='blue')                    
        for x in range(len(sensitivities)):            
            number_of_observables_in_simulation = len(sensitivities[x])
            subplot_function(number_of_observables_in_simulation,time_profiles[x],sensitivities[x],top_sensitivities[x],observables_list_for_legend,list_of_experiment_observables[x],x)
                       
        return 
        
        
    def plotting_normal_distributions(self,
                                      paramter_list,
                                      optimized_cti_file='',
                                      pdf_distribution_file='',
                                      shock_tube_instance=None):
        
        all_parameters = shock_tube_instance.posterior_diag_df['parameter'].tolist()
        df = shock_tube_instance.posterior_diag_df
        gas_optimized = ct.Solution(optimized_cti_file)
        
        for parameter in paramter_list:
            indx = all_parameters.index(parameter)
            variance = df['value'][indx]
            if parameter[0]=='A' or parameter[0]=='n' or parameter[0]=='E':
                letter,number = parameter.split('_')
                number = int(number)
                if 'ElementaryReaction' in str(type(gas_optimized.reaction(number))):
                    A=gas_optimized.reaction(number).rate.pre_exponential_factor
                    n=gas_optimized.reaction(number).rate.temperature_exponent
                    Ea=gas_optimized.reaction(number).rate.activation_energy
                if 'FalloffReaction' in str(type(gas_optimized.reaction(number))):
                    A=gas_optimized.reaction(number).high_rate.pre_exponential_factor
                    n=gas_optimized.reaction(number).high_rate.temperature_exponent
                    Ea=gas_optimized.reaction(number).high_rate.activation_energy
                if 'ThreeBodyReaction' in   str(type(gas_optimized.reaction(number))):
                    A=gas_optimized.reaction(number).rate.pre_exponential_factor
                    n=gas_optimized.reaction(number).rate.temperature_exponent
                    Ea=gas_optimized.reaction(number).rate.activation_energy
            else: 
                letter = None
                
            if letter =='A':
                mu = np.log(A*1000)
                sigma = math.sqrt(variance)
                sigma = sigma
                
            elif letter == 'n':
                mu = n
                sigma = math.sqrt(variance)
                #sigma = sigma/2
            elif letter == 'Ea':
                mu=Ea/1000/4.184            
                sigma = math.sqrt(variance)
                sigma = sigma*ct.gas_constant/(1000*4.184)
                #sigma = sigma/2
            else:
                mu= 0 
                sigma = math.sqrt(variance)
                

            
            
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.figure()
            plt.plot(x, stats.norm.pdf(x, mu, sigma))
            plt.xlabel(parameter)
            plt.ylabel('pdf')
            
            plt.savefig(self.out_path+'/'+parameter+'_distribution'+'.pdf',bbox_inches='tight',dpi=1000)
            plt.savefig(self.out_path+'/'+parameter+'_distribution'+'.png',bbox_inches='tight',dpi=1000)
            plt.savefig(self.out_path+'/'+parameter+'_distribution'+'.svg',bbox_inches='tight',dpi=1000,transparent=True)

            if bool(pdf_distribution_file):
                df2 = pd.read_csv(pdf_distribution_file)
                #temp = np.log(np.exp(df2[parameter].values)/9.33e13)
                #plt.plot(temp,df2['pdf_'+parameter])
                plt.plot(df2[parameter],df2['pdf_'+parameter])
                plt.savefig(self.out_path+'/'+parameter+'_distribution'+'.pdf',bbox_inches='tight',dpi=1000)
                plt.savefig(self.out_path+'/'+parameter+'_distribution'+'.png',bbox_inches='tight',dpi=1000)
                plt.savefig(self.out_path+'/'+parameter+'_distribution'+'.svg',bbox_inches='tight',dpi=1000,transparent=True)

    

    
    def plotting_joint_normal_distributions(self,
                                            coupled_parameters,
                                            optimized_cti_file='',
                                            joint_data_csv=''):
                
        all_parameters = self.shock_tube_instance.posterior_diag_df['parameter'].tolist()
        df = self.shock_tube_instance.posterior_diag_df
        gas_optimized = ct.Solution(optimized_cti_file)
        for couple in coupled_parameters:
            indx1 = all_parameters.index(couple[0])
            indx2 = all_parameters.index(couple[1])
            variance1 = df['value'][indx1]
            variance2 = df['value'][indx2]
            if couple[0][0]=='A' or couple[0][0]=='n' or couple[0][0]=='E':
           
                letter1,number1 = couple[0].split('_')
                number1 = int(number1)
                number1_covariance = number1
                if letter1=='n':
                    number1_covariance = number1+len(gas_optimized.reaction_equations())
                if letter1=='Ea':
                    number1_covariance = number1+len(gas_optimized.reaction_equations())*2
                    

                    
                if 'ElementaryReaction' in str(type(gas_optimized.reaction(number1))):
                    A1=gas_optimized.reaction(number1).rate.pre_exponential_factor
                    n1=gas_optimized.reaction(number1).rate.temperature_exponent
                    Ea1=gas_optimized.reaction(number1).rate.activation_energy
                if 'FalloffReaction' in str(type(gas_optimized.reaction(number1))):
                    A1=gas_optimized.reaction(number1).high_rate.pre_exponential_factor
                    n1=gas_optimized.reaction(number1).high_rate.temperature_exponent
                    Ea1=gas_optimized.reaction(number1).high_rate.activation_energy
                if 'ThreeBodyReaction' in   str(type(gas_optimized.reaction(number1))):
                    A1=gas_optimized.reaction(number1).rate.pre_exponential_factor
                    n1=gas_optimized.reaction(number1).rate.temperature_exponent
                    Ea1=gas_optimized.reaction(number1).rate.activation_energy
            else:
                letter1 = None
                mu1=0
                mu_x=0
                sigma1= math.sqrt(variance1)
                number1_covariance = indx1
                variance_x = variance1
            if couple[1][0]=='A' or couple[1][0]=='n' or couple[1][0]=='E':
                letter2,number2 = couple[1].split('_')
                number2 = int(number2)
                number2_covariance = number2
                if letter2=='n':
                    number2_covariance = number2+len(gas_optimized.reaction_equations())
                if letter2 == 'Ea':
                    number2_covariance = number2+len(gas_optimized.reaction_equations())*2
                    
                if 'ElementaryReaction' in str(type(gas_optimized.reaction(number2))):   
                    A2=gas_optimized.reaction(number2).rate.pre_exponential_factor
                    n2=gas_optimized.reaction(number2).rate.temperature_exponent
                    Ea2=gas_optimized.reaction(number2).rate.activation_energy 
                if 'FalloffReaction' in str(type(gas_optimized.reaction(number2))):
                    A2=gas_optimized.reaction(number2).high_rate.pre_exponential_factor
                    n2=gas_optimized.reaction(number2).high_rate.temperature_exponent
                    Ea2=gas_optimized.reaction(number2).high_rate.activation_energy
                if 'ThreeBodyReaction' in   str(type(gas_optimized.reaction(number2))):
                    A2=gas_optimized.reaction(number2).rate.pre_exponential_factor
                    n2=gas_optimized.reaction(number2).rate.temperature_exponent
                    Ea2=gas_optimized.reaction(number2).rate.activation_energy
            else:
                mu_y=0
                mu2=0
                letter2=None
                variance_y = variance2
                sigma = math.sqrt(variance2)
                number2_covariance = indx2

            
            
            covariance_couple = self.covariance[number1_covariance,number2_covariance]
           # print(number1_covariance,number2_covariance)
            #covariance_couple = .00760122
            if letter1 =='A':
                mu1 = np.log(A1*1000)
                mu_x = mu1
                variance_x = variance1
                sigma = np.sqrt(variance_x)
                
                #sigma = np.exp(sigma)
                #sigma = sigma*1000
                #sigma = np.log(sigma)
                #sigma = sigma/2
                variance_x = sigma**2
                #convert to chemkin units
            if letter1 == 'n':
                mu1 = n1
                mu_x = mu1
                variance_x = variance1
                sigma = np.sqrt(variance_x)
                #sigma = sigma/2
                variance_x = sigma**2
            if letter1 == 'Ea':
                mu1=Ea1/1000/4.184 
                mu_x = mu1                
                variance_x = variance1
                sigma = math.sqrt(variance_x)
                sigma = sigma*ct.gas_constant/(1000*4.184)
                #sigma = sigma/2
                variance_x = sigma**2
   
            
            if letter2 =='A':
                mu2 = np.log(A2*1000)
                mu_y = mu2
                variance_y = variance2
                sigma = np.sqrt(variance_y)
                sigma = sigma
                #sigma = np.exp(sigma)
                #sigma = sigma*1000
                #sigma = np.log(sigma)
                #sigma = sigma/2
                variance_y = sigma**2
                #convert to chemkin units
            if letter2 == 'n':
                mu2 = n2
                mu_y = mu2
                variance_y = variance2      
                sigma = np.sqrt(variance_y)
                #sigma = sigma/2
                variance_y = sigma**2
                
            if letter2 == 'Ea':
                mu2 = Ea2/1000/4.184 
                mu_y = mu2
                variance_y = variance2
                sigma = math.sqrt(variance_y)
                sigma = sigma*ct.gas_constant/(1000*4.184)
                #sigma = sigma/2
                variance_y = sigma**2

            
            if letter2 =='Ea' or letter1 == 'Ea':
                covariance_couple = covariance_couple*ct.gas_constant/(1000*4.184)
                if letter2=='Ea' and letter1=='Ea':
                    covariance_couple = np.sqrt(covariance_couple)
                    covariance_couple = covariance_couple*ct.gas_constant/(1000*4.184)
                    covariance_couple = covariance_couple**2

            #if letter1=='A' or letter2=='A':
                #covariance_couple = np.exp(covariance_couple)
                #covariance_couple  = covariance_couple/2
                #covariance_couple = np.log(covariance_couple)
                
                
           
            x = np.linspace(mu1 - 3*np.sqrt(variance_x), mu1 + 3*np.sqrt(variance_x),1000)
            y = np.linspace(mu2 - 3*np.sqrt(variance_y), mu2 + 3*np.sqrt(variance_y),1000)
            
            #x = np.linspace(mu1 - 2*np.sqrt(variance_x), mu1 + 2*np.sqrt(variance_x),1000)
            #y = np.linspace(mu2 - 2*np.sqrt(variance_y), mu2 + 2*np.sqrt(variance_y),1000)
            #TEST

            
            
            X,Y = np.meshgrid(x,y)
            #X, Y = np.meshgrid(x,y)


            
            
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X; pos[:, :, 1] = Y
            rv = multivariate_normal([mu_x, mu_y], [[variance_x, covariance_couple], [covariance_couple, variance_y]])
            # print(couple,[mu_x, mu_y], [[variance_x, covariance_couple], [covariance_couple, variance_y]])
            fig = plt.figure()

            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
            ax.set_xlabel(couple[0])
            ax.set_ylabel(couple[1])
            ax.set_zlabel('Z axis')
            plt.show()

            additional_dictionary = {'A_5':{'reaction':'H2O2 + M = 2OH + M','our_value':np.log(4.99999e8),'hong_value':np.log(5.60e8)},
                                     'A_6':{'reaction':'OH + H2O2 = H2O + HO2','our_value':np.log(5624842396127.52),'hong_value':np.log(6.93e12)},
                                     'A_7':{'reaction': 'OH + HO2 = H2O + O2' , 'our_value':np.log(16646221572429.6),'hong_value':np.log(1.82e13)},
                                     'A_8':{'reaction':'2HO2 = H2O2 + O2','our_value':np.log(806831822530.157),'hong_value':np.log(3.17e12)},
                                     'A_11':{'reaction':'2OH = H2O + O','our_value':np.log(1730749579423.63),'hong_value':np.log(2.355e12)},
                                     'Sigma_1':{'reaction':'sigma H2O2','our_value':-.03846,'hong_value':0},
                                     'Sigma_2':{'reaction':'sigma_HO2','our_value':.0721,'hong_value':0}}
           
            additional_dictionary = {'A_5':{'reaction':'H2O2 + M = 2OH + M','our_value':np.log(4.99999e8),'hong_value':np.log(5.60e8)},
                                     'A_6':{'reaction':'OH + H2O2 = H2O + HO2','our_value':np.log(5917630773605.197),'hong_value':np.log(6.93e12)},
                                     'A_7':{'reaction': 'OH + HO2 = H2O + O2' , 'our_value':np.log(18236369573049.9),'hong_value':np.log(1.82e13)},
                                     'A_8':{'reaction':'2HO2 = H2O2 + O2','our_value':np.log(863643827140.3533),'hong_value':np.log(3.17e12)},
                                     'A_11':{'reaction':'2OH = H2O + O','our_value':np.log(1734217478483.0261),'hong_value':np.log(2.355e12)},
                                     'Sigma_1':{'reaction':'sigma H2O2','our_value':-.03846,'hong_value':0},
                                     'Sigma_2':{'reaction':'sigma_HO2','our_value':.0721,'hong_value':0}}
            
            error_dictonary =  {'A_5':{'reaction':'H2O2 + M = 2OH + M','our_value':None,'hong_value':0},
                                     'A_6':{'reaction':'OH + H2O2 = H2O + HO2','our_value':np.log(5624842396127.52),'hong_value':0},
                                     'A_7':{'reaction': 'OH + HO2 = H2O + O2' , 'our_value':np.log(16646221572429.6),'hong_value':0},
                                     'A_8':{'reaction':'2HO2 = H2O2 + O2','our_value':np.log(806831822530.157),'hong_value':0},
                                     'A_11':{'reaction':'2OH = H2O + O','our_value':np.log(1730749579423.63),'hong_value':0},
                                     'Sigma_1':{'reaction':'sigma H2O2','our_value':-.03846,'hong_value':0},
                                     'Sigma_2':{'reaction':'sigma_HO2','our_value':.0721,'hong_value':0}}
            Z = rv.pdf(pos)
            plt.figure()
            levels = [.65,.95,.99]
            #contour = plt.contour(X, Y, Z, levels, colors='k')
            #plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
           # plt.colorbar(contour_filled)
            plt.contour(X,Y,Z)
            plt.xlabel(couple[0])
            plt.ylabel(couple[1])

        
#            
#            plt.figure()
#            
#            Z_test = mlab.bivariate_normal(X, Y,np.sqrt(covariance_couple),np.sqrt(covariance_couple),mu_x,mu_y)
#            z1 = mlab.bivariate_normal(0, 1 * np.sqrt(covariance_couple), np.sqrt(covariance_couple), np.sqrt(covariance_couple),mu_x,mu_y)
#            z2 = mlab.bivariate_normal(0, 2 * np.sqrt(covariance_couple), np.sqrt(covariance_couple), np.sqrt(covariance_couple),mu_x,mu_y)
#            z3 = mlab.bivariate_normal(0, 3 * np.sqrt(covariance_couple), np.sqrt(covariance_couple), np.sqrt(covariance_couple),mu_x,mu_y)
#            
##plot Gaussian:
#            im = plt.imshow(Z_test,interpolation='bilinear', origin='lower',
#                 extent=(-50,50,-50,50),cmap=cm.gray)
##Plot contours at whatever z values we want:
#            CS = plt.contour(Z_test, [z1, z2, z3], origin='lower', extent=(-50,50,-50,50),colors='red')            
            
            
            
            
            
            if bool(additional_dictionary):
                plt.xlabel(additional_dictionary[couple[0]]['reaction'])
                plt.ylabel(additional_dictionary[couple[1]]['reaction'])
                x_error = (additional_dictionary[couple[0]]['hong_value'])*(error_dictonary[couple[0]]['hong_value'])
                print(x_error,'this is the x error')

                y_error = (additional_dictionary[couple[1]]['hong_value'])*(error_dictonary[couple[1]]['hong_value'])
                print(y_error,'this is the y error')
                plt.errorbar(additional_dictionary[couple[0]]['hong_value'],additional_dictionary[couple[1]]['hong_value'],xerr=x_error,yerr=y_error)
                
                plt.scatter(additional_dictionary[couple[0]]['hong_value'],additional_dictionary[couple[1]]['hong_value'],zorder=4,label='Hong Values From Table')
                
                plt.scatter(additional_dictionary[couple[0]]['our_value'],additional_dictionary[couple[1]]['our_value'],zorder=4,marker='x',label='MSI Values')
                plt.legend()

            if bool(joint_data_csv):
                df2 = pd.read_csv(joint_data_csv)
                #plt.figure()
                plt.scatter(df2[couple[0]], df2[couple[1]])
                
                plt.savefig(self.out_path+'/'+couple[0]+'_'+couple[1]+'_distribution'+'.pdf',bbox_inches='tight',dpi=1000)
                plt.savefig(self.out_path+'/'+couple[0]+'_'+couple[1]+'_distribution'+'.png',bbox_inches='tight',dpi=1000)
                plt.savefig(self.out_path+'/'+couple[0]+'_'+couple[1]+'_distribution'+'.svg',bbox_inches='tight',dpi=1000,transparent=True)
    def plotting_physical_model_parameter_distributions(self,
                           paramter_list,
                           shock_tube_instance,
                           optimized_X,
                           original_experimental_conditions,
                           T_uncertainty=.005,
                           P_uncertainty=.01,
                           X_uncertainty=.025,
                           directory_to_save_images='',
                           experiments_want_to_plot_data_from=[]):
        
        if bool(experiments_want_to_plot_data_from)==False:
            experiments_want_to_plot_data_from = np.arange(0,len(self.exp_dict_list_optimized))
        try:
            all_parameters = shock_tube_instance.posterior_diag_df['parameter'].tolist()
        except:
            all_parameters = shock_tube_instance.prior_diag_df['parameter'].tolist()

        parameter_groups = ['T','P','Time']
        #print(all_parameters)
        list_of_species = []
        for parameter in all_parameters:
            if parameter[0] == 'X':
                list_of_species.append(parameter.split('_')[1])
        
        output = []
        for x in list_of_species:
            if x not in output:
                output.append(x)
        parameter_groups = parameter_groups + output
        
        for parameter in parameter_groups:
            temp_list = []
            parameter_counter = 0
            for i,p in enumerate(all_parameters):
                if parameter == 'T':
                    if p[0] == 'T' and p[1] != 'i':
                        yaml_file = int(p.split('_')[2])
                        if parameter_counter in experiments_want_to_plot_data_from: 
                            temp_list.append(optimized_X[i][0])
                            prior_sigma=T_uncertainty
                        parameter_counter+=1
            
                elif parameter == 'Time':
                    if p[0] == 'T' and p[1] == 'i':
                        yaml_file = int(p.split('_')[3])
                        if parameter_counter in experiments_want_to_plot_data_from: 
                            temp_list.append(optimized_X[i][0])
                            prior_sigma=T_uncertainty
                        parameter_counter+=1                
                        
                elif parameter == 'P':        
                    if p[0] == 'P':
                        yaml_file = int(p.split('_')[2])
                        pressure_original = original_experimental_conditions[yaml_file]['pressure']
                        #temp_list.append(temp_original*np.exp(optimized_X[i]) - temp_original) 
                        if parameter_counter in experiments_want_to_plot_data_from: 
                            temp_list.append(optimized_X[i][0])  
                            prior_sigma = P_uncertainty
                        parameter_counter+=1

                elif parameter =='H2O':
                    if p[0] == 'X' and p[2:5] == 'H2O' and p[5]== '_':
                        yaml_file = int(p.split('_')[3])
                        specie_original = original_experimental_conditions[yaml_file]['conditions']['H2O'] 
                        if parameter_counter in experiments_want_to_plot_data_from:                            
                            temp_list.append(optimized_X[i][0])     
                            prior_sigma=X_uncertainty
                        parameter_counter+=1
   
                elif parameter =='H2O2':
                    if p[0] == 'X' and p[2:6] == 'H2O2' and p[6]== '_':
                        yaml_file = int(p.split('_')[3])
                        specie_original = original_experimental_conditions[yaml_file]['conditions']['H2O2']                            
                        if parameter_counter in experiments_want_to_plot_data_from:                            
                            temp_list.append(optimized_X[i][0])     
                            prior_sigma=X_uncertainty
                        parameter_counter+=1

                elif parameter =='O2':
                    if p[0] == 'X' and p[2:4] == 'O2' and p[4]== '_':
                        yaml_file = int(p.split('_')[3])
                        specie_original = original_experimental_conditions[yaml_file]['conditions']['O2']                            
                        if parameter_counter in experiments_want_to_plot_data_from:                            
                            temp_list.append(optimized_X[i][0])     
                            prior_sigma=X_uncertainty
                        parameter_counter+=1
                            
                elif parameter =='H':
                    if p[0] == 'X' and p[2:3] == 'H' and p[3]== '_':
                        yaml_file = int(p.split('_')[3])
                        specie_original = original_experimental_conditions[yaml_file]['conditions']['H']                            
                        if parameter_counter in experiments_want_to_plot_data_from:                            
                            temp_list.append(optimized_X[i][0])     
                            prior_sigma=X_uncertainty
                        parameter_counter+=1
                        
                elif parameter =='CH4':
                    if p[0] == 'X' and p[2:5] == 'CH4' and p[5]== '_':
                        yaml_file = int(p.split('_')[3])
                        specie_original = original_experimental_conditions[yaml_file]['conditions']['CH4']                            
                        if parameter_counter in experiments_want_to_plot_data_from:                            
                            temp_list.append(optimized_X[i][0])     
                            prior_sigma=X_uncertainty                         
                        parameter_counter+=1
                else:
                    parameter_counter+=1
                

            plt.figure()
            mu2=0 
            sigma2=prior_sigma
            n, bins, patches=plt.hist(temp_list,bins='auto',density=True,color='g') 
            (mu, sigma) = norm.fit(temp_list)
            #y = mlab.normpdf( bins, mu, sigma)
            y = norm.pdf(bins,mu,sigma)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            l = plt.plot(bins, y, 'b--', linewidth=2)
            plt.plot(x, stats.norm.pdf(x, mu, sigma),'b')
            x2 = np.linspace(mu2 - 3*sigma2, mu2 + 3*sigma2, 100)
            plt.plot(x2, stats.norm.pdf(x2, mu2, sigma2),'r')

    #plot
            plt.xlabel(parameter)
            #plt.ylabel('Probability')
            plt.title(r'$\mathrm{Histogram\ of\ physical\ model\ parameter:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
            plt.grid(True)
            #plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+parameter+'_.pdf',dpi=1000,bbox_inches='tight')


    def difference_plotter(self,
                           paramter_list,
                           optimized_cti_file='',
                           pdf_distribution_file=''):
                        
            
        all_parameters = self.shock_tube_instance.posterior_diag_df['parameter'].tolist()
        df = self.shock_tube_instance.posterior_diag_df
        gas_optimized = ct.Solution(optimized_cti_file)
        
        for parameter in paramter_list:
            indx = all_parameters.index(parameter)
            variance = df['value'][indx]
            letter,number = parameter.split('_')
            number = int(number)
            A=gas_optimized.reaction(number).rate.pre_exponential_factor
            n=gas_optimized.reaction(number).rate.temperature_exponent
            Ea=gas_optimized.reaction(number).rate.activation_energy
            
            if letter =='A':
                mu = np.log(A*1000)
                sigma = math.sqrt(variance)
                sigma = sigma
                
            if letter == 'n':
                mu = n
                sigma = math.sqrt(variance)
                #sigma = sigma/2
            if letter == 'Ea':
                mu=Ea/1000/4.184            
                sigma = math.sqrt(variance)
                sigma = sigma*ct.gas_constant/(1000*4.184)
                #sigma = sigma/2

            
            
            x = np.linspace(mu - 6*sigma, mu + 6*sigma, 100)
            #plt.figure()
            #plt.plot(x, stats.norm.pdf(x, mu, sigma))
           # plt.xlabel(parameter)
           # plt.ylabel('pdf')
           # plt.savefig(self.out_path+'/'+parameter+'_distribution'+'_.pdf',bbox_inches='tight')

            if bool(pdf_distribution_file):
                df2 = pd.read_csv(pdf_distribution_file)
                #temp = np.log(np.exp(df2[parameter].values)/9.33e13)
                #plt.plot(temp,df2['pdf_'+parameter])
                interp_y = np.interp(df2[parameter],x,stats.norm.pdf(x, mu, sigma))
                plt.figure()
                plt.plot(df2[parameter],interp_y)
                plt.plot(df2[parameter],df2['pdf_'+parameter])
                interp_x = np.interp(df2['pdf_'+parameter],stats.norm.pdf(x,mu,sigma),x)
                y_shift = np.divide((df2['pdf_'+parameter] - interp_y),df2['pdf_'+parameter])
                x_shift = np.divide((df2[parameter] - interp_x),df2[parameter])
                plt.figure()
                plt.title('Percent Difference In Y')
                plt.plot(y_shift)
                plt.xlabel(parameter)
                plt.figure()
                plt.plot(x_shift)
                plt.title('Percent Difference In X')
                plt.xlabel(parameter)
              
    def plotting_histograms_of_MSI_simulations(self,experiments_want_to_plot_data_from=[],bins='auto',directory_to_save_images=''):
        s_shape = self.S_matrix.shape[1]
        if self.k_target_value_S_matrix.any():
            target_values_for_s = self.k_target_value_S_matrix
            s_shape = s_shape+target_values_for_s.shape[0]
        y_shape = self.y_matrix.shape[0]
        difference = y_shape-s_shape
        y_values = self.y_matrix[0:difference,0]
        Y_values = self.Y_matrix[0:difference,0]
        self.lengths_of_experimental_data()

        #plotting_Y Histagrams 
        if bool(experiments_want_to_plot_data_from):
            y_values = []
            Y_values = []
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from:
                        temp = self.Y_matrix[start:stop,:]
                        Y_values.append(temp)
                        temp2 = self.y_matrix[start:stop,:]
                        y_values.append(temp2)
                    
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]              
          
                    
                    
                    
                    
            Y_values = np.vstack((Y_values))
            y_values = np.vstack((y_values))
            plt.figure()            
            plt.subplot(2,2,1)

            n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid')
            min_value = min(Y_values)
            max_value=max(Y_values)
            plt.xlim([min_value,max_value])
            plt.xlabel('Y')
            plt.suptitle('Including Experiments_'+ str(experiments_want_to_plot_data_from), fontsize=10)

            plt.subplot(2,2,2)
            plt.hist(y_values,bins=bins,align='mid')
            plt.xlabel('y')

            plt.subplot(2,2,3)
            plt.hist(Y_values,bins=bins,density=True,align='mid')
            plt.xlabel('Y')
            plt.ylabel('normalized')

            plt.subplot(2,2,4)
            plt.hist(y_values,bins=bins,density=True,align='mid')
            plt.xlabel('y')      
            plt.ylabel('normalized')
            
            
            
            
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
            plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_4.pdf',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_4.png',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_4.svg',dpi=1000,bbox_inches='tight',transparent=True)
            
            
            
            #plotting two fold plots 
            plt.figure()            
            plt.subplot(2,1,1)
            plt.title('Including Experiments_'+ str(experiments_want_to_plot_data_from))

            n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid')
            plt.xlabel('Y')
            #plt.xlim([-1,1])

            plt.subplot(2,1,2)
            plt.hist(y_values,bins=bins,align='mid')
            plt.xlabel('y')
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
            plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2.pdf',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2.png',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2.svg',dpi=1000,bbox_inches='tight',transparent=True)

#plotting normalized values
            plt.figure()            
            plt.subplot(2,1,1)
            n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid',density=True)
            plt.xlabel('Y')
            plt.title('Including Experiments_'+ str(experiments_want_to_plot_data_from))
            plt.ylabel('normalized')

            plt.subplot(2,1,2)
            plt.hist(y_values,bins=bins,align='mid',density=True)
            plt.xlabel('y')
            plt.ylabel('normalized')
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
            plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2_normalized.pdf',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2_normalized.png',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2_normalized.svg',dpi=1000,bbox_inches='tight',transparent=True)

            



        else:
            plt.figure()            
            plt.subplot(2,2,1)
            min_value = min(Y_values)
            max_value=max(Y_values)
            plt.xlim([min_value,max_value])
            n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid')
            #plt.xlim([min_value,max_value])
            plt.xlabel('Y')
            plt.suptitle("Including All Experiments", fontsize=10)

            plt.subplot(2,2,2)
            plt.hist(y_values,bins=bins,align='mid')
            plt.xlabel('y')

            plt.subplot(2,2,3)
            plt.hist(Y_values,bins=bins,density=True,align='mid')
            plt.xlabel('Y')
            plt.ylabel('normalized')

            plt.subplot(2,2,4)
            plt.hist(y_values,bins=bins,density=True,align='mid')
            plt.xlabel('y')      
            plt.ylabel('normalized')
            
            
            
            
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
            plt.savefig(directory_to_save_images+'/'+'Including all Experiments'+'_Yy_hist_4.pdf',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including all Experiments'+'_Yy_hist_4.png',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including all Experiments'+'_Yy_hist_4.svg',dpi=1000,bbox_inches='tight',transparent=True)
            
            
            
            #plotting two fold plots 
            plt.figure()            
            plt.subplot(2,1,1)
            min_value = np.min(Y_values)
            max_value = np.max(Y_values)
            plt.title('Including all Experiments')

            n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid')
            plt.xlabel('Y')
            #plt.xlim([-1,1])

            plt.subplot(2,1,2)
            plt.hist(y_values,bins=bins,align='mid')
            plt.xlabel('y')
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
            plt.savefig(directory_to_save_images+'/'+'Including all Experiments'+'_Yy_hist_2.pdf',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including all Experiments'+'_Yy_hist_2.png',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including all Experiments'+'_Yy_hist_2.svg',dpi=1000,bbox_inches='tight',transparent=True)
            
#plotting normalized values
            plt.figure()            
            plt.subplot(2,1,1)
            min_value = np.min(Y_values)
            max_value = np.max(Y_values)
            n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid',density=True)
            plt.xlabel('Y')
            plt.title('Including all Experiments')
            plt.ylabel('normalized')

            plt.subplot(2,1,2)
            plt.hist(y_values,bins=bins,align='mid',density=True)
            plt.xlabel('y')
            plt.ylabel('normalized')
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
            plt.savefig(directory_to_save_images+'/'+'Including all Experiments'+'_Yy_hist_2_normalized.pdf',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including all Experiments'+'_Yy_hist_2_normalized.png',dpi=1000,bbox_inches='tight')
            plt.savefig(directory_to_save_images+'/'+'Including all Experiments'+'_Yy_hist_2_normalized.svg',dpi=1000,bbox_inches='tight',transparent=True)

    def plotting_T_and_time_full_simulation(self,experiments_want_to_plot_data_from=[],directory_to_save_images=''):
        init_temperature_list = []
        for exp in self.exp_dict_list_original:
            init_temperature_list.append(exp['simulation'].temperature)
        total_times = []
        temperature_list_full_simulation = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            single_exp_dict = []
            temp_list_single_experiment = []
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    single_exp_dict.append(exp['experimental_data'][observable_counter]['Time']*1e3)
                    interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                    temp_list_single_experiment.append(interploated_temp)

                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:
                    single_exp_dict.append(exp['experimental_data'][observable_counter]['Time']*1e3)
                    interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                    temp_list_single_experiment.append(interploated_temp) 
                    #print(interploated_temp.shape ,exp['experimental_data'][observable_counter]['Time'].shape )


                    observable_counter+=1
                    
            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    single_exp_dict.append(exp['absorbance_experimental_data'][k]['time']*1e3)
                    interploated_temp = np.interp(exp['absorbance_experimental_data'][k]['time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                    temp_list_single_experiment.append(interploated_temp)
                    #print(interploated_temp.shape, exp['absorbance_experimental_data'][k]['time'].shape )
  

            total_times.append(single_exp_dict)
            temperature_list_full_simulation.append(temp_list_single_experiment)
            
            
        if bool(experiments_want_to_plot_data_from)==False:
            experiments_want_to_plot_data_from = np.arange(0,len(self.exp_dict_list_optimized))
        else:
            experiments_want_to_plot_data_from = experiments_want_to_plot_data_from
           
        y_values = []
        Y_values = []
        temperature_values_list = []
        time_values_list = []
        full_temperature_range_list = []
        start = 0
        stop = 0             
        for x in range(len(self.simulation_lengths_of_experimental_data)):
            single_experiment_Y =[]
            single_experiment_y =[]
            single_experiment_temperature_values_list=[]
            single_experiment_time_values_list=[]
            single_experiment_full_temp_range=[]
            for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                stop = self.simulation_lengths_of_experimental_data[x][y] + start
                if x in experiments_want_to_plot_data_from:
                    temp = self.Y_matrix[start:stop,:]
                    single_experiment_Y.append(temp)
                    temp2 = self.y_matrix[start:stop,:]
                    single_experiment_y.append(temp2)
                    intial_temp = np.array(([init_temperature_list[x]]*temp.shape[0]))
                    intial_temp = intial_temp.reshape((intial_temp.shape[0],1))
                    single_experiment_temperature_values_list.append(intial_temp)
                    
                    
                    time_values = total_times[x][y].values
                    time_values = time_values.reshape((time_values.shape[0],1))
                    single_experiment_time_values_list.append(time_values)
                    
                    temperature_full = temperature_list_full_simulation[x][y]
                    temperature_full = temperature_full.reshape((temperature_full.shape[0],1))
                    single_experiment_full_temp_range.append(temperature_full)

                    start = start + self.simulation_lengths_of_experimental_data[x][y]
                else:
                    start = start + self.simulation_lengths_of_experimental_data[x][y] 
            Y_values.append(single_experiment_Y)
            y_values.append(single_experiment_y)
            temperature_values_list.append(single_experiment_temperature_values_list)
            time_values_list.append(single_experiment_time_values_list)
            full_temperature_range_list.append(single_experiment_full_temp_range)
        
        x = np.arange(10)
        ys = [i+x+(i*x)**2 for i in range(10)]
        colors=cm.rainbow(np.linspace(0,1,30))

        #colors = cm.rainbow(np.linspace(0, 1, len(ys)))

        plt.figure() 
        for x,simulation_list in enumerate(Y_values):
            for y,lst in enumerate(Y_values[x]):
                plt.subplot(2,1,1)
                plt.xlabel('Y')
                plt.ylabel('Time')
                plt.scatter(Y_values[x][y],time_values_list[x][y],label='Experiment_'+str(x)+'_observable_'+str(y),color=colors[x])
                plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                plt.subplot(2,1,2)
                plt.scatter(y_values[x][y],time_values_list[x][y],color=colors[x])
                plt.xlabel('y')
                plt.ylabel('Time')
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
                plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_vs_time.pdf',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_vs_time.png',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_vs_time.svg',dpi=1000,bbox_inches='tight',transparent=True)

    
                
                
        plt.figure() 
    
        for x,simulation_list in enumerate(Y_values):
            for y,lst in enumerate(Y_values[x]):
                plt.subplot(2,1,1)
                plt.scatter(Y_values[x][y],temperature_values_list[x][y],label='Experiment_'+str(x)+'_observable_'+str(y),color=colors[x])
                plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                plt.xlabel('Y')
                plt.ylabel('Initial Simulation Temp')
                plt.subplot(2,1,2)
                plt.scatter(y_values[x][y],temperature_values_list[x][y],color=colors[x])    
                plt.xlabel('y')
                plt.ylabel('Initial Simulation Temp')
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
                plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_vs_init_temp.pdf',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_vs_init_temp.png',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_vs_init_temp.svg',dpi=1000,bbox_inches='tight',transparent=True)

        plt.figure() 
        for x,simulation_list in enumerate(Y_values):
            for y,lst in enumerate(Y_values[x]):
                plt.subplot(2,1,1)
                plt.scatter(Y_values[x][y],full_temperature_range_list[x][y],label='Experiment_'+str(x)+'_observable_'+str(y),color=colors[x])
                plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))

                plt.xlabel('Y')
                plt.ylabel('Temperature')
                plt.subplot(2,1,2)
                plt.scatter(y_values[x][y],full_temperature_range_list[x][y],color=colors[x])      
                plt.xlabel('y')
                plt.ylabel('Temperature')
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
                plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_vs_temperature.pdf',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_vs_temperature.png',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+'Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_vs_temperature.svg',dpi=1000,bbox_inches='tight',transparent=True)

        return 

    def plotting_histograms_of_individual_observables(self,experiments_want_to_plot_data_from,bins='auto',directory_to_save_images='',csv=''):
        s_shape = self.S_matrix.shape[1]
        if self.k_target_value_S_matrix.any():
            target_values_for_s = self.k_target_value_S_matrix
            s_shape = s_shape+target_values_for_s.shape[0]
        y_shape = self.y_matrix.shape[0]
        difference = y_shape-s_shape
        y_values = self.y_matrix[0:difference,0]
        Y_values = self.Y_matrix[0:difference,0]

        self.lengths_of_experimental_data()

        #plotting_Y Histagrams 
        #obserervable_list = []
        
        observables_total = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            single_experiment = []
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    single_experiment.append(observable)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:

                    single_experiment.append(observable)
                    
                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    single_experiment.append(wl)
                    
            observables_total.append(single_experiment)
        
        observables_flatten = [item for sublist in observables_total for item in sublist]
        from collections import OrderedDict
        observables_unique = list(OrderedDict.fromkeys(observables_flatten))
        
        empty_nested_observable_list_Y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y = [[] for x in range(len(observables_unique))]

        
        if bool(experiments_want_to_plot_data_from):
            
            y_values = []
            Y_values = []
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from:
                        temp = self.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = self.y_matrix[start:stop,:]
                        empty_nested_observable_list_y[observables_unique.index(current_observable)].append(temp2)

                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]              
          
                    
                    
                    
                    
            for i,observable in enumerate(empty_nested_observable_list_Y):
                if bool(observable):
                    Y_values = np.vstack((observable))
                    y_values = np.vstack((empty_nested_observable_list_y[i]))
                    
                    plt.figure()            
                    plt.subplot(2,2,1)
        
                    n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid')
                    min_value = min(Y_values)
                    max_value=max(Y_values)
                    plt.xlim([min_value,max_value])
                    plt.xlabel('Y')
                    plt.suptitle(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from), fontsize=10)
        
                    plt.subplot(2,2,2)
                    plt.hist(y_values,bins=bins,align='mid')
                    plt.xlabel('y')
        
                    plt.subplot(2,2,3)
                    plt.hist(Y_values,bins=bins,density=True,align='mid')
                    plt.xlabel('Y')
                    plt.ylabel('normalized')
        
                    plt.subplot(2,2,4)
                    plt.hist(y_values,bins=bins,density=True,align='mid')
                    plt.xlabel('y')      
                    plt.ylabel('normalized')
            
            
            
                    
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
                    #plt.savefig(directory_to_save_images+'/'+str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_4.pdf',dpi=1000,bbox_inches='tight')
                    
                    
                    
                    #plotting two fold plots 
                    plt.figure()            
                    plt.subplot(2,1,1)
                    plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
        
                    n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid')
                    plt.xlabel('Y')
                    #plt.xlim([-1,1])
        
                    plt.subplot(2,1,2)
                    plt.hist(y_values,bins=bins,align='mid')
                    plt.xlabel('y')
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
                    #plt.savefig(directory_to_save_images+'/'+str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2.pdf',dpi=1000,bbox_inches='tight')
        
        #plotting normalized values
                    plt.figure()            
                    plt.subplot(2,1,1)
                    n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid',density=True)
                    plt.xlabel('Y')
                    plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
                    plt.ylabel('normalized')
        
                    plt.subplot(2,1,2)
                    plt.hist(y_values,bins=bins,align='mid',density=True)
                    plt.xlabel('y')
                    plt.ylabel('normalized')
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
                    

                    #plotting two fold plots 
                    plt.figure()            
                    plt.subplot(2,1,1)
                    plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
        
                    n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid')
                    plt.xlabel('Y')
                    #plt.xlim([-1,1])
        
                    plt.subplot(2,1,2)
                    plt.hist(y_values,bins=bins,align='mid')
                    plt.xlabel('y')
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
                    #plt.savefig(directory_to_save_images+'/'+str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2.pdf',dpi=1000,bbox_inches='tight')
        
        #plotting normalized values
                    plt.figure()            
                    plt.subplot(2,1,1)
                    n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid',density=True)
                    plt.xlabel('Y')
                    plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
                    plt.ylabel('normalized')
        
                    plt.subplot(2,1,2)
                    plt.hist(y_values,bins=bins,align='mid',density=True)
                    plt.xlabel('y')
                    plt.ylabel('normalized')
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
                   # plt.savefig(directory_to_save_images+'/'+str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2_normalized.pdf',dpi=1000,bbox_inches='tight')

    def plotting_histograms_of_individual_observables_for_paper_2(self,experiments_want_to_plot_data_from,experiments_want_to_plot_data_from_2=[],bins='auto',directory_to_save_images='',csv=''):
        s_shape = self.S_matrix.shape[1]
        if self.k_target_value_S_matrix.any():
            target_values_for_s = self.k_target_value_S_matrix
            s_shape = s_shape+target_values_for_s.shape[0]
        y_shape = self.y_matrix.shape[0]
        difference = y_shape-s_shape
        y_values = self.y_matrix[0:difference,0]
        Y_values = self.Y_matrix[0:difference,0]
        self.lengths_of_experimental_data()

        #plotting_Y Histagrams 
        #edit this part 
        #obserervable_list = []
        
        observables_total = []
        
        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            single_experiment = []
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    single_experiment.append(observable)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:

                    single_experiment.append(observable)
                    
                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    single_experiment.append(wl)
                    
            observables_total.append(single_experiment)
        observables_flatten = [item for sublist in observables_total for item in sublist]
        from collections import OrderedDict
        observables_unique = list(OrderedDict.fromkeys(observables_flatten))
        
        
        empty_nested_observable_list_Y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y = [[] for x in range(len(observables_unique))]
        
        empty_nested_observable_list_Z = [[] for x in range(len(observables_unique))]
        
        empty_nested_observable_list_Y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_Z_2 = [[] for x in range(len(observables_unique))]

        if bool(experiments_want_to_plot_data_from):
            # print('inside here')
            y_values = []
            Y_values = []
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from:
                        temp = self.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = self.y_matrix[start:stop,:]
                        empty_nested_observable_list_y[observables_unique.index(current_observable)].append(temp2)

                        temp3 = self.Z_matrix[start:stop,:]
                        empty_nested_observable_list_Z[observables_unique.index(current_observable)].append(temp3)
                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]              
          
        if bool(experiments_want_to_plot_data_from_2):
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from_2:


                        temp = self.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y_2[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = self.y_matrix[start:stop,:]
                        empty_nested_observable_list_y_2[observables_unique.index(current_observable)].append(temp2)

                        temp3 = self.Z_matrix[start:stop,:]
                        empty_nested_observable_list_Z_2[observables_unique.index(current_observable)].append(temp3)
                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]                     
                    
        import matplotlib.gridspec as gridspec
    
        fig = plt.figure(figsize=(6,7))

        gs = gridspec.GridSpec(3, 1,height_ratios=[3,3,3],wspace=0.1,hspace=0.1)
        gs.update(wspace=0, hspace=0.7)
        ax1=plt.subplot(gs[0])
        ax2=plt.subplot(gs[1])
        ax3=plt.subplot(gs[2])  
        for i,observable in enumerate(empty_nested_observable_list_Y):
            new_Y_test_2 =[]
            if bool(observable):
                Y_values = np.vstack((observable))
                y_values = np.vstack((empty_nested_observable_list_y[i]))
                z_values = np.vstack((empty_nested_observable_list_Z[i]))
                indecies = np.argwhere(z_values > 100)
                new_y_test = copy.deepcopy(Y_values)
                new_y_test = np.delete(new_y_test,indecies)
             #   print(indecies.shape)
            #    print(indecies)
            #    print(i)
                if bool(experiments_want_to_plot_data_from_2) and bool(empty_nested_observable_list_y_2[i]):
                    Y_values_2 = np.vstack((empty_nested_observable_list_Y_2[i]))
                    y_values_2 = np.vstack((empty_nested_observable_list_y_2[i]))
                    z_values_2 = np.vstack((empty_nested_observable_list_Z_2[i]))
                    indecies_2 = np.argwhere(z_values_2 > 100)
                    new_Y_test_2 = copy.deepcopy(Y_values_2)
                    new_Y_test_2 = np.delete(new_Y_test_2,indecies_2)
                    
            
                #plt.figure()            
                #plt.subplot(1,1,1)
                #plt.subplots(3,1,1)
                #n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid',density=True,label='Hong Experiments')
                test = [-0.06402874, -0.05325865, -0.04248857, -0.03171848, -0.02094839, -0.0101783,
                        0.00059179,  0.01136188,  0.02213197,  0.03290205,  0.04367214,  0.05444223,
                        0.06521232,  0.07598241,  0.0867525,   0.09752259,  0.10829268]
                if i ==0:
                #n, bins2, patches = plt.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    #ax1.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    n,bins_test_1,patches = ax1.hist(new_y_test,bins=bins ,align='mid',density=True,label='#1')
                    ax1.set_xlim(left=-.3, right=.3, emit=True, auto=False)
                    ax1.set_ylim(top=15,bottom=0)

                    ax1.set_xlabel('Y')
                    ax1.set_xlabel('Relative Difference')
                #plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
                    ax1.set_title(str(observables_unique[i]))
                    ax1.set_ylabel('pdf')

                #plt.ylabel('normalized')
                    if bool(experiments_want_to_plot_data_from_2):
                   # plt.hist(Y_values_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        #ax1.hist(new_Y_test_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        ax1.hist(new_Y_test_2,bins=bins ,align='mid',density=True,alpha=0.5,label='#2')

                    if bool(csv):
                        df = pd.read_csv(csv)
                        #ax1.hist(df[str(observables_unique[i])+'_Y'].dropna()*-1,bins=bins ,align='mid',density=True,alpha=0.5,label='Hong vs. Hong')
                        #ax1.hist(df[str(observables_unique[i])+'_Y'].dropna()*-1,bins=bins ,align='mid',density=True,alpha=0.5,label='#3')

                    ax1.legend()
                if i ==1:
                #n, bins2, patches = plt.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    n,bins_test_2,patches = ax2.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    ax2.set_xlim(left=-.08, right=.08, emit=True, auto=False)
                    ax2.set_ylim(top=28,bottom=0)
                    ax2.set_xlabel('Y')
                    ax2.set_xlabel('Relative Difference')
                #plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
                    #ax2.set_title(str(observables_unique[i]))
                    ax2.set_title(r'H$_2$O')
                    ax2.set_ylabel('pdf')
                    
                #plt.ylabel('normalized')
                    if bool(experiments_want_to_plot_data_from_2):
                   # plt.hist(Y_values_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        ax2.hist(new_Y_test_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        
                    if bool(csv):
                        df = pd.read_csv(csv)
                        #ax2.hist(df[str(observables_unique[i])+'_Y'].dropna()*-1,bins=bins ,align='mid',density=True,alpha=0.5,label='Hong vs. Hong')
                
                
                if i ==3:
                #n, bins2, patches = plt.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    n,bins_test_3,patches = ax3.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    ax3.set_xlim(left=-.15, right=.15, emit=True, auto=False)
                    ax3.set_ylim(top=12,bottom=0)

                    ax3.set_xlabel('Y')
                    ax3.set_xlabel('Relative Difference')
                    ax3.set_ylabel('pdf')
                #plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
                    ax3.set_title(str(observables_unique[i]))
                    ax3.set_title('Absorbance '+ str(observables_unique[i])+ ' nm')

                #plt.ylabel('normalized')
                    if bool(experiments_want_to_plot_data_from_2):
                        # print('inside here')
                        # print(experiments_want_to_plot_data_from_2)
                   # plt.hist(Y_values_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        ax3.hist(new_Y_test_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        
                    if bool(csv):
                        df = pd.read_csv(csv)
                        #ax3.hist(df[str(observables_unique[i])+'_Y'].dropna()*-1,bins=bins ,align='mid',density=True,alpha=0.5,label='Hong vs. Hong')                        
                
                
                

                    plt.savefig(directory_to_save_images+'/'+str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2_normalized.pdf',dpi=1000,bbox_inches='tight')   
                    plt.savefig(directory_to_save_images+'/'+str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2_normalized.png',dpi=1000,bbox_inches='tight')   
                    plt.savefig(directory_to_save_images+'/'+str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2_normalized.svg',dpi=1000,bbox_inches='tight',transparent=True)    
    
    
    
    def plotting_histograms_of_individual_observables_for_paper(self,experiments_want_to_plot_data_from,experiments_want_to_plot_data_from_2=[],bins='auto',directory_to_save_images='',csv=''):
        s_shape = self.S_matrix.shape[1]
        if self.k_target_value_S_matrix.any():
            target_values_for_s = self.k_target_value_S_matrix
            s_shape = s_shape+target_values_for_s.shape[0]
        y_shape = self.y_matrix.shape[0]
        difference = y_shape-s_shape
        y_values = self.y_matrix[0:difference,0]
        Y_values = self.Y_matrix[0:difference,0]
        self.lengths_of_experimental_data()

        #plotting_Y Histagrams 
        #obserervable_list = []
        
        observables_total = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            single_experiment = []
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    single_experiment.append(observable)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:

                    single_experiment.append(observable)
                    
                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    single_experiment.append(wl)
                    
            observables_total.append(single_experiment)
        
        observables_flatten = [item for sublist in observables_total for item in sublist]
        from collections import OrderedDict
        observables_unique = list(OrderedDict.fromkeys(observables_flatten))
        
        empty_nested_observable_list_Y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y = [[] for x in range(len(observables_unique))]
        
        empty_nested_observable_list_Z = [[] for x in range(len(observables_unique))]
        
        empty_nested_observable_list_Y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_Z_2 = [[] for x in range(len(observables_unique))]

        if bool(experiments_want_to_plot_data_from):
            # print('inside here')
            y_values = []
            Y_values = []
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from:
                        temp = self.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = self.y_matrix[start:stop,:]
                        empty_nested_observable_list_y[observables_unique.index(current_observable)].append(temp2)

                        temp3 = self.Z_matrix[start:stop,:]
                        empty_nested_observable_list_Z[observables_unique.index(current_observable)].append(temp3)
                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]              
          
        if bool(experiments_want_to_plot_data_from_2):
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from_2:
                        temp = self.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y_2[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = self.y_matrix[start:stop,:]
                        empty_nested_observable_list_y_2[observables_unique.index(current_observable)].append(temp2)

                        temp3 = self.Z_matrix[start:stop,:]
                        empty_nested_observable_list_Z_2[observables_unique.index(current_observable)].append(temp3)
                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]                     
                    
            import matplotlib.gridspec as gridspec
       
            for i,observable in enumerate(empty_nested_observable_list_Y):
                if bool(observable):
                    Y_values = np.vstack((observable))
                    y_values = np.vstack((empty_nested_observable_list_y[i]))
                    z_values = np.vstack((empty_nested_observable_list_Z[i]))
                    indecies = np.argwhere(z_values > 100)
                    new_y_test = copy.deepcopy(Y_values)
                    new_y_test = np.delete(new_y_test,indecies)

                    if bool(experiments_want_to_plot_data_from_2) and bool(empty_nested_observable_list_y_2[i]):
                        
                        Y_values_2 = np.vstack((empty_nested_observable_list_Y_2[i]))
                        y_values_2 = np.vstack((empty_nested_observable_list_y_2[i]))
                        z_values_2 = np.vstack((empty_nested_observable_list_Z_2[i]))
                        indecies_2 = np.argwhere(z_values_2 > 100)
                        new_Y_test_2 = copy.deepcopy(Y_values_2)
                        new_Y_test_2 = np.delete(new_Y_test_2,indecies_2)
                
                    plt.figure()            
                    plt.subplot(1,1,1)
                    #plt.subplots(3,1,1)
                    #n, bins2, patches = plt.hist(Y_values,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    n, bins2, patches = plt.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')

                    plt.xlabel('Y')
                    #plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
                    plt.title(str(observables_unique[i]))
                    #plt.ylabel('normalized')
                    if bool(experiments_want_to_plot_data_from_2):
                       # plt.hist(Y_values_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        plt.hist(new_Y_test_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')

                    if bool(csv):
                        df = pd.read_csv(csv)
                        plt.hist(df[str(observables_unique[i])+'_Y'].dropna()*-1,bins=bins ,align='mid',density=True,alpha=0.5,label='Hong vs. Hong')
                    plt.legend()
        return 

    def plotting_T_and_time_full_simulation_individual_observables(self,experiments_want_to_plot_data_from,bins='auto',directory_to_save_images=''):
#working_here

        observables_total = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            single_experiment = []
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    single_experiment.append(observable)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:

                    single_experiment.append(observable)
                    
                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    single_experiment.append(wl)
                    
            observables_total.append(single_experiment)
        
        observables_flatten = [item for sublist in observables_total for item in sublist]
        from collections import OrderedDict
        observables_unique = list(OrderedDict.fromkeys(observables_flatten))
        
        empty_nested_observable_list_Y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_time = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_temperature = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_initial_temperature = [[] for x in range(len(observables_unique))]


        if bool(experiments_want_to_plot_data_from):
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from:
                        temp = self.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = self.y_matrix[start:stop,:]
                        empty_nested_observable_list_y[observables_unique.index(current_observable)].append(temp2)

                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]  


        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                if i in experiments_want_to_plot_data_from:
                    if observable in exp['mole_fraction_observables']:
                        empty_nested_observable_list_time[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
                        observable_counter+=1
                        
                    if observable in exp['concentration_observables']:
                        empty_nested_observable_list_time[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
    
    
                        observable_counter+=1
            if i in experiments_want_to_plot_data_from:        
                if 'perturbed_coef' in exp.keys():
                    wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                    for k,wl in enumerate(wavelengths):
                        empty_nested_observable_list_time[observables_unique.index(wl)].append(exp['absorbance_experimental_data'][k]['time']*1e3)
    
                        interploated_temp = np.interp(exp['absorbance_experimental_data'][k]['time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(wl)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(wl)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
    
    
                        #print(interploated_temp.shape, exp['absorbance_experimental_data'][k]['time'].shape )
  
        
        x = np.arange(10)
        ys = [i+x+(i*x)**2 for i in range(10)]
        colors=cm.rainbow(np.linspace(0,1,30))

        #colors = cm.rainbow(np.linspace(0, 1, len(ys)))

        
        for x,observable in enumerate(empty_nested_observable_list_Y):
            if bool(observable):
                plt.figure()
                for y,array in enumerate(empty_nested_observable_list_Y[x]):
                        plt.subplot(2,1,1)
                        plt.xlabel('Y')
                        plt.ylabel('Time')
                        plt.scatter(empty_nested_observable_list_Y[x][y],empty_nested_observable_list_time[x][y],label='Experiment_'+str(x)+'_observable_'+str(y),color=colors[x])
                        #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                        plt.title(observables_unique[x])

                        plt.subplot(2,1,2)
                        plt.scatter(empty_nested_observable_list_y[x][y],empty_nested_observable_list_time[x][y],color=colors[x])
                        plt.xlabel('y')
                        plt.ylabel('Time')
                        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)

                plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_time.pdf',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_time.png',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_time.svg',dpi=1000,bbox_inches='tight',transparent=True)

        for x,observable in enumerate(empty_nested_observable_list_Y):
            if bool(observable):
                plt.figure()
                for y,array in enumerate(empty_nested_observable_list_Y[x]):
                    plt.subplot(2,1,1)
                    plt.scatter(empty_nested_observable_list_Y[x][y],empty_nested_observable_list_temperature[x][y],label='Experiment_'+str(x)+'_observable_'+str(y),color=colors[x])
                    #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                    plt.xlabel('Y')
                    plt.ylabel('Temperature')
                    plt.title(observables_unique[x])

                    plt.subplot(2,1,2)
                    
                    plt.scatter(empty_nested_observable_list_y[x][y],empty_nested_observable_list_temperature[x][y],color=colors[x])    
                    plt.xlabel('y')
                    plt.ylabel('Temperature')
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
                plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_temperature.pdf',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_temperature.png',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_temperature.svg',dpi=1000,bbox_inches='tight',transparent=True)
        for x,observable in enumerate(empty_nested_observable_list_Y):
            if bool(observable):
                plt.figure()
                for y,array in enumerate(empty_nested_observable_list_Y[x]):
                    plt.subplot(2,1,1)
                    plt.scatter(empty_nested_observable_list_Y[x][y],empty_nested_observable_list_initial_temperature[x][y],label='Experiment_'+str(x)+'_observable_'+str(y),color=colors[x])
                    #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                    plt.xlabel('Y')
                    plt.ylabel('Initial Temperature')
                    plt.title(observables_unique[x])

                    plt.subplot(2,1,2)
                    
                    plt.scatter(empty_nested_observable_list_y[x][y],empty_nested_observable_list_initial_temperature[x][y],color=colors[x])    
                    plt.xlabel('y')
                    plt.ylabel('Initial Temperature')
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)    
                plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_initial_temperature.pdf',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_initial_temperature.png',dpi=1000,bbox_inches='tight')
                plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_initial_temperature.svg',dpi=1000,bbox_inches='tight',transparent=True)





    def plotting_T_and_time_full_simulation_individual_observables_for_paper(self,experiments_want_to_plot_data_from,
                                                                             bins='auto',
                                                                             directory_to_save_images='',csv='',experiments_want_to_plot_data_from_2=[]):
#working_here

        observables_total = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            single_experiment = []
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    single_experiment.append(observable)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:

                    single_experiment.append(observable)
                    
                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    single_experiment.append(wl)
                    
            observables_total.append(single_experiment)
        
        observables_flatten = [item for sublist in observables_total for item in sublist]
        from collections import OrderedDict
        observables_unique = list(OrderedDict.fromkeys(observables_flatten))
        
        empty_nested_observable_list_Y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_time = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_temperature = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_initial_temperature = [[] for x in range(len(observables_unique))]


        if bool(experiments_want_to_plot_data_from):
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from:
                        temp = self.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = self.y_matrix[start:stop,:]
                        empty_nested_observable_list_y[observables_unique.index(current_observable)].append(temp2)

                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]  


        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                if i in experiments_want_to_plot_data_from:
                    if observable in exp['mole_fraction_observables']:
                        empty_nested_observable_list_time[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
                        observable_counter+=1
                        
                    if observable in exp['concentration_observables']:
                        empty_nested_observable_list_time[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
    
    
                        observable_counter+=1
            if i in experiments_want_to_plot_data_from:        
                if 'perturbed_coef' in exp.keys():
                    wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                    for k,wl in enumerate(wavelengths):
                        empty_nested_observable_list_time[observables_unique.index(wl)].append(exp['absorbance_experimental_data'][k]['time']*1e3)
    
                        interploated_temp = np.interp(exp['absorbance_experimental_data'][k]['time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(wl)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(wl)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
####################################################################################################################################################################################################################    
        empty_nested_observable_list_Y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_time_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_temperature_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_initial_temperature_2 = [[] for x in range(len(observables_unique))]


        if bool(experiments_want_to_plot_data_from_2):
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from_2:
                        temp = self.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y_2[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = self.y_matrix[start:stop,:]
                        empty_nested_observable_list_y_2[observables_unique.index(current_observable)].append(temp2)

                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]  


        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                if i in experiments_want_to_plot_data_from_2:
                    if observable in exp['mole_fraction_observables']:
                        empty_nested_observable_list_time_2[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature_2[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature_2[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
                        observable_counter+=1
                        
                    if observable in exp['concentration_observables']:
                        empty_nested_observable_list_time_2[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature_2[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature_2[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
    
    
                        observable_counter+=1
            if i in experiments_want_to_plot_data_from_2:        
                if 'perturbed_coef' in exp.keys():
                    wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                    for k,wl in enumerate(wavelengths):
                        empty_nested_observable_list_time_2[observables_unique.index(wl)].append(exp['absorbance_experimental_data'][k]['time']*1e3)
    
                        interploated_temp = np.interp(exp['absorbance_experimental_data'][k]['time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature_2[observables_unique.index(wl)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature_2[observables_unique.index(wl)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])    
###################################################################################################################################################################################################################  
        
        x = np.arange(10)
        ys = [i+x+(i*x)**2 for i in range(10)]
        colors=cm.rainbow(np.linspace(0,1,30))

        #colors = cm.rainbow(np.linspace(0, 1, len(ys)))

        
        for x,observable in enumerate(empty_nested_observable_list_Y):
            length_of_2nd_list = len(empty_nested_observable_list_Y_2[x])
            if bool(observable):
                plt.figure()
                if bool(csv):
                    df = pd.read_csv(csv)
                    plt.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_time'].dropna()*1e3,alpha=0.5,color='k',zorder=4)      
                for y,array in enumerate(empty_nested_observable_list_Y[x]):
                        plt.subplot(1,1,1)
                        plt.xlabel('Y')
                        plt.ylabel('Time')
                        plt.scatter(empty_nested_observable_list_Y[x][y],empty_nested_observable_list_time[x][y],label='Experiment_'+str(x)+'_observable_'+str(y),color='blue')
                        

                        if y<length_of_2nd_list:
                            plt.scatter(empty_nested_observable_list_Y_2[x][y],empty_nested_observable_list_time_2[x][y],color='red',zorder=4)

                        #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                        plt.title(observables_unique[x])
                

#                        plt.subplot(2,1,2)
#                        plt.scatter(empty_nested_observable_list_y[x][y],empty_nested_observable_list_time[x][y],color=colors[x])
#                        plt.xlabel('y')
#                        plt.ylabel('Time')
#                        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)

                #plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_time.pdf',dpi=1000,bbox_inches='tight')

        for x,observable in enumerate(empty_nested_observable_list_Y):
            length_of_2nd_list = len(empty_nested_observable_list_Y_2[x])

            if bool(observable):
                plt.figure()
                if bool(csv):
                    df = pd.read_csv(csv)
                    plt.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_Temperature'].dropna(),alpha=0.5,color='k',zorder=4)                    
                for y,array in enumerate(empty_nested_observable_list_Y[x]):
                    plt.subplot(1,1,1)
                    plt.scatter(empty_nested_observable_list_Y[x][y],empty_nested_observable_list_temperature[x][y],label='Experiment_'+str(x)+'_observable_'+str(y),color='blue')
                    #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                    plt.xlabel('Y')
                    plt.ylabel('Temperature')
                    plt.title(observables_unique[x])
                    
                    if y<length_of_2nd_list:
                        plt.scatter(empty_nested_observable_list_Y_2[x][y],empty_nested_observable_list_temperature_2[x][y],color='red',zorder=4)


#                    plt.subplot(2,1,2)
#                    
#                    plt.scatter(empty_nested_observable_list_y[x][y],empty_nested_observable_list_temperature[x][y],color=colors[x])    
#                    plt.xlabel('y')
#                    plt.ylabel('Temperature')
#                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
                #plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_temperature.pdf',dpi=1000,bbox_inches='tight')
        for x,observable in enumerate(empty_nested_observable_list_Y):
            length_of_2nd_list = len(empty_nested_observable_list_Y_2[x])
            
            if bool(observable):
                plt.figure()
                if bool(csv):
                    df = pd.read_csv(csv)
                    plt.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_initial_Temperature'].dropna(),alpha=0.5,color='k',zorder=4)                 
                for y,array in enumerate(empty_nested_observable_list_Y[x]):
                    plt.subplot(1,1,1)
                    plt.scatter(empty_nested_observable_list_Y[x][y],empty_nested_observable_list_initial_temperature[x][y],label='Experiment_'+str(x)+'_observable_'+str(y),color='blue')
                    #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                    plt.xlabel('Y')
                    plt.ylabel('Initial Temperature')
                    plt.title(observables_unique[x])
                    if y<length_of_2nd_list:
                        plt.scatter(empty_nested_observable_list_Y_2[x][y],empty_nested_observable_list_initial_temperature_2[x][y],color='red',zorder=4)
#                    plt.subplot(2,1,2)
#                    
#                    plt.scatter(empty_nested_observable_list_y[x][y],empty_nested_observable_list_initial_temperature[x][y],color=colors[x])    
#                    plt.xlabel('y')
#                    plt.ylabel('Initial Temperature')
#                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)    
                #plt.savefig(directory_to_save_images+'/'+str(observables_unique[x])+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_initial_temperature.pdf',dpi=1000,bbox_inches='tight')

    def plotting_T_and_time_full_simulation_individual_observables_for_paper_2(self,experiments_want_to_plot_data_from,
                                                                             bins='auto',
                                                                             directory_to_save_images='',csv='',experiments_want_to_plot_data_from_2=[]):
#working_here

        observables_total = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            single_experiment = []
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    single_experiment.append(observable)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:

                    single_experiment.append(observable)
                    
                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    single_experiment.append(wl)
                    
            observables_total.append(single_experiment)
        
        observables_flatten = [item for sublist in observables_total for item in sublist]
        from collections import OrderedDict
        observables_unique = list(OrderedDict.fromkeys(observables_flatten))
        
        empty_nested_observable_list_Y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_time = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_temperature = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_initial_temperature = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_z = [[] for x in range(len(observables_unique))]


        if bool(experiments_want_to_plot_data_from):
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from:
                        temp = self.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = self.y_matrix[start:stop,:]
                        empty_nested_observable_list_y[observables_unique.index(current_observable)].append(temp2)
                        
                        temp3 = self.Z_matrix[start:stop,:]
                        empty_nested_observable_list_z[observables_unique.index(current_observable)].append(temp3)
                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]  


        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                if i in experiments_want_to_plot_data_from:
                    if observable in exp['mole_fraction_observables']:
                        empty_nested_observable_list_time[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
                        observable_counter+=1
                        
                    if observable in exp['concentration_observables']:
                        empty_nested_observable_list_time[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
    
    
                        observable_counter+=1
            if i in experiments_want_to_plot_data_from:        
                if 'perturbed_coef' in exp.keys():
                    wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                    for k,wl in enumerate(wavelengths):
                        empty_nested_observable_list_time[observables_unique.index(wl)].append(exp['absorbance_experimental_data'][k]['time']*1e3)
    
                        interploated_temp = np.interp(exp['absorbance_experimental_data'][k]['time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(wl)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(wl)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
####################################################################################################################################################################################################################    
        empty_nested_observable_list_Y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_time_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_temperature_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_initial_temperature_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_z_2 = [[] for x in range(len(observables_unique))]


        if bool(experiments_want_to_plot_data_from_2):
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from_2:
                        temp = self.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y_2[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = self.y_matrix[start:stop,:]
                        empty_nested_observable_list_y_2[observables_unique.index(current_observable)].append(temp2)
                        
                        temp3 = self.Z_matrix[start:stop,:]
                        empty_nested_observable_list_z_2[observables_unique.index(current_observable)].append(temp3)
                        
                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]  


        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                if i in experiments_want_to_plot_data_from_2:
                    if observable in exp['mole_fraction_observables']:
                        empty_nested_observable_list_time_2[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature_2[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature_2[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
                        observable_counter+=1
                        
                    if observable in exp['concentration_observables']:
                        empty_nested_observable_list_time_2[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature_2[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature_2[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
    
    
                        observable_counter+=1
            if i in experiments_want_to_plot_data_from_2:        
                if 'perturbed_coef' in exp.keys():
                    wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                    for k,wl in enumerate(wavelengths):
                        empty_nested_observable_list_time_2[observables_unique.index(wl)].append(exp['absorbance_experimental_data'][k]['time']*1e3)
    
                        interploated_temp = np.interp(exp['absorbance_experimental_data'][k]['time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature_2[observables_unique.index(wl)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature_2[observables_unique.index(wl)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])    
###################################################################################################################################################################################################################  
        
        x = np.arange(10)
        ys = [i+x+(i*x)**2 for i in range(10)]
        colors=cm.rainbow(np.linspace(0,1,30))

        #colors = cm.rainbow(np.linspace(0, 1, len(ys)))
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(6,7))
        gs = gridspec.GridSpec(3, 1,height_ratios=[3,3,3],wspace=0.025,hspace=0.1)
        gs.update(wspace=0, hspace=0.7)
        ax1=plt.subplot(gs[0])
        ax2=plt.subplot(gs[1])
        ax3=plt.subplot(gs[2]) 
        
        fig2 = plt.figure(figsize=(6,7))
        gs2 = gridspec.GridSpec(3, 1,height_ratios=[3,3,3],wspace=0.025,hspace=0.1)
        gs2.update(wspace=0, hspace=0.7)
        ax4=plt.subplot(gs2[0])
        ax5=plt.subplot(gs2[1])
        ax6=plt.subplot(gs2[2]) 


        
        for x,observable in enumerate(empty_nested_observable_list_Y):
            length_of_2nd_list = len(empty_nested_observable_list_Y_2[x])
            if bool(observable):

                if x ==0:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax1.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_time'].dropna()*1e3,alpha=1,color='g',zorder=4,label='_nolegend_')      
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                            
                        
                            z_values = empty_nested_observable_list_z[x][y]
                            indecies = np.argwhere(z_values > 100)
                            new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                            new_y_test = np.delete(new_y_test,indecies)
                            new_time_test = copy.deepcopy(empty_nested_observable_list_time[x][y])
                            new_time_test = new_time_test.values
                            new_time_test = new_time_test.reshape((new_time_test.shape[0],1))
                            new_time_test  = np.delete(new_time_test,indecies)
                            ax1.scatter(new_y_test,new_time_test, c='#1f77b4',alpha=1)
                            ax1.set_xlabel('Relative Difference')
                            ax1.set_ylabel('Time (ms)')
                            
    
                            if y<length_of_2nd_list:
                                z_values_2 = empty_nested_observable_list_z_2[x][y]
                                indecies_2 = np.argwhere(z_values_2 > 100)
                                new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                                new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                                new_time_test_2 = copy.deepcopy(empty_nested_observable_list_time_2[x][y])
                                new_time_test_2 = new_time_test_2.values
                                new_time_test_2 = new_time_test_2.reshape((new_time_test_2.shape[0],1))
                                new_time_test_2  = np.delete(new_time_test_2,indecies_2)
                                ax1.scatter(new_y_test_2,new_time_test_2,color='orange',zorder=3,alpha=.15)
    
                            ax1.set_title(observables_unique[x])
                            ax1.set_xlim(left=-.25, right=.25, emit=True, auto=False)
                    ax1.scatter([],[],c='#1f77b4',label='#1')
                    ax1.scatter([],[],color='orange',label='#2')                                
                    #ax1.scatter([],[],color='green',label='#3')
                    ax1.legend(frameon=False)




                if x ==1:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax2.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_time'].dropna()*1e3,alpha=1,color='g',zorder=4)      
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                            
                        
                            z_values = empty_nested_observable_list_z[x][y]
                            indecies = np.argwhere(z_values > 100)
                            new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                            new_y_test = np.delete(new_y_test,indecies)
                            new_time_test = copy.deepcopy(empty_nested_observable_list_time[x][y])
                            new_time_test = new_time_test.values
                            new_time_test = new_time_test.reshape((new_time_test.shape[0],1))
                            new_time_test  = np.delete(new_time_test,indecies)
                            ax2.scatter(new_y_test,new_time_test, c='#1f77b4',alpha=1)
                            ax2.set_xlabel('Relative Difference')
                            ax2.set_ylabel('Time (ms)')
                            ax2.set_xlim(left=-.09, right=.09, emit=True, auto=False)
                            
    
                            if y<length_of_2nd_list:
                                z_values_2 = empty_nested_observable_list_z_2[x][y]
                                indecies_2 = np.argwhere(z_values_2 > 100)
                                new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                                new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                                new_time_test_2 = copy.deepcopy(empty_nested_observable_list_time_2[x][y])
                                new_time_test_2 = new_time_test_2.values
                                new_time_test_2 = new_time_test_2.reshape((new_time_test_2.shape[0],1))
                                new_time_test_2  = np.delete(new_time_test_2,indecies_2)
                                ax2.scatter(new_y_test_2,new_time_test_2,color='orange',zorder=3,alpha=.15)
    
                            #ax2.set_title(observables_unique[x])
                            ax2.set_title(r'H$_2$O')
                            
                            


                if x ==3:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax3.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_time'].dropna()*1e3,alpha=1,color='g',zorder=4,)      
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                            
                        
                            z_values = empty_nested_observable_list_z[x][y]
                            indecies = np.argwhere(z_values > 100)
                            new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                            new_y_test = np.delete(new_y_test,indecies)
                            new_time_test = copy.deepcopy(empty_nested_observable_list_time[x][y])
                            new_time_test = new_time_test.values
                            new_time_test = new_time_test.reshape((new_time_test.shape[0],1))
                            new_time_test  = np.delete(new_time_test,indecies)
                            ax3.scatter(new_y_test,new_time_test,c='#1f77b4',alpha=1)
                            ax3.set_xlabel('Relative Difference')
                            ax3.set_ylabel('Time (ms)')
                            ax3.set_xlim(left=-.3, right=.3, emit=True, auto=False)


                            
    
                            if y<length_of_2nd_list:
                                z_values_2 = empty_nested_observable_list_z_2[x][y]
                                indecies_2 = np.argwhere(z_values_2 > 100)
                                new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                                new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                                new_time_test_2 = copy.deepcopy(empty_nested_observable_list_time_2[x][y])
                                new_time_test_2 = new_time_test_2.values
                                new_time_test_2 = new_time_test_2.reshape((new_time_test_2.shape[0],1))
                                new_time_test_2  = np.delete(new_time_test_2,indecies_2)
                                ax3.scatter(new_y_test_2,new_time_test_2,color='orange',zorder=3,alpha=.15)
    
                            #ax3.set_title(observables_unique[x])
                            ax3.set_title('Absorbance '+ str(observables_unique[x])+str(' nm'))

                fig.savefig(directory_to_save_images+'/'+'Three_pannel_plot_'+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_time.pdf',dpi=1000,bbox_inches='tight')
                fig.savefig(directory_to_save_images+'/'+'Three_pannel_plot_'+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_time.png',dpi=1000,bbox_inches='tight')
                fig.savefig(directory_to_save_images+'/'+'Three_pannel_plot_'+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_time.svg',dpi=1000,bbox_inches='tight',transparent=True)

        for x,observable in enumerate(empty_nested_observable_list_Y):
            length_of_2nd_list = len(empty_nested_observable_list_Y_2[x])

            if bool(observable):
                
                if x==0:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax4.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_Temperature'].dropna(),label='_nolegend_',alpha=1,color='green',zorder=4)                    
                    
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                        
                        z_values = empty_nested_observable_list_z[x][y]
                        indecies = np.argwhere(z_values > 100)
                        new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                        new_y_test = np.delete(new_y_test,indecies)
                        new_temperature_test = copy.deepcopy(empty_nested_observable_list_temperature[x][y])
                        new_temperature_test = new_temperature_test.reshape((new_temperature_test.shape[0],1))
                        new_temperature_test  = np.delete(new_temperature_test,indecies)                    
                        
                        ax4.scatter(new_y_test,new_temperature_test,c='#1f77b4',alpha=1,label='_nolegend_')
                        #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                        ax4.set_xlabel('Relative Difference')
                        ax4.set_ylabel('Temperature (K)')
                        ax4.set_title(observables_unique[x])
                        ax4.set_xlim(left=-.25, right=.25, emit=True, auto=False)
                        
                        if y<length_of_2nd_list:
                            z_values_2 = empty_nested_observable_list_z_2[x][y]
                            indecies_2 = np.argwhere(z_values_2 > 100)
                            new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                            new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                            new_temperature_test_2 = copy.deepcopy(empty_nested_observable_list_temperature_2[x][y])
                            new_temperature_test_2 = new_temperature_test_2.reshape((new_temperature_test_2.shape[0],1))
                            new_temperature_test_2  = np.delete(new_temperature_test_2,indecies_2)                              
                            ax4.scatter(new_y_test_2,new_temperature_test_2,color='orange',zorder=3,alpha=1,label='_nolegend_')
                        
                    ax4.scatter([],[],c='#1f77b4',label='#1')
                    ax4.scatter([],[],c='orange',label='#2')
                    #ax4.scatter([],[],c='green',label='#3')
                    ax4.legend(frameon=False)
   
                                    
                if x==1:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax5.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_Temperature'].dropna(),alpha=1,color='green',zorder=4)                    
                    
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                        
                        z_values = empty_nested_observable_list_z[x][y]
                        indecies = np.argwhere(z_values > 100)
                        new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                        new_y_test = np.delete(new_y_test,indecies)
                        new_temperature_test = copy.deepcopy(empty_nested_observable_list_temperature[x][y])
                        new_temperature_test = new_temperature_test.reshape((new_temperature_test.shape[0],1))
                        new_temperature_test  = np.delete(new_temperature_test,indecies)                    
                        
                        ax5.scatter(new_y_test,new_temperature_test,c='#1f77b4',alpha=1)
                        #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                        ax5.set_xlabel('Relative Difference')
                        ax5.set_ylabel('Temperature (K)')
                        #ax5.set_title(observables_unique[x])
                        ax5.set_title(r'H$_2$O')
                        ax5.set_xlim(left=-.09, right=.09, emit=True, auto=False)

                        
                        if y<length_of_2nd_list:
                            z_values_2 = empty_nested_observable_list_z_2[x][y]
                            indecies_2 = np.argwhere(z_values_2 > 100)
                            new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                            new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                            new_temperature_test_2 = copy.deepcopy(empty_nested_observable_list_temperature_2[x][y])
                            new_temperature_test_2 = new_temperature_test_2.reshape((new_temperature_test_2.shape[0],1))
                            new_temperature_test_2  = np.delete(new_temperature_test_2,indecies_2)                              
                            ax5.scatter(new_y_test_2,new_temperature_test_2,color='orange',zorder=3,alpha=1)

                if x==3:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax6.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_Temperature'].dropna(),alpha=1,color='green',zorder=4)                    
                    
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                        
                        z_values = empty_nested_observable_list_z[x][y]
                        indecies = np.argwhere(z_values > 100)
                        new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                        new_y_test = np.delete(new_y_test,indecies)
                        new_temperature_test = copy.deepcopy(empty_nested_observable_list_temperature[x][y])
                        new_temperature_test = new_temperature_test.reshape((new_temperature_test.shape[0],1))
                        new_temperature_test  = np.delete(new_temperature_test,indecies)                    
                        
                        ax6.scatter(new_y_test,new_temperature_test,label='Experiment_'+str(x)+'_observable_'+str(y),c='#1f77b4',alpha=1)
                        #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                        ax6.set_xlabel('Relative Difference')
                        ax6.set_ylabel('Temperature (K)')
                        ax6.set_title('Absorbance '+ str(observables_unique[x])+str(' nm'))
                        ax6.set_xlim(left=-.3, right=.3, emit=True, auto=False)

                        if y<length_of_2nd_list:
                            z_values_2 = empty_nested_observable_list_z_2[x][y]
                            indecies_2 = np.argwhere(z_values_2 > 100)
                            new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                            new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                            new_temperature_test_2 = copy.deepcopy(empty_nested_observable_list_temperature_2[x][y])
                            new_temperature_test_2 = new_temperature_test_2.reshape((new_temperature_test_2.shape[0],1))
                            new_temperature_test_2  = np.delete(new_temperature_test_2,indecies_2)                              
                            ax6.scatter(new_y_test_2,new_temperature_test_2,color='orange',zorder=3,alpha=.5)
  
                fig2.savefig(directory_to_save_images+'/'+'Three_pannel_plot'+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_initial_temperature.pdf',dpi=1000,bbox_inches='tight')
                fig2.savefig(directory_to_save_images+'/'+'Three_pannel_plot'+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_initial_temperature.png',dpi=1000,bbox_inches='tight')
                fig2.savefig(directory_to_save_images+'/'+'Three_pannel_plot'+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_initial_temperature.svg',dpi=1000,bbox_inches='tight',transparent=True)

    def residual_sum_of_squares(self):
        overall_list = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            single_exp_dict = []
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    difference = exp['experimental_data'][observable_counter][observable].values - (exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6
                    square_of_differences = np.square(difference)
                    sum_of_squares = sum(square_of_differences)
                    single_exp_dict.append(sum_of_squares)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:
                    difference = exp['experimental_data'][observable_counter][observable+'_ppm'].values - (exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6
                    square_of_differences = np.square(difference)
                    sum_of_squares = sum(square_of_differences)
                    single_exp_dict.append(sum_of_squares)

                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    difference = exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values- exp['absorbance_model_data'][wl]
                    square_of_differences = np.square(difference)
                    sum_of_squares = sum(square_of_differences)
                    single_exp_dict.append(sum_of_squares)
            
            overall_list.append(single_exp_dict)
        
        return overall_list
                    
                    
    def sum_of_squares_of_Y(self):
        overall_list = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            single_exp_dict = []
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                plt.figure()
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    mean_calculated_experimental = np.mean(exp['experimental_data'][observable_counter][observable].values)
                    #mean_calculated_predicted = (exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6
                    
                    difference = exp['experimental_data'][observable_counter][observable].values - mean_calculated_experimental
                    square_of_differences = np.square(difference)
                    sum_of_squares = sum(square_of_differences)
                    single_exp_dict.append(sum_of_squares)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:
                    mean_calculated_experimental = np.mean(exp['experimental_data'][observable_counter][observable+'_ppm'].values)
                    difference = exp['experimental_data'][observable_counter][observable+'_ppm'].values - mean_calculated_experimental
                    square_of_differences = np.square(difference)
                    sum_of_squares = sum(square_of_differences)
                    #print(sum_of_squares)
                    single_exp_dict.append(sum_of_squares)

                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    mean_calculated_experimental = np.mean(exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values)
                    difference = exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values- mean_calculated_experimental
                    square_of_differences = np.square(difference)
                    sum_of_squares = sum(square_of_differences)
                    single_exp_dict.append(sum_of_squares)
            
            overall_list.append(single_exp_dict)
        
        return overall_list                    



    
    def calculating_R_squared(self,RSS_list,SYY_list):
        overall_list = []
        for i,lst in enumerate(RSS_list):
            single_exp_dict = []
            for j,value in enumerate(RSS_list[i]):
                single_exp_dict.append(1-(RSS_list[i][j]/SYY_list[i][j]))
            overall_list.append(single_exp_dict)

        
        return overall_list        
    
    
    def weighted_sum_of_squares(self):
        
        def uncertainty_calc(relative_uncertainty,absolute_uncertainty,data,experimental_data):

            if 'W' in list(experimental_data.columns):
                weighting_factor = experimental_data['W'].values
                if 'Relative_Uncertainty' in list(experimental_data.columns):
                    time_dependent_uncertainty = experimental_data['Relative_Uncertainty'].values
                    un_weighted_uncertainty = copy.deepcopy(time_dependent_uncertainty)
                    total_uncertainty = time_dependent_uncertainty/weighting_factor
                    
                else:
                    length_of_data = data.shape[0]
                    relative_uncertainty_array = np.full((length_of_data,1),relative_uncertainty) 
                    un_weighted_uncertainty = copy.deepcopy(relative_uncertainty_array)
                    total_uncertainty = un_weighted_uncertainty/weighting_factor
                
            
            
            elif 'Relative_Uncertainty' in list(experimental_data.columns):  
                
                time_dependent_uncertainty = experimental_data['Relative_Uncertainty'].values
                #do we need to take the natrual log of this?
                time_dependent_uncertainty = np.log(time_dependent_uncertainty+1)
                #do we need to take the natrual log of this?
                length_of_data = data.shape[0]
                un_weighted_uncertainty = copy.deepcopy(time_dependent_uncertainty)
                total_uncertainty = np.divide(time_dependent_uncertainty,(1/length_of_data**.5) )
#                

               
            else:
                length_of_data = data.shape[0]
                relative_uncertainty_array = np.full((length_of_data,1),relative_uncertainty)
                
                if absolute_uncertainty != 0:
                #check if this weighting factor is applied in the correct place 
                #also check if want these values to be the natural log values 
                    absolute_uncertainty_array = np.divide(data,absolute_uncertainty)
                    total_uncertainty = np.log(1 + np.sqrt(np.square(relative_uncertainty_array) + np.square(absolute_uncertainty_array)))
                    un_weighted_uncertainty = copy.deepcopy(total_uncertainty)
                     #weighting factor
                    total_uncertainty = np.divide(total_uncertainty,(1/length_of_data**.5) )
                
                else:
                    #total_uncertainty = np.log(1 + np.sqrt(np.square(relative_uncertainty_array)))
                    total_uncertainty = relative_uncertainty_array
                    #weighting factor
                    
                    un_weighted_uncertainty = copy.deepcopy(total_uncertainty)
                    total_uncertainty = np.divide(total_uncertainty,(1/length_of_data**.5) )

            #make this return a tuple 
            return total_uncertainty,un_weighted_uncertainty
        
        
        overall_list = []
        overall_list_maximum_deviation = []
        overall_list_weighted_uncertainty = []
        overall_percent_difference = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            single_exp_dict = []
            single_exp_dict_maximum_deviation = []
            single_experiment_weighted_uncertainty = []
            single_experiment_percent_difference = []
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    difference = exp['experimental_data'][observable_counter][observable].values - (exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['mole_fraction_relative_uncertainty'][observable_counter],
                        exp['uncertainty']['mole_fraction_absolute_uncertainty'][observable_counter],
                        exp['experimental_data'][observable_counter][observable].values,exp['experimental_data'][observable_counter])
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))
                    un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0],
                                                                               1))
                    
                    divided = np.divide(difference,un_weighted_uncertainty)
                    numerator = exp['experimental_data'][observable_counter][observable].values
                    numerator = numerator.reshape((numerator.shape[0],
                                                                               1))   
                    
                    percent_difference = np.divide(divided,np.divide(numerator,un_weighted_uncertainty))
                    single_experiment_percent_difference.append(np.max(np.absolute(percent_difference)))
                    
                    divided2 = np.divide(difference, total_uncertainty)
                    
                    single_exp_dict_maximum_deviation.append(np.max(np.absolute(divided)))
                    square_of_differences = np.square(divided)
                    square_of_differences2 = np.square(divided2)

                    
                    sum_of_squares = sum(square_of_differences)
                    sum_of_squares2 = sum(square_of_differences2)

                    sqrt_of_difference = np.sqrt(sum_of_squares)                    
                    single_exp_dict.append(sqrt_of_difference)
                    single_experiment_weighted_uncertainty.append(sum_of_squares2)
                    
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:
                    difference = exp['experimental_data'][observable_counter][observable+'_ppm'].values - (exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['concentration_relative_uncertainty'][observable_counter],
                         exp['uncertainty']['concentration_absolute_uncertainty'][observable_counter],
                         exp['experimental_data'][observable_counter][observable+'_ppm'].values,exp['experimental_data'][observable_counter])    
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))
                    un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0],
                                                                               1))                    
                    divided = np.divide(difference,un_weighted_uncertainty)
                    numerator = exp['experimental_data'][observable_counter][observable+'_ppm'].values
                    numerator = numerator.reshape((numerator.shape[0],
                                                                               1))   
                    
                    percent_difference = np.divide(divided,np.divide(numerator,un_weighted_uncertainty))
                    single_experiment_percent_difference.append(np.max(np.absolute(percent_difference)))                    
                    
                    divided2 = np.divide(difference, total_uncertainty)

                    single_exp_dict_maximum_deviation.append(np.max(np.absolute(divided)))

                    square_of_differences = np.square(divided)
                    square_of_differences2 = np.square(divided2)
                    sum_of_squares2 = sum(square_of_differences2)

                    sum_of_squares = sum(square_of_differences)
                    sqrt_of_difference = np.sqrt(sum_of_squares)                    
                    single_exp_dict.append(sqrt_of_difference)
                    single_experiment_weighted_uncertainty.append(sum_of_squares2)

                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    difference = exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values- exp['absorbance_model_data'][wl]
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['absorbance_relative_uncertainty'][k],
                                                             exp['uncertainty']['absorbance_absolute_uncertainty'][k],
                                                             exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values,exp['absorbance_experimental_data'][k])                    
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))
                    un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0],
                                                                               1))                    
                    
                    divided = np.divide(difference,un_weighted_uncertainty)
                    numerator =  exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values
                    numerator = numerator.reshape((numerator.shape[0],
                                                                               1))   
                    
                    percent_difference = np.divide(divided,np.divide(numerator,un_weighted_uncertainty))
                    single_experiment_percent_difference.append(np.max(np.absolute(percent_difference)))                    
                    single_exp_dict_maximum_deviation.append(np.max(np.absolute(divided)))
                    divided2 = np.divide(difference, total_uncertainty)
                    square_of_differences2 = np.square(divided2)
                    sum_of_squares2 = sum(square_of_differences2)
                    
                    square_of_differences = np.square(divided)
                    sum_of_squares = sum(square_of_differences)
                    sqrt_of_difference = np.sqrt(sum_of_squares)                    
                    single_exp_dict.append(sqrt_of_difference)
                    single_experiment_weighted_uncertainty.append(sum_of_squares2)

            overall_list.append(single_exp_dict)
            overall_list_maximum_deviation.append(single_exp_dict_maximum_deviation)
            overall_list_weighted_uncertainty.append(single_experiment_weighted_uncertainty)
            overall_percent_difference.append(single_experiment_percent_difference)
        return overall_list,overall_list_maximum_deviation,overall_list_weighted_uncertainty,overall_percent_difference
    
    def weighted_sum_of_squares_of_Y(self):
        
        def uncertainty_calc(relative_uncertainty,absolute_uncertainty,data,experimental_data):

            if 'W' in list(experimental_data.columns):
                weighting_factor = experimental_data['W'].values
                if 'Relative_Uncertainty' in list(experimental_data.columns):
                    time_dependent_uncertainty = experimental_data['Relative_Uncertainty'].values
                    un_weighted_uncertainty = copy.deepcopy(time_dependent_uncertainty)
                    total_uncertainty = time_dependent_uncertainty/weighting_factor
                    
                else:
                    length_of_data = data.shape[0]
                    relative_uncertainty_array = np.full((length_of_data,1),relative_uncertainty) 
                    un_weighted_uncertainty = copy.deepcopy(relative_uncertainty_array)
                    total_uncertainty = un_weighted_uncertainty/weighting_factor
                
            
            
            elif 'Relative_Uncertainty' in list(experimental_data.columns):  
                
                time_dependent_uncertainty = experimental_data['Relative_Uncertainty'].values
                #do we need to take the natrual log of this?
                time_dependent_uncertainty = np.log(time_dependent_uncertainty+1)
                #do we need to take the natrual log of this?
                length_of_data = data.shape[0]
                un_weighted_uncertainty = copy.deepcopy(time_dependent_uncertainty)
                total_uncertainty = np.divide(time_dependent_uncertainty,(1/length_of_data**.5) )
#                

               
            else:
                length_of_data = data.shape[0]
                relative_uncertainty_array = np.full((length_of_data,1),relative_uncertainty)
                
                if absolute_uncertainty != 0:
                #check if this weighting factor is applied in the correct place 
                #also check if want these values to be the natural log values 
                    absolute_uncertainty_array = np.divide(data,absolute_uncertainty)
                    total_uncertainty = np.log(1 + np.sqrt(np.square(relative_uncertainty_array) + np.square(absolute_uncertainty_array)))
                    un_weighted_uncertainty = copy.deepcopy(total_uncertainty)
                     #weighting factor
                    total_uncertainty = np.divide(total_uncertainty,(1/length_of_data**.5) )
                
                else:
                    #total_uncertainty = np.log(1 + np.sqrt(np.square(relative_uncertainty_array)))
                    total_uncertainty = relative_uncertainty_array
                    #weighting factor
                    
                    un_weighted_uncertainty = copy.deepcopy(total_uncertainty)
                    total_uncertainty = np.divide(total_uncertainty,(1/length_of_data**.5) )

            #make this return a tuple 
            return total_uncertainty,un_weighted_uncertainty        
        
        overall_list = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            single_exp_dict = []
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    mean_calculated_experimental = np.mean(exp['experimental_data'][observable_counter][observable].values)
                    #mean_calculated_predicted = (exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['mole_fraction_relative_uncertainty'][observable_counter],
                        exp['uncertainty']['mole_fraction_absolute_uncertainty'][observable_counter],
                        exp['experimental_data'][observable_counter][observable].values,exp['experimental_data'][observable_counter])                    
                    difference = exp['experimental_data'][observable_counter][observable].values - mean_calculated_experimental
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))                    
                    weighted_difference = np.divide(difference,total_uncertainty)
                    square_of_differences = np.square(weighted_difference)
                    sum_of_squares = sum(square_of_differences)
                    single_exp_dict.append(sum_of_squares)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:
                    mean_calculated_experimental = np.mean(exp['experimental_data'][observable_counter][observable+'_ppm'].values)
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['concentration_relative_uncertainty'][observable_counter],
                         exp['uncertainty']['concentration_absolute_uncertainty'][observable_counter],
                         exp['experimental_data'][observable_counter][observable+'_ppm'].values,exp['experimental_data'][observable_counter])    
                    
                    difference = exp['experimental_data'][observable_counter][observable+'_ppm'].values - mean_calculated_experimental
                    
                                        
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))                    
                    weighted_difference = np.divide(difference,total_uncertainty)

                    square_of_differences = np.square(weighted_difference)
                    sum_of_squares = sum(square_of_differences)
                    single_exp_dict.append(sum_of_squares)

                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    mean_calculated_experimental = np.mean(exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values)
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['absorbance_relative_uncertainty'][k],
                                                             exp['uncertainty']['absorbance_absolute_uncertainty'][k],
                                                             exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values,exp['absorbance_experimental_data'][k])                      
                    difference = exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values- mean_calculated_experimental
                    
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))  
                    weighted_difference = np.divide(difference,total_uncertainty)
                    square_of_differences = np.square(weighted_difference)
                    sum_of_squares = sum(square_of_differences)
                    single_exp_dict.append(sum_of_squares)
            
            overall_list.append(single_exp_dict)
        
        return overall_list    


    def plotting_individual_histograms(self,experimental_dict_list,parsed_yaml_list_optimized,directory_to_save_images=''):
        
        def uncertainty_calc(relative_uncertainty,absolute_uncertainty,data,experimental_data):

            if 'W' in list(experimental_data.columns):
                weighting_factor = experimental_data['W'].values
                if 'Relative_Uncertainty' in list(experimental_data.columns):
                    time_dependent_uncertainty = experimental_data['Relative_Uncertainty'].values
                    un_weighted_uncertainty = copy.deepcopy(time_dependent_uncertainty)
                    total_uncertainty = time_dependent_uncertainty/weighting_factor
                    
                else:
                    length_of_data = data.shape[0]
                    relative_uncertainty_array = np.full((length_of_data,1),relative_uncertainty) 
                    un_weighted_uncertainty = copy.deepcopy(relative_uncertainty_array)
                    total_uncertainty = un_weighted_uncertainty/weighting_factor
                
            
            
            elif 'Relative_Uncertainty' in list(experimental_data.columns):  
                
                time_dependent_uncertainty = experimental_data['Relative_Uncertainty'].values
                #do we need to take the natrual log of this?
                time_dependent_uncertainty = np.log(time_dependent_uncertainty+1)
                #do we need to take the natrual log of this?
                length_of_data = data.shape[0]
                un_weighted_uncertainty = copy.deepcopy(time_dependent_uncertainty)
                total_uncertainty = np.divide(time_dependent_uncertainty,(1/length_of_data**.5) )
#                

               
            else:
                length_of_data = data.shape[0]
                relative_uncertainty_array = np.full((length_of_data,1),relative_uncertainty)
                
                if absolute_uncertainty != 0:
                #check if this weighting factor is applied in the correct place 
                #also check if want these values to be the natural log values 
                    absolute_uncertainty_array = np.divide(data,absolute_uncertainty)
                    total_uncertainty = np.log(1 + np.sqrt(np.square(relative_uncertainty_array) + np.square(absolute_uncertainty_array)))
                    un_weighted_uncertainty = copy.deepcopy(total_uncertainty)
                     #weighting factor
                    total_uncertainty = np.divide(total_uncertainty,(1/length_of_data**.5) )
                
                else:
                    #total_uncertainty = np.log(1 + np.sqrt(np.square(relative_uncertainty_array)))
                    total_uncertainty = relative_uncertainty_array
                    #weighting factor
                    
                    un_weighted_uncertainty = copy.deepcopy(total_uncertainty)
                    total_uncertainty = np.divide(total_uncertainty,(1/length_of_data**.5) )

            #make this return a tuple 
            return total_uncertainty,un_weighted_uncertainty
        
        
        for i,exp in enumerate(experimental_dict_list):
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                if observable in exp['mole_fraction_observables']:
                    difference = exp['experimental_data'][observable_counter][observable].values - (exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6
                    log_difference = np.log(exp['experimental_data'][observable_counter][observable].values)-np.log((exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6)
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['mole_fraction_relative_uncertainty'][observable_counter],
                        exp['uncertainty']['mole_fraction_absolute_uncertainty'][observable_counter],
                        exp['experimental_data'][observable_counter][observable].values,exp['experimental_data'][observable_counter])
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))
                    un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0],
                                                                               1))
                    
                    divided = np.divide(difference,un_weighted_uncertainty)
                    numerator = exp['experimental_data'][observable_counter][observable].values
                    numerator = numerator.reshape((numerator.shape[0],
                                                                               1))   
                    
                    log_difference = log_difference.reshape((log_difference.shape[0],
                                                                               1))                       
                    
                    weighted_log_difference = np.divide(log_difference,total_uncertainty)
                    
                    plt.subplot(2,1,1)
                    plt.hist(difference)
                    plt.xlabel('Y')
                    plt.title(observable)
                    
                    plt.subplot(2,1,2)
                    plt.xlabel('y')
                    plt.hist(weighted_log_difference)
                    plt.title(observable)
                    
                    
                    plt.subplot(2,1,1)
                    plt.figure()
                    plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,difference)
                    plt.ylabel('Y')
                    plt.xlabel('Time (ms)')
                    plt.title(observable)
                    plt.subplot(2,1,2)
                    plt.figure(exp['experimental_data'][observable_counter]['Time']*1e3,weighted_log_difference)
                    plt.xlabel('Time (ms)')
                    plt.ylabel('y')
                    
                    
                    plt.subplot(2,1,1)
                    plt.figure(exp['simulation'].timeHistoryInterpToExperiment['temperature'].dropna().values, difference)
                    plt.ylabel('Y')
                    plt.xlabel('Temperature')                    
                    plt.subplot(2,1,2)
                    plt.figure(exp['simulation'].timeHistoryInterpToExperiment['temperature'].dropna().values, weighted_log_difference)
                    plt.ylabel('y')
                    plt.xlabel('Temperature')                         
                    
                    
                    
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:
                    difference = exp['experimental_data'][observable_counter][observable+'_ppm'].values - (exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6
                    log_difference = np.log(exp['experimental_data'][observable_counter][observable+'_ppm'].values) - np.log((exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6)
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['concentration_relative_uncertainty'][observable_counter],
                         exp['uncertainty']['concentration_absolute_uncertainty'][observable_counter],
                         exp['experimental_data'][observable_counter][observable+'_ppm'].values,exp['experimental_data'][observable_counter])    
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))
                    un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0],
                                                                               1))                    
                    divided = np.divide(difference,un_weighted_uncertainty)
                    numerator = exp['experimental_data'][observable_counter][observable+'_ppm'].values
                    numerator = numerator.reshape((numerator.shape[0],
                                                                               1))   
                    
                    log_difference = log_difference.reshape((log_difference.shape[0],
                                                                               1))                       
                    
                    weighted_log_difference = np.divide(log_difference,total_uncertainty)
                    
                    plt.figure()
                    plt.subplot(2,1,1)
                    plt.hist(difference)
                    plt.xlabel('Y')
                    plt.title(observable + ' ' + 'Experiment Number' + ' '+str(i))
                    
                    plt.subplot(2,1,2)
                    plt.xlabel('y')
                    plt.hist(weighted_log_difference)
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)

                    plt.savefig(directory_to_save_images+'/'+observable+'_'+'Experiment Number_' +str(i)+'_hist.pdf',dpi=1000,bbox_inches='tight')
                    plt.savefig(directory_to_save_images+'/'+observable+'_'+'Experiment Number_' +str(i)+'_hist.svg',dpi=1000,bbox_inches='tight',transparent=True)
                    plt.savefig(directory_to_save_images+'/'+observable+'_'+'Experiment Number_' +str(i)+'_hist.png',dpi=1000,bbox_inches='tight')
                    
                    #plt.subplots_adjust( hspace=.2)

                    plt.figure()
                    plt.subplot(2,1,1)
                    plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,difference)
                    plt.ylabel('Y')
                    plt.title(observable+ ' ' + 'Experiment Number' + ' '+str(i))
                    plt.subplot(2,1,2)
                    plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,weighted_log_difference)
                    plt.xlabel('Time (ms)')
                    plt.ylabel('y')
                    plt.savefig(directory_to_save_images+'/'+observable+'_'+'Experiment Number_'+str(i)+'_timeVSy.pdf',dpi=1000,bbox_inches='tight')
                    plt.savefig(directory_to_save_images+'/'+observable+'_'+'Experiment Number_'+str(i)+'_timeVSy.svg',dpi=1000,bbox_inches='tight',transparent=True)
                    plt.savefig(directory_to_save_images+'/'+observable+'_'+'Experiment Number_'+str(i)+'_timeVSy.png',dpi=1000,bbox_inches='tight')
                    
                    
                    plt.figure()
                    plt.subplot(2,1,1)
                    interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                    plt.plot(interploated_temp, difference)
                    plt.ylabel('Y')
                    plt.title(observable+ ' ' + 'Experiment Number' + ' '+str(i))
                    plt.subplot(2,1,2)
                    plt.plot(interploated_temp, weighted_log_difference)
                    plt.ylabel('y')
                    plt.xlabel('Temperature') 
                    plt.savefig(directory_to_save_images+'/'+observable+'_'+'Experiment Number_'+str(i)+'_tempVSy.pdf',dpi=1000,bbox_inches='tight')
                    plt.savefig(directory_to_save_images+'/'+observable+'_'+'Experiment Number_'+str(i)+'_tempVSy.svg',dpi=1000,bbox_inches='tight',transparent=True)
                    plt.savefig(directory_to_save_images+'/'+observable+'_'+'Experiment Number_'+str(i)+'_tempVSy.png',dpi=1000,bbox_inches='tight')

                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                plt.figure()
                for k,wl in enumerate(wavelengths):
                    difference = exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values- exp['absorbance_model_data'][wl]
                    log_difference = np.log(exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values) - np.log(exp['absorbance_model_data'][wl])
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['absorbance_relative_uncertainty'][k],
                                                             exp['uncertainty']['absorbance_absolute_uncertainty'][k],
                                                             exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values,exp['absorbance_experimental_data'][k])                    
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))
                    un_weighted_uncertainty = un_weighted_uncertainty.reshape((un_weighted_uncertainty.shape[0],
                                                                               1))                    
                    
                    divided = np.divide(difference,un_weighted_uncertainty)
                    numerator =  exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values
                    numerator = numerator.reshape((numerator.shape[0],
                                                                               1))   
                    log_difference = log_difference.reshape((log_difference.shape[0],
                                                                               1))                       
                    
                    weighted_log_difference = np.divide(log_difference,total_uncertainty)
                    
                    plt.figure()
                    plt.subplot(2,1,1)
                    plt.title('Absorbance'+ ' ' +str(wl)+ ' ' + 'Experiment Number' + ' '+str(i))
                    plt.xlabel('Y')

                    plt.hist(log_difference)
                    
                    plt.subplot(2,1,2)
                    plt.hist(weighted_log_difference)
                    plt.xlabel('y')
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)
                    plt.savefig(directory_to_save_images+'/'+'Absorbance'+ '_' +str(wl)+ '_' + 'Experiment Number' + '_'+str(i)+'_hist.pdf',dpi=1000,bbox_inches='tight')
                    plt.savefig(directory_to_save_images+'/'+'Absorbance'+ '_' +str(wl)+ '_' + 'Experiment Number' + '_'+str(i)+'_hist.svg',dpi=1000,bbox_inches='tight',transparent=True)
                    plt.savefig(directory_to_save_images+'/'+'Absorbance'+ '_' +str(wl)+ '_' + 'Experiment Number' + '_'+str(i)+'_hist.png',dpi=1000,bbox_inches='tight')                     

                    plt.figure()
                    plt.subplot(2,1,1)
                    plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,difference)
                    plt.ylabel('Y')
                    plt.title('Absorbance'+' '+str(wl)+ ' ' + 'Experiment Number' + ' '+str(i))
                    plt.subplot(2,1,2)
                    plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,weighted_log_difference)
                    plt.xlabel('Time (ms)')
                    plt.ylabel('y')
                    plt.savefig(directory_to_save_images+'/'+'Absorbance'+ '_' +str(wl)+ ' ' + 'Experiment Number' + '_'+str(i)+'_timeVSy.pdf',dpi=1000,bbox_inches='tight')
                    plt.savefig(directory_to_save_images+'/'+'Absorbance'+ '_' +str(wl)+ ' ' + 'Experiment Number' + '_'+str(i)+'_timeVSy.svg',dpi=1000,bbox_inches='tight',transparent=True)
                    plt.savefig(directory_to_save_images+'/'+'Absorbance'+ '_' +str(wl)+ ' ' + 'Experiment Number' + '_'+str(i)+'_timeVSy.png',dpi=1000,bbox_inches='tight')
 
                    
                    plt.figure()                    
                    plt.subplot(2,1,1)
                    interploated_temp = np.interp(exp['absorbance_experimental_data'][k]['time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                    plt.plot(interploated_temp,difference)
                    plt.title('Absorbance'+' '+str(wl)+ ' ' + 'Experiment Number' + ' '+str(i))
                    plt.ylabel('Y')
                    plt.subplot(2,1,2)
                    plt.plot(interploated_temp,weighted_log_difference)
                    plt.xlabel('Temperature') 
                    plt.ylabel('y')
                    plt.savefig(directory_to_save_images+'/'+'Absorbance'+ '_' +str(wl)+ '_' + 'Experiment Number' + '_'+str(i)+'_tempVSy.pdf',dpi=1000,bbox_inches='tight')
                    plt.savefig(directory_to_save_images+'/'+'Absorbance'+ '_' +str(wl)+ '_' + 'Experiment Number' + '_'+str(i)+'_tempVSy.svg',dpi=1000,bbox_inches='tight',transparent=True)
                    plt.savefig(directory_to_save_images+'/'+'Absorbance'+ '_' +str(wl)+ '_' + 'Experiment Number' + '_'+str(i)+'_tempVSy.png',dpi=1000,bbox_inches='tight')

        return 
    
    
    def objective_functions(self,exp_dict_list):
        
        def uncertainty_calc(relative_uncertainty,absolute_uncertainty,data,experimental_data):

            if 'W' in list(experimental_data.columns):
                weighting_factor = experimental_data['W'].values
                if 'Relative_Uncertainty' in list(experimental_data.columns):
                    time_dependent_uncertainty = experimental_data['Relative_Uncertainty'].values
                    un_weighted_uncertainty = copy.deepcopy(time_dependent_uncertainty)
                    total_uncertainty = time_dependent_uncertainty/weighting_factor
                    
                else:
                    length_of_data = data.shape[0]
                    relative_uncertainty_array = np.full((length_of_data,1),relative_uncertainty) 
                    un_weighted_uncertainty = copy.deepcopy(relative_uncertainty_array)
                    total_uncertainty = un_weighted_uncertainty/weighting_factor
                
            
            
            elif 'Relative_Uncertainty' in list(experimental_data.columns):  
                
                time_dependent_uncertainty = experimental_data['Relative_Uncertainty'].values
                #do we need to take the natrual log of this?
                time_dependent_uncertainty = np.log(time_dependent_uncertainty+1)
                #do we need to take the natrual log of this?
                length_of_data = data.shape[0]
                un_weighted_uncertainty = copy.deepcopy(time_dependent_uncertainty)
                total_uncertainty = np.divide(time_dependent_uncertainty,(1/length_of_data**.5) )
#                

               
            else:
                length_of_data = data.shape[0]
                relative_uncertainty_array = np.full((length_of_data,1),relative_uncertainty)
                
                if absolute_uncertainty != 0:
                #check if this weighting factor is applied in the correct place 
                #also check if want these values to be the natural log values 
                    absolute_uncertainty_array = np.divide(data,absolute_uncertainty)
                    total_uncertainty = np.log(1 + np.sqrt(np.square(relative_uncertainty_array) + np.square(absolute_uncertainty_array)))
                    un_weighted_uncertainty = copy.deepcopy(total_uncertainty)
                     #weighting factor
                    total_uncertainty = np.divide(total_uncertainty,(1/length_of_data**.5) )
                
                else:
                    #total_uncertainty = np.log(1 + np.sqrt(np.square(relative_uncertainty_array)))
                    total_uncertainty = relative_uncertainty_array
                    #weighting factor
                    
                    un_weighted_uncertainty = copy.deepcopy(total_uncertainty)
                    total_uncertainty = np.divide(total_uncertainty,(1/length_of_data**.5) )

            #make this return a tuple 
            return total_uncertainty,un_weighted_uncertainty        
        
        objective_function_w_inside = []
        objective_function_w_outside = []
        
        for i,exp in enumerate(exp_dict_list):
            single_exp_dict_w_inside = []
            single_exp_dict_w_outside = []
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['mole_fraction_relative_uncertainty'][observable_counter],
                        exp['uncertainty']['mole_fraction_absolute_uncertainty'][observable_counter],
                        exp['experimental_data'][observable_counter][observable].values,exp['experimental_data'][observable_counter])                    
                    difference = np.log(exp['experimental_data'][observable_counter][observable].values) - np.log((exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6)
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))                    
                    weighted_difference = np.divide(difference,total_uncertainty)
                    square_unweighted_differences = np.square(difference)
                    square_of_differences = np.square(weighted_difference)
                    sum_of_squares = sum(square_of_differences)
                    square_unweighted_differences = difference*total_uncertainty
                    sum_of_squares_w_outside = sum(square_unweighted_differences)
                    single_exp_dict_w_outside.append((sum_of_squares_w_outside[0],np.shape(total_uncertainty[0])))
                    single_exp_dict_w_inside.append((sum_of_squares[0],np.shape(total_uncertainty)[0]))
                    
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['concentration_relative_uncertainty'][observable_counter],
                         exp['uncertainty']['concentration_absolute_uncertainty'][observable_counter],
                         exp['experimental_data'][observable_counter][observable+'_ppm'].values,exp['experimental_data'][observable_counter])    
                    
                    difference = np.log(exp['experimental_data'][observable_counter][observable+'_ppm'].values) - np.log((exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)*1e6)
                    
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))                    
                    weighted_difference = np.divide(difference,total_uncertainty)
                    square_unweighted_differences = np.square(difference)
                    square_of_differences = np.square(weighted_difference)
                    sum_of_squares = sum(square_of_differences)
                    square_unweighted_differences = difference*total_uncertainty
                    sum_of_squares_w_outside = sum(square_unweighted_differences)
                    single_exp_dict_w_outside.append((sum_of_squares_w_outside[0],np.shape(difference)[0]))
                    single_exp_dict_w_inside.append((sum_of_squares[0],np.shape(difference)[0]))
                    
                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    total_uncertainty,un_weighted_uncertainty = uncertainty_calc(exp['uncertainty']['absorbance_relative_uncertainty'][k],
                                                             exp['uncertainty']['absorbance_absolute_uncertainty'][k],
                                                             exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values,exp['absorbance_experimental_data'][k])                      
                    difference = np.log(exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)].values)- np.log(exp['absorbance_model_data'][wl])
                    
                    difference = difference.reshape((difference.shape[0],
                                                                               1))
                                        
                    total_uncertainty = total_uncertainty.reshape((total_uncertainty.shape[0],
                                                                               1))                    
                    weighted_difference = np.divide(difference,total_uncertainty)
                    square_unweighted_differences = np.square(difference)
                    square_of_differences = np.square(weighted_difference)
                    sum_of_squares = sum(square_of_differences)
                    square_unweighted_differences = difference*total_uncertainty
                    sum_of_squares_w_outside = sum(square_unweighted_differences)
                    single_exp_dict_w_outside.append((sum_of_squares_w_outside[0],np.shape(difference)[0]))
                    single_exp_dict_w_inside.append((sum_of_squares[0],np.shape(difference)[0]))
                    
            
            objective_function_w_inside.append(single_exp_dict_w_inside)
            objective_function_w_outside.append(single_exp_dict_w_outside)
        
        return objective_function_w_inside,objective_function_w_outside
    
    def post_proessing_computing_cost_function(self,objective_function_list,experiments_to_consider=[]):
        #experiments_to_consider = [0,1,2,3,4,5,6,7,8]
        
        overall_list = []
        total_sum = []
        for i,value in enumerate(experiments_to_consider):
            for j,tupl in enumerate(objective_function_list[value]):
                overall_list.append(tupl[0])
                total_sum.append(tupl[1])
        

        objective_function_weighted = sum(overall_list)/sum(total_sum)
        objective_function_not_weighted = sum(overall_list)
        return objective_function_weighted , objective_function_not_weighted
    
    
    
    
    
    
    
    
    
    
    
    def plotting_histograms_of_individual_observables_for_hong_data(self,MSI_instance_one,MSI_instance_two,
                                                                    experiments_want_to_plot_data_from,
                                                                    experiments_want_to_plot_data_from_2=[],
                                                                    bins='auto',directory_to_save_images='',csv=''):
        s_shape = MSI_instance_one.S_matrix.shape[1]
        #if MSI_instance_one.k_target_value_S_matrix.any():
            #target_values_for_s = MSI_instance_one.k_target_value_S_matrix
            #s_shape = s_shape+target_values_for_s.shape[0]
        y_shape = MSI_instance_one.y_matrix.shape[0]
        difference = y_shape-s_shape
        y_values = MSI_instance_one.y_matrix[0:difference,0]
        Y_values = MSI_instance_one.Y_matrix[0:difference,0]
        self.lengths_of_experimental_data()

        #plotting_Y Histagrams 
        #edit this part 
        #obserervable_list = []
        
        observables_total = []
        
        for i,exp in enumerate(self.exp_dict_list_original):
            observable_counter=0
            single_experiment = []
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    single_experiment.append(observable)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:

                    single_experiment.append(observable)
                    
                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    single_experiment.append(wl)
                    
            observables_total.append(single_experiment)
        observables_flatten = [item for sublist in observables_total for item in sublist]
        from collections import OrderedDict
        observables_unique = list(OrderedDict.fromkeys(observables_flatten))
        
        
        empty_nested_observable_list_Y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y = [[] for x in range(len(observables_unique))]
        
        empty_nested_observable_list_Z = [[] for x in range(len(observables_unique))]
        
        empty_nested_observable_list_Y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_Z_2 = [[] for x in range(len(observables_unique))]

        if bool(experiments_want_to_plot_data_from):
            # print('inside here')
            y_values = []
            Y_values = []
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from:
                        temp = MSI_instance_two.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = MSI_instance_two.y_matrix[start:stop,:]
                        empty_nested_observable_list_y[observables_unique.index(current_observable)].append(temp2)

                        temp3 = MSI_instance_two.Z_matrix[start:stop,:]
                        empty_nested_observable_list_Z[observables_unique.index(current_observable)].append(temp3)
                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]              
          
        if bool(experiments_want_to_plot_data_from_2):
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from_2:
                        #print(x)
                        #print(current_observable,'this is current')

                        temp = MSI_instance_one.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y_2[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = MSI_instance_one.y_matrix[start:stop,:]
                        empty_nested_observable_list_y_2[observables_unique.index(current_observable)].append(temp2)

                        temp3 = MSI_instance_one.Z_matrix[start:stop,:]
                        empty_nested_observable_list_Z_2[observables_unique.index(current_observable)].append(temp3)
                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]                     
                    
        import matplotlib.gridspec as gridspec
    
        fig = plt.figure(figsize=(6,7))

        gs = gridspec.GridSpec(3, 1,height_ratios=[3,3,3],wspace=0.1,hspace=0.1)
        gs.update(wspace=0, hspace=0.7)
        ax1=plt.subplot(gs[0])
        ax2=plt.subplot(gs[1])
        ax3=plt.subplot(gs[2])  
        for i,observable in enumerate(empty_nested_observable_list_Y):
            new_Y_test_2 =[]
            if bool(observable):
                Y_values = np.vstack((observable))
                y_values = np.vstack((empty_nested_observable_list_y[i]))
                z_values = np.vstack((empty_nested_observable_list_Z[i]))
                indecies = np.argwhere(z_values > 100)
                new_y_test = copy.deepcopy(Y_values)
                new_y_test = np.delete(new_y_test,indecies)
             #   print(indecies.shape)
            #    print(indecies)
            #    print(i)
                if bool(experiments_want_to_plot_data_from_2) and bool(empty_nested_observable_list_y_2[i]):
                    Y_values_2 = np.vstack((empty_nested_observable_list_Y_2[i]))
                    y_values_2 = np.vstack((empty_nested_observable_list_y_2[i]))
                    z_values_2 = np.vstack((empty_nested_observable_list_Z_2[i]))
                    indecies_2 = np.argwhere(z_values_2 > 100)
                    new_Y_test_2 = copy.deepcopy(Y_values_2)
                    new_Y_test_2 = np.delete(new_Y_test_2,indecies_2)
                    

                if i ==0:
                #n, bins2, patches = plt.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    #ax1.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    n,bins_test_1,patches = ax1.hist(new_y_test,bins=bins ,align='mid',density=True,label='#1')
                    #comment
                    ax1.set_xlim(left=-1, right=1, emit=True, auto=False)
                    ax1.set_ylim(top=7,bottom=0)

                    ax1.set_xlabel('Y')
                    ax1.set_xlabel('Relative Difference')
                #plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
                    ax1.set_title(str(observables_unique[i]))
                    ax1.set_ylabel('pdf')
                    
                    (mu, sigma) = norm.fit(new_y_test)
                    yy = norm.pdf(bins_test_1,mu,sigma)
                    xx = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                    #l = ax1.plot(bins_test_1, yy, color='blue', linestyle='--',linewidth=2)
                    ax1.plot(xx, stats.norm.pdf(xx, mu, sigma),color='blue')
                    ax1.text(.04,3,r'$\mu=%.3f,\ \sigma=%.3f$' %(mu, sigma),color='blue', fontsize=11)

                #plt.ylabel('normalized')
                    if bool(experiments_want_to_plot_data_from_2):
                   # plt.hist(Y_values_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        #ax1.hist(new_Y_test_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        n,bins_test_1,patches = ax1.hist(new_Y_test_2,bins=bins ,align='mid',density=True,alpha=0.5,label='#2')
                        
                            
                        (mu, sigma) = norm.fit(new_Y_test_2)
                        yy = norm.pdf(bins_test_1,mu,sigma)
                        xx = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                        #l = ax1.plot(bins_test_1, yy, color='orange',linestyle='--', linewidth=2)
                        ax1.plot(xx, stats.norm.pdf(xx, mu, sigma),color='orange')
                        ax1.text(.04,2,r'$\mu=%.3f,\ \sigma=%.3f$' %(mu, sigma),color='orange', fontsize=11)
                    if bool(csv):
                        df = pd.read_csv(csv)
                        #ax1.hist(df[str(observables_unique[i])+'_Y'].dropna()*-1,bins=bins ,align='mid',density=True,alpha=0.5,label='Hong vs. Hong')
                        #ax1.hist(df[str(observables_unique[i])+'_Y'].dropna()*-1,bins=bins ,align='mid',density=True,alpha=0.5,label='#3')

                    ax1.legend()
                if i ==1:
                #n, bins2, patches = plt.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    n,bins_test_2,patches = ax2.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    #comment
                    #ax2.set_xlim(left=-.08, right=.08, emit=True, auto=False)
                    ax2.set_ylim(top=17,bottom=0)
                    ax2.set_xlabel('Y')
                    ax2.set_xlabel('Relative Difference')
                #plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
                    #ax2.set_title(str(observables_unique[i]))
                    ax2.set_title(r'H$_2$O')
                    ax2.set_ylabel('pdf')
                    
                    (mu, sigma) = norm.fit(new_y_test)
                    yy = norm.pdf(bins_test_2,mu,sigma)
                    xx = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                    #l = ax2.plot(bins_test_2, yy, color='blue', linestyle='--', linewidth=2)
                    ax2.plot(xx, stats.norm.pdf(xx, mu, sigma),color='blue')
                    ax2.text(.015,15,r'$\mu=%.3f,\ \sigma=%.3f$' %(mu, sigma),color='blue', fontsize=11)
                    
                    
                #plt.ylabel('normalized')
                    if bool(experiments_want_to_plot_data_from_2):
                   # plt.hist(Y_values_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        n,bins_test_2,patches = ax2.hist(new_Y_test_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        
                        (mu, sigma) = norm.fit(new_Y_test_2)
                        yy = norm.pdf(bins_test_2,mu,sigma)
                        xx = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                        #l = ax2.plot(bins_test_2, yy, color='orange',linestyle='--', linewidth=2)
                        ax2.plot(xx, stats.norm.pdf(xx, mu, sigma),color='orange')
                        ax2.text(.015,12.8,r'$\mu=%.3f,\ \sigma=%.3f$' %(mu, sigma),color='orange', fontsize=11)
                    if bool(csv):
                        df = pd.read_csv(csv)
                        #ax2.hist(df[str(observables_unique[i])+'_Y'].dropna()*-1,bins=bins ,align='mid',density=True,alpha=0.5,label='Hong vs. Hong')
                
                
                if i ==3:
                #n, bins2, patches = plt.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    n,bins_test_3,patches = ax3.hist(new_y_test,bins=bins ,align='mid',density=True,label='Hong Experiments')
                    #comment
                    #ax3.set_xlim(left=-.15, right=.15, emit=True, auto=False)
                    ax3.set_ylim(top=12,bottom=0)

                    ax3.set_xlabel('Y')
                    ax3.set_xlabel('Relative Difference')
                    ax3.set_ylabel('pdf')
                #plt.title(str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from))
                    ax3.set_title(str(observables_unique[i]))
                    ax3.set_title('Absorbance '+ str(observables_unique[i])+ ' nm')

                    (mu, sigma) = norm.fit(new_y_test)
                    yy = norm.pdf(bins_test_3,mu,sigma)
                    xx = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                    #l = ax3.plot(bins_test_3, yy, color='blue',linestyle='--',linewidth=2)
                    ax3.plot(xx, stats.norm.pdf(xx, mu, sigma),color='blue')
                    ax3.text(.037,10.5,r'$\mu=%.3f,\ \sigma=%.3f$' %(mu, sigma),color='blue', fontsize=11)


                #plt.ylabel('normalized')
                    if bool(experiments_want_to_plot_data_from_2):
                        #print('inside here')
                        #print(experiments_want_to_plot_data_from_2)
                   # plt.hist(Y_values_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        n,bins_test_3,patches = ax3.hist(new_Y_test_2,bins=bins ,align='mid',density=True,alpha=0.5,label='Extra Experiments')
                        
                        (mu, sigma) = norm.fit(new_Y_test_2)
                        yy = norm.pdf(bins_test_3,mu,sigma)
                        xx = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                        #l = ax3.plot(bins_test_3, yy, color='orange',linestyle='--', linewidth=2)
                        ax3.plot(xx, stats.norm.pdf(xx, mu, sigma),color='orange')
                        ax3.text(.037,8.5,r'$\mu=%.3f,\ \sigma=%.3f$' %(mu, sigma),color='orange', fontsize=11)
                    if bool(csv):
                        df = pd.read_csv(csv)
                        #ax3.hist(df[str(observables_unique[i])+'_Y'].dropna()*-1,bins=bins ,align='mid',density=True,alpha=0.5,label='Hong vs. Hong')                        
                
                
                

                    #plt.savefig(directory_to_save_images+'/'+str(observables_unique[i])+'_Including Experiments_'+ str(experiments_want_to_plot_data_from)+'_Yy_hist_2_normalized.pdf',dpi=1000,bbox_inches='tight')    
        
    
    
    
    
    def plotting_T_and_time_full_simulation_individual_observables_for_hong_data(self,MSI_instance_one,MSI_instance_two,experiments_want_to_plot_data_from,
                                                                             bins='auto',
                                                                             directory_to_save_images='',csv='',experiments_want_to_plot_data_from_2=[]):

        observables_total = []
        for i,exp in enumerate(self.exp_dict_list_original):
            observable_counter=0
            single_experiment = []
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                
                if observable == None:
                    continue
                
                if observable in exp['mole_fraction_observables']:
                    single_experiment.append(observable)
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:

                    single_experiment.append(observable)
                    
                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    single_experiment.append(wl)
                    
            observables_total.append(single_experiment)
        
        observables_flatten = [item for sublist in observables_total for item in sublist]
        from collections import OrderedDict
        observables_unique = list(OrderedDict.fromkeys(observables_flatten))
        
        empty_nested_observable_list_Y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_time = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_temperature = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_initial_temperature = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_z = [[] for x in range(len(observables_unique))]


        if bool(experiments_want_to_plot_data_from):
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from:
                        temp = MSI_instance_one.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = MSI_instance_one.y_matrix[start:stop,:]
                        empty_nested_observable_list_y[observables_unique.index(current_observable)].append(temp2)
                        
                        temp3 = MSI_instance_one.Z_matrix[start:stop,:]
                        empty_nested_observable_list_z[observables_unique.index(current_observable)].append(temp3)
                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]  


        for i,exp in enumerate(self.exp_dict_list_original):
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                if i in experiments_want_to_plot_data_from:
                    if observable in exp['mole_fraction_observables']:
                        empty_nested_observable_list_time[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
                        observable_counter+=1
                        
                    if observable in exp['concentration_observables']:
                        empty_nested_observable_list_time[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
    
    
                        observable_counter+=1
            if i in experiments_want_to_plot_data_from:        
                if 'perturbed_coef' in exp.keys():
                    wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                    for k,wl in enumerate(wavelengths):
                        empty_nested_observable_list_time[observables_unique.index(wl)].append(exp['absorbance_experimental_data'][k]['time']*1e3)
    
                        interploated_temp = np.interp(exp['absorbance_experimental_data'][k]['time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature[observables_unique.index(wl)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature[observables_unique.index(wl)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
####################################################################################################################################################################################################################    
        empty_nested_observable_list_Y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_y_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_time_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_temperature_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_initial_temperature_2 = [[] for x in range(len(observables_unique))]
        empty_nested_observable_list_z_2 = [[] for x in range(len(observables_unique))]


        if bool(experiments_want_to_plot_data_from_2):
            start = 0
            stop = 0 
            for x in range(len(self.simulation_lengths_of_experimental_data)):
                for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                    current_observable = observables_total[x][y]
                    stop = self.simulation_lengths_of_experimental_data[x][y] + start
                    if x in experiments_want_to_plot_data_from_2:
                        temp = MSI_instance_two.Y_matrix[start:stop,:]
                        empty_nested_observable_list_Y_2[observables_unique.index(current_observable)].append(temp)
                        
                        temp2 = MSI_instance_two.y_matrix[start:stop,:]
                        empty_nested_observable_list_y_2[observables_unique.index(current_observable)].append(temp2)
                        
                        temp3 = MSI_instance_two.Z_matrix[start:stop,:]
                        empty_nested_observable_list_z_2[observables_unique.index(current_observable)].append(temp3)
                        
                        
                        start = start + self.simulation_lengths_of_experimental_data[x][y]
                    else:
                        start = start + self.simulation_lengths_of_experimental_data[x][y]  


        for i,exp in enumerate(self.exp_dict_list_original):
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                if i in experiments_want_to_plot_data_from_2:
                    if observable in exp['mole_fraction_observables']:
                        empty_nested_observable_list_time_2[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature_2[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature_2[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
                        observable_counter+=1
                        
                    if observable in exp['concentration_observables']:
                        empty_nested_observable_list_time_2[observables_unique.index(observable)].append(exp['experimental_data'][observable_counter]['Time']*1e3)
                        interploated_temp = np.interp(exp['experimental_data'][observable_counter]['Time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature_2[observables_unique.index(observable)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature_2[observables_unique.index(observable)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])
    
    
                        observable_counter+=1
            if i in experiments_want_to_plot_data_from_2:        
                if 'perturbed_coef' in exp.keys():
                    wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                    for k,wl in enumerate(wavelengths):
                        empty_nested_observable_list_time_2[observables_unique.index(wl)].append(exp['absorbance_experimental_data'][k]['time']*1e3)
    
                        interploated_temp = np.interp(exp['absorbance_experimental_data'][k]['time'],exp['simulation'].timeHistories[0]['time'],exp['simulation'].timeHistories[0]['temperature'])
                        empty_nested_observable_list_temperature_2[observables_unique.index(wl)].append(interploated_temp)
                        empty_nested_observable_list_initial_temperature_2[observables_unique.index(wl)].append([self.exp_dict_list_original[i]['simulation'].temperature]*np.shape(interploated_temp)[0])    
###################################################################################################################################################################################################################  
        
        x = np.arange(10)
        ys = [i+x+(i*x)**2 for i in range(10)]
        colors=cm.rainbow(np.linspace(0,1,30))

        #colors = cm.rainbow(np.linspace(0, 1, len(ys)))
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(6,7))
        gs = gridspec.GridSpec(3, 1,height_ratios=[3,3,3],wspace=0.025,hspace=0.1)
        gs.update(wspace=0, hspace=0.7)
        ax1=plt.subplot(gs[0])
        ax2=plt.subplot(gs[1])
        ax3=plt.subplot(gs[2]) 
        
        fig2 = plt.figure(figsize=(6,7))
        gs2 = gridspec.GridSpec(3, 1,height_ratios=[3,3,3],wspace=0.025,hspace=0.1)
        gs2.update(wspace=0, hspace=0.7)
        ax4=plt.subplot(gs2[0])
        ax5=plt.subplot(gs2[1])
        ax6=plt.subplot(gs2[2]) 


        
        for x,observable in enumerate(empty_nested_observable_list_Y):
            length_of_2nd_list = len(empty_nested_observable_list_Y_2[x])
            if bool(observable):

                if x ==0:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax1.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_time'].dropna()*1e3,alpha=1,color='g',zorder=4,label='_nolegend_')      
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                            
                        
                            z_values = empty_nested_observable_list_z[x][y]
                            indecies = np.argwhere(z_values > 100)
                            new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                            new_y_test = np.delete(new_y_test,indecies)
                            new_time_test = copy.deepcopy(empty_nested_observable_list_time[x][y])
                            new_time_test = new_time_test.values
                            new_time_test = new_time_test.reshape((new_time_test.shape[0],1))
                            new_time_test  = np.delete(new_time_test,indecies)
                            ax1.scatter(new_y_test,new_time_test, c='#1f77b4',alpha=1)
                            ax1.set_xlabel('Relative Difference')
                            ax1.set_ylabel('Time (ms)')
                            
    
                            if y<length_of_2nd_list:
                                z_values_2 = empty_nested_observable_list_z_2[x][y]
                                indecies_2 = np.argwhere(z_values_2 > 100)
                                new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                                new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                                new_time_test_2 = copy.deepcopy(empty_nested_observable_list_time_2[x][y])
                                new_time_test_2 = new_time_test_2.values
                                new_time_test_2 = new_time_test_2.reshape((new_time_test_2.shape[0],1))
                                new_time_test_2  = np.delete(new_time_test_2,indecies_2)
                                #commented
                                ax1.scatter(new_y_test_2,new_time_test_2,color='orange',zorder=3,alpha=.25)
    
                            ax1.set_title(observables_unique[x])
                            ax1.set_xlim(left=-.25, right=.25, emit=True, auto=False)
                    ax1.scatter([],[],c='#1f77b4',label='#1')
                    ax1.scatter([],[],color='orange',label='#2')                                
                    #ax1.scatter([],[],color='green',label='#3')
                    ax1.legend(frameon=False)




                if x ==1:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax2.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_time'].dropna()*1e3,alpha=1,color='g',zorder=4)      
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                            
                        
                            z_values = empty_nested_observable_list_z[x][y]
                            indecies = np.argwhere(z_values > 100)
                            new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                            new_y_test = np.delete(new_y_test,indecies)
                            new_time_test = copy.deepcopy(empty_nested_observable_list_time[x][y])
                            new_time_test = new_time_test.values
                            new_time_test = new_time_test.reshape((new_time_test.shape[0],1))
                            new_time_test  = np.delete(new_time_test,indecies)
                            ax2.scatter(new_y_test,new_time_test, c='#1f77b4',alpha=1)
                            ax2.set_xlabel('Relative Difference')
                            ax2.set_ylabel('Time (ms)')
                            ax2.set_xlim(left=-.09, right=.09, emit=True, auto=False)
                            
    
                            if y<length_of_2nd_list:
                                z_values_2 = empty_nested_observable_list_z_2[x][y]
                                indecies_2 = np.argwhere(z_values_2 > 100)
                                new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                                new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                                new_time_test_2 = copy.deepcopy(empty_nested_observable_list_time_2[x][y])
                                new_time_test_2 = new_time_test_2.values
                                new_time_test_2 = new_time_test_2.reshape((new_time_test_2.shape[0],1))
                                new_time_test_2  = np.delete(new_time_test_2,indecies_2)
                                #commented
                                ax2.scatter(new_y_test_2,new_time_test_2,color='orange',zorder=3,alpha=.25)
    
                            #ax2.set_title(observables_unique[x])
                            ax2.set_title(r'H$_2$O')
                            
                            


                if x ==3:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax3.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_time'].dropna()*1e3,alpha=1,color='g',zorder=4,)      
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                            
                        
                            z_values = empty_nested_observable_list_z[x][y]
                            indecies = np.argwhere(z_values > 100)
                            new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                            new_y_test = np.delete(new_y_test,indecies)
                            new_time_test = copy.deepcopy(empty_nested_observable_list_time[x][y])
                            new_time_test = new_time_test.values
                            new_time_test = new_time_test.reshape((new_time_test.shape[0],1))
                            new_time_test  = np.delete(new_time_test,indecies)
                            ax3.scatter(new_y_test,new_time_test,c='#1f77b4',alpha=1)
                            ax3.set_xlabel('Relative Difference')
                            ax3.set_ylabel('Time (ms)')
                            ax3.set_xlim(left=-.3, right=.3, emit=True, auto=False)


                            
    
                            if y<length_of_2nd_list:
                                z_values_2 = empty_nested_observable_list_z_2[x][y]
                                indecies_2 = np.argwhere(z_values_2 > 100)
                                new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                                new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                                new_time_test_2 = copy.deepcopy(empty_nested_observable_list_time_2[x][y])
                                new_time_test_2 = new_time_test_2.values
                                new_time_test_2 = new_time_test_2.reshape((new_time_test_2.shape[0],1))
                                new_time_test_2  = np.delete(new_time_test_2,indecies_2)
                                #commented
                                ax3.scatter(new_y_test_2,new_time_test_2,color='orange',zorder=3,alpha=.25)
    
                            #ax3.set_title(observables_unique[x])
                            ax3.set_title('Absorbance '+ str(observables_unique[x])+str(' nm'))

                #fig.savefig(directory_to_save_images+'/'+'Three_pannel_plot_'+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_time.pdf',dpi=1000,bbox_inches='tight')

        for x,observable in enumerate(empty_nested_observable_list_Y):
            length_of_2nd_list = len(empty_nested_observable_list_Y_2[x])

            if bool(observable):
                
                if x==0:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax4.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_Temperature'].dropna(),label='_nolegend_',alpha=1,color='green',zorder=4)                    
                    
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                        
                        z_values = empty_nested_observable_list_z[x][y]
                        indecies = np.argwhere(z_values > 100)
                        new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                        new_y_test = np.delete(new_y_test,indecies)
                        new_temperature_test = copy.deepcopy(empty_nested_observable_list_temperature[x][y])
                        new_temperature_test = new_temperature_test.reshape((new_temperature_test.shape[0],1))
                        new_temperature_test  = np.delete(new_temperature_test,indecies)                    
                        
                        ax4.scatter(new_y_test,new_temperature_test,c='#1f77b4',alpha=1,label='_nolegend_')
                        #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                        ax4.set_xlabel('Relative Difference')
                        ax4.set_ylabel('Temperature (K)')
                        ax4.set_title(observables_unique[x])
                        ax4.set_xlim(left=-.25, right=.25, emit=True, auto=False)
                        
                        if y<length_of_2nd_list:
                            z_values_2 = empty_nested_observable_list_z_2[x][y]
                            indecies_2 = np.argwhere(z_values_2 > 100)
                            new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                            new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                            new_temperature_test_2 = copy.deepcopy(empty_nested_observable_list_temperature_2[x][y])
                            new_temperature_test_2 = new_temperature_test_2.reshape((new_temperature_test_2.shape[0],1))
                            new_temperature_test_2  = np.delete(new_temperature_test_2,indecies_2)                              
                            #commented
                            ax4.scatter(new_y_test_2,new_temperature_test_2,color='orange',zorder=3,alpha=.25,label='_nolegend_')
                        
                    ax4.scatter([],[],c='#1f77b4',label='#1')
                    ax4.scatter([],[],c='orange',label='#2')
                    #ax4.scatter([],[],c='green',label='#3')
                    ax4.legend(frameon=False)
   
                                    
                if x==1:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax5.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_Temperature'].dropna(),alpha=1,color='green',zorder=4)                    
                    
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                        
                        z_values = empty_nested_observable_list_z[x][y]
                        indecies = np.argwhere(z_values > 100)
                        new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                        new_y_test = np.delete(new_y_test,indecies)
                        new_temperature_test = copy.deepcopy(empty_nested_observable_list_temperature[x][y])
                        new_temperature_test = new_temperature_test.reshape((new_temperature_test.shape[0],1))
                        new_temperature_test  = np.delete(new_temperature_test,indecies)                    
                        
                        ax5.scatter(new_y_test,new_temperature_test,c='#1f77b4',alpha=1)
                        #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                        ax5.set_xlabel('Relative Difference')
                        ax5.set_ylabel('Temperature (K)')
                        #ax5.set_title(observables_unique[x])
                        ax5.set_title(r'H$_2$O')
                        ax5.set_xlim(left=-.09, right=.09, emit=True, auto=False)

                        
                        if y<length_of_2nd_list:
                            z_values_2 = empty_nested_observable_list_z_2[x][y]
                            indecies_2 = np.argwhere(z_values_2 > 100)
                            new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                            new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                            new_temperature_test_2 = copy.deepcopy(empty_nested_observable_list_temperature_2[x][y])
                            new_temperature_test_2 = new_temperature_test_2.reshape((new_temperature_test_2.shape[0],1))
                            new_temperature_test_2  = np.delete(new_temperature_test_2,indecies_2)         
                            #commented
                            ax5.scatter(new_y_test_2,new_temperature_test_2,color='orange',zorder=3,alpha=.15)

                if x==3:
                    if bool(csv):
                        df = pd.read_csv(csv)
                        ax6.scatter(df[str(observables_unique[x])+'_Y'].dropna()*-1,df[str(observables_unique[x])+'_Temperature'].dropna(),alpha=1,color='green',zorder=4)                    
                    
                    for y,array in enumerate(empty_nested_observable_list_Y[x]):
                        
                        z_values = empty_nested_observable_list_z[x][y]
                        indecies = np.argwhere(z_values > 100)
                        new_y_test = copy.deepcopy(empty_nested_observable_list_Y[x][y])
                        new_y_test = np.delete(new_y_test,indecies)
                        new_temperature_test = copy.deepcopy(empty_nested_observable_list_temperature[x][y])
                        new_temperature_test = new_temperature_test.reshape((new_temperature_test.shape[0],1))
                        new_temperature_test  = np.delete(new_temperature_test,indecies)                    
                        
                        ax6.scatter(new_y_test,new_temperature_test,label='Experiment_'+str(x)+'_observable_'+str(y),c='#1f77b4',alpha=1)
                        #plt.legend(ncol=2,bbox_to_anchor=(1, 0.5))
                        ax6.set_xlabel('Relative Difference')
                        ax6.set_ylabel('Temperature (K)')
                        ax6.set_title('Absorbance '+ str(observables_unique[x])+str(' nm'))
                        ax6.set_xlim(left=-.3, right=.3, emit=True, auto=False)

                        if y<length_of_2nd_list:
                            z_values_2 = empty_nested_observable_list_z_2[x][y]
                            indecies_2 = np.argwhere(z_values_2 > 100)
                            new_y_test_2 = copy.deepcopy(empty_nested_observable_list_Y_2[x][y])
                            new_y_test_2 = np.delete(new_y_test_2,indecies_2)
                            new_temperature_test_2 = copy.deepcopy(empty_nested_observable_list_temperature_2[x][y])
                            new_temperature_test_2 = new_temperature_test_2.reshape((new_temperature_test_2.shape[0],1))
                            new_temperature_test_2  = np.delete(new_temperature_test_2,indecies_2)      
                            #commented
                            ax6.scatter(new_y_test_2,new_temperature_test_2,color='orange',zorder=3,alpha=.25)
  
                #fig2.savefig(directory_to_save_images+'/'+'Three_pannel_plot'+'_Including Experiments_'+str(experiments_want_to_plot_data_from)+'_Yy_vs_initial_temperature.pdf',dpi=1000,bbox_inches='tight')

    def plotting_3_figre_observables(self,experiment_number_want_to_plot,sigmas_original=[],sigmas_optimized=[]):
        
        
        df = pd.read_csv('MSI/data/klip_optimization_comparison/graph_read_hong_data_1072.csv')
        df = pd.read_csv('MSI/data/klip_optimization_comparison/graph_read_hong_1283.csv')

        for i,exp in enumerate(self.exp_dict_list_optimized):
            if i==experiment_number_want_to_plot:
                observable_counter=0
                plt.figure(figsize=(10,10))
               # plt.figure()
                for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                    if observable == None:
                        continue

                        
                    if observable in exp['concentration_observables']:
                        if observable == 'H2O':
                            plt.subplot(3,1,1)
                            #plt.xlim(-.005,3.5)
                            #plt.ylim(1000,3500)
                            
                            plt.xlim(-.005,1)
                            plt.plot(df['hong_h2o_time'],df['hong_h2o'],'g:')
                    
                        if observable =='OH':
                            plt.subplot(3,1,2)
                            #plt.xlim(-.005,1)
                            #plt.ylim(0,25)
                           
                            plt.xlim(-.005,1)
                            plt.plot(df['hong_oh_time'],df['hong_oh'],'g:')
                            #plt.plot(df['hong_oh_time_2'],df['hong_oh_2'],'g:')
                        if observable == 'H2O2':
                            plt.subplot(3,1,1)
                            plt.xlim(-.005,.8)

                        plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable]*1e6,'b',label='MSI')
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable]*1e6,'r',label= "$\it{a}$ $\it{priori}$ model")
                        plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable+'_ppm'],'o',color='black',label='Hong et al. experiment') 
                       # plt.xlabel('Time (ms)')
                        if observable == 'H2O':
                            plt.ylabel(r'H$_2$O [ppm]', fontsize=18)
                        if observable == 'OH':
                            plt.ylabel('OH [ppm]',fontsize=18)
                        if observable == 'H2O2':
                            plt.ylabel(r'H$_2$O$_2$ [ppm]', fontsize=18)
                            plt.xlabel('Time [ms]',fontsize=18)

                        else:                        
                            plt.ylabel(str(observable) +' '+ '[ppm]')
                        #plt.title('Experiment_'+str(i+1))
                        
                        if bool(sigmas_optimized)==True:
                            high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                            high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                            low_error_optimized = np.exp(np.array(sigmas_optimized[i][observable_counter])*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                            
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_optimized,'b--')
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_optimized,'b--')                    
                            
        
        
                            high_error_original = np.exp(sigmas_original[i][observable_counter])
                            high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                            low_error_original = np.exp(np.array(sigmas_original[i][observable_counter])*-1)
                            low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                            

                        key_list = []
                        for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():
                            
                            #plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                            key_list.append(key)
                       
                        #plt.legend(handlelength=3)
                        #plt.legend(ncol=1)
                        sp = '_'.join(key_list)
                        #print(sp)
                        #plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K'+'_'+str(self.exp_dict_list_original[i]['simulation'].pressure)+'_'+sp+'_'+'.pdf', bbox_inches='tight')
                        
                        #stub
                        plt.tick_params(direction='in')
                        
                        
                        #plt.savefig('Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        #plt.savefig('Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)
                        #plt.savefig(self.out_path+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=1000)
    
    

                        observable_counter+=1
                        
    
                if 'perturbed_coef' in exp.keys():
                    wavelengths = self.parsed_yaml_list_optimized[i]['absorbanceCsvWavelengths']
                    plt.subplot(3,1,3)
                    #plt.xlim(-.005,3)
                    
                    plt.xlim(-.005,.2)
                    #plt.xlim(-.005,.6)
                    #plt.xlim(-.005,1.5)

                    plt.plot(df['hong_abs_time'],df['hong_abs'],'g:',zorder=10,label='Hong et al. model')
                    


                    for k,wl in enumerate(wavelengths):
                        if wl == 227:
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Hong et al. experiment')
                        if wl == 215:
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Kappel et al. experiment')
    
                        plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['absorbance_calculated_from_model'][wl],'b',label='MSI')
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['absorbance_calculated_from_model'][wl],'r',label= "$\it{a}$ $\it{priori}$ model")
                        #plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Experimental Data')
                        plt.xlabel('Time [ms]',fontsize=18)
                        #plt.ylabel('Absorbance'+' '+str(wl)+' nm')
                        plt.ylabel('Absorbance',fontsize=18)
                        #plt.title('Experiment_'+str(i+1))
                        
                        if bool(sigmas_optimized)==True:
                            high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])
                            high_error_optimized = np.multiply(high_error_optimized,exp['absorbance_model_data'][wl])
                            low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp['absorbance_model_data'][wl])
                            
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,high_error_optimized,'b--')
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,low_error_optimized,'b--')
                            
                            high_error_original = np.exp(sigmas_original[i][observable_counter])
                            high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['absorbance_model_data'][wl])
                            low_error_original =  np.exp(sigmas_original[i][observable_counter]*-1)
                            low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['absorbance_model_data'][wl])
                            
                            
                            #plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,high_error_original,'r--')
                            #plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,low_error_original,'r--')
    
    #                    if bool(sigmas_optimized)==True and  i+1 == 11:    
    #                        plt.ylim(top=.35)
                        
                        #start here
                        
                        #plt.plot([],'w' ,label= 'T:'+ str(self.exp_dict_list_original[i]['simulation'].temperature))
                        #plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
                        #for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():                        
                            #plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                            
    
                        #plt.legend(handlelength=3)
                        plt.legend(ncol=2, prop={'size': 12})
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(experiment_number_want_to_plot)+'_three_figure_obervables.pdf', bbox_inches='tight',dpi=1000)
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(experiment_number_want_to_plot)+'_three_figure_obervables.png', bbox_inches='tight',dpi=1000)
                        plt.savefig(self.out_path+'/'+'Experiment_'+str(experiment_number_want_to_plot)+'_three_figure_obervables.svg', bbox_inches='tight',dpi=1000,transparent=True)
                        plt.tick_params(direction='in')
   

                        #plt.savefig('Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        #plt.savefig('Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)
                        
                        
                        
                        
    def plotting_rate_constants_combined_channels(self,
                                initial_temperature=300,
                                final_temperature=3000,
                                temperature_range_to_plot_over=None):
        
        print('\n')
        print('--------------------------------------------------------------------------')
        print('Rate Constant Plots')
        print('--------------------------------------------------------------------------')         

        gas_optimized = ct.Solution(self.new_cti)
        gas_original = ct.Solution(self.nominal_cti)

        

        def unique_list(seq):
            checked = []
            for e in seq:
                if e not in checked:
                    checked.append(e)
            
            for j,item in enumerate(checked):
                if type(item)==tuple:
                    sorted_tuple = tuple(sorted(item))
                    checked[j] = sorted_tuple
                    
            return checked
        
        
        def filter_range_for_plotting(Temp_optimized,k_optimized,Temp_original,k_original,
                              Temp_high_error_optimized, k_high_error_optimized,
                              Temp_low_error_optimized, k_low_error_optimized,
                              Temp_low_error_original, k_low_error_original,
                              Temp_high_error_original, k_high_error_original,
                              low_temp=900,high_temp=1500):
            def result_high_and_low(arr_temp,arr_k):
                arr_temp = np.array(arr_temp)
                arr_k = np.array(arr_k)
                result_high = np.where(arr_temp == high_temp)[0][0]
                result_low = np.where(arr_temp == low_temp)[0][0]
                
                arr_k_filtered = arr_k[result_low:result_high+1]
                max_k_in_range = np.max(arr_k_filtered)
                min_k_in_range = np.min(arr_k_filtered)
                
                return [min_k_in_range,max_k_in_range]
            
            optimized_tuple = result_high_and_low(Temp_optimized,k_optimized)
            optimized_high_error_bar_tuple = result_high_and_low(Temp_high_error_optimized, k_high_error_optimized)
            optimized_low_error_bar_tuple = result_high_and_low(Temp_low_error_optimized, k_low_error_optimized)
            
            original_tuple = result_high_and_low(Temp_optimized,k_optimized)
            original_low_error_bar_tuple=result_high_and_low(Temp_low_error_original, k_low_error_original)
            original_high_error_bar_tuple = result_high_and_low(Temp_high_error_original, k_high_error_original)
            
            final_list = optimized_tuple+optimized_high_error_bar_tuple+optimized_low_error_bar_tuple+original_tuple+original_low_error_bar_tuple+original_high_error_bar_tuple
            
            final_array = np.array(final_list)
            max_value = np.max(final_array)
            min_value = np.min(final_array)
            
            return min_value,max_value
        
        

        def target_values_for_S(rate_constant_plots_df,
                                exp_dict_list,
                                S_matrix,
                                master_equation_reaction_list = [],
                                master_equation_sensitivites = {}):
                    
                    
                
                # target_value_csv = pd.read_csv(target_value_csv)
                target_reactions = rate_constant_plots_df['Reaction']
                target_temp = rate_constant_plots_df['temperature']
                target_press = rate_constant_plots_df['pressure']
                # target_k = rate_constant_plots_df['k']
                bath_gas = rate_constant_plots_df['M']
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
                        
                    # if bath_gas !=0:
                    #     gas.TPX = temperature,pressure*101325,{'H2O':.013,'O2':.0099,'H':.0000007,'Ar':.9770993}   
                    # else:
                    #     gas.TPX = temperature,pressure*101325,{'Ar':.99}
                    if bath_gas ==1:
                        gas.TPX = temperature,pressure*101325,{'Ar':1}   
                    if bath_gas ==2:
                        gas.TPX = temperature,pressure*101325,{'He':1}   
                    else:
                        gas.TPX = temperature,pressure*101325,{'Ar':.99}                    
                    
                    total_k = []    
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
                        total_k.append(k)
                        
                                
                                    #check and make sure we are subtracting in the correct order 
                    k_summation=sum(total_k)    
                    
                    weighting_factor_dict = {}
                    for reaction in rate_constant_list:
                        weighting_factor_dict[reaction] = original_rc_dict[reaction] / k_summation
                        
                    return weighting_factor_dict
                
                def calculate_weighting_factor_summation_with_denominator(numerator_rate_constant_list,denominator_rate_constant_list,gas,temperature,Press,bath_gas):
                    if Press == 0:
                        pressure = 1e-9
                    else:
                        pressure = Press
                        
                        
                    if bath_gas ==1:
                        gas.TPX = temperature,pressure*101325,{'Ar':1}   
                    if bath_gas ==2:
                        gas.TPX = temperature,pressure*101325,{'He':1}                           
                    else:
                        gas.TPX = temperature,pressure*101325,{'Ar':.99}
                                                
                    # if bath_gas !=0:
                    #     gas.TPX = temperature,pressure*101325,{'H2O':.013,'O2':.0099,'H':.0000007,'Ar':.9770993}   
                    # else:
                    #     gas.TPX = temperature,pressure*101325,{'Ar':.99}
                    
                    total_k_numerator = []    
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
                        total_k_numerator.append(k)
                        
                                
                                    #check and make sure we are subtracting in the correct order 
                    k_summation_numerator=sum(total_k_numerator)    
                    
                    weighting_factor_dict_numerator = {}
                    for reaction in numerator_rate_constant_list:
                        weighting_factor_dict_numerator[reaction] = original_rc_dict[reaction] / k_summation_numerator
    
                    total_k_denominator = []    
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
                        total_k_denominator.append(k)
                        
                                
                                    #check and make sure we are subtracting in the correct order 
                    k_summation_denominator=sum(total_k_denominator)    
                    
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
                        for i,MP in enumerate(master_equation_sensitivites[reaction]):
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
                    if '[/]' in reaction:
                        #check numerator and denominator
                        reactants_in_numerator = reaction.split('[/]')[0].rstrip()
                        reactants_in_numerator = reactants_in_numerator.lstrip()
                        
                        reactants_in_denominator = reaction.split('[/]')[1].rstrip()
                        reactants_in_denominator = reactants_in_denominator.lstrip()
    
                        if '[*]' in reactants_in_numerator and '[+]' not in reactants_in_numerator:
    
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
                            
                            
                            
                            
                            
    
    
    
                        elif '[+]' in reactants_in_numerator and '[*]' not in reactants_in_numerator:
                            list_of_reactions_in_numerator = reactants_in_numerator.split('[+]')
                            list_of_reactions_in_numerator_cleaned=[]
                            for reaction in list_of_reactions_in_numerator:
                                reaction = reaction.rstrip()
                                reaction = reaction.lstrip()
                                list_of_reactions_in_numerator_cleaned.append(reaction)
                                    
                            reactions_in_numerator_with_these_reactants  =    list_of_reactions_in_numerator_cleaned 
    
    
    
    
                        elif '[+]' in reactants_in_numerator and '[*]' in reactants_in_numerator:
                            print('need to make rule')
                        else:
                            reactions_in_numerator_with_these_reactants = []
                            reactions_in_numerator_with_these_reactants.append(reactants_in_numerator)
    
                        
    
    
    
    
                        #check reactants in numerator 
                        if '[*]' in reactants_in_denominator and '[+]' not in reactants_in_denominator:
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
    
    
    
    
                        elif '[+]' in reactants_in_denominator and '[*]' not in reactants_in_denominator:
                            list_of_reactions_in_denominator = reactants_in_numerator.split('[+]')
                            list_of_reactions_in_denominator_cleaned=[]
                            for reaction in list_of_reactions_in_denominator:
                                reaction = reaction.rstrip()
                                reaction = reaction.lstrip()
                                list_of_reactions_in_denominator_cleaned.append(reaction)
                                    
                            reactions_in_denominator_with_these_reactants  =    list_of_reactions_in_denominator_cleaned 
                        
    
                        elif '[+]' in reactants_in_denominator and '[*]' in reactants_in_denominator:
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
    
    
                    else:

                        if '[+]' in reaction:
                            list_of_reactions = reaction.split('[+]')
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
        
        
        
                        elif '[*]' in reaction:
        
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
    
    
    
                Trigger = False
                MP_stack = []
                target_values_to_stack =  []
                for i,reaction in enumerate(target_reactions):
                    type_of_reaction, reaction_tuple = check_if_reaction_is_theory_or_not(reaction)
    
                    # print(reaction_tuple)
                    if type_of_reaction== 'master_equations_only':
                        
                        if len(reaction_tuple)==1:
                            if len(reaction_tuple[0])==1:
                                
                                
    
                                nested_reaction_list = create_empty_nested_reaction_list()
                                for j, MP_array in enumerate(master_equation_sensitivites[reaction]):
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
                                    #if reaction =='2 HO2 <=> H2O2 + O2X' and j==0 and target_temp[i]==250 :
                                        #print(sum(temp),'sum temp')
                                        
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
                                weighting_factor_dictonary = calculate_weighting_factor_summation(reactions_in_cti_file_with_these_reactants,
                                                                                              gas,
                                                                                              target_temp[i],
                                                                                              target_press[i],
                                                                                              bath_gas[i])
                                nested_reaction_list = create_empty_nested_reaction_list()
                            
                                for secondary_reaction in reactions_in_cti_file_with_these_reactants:
                                    for j, MP_array in enumerate(master_equation_sensitivites[secondary_reaction]):
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
                                        
                                    sub_array_to_apply_weighting_factor_to = list(np.array(nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)])*weighting_factor_dictonary[secondary_reaction])
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
    
                            weighting_factor_dictonary = calculate_weighting_factor_summation_with_denominator(reactions_in_cti_file_with_these_reactants_numerator,
                                                                                              reactions_in_cti_file_with_these_reactants_denominator,                 
                                                                                              gas,
                                                                                              target_temp[i],
                                                                                              target_press[i],
                                                                                              bath_gas[i])
    
                            #now need to add to S matrix 
                            for secondary_reaction in (reactions_in_cti_file_with_these_reactants_numerator+reactions_in_cti_file_with_these_reactants_denominator):
                                for j, MP_array in enumerate(master_equation_sensitivites[secondary_reaction]):
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
                                    
                                sub_array_to_apply_weighting_factor_to = list(np.array(nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)])*weighting_factor_dictonary[secondary_reaction])
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
    
    
                                weighting_factor_dictonary = calculate_weighting_factor_summation(reactions_in_cti_file_with_these_reactants,
                                                                                                  gas,
                                                                                                  target_temp[i],
                                                                                                  target_press[i],
                                                                                                  bath_gas[i])
                                
                                A_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))        
                                N_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
                                Ea_temp = np.zeros((1,number_of_reactions_in_cti-len(flattened_master_equation_reaction_list)))
                                
                                for secondary_reaction in reactions_in_cti_file_with_these_reactants:
                                    #need to multiply by the weighting factor for the reaction
                                    
                                    A_temp[0,reactions_in_cti_file.index(secondary_reaction)] = 1 * weighting_factor_dictonary[secondary_reaction]
                                    N_temp [0,reactions_in_cti_file.index(secondary_reaction)] = np.log(target_temp[i]) * weighting_factor_dictonary[secondary_reaction]
                                    Ea_temp[0,reactions_in_cti_file.index(secondary_reaction)] = (-1/target_temp[i]) * weighting_factor_dictonary[secondary_reaction]
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
                            weighting_factor_dictonary = calculate_weighting_factor_summation_with_denominator(reactions_in_cti_file_with_these_reactants_numerator,
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
                                    A_temp[0,reactions_in_cti_file.index(secondary_reaction)] = 1 * weighting_factor_dictonary[secondary_reaction]
                                    N_temp [0,reactions_in_cti_file.index(secondary_reaction)] = np.log(target_temp[i]) * weighting_factor_dictonary[secondary_reaction]
                                    Ea_temp[0,reactions_in_cti_file.index(secondary_reaction)] = (-1/target_temp[i]) * weighting_factor_dictonary[secondary_reaction]
                                    
                            
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
                            weighting_factor_dictonary = calculate_weighting_factor_summation(reactions_in_cti_file_with_these_reactants,
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
                                    A_temp[0,reactions_in_cti_file.index(secondary_reaction)] = 1 * weighting_factor_dictonary[secondary_reaction]
                                    N_temp [0,reactions_in_cti_file.index(secondary_reaction)] = np.log(target_temp[i]) * weighting_factor_dictonary[secondary_reaction]
                                    Ea_temp[0,reactions_in_cti_file.index(secondary_reaction)] = (-1/target_temp[i]) * weighting_factor_dictonary[secondary_reaction]
                      
                                
                                elif secondary_reaction in flattened_master_equation_reaction_list:
                                    for j, MP_array in enumerate(master_equation_sensitivites[secondary_reaction]):
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
                                        
                                    sub_array_to_apply_weighting_factor_to = list(np.array(nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)])*weighting_factor_dictonary[secondary_reaction])
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
    
                            weighting_factor_dictonary = calculate_weighting_factor_summation_with_denominator(reactions_in_cti_file_with_these_reactants_numerator,
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
                                    A_temp[0,reactions_in_cti_file.index(secondary_reaction)] = 1 * weighting_factor_dictonary[secondary_reaction]
                                    N_temp [0,reactions_in_cti_file.index(secondary_reaction)] = np.log(target_temp[i]) * weighting_factor_dictonary[secondary_reaction]
                                    Ea_temp[0,reactions_in_cti_file.index(secondary_reaction)] = (-1/target_temp[i]) * weighting_factor_dictonary[secondary_reaction]
                      
                                
                                elif secondary_reaction in flattened_master_equation_reaction_list:
                                    for j, MP_array in enumerate(master_equation_sensitivites[secondary_reaction]):
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
                                        
                                    sub_array_to_apply_weighting_factor_to = list(np.array(nested_reaction_list[flattened_master_equation_reaction_list.index(secondary_reaction)])*weighting_factor_dictonary[secondary_reaction])
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






            
        def sort_rate_constant_target_values(parsed_csv,unique_reactions,gas):
            reaction_list_from_mechanism = gas.reaction_equations()
            target_value_ks = [[] for reaction in range(len(unique_reactions))]
            target_value_temps = [[] for reaction in range(len(unique_reactions))]
            # if 'ref' in list(parsed_csv.columns):
            target_value_refs = [[] for reaction in range(len(unique_reactions))]
            
            condition_list = []
            
            
            for i,reaction in enumerate(parsed_csv['Reaction']):

                if '[/]' in reaction:
                    numerator = reaction.split('[/]')[0].rstrip().lstrip()
                    denominator = reaction.split('[/]')[1].rstrip().lstrip()
                    idx = []
                    for frac_reaction in [numerator, denominator]:
                        if '[*]' in frac_reaction:
                            jdx = []
                            reactions_in_cti_file_with_these_reactants = []
                            reaction_number_in_cti_file_with_these_reactants = []  
                            reactions_in_cti_file_with_these_products = []
                            reaction_number_in_cti_file_with_these_products = []   
                            reactants_in_target_reactions = frac_reaction.split('<=>')[0].rstrip()
                            reverse_reactants_in_target_reaction=None
                            if len(reactants_in_target_reactions.split('+'))>1:
                                reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                                temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                                temp = temp.lstrip()
                                temp = temp.rstrip()
                                reverse_reactants_in_target_reaction = temp
                            if reverse_reactants_in_target_reaction !=None:
                                for reaction_number_in_cti_file in range(gas.n_reactions):
                                    if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                                            
                                            reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)  
                                    elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                                            
                                            reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)
                            elif reverse_reactants_in_target_reaction ==None:
                                for reaction_number_in_cti_file in range(gas.n_reactions):
                                    if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'): 
                                            reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                    elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'): 
                                            reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file) 
                            jdx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                            jdx.append(tuple(sorted(reaction_number_in_cti_file_with_these_products)))
                            idx.append(jdx)
                        elif '[+]' in frac_reaction:
                            jdx = []
                            reactions_in_cti_file_with_these_reactants = []
                            reaction_number_in_cti_file_with_these_reactants = []    
                            list_of_reactions = frac_reaction.split('[+]')
                            list_of_reactions_cleaned=[]
                            for reac in list_of_reactions:
                                reac = reac.rstrip()
                                reac = reac.lstrip()
                                list_of_reactions_cleaned.append(reac)
                            for reac in list_of_reactions_cleaned:
                                reaction_number_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism.index(reac))
                                reactions_in_cti_file_with_these_reactants.append(reac)
                            if frac_reaction == denominator:
                                jdx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                                idx.append(jdx)
                            else:
                                idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                        else:
                            idx.append(reaction_list_from_mechanism.index(frac_reaction))              

                else:
                    if '[*]' in reaction:
                        idx = []
                        reactions_in_cti_file_with_these_reactants = []
                        reaction_number_in_cti_file_with_these_reactants = [] 
                        reactions_in_cti_file_with_these_products = []
                        reaction_number_in_cti_file_with_these_products = []    
                        reactants_in_target_reactions = reaction.split('<=>')[0].rstrip()
                        reverse_reactants_in_target_reaction=None
                        if len(reactants_in_target_reactions.split('+'))>1:
                            reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                            temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                            temp = temp.lstrip()
                            temp = temp.rstrip()
                            reverse_reactants_in_target_reaction = temp
                        if reverse_reactants_in_target_reaction !=None:
                            for reaction_number_in_cti_file in range(gas.n_reactions):
                                if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                                            
                                        reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):
                                        reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)
                        elif reverse_reactants_in_target_reaction ==None:
                            for reaction_number_in_cti_file in range(gas.n_reactions):
                                if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'): 
                                        reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file) 
                                elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or  
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):
                                        reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)                            
                        idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                        idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_products)))
                    elif '[+]' in reaction:
                        reactions_in_cti_file_with_these_reactants = []
                        reaction_number_in_cti_file_with_these_reactants = []
                        list_of_reactions = reaction.split('[+]')
                        list_of_reactions_cleaned=[]
                        for reac in list_of_reactions:
                            reac = reac.rstrip()
                            reac = reac.lstrip()
                            list_of_reactions_cleaned.append(reac)
                        for reac in list_of_reactions_cleaned:
                            reaction_number_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism.index(reac))
                            reactions_in_cti_file_with_these_reactants.append(reac)
                        idx = tuple(sorted(reaction_number_in_cti_file_with_these_reactants))
                    else:
                        idx = reaction_list_from_mechanism.index(reaction)
#Obama
                target_value_ks[unique_reactions.index(idx)].append(parsed_csv['k'][i])
                target_value_temps[unique_reactions.index(idx)].append(parsed_csv['temperature'][i])
                        
                if 'ref' in list(parsed_csv.columns): 
                    target_value_refs[unique_reactions.index(idx)].append(parsed_csv['ref'][i])
                else: 
                    target_value_refs[unique_reactions.index(idx)].append(np.nan)       
       

            return target_value_temps, target_value_ks, target_value_refs
        
        def rate_constant_over_temperature_range_from_cantera(reaction_number,
                                                              gas,
                                                              initial_temperature=300,
                                                              final_temperature=3000,
                                                              pressure=1,
                                                              conditions = {'Ar':1}):
            Temp = []
            k = []
            if type(reaction_number) == list: #Total Rates & Ratios
                if type(reaction_number[1]) == tuple: #Total Rates
                    k_total = []
                    for i,sub_number in enumerate(reaction_number[0]):
                        k_temp=[]
                        for temperature in np.arange(initial_temperature,final_temperature+1,1):
                            gas.TPX = temperature,pressure*101325,conditions
                            coeff_sum = sum(gas.reaction(sub_number).reactants.values())
                            rc = gas.forward_rate_constants[sub_number]
                            if coeff_sum==1:
                                rc = rc
                            elif coeff_sum==2:
                                rc = rc*1000
                            elif coeff_sum==3:
                                rc = rc*1000000
                            k_temp.append(rc)
                            if i == 0:
                               Temp.append(temperature)
                        k_temp = np.array(k_temp)
                        k_total.append(k_temp)
                    for i,sub_number in enumerate(reaction_number[1]):
                        k_temp=[]
                        for temperature in np.arange(initial_temperature,final_temperature+1,1):
                            gas.TPX = temperature,pressure*101325,conditions
                            coeff_sum = sum(gas.reaction(sub_number).products.values())
                            rc = gas.reverse_rate_constants[sub_number]
                            if coeff_sum==1:
                                rc = rc
                            elif coeff_sum==2:
                                rc = rc*1000
                            elif coeff_sum==3:
                                rc = rc*1000000
                            k_temp.append(rc)
                        k_temp = np.array(k_temp)
                        k_total.append(k_temp)                    
                    k = list(sum(k_total))
                
                elif type(reaction_number[1]) == list: #Ratio over a sum
                    
                    if len(reaction_number[1]) == 1: #Ratio over partial sum
                        
                        if type(reaction_number[0]) == tuple: #Partial sum over partial sum
                            k_num_total = []
                            for i,sub_number in enumerate(reaction_number[0]):
                                k_num_temp=[]
                                for temperature in np.arange(initial_temperature,final_temperature+1,1):
                                    gas.TPX = temperature,pressure*101325,conditions
                                    coeff_sum = sum(gas.reaction(sub_number).reactants.values())
                                    rc = gas.forward_rate_constants[sub_number]
                                    if coeff_sum==1:
                                        rc = rc
                                    elif coeff_sum==2:
                                        rc = rc*1000
                                    elif coeff_sum==3:
                                        rc = rc*1000000
                                    k_num_temp.append(rc)
                                    if i ==0:
                                        Temp.append(temperature)
                                k_num_temp = np.array(k_num_temp)
                                k_num_total.append(k_num_temp)
                            k_num = list(sum(k_num_total))
                        
                        else:   #Single rate over partial sum
                            k_num = []
                            for temperature in np.arange(initial_temperature,final_temperature+1,1):
                                Temp.append(temperature)
                                gas.TPX = temperature,pressure*101325,conditions
                                k_num.append(gas.forward_rate_constants[reaction_number[0]])
                        
                        k_den_total = []
                        for i,sub_number in enumerate(reaction_number[1][0]):
                            k_den_temp=[]
                            for temperature in np.arange(initial_temperature,final_temperature+1,1):
                                gas.TPX = temperature,pressure*101325,conditions
                                coeff_sum = sum(gas.reaction(sub_number).reactants.values())
                                rc = gas.forward_rate_constants[sub_number]
                                if coeff_sum==1:
                                    rc = rc
                                elif coeff_sum==2:
                                    rc = rc*1000
                                elif coeff_sum==3:
                                    rc = rc*1000000
                                k_den_temp.append(rc)
                            k_den_temp = np.array(k_den_temp)
                            k_den_total.append(k_den_temp)
                        k_den = list(sum(k_den_total))

                    else:   #Ratio over a total sum
                        if type(reaction_number[0]) == tuple: #Partial sum over total sum
                            k_num_total = []
                            for i,sub_number in enumerate(reaction_number[0]):
                                k_num_temp=[]
                                for temperature in np.arange(initial_temperature,final_temperature+1,1):
                                    gas.TPX = temperature,pressure*101325,conditions
                                    coeff_sum = sum(gas.reaction(sub_number).reactants.values())
                                    rc = gas.forward_rate_constants[sub_number]
                                    if coeff_sum==1:
                                        rc = rc
                                    elif coeff_sum==2:
                                        rc = rc*1000
                                    elif coeff_sum==3:
                                        rc = rc*1000000
                                    k_num_temp.append(rc)
                                    if i ==0:
                                        Temp.append(temperature)
                                k_num_temp = np.array(k_num_temp)
                                k_num_total.append(k_num_temp)
                            k_num = list(sum(k_num_total))
                        
                        else:   #Single rate over total sum
                            k_num = []
                            for temperature in np.arange(initial_temperature,final_temperature+1,1):
                                Temp.append(temperature)
                                gas.TPX = temperature,pressure*101325,conditions
                                k_num.append(gas.forward_rate_constants[reaction_number[0]])
                        
                        k_den_total = []
                        for i, sub_number in enumerate(reaction_number[1][0]):
                            k_den_temp=[]
                            for temperature in np.arange(initial_temperature,final_temperature+1,1):
                                gas.TPX = temperature,pressure*101325,conditions
                                coeff_sum = sum(gas.reaction(sub_number).reactants.values())
                                rc = gas.forward_rate_constants[sub_number]
                                if coeff_sum==1:
                                    rc = rc
                                elif coeff_sum==2:
                                    rc = rc*1000
                                elif coeff_sum==3:
                                    rc = rc*1000000
                                k_den_temp.append(rc)
                            k_den_temp = np.array(k_den_temp)
                            k_den_total.append(k_den_temp)
                        for i, sub_number in enumerate(reaction_number[1][1]):
                            k_den_temp=[]
                            for temperature in np.arange(initial_temperature,final_temperature+1,1):
                                gas.TPX = temperature,pressure*101325,conditions
                                coeff_sum = sum(gas.reaction(sub_number).products.values())
                                rc = gas.reverse_rate_constants[sub_number]
                                if coeff_sum==1:
                                    rc = rc
                                elif coeff_sum==2:
                                    rc = rc*1000
                                elif coeff_sum==3:
                                    rc = rc*1000000
                                k_den_temp.append(rc)
                            k_den_temp = np.array(k_den_temp)
                            k_den_total.append(k_den_temp)
                        k_den = list(sum(k_den_total))
                    
                    k = np.divide(k_num,k_den)

                else:   #Ratio over a single rate
                    
                    if type(reaction_number[0]) == tuple: #Partial sum over single rate
                        k_num_total = []
                        for i,sub_number in enumerate(reaction_number[0]):
                            k_num_temp=[]
                            for temperature in np.arange(initial_temperature,final_temperature+1,1):
                                gas.TPX = temperature,pressure*101325,conditions
                                coeff_sum = sum(gas.reaction(sub_number).reactants.values())
                                rc = gas.forward_rate_constants[sub_number]
                                if coeff_sum==1:
                                    rc = rc
                                elif coeff_sum==2:
                                    rc = rc*1000
                                elif coeff_sum==3:
                                    rc = rc*1000000
                                k_num_temp.append(rc)
                                if i ==0:
                                    Temp.append(temperature)
                            k_num_temp = np.array(k_num_temp)
                            k_num_total.append(k_num_temp)
                        k_num = list(sum(k_num_total))
                    
                    else:   #Single rate over single rate 
                        k_num = []
                        for temperature in np.arange(initial_temperature,final_temperature+1,1):
                            Temp.append(temperature)
                            gas.TPX = temperature,pressure*101325,conditions
                            k_num.append(gas.forward_rate_constants[reaction_number[0]])
                    
                    k_den = []
                    for temperature in np.arange(initial_temperature,final_temperature+1,1):
                        gas.TPX = temperature,pressure*101325,conditions
                        k_den.append(gas.forward_rate_constants[reaction_number[1]])    
                    
                    k = np.divide(k_num,k_den)    
            
            elif type(reaction_number) == tuple: #Partial Sums
                
                k_total=[]
                for i,sub_number in enumerate(reaction_number):
                    k_temp=[]
                    for temperature in np.arange(initial_temperature,final_temperature+1,1):
                        gas.TPX = temperature,pressure*101325,conditions
   
                        coeff_sum = sum(gas.reaction(sub_number).reactants.values())
    
                        rc = gas.forward_rate_constants[sub_number]
                        if coeff_sum==1:
                            rc = rc
                        elif coeff_sum==2:
                            rc = rc*1000
                        elif coeff_sum==3:
                            rc = rc*1000000
         
                        k_temp.append(rc)
                        
                        if i ==0:
                            Temp.append(temperature)
                    k_temp = np.array(k_temp)
                    k_total.append(k_temp)                
                k = list(sum(k_total))
            else:   #Single Rate
                for temperature in np.arange(initial_temperature,final_temperature+1,1):
                    gas.TPX = temperature,pressure*101325,conditions
                    Temp.append(temperature)
                    coeff_sum = sum(gas.reaction(reaction_number).reactants.values())
                    rc = gas.forward_rate_constants[reaction_number]
                    if coeff_sum==1:
                        rc=rc
                    elif coeff_sum==2:
                        rc = rc*1000
                    elif coeff_sum==3:
                        rc = rc*1000000
                    
                    k.append(rc)
                    
            return Temp,k

        
        def calculate_sigmas_for_rate_constants(k_target_value_S_matrix,k_target_values_parsed_csv,unique_reactions,gas,covariance):

            
            reaction_list_from_mechanism = gas.reaction_equations()
            sigma_list_for_target_ks = [[] for reaction in range(len(unique_reactions))]
            shape = k_target_value_S_matrix.shape
            
            
            for row in range(shape[0]):
                #print(row)
                SC = np.dot(k_target_value_S_matrix[row,:],covariance)
                sigma_k = np.dot(SC,np.transpose(k_target_value_S_matrix[row,:]))
                sigma_k = np.sqrt(sigma_k)
                #print(row)
                #print(k_target_values_parsed_csv['Reaction'][row])
                if '[/]' in k_target_values_parsed_csv['Reaction'][row]:

                    numerator = k_target_values_parsed_csv['Reaction'][row].split('[/]')[0].rstrip().lstrip()
                    denominator = k_target_values_parsed_csv['Reaction'][row].split('[/]')[1].rstrip().lstrip()
                    indx = []
                    for frac_reaction in [numerator, denominator]:
                        if '[*]' in frac_reaction:
                            idx = []
                            reactions_in_cti_file_with_these_reactants = []
                            reaction_number_in_cti_file_with_these_reactants = []
                            reactions_in_cti_file_with_these_products = []
                            reaction_number_in_cti_file_with_these_products = []
                            reactants_in_target_reactions = frac_reaction.split('<=>')[0].rstrip()
                            reverse_reactants_in_target_reaction=None
                            if len(reactants_in_target_reactions.split('+'))>1:
                                reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                                temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                                temp = temp.lstrip()
                                temp = temp.rstrip()
                                reverse_reactants_in_target_reaction = temp
                            if  reverse_reactants_in_target_reaction != None:       
                                for reaction_number_in_cti_file in range(gas.n_reactions):
                                    if  (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                                     
                                            reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                    elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                                     
                                            reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)         
                            elif reverse_reactants_in_target_reaction == None:
                                for reaction_number_in_cti_file in range(gas.n_reactions):
                                    if  (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'): 
                                            reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)    
                                    elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'): 
                                            reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)
                            idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))                
                            idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_products)))
                            indx.append(idx)
                        elif '[+]' in frac_reaction:
                            idx = []
                            reactions_in_cti_file_with_these_reactants = []
                            reaction_number_in_cti_file_with_these_reactants = []
                            list_of_reactions = frac_reaction.split('[+]')
                            list_of_reactions_cleaned=[]
                            for reac in list_of_reactions:
                                reac = reac.rstrip()
                                reac = reac.lstrip()
                                list_of_reactions_cleaned.append(reac)
                            for reac in list_of_reactions_cleaned:
                                reaction_number_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism.index(reac))
                                reactions_in_cti_file_with_these_reactants.append(reac)
                            if frac_reaction == denominator:
                                idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                                indx.append(idx)
                            else:    
                                indx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))     
                        else:
                            indx.append(reaction_list_from_mechanism.index(frac_reaction))

                    sigma_list_for_target_ks[unique_reactions.index(indx)].append(sigma_k)
                        
                else: 
                           
                    if '[*]' in k_target_values_parsed_csv['Reaction'][row]:
                        indx = []
                        reactions_in_cti_file_with_these_reactants = []
                        reaction_number_in_cti_file_with_these_reactants = []
                        reactions_in_cti_file_with_these_products = []
                        reaction_number_in_cti_file_with_these_products = []
                        reactants_in_target_reactions = k_target_values_parsed_csv['Reaction'][row].split('<=>')[0].rstrip()
                        reverse_reactants_in_target_reaction=None
                        if len(reactants_in_target_reactions.split('+'))>1:
                            reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                            temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                            temp = temp.lstrip()
                            temp = temp.rstrip()
                            reverse_reactants_in_target_reaction = temp
                        if  reverse_reactants_in_target_reaction != None:       
                            for reaction_number_in_cti_file in range(gas.n_reactions):
                                if  (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                                     
                                        reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                                     
                                        reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)        
                        elif reverse_reactants_in_target_reaction == None:
                            for reaction_number_in_cti_file in range(gas.n_reactions):
                                if  (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'): 
                                        reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)'  or 
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'): 
                                        reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)                  
                        indx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                        indx.append(tuple(sorted(reaction_number_in_cti_file_with_these_products)))                       
                        sigma_list_for_target_ks[unique_reactions.index(indx)].append(sigma_k)
                    elif '[+]' in k_target_values_parsed_csv['Reaction'][row]:
                        reactions_in_cti_file_with_these_reactants = []
                        reaction_number_in_cti_file_with_these_reactants = []
                        list_of_reactions = self.rate_constant_plots_df['Reaction'][row].split('[+]')
                        list_of_reactions_cleaned=[]
                        for reac in list_of_reactions:
                            reac = reac.rstrip()
                            reac = reac.lstrip()
                            list_of_reactions_cleaned.append(reac)
                        for reac in list_of_reactions_cleaned:
                            reaction_number_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism.index(reac))
                            reactions_in_cti_file_with_these_reactants.append(reac)    
                        indx = tuple(sorted(reaction_number_in_cti_file_with_these_reactants))
                        sigma_list_for_target_ks[unique_reactions.index(indx)].append(sigma_k)                            
                    else:       
                        indx = reaction_list_from_mechanism.index(k_target_values_parsed_csv['Reaction'][row])
                        sigma_list_for_target_ks[unique_reactions.index(indx)].append(sigma_k)
                
            return sigma_list_for_target_ks
#Trump         
        def calculating_target_value_ks_from_cantera_for_sigmas(k_target_values_parsed_csv,gas,unique_reactions):
            target_value_ks = [[] for reaction in range(len(unique_reactions))]
            
            target_reactions = k_target_values_parsed_csv['Reaction']
            target_temp = k_target_values_parsed_csv['temperature']
            target_press = k_target_values_parsed_csv['pressure']
            reactions_in_cti_file = gas.reaction_equations()

            for i,reaction in enumerate(target_reactions): 
                if '[/]' in reaction:
                    numerator = reaction.split('[/]')[0].rstrip().lstrip()
                    denominator = reaction.split('[/]')[1].rstrip().lstrip()
                    indx = []
                    for frac_reaction in [numerator, denominator]:
                        if "[*]" in frac_reaction:
                            if target_press[i] == 0:
                                pressure = 1e-9
                            else:
                                pressure = target_press[i]
                            pressure = 1
                            gas.TPX = target_temp[i],pressure*101325,{'Ar':1}
                            reactions_in_cti_file_with_these_reactants = []
                            reaction_number_in_cti_file_with_these_reactants = []
                            reactions_in_cti_file_with_these_products = []
                            reaction_number_in_cti_file_with_these_products = []
                            idx = []
                            reactants_in_target_reactions = frac_reaction.split('<=>')[0].rstrip()
                            reverse_reactants_in_target_reaction=None
                            if len(reactants_in_target_reactions.split('+'))>1:
                                reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                                temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                                temp = temp.lstrip()
                                temp = temp.rstrip()
                                reverse_reactants_in_target_reaction = temp      
                            if reverse_reactants_in_target_reaction !=None:
                                for reaction_number_in_cti_file in range(gas.n_reactions):
                                    if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or                  
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                        
                                            reactions_in_cti_file_with_these_reactants.append(reactions_in_cti_file[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                    elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or                  
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                        
                                            reactions_in_cti_file_with_these_products.append(reactions_in_cti_file[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)  
                            elif reverse_reactants_in_target_reaction ==None:
                                for reaction_number_in_cti_file in range(gas.n_reactions):
                                    if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or                 
                                        gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):                        
                                            reactions_in_cti_file_with_these_reactants.append(reactions_in_cti_file[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                    elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or                  
                                        gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):                        
                                            reactions_in_cti_file_with_these_products.append(reactions_in_cti_file[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)
                                                       
                            total_k = []
                            for rnum, secondary_reaction in enumerate(reactions_in_cti_file_with_these_reactants):
                                reaction_number_in_cti = reactions_in_cti_file.index(secondary_reaction)
                                coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())
            
                                k = gas.forward_rate_constants[reaction_number_in_cti]
                                if coeff_sum==1:
                                    k=k
                                elif coeff_sum==2:
                                    k = k*1000
                                elif coeff_sum==3:
                                    k = k*1000000

                                total_k.append(k)
                            
                            k = sum(total_k)
                            idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                            idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_products)))
                            indx.append(idx)

                        elif '[+]' in frac_reaction: 
                            idx = []
                            total_k=[]
                            reactions_in_cti_file_with_these_reactants = []
                            reaction_number_in_cti_file_with_these_reactants = []                            
                            if target_press[i] == 0:
                                pressure = 1e-9
                            else:
                                pressure = target_press[i]
                            #this is a temporary fix need to figure out how to pass this in 
                            pressure=1
                            # gas.TPX = target_temp[i],pressure*101325,{'H2O2':0.003094,'O2':0.000556,'H2O':0.001113,'Ar':0.995237}
                            gas.TPX = target_temp[i],pressure*101325,{'Ar':1}                            
                            list_of_reactions = frac_reaction.split('[+]')
                            list_of_reactions_cleaned=[]                            
                            for reac in list_of_reactions:
                                reac = reac.rstrip()
                                reac = reac.lstrip()
                                list_of_reactions_cleaned.append(reac)
                            for reac in list_of_reactions_cleaned:
                                reaction_number_in_cti = reactions_in_cti_file.index(reac)
                                reactions_in_cti_file_with_these_reactants.append(reac)
                                reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti)
                                coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())
            
                                k = gas.forward_rate_constants[reaction_number_in_cti]
                                if coeff_sum==1:
                                    k=k
                                elif coeff_sum==2:
                                    k = k*1000
                                elif coeff_sum==3:
                                    k = k*1000000

                                total_k.append(k)                        
                                                
                            k = sum(total_k)
                            if frac_reaction == denominator:
                                idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                                indx.append(idx)
                            else:
                                indx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                            # target_value_ks[unique_reactions.index(indx)].append(k)        
                        else:
                            if target_press[i] == 0:
                                pressure = 1e-9
                            else:
                                pressure = target_press[i]
                            #this is a temporary fix need to figure out how to pass this in 
                            pressure=1
                            # gas.TPX = target_temp[i],pressure*101325,{'H2O2':0.003094,'O2':0.000556,'H2O':0.001113,'Ar':0.995237}
                            gas.TPX = target_temp[i],pressure*101325,{'Ar':1}
                            
                            reaction_number_in_cti = reactions_in_cti_file.index(frac_reaction)
                            indx.append(reactions_in_cti_file.index(frac_reaction))
                            coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())
            
                            k = gas.forward_rate_constants[reaction_number_in_cti]
                            if coeff_sum==1:
                                k=k
                            elif coeff_sum==2:
                                k = k*1000
                            elif coeff_sum==3:
                                k = k*1000000


                    target_value_ks[unique_reactions.index(indx)].append(k)


                else:
                                                        
                    if "[*]" in reaction:
                        indx = []
                        if target_press[i] == 0:
                            pressure = 1e-9
                        else:
                            pressure = target_press[i]
                        pressure = 1
                        gas.TPX = target_temp[i],pressure*101325,{'Ar':1}
                        reactions_in_cti_file_with_these_reactants = []
                        reaction_number_in_cti_file_with_these_reactants = []
                        reactions_in_cti_file_with_these_products = []
                        reaction_number_in_cti_file_with_these_products = []
                        reactants_in_target_reactions = reaction.split('<=>')[0].rstrip()
                        reverse_reactants_in_target_reaction=None
                        if len(reactants_in_target_reactions.split('+'))>1:
                            reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                            temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                            temp = temp.lstrip()
                            temp = temp.rstrip()
                            reverse_reactants_in_target_reaction = temp
                                
                        if reverse_reactants_in_target_reaction !=None:
                            for reaction_number_in_cti_file in range(gas.n_reactions):
                                if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or                  
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                        
                                        reactions_in_cti_file_with_these_reactants.append(reactions_in_cti_file[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or                  
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                        
                                        reactions_in_cti_file_with_these_products.append(reactions_in_cti_file[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)          
                        elif reverse_reactants_in_target_reaction ==None:
                            for reaction_number_in_cti_file in range(gas.n_reactions):
                                if (gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or                 
                                    gas.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):                        
                                        reactions_in_cti_file_with_these_reactants.append(reactions_in_cti_file[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                elif(gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or                 
                                    gas.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):                        
                                        reactions_in_cti_file_with_these_products.append(reactions_in_cti_file[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)                                  
                        indx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                        indx.append(tuple(sorted(reaction_number_in_cti_file_with_these_products)))
                        total_k = []
                        for rnum, secondary_reaction in enumerate(reactions_in_cti_file_with_these_reactants):
                            reaction_number_in_cti = reactions_in_cti_file.index(secondary_reaction)
                            coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())
                            k = gas.forward_rate_constants[reaction_number_in_cti]
                            if coeff_sum==1:
                                k=k
                            elif coeff_sum==2:
                                k = k*1000
                            elif coeff_sum==3:
                                k = k*1000000
                            total_k.append(k)
                        for rnum, secondary_reaction in enumerate(reactions_in_cti_file_with_these_products):
                            reaction_number_in_cti = reactions_in_cti_file.index(secondary_reaction)
                            coeff_sum = sum(gas.reaction(reaction_number_in_cti).products.values())
                            k = gas.reverse_rate_constants[reaction_number_in_cti]
                            if coeff_sum==1:
                                k=k
                            elif coeff_sum==2:
                                k = k*1000
                            elif coeff_sum==3:
                                k = k*1000000
                            total_k.append(k)                       
                        k = sum(total_k)
                        target_value_ks[unique_reactions.index(indx)].append(k)
                        
                    elif '[+]' in reaction:
                        reactions_in_cti_file_with_these_reactants = []
                        reaction_number_in_cti_file_with_these_reactants = []
                        if target_press[i] == 0:
                            pressure = 1e-9
                        else:
                            pressure = target_press[i] 
                        pressure=1
                        gas.TPX = target_temp[i],pressure*101325,{'Ar':1}
                        list_of_reactions = reaction.split('[+]')
                        list_of_reactions_cleaned=[]
                        
                        
                        for reac in list_of_reactions:
                            reac = reac.rstrip()
                            reac = reac.lstrip()
                            list_of_reactions_cleaned.append(reac)
                        
                        total_k=[]    
                        for reac in list_of_reactions_cleaned:
                            reaction_number_in_cti = reactions_in_cti_file.index(reac)
                            reactions_in_cti_file_with_these_reactants.append(reac)
                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti)
                            coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())
        
                            k = gas.forward_rate_constants[reaction_number_in_cti]
                            if coeff_sum==1:
                                k=k
                            elif coeff_sum==2:
                                k = k*1000
                            elif coeff_sum==3:
                                k = k*1000000

                            total_k.append(k)                        
                                            
                        k = sum(total_k)
                        indx = tuple(sorted(reaction_number_in_cti_file_with_these_reactants))
                        target_value_ks[unique_reactions.index(indx)].append(k)                    
                        
                        
                    else:

                        if target_press[i] == 0:
                            pressure = 1e-9
                        else:
                            pressure = target_press[i]
                        #this is a temporary fix need to figure out how to pass this in 
                        pressure=1
                        # gas.TPX = target_temp[i],pressure*101325,{'H2O2':0.003094,'O2':0.000556,'H2O':0.001113,'Ar':0.995237}
                        gas.TPX = target_temp[i],pressure*101325,{'Ar':1}
                        
                        reaction_number_in_cti = reactions_in_cti_file.index(reaction)
                        indx = reactions_in_cti_file.index(reaction)
                        coeff_sum = sum(gas.reaction(reaction_number_in_cti).reactants.values())
        
                        k = gas.forward_rate_constants[reaction_number_in_cti]
                        if coeff_sum==1:
                            k=k
                        elif coeff_sum==2:
                            k = k*1000
                        elif coeff_sum==3:
                            k = k*1000000
                        target_value_ks[unique_reactions.index(indx)].append(k)


            return target_value_ks
#Pence     
    
        if bool(self.rate_constant_plots_csv):

            df = pd.read_csv(os.path.join(self.working_directory,self.rate_constant_plots_csv))

            rate_constant_plots_dict = {'Reaction':[],'temperature':[],'pressure':[],'M':[],'k':[],'ln_unc_k':[],'W':[]}
            for i, reaction in enumerate(df['Reaction']):
                for Temp in np.linspace(df.iloc[i]['Tmin'],df.iloc[i]['Tmax'],int(np.divide(df.iloc[i]['Tmax']-df.iloc[i]['Tmin'],50))+1):
                        rate_constant_plots_dict['Reaction'].append(reaction)
                        rate_constant_plots_dict['temperature'].append(Temp)
                        rate_constant_plots_dict['pressure'].append(df.iloc[i]['pressure'])
                        rate_constant_plots_dict['M'].append(df.iloc[i]['M'])
                        rate_constant_plots_dict['k'].append(df.iloc[i]['k'])
                        rate_constant_plots_dict['ln_unc_k'].append(df.iloc[i]['ln_unc_k'])
                        rate_constant_plots_dict['W'].append(df.iloc[i]['W'])
                        
            self.rate_constant_plots_df = pd.DataFrame(rate_constant_plots_dict)            
            
                                
            S_matrix_k_target_values_extra = target_values_for_S(self.rate_constant_plots_df,
                                                                 self.exp_dict_list_optimized,
                                                                 self.S_matrix,
                                                                 master_equation_reaction_list = self.master_equation_reactions,
                                                                 master_equation_sensitivites=self.cheby_sensitivity_dict)
                        
            unique_reactions_optimized=[]
            unique_reactions_original = []
            
            reaction_list_from_mechanism_original = gas_original.reaction_equations()
            reaction_list_from_mechanism = gas_optimized.reaction_equations()
            # k_target_value_csv_extra = rate_constant_plots_df   
            if bool(self.target_value_rate_constant_csv):
                k_target_value_csv = pd.read_csv(os.path.join(self.working_directory,self.target_value_rate_constant_csv))
            else:
                k_target_value_csv = pd.DataFrame(columns=['Reaction','temperature','pressure','M','k','ln_unc_k','W','ref'])
        #edit here to skip plotting ?

            for row in range(self.rate_constant_plots_df.shape[0]):
                if '[/]' in self.rate_constant_plots_df['Reaction'][row]:                    
                    numerator = self.rate_constant_plots_df['Reaction'][row].split('[/]')[0].rstrip().lstrip()
                    denominator = self.rate_constant_plots_df['Reaction'][row].split('[/]')[1].rstrip().lstrip()
                    unique_fraction_reaction_optimized = []
                    unique_fraction_reaction_original = []
                    for frac_reaction in [numerator, denominator]:
                        if "[*]" in frac_reaction:
                            idx = []
                            Idx = []
                            reactions_in_cti_file_with_these_reactants = []
                            reaction_number_in_cti_file_with_these_reactants = []
                            reactions_in_cti_file_with_these_reactants_original = []
                            reaction_number_in_cti_file_with_these_reactants_original = []
                            reactions_in_cti_file_with_these_products = []
                            reaction_number_in_cti_file_with_these_products = []
                            reactions_in_cti_file_with_these_products_original = []
                            reaction_number_in_cti_file_with_these_products_original = []                    
                            reactants_in_target_reactions = frac_reaction.split('<=>')[0].rstrip()
                            reverse_reactants_in_target_reaction=None
                            if len(reactants_in_target_reactions.split('+'))>1:
                                reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                                temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                                temp = temp.lstrip()
                                temp = temp.rstrip()
                                reverse_reactants_in_target_reaction = temp
                            if  reverse_reactants_in_target_reaction !=None:
                                for reaction_number_in_cti_file in range(gas_optimized.n_reactions):
                                    if (gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' (+M)' or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M' or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' + M'):    
                                            reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                    elif(gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_optimized.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                        gas_optimized.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' (+M)' or 
                                        gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M' or 
                                        gas_optimized.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' + M'):    
                                            reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)                            
                                for reaction_number_in_cti_file in range(gas_original.n_reactions):
                                    if (gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                               
                                            reactions_in_cti_file_with_these_reactants_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants_original.append(reaction_number_in_cti_file)
                                    elif(gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_original.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas_original.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                        gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                        gas_original.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                               
                                            reactions_in_cti_file_with_these_products_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_products_original.append(reaction_number_in_cti_file) 
                            elif  reverse_reactants_in_target_reaction ==None:
                                for reaction_number_in_cti_file in range(gas_optimized.n_reactions):
                                    if (gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M'):
                                            reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                                    elif(gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_optimized.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                        gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M'):
                                            reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)                    
                                for reaction_number_in_cti_file in range(gas_original.n_reactions):
                                    if (gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):
                                            reactions_in_cti_file_with_these_reactants_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants_original.append(reaction_number_in_cti_file)
                                    elif(gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_original.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):
                                            reactions_in_cti_file_with_these_products_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_products_original.append(reaction_number_in_cti_file)                      
                            idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                            Idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants_original)))
                            idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_products)))
                            Idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_products_original)))
                            unique_fraction_reaction_optimized.append(idx)
                            unique_fraction_reaction_original.append(Idx)
                        elif "[+]" in frac_reaction:
                            idx = []
                            Idx = []
                            reactions_in_cti_file_with_these_reactants = []
                            reaction_number_in_cti_file_with_these_reactants = []
                            reactions_in_cti_file_with_these_reactants_original = []
                            reaction_number_in_cti_file_with_these_reactants_original = []   
                            list_of_reactions = frac_reaction.split('[+]')
                            list_of_reactions_cleaned=[]
                            for reac in list_of_reactions:
                                reac = reac.rstrip()
                                reac = reac.lstrip()
                                list_of_reactions_cleaned.append(reac)
                            for reac in list_of_reactions_cleaned:
                                reaction_number_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism.index(reac))
                                reactions_in_cti_file_with_these_reactants.append(reac)                    
                                reaction_number_in_cti_file_with_these_reactants_original.append(reaction_list_from_mechanism_original.index(reac))
                                reactions_in_cti_file_with_these_reactants_original.append(reac)
                            if frac_reaction == denominator:
                                idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                                unique_fraction_reaction_optimized.append(idx)
                                Idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants_original)))
                                unique_fraction_reaction_original.append(Idx)
                            else:   
                                unique_fraction_reaction_optimized.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                                unique_fraction_reaction_original.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants_original)))    
                        else:
                            unique_fraction_reaction_optimized.append(reaction_list_from_mechanism.index(frac_reaction))
                            unique_fraction_reaction_original.append(reaction_list_from_mechanism_original.index(frac_reaction)) 

                    unique_reactions_optimized.append(unique_fraction_reaction_optimized)
                    unique_reactions_original.append(unique_fraction_reaction_original)

                else:                    
                    if "[*]" in self.rate_constant_plots_df['Reaction'][row]:
                        idx = []
                        Idx = []
                        reactions_in_cti_file_with_these_reactants = []
                        reaction_number_in_cti_file_with_these_reactants = []
                        reactions_in_cti_file_with_these_reactants_original = []
                        reaction_number_in_cti_file_with_these_reactants_original = []
                        reactions_in_cti_file_with_these_products = []
                        reaction_number_in_cti_file_with_these_products = []
                        reactions_in_cti_file_with_these_products_original = []
                        reaction_number_in_cti_file_with_these_products_original = []                    
                        reactants_in_target_reactions = self.rate_constant_plots_df['Reaction'][row].split('<=>')[0].rstrip()
                        reverse_reactants_in_target_reaction=None
                        if len(reactants_in_target_reactions.split('+'))>1:
                            reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                            temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                            temp = temp.lstrip()
                            temp = temp.rstrip()
                            reverse_reactants_in_target_reaction = temp
                        if  reverse_reactants_in_target_reaction !=None:
                            for reaction_number_in_cti_file in range(gas_optimized.n_reactions):
                                if (gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' (+M)' or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M' or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' + M'):    
                                        reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file) 
                                elif(gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_optimized.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                    gas_optimized.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' (+M)' or 
                                    gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M' or 
                                    gas_optimized.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' + M'):    
                                        reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)                    
                            for reaction_number_in_cti_file in range(gas_original.n_reactions):
                                if (gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                               
                                        reactions_in_cti_file_with_these_reactants_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants_original.append(reaction_number_in_cti_file)
                                elif(gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_original.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas_original.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                    gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                    gas_original.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                               
                                        reactions_in_cti_file_with_these_products_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_products_original.append(reaction_number_in_cti_file)         
                        elif  reverse_reactants_in_target_reaction ==None:
                            for reaction_number_in_cti_file in range(gas_optimized.n_reactions):
                                if (gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M'):
                                        reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file) 
                                elif(gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_optimized.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                    gas_optimized.products(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M'):
                                        reactions_in_cti_file_with_these_products.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_products.append(reaction_number_in_cti_file)                            
                            for reaction_number_in_cti_file in range(gas_original.n_reactions):
                                if (gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):
                                        reactions_in_cti_file_with_these_reactants_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants_original.append(reaction_number_in_cti_file) 
                                elif(gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_original.products(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas_original.products(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):
                                        reactions_in_cti_file_with_these_products_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_products_original.append(reaction_number_in_cti_file)
                        idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                        Idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants_original)))
                        idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_products)))
                        Idx.append(tuple(sorted(reaction_number_in_cti_file_with_these_products_original)))
                        unique_reactions_optimized.append(idx)
                        unique_reactions_original.append(Idx)                       
                    elif "[+]" in self.rate_constant_plots_df['Reaction'][row]:
                        reactions_in_cti_file_with_these_reactants = []
                        reaction_number_in_cti_file_with_these_reactants = []
                        reactions_in_cti_file_with_these_reactants_original = []
                        reaction_number_in_cti_file_with_these_reactants_original = []   
                        list_of_reactions = self.rate_constant_plots_df['Reaction'][row].split('[+]')
                        list_of_reactions_cleaned=[]
                        for reac in list_of_reactions:
                            reac = reac.rstrip()
                            reac = reac.lstrip()
                            list_of_reactions_cleaned.append(reac)
                        for reac in list_of_reactions_cleaned:
                            reaction_number_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism.index(reac))
                            reactions_in_cti_file_with_these_reactants.append(reac)                    
                            reaction_number_in_cti_file_with_these_reactants_original.append(reaction_list_from_mechanism_original.index(reac))
                            reactions_in_cti_file_with_these_reactants_original.append(reac)
                        unique_reactions_optimized.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                        unique_reactions_original.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants_original)))
                    else:
                        unique_reactions_optimized.append(reaction_list_from_mechanism.index(self.rate_constant_plots_df['Reaction'][row]))
                        unique_reactions_original.append(reaction_list_from_mechanism_original.index(self.rate_constant_plots_df['Reaction'][row]))
#Biden          
            unique_reactions_optimized = unique_list(unique_reactions_optimized)
            unique_reactions_original = unique_list(unique_reactions_original)
            
            sigma_list_for_target_ks_optimized = calculate_sigmas_for_rate_constants(S_matrix_k_target_values_extra,self.rate_constant_plots_df,unique_reactions_optimized,gas_optimized,self.covariance)
          
            self.sigma_list_for_target_ks_optimized = sigma_list_for_target_ks_optimized
            
            sigma_list_for_target_ks_original = calculate_sigmas_for_rate_constants(S_matrix_k_target_values_extra,self.rate_constant_plots_df,unique_reactions_original,gas_original,self.covariance_original)
            self.sigma_list_for_target_ks_original = sigma_list_for_target_ks_original
           #  ######################  
            
            target_value_temps_optimized,target_value_ks_optimized,target_value_refs_optimized = sort_rate_constant_target_values(self.rate_constant_plots_df,unique_reactions_optimized,gas_optimized)
            target_value_temps_original,target_value_ks_original,target_value_refs_original = sort_rate_constant_target_values(self.rate_constant_plots_df,unique_reactions_original,gas_original)
           
            
            
           #  ############################################# 
            unique_reactions_optimized_for_plotting=[]
            unique_reactions_original_for_plotting = []
            reactions_in_cti_file_with_these_reactants_original = []
            reaction_number_in_cti_file_with_these_reactants_original = []  
            
            
            
            
            
            #need to start editing here tomorrow
            for row in range(k_target_value_csv.shape[0]):
                if '[/]' in k_target_value_csv['Reaction'][row]:
                    
                    numerator = k_target_value_csv['Reaction'][row].split('[/]')[0].rstrip().lstrip()
                    denominator = k_target_value_csv['Reaction'][row].split('[/]')[1].rstrip().lstrip()

                    unique_fraction_reaction_optimized_for_plotting = []
                    unique_fraction_reaction_original_for_plotting = []
                    for frac_reaction in [numerator, denominator]:

                        if "[*]" in frac_reaction:
                            reactions_in_cti_file_with_these_reactants = []
                            reaction_number_in_cti_file_with_these_reactants = []
                                            
                            #might be a more comprehensive way to do this 
                            reactants_in_target_reactions = frac_reaction.split('<=>')[0].rstrip()
                            reverse_reactants_in_target_reaction=None
                            if len(reactants_in_target_reactions.split('+'))>1:
                                reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                                temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                                temp = temp.lstrip()
                                temp = temp.rstrip()
                                reverse_reactants_in_target_reaction = temp
                            if reverse_reactants_in_target_reaction !=None:
                                
                                
                                for reaction_number_in_cti_file in range(gas_optimized.n_reactions):
                                    if (gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' (+M)' or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M' or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' + M'):                                   
                                        
                                        
                                            reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)  
                            
                                for reaction_number_in_cti_file in range(gas_original.n_reactions):
                                    if (gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                                 
                                        
                                        
                                        
                                            reactions_in_cti_file_with_these_reactants_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants_original.append(reaction_number_in_cti_file)                     


                            elif reverse_reactants_in_target_reaction ==None:
                                
                                for reaction_number_in_cti_file in range(gas_optimized.n_reactions):
                                    if (gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                        gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M'):                                   
                                        
                                        
                                            reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)  
                            
                                for reaction_number_in_cti_file in range(gas_original.n_reactions):
                                    if (gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                        gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):                                 
                                        
                                        
                                        
                                            reactions_in_cti_file_with_these_reactants_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                            reaction_number_in_cti_file_with_these_reactants_original.append(reaction_number_in_cti_file)  
                            
                            unique_fraction_reaction_optimized_for_plotting.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                            unique_fraction_reaction_original_for_plotting.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants_original)))
                        elif '[+]' in frac_reaction:
                            reactions_in_cti_file_with_these_reactants = []
                            reaction_number_in_cti_file_with_these_reactants = []
                            reactions_in_cti_file_with_these_reactants_original =[]
                            reaction_number_in_cti_file_with_these_reactants_original = []
                            
                            list_of_reactions = frac_reaction.split('[+]')
                            list_of_reactions_cleaned=[]
                            
                            for reaction in list_of_reactions:
                                reaction = reaction.rstrip()
                                reaction = reaction.lstrip()
                                list_of_reactions_cleaned.append(reaction)
            
                
                            for reaction in list_of_reactions_cleaned:
                                reaction_number_in_cti_file = reaction_list_from_mechanism.index(reaction) 

                                reactions_in_cti_file_with_these_reactants.append(reaction)                    
                                reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                            
                            
                            for reaction in list_of_reactions_cleaned:
                                reaction_number_in_cti_file = reaction_list_from_mechanism_original.index(reaction) 

                                reactions_in_cti_file_with_these_reactants_original.append(reaction)                    
                                reaction_number_in_cti_file_with_these_reactants_original.append(reaction_number_in_cti_file)  

                            
                            
                            
                            unique_fraction_reaction_optimized_for_plotting.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                            unique_fraction_reaction_original_for_plotting.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants_original)))
                        else:
                            unique_fraction_reaction_optimized_for_plotting.append(reaction_list_from_mechanism.index(frac_reaction))
                            unique_fraction_reaction_original_for_plotting.append(reaction_list_from_mechanism_original.index(frac_reaction))

                    unique_reactions_optimized_for_plotting.append(unique_fraction_reaction_optimized_for_plotting)
                    unique_reactions_original_for_plotting.append(unique_fraction_reaction_original_for_plotting)
                
                else:

                    if "[*]" in k_target_value_csv['Reaction'][row]:
                        reactions_in_cti_file_with_these_reactants = []
                        reaction_number_in_cti_file_with_these_reactants = []
                                        
                        #might be a more comprehensive way to do this 
                        reactants_in_target_reactions = k_target_value_csv['Reaction'][row].split('<=>')[0].rstrip()
                        reverse_reactants_in_target_reaction=None
                        if len(reactants_in_target_reactions.split('+'))>1:
                            reverse_reactants_in_target_reaction = reactants_in_target_reactions.split('+')
                            temp = reverse_reactants_in_target_reaction[1] + ' '+ '+' +' '+ reverse_reactants_in_target_reaction[0]
                            temp = temp.lstrip()
                            temp = temp.rstrip()
                            reverse_reactants_in_target_reaction = temp
                        if reverse_reactants_in_target_reaction !=None:
                            
                            
                            for reaction_number_in_cti_file in range(gas_optimized.n_reactions):
                                if (gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' (+M)' or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M' or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction +   ' + M'):                                   
                                    
                                    
                                        reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)  
                        
                            for reaction_number_in_cti_file in range(gas_original.n_reactions):
                                if (gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' (+M)' or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M' or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction + ' + M'):                                 
                                    
                                    
                                    
                                        reactions_in_cti_file_with_these_reactants_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants_original.append(reaction_number_in_cti_file)                     


                        elif reverse_reactants_in_target_reaction ==None:
                            
                            for reaction_number_in_cti_file in range(gas_optimized.n_reactions):
                                if (gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' (+M)' or 
                                    gas_optimized.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions +  ' + M'):                                   
                                    
                                    
                                        reactions_in_cti_file_with_these_reactants.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)  
                        
                            for reaction_number_in_cti_file in range(gas_original.n_reactions):
                                if (gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reverse_reactants_in_target_reaction or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' (+M)' or 
                                    gas_original.reactants(reaction_number_in_cti_file) == reactants_in_target_reactions + ' + M'):                                 
                                    
                                    
                                    
                                        reactions_in_cti_file_with_these_reactants_original.append(reaction_list_from_mechanism[reaction_number_in_cti_file])                    
                                        reaction_number_in_cti_file_with_these_reactants_original.append(reaction_number_in_cti_file)  
                        
                        unique_reactions_optimized_for_plotting.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                        unique_reactions_original_for_plotting.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants_original)))

                    elif '[+]' in k_target_value_csv['Reaction'][row]:
                        reactions_in_cti_file_with_these_reactants = []
                        reaction_number_in_cti_file_with_these_reactants = []
                        reactions_in_cti_file_with_these_reactants_original =[]
                        reaction_number_in_cti_file_with_these_reactants_original = []
                        
                        list_of_reactions = k_target_value_csv['Reaction'][row].split('[+]')
                        list_of_reactions_cleaned=[]
                        
                        for reaction in list_of_reactions:
                            reaction = reaction.rstrip()
                            reaction = reaction.lstrip()
                            list_of_reactions_cleaned.append(reaction)
        
            
                        for reaction in list_of_reactions_cleaned:
                            reaction_number_in_cti_file = reaction_list_from_mechanism.index(reaction) 

                            reactions_in_cti_file_with_these_reactants.append(reaction)                    
                            reaction_number_in_cti_file_with_these_reactants.append(reaction_number_in_cti_file)
                        
                        
                        for reaction in list_of_reactions_cleaned:
                            reaction_number_in_cti_file = reaction_list_from_mechanism_original.index(reaction) 

                            reactions_in_cti_file_with_these_reactants_original.append(reaction)                    
                            reaction_number_in_cti_file_with_these_reactants_original.append(reaction_number_in_cti_file)  

                        
                        
                        
                        unique_reactions_optimized_for_plotting.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants)))
                        unique_reactions_original_for_plotting.append(tuple(sorted(reaction_number_in_cti_file_with_these_reactants_original)))
        
        
                    else:
                        unique_reactions_optimized_for_plotting.append(reaction_list_from_mechanism.index(k_target_value_csv['Reaction'][row]))
                        unique_reactions_original_for_plotting.append(reaction_list_from_mechanism_original.index(k_target_value_csv['Reaction'][row]))
            
            #this could be wrong
            
            unique_reactions_optimized_for_plotting = unique_list(unique_reactions_optimized)
            
            unique_reactions_original_for_plotting = unique_list(unique_reactions_original)            
            
            target_value_temps_optimized_for_plotting,target_value_ks_optimized_for_plotting,target_value_refs_optimized_for_plotting = sort_rate_constant_target_values(k_target_value_csv,unique_reactions_optimized_for_plotting,gas_optimized)
            target_value_temps_original_for_plotting,target_value_ks_original_for_plotting,target_value_refs_original_for_plotting = sort_rate_constant_target_values(k_target_value_csv,unique_reactions_original_for_plotting,gas_original)
           # #############################################
           
           
            target_value_ks_calculated_with_cantera_optimized = calculating_target_value_ks_from_cantera_for_sigmas(self.rate_constant_plots_df,gas_optimized,unique_reactions_optimized)
            target_value_ks_calculated_with_cantera_original = calculating_target_value_ks_from_cantera_for_sigmas(self.rate_constant_plots_df,gas_original,unique_reactions_original)    
            
            #print(unique_reactions_optimized)
            self.unique_reactions_optimized = unique_reactions_optimized
            
            marker_list = [
                           's', '^', 'd', '8', 'v', 'X', 'h', '<', 'p', '*', '>', 'P', 'D',
                           's', '^', 'd', '8', 'v', 'X', 'h', '<', 'p', '*', '>', 'P', 'D',
                           's', '^', 'd', '8', 'v', 'X', 'h', '<', 'p', '*', '>', 'P', 'D',
                           's', '^', 'd', '8', 'v', 'X', 'h', '<', 'p', '*', '>', 'P', 'D',
                           's', '^', 'd', '8', 'v', 'X', 'h', '<', 'p', '*', '>', 'P', 'D',
                           's', '^', 'd', '8', 'v', 'X', 'h', '<', 'p', '*', '>', 'P', 'D',
                           's', '^', 'd', '8', 'v', 'X', 'h', '<', 'p', '*', '>', 'P', 'D',
                          ]     
            marker_refs_list = []       
            
            self.rate_loop = self.manager.counter(total=len(unique_reactions_optimized), desc='Rate Constant Plots:', unit='plots', color='green') 
                                
            rates_csv = pd.read_csv(self.rate_constant_plots_csv)
            
                                
            for i,reaction in enumerate(unique_reactions_optimized):
                
                initial_temperature = rates_csv.iloc[i]['Tmin']
                final_temperature = rates_csv.iloc[i]['Tmax']
                
                plt.figure()
                optimized_rate_constant_df = pd.DataFrame()
                original_rate_constant_df = pd.DataFrame()
                
                if type(reaction)==list:
                    if type(reaction[1]) == tuple: # Total Rate
                        new_list = []
                        for sub_number in reaction:
                            new_tuple = []
                            for bub_number in sub_number:
                                new_tuple.append(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[bub_number]))
                            new_tuple = tuple(sorted(new_tuple))
                            new_list.append(new_tuple)
                        Temp_optimized,k_optimized = rate_constant_over_temperature_range_from_cantera(reaction,
                                                                        gas_optimized,
                                                                        initial_temperature=initial_temperature,
                                                                        final_temperature=final_temperature,
                                                                        pressure=1, 
                                                                        conditions={'Ar':1})  
                        
                        Temp_original,k_original = rate_constant_over_temperature_range_from_cantera(new_list,
                                                                            gas_original,
                                                                            initial_temperature=initial_temperature,
                                                                            final_temperature=final_temperature,
                                                                            pressure=1,
                                                                            conditions={'Ar':1}) 
                        
                        high_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[i]))
                        high_error_optimized = np.multiply(high_error_optimized,target_value_ks_calculated_with_cantera_optimized[i])
                        low_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[i])*-1)
                        low_error_optimized = np.multiply(low_error_optimized,target_value_ks_calculated_with_cantera_optimized[i])      
                        a, b = zip(*sorted(zip(target_value_temps_optimized[i],high_error_optimized)))
                        aa, bb = zip(*sorted(zip(target_value_temps_optimized[i],low_error_optimized)))                                        

                        high_error_original = np.exp(sigma_list_for_target_ks_original[unique_reactions_original.index(new_list)])
                        high_error_original = np.multiply(high_error_original,target_value_ks_calculated_with_cantera_original[unique_reactions_original.index(new_list)])
                        low_error_original = np.exp(np.array(sigma_list_for_target_ks_original[unique_reactions_original.index(new_list)])*-1)
                        low_error_original = np.multiply(low_error_original,target_value_ks_calculated_with_cantera_original[unique_reactions_original.index(new_list)]) 
                        c, d = zip(*sorted(zip(target_value_temps_original[unique_reactions_original.index(new_list)],high_error_original)))    
                        cc, dd = zip(*sorted(zip(target_value_temps_original[unique_reactions_original.index(new_list)],low_error_original)))                      

                    else: #Ratio of rates
                        numerator = reaction[0]
                        denominator = reaction[1]

                        new_list =[]
                        for frac_reaction in [numerator, denominator]:
                            if type(frac_reaction) == tuple:
                                new_tuple =[]
                                for sub_number in frac_reaction:
                                    new_tuple.append(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[sub_number]))
                                new_tuple = tuple(sorted(new_tuple))
                                new_list.append(new_tuple)
                            elif type(frac_reaction) == list:
                                nu_list = []
                                for denom in range(len(frac_reaction)):
                                    new_tuple = []
                                    for nom in range(len(frac_reaction[denom])):
                                        new_tuple.append(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[frac_reaction[denom][nom]]))
                                    new_tuple = tuple(sorted(new_tuple))
                                    nu_list.append(new_tuple)
                                new_list.append(nu_list)
                            else:
                                new_list.append(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[frac_reaction]))
                                
                        Temp_optimized,k_optimized = rate_constant_over_temperature_range_from_cantera(reaction,
                                                                        gas_optimized,
                                                                        initial_temperature=initial_temperature,
                                                                        final_temperature=final_temperature,
                                                                        pressure=1,
                                                                        #   conditions={'H2O2':0.003094,'O2':0.000556,'H2O':0.001113,'Ar':0.995237})
                                                                        conditions={'Ar':1})  
                        
                        Temp_original,k_original = rate_constant_over_temperature_range_from_cantera(new_list,
                                                                            gas_original,
                                                                            initial_temperature=initial_temperature,
                                                                            final_temperature=final_temperature,
                                                                            pressure=1,
                                                                            #   conditions={'H2O2':0.003094,'O2':0.000556,'H2O':0.001113,'Ar':0.995237}) 
                                                                            conditions={'Ar':1})    
                        
                        # high_error_optimized = np.exp(sigma_list_for_target_ks_optimized[unique_reactions_optimized.index(new_list)])
                        # high_error_optimized = np.multiply(high_error_optimized,target_value_ks_calculated_with_cantera_optimized[unique_reactions_optimized.index(new_list)])
                        # low_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[unique_reactions_optimized.index(new_list)])*-1)
                        # low_error_optimized = np.multiply(low_error_optimized,target_value_ks_calculated_with_cantera_optimized[unique_reactions_optimized.index(new_list)]) 
                        # a, b = zip(*sorted(zip(target_value_temps_optimized[unique_reactions_optimized.index(new_list)],high_error_optimized)))    
                        # aa, bb = zip(*sorted(zip(target_value_temps_optimized[unique_reactions_optimized.index(new_list)],low_error_optimized)))    
                        
                        high_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[i]))
                        high_error_optimized = np.multiply(high_error_optimized,target_value_ks_calculated_with_cantera_optimized[i])
                        low_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[i])*-1)
                        low_error_optimized = np.multiply(low_error_optimized,target_value_ks_calculated_with_cantera_optimized[i])      
                        a, b = zip(*sorted(zip(target_value_temps_optimized[i],high_error_optimized)))
                        aa, bb = zip(*sorted(zip(target_value_temps_optimized[i],low_error_optimized)))                                        

                        high_error_original = np.exp(sigma_list_for_target_ks_original[unique_reactions_original.index(new_list)])
                        high_error_original = np.multiply(high_error_original,target_value_ks_calculated_with_cantera_original[unique_reactions_original.index(new_list)])
                        low_error_original = np.exp(np.array(sigma_list_for_target_ks_original[unique_reactions_original.index(new_list)])*-1)
                        low_error_original = np.multiply(low_error_original,target_value_ks_calculated_with_cantera_original[unique_reactions_original.index(new_list)]) 
                        c, d = zip(*sorted(zip(target_value_temps_original[unique_reactions_original.index(new_list)],high_error_original)))    
                        cc, dd = zip(*sorted(zip(target_value_temps_original[unique_reactions_original.index(new_list)],low_error_original)))                      

                elif type(reaction)==tuple: # Partial Sum of Rates
                    new_tuple =[]
                    for sub_number in reaction:
                        new_tuple.append(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[sub_number]))
                    new_tuple = tuple(sorted(new_tuple))
                    
                    Temp_optimized,k_optimized = rate_constant_over_temperature_range_from_cantera(reaction,
                                                                    gas_optimized,
                                                                    initial_temperature=initial_temperature,
                                                                    final_temperature=final_temperature,
                                                                    pressure=1,
                                                                    #   conditions={'H2O2':0.003094,'O2':0.000556,'H2O':0.001113,'Ar':0.995237})
                                                                    conditions={'Ar':1})  
                                        
                    Temp_original,k_original = rate_constant_over_temperature_range_from_cantera(new_tuple,
                                                                        gas_original,
                                                                        initial_temperature=initial_temperature,
                                                                        final_temperature=final_temperature,
                                                                        pressure=1,
                                                                        #   conditions={'H2O2':0.003094,'O2':0.000556,'H2O':0.001113,'Ar':0.995237})                    
                                                                        conditions={'Ar':1})  
                    
                    # high_error_optimized = np.exp(sigma_list_for_target_ks_optimized[unique_reactions_optimized.index(new_tuple)])
                    # high_error_optimized = np.multiply(high_error_optimized,target_value_ks_calculated_with_cantera_optimized[unique_reactions_optimized.index(new_tuple)])
                    # low_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[unique_reactions_optimized.index(new_tuple)])*-1)
                    # low_error_optimized = np.multiply(low_error_optimized,target_value_ks_calculated_with_cantera_optimized[unique_reactions_optimized.index(new_tuple)]) 
                    # a, b = zip(*sorted(zip(target_value_temps_optimized[unique_reactions_optimized.index(new_tuple)],high_error_optimized)))    
                    # aa, bb = zip(*sorted(zip(target_value_temps_optimized[unique_reactions_optimized.index(new_tuple)],low_error_optimized)))              
                      
                    high_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[i]))
                    high_error_optimized = np.multiply(high_error_optimized,target_value_ks_calculated_with_cantera_optimized[i])
                    low_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[i])*-1)
                    low_error_optimized = np.multiply(low_error_optimized,target_value_ks_calculated_with_cantera_optimized[i])      
                    a, b = zip(*sorted(zip(target_value_temps_optimized[i],high_error_optimized)))
                    aa, bb = zip(*sorted(zip(target_value_temps_optimized[i],low_error_optimized)))                                
                    
                    high_error_original = np.exp(sigma_list_for_target_ks_original[unique_reactions_original.index(new_tuple)])
                    high_error_original = np.multiply(high_error_original,target_value_ks_calculated_with_cantera_original[unique_reactions_original.index(new_tuple)])
                    low_error_original = np.exp(np.array(sigma_list_for_target_ks_original[unique_reactions_original.index(new_tuple)])*-1)
                    low_error_original = np.multiply(low_error_original,target_value_ks_calculated_with_cantera_original[unique_reactions_original.index(new_tuple)]) 
                    c, d = zip(*sorted(zip(target_value_temps_original[unique_reactions_original.index(new_tuple)],high_error_original)))  
                    cc, dd = zip(*sorted(zip(target_value_temps_original[unique_reactions_original.index(new_tuple)],low_error_original)))  
                                        
                else:

                    Temp_optimized,k_optimized = rate_constant_over_temperature_range_from_cantera(reaction,
                                                                    gas_optimized,
                                                                    initial_temperature=initial_temperature,
                                                                    final_temperature=final_temperature,
                                                                    pressure=1,
                                                                    #   conditions={'H2O2':0.003094,'O2':0.000556,'H2O':0.001113,'Ar':0.995237})
                                                                    conditions={'Ar':1})  
                    
                    Temp_original,k_original = rate_constant_over_temperature_range_from_cantera(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]),
                                                                        gas_original,
                                                                        initial_temperature=initial_temperature,
                                                                        final_temperature=final_temperature,
                                                                        pressure=1,
                                                                        #   conditions={'H2O2':0.003094,'O2':0.000556,'H2O':0.001113,'Ar':0.995237})
                                                                        conditions={'Ar':1})  
                    
                    # high_error_optimized = np.exp(sigma_list_for_target_ks_optimized[unique_reactions_optimized.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))])  
                    # high_error_optimized = np.multiply(high_error_optimized,target_value_ks_calculated_with_cantera_optimized[unique_reactions_optimized.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))])
                    # low_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[unique_reactions_optimized.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))])*-1)
                    # low_error_optimized = np.multiply(low_error_optimized,target_value_ks_calculated_with_cantera_optimized[unique_reactions_optimized.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))])  
                    # a, b = zip(*sorted(zip(target_value_temps_optimized[unique_reactions_optimized.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))],high_error_optimized)))              
                    # aa, bb = zip(*sorted(zip(target_value_temps_optimized[unique_reactions_optimized.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))],low_error_optimized)))  
                                        
                    high_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[i]))
                    high_error_optimized = np.multiply(high_error_optimized,target_value_ks_calculated_with_cantera_optimized[i])
                    low_error_optimized = np.exp(np.array(sigma_list_for_target_ks_optimized[i])*-1)
                    low_error_optimized = np.multiply(low_error_optimized,target_value_ks_calculated_with_cantera_optimized[i])      
                    a, b = zip(*sorted(zip(target_value_temps_optimized[i],high_error_optimized)))
                    aa, bb = zip(*sorted(zip(target_value_temps_optimized[i],low_error_optimized)))                                          
                                        
                                        
                    high_error_original = np.exp(sigma_list_for_target_ks_original[unique_reactions_original.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))])  
                    high_error_original = np.multiply(high_error_original,target_value_ks_calculated_with_cantera_original[unique_reactions_original.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))])
                    low_error_original = np.exp(np.array(sigma_list_for_target_ks_original[unique_reactions_original.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))])*-1)
                    low_error_original = np.multiply(low_error_original,target_value_ks_calculated_with_cantera_original[unique_reactions_original.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))])  
                    c, d = zip(*sorted(zip(target_value_temps_original[unique_reactions_original.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))],high_error_original)))              
                    cc, dd = zip(*sorted(zip(target_value_temps_original[unique_reactions_original.index(reaction_list_from_mechanism_original.index(reaction_list_from_mechanism[reaction]))],low_error_original)))  
                    
  
                       
                       
                       
                optimized_rate_constant_df['high optimized temp'] = pd.Series(a)  
                optimized_rate_constant_df['high optimized rate'] = pd.Series(b)  
                optimized_rate_constant_df['low optimized temp'] = pd.Series(aa)  
                optimized_rate_constant_df['low optimized rate'] = pd.Series(bb)  
                optimized_rate_constant_df['Temperature [K]'] = pd.Series(Temp_optimized)
                optimized_rate_constant_df['optimized rate'] = pd.Series(k_optimized)                    
                optimized_rate_constant_df.to_csv(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'_optimized.csv',index=False)                                             

                original_rate_constant_df['high original temp'] = pd.Series(c) 
                original_rate_constant_df['high original rate'] = pd.Series(d) 
                original_rate_constant_df['low original temp'] = pd.Series(cc) 
                original_rate_constant_df['low original rate'] = pd.Series(dd)    
                original_rate_constant_df['Temperature [K]'] = pd.Series(Temp_original)
                original_rate_constant_df['original rate'] = pd.Series(k_original)
                original_rate_constant_df.to_csv(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'_original.csv',index=False) 

                
                                       
                
                if np.inf in b:
                    pass
                elif np.nan in b:
                    pass
                elif min(b) < 1e-100:
                    pass
                elif max(b) > 1e+100:
                    pass
                elif type(reaction)==list:
                    if type(reaction[1]) == tuple:
                        plt.semilogy(a,b,'b--')
                    else:
                        pass
                else:
                    plt.semilogy(a,b,'b--')   
                            
                if np.inf in bb:
                    pass
                elif np.nan in bb:
                    pass
                elif min(bb) < 1e-100:
                    pass
                elif max(bb) > 1e+100:
                    pass      
                elif type(reaction)==list:
                    if type(reaction[1]) == tuple:
                        plt.semilogy(aa,bb,'b--')
                    else:
                        pass                          
                else:                         
                    plt.semilogy(aa,bb,'b--')                
                
                plt.semilogy(Temp_original,k_original,'r',label=r"$\it{A}$ $\it{priori}$ model")
                
                plt.semilogy(Temp_optimized,k_optimized,'b', label='MSI')            
                
                for j, tvt in enumerate(target_value_temps_optimized_for_plotting[i]):
                                        
                    # print(j)
                    # print(target_value_papers_optimized_for_plotting[i][j])
                    
                    if isinstance(target_value_refs_optimized_for_plotting[i][j],str):
                    
                        if target_value_refs_optimized_for_plotting[i][j] in marker_refs_list:
                            m = marker_list[marker_refs_list.index(target_value_refs_optimized_for_plotting[i][j])]
                            if j > 0:
                                if target_value_refs_optimized_for_plotting[i][j] == target_value_refs_optimized_for_plotting[i][j-1]:
                                    plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none')
                                else:
                                    plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none', label=target_value_refs_optimized_for_plotting[i][j])
                            else:
                                plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none', label=target_value_refs_optimized_for_plotting[i][j])
                        elif target_value_refs_optimized_for_plotting[i][j].lower() == 'nan':
                            plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], 'o', color='black', markerfacecolor='none')                        
                        else:
                            marker_refs_list.append(target_value_refs_optimized_for_plotting[i][j])
                            m = marker_list[marker_refs_list.index(target_value_refs_optimized_for_plotting[i][j])]
                            if j > 0:
                                if target_value_refs_optimized_for_plotting[i][j] == target_value_refs_optimized_for_plotting[i][j-1]:
                                    plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none')
                                else:
                                    plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none', label=target_value_refs_optimized_for_plotting[i][j])
                            else:
                                plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none', label=target_value_refs_optimized_for_plotting[i][j])
                    else:
                        if target_value_refs_optimized_for_plotting[i][j] in marker_refs_list:
                            m = marker_list[marker_refs_list.index(target_value_refs_optimized_for_plotting[i][j])]
                            if j > 0:
                                if target_value_refs_optimized_for_plotting[i][j] == target_value_refs_optimized_for_plotting[i][j-1]:
                                    plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none')
                                else:
                                    plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none', label=target_value_refs_optimized_for_plotting[i][j])
                            else:
                                plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none', label=target_value_refs_optimized_for_plotting[i][j])
                        elif math.isnan(target_value_refs_optimized_for_plotting[i][j]):
                            plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], 'o', color='black', markerfacecolor='none')
                        else:
                            marker_refs_list.append(target_value_refs_optimized_for_plotting[i][j])
                            m = marker_list[marker_refs_list.index(target_value_refs_optimized_for_plotting[i][j])]
                            if j > 0:
                                if target_value_refs_optimized_for_plotting[i][j] == target_value_refs_optimized_for_plotting[i][j-1]:
                                    plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none')
                                else:
                                    plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none', label=target_value_refs_optimized_for_plotting[i][j])
                            else:
                                plt.semilogy(target_value_temps_optimized_for_plotting[i][j],target_value_ks_optimized_for_plotting[i][j], m, color='black', markerfacecolor='none', label=target_value_refs_optimized_for_plotting[i][j])
                
                
                
                self.target_value_temps_original = target_value_temps_original
                self.unique_reactions_original  = unique_reactions_original
                self.S_matrix_k_target_values_extra = S_matrix_k_target_values_extra
                
                if temperature_range_to_plot_over is not None:
                    low_value_axis , high_value_axis = filter_range_for_plotting(Temp_optimized,k_optimized,Temp_original,k_original,
                                  a, b,
                                  aa, bb,
                                  c, d,
                                  cc, dd,
                                  low_temp=temperature_range_to_plot_over[0],high_temp=temperature_range_to_plot_over[1])

                
                plt.xlabel('Temperature [K]',fontsize=15)
                plt.ylabel(r'k',fontsize=15)
                plt.tick_params(which='both',direction='in')
                if type(reaction) == list: #Total Rates & Ratios of Rates
                    if type(reaction[1]) == tuple: #Total Rates
                        reactants = []
                        reactants.append(reaction_list_from_mechanism[reaction[0][0]].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]')
                        plt.title(reactants[0])
                        print(reactants[0])
                        plt.legend(ncol=2)
                        if temperature_range_to_plot_over is not None:
                            plt.xlim(temperature_range_to_plot_over[0],temperature_range_to_plot_over[1])
                            plt.ylim(low_value_axis,high_value_axis)
                        if self.pdf == True:
                            plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.pdf', bbox_inches='tight',dpi=self.dpi)                          
                        if self.png == True:
                            plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.png', bbox_inches='tight',dpi=self.dpi)                                 
                        if self.svg == True:
                            plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)
                    else: #Ratio of rates
                        numerator = reaction[0]
                        denominator = reaction[1]
                        reactants = []
                        for frac_reaction in [numerator, denominator]:
                            if type(frac_reaction) == list: #denominator is a sum (total or partial)
                                if len(frac_reaction) == 2: #denominator is a total sum
                                    reactants.append(reaction_list_from_mechanism[frac_reaction[0][0]].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]')
                                else:   #denominator is a partial sum  
                                    name =  '' 
                                    reax = []
                                    for sub_number in frac_reaction[0]:
                                        reax.append(reaction_list_from_mechanism[sub_number].rstrip().lstrip())
                                    for q,sub_number in enumerate(reax):
                                        if q == 0:
                                            name = name + sub_number
                                        else:
                                            name = name + ' [+] ' + sub_number
                                    reactants.append(name)
                            elif type(frac_reaction) == tuple:  #numerator is sum
                                name =  '' 
                                reax = []
                                for sub_number in frac_reaction:
                                    reax.append(reaction_list_from_mechanism[sub_number].rstrip().lstrip())
                                for q,sub_number in enumerate(reax):
                                    if q == 0:
                                        name = name + sub_number
                                    else:
                                        name = name + ' [+] ' + sub_number
                                reactants.append(name)
                            else: #numerator or denominator is a single rate
                                reactants.append(reaction_list_from_mechanism[frac_reaction])
                        plt.title(reactants[0] + ' [/] ' + reactants[1])
                        print(reactants[0] + ' [/] ' + reactants[1])
                        plt.legend(ncol=2)
                        if temperature_range_to_plot_over is not None:
                            plt.xlim(temperature_range_to_plot_over[0],temperature_range_to_plot_over[1])
                            plt.ylim(low_value_axis,high_value_axis)
                        if self.pdf == True:
                            plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.pdf', bbox_inches='tight',dpi=self.dpi)                          
                        if self.png == True:
                            plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.png', bbox_inches='tight',dpi=self.dpi)                                 
                        if self.svg == True:
                            plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)            
                elif type(reaction)==tuple: #Partial sum of rates w/ old total rate functionality
                    if len(reaction) == 2:
                        reactants = reaction_list_from_mechanism[reaction[0]].rstrip().lstrip() + ' [+] ' + reaction_list_from_mechanism[reaction[1]].rstrip().lstrip()
                    else:
                        ttl_reaction_string=[]
                        for sub_number in reaction:
                            ttl_reaction_string.append(reaction_list_from_mechanism[sub_number])
                        reactants = ttl_reaction_string[0].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]' 
                    plt.title(reactants)
                    print(reactants)
                    plt.legend(ncol=2)
                    if temperature_range_to_plot_over is not None:
                        plt.xlim(temperature_range_to_plot_over[0],temperature_range_to_plot_over[1])
                        plt.ylim(low_value_axis,high_value_axis)
                    if self.pdf == True:
                        plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.pdf', bbox_inches='tight',dpi=self.dpi)       
                    if self.png == True:
                        plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.png', bbox_inches='tight',dpi=self.dpi)            
                    if self.svg == True:
                        plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)                                                              
                else: #Single Rates
                    if temperature_range_to_plot_over is not None:
                        plt.xlim(temperature_range_to_plot_over[0],temperature_range_to_plot_over[1])
                        plt.ylim(low_value_axis,high_value_axis)
                    plt.title(reaction_list_from_mechanism[reaction])
                    print(reaction_list_from_mechanism[reaction])
                    plt.legend(ncol=2)

                    if self.pdf == True:
                        plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.pdf', bbox_inches='tight',dpi=self.dpi)                             
                    if self.png == True:
                        plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.png', bbox_inches='tight',dpi=self.dpi)                              
                    if self.svg == True:
                        plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True)   
                        
                self.rate_loop.update()                         

    def plotting_uncertainty_weighted_sens_rate_constant(self,top_sensitivity=10,
                                                         reactions_for_legend = [],
                                                         observable_list_for_legend_csv_path=None):
            
        print('\n')
        print('--------------------------------------------------------------------------')
        print('Rate Constant UWSA Plots')
        print('--------------------------------------------------------------------------')

        
        if observable_list_for_legend_csv_path != None:
        
            df = pd.read_csv(observable_list_for_legend_csv_path)
            column_name_list = df.columns.tolist()
            
            
            included_uncertainty_flag=False
            if 'uncertainty' in column_name_list:
                uncertainty_from_csv = df['uncertainty'].to_numpy()
                uncertainty_from_csv = uncertainty_from_csv.reshape((uncertainty_from_csv.shape[0],1))
                included_uncertainty_flag=True
                
        self.target_value_temps_original
        
        self.unique_reactions_original
        #print(len(self.target_value_temps_original[0]))
        S_matrix_copy = copy.deepcopy(self.S_matrix_k_target_values_extra)
        # self.get_observables_list()
        # sum_arrhenius_observables = self.get_sum_arrhenius_observables()
        
        length_of_temperature_vectors =[[] for x in range(len(self.target_value_temps_original))]
        for i,temperature_list in enumerate(self.target_value_temps_original):
            length_of_temperature_vectors[i].append(len(temperature_list))
        
        #print(observables_list)

        # if included_uncertainty_flag==True:
        #     Sig = uncertainty_from_csv
        # else:
        Sig = self.short_sigma
        
        # print(len(Sig))
        
        for pp  in range(np.shape(S_matrix_copy)[1]):

            S_matrix_copy[:,pp] *=Sig[pp]

        sensitivitys =[[] for x in range(len(self.target_value_temps_original))]
        topSensitivities = [[] for x in range(len(self.target_value_temps_original))]   
        start=0
        stop = 0
        
        for x in range(len(length_of_temperature_vectors)):
            for y in range(len(length_of_temperature_vectors[x])):
    
                stop = length_of_temperature_vectors[x][y] + start

                
                temp = S_matrix_copy[int(start):int(stop),:]
                sort_s= pd.DataFrame(temp).reindex(pd.DataFrame(temp).abs().max().sort_values(ascending=False).index, axis=1)
                # cc=pd.DataFrame(sort_s).iloc[:,:top_sensitivity]
                cc=pd.DataFrame(sort_s).iloc[:,:]
                top_five_reactions=cc.columns.values.tolist()
                topSensitivities[x].append(top_five_reactions)
                #ccn=pd.DataFrame(cc).as_matrix()
                ccn=pd.DataFrame(cc).to_numpy()

                sensitivitys[x].append(ccn)           
                start = start + length_of_temperature_vectors[x][y]
                
               
        list_of_reaction_strings = []
        gas = ct.Solution(self.nominal_cti)
        reaction_equations_list = gas.reaction_equations()
        for reaction_number in self.unique_reactions_original:

            if type(reaction_number) == list: #Total rate or Ratio
                if type(reaction_number[1]) == tuple: #Total rate
                    list_of_reaction_strings.append(reaction_equations_list[reaction_number[0][0]].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]')
                else:   #Ratio of rates
                    numerator = reaction_number[0]
                    denominator = reaction_number[1]
                    temp_list = []
                    for frac_reaction in [numerator, denominator]:
                        if type (frac_reaction) == list: # denominator is a sum (total or partial)
                            if len(frac_reaction) == 2: # denominator is a total sum
                                temp_list.append(reaction_equations_list[frac_reaction[0][0]].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]')
                            else:   #denominator is a partial sum
                                name = ''
                                reax = []
                                for sub_number in frac_reaction[0]:
                                    reax.append(reaction_equations_list[sub_number].rstrip().lstrip())
                                for q,sub_number in enumerate(reax):
                                    if q == 0:
                                        name = name + sub_number
                                    else:
                                        name = name + ' [+] ' + sub_number
                                temp_list.append(name)
                        elif type(frac_reaction) == tuple: # numerator is a partial sum of rates
                            temp_tuple = []
                            if len(frac_reaction) == 2:
                                total_reaction_string = reaction_equations_list[frac_reaction[0]].rstrip().lstrip() + ' [+] ' + reaction_equations_list[frac_reaction[1]].rstrip().lstrip()
                            else:
                                for sub_reaction in frac_reaction:
                                    temp_tuple.append(reaction_equations_list[sub_reaction])
                                total_reaction_string = temp_tuple[0].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]' 

                            temp_list.append(total_reaction_string)
                        else:   #numerator or denominator is a single rate
                            temp_list.append(reaction_equations_list[frac_reaction])

                    list_of_reaction_strings.append(temp_list[0] + ' [/] ' + temp_list[1])            
            
            elif type(reaction_number) == tuple: #Sum of rates w/ old totaL rate functionality
                temp_tuple = []

                if len(reaction_number) == 2:
                    total_reaction_string = reaction_equations_list[reaction_number[0]].rstrip().lstrip() + ' [+] ' + reaction_equations_list[reaction_number[1]].rstrip().lstrip()
                else:
                    for sub_reaction in reaction_number:
                        temp_tuple.append(reaction_equations_list[sub_reaction])
                    total_reaction_string = temp_tuple[0].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]' 

                list_of_reaction_strings.append(total_reaction_string)
                
            else: #Single Rate
                
                list_of_reaction_strings.append(reaction_equations_list[reaction_number])
        

        self.rate_UWSA_loop = self.manager.counter(total=len(self.unique_reactions_original), desc='Rate Constant UWSA Plots:', unit='plots', color='green') 
        
        for i,reaction in enumerate(self.unique_reactions_original):
            
            
            #plt.figure()
            
            fig = plt.figure()
            # plt.subplot(3,1,1)
            #fig = plt.figure(figsize=(20, 10))
            colors=['k','r','b','g','m']
            line_type=['-.','-','--',(0,(5,10)),':']
            marker_list = [None,None,None,None,None]
            #line_type  = ['--', '-.', '-', ':','-']

            UWSA_Rate_Constant_df = pd.DataFrame()
            UWSA_Rate_Constant_df['T'] = pd.Series(self.target_value_temps_original[i])     
            
            for ccc, top_columns in enumerate(topSensitivities[i][0]):
                
                if ccc < top_sensitivity:
                    c, d = zip(*sorted(zip(self.target_value_temps_original[i],sensitivitys[i][0][:,ccc])))  

                    #plt.plot(self.target_value_temps_original[i],sensitivitys[i][0][:,c],label = observables_list[top_columns] +'_'+str(Sig[top_columns])) 
                    if observable_list_for_legend_csv_path != None:
                        df = pd.read_csv(observable_list_for_legend_csv_path)
                        observable_list_for_legend = df['optimization_variable'].tolist()
                        #print(observable_list_for_legend)
                        plt.plot(c,d,marker=marker_list[ccc],color=colors[ccc],linestyle=line_type[ccc],label = observable_list_for_legend[top_columns] +' '+str(Sig[top_columns])) 
                    else:
                        observables_list_for_legend = self.active_parameters
                        plt.plot(c,d,label = observables_list_for_legend[top_columns] +' '+str(Sig[top_columns])) 
                
               
                UWSA_Rate_Constant_df[observables_list_for_legend[top_columns] +' '+str(Sig[top_columns])] = pd.Series(sensitivitys[i][0][:,ccc])   
                
            UWSA_Rate_Constant_df.to_csv(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'_UWSA.csv',index=False)   
            
            if bool(reactions_for_legend):
                print(str(reactions_for_legend[i]))
                plt.title(str(reactions_for_legend[i]))
            else:
                print(str(list_of_reaction_strings[i]))
                plt.title(str(list_of_reaction_strings[i]))
            plt.xlabel('Temperature [K]')
            plt.ylabel(r'$\frac{\partial(\rm k)}{\partial(\rm x_j)} \rm \sigma_j$')
            # plt.ylabel(r'$\frac{\partial(k)}{\partial('+str(list_of_experiment_observables[plot_number])+')} \sigma_j$')
            plt.tick_params(direction='in')
            # plt.xticks(fontsize= 10)
            # plt.yticks(fontsize= 10)
            # plt.legend(fontsize= 8)
            plt.legend(ncol=1, loc='upper left',bbox_to_anchor=(1,1))

            # plt.savefig(self.out_path+'/'+'UWSA_'+str(list_of_reaction_strings[i])+'.pdf', bbox_inches='tight')
            # plt.savefig(self.out_path+'/'+'UWSA_'+str(list_of_reaction_strings[i])+'.png', bbox_inches='tight')
            # plt.savefig(self.out_path+'/'+'UWSA_'+str(list_of_reaction_strings[i])+'.svg', bbox_inches='tight',transparent=True)
            
            if self.pdf == True:
                plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'_UWSA'+'.pdf', bbox_inches='tight',dpi=self.dpi)
            if self.png == True:
                plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'_UWSA'+'.png', bbox_inches='tight',dpi=self.dpi)
            if self.svg == True:
                plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'_UWSA'+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True) 
            
            self.rate_UWSA_loop.update()                            

            # for plot_number in range(number_of_observables_in_simulation):
            #     for c,top_columns in enumerate(top_sensitivity_single_exp[plot_number]):
            #         plt.subplot(number_of_observables_in_simulation,1,plot_number+1)
            #         if plot_number==0:
            #             plt.title('Experiment_'+str(experiment_number+1))
            #         plt.plot(time_profiles[plot_number],sensitivities[plot_number][:,c],label = observables_list[top_columns] +'_'+str(sigma_list[top_columns])) 
            #         plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.5)
            #         plt.ylabel(list_of_experiment_observables[plot_number])
            #         top,bottom = plt.ylim()
            #         left,right = plt.xlim()
            #         plt.legend(ncol=5, loc='upper left',bbox_to_anchor=(-.5,-.3))                       
            
    def plotting_Sdx_rate_constant(self,top_sensitivity=10,
                                                         reactions_for_legend = [],
                                                         observable_list_for_legend_csv_path=None):
             
        print('\n')
        print('--------------------------------------------------------------------------')
        print('Rate Constant Sdx Plots')
        print('--------------------------------------------------------------------------')
            
        if observable_list_for_legend_csv_path != None:
        
            df = pd.read_csv(observable_list_for_legend_csv_path)
            column_name_list = df.columns.tolist()
            
            included_uncertainty_flag=False
            if 'uncertainty' in column_name_list:
                uncertainty_from_csv = df['uncertainty'].to_numpy()
                uncertainty_from_csv = uncertainty_from_csv.reshape((uncertainty_from_csv.shape[0],1))
                included_uncertainty_flag=True
                
        self.target_value_temps_original
        
        self.unique_reactions_original
        S_matrix_copy = copy.deepcopy(self.S_matrix_k_target_values_extra)
        flat_list = [item for sublist in self.simulation_lengths_of_experimental_data for item in sublist]
        length = sum(flat_list)
        observables_list = self.target_parameters[length:]
        # observables_list = self.sum_arrhenius_observables[length:]
        
        reactions_in_cti_file = self.exp_dict_list_original[0]['simulation'].processor.solution.reaction_equations()
        flatten = lambda *n: (e for a in n
            for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))        
        flattened_master_equation_reaction_list = list(flatten(self.master_equation_reactions))     
        A_n_Ea_length = int((len(reactions_in_cti_file) - len(flattened_master_equation_reaction_list))*3)
                
        
        
        length_of_temperature_vectors =[[] for x in range(len(self.target_value_temps_original))]
        for i,temperature_list in enumerate(self.target_value_temps_original):
            length_of_temperature_vectors[i].append(len(temperature_list))
        
        
        Sig = self.short_sigma
        X = self.X
        
        # print(len(Sig))
        # print(len(X))
        
        for pp  in range(np.shape(S_matrix_copy)[1]):

            S_matrix_copy[:,pp] *=X[pp]

        sensitivitys =[[] for x in range(len(self.target_value_temps_original))]
        topSensitivities = [[] for x in range(len(self.target_value_temps_original))]   
        summed_observable_list = []
        # num_rxn = []

        start=0
        stop = 0
        
        for x in range(len(length_of_temperature_vectors)):
            for y in range(len(length_of_temperature_vectors[x])):

                stop = length_of_temperature_vectors[x][y] + start
                temp = S_matrix_copy[int(start):int(stop),:]

                temp_df = pd.DataFrame(temp, columns=observables_list[:len(temp.T)])
                
                k_target_length = len(self.k_target_value_S_matrix)
                num_rxns = len(observables_list[:-k_target_length])/3
                
                A_temp_df = temp_df.iloc[:,:int(A_n_Ea_length/3)]
                n_temp_df = temp_df.iloc[:,int(A_n_Ea_length/3):int(2*A_n_Ea_length/3)]
                Ea_temp_df = temp_df.iloc[:,int(2*A_n_Ea_length/3):int(3*A_n_Ea_length/3)]

                if len(A_temp_df.T) != len(n_temp_df.T):
                    print('Length of A_temp_df and n_temp_df not equal')
                if len(A_temp_df.T) != len(Ea_temp_df.T):
                    print('Length of A_temp_df and Ea_temp_df not equal')
                if len(n_temp_df.T) != len(Ea_temp_df.T):
                    print('Length of n_temp_df and Ea_temp_df not equal')                                        

                # sum_temp_df = pd.DataFrame()

                # for i, col in enumerate(A_temp_df.columns):
                #     sum_temp_df['k_'+str(i)] = A_temp_df['A_'+str(i)] + n_temp_df['n_'+str(i)] + Ea_temp_df['Ea_'+str(i)] 
                
                # new_temp_df = pd.concat([sum_temp_df, temp_df.iloc[:,len(A_temp_df.T)+len(n_temp_df.T)+len(Ea_temp_df.T):]], axis=1)
                # sort_s= new_temp_df.reindex(new_temp_df.abs().max().sort_values(ascending=False).index, axis=1)


                sum_temp_list = []
                col_reactions = []
                for i, col in enumerate(A_temp_df.columns):
                    col_reactions.append(col[2:])
                    sum_temp_list.append(list(A_temp_df.iloc[:, i] + n_temp_df.iloc[:, i]+ Ea_temp_df.iloc[:, i]))
                    
                new_temp_array = np.concatenate((np.array(sum_temp_list),temp_df.iloc[:,A_n_Ea_length:].to_numpy().T))
                new_temp_df = pd.DataFrame(new_temp_array.T)
                
                new_temp_df_columns = col_reactions + list(temp_df.iloc[:,A_n_Ea_length:].columns)
                # new_temp_df = pd.concat([sum_temp_df, temp_df.iloc[:,A_n_Ea_length:]], axis=1)
                # self.summed_observable_list = new_temp_df.columns
                sort_s= new_temp_df.reindex(new_temp_df.abs().max().sort_values(ascending=False).index, axis=1)
                sort_s.columns = [new_temp_df_columns[i] for i in list(sort_s.columns)]
                
                
                
                # sort_s= pd.DataFrame(temp).reindex(pd.DataFrame(temp).abs().max().sort_values(ascending=False).index, axis=1)

                # cc=pd.DataFrame(sort_s).iloc[:,:top_sensitivity]
                cc=pd.DataFrame(sort_s).iloc[:,:]
                top_five_reactions=cc.columns.values.tolist()
                topSensitivities[x].append(top_five_reactions)
                #ccn=pd.DataFrame(cc).as_matrix()
                ccn=pd.DataFrame(cc).to_numpy()

                sensitivitys[x].append(ccn)           
                start = start + length_of_temperature_vectors[x][y]

                summed_observable_list.append(new_temp_df_columns)
        num_rxn = len(A_temp_df.T)
                
               
        list_of_reaction_strings = []
        gas = ct.Solution(self.nominal_cti)
        reaction_equations_list = gas.reaction_equations()
        for reaction_number in self.unique_reactions_original:

            if type(reaction_number) == list: # Total sum or ratio of rates
                if type(reaction_number[1]) == tuple: #Total rate
                    list_of_reaction_strings.append(reaction_equations_list[reaction_number[0][0]].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]')
                else:   #Ratio of rates
                    numerator = reaction_number[0]
                    denominator = reaction_number[1]
                    temp_list = []
                    for frac_reaction in [numerator, denominator]:
                        if type (frac_reaction) == list: # denominator is a sum (total or partial)
                            if len(frac_reaction) == 2: # denominator is a total sum
                                temp_list.append(reaction_equations_list[frac_reaction[0][0]].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]')
                            else:   #denominator is a partial sum
                                name = ''
                                reax = []
                                for sub_number in frac_reaction[0]:
                                    reax.append(reaction_equations_list[sub_number].rstrip().lstrip())
                                for q,sub_number in enumerate(reax):
                                    if q == 0:
                                        name = name + sub_number
                                    else:
                                        name = name + ' [+] ' + sub_number
                                temp_list.append(name)
                        elif type(frac_reaction) == tuple: # numerator is a partial sum of rates
                            temp_tuple = []
                            if len(frac_reaction) == 2:
                                total_reaction_string = reaction_equations_list[frac_reaction[0]].rstrip().lstrip() + ' [+] ' + reaction_equations_list[frac_reaction[1]].rstrip().lstrip()
                            else:
                                for sub_reaction in frac_reaction:
                                    temp_tuple.append(reaction_equations_list[sub_reaction])
                                total_reaction_string = temp_tuple[0].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]' 

                            temp_list.append(total_reaction_string)
                        else: # numerator or denominator is a single rate
                            temp_list.append(reaction_equations_list[frac_reaction])
                    list_of_reaction_strings.append(temp_list[0] + ' [/] ' + temp_list[1])            
            elif type(reaction_number) == tuple: # Partial sum of rate w/ old total rate functionality
                temp_tuple = []

                if len(reaction_number) == 2:
                    total_reaction_string = reaction_equations_list[reaction_number[0]].rstrip().lstrip() + ' [+] ' + reaction_equations_list[reaction_number[1]].rstrip().lstrip()
                else:
                    for sub_reaction in reaction_number:
                        temp_tuple.append(reaction_equations_list[sub_reaction])
                    total_reaction_string = temp_tuple[0].split('<=>')[0].rstrip().lstrip().replace(' (+M)','') + ' <=> [*]' 

                list_of_reaction_strings.append(total_reaction_string)
                
            else: # Single Rate
                
                list_of_reaction_strings.append(reaction_equations_list[reaction_number])
        
                
        self.rate_Sdx_loop = self.manager.counter(total=len(self.unique_reactions_original), desc='Rate Constant Sdx Plots:', unit='plots', color='green') 
        
        for i,reaction in enumerate(self.unique_reactions_original):
            
            #plt.figure()
            
            fig = plt.figure()
            # plt.subplot(3,1,1)
            #fig = plt.figure(figsize=(20, 10))
            colors=['k','r','b','g','m']
            line_type=['-.','-','--',(0,(5,10)),':']
            marker_list = [None,None,None,None,None]
            #line_type  = ['--', '-.', '-', ':','-']

            Sdx_Rate_Constant_df = pd.DataFrame()
            Sdx_Rate_Constant_df['T'] = pd.Series(self.target_value_temps_original[i])     
            
            for ccc, top_columns in enumerate(topSensitivities[i][0]):

                top_columns_index = summed_observable_list[i].index(top_columns)                
                
                if ccc < top_sensitivity:
                    c, d = zip(*sorted(zip(self.target_value_temps_original[i],sensitivitys[i][0][:,ccc])))  

                    # sum_arrhenius_observables = self.get_sum_arrhenius_observables()
                    # if top_columns.split('_')[0] == 'k':
                    if top_columns_index < num_rxn:
                        X_A = X[top_columns_index][0]
                        X_n = X[top_columns_index+num_rxn][0]
                        X_Ea = X[top_columns_index+2*num_rxn][0]
                        summed_X = X_A + X_n + X_Ea
                        sigma_A = Sig[top_columns_index][0]
                        sigma_n = Sig[top_columns_index+num_rxn][0]
                        sigma_Ea = Sig[top_columns_index+2*num_rxn][0]

                        plt.plot(c,d,
                        label = summed_observable_list[i][top_columns_index] +' [A:'+str(sigma_A)+', n:'+str(sigma_n)+', Ea:'+str(sigma_Ea)+', '+str(summed_X)+']') 
                    else:
                        plt.plot(c,d,
                        label = summed_observable_list[i][top_columns_index] +' ['+str(Sig[top_columns_index+2*num_rxn][0])+', '+str(X[top_columns_index+2*num_rxn][0])+']') 

                    # plt.plot(c,d,label = observables_list_for_legend[top_columns]+' ['+str(Sig[top_columns][0])+', '+str(X[top_columns][0])+']') 
                
                if top_columns.split('_')[0] == 'k':
                    X_A = X[top_columns_index][0]
                    X_n = X[top_columns_index+num_rxn][0]
                    X_Ea = X[top_columns_index+2*num_rxn][0]
                    summed_X = X_A + X_n + X_Ea
                    sigma_A = Sig[top_columns_index][0]
                    sigma_n = Sig[top_columns_index+num_rxn][0]
                    sigma_Ea = Sig[top_columns_index+2*num_rxn][0]

                    Sdx_Rate_Constant_df[summed_observable_list[i][top_columns_index] +' [A:'+str(sigma_A)+', n:'+str(sigma_n)+', Ea:'+str(sigma_Ea)+', '+str(summed_X)+']'] = pd.Series(sensitivitys[i][0][:,ccc])   
                else:           
                    Sdx_Rate_Constant_df[summed_observable_list[i][top_columns_index] +' ['+str(Sig[top_columns_index+2*num_rxn][0])+', '+str(X[top_columns_index+2*num_rxn][0])+']'] = pd.Series(sensitivitys[i][0][:,ccc])   
                
            Sdx_Rate_Constant_df.to_csv(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'_Sdx.csv',index=False)   
            
            if bool(reactions_for_legend):
                print(str(reactions_for_legend[i]))
                plt.title(str(reactions_for_legend[i]))
            else:
                print(str(list_of_reaction_strings[i]))
                plt.title(str(list_of_reaction_strings[i]))
            plt.xlabel('Temperature [K]')
            plt.ylabel(r'$\frac{\partial(\rm k)}{\partial(\rm x_j)} \rm \Delta x_j$')
            # plt.ylabel(r'$\frac{\partial(k)}{\partial('+str(list_of_experiment_observables[plot_number])+')} \sigma_j$')
            plt.tick_params(direction='in')
            # plt.xticks(fontsize= 10)
            # plt.yticks(fontsize= 10)
            # plt.legend(fontsize= 8)
            plt.legend(ncol=1, loc='upper left',bbox_to_anchor=(1,1))

            # plt.savefig(self.out_path+'/'+'Sdx_'+str(list_of_reaction_strings[i])+'.pdf', bbox_inches='tight')
            # plt.savefig(self.out_path+'/'+'Sdx_'+str(list_of_reaction_strings[i])+'.png', bbox_inches='tight')
            # plt.savefig(self.out_path+'/'+'Sdx_'+str(list_of_reaction_strings[i])+'.svg', bbox_inches='tight',transparent=True)
            
            if self.pdf == True:
                plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'_Sdx'+'.pdf', bbox_inches='tight',dpi=self.dpi)
            if self.png == True:
                plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'_Sdx'+'.png', bbox_inches='tight',dpi=self.dpi)
            if self.svg == True:
                plt.savefig(self.out_path+'/'+'Rate_Constant_'+str(i+1)+'_Sdx'+'.svg', bbox_inches='tight',dpi=self.dpi,transparent=True) 
                                        
            self.rate_Sdx_loop.update()
            
            # for plot_number in range(number_of_observables_in_simulation):
            #     for c,top_columns in enumerate(top_sensitivity_single_exp[plot_number]):
            #         plt.subplot(number_of_observables_in_simulation,1,plot_number+1)
            #         if plot_number==0:
            #             plt.title('Experiment_'+str(experiment_number+1))
            #         plt.plot(time_profiles[plot_number],sensitivities[plot_number][:,c],label = observables_list[top_columns] +'_'+str(sigma_list[top_columns])) 
            #         plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.5)
            #         plt.ylabel(list_of_experiment_observables[plot_number])
            #         top,bottom = plt.ylim()
            #         left,right = plt.xlim()
            #         plt.legend(ncol=5, loc='upper left',bbox_to_anchor=(-.5,-.3))                   
            
    def merge_observable_pdfs(self):
        
        print('\n')
        print('--------------------------------------------------------------------------')
        print('Merging pdfs for obervables')
        print('--------------------------------------------------------------------------')

        
        Experiment_pdfs = set(glob.glob(os.path.join(self.out_path, "Experiment_[0-9]_[!US]*.pdf")) 
                            + glob.glob(os.path.join(self.out_path, "Experiment_[0-9][0-9]_[!US]*.pdf")) 
                            + glob.glob(os.path.join(self.out_path, "Experiment_[0-9][0-9][0-9]_[!US]*.pdf")))
        Experiment_pdfs = natsort.natsorted(Experiment_pdfs)
        merger = PdfMerger()
        for pdf in Experiment_pdfs:
            merger.append(pdf)
        merger.write(os.path.join(self.out_path, "Experiment.pdf"))
        merger.close()
        for pdf in Experiment_pdfs:
            os.remove(pdf)

        Experiment_UWSA_pdfs = set(glob.glob(os.path.join(self.out_path, "Experiment_[0-9]_[U]*.pdf")) 
                            + glob.glob(os.path.join(self.out_path, "Experiment_[0-9][0-9]_[U]*.pdf")) 
                            + glob.glob(os.path.join(self.out_path, "Experiment_[0-9][0-9][0-9]_[U]*.pdf")))
        Experiment_UWSA_pdfs = natsort.natsorted(Experiment_UWSA_pdfs)
        merger = PdfMerger()
        for pdf in Experiment_UWSA_pdfs:
            merger.append(pdf)
        merger.write(os.path.join(self.out_path, "Experiment_UWSA.pdf"))
        merger.close()
        for pdf in Experiment_UWSA_pdfs:
            os.remove(pdf)

        Experiment_Sdx_pdfs = set(glob.glob(os.path.join(self.out_path, "Experiment_[0-9]_[Sdx]*.pdf")) 
                            + glob.glob(os.path.join(self.out_path, "Experiment_[0-9][0-9]_[Sdx]*.pdf")) 
                            + glob.glob(os.path.join(self.out_path, "Experiment_[0-9][0-9][0-9]_[Sdx]*.pdf")))
        Experiment_Sdx_pdfs = natsort.natsorted(Experiment_Sdx_pdfs)
        merger = PdfMerger()
        for pdf in Experiment_Sdx_pdfs:
            merger.append(pdf)
        merger.write(os.path.join(self.out_path, "Experiment_Sdx.pdf"))
        merger.close()
        for pdf in Experiment_Sdx_pdfs:
            os.remove(pdf)        
            
    def merge_rate_constant_pdfs(self):
        
        print('\n')
        print('--------------------------------------------------------------------------')
        print('Merging pdfs for rate constants')
        print('--------------------------------------------------------------------------')

        
        Rate_Constant_pdfs = set(glob.glob(os.path.join(self.out_path, "Rate_Constant_[0-9].pdf"))
                                + glob.glob(os.path.join(self.out_path, "Rate_Constant_[0-9][0-9].pdf")))
        Rate_Constant_pdfs = natsort.natsorted(Rate_Constant_pdfs)
        merger = PdfMerger()
        for pdf in Rate_Constant_pdfs:
            merger.append(pdf)
        merger.write(os.path.join(self.out_path, "Rate_Constant.pdf"))
        merger.close()
        for pdf in Rate_Constant_pdfs:
            os.remove(pdf)
            
        Rate_Constant_UWSA_pdfs = set(glob.glob(os.path.join(self.out_path, "Rate_Constant_[0-9]_[U]*.pdf"))
                                    + glob.glob(os.path.join(self.out_path, "Rate_Constant_[0-9][0-9]_[U]*.pdf")) )
        Rate_Constant_UWSA_pdfs = natsort.natsorted(Rate_Constant_UWSA_pdfs)
        merger = PdfMerger()
        for pdf in Rate_Constant_UWSA_pdfs:
            merger.append(pdf)
        merger.write(os.path.join(self.out_path, "Rate_Constant_UWSA.pdf"))
        merger.close()
        for pdf in Rate_Constant_UWSA_pdfs:
            os.remove(pdf)
            
        Rate_Constant_Sdx_pdfs = set(glob.glob(os.path.join(self.out_path, "Rate_Constant_[0-9]_[Sdx]*.pdf"))
                                    + glob.glob(os.path.join(self.out_path, "Rate_Constant_[0-9][0-9]_[Sdx]*.pdf")) )
        Rate_Constant_Sdx_pdfs = natsort.natsorted(Rate_Constant_Sdx_pdfs)
        merger = PdfMerger()
        for pdf in Rate_Constant_Sdx_pdfs:
            merger.append(pdf)
        merger.write(os.path.join(self.out_path, "Rate_Constant_Sdx.pdf"))
        merger.close()
        for pdf in Rate_Constant_Sdx_pdfs:
            os.remove(pdf)
            
    def plotting_convergence(self, convergence_sorted, number_of_plots=50):
        
        print('\n')
        print('--------------------------------------------------------------------------')
        print('Plotting convergence')
        print('--------------------------------------------------------------------------')
        
        if number_of_plots > len(convergence_sorted):
            for i in range(len(convergence_sorted)):
                plt.figure()
                plt.scatter(np.arange(1,self.number_of_iterations+1), convergence_sorted.iloc[i])
                plt.xlabel('Number of Iterations')
                plt.ylabel(r'ln($X$)')
                plt.title(convergence_sorted.index[i])
                plt.savefig(os.path.join(self.matrix_path,'convergence_plot_' + str(i) + '.pdf'), bbox_inches='tight', dpi=1)
        else:
            for i in range(number_of_plots):
                plt.figure()
                plt.scatter(np.arange(1,self.number_of_iterations+1), convergence_sorted.iloc[i])
                plt.xlabel('Number of Iterations')
                plt.ylabel(r'ln($X$)')
                plt.title(convergence_sorted.index[i])
                plt.savefig(os.path.join(self.matrix_path,'convergence_plot_' + str(i) + '.pdf'), bbox_inches='tight', dpi=1)
            
        convergece_plots_pdfs = set(glob.glob(os.path.join(self.matrix_path, "convergence_plot*.pdf")))
        convergece_plots_pdfs = natsort.natsorted(convergece_plots_pdfs)
        merger = PdfMerger()
        for pdf in convergece_plots_pdfs:
            merger.append(pdf)
        merger.write(os.path.join(self.matrix_path, "convergence_plots.pdf"))
        merger.close()
        for i, convergence_plot_file in enumerate(list(set(glob.glob(os.path.join(self.matrix_path, "convergence_plot_*.pdf"))))):
            os.remove(convergence_plot_file)
            