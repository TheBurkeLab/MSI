#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:02:43 2020

@author: carly
"""

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
import os 
import re






class Plotting_for_2020_eastern_states(object):
    def __init__(self,S_matrix,
                  s_matrix,
                  Y_matrix,
                  y_matrix,
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
                  target_value_rate_constant_csv='',
                  target_value_rate_constant_csv_extra_values = '',
                  k_target_value_S_matrix = None,
                  k_target_values='Off',
                  working_directory='',
                  sigma_uncertainty_weighted_sensitivity_csv='',
                  simulation_run=None,
                  shock_tube_instance = None,
                  cheby_sensitivity_dict = None,
                  mapped_to_alpha_full_simulation=None,
                  optimized_cti_file='',
                  original_cti_file='',
                  sigma_ones=False,
                  T_min=200,
                  T_max=3000,
                  P_min=1013.25,
                  P_max=1.013e+6):
        self.S_matrix = S_matrix
        self.s_matrix = s_matrix
        self.Y_matrix = Y_matrix
        self.y_matrix = y_matrix
        self.z_matrix = z_matrix
        self.X = X
        self.sigma = sigma
        #self.sigma = sigma
        self.covarience=covarience
        self.original_covariance=original_covariance
        #original
        self.S_matrix_original=S_matrix_original
        self.exp_dict_list_optimized = exp_dict_list_optimized
        self.exp_dict_list_original = exp_dict_list_original
        self.parsed_yaml_list = parsed_yaml_list
        self.target_value_rate_constant_csv = target_value_rate_constant_csv
        self.k_target_value_S_matrix = k_target_value_S_matrix
        self.Ydf = Ydf
        self.k_target_values=k_target_values
        self.target_value_rate_constant_csv_extra_values = target_value_rate_constant_csv_extra_values
        self.working_directory = working_directory
        self.sigma_uncertainty_weighted_sensitivity_csv  = sigma_uncertainty_weighted_sensitivity_csv
        self.simulation_run = simulation_run
        self.shock_tube_instance = shock_tube_instance
        self.cheby_sensitivity_dict=cheby_sensitivity_dict
        self.mapped_to_alpha_full_simulation = mapped_to_alpha_full_simulation,
        self.new_cti=optimized_cti_file
        self.nominal_cti=original_cti_file
        self.sigma_ones = sigma_ones
        self.T_min = T_min
        self.T_max = T_max
        self.P_min = P_min
        self.P_max = P_max
        
 #fix all the indexing to have a captial or lowercase time situation or add the module that lets you do either to all the scripts  

    def lengths_of_experimental_data(self):
        simulation_lengths_of_experimental_data = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            length_of_experimental_data=[]
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                if observable in exp['mole_fraction_observables']:
                    length_of_experimental_data.append(exp['experimental_data'][observable_counter]['Time'].shape[0])
                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:
                    length_of_experimental_data.append(exp['experimental_data'][observable_counter]['Time'].shape[0])
                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list[i]['absorbanceCsvWavelengths']
                absorbance_wl=0
                for k,wl in enumerate(wavelengths):
                    length_of_experimental_data.append(exp['absorbance_experimental_data'][k]['time'].shape[0])
                    absorbance_wl+=1
            else:
                absorbance_wl=0
                    
            simulation_lengths_of_experimental_data.append(length_of_experimental_data)
            
                    
        self.simulation_lengths_of_experimental_data=simulation_lengths_of_experimental_data
        
        return observable_counter+absorbance_wl,length_of_experimental_data
                    

        
    def calculating_sigmas(self,S_matrix,covarience):  
        sigmas =[[] for x in range(len(self.simulation_lengths_of_experimental_data))]
                 
        counter=0
        for x in range(len(self.simulation_lengths_of_experimental_data)):
            for y in range(len(self.simulation_lengths_of_experimental_data[x])):
                temp=[]
                for z in np.arange(counter,(self.simulation_lengths_of_experimental_data[x][y]+counter)):       
                    SC = np.dot(S_matrix[z,:],covarience)
                    sigma = np.dot(SC,np.transpose(S_matrix[z,:]))
                    test = sigma
                    sigma = np.sqrt(sigma)
                    temp.append(sigma)
                temp = np.array(temp)            
                sigmas[x].append(temp)
        
                
                counter = counter + self.simulation_lengths_of_experimental_data[x][y]
        
        return sigmas, test
    
    
    
    def plotting_3_figre_observables(self,experiment_number_want_to_plot,sigmas_original=[],sigmas_optimized=[]):
        
        
        
        for i,exp in enumerate(self.exp_dict_list_optimized):
            if i==experiment_number_want_to_plot:
                observable_counter=0
                plt.figure(figsize=(5,6))
                #plt.figure()
                for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                    if observable == None:
                        continue

                        
                    if observable in exp['concentration_observables']:
                        if observable == 'H2O':
                            plt.subplot(3,1,1)
                        if observable =='OH':
                            plt.subplot(3,1,2)
                            #plt.xlim(-.005,.5)
                            #plt.ylim(1,7)

                        plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable]*1e6,'b',label='MSI')
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable]*1e6,'r',label= "$\it{a}$ $\it{priori}$ model")
                        plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable+'_ppm'],'o',color='black',label='Hong et al. experiment') 
                       # plt.xlabel('Time (ms)')
                        if observable == 'H2O':
                            plt.ylabel(r'H$_2$O ppm')
                        if observable == 'OH':
                            plt.ylabel('OH ppm')
                            
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
                        plt.legend(ncol=1)
                        sp = '_'.join(key_list)
                        #print(sp)
                        #plt.savefig(self.working_directory+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K'+'_'+str(self.exp_dict_list_original[i]['simulation'].pressure)+'_'+sp+'_'+'.pdf', bbox_inches='tight')
                        
                        #stub
                        plt.tick_params(direction='in')
                        
                        
                        #plt.savefig('Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        #plt.savefig('Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)
                        #plt.savefig(self.working_directory+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=1000)
    
    
    
                        observable_counter+=1
                        
    
                if 'perturbed_coef' in exp.keys():
                    
                    wavelengths = self.parsed_yaml_list[i]['absorbanceCsvWavelengths']
                    print(wavelengths)
                    plt.subplot(3,1,3)
                    plt.xlim(0,20)
                    plt.ylim(0,.15)
                    plt.xticks(size=12)
                    plt.yticks(size=12)

                    for k,wl in enumerate(wavelengths):
                        if wl == 227:
                            df = pd.read_csv('/Users/carlylagrotta/Desktop/neg_one.csv')
                            neg_ones = df['neg_ones'].to_numpy()
                            temp = np.multiply( np.array(exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)]) ,neg_ones)
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,temp,'o',color='black',label='Kircher et al. generated experiment')
                            
                            #plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Hong et al. experiment')
                        if wl == 215:
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Kappel et al. experiment')
    
                        plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['absorbance_calculated_from_model'][wl],'b',label='MSI')
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['absorbance_calculated_from_model'][wl],'r',label= "$\it{a}$ $\it{priori}$ model")
                        #plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Experimental Data')
                        plt.xlabel('Time [ms]',size=15)
                        plt.ylabel('Absorbance',size=15)
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
                        plt.legend(ncol=1,prop={'size': 6})
                        #plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        plt.tick_params(direction='in')
                        
                        
                        plt.savefig('Kircher_lowtemp_Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+'.pdf', bbox_inches='tight')
                        #plt.savefig('Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)
                        
                        
                        
    def shorten_sigma(self):
        flat_list = [item for sublist in self.simulation_lengths_of_experimental_data for item in sublist]
        length = sum(flat_list)
        observables_list = self.Ydf['value'].tolist()[length:]
        short_sigma = list(self.sigma)[length:]
        #print(flat_list)
        if bool(self.target_value_rate_constant_csv) and self.k_target_values=='On':
           
           k_target_value_csv = pd.read_csv(self.target_value_rate_constant_csv) 
           shape = k_target_value_csv.shape[0]
           slc = len(observables_list) - shape
           observables_list = observables_list[:slc]
           short_sigma = short_sigma[:slc]
           short_sigma = np.array(short_sigma)
        self.short_sigma =  short_sigma
           
        
        return 
            
    def sort_top_uncertainty_weighted_sens(self,top_sensitivity=6):
        S_matrix_copy = copy.deepcopy(self.S_matrix)
        self.shorten_sigma()
        sigma_csv = self.sigma_uncertainty_weighted_sensitivity_csv
        if bool(sigma_csv):
            df = pd.read_csv(sigma_csv)
            Sig = np.array(df['Sigma'])
            Sig = Sig.reshape((Sig.shape[0],1))
        elif self.sigma_ones==True:
            shape = len(self.short_sigma)
            Sig = np.ones((shape,1))

                       
            
        else:
            Sig = self.short_sigma
            
            
        #Sig = self.sigma
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
                cc=pd.DataFrame(sort_s).iloc[:,:top_sensitivity]
                top_five_reactions=cc.columns.values.tolist()
                topSensitivities[x].append(top_five_reactions)
                #ccn=pd.DataFrame(cc).as_matrix()
                ccn=pd.DataFrame(cc).to_numpy()

                sensitivitys[x].append(ccn)           
                start = start + self.simulation_lengths_of_experimental_data[x][y]
                
               
        return sensitivitys,topSensitivities
    
    
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
                wavelengths = self.parsed_yaml_list[i]['absorbanceCsvWavelengths']
                for k,wl in enumerate(wavelengths):
                    time_profiles[i].append(exp['absorbance_experimental_data'][k]['time']*1e3)       
                    observables[i].append('Absorbance_'+str(wl))                                 
        self.time_profiles = time_profiles
        self.observable_list = observables
        return time_profiles
    
    
    def get_observables_list(self):
        #use this function to return observable list and uncertainty  pass in csv and get unc and csv
        sigma_csv = self.sigma_uncertainty_weighted_sensitivity_csv
        if bool(sigma_csv):
            df = pd.read_csv(sigma_csv)
            Sig = df['Sigma'].values
            Sig = np.array(Sig)
            Sig = Sig.reshape((Sig.shape[0],1))
            observable_list  = df['Observable'].tolist()
            self.sigma_list = Sig
            
            #print(self.sigma_list)
            return observable_list
            
        else:    
            flat_list = [item for sublist in self.simulation_lengths_of_experimental_data for item in sublist]
            length = sum(flat_list)
            observables_list = self.Ydf['value'].tolist()[length:]
            
            if bool(self.target_value_rate_constant_csv) and self.k_target_values=='On':
               
               k_target_value_csv = pd.read_csv(self.target_value_rate_constant_csv) 
               shape = k_target_value_csv.shape[0]
               slc = len(observables_list) - shape
               observables_list = observables_list[:slc]
        
           
        
            return observables_list
    
    
    def plotting_uncertainty_weighted_sens(self):
        sensitivities,top_sensitivities = self.sort_top_uncertainty_weighted_sens()
        observables_list = self.get_observables_list()
        if bool(self.sigma_uncertainty_weighted_sensitivity_csv):
            
            sigma_list = self.sigma_list
        else:
            sigma_list = list(self.short_sigma)
        #start here
        time_profiles = self.getting_time_profiles_for_experiments(self.exp_dict_list_optimized)
        list_of_experiment_observables = self.observable_list
        def subplot_function(number_of_observables_in_simulation,time_profiles,sensitivities,top_sensitivity_single_exp,observables_list,list_of_experiment_observables,experiment_number):
            #plt.figure(figsize=(2,6))
            #stub
            plt.figure()
            for plot_number in range(number_of_observables_in_simulation):
                for c,top_columns in enumerate(top_sensitivity_single_exp[plot_number]):
                    plt.subplot(number_of_observables_in_simulation,1,plot_number+1)
                    if plot_number==0:
                       # plt.title('Experiment_'+str(experiment_number+1))
                        if experiment_number==26:
                            label_list = [r'ln(M$_{Cl,o,27})$',
                                          r'ln(M$_{CH_3OH,o,27})$',
                                          r'ln(P$_{11}$)',
                                          r'ln(M$_{O_2,o,27})$',
                                          r'ln(k$_2$)',
                                          r'ln($\sigma_{HO_2}$)']
                            line_color = ['red','black','green','purple','blue','brown']
                            line_type=['-',':','--','-','-','-']
                            marker_list=[None,None,None,None,'o','^']
                            plt.plot(time_profiles[plot_number],sensitivities[plot_number][:,c],label=label_list[c],color=line_color[c],linestyle=line_type[c],marker=marker_list[c],markersize=2) 
                        else:
                            plt.plot(time_profiles[plot_number],sensitivities[plot_number][:,c],label = observables_list[top_columns] +'_'+str(sigma_list[top_columns])) 
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.5)
                    plt.ylabel(list_of_experiment_observables[plot_number])
                    top,bottom = plt.ylim()
                    left,right = plt.xlim()
                    #plt.legend(ncol=2, loc='upper left',bbox_to_anchor=(-.5,-.3))
                    plt.legend(ncol=2, loc='best')
                    plt.xlim(-.1,10)
                    plt.ylim(top=1,bottom=-0.35)
                    plt.tick_params(direction='in')
                    plt.xlabel('Time [ms]',size=15)
                    plt.ylabel(r'$\dfrac{\partial ln(abs)}{\partial (x_j)}$ $\sigma_j$',size=15)
                    plt.xticks(size=15)
                    plt.yticks(size=15)
                    
                    #plt.legend(ncol=3, loc='upper left',bbox_to_anchor=(1.2,2),fontsize=2)

            if self.simulation_run==None:
                
                plt.savefig(self.working_directory+'/'+'Experiment_'+str(experiment_number+1)+'.pdf', bbox_inches='tight')
            else:
                
                plt.title('Experiment_'+str(self.simulation_run))
                plt.savefig(self.working_directory+'/'+'Experiment_'+str(self.simulation_run)+'.pdf', bbox_inches='tight')
                #plt.savefig('Experiment_'+str(self.simulation_run)+'.pdf', bbox_inches='tight')

               
        for x in range(len(sensitivities)):            
            number_of_observables_in_simulation = len(sensitivities[x])
            subplot_function(number_of_observables_in_simulation,time_profiles[x],sensitivities[x],top_sensitivities[x],observables_list,list_of_experiment_observables[x],x)
            
        return 
            
                       