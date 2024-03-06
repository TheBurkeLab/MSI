from shutil import which
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







class Plotting_for_2020_symposium(object):
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
                 shock_tube_instance = None):
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
    
    
    
    def plotting_observables(self,sigmas_original=[],sigmas_optimized=[]):
        
        
        
        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                if observable == None:
                    continue
                plt.figure()
                
                if observable in exp['mole_fraction_observables']:
                    plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable],'b',label='MSI')
                    plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original['simulation'].timeHistories[0][observable],'r',label= "$\it{a}$ $\it{priori}$ model")
                    plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable],'o',color='black',label='Experimental Data')
                    plt.xlabel('Time (ms)')
                    plt.ylabel('Mole Fraction'+''+str(observable))
                    plt.title('Experiment_'+str(i+1))
                    
                    
                    
                    

                    
                    if bool(sigmas_optimized) == True:
                        
                        high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                        high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                        low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                        low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                        plt.figure()
                        plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_optimized,'b--')
                        plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_optimized,'b--')
                        
                        
                        
                        high_error_original = np.exp(sigmas_original[i][observable_counter])
                        high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                        low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                        low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                        plt.figure()
                        plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_original,'r--')
                        plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_original,'r--')
                    
                    plt.savefig(self.working_directory+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=1000)
                    

                    observable_counter+=1
                    
                if observable in exp['concentration_observables']:
                    plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable]*1e6,'b',label='MSI')
                    plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable]*1e6,'r',label= "$\it{a}$ $\it{priori}$ model")
                    plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable+'_ppm'],'o',color='black',label='Hong et al. Experimental') 
                    plt.xlabel('Time (ms)')
                    if observable =='H2O':
                        plt.ylabel(r'H$_2$O (ppm)')
                    if observable == 'OH':
                        plt.ylabel('OH (ppm)')
                    else:                        
                        plt.ylabel(r'H$_2$O (ppm)')
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
                        
                        #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_original,'r--')
                        #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_original,'r--')
                    
                    #plt.plot([],'w' ,label= 'T:'+ str(self.exp_dict_list_original[i]['simulation'].temperature))
                    #plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
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
                    
                    
                    plt.savefig('Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                    plt.savefig('Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)
                    #plt.savefig(self.working_directory+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=1000)



                    observable_counter+=1
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list[i]['absorbanceCsvWavelengths']
                plt.figure()
                for k,wl in enumerate(wavelengths):
                    if wl == 227:
                        plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Hong et al. Experimental')
                    if wl == 215:
                        plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Kappel et al. Experimental')

                    plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['absorbance_calculated_from_model'][wl],'b',label='MSI')
                    plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['absorbance_calculated_from_model'][wl],'r',label= "$\it{a}$ $\it{priori}$ model")
                    #plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Experimental Data')
                    plt.xlabel('Time (ms)')
                    plt.ylabel('Absorbance'+' ('+str(wl)+' nm)')
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
                    plt.legend(ncol=1)
                    #plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                    plt.tick_params(direction='in')
                    
                    
                    plt.savefig('Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                    plt.savefig('Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)

    def plotting_n_figre_observables(self,n,experiment_number_want_to_plot,sigmas_original=[],sigmas_optimized=[]):
        
        
        
        for i,exp in enumerate(self.exp_dict_list_optimized):
            if i==experiment_number_want_to_plot:
                observable_counter=0
                plt.figure()
                for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                    if observable == None:
                        continue
                    plt.subplot(n,1,observable_counter+1)

                    if observable in exp['mole_fraction_observables']:
                        if re.match('[Ss]hock [Tt]ube',exp['simulation_type']) and re.match('[Ss]pecies[ -][Pp]rofile',exp['experiment_type']):
                            plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable],'b',label='MSI')
                            plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable],'r',label= "$\it{A priori}$ model")
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable],'o',color='black',label='Experimental Data')
                            plt.xlabel('Time (ms)')
                            plt.ylabel('Mole Fraction '+''+str(observable))
                            plt.title('Experiment_'+str(i+1))
                            
                            
                            
                            
        
                            
                            if bool(sigmas_optimized) == True:
                                
                                high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                                high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                                low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                                low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_optimized,'b--')
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_optimized,'b--')


                    if observable in exp['concentration_observables']:
                        
                        if observable+'_ppm' in exp['experimental_data'][observable_counter].columns:
                            plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable]*1e6,'b',label='MSI')
                            plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable]*1e6,'r',label= "$\it{a}$ $\it{priori}$ model")
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable+'_ppm'],'o',color='black',label='Hong et al. experiment') 

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
                        elif observable+'_mol/cm^3' in exp['experimental_data'][observable_counter].columns:
                            concentration_optimized = np.true_divide(1,exp['simulation'].timeHistories[0]['temperature'].to_numpy())*exp['simulation'].timeHistories[0]['pressure'].to_numpy()
                           
                            concentration_optimized *= (1/(8.314e6))*exp['simulation'].timeHistories[0][observable].dropna().to_numpy()
                            concentration_original = np.true_divide(1,self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'].to_numpy())*self.exp_dict_list_original[i]['simulation'].timeHistories[0]['pressure'].to_numpy()
                           
                            concentration_original *= (1/(8.314e6))*self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable].dropna().to_numpy()
                            
                            plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,concentration_optimized,'b',label='MSI')
                            plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,concentration_original,'r',label= "$\it{A priori}$ model")
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable+'_mol/cm^3'],'o',color='black',label='Experimental Data') 
                            plt.xlabel('Time (ms)')
                            plt.ylabel(r'$\frac{mol}{cm^3}$'+''+str(observable))
                            plt.title('Experiment_'+str(i+1))
                            
                            if bool(sigmas_optimized)==True:
                                concentration_sig = np.true_divide(1,exp['simulation'].pressureAndTemperatureToExperiment[observable_counter]['temperature'].to_numpy())*exp['simulation'].pressureAndTemperatureToExperiment[observable_counter]['pressure'].to_numpy()
                        
                                concentration_sig *= (1/(8.314e6))*exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().to_numpy()
                                high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                                high_error_optimized = np.multiply(high_error_optimized,concentration_sig)
                                low_error_optimized = np.exp(np.array(sigmas_optimized[i][observable_counter])*-1)
                                low_error_optimized = np.multiply(low_error_optimized,concentration_sig)
                                
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_optimized,'b--')
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_optimized,'b--')                        
                                plt.tick_params(direction='in')
                        
    
                        observable_counter+=1
                        
    
                if 'perturbed_coef' in exp.keys():
                    wavelengths = self.parsed_yaml_list[i]['absorbanceCsvWavelengths']
                    plt.subplot(3,1,3)
                    plt.xlim(-.005,.5)


                    for k,wl in enumerate(wavelengths):
                        if wl == 227:
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Hong et al. experiment')
                        if wl == 215:
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Kappel et al. experiment')
    
                        plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['absorbance_calculated_from_model'][wl],'b',label='MSI')
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['absorbance_calculated_from_model'][wl],'r',label= "$\it{a}$ $\it{priori}$ model")
                        #plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Experimental Data')
                        plt.xlabel('Time [ms]')
                        plt.ylabel('Absorbance'+' '+str(wl)+' nm')
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
                        plt.legend(ncol=1)
                        #plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        plt.tick_params(direction='in')
                        
                        
                        #plt.savefig('Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        #plt.savefig('Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)
    def plotting_3_figre_observables(self,experiment_number_want_to_plot,sigmas_original=[],sigmas_optimized=[]):
        
        
        
        for i,exp in enumerate(self.exp_dict_list_optimized):
            if i==experiment_number_want_to_plot:
                observable_counter=0
                plt.figure()
                for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                    if observable == None:
                        continue

                        
                    if observable in exp['concentration_observables']:
                        if observable == 'H2O':
                            plt.subplot(3,1,1)
                        if observable =='OH':
                            plt.subplot(3,1,2)
                            plt.xlim(-.005,.5)
                            plt.ylim(1,7)

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
                    plt.subplot(3,1,3)
                    plt.xlim(-.005,.5)


                    for k,wl in enumerate(wavelengths):
                        if wl == 227:
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Hong et al. experiment')
                        if wl == 215:
                            plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Kappel et al. experiment')
    
                        plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['absorbance_calculated_from_model'][wl],'b',label='MSI')
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['absorbance_calculated_from_model'][wl],'r',label= "$\it{a}$ $\it{priori}$ model")
                        #plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Experimental Data')
                        plt.xlabel('Time [ms]')
                        plt.ylabel('Absorbance'+' '+str(wl)+' nm')
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
                        plt.legend(ncol=1)
                        #plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        plt.tick_params(direction='in')
                        
                        
                        #plt.savefig('Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        #plt.savefig('Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)


    def plotting_observables_subplots(self,sigmas_original=[],sigmas_optimized=[],file_identifier='',filetype='.jpg'):
                
        
        for i,exp in enumerate(self.exp_dict_list_optimized):
            
            observable_counter=0
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables'] + exp['ignition_delay_observables']):
                if observable == None:
                    continue
                plt.figure()
                plt.subplot(3,1,1)
                
                if observable in exp['mole_fraction_observables']:
                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']) and re.match('[Ss]pecies[ -][Pp]rofile',exp['experiment_type']):
                        plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable],'b',label='MSI')
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable],'r',label= "$\it{a priori}$ model")
                        plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable],'o',color='black',label='Experimental Data')
                        plt.xlabel('Time (ms)',fontsize=15)
                        plt.tick_params(direction='in',which='both',labelsize=15)
                        plt.ylabel(str(observable)+' '+ '[mole fraction]',fontsize=15 )
                        
                        
                        
                        
                        
    
                        
                        if bool(sigmas_optimized) == True:
                            
                            high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                            high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_optimized,'b--')
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_optimized,'b--')
                            

                        plt.savefig(self.working_directory+'/'+'subplot_Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        plt.savefig(self.working_directory+'/'+'subplot_Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)                      
    
                        observable_counter+=1
                    elif re.match('[Jj][Ss][Rr]',exp['simulation_type']):
                        nominal=self.run_jsr(self.exp_dict_list_original[i],self.nominal_cti)
                        
                        MSI_model=self.run_jsr(exp,self.new_cti)
                        plt.plot(MSI_model['temperature'],MSI_model[observable],'b',label='MSI')
                        plt.plot(nominal['temperature'],nominal[observable],'r',label= "$\it{A priori}$ model")
                        plt.plot(exp['experimental_data'][observable_counter]['Temperature'],exp['experimental_data'][observable_counter][observable],'o',color='black',label='Experimental Data')
                        plt.xlabel('Temperature (K)')
                        plt.ylabel('Mole Fraction '+''+str(observable))
                        plt.title('Experiment_'+str(i+1))
                        
                        if bool(sigmas_optimized) == True:
                            
                            high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])
                            print(high_error_optimized)
                            high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values)
                            low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values)
                            #plt.figure()
                            if len(high_error_optimized)>1 and len(low_error_optimized) > 1:
                                plt.plot(exp['experimental_data'][observable_counter]['Temperature'],  high_error_optimized,'b--')
                                plt.plot(exp['experimental_data'][observable_counter]['Temperature'],low_error_optimized,'b--')
                            else:
                                print(high_error_optimized,observable,exp['simulation'].timeHistories[0][observable].dropna().values)
                                plt.plot(exp['experimental_data'][observable_counter]['Temperature'],  high_error_optimized,'rX')
                                plt.plot(exp['experimental_data'][observable_counter]['Temperature'],low_error_optimized,'bX')
                            
                            
                            
                            #high_error_original = np.exp(sigmas_original[i][observable_counter])
                           # high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            #low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                            #low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            #plt.figure()
                           # plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_original,'r--')
                            #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_original,'r--')
                        
                        plt.savefig(os.path.join(self.working_directory,'Experiment_'+str(i+1)+'_'+str(observable)+file_identifier+filetype), bbox_inches='tight',dpi=1200)
                        observable_counter+=1
                    elif re.match('[Ss]pecies[- ][Pp]rofile',exp['experiment_type']) and re.match('[Ff]low[ -][Rr]eactor',exp['simulation_type']):
                        plt.plot(exp['simulation'].timeHistories[0]['temperature'],exp['simulation'].timeHistories[0][observable],'b',label='MSI')
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'],self.exp_dict_list_original['simulation'].timeHistories[0][observable],'r',label= "$\it{A priori}$ model")
                        plt.plot(exp['experimental_data'][observable_counter]['Temperature'],exp['experimental_data'][observable_counter][observable],'o',color='black',label='Experimental Data')
                        plt.xlabel('Time (ms)')
                        plt.ylabel('Mole Fraction '+''+str(observable))
                        plt.title('Experiment_'+str(i+1))
                        
                        
                        
                        
    
                        
                        if bool(sigmas_optimized) == True:
                            
                            high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                            high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values)
                            low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values)
                            plt.plot(exp['experimental_data'][observable_counter]['Temperature']*1e3,  high_error_optimized,'b--')
                            plt.plot(exp['experimental_data'][observable_counter]['Temperature']*1e3,low_error_optimized,'b--')
                            
                            
                            
                            #high_error_original = np.exp(sigmas_original[i][observable_counter])
                            #high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable].dropna().values)
                            #low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                            #low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            
                            #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_original,'r--')
                            #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_original,'r--')
                        
                        plt.savefig(self.working_directory+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=1000)                       
                        
                        
                if observable in exp['concentration_observables']:
                    #print(observable_counter,'THIS IS OBSERVABLE COUNTER')
                    if re.match('[Ss]hock [Tt]ube',exp['simulation_type']) and re.match('[Ss]pecies[ -][Pp]rofile',exp['experiment_type']):
                        #print(observable_counter)
                        if observable+'_ppm' in exp['experimental_data'][observable_counter].columns:
                            plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable]*1e6,'b',label='MSI')
                            plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable]*1e6,'r',label= "$\it{A priori}$ model")
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable+'_ppm'],'o',color='black',label='Experimental Data') 
                            plt.xlabel('Time (ms)')
                            plt.ylabel(str(observable)+ ' '+ 'ppm')
                            plt.title('Experiment_'+str(i+1))
                            
                            if bool(sigmas_optimized)==True:
                                high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                                high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                                low_error_optimized = np.exp(np.array(sigmas_optimized[i][observable_counter])*-1)
                                low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                                
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_optimized,'b--')
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_optimized,'b--')                    
                                
            
            
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
                            plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,concentration_original,'r',label= "$\it{A priori}$ model")
                            plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,exp['experimental_data'][observable_counter][observable+'_mol/cm^3'],'o',color='black',label='Experimental Data') 
                            plt.xlabel('Time (ms)')
                            plt.ylabel(r'$\frac{mol}{cm^3}$'+''+str(observable))
                            plt.title('Experiment_'+str(i+1))
                            
                            if bool(sigmas_optimized)==True:
                                concentration_sig = np.true_divide(1,exp['simulation'].pressureAndTemperatureToExperiment[observable_counter]['temperature'].to_numpy())*exp['simulation'].pressureAndTemperatureToExperiment[observable_counter]['pressure'].to_numpy()
                        
                                concentration_sig *= (1/(8.314e6))*exp['simulation'].timeHistoryInterpToExperiment[observable].dropna().to_numpy()
                                high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                                high_error_optimized = np.multiply(high_error_optimized,concentration_sig)
                                low_error_optimized = np.exp(np.array(sigmas_optimized[i][observable_counter])*-1)
                                low_error_optimized = np.multiply(low_error_optimized,concentration_sig)
                                
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_optimized,'b--')
                                plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_optimized,'b--') 
                        
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
                        #plt.savefig(self.working_directory+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K'+'_'+str(self.exp_dict_list_original[i]['simulation'].pressure)+'_'+sp+'_'+'.pdf', bbox_inches='tight')
                        
                        #stub
                        plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                        plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)
                    


                        observable_counter+=1
                    
                    elif re.match('[Ff]low [Rr]eactor',exp['simulation_type']) and re.match('[Ss]pecies[ -][Pp]rofile',exp['experiment_type']):
                        plt.plot(exp['simulation'].timeHistories[0]['initial_temperature'],exp['simulation'].timeHistories[0][observable]*1e6,'b',label='MSI')
                        plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['initial_temperature'],self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable]*1e6,'r',label= "$\it{A priori}$ model")
                        plt.plot(exp['experimental_data'][observable_counter]['Temperature'],exp['experimental_data'][observable_counter][observable+'_ppm'],'o',color='black',label='Experimental Data')
                        plt.xlabel('Temperature (K)')
                        plt.ylabel('ppm '+''+str(observable))
                        plt.title('Experiment_'+str(i+1))
                        
                        
                        
                        
    
                        
                        if bool(sigmas_optimized) == True:
                            #stub
                            high_error_optimized = np.exp(sigmas_optimized[i][observable_counter])                   
                            high_error_optimized = np.multiply(high_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values*1e6)
                            low_error_optimized = np.exp(sigmas_optimized[i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp['simulation'].timeHistories[0][observable].dropna().values*1e6)
                            plt.plot(exp['experimental_data'][observable_counter]['Temperature'],  high_error_optimized,'b--')
                            plt.plot(exp['experimental_data'][observable_counter]['Temperature'],low_error_optimized,'b--')
                            
                            
                            
                            #high_error_original = np.exp(sigmas_original[i][observable_counter])
                            #high_error_original = np.multiply(high_error_original,self.exp_dict_list_original[i]['simulation'].timeHistories[0][observable].dropna().values)
                            #low_error_original = np.exp(sigmas_original[i][observable_counter]*-1)
                            #low_error_original = np.multiply(low_error_original,self.exp_dict_list_original[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values)
                            
                            #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,  high_error_original,'r--')
                            #plt.plot(exp['experimental_data'][observable_counter]['Time']*1e3,low_error_original,'r--')
                        
                        #plt.savefig(self.working_directory+'/'+'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf', bbox_inches='tight',dpi=1000)                     
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

                            plt.semilogy(1000/nominal['temperature'],nominal['delay'],'r',label= "$\it{A priori}$ model")
                            
                            #plt.semilogy(1000/exp['simulation'].timeHistories[0]['temperature'],exp['simulation'].timeHistories[0]['delay'],'b',label='MSI')
                            #plt.semilogy(1000/self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'],self.exp_dict_list_original[i]['simulation'].timeHistories[0]['delay'],'r',label= "$\it{A priori}$ model")
                            plt.semilogy(1000/exp['experimental_data'][observable_counter]['temperature'],exp['experimental_data'][observable_counter][observable+'_s'],'o',color='black',label='Experimental Data')
                            plt.xlabel('1000/T')
                            plt.ylabel('Time (s)')
                            plt.title('Experiment_'+str(i+1))
                            
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
                            
                            
                            
                            
                            plt.savefig(os.path.join(self.working_directory,'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf'), bbox_inches='tight',dpi=1000)
                            plt.savefig(os.path.join(self.working_directory,'Experiment_'+str(i+1)+'_'+str(observable)+'.svg'), bbox_inches='tight',dpi=1000)
                            observable_counter+=1
                    elif re.match('[Rr][Cc][Mm]',exp['simulation_type']):
                        if len(exp['simulation'].temperatures)>1:
                            
                            plt.semilogy(1000/exp['simulation'].timeHistories[0]['ignition_temperature'],exp['simulation'].timeHistories[0]['delay']-exp['simulation'].timeHistories[0]['end_of_compression_time'],'b',label='MSI')
                            plt.semilogy(1000/self.exp_dict_list_original[i]['simulation'].timeHistories[0]['ignition_temperature'],self.exp_dict_list_original[i]['simulation'].timeHistories[0]['delay']-self.exp_dict_list_original[i]['simulation'].timeHistories[0]['end_of_compression_time'],'r',label= "$\it{A priori}$ model")
    
                            #plt.semilogy(1000/exp['simulation'].timeHistories[0]['temperature'],exp['simulation'].timeHistories[0]['delay'],'b',label='MSI')
                            #plt.semilogy(1000/self.exp_dict_list_original[i]['simulation'].timeHistories[0]['temperature'],self.exp_dict_list_original[i]['simulation'].timeHistories[0]['delay'],'r',label= "$\it{A priori}$ model")
                            plt.semilogy(1000/exp['experimental_data'][observable_counter]['temperature'],exp['experimental_data'][observable_counter][observable+'_s'],'o',color='black',label='Experimental Data')
                            plt.xlabel('1000/T (1000/K)')
                            plt.ylabel('Time (ms)')
                            plt.title('Experiment_'+str(i+1))
                            
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
                            
                            plt.savefig(os.path.join(self.working_directory,'Experiment_'+str(i+1)+'_'+str(observable)+'.pdf'), bbox_inches='tight',dpi=1000)
                            observable_counter+=1                            
                        
                    

            if 'perturbed_coef' in exp.keys():
                wavelengths = self.parsed_yaml_list[i]['absorbanceCsvWavelengths']
                plt.figure()
                for k,wl in enumerate(wavelengths):
                    plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Experimental Data')

                    plt.plot(exp['simulation'].timeHistories[0]['time']*1e3,exp['absorbance_calculated_from_model'][wl],'b',label='MSI')
                    plt.plot(self.exp_dict_list_original[i]['simulation'].timeHistories[0]['time']*1e3,self.exp_dict_list_original[i]['absorbance_calculated_from_model'][wl],'r',label= "$\it{A priori}$ model")
                    #plt.plot(exp['absorbance_experimental_data'][k]['time']*1e3,exp['absorbance_experimental_data'][k]['Absorbance_'+str(wl)],'o',color='black',label='Experimental Data')
                    plt.xlabel('Time (ms)')
                    plt.ylabel('Absorbance'+''+str(wl))
                    plt.title('Experiment_'+str(i+1))
                    
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





                    key_list=[]
                    plt.plot([],'w' ,label= 'T:'+ str(self.exp_dict_list_original[i]['simulation'].temperature))
                    plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
                    for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():                        
                        plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                        key_list.append(key)

                    #plt.legend(handlelength=3)
                    plt.legend(ncol=2)
                    #plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                    sp = '_'.join(key_list)
                    
                    
                    
                    
                    plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.pdf', bbox_inches='tight')
                    plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'K_'+sp+'.svg', bbox_inches='tight',transparent=True)
