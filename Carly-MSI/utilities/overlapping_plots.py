
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors 
class overlapping_plots(object):
    def __init__(self,
                 exp_dict_list_optimized,
                 exp_dict_list_original,
                 parsed_yaml_list,
                 list_of_exp_dict_list_optimized=[],
                 list_of_exp_dict_list_original=[],
                 working_directory='/home/carly/Dropbox/Columbia'):



        self.exp_dict_list_optimized = exp_dict_list_optimized
        self.exp_dict_list_original = exp_dict_list_original
        self.parsed_yaml_list = parsed_yaml_list
        self.list_of_exp_dict_list_optimized = list_of_exp_dict_list_optimized
        self.list_of_exp_dict_list_original = list_of_exp_dict_list_original
        self.working_directory = working_directory
    
    def plotting_observables(self,sigmas_original_list=[],sigmas_optimized_list=[]):
        number = 6
        cmap = plt.get_cmap('tab10')
        cmap2 = plt.get_cmap('autumn')
        blue_colors = [cmap(i) for i in np.linspace(0, 1, number)]
        red_colors = [cmap2(i) for i in np.linspace(0, 1, number)]
        marker_list = ['+','o','D','*','^']
        line_type = ['-', '--', '-.', ':','-']
        line_width = [1,1,1,1,1]
     
        
        overall_observables = []
        for i,exp in enumerate(self.exp_dict_list_optimized):
            observable_counter=0
            temp = []
            for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                temp.append(observable)
            overall_observables.append(temp)
                
        
        
            
#            # shouldn't be looping over all of this 
        for i,simulation in enumerate(overall_observables):
            if 'perturbed_coef' in self.list_of_exp_dict_list_optimized[0][i].keys():
                absorb = True
                wavelengths = self.parsed_yaml_list[i]['absorbanceCsvWavelengths']
            else:
                absorb = False
            observable_counter=0

            for j,observable in enumerate(overall_observables[i]):
                if observable==None:
                    continue
                #print(observable,i)
                
                if observable in self.list_of_exp_dict_list_optimized[0][i]['concentration_observables']:
                    plt.figure()
                    for k,exp in enumerate(self.list_of_exp_dict_list_optimized):
                        #print(observable,k)
                        plt.plot(self.list_of_exp_dict_list_original[k][i]['simulation'].timeHistories[0]['time']*1e3,self.list_of_exp_dict_list_original[k][i]['simulation'].timeHistories[0][observable]*1e6,linewidth=1,color = 'r',label= "$\it{A priori}$ model_"+str(k))

                        plt.plot(exp[i]['simulation'].timeHistories[0]['time']*1e3,exp[i]['simulation'].timeHistories[0][observable]*1e6,color = blue_colors[k],linestyle=line_type[k],marker=marker_list[k],markersize=1.5,linewidth=line_width[k],label='MSI_'+str(k))
                        #k+1+k*-5
                        if k ==0:
                            plt.plot(exp[i]['experimental_data'][observable_counter]['Time']*1e3,exp[i]['experimental_data'][observable_counter][observable+'_ppm'],'o',color='black',label='Experimental Data') 
                            plt.ylabel(observable+' ' + 'ppm')
                            plt.xlabel('Time (ms)')
                            plt.plot([],'w' ,label= 'T:'+ str(self.exp_dict_list_original[i]['simulation'].temperature))
                            plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
                            for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():                        
                                plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                        plt.legend(ncol=5, loc='upper left',bbox_to_anchor=(-.2,-.2))

                        if bool(sigmas_optimized_list) == True:
                        
                            high_error_optimized = np.exp(sigmas_optimized_list[k][i][observable_counter])                   
                            high_error_optimized = np.multiply(high_error_optimized,exp[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                            low_error_optimized = np.exp(sigmas_optimized_list[k][i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp[i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                            plt.plot(exp[i]['experimental_data'][observable_counter]['Time']*1e3,  high_error_optimized,color = blue_colors[k],linestyle='--')
                            plt.plot(exp[i]['experimental_data'][observable_counter]['Time']*1e3,low_error_optimized,color = blue_colors[k],linestyle='--')
                            
                            
                            
                            high_error_original = np.exp(sigmas_original_list[k][i][observable_counter])
                            high_error_original = np.multiply(high_error_original,self.list_of_exp_dict_list_original[k][i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                            low_error_original = np.exp(sigmas_original_list[k][i][observable_counter]*-1)
                            low_error_original = np.multiply(low_error_original,self.list_of_exp_dict_list_original[k][i]['simulation'].timeHistoryInterpToExperiment[observable].dropna().values*1e6)
                            #plt.plot(exp[i]['experimental_data'][observable_counter]['Time']*1e3,  high_error_original,color = red_colors[k+1+k*-5],linestyle='--')
                            #plt.plot(exp[i]['experimental_data'][observable_counter]['Time']*1e3,low_error_original,color = red_colors[k+1+k*-5],linestyle='--')  
                plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+'_'+str(observable)+'_'+str(self.exp_dict_list_original[i]['simulation'].temperature)+'.pdf', bbox_inches='tight')
                observable_counter+=1
                
                
            if absorb == True:
                plt.figure()
                for k,exp in enumerate(self.list_of_exp_dict_list_optimized):
                    for p,wl in enumerate(wavelengths):
                        plt.plot(self.list_of_exp_dict_list_original[k][i]['simulation'].timeHistories[0]['time']*1e3,self.list_of_exp_dict_list_original[k][i]['absorbance_calculated_from_model'][wl],color = 'r',linewidth=1,label= "$\it{A priori}$ model_"+str(k))
                        plt.plot(exp[i]['simulation'].timeHistories[0]['time']*1e3,exp[i]['absorbance_calculated_from_model'][wl],color = blue_colors[k],linestyle=line_type[k],marker=marker_list[k],linewidth=line_width[k],markersize=1.5,label='MSI_'+str(k))                               
                        if k ==0:
                            plt.plot(exp[i]['absorbance_experimental_data'][p]['time']*1e3,exp[i]['absorbance_experimental_data'][p]['Absorbance_'+str(wl)],'o',color='black',label='Experimental Data')
                            plt.ylabel('Absorbance'+''+str(wl))
                            plt.xlabel('Time (ms)')
                            plt.plot([],'w' ,label= 'T:'+ str(self.exp_dict_list_original[i]['simulation'].temperature))
                            plt.plot([],'w', label= 'P:'+ str(self.exp_dict_list_original[i]['simulation'].pressure))
                            for key in self.exp_dict_list_original[i]['simulation'].conditions.keys():                        
                                plt.plot([],'w',label= key+': '+str(self.exp_dict_list_original[i]['simulation'].conditions[key]))
                        plt.legend(ncol=5, loc='upper left',bbox_to_anchor=(-.2,-.2))
                        
                        
                        if bool(sigmas_optimized_list)==True:
                            high_error_optimized = np.exp(sigmas_optimized_list[k][i][observable_counter])
                            high_error_optimized = np.multiply(high_error_optimized,exp[i]['absorbance_model_data'][wl])
                            low_error_optimized = np.exp(sigmas_optimized_list[k][i][observable_counter]*-1)
                            low_error_optimized = np.multiply(low_error_optimized,exp[i]['absorbance_model_data'][wl])
                            
                            plt.plot(exp[i]['absorbance_experimental_data'][p]['time']*1e3,high_error_optimized,color = blue_colors[k],linestyle='--')
                            plt.plot(exp[i]['absorbance_experimental_data'][p]['time']*1e3,low_error_optimized,color = blue_colors[k],linestyle='--')
                        
                        #plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+str(self.exp_dict_list_original[i]['simulation'].temperature)+'.pdf', bbox_inches='tight')

                            high_error_original = np.exp(sigmas_original_list[k][i][observable_counter])
                            high_error_original = np.multiply(high_error_original,self.list_of_exp_dict_list_original[k][i]['absorbance_model_data'][wl])
                            low_error_original =  np.exp(sigmas_original_list[k][i][observable_counter]*-1)
                            low_error_original = np.multiply(low_error_original,self.list_of_exp_dict_list_original[k][i]['absorbance_model_data'][wl])
                            
                            
                            #plt.plot(exp[i]['absorbance_experimental_data'][p]['time']*1e3,high_error_original,color = red_colors[k+1+k*-5],linestyle='--')
                            #plt.plot(exp[i]['absorbance_experimental_data'][p]['time']*1e3,low_error_original,color = red_colors[k+1+k*-5],linestyle='--')
                plt.savefig(self.working_directory+'/'+'Exp_'+str(i+1)+' '+'Absorb at'+'_'+str(wl)+str(self.exp_dict_list_original[i]['simulation'].temperature)+'.pdf', bbox_inches='tight')
                

        
        

    
    def maximum_difference_between_perdictions(self,experimental_dict_list_one=[],experimental_dict_list_two=[],experiments_of_interst=[]):
        
        
        overall_list_temperature = []
        overall_list_percent_difference = []
        overall_list_difference= []
        overall_pressure_list = []
        for i,exp in enumerate(experimental_dict_list_one):
            observable_counter=0
            experiment_list_percent_difference = []
            experiment_list_difference = []
            temperature_list = []
            pressure_list = []
            if i in experiments_of_interst:
                for j,observable in enumerate(exp['mole_fraction_observables'] + exp['concentration_observables']):
                    if observable == None:
                        continue
                    if observable in exp['mole_fraction_observables']:
                        time1 = exp['simulation'].timeHistories[0]['time']*1e3
                        perdiction1 = exp['simulation'].timeHistories[0][observable]
                        
                        time2 = experimental_dict_list_two[i]['simulation'].timeHistories[0]['time']*1e3,exp['simulation'].timeHistories[0][observable]
                        perdiction2 = experimental_dict_list_two[i]['simulation'].timeHistories[0][observable]

                        interpolate_2_to_1 = np.interp(time1,time2,perdiction2)
                        relative_difference = perdiction1 - interpolate_2_to_1
                        percent_difference = ((perdiction1 - interpolate_2_to_1)/perdiction1)*100
                        experiment_list_difference.append(np.max(np.abs(relative_difference)))
                        experiment_list_percent_difference.append(np.max(np.abs(percent_difference)))
                        temperature_list.append(experimental_dict_list_one[i]['simulation'].temperature)
                        pressure_list.append(experimental_dict_list_one[i]['simulation'].pressure)
                        
                        
                        observable_counter+=1
                        
                    if observable in exp['concentration_observables']:
                        time1 = exp['simulation'].timeHistories[0]['time']*1e3
                        perdiction1 = exp['simulation'].timeHistories[0][observable]
                        
                        time2 = experimental_dict_list_two[i]['simulation'].timeHistories[0]['time']*1e3
                        perdiction2 = experimental_dict_list_two[i]['simulation'].timeHistories[0][observable]

                        interpolate_2_to_1 = np.interp(time1.dropna().values,time2.dropna().values,perdiction2.values)
                        relative_difference = perdiction1 - interpolate_2_to_1
                        percent_difference = ((perdiction1 - interpolate_2_to_1)/perdiction1)*100
                        experiment_list_difference.append(np.max(np.abs(relative_difference)))
                        experiment_list_percent_difference.append(np.max(np.abs(percent_difference)))
                        
                        temperature_list.append(experimental_dict_list_one[i]['simulation'].temperature)
                        pressure_list.append(experimental_dict_list_one[i]['simulation'].pressure)
                        
                        observable_counter+=1
                        
    
                if 'perturbed_coef' in exp.keys():
                    wavelengths = self.parsed_yaml_list[i]['absorbanceCsvWavelengths']
                    for k,wl in enumerate(wavelengths):
                        time1 = exp['simulation'].timeHistories[0]['time']*1e3
                        perdiction1 = exp['absorbance_calculated_from_model'][wl]
                        
                        time2 = experimental_dict_list_two[i]['simulation'].timeHistories[0]['time']*1e3
                        perdiction2 = experimental_dict_list_two[i]['absorbance_calculated_from_model'][wl]
                        interpolate_2_to_1 = np.interp(time1,time2,perdiction2)
                        relative_difference = perdiction1 - interpolate_2_to_1
                        percent_difference = ((perdiction1 - interpolate_2_to_1)/perdiction1)*100
                        experiment_list_difference.append(np.max(np.abs(relative_difference)))
                        experiment_list_percent_difference.append(np.max(np.abs(percent_difference)))
                        pressure_list.append(experimental_dict_list_one[i]['simulation'].pressure)
                        temperature_list.append(experimental_dict_list_one[i]['simulation'].temperature)
            
        

                overall_list_difference.append(experiment_list_difference)
                overall_list_percent_difference.append(experiment_list_percent_difference)
                overall_list_temperature.append(temperature_list)
                overall_pressure_list.append(pressure_list)
        return overall_list_temperature,overall_list_difference,overall_list_percent_difference,overall_pressure_list

    def plotting_differences(self,temperature =[],
                             difference =[], 
                             percent_difference = [],
                             difference_2=[],
                             percent_difference_2 = [],
                             difference_3 = [],
                             observable_number =None,
                             pressure_list=[]):
        temp_to_plot = []
        difference_to_plot = []
        percent_difference_to_plot = []
        difference_to_plot_2 =[]
        percent_difference_to_plot_2 = []
        difference_to_plot_3=[]
        pressure_to_plot = []
        for i, experiment in enumerate(temperature):
            for j, observable in enumerate(temperature):
                if j==observable_number:
                    temp_to_plot.append(temperature[i][j])
                    difference_to_plot.append(difference[i][j])
                    percent_difference_to_plot.append(percent_difference[i][j])
                    percent_difference_to_plot_2.append(percent_difference_2[i][j])
                    difference_to_plot_2.append(difference_2[i][j])
                    difference_to_plot_3.append(difference_3[i][j])
                    pressure_to_plot.append(pressure_list[i][j])
        
        plt.figure()
        a, b = zip(*sorted(zip(temp_to_plot,difference_to_plot)))  
        aa,bb =  zip(*sorted(zip(temp_to_plot,difference_to_plot_2)))
        aaa,bbb = zip(*sorted(zip(temp_to_plot,difference_to_plot_3)))
        #plt.title('Absorbance 227 nm')
        plt.xlabel('Temperature')
        plt.ylabel('Relative Difference')
        plt.plot(a,b,marker='o')
        plt.plot(aa,bb,marker='o')
        plt.plot(aaa,bbb,marker='o')
        
        
        max_diference = max(difference_to_plot)
        max_diference_2 = max(difference_to_plot_2)
        max_diference_3 = max(difference_to_plot_3)
        normalized_difference = np.array(difference_to_plot)/max_diference
        normalized_difference_2 = np.array(difference_to_plot_2)/max_diference_2
        normalized_difference_3 = np.array(difference_to_plot_3)/max_diference_3

        plt.figure()
        #plt.title('Absorbance 227 nm')
        plt.xlabel('Temperature')
        plt.ylabel('Normalized Relative Difference')
        c, d = zip(*sorted(zip(temp_to_plot,normalized_difference)))  
        cc, dd = zip(*sorted(zip(temp_to_plot,normalized_difference_2)))  
        ccc,ddd=zip(*sorted(zip(temp_to_plot,normalized_difference_3))) 

        plt.plot(c,d,marker='o')
        
        plt.plot(cc,dd,marker='o')
        plt.plot(ccc,ddd,marker='o')
        
        plt.figure()
        e, f = zip(*sorted(zip(temp_to_plot,pressure_to_plot)))  
        plt.plot(e,f,marker='o')
        #plt.title('Absorbance 215 nm')
        plt.xlabel('Temperature')
        plt.ylabel('Pressure')
    
    
