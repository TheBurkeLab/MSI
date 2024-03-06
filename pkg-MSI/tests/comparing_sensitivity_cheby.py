
import sys
import os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')
import numpy as np
import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import cantera as ct
import pandas as pd
import matplotlib.pyplot as plt

obvservable_number=0
test_p_arrhenius = pr.Processor('MSI/data/test_data/FFCM1_custom.cti')
test_tube_arrhenius = st.shockTube(pressure=1.909,
                         temperature=1398,
                         observables=['H2O','OH'],
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'H2O':0.001234,
                                     'H2O2': 0.00254 ,
                                     'O2':0.00062,
                                     'Ar':0.995606,
                                     },
                         initialTime=0,
                         finalTime=0.001,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant volume',
                         processor=test_p_arrhenius,
                         save_timeHistories=1,
                         save_physSensHistories=1)



test_tube_arrhenius.run()
k_sens_arrhenius = test_tube_arrhenius.kineticSensitivities
reactions = test_tube_arrhenius.processor.solution.reaction_equations()
time_arrhenius = test_tube_arrhenius.timeHistories[0]['time']




test_p_chevy = pr.Processor('MSI/data/chebyshev_data/FFCM1_custom_cheb_test_extra_zeros.cti')
test_tube_chevy = st.shockTube(pressure=1.909,
                         temperature=1398,
                         observables=['H2O','OH'],
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'H2O':0.001234,
                                     'H2O2': 0.00254 ,
                                     'O2':0.00062,
                                     'Ar':0.995606,
                                     },
                         initialTime=0,
                         finalTime=0.001,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant volume',
                         processor=test_p_chevy,
                         save_timeHistories=1,
                         save_physSensHistories=1)



test_tube_chevy.run()
k_sens_chevy = test_tube_chevy.kineticSensitivities
reactions = test_tube_chevy.processor.solution.reaction_equations()
time_chevy = test_tube_chevy.timeHistories[0]['time']


##########################################################################################

reaction_0_chevy = k_sens_chevy[:,:,obvservable_number]
reaction_0_chevy = reaction_0_chevy[:,0]
interpolate_chevy_0 = np.interp(time_arrhenius.values,time_chevy.values,reaction_0_chevy)

reaction_0_spf = k_sens_arrhenius[:,:,obvservable_number]
reaction_0_spf = reaction_0_spf[:,0]

percent_difference_0 = ((reaction_0_spf-interpolate_chevy_0)/reaction_0_spf)*100
print(reactions[0])
print('This is the maximum sensitivity difference:',percent_difference_0.max())
print('This is the mean sensitivity difference:',percent_difference_0.mean())
print('')




reaction_1_chevy = k_sens_chevy[:,:,obvservable_number]
reaction_1_chevy = reaction_1_chevy[:,1]
interpolate_chevy_1 = np.interp(time_arrhenius.values,time_chevy.values,reaction_1_chevy)

reaction_1_spf = k_sens_arrhenius[:,:,obvservable_number]
reaction_1_spf = reaction_1_spf[:,1]

percent_difference_1 = ((reaction_1_spf-interpolate_chevy_1)/reaction_1_spf)*100
print(reactions[1])
print('This is the maximum sensitivity difference:',percent_difference_1.max())
print('This is the mean sensitivity difference:',percent_difference_1.mean())
print('')



reaction_2_chevy = k_sens_chevy[:,:,obvservable_number]
reaction_2_chevy = reaction_2_chevy[:,2]
interpolate_chevy_2 = np.interp(time_arrhenius.values,time_chevy.values,reaction_2_chevy)

reaction_2_spf = k_sens_arrhenius[:,:,obvservable_number]
reaction_2_spf = reaction_2_spf[:,2]

percent_difference_2 = ((reaction_2_spf-interpolate_chevy_2)/reaction_2_spf)*100
print(reactions[2])
print('This is the maximum sensitivity difference:',percent_difference_2.max())
print('This is the mean sensitivity difference:',percent_difference_2.mean())
print('')


reaction_3_chevy = k_sens_chevy[:,:,obvservable_number]
reaction_3_chevy = reaction_3_chevy[:,3]
interpolate_chevy_3 = np.interp(time_arrhenius.values,time_chevy.values,reaction_3_chevy)

reaction_3_spf = k_sens_arrhenius[:,:,obvservable_number]
reaction_3_spf = reaction_3_spf[:,3]

percent_difference_3 = ((reaction_3_spf-interpolate_chevy_3)/reaction_3_spf)*100

print(reactions[3])
print('This is the maximum sensitivity difference:',percent_difference_3.max())
print('This is the mean sensitivity difference:',percent_difference_3.mean())
print('')



reaction_4_chevy = k_sens_chevy[:,:,obvservable_number]
reaction_4_chevy = reaction_4_chevy[:,4]
interpolate_chevy_4 = np.interp(time_arrhenius.values,time_chevy.values,reaction_4_chevy)

reaction_4_spf = k_sens_arrhenius[:,:,obvservable_number]
reaction_4_spf = reaction_4_spf[:,4]

percent_difference_4 = ((reaction_4_spf-interpolate_chevy_4)/reaction_4_spf)*100

print(reactions[4])
print('This is the maximum sensitivity difference:',percent_difference_4.max())
print('This is the mean sensitivity difference:',percent_difference_4.mean())
print('')





#reaction_5_chevy = k_sens_chevy[:,:,obvservable_number]
#reaction_5_chevy = reaction_5_chevy[:,5]
#interpolate_chevy_5 = np.interp(time_arrhenius.values,time_chevy.values,reaction_5_chevy)
#
#reaction_5_spf = k_sens_arrhenius[:,:,obvservable_number]
#reaction_5_spf = reaction_5_spf[:,5]
#
#percent_difference_5 = ((reaction_5_spf-interpolate_chevy_5)/reaction_5_spf)*100
#
#print(reactions[5])
#print('This is the maximum sensitivity difference:',percent_difference_5.max())
#print('This is the mean sensitivity difference:',percent_difference_5.mean())
#print('')






#reaction_6_chevy = k_sens_chevy[:,:,obvservable_number]
#reaction_6_chevy = reaction_6_chevy[:,6]
#interpolate_chevy_6 = np.interp(time_arrhenius.values,time_chevy.values,reaction_6_chevy)
#
#reaction_6_spf = k_sens_arrhenius[:,:,obvservable_number]
#reaction_6_spf = reaction_6_spf[:,6]
#
#percent_difference_6 = ((reaction_6_spf-interpolate_chevy_6)/reaction_6_spf)*100
#
#print(percent_difference_6.max())
#print(percent_difference_6.mean())
#print('')
#
#
#
#
#reaction_7_chevy = k_sens_chevy[:,:,obvservable_number]
#reaction_7_chevy = reaction_7_chevy[:,7]
#interpolate_chevy_7 = np.interp(time_arrhenius.values,time_chevy.values,reaction_7_chevy)
#
#reaction_7_spf = k_sens_arrhenius[:,:,obvservable_number]
#reaction_7_spf = reaction_7_spf[:,7]
#
#percent_difference_7 = ((reaction_7_spf-interpolate_chevy_7)/reaction_7_spf)*100
#
#print(percent_difference_7.max())
#print(percent_difference_7.mean())
#print('')





#def sort_top_uncertainty_weighted_sens(observables,S_matrix,top_sensitivity=10):
#    sensitivitys =[[] for x in range(len(observables))]
#    topSensitivities = [[] for x in range(len(observables))]  
#    for i,observable in enumerate(observables):
#        
#        temp = S_matrix[:,:,i] 
#        sort_s= pd.DataFrame(temp).reindex(pd.DataFrame(temp).abs().max().sort_values(ascending=False).index, axis=1)
#        cc=pd.DataFrame(sort_s).iloc[:,:top_sensitivity]
#        top_five_reactions=cc.columns.values.tolist()
#        topSensitivities[i].append(top_five_reactions)
#        ccn=pd.DataFrame(cc).as_matrix()
#        sensitivitys[i].append(ccn) 
#    return sensitivitys,topSensitivities
#    
#    
#
#sensitivitys,topSensitivities = sort_top_uncertainty_weighted_sens(test_tube.observables,k_sens_new,top_sensitivity=5)
#
#def plotting_senstivities(sensitivitys,
#                          topSensitivities,
#                          reactions,observables,
#                          time,
#                          working_directory):
#    for i,observable in enumerate(observables):
#        plt.figure()
#        for column in range(np.shape(sensitivitys[i][0])[1]):
#            plt.plot(time,sensitivitys[i][0][:,column],label=reactions[topSensitivities[i][0][column]])
#            plt.xlabel('time (s)')
#            plt.ylabel(observable)
#            #plt.legend(loc='best',ncol=2)
#            plt.legend(ncol=5, loc='upper left',bbox_to_anchor=(-.5,-.15))
#            plt.savefig(working_directory+'/'+observable+'_ksens'+'.pdf', bbox_inches='tight')
#            plt.title(str(test_tube.temperature)+'_'+str(test_tube.conditions))
#
##plotting_senstivities(sensitivitys,topSensitivities,reactions,
##                      test_tube.observables,
##                      test_tube.timeHistories[0]['time'],'/home/carly/Dropbox/Columbia')
#
#def plotting_senstivities_for_paper(sensitivitys,
#                          topSensitivities,
#                          reactions,observables,
#                          time,
#                          working_directory):
#    marker_list = ['o', 'v', '^', '<', '>', '8', 's']
#    line_type = [ '-', '--', '-.', ':','*']
#    label_list_gri30 = [[r'H$_2$O$_2$ + OH <=> H$_2$O + HO$_2$',
#                         r'2HO$_2$ <=> H$_2$O$_2$ + O$_2$',
#                         r'CH$_4$ + OH <=> CH$_3$ + H$_2$O',
#                        r'HO$_2$ + OH <=> H$_2$O + O$_2$',
#                        r'CH$_3$ + HO$_2$ <=> CH$_3$O + OH'],
#                        [r'H$_2$O$_2$ + OH <=> H$_2$O + HO$_2$',
#                         r'CH$_3$ + HO$_2$ <=> CH$_3$O + OH',
#                         r'CH$_4$ + OH <=> CH$_3$ + H$_2$O',
#                         r'2CH$_3$ (+M) <=> C$_2$H$_6$ (+M)',
#                         r'2HO$_2$ <=> H$_2$O$_2$ + O$_2$']]
#
#    label_list_FFCM = [[r'H$_2$O$_2$ + OH <=> H$_2$O + HO$_2$',
#                         r'HO$_2$ + OH <=> H$_2$O + O$_2$',
#                         r'2CH$_3$ (+M) <=> C$_2$H$_6$ (+M)',
#                        r'CH$_4$ + OH <=> CH$_3$ + H$_2$O',
#                        r'CH$_3$ + HO$_2$ <=> CH$_4$ + O$_2$'],
#                        [r'H$_2$O$_2$ + OH <=> H$_2$O + HO$_2$',
#                         r'HO$_2$ + OH <=> H$_2$O + O$_2$',
#                         r'2CH$_3$ (+M) <=> C$_2$H$_6$ (+M)',
#                         r'CH$_3$ + HO$_2$ <=> CH$_3$O + OH',
#                         r'CH$_4$ + OH <=> CH$_3$ + H$_2$O']]
#    ffcm_HO2_color_list = ['black','darkblue','blue','royalblue','cyan']
#    ffcm_HO2_line_list = [ '-', '--', '-.', ':','*']
#    ffcm_CH3_color_list = ['black','darkblue','blue','cyan','royalblue']
#    ffcm_CH3_line_list = [ '-', '--', '-.', '*',':']
#    
#    
#    gri30_HO2_color_list = ['black','blue','royalblue','darkblue','cyan']
#    gri30_HO2_line_list = ['-','-.',':','--','*']
#    gri30_CH3_color_list = ['black','cyan','royalblue','darkblue','blue']
#    gri30_CH3_line_list = ['-','*',':','--','-.']
#
#    for i,observable in enumerate(observables):
#        #plt.figure()
#        plt.subplot(2,1,i+1)
#        counter_marker = 0
#        for column in range(np.shape(sensitivitys[i][0])[1]):
#            #plt.plot(time*1e3,sensitivitys[i][0][:,column],line_type[counter_marker],label=reactions[topSensitivities[i][0][column]],)
#            if i==0:
#                #plt.plot(time*1e3,sensitivitys[i][0][:,column],ffcm_HO2_line_list[counter_marker],label=label_list_FFCM[i][counter_marker],color=ffcm_HO2_color_list[counter_marker])
#                plt.plot(time*1e3,sensitivitys[i][0][:,column],gri30_HO2_line_list[counter_marker],label=label_list_gri30[i][counter_marker],color=gri30_HO2_color_list[counter_marker])
#                
#            if i ==1:
#                #plt.plot(time*1e3,sensitivitys[i][0][:,column],ffcm_CH3_line_list[counter_marker],label=label_list_FFCM[i][counter_marker],color=ffcm_CH3_color_list[counter_marker])
#                plt.plot(time*1e3,sensitivitys[i][0][:,column],gri30_CH3_line_list[counter_marker],label=label_list_gri30[i][counter_marker],color=gri30_CH3_color_list[counter_marker])
#
#            plt.tick_params(axis ='both', direction ='in') 
#
#            
#            if i==0:
#                plt.ylabel(observable)
#                #plt.ylabel(r'$\frac{\partial(H$_2$O)}{X}$')
#                plt.ylabel(r'$\frac{\partial(HO_2)}{\partial(X_j)}$',fontsize=12)
#                
#            if i==1:
#                plt.ylabel(r'$\frac{\partial(CH_3)}{\partial(X_j)}$',fontsize=12)
#                plt.xlabel('Time (ms)')
#            #plt.legend(loc='best',ncol=2)
#            #plt.legend(ncol=5, loc='upper left',bbox_to_anchor=(-.5,-.15))
#            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
#
#            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1)
#            plt.savefig(working_directory+'/'+'srinivasan_gri30'+'_ksens'+'.pdf', bbox_inches='tight')
#            plt.savefig(working_directory+'/'+'srinivasan_gri30'+'_ksens'+'.svg', bbox_inches='tight')
#            counter_marker+=1
#            #plt.title(str(test_tube.temperature)+'_'+str(test_tube.conditions))
#
#plotting_senstivities_for_paper(sensitivitys,topSensitivities,reactions,
#                      test_tube.observables,
#                      test_tube.timeHistories[0]['time'],
#                      '/home/carly/Dropbox/Columbia/2020_Combustion_Symposium')