import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built
import numpy as np
import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import cantera as ct
import pandas as pd
import matplotlib.pyplot as plt
test_p = pr.Processor('MSI/data/test_data/FFCM1_custom_updated.cti')

#test_tube = st.shockTube(pressure=1.672,
#                         temperature=1182,
#                         observables=['OH','H2O'],
#                         kineticSens=1,
#                         physicalSens=0,
#                         conditions={'H2O':0.001113,
#                                     'H2O2':0.002046,
#                                     'O2':0.000556,
#                                     'Ar':0.996285},
#                         initialTime=0,
#                         finalTime=0.003,
#                         thermalBoundary='Adiabatic',
#                         mechanicalBoundary='constant pressure',
#                         processor=test_p,
#                         save_timeHistories=1,
#                         save_physSensHistories=1)
#test_tube = st.shockTube(pressure=1.635,
#                         temperature=1283,
#                         observables=['OH','H2O'],
#                         kineticSens=1,
#                         physicalSens=0,
#                         conditions={'H2O':0.001113,
#                                     'H2O2':0.003094 ,
#                                     'O2':0.000556,
#                                     'Ar':0.996285},
#                         initialTime=0,
#                         finalTime=0.003,
#                         thermalBoundary='Adiabatic',
#                         mechanicalBoundary='constant pressure',
#                         processor=test_p,
#                         save_timeHistories=1,
#                         save_physSensHistories=1)

#test_p = pr.Processor('MSI/data/test_data/FFCM1_custom.cti')
#test_p = pr.Processor('MSI/data/chebyshev_data/FFCM1_custom_cheb.cti')
#'CH3':  0.0000012094157562676408,
#                                     'HO2': 0.0000007614839946870331,
#                                     'OH': 0.000003041863871824672,
#                                     'H2O2':0.0001531112203220986,
#                                     'CH4':0.0174693387016437,
#                                     'H2O':0.014293095301344845,
#                                     'He':0.9990681757005978
#OH
                                     #,
                                     #'N2O':0.0007737003154574132,
#
test_p = pr.Processor('MSI/data/test_data/FFCM1_custom.cti')
test_p = pr.Processor('gri30.cti')
test_tube = st.shockTube(pressure=.98692,
                         temperature=295,
                         observables=['HO2','CH3'],
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'CH3':  0.0000012094157562676408,
                                     'HO2': 0.0000007614839946870331,
                                     'OH': 0.000003041863871824672,
                                     'H2O2':0.0001531112203220986,
                                     'CH4':0.0174693387016437,
                                     'H2O':0.014293095301344845,
                                     'N2O':0.0007737003154574132,
                                     'Ar':0.9990681757005978
                                     },
                         initialTime=0,
                         finalTime=0.003,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant volume',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1)



#test_tube = st.shockTube(pressure=1.635,
#                         temperature=1283,
#                         observables=['OH','H2O','H2O2','HO2'],
#                         kineticSens=1,
#                         physicalSens=0,
#                         conditions={'H2O2':0.003049 ,'H2O':0.001113,
#                                     'O2':0.000556,'Ar':0.995212},
#                         initialTime=0,
#                         finalTime=0.001,
#                         thermalBoundary='Adiabatic',
#                         mechanicalBoundary='constant pressure',
#                         processor=test_p,
#                         save_timeHistories=1,
#                         save_physSensHistories=1)

test_tube.run()
k_sens_new = test_tube.kineticSensitivities
reactions = test_tube.processor.solution.reaction_equations()


def sort_top_uncertainty_weighted_sens(observables,S_matrix,top_sensitivity=10):
    sensitivitys =[[] for x in range(len(observables))]
    topSensitivities = [[] for x in range(len(observables))]  
    for i,observable in enumerate(observables):
        
        temp = S_matrix[:,:,i] 
        sort_s= pd.DataFrame(temp).reindex(pd.DataFrame(temp).abs().max().sort_values(ascending=False).index, axis=1)
        cc=pd.DataFrame(sort_s).iloc[:,:top_sensitivity]
        top_five_reactions=cc.columns.values.tolist()
        topSensitivities[i].append(top_five_reactions)
        ccn=pd.DataFrame(cc).as_matrix()
        sensitivitys[i].append(ccn) 
    return sensitivitys,topSensitivities
    
    

sensitivitys,topSensitivities = sort_top_uncertainty_weighted_sens(test_tube.observables,k_sens_new,top_sensitivity=5)

def plotting_senstivities(sensitivitys,
                          topSensitivities,
                          reactions,observables,
                          time,
                          working_directory):
    for i,observable in enumerate(observables):
        plt.figure()
        for column in range(np.shape(sensitivitys[i][0])[1]):
            plt.plot(time,sensitivitys[i][0][:,column],label=reactions[topSensitivities[i][0][column]])
            plt.xlabel('time (s)')
            plt.ylabel(observable)
            #plt.legend(loc='best',ncol=2)
            plt.legend(ncol=5, loc='upper left',bbox_to_anchor=(-.5,-.15))
            plt.savefig(working_directory+'/'+observable+'_ksens'+'.pdf', bbox_inches='tight')
            plt.title(str(test_tube.temperature)+'_'+str(test_tube.conditions))

#plotting_senstivities(sensitivitys,topSensitivities,reactions,
#                      test_tube.observables,
#                      test_tube.timeHistories[0]['time'],'/home/carly/Dropbox/Columbia')

def plotting_senstivities_for_paper(sensitivitys,
                          topSensitivities,
                          reactions,observables,
                          time,
                          working_directory):
    marker_list = ['o', 'v', '^', '<', '>', '8', 's']
    line_type = [ '-', '--', '-.', ':','*']
    label_list_gri30 = [[r'H$_2$O$_2$ + OH <=> H$_2$O + HO$_2$',
                         r'2HO$_2$ <=> H$_2$O$_2$ + O$_2$',
                         r'CH$_4$ + OH <=> CH$_3$ + H$_2$O',
                        r'HO$_2$ + OH <=> H$_2$O + O$_2$',
                        r'CH$_3$ + HO$_2$ <=> CH$_3$O + OH'],
                        [r'H$_2$O$_2$ + OH <=> H$_2$O + HO$_2$',
                         r'CH$_3$ + HO$_2$ <=> CH$_3$O + OH',
                         r'CH$_4$ + OH <=> CH$_3$ + H$_2$O',
                         r'2CH$_3$ (+M) <=> C$_2$H$_6$ (+M)',
                         r'2HO$_2$ <=> H$_2$O$_2$ + O$_2$']]

    label_list_FFCM = [[r'H$_2$O$_2$ + OH <=> H$_2$O + HO$_2$',
                         r'HO$_2$ + OH <=> H$_2$O + O$_2$',
                         r'2CH$_3$ (+M) <=> C$_2$H$_6$ (+M)',
                        r'CH$_4$ + OH <=> CH$_3$ + H$_2$O',
                        r'CH$_3$ + HO$_2$ <=> CH$_4$ + O$_2$'],
                        [r'H$_2$O$_2$ + OH <=> H$_2$O + HO$_2$',
                         r'HO$_2$ + OH <=> H$_2$O + O$_2$',
                         r'2CH$_3$ (+M) <=> C$_2$H$_6$ (+M)',
                         r'CH$_3$ + HO$_2$ <=> CH$_3$O + OH',
                         r'CH$_4$ + OH <=> CH$_3$ + H$_2$O']]
    ffcm_HO2_color_list = ['black','darkblue','blue','royalblue','cyan']
    ffcm_HO2_line_list = [ '-', '--', '-.', ':','*']
    ffcm_CH3_color_list = ['black','darkblue','blue','cyan','royalblue']
    ffcm_CH3_line_list = [ '-', '--', '-.', '*',':']
    
    
    gri30_HO2_color_list = ['black','blue','royalblue','darkblue','cyan']
    gri30_HO2_line_list = ['-','-.',':','--','*']
    gri30_CH3_color_list = ['black','cyan','royalblue','darkblue','blue']
    gri30_CH3_line_list = ['-','*',':','--','-.']

    for i,observable in enumerate(observables):
        #plt.figure()
        plt.subplot(2,1,i+1)
        counter_marker = 0
        for column in range(np.shape(sensitivitys[i][0])[1]):
            #plt.plot(time*1e3,sensitivitys[i][0][:,column],line_type[counter_marker],label=reactions[topSensitivities[i][0][column]],)
            if i==0:
                #plt.plot(time*1e3,sensitivitys[i][0][:,column],ffcm_HO2_line_list[counter_marker],label=label_list_FFCM[i][counter_marker],color=ffcm_HO2_color_list[counter_marker])
                plt.plot(time*1e3,sensitivitys[i][0][:,column],gri30_HO2_line_list[counter_marker],label=label_list_gri30[i][counter_marker],color=gri30_HO2_color_list[counter_marker])
                
            if i ==1:
                #plt.plot(time*1e3,sensitivitys[i][0][:,column],ffcm_CH3_line_list[counter_marker],label=label_list_FFCM[i][counter_marker],color=ffcm_CH3_color_list[counter_marker])
                plt.plot(time*1e3,sensitivitys[i][0][:,column],gri30_CH3_line_list[counter_marker],label=label_list_gri30[i][counter_marker],color=gri30_CH3_color_list[counter_marker])

            plt.tick_params(axis ='both', direction ='in') 

            
            if i==0:
                plt.ylabel(observable)
                #plt.ylabel(r'$\frac{\partial(H$_2$O)}{X}$')
                plt.ylabel(r'$\frac{\partial(HO_2)}{\partial(X_j)}$',fontsize=12)
                
            if i==1:
                plt.ylabel(r'$\frac{\partial(CH_3)}{\partial(X_j)}$',fontsize=12)
                plt.xlabel('Time (ms)')
            #plt.legend(loc='best',ncol=2)
            #plt.legend(ncol=5, loc='upper left',bbox_to_anchor=(-.5,-.15))
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1)
            plt.savefig(working_directory+'/'+'srinivasan_gri30'+'_ksens'+'.pdf', bbox_inches='tight')
            plt.savefig(working_directory+'/'+'srinivasan_gri30'+'_ksens'+'.svg', bbox_inches='tight')
            counter_marker+=1
            #plt.title(str(test_tube.temperature)+'_'+str(test_tube.conditions))

plotting_senstivities_for_paper(sensitivitys,topSensitivities,reactions,
                      test_tube.observables,
                      test_tube.timeHistories[0]['time'],
                      '/home/carly/Dropbox/Columbia/2020_Combustion_Symposium')