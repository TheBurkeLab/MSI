#%%
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import pandas as pd

plot_arrhenius_fit = False
save_fig = False

gas_cheby = ct.Solution('testing_chevy_fits.cti')
gas_original = ct.Solution('gonzalez.cti')

data_directory = '/home/jl/MSI_Theory/PAPR-MESS_calculation/N2+O2/calculation_15/nominal/T_rate.csv'

gas_cheby.reaction_equations()

reaction = 'N2O + O <=> N2 + O2'

temperature_list = np.arange(200,4000)
pressure_list = [1,5,20]
lines = ['solid', (0, (5, 10)), 'dotted']
colors = ['r','b','g','k']
cheby_index = gas_cheby.reaction_equations().index(reaction)
k=[]
for i,p in enumerate(pressure_list):
    k_temp = []
    for j,temperature in enumerate(temperature_list):
        gas_cheby.TPX = temperature,p*101325,{'Ar':1}
        k_temp.append(gas_cheby.forward_rate_constants[cheby_index]*1000)
    k.append(k_temp)
    
regular_index = gas_original.reaction_equations().index(reaction)
k2=[]
for i,p in enumerate(pressure_list):
    k_temp2 = []
    for j,temperature in enumerate(temperature_list):
        gas_original.TPX = temperature,p*101325,{'Ar':1}
        k_temp2.append(gas_original.forward_rate_constants[regular_index]*1000)
    k2.append(k_temp2)    
    
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7,7))
fig.subplots_adjust(hspace=0)

axs[0].set_title(reaction)
axs[0].set_xlabel('Temperature [K]')
axs[0].set_ylabel('k [cm^3/mol*s]')
axs[0].tick_params(axis = 'x', direction='in')
axs[0].tick_params(axis = 'y', direction='in')
    

df = pd.read_csv(data_directory, skiprows=3)

k_lists = []
T_list = []
k_list = []
signal = -4
for i, val in enumerate(df.iloc[:,0]):
      
    if val == "========================================":
        signal = i
        k_lists.append([T_list,k_list])
        T_list = []
        k_list = []
        pass
    elif i - signal <= 3:
        pass
    else:
        T_list.append(eval(val))
        k_list.append(eval(df.iloc[i,1]))
k_lists.append([T_list,k_list])

for i,p in enumerate(pressure_list):

    df_k_converted = np.multiply(k_lists[i][1],6.022e23)
    df_k_converted_0 = np.multiply(k_lists[0][1],6.022e23)

    axs[0].semilogy(k_lists[i][0], df_k_converted, linestyle = lines[i], label='raw data ' + str(p) + ' atm', color=colors[i])
        
    if i != 0:

        percent_change = 100 * np.divide(np.subtract(df_k_converted, df_k_converted_0), df_k_converted)  

        axs[1].plot(k_lists[i][0],percent_change, linestyle = lines[i],  color=colors[i])
        

if plot_arrhenius_fit == True:

    axs[0].semilogy(temperature_list,k2[0],linestyle = 'dashed',color='k',label='gonzalez')
    k2_reshaped = np.interp(k_lists[0][0], temperature_list, k2[i])
    percent_change2 = 100 * np.divide(np.subtract(k2_reshaped, df_k_converted_0), k2_reshaped)  
    axs[1].plot(k_lists[0][0],percent_change2, linestyle = 'dashed',  color='k')
        

axs[0].legend()
axs[1].set_xlabel('Temperature [K]')
axs[1].set_ylabel('Percent Change')
axs[1].tick_params(axis = 'x', direction='in')
axs[1].tick_params(axis = 'y', direction='in')


if save_fig == True:
    fig.savefig('pressure_dependence.pdf',dpi=1000,bbox_inches='tight')