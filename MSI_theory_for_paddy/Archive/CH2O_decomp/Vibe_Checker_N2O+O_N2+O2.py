#%%
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import pandas as pd

plot_chevy_fit = False
plot_arrhenius_fit = True
plot_mess_data = True

gas_cheby = ct.Solution('testing_chevy_fits.cti')
gas_original = ct.Solution('gonzalez.cti')

save_fig = False

gas_cheby.reaction_equations()

reaction = 'N2O + O <=> N2 + O2'

temperature_list = np.arange(200,4000)
pressure_list = [1]
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

color_list = ['b','k','r','g','pink']
        
for i,p in enumerate(pressure_list):
    
    axs[0].semilogy(temperature_list,k2[i],color='r',label='arrhenius')
    axs[0].semilogy(temperature_list,k[i],'--',color='b',label='chebyshev')
    # plt.xlim(1000,2500)
    # plt.ylim(.01,10000000)
    axs[0].set_title(reaction)
    axs[0].set_xlabel('Temperature [K]')
    axs[0].set_ylabel('k [cm^3/mol*s]')
    axs[0].tick_params(axis = 'x', direction='in')
    axs[0].tick_params(axis = 'y', direction='in')
    

    
df = pd.read_csv('/home/jl/MSI_Theory/PAPR-MESS_calculation/N2+O2/calculation_16/nominal/T_rate.csv', skiprows=3)
# df2 = df.tail(df.shape[0] -3)

df_temp = df['T']
df_k = df['P1->P2']

df_k_converted = np.multiply(df_k,6.022e23)

axs[0].scatter(df_temp, df_k_converted,label='raw data',color='k')
axs[0].legend()


# plt.figure()    
for i,p in enumerate(pressure_list):
    
    
    k_reshaped = np.interp(df_temp, temperature_list, k[i])
    percent_change = 100 * np.divide(np.subtract(df_k_converted, k_reshaped), df_k_converted)  
    # difference = np.subtract(k_reshaped,df_k_converted)
    
    k2_reshaped = np.interp(df_temp, temperature_list, k2[i])
    percent_change2 = 100 * np.divide(np.subtract(df_k_converted, k2_reshaped), df_k_converted)         
    
    
    # axs[0].title(reaction + ' Percent Change')
    axs[1].plot(df_temp,percent_change, 'b--')
    axs[1].plot(df_temp,percent_change2, 'r-')
    axs[1].set_xlabel('Temperature [K]')
    axs[1].set_ylabel('Percent Change')
    axs[1].tick_params(axis = 'x', direction='in')
    axs[1].tick_params(axis = 'y', direction='in')
    
if save_fig == True:
    fig.savefig('vibe_check.pdf',dpi=1000,bbox_inches='tight')