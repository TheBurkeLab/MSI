#%%
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import pandas as pd

gas_cheby = ct.Solution('testing_chevy_fits.cti')
gas_original = ct.Solution('comparision_rate_constants.cti')

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
    
        
color_list = ['b','k','r','g','pink']
        
for i,p in enumerate(pressure_list):
    
    plt.semilogy(temperature_list,k2[i],color='r',label='glarborg')
    plt.semilogy(temperature_list,k[i],'--',color='b',label='mess')
    # plt.xlim(1000,2500)
    # plt.ylim(.01,10000000)
    plt.title(reaction)
    
# plt.figure()    
# for i,p in enumerate(pressure_list):
#     plt.plot(temperature_list,np.array(k2[i])/np.array(k[i]),color=color_list[i])
    
df = pd.read_csv('/home/jl/MSI_Theory/PAPR-MESS_calculation/N2+O2/calculation_11/nominal/T_rate_plot.csv')
df_temp = df['T']
df_raw = df['P1->P2']

plt.scatter(df_temp, np.multiply(df_raw,6.022e23),label='raw',color='k')
plt.legend()
plt.savefig('Vibe_Checker_N2O+O_N2+O2.pdf',dpi=1000,bbox_inches='tight')