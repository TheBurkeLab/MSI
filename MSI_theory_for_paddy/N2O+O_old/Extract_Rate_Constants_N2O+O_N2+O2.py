#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_text_file(file_name):
    with open(file_name) as f:
        lines = [line.rstrip() for line in f]
    return lines

def pull_out_chebyshev_fit(lines,p_list,shape):
    chebyshev_fit=[]
    reaction_label=[]
    counter=None
    for i,l in enumerate(lines):
        if l.strip() in p_list:
            reaction_label.append(l.strip())
            temp = lines[i+2]
            temp= temp.replace('[[','')
            temp= temp.replace(']]','')
            l_split = temp.split(' ')
            temp_list = []
            for i,nmb in enumerate(l_split):
                if nmb != '':
                    nmb = nmb.replace(',','')
                    temp_list.append(float(nmb))
            
            arr = np.array(temp_list)
            arr = arr.reshape(shape)
            chebyshev_fit.append(arr)

    return reaction_label,chebyshev_fit

def build_reaction_string(p_list,temp_dict):
    reactions=[]
    for string in p_list:
        split = string.split('->')
        reactant = temp_dict[split[0]]
        product = temp_dict[split[1]]
        final = reactant +' <=> ' + product
        reactions.append(final)
    return reactions

def convert_units(dict_of_reactions,constant = np.log10(6.0221409e+23)):
    for key in dict_of_reactions.keys():
        temp = key.split('<=>')
        #print(temp[0][0])
        if '+' in temp[0] or temp[0][0]=='2':
            print(key)
            cheby_fit = dict_of_reactions[key]
            cheby_fit[0,0] = cheby_fit[0,0]+constant
            dict_of_reactions[key] = cheby_fit
        
    return dict_of_reactions

def write_out_reactions(dict_of_reactions_unites_converted,Tmin,Tmax,P_min,P_max):
    with open('readme.txt', 'w') as f:
        for i, reaction in enumerate(dict_of_reactions_unites_converted.keys()):
            f.write('#  Reaction '+str(i)+'\n')
            f.write('chebyshev_reaction(\''+reaction+'\',\n')
            f.write('Tmin='+'%s' % float('%.5g' % Tmin)+', Tmax='+'%s' % float('%.5g' % Tmax)+',\n')
            f.write('Pmin=('+'%s' % float('%.5g' % P_min)+', \'atm\'),')
            f.write('Pmax=('+'%s' % float('%.5g' % P_max)+', \'atm\'),\n')
            f.write('coeffs=[')
            arry_temp = dict_of_reactions_unites_converted[reaction]
            tempvar=np.arange(len(arry_temp[:,0]))
            for i in np.arange(len(arry_temp[:,0])):
                if i==0:
                    converted_first_row = arry_temp[i,:]
                    if converted_first_row.ndim ==1:
                        converted_first_row[0] = converted_first_row[0]+0
                    else:
                        converted_first_row[0,0] = converted_first_row[0,0] +0
                    f.write(str(list(converted_first_row)))
                elif i!=0 and i!=tempvar[-1]:
                        f.write('                           '+str(list(arry_temp[i,:])))
                if i!=tempvar[-1]:
                        f.write(',\n')
                elif i==tempvar[-1]:
                        f.write('                           '+str(list(arry_temp[i,:])))
                        f.write('])\n\n')
            f.write('\n')
            f.write('\n')
            
#needs to be in the same order as the rate constnat file
p_list =['P1->P2']

file_name = 'Chebyshev_fit.txt'
lines=load_text_file('/home/jl/MSI_Theory/PAPR-MESS_calculation/N2+O2/calculation_13/nominal/'+file_name)
reaction_label, fit = pull_out_chebyshev_fit(lines,p_list,shape=(7,1))
reactions = build_reaction_string(p_list, temp_dict = {'P1':'N2O + O', 'P2':'N2 + O2'})
dict_of_reactions = dict(zip(reactions,fit))
dict_of_reactions_unites_converted = convert_units(dict_of_reactions)

write_out_reactions(dict_of_reactions_unites_converted,Tmin=200,Tmax=4000,P_min=0.0001,P_max=100)