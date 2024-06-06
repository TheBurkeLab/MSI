#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_text_file(file_name):
    with open(file_name) as f:
        lines = [line.rstrip() for line in f]
    return lines

def build_dictonary_by_key_word_paramter(key_words,number_of_sets,lines):
    key_word_stored_sens_lists=[]
    counter=None
    for l in lines:
        if l in key_words:
            counter = 0
            key_word_temp_list = []
        if '[[' in l and ']]' in l:
            counter+=1
            temp_list = []
            appnd = True
            l = l.replace('[[','')
            l = l.replace(']]','')
            l_split = l.split(' ')
            for i,nmb in enumerate(l_split):
                if nmb != '':
                    nmb = nmb.replace(',','')
                    temp_list.append(float(nmb))
            key_word_temp_list.append(temp_list)
            #counter+=1
            
        if counter == number_of_sets:
            key_word_stored_sens_lists.append(key_word_temp_list)
            counter=0
    
    dict_by_key_words = dict(zip(key_words,key_word_stored_sens_lists))
    return dict_by_key_words

def build_temp_dict(key_words):
    temp_dict = {}
    for word in key_words:
        temp_dict[word] = []
    
    return temp_dict

def convert_units(dict_by_key_words,temp_dict):
    for key in dict_by_key_words:
        for i,reaction_set in enumerate(dict_by_key_words[key]):
            if 'energy' in key or 'Energy' in key:
                temp_arr = np.array(reaction_set)
                temp_arr = temp_arr.reshape((temp_arr.shape[0],1))
                zeros = np.zeros((temp_arr.shape[0],1))
                temp_arr = np.hstack((temp_arr,zeros))
                temp_arr = temp_arr* np.log(10)* 349.757
                temp_dict[key].append(temp_arr)
            else:
                temp_arr = np.array(reaction_set)
                temp_arr = temp_arr.reshape((temp_arr.shape[0],1))
                zeros = np.zeros((temp_arr.shape[0],1))
                temp_arr = np.hstack((temp_arr,zeros))
                temp_arr = temp_arr* np.log(10)
                temp_dict[key].append(temp_arr)

    return temp_dict

def convert_units_two(dict_by_key_words,temp_dict,shape=(6,3)):
    for key in dict_by_key_words:
        for i,reaction_set in enumerate(dict_by_key_words[key]):
            if 'energy' in key or 'Energy' in key:
                temp_arr = np.array(reaction_set)
                temp_arr = temp_arr.reshape(shape)
                temp_arr = temp_arr* np.log(10)* 349.757
                temp_dict[key].append(temp_arr)
            else:
                temp_arr = np.array(reaction_set)
                temp_arr = temp_arr.reshape(shape)
                temp_arr = temp_arr* np.log(10)
                temp_dict[key].append(temp_arr)

    return temp_dict

def build_empty_nested_list(temp_dict,key_words):
    empty_nested_list =[]
    for nmb_reactions in range(len(temp_dict[key_words[0]])):
        empty_nested_list.append([])
    return empty_nested_list
    
    
def breakup_by_reaction(empty_nested_list,key_words,temp_dict):
    for i in range(len(empty_nested_list)):
        for word in key_words:
            empty_nested_list[i].append(temp_dict[word][i])
    return empty_nested_list

def pull_out_key_words(lines):
    key_word=[]
    counter=None
    for i,l in enumerate(lines):
        if '====' in l:
            key_word.append(lines[i+1])

    return key_word

def build_reaction_string(p_list,temp_dict):
    reactions=[]
    
    for string in p_list:
        split = string.split('->')
        print(split)
        reactant = temp_dict[split[0]]
        product = temp_dict[split[1]]
        final = reactant +' <=> ' + product
        reactions.append(final)
    return reactions

def reduced_T( T, T_min, T_max):
    '''Calculate the reduced temperature.'''
    T = np.array(T)
    T_tilde = 2. * T ** (-1.0) - T_min ** (-1.0) - T_max ** (-1.0)
    T_tilde /= (T_max ** (-1.0) - T_min ** (-1.0))
    return T_tilde

def calc_reduced_P(P,P_min,P_max):
        
    numerator = 2*np.log10(P) - np.log10(P_min) - np.log10(P_max)
    denominator = np.log10(P_max) - np.log10(P_min)
    P_reduced = np.divide(numerator,denominator)
    return P_reduced

def calc_chevy(T,P,alpha):
    #calculate rate constants helper function
    T_reduced_list = reduced_T(T,200,2400)
    P_reduced_list = calc_reduced_P(P,.0001,10.0)
    values = np.polynomial.chebyshev.chebval2d(T_reduced_list,P_reduced_list,alpha)

    return values

def plot_sens(dictonary,path,key_words):
    for key in dictonary.keys():
        for j, sens in enumerate(dictonary[key]):
            plt.figure()
            values = calc_chevy(np.arange(500,2400),[1]*(2400-500),sens)
            temp_list = np.arange(500,2400)
            #p_list = np.arange(.0001,10.0)
            plt.plot(temp_list,values)
            plt.title('Reaction: '+key+' '+ 'Parameter: '+ key_words[j])
            plt.xlabel('Temperature (K)')
            plt.ylabel('dln(k)/dln(paramter)')
            plt.savefig(path+'/'+key+'_'+key_words[j]+'.pdf',bbox_inches='tight')
            
def plot_sens2(dictonary,path,key_words,pressure_list):
    for key in dictonary.keys():
        for j, sens in enumerate(dictonary[key]):
            plt.figure()
            for pressure in pressure_list:          
                values = calc_chevy(np.arange(290,3000),[pressure]*2710,sens)
                temp_list = np.arange(290,3000)
                plt.plot(temp_list,values,label=str(pressure)+' atm')
                plt.legend()
                
            plt.title('Reaction: '+key+' '+ 'Parameter: '+ key_words[j])
            plt.xlabel('Temperature (K)')
            plt.ylabel('dln(k)/dln(paramter)')
            plt.savefig(path+'/'+key+'_'+key_words[j]+'.pdf',bbox_inches='tight')


file_name = 'Chebyshev_sens.txt'
lines=load_text_file('/home/jl/MSI_Theory/PAPR-MESS_calculation/NO+NO/calculation_9/'+file_name)
key_words = pull_out_key_words(lines)

p_list = ['P1->P2']

build_reaction_string(p_list, temp_dict = {'P1':'N2O + O', 'P2':'2 NO'})

number_of_sets = len(p_list)

dict_by_key_words = build_dictonary_by_key_word_paramter(key_words,number_of_sets,lines)
temp_dict = build_temp_dict(key_words) 
temp_dict = convert_units(dict_by_key_words,temp_dict) # 
# temp_dict = convert_units_two(dict_by_key_words,temp_dict,shape=(7,1))

empty_nested_list = build_empty_nested_list(temp_dict,key_words)

breakup_by_reaction = breakup_by_reaction(empty_nested_list,key_words,temp_dict)
#number_of_sets = 1 

reaction_list = ['N2O + O <=> 2 NO']
test = dict(zip(reaction_list,breakup_by_reaction))

len(key_words)

print(test)

#%%
                       
plot_sens2(test,'/Users/carlylagrotta/Dropbox/Columbia/MSI/data/branching_reaction_study/Reaction Set G/plots',key_words,[.5,1,4]) 