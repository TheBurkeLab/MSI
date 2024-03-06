# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 14:32:24 2017

@author: Mark Barbet
"""
"""Active parameter CTI writer.  Function takes a subset of reactions from an already modified cti file and
writes them to internal memory.  It then reads an input from a portion of the code dealing with master equation simulation 
and adds those reactions to create a complete internal mechanism. This internal mechanism is also updated with x vector values if provided.

Parameters
    ----------
    x : dict
        Dictonary of kinetic paramter updates for non theory treated reactions.
    original_cti : str
        Name of original cti.
    master_rxns : str
        Name of cti file with master reactions.
    master_index : list
        List of index(s) of master reactions in original cti file.
    MP : dict
        Dictonary of kinetic paramter updates for theory treated reactions.
    working_directory: str
        Working directory in which the cti files reside.
    file_name: str
        File name of original cti file. This will get passed to the next function which writes
        the updated cti file with the name ...original_cti_updated.

"""

import numpy as np
import cantera as ct
from ..utilities import soln2cti_py3 as ctiw


def cti_write2(x={},original_cti='',master_rxns='',master_index=[],MP={},working_directory='',file_name=''):
    #print(MP)
    flatten = lambda *n: (e for a in n
        for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))  
    #flatten master index 
    master_index = list(flatten(master_index))

    print(bool(x))
    if not original_cti:
        raise Exception('Please provide a name for the original mechanism file and try again.')
    if not master_rxns and np.any(master_index):
        raise Exception('Please provide a mechanism file for reactions analysed with master equation or leave master_index empty')
    if master_rxns and not np.any(master_index):
        raise Exception('Please provide master_index, a non-empty list of reaction numbers from original file which are analysed with master equation.')
        
    if not master_rxns and not master_index:
        master_index=np.ones(ct.Solution(original_cti).n_reactions,dtype=bool)
    elif master_rxns and np.any(master_index):
        temp=np.ones(ct.Solution(original_cti).n_reactions,dtype=bool)
        for j in np.arange(len(master_index)):
            
            temp[master_index[j]-1]=False        
        master_index=temp
    lineList=[]
    with open(original_cti) as f:
        lineList=f.readlines()        
    done=False
    count=0
    while not done or count<len(lineList):
         if 'Reaction data' in lineList[count] or 'Reaction Data' in lineList[count] or 'reaction data' in lineList[count]:
             done=True
             lineList=lineList[0:count-1]
         else:count+=1
    #with open('tempcti.cti','w') as p:
         #p.writelines(lineList)
        
   #Attempt to get rid of temp cti files:
    NewModelSpecies=ct.Species.listFromFile(original_cti)
    NewModel=ct.Solution(thermo='IdealGas',kinetics='GasKinetics',
                         species=NewModelSpecies)
        
        
    
    #NewModel=ct.Solution('tempcti.cti')

    original_mechanism=ct.Solution(original_cti)
    original_rxn_count=0
    master_rxn_eqs=[]
    if master_rxns:
        with open(master_rxns) as f:
            reactionsList=f.readlines()
        lineList=lineList+reactionsList
        with open('masterTemp.cti','w') as f:
            f.writelines(lineList)
        master_reactions=ct.Solution('masterTemp.cti')
        master_rxn_eqs=master_reactions.reaction_equations()
        
    original_rxn_eqs=[]
    for i in np.arange(original_mechanism.n_reactions):
        if master_index[i]:
            NewModel.add_reaction(original_mechanism.reaction(i))
            original_rxn_count+=1
            original_rxn_eqs.append(original_mechanism.reaction_equation(i))
#            if 'FalloffReaction' in str(type(original_mechanism.reaction(i))):
#                print(original_mechanism.reaction(i).high_rate)
#                print(original_mechanism.reaction(i).low_rate)
    if master_rxns:
        for i in np.arange(master_reactions.n_reactions):
#            print(master_reactions.reaction(i).rate)
            NewModel.add_reaction(master_reactions.reaction(i))
    
    #
    #print(master_reactions.reaction(0).rate)
    

    
    
    if x=={}:
        
        
        for j in np.arange(original_rxn_count-1):
           
           #if master_index[j]:
               #print(str(type(original_mechanism.reaction(j))),str(type(NewModel.reaction(j))))
               if 'ThreeBodyReaction' in str(type(NewModel.reaction(j))):
                   NewModel.reaction(j).rate=NewModel.reaction(j).rate
               elif 'ElementaryReaction' in str(type(NewModel.reaction(j))):
                   NewModel.reaction(j).rate=NewModel.reaction(j).rate
               elif 'FalloffReaction' in str(type(NewModel.reaction(j))):
                   NewModel.reaction(j).high_rate=NewModel.reaction(j).high_rate
                   NewModel.reaction(j).low_rate=NewModel.reaction(j).low_rate

                   if NewModel.reaction(j).falloff.type=='Troe':
                       NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                   if NewModel.reaction(j).falloff.type=='Sri':
                       NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
               elif 'ChemicallyActivatedReaction' in str(type(NewModel.reaction(j))):
                   NewModel.reaction(j).high_rate=NewModel.reaction(j).high_rate
                   NewModel.reaction(j).low_rate=NewModel.reaction(j).low_rate
                   if NewModel.reaction(j).falloff.type=='Troe':
                       NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                   if NewModel.reaction(j).falloff.type=='Sri':
                       NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
               elif 'PlogReaction' in str(type(NewModel.reaction(j))):
                   NewModel.reaction(j).rates=NewModel.reaction(j).rates
               elif 'ChebyshevReaction' in str(type(NewModel.reaction(j))):
                   NewModel.reaction(j).set_parameters(NewModel.reaction(j).Tmin,NewModel.reaction(j).Tmax,NewModel.reaction(j).Pmin,NewModel.reaction(j).Pmax,NewModel.reaction(j).coeffs)
                   
    
    E = 1 #going test for energy
    
    #change x value to update Ea/R to just Ea
    T=ct.gas_constant 
    if x!={}:
        
        for j in np.arange(original_rxn_count-1):
           #if master_index[j]:
               try:
                   if 'ThreeBodyReaction' in str(type(NewModel.reaction(j))):
                       A=NewModel.reaction(j).rate.pre_exponential_factor
                       n=NewModel.reaction(j).rate.temperature_exponent
                       Ea=NewModel.reaction(j).rate.activation_energy
                       NewModel.reaction(j).rate=ct.Arrhenius(A*np.exp(x['r'+str(j)]['A']),n+x['r'+str(j)]['n'],Ea+x['r'+str(j)]['Ea']*T)
                   elif 'ElementaryReaction' in str(type(NewModel.reaction(j))):
                       A=NewModel.reaction(j).rate.pre_exponential_factor
                       n=NewModel.reaction(j).rate.temperature_exponent
                       Ea=NewModel.reaction(j).rate.activation_energy
                       NewModel.reaction(j).rate=ct.Arrhenius(A*np.exp(x['r'+str(j)]['A']),n+x['r'+str(j)]['n'],Ea+x['r'+str(j)]['Ea']*T)
                   elif 'FalloffReaction' in str(type(NewModel.reaction(j))):
                       A=NewModel.reaction(j).high_rate.pre_exponential_factor
                       n=NewModel.reaction(j).high_rate.temperature_exponent
                       Ea=NewModel.reaction(j).high_rate.activation_energy
                       NewModel.reaction(j).high_rate=ct.Arrhenius(A*np.exp(x['r'+str(j)]['A']),n+x['r'+str(j)]['n'],Ea+x['r'+str(j)]['Ea']*T)
                       A=NewModel.reaction(j).low_rate.pre_exponential_factor
                       n=NewModel.reaction(j).low_rate.temperature_exponent
                       Ea=NewModel.reaction(j).low_rate.activation_energy
                       NewModel.reaction(j).low_rate=ct.Arrhenius(A*np.exp(x['r'+str(j)]['A']),n+x['r'+str(j)]['n'],Ea+x['r'+str(j)]['Ea']*T)
                       if NewModel.reaction(j).falloff.type=='Troe':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                       if NewModel.reaction(j).falloff.type=='Sri':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                   elif 'ChemicallyActivatedReaction' in str(type(NewModel.reaction(j))):
                       A=NewModel.reaction(j).high_rate.pre_exponential_factor
                       n=NewModel.reaction(j).high_rate.temperature_exponent
                       Ea=NewModel.reaction(j).high_rate.activation_energy
                       NewModel.reaction(j).high_rate=ct.Arrhenius(A*np.exp(x['r'+str(j)]['A']),n+x['r'+str(j)]['n'],Ea+x['r'+str(j)]['Ea']*T)
                       A=NewModel.reaction(j).low_rate.pre_exponential_factor
                       n=NewModel.reaction(j).low_rate.temperature_exponent
                       Ea=NewModel.reaction(j).low_rate.activation_energy
                       NewModel.reaction(j).low_rate=ct.Arrhenius(A*np.exp(x['r'+str(j)]['A']),n+x['r'+str(j)]['n'],Ea+x['r'+str(j)]['Ea']*T)
                       if NewModel.reaction(j).falloff.type=='Troe':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                       if NewModel.reaction(j).falloff.type=='Sri':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                   elif 'PlogReaction' in str(type(NewModel.reaction(j))):
                       
                       temp_rate=[]
                       for number, reactions in enumerate(NewModel.reaction(j).rates):
                           A = NewModel.reaction(j).rates[number][1].pre_exponential_factor
                           n = NewModel.reaction(j).rates[number][1].temperature_exponent
                           Ea = NewModel.reaction(j).rates[number][1].activation_energy
                           
                           pressure =  NewModel.reaction(j).rates[number][0]
                           temp_rate.append((pressure, ct.Arrhenius(A*np.exp(x['r'+str(j)]['A']),n+x['r'+str(j)]['n'],Ea+x['r'+str(j)]['Ea']*T)))

                       NewModel.reaction(j).rates = temp_rate     
                          
                       #NewModel.reaction(j).rates=NewModel.reaction(j).rates                      
                   elif 'ChebyshevReaction' in str(type(original_mechanism.reaction(j))):
                       NewModel.reaction(j).set_parameters(NewModel.reaction(j).Tmin,NewModel.reaction(j).Tmax,NewModel.reaction(j).Pmin,NewModel.reaction(j).Pmax,NewModel.reaction(j).coeffs)
                       
                       
               except:
                   print ('we are in the except statment in marks code',j)
                   if 'ThreeBodyReaction' in str(type(NewModel.reaction(j))):
                       NewModel.reaction(j).rate=NewModel.reaction(j).rate
                   elif 'ElementaryReaction' in str(type(NewModel.reaction(j))):
                       NewModel.reaction(j).rate=NewModel.reaction(j).rate
                   elif 'FalloffReaction' in str(type(NewModel.reaction(j))):
                       NewModel.reaction(j).high_rate=NewModel.reaction(j).high_rate
                       NewModel.reaction(j).low_rate=NewModel.reaction(j).low_rate
                       if NewModel.reaction(j).falloff.type=='Troe':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                       if NewModel.reaction(j).falloff.type=='Sri':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                   elif 'ChemicallyActivatedReaction' in str(type(NewModel.reaction(j))):
                      NewModel.reaction(j).high_rate=NewModel.reaction(j).high_rate
                      NewModel.reaction(j).low_rate=NewModel.reaction(j).low_rate
                      if NewModel.reaction(j).falloff.type=='Troe':
                          NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                      if NewModel.reaction(j).falloff.type=='Sri':
                          NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                   elif 'PlogReaction' in str(type(NewModel.reaction(j))):
                      NewModel.reaction(j).rates=NewModel.reaction(j).rates
                   elif 'ChebyshevReaction' in str(type(NewModel.reaction(j))):
                      NewModel.reaction(j).set_parameters(NewModel.reaction(j).Tmin,NewModel.reaction(j).Tmax,NewModel.reaction(j).Pmin,NewModel.reaction(j).Pmax,NewModel.reaction(j).coeffs)
    

    if MP!={}:   
        print('insdie the MP if statment')        

        #print(MP)
        for j in np.arange(original_rxn_count,NewModel.n_reactions):
               
           
               try:
                   if 'ThreeBodyReaction' in str(type(NewModel.reaction(j))):
                       A=NewModel.reaction(j).rate.pre_exponential_factor
                       n=NewModel.reaction(j).rate.temperature_exponent
                       Ea=NewModel.reaction(j).rate.activation_energy
                       NewModel.reaction(j).rate=ct.Arrhenius(A*np.exp(MP['r'+str(j)]['A']),n+MP['r'+str(j)]['n'],Ea+MP['r'+str(j)]['Ea']*E)
                   elif 'ElementaryReaction' in str(type(NewModel.reaction(j))):
                       A=NewModel.reaction(j).rate.pre_exponential_factor
                       n=NewModel.reaction(j).rate.temperature_exponent
                       Ea=NewModel.reaction(j).rate.activation_energy
                       NewModel.reaction(j).rate=ct.Arrhenius(A*np.exp(MP['r'+str(j)]['A']),n+MP['r'+str(j)]['n'],Ea+MP['r'+str(j)]['Ea']*E)
                   elif 'FalloffReaction' in str(type(NewModel.reaction(j))):
                       A=NewModel.reaction(j).high_rate.pre_exponential_factor
                       n=NewModel.reaction(j).high_rate.temperature_exponent
                       Ea=NewModel.reaction(j).high_rate.activation_energy
                       
                       NewModel.reaction(j).high_rate=ct.Arrhenius(A*np.exp(MP['r'+str(j)]['A']),n+MP['r'+str(j)]['n'],Ea+MP['r'+str(j)]['Ea']*E)
                       A=NewModel.reaction(j).low_rate.pre_exponential_factor
                       n=NewModel.reaction(j).low_rate.temperature_exponent
                       Ea=NewModel.reaction(j).low_rate.activation_energy
                       NewModel.reaction(j).low_rate=ct.Arrhenius(A*np.exp(MP['r'+str(j)]['A']),n+MP['r'+str(j)]['n'],Ea+MP['r'+str(j)]['Ea']*E)
                       if NewModel.reaction(j).falloff.type=='Troe':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                       if NewModel.reaction(j).falloff.type=='Sri':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                   elif 'ChemicallyActivatedReaction' in str(type(NewModel.reaction(j))):
                       A=NewModel.reaction(j).high_rate.pre_exponential_factor
                       n=NewModel.reaction(j).high_rate.temperature_exponent
                       Ea=NewModel.reaction(j).high_rate.activation_energy
                       NewModel.reaction(j).high_rate=ct.Arrhenius(A*np.exp(MP['r'+str(j)]['A']),n+MP['r'+str(j)]['n'],Ea+MP['r'+str(j)]['Ea']*E)
                       A=NewModel.reaction(j).low_rate.pre_exponential_factor
                       n=NewModel.reaction(j).low_rate.temperature_exponent
                       Ea=NewModel.reaction(j).low_rate.activation_energy
                       NewModel.reaction(j).low_rate=ct.Arrhenius(A*np.exp(MP['r'+str(j)]['A']),n+MP['r'+str(j)]['n'],Ea+MP['r'+str(j)]['Ea']*E)
                       if NewModel.reaction(j).falloff.type=='Troe':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                       if NewModel.reaction(j).falloff.type=='Sri':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                   elif 'PlogReaction' in str(type(NewModel.reaction(j))):
                       for number, reactions in enumerate(NewModel.reaction(j).rates):
                           A = NewModel.reaction(j)[number][1].pre_exponential_factor
                           n = NewModel.reaction(j)[number][1].temperature_exponent
                           Ea = NewModel.reaction(j)[number][1].activation_energy
                           NewModel.reaction(j)[number][1] = ct.Arrhenius(A*np.exp(MP['r'+str(j)]['A']),n+MP['r'+str(j)]['n'],Ea+MP['r'+str(j)]['Ea']*E)                      
                       NewModel.reaction(j).rates=NewModel.reaction(j).rates                      
                   elif 'ChebyshevReaction' in str(type(NewModel.reaction(j))):

                       converted = MP['r'+str(j)]/np.log(10)
                       test = NewModel.reaction(j).coeffs +converted                                                                    
                       NewModel.reaction(j).set_parameters(NewModel.reaction(j).Tmin,NewModel.reaction(j).Tmax,NewModel.reaction(j).Pmin,NewModel.reaction(j).Pmax,(test))
                       
               except:
                   print ('we are in the except statment in marks code',j)
                   
                   
                   
                   if 'ThreeBodyReaction' in str(type(NewModel.reaction(j))):
                       NewModel.reaction(j).rate=NewModel.reaction(j).rate
                   elif 'ElementaryReaction' in str(type(NewModel.reaction(j))):
                       NewModel.reaction(j).rate=NewModel.reaction(j).rate
                   elif 'FalloffReaction' in str(type(NewModel.reaction(j))):
                       NewModel.reaction(j).high_rate=NewModel.reaction(j).high_rate
                       NewModel.reaction(j).low_rate=NewModel.reaction(j).low_rate
                       if NewModel.reaction(j).falloff.type=='Troe':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                       if NewModel.reaction(j).falloff.type=='Sri':
                           NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                   elif 'ChemicallyActivatedReaction' in str(type(NewModel.reaction(j))):
                      NewModel.reaction(j).high_rate=NewModel.reaction(j).high_rate
                      NewModel.reaction(j).low_rate=NewModel.reaction(j).low_rate
                      if NewModel.reaction(j).falloff.type=='Troe':
                          NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                      if NewModel.reaction(j).falloff.type=='Sri':
                          NewModel.reaction(j).falloff=NewModel.reaction(j).falloff
                   elif 'PlogReaction' in str(type(NewModel.reaction(j))):
                      NewModel.reaction(j).rates=NewModel.reaction(j).rates
                   elif 'ChebyshevReaction' in str(type(NewModel.reaction(j))):
                      NewModel.reaction(j).set_parameters(NewModel.reaction(j).Tmin,NewModel.reaction(j).Tmax,NewModel.reaction(j).Pmin,NewModel.reaction(j).Pmax,NewModel.reaction(j).coeffs)                      
                   
               
    
    new_file=ctiw.write(NewModel, cwd=working_directory,file_name=file_name,original_cti=original_cti)
    #tab
    return new_file,original_rxn_eqs,master_rxn_eqs
