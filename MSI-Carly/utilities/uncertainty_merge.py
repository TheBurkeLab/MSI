# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:39:06 2018

@author: justin

Create and return dataframe with kinetic parameter uncertainties for each 
    reaction.
"""

import pandas as pd
import numpy as np
import re

def react_split(reaction):
    '''
    Splits a chem. eqn into products and reactants.
    
    Parameters: chemical reaction equation (string)
    
    Returns: list of products and reactants
            - list[0] = reactants
            - list[1] = products
    '''
    sList = re.split('=|<|>', reaction)
    sList = list(filter(None, sList))
    return sList

def species_split(species):
    '''
    Removes symbols from string of chemical species
    
    Parameters: products or reactants string with symbols
    
    Returns: string containing only chemical species
    '''
    sList = re.split(' |\+|\(|\)',species)
    sList = list(filter(None, sList))
    return ' '.join(sorted(sList))  

def get_uncertainties(
        ct_eqns, 
        uncert_file = False,
        A_uncert = 5,
        N_uncert = 5,
        Ea_uncert = 1000,
        csvName = 'uncert_temp.csv'
    ):
    '''
    Creates a data frame of uncertainties based on a given set of chemical 
    equations and merges them with known uncertainties from a separate .csv
    file, if one is provided. Unknown uncertainties are given a predetermined
    value.
    
    Parameters:
        - ct_eqns (list): chemical equations in the given model (.cti) file
        - uncert_file (str): name of the .csv file that has the known 
          uncertainties
        - A_uncert (int or float): default uncertainty for unknown pre-
          exponential uncertainty
        - N_uncert (int or float): default uncertainty for unknown temperature
          exponential uncertainty
        - Ea_uncert (int or float): default uncertainty for unknown activation
          energy uncertainty
        - csvName (str): name of the output csv file
    
    Returns:
        - .csv file with reactions, reactants, products, and uncertainties
        - pandas data frame file with reactions, reactants, products, and 
          uncertainties
    '''
    
    #create df with all chemical equations from mechanism file and convert to
    # lowercase
    df_reactions = pd.DataFrame(ct_eqns)
    df_reactions['Reaction'] = df_reactions[0].apply(lambda x: x.lower())
    
    #apply react_split and species_split to prep reactions for comparison
    #create df columns for reactants and products
    #stub: this should be its own function
    df_reactions = df_reactions.merge(
        df_reactions.Reaction.apply(
            lambda x: pd.Series(
                {
                    'Reactants':react_split(x)[0],
                    'Products':react_split(x)[1]
                }
            )
        ), 
        left_index=True,
        right_index=True
    )
    
    #remove symbols so that reactants and products columns contain only species
    df_reactions['Reactants'] = df_reactions['Reactants'].apply(lambda x: species_split(x))
    df_reactions['Products'] = df_reactions['Products'].apply(lambda x: species_split(x))
    
    df_reactions['Reaction_ID'] = df_reactions.index
    
    try:
        #import uncertainty.csv file and create equivalent reactant and product columns
        df_uncert = pd.read_csv(uncert_file)
        df_uncert['Reaction'] = df_uncert['Reaction'].apply(lambda x: x.lower())
        
        #apply react_split and species_split to prep reactions for comparison
        df_uncert = df_uncert.merge(
            df_uncert.Reaction.apply(
                lambda x: pd.Series(
                    {
                        'Reactants':react_split(x)[0],
                        'Products':react_split(x)[1]
                    }
                )
            ), 
            left_index=True,
            right_index=True
        )
        
        df_uncert['Reactants'] = df_uncert['Reactants'].apply(lambda x: species_split(x))
        df_uncert['Products'] = df_uncert['Products'].apply(lambda x: species_split(x))
        
        #merge uncertainties to mech file for reactions progressing in SAME 
        # direction
        df_combined  = pd.merge(
            df_reactions[[
                'Reaction_ID',
                'Products',
                'Reactants',
                'Reaction'
            ]],
            df_uncert[[
                'Uncertainty A (unit)',
                'Uncertainty N (unit)',
                'Uncertainty Ea (unit)',
                'Products',
                'Reactants'
            ]],
            how='left',
            left_on=['Products','Reactants'],
            right_on=['Products','Reactants']
        )
        
        #merge uncertainties to mech file for reactions progressing in OPPOSITE
        # direction
        df_combined2 = pd.merge(
            df_combined[df_combined.isnull().any(axis=1)][[
                'Reaction_ID',
                'Products',
                'Reactants',
                'Reaction'
            ]],
            df_uncert[[
                'Uncertainty A (unit)',
                'Uncertainty N (unit)',
                'Uncertainty Ea (unit)',
                'Products',
                'Reactants'
            ]],
            how='left',
            left_on=['Reactants','Products'],
            right_on=['Products','Reactants']
        )
        
        df_combined2.drop(['Products_y','Reactants_y'],axis=1, inplace=True)
        df_combined2.rename(columns={'Products_x':'Products','Reactants_x':'Reactants'}, inplace=True)
        
        #merge both combined uncertainties dfs to create final uncertainties df
        df_combined3 = pd.merge(
            df_combined,
            df_combined2[[
                'Reaction_ID',
                'Uncertainty A (unit)',
                'Uncertainty N (unit)',
                'Uncertainty Ea (unit)'
            ]],
            how='left',
            left_on=['Reaction_ID'],
            right_on=['Reaction_ID']
        )
        
        #coalesce uncertainty columns to keep first non-null value per row
        for u in ['A','N','Ea']:
            df_combined3['Uncertainty '+u+' (unit)_x'] = df_combined3[
                'Uncertainty '+u+' (unit)_x'
            ].fillna(df_combined3['Uncertainty '+u+' (unit)_y'])
        
        df_combined3.drop(
            [
                'Uncertainty A (unit)_y',
                'Uncertainty N (unit)_y',
                'Uncertainty Ea (unit)_y'
            ],
            inplace=True,
            axis=1
        )
        
        df_combined3.rename(
            columns={
                'Uncertainty A (unit)_x':'Uncertainty A (unit)',
                'Uncertainty N (unit)_x':'Uncertainty N (unit)',
                'Uncertainty Ea (unit)_x':'Uncertainty Ea (unit)'
            },
            inplace=True
        )
        
        df_combined3.drop_duplicates(inplace=True)
    
    #if no uncertainty reference file exists...
    except:
        if uncert_file:
            raise FileNotFoundError('No file named %s found' %(uncert_file))
        for u in ['A','N','Ea']:
            df_combined3 = df_reactions
            df_combined3['Uncertainty '+u+' (unit)'] = np.nan
    
    #assign default values to null uncertainty fields
    for u in ['A','N','Ea']:
        col = 'Uncertainty '+u+' (unit)'
        if u == 'A':
            df_combined3[col].fillna(A_uncert, inplace=True)
        elif u == 'N':
            df_combined3[col].fillna(N_uncert, inplace=True)
        elif u == 'Ea':
            df_combined3[col].fillna(Ea_uncert, inplace=True)
    
    df_combined3.set_index('Reaction_ID', inplace=True)
    
    df_combined3.to_csv(csvName)  
    
    return df_combined3





