# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:09:43 2018

@author: Mark Barbet
"""

import pandas as pd
import cantera as ct


def uncertainty_template(cti_file,output_csv,headers=['Reaction','Uncertainty A (unit)','Uncertainty N (unit)','Uncertainty Ea (unit)']):
    gas=ct.Solution(cti_file)
    outputData=pd.DataFrame(columns=headers)
    outputData[headers[0]]=gas.reaction_equations()
    outputData.to_csv(output_csv, sep=',',index=False)
    
