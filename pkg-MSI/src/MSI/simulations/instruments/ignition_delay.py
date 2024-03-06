import cantera as ct
from .. import simulation as sim
from ...cti_core import cti_processor as ctp
import pandas as pd
import numpy as np
import time
import copy
import re

#from . import batch_reactor as st

import MSI.simulations.instruments.batch_reactor as st
import MSI.simulations.instruments.RCM as rcmsim


import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

class ignition_delay(sim.Simulation):
    
    
    def __init__(self,pressure:float,temperature:float,observables:list,
                 kineticSens:int,physicalSens:int,conditions:dict,thermalBoundary,
                 mechanicalBoundary,
                 processor:ctp.Processor=None,cti_path="", 
                 save_physSensHistories=0,moleFractionObservables:list=[],
                 absorbanceObservables:list=[],concentrationObservables:list=[],
                 fullParsedYamlFile:dict={}, save_timeHistories:int=0,
                 log_file=True,log_name='log.txt',timeshift:float=0.0,initialTime:float=0.0,
                 finalTime:float=1.0,target:str='temperature',target_type:str='max derivative',n_processors:int=2,
                 volumeTrace=''):
        
        
        if processor!=None and cti_path!="":
            print("Error: Cannot give both a processor and a cti file path, pick one")
        elif processor==None and cti_path=="":
            print("Error: Must give either a processor or a cti file path")
        if processor != None:
            self.processor = processor 
        elif cti_path!="":
            self.processor = ctp.Processor(cti_path)
        self.n_processors=n_processors
        self.pressure=pressure
        self.temperature=temperature
        self.observables=observables
        self.kineticSens=kineticSens
        self.physicalSens=physicalSens
        self.conditions=conditions
        self.cti_path=cti_path
        self.thermalBoundary = thermalBoundary
        self.mechanicalBoundary=mechanicalBoundary
        self.kineticSensitivities= None
        self.experimentalData = None
        self.concentrationObservables = concentrationObservables
        self.moleFractionObservables = moleFractionObservables
        self.absorbanceObservables = absorbanceObservables
        self.fullParsedYamlFile =  fullParsedYamlFile
        #self.energycon='off'
        self.timeshift=timeshift
        self.timeHistory = None
        self.experimentalData = None
        self.initialTime=initialTime
        self.finalTime=finalTime
        self.log_name=log_name
        self.log_file=log_file
        self.ignitionDelayObservables=['tau']
        self.volumeTrace = volumeTrace
        #self.yaml_file=yaml_file
        if save_timeHistories == 1:
            self.timeHistories=[]
            self.timeHistoryInterpToExperiment = None
            self.pressureAndTemperatureToExperiment = None
        else:
            self.timeHistories=None
        if save_physSensHistories == 1:
            self.physSensHistories = []
        self.setTPX()
        self.dk = 0.01
        self.solution=None
        self.target=target
        self.target_type=target_type
    

    def run_RCM(self,args=[],temp_proc=None):
        
        if not args:
            rcm = rcmsim.RCM(pressure =self.pressure,
                         temperature = self.temperature,
                         observables = self.observables,
                         kineticSens = 0,
                         physicalSens = 0,
                         conditions = self.conditions,
                         initialTime = self.initialTime,
                         finalTime = self.finalTime,
                         thermalBoundary = self.thermalBoundary,
                         mechanicalBoundary = self.mechanicalBoundary,
                         processor = self.processor,
                         save_timeHistories = 1,
                         save_physSensHistories = 0,
                         moleFractionObservables = self.moleFractionObservables,
                         concentrationObservables = self.concentrationObservables,
                         fullParsedYamlFile = self.fullParsedYamlFile,
                         time_shift_value = self.timeshift,
                         volumeTrace = self.volumeTrace)
            rcm.run()
            return rcm


    
    def run_batchReactor_ksens(self,observables=['temperature'],temp_proc=None):
        
        batch_reactor = st.batchReactor(pressure =self.pressure,
                         temperature = self.temperature,
                         observables = observables,
                         kineticSens = 1,
                         physicalSens = 0,
                         conditions = self.conditions,
                         initialTime = self.initialTime,
                         finalTime = self.finalTime,
                         thermalBoundary = self.thermalBoundary,
                         mechanicalBoundary = self.mechanicalBoundary,
                         processor = self.processor,
                         save_timeHistories = 1,
                         save_physSensHistories = 0,
                         moleFractionObservables = self.moleFractionObservables,
                         concentrationObservables = self.concentrationObservables,
                         fullParsedYamlFile = self.fullParsedYamlFile,
                         time_shift_value = self.timeshift,
                         rtol=1e-9,atol=1e-15,rtol_sens=0.0001,atol_sens=1e-6)
        
        batch_reactor.run()
        return batch_reactor
    
    def run_batchReactor(self,args=[],temp_proc=None):
        
        if not args:
            batch_reactor = st.batchReactor(pressure =self.pressure,
                         temperature = self.temperature,
                         observables = self.observables,
                         kineticSens = 0,
                         physicalSens = 0,
                         conditions = self.conditions,
                         initialTime = self.initialTime,
                         finalTime = self.finalTime,
                         thermalBoundary = self.thermalBoundary,
                         mechanicalBoundary = self.mechanicalBoundary,
                         processor = self.processor,
                         save_timeHistories = 1,
                         save_physSensHistories = 0,
                         moleFractionObservables = self.moleFractionObservables,
                         concentrationObservables = self.concentrationObservables,
                         fullParsedYamlFile = self.fullParsedYamlFile,
                         time_shift_value = self.timeshift)
        elif args:
            batch_reactor = st.batchReactor(pressure =args[0],
                         temperature = args[1],
                         observables = args[2],
                         kineticSens = 0,
                         physicalSens = 0,
                         conditions = args[4],
                         initialTime = args[5],
                         finalTime = args[6],
                         thermalBoundary = args[7],
                         mechanicalBoundary = args[8],
                         processor = temp_proc,
                         save_timeHistories = 1,
                         save_physSensHistories = 0,
                         moleFractionObservables = args[9],
                         concentrationObservables = args[10],
                         fullParsedYamlFile = args[11],
                         time_shift_value = args[12])
        
        batch_reactor.run()
        return batch_reactor
    
    
    def run_single(self):
        
        tic=time.time()
        
        #check if shock tube or rcm
       
        if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
            s=self.run_batchReactor()
            
        elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
            s=self.run_RCM()
        
        self.timehistory=copy.deepcopy(s.timeHistory)
        delay=None
        if re.match('^[Mm]ax[ -][Dd]erivative$',self.target_type):
            
            if re.match('[Tt]emperature',self.target):
                
                delay=self.ig_dTdt(self.timehistory)
                
            elif re.match('[Pp]ressure',self.target):
                if re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                                    
                    delay,ignition_temperature,ignition_pressure,end_of_compression_time=self.ig_dPdt(self.timehistory,return_ignition_temp_and_pressure=True)
                else:
                    delay=self.ig_dPdt(self.timehistory)
                                    
            elif not re.match('[Tt]emperature',self.target) and not re.match('[Pp]ressure',self.target):
                delay=self.ig_dXdt(self.timehistory,self.target)
                
                
        elif re.match('[Mm]ax[ -][Vv]alue', self.target_type):
            if re.match('[Tt]emperature',self.target):
                 delay=self.ig_Tmax(self.timehistory)
            elif re.match('[Pp]ressure',self.target):
                 delay=self.ig_Pmax(self.timehistory)
            elif not re.match('[Tt]emperature',self.target) and not re.match('[Pp]ressure',self.target):
                 delay=self.ig_Xmax(self.timehistory,self.target)
        
        
        elif re.match('[Ss]pecific[ -][Vv]alue',self.target_type):
           
            split_item = self.target.split(' ')
            
            if len(split_item) ==2:
                split_item.append('')
            
            if re.match('[Tt]emperature',split_item[0]):
                delay = self.ig_T_specific_value(self.timehistory, float(split_item[1]), split_item[2])
            elif re.match('[Pp]ressure',split_item[0]):
                delay = self.ig_P_specific_value(self.timehistory, float(split_item[1]), split_item[2])
            elif not re.match('[Tt]emperature',split_item[0]) and not re.match('[Pp]ressure',split_item[0]):
                
                delay = self.ig_X_specific_value(self.timehistory, split_item[0], float(split_item[1]),unit = split_item[2])
       
        
       
        elif re.match('^[Mm]ax[ -][Dd]erivative[ -][Tt]angent[ -][Ii]ntercept$',self.target_type):

            if re.match('[Tt]emperature',self.target):
                delay = self.ig_dTdt_int(self.timehistory)
            elif re.match('[Pp]ressure',self.target):
                delay =self.ig_dPdt_int(self.timehistory)
            elif not re.match('[Tt]emperature',self.target) and not re.match('[Pp]ressure',self.target):
                delay = self.ig_dXdt_int(self.timehistory,self.target)                
                
        sens=[]        
        #self.direct_ksens('a',dk=self.dk)
        if self.kineticSens:   
                ##############################################################
                #Function to calculate sensitivity
                #Replace to use different method
                

                #sens=self.direct_ksens('a',dk=self.dk)
                #sens=self.BFM(delay)

                #sens=self.direct_ksens('a',dk=self.dk)
                sens=self.BFM(delay,simulation_type = '[Bb]atch[ -][Rr]eactor')
                #sens=self.BFM_pool(delay,self.n_processors)
                ##############################################################
                
                
                dfs = [pd.DataFrame() for x in range(len(self.ignitionDelayObservables))]
                dfs[0] = dfs[0].append(((pd.DataFrame(sens)).transpose()),ignore_index=True)
                numpyMatrixsksens = [dfs[dataframe].values for dataframe in range(len(dfs))]
                self.kineticSensitivities = np.dstack(numpyMatrixsksens)
                
        
        toc=time.time()
        #stub
        #print('Simulation took '+str(round(toc-tic,9))+' seconds.')      
        if re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
            columns=['temperature','pressure']
            for species in self.conditions.keys():
                columns=columns+[species]
            columns=columns+['delay']
            columns = columns+['ignition_temperature','ignition_pressure','end_of_compression_time']
            data=pd.DataFrame(columns=columns)    
            data['temperature']=[self.temperature]
            data['pressure']=[self.pressure]
            data['ignition_temperature'] = ignition_temperature
            data['ignition_pressure'] = ignition_pressure
            data['end_of_compression_time'] = end_of_compression_time
            for species in self.conditions.keys():
                data[species]=[self.conditions[species]]
            data['delay']=[delay]
        else:
            
            columns=['temperature','pressure']
            for species in self.conditions.keys():
                columns=columns+[species]
            columns=columns+['delay']
            data=pd.DataFrame(columns=columns)    
            data['temperature']=[self.temperature]
            data['pressure']=[self.pressure]
            for species in self.conditions.keys():
                data[species]=[self.conditions[species]]
            data['delay']=[delay]
        #('Delay: '+str(delay))
        if self.kineticSens:  
            #print(data)
            return data,self.kineticSensitivities
        else:
            return data,[]
            
        
        
                
        
    def ig_dTdt(self,data):
        delay=[]
        #print(type(data))
        tt=data['time'].values
        TT=data['temperature'].values
        #PP=data['pressure'].values
        dTdt=np.zeros(np.array(tt).shape,np.float)
        dTdt[0:-1]=np.diff(TT)/np.diff(tt)
        dTdt[-1]=(TT[-1] - TT[-2])/(tt[-1] - tt[-2])
        delay=tt[np.argmax(dTdt)]
        return delay

        
    def ig_dPdt(self,data,return_ignition_temp_and_pressure=False):
        
        delay=[]        
        tt=data['time'].values
        PP=data['pressure'].values
        dPdt=np.zeros(np.array(tt).shape,np.float)
        dPdt[0:-1]=np.diff(PP)/np.diff(tt)
        dPdt[-1]=(PP[-1] - PP[-2])/(tt[-1] - tt[-2])
        delay=tt[np.argmax(dPdt)]
        if return_ignition_temp_and_pressure==False:           
            return delay
        elif return_ignition_temp_and_pressure==True:
            value_to_filter_below= delay-.01
            filtered_df = copy.deepcopy(data)
            filtered_df[['time']] = filtered_df[filtered_df[['time']] < value_to_filter_below][['time']]
            filtered_df=filtered_df.dropna()
            pressure_max_index = filtered_df['pressure'].idxmax()
            
            
            igntion_temp = filtered_df['temperature'][pressure_max_index]
            ignition_pressure = filtered_df['pressure'][pressure_max_index] 
            end_of_compression_time = filtered_df['time'][pressure_max_index]                                
            return delay,igntion_temp,ignition_pressure,end_of_compression_time
        
    def ig_dXdt(self,data,target):
        
        delay=[]
        tt=data['time'].values
        XX=data[target].values
        dXdt=np.zeros(np.array(tt).shape,np.float)
        dXdt[0:-1]=np.diff(XX)/np.diff(tt)
        dXdt[-1]=(XX[-1] - XX[-2])/(tt[-1] - tt[-2])
        delay=tt[np.argmax(dXdt)]
        return delay
    
    def ig_Xmax(self,data,target):
        delay=[]
        tt=data['time'].values
        XX=data[target].values
        delay=tt[np.argmax(XX)]
        
        
        return delay
    
    def ig_Tmax(self,data):
        delay=[]
        tt=data['time'].values
        XX=data['temperature'].values
        delay=tt[np.argmax(XX)]
        
        return delay

    def ig_Pmax(self,data):
        delay=[]
        tt=data['time'].values
        XX=data['pressure'].values
        delay=tt[np.argmax(XX)]
        
        return delay


    def ig_dTdt_int(self,data):
        delay=[]
        #print(type(data))
        tt=data['time'].values
        TT=data['temperature'].values
        #PP=data['pressure'].values
        dTdt=np.zeros(np.array(tt).shape,np.float)
        dTdt[0:-1]=np.diff(TT)/np.diff(tt)
        dTdt[-1]=(TT[-1] - TT[-2])/(tt[-1] - tt[-2])
        slope = np.max(dTdt)
        y2 = TT[np.argmax(dTdt)]
        x2 = tt[np.argmax(dTdt)]
        b = y2 - slope*x2
        delay = -b/slope
        #delay=tt[np.argmax(dTdt)]
        return delay

        
    def ig_dPdt_int(self,data):
        
        delay=[]        
        tt=data['time'].values
        PP=data['pressure'].values
        dPdt=np.zeros(np.array(tt).shape,np.float)
        dPdt[0:-1]=np.diff(PP)/np.diff(tt)
        dPdt[-1]=(PP[-1] - PP[-2])/(tt[-1] - tt[-2])
        slope = np.max(dPdt)
        y2 = PP[np.argmax(dPdt)]
        x2 = tt[np.argmax(dPdt)]
        b = y2 - slope*x2
        delay = -b/slope
        
        
        #if return_ignition_temp_and_pressure==False:           
        return delay
        # elif return_ignition_temp_and_pressure==True:
        #     value_to_filter_below= delay-.01
        #     filtered_df = copy.deepcopy(data)
        #     filtered_df[['time']] = filtered_df[filtered_df[['time']] < value_to_filter_below][['time']]
        #     filtered_df=filtered_df.dropna()
        #     pressure_max_index = filtered_df['pressure'].idxmax()
            
            
        #     igntion_temp = filtered_df['temperature'][pressure_max_index]
        #     ignition_pressure = filtered_df['pressure'][pressure_max_index] 
        #     end_of_compression_time = filtered_df['time'][pressure_max_index]                                
        #     return delay,igntion_temp,ignition_pressure,end_of_compression_time
        
    def ig_dXdt_int(self,data,target):
        
        delay=[]
        tt=data['time'].values
        XX=data[target].values
        dXdt=np.zeros(np.array(tt).shape,np.float)
        dXdt[0:-1]=np.diff(XX)/np.diff(tt)
        dXdt[-1]=(XX[-1] - XX[-2])/(tt[-1] - tt[-2])
        slope = np.max(dXdt)
        y2 = XX[np.argmax(dXdt)]
        x2 = tt[np.argmax(dXdt)]
        b = y2 - slope*x2
        delay = -b/slope
        return delay


    def ig_X_specific_value(self,data,target,value,unit=''):
        
        tt=data['time'].values
        XX=data[target].values
        #if re.match('',unit):
        if unit =='':
            value = value
        elif re.match('[Pp][Pp][Mm]',unit) or re.match('[Pp]arts[ -][Pp]er[ -][Mm]illion',unit):
            value = value/1e6
        #elif re.match('mol/cm^3',unit):
        elif unit == 'mol/cm^3':
            temperature_matrix = data['temperature'].values
            pressure_matrix = data['pressure'].values
            species_array = data[target].values
            
            concentration = np.true_divide(1,temperature_matrix.flatten())*pressure_matrix.flatten()
            concentration *= (1/(8.314e6))*species_array.flatten()     
            max_value_index_in_array = np.argmax(concentration) + 1
            slice_concentration = concentration[0:max_value_index_in_array]
            slice_time = tt[0:max_value_index_in_array]
            
            
            
            
                                                 
                                                      
            
        
        
        delay=[]

        delay = np.interp(value,slice_concentration,slice_time)
        
        return delay
    
    def ig_T_specific_value(self,data,value,unit):
        if re.match('K',unit):
            value = value
        elif re.match('C',unit):
            value = value +273.15         
            
            
        delay=[]
        tt=data['time'].values
        XX=data['temperature'].values
        
        
        delay = np.interp(value,XX,tt)

        
        return delay

    def ig_P_specific_value(self,data,value,unit):
        if re.match('atm',unit):
            value = value*ct.one_atm
        elif re.match('[Pp][a]',unit) or re.match('[Pp]ascal',unit):
            value = value
        elif re.match('[Pp][Ss][Ii]',unit):
            value = 6894.76*value
        elif re.match('[Tt]orr',unit):
            value = 133.322*value
        
        delay=[]
        tt=data['time'].values
        XX=data['pressure'].values
        delay = np.interp(value,XX,tt)        
        return delay


    def sensitivityCalculation(self,originalValues,newValues,dk=.01):
        sensitivity=(np.log(newValues)-np.log(originalValues))/dk
                           
        return sensitivity
    

    def direct_ksens(self,nominal,dk=0.01,observable=['temperature']):
        
        temp_st=self.run_batchReactor_ksens(observables=observable)
        
        sens=temp_st.kineticSensitivities
        
        nom_soln=np.array(temp_st.timeHistory['temperature'])
        
        nom_delay=self.ig_dTdt(temp_st.timeHistory)
        import matplotlib.pyplot as plt
        plt.plot(temp_st.timeHistory['time'],nom_soln,label='Nominal T')
        #print(sens[:,0,0])
        ksens=np.zeros(temp_st.processor.solution.n_reactions)
        for i in range(temp_st.processor.solution.n_reactions):
            if i==0:
                self.processor.solution.set_multiplier(1+0.001,0)
                temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                self.processor.solution.set_multiplier(1.0,0)
            tempsen=sens[:,i,0]
            sens_add=0.0
            Tprime=copy.deepcopy(nom_soln)
            # for j in range(len(sens[:,i,0])):
            #     sens_add=sens_add+sens[j,i,0]*0.05
            #     Tprime[j]=Tprime[j]+sens_add
            
            Tprime=nom_soln+np.multiply(tempsen,0.001)
            tempdata=pd.DataFrame(columns=['time','temperature'])
            tempdata['time']=temp_st.timeHistory['time']
            tempdata['temperature']=Tprime
            
            delayprime = self.ig_dTdt(tempdata)
            ksens[i]=self.sensitivityCalculation(nom_delay,delayprime,0.001)
            #print(tempsen)
            if i==0:
                plt.plot(tempdata['time'],tempdata['temperature'],':',label=r'Fast T prime')
                plt.plot(temp_history['time'],temp_history['temperature'],'-.',label=r'Brute Force T prime')
                plt.legend()
                #plt.xlim((0.7,0.9))
                #plt.ylim((0,2000))
                plt.figure()
                plt.plot(tempdata['time'],np.array(tempdata['temperature'])-nom_soln)
                plt.figure()
                plt.plot(tempdata['time'],tempsen)
                #print('Oy '+str(ksens[i]))
        #print(ksens)
        return ksens
        #print(temp_st.kineticSensitivities)
        #print(temp_st.kineticSensitivities.shape)
        
    
        
    def BFM(self,nominal,simulation_type = ''):
         
         sens=np.zeros(self.processor.solution.n_reactions)
         if re.match('^[Mm]ax [Dd]erivative$',self.target_type):
            if re.match('[Tt]emperature',self.target):
                
                for i in range(self.processor.solution.n_reactions):
                    #stub
                    print('Solving kinetic sensitivity for reaction '+str(i+1))
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)

                    delay=self.ig_dTdt(temp_history)
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    self.processor.solution.set_multiplier(1.0,i)
                    
            elif re.match('[Pp]ressure',self.target):
                
                for i in range(self.processor.solution.n_reactions):
                    #stub
                    #print('Solving kinetic sensitivity for reaction '+str(i+1))
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)                    
                    delay=self.ig_dPdt(temp_history)
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    self.processor.solution.set_multiplier(1.0,i)
                    
            elif not re.match('[Tt]emperature',self.target) and not re.match('[Pp]ressure',self.target):
                
                for i in range(self.processor.solution.n_reactions):
                    print('Solving kinetic sensitivity for reaction '+str(i+1))
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)                    
                    delay=self.ig_dXdt(temp_history,self.target)
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    #print(nominal,delay,sens[i])
                    self.processor.solution.set_multiplier(1.0,i)
         elif re.match('[Mm]ax[ -][Vv]alue',self.target_type):
            if re.match('[Tt]emperature',self.target):
                
                for i in range(self.processor.solution.n_reactions):
                    print('Solving kinetic sensitivity for reaction '+str(i+1))
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)

                    delay=self.ig_Tmax(temp_history)
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    self.processor.solution.set_multiplier(1.0,i)
                    
            elif re.match('[Pp]ressure',self.target):
                
                for i in range(self.processor.solution.n_reactions):
                    print('Solving kinetic sensitivity for reaction '+str(i+1))
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)                    
                    delay=self.ig_Pmax(temp_history)
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    self.processor.solution.set_multiplier(1.0,i)
                    
            elif not re.match('[Tt]emperature',self.target) and not re.match('[Pp]ressure',self.target):
                
                for i in range(self.processor.solution.n_reactions):
                    print('Solving kinetic sensitivity for reaction '+str(i+1))
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)                    
                    delay=self.ig_Xmax(temp_history,self.target)
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    #print(nominal,delay,sens[i])
                    self.processor.solution.set_multiplier(1.0,i)             
         elif re.match('^[Mm]ax[ -][Dd]erivative[- ][Tt]angent[ -][Ii]ntercept$',self.target_type):
             
            if re.match('[Tt]emperature',self.target):
                
                for i in range(self.processor.solution.n_reactions):
                    print('Solving kinetic sensitivity for reaction '+str(i+1))

                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)

                    delay=self.ig_dTdt_int(temp_history)
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    self.processor.solution.set_multiplier(1.0,i)
                    
            elif re.match('[Pp]ressure',self.target):
                
                for i in range(self.processor.solution.n_reactions):
                    print('Solving kinetic sensitivity for reaction '+str(i+1))
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)                    
                    delay=self.ig_dPdt_int(temp_history)
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    self.processor.solution.set_multiplier(1.0,i)
                    
            elif not re.match('[Tt]emperature',self.target) and not re.match('[Pp]ressure',self.target):
                
                for i in range(self.processor.solution.n_reactions):
                    #stub
                    #print('Solving kinetic sensitivity for reaction '+str(i+1))
                    #print('INSIDE HERE')
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)                    
                    delay=self.ig_dXdt_int(temp_history,self.target)
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    #print(nominal,delay,sens[i])
                    self.processor.solution.set_multiplier(1.0,i)   
         elif re.match('[Ss]pecific[ -][Vv]alue',self.target_type):
            split_item = self.target.split(' ')
            if len(split_item) ==2:
                split_item.append('')
                
            if re.match('[Tt]emperature',split_item[0]):
                
                for i in range(self.processor.solution.n_reactions):
                    print('Solving kinetic sensitivity for reaction '+str(i+1))
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)

                    delay=self.ig_T_specific_value(temp_history, split_item[1], split_item[2])
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    self.processor.solution.set_multiplier(1.0,i)
                    
            elif re.match('[Pp]ressure',split_item[0]):
                
                for i in range(self.processor.solution.n_reactions):
                    print('Solving kinetic sensitivity for reaction '+str(i+1))
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)                    
                    delay=self.ig_P_specific_value(temp_history, split_item[1], split_item[2])
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    self.processor.solution.set_multiplier(1.0,i)
                    
            elif not re.match('[Tt]emperature',split_item[0]) and not re.match('[Pp]ressure',split_item[0]):
                
                for i in range(self.processor.solution.n_reactions):
                    print('Solving kinetic sensitivity for reaction '+str(i+1))
                    self.processor.solution.set_multiplier(1+self.dk,i)
                    if re.match('[Bb]atch[ -][Rr]eactor',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_batchReactor().timeHistory)
                    elif re.match('[Rr][Cc][Mm]',self.fullParsedYamlFile['simulationType']):
                        temp_history=copy.deepcopy(self.run_RCM().timeHistory)    
                    delay = self.ig_X_specific_value(temp_history, split_item[0], split_item[1],unit=split_item[2])
                    
                    sens[i]=self.sensitivityCalculation(nominal,delay,self.dk)
                    #print(nominal,delay,sens[i])
                    self.processor.solution.set_multiplier(1.0,i)   
                                        
                    
                    
                    
         return sens
     
        
        
        
        
    def BFM_pool(self,nominal,n_procs):
         info=[]
         sens=[]
         args=(self.pressure,self.temperature,self.observables,0,self.conditions,self.initialTime,self.finalTime,
               self.thermalBoundary,self.mechanicalBoundary,self.moleFractionObservables,self.concentrationObservables,
               self.fullParsedYamlFile,self.timeshift)
         for i in range(self.processor.solution.n_reactions):
             info.append([self.dk,self.processor.cti_path,nominal,i,self.target_type,args,self.target])
         def solver(info):
             if re.match('[Mm]ax [Dd]erivative',info[4]):
                 if re.match('[Tt]emperature',info[6]):
             
                     print('Solving kinetic sensitivity for reaction '+str(info[3]+1))
                     temp_processor=ctp.Processor(info[1])
                     temp_processor.solution.set_multiplier(1+info[0],info[3])
                     temp_history=copy.deepcopy(self.run_batchReactor(info[5],temp_processor).timeHistory)                    
                     delay=self.ig_dTdt(temp_history)
                     sens=self.sensitivityCalculation(nominal,delay,info[0])
                     
                 elif re.match('[Pp]ressure',info[6]):
             
                     print('Solving kinetic sensitivity for reaction '+str(info[3]+1))
                     temp_processor=ctp.Processor(info[1])
                     temp_processor.solution.set_multiplier(1+info[0],info[3])
                     temp_history=copy.deepcopy(self.run_batchReactor(info[5],temp_processor).timeHistory)                    
                     delay=self.ig_dPdt(temp_history)
                     sens=self.sensitivityCalculation(nominal,delay,info[0])
                     
                 elif not re.match('[Tt]emperature',info[6]) and not re.match('[Pp]ressure',info[6]):
             
                     print('Solving kinetic sensitivity for reaction '+str(info[3]+1))
                     temp_processor=ctp.Processor(info[1])
                     temp_processor.solution.set_multiplier(1+info[0],info[3])
                     temp_history=copy.deepcopy(self.run_batchReactor(info[5],temp_processor).timeHistory)                    
                     delay=self.ig_dXdt(temp_history,info[6])
                     sens=self.sensitivityCalculation(nominal,delay,info[0])
             return sens 
                     
                     
                     
         pool = ThreadPool(n_procs) 
         #results = results+pool.map(solver,conditionsTups)
         sens=pool.map(solver,info)
         #sens=np.zeros(self.processor.solution.n_reactions)
         
         return sens
        
class ignition_delay_wrapper(sim.Simulation):
    
    def __init__(self,pressures,temperatures,observables:list,
                 kineticSens:int,physicalSens:int,conditions,thermalBoundary,
                 mechanicalBoundary,
                 processor:ctp.Processor=None,cti_path="", 
                 save_physSensHistories=0,moleFractionObservables:list=[],
                 absorbanceObservables:list=[],concentrationObservables:list=[],
                 fullParsedYamlFile:dict={}, save_timeHistories:int=0,
                 log_file=True,log_name='log.txt',timeshift:float=0.0,initialTime:float=0.0,
                 finalTime:float=1.0,target:str='temperature',
                 target_type:str='max derivative',n_processors:int=2, 
                 volumeTraceList=[]):
        
        
        
        
        if processor!=None and cti_path!="":
            print("Error: Cannot give both a processor and a cti file path, pick one")
        elif processor==None and cti_path=="":
            print("Error: Must give either a processor or a cti file path")
        if processor != None:
            self.processor = processor 
        elif cti_path!="":
            self.processor = ctp.Processor(cti_path)
        self.n_processors=n_processors
        self.pressures=pressures
        self.temperatures=temperatures
        self.observables=observables
        self.kineticSens=kineticSens
        self.physicalSens=physicalSens
        self.save_physSensHistories=save_physSensHistories
        self.conditions=conditions
        self.cti_path=cti_path
        self.thermalBoundary = thermalBoundary
        self.mechanicalBoundary=mechanicalBoundary
        self.kineticSensitivities= None
        self.experimentalData = None
        self.concentrationObservables = concentrationObservables
        self.moleFractionObservables = moleFractionObservables
        self.absorbanceObservables = absorbanceObservables
        self.fullParsedYamlFile =  fullParsedYamlFile
        #self.energycon='off'
        self.timeshift=timeshift
        self.timeHistory = None
        self.experimentalData = None
        self.initialTime=initialTime
        self.finalTime=finalTime
        self.log_name=log_name
        self.log_file=log_file
        self.save_timeHistories=save_timeHistories
        self.ignitionDelayObservables=['tau']
        #self.yaml_file=yaml_file
        if save_timeHistories == 1:
            self.timeHistories=[]
            self.timeHistoryInterpToExperiment = None
            self.pressureAndTemperatureToExperiment = None
        else:
            self.timeHistories=None
        if save_physSensHistories == 1:
            self.physSensHistories = []
        #self.setTPX()
        self.dk = [0]
        self.solution=None
        self.target=target
        self.target_type=target_type
        self.volumeTraceList = volumeTraceList
        
    def run(self):
        
        
        solution=[]
        ksens=[]
        ksens_1stIter=False
        if self.volumeTraceList: 
        #print(self.conditions)
            for i in range(len(self.temperatures)):
                for k in range(len(self.conditions)):
                    temp_ig=ignition_delay(pressure=self.pressures[i],
                                                   temperature=self.temperatures[i],
                                                   observables=self.observables,
                                                   kineticSens=self.kineticSens,
                                                   physicalSens=self.physicalSens,
                                                   conditions=self.conditions[k],
                                                   thermalBoundary=self.thermalBoundary,
                                                   mechanicalBoundary=self.mechanicalBoundary,
                                                   processor=self.processor,
                                                   cti_path=self.cti_path, 
                                                   save_physSensHistories=self.save_physSensHistories,
                                                   moleFractionObservables=self.moleFractionObservables,
                                                   absorbanceObservables=self.absorbanceObservables,
                                                   concentrationObservables=self.concentrationObservables,
                                                   fullParsedYamlFile=self.fullParsedYamlFile, 
                                                   save_timeHistories=self.save_timeHistories,
                                                   log_file=self.log_file,
                                                   log_name=self.log_name,
                                                   timeshift=self.timeshift,
                                                   initialTime=self.initialTime,
                                                   finalTime=self.finalTime,
                                                   target=self.target,
                                                   target_type=self.target_type,
                                                   n_processors=self.n_processors,
                                                   volumeTrace = self.volumeTraceList[i])
                        
                    a,b=temp_ig.run_single()
                        
                    temp=[]
                    temp1=[]
                    temp=copy.deepcopy(a)
                        #print(temp)
                    temp1=copy.deepcopy(b)
                        #print(a)
                    solution.append(temp)
                        
   
                    if not ksens_1stIter and self.kineticSens==1:
                        ksens=temp1
                        ksens_1stIter=True
                    elif self.kineticSens==1 and ksens_1stIter:
                        ksens=np.vstack([ksens,temp1])                        
                        
        elif len(self.pressures) == len(self.temperatures) and len(self.pressures)>1 and len(self.temperatures)>1 and not self.volumeTraceList :
            print('T and P Itterate Together')
            for i in range(len(self.temperatures)):
                for k in range(len(self.conditions)):
                    temp_ig=ignition_delay(pressure=self.pressures[i],
                                                   temperature=self.temperatures[i],
                                                   observables=self.observables,
                                                   kineticSens=self.kineticSens,
                                                   physicalSens=self.physicalSens,
                                                   conditions=self.conditions[k],
                                                   thermalBoundary=self.thermalBoundary,
                                                   mechanicalBoundary=self.mechanicalBoundary,
                                                   processor=self.processor,
                                                   cti_path=self.cti_path, 
                                                   save_physSensHistories=self.save_physSensHistories,
                                                   moleFractionObservables=self.moleFractionObservables,
                                                   absorbanceObservables=self.absorbanceObservables,
                                                   concentrationObservables=self.concentrationObservables,
                                                   fullParsedYamlFile=self.fullParsedYamlFile, 
                                                   save_timeHistories=self.save_timeHistories,
                                                   log_file=self.log_file,
                                                   log_name=self.log_name,
                                                   timeshift=self.timeshift,
                                                   initialTime=self.initialTime,
                                                   finalTime=self.finalTime,
                                                   target=self.target,
                                                   target_type=self.target_type,
                                                   n_processors=self.n_processors)
                        
                    a,b=temp_ig.run_single()
                        
                    temp=[]
                    temp1=[]
                    temp=copy.deepcopy(a)
                        #print(temp)
                    temp1=copy.deepcopy(b)
                        #print(a)
                    solution.append(temp)
                        
   
                    if not ksens_1stIter and self.kineticSens==1:
                        ksens=temp1
                        ksens_1stIter=True
                    elif self.kineticSens==1 and ksens_1stIter:
                        ksens=np.vstack([ksens,temp1])               
        
        else:
            for i in range(len(self.temperatures)):
                for j in range(len(self.pressures)):
                    for k in range(len(self.conditions)):
                        temp_ig=ignition_delay(pressure=self.pressures[j],
                                               temperature=self.temperatures[i],
                                               observables=self.observables,
                                               kineticSens=self.kineticSens,
                                               physicalSens=self.physicalSens,
                                               conditions=self.conditions[k],
                                               thermalBoundary=self.thermalBoundary,
                                               mechanicalBoundary=self.mechanicalBoundary,
                                               processor=self.processor,
                                               cti_path=self.cti_path, 
                                               save_physSensHistories=self.save_physSensHistories,
                                               moleFractionObservables=self.moleFractionObservables,
                                               absorbanceObservables=self.absorbanceObservables,
                                               concentrationObservables=self.concentrationObservables,
                                               fullParsedYamlFile=self.fullParsedYamlFile, 
                                               save_timeHistories=self.save_timeHistories,
                                               log_file=self.log_file,
                                               log_name=self.log_name,
                                               timeshift=self.timeshift,
                                               initialTime=self.initialTime,
                                               finalTime=self.finalTime,
                                               target=self.target,
                                               target_type=self.target_type,
                                               n_processors=self.n_processors,
                                               volumeTrace = '')                            
                           
            
                        a,b=temp_ig.run_single()
                        
                        temp=[]
                        temp1=[]
                        temp=copy.deepcopy(a)
                        #print(temp)
                        temp1=copy.deepcopy(b)
                        #print(a)
                        solution.append(temp)
                        
                        
                        
                        
                        if not ksens_1stIter and self.kineticSens==1:
                            ksens=temp1
                            ksens_1stIter=True
                        elif self.kineticSens==1 and ksens_1stIter:
                            ksens=np.vstack([ksens,temp1])
                    #print(ksens)
        solution=pd.concat(solution)
        #print(np.shape(ksens))
        #print(self.timeHistories)
        #print(solution)
        if self.timeHistories != None:
            self.timeHistories.append(solution)
        self.kineticSensitivities=ksens

        if self.pressureAndTemperatureToExperiment == None:
            self.pressureAndTemperatureToExperiment = solution[['temperature','pressure']]

        return (solution,ksens)        
        
    # def run(self):
        
        
        
    #     solution=[]
    #     ksens=[]
    #     ksens_1stIter=False
    #     #print(self.conditions)
    #     for i in range(len(self.temperatures)):
    #         for j in range(len(self.pressures)):
    #             for k in range(len(self.conditions)):
    #                 if self.volumeTraceList:
    #                     temp_ig=ignition_delay(pressure=self.pressures[j],
    #                                            temperature=self.temperatures[i],
    #                                            observables=self.observables,
    #                                            kineticSens=self.kineticSens,
    #                                            physicalSens=self.physicalSens,
    #                                            conditions=self.conditions[k],
    #                                            thermalBoundary=self.thermalBoundary,
    #                                            mechanicalBoundary=self.mechanicalBoundary,
    #                                            processor=self.processor,
    #                                            cti_path=self.cti_path, 
    #                                            save_physSensHistories=self.save_physSensHistories,
    #                                            moleFractionObservables=self.moleFractionObservables,
    #                                            absorbanceObservables=self.absorbanceObservables,
    #                                            concentrationObservables=self.concentrationObservables,
    #                                            fullParsedYamlFile=self.fullParsedYamlFile, 
    #                                            save_timeHistories=self.save_timeHistories,
    #                                            log_file=self.log_file,
    #                                            log_name=self.log_name,
    #                                            timeshift=self.timeshift,
    #                                            initialTime=self.initialTime,
    #                                            finalTime=self.finalTime,
    #                                            target=self.target,
    #                                            target_type=self.target_type,
    #                                            n_processors=self.n_processors,
    #                                            volumeTrace = self.volumeTraceList[i])
    #                 else:
    #                     temp_ig=ignition_delay(pressure=self.pressures[j],
    #                                            temperature=self.temperatures[i],
    #                                            observables=self.observables,
    #                                            kineticSens=self.kineticSens,
    #                                            physicalSens=self.physicalSens,
    #                                            conditions=self.conditions[k],
    #                                            thermalBoundary=self.thermalBoundary,
    #                                            mechanicalBoundary=self.mechanicalBoundary,
    #                                            processor=self.processor,
    #                                            cti_path=self.cti_path, 
    #                                            save_physSensHistories=self.save_physSensHistories,
    #                                            moleFractionObservables=self.moleFractionObservables,
    #                                            absorbanceObservables=self.absorbanceObservables,
    #                                            concentrationObservables=self.concentrationObservables,
    #                                            fullParsedYamlFile=self.fullParsedYamlFile, 
    #                                            save_timeHistories=self.save_timeHistories,
    #                                            log_file=self.log_file,
    #                                            log_name=self.log_name,
    #                                            timeshift=self.timeshift,
    #                                            initialTime=self.initialTime,
    #                                            finalTime=self.finalTime,
    #                                            target=self.target,
    #                                            target_type=self.target_type,
    #                                            n_processors=self.n_processors,
    #                                            volumeTrace = '')                            
                           
            
    #                 a,b=temp_ig.run_single()
                    
    #                 temp=[]
    #                 temp1=[]
    #                 temp=copy.deepcopy(a)
    #                 #print(temp)
    #                 temp1=copy.deepcopy(b)
    #                 #print(a)
    #                 solution.append(temp)
    #                 if not ksens_1stIter and self.kineticSens==1:
    #                     ksens=temp1
    #                     ksens_1stIter=True
    #                 elif self.kineticSens==1 and ksens_1stIter:
    #                     ksens=np.vstack([ksens,temp1])
    #                 #print(ksens)
    #     solution=pd.concat(solution)
    #     #print(np.shape(ksens))
    #     #print(self.timeHistories)
    #     #print(solution)
    #     if self.timeHistories != None:
    #         self.timeHistories.append(solution)
    #     self.kineticSensitivities=ksens
    #     return (solution,ksens)
        
        
    def sensitivity_adjustment(self,temp_del:float=0.0,
                               pres_del:float=0.0,
                               spec_triplet:(str,float,int)=('',0.0,0),
                               res_del:float=0.0):
        
        #this is where we would make the dk fix
        if temp_del != 0.0:
            self.dk.append(temp_del)
        if pres_del != 0.0:       
            self.dk.append(pres_del) 
        if spec_triplet[1] != 0.0:
            self.dk.append(spec_triplet[1])
        
        temptemp=copy.deepcopy(self.temperatures)
        temppres=copy.deepcopy(self.pressures)
        tempcond=copy.deepcopy(self.conditions[spec_triplet[2]])
        kin_temp = self.kineticSens
        self.kineticSens = 0
        '''
          Passes the Perturbed observable to the setTPX function. Temperature and pressure 
        are passed and set directly species need to go through an additional step in the 
        setTPX function. 
        '''
        if spec_triplet[0] != '':
            self.temperatures=np.array(self.temperatures)+temp_del*np.array(self.temperatures)
            self.pressures=np.array(self.pressures)+pres_del*np.array(self.pressures)
            #self.pressure=self.pressure+pres_del*self.pressure
            xj=self.conditions[spec_triplet[2]][spec_triplet[0]]
            delxj=spec_triplet[1]*self.conditions[spec_triplet[2]][spec_triplet[0]]
            #print(xj,delxj)
            self.conditions[spec_triplet[2]][spec_triplet[0]]=np.divide(np.multiply(xj+delxj,1-xj),1-xj-delxj)
#           self.setTPX(self.temperature+self.temperature*temp_del,
#                   self.pressure+self.pressure*pres_del,
#                   {spec_pair[0]:self.conditions[spec_pair[0]]*spec_pair[1]})
          
           
        else:
           self.temperatures=np.array(self.temperatures)+temp_del*np.array(self.temperatures)
           self.pressures=np.array(self.pressures)+pres_del*np.array(self.pressures)
           #self.pressure=self.pressure+pres_del*self.pressure
           #self.residence_time=self.residence_time+res_del*self.residence_time
           
#           self.setTPX(self.temperature+self.temperature*temp_del,
#                       self.pressure+self.pressure*pres_del)
        
        data,trash = self.run() #Ignore trash, just temp storage for empty kinetic sens array
        #print(data)
        
        #data = sim.Simulation.sensitivity_adjustment(self,temp_del,pres_del,spec_pair)
        self.temperatures=temptemp
        self.pressures=temppres
        self.conditions[spec_triplet[2]]=tempcond
        self.kineticSens = kin_temp
        
        
        return data
    
    # def species_adjustment(self,spec_del:float=0.0):
    #     inert_species=['Ar','AR','HE','He','Kr','KR',
    #                    'Xe','XE','NE','Ne']
        
    #     '''
    #     Creates tuples of specie that need to be perturbed and the
    #     percent value by which to perturb its mole fraction 
    #     '''
    #     # gets the mole fraction and the species which are going to be 
    #     #perturbed in order to run a sensitivity calculation 
    #     data = []
    #     for i in range(len(self.conditions)):
    #         for x in self.conditions[i].keys():
    #             if x not in inert_species:
    #                 data.append(self.sensitivity_adjustment(spec_triplet=(x,spec_del,i)))

    #     return data
    
    
    def species_adjustment(self,spec_del:float=0.0, diluents=[]):
        # inert_species=['Ar','AR','HE','He','Kr','KR',
        #                'Xe','XE','NE','Ne']
        inert_species=diluents
        '''
        Creates tuples of specie that need to be perturbed and the
        percent value by which to perturb its mole fraction 
        '''
        # gets the mole fraction and the species which are going to be 
        #perturbed in order to run a sensitivity calculation 
        data = []
        for i in range(len(self.conditions)):
            for x in self.conditions[i].keys():
                if x not in inert_species:
                    data.append(self.sensitivity_adjustment(spec_triplet=(x,spec_del,i)))
        #print(len(data))
        return data
    
    def importExperimentalData(self,csvFileList):
        print('Importing ignition delay data the following csv files...') 
        print(csvFileList)
        experimentalData = [pd.read_csv(csv) for csv in csvFileList]
        experimentalData = [experimentalData[x].dropna(how='any') for x in range(len(experimentalData))]
        experimentalData = [experimentalData[x].apply(pd.to_numeric, errors = 'coerce').dropna() for x in range(len(experimentalData))]
        for x in range(len(experimentalData)):
            experimentalData[x] = experimentalData[x][~(experimentalData[x][experimentalData[x].columns[1]] < 0)]
        self.experimentalData = experimentalData
        return experimentalData
    
    def map_and_interp_ksens(self,temp_history=None):
        A = self.kineticSensitivities
        #print(np.shape(A))
        N = np.zeros(A.shape)
        Ea = np.zeros(A.shape)
        for i in range(0,A.shape[2]):
            sheetA = A[:,:,i] #sheet for specific observable

            #print(sheetA)
            for x,column in enumerate(sheetA.T):
                N[:,x,i]= np.multiply(column,np.log(self.timeHistories[0]['temperature'])) if temp_history is None else np.multiply(column,np.log(temp_history['temperature']))
                #not sure if this mapping is correct, check with burke and also update absorption mapping
                #to_mult_ea = np.divide(-1,np.multiply(1/ct.gas_constant,self.timeHistories[0]['temperature'])) if time_history is None else np.divide(-1,np.multiply(ct.gas_constant,time_history['temperature']))
                to_mult_ea = np.divide(-1,np.multiply(1,self.timeHistories[0]['temperature'])) if temp_history is None else np.divide(-1,np.multiply(1,temp_history['temperature']))
                Ea[:,x,i]= np.multiply(column,to_mult_ea)
        #print(np.shape(A))
        tempA=[]
        tempn=[]
        tempEa=[]
        for i in range(0,A.shape[2]):
            tempA.append(A[:,:,i])
            tempn.append(N[:,:,i])
            tempEa.append(Ea[:,:,i])
        A=tempA
        N=tempn
        Ea=tempEa
        return {'A':A,
                'N':N,
                'Ea':Ea}
    def sensitivityCalculation(self,originalValues,newValues,dk=.01):
        #print(originalValues)
        #print(newValues)
        if isinstance(originalValues,pd.DataFrame) and isinstance(newValues,pd.DataFrame) or isinstance(originalValues,pd.Series) and isinstance(newValues,pd.Series):
            if isinstance(originalValues,pd.Series) or isinstance(newValues,pd.Series):
                originalValues=originalValues.to_frame()
                newValues=newValues.to_frame()
            #newValues.columns = thingToFindSensitivtyOf
            
            newValues = newValues.applymap(np.log)
            originalValues = originalValues.applymap(np.log)
            #tab
            
            sensitivity = (newValues.subtract(originalValues)/dk)
            return sensitivity
        else:
            print("Error: wrong datatype, both must be pandas data frames")
            return -1
    
    def calculate_time_shift_sens(self,nominal, dtau=1e-8):
        
        new_delay=np.array(nominal)+dtau*np.ones(len(np.array(nominal)))
        sens=(np.log(new_delay)-np.log(np.array(nominal)))/dtau
        sensdata=pd.DataFrame(columns=['delay'])
        sensdata['delay']=sens
        
        return new_delay,sensdata
        
        
        
        
        
        
        
        
        
        
        
        
