# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:33:46 2018

@author: Mark Barbet
"""

import cantera as ct
from .. import simulation as sim
from ...cti_core import cti_processor as ctp
import pandas as pd
import numpy as np
import time
import copy
import re

class JSR_steadystate(sim.Simulation):
    
    '''Child class of sim.Simulaton.  Inherits all attributes and methods including __init__().  
    Also has internal init due to data requirements.  JSR_steadystate is a single jet-stirred-reactor simulation 
    and may be used to simulate a jet-stirred-reactor experiment at a single constant temperature, pressure, and 
    mole fraction conditions.'''
    
    
    
    
    
    def __init__(self,pressure:float,temperature:float,observables:list,
                 kineticSens:int,physicalSens:int,conditions:dict,thermalBoundary,mechanicalBoundary,
                 processor:ctp.Processor=None,cti_path="", 
                 save_physSensHistories=0,moleFractionObservables:list=[],
                 absorbanceObservables:list=[],concentrationObservables:list=[],
                 fullParsedYamlFile:dict={},residence_time:float=1.0,pvalveCoefficient:float=0.01,
                 maxpRise:float=0.001,save_timeHistories:int=0,rtol:float=1e-14,atol:float=1e-15):
        
        #sim.Simulation.__init__(self,pressure,temperature,observables,kineticSens,physicalSens,
        #                        conditions,processor,cti_path)
        #set up processor and initialize all variables  common to every simulation  
        if processor!=None and cti_path!="":
            print("Error: Cannot give both a processor and a cti file path, pick one")
        elif processor==None and cti_path=="":
            print("Error: Must give either a processor or a cti file path")
        if processor != None:
            self.processor = processor 
        elif cti_path!="":
            self.processor = ctp.Processor(cti_path)
        
        self.pressure=pressure
        '''Pressure of the reactor in atm'''
        self.temperature=temperature
        '''Temperature of the reactor in K'''
        self.observables=observables
        '''Observables to be used in kinetic sensitivity analysis.  Required only if `self.kineticSens` is set to 1'''
        self.kineticSens=kineticSens
        '''Set to 1 to run kinetic sensitivity analysis, 0 otherwise'''
        self.physicalSens=physicalSens
        '''Deprecated'''
        self.conditions=conditions
        '''Mole fractions for the gas flow into the reactor.  Accepts a dictionary similarly to {\'H2\':0.1,\'O2\':0.1,\'Ar\':0.8}'''
        self.cti_path=cti_path
        '''Full file path for Cantera cti file.  Only needed if `self.processor` has not been defined.'''
        self.thermalBoundary = thermalBoundary
        '''Deprecated '''
        self.mechanicalBoundary = mechanicalBoundary
        '''Deprecated'''
        self.kineticSensitivities= None
        '''Stores an array of the kinetic sensitivities after calculation'''
        self.experimentalData = None
        '''Deprecated'''
        self.concentrationObservables = concentrationObservables
        self.moleFractionObservables = moleFractionObservables
        self.absorbanceObservables = absorbanceObservables
        self.fullParsedYamlFile =  fullParsedYamlFile
        self.pvalveCoefficient=pvalveCoefficient
        '''Controls the sensitivity of the pressure valve in the reactor network.  For most use cases, leave default value as is.  
        Only adjust if final pressure substantially deviates from value set in `self.pressure`'''
        self.maxPrise=maxpRise
        '''Max pressure rise allowed.  Leave unchanged in most use cases. 
        Only adjust if final pressure substantially deviates from value set in `self.pressure`'''
        self.energycon='off'
        '''Always set to \'off\' '''
        self.residence_time=residence_time
        '''Residence time of the reactor'''
        self.timeHistory = None
        self.experimentalData = None
        
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
        self.rtol=rtol
        self.atol=atol
        self.solution=None
        
        
    def set_geometry(self,volume=0.1):
        self.reactor_volume=volume
        
        
    def printVars(self):
        print()
        
    def settingJSRConditions(self):
        '''
        Determine the mechanical and thermal boundary conditions for a 
        shock tube.
        '''
        
        #assigning the thermal boundary variable
        if re.match('[aA]diabatic',self.thermalBoundary):
            energy = 'on'
        elif re.match('[iI]sothermal',self.thermalBoundary):
            energy = 'off'
        else:
            raise Exception('Please specify a thermal boundary condition, adiabatic or isothermal')
        #assigning the mehcanical boundary variable 
        if re.match('[Cc]onstant [Pp]ressure',self.mechanicalBoundary):
            mechBoundary = 'constant pressure'
        elif re.match('[Cc]onstant [Vv]olume',self.mechanicalBoundary):
            mechBoundary = 'constant volume'
        else:
            raise Exception('Please specifiy a mehcanical boundary condition, constant pressure or constant volume')
        #return the thermal and mechanical boundary of the shock tube 
        return energy,mechBoundary
    
    def sensitivity_adjustment(self,temp_del:float=0.0,
                               pres_del:float=0.0,
                               spec_pair:(str,float)=('',0.0)):
        
        #this is where we would make the dk fix
        if temp_del != 0.0:
            self.dk.append(temp_del)
        if pres_del != 0.0:       
            self.dk.append(pres_del) 
        if spec_pair[1] != 0.0:
            self.dk.append(spec_pair[1])
        
        
        kin_temp = self.kineticSens
        self.kineticSens = 0
        data = sim.Simulation.sensitivity_adjustment(self,temp_del,pres_del,spec_pair)
        self.kineticSens = kin_temp
        return data
    
    def run_single(self):
        
        gas=self.processor.solution
        reactorPressure=gas.P
        self.reactorPressure=self.processor.solution.P
        pressureValveCoefficient=self.pvalveCoefficient
        maxPressureRiseAllowed=self.maxPrise
        
        print(maxPressureRiseAllowed,self.reactorPressure,pressureValveCoefficient)
        #Build the system components for JSR
        pretic=time.time()
        
        if bool(self.observables) and self.kineticSens==1:
            ###################################################################
            #Block to create temp reactor network to pre-solve JSR without kinetic sens
            ct.suppress_thermo_warnings()
            tempgas=ct.Solution(self.processor.cti_path)
            tempgas.TPX=self.processor.solution.TPX
            tempfuelAirMixtureTank=ct.Reservoir(tempgas)
            tempexhaust=ct.Reservoir(tempgas)
            tempstirredReactor=ct.IdealGasReactor(tempgas,energy=self.energycon,
                                                  volume=self.reactor_volume)
            tempmassFlowController=ct.MassFlowController(upstream=tempfuelAirMixtureTank,
                                                         downstream=tempstirredReactor,
                                                         mdot=tempstirredReactor.mass/self.residence_time)
            tempPressureRegulator=ct.Valve(upstream=tempstirredReactor,downstream=tempexhaust,
                                           K=pressureValveCoefficient)
            
            tempreactorNetwork=ct.ReactorNet([tempstirredReactor])
            tempreactorNetwork.rtol = self.rtol
            tempreactorNetwork.atol = self.atol
            print(self.rtol,self.atol)
            tempreactorNetwork.advance_to_steady_state()
            ###################################################################
            #reactorNetwork.advance_to_steady_state()
            #reactorNetwork.reinitialize()
        
        elif self.kineticSens and bool(self.observables)==False:
            #except:
                print('Please supply a non-empty list of observables for sensitivity analysis or set kinetic_sens=0')        
        pretoc=time.time()
        
        print('Presolving Took {:3.2f}s to compute'.format(pretoc-pretic))        
        fuelAirMixtureTank=ct.Reservoir(self.processor.solution)
        exhaust=ct.Reservoir(self.processor.solution)
        if bool(self.observables) and self.kineticSens==1:
            stirredReactor=ct.IdealGasReactor(tempgas,energy=self.energycon,
                                          volume=self.reactor_volume)
        else:
            stirredReactor=ct.IdealGasReactor(self.processor.solution,energy=self.energycon,
                                          volume=self.reactor_volume)
        #stirredReactor=ct.IdealGasReactor(self.processor.solution,energy=self.energycon,
        #                                  volume=self.reactor_volume)    
        massFlowController=ct.MassFlowController(upstream=fuelAirMixtureTank,
                                                 downstream=stirredReactor,
                                                 mdot=stirredReactor.mass/self.residence_time)
        pressureRegulator=ct.Valve(upstream=stirredReactor,downstream=exhaust,K=pressureValveCoefficient)
        reactorNetwork=ct.ReactorNet([stirredReactor])
        if bool(self.observables) and self.kineticSens==1:
            for i in range(gas.n_reactions):
                stirredReactor.add_sensitivity_reaction(i)
        reactorNetwork.rtol_sensitivity=0.0000001
        reactorNetwork.atol_sensitivity=0.00000001
        print('Sens tols:'+str(reactorNetwork.atol_sensitivity)+', '+str(reactorNetwork.rtol_sensitivity))
        # now compile a list of all variables for which we will store data
        columnNames = [stirredReactor.component_name(item) for item in range(stirredReactor.n_vars)]
        columnNames = ['pressure'] + columnNames

        # use the above list to create a DataFrame
        timeHistory = pd.DataFrame(columns=columnNames)

        # Start the stopwatch
        tic = time.time()
        reactorNetwork.rtol = self.rtol
        reactorNetwork.atol = self.atol    
        #reactorNetwork.max_err_test_fails= 10000
        #print(reactorNetwork.max_err_test_fails)
        if self.physicalSens==1 and bool(self.observables)==False:
            #except:
                print('Please supply a non-empty list of observables for sensitivity analysis or set physical_sens=0')
        
        #Establish a matrix to hold sensitivities for kinetic parameters, along with tolerances
        if self.kineticSens==1 and bool(self.observables):
            #senscolumnNames = ['Reaction']+observables     
            senscolumnNames = self.observables
            #sensArray = pd.DataFrame(columns=senscolumnNames)
            #senstempArray = np.zeros((gas.n_reactions,len(observables)))
            dfs = [pd.DataFrame() for x in range(len(self.observables))]
            #tempArray = [np.zeros(self.processor.solution.n_reactions) for x in range(len(self.observables))]
            #stirredReactor.thermo.X=tempstirredReactor.thermo.X
                
        
        posttic=time.time()
#        for steps in range(10):
#            reactorNetwork.step()
        reactorNetwork.advance_to_steady_state()
        posttoc=time.time()
        print('Main Solver Took {:3.2f}s to compute'.format(posttoc-posttic))
        final_pressure=stirredReactor.thermo.P
        sens=reactorNetwork.sensitivities()
        #print(sens[:,787])
        #print(gas.species_names)
        #print(self.observables)
        #print(sens)
        if self.kineticSens==1 and bool(self.observables):
            #print((pd.DataFrame(sens[0,:])).transpose())
            #test=pd.concat([pd.DataFrame(),pd.DataFrame(sens[0,:]).transpose()])
            #print(test)
            for k in range(len(self.observables)):
                index=gas.species_names.index(self.observables[k])
                dfs[k] = dfs[k].append(((pd.DataFrame(sens[index+3,:])).transpose()),ignore_index=True)
                #dfs[k]=pd.concat([dfs[k],pd.DataFrame(sens[k,:]).transpose()])
                #dfs[k]=pd.DataFrame(sens[k,:]).transpose()
            #print(dfs)  
        toc = time.time()
        print('Simulation Took {:3.2f}s to compute'.format(toc-tic)+' at T = '+str(stirredReactor.T))
        #print(dfs[0])
        columnNames = []
        #Store solution to a solution array
        #for l in np.arange(stirredReactor.n_vars):
            #columnNames.append(stirredReactor.component_name(l))
        columnNames=[stirredReactor.component_name(item) for item in range(stirredReactor.n_vars)]
        #state=stirredReactor.get_state()
        state=np.hstack([stirredReactor.mass, 
                   stirredReactor.volume, stirredReactor.T, stirredReactor.thermo.X])
        data=pd.DataFrame(state).transpose()
        data.columns=columnNames
        pressureDifferential = timeHistory['pressure'].max()-timeHistory['pressure'].min()
        if(abs(pressureDifferential/self.reactorPressure) > maxPressureRiseAllowed):
            #except:
                print("WARNING: Non-trivial pressure rise in the reactor. Adjust K value in valve")
        
        if self.kineticSens==1:
            numpyMatrixsksens = [dfs[dataframe].values for dataframe in range(len(dfs))]
            self.kineticSensitivities = np.dstack(numpyMatrixsksens)
            #print(np.shape(self.kineticSensitivities))
            self.solution=data
            self.solution['pressure']=[self.pressure]
            return (self.solution,self.kineticSensitivities)
        else:
            self.solution=data
            self.solution['pressure']=[self.pressure]
            return (self.solution,[])
        
        
        
        
        
#class JSR_multiTemp_steadystate2(sim.Simulation):
#        
#    def __init__(self,volume:float,pressure:float,temperatures:list,observables:list,
#                 kineticSens:int,physicalSens:int,conditions:dict,thermalBoundary,mechanicalBoundary,
#                 processor:ctp.Processor=None,cti_path="", 
#                 save_physSensHistories=0,moleFractionObservables:list=[],
#                 absorbanceObservables:list=[],concentrationObservables:list=[],
#                 fullParsedYamlFile:dict={},residence_time:float=1.0,pvalveCoefficient:float=0.01,maxpRise:float=0.001):
#        
##    sim.Simulation.__init__(self,pressure,temperature,observables,kineticSens,physicalSens,
##                                conditions,processor,cti_path)
#        self.volume=volume
#        self.temperatures=temperatures
#        self.JSR_objects=[]
#        for i in range(len(self.temperatures)):
#            a=[]
#            a=copy.deepcopy(JSR_steadystate(pressure,self.temperatures[i],observables,
#                 kineticSens,physicalSens,conditions,thermalBoundary,mechanicalBoundary,
#                 processor,cti_path, 
#                 save_physSensHistories,moleFractionObservables,
#                 absorbanceObservables,concentrationObservables,
#                 fullParsedYamlFile,residence_time,pvalveCoefficient,maxpRise))
#            self.JSR_objects.append(a)
#        for i in self.JSR_objects:
#            print(i.temperature)
#            
#            
#    def run_multitemp(self):
#        
#        for i in range(len(self.JSR_objects)):
#            self.JSR_objects[i].set_geometry(self.volume)
#            
#            self.JSR_objects[i].run()
#        
#        
#        
class JSR_multiTemp_steadystate(sim.Simulation):
        
    def __init__(self,volume:float,pressure:float,temperatures:list,observables:list,
                 kineticSens:int,physicalSens:int,conditions:dict,thermalBoundary,mechanicalBoundary,
                 processor:ctp.Processor=None,cti_path="", 
                 save_physSensHistories=0,moleFractionObservables:list=[],save_timeHistories:int=0,
                 absorbanceObservables:list=[],concentrationObservables:list=[],
                 fullParsedYamlFile:dict={},residence_time:float=1.0,pvalveCoefficient:float=0.01,
                 maxpRise:float=0.001,atol:float=1e-15,rtol:float=1e-14):
        
#    sim.Simulation.__init__(self,pressure,temperature,observables,kineticSens,physicalSens,
#                                conditions,processor,cti_path)
        
        if processor!=None and cti_path!="":
            print("Error: Cannot give both a processor and a cti file path, pick one")
        elif processor==None and cti_path=="":
            print("Error: Must give either a processor or a cti file path")
        if processor != None:
            self.processor = processor 
        elif cti_path!="":
            self.processor = ctp.Processor(cti_path)        
        
        self.save_physSensHistories=save_physSensHistories
        self.volume=volume
        self.temperatures=temperatures
        self.JSR_objects=[]
        self.pressure=pressure
        self.observables=observables
        self.kineticSens=kineticSens
        self.physicalSens=physicalSens
        self.conditions=conditions
        self.thermalBoundary=thermalBoundary
        self.mechanicalBoundary = mechanicalBoundary
        self.kineticSensitivities= None
        self.experimentalData = None
        self.concentrationObservables = concentrationObservables
        self.moleFractionObservables = moleFractionObservables
        self.absorbanceObservables = absorbanceObservables
        self.fullParsedYamlFile =  fullParsedYamlFile
        self.pvalveCoefficient=pvalveCoefficient
        self.maxPrise=maxpRise
        self.energycon='off'
        self.residence_time=residence_time
        
               
        if save_timeHistories == 1:
            self.timeHistories=[]
            self.timeHistoryInterpToExperiment = None
            self.pressureAndTemperatureToExperiment = pd.DataFrame()
        else:
            self.timeHistories=None
        
        #self.setTPX()
        self.dk = [0]
        self.rtol=rtol
        self.atol=atol
            
    def run(self):
        solution=[]
        ksens=[]
        ksens_1stIter=False
        for i in range(len(self.temperatures)):
            temp_jsr=JSR_steadystate(pressure=self.pressure,
                                     temperature=self.temperatures[i],
                                     observables=self.observables,
                                     kineticSens=self.kineticSens,
                                     physicalSens=self.physicalSens,
                                     conditions=self.conditions,
                                     thermalBoundary=self.thermalBoundary,
                                     mechanicalBoundary=self.mechanicalBoundary,
                                     processor=self.processor,
                                     save_physSensHistories=self.save_physSensHistories,
                                     residence_time=self.residence_time,
                                     rtol=self.rtol,
                                     atol=self.atol)
            temp_jsr.set_geometry(volume=self.volume)
            try:
                a,b=temp_jsr.run_single()
            except Exception as e:
                print(e)
                temp_jsr=JSR_steadystate(pressure=self.pressure,
                                     temperature=self.temperatures[i],
                                     observables=self.observables,
                                     kineticSens=self.kineticSens,
                                     physicalSens=self.physicalSens,
                                     conditions=self.conditions,
                                     thermalBoundary=self.thermalBoundary,
                                     mechanicalBoundary=self.mechanicalBoundary,
                                     processor=self.processor,
                                     save_physSensHistories=self.save_physSensHistories,
                                     residence_time=self.residence_time,
                                     rtol=self.rtol*10.0,
                                     atol=self.atol*10.0)
                temp_jsr.set_geometry(volume=self.volume)
                a,b=temp_jsr.run_single()
                print('Completed')
            temp=[]
            temp1=[]
            temp=copy.deepcopy(a)
            temp1=copy.deepcopy(b)
            solution.append(temp)
            if not ksens_1stIter and self.kineticSens==1:
                ksens=temp1
                ksens_1stIter=True
            elif self.kineticSens==1 and ksens_1stIter:
                ksens=np.vstack([ksens,temp1])
            #print(ksens)
        solution=pd.concat(solution)
        #print(np.shape(ksens))
        if self.timeHistories != None:
            self.timeHistories.append(solution)
        self.kineticSensitivities=ksens
        if 'pressureAndTemperatureToExperiment' in vars(self):
            if self.pressureAndTemperatureToExperiment.empty:
                self.pressureAndTemperatureToExperiment = solution[['temperature','pressure']]


        return (solution,ksens)
        
    def sensitivity_adjustment(self,temp_del:float=0.0,
                               pres_del:float=0.0,
                               spec_pair:(str,float)=('',0.0),
                               res_del:float=0.0):
        
        #this is where we would make the dk fix
        if temp_del != 0.0:
            self.dk.append(temp_del)
        if pres_del != 0.0:       
            self.dk.append(pres_del) 
        if spec_pair[1] != 0.0:
            self.dk.append(spec_pair[1])
        
        temptemp=copy.deepcopy(self.temperatures)
        temppres=copy.deepcopy(self.pressure)
        tempcond=copy.deepcopy(self.conditions)
        kin_temp = self.kineticSens
        self.kineticSens = 0
        '''
          Passes the Perturbed observable to the setTPX function. Temperature and pressure 
        are passed and set directly species need to go through an additional step in the 
        setTPX function. 
        '''
        if spec_pair[0] != '':
            self.temperatures=np.array(self.temperatures)+temp_del*np.array(self.temperatures)
            self.pressure=self.pressure+pres_del*self.pressure
            xj=self.conditions[spec_pair[0]]
            delxj=spec_pair[1]*self.conditions[spec_pair[0]]
            #print(xj,delxj)
            self.conditions[spec_pair[0]]=np.divide(np.multiply(xj+delxj,1-xj),1-xj-delxj)
#           self.setTPX(self.temperature+self.temperature*temp_del,
#                   self.pressure+self.pressure*pres_del,
#                   {spec_pair[0]:self.conditions[spec_pair[0]]*spec_pair[1]})
          
           
        else:
           self.temperatures=np.array(self.temperatures)+temp_del*np.array(self.temperatures)
           self.pressure=self.pressure+pres_del*self.pressure
           self.residence_time=self.residence_time+res_del*self.residence_time
           
#           self.setTPX(self.temperature+self.temperature*temp_del,
#                       self.pressure+self.pressure*pres_del)
        
        data,trash = self.run() #Ignore trash, just temp storage for empty kinetic sens array
        #print(data)
        
        #data = sim.Simulation.sensitivity_adjustment(self,temp_del,pres_del,spec_pair)
        self.temperatures=temptemp
        self.pressure=temppres
        self.conditions=tempcond
        self.kineticSens = kin_temp
        
        
        return data
    
    def species_adjustment(self,spec_del:float=0.0):
        inert_species=['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']
        
        '''
        Creates tuples of specie that need to be perturbed and the
        percent value by which to perturb its mole fraction 
        '''
        # gets the mole fraction and the species which are going to be 
        #perturbed in order to run a sensitivity calculation 
        data = []
        for x in self.conditions.keys():
            if x not in inert_species:
                data.append(self.sensitivity_adjustment(spec_pair=(x,spec_del)))

        return data
    
    def importExperimentalData(self,csvFileList):
        print('Importing jsr data the following csv files...') 
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
        N = np.zeros(A.shape)
        Ea = np.zeros(A.shape)
        for i in range(0,A.shape[2]):
            sheetA = A[:,:,i] #sheet for specific observable
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
    def sensitivityCalculation(self,originalValues,newValues,thingToFindSensitivtyOf,dk=.01):
        if isinstance(originalValues,pd.DataFrame) and isinstance(newValues,pd.DataFrame):
            
            #newValues.columns = thingToFindSensitivtyOf
            
            newValues = newValues.applymap(np.log)
            originalValues = originalValues.applymap(np.log)
            #tab
            
            sensitivity = (newValues.subtract(originalValues)/dk)
            return sensitivity
        else:
            print("Error: wrong datatype, both must be pandas data frames")
            return -1
        
        