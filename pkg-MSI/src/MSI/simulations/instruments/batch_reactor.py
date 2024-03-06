import itertools
import numpy as np
import cantera as ct
import pandas as pd
import re
import pickle
from .. import simulation as sim
from ...cti_core import cti_processor as ctp

class batchReactor(sim.Simulation):
    
    def __init__(self,pressure:float,temperature:float,observables:list,
                 kineticSens:int,physicalSens:int,conditions:dict,
                 initialTime,finalTime,thermalBoundary,mechanicalBoundary,
                 processor:ctp.Processor=None,cti_path="",save_timeHistories:int=0, 
                 save_physSensHistories=0,moleFractionObservables:list=[],
                 absorbanceObservables:list=[],concentrationObservables:list=[],
                 fullParsedYamlFile:dict={},
                 time_shift_value:float = 0, atol:float=1e-15, rtol:float=1e-9,
                 rtol_sens:float=0.0001,
                 atol_sens:float=1e-6):
        '''
        Child class pertaining to shock tube simulations. Inherits all 
        attributes and methods from simulations class including __init__(). 
        Also has its own internal init method due to additional data 
        requirements.
    

        Parameters
        ----------
        pressure : float
            Pressure in [atm].
        temperature : float
            Temperature in [K].
        observables : list
            Species which sensitivity analysis is performed for.
        kineticSens : int
            0 for off, 1 for on.
        physicalSens : int
            0 for off, 1 for on.
        conditions : dict
            Initial mole fractions for species in simulation.
        initialTime : float
            Time to begin simulation from (s).
        finalTime : float
            Time to end simulation (s).
        thermalBoundary : str
            Thermal boundary condition inside the reactor. Shock tubes can
            either be adiabatic or isothermal.
        mechanicalBoundary : TYPE
            Mechanical bondary condition inside the reactor. Shock tubes can
            either be constant pressure or constant volume.
        processor : ctp.Processor, optional
            Loaded cti file. The default is None.
        cti_path : str, optional
            Path of cti file for running. If processor is provided this is not 
            needed. The default is "".
        save_timeHistories : int, optional
            Boolean variable describing if time histories for simulation runs
            are saved. 0 for not saved, 1 for saved. The default is 0.
        save_physSensHistories : TYPE, optional
            Boolean variable describing if physical sensitivity time histories
            are saved. 0 for not saved, 1 for saved. The default is 0.
        moleFractionObservables : list, optional
            Species for which experimental data in the form of mole fraction
            time histories will be provided for optimization.
            Kinetic sensitivities are calculated for all these species. 
            The default is [].
        absorbanceObservables : list, optional
            Species for which experimental data in the form of summed
            absorption time histories (from some or all of the species) will be
            provided for optimization.
            Kinetic sensitivities are calculated for all these species. 
            The default is [].
        concentrationObservables : list, optional
            Species for which experimental data in the form of concentration
            time histories will be provided for optimization.
            Kinetic sensitivities are calculated for all these species. 
            The default is [].
        fullParsedYamlFile : dict, optional
            Full dictionary from the parsed shock tube yaml file. 
            The default is {}.
        time_shift_value : float, optional
            The numerical value by which the time vector of the simulation
            is shifted in seconds. The default is 0.
        atol : float, optional
            Get the absolute error tolerance. The default is 1e-15.
        rtol : float, optional
            Get the relative error tolerance. The default is 1e-9.
        rtol_sens : float, optional
            Scalar relative error tolerance for sensitivity. The default is 0.0001.
        atol_sens : float, optional
            Scalar absolute error tolerance for sensitivity. The default is 1e-6.

        Returns
        -------
        None.

        '''

        sim.Simulation.__init__(self,pressure,temperature,observables,kineticSens,physicalSens,
                                conditions,processor,cti_path)
        self.initialTime = initialTime
        self.finalTime = finalTime
        self.thermalBoundary = thermalBoundary
        self.mechanicalBoundary = mechanicalBoundary
        self.kineticSensitivities= None
        self.timeHistory = None
        self.experimentalData = None
        self.concentrationObservables = concentrationObservables
        self.moleFractionObservables = moleFractionObservables
        self.absorbanceObservables = absorbanceObservables
        self.fullParsedYamlFile =  fullParsedYamlFile
        self.time_shift_value = time_shift_value

        if save_timeHistories == 1:
            self.timeHistories=[]
            self.timeHistoryInterpToExperiment = None
            self.pressureAndTemperatureToExperiment = None
        else:
            self.timeHistories=None
        if save_physSensHistories == 1:
            self.physSensHistories = []
        self.setTPX()
        self.dk = [0]
        self.atol=atol
        self.rtol=rtol
        self.rtol_sensitivity=rtol_sens
        self.atol_sensitivity=atol_sens
        
    def printVars(self):
        '''
        Prints variables associated with the reactor initial conditions.
    
        '''        
        
        
        print('initial time: {0}\nfinal time: {1}\n'.format(self.initialTime,self.finalTime),
              '\nthermalBoundary: {0}\nmechanicalBoundary: {1}'.format(self.thermalBoundary,self.mechanicalBoundary),
              '\npressure: {0}\ntemperature: {1}\nobservables: {2}'.format(self.pressure,self.temperature,self.observables),
              '\nkineticSens: {0}\nphysicalSens: {1}'.format(self.kineticSens,self.physicalSens),
              '\nTPX: {0}'.format(self.processor.solution.TPX)
              )
    #maybe unify paths with cti file?, also really fix the python styling
    def write_time_histories(self, path=''):
        if self.timeHistories == None:
            print("Error: this simulation is not saving time histories, reinitialize with flag")
            return -1
        if path=='':
            path = './time_histories.time'
        pickle.dump(self.timeHistories,open(path,'wb'))
        return 0

    def write_physSensHistories(self, path=''):
        if self.physSensHistories == None:
            print("Error: this simulation is not saving time histories, reinitialize with flag")
            return -1
        if path=='':
            path = './physSensHistories.sens'
            pickle.dump(self.physSensHistories,open(path,'wb'))
        return 0

    def load_physSensHistories(self, path=''):
        if self.physSensHistories == None:
            print("Error: this simulation is not saving time histories, reinitialize with flag")
            return -1
        if path=='':
            path = './physSensHistories.sens'
        pickle.load(self.physSensHistories,open(path,'wb'))
        return 0

    #note this is destructive, the original timeHistories are overwritten, run before any runs
    #same write is destructive by default
    def load_time_histories(self, path=''):
        if self.timeHistories == None:
            print("Error: this simulation is not saving time histories, reinitialize with flag")
            return -1
        if path=='':
            path = './time_histories.time'
        pickle.load(self.timeHistories,open(path,'wb'))
        return 0
    def settingbatchReactorConditions(self):
        '''
        Determine the mechanical and thermal boundary conditions for a 
        shock tube based on what was initialized.
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
        '''
        Appends the sensitivity adjustment to list, and calls sensitivity adjustment
        function from the simulations class, to adjust P,T,X for the sensitivity
        calculation       

        Parameters
        ----------
        temp_del : float, optional
            The decimal value of the percent by which temperature is perturbed.
            The default is 0.0.
        pres_del : float, optional
            The decimal value of the percent by which pressure is perturbed. 
            The default is 0.0.
        spec_pair : (str,float), optional
             The string of a species and the decimal value of the percent by 
             which that species is perturbed . The default is ('',0.0).

        Returns
        -------
        data : Pandas Data Frame
            Time history of the perturbed simulation.

        '''

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

    def run(self,initialTime:float=-1.0, finalTime:float=-1.0):
        '''
        Run the shock tube simulation


        Parameters
        ----------
        initialTime : float, optional
            The time at which the reactor simulation begins, in seconds. The default is -1.0.
        finalTime : float, optional
            The time at which the reactor simulation ends, in seconds. The default is -1.0.

        Returns
        -------
        timeHistory: Pandas DataFrame
            Time history of simulation containing temperature, pressurea and
            species results.
        kineticSensitivities: numpy array
            three dimensional numpy array: (time x reaction x observable).

        '''

        if initialTime == -1.0:
            initialTime = self.initialTime 
        if finalTime == -1.0:
            finalTime = self.finalTime
        self.timeHistory = None
        self.kineticSensitivities= None #3D numpy array, columns are reactions with timehistories, depth gives the observable for those histories
        conditions = self.settingbatchReactorConditions()
        mechanicalBoundary = conditions[1]
        #same solution for both cp and cv sims
        if mechanicalBoundary == 'constant pressure':
            batchReactor = ct.IdealGasConstPressureReactor(self.processor.solution,
                                                        name = 'R1',
                                                        energy = conditions[0])
        else:
            batchReactor = ct.IdealGasReactor(self.processor.solution,
                                           name = 'R1',
                                           energy = conditions[0])
        sim = ct.ReactorNet([batchReactor])
        sim.rtol=self.rtol
        sim.atol=self.atol
        #print(sim.rtol_sensitivity,sim.atol_sensitivity)
        sim.rtol_sensitivity=self.rtol_sensitivity
        sim.atol_sensitivity=self.atol_sensitivity
        
        columnNames = [batchReactor.component_name(item) for item in range(batchReactor.n_vars)]
        columnNames = ['time']+['pressure']+columnNames
        self.timeHistory = pd.DataFrame(columns=columnNames)

        if self.kineticSens == 1:
            for i in range(self.processor.solution.n_reactions):
                batchReactor.add_sensitivity_reaction(i)
            dfs = [pd.DataFrame() for x in range(len(self.observables))]
            tempArray = [np.zeros(self.processor.solution.n_reactions) for x in range(len(self.observables))]

        t = self.initialTime
        counter = 0
        #print(sim.rtol_sensitivity,sim.atol_sensitivity)
        while t < self.finalTime:
            t = sim.step()
            if mechanicalBoundary =='constant volume':
                state = np.hstack([t,batchReactor.thermo.P,batchReactor.mass,batchReactor.volume,
                               batchReactor.T, batchReactor.thermo.X])
            else:
                state = np.hstack([t,batchReactor.thermo.P, batchReactor.mass,
                               batchReactor.T, batchReactor.thermo.X])

            self.timeHistory.loc[counter] = state
            if self.kineticSens == 1:
                counter_1 = 0
                for observable,reaction in itertools.product(self.observables, range(self.processor.solution.n_reactions)):
                    tempArray[self.observables.index(observable)][reaction] = sim.sensitivity(observable,
                                                                                                    reaction)
                    counter_1 +=1
                    if counter_1 % self.processor.solution.n_reactions == 0:
                        dfs[self.observables.index(observable)] = dfs[self.observables.index(observable)].append(((
                            pd.DataFrame(tempArray[self.observables.index(observable)])).transpose()),
                            ignore_index=True)
            counter+=1
        
 
        
        if self.timeHistories != None:

            self.timeHistory.time = self.timeHistory.time + self.time_shift_value
            
            #self.timeHistory.time = self.timeHistory.time + 0
            
            self.timeHistories.append(self.timeHistory)
            ############################################################

        if self.kineticSens == 1:
            numpyMatrixsksens = [dfs[dataframe].values for dataframe in range(len(dfs))]
            self.kineticSensitivities = np.dstack(numpyMatrixsksens)
            return self.timeHistory,self.kineticSensitivities
        else:
            return self.timeHistory

    #interpolate the most recent time history against the oldest by default
    #working_data used if have list not pandas frame
    #return more data about what was interpolated in a tuple?
    def interpolate_time(self,index:int=None,time_history=None):
        '''
        This function interpolates and returns the most recent time history 
        against the original time history by default, unless a specific time 
        history index  or time history is passed in. If an index is passed in 
        then an interpolated time history associated with that index in the
        list is returned. If a specific time_history is passed in then an
        interpolated version of that time history is returned.         

        Parameters
        ----------
        index : int, optional
            The index value of the specific time history to be interpolated. 
            The default is None.
        time_history : Pandas Data Frame, optional
            Pandas dataframe containing the time history from a simulation.
            The default is None.

        Returns
        -------
        interpolatedTimeHistory: Pandas Data Frame
            Interpolated time history.

        '''

        if self.timeHistories == None:
            print("Error: this simulation is not saving time histories, reinitialize with flag")
            return -1
        else:
            if index is not None and time_history is not None:
                print("Error: can only specify one of index, time_history")
                return -1
            if index is None:
                index = -1
            
            if time_history is None:
                return self.interpolation(self.timeHistories[0],self.timeHistories[index],["temperature","pressure"]+self.observables)
            else:
                return self.interpolation(self.timeHistories[0],time_history,["temperature","pressure"]+self.observables)
            
    #assumes most recent time histories are the correct ones to interpolate on
    #interpolates agains the original time history
    def interpolate_species_adjustment(self):
        '''
        This function interpolates the time history of a species adjustment run, 
        against the original time history.
        '''
        interpolated_data = []
        species_to_loop = set(self.conditions.keys()).difference(['Ar','AR','HE','He','Kr','KR','Xe','XE','NE','Ne'])

        for x in range(0,len(species_to_loop)):
            interpolated_data.insert(0,self.interpolate_time(index=-1-x))

        return interpolated_data
    def interpolate_species_sensitivities(self):
        '''
        This function interpolates the time history of a species sensitivity 
        against the original time history.
        '''        
        interpolated_data = self.interpolate_species_adjustment()
        interpolated_sens = []
        
        ind_off = len(self.timeHistories)-len(set(self.conditions.keys()).difference(['Ar','AR','HE','He','Kr','KR','Xe','XE','NE','Ne']))
        for i,th in enumerate(interpolated_data):
            ind = ind_off + i 

            interpolated_sens.append(self.interpolate_physical_sensitivities(index=ind,time_history=th))

        return interpolated_sens

    #interpolate a range of time histories agains the original
    #possibly add experimental flag to do range with exp data                  
    #end is exclusive
    #start here tomorrow
    def interpolate_range(self,begin:int,end:int):
        '''
        Function that defines a time range for interpolation.

        Parameters
        ----------
        begin : int
            Time to begin interpolation.
        end : int
            Time to end interpolation.

        Returns
        -------
        interpolated_data : Pandas Data Frame
            Interpolated time history.

        '''
        if begin<0 or end>len(self.timeHistories):
            print("Error: invalid indices")
        if self.timeHistories == None:
            print("Error: simulation is not saving time histories")

        interpolated_data = []
        for x in range(begin,end):
            interpolated_data.append(self.interpolate_time(index=x))
        return interpolated_data
        
    #interpolates agains the original time history
    def interpolate_physical_sensitivities(self, index:int=-1,
                                           time_history=None):
        '''
        This function interpolates the time history of a physical sensitivity, 
        excluding species, against the original time history and returns a 
        numpy array.        

        Parameters
        ----------
        index : int, optional
            The index value of the specific time history to be interpolated. 
            The default is -1.
        time_history : Pandas Data Frame, optional
            Pandas dataframe containing the time history from a simulation.
            The default is None.

        Returns
        -------
        sensitivity: Pandas Data Frame.
        Physical sensitivity Data Frame.

        '''
          
        interpolated_time = self.interpolate_time(index) if time_history is None else time_history
        #print(interpolated_time)
        #calculate which dk
        dk = self.dk[index]
        
        print()
        sensitivity = self.sensitivityCalculation(self.timeHistories[0][self.observables],
                                                  interpolated_time[self.observables],self.observables,dk)
        if self.physSensHistories != None:
            self.physSensHistories.append(sensitivity)
        #print('this is sensitivity')
        #print(sensitivity)
        return sensitivity
    
    #returns a 3D array of interpolated time histories corrosponding to physical sensitivities
    def interpolate_experimental_kinetic(self, pre_interpolated = []):
        '''
        This function interpolates kinetic sensitivities to experimental data.

        Parameters
        ----------
        pre_interpolated : list, optional
            List of kinetic sensitivties to be interpolated. If not provided
            the function will look for kinetic sensitivities that were saved
            as an attribute.
            The default is [].

        Returns
        -------
        flipped: numpy array
            Interpolated 3d array containing sensitivities.

        '''
           
        if self.experimentalData == None:
            print("Error: experimental data must be loaded")
            return -1
        if len(pre_interpolated) == 0 and not self.kineticSensitivities.any():
            print("Error: must specify pre_interpolated or have kineticSensitivities run first")
            return -1
        elif len(pre_interpolated)>0:
            array = pre_interpolated
        else:
            array = self.kineticSensitivities
        exp_interp_array = []
        #if len(self.experimentalData) < array.shape[2]:
        #    print("Error: mismatch between kineticSensitivities observables and given experimental data")
        #    return -1
        #exp data and kineticSensitivities must match in size and order of observables
        for i,frame in enumerate(self.experimentalData):
            if i > array.shape[2]:
                break
            sheet = array[:,:,i]
            exp_interp_array.append([])
            for time_history in sheet.T:
                # new_history = np.interp(frame.ix[:,0],
                #                        self.timeHistories[0]['time'],
                #                        time_history)
                new_history = np.interp(frame.iloc[:,0],
                             self.timeHistories[0]['time'],
                             time_history)
                new_history = new_history.reshape((new_history.shape[0],
                                                  1))
                exp_interp_array[-1].append(new_history)
        flipped = [] 
        for x in exp_interp_array:
            flipped.append(np.hstack(x))
        return flipped
   
    def map_and_interp_ksens(self,time_history=None):
        '''
        This function maps kinetic sensitivity calculations returned from cantera
        to kineitcs parameters A,n and Ea, as well as interpolates them to the 
        corresponding experimental data. It returns a dictonary containing the 
        interpolated kinetic senstivities.        

        Parameters
        ----------
        time_history : Pandas Data Frame, optional
            The original time history is required for obtaining temperature 
            values in order to do the mapping. The default is None.

        Returns
        -------
        dict
            Dictionary containing the mapping for kinetic sensitivities.

        '''
        
        A = self.kineticSensitivities
        N = np.zeros(A.shape)
        Ea = np.zeros(A.shape)
        for i in range(0,A.shape[2]):
            sheetA = A[:,:,i] #sheet for specific observable
            for x,column in enumerate(sheetA.T):
                N[:,x,i]= np.multiply(column,np.log(self.timeHistories[0]['temperature'])) if time_history is None else np.multiply(column,np.log(time_history['temperature']))
                #not sure if this mapping is correct, check with burke and also update absorption mapping
                #to_mult_ea = np.divide(-1,np.multiply(1/ct.gas_constant,self.timeHistories[0]['temperature'])) if time_history is None else np.divide(-1,np.multiply(ct.gas_constant,time_history['temperature']))
                to_mult_ea = np.divide(-1,np.multiply(1,self.timeHistories[0]['temperature'])) if time_history is None else np.divide(-1,np.multiply(1,time_history['temperature']))
                Ea[:,x,i]= np.multiply(column,to_mult_ea)
                
                
        return {'A':self.interpolate_experimental_kinetic(A),
                'N':self.interpolate_experimental_kinetic(N),
                'Ea':self.interpolate_experimental_kinetic(Ea)}
            

    #assumes pre_interpolated has been interpolated against the original time history
    #assumes pre_interpolated is a list of dataframes where each dataframe is a time history
    #single is a single dataframe representing one time history/run of the simulation
    def interpolate_experimental(self,pre_interpolated = [], single = None):
        '''
        This function interpolates the time history of a physical sensitivity 
        against the corresponding experimental data and returns a pandas 
        dataframe.        

        Parameters
        ----------
        pre_interpolated : list, optional
            has been interpolated against the original time history,
        assumes pre_interpolated is a list of dataframes where each dataframe 
        is a time history. The default is [].
        single : Pandas Data Frame, optional
            Single Pandas Data Frame to be interpolated against experimental
            data. The default is None.

        Returns
        -------
        int_exp: Pandas Data Frame
            Interpolated data frame.

        '''
         
        if self.timeHistories == None:
            print("Error: can't interpolate without time histories")
            return -1
        if self.experimentalData == None:
            print("Error: must have experimental data before interpolation")
            return -1
        if len(pre_interpolated)!=0 and single != None:
            print("Error: can only specify one of pre_interpolated, single")
        if single is not  None:
            pre_interpolated = [single]
        
        int_exp = []
        #This portion of the function removes the pressure and the temperature from the pre_interpolated frames
        #so that when it gets interpolated against experimental data the temperature and pressure are not there
        if isinstance(pre_interpolated[0],pd.DataFrame):
            if 'pressure' in pre_interpolated[0].columns.tolist() and 'temperature' in pre_interpolated[0].columns.tolist():
                pre_interpolated = [df.drop(columns=['temperature','pressure']) for df in pre_interpolated]

        #make sure you put the observables list in the correct order
        #check what order the experimental list is parsed in
        #making new observables list 
        if single is not None:
            mole_fraction_and_concentration_observables= self.moleFractionObservables + self.concentrationObservables
            
            mole_fraction_and_concentration_observables   = [x for x in mole_fraction_and_concentration_observables if x is not None]      
            for time_history in pre_interpolated:
                array_list = []
                max_size = 0
                for i, observable in enumerate(mole_fraction_and_concentration_observables):
                    interpolated_column = np.interp(self.experimentalData[i]['Time'].values,
                                                    self.timeHistories[0]['time'].values,
                                                    time_history[observable].values)
                    
                    interpolated_column = np.reshape(interpolated_column,((interpolated_column.shape[0],1)))
                    array_list.append(interpolated_column)
                    max_size = max(interpolated_column.shape[0],max_size)
                padded_arrays = []
                for arr in array_list:
                    if arr.shape[0] < max_size:
                        padded_arrays.append(np.pad(arr,
                                                    ((0,max_size-arr.shape[0]),(0,0)),
                                                    'constant',constant_values = np.nan))
                    else:
                        padded_arrays.append(arr)
                    
                np_array = np.hstack((padded_arrays))
                new_frame = pd.DataFrame(np_array)
                int_exp.append(new_frame)
                
            for x in int_exp:
                x.columns = mole_fraction_and_concentration_observables
            
            return int_exp[0]
                    
                    
        #check and make sure this part actually works for what we want  for interpolating the pre interpolated time histories 
        #make sure we are getting the correct columns            
        else:
            for time_history in pre_interpolated:
                array_list = []
                max_size = 0
                for i,frame in enumerate(self.experimentalData): #each frame is data for one observable
                    if i>len(self.observables):
                        break
                    #change these bboth to use observable 
                    # interpolated_column= np.interp(frame.ix[:,0],
                    #                                self.timeHistories[0]['time'],
                    #                                time_history.ix[:,i])
                    
                    interpolated_column= np.interp(frame.iloc[:,0],
                                                   self.timeHistories[0]['time'],
                                                   time_history.iloc[:,i])

                    

                    interpolated_column= np.reshape(interpolated_column,
                                                ((interpolated_column.shape[0],1)))
                    array_list.append(interpolated_column)
                    max_size = max(interpolated_column.shape[0],max_size)
                padded_arrays= []
                for arr in array_list:
                    if arr.shape[0] < max_size:
                        padded_arrays.append(np.pad(arr,
                                            ((0,max_size - arr.shape[0]),(0,0)),
                                            'constant',constant_values=np.nan))
                    else:
                        padded_arrays.append(arr)
                np_array = np.hstack((padded_arrays))
                new_frame = pd.DataFrame(np_array)
                int_exp.append(new_frame)
            
            for x in int_exp:
                x.columns = self.observables[0:len(self.experimentalData)]
            if single is not  None:
                return int_exp[0]
            else:
                return int_exp

    def interpolation(self,originalValues,newValues, thingBeingInterpolated):   
        '''
        This function is the base interpolation function, interpolating one set 
        of data to another on a per time basis and returns a pandas dataframe.         

        Parameters
        ----------
        originalValues : Pandas Data Frame
            Original dataframe of time history.
        newValues : Pandas Data Frame
            New dataframe of time history.
        thingBeingInterpolated : list
            List of observable names.

        Returns
        -------
        interpolatedData: Pandas Data Frame
            Interpolated Data Frame.

        '''
        #interpolating time histories to original time history      
        
        if isinstance(originalValues,pd.DataFrame) and isinstance(newValues,pd.DataFrame):
            tempDfForInterpolation = newValues[thingBeingInterpolated]
            #tempListForInterpolation = [tempDfForInterpolation.ix[:,x].values for x in range(tempDfForInterpolation.shape[1])]
            tempListForInterpolation = [tempDfForInterpolation.iloc[:,x].values for x in range(tempDfForInterpolation.shape[1])]
            interpolatedData = [np.interp(originalValues['time'].values,newValues['time'].values,tempListForInterpolation[x]) for x in range(len(tempListForInterpolation))]
            interpolatedData = [pd.DataFrame(interpolatedData[x]) for x in range(len(interpolatedData))]
            interpolatedData = pd.concat(interpolatedData, axis=1,ignore_index=True)
            interpolatedData.columns = thingBeingInterpolated
        else:
            print("Error: values must be pandas dataframes")
            return -1
        return interpolatedData


    def sensitivityCalculation(self,originalValues,newValues,thingToFindSensitivtyOf,dk=.01):
        '''
         This function calculates log/log sensitivity and returns a
         pandas dataframe.

        Parameters
        ----------
        originalValues : Pandas Data Frame
            Original dataframe of time history.
        newValues : Pandas Data Frame
             New dataframe of time history.
        thingToFindSensitivtyOf : list
            List of observable names.
        dk : float, optional
            The decimal value of the percentage by which the original value 
            was perturbed. The default is .01.

        Returns
        -------
        sensitivity: Pandas Data Frame
            Sensitivity of observables.

        '''
     
        if isinstance(originalValues,pd.DataFrame) and isinstance(newValues,pd.DataFrame):
            newValues.columns = thingToFindSensitivtyOf
            newValues = newValues.applymap(np.log)
            originalValues = originalValues.applymap(np.log)
            #tab
            
            sensitivity = (newValues.subtract(originalValues)/dk)
            return sensitivity
        else:
            print("Error: wrong datatype, both must be pandas data frames")
            return -1

    
    def importExperimentalData(self,csvFileList):
        '''
        This function imports experimental data in csv format and returns a 
        list of pandas dataframes.        

        Parameters
        ----------
        csvFileList : list
            List of csv file directories.

        Returns
        -------
        experimentalData : list
            Experimental data from csv files as pandas data frames 
            stored in a list.

        '''
           
        print('Importing shock tube data the following csv files...') 
        print(csvFileList)
        experimentalData = [pd.read_csv(csv) for csv in csvFileList]
        experimentalData = [experimentalData[x].dropna(how='any') for x in range(len(experimentalData))]
        experimentalData = [experimentalData[x].apply(pd.to_numeric, errors = 'coerce').dropna() for x in range(len(experimentalData))]
        for x in range(len(experimentalData)):
            experimentalData[x] = experimentalData[x][~(experimentalData[x][experimentalData[x].columns[1]] < 0)]
        self.experimentalData = experimentalData
        return experimentalData

    def savingInterpTimeHistoryAgainstExp(self,timeHistory):
        '''
        This function writes the time history which is interpolated against 
        experimetnal data as a class object.        

        Parameters
        ----------
        timeHistory : Pandas Data Frame
            Time history interpolated against experimental data .

        Returns
        -------
        timeHistoryInterpToExperiment: Pandas Data Frame
            Time history that is being saved.

        '''

        self.timeHistoryInterpToExperiment = timeHistory
        
    def interpolatePressureandTempToExperiment(self,simulation,experimental_data):
        '''
        This function interpolates the pressure and temperature time history 
        from a simulation against the corresponding experimental data.        

        Parameters
        ----------
        simulation : class variable
             The simulation assoicated with the time history to be 
             interpolated.
        experimental_data : list
            List of corresponding experimental data dataframtes .

        Returns
        -------
        list_of_df : list
            List of interpolated data frames for pressure and temperature.

        '''
        
        p_and_t = ['pressure','temperature']
        list_of_df = []
        for df in experimental_data:
            temp = []
            for variable in p_and_t:
                interpolated_data = np.interp(df['Time'],simulation.timeHistories[0]['time'],simulation.timeHistories[0][variable])
                interpolated_data = interpolated_data.reshape((interpolated_data.shape[0],1))
                temp.append(interpolated_data)                
            temp = np.hstack(temp)    
            temp = pd.DataFrame(temp)
            temp.columns = p_and_t
            list_of_df.append(temp)
            self.pressureAndTemperatureToExperiment = list_of_df
            
        return list_of_df
    
    def calculate_time_shift_sensitivity(self,simulation,experimental_data,dk=1e-8):
        '''
        This function interpolates the pressure and temperature time history 
        from a simulation against the corresponding experimental data.        

        Parameters
        ----------
        simulation : class variable
            The simulation assoicated with the time history to 
            be interpolated.
        experimental_data : list
             List of corresponding experimental data dataframtes .
        dk : float, optional
            Decimal percentage by which time values are perturbed. Default is 
            1e-8.

        Returns
        -------
        time_shift_sensitivity : Pandas Data Frame
            List of Pandas Data Frames containing time shift sensitivity 
            values.

        '''
         
        lst_obs = simulation.moleFractionObservables + simulation.concentrationObservables
        lst_obs = [i for i in lst_obs if i] 

            
        one_percent_of_average = dk
                        
        
        original_time = simulation.timeHistories[0]['time']
        new_time = original_time + one_percent_of_average
            

        #interpolate to the orignal time 
        interpolated_against_original_time = []
        for i,obs in enumerate(lst_obs):
            interpolated_original_observable_against_original_time = np.interp(original_time,new_time,simulation.timeHistories[0][lst_obs[i]])
            s1 = pd.Series(interpolated_original_observable_against_original_time,name=lst_obs[i])
            interpolated_against_original_time.append(s1)
        
        observables_interpolated_against_original_time_df = pd.concat(interpolated_against_original_time,axis=1)
        
        #calculate sensitivity
        
        calculated_sensitivity = []
        for i,obs in enumerate(lst_obs):
                      
           sens = (observables_interpolated_against_original_time_df[obs].apply(np.log) - simulation.timeHistories[0][obs].apply(np.log))/one_percent_of_average
           s1 = pd.Series(sens,name=lst_obs[i])
           calculated_sensitivity.append(s1)
            
        calculated_sensitivity_df = pd.concat(calculated_sensitivity,axis=1)

        
        interpolated_sensitivity = []
        for i,obs in enumerate(lst_obs): 
            interpolated_sensitivty_per_original_observable = np.interp(experimental_data[i]['Time'],simulation.timeHistories[0]['time'],calculated_sensitivity_df[obs])
            s1 = pd.Series(interpolated_sensitivty_per_original_observable,name=obs)
            interpolated_sensitivity.append(s1)

        
        interpolated_sensitivity_df = pd.concat(interpolated_sensitivity,axis=1)

        time_shift_sensitivity = interpolated_sensitivity_df
        


        self.time_shift_sensitivity = time_shift_sensitivity
        average_time=1
        self.average_time = average_time
        return time_shift_sensitivity
    
