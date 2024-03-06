import itertools
import numpy as np
import cantera as ct
import pandas as pd
import re
import pickle
import numpy.polynomial.polynomial as poly

from .. import simulation as sim
from ...cti_core import cti_processor as ctp

class VariablePressurebatchReactor(sim.Simulation):
    
    def __init__(self,pressure:float,temperature:float,observables:list,
                 kineticSens:int,physicalSens:int,conditions:dict,
                 initialTime,finalTime,thermalBoundary,mechanicalBoundary,
                 processor:ctp.Processor=None,cti_path="",save_timeHistories:int=0, 
                 save_physSensHistories=0,moleFractionObservables:list=[],
                 absorbanceObservables:list=[],concentrationObservables:list=[],
                 fullParsedYamlFile:dict={},
                 time_shift_value = 0, atol=1e-15, rtol=1e-9,rtol_sens=0.0001,
                 atol_sens=1e-6,volumeTrace='',exactDerivFlag=False):

        '''
        Child class of shock Tubes. Inherits all attributes and
        methods including __init__(). Also has its own internal
        init method due to additional data requirements
    
        Input:
            - initialTime = float, time the simulation will begin at in [s]
            - finalTime = float, time the simulation will end at in [s]
            - thermalBoundary = string, the boundary condtion for the batchReactor.
              For example, adiabatic, or isothermal
            - mechanicalBoundary = string, the thermal boundary condition for
              the batchReactor. For example, constant pressure or constant volume
            - histories: save the timehistories of all runs of the simulation
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
        self.volume_trace_csv = volumeTrace
        self.volume_trace_class = VolumeProfile(self.volume_trace_csv)
        self.volume_trace_class_exact_derivitive = VolumeProfileExactDerivative(self.volume_trace_csv)
        self.exact_deriv_flag = exactDerivFlag
        

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


    def run(self,initialTime:float=-1.0, finalTime:float=-1.0):
        '''
        Run the shock tube simulation
        '''
        
        #p_alpha = meq.Master_Equation.chebyshev_specific_poly(self,l,meq.Master_Equation.calc_reduced_P(self,target_press_new*101325))
        if self.exact_deriv_flag==False:   
            inp_time = self.volume_trace_class.time
            inp_vol = self.volume_trace_class.volume
        if self.exact_deriv_flag==True:
            inp_time = self.volume_trace_class_exact_derivitive.time
            inp_vol = self.volume_trace_class_exact_derivitive.volume
        
        if initialTime == -1.0:
            initialTime = self.initialTime 
        if finalTime == -1.0:
            finalTime = self.finalTime
        self.timeHistory = None
        self.kineticSensitivities= None #3D numpy array, columns are reactions with timehistories, depth gives the observable for those histories
        conditions = self.settingVariablePressurebatchReactorConditions()
        mechanicalBoundary = conditions[1]
        #same solution for both cp and cv sims
        if mechanicalBoundary == 'constant pressure':

            print('variable pressure shock tube is not a constant pressure simulation')
        else:
            if self.exact_deriv_flag == False:
                vpst = ct.IdealGasReactor(self.processor.solution)
                env = ct.Reservoir(ct.Solution('air.xml'))
                wall = ct.Wall(vpst, env, A=1.0, velocity=self.volume_trace_class)
                sim = ct.ReactorNet([vpst])
                sim.set_max_time_step(inp_time[1])
            if self.exact_deriv_flag == True:
                vpst = ct.IdealGasReactor(self.processor.solution)
                env = ct.Reservoir(ct.Solution('air.xml'))
                wall = ct.Wall(vpst, env, A=1.0, velocity=self.volume_trace_class_exact_derivitive)
                sim = ct.ReactorNet([vpst])
                sim.set_max_time_step(inp_time[1])            

        sim.rtol=self.rtol
        sim.atol=self.atol
        sim.rtol_sensitivity=self.rtol_sensitivity
        sim.atol_sensitivity=self.atol_sensitivity
        
        columnNames = [vpst.component_name(item) for item in range(vpst.n_vars)]
        columnNames = ['time']+['pressure']+columnNames
        self.timeHistory = pd.DataFrame(columns=columnNames)

        if self.kineticSens == 1:
            for i in range(self.processor.solution.n_reactions):
                vpst.add_sensitivity_reaction(i)
            dfs = [pd.DataFrame() for x in range(len(self.observables))]
            tempArray = [np.zeros(self.processor.solution.n_reactions) for x in range(len(self.observables))]

        t = self.initialTime
        counter = 0
        #print(sim.rtol_sensitivity,sim.atol_sensitivity)
        
     
        vol_sol = ct.SolutionArray(self.processor.solution, extra=["time", "volume"])
        
        self.vol_sol = vol_sol

        #while t < self.finalTime and RCM.T < 2500:
        #should we have this temperature limiting factor?
        #while t < self.finalTime and RCM.T<2500:
        while t < self.finalTime:
            t = sim.step()
            if mechanicalBoundary =='constant volume':
                state = np.hstack([t,vpst.thermo.P,vpst.mass,vpst.volume,
                               vpst.T, vpst.thermo.X])
                vol_sol.append(time=sim.time, T=vpst.T, P=vpst.thermo.P, X=vpst.thermo.X, volume=vpst.volume)
                #t = sim.step()
            else:
                print('variable pressure shock tube has variable volume')

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
            #print(self.timeHistory)
            ############################################################

        if self.kineticSens == 1:
            numpyMatrixsksens = [dfs[dataframe].values for dataframe in range(len(dfs))]
            self.kineticSensitivities = np.dstack(numpyMatrixsksens)
            return self.timeHistory,self.kineticSensitivities
        else:
            return self.timeHistory , self.vol_sol

    #interpolate the most recent time history against the oldest by default
    #working_data used if have list not pandas frame
    #return more data about what was interpolated in a tuple?
    def settingVariablePressurebatchReactorConditions(self):
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
    
class VolumeProfile(object):
    """
    Set the velocity of the piston by using a user specified volume
    profile. The initialization and calling of this class are handled
    by the `cantera.Func1` interface of Cantera.
    The velocity is calculated by assuming a unit area and using the
    forward difference, calculated by `numpy.diff`. This function is
    only called once when the class is initialized at the beginning of
    a problem so it is efficient.
    Parameters
    ----------
    time: `numpy.ndarray`
        Array or list of time values
    volume: `numpy.ndarray`
        Array or list of volume values
    Attributes
    ----------
    time: `numpy.ndarray`
        Array of time values
    volume: `numpy.ndarray`
        Array of volume values
    velocity: `numpy.ndarray`
        Array of velocity values
    """

    def __init__(self, volumeProfileCsv):
        # The time and volume are stored as lists in the keywords
        # dictionary. The volume is normalized by the first volume
        # element so that a unit area can be used to calculate the
        # velocity.
        self.df = pd.read_csv(volumeProfileCsv,float_precision='round_trip')
        self.arr = self.df.to_numpy()

        
        self.time = self.arr[:, 0]
        self.vol_temp = self.arr[:, 1]
        self.volume = self.vol_temp/self.vol_temp[0]

        # The velocity is calculated by the forward difference.
        # numpy.diff returns an array one element smaller than the
        # input array, so we append a zero to match the length of the
        # self.time array.
        self.velocity = np.diff(self.volume)/np.diff(self.time)
        self.velocity = np.append(self.velocity, 0)

    def __call__(self, t):
        """Return the velocity when called during a time step.
        Parameters
        ----------
        t : `float`
            Current simulation time.
        """

        if t <= self.time[-1] and t >= self.time[0]:
            # prev_time_point is the previous value in the time array
            # after the current simulation time
            prev_time_point = self.time[self.time <= t][-1]
            # index is the index of the time array where
            # prev_time_point occurs
            index = np.where(self.time == prev_time_point)[0][0]
            return self.velocity[index]
        else:
            return 0
        
        
        
        
class VolumeProfileExactDerivative(object):


    """
    Set the velocity of the piston by using a user specified volume
    profile. The initialization and calling of this class are handled
    by the `cantera.Func1` interface of Cantera.
    The velocity is calculated by assuming a unit area and using the
    forward difference, calculated by `numpy.diff`. This function is
    only called once when the class is initialized at the beginning of
    a problem so it is efficient.
    Parameters
    ----------
    time: `numpy.ndarray`
        Array or list of time values
    volume: `numpy.ndarray`
        Array or list of volume values
    Attributes
    ----------
    time: `numpy.ndarray`
        Array of time values
    volume: `numpy.ndarray`
        Array of volume values
    velocity: `numpy.ndarray`
        Array of velocity values
    """

    def __init__(self, volumeProfileCsv):
        # The time and volume are stored as lists in the keywords
        # dictionary. The volume is normalized by the first volume
        # element so that a unit area can be used to calculate the
        # velocity.
        self.df = pd.read_csv(volumeProfileCsv,float_precision='round_trip')
        self.arr = self.df.to_numpy()
        
        self.time = self.arr[:, 0]

        self.vol_temp = self.arr[:, 1]
        
        self.volume = self.vol_temp/self.vol_temp[0]   
        
        self.volume = pd.Series(self.volume)
        
        self.time = pd.Series(self.time)
        #normalize the volume
        min_index = self.volume.idxmin()

        
        coeffs = np.polynomial.chebyshev.chebfit(self.time[:min_index], self.volume[:min_index], 15, rcond=None, full=False, w=None)
        y_calculated1 = np.polynomial.chebyshev.chebval(self.time[:min_index], coeffs, tensor=True)
        
        coeffs2 = np.polynomial.chebyshev.chebfit(self.time[min_index:], self.volume[min_index:], 15, rcond=None, full=False, w=None)
        y_calculated2 = np.polynomial.chebyshev.chebval(self.time[min_index:], coeffs2, tensor=True)        
        
        
        deriv1 = np.polynomial.chebyshev.chebder(coeffs)
        y_deriv_calculated = np.polynomial.chebyshev.chebval(self.time[:min_index], deriv1, tensor=True)        
        
        
        
        deriv2 = np.polynomial.chebyshev.chebder(coeffs2)
        y_deriv_calculated2 = np.polynomial.chebyshev.chebval(self.time[min_index:], deriv2, tensor=True)
        
        y_deriv_calculated = y_deriv_calculated.to_numpy()
        y_deriv_calculated2 = y_deriv_calculated2.to_numpy()
    
        y_deriv_calculated = y_deriv_calculated.reshape((y_deriv_calculated.shape[0],1))
        y_deriv_calculated2 = y_deriv_calculated2.reshape((y_deriv_calculated2.shape[0],1))
        deriv=np.vstack((y_deriv_calculated,y_deriv_calculated2))        
        deriv = deriv.flatten()
        self.time = self.time.to_numpy()

        self.velocity = deriv

        # The velocity is calculated by fitting the volume function to a polynomial.
        # The derivitive is then calculated of this fitted polynomial to get the velocity

    def __call__(self, t):
        """Return the velocity when called during a time step.
        Parameters
        ----------
        t : `float`
            Current simulation time.
        """

        if t <= self.time[-1] and t >= self.time[0]:
            # prev_time_point is the previous value in the time array
            # after the current simulation time
            prev_time_point = self.time[self.time <= t][-1]
            # index is the index of the time array where
            # prev_time_point occurs
            index = np.where(self.time == prev_time_point)[0][0]
            return self.velocity[index]
        else:
            return 0      
        
        
        
        
        
