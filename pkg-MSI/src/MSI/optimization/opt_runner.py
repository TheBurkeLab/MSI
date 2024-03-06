import MSI.simulations as sim
import re
import MSI.cti_core.cti_processor as pr
import MSI.optimization.matrix_loader as ml
import MSI.simulations.absorbance.curve_superimpose as csp
import MSI.simulations.yaml_parser as yp
import MSI.simulations.instruments.batch_reactor as st
import MSI.simulations.instruments.jsr_steadystate as jsr
import MSI.simulations.instruments.flames as fl
import MSI.simulations.instruments.ignition_delay as ig
import MSI.simulations.instruments.flow_reactor as fr
import pandas as pd
import numpy as np
import os

#acts as front end to the rest of the system


# takes the data from one experiment and puts it in a dict of dicts
# that follows the format of the S matrix
# only need the last 3 elements of the interpolated absorbance
# absorbance is of form [interp_original,interp_abs_kinetic_sens,interp_abs_phys_sens,interp_abs_coef_sens]
# where each list element is a dict. keys are wavelengths, values are the sensitivities for that wavelength 
# psens should match interpolated_tp and species sens in size, but again is dict with wavelength keys
# index from 1, so if you have 3 experiments, their indices will be 1,2,3
class Optimization_Utility(object):
    def __init__(self,yaml_list=None):
        self.matrix = None
        self.yaml_list=yaml_list
        
    def build_single_exp_dict(self,exp_index:int,
                              simulation,
                              interpolated_kinetic_sens:dict,
                              interpolated_tp_sens:list,
                              interpolated_species_sens:list,
                              interpolated_absorbance:list=[],
                              experimental_data:list =[],
                              absorbance_experimental_data:list=[],
                              time_history_interpolated_against_absorbance_experiment:dict={},
                              absorbance_calculated_from_model=None,
                              yaml_dict:dict={},
                              interpolated_time_shift_sens=None,
                              interpolated_abs_time_shift = None):
        exp_dict = {}
        exp_dict['index']              = exp_index
        exp_dict['simulation']         = simulation
        #print(yaml_dict,'wtf is going on')
        #print('HELLO I AM TRAPPED IN A COMPUTER')
        if interpolated_kinetic_sens==None:
            exp_dict['ksens']  = None
            exp_dict['temperature'] = None
            exp_dict['pressure'] = None
            exp_dict['species'] = None
        else:
            #print('I LIKE POTATOES')
            exp_dict['ksens']              = interpolated_kinetic_sens
            exp_dict['temperature']        = interpolated_tp_sens[0]
            exp_dict['pressure']           = interpolated_tp_sens[1]
            
            exp_dict['species']            = interpolated_species_sens
            
        exp_dict['simulation_type'] = simulation.fullParsedYamlFile['simulationType']
        exp_dict['experiment_type'] = simulation.fullParsedYamlFile['experimentType']
        exp_dict['observables']        = simulation.observables

        #needs to be in the order of mole fraction csv files + concentration csv files 
        exp_dict['experimental_data']  = experimental_data
        # start here 
        #print(exp_dict['simulation_type'])
        
        if re.match('[Ss]pecies[- ][Pp]rofile',simulation.fullParsedYamlFile['experimentType']) and re.match('[Bb]atch[- ][Rr]eactor',simulation.fullParsedYamlFile['simulationType']):
            exp_dict['concentration_observables'] = simulation.concentrationObservables
            exp_dict['mole_fraction_observables'] = simulation.moleFractionObservables
            exp_dict['time_shift'] = interpolated_time_shift_sens
            exp_dict['uncertainty']        = self.build_uncertainty_batch_reactor_dict(exp_dict['simulation'].fullParsedYamlFile)
            exp_dict['simulation_type'] = simulation.fullParsedYamlFile['simulationType']
            exp_dict['flame_speed_observables']= [None]
            exp_dict['ignition_delay_observables'] = [None]
            #print('FUCK SPYDER')

        #decide how we want to build uncertainty dict and if we want to pass in the parsed yaml file?
        elif re.match('[Ii]gnition[- ][Dd]elay',simulation.fullParsedYamlFile['experimentType']) and re.match('[Bb]atch[- ][Rr]eactor',simulation.fullParsedYamlFile['simulationType']):
            exp_dict['time_shift'] = interpolated_time_shift_sens
            exp_dict['uncertainty']= self.build_uncertainty_ignition_delay_dict(exp_dict['simulation'].fullParsedYamlFile)
            exp_dict['flame_speed_observables']= [None]
            exp_dict['concentration_observables'] = [None]
            exp_dict['mole_fraction_observables'] = [None]
            exp_dict['ignition_delay_observables'] = simulation.ignitionDelayObservables
            exp_dict['conditions_dict_list'] = simulation.fullParsedYamlFile['conditions_dict_list']
            exp_dict['conditions_to_run']=simulation.fullParsedYamlFile['conditions_to_run']

        elif re.match('[Jj][Ss][Rr]',yaml_dict['simulationType']) or  re.match('[Jj]et[- ][Ss]tirred[- ][Rr]eactor',yaml_dict['simulationType']):
            exp_dict['concentration_observables'] = simulation.concentrationObservables
            exp_dict['mole_fraction_observables'] = simulation.moleFractionObservables
            exp_dict['restime_sens']=interpolated_tp_sens[2]
            exp_dict['volume']=yaml_dict['volume']
            exp_dict['residence_time']=yaml_dict['residence_time']
            exp_dict['uncertainty']=self.build_uncertainty_jsr_dict(exp_dict['simulation'].fullParsedYamlFile)
            exp_dict['simulation_type'] = yaml_dict['simulationType']
            exp_dict['flame_speed_observables']= [None]
            exp_dict['ignition_delay_observables'] = [None]

        elif re.match('[Ff]lame[ -][Ss]peed',yaml_dict['simulationType']) and re.match('[Oo][Nn][Ee]|[1][ -][dD][ -][Ff]lame',yaml_dict['experimentType']):
            
            exp_dict['flame_speed_observables']= simulation.flameSpeedObservables
            exp_dict['concentration_observables'] = [None]
            exp_dict['mole_fraction_observables'] = [None]
            exp_dict['uncertainty']=self.build_uncertainty_flame_speed_dict(exp_dict['simulation'].fullParsedYamlFile)
            exp_dict['ignition_delay_observables'] = [None]
        
        elif re.match('[Ss]pecies[- ][Pp]rofile',simulation.fullParsedYamlFile['experimentType']) and re.match('[Ff]low[ -][Rr]eactor',simulation.fullParsedYamlFile['simulationType']):
            exp_dict['concentration_observables'] = simulation.concentrationObservables
            exp_dict['mole_fraction_observables'] = simulation.moleFractionObservables
            exp_dict['time_shift'] = interpolated_time_shift_sens
            exp_dict['uncertainty']        = self.build_uncertainty_flow_reactor_dict(exp_dict['simulation'].fullParsedYamlFile)
            exp_dict['simulation_type'] = simulation.fullParsedYamlFile['simulationType']
            exp_dict['flame_speed_observables']= [None]
            exp_dict['ignition_delay_observables'] = [None]   
            
        elif re.match('[Ss]pecies[- ][Pp]rofile',simulation.fullParsedYamlFile['experimentType']) and re.match('[Vv]ariable[ -][Pp]ressure[ -][Bb]atch[- ][Rr]eactor',simulation.fullParsedYamlFile['simulationType']):
            exp_dict['concentration_observables'] = simulation.concentrationObservables
            exp_dict['mole_fraction_observables'] = simulation.moleFractionObservables
            exp_dict['time_shift'] = interpolated_time_shift_sens
            exp_dict['uncertainty']        = self.build_uncertainty_batch_reactor_dict(exp_dict['simulation'].fullParsedYamlFile)
            exp_dict['simulation_type'] = simulation.fullParsedYamlFile['simulationType']
            exp_dict['flame_speed_observables']= [None]
            exp_dict['ignition_delay_observables'] = [None]
            
        elif re.match('[Ii]gnition[- ][Dd]elay',simulation.fullParsedYamlFile['experimentType']) and re.match('[Rr][Cc][Mm]',simulation.fullParsedYamlFile['simulationType']):
            exp_dict['time_shift'] = interpolated_time_shift_sens
            exp_dict['uncertainty']= self.build_uncertainty_ignition_delay_dict(exp_dict['simulation'].fullParsedYamlFile)
            exp_dict['flame_speed_observables']= [None]
            exp_dict['concentration_observables'] = [None]
            exp_dict['mole_fraction_observables'] = [None]
            exp_dict['ignition_delay_observables'] = simulation.ignitionDelayObservables
            exp_dict['conditions_dict_list'] = simulation.fullParsedYamlFile['conditions_dict_list']
            exp_dict['conditions_to_run']=simulation.fullParsedYamlFile['conditions_to_run']  
            
        if len(interpolated_absorbance) != 0:
            exp_dict['absorbance_model_data'] = interpolated_absorbance[0]
            exp_dict['absorbance_ksens']   = interpolated_absorbance[1]
            exp_dict['absorbance_psens']   = interpolated_absorbance[2]
            exp_dict['absorbance_time_shift'] = interpolated_abs_time_shift
            exp_dict['perturbed_coef']     = interpolated_absorbance[3]
            exp_dict['absorbance_observables'] = simulation.absorbanceObservables
            exp_dict['absorbance_experimental_data'] = absorbance_experimental_data
            exp_dict['absorbance_calculated_from_model'] = absorbance_calculated_from_model
            exp_dict['time_history_interpolated_against_abs'] = time_history_interpolated_against_absorbance_experiment

           
            
        return exp_dict
    
    
    def load_exp_from_file(self,yaml_exp_file_list = []):
        for file in yaml_exp_file_list:
            continue
        
    def build_uncertainty_flame_speed_dict(self,experiment_dictionary:dict={}):
        uncertainty_dict={}

        uncertainty_dict['temperature_relative_uncertainty'] = experiment_dictionary['inletTemperatureRelativeUncertainty']
        uncertainty_dict['pressure_relative_uncertainty'] = experiment_dictionary['pressureRelativeUncertainty']
        uncertainty_dict['species_relative_uncertainty'] = {'dictionary_of_values':experiment_dictionary['relativeUncertaintyBySpecies'],
                        'species':experiment_dictionary['species'], 'type_dict':experiment_dictionary['typeDict']}
        uncertainty_dict['flame_speed_relative_uncertainty'] = experiment_dictionary['flameSpeedRelativeUncertainity']
        uncertainty_dict['flame_speed_absolute_uncertainty'] = experiment_dictionary['flameSpeedAbsoluteUncertainty']        

        return uncertainty_dict
    
    def build_uncertainty_ignition_delay_dict(self,experiment_dictionary:dict={}):
        uncertainty_dict={}
        
        uncertainty_dict['temperature_relative_uncertainty'] = experiment_dictionary['tempRelativeUncertainty']
        uncertainty_dict['pressure_relative_uncertainty'] = experiment_dictionary['pressureRelativeUncertainty']
        uncertainty_dict['species_relative_uncertainty'] = {'dictionary_of_values':experiment_dictionary['relativeUncertaintyBySpecies'],
                        'type_dict':experiment_dictionary['typeToSpeciesDict']}
        uncertainty_dict['ignition_delay_relative_uncertainty'] = experiment_dictionary['ignitionDelayRelativeUncertainty']
        uncertainty_dict['ignition_delay_absolute_uncertainty'] = experiment_dictionary['ignitionDelayAbsoluteUncertainty']
        uncertainty_dict['time_shift_absolute_uncertainty'] = experiment_dictionary['timeShiftUncertainty']

        
        return uncertainty_dict
    
    def build_uncertainty_jsr_dict(self,experiment_dictionary:dict={}):
        uncertainty_dict={}
        #Don't worry about absorbance for now
        uncertainty_dict['temperature_relative_uncertainty'] = experiment_dictionary['tempRelativeUncertainty']
        uncertainty_dict['pressure_relative_uncertainty'] = experiment_dictionary['pressureRelativeUncertainty']
        uncertainty_dict['species_relative_uncertainty'] = {'dictionary_of_values':experiment_dictionary['speciesUncertaintys'],
                        'species':experiment_dictionary['speciesNames']}
        uncertainty_dict['restime_relative_uncertainty'] = experiment_dictionary['residenceTimeRelativeUncertainty']
        uncertainty_dict['mole_fraction_relative_uncertainty'] = experiment_dictionary['moleFractionRelativeUncertainty']
        uncertainty_dict['mole_fraction_absolute_uncertainty'] = experiment_dictionary['moleFractionAbsoluteUncertainty'] 
        uncertainty_dict['concentration_relative_uncertainty'] = experiment_dictionary['concentrationRelativeUncertainty']
        uncertainty_dict['concentration_absolute_uncertainty'] = experiment_dictionary['concentrationAbsoluteUncertainty']        
        
        return uncertainty_dict
    
    def build_uncertainty_flow_reactor_dict(self,experiment_dictionary:dict={}):
        uncertainty_dict={}
        #Don't worry about absorbance for now
        uncertainty_dict['temperature_relative_uncertainty'] = experiment_dictionary['tempRelativeUncertainty']
        uncertainty_dict['pressure_relative_uncertainty'] = experiment_dictionary['pressureRelativeUncertainty']
        uncertainty_dict['species_relative_uncertainty'] = {'dictionary_of_values':experiment_dictionary['speciesUncertaintys'],
                        'species':experiment_dictionary['speciesNames']}
        uncertainty_dict['mole_fraction_relative_uncertainty'] = experiment_dictionary['moleFractionRelativeUncertainty']
        uncertainty_dict['mole_fraction_absolute_uncertainty'] = experiment_dictionary['moleFractionAbsoluteUncertainty'] 
        uncertainty_dict['concentration_relative_uncertainty'] = experiment_dictionary['concentrationRelativeUncertainity']
        uncertainty_dict['concentration_absolute_uncertainty'] = experiment_dictionary['concentrationAbsoluteUncertainty']
        uncertainty_dict['time_shift_uncertainty'] = experiment_dictionary['timeShiftUncertainty']
        return uncertainty_dict    
    
    def build_uncertainty_batch_reactor_dict(self,experiment_dictonarie:dict={}):
        uncertainty_dict = {}
        #need to make an exception to this if there is no absortpion in dict
        if 'coupledCoefficients' in experiment_dictonarie.keys():
            coupled_coefficients = experiment_dictonarie['coupledCoefficients']
            coupled_coefficients = [item for sublist in coupled_coefficients for item in sublist]
            uncertain_parameters_ones = experiment_dictonarie['uncertaintyParameterOnes']
            uncertain_parameter_twos = experiment_dictonarie['uncertaintyParameterTwos']
            zip_uncertain_paramters = list(zip(uncertain_parameters_ones,uncertain_parameter_twos))
            dict_of_coupled_unc_and_param = dict(zip(coupled_coefficients,zip_uncertain_paramters))
            
            
            uncertainty_dict['coupled_coef_and_uncertainty'] = dict_of_coupled_unc_and_param
            uncertainty_dict['absorbance_relative_uncertainty'] = experiment_dictonarie['absorbanceRelativeUncertainty']
            uncertainty_dict['absorbance_absolute_uncertainty'] = experiment_dictonarie['absorbanceAbsoluteUncertainty']
        #finish making this dictionary
        uncertainty_dict['temperature_relative_uncertainty'] = experiment_dictonarie['tempRelativeUncertainty']
        uncertainty_dict['pressure_relative_uncertainty'] = experiment_dictonarie['pressureRelativeUncertainty']
        uncertainty_dict['species_relative_uncertainty'] = {'dictionary_of_values':experiment_dictonarie['speciesUncertaintys'],
                        'species':experiment_dictonarie['speciesNames']}
        uncertainty_dict['time_shift_absolute_uncertainty'] = experiment_dictonarie['timeShiftUncertainty']
        uncertainty_dict['mole_fraction_relative_uncertainty'] = experiment_dictonarie['moleFractionRelativeUncertainty']
        uncertainty_dict['mole_fraction_absolute_uncertainty'] = experiment_dictonarie['moleFractionAbsoluteUncertainty'] 
        uncertainty_dict['concentration_relative_uncertainty'] = experiment_dictonarie['concentrationRelativeUncertainity']
        uncertainty_dict['concentration_absolute_uncertainty'] = experiment_dictonarie['concentrationAbsoluteUncertainty']
        
        return uncertainty_dict
    
    def running_full_flame_speed(self,processor=None,
                                 experiment_dictionary:dict={},
                                 kineticSens = 1,
                                 physicalSens =1,
                                 dk =0.01,
                                 exp_number = 1):
        
        # flame_speed=fl.flamespeed_multi_condition(pressures:float,
        #                                           temperatures:float,
        #                                           observables:list,
        #                                           kineticSens:int,
        #                                           physicalSens:int,
        #                                           conditions:dict,
        #                                           thermalBoundary='Adiabatic',
        #                                           processor:ctp.Processor=None,
        #                                           save_physSensHistories=0,
        #                                           moleFractionObservables:list=[],
        #                                           absorbanceObservables:list=[],
        #                                           concentrationObservables:list=[],
        #                                           fullParsedYamlFile:dict={},
        #                                           flame_width:float=1.0,
        #                                           save_timeHistories:int=0,
        #                                           T_profile=pd.DataFrame(columns=['z','T']),
        #                                           soret=True,
        #                                           tol_ss=[1.0e-5, 1.0e-13],
        #                                           tol_ts=[1.0e-4, 1.0e-10],
        #                                           loglevel=1,
        #                                           flametype='Flame Speed',
        #                                           cti_path="")
        experiment = 'not yet installed'
        return experiment
    def running_ignition_delay(self,processor=None,
                               experiment_dictionary:dict={},
                               kineticSens=1,
                               physicalSens=1,
                               dk=0.01,
                               exp_number=1):
        
        
        if 'volumeTraceCsvList' in experiment_dictionary.keys():
            
            ig_delay=ig.ignition_delay_wrapper(pressures=experiment_dictionary['pressures'],
                                               temperatures=experiment_dictionary['temperatures'],
                                               observables=experiment_dictionary['observables'],
                                               kineticSens=kineticSens,
                                               physicalSens=physicalSens,
                                               conditions=experiment_dictionary['conditions_to_run'],
                                               thermalBoundary=experiment_dictionary['thermalBoundary'],
                                               mechanicalBoundary=experiment_dictionary['mechanicalBoundary'],
                                               processor=processor,
                                               cti_path="", 
                                               save_physSensHistories=1,
                                               fullParsedYamlFile=experiment_dictionary, 
                                               save_timeHistories=1,
                                               log_file=False,
                                               log_name='log.txt',
                                               timeshift=experiment_dictionary['time_shift'],
                                               initialTime=experiment_dictionary['initialTime'],
                                               finalTime=experiment_dictionary['finalTime'],
                                               target=experiment_dictionary['target'],
                                               target_type=experiment_dictionary['target_type'],
                                               n_processors=2,
                                               volumeTraceList = experiment_dictionary['volumeTraceCsvList'])
        else:
            ig_delay=ig.ignition_delay_wrapper(pressures=experiment_dictionary['pressures'],
                                               temperatures=experiment_dictionary['temperatures'],
                                               observables=experiment_dictionary['observables'],
                                               kineticSens=kineticSens,
                                               physicalSens=physicalSens,
                                               conditions=experiment_dictionary['conditions_to_run'],
                                               thermalBoundary=experiment_dictionary['thermalBoundary'],
                                               mechanicalBoundary=experiment_dictionary['mechanicalBoundary'],
                                               processor=processor,
                                               cti_path="", 
                                               save_physSensHistories=1,
                                               fullParsedYamlFile=experiment_dictionary, 
                                               save_timeHistories=1,
                                               log_file=False,
                                               log_name='log.txt',
                                               timeshift=experiment_dictionary['time_shift'],
                                               initialTime=experiment_dictionary['initialTime'],
                                               finalTime=experiment_dictionary['finalTime'],
                                               target=experiment_dictionary['target'],
                                               target_type=experiment_dictionary['target_type'],
                                               n_processors=2)
        
        
        soln,ksen=ig_delay.run()
        if 'volumeTraceCsvList' in experiment_dictionary.keys(): 
            int_ksens_exp_mapped= ig_delay.map_and_interp_ksens()
            tsoln=ig_delay.sensitivity_adjustment(temp_del = dk)
            psoln=ig_delay.sensitivity_adjustment(pres_del = dk)
            diluent=[]
            if 'Diluent' in experiment_dictionary['typeToSpeciesDict'].keys() or 'diluent' in experiment_dictionary['typeToSpeciesDict'].keys():
                            diluent.append(experiment_dictionary['typeToSpeciesDict']['diluent'])
            diluent=[item for sublist in diluent for item in sublist]
            ssoln=ig_delay.species_adjustment(dk,diluents=diluent)
            deltatsoln,deltatausens=ig_delay.calculate_time_shift_sens(soln['delay'].values,dtau=1e-8)
            tsen=ig_delay.sensitivityCalculation(soln['delay'],tsoln['delay'])
            psen=ig_delay.sensitivityCalculation(soln['delay'],psoln['delay'])
            ssens=[]
            
            # for j in range(len(experiment_dictionary['conditions_to_run'])):
            for i in range(len(ssoln)):                  
                    ssens.append(ig_delay.sensitivityCalculation(soln['delay'],ssoln[i]['delay']))
            species_length=len(set(experiment_dictionary['speciesNames']).difference(diluent))
            list_of_ssens=[]
            chunksize=int(len(ssens)/species_length)
            #print(species_length,chunksize)
            for i in range(species_length):
                tempdata=[]
                tempdata=pd.DataFrame(columns=['delay'])

                tempdata['delay']=np.zeros(len(experiment_dictionary['conditions_to_run'])*len(experiment_dictionary['temperatures']))
                for k in range(chunksize):
                    #print(ssens[i+int(k*(chunksize))]['delay'])
                    #print('Second array')
                    #print(np.array(tempdata['delay']))
                    tempdata['delay']=np.array(ssens[i+int(k*(chunksize))]['delay'])+np.array(tempdata['delay'])
                    
                #print(tempdata)
                list_of_ssens.append(tempdata)
            ssens=list_of_ssens
                   
                    
            csv_paths = [x for x in  experiment_dictionary['ignitionDelayCsvFiles'] if x is not None]
            exp_data = ig_delay.importExperimentalData(csv_paths)
            
        elif len(experiment_dictionary['temperatures']) == len(experiment_dictionary['pressures']) and len(experiment_dictionary['temperatures'])>1 and len(experiment_dictionary['pressures']) > 1 and 'volumeTraceCsvList' not in experiment_dictionary.keys():
            
            
            int_ksens_exp_mapped= ig_delay.map_and_interp_ksens()
            tsoln=ig_delay.sensitivity_adjustment(temp_del = dk)
            psoln=ig_delay.sensitivity_adjustment(pres_del = dk)
            diluent=[]
            if 'Diluent' in experiment_dictionary['typeToSpeciesDict'].keys() or 'diluent' in experiment_dictionary['typeToSpeciesDict'].keys():
                            diluent.append(experiment_dictionary['typeToSpeciesDict']['diluent'])
            diluent=[item for sublist in diluent for item in sublist]
            ssoln=ig_delay.species_adjustment(dk,diluents=diluent)
            deltatsoln,deltatausens=ig_delay.calculate_time_shift_sens(soln['delay'].values,dtau=1e-8)
            tsen=ig_delay.sensitivityCalculation(soln['delay'],tsoln['delay'])
            psen=ig_delay.sensitivityCalculation(soln['delay'],psoln['delay'])
            ssens=[]
            
            # for j in range(len(experiment_dictionary['conditions_to_run'])):
            for i in range(len(ssoln)):                  
                    ssens.append(ig_delay.sensitivityCalculation(soln['delay'],ssoln[i]['delay']))
            species_length=len(set(experiment_dictionary['speciesNames']).difference(diluent))
            list_of_ssens=[]
            chunksize=int(len(ssens)/species_length)
            #print(species_length,chunksize)
            for i in range(species_length):
                tempdata=[]
                tempdata=pd.DataFrame(columns=['delay'])

                tempdata['delay']=np.zeros(len(experiment_dictionary['conditions_to_run'])*len(experiment_dictionary['temperatures']))
                for k in range(chunksize):
                    #print(ssens[i+int(k*(chunksize))]['delay'])
                    #print('Second array')
                    #print(np.array(tempdata['delay']))
                    tempdata['delay']=np.array(ssens[i+int(k*(chunksize))]['delay'])+np.array(tempdata['delay'])
                    
                #print(tempdata)
                list_of_ssens.append(tempdata)
            ssens=list_of_ssens
                   
                    
            csv_paths = [x for x in  experiment_dictionary['ignitionDelayCsvFiles'] if x is not None]
            exp_data = ig_delay.importExperimentalData(csv_paths)            
        else:
            int_ksens_exp_mapped= ig_delay.map_and_interp_ksens()
            tsoln=ig_delay.sensitivity_adjustment(temp_del = dk)
            psoln=ig_delay.sensitivity_adjustment(pres_del = dk)
            diluent=[]
            if 'Diluent' in experiment_dictionary['typeToSpeciesDict'].keys() or 'diluent' in experiment_dictionary['typeToSpeciesDict'].keys():
                            diluent.append(experiment_dictionary['typeToSpeciesDict']['diluent'])
            diluent=[item for sublist in diluent for item in sublist]
            ssoln=ig_delay.species_adjustment(dk,diluents=diluent)
            deltatsoln,deltatausens=ig_delay.calculate_time_shift_sens(soln['delay'].values,dtau=1e-8)
            tsen=ig_delay.sensitivityCalculation(soln['delay'],tsoln['delay'])
            psen=ig_delay.sensitivityCalculation(soln['delay'],psoln['delay'])
            ssens=[]
            
            # for j in range(len(experiment_dictionary['conditions_to_run'])):
            for i in range(len(ssoln)):                  
                    ssens.append(ig_delay.sensitivityCalculation(soln['delay'],ssoln[i]['delay']))
            species_length=len(set(experiment_dictionary['speciesNames']).difference(diluent))
            list_of_ssens=[]
            chunksize=int(len(ssens)/species_length)
            #print(species_length,chunksize)
            for i in range(species_length):
                tempdata=[]
                tempdata=pd.DataFrame(columns=['delay'])
                tempdata['delay']=np.zeros(len(experiment_dictionary['conditions_to_run'])*len(experiment_dictionary['temperatures'])*len(experiment_dictionary['pressures']))
                for k in range(chunksize):
                    #print(ssens[i+int(k*(chunksize))]['delay'])
                    #print('Second array')
                    #print(np.array(tempdata['delay']))
                    tempdata['delay']=np.array(ssens[i+int(k*(chunksize))]['delay'])+np.array(tempdata['delay'])
                    
                #print(tempdata)
                list_of_ssens.append(tempdata)
            ssens=list_of_ssens
                   
                    
            csv_paths = [x for x in  experiment_dictionary['ignitionDelayCsvFiles'] if x is not None]
            exp_data = ig_delay.importExperimentalData(csv_paths)            
            
            
        experiment = self.build_single_exp_dict(exp_number,
                                           ig_delay,
                                           int_ksens_exp_mapped,
                                           [tsen,psen],
                                           ssens,
                                           experimental_data = exp_data,
                                           yaml_dict=experiment_dictionary,
                                           interpolated_time_shift_sens=deltatausens)
        return experiment
        
        
    def running_flow_reactor(self,processor=None,
                             experiment_dictionary:dict={},
                             kineticSens = 1,
                             physicalSens = 1,
                             dk = 0.01,
                             exp_number = 1):
        flow_reactor = fr.flow_reactor_wrapper(pressure = experiment_dictionary['pressure'],
                                                temperatures = experiment_dictionary['temperatures'],
                                                observables = experiment_dictionary['observables'],
                                                moleFractionObservables = experiment_dictionary['moleFractionObservables'],
                                                concentrationObservables = experiment_dictionary['concentrationObservables'],
                                                fullParsedYamlFile = experiment_dictionary,
                                                kineticSens=kineticSens,
                                                physicalSens=physicalSens,
                                                conditions=experiment_dictionary['conditions'],
                                                thermalBoundary=experiment_dictionary['thermalBoundary'],
                                                mechanicalBoundary=experiment_dictionary['mechanicalBoundary'],
                                                processor=processor,
                                                cti_path="", 
                                                save_physSensHistories=1,
                                                save_timeHistories=1,
                                                timeshifts=experiment_dictionary['timeShift'],
                                                initialTime=experiment_dictionary['initialTime'],
                                                residenceTimes=experiment_dictionary['residenceTimes'])
        soln,ksen=flow_reactor.run(ksens_marker=kineticSens ,psens_marker=physicalSens)

        int_ksens_exp_mapped= flow_reactor.map_and_interp_ksens()
        tsoln=flow_reactor.sensitivity_adjustment(temp_del = dk)
        psoln=flow_reactor.sensitivity_adjustment(pres_del = dk)
        ssoln=flow_reactor.species_adjustment(dk)
        tsen=flow_reactor.sensitivityCalculation(soln[flow_reactor.observables],
                                                 tsoln[flow_reactor.observables],
                                                 dk=dk)
                                                 
        
        psen=flow_reactor.sensitivityCalculation(soln[flow_reactor.observables],
                                                 psoln[flow_reactor.observables],dk=dk)
        
        
        ssens=[]
        for i in range(len(ssoln)):            
            ssens.append(flow_reactor.sensitivityCalculation(soln[flow_reactor.observables],
                                                             ssoln[i][flow_reactor.observables],
                                                             dk=dk))
        time_shift_sens =[]
        for i,timehist in enumerate(flow_reactor.fullTimeHistories):
            time_shift_sens.append(flow_reactor.calculate_time_shift_sensitivity(flow_reactor,timehist,1e-8,flow_reactor.finalTimes[i]))
            
        time_shift_sens_df = pd.concat(time_shift_sens,ignore_index=True)    
        #print(time_shift_sens_df)
            
        csv_paths = [x for x in  experiment_dictionary['moleFractionCsvFiles'] + experiment_dictionary['concentrationCsvFiles'] if x is not None]
        #print(csv_paths)
        exp_data = flow_reactor.importExperimentalData(csv_paths)
        
        experiment = self.build_single_exp_dict(exp_number,
                                            flow_reactor,
                                            int_ksens_exp_mapped,
                                            [tsen,psen],
                                            ssens,
                                            interpolated_time_shift_sens = time_shift_sens_df,
                                            experimental_data = exp_data,
                                            yaml_dict=experiment_dictionary)                                                
                                               
                                        
            
    
        return experiment
    
    
    
            
    def running_full_jsr(self,processor=None,
                             experiment_dictionary:dict={},
                             kineticSens = 1,
                             physicalSens = 1,
                             dk = 0.01,
                             exp_number = 1):
        
        jet_stirred_reactor = jsr.JSR_multiTemp_steadystate(volume=experiment_dictionary['volume'],
                    pressure=experiment_dictionary['pressure'],
                    temperatures=experiment_dictionary['temperatures'],
                    observables=experiment_dictionary['observables'],
                    kineticSens=kineticSens,
                    physicalSens=physicalSens,
                    conditions=experiment_dictionary['conditions'],
                    thermalBoundary=experiment_dictionary['thermalBoundary'],
                    mechanicalBoundary=experiment_dictionary['mechanicalBoundary'],
                    processor=processor,
                    save_physSensHistories=1,
                    save_timeHistories=1,
                    residence_time=experiment_dictionary['residence_time'],
                    moleFractionObservables = experiment_dictionary['moleFractionObservables'],
                    concentrationObservables = experiment_dictionary['concentrationObservables'],
                    fullParsedYamlFile = experiment_dictionary)
        
        soln,ksen=jet_stirred_reactor.run()
        #print(experiment_dictionary)
        for i,species in enumerate(experiment_dictionary['observables']):
            #print(os.path.split(experiment_dictionary['moleFractionCsvFiles']))
            tempsoln=pd.DataFrame(columns=('Temperature',species))
            tempsoln['Temperature']=soln['temperature']
            tempsoln[species]=soln[species]
            #print(self.yaml_list[exp_number])
            tempsoln.to_csv(os.path.join(os.path.splitext(self.yaml_list[exp_number][0])[0]+'_'+species+'.csv'),index=False)
    
        int_ksens_exp_mapped= jet_stirred_reactor.map_and_interp_ksens()
        tsoln=jet_stirred_reactor.sensitivity_adjustment(temp_del = dk)
        psoln=jet_stirred_reactor.sensitivity_adjustment(pres_del = dk)
        ssoln=jet_stirred_reactor.species_adjustment(dk)
        rsoln=jet_stirred_reactor.sensitivity_adjustment(res_del = dk)

        
        tsen=jet_stirred_reactor.sensitivityCalculation(soln[jet_stirred_reactor.observables],tsoln[jet_stirred_reactor.observables],jet_stirred_reactor.observables)
        psen=jet_stirred_reactor.sensitivityCalculation(soln[jet_stirred_reactor.observables],psoln[jet_stirred_reactor.observables],jet_stirred_reactor.observables)
        ssens=[]
        for i in range(len(ssoln)):            
            ssens.append(jet_stirred_reactor.sensitivityCalculation(soln[jet_stirred_reactor.observables],ssoln[i][jet_stirred_reactor.observables],jet_stirred_reactor.observables))
        rsens=jet_stirred_reactor.sensitivityCalculation(soln[jet_stirred_reactor.observables],rsoln[jet_stirred_reactor.observables],jet_stirred_reactor.observables)
        #print(ssens)
        #print(jet_stirred_reactor.physicalSens)
        csv_paths = [x for x in  (experiment_dictionary['moleFractionCsvFiles']+ experiment_dictionary['concentrationCsvFiles']) if x is not None]
        #print(csv_paths)
        exp_data = jet_stirred_reactor.importExperimentalData(csv_paths)
        
        experiment = self.build_single_exp_dict(exp_number,
                                           jet_stirred_reactor,
                                           int_ksens_exp_mapped,
                                           [tsen,psen,rsens],
                                           ssens,
                                           experimental_data = exp_data,
                                           yaml_dict=experiment_dictionary)
        return experiment
        
    def running_full_batch_reactor(self,processor=None,
                                           experiment_dictionary:dict={},
                                           kineticSens = 1,
                                           physicalSens = 1,
                                           dk = .01,
                                           exp_number=1):
        batch_reactor = st.batchReactor(pressure = experiment_dictionary['pressure'],
                     temperature = experiment_dictionary['temperature'],
                     observables = experiment_dictionary['observables'],
                     kineticSens = kineticSens,
                     physicalSens = physicalSens,
                     conditions = experiment_dictionary['conditions'],
                     initialTime = experiment_dictionary['initialTime'],
                     finalTime = experiment_dictionary['finalTime'],
                     thermalBoundary = experiment_dictionary['thermalBoundary'],
                     mechanicalBoundary = experiment_dictionary['mechanicalBoundary'],
                     processor = processor,
                     save_timeHistories = 1,
                     save_physSensHistories = 1,
                     moleFractionObservables = experiment_dictionary['moleFractionObservables'],
                     concentrationObservables = experiment_dictionary['concentrationObservables'],
                     fullParsedYamlFile = experiment_dictionary,
                     time_shift_value = experiment_dictionary['timeShift'])
        
        csv_paths = [x for x in  experiment_dictionary['moleFractionCsvFiles'] + experiment_dictionary['concentrationCsvFiles'] if x is not None]
        exp_data = batch_reactor.importExperimentalData(csv_paths)
        
        batch_reactor.run()
        ########################################################################

        
        ################################################################################
        int_ksens_exp_mapped= batch_reactor.map_and_interp_ksens()#ksens is wiped on rerun so int it before
        batch_reactor.sensitivity_adjustment(temp_del = dk)
        batch_reactor.sensitivity_adjustment(pres_del = dk)
        batch_reactor.species_adjustment(dk)
        ############################################### check to make sure these aren't effected 
        int_tp_psen_against_experimental = batch_reactor.interpolate_experimental([batch_reactor.interpolate_physical_sensitivities(index=1),
                                                                           batch_reactor.interpolate_physical_sensitivities(index=2)])
        
    
        int_spec_psen_against_experimental = batch_reactor.interpolate_experimental(pre_interpolated=batch_reactor.interpolate_species_sensitivities())
    ###############saving the shock tube experimental interpolated time history     
        single_data = batch_reactor.interpolate_experimental(single=batch_reactor.timeHistories[0])
        batch_reactor.savingInterpTimeHistoryAgainstExp(single_data)
        #tab starting here tomorrow
        batch_reactor.interpolatePressureandTempToExperiment(batch_reactor,exp_data)
        time_shift_sensitivity = batch_reactor.calculate_time_shift_sensitivity(batch_reactor,exp_data,dk)
        
    ###############  ###############  
        experiment = self.build_single_exp_dict(exp_number,
                                           batch_reactor,
                                           int_ksens_exp_mapped,
                                           int_tp_psen_against_experimental,
                                           int_spec_psen_against_experimental,
                                           experimental_data = exp_data,
                                           yaml_dict=experiment_dictionary,
                                           interpolated_time_shift_sens = time_shift_sensitivity)
        
        #write test case and check if we can get as far as just returnign the experiment
        return experiment
    
    def running_full_batch_reactor_absorption(self,processor=None,
                                           experiment_dictionary:dict={},
                                           absorbance_yaml_file_path = '',
                                           kineticSens = 1,
                                           physicalSens = 1,
                                           dk = .01,
                                           exp_number=1):
        batch_reactor = st.batchReactor(pressure = experiment_dictionary['pressure'],
                     temperature = experiment_dictionary['temperature'],
                     observables = experiment_dictionary['observables'],
                     kineticSens = kineticSens,
                     physicalSens = physicalSens,
                     conditions = experiment_dictionary['conditions'],
                     initialTime = experiment_dictionary['initialTime'],
                     finalTime = experiment_dictionary['finalTime'],
                     thermalBoundary = experiment_dictionary['thermalBoundary'],
                     mechanicalBoundary = experiment_dictionary['mechanicalBoundary'],
                     processor = processor,
                     save_timeHistories = 1,
                     save_physSensHistories = 1,
                     moleFractionObservables = experiment_dictionary['moleFractionObservables'],
                     absorbanceObservables = experiment_dictionary['absorbanceObservables'],
                     concentrationObservables = experiment_dictionary['concentrationObservables'],
                     fullParsedYamlFile = experiment_dictionary,
                     time_shift_value = experiment_dictionary['timeShift'])
    
        
        csv_paths = [x for x in  experiment_dictionary['moleFractionCsvFiles'] + experiment_dictionary['concentrationCsvFiles'] if x is not None]
        
        exp_data = batch_reactor.importExperimentalData(csv_paths)
        batch_reactor.run()
        #this might be in the wrong spot 
        int_ksens_exp_mapped= batch_reactor.map_and_interp_ksens()
    
    
        abs_instance = csp.Absorb()
        parser = yp.Parser()
        abs_loaded = parser.load_to_obj(absorbance_yaml_file_path)
        abs_data = abs_instance.superimpose_batch_reactor(batch_reactor,abs_loaded,experiment_dictionary['pathLength'],
                                                       kinetic_sens=kineticSens)    
        
        
        perturbed_coef = abs_instance.perturb_abs_coef(dk,
                                              batch_reactor,
                                              abs_loaded,
                                              experiment_dictionary['pathLength'],
                                              summed_data = abs_data[0]) 
        
        
        
        batch_reactor.sensitivity_adjustment(temp_del = dk)
        batch_reactor.sensitivity_adjustment(pres_del = dk)
        batch_reactor.species_adjustment(dk)
        int_tp_psen_against_experimental = batch_reactor.interpolate_experimental([batch_reactor.interpolate_physical_sensitivities(index=1),
                                                                                 batch_reactor.interpolate_physical_sensitivities(index=2)])

        
        int_spec_psen_against_experimental = batch_reactor.interpolate_experimental(pre_interpolated=batch_reactor.interpolate_species_sensitivities())
        abs_phys_sens = abs_instance.absorb_phys_sensitivities(batch_reactor,abs_data[0],abs_loaded,
                                                               experiment_dictionary['pathLength'],
                                                               dk = dk)
       


        loaded_experimental_data_absorbance = abs_instance.import_experimental_data(experiment_dictionary['absorbanceCsvFiles'])
        
        interp_abs_exp= abs_instance.interpolate_experimental(batch_reactor,loaded_experimental_data_absorbance,
                                                            original_summed_absorption=abs_data[0],
                                                            abs_kinetic_sens = abs_data[1],
                                                            abs_phys_sens = abs_phys_sens,
                                                            abs_coef_sens = perturbed_coef)

        time_history_interp_against_experiment_dict = abs_instance.interpolate_experimental(batch_reactor,
                                                                                            loaded_experimental_data_absorbance,
                                                                                            time_history = batch_reactor.timeHistories[0])
     #################################################################################   
        single_data = batch_reactor.interpolate_experimental(single=batch_reactor.timeHistories[0])
        batch_reactor.savingInterpTimeHistoryAgainstExp(single_data)
        batch_reactor.interpolatePressureandTempToExperiment(batch_reactor,exp_data)
        time_shift_sensitivity = batch_reactor.calculate_time_shift_sensitivity(batch_reactor,exp_data,dk)
    ####################################################################################    
        time_shift_abs_senstivity = abs_instance.calculate_time_shift_sensitivity_abs(abs_data[0],
                                                                                       loaded_experimental_data_absorbance,
                                                                                       batch_reactor,dk)
        experiment = self.build_single_exp_dict(exp_number,
                                                batch_reactor,
                                      int_ksens_exp_mapped,
                                      int_tp_psen_against_experimental,
                                      int_spec_psen_against_experimental,
                                      interpolated_absorbance=interp_abs_exp,
                                      experimental_data = exp_data,
                                      absorbance_experimental_data = loaded_experimental_data_absorbance,
                                      time_history_interpolated_against_absorbance_experiment = time_history_interp_against_experiment_dict,
                                      absorbance_calculated_from_model = abs_data[0],
                                      yaml_dict=experiment_dictionary,
                                      interpolated_time_shift_sens = time_shift_sensitivity,
                                      interpolated_abs_time_shift = time_shift_abs_senstivity)
        
        return experiment
    
    
    
    def running_batch_reactor_absorption_only(self,processor=None,
                                           experiment_dictionary:dict={},
                                           absorbance_yaml_file_path = '',
                                           kineticSens = 1,
                                           physicalSens = 1,
                                           dk = .01,
                                           exp_number=1):
        batch_reactor = st.batchReactor(pressure = experiment_dictionary['pressure'],
                     temperature = experiment_dictionary['temperature'],
                     observables = experiment_dictionary['observables'],
                     kineticSens = kineticSens,
                     physicalSens = physicalSens,
                     conditions = experiment_dictionary['conditions'],
                     initialTime = experiment_dictionary['initialTime'],
                     finalTime = experiment_dictionary['finalTime'],
                     thermalBoundary = experiment_dictionary['thermalBoundary'],
                     mechanicalBoundary = experiment_dictionary['mechanicalBoundary'],
                     processor = processor,
                     save_timeHistories = 1,
                     save_physSensHistories = 1,
                     moleFractionObservables = experiment_dictionary['moleFractionObservables'],
                     absorbanceObservables = experiment_dictionary['absorbanceObservables'],
                     concentrationObservables = experiment_dictionary['concentrationObservables'],
                     fullParsedYamlFile = experiment_dictionary,
                     time_shift_value = experiment_dictionary['timeShift'])

    
        batch_reactor.run()
        abs_instance = csp.Absorb()
        parser = yp.Parser()
        abs_loaded = parser.load_to_obj(absorbance_yaml_file_path)
        abs_data = abs_instance.superimpose_batch_reactor(batch_reactor,abs_loaded,experiment_dictionary['pathLength'],
                                                       kinetic_sens=kineticSens)
        
        #print('first go')
        
        
        perturbed_coef = abs_instance.perturb_abs_coef(dk,
                                              batch_reactor,
                                              abs_loaded,
                                              experiment_dictionary['pathLength'],
                                              summed_data = abs_data[0]) 
        
        #print('second go')
       
        #print(perturbed_coef)
        
        
        batch_reactor.sensitivity_adjustment(temp_del = dk)
        batch_reactor.sensitivity_adjustment(pres_del = dk)
        batch_reactor.species_adjustment(dk)        
        abs_phys_sens = abs_instance.absorb_phys_sensitivities(batch_reactor,abs_data[0],abs_loaded,
                                                               experiment_dictionary['pathLength'],
                                                               dk = dk)
       

        
        loaded_experimental_data_absorbance = abs_instance.import_experimental_data(experiment_dictionary['absorbanceCsvFiles'])
        
        interp_abs_exp= abs_instance.interpolate_experimental(batch_reactor,loaded_experimental_data_absorbance,
                                                            original_summed_absorption=abs_data[0],
                                                            abs_kinetic_sens = abs_data[1],
                                                            abs_phys_sens = abs_phys_sens,
                                                            abs_coef_sens = perturbed_coef)
        
        
        time_history_interp_against_experiment_dict = abs_instance.interpolate_experimental(batch_reactor,
                                                                                            loaded_experimental_data_absorbance,
                                                                                            time_history = batch_reactor.timeHistories[0])
        time_shift_abs_senstivity = abs_instance.calculate_time_shift_sensitivity_abs(abs_data[0],
                                                                                       loaded_experimental_data_absorbance,
                                                                                       batch_reactor,dk)
        
        
                                
        experiment = self.build_single_exp_dict(exp_number,
                                                batch_reactor,
                                      None,
                                      None,
                                      None,
                                      interpolated_absorbance=interp_abs_exp,
                                      absorbance_experimental_data = loaded_experimental_data_absorbance,
                                      time_history_interpolated_against_absorbance_experiment = time_history_interp_against_experiment_dict,
                                      absorbance_calculated_from_model = abs_data[0],
                                      yaml_dict=experiment_dictionary,
                                      interpolated_abs_time_shift = time_shift_abs_senstivity)
        
        return experiment    
    
    
    def running_full_variable_pressure_batch_reactor(self,processor=None,
                                           experiment_dictionary:dict={},
                                           kineticSens = 1,
                                           physicalSens = 1,
                                           dk = .01,
                                           exp_number=1):
        batch_reactor = st.batchReactor(pressure = experiment_dictionary['pressure'],
                     temperature = experiment_dictionary['temperature'],
                     observables = experiment_dictionary['observables'],
                     kineticSens = kineticSens,
                     physicalSens = physicalSens,
                     conditions = experiment_dictionary['conditions'],
                     initialTime = experiment_dictionary['initialTime'],
                     finalTime = experiment_dictionary['finalTime'],
                     thermalBoundary = experiment_dictionary['thermalBoundary'],
                     mechanicalBoundary = experiment_dictionary['mechanicalBoundary'],
                     processor = processor,
                     save_timeHistories = 1,
                     save_physSensHistories = 1,
                     moleFractionObservables = experiment_dictionary['moleFractionObservables'],
                     concentrationObservables = experiment_dictionary['concentrationObservables'],
                     fullParsedYamlFile = experiment_dictionary,
                     time_shift_value = experiment_dictionary['timeShift'],
                     volumeTrace=experiment_dictionary['volumeTraceCsv'],
                     exactDerivFlag=False)
        
        csv_paths = [x for x in  experiment_dictionary['moleFractionCsvFiles'] + experiment_dictionary['concentrationCsvFiles'] if x is not None]
        exp_data = batch_reactor.importExperimentalData(csv_paths)
        
        batch_reactor.run()
        ########################################################################

        
        ################################################################################
        int_ksens_exp_mapped= batch_reactor.map_and_interp_ksens()#ksens is wiped on rerun so int it before
        batch_reactor.sensitivity_adjustment(temp_del = dk)
        batch_reactor.sensitivity_adjustment(pres_del = dk)
        batch_reactor.species_adjustment(dk)
        ############################################### check to make sure these aren't effected 
        int_tp_psen_against_experimental = batch_reactor.interpolate_experimental([batch_reactor.interpolate_physical_sensitivities(index=1),
                                                                           batch_reactor.interpolate_physical_sensitivities(index=2)])
        
    
        int_spec_psen_against_experimental = batch_reactor.interpolate_experimental(pre_interpolated=batch_reactor.interpolate_species_sensitivities())
    ###############saving the shock tube experimental interpolated time history     
        single_data = batch_reactor.interpolate_experimental(single=batch_reactor.timeHistories[0])
        batch_reactor.savingInterpTimeHistoryAgainstExp(single_data)
        #tab starting here tomorrow
        batch_reactor.interpolatePressureandTempToExperiment(batch_reactor,exp_data)
        time_shift_sensitivity = batch_reactor.calculate_time_shift_sensitivity(batch_reactor,exp_data,dk)
        
    ###############  ###############  
        experiment = self.build_single_exp_dict(exp_number,
                                           batch_reactor,
                                           int_ksens_exp_mapped,
                                           int_tp_psen_against_experimental,
                                           int_spec_psen_against_experimental,
                                           experimental_data = exp_data,
                                           yaml_dict=experiment_dictionary,
                                           interpolated_time_shift_sens = time_shift_sensitivity)
        
        #write test case and check if we can get as far as just returnign the experiment
        return experiment    
    
    
    def looping_over_parsed_yaml_files(self,list_of_parsed_yamls,list_of_yaml_paths,processor=None,kineticSens=1,physicalSens=1,dk=.01):
       
        experiment_list = []
        for i,yamlDict in enumerate(list_of_parsed_yamls):
         
            simulation_type = yamlDict['simulationType']
            experiment_type = yamlDict['experimentType']
            
            if re.match('[Bb]atch[ _-][Rr]eactor',simulation_type) and re.match('[Ss]pecies[- ][Pp]rofile',experiment_type):
                

                
                    if 'absorbanceObservables' not in yamlDict.keys():
                        experiment = self.running_full_batch_reactor(processor=processor,
                                           experiment_dictionary=yamlDict,
                                           kineticSens = kineticSens,
                                           physicalSens = physicalSens,
                                           dk = dk,
                                           exp_number=i)
                        experiment_list.append(experiment)
                        
                    elif 'absorbanceObservables' in yamlDict.keys() and yamlDict['moleFractionObservables'][0] == None and yamlDict['concentrationObservables'][0]==None:
                        path = list_of_yaml_paths[i][1]
                        
                        experiment = self.running_batch_reactor_absorption_only(processor=processor,
                                                                             experiment_dictionary = yamlDict,
                                                                             absorbance_yaml_file_path = path,
                                                                             kineticSens = kineticSens,
                                                                             physicalSens = physicalSens,
                                                                             dk = dk,
                                                                             exp_number=i)
                        experiment_list.append(experiment)
                    
                    else:
                        path = list_of_yaml_paths[i][1]
                        experiment = self.running_full_batch_reactor_absorption(processor=processor,
                                           experiment_dictionary=yamlDict,
                                           absorbance_yaml_file_path = path,
                                           kineticSens = kineticSens,
                                           physicalSens = physicalSens,
                                           dk = dk,
                                           exp_number=i)
                        experiment_list.append(experiment)
                        
                        
            elif re.match('[Bb]atch[ _-][Rr]eactor',simulation_type) and re.match('[Ii]gnition[- ][Dd]elay',experiment_type):
                 if 'absorbanceObservables' not in yamlDict.keys():
                        experiment = self.running_ignition_delay(processor=processor,
                                           experiment_dictionary=yamlDict,
                                           kineticSens = kineticSens,
                                           physicalSens = physicalSens,
                                           dk = dk,
                                           exp_number=i)
                        experiment_list.append(experiment)
                
                 elif 'absorbanceObservables' in yamlDict.keys() and yamlDict['moleFractionObservables'][0] == None and yamlDict['concentrationObservables'][0]==None:
#                        path = list_of_yaml_paths[i][1]
#                        print(path)
#                        experiment = self.running_batch_reactor_absorption_only(processor=processor,
#                                                                             experiment_dictionary = yamlDict,
#                                                                             absorbance_yaml_file_path = path,
#                                                                             kineticSens = kineticSens,
#                                                                             physicalSens = physicalSens,
#                                                                             dk = dk,
#                                                                             exp_number=i)
#                        experiment_list.append(experiment)
                        print('Absorbance currently not enabled for ignition delay')
                 else:
#                        path = list_of_yaml_paths[i][1]
#                        experiment = self.running_full_batch_reactor_absorption(processor=processor,
#                                           experiment_dictionary=yamlDict,
#                                           absorbance_yaml_file_path = path,
#                                           kineticSens = kineticSens,
#                                           physicalSens = physicalSens,
#                                           dk = dk,
#                                           exp_number=i)
#                        experiment_list.append(experiment)
                        print('Absorbance currently not enabled for ignition delay')


            elif re.match('[Rr][Cc][Mm]',simulation_type) and re.match('[Ii]gnition[- ][Dd]elay',experiment_type):
                 
                 if 'absorbanceObservables' not in yamlDict.keys():
                        experiment = self.running_ignition_delay(processor=processor,
                                           experiment_dictionary=yamlDict,
                                           kineticSens = kineticSens,
                                           physicalSens = physicalSens,
                                           dk = dk,
                                           exp_number=i)
                        experiment_list.append(experiment)
                
                 elif 'absorbanceObservables' in yamlDict.keys() and yamlDict['moleFractionObservables'][0] == None and yamlDict['concentrationObservables'][0]==None:
#                        path = list_of_yaml_paths[i][1]
#                        print(path)
#                        experiment = self.running_batch_reactor_absorption_only(processor=processor,
#                                                                             experiment_dictionary = yamlDict,
#                                                                             absorbance_yaml_file_path = path,
#                                                                             kineticSens = kineticSens,
#                                                                             physicalSens = physicalSens,
#                                                                             dk = dk,
#                                                                             exp_number=i)
#                        experiment_list.append(experiment)
                        print('Absorbance currently not enabled for ignition delay')
                 else:
#                        path = list_of_yaml_paths[i][1]
#                        experiment = self.running_full_batch_reactor_absorption(processor=processor,
#                                           experiment_dictionary=yamlDict,
#                                           absorbance_yaml_file_path = path,
#                                           kineticSens = kineticSens,
#                                           physicalSens = physicalSens,
#                                           dk = dk,
#                                           exp_number=i)
#                        experiment_list.append(experiment)
                        print('Absorbance currently not enabled for ignition delay')
                        
                
            elif re.match('[Jj][Ss][Rr]',simulation_type) or re.match('[Jj]et[- ][Ss]tirred[- ][Rr]eactor',simulation_type):
                
                
                    if 'absorbanceObservables' not in yamlDict.keys():
                        
                        experiment = self.running_full_jsr(processor=processor,
                                           experiment_dictionary=yamlDict,
                                           kineticSens = kineticSens,
                                           physicalSens = physicalSens,
                                           dk = dk,
                                           exp_number=i)
                        experiment_list.append(experiment)
                        ####FINISH writing this function and start writing main function tomorrow 
                    elif 'absorbanceObservables' in yamlDict.keys() and yamlDict['moleFractionObservables'][0] == None and yamlDict['concentrationObservables'][0]==None:
#                        path = list_of_yaml_paths[i][1]
#                        print(path)
#                        experiment = self.running_batch_reactor_absorption_only(processor=processor,
#                                                                             experiment_dictionary = yamlDict,
#                                                                             absorbance_yaml_file_path = path,
#                                                                             kineticSens = kineticSens,
#                                                                             physicalSens = physicalSens,
#                                                                             dk = dk,
#                                                                             exp_number=i)
#                        experiment_list.append(experiment)
                        print('Absorbance currently not enabled for jsr')
                    else:
#                        path = list_of_yaml_paths[i][1]
#                        experiment = self.running_full_batch_reactor_absorption(processor=processor,
#                                           experiment_dictionary=yamlDict,
#                                           absorbance_yaml_file_path = path,
#                                           kineticSens = kineticSens,
#                                           physicalSens = physicalSens,
#                                           dk = dk,
#                                           exp_number=i)
#                        experiment_list.append(experiment)
                        print('Absorbance currently not enabled for jsr')
            elif re.match('[Ff]lame[ -][Ss]peed',simulation_type) and re.match('[Oo][Nn][Ee]|[1][ -][dD][ -][Ff]lame',experiment_type):
                print("ADD FLAME SPEED DICT HERE")           
            
            
            
            elif re.match('[Ff]low[ -][Rr]eactor',simulation_type) and re.match('[Ss]pecies[- ][Pp]rofile',experiment_type):
                
                
                    if 'absorbanceObservables' not in yamlDict.keys():
                        experiment= self.running_flow_reactor(processor=processor,
                                           experiment_dictionary=yamlDict,
                                           kineticSens = kineticSens,
                                           physicalSens = physicalSens,
                                           dk = dk,
                                           exp_number=i)
                        experiment_list.append(experiment)
                    elif 'absorbanceObservables' in yamlDict.keys() and yamlDict['moleFractionObservables'][0] == None and yamlDict['concentrationObservables'][0]==None:

                        print('Absorbance currently not enabled for flow reactor')
                    else:

                        print('Absorbance currently not enabled for jsr')                
            else:
                print('We do not have this simulation installed yet')
            
        return experiment_list
    def saving_experimental_dict(self,list_off_experiment_dictonaires):
        uncertainty_list = []
        for i,exp in enumerate(list_off_experiment_dictonaires):
            if 'perturbed_coef' not in exp.keys():
                uncertainty_list.append({})
            else:
                uncertainty_list.append(exp['uncertainty'])
                            
        return uncertainty_list
    


   
    

    
    
