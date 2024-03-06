import yaml 
import shutil
import numpy as np
import copy
import re

# subpackage for reading yaml files that describe simulations and absorbance data
class Parser(object):
    """Class to read and write yaml input files for MSI for a variety of experiment types."""
    def __init__(self,original_experimental_conditions=None):
        """Internal varibale 'Parser.original_experimenntal_conditions' stores original parsed yaml file from experiment."""
        self.original_experimental_conditions =  original_experimental_conditions
       

    #config is a dict containing the yaml information
    def load_to_obj(self, path:str = ''):
        """
        Takes in a file path for a yaml file and returns a dictionary of 
        simulation information.

        Parameters
        ----------
        path : str, optional
            The path to where the yaml file is stored. The default is ''.

        Returns
        -------
        config : dictionary
            An unorganized dictionary that contains information reguarding
            the experiment the yaml input file was written for.

        """
        with open(path) as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        return config
    
    def get_sim_type(self,loaded_exp:dict={}):
        """
        Takes in an unorganized dictonary for a yaml file and returns the 
        simulation type as well as the experiment type contained in the file. This function
        is used as a helper function to determine which of the parser helper
        functions the dictonary will be passed to. Since the structure of yaml
        files and dictionaries varry based on experiment type.  

        Parameters
        ----------
        loaded_exp : dict, optional
            An unorganized dictionary that contains information reguarding
            the experiment the yaml input file was written for. The default is {}.

        Returns
        -------
        simtype : str
            The simulation type contained in the unorganized yaml file 
            dictionary.
        experiment_type : str
            The experiment type contained in the unorganized yaml file 
            dictionary.

        """
        #parse into unorganized yaml dict (regardless of simulation)
        #to determine the experiment and simultion type
        simtype = loaded_exp['apparatus']['kind']
        experiment_type = loaded_exp['experiment-type']
        return simtype,experiment_type
    
    def parse_flame_speed_obj(self,loaded_exp:dict={}, loaded_absorption:dict={}):
        """
        Takes in an unorganized dictonary for a yaml file containing 
        experimental information relating to flame speeds and returns an
        organized dictonary with the necessary information to run an MSI
        flame speed optimization.

        Parameters
        ----------
        loaded_exp : dict, optional
           Unorganized dictonary for a yaml file containing 
           experimental information relating to flame speeds. The default is 
           {}.
        loaded_absorption : dict, optional
            Unorganized dictonary for a yaml file containing experimental 
            information relating to absorption in a flame speed simulation. 
            The default is {}.

        Returns
        -------
        dict
            Organized dictonary with the necessary information to run an MSI
            flame speed optimization.

        """
        #begin defining importnat variables and parsing into unorganized
        #yaml file to return information to run MSI simulations for
        #flame speeds
        simulation_type = loaded_exp['apparatus']['kind']
        experiment_type = loaded_exp['experiment-type']
        flame_width = loaded_exp['apparatus']['flame_width']['value']
        flame_width_relative_uncertainty = loaded_exp['apparatus']['flame_width']['relative-uncertainty']
        pressure = loaded_exp['common-properties']['pressure']['value-list']
        inlet_temperature = loaded_exp['common-properties']['inlet-temperature']['value-list']
        inlet_temperature_relative_uncertainty = loaded_exp['common-properties']['inlet-temperature']['relative-uncertainty']
        species_group_numbers = [(species['species-group']) for species in loaded_exp['common-properties']['composition']]
        flame_speed_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['flame-speed']]
        flame_speed_relative_uncertainty = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['flame-speed']]    
        flame_speed_observables = [point['targets'][0]['name'] for point in loaded_exp['datapoints']['flame-speed']]
        pressure_relative_uncertainty = loaded_exp['common-properties']['pressure']['relative-uncertainty']
        pressure_relative_uncertainty = float(pressure_relative_uncertainty)
        thermal_boundary = loaded_exp['common-properties']['assumptions']['thermal-boundary']
        mechanical_boundary = loaded_exp['common-properties']['assumptions']['mechanical-boundary']
        flame_speed_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['flame-speed']]

        
        species_groups = [(species['mixture']) for species in loaded_exp['common-properties']['composition']]
        attribute_group = [(attribute['attributes']) for attribute in loaded_exp['common-properties']['composition']]
        species_in_group_list = [[] for groups in species_groups]
        type_in_group_list = [[] for groups in species_groups]
        relative_uncertainty_in_group_list = [[] for groups in species_groups]
        species = []
        mole_fractions = []
    
    
        for i,group in enumerate(species_groups):
            for j, dictonary in enumerate(group):
                species_in_group_list[i].append(dictonary['name'])
                species.append(dictonary['name'])
                mole_fractions.append(dictonary['mole-fraction']['value-list'])
        
        
        conditions = dict(zip(species,mole_fractions))
        species_by_group = dict(zip(species_group_numbers,species_in_group_list))
    
        for i,group in enumerate(attribute_group):
            type_in_group_list[i].append(group['type'])
            relative_uncertainty_in_group_list[i].append(group['relative-uncertainty'])
    
            
        flat_type_list  = [item for sublist in type_in_group_list for item in sublist]
        type_dict = dict(zip(flat_type_list,species_in_group_list))
        relative_uncertainty_by_species = {}
        
        for i, grp in enumerate(species_in_group_list):
            for j,s in enumerate(grp):    
                relative_uncertainty_by_species[s] = relative_uncertainty_in_group_list[i][0]
        
    
        group_lst = []
        for i in species_group_numbers:
            group_lst.append('group_'+str(i))
            
        overall_dict = {}
        
        for i, group in enumerate(group_lst):
            overall_dict[group]= {'species': species_in_group_list[i]}
            overall_dict[group].update({'type': type_in_group_list[i][0]})
            overall_dict[group].update({'relative_uncertainty':relative_uncertainty_in_group_list[i][0]})
    
        if loaded_absorption == {}:
            #build dict and return information for flame speed simulation
            return {'simulationType':simulation_type,
                    'experimentType':experiment_type,
                    'flameWidth': flame_width,
                    'flameWidthrelativeUncertainty':flame_width_relative_uncertainty,
                    'pressureList':pressure,
                    'pressureRelativeUncertainty':pressure_relative_uncertainty,
                    'inletTemperatureList':inlet_temperature,
                    'inletTemperatureRelativeUncertainty':inlet_temperature_relative_uncertainty,
                    'speciesGroupNumbers':species_group_numbers,
                    'speciesInGroupList':species_in_group_list,
                    'conditions':conditions,
                    'species':species,
                    'relativeUncertaintyBySpecies':relative_uncertainty_by_species,
                    'speciesByGroup':species_by_group,
                    'overallDict': overall_dict,
                    'flameSpeedRelativeUncertainty':flame_speed_relative_uncertainty,
                    'flameSpeedAbsoluteUncertainty':flame_speed_absolute_uncertainty,
                    'flameSpeedObservables':flame_speed_observables,
                    'flameSpeedCsvFiles':flame_speed_csv_files,
                    'thermalBoundary':thermal_boundary,
                    'mechanicalBoundary':mechanical_boundary,
                    'typeToSpeciesDict':type_dict
                    }
        else:
            print('Placeholder: no Flame Speed absorption')
        
    def parse_jsr_obj(self,loaded_exp:dict={}, loaded_absorption:dict={}):
        """
        Takes in an unorganized dictonary for a yaml file containing 
        experimental information relating to jet stirred reactors (JSR) and 
        returns an organized dictonary with the necessary information 
        to run an MSI JSR optimization.
        

        Parameters
        ----------
        loaded_exp : dict, optional
           Unorganized dictonary for a yaml file containing 
           experimental information relating to JSR. The default is 
           {}.
        loaded_absorption : dict, optional
            Unorganized dictonary for a yaml file containing experimental 
            information relating to absorption in a JSR. 
            The default is {}.

        Returns
        -------
        dict
            Organized dictonary with the necessary information to run an MSI
            JSR optimization.

        """
        #begin defining importnat variables and parsing into unorganized
        #yaml file to return information to run MSI simulations for
        #JSR       

        simulation_type = loaded_exp['apparatus']['kind']
        pressure = loaded_exp['common-properties']['pressure']['value']
        temperatures = loaded_exp['common-properties']['temperature']['value-list']
        mole_fractions = [((concentration['mole-fraction'])) for concentration in loaded_exp['common-properties']['composition']]
        mole_fractions = [float(elm) for elm in mole_fractions]
        species_names = [(species['species']) for species in loaded_exp['common-properties']['composition']]
        conditions = dict(zip(species_names,mole_fractions))
        thermal_boundary = loaded_exp['common-properties']['assumptions']['thermal-boundary']
        mechanical_boundary = loaded_exp['common-properties']['assumptions']['mechanical-boundary']
        experiment_type = loaded_exp['experiment-type']
        
        mole_fraction_observables = [point['targets'][0]['name'] for point in loaded_exp['datapoints']['mole-fraction']]

        if mole_fraction_observables[0]!=None:
            for i in range(len(mole_fraction_observables)):
                if not mole_fraction_observables[i]:
                    mole_fraction_observables[i]='NO'

        species_uncertainties = [uncert['relative-uncertainty'] for uncert in loaded_exp['common-properties']['composition']]
        species_uncertainties = [float(elm) for elm in species_uncertainties]
        species_uncertainties = dict(zip(species_names,species_uncertainties))
        #observables = [x for x in mole_fraction_observables if x is not None]
     
        concentration_observables = [datapoint['targets'][0]['name'] for datapoint in loaded_exp['datapoints']['concentration']] 

        if concentration_observables[0]!=None:
            for i in range(len(concentration_observables)):
                if not concentration_observables[i]:
                    concentration_observables[i]='NO'    

        #print(concentration_observables,len(concentration_observables))
        observables = [x for x in (mole_fraction_observables + concentration_observables) if x is not None]                
        # for i in range(len(observables)):
        #     if not observables[i]:
        #         observables[i]='NO'                
                
                
                
        #print(observables)
        mole_fraction_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['mole-fraction']]
        concentration_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['concentration']]
        
        csv_files = [x for x in (mole_fraction_csv_files+concentration_csv_files) if x is not None]

        temp_relative_uncertainty = loaded_exp['common-properties']['temperature']['relative-uncertainty']
        temp_relative_uncertainty = float(temp_relative_uncertainty)
        pressure_relative_uncertainty = loaded_exp['common-properties']['pressure']['relative-uncertainty']
        pressure_relative_uncertainty = float(pressure_relative_uncertainty)
        mole_fraction_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['mole-fraction']]
        concentration_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['concentration']]
        volume=loaded_exp['apparatus']['reactor-volume']['value']
        mole_fraction_relative_uncertainty = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['mole-fraction']] 
        concentration_relative_uncertainty = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['concentration']]       

        residence_time=loaded_exp['apparatus']['residence-time']['value']
        restime_relative_uncertainty=loaded_exp['apparatus']['residence-time']['relative-uncertainty']
        #print(csv_files)
        #print(mole_fraction_csv_files,'MOLE FRACTION')
        #print(concentration_csv_files,'CONCENTATIOn')
        
        if loaded_absorption == {}:
             #build dict and return information for flame speed simulation
            return{
               'pressure':pressure,
               'temperatures':temperatures,
               'conditions':conditions,
               'speciesUncertaintys':species_uncertainties,
               'thermalBoundary':thermal_boundary,
               'mechanicalBoundary':mechanical_boundary,
               'moleFractionObservables':mole_fraction_observables,
               'concentrationObservables':concentration_observables,
               'observables':observables,
               'speciesNames':species_names,
               'MoleFractions':mole_fractions,
               'moleFractionCsvFiles':mole_fraction_csv_files,
               'concentrationCsvFiles':concentration_csv_files,
               'tempRelativeUncertainty':temp_relative_uncertainty,
               'pressureRelativeUncertainty': pressure_relative_uncertainty,
               'moleFractionAbsoluteUncertainty':mole_fraction_absolute_uncertainty,
               'moleFractionRelativeUncertainty':mole_fraction_relative_uncertainty,
               'concentrationAbsoluteUncertainty':concentration_absolute_uncertainty,
               'concentrationRelativeUncertainty':concentration_relative_uncertainty,               
               'csvFiles': csv_files,
               'simulationType':  simulation_type,
               'volume': volume,
               'residence_time': residence_time,
               'experimentType':experiment_type,
               'residenceTimeRelativeUncertainty':restime_relative_uncertainty
               #'concentrationObservables': [None]
           }
        else:
            print('Placeholder: no JSR absorption')
        
    def parse_variable_pressure_batch_reactor_obj(self,loaded_exp:dict={}, loaded_absorption:dict={}):
        
        """
        Takes in an unorganized dictonary for a yaml file containing 
        experimental information relating to a variable pressure shock tube and 
        returns an organized dictonary with the necessary information 
        to run an MSI variable pressure shock tube optimization.
        

        Parameters
        ----------
        loaded_exp : dict, optional
           Unorganized dictonary for a yaml file containing 
           experimental information relating to variable pressure shock tube. The default is 
           {}.
        loaded_absorption : dict, optional
            Unorganized dictonary for a yaml file containing experimental 
            information relating to absorption in a variable pressure shock tube. 
            The default is {}.

        Returns
        -------
        dict
            Organized dictonary with the necessary information to run an MSI
            variable pressure shock tube optimization.

        """
        #begin defining importnat variables and parsing into unorganized
        #yaml file to return information to run MSI simulations for
        #variable pressure shock tube         
        
        
        simulation_type = loaded_exp['apparatus']['kind']
        pressure = loaded_exp['common-properties']['pressure']['value']
        temperature = loaded_exp['common-properties']['temperature']['value']
        mole_fractions = [((concentration['mole-fraction'])) for concentration in loaded_exp['common-properties']['composition']]
        mole_fractions = [float(elm) for elm in mole_fractions]
        species_names = [(species['species']) for species in loaded_exp['common-properties']['composition']]
        conditions = dict(zip(species_names,mole_fractions))
        thermal_boundary = loaded_exp['common-properties']['assumptions']['thermal-boundary']
        mechanical_boundary = loaded_exp['common-properties']['assumptions']['mechanical-boundary']
        experiment_type = loaded_exp['experiment-type']
        mole_fraction_observables = [point['targets'][0]['name'] for point in loaded_exp['datapoints']['mole-fraction']]

        for i in range(len(mole_fraction_observables)):
            if mole_fraction_observables[i]==False:
                mole_fraction_observables[i]='NO'
        #print(mole_fraction_observables, len(mole_fraction_observables))
        species_uncertainties = [uncert['relative-uncertainty'] for uncert in loaded_exp['common-properties']['composition']]
        species_uncertainties = [float(elm) for elm in species_uncertainties]
        species_uncertainties = dict(zip(species_names,species_uncertainties))
        
        
        concentration_observables = [datapoint['targets'][0]['name'] for datapoint in loaded_exp['datapoints']['concentration']] 
        for i in range(len(concentration_observables)):
            if concentration_observables[i] == False:
                concentration_observables[i]='NO'      
        #print(concentration_observables,len(concentration_observables))
        observables = [x for x in (mole_fraction_observables + concentration_observables) if x is not None]
        
        volume_trace_csv= loaded_exp['common-properties']['volume-trace']['csvfile']

        initial_time = loaded_exp['common-properties']['time']['initial-time']['value']
        #eventually going to get this from a csv file 
        final_time = loaded_exp['common-properties']['time']['final-time']['value']
    
   
        mole_fraction_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['mole-fraction']]
        concentration_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['concentration']]
        path_length = loaded_exp['apparatus']['inner-diameter']['value']
        csv_files = [x for x in (mole_fraction_csv_files + concentration_csv_files) if x is not None]


        #importing unceratinty values 
        temp_relative_uncertainty = loaded_exp['common-properties']['temperature']['relative-uncertainty']
        temp_relative_uncertainty = float(temp_relative_uncertainty)
        pressure_relative_uncertainty = loaded_exp['common-properties']['pressure']['relative-uncertainty']
        pressure_relative_uncertainty = float(pressure_relative_uncertainty)
        time_shift = float(loaded_exp['common-properties']['time-shift']['value'])
        time_shift_uncertainty = loaded_exp['common-properties']['time-shift']['absolute-uncertainty']['value']
        concentration_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['concentration']]
        concentration_relative_uncertainity = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['concentration']]

        mole_fraction_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['mole-fraction']]

        mole_fraction_relative_uncertainty = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['mole-fraction']]        

        if loaded_absorption == {}:
            return{
               'pressure':pressure,
               'temperature':temperature,
               'conditions':conditions,
               'speciesUncertaintys':species_uncertainties,
               'thermalBoundary':thermal_boundary,
               'mechanicalBoundary':mechanical_boundary,
               'moleFractionObservables':mole_fraction_observables,
               'concentrationObservables': concentration_observables,               
               'observables':observables,
               'initialTime':initial_time,
               'finalTime':final_time,
               'speciesNames':species_names,
               'pathLength':path_length,
               'MoleFractions':mole_fractions,
               'moleFractionCsvFiles':mole_fraction_csv_files,
               'concentrationCsvFiles':concentration_csv_files,
               'tempRelativeUncertainty':temp_relative_uncertainty,
               'pressureRelativeUncertainty': pressure_relative_uncertainty,
               'timeShiftUncertainty':time_shift_uncertainty,
               'concentrationAbsoluteUncertainty':concentration_absolute_uncertainty,
               'concentrationRelativeUncertainity':concentration_relative_uncertainity,
               'moleFractionAbsoluteUncertainty':mole_fraction_absolute_uncertainty,
               'moleFractionRelativeUncertainty':mole_fraction_relative_uncertainty,
               'csvFiles': csv_files,
               'simulationType':  simulation_type,
               'timeShift':time_shift,
               'experimentType':experiment_type,
               'ignitionDelayObservables':[None],
               'volumeTraceCsv':volume_trace_csv

           }
        
        else: 
            print('Placeholder: no JSR absorption')
    def parse_batch_reactor_obj(self,loaded_exp:dict={}, loaded_absorption:dict={}):
        
        
        """
        Takes in an unorganized dictonary for a yaml file containing 
        experimental information relating to a shock tube and 
        returns an organized dictonary with the necessary information 
        to run an MSI shock tube optimization.
        

        Parameters
        ----------
        loaded_exp : dict, optional
           Unorganized dictonary for a yaml file containing 
           experimental information relating to a shock tube. The default is 
           {}.
        loaded_absorption : dict, optional
            Unorganized dictonary for a yaml file containing experimental 
            information relating to absorption in a shock tube. 
            The default is {}.

        Returns
        -------
        dict
            Organized dictonary with the necessary information to run an MSI
            shock tube optimization.

        """
        #begin defining importnat variables and parsing into unorganized
        #yaml file to return information to run MSI simulations for
        #shock tube            
        
        
        
        
        
        
        simulation_type = loaded_exp['apparatus']['kind']
        pressure = loaded_exp['common-properties']['pressure']['value']
        temperature = loaded_exp['common-properties']['temperature']['value']
        mole_fractions = [((concentration['mole-fraction'])) for concentration in loaded_exp['common-properties']['composition']]
        mole_fractions = [float(elm) for elm in mole_fractions]
        species_names = [(species['species']) for species in loaded_exp['common-properties']['composition']]
        conditions = dict(zip(species_names,mole_fractions))
        thermal_boundary = loaded_exp['common-properties']['assumptions']['thermal-boundary']
        mechanical_boundary = loaded_exp['common-properties']['assumptions']['mechanical-boundary']
        experiment_type = loaded_exp['experiment-type']
        mole_fraction_observables = [point['targets'][0]['name'] for point in loaded_exp['datapoints']['mole-fraction']]
        
        for i in range(len(mole_fraction_observables)):
            if mole_fraction_observables[i]==False:
                mole_fraction_observables[i]='NO'
        #print(mole_fraction_observables, len(mole_fraction_observables))
        species_uncertainties = [uncert['relative-uncertainty'] for uncert in loaded_exp['common-properties']['composition']]
        species_uncertainties = [float(elm) for elm in species_uncertainties]
        species_uncertainties = dict(zip(species_names,species_uncertainties))
        
        
        concentration_observables = [datapoint['targets'][0]['name'] for datapoint in loaded_exp['datapoints']['concentration']] 
        
        for i in range(len(concentration_observables)):
            if concentration_observables[i]==False:
                concentration_observables[i]='NO'    
        #print(concentration_observables,len(concentration_observables))
        observables = [x for x in (mole_fraction_observables + concentration_observables) if x is not None]
        

        initial_time = loaded_exp['common-properties']['time']['initial-time']['value']
        #eventually going to get this from a csv file 
        final_time = loaded_exp['common-properties']['time']['final-time']['value']
    
   
        mole_fraction_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['mole-fraction']]
        concentration_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['concentration']]
        path_length = loaded_exp['apparatus']['inner-diameter']['value']
        csv_files = [x for x in (mole_fraction_csv_files + concentration_csv_files) if x is not None]


        #importing unceratinty values 
        temp_relative_uncertainty = loaded_exp['common-properties']['temperature']['relative-uncertainty']
        temp_relative_uncertainty = float(temp_relative_uncertainty)
        pressure_relative_uncertainty = loaded_exp['common-properties']['pressure']['relative-uncertainty']
        pressure_relative_uncertainty = float(pressure_relative_uncertainty)
        time_shift = float(loaded_exp['common-properties']['time-shift']['value'])
        time_shift_uncertainty = loaded_exp['common-properties']['time-shift']['absolute-uncertainty']['value']
        concentration_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['concentration']]
        concentration_relative_uncertainity = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['concentration']]

        mole_fraction_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['mole-fraction']]

        mole_fraction_relative_uncertainty = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['mole-fraction']]        

        if loaded_absorption == {}:
            return{
               'pressure':pressure,
               'temperature':temperature,
               'conditions':conditions,
               'speciesUncertaintys':species_uncertainties,
               'thermalBoundary':thermal_boundary,
               'mechanicalBoundary':mechanical_boundary,
               'moleFractionObservables':mole_fraction_observables,
               'concentrationObservables': concentration_observables,               
               'observables':observables,
               'initialTime':initial_time,
               'finalTime':final_time,
               'speciesNames':species_names,
               'pathLength':path_length,
               'MoleFractions':mole_fractions,
               'moleFractionCsvFiles':mole_fraction_csv_files,
               'concentrationCsvFiles':concentration_csv_files,
               'tempRelativeUncertainty':temp_relative_uncertainty,
               'pressureRelativeUncertainty': pressure_relative_uncertainty,
               'timeShiftUncertainty':time_shift_uncertainty,
               'concentrationAbsoluteUncertainty':concentration_absolute_uncertainty,
               'concentrationRelativeUncertainity':concentration_relative_uncertainity,
               'moleFractionAbsoluteUncertainty':mole_fraction_absolute_uncertainty,
               'moleFractionRelativeUncertainty':mole_fraction_relative_uncertainty,
               'csvFiles': csv_files,
               'simulationType':  simulation_type,
               'timeShift':time_shift,
               'experimentType':experiment_type,
               'ignitionDelayObservables':[None]
           }
        
        else: #absorbtion file given
            absorbance_absolute_uncertainty = [point['absolute-uncertainty'] for point in loaded_exp['datapoints']['absorbance']]
            absorbance_relative_uncertainty = [point['relative-uncertainty'] for point in loaded_exp['datapoints']['absorbance']]
            #importing absorbance uncertainty 

            absorbance_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['absorbance']]
            absorbance_csv_wavelengths = [csvfile['wavelength']['value'] for csvfile in loaded_exp['datapoints']['absorbance']]
            absorption_observables = [species['species'] for species in loaded_absorption['Absorption-coefficients']]

            observables = [x for x in (mole_fraction_observables + concentration_observables + absorption_observables) if x is not None]


            uncertainty_parameter_ones = [[] for i in range(len(loaded_absorption['Absorption-coefficients']))]
            for uncertainty in range(len(loaded_absorption['Absorption-coefficients'])):
                temp = [wavelength['parameter-one']['absolute-uncertainty']['value'] for wavelength in loaded_absorption['Absorption-coefficients'][uncertainty]['wave-lengths']]
                uncertainty_parameter_ones[uncertainty] = temp
                
            uncertainty_parameter_twos = [[] for i in range(len(loaded_absorption['Absorption-coefficients']))]
            for uncertainty in range(len(loaded_absorption['Absorption-coefficients'])):
                temp = [wavelength['parameter-two']['absolute-uncertainty']['value'] for wavelength in loaded_absorption['Absorption-coefficients'][uncertainty]['wave-lengths']]
                uncertainty_parameter_twos[uncertainty] = temp        
                
                
            # add the function which will return the coupled paramters here 
            parameter_ones = []
            for p1 in range(len(loaded_absorption['Absorption-coefficients'])):
                temp = [wl['parameter-one']['value'] for wl in loaded_absorption['Absorption-coefficients'][p1]['wave-lengths']]
                parameter_ones.append(temp)

            parameter_twos = [] 
            for p2 in range(len(loaded_absorption['Absorption-coefficients'])):
                temp = [wl['parameter-two']['value'] for wl in loaded_absorption['Absorption-coefficients'][p2]['wave-lengths']]
                parameter_twos.append(temp)
                
            coupledCoefficients = [list(zip(parameter_ones[x],parameter_twos[x])) for x in range(len(parameter_ones))]
            functional_form = []
            for form in range(len(loaded_absorption['Absorption-coefficients'])):
                temp = [wl['functional-form'] for wl in loaded_absorption['Absorption-coefficients'][form]['wave-lengths']]
                functional_form.append(temp)

   
            return {
                   'pressure':pressure,
                   'temperature':temperature,
                   'conditions':conditions,
                   'thermalBoundary':thermal_boundary,
                   'mechanicalBoundary':mechanical_boundary,
                   'speciesNames': species_names,
                   'observables': observables,
                   'moleFractionObservables':mole_fraction_observables,
                   'concentrationObservables':concentration_observables, 
                   'absorbanceObservables':absorption_observables,
                   'initialTime': initial_time,
                   'finalTime':final_time,
                   'speciesNames': species_names,
                   'MoleFractions':mole_fractions,
                   'absorbanceCsvFiles': absorbance_csv_files,
                   'moleFractionCsvFiles':mole_fraction_csv_files,
                   'concentrationCsvFiles':concentration_csv_files,
                   'absorbanceCsvWavelengths': absorbance_csv_wavelengths,
                   'pathLength':path_length,
                   'tempRelativeUncertainty': temp_relative_uncertainty,
                   'pressureRelativeUncertainty': pressure_relative_uncertainty,
                   'speciesUncertaintys': species_uncertainties,
                   'timeShiftUncertainty': time_shift_uncertainty,
                   'concentrationAbsoluteUncertainty': concentration_absolute_uncertainty,
                   'concentrationRelativeUncertainity': concentration_relative_uncertainity,
                   'moleFractionAbsoluteUncertainty': mole_fraction_absolute_uncertainty,
                   'moleFractionRelativeUncertainty': mole_fraction_relative_uncertainty,
                   'absorbanceAbsoluteUncertainty': absorbance_absolute_uncertainty,
                   'absorbanceRelativeUncertainty': absorbance_relative_uncertainty,
                   'uncertaintyParameterOnes':uncertainty_parameter_ones,
                   'uncertaintyParameterTwos':uncertainty_parameter_twos,
                   'coupledCoefficients':coupledCoefficients,
                   'simulationType':  simulation_type,
                   'parameterOnes':parameter_ones,
                   'parameterTwos':parameter_twos,
                   'functionalForm':functional_form,
                   'simulationType':  simulation_type,
                   'timeShift':time_shift,
                   'experimentType':experiment_type,
                   'ignitionDelayObservables':[None]
                   }
        
        
    def parse_RCM_obj(self, loaded_exp:dict={}, loaded_absorption:dict={}):
        
        """
        Takes in an unorganized dictonary for a yaml file containing 
        experimental information relating to a shock tube and 
        returns an organized dictonary with the necessary information 
        to run an MSI RCM optimization.
        

        Parameters
        ----------
        loaded_exp : dict, optional
           Unorganized dictonary for a yaml file containing 
           experimental information relating to an RCM. The default is 
           {}.
        loaded_absorption : dict, optional
            Unorganized dictonary for a yaml file containing experimental 
            information relating to absorption in an RCM. 
            The default is {}.

        Returns
        -------
        dict
            Organized dictonary with the necessary information to run an RCM
            optimization.

        """
        #begin defining importnat variables and parsing into unorganized
        #yaml file to return information to run MSI simulations for
        #RCM     
    
        simulation_type = loaded_exp['apparatus']['kind']
        pressure_list = loaded_exp['common-properties']['pressure']['value-list']
        temperature_list = loaded_exp['common-properties']['temperature']['value-list']
        temp_relative_uncertainty = loaded_exp['common-properties']['temperature']['relative-uncertainty']
        temp_relative_uncertainty = float(temp_relative_uncertainty)
        pressure_relative_uncertainty = loaded_exp['common-properties']['pressure']['relative-uncertainty']
        pressure_relative_uncertainty = float(pressure_relative_uncertainty)
        initial_time = loaded_exp['common-properties']['time']['initial-time']['value']
            #eventually going to get this from a csv file 
        final_time = loaded_exp['common-properties']['time']['final-time']['value']
        
        time_shift = float(loaded_exp['common-properties']['time-shift']['value'])
        time_shift_uncertainty = loaded_exp['common-properties']['time-shift']['absolute-uncertainty']['value']
        species_group_numbers = [(species['species-group']) for species in loaded_exp['common-properties']['composition']]
    
        thermal_boundary = loaded_exp['common-properties']['assumptions']['thermal-boundary']
        mechanical_boundary = loaded_exp['common-properties']['assumptions']['mechanical-boundary']
        experiment_type = loaded_exp['experiment-type']
        
            
        ignition_delay_observables = [datapoint['targets'][0]['name'] for datapoint in loaded_exp['datapoints']['ignition-delay']]            
        observables = [x for x in (ignition_delay_observables) if x is not None]
            
    
        volume_trace_csv_files= loaded_exp['common-properties']['volume-trace']['csvfile-list']

        
        

        ignition_delay_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['ignition-delay']]
        ignition_dealy_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['ignition-delay']]
        ignition_dealy_relative_uncertainity = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['ignition-delay']]
    
        path_length = loaded_exp['apparatus']['inner-diameter']['value']
        csv_files = [x for x in (ignition_delay_csv_files) if x is not None]
            
        ignition_type = loaded_exp['common-properties']['ignition-type']['type']
        ignition_target = loaded_exp['common-properties']['ignition-type']['target']
        
        species_groups = [(species['mixture']) for species in loaded_exp['common-properties']['composition']]
        attribute_group = [(attribute['attributes']) for attribute in loaded_exp['common-properties']['composition']]
        species_in_group_list = [[] for groups in species_groups]
        type_in_group_list = [[] for groups in species_groups]
        relative_uncertainty_in_group_list = [[] for groups in species_groups]
        species = []
        mole_fractions = []
        species_names = []
        for i,group in enumerate(species_groups):
            for j, dictonary in enumerate(group):
                species_names.append(dictonary['name'])
    
        for i,group in enumerate(species_groups):
            for j, dictonary in enumerate(group):
                species_in_group_list[i].append(dictonary['name'])
                species.append(dictonary['name'])
                mole_fractions.append(dictonary['mole-fraction']['value-list'])
        
        conditions = dict(zip(species,mole_fractions))
        conditions_dict_list=copy.deepcopy(conditions)
        max_cond_length=0
        for species in conditions.keys():
            if len(conditions[species])>max_cond_length:
                max_cond_length=len(conditions[species])
        temp_dict={}
        for species in conditions.keys():
            if len(conditions[species])==1:
                temp_dict[species]=max_cond_length*[conditions[species]]
            else:
                temp_dict[species]=conditions[species]
                
        conditions_to_run = [{key:value[index] for key, value in temp_dict.items()}
                             for index in range(len(list(temp_dict.values())[0]))]
        for i in range(len(conditions_to_run)):
            for j in conditions_to_run[i].keys():
                if type(conditions_to_run[i][j])==list:
                    conditions_to_run[i][j]=conditions_to_run[i][j][0]
        
        species_by_group = dict(zip(species_group_numbers,species_in_group_list))
    
        for i,group in enumerate(attribute_group):
            type_in_group_list[i].append(group['type'])
            relative_uncertainty_in_group_list[i].append(group['relative-uncertainty'])
    
            
        flat_type_list  = [item for sublist in type_in_group_list for item in sublist]
        type_dict = dict(zip(flat_type_list,species_in_group_list))
        relative_uncertainty_by_species = {}
        for i, grp in enumerate(species_in_group_list):
            for j,s in enumerate(grp):    
                relative_uncertainty_by_species[s] = relative_uncertainty_in_group_list[i][0]
        
    
        group_lst = []
        for i in species_group_numbers:
            group_lst.append('group_'+str(i))
            
        overall_dict = {}
        
        for i, group in enumerate(group_lst):
            overall_dict[group]= {'species': species_in_group_list[i]}
            overall_dict[group].update({'type': type_in_group_list[i][0]})
            overall_dict[group].update({'relative_uncertainty':relative_uncertainty_in_group_list[i][0]})
    
        if loaded_absorption == {}:
            return{
                   'pressures':pressure_list,
                   'pressureRelativeUncertainty': [pressure_relative_uncertainty],
                   'temperatures':temperature_list,
                    'tempRelativeUncertainty':[temp_relative_uncertainty],
                   'conditions':conditions,
                   'thermalBoundary':thermal_boundary,
                   'mechanicalBoundary':mechanical_boundary,
                   'ignitionDelayObservables': ignition_delay_observables,               
                   'observables':observables,
                   'initialTime':initial_time,
                   'finalTime':final_time,
                   'speciesNames':species_names,
                   'pathLength':path_length,
                   'MoleFractions':mole_fractions,
                   'ignitionDelayCsvFiles':ignition_delay_csv_files,
                   'timeShiftUncertainty':time_shift_uncertainty,
                   'ignitionDelayRelativeUncertainty':ignition_dealy_relative_uncertainity,
                   'ignitionDelayAbsoluteUncertainty':ignition_dealy_absolute_uncertainty,
                   'csvFiles': csv_files,
                   'simulationType':  simulation_type,
                   'time_shift':time_shift,
                   'experimentType':experiment_type,
                    'speciesGroupNumbers':species_group_numbers,
                    'speciesInGroupList':species_in_group_list,
                    'conditions':conditions,
                    'relativeUncertaintyBySpecies':relative_uncertainty_by_species,
                    'speciesByGroup':species_by_group,
                    'overallDict': overall_dict,
                    'typeToSpeciesDict':type_dict,
                    'target':ignition_target,
                    'target_type':ignition_type,
                    'conditions_to_run':conditions_to_run,
                    'conditions_dict_list':conditions_dict_list,
                    'concentrationObservables': [None],
                    'moleFractionObservables': [None],
                    'volumeTraceCsvList':volume_trace_csv_files
                   }       
        else:
            print('We do not have absorbance installed for ignition delay')       
        
    def parse_ignition_delay_obj(self, loaded_exp:dict={}, loaded_absorption:dict={}):
        """
        Takes in an unorganized dictonary for a yaml file containing 
        experimental information relating to a shock tube and 
        returns an organized dictonary with the necessary information 
        to run an MSI igntion dealy optimization.
        

        Parameters
        ----------
        loaded_exp : dict, optional
           Unorganized dictonary for a yaml file containing 
           experimental information relating to an RCM. The default is 
           {}.
        loaded_absorption : dict, optional
            Unorganized dictonary for a yaml file containing experimental 
            information relating to absorption in an igntion dealy. 
            The default is {}.

        Returns
        -------
        dict
            Organized dictonary with the necessary information to run an
            igntion dealy optimization.

        """
        #begin defining importnat variables and parsing into unorganized
        #yaml file to return information to run MSI simulations for
        #igntion dealy     
    
        simulation_type = loaded_exp['apparatus']['kind']
        pressure_list = loaded_exp['common-properties']['pressure']['value-list']
        temperature_list = loaded_exp['common-properties']['temperature']['value-list']
        temp_relative_uncertainty = loaded_exp['common-properties']['temperature']['relative-uncertainty']
        temp_relative_uncertainty = float(temp_relative_uncertainty)
        pressure_relative_uncertainty = loaded_exp['common-properties']['pressure']['relative-uncertainty']
        pressure_relative_uncertainty = float(pressure_relative_uncertainty)
        initial_time = loaded_exp['common-properties']['time']['initial-time']['value']
            #eventually going to get this from a csv file 
        final_time = loaded_exp['common-properties']['time']['final-time']['value']
        
        time_shift = float(loaded_exp['common-properties']['time-shift']['value'])
        time_shift_uncertainty = loaded_exp['common-properties']['time-shift']['absolute-uncertainty']['value']
        species_group_numbers = [(species['species-group']) for species in loaded_exp['common-properties']['composition']]
    
        thermal_boundary = loaded_exp['common-properties']['assumptions']['thermal-boundary']
        mechanical_boundary = loaded_exp['common-properties']['assumptions']['mechanical-boundary']
        experiment_type = loaded_exp['experiment-type']
        
            
        ignition_delay_observables = [datapoint['targets'][0]['name'] for datapoint in loaded_exp['datapoints']['ignition-delay']]            
        observables = [x for x in (ignition_delay_observables) if x is not None]
            
    
        
       
        ignition_delay_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['ignition-delay']]
        ignition_dealy_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['ignition-delay']]
        ignition_dealy_relative_uncertainity = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['ignition-delay']]
    
        path_length = loaded_exp['apparatus']['inner-diameter']['value']
        csv_files = [x for x in (ignition_delay_csv_files) if x is not None]
            
        ignition_type = loaded_exp['common-properties']['ignition-type']['type']
        ignition_target = loaded_exp['common-properties']['ignition-type']['target']
        
        species_groups = [(species['mixture']) for species in loaded_exp['common-properties']['composition']]
        attribute_group = [(attribute['attributes']) for attribute in loaded_exp['common-properties']['composition']]
        species_in_group_list = [[] for groups in species_groups]
        type_in_group_list = [[] for groups in species_groups]
        relative_uncertainty_in_group_list = [[] for groups in species_groups]
        species = []
        mole_fractions = []
        species_names = []
        for i,group in enumerate(species_groups):
            for j, dictonary in enumerate(group):
                species_names.append(dictonary['name'])
    
        for i,group in enumerate(species_groups):
            for j, dictonary in enumerate(group):
                species_in_group_list[i].append(dictonary['name'])
                species.append(dictonary['name'])
                mole_fractions.append(dictonary['mole-fraction']['value-list'])
        
        conditions = dict(zip(species,mole_fractions))
        conditions_dict_list=copy.deepcopy(conditions)
        max_cond_length=0
        for species in conditions.keys():
            if len(conditions[species])>max_cond_length:
                max_cond_length=len(conditions[species])
        temp_dict={}
        for species in conditions.keys():
            if len(conditions[species])==1:
                temp_dict[species]=max_cond_length*[conditions[species]]
            else:
                temp_dict[species]=conditions[species]
                
        conditions_to_run = [{key:value[index] for key, value in temp_dict.items()}
                             for index in range(len(list(temp_dict.values())[0]))]
        for i in range(len(conditions_to_run)):
            for j in conditions_to_run[i].keys():
                if type(conditions_to_run[i][j])==list:
                    conditions_to_run[i][j]=conditions_to_run[i][j][0]
        
        species_by_group = dict(zip(species_group_numbers,species_in_group_list))
    
        for i,group in enumerate(attribute_group):
            type_in_group_list[i].append(group['type'])
            relative_uncertainty_in_group_list[i].append(group['relative-uncertainty'])
    
            
        flat_type_list  = [item for sublist in type_in_group_list for item in sublist]
        type_dict = dict(zip(flat_type_list,species_in_group_list))
        relative_uncertainty_by_species = {}
        for i, grp in enumerate(species_in_group_list):
            for j,s in enumerate(grp):    
                relative_uncertainty_by_species[s] = relative_uncertainty_in_group_list[i][0]
        
    
        group_lst = []
        for i in species_group_numbers:
            group_lst.append('group_'+str(i))
            
        overall_dict = {}
        
        for i, group in enumerate(group_lst):
            overall_dict[group]= {'species': species_in_group_list[i]}
            overall_dict[group].update({'type': type_in_group_list[i][0]})
            overall_dict[group].update({'relative_uncertainty':relative_uncertainty_in_group_list[i][0]})
    
        if loaded_absorption == {}:
            return{
                   'pressures':pressure_list,
                   'pressureRelativeUncertainty': [pressure_relative_uncertainty],
                   'temperatures':temperature_list,
                    'tempRelativeUncertainty':[temp_relative_uncertainty],
                   'conditions':conditions,
                   'thermalBoundary':thermal_boundary,
                   'mechanicalBoundary':mechanical_boundary,
                   'ignitionDelayObservables': ignition_delay_observables,               
                   'observables':observables,
                   'initialTime':initial_time,
                   'finalTime':final_time,
                   'speciesNames':species_names,
                   'pathLength':path_length,
                   'MoleFractions':mole_fractions,
                   'ignitionDelayCsvFiles':ignition_delay_csv_files,
                   'timeShiftUncertainty':time_shift_uncertainty,
                   'ignitionDelayRelativeUncertainty':ignition_dealy_relative_uncertainity,
                   'ignitionDelayAbsoluteUncertainty':ignition_dealy_absolute_uncertainty,
                   'csvFiles': csv_files,
                   'simulationType':  simulation_type,
                   'time_shift':time_shift,
                   'experimentType':experiment_type,
                    'speciesGroupNumbers':species_group_numbers,
                    'speciesInGroupList':species_in_group_list,
                    'conditions':conditions,
                    'relativeUncertaintyBySpecies':relative_uncertainty_by_species,
                    'speciesByGroup':species_by_group,
                    'overallDict': overall_dict,
                    'typeToSpeciesDict':type_dict,
                    'target':ignition_target,
                    'target_type':ignition_type,
                    'conditions_to_run':conditions_to_run,
                    'conditions_dict_list':conditions_dict_list,
                    'concentrationObservables': [None],
                    'moleFractionObservables': [None]
                   }       
        else:
            print('We do not have absorbance installed for ignition delay')
            
    def parse_flow_reactor_obj(self, loaded_exp:dict={}, loaded_absorption:dict={}):
        """
        Takes in an unorganized dictonary for a yaml file containing 
        experimental information relating to a shock tube and 
        returns an organized dictonary with the necessary information 
        to run an MSI flow reactor optimization.
        

        Parameters
        ----------
        loaded_exp : dict, optional
           Unorganized dictonary for a yaml file containing 
           experimental information relating to a flow reactor. The default is 
           {}.
        loaded_absorption : dict, optional
            Unorganized dictonary for a yaml file containing experimental 
            information relating to absorption in a flow reactor. 
            The default is {}.

        Returns
        -------
        dict
            Organized dictonary with the necessary information to run a
           flow reactor optimization.

        """
        #begin defining importnat variables and parsing into unorganized
        #yaml file to return information to run MSI simulations for
        #flow reactor          
        

        
        residence_time_list = loaded_exp['apparatus']['residence-time']['value-list']
        simulation_type = loaded_exp['apparatus']['kind']
        pressure = loaded_exp['common-properties']['pressure']['value']
        temperature_list = loaded_exp['common-properties']['temperature']['value-list']
        mole_fractions = [((concentration['mole-fraction'])) for concentration in loaded_exp['common-properties']['composition']]
        mole_fractions = [float(elm) for elm in mole_fractions]
        species_names = [(species['species']) for species in loaded_exp['common-properties']['composition']]
        conditions = dict(zip(species_names,mole_fractions))
        thermal_boundary = loaded_exp['common-properties']['assumptions']['thermal-boundary']
        mechanical_boundary = loaded_exp['common-properties']['assumptions']['mechanical-boundary']
        experiment_type = loaded_exp['experiment-type']
        mole_fraction_observables = [point['targets'][0]['name'] for point in loaded_exp['datapoints']['mole-fraction']]
        
        for i in range(len(mole_fraction_observables)):
            if mole_fraction_observables[i]==False:
                mole_fraction_observables[i]='NO'
                
        species_uncertainties = [uncert['relative-uncertainty'] for uncert in loaded_exp['common-properties']['composition']]
        species_uncertainties = [float(elm) for elm in species_uncertainties]
        species_uncertainties = dict(zip(species_names,species_uncertainties))
            
            
        concentration_observables = [datapoint['targets'][0]['name'] for datapoint in loaded_exp['datapoints']['concentration']]
        
        for i in range(len(concentration_observables)):
            if concentration_observables[i] == False:
                    concentration_observables[i]='NO'               
        observables = [x for x in (mole_fraction_observables + concentration_observables) if x is not None]
     
        initial_time = loaded_exp['common-properties']['time']['initial-time']['value']
            #eventually going to get this from a csv file 
        
       
        mole_fraction_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['mole-fraction']]
        concentration_csv_files = [csvfile['csvfile'] for csvfile in loaded_exp['datapoints']['concentration']]
        csv_files = [x for x in (mole_fraction_csv_files + concentration_csv_files) if x is not None]
    
    
            #importing unceratinty values 
        temp_relative_uncertainty = loaded_exp['common-properties']['temperature']['relative-uncertainty']
        temp_relative_uncertainty = float(temp_relative_uncertainty)
        pressure_relative_uncertainty = loaded_exp['common-properties']['pressure']['relative-uncertainty']
        pressure_relative_uncertainty = float(pressure_relative_uncertainty)
        time_shift_list = loaded_exp['common-properties']['time-shift']['value-list']
        for i,value in enumerate(time_shift_list):
            time_shift_list[i]=float(value)
        time_shift_uncertainty = loaded_exp['common-properties']['time-shift']['absolute-uncertainty']['value']
        concentration_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['concentration']]
        concentration_relative_uncertainity = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['concentration']]
    
        mole_fraction_absolute_uncertainty = [point['targets'][0]['absolute-uncertainty'] for point in loaded_exp['datapoints']['mole-fraction']]
    
        mole_fraction_relative_uncertainty = [point['targets'][0]['relative-uncertainty'] for point in loaded_exp['datapoints']['mole-fraction']]        
        
        time_shift_original = loaded_exp['common-properties']['time-shift']['value-list']
        if len(temperature_list) > 1 and len(residence_time_list) ==1:
            residence_time_list = residence_time_list*len(temperature_list)
        if len(temperature_list) > 1 and len(time_shift_list) ==1:
            time_shift_list = time_shift_list*len(temperature_list)            
            
        if loaded_absorption=={}:
            return{
                   'pressure':pressure,
                   'temperatures':temperature_list,
                   'residenceTimes':residence_time_list,
                   'conditions':conditions,
                   'speciesUncertaintys':species_uncertainties,
                   'thermalBoundary':thermal_boundary,
                   'mechanicalBoundary':mechanical_boundary,
                   'moleFractionObservables':mole_fraction_observables,
                   'concentrationObservables': concentration_observables,               
                   'observables':observables,
                   'initialTime':initial_time,
                   'speciesNames':species_names,
                   'MoleFractions':mole_fractions,
                   'moleFractionCsvFiles':mole_fraction_csv_files,
                   'concentrationCsvFiles':concentration_csv_files,
                   'tempRelativeUncertainty':temp_relative_uncertainty,
                   'pressureRelativeUncertainty': pressure_relative_uncertainty,
                   'timeShiftUncertainty':time_shift_uncertainty,
                   'concentrationAbsoluteUncertainty':concentration_absolute_uncertainty,
                   'concentrationRelativeUncertainity':concentration_relative_uncertainity,
                   'moleFractionAbsoluteUncertainty':mole_fraction_absolute_uncertainty,
                   'moleFractionRelativeUncertainty':mole_fraction_relative_uncertainty,
                   'csvFiles': csv_files,
                   'simulationType':  simulation_type,
                   'timeShift':time_shift_list,
                   'experimentType':experiment_type,
                   'timeShiftOriginal':time_shift_original
               }        
        else:
            print('We do not have absorbance installed for ignition delay')
        
    
    
    
    
    
    
    
    def load_yaml_list(self, yaml_list:list = []):
        '''
        Takes in a nested list of yaml file strings and returns a list of
        unorganized dictonaries for all the yaml files passed in.

        Parameters
        ----------
        yaml_list : list, optional
            This is a nested list of tuples Where the yaml files pertaining
            to a single experiment are in their own tuple. For example, 
            [('Hong_0.yaml','Hong_0_abs.yaml'), ('Hong_1.yaml')] 
            The default is [].

        Returns
        -------
        list_of_yaml_objects: list
        A nested list structured the same as the input,
        of unorganized dictonaries for all the yaml files passed in.

        '''
        #make empty list to append objects to 
        list_of_yaml_objects = []
        #iterate over nested list
        for tup in yaml_list:
            #iterate over tuple
            temp = []
            for file in tup:
                temp.append(self.load_to_obj(file))
            list_of_yaml_objects.append(temp) 
            #load file
            #change tuple to a list for further use
        list_of_yaml_objects = [tuple(lst) for lst in list_of_yaml_objects ] 
        #print(list_of_yaml_objects[0][0])              
        return list_of_yaml_objects
    
    def parsing_multiple_dictonaries(self,list_of_yaml_objects:list = [],loop_counter:int=0):
        '''
        This function takes in a list of nested and unorganized yaml 
        dictonaries and sorts them into the correct parser based on their
        simulation and experiment type. It returns a nested list of organized
        and parsed dictonaries which contain all the necessary variables to
        run a simulation for the experiment as well as an optimization. 

        Parameters
        ----------
        list_of_yaml_objects : list, optional
            Nested list of unorganized yaml dictonaries which could be for a
            variety of differnt experiments and simulation types. The default 
            is [].
            
        loop_counter : int, optional
             The loop counter keeps track of the current iteration number 
             of the optimization. If it is zero this function saves the 
             original set of parsed experimental dictonaries for use in other 
             functions. The default is 0.

        Returns
        -------
        experiment_dictonaries : list
            A nested list of organized and parsed dictonaries which contain
            all the necessary variables to run a simulation for the experiment 
            as well as an optimization. 

        '''
        
        
        experiment_dictonaries = []
        counter=0
        for tup in list_of_yaml_objects:
            
            simtype, experiment_type= self.get_sim_type(tup[0])
           # print(simtype)
            simtype=str(simtype)
            experiment_type=str(experiment_type)

            #simtype = 'shock tube'
            #if simtype=='shock tube' or simtype=='Shock Tube' or simtype=='Shock tube':
            if re.match('[Bb]atch[ -][Rr]eactor',simtype) and re.match('[Ss]pecies[ -][Pp]rofile',experiment_type):
                if len(tup)>1:
                    experiment_dictonaries.append(self.parse_batch_reactor_obj(loaded_exp = tup[0],
                                                                            loaded_absorption = tup[1]))
    
                else:
                    
                    experiment_dictonaries.append(self.parse_batch_reactor_obj(loaded_exp = tup[0]))
            elif re.match('[Jj][Ss][Rr]',simtype) or re.match('[Jj]et[- ][Ss]tirred[- ][Rr]eactor',simtype) and re.match('[Ss]pecies[ -][Pp]rofile',experiment_type):
                if len(tup)>1:
                    experiment_dictonaries.append(self.parse_jsr_obj(loaded_exp = tup[0],
                                                                            loaded_absorption = tup[1]))
    
                else:
                    
                    experiment_dictonaries.append(self.parse_jsr_obj(loaded_exp = tup[0]))
            elif re.match('[Ff]lame[- ][Ss]peed',simtype) and re.match('[Oo][Nn][Ee]|[1][ -][dD][ -][Ff]lame',experiment_type):
                if len(tup)>1:
                    experiment_dictonaries.append(self.parse_flame_speed_obj(loaded_exp = tup[0],
                                                                            loaded_absorption = tup[1]))
                else:
                    experiment_dictonaries.append(self.parse_flame_speed_obj(loaded_exp = tup[0]))
                
            elif re.match('[Bb]atch[ -][Rr]eactor',simtype) and re.match('[Ii]gnition[ -][Dd]elay',experiment_type):
                if len(tup)>1:
                    experiment_dictonaries.append(self.parse_ignition_delay_obj(loaded_exp = tup[0],
                                                                            loaded_absorption = tup[1]))
                else:
                    experiment_dictonaries.append(self.parse_ignition_delay_obj(loaded_exp = tup[0])) 
                    
                
            elif re.match('[Rr][Cc][Mm]',simtype) and re.match('[Ii]gnition[ -][Dd]elay',experiment_type):
                if len(tup)>1:
                    experiment_dictonaries.append(self.parse_RCM_obj(loaded_exp = tup[0],
                                                                            loaded_absorption = tup[1]))
                else:
                    experiment_dictonaries.append(self.parse_RCM_obj(loaded_exp = tup[0]))                     
                
            elif re.match('[Vv]ariable[ -][Pp]ressure[ -][Bb]atch[ -][Rr]eactor',simtype) and re.match('[Ss]pecies[ -][Pp]rofile',experiment_type):
                if len(tup)>1:
                    experiment_dictonaries.append(self.parse_variable_pressure_batch_reactor_obj(loaded_exp = tup[0],
                                                                            loaded_absorption = tup[1]))
                else:
                    experiment_dictonaries.append(self.parse_variable_pressure_batch_reactor_obj(loaded_exp = tup[0]))                     
                    
            elif  re.match('[Ff]low[- ][Rr]eactor',simtype) and re.match('[Ss]pecies[ -][Pp]rofile',experiment_type):
                if len(tup)>1:
                    experiment_dictonaries.append(self.parse_flow_reactor_obj()(loaded_exp = tup[0],
                                                                            loaded_absorption = tup[1]))
                else:
                    experiment_dictonaries.append(self.parse_flow_reactor_obj(loaded_exp = tup[0])) 

                    
            else:
                print('Failed to parse Yaml files- unrecognized simulation type for tuple index: '+str(counter))
            counter=counter+1
        if loop_counter == 0   :     
            self.original_experimental_conditions = experiment_dictonaries
            
        return experiment_dictonaries
    
    def assemble_dicts_for_master_equation(self,experiment_dictonaries:list=[],
                                           master_equation_reactions:list=[],
                                           additional_parameters:dict={}):
        '''
        

        Parameters
        ----------
        experiment_dictonaries : list, optional
            DESCRIPTION. The default is [].
        master_equation_reactions : list, optional
            DESCRIPTION. The default is [].
        additional_parameters : dict, optional
            DESCRIPTION. The default is {}.

        Returns
        -------
        master_equation_parameters : TYPE
            DESCRIPTION.

        '''
        
        
        temperatures = []
        pressures = []
        conditions = []
        master_equation_parameters = []
        for exp in experiment_dictonaries:
            temperatures.append(exp['temperature'])
            pressures.append(exp['pressure'])
            conditions.append(exp['conditions'])
            
        if bool(additional_parameters) == False:
            parameters = {'W':['Energy','Frequencies','SymmetryFactor'],
                          'B':['ImaginaryFrequency']}
            
        for reaction in range(len(master_equation_reactions)):
            temp_dict = {}
            for key in parameters.keys():
                for param in parameters[key]:
                    string = str(key+str(reaction)+'_'+param)
                    temp_dict[string] = [temperatures,pressures,conditions]
            master_equation_parameters.append(temp_dict)
        
        return master_equation_parameters
    
    
    def yaml_file_copy(self,fileName:str):
        '''
        This function makes a copy of a yaml file and writes a new file using
        that copy. The new files name is: original_file_name_updated.yaml.
        The function returns a string of the new file name.

        Parameters
        ----------
        fileName : str
            Name of yaml file that a copy is being made of.

        Returns
        -------
        NewName : str
            Name of new yaml file.

        '''
    
        tempName = fileName[0:(len(fileName)-5)]
        yamlExtention = fileName[(len(fileName)-5):]
        NewName = tempName +'_updated'+yamlExtention
        shutil.copy2(fileName, NewName) 
    
        return NewName
    
    def reorder_temps_from_dict(self,phys_updates, 
                                temperature_list, 
                                yaml_number):
        '''
        Function takes the physical_observables_updates_list, temperature_list,
        and yaml_number and ensures that it returns all the temperature 
        updates in the correct order. 
        NOTE:This function is not currently being used. It seems like it was
        written and then we realized we didn't actually need it.

        Parameters
        ----------
        phys_updates : list
            List of physical updates.
        temperature_list : list
            List of temperatures.
        yaml_number : int
            Yaml file number.

        Returns
        -------
        values : list
            List of temperature updates.

        '''
        values=[]
        strings=[]
        for i in temperature_list:
            strings.append('T_experiment_'+str(yaml_number)+'_'+str(i))
            value=phys_updates[strings[-1]]
            values.append(value)
        return values
    
    def yaml_file_updates(self,file_name_list,
                          parsed_yaml_list,
                          experiment_dict_list,
                          physical_observables_updates_list,
                          loop_counter=0):
        '''
        This function updates the physical model parameters (excluding absorption cofficents)
        stored insdie the  yaml files after an iteration of the optimization 
        is finished running.
        
        This function takes the list of yaml files names, the organized 
        parsed yaml file dictonaries list, the experiment dictonaries list,
        the physical observables updates list, and the loop counter as 
        arguments and returns the file name list of updated files. 
        
        Parameters
        ----------
        file_name_list : list
            Nested list of yaml file names to be updated.
        parsed_yaml_list : list
            Nested list (same order as the file name list) of organized
            and parsed yaml file dictonaries.
        experiment_dict_list : list
            List of dictonaries pertaining to the simulations run in the 
            optimization that contain relevent information. The list is in the
            same order as the file name list.
        physical_observables_updates_list : list
            List of dictonaries that contain the updates (delta x) for each
            experiment. The list is in the same order as file name list.
        loop_counter : int, optional
            Integer tht keeps track of what iteration the optimization is on.
            If this value is zero copies of the original yaml files will
            be made, and the copy will be updated instead of the original.
            The default is 0.

        Returns
        -------
        file_name_list: list
            List of file names that have been updated.

        '''
        
        #always pass in the updated file name list except for the first run of the code
        if loop_counter == 0:
            updated_file_name_list = []
        
        for yaml_file in range(len(file_name_list)):
            temp = []
            if loop_counter == 0: 
                new_file_name = self.yaml_file_copy(file_name_list[yaml_file][0])                 
                temp.append(new_file_name)
                updated_file_name_list.append(temp)
                if len(file_name_list[yaml_file])>1:
                    temp.append(file_name_list[yaml_file][1])
                
            else: 
                new_file_name = file_name_list[yaml_file][0]
    
            if experiment_dict_list[0]['simulation'].physicalSens ==1 :
                if re.match('[Bb]atch[ -][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']) and re.match('[Ss]pecies[- ][Pp]rofile',self.original_experimental_conditions[yaml_file]['experimentType']):
                    temp = self.original_experimental_conditions[yaml_file]['temperature']
                    time_shift = self.original_experimental_conditions[yaml_file]['timeShift']
                    press = self.original_experimental_conditions[yaml_file]['pressure']
                    conditions = self.original_experimental_conditions[yaml_file]['conditions']
                    
                elif re.match('[Bb]atch[ -][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']) and re.match('[Ii]gnition[- ][Dd]elay',self.original_experimental_conditions[yaml_file]['experimentType']):
                    temp = self.original_experimental_conditions[yaml_file]['temperatures']
                    time_shift = self.original_experimental_conditions[yaml_file]['time_shift']
                    press = self.original_experimental_conditions[yaml_file]['pressures']
                    conditions = self.original_experimental_conditions[yaml_file]['conditions']

                elif re.match('[Rr][Cc][Mm]',self.original_experimental_conditions[yaml_file]['simulationType']) and re.match('[Ii]gnition[- ][Dd]elay',self.original_experimental_conditions[yaml_file]['experimentType']):
                    temp = self.original_experimental_conditions[yaml_file]['temperatures']
                    time_shift = self.original_experimental_conditions[yaml_file]['time_shift']
                    press = self.original_experimental_conditions[yaml_file]['pressures']
                    conditions = self.original_experimental_conditions[yaml_file]['conditions']

                    
                    
                elif re.match('[Jj][Ss][Rr]', self.original_experimental_conditions[yaml_file]['simulationType']) or re.match('[Jj]et[- ][Ss]tirred[- ][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']):
                    temp = self.original_experimental_conditions[yaml_file]['temperatures']
                    res = self.original_experimental_conditions[yaml_file]['residence_time']
                    press = self.original_experimental_conditions[yaml_file]['pressure']
                    conditions = self.original_experimental_conditions[yaml_file]['conditions']
                    
                elif re.match('[Ff]low[- ][Rr]eactor', self.original_experimental_conditions[yaml_file]['simulationType']) or re.match('[Ss]pecies[ -][Pp]rofile',self.original_experimental_conditions[yaml_file]['experimentType']):
                    
                    temp = self.original_experimental_conditions[yaml_file]['temperatures']
                    press = self.original_experimental_conditions[yaml_file]['pressure']
                    conditions = self.original_experimental_conditions[yaml_file]['conditions']
                    time_shifts = self.original_experimental_conditions[yaml_file]['timeShiftOriginal']

                #mole_fractions = self.original_experimental_conditions[yaml_file]['MoleFractions']
                
                print('__________________________________________________________________________')
                print('loop:',loop_counter)
                print(temp)
                print(press)
                print(conditions)
                if re.match('[Jj][Ss][Rr]', self.original_experimental_conditions[yaml_file]['simulationType']):
                    print(res)
                print('__________________________________________________________________________')
                if re.match('[Bb]atch[ -][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']) and re.match('[Ss]pecies[- ][Pp]rofile',self.original_experimental_conditions[yaml_file]['experimentType']):
                    updatedTemp = np.exp(physical_observables_updates_list[yaml_file]['T_experiment_'+str(yaml_file)]) * temp
                    updatedTemp = round(updatedTemp,9)
                    #updatedTimeShift =  (np.exp(physical_observables_updates_list[yaml_file]['Time_shift_experiment_'+str(yaml_file)]) * experiment_dict_list[yaml_file]['average_time']) - experiment_dict_list[yaml_file]['average_time']
                    updatedPress = np.exp(physical_observables_updates_list[yaml_file]['P_experiment_'+str(yaml_file)]) * press
                    updatedPress = round(updatedPress,9)
                    updatedTimeShift = physical_observables_updates_list[yaml_file]['Time_shift_experiment_'+str(yaml_file)] + time_shift

                    updatedTimeShift = round(updatedTimeShift,9)
                    species_to_loop =  experiment_dict_list[yaml_file]['uncertainty']['species_relative_uncertainty']['species']
                    dilluant = ['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']
                    updated_mole_fractions = {}
                    count = 0
                    for specie in species_to_loop:
                        if specie in dilluant:
                            continue
                        updated = np.exp(physical_observables_updates_list[yaml_file]['X_'+str(count)+'_experiment_'+str(yaml_file)])*conditions[specie]
                        updated = round(updated,9)
                        updated_mole_fractions[specie] = updated
                        count+=1
    
                    for specie in species_to_loop:
                        if specie in dilluant:
                            updated_mole_fractions[specie] = conditions[specie]
                    
                    updated_mole_fraction_list = []
                    for specie in species_to_loop:
                        updated_mole_fraction_list.append(updated_mole_fractions[specie])
                    
                elif re.match('[Jj][Ss][Rr]', self.original_experimental_conditions[yaml_file]['simulationType']) or  re.match('[Jj]et[- ][Ss]tirred[- ][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']):
                    updatedTemp=[]
                    for i,T in enumerate(temp):
                        updatedTemp.append(float(round(np.exp(physical_observables_updates_list[yaml_file]['T'+str(i+1)+'_experiment_'+str(yaml_file)]) * T,9)))
                    updatedResTime=np.exp(physical_observables_updates_list[yaml_file]['R_experiment_'+str(yaml_file)])*res
                    updatedResTime=round(updatedResTime,9)
                    updatedPress = np.exp(physical_observables_updates_list[yaml_file]['P_experiment_'+str(yaml_file)]) * press
                    updatedPress = round(updatedPress,9)
                    species_to_loop =  experiment_dict_list[yaml_file]['uncertainty']['species_relative_uncertainty']['species']
                    dilluant = ['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']
                    updated_mole_fractions = {}
                    count = 0
                    for specie in species_to_loop:
                        if specie in dilluant:
                            continue
                        updated = np.exp(physical_observables_updates_list[yaml_file]['X_'+str(count)+'_experiment_'+str(yaml_file)])*conditions[specie]
                        updated = round(updated,9)
                        updated_mole_fractions[specie] = updated
                        count+=1
    
                    for specie in species_to_loop:
                        if specie in dilluant:
                            updated_mole_fractions[specie] = conditions[specie]
                    
                    updated_mole_fraction_list = []
                    for specie in species_to_loop:
                        updated_mole_fraction_list.append(updated_mole_fractions[specie])
                elif re.match('[Ff]low[ -][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']) and re.match('[Ss]pecies[- ][Pp]rofile',self.original_experimental_conditions[yaml_file]['experimentType']):
                    updatedTemp=[]
                    for i,T in enumerate(temp):
                        updatedTemp.append(float(round(np.exp(physical_observables_updates_list[yaml_file]['T'+str(i+1)+'_experiment_'+str(yaml_file)]) * T,9)))
                    updatedPress = np.exp(physical_observables_updates_list[yaml_file]['P_experiment_'+str(yaml_file)]) * press
                    updatedPress = round(updatedPress,9)
                    
                    updatedTimeShift = []
                    for i,time in enumerate(time_shifts):
                        updatedTimeShift.append(float(round((physical_observables_updates_list[yaml_file]['Time_Shift'+str(i+1)+'_experiment_'+str(yaml_file)] + time),9)))
                    
                    species_to_loop =  experiment_dict_list[yaml_file]['uncertainty']['species_relative_uncertainty']['species']
                    dilluant = ['Ar','AR','ar','HE','He','he','Kr','KR','kr','Xe','XE','xe','NE','Ne','ne']
                    updated_mole_fractions = {}
                    count = 0
                    for specie in species_to_loop:
                        if specie in dilluant:
                            continue
                        updated = np.exp(physical_observables_updates_list[yaml_file]['X_'+str(count)+'_experiment_'+str(yaml_file)])*conditions[specie]
                        updated = round(updated,9)
                        updated_mole_fractions[specie] = updated
                        count+=1
    
                    for specie in species_to_loop:
                        if specie in dilluant:
                            updated_mole_fractions[specie] = conditions[specie]
                    
                    updated_mole_fraction_list = []
                    for specie in species_to_loop:
                        updated_mole_fraction_list.append(updated_mole_fractions[specie])                    
                elif re.match('[Bb]atch[ -][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']) and re.match('[Ii]gnition[- ][Dd]elay',self.original_experimental_conditions[yaml_file]['experimentType']):
                    updatedTemp=[]
                    diluent=[]   
                    if 'Diluent' in experiment_dict_list[yaml_file]['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in experiment_dict_list[yaml_file]['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                        diluent = experiment_dict_list[yaml_file]['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                    for i,T in enumerate(temp):
                        updatedTemp.append(float(round(np.exp(physical_observables_updates_list[yaml_file]['T'+str(i+1)+'_experiment_'+str(yaml_file)]) * T,9)))
                    updatedPress=[]
                    for i,P in enumerate(press):
                        updatedPress.append(float(round(np.exp(physical_observables_updates_list[yaml_file]['P'+str(i+1)+'_experiment_'+str(yaml_file)]) * P,9)))
                    updated_mole_fractions = {}
                    #print(physical_observables_updates_list[yaml_file])
                    counter=0
                    for i,species in enumerate(self.original_experimental_conditions[yaml_file]['conditions'].keys()):
                        temp_spec_list=[]
                        if species in diluent:
                            continue
                        else:   
                            for j,value in enumerate(self.original_experimental_conditions[yaml_file]['conditions'][species]):
                                temp_spec_list.append(np.exp(physical_observables_updates_list[yaml_file]['X'+str(counter+1)+'_cond'+str(j)+'_'+species+'_experiment_'+str(yaml_file)])*self.original_experimental_conditions[yaml_file]['conditions'][species][j])
                                temp_spec_list[-1]=float(round(temp_spec_list[-1],9))
                            counter=counter+1
                        updated_mole_fractions[species]=copy.deepcopy(temp_spec_list)
                    updatedTimeShift = physical_observables_updates_list[yaml_file]['Time_shift_experiment_'+str(yaml_file)] + time_shift

                    updatedTimeShift = float(round(updatedTimeShift,9))
                elif re.match('[Rr][Cc][Mm]',self.original_experimental_conditions[yaml_file]['simulationType']) and re.match('[Ii]gnition[- ][Dd]elay',self.original_experimental_conditions[yaml_file]['experimentType']):
                    updatedTemp=[]
                    diluent=[]   
                    if 'Diluent' in experiment_dict_list[yaml_file]['uncertainty']['species_relative_uncertainty']['type_dict'].keys() or 'diluent' in experiment_dict_list[yaml_file]['uncertainty']['species_relative_uncertainty']['type_dict'].keys():
                        diluent = experiment_dict_list[yaml_file]['uncertainty']['species_relative_uncertainty']['type_dict']['diluent']
                    for i,T in enumerate(temp):
                        updatedTemp.append(float(round(np.exp(physical_observables_updates_list[yaml_file]['T'+str(i+1)+'_experiment_'+str(yaml_file)]) * T,9)))
                    updatedPress=[]
                    for i,P in enumerate(press):
                        updatedPress.append(float(round(np.exp(physical_observables_updates_list[yaml_file]['P'+str(i+1)+'_experiment_'+str(yaml_file)]) * P,9)))
                    updated_mole_fractions = {}
                    #print(physical_observables_updates_list[yaml_file])
                    counter=0
                    for i,species in enumerate(self.original_experimental_conditions[yaml_file]['conditions'].keys()):
                        temp_spec_list=[]
                        if species in diluent:
                            continue
                        else:   
                            for j,value in enumerate(self.original_experimental_conditions[yaml_file]['conditions'][species]):
                                temp_spec_list.append(np.exp(physical_observables_updates_list[yaml_file]['X'+str(counter+1)+'_cond'+str(j)+'_'+species+'_experiment_'+str(yaml_file)])*self.original_experimental_conditions[yaml_file]['conditions'][species][j])
                                temp_spec_list[-1]=float(round(temp_spec_list[-1],9))
                            counter=counter+1
                        updated_mole_fractions[species]=copy.deepcopy(temp_spec_list)
                    updatedTimeShift = physical_observables_updates_list[yaml_file]['Time_shift_experiment_'+str(yaml_file)] + time_shift

                    updatedTimeShift = float(round(updatedTimeShift,9))                        
                
 # starting to do file updates here 
               
                with open(new_file_name) as f:
                    config2 = yaml.safe_load(f)
                
                
                if re.match('[Bb]atch[ -][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']) and re.match('[Ss]pecies[- ][Pp]rofile',self.original_experimental_conditions[yaml_file]['experimentType']):
                    config2['common-properties']['temperature']['value']=float(updatedTemp)
                    config2['common-properties']['time-shift']['value']=float(updatedTimeShift)
                    config2['common-properties']['pressure']['value']=float(updatedPress)
                    for i,moleFraction in enumerate(updated_mole_fraction_list):
                        config2['common-properties']['composition'][i]['mole-fraction']=float(moleFraction)
                elif re.match('[Jj][Ss][Rr]', self.original_experimental_conditions[yaml_file]['simulationType']) or re.match('[Jj]et[- ][Ss]tirred[- ][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']):    
                    config2['common-properties']['temperature']['value-list']=updatedTemp
                    config2['apparatus']['residence-time']['value']=float(updatedResTime)
                    config2['common-properties']['pressure']['value']=float(updatedPress)
                    for i,moleFraction in enumerate(updated_mole_fraction_list):
                        config2['common-properties']['composition'][i]['mole-fraction']=float(moleFraction)
                    
                elif re.match('[Ss]pecies[- ][Pp]rofile', self.original_experimental_conditions[yaml_file]['simulationType']) or re.match('[Ff]low[ -][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']):    
                    config2['common-properties']['temperature']['value-list']=updatedTemp
                    config2['common-properties']['pressure']['value']=float(updatedPress)
                    config2['common-properties']['time-shift']['value-list']=updatedTimeShift
                    for i,moleFraction in enumerate(updated_mole_fraction_list):
                        config2['common-properties']['composition'][i]['mole-fraction']=float(moleFraction)
                
                elif re.match('[Bb]atch[ -][Rr]eactor',self.original_experimental_conditions[yaml_file]['simulationType']) and re.match('[Ii]gnition[- ][Dd]elay',self.original_experimental_conditions[yaml_file]['experimentType']):
                    config2['common-properties']['temperature']['value-list']=updatedTemp
                    config2['common-properties']['pressure']['value-list']=updatedPress
                    
                    for j,group in enumerate(config2['common-properties']['composition']):
                       #print(group['mixture'])
                       for n,obj in enumerate(group['mixture']):
                           if obj['name'] not in diluent:
                               obj['mole-fraction']['value-list']=updated_mole_fractions[obj['name']]
                           #print(updated_mole_fractions[obj['name']])
                    #config2['common-properties']['composition'][i]['mole-fraction']=updated_mole_fractions[species]
                    config2['common-properties']['time-shift']['value']=float(updatedTimeShift)

                elif re.match('[Rr][Cc][Mm]',self.original_experimental_conditions[yaml_file]['simulationType']) and re.match('[Ii]gnition[- ][Dd]elay',self.original_experimental_conditions[yaml_file]['experimentType']):
                    config2['common-properties']['temperature']['value-list']=updatedTemp
                    config2['common-properties']['pressure']['value-list']=updatedPress
                    
                    for j,group in enumerate(config2['common-properties']['composition']):
                       #print(group['mixture'])
                       for n,obj in enumerate(group['mixture']):
                           if obj['name'] not in diluent:
                               obj['mole-fraction']['value-list']=updated_mole_fractions[obj['name']]
                           #print(updated_mole_fractions[obj['name']])
                    #config2['common-properties']['composition'][i]['mole-fraction']=updated_mole_fractions[species]
                    config2['common-properties']['time-shift']['value']=float(updatedTimeShift)                
                    
                with open(new_file_name,'w') as f:
                    yaml.safe_dump(config2, f,default_flow_style=False)
                    
  # make a list of updated yaml files here that we return from this and then use these names       
        if loop_counter ==0:  
            return updated_file_name_list
        else:
            return file_name_list
    
    def absorption_file_updates(self,file_name_list,
                          parsed_yaml_list,
                          experiment_dict_list,
                          absorption_observables_updates_dict,
                          loop_counter=0):
        '''
        This function updates the absorption physical model parameters stored
        inside the absorption specific yaml files.
        
        This function takes the list of yaml files names, the organized 
        parsed yaml file dictonaries list, the experiment dictonaries list,
        the physical observables updates list, and the loop counter as 
        arguments and returns the file name list of updated files.         

        Parameters
        ----------
        file_name_list : list
            Nested list of yaml file names to be updated..
        parsed_yaml_list : list
            Nested list (same order as the file name list) of organized
            and parsed yaml file dictonaries.
        experiment_dict_list : list
            List of dictonaries pertaining to the simulations run in the 
            optimization that contain relevent information. The list is in the
            same order as the file name list.
        absorption_observables_updates_dict : dictionary
            A dictionary that contains the update (delta x) values for
            absorption coefficients.
        loop_counter : int, optional
            Integer tht keeps track of what iteration the optimization is on.
            If this value is zero copies of the original yaml files will
            be made, and the copy will be updated instead of the original.
            The default is 0.
        Returns
        -------
        file_name_list: list
            List of file names that have been updated.

        '''
        
        
        
        
        #change is happening somewhere after here 
        for yaml_file in range(len(file_name_list)):
            if len(file_name_list[yaml_file])<2:
                continue
            
            if loop_counter ==0:
                new_absorption_file_name = self.yaml_file_copy(file_name_list[yaml_file][1])
                file_name_list[yaml_file][1] = new_absorption_file_name
                
                
            else:
                new_absorption_file_name = file_name_list[yaml_file][1]
                
            
            coupledCoefficients = self.original_experimental_conditions[yaml_file]['coupledCoefficients']
            coupledCoefficentsUpdated = copy.deepcopy(coupledCoefficients)
            ########changes somewhere down there
            
            for species in range(len(coupledCoefficients)):
                for wavelength in range(len(coupledCoefficients[species])):
                    lst = list(coupledCoefficients[species][wavelength])
                    tup = tuple(lst)

                    temp = []
                    for i,values in enumerate(absorption_observables_updates_dict[tup]):
                        
                        temp.append(np.exp(values)*lst[i])

                    coupledCoefficentsUpdated[species][wavelength] = tuple(temp)
                    
            combinationOfNewParameters = list(map(list, list(zip(*(list(map(list, list(zip(*x)))) for x in coupledCoefficentsUpdated)))))
            parameterOnesUpdated = combinationOfNewParameters[0]
             

            for x in range(len(parameterOnesUpdated)):
                for y in range(len(parameterOnesUpdated[x])):
                    parameterOnesUpdated[x][y] = round(parameterOnesUpdated[x][y], 8)

        

            parameterTwosUpdated = combinationOfNewParameters[1]
            for x in range(len(parameterTwosUpdated)):
                for y in range(len(parameterTwosUpdated[x])):
                    parameterTwosUpdated[x][y] = round(parameterTwosUpdated[x][y], 8)
            

            with open(new_absorption_file_name)as f:
                config3 = yaml.safe_load(f)
            
            
            
            for parameterOne in range(len(config3['Absorption-coefficients'])):
                for wavelength in range(len(config3['Absorption-coefficients'][parameterOne]['wave-lengths'])):        
                    config3['Absorption-coefficients'][parameterOne]['wave-lengths'][wavelength]['parameter-one']['value'] = float(parameterOnesUpdated[parameterOne][0])

        
     

            for parameterTwo in range(len(config3['Absorption-coefficients'])):
                for wavelength in range(len(config3['Absorption-coefficients'][parameterTwo]['wave-lengths'])):
                    config3['Absorption-coefficients'][parameterTwo]['wave-lengths'][wavelength]['parameter-two']['value'] = float(parameterTwosUpdated[parameterTwo][0])
            
            with open(new_absorption_file_name,'w') as f:
                yaml.safe_dump(config3, f,default_flow_style=False)
        
        return file_name_list