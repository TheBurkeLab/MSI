from ... import simulations as sim
import numpy as np
import pandas as pd
import cantera as ct

class Absorb:
    def __init__(self):
        self.saved_abs_data = []
        self.saved_perturb_data = [] 
        self.timeHistoryInterpToAbsorbaceExperiment = None
    def perturb_abs_coef(self,del_param:float,simulation:sim.instruments.batch_reactor.batchReactor,
                    absorb:dict,pathlength:float,
                    absorbance_csv_files:list=[],
                    absorbance_csv_wavelengths:list=[],
                    summed_data = None):

        #modify absorb dict, modify each coef
        #remember to put it back
        #essentially modify the coupled coefficient before calculating the absorbance data
        coupled_coefficients = self.couple_parameters(absorb) 
        #loop over the coupled_coefficients, replace, then put back
        self.saved_perturb_data.clear()
        species = [species['species'] for species in absorb['Absorption-coefficients']]
        species_and_coupled_coefficients = dict(list(zip(species,coupled_coefficients)))
        species_and_wavelengths = dict(list(zip(species, self.get_wavelengths(absorb))))
        species_and_functional_form = dict(list(zip(species,self.get_functional(absorb)))) 
        for species in list(species_and_coupled_coefficients):
            index = 0
            for i in range(0,len(species_and_coupled_coefficients[species])):
                orig_cc = species_and_coupled_coefficients[species][i]
                if orig_cc[0] == 0 and orig_cc[1] == 0:
                    pass
               # print(orig_cc[0],'this is cc0')
                if orig_cc[0] != 0:
                    cc = (orig_cc[0]+orig_cc[0]*del_param,orig_cc[1])
                    species_and_coupled_coefficients[species][i] = cc
                    changed_data = self.get_abs_data(simulation,
                                                     absorb,
                                                     pathlength,
                                                     kinetic_sens = 0,
                                                     pert_spec_coef = species_and_coupled_coefficients)
                    #changed_data is a dict, keys are wavelengths, values are absorbances
                    if summed_data is None:
                        self.saved_perturb_data.append([(species,species_and_wavelengths[species][index],cc),
                                                        changed_data])
                    else:                                
                        self.saved_perturb_data.append([(species,species_and_wavelengths[species][index],cc),
                                                        self.ln_abs(changed_data,summed_data,dk=orig_cc[0]*del_param)])
                        #print(changed_data)
                # start checking data somewhere around here
                if orig_cc[1] != 0:
                    #print(orig_cc[1],'this is cc1')
                    if species_and_functional_form[species][i] != 'F':
                        #print('in here')
                        cc = (orig_cc[0],orig_cc[1]+orig_cc[1]*del_param)

                        species_and_coupled_coefficients[species][i] = cc
                        #print(cc)
                        changed_data = self.get_abs_data(simulation,
                                                         absorb,
                                                         pathlength,
                                                         kinetic_sens = 0,
                                                         pert_spec_coef = species_and_coupled_coefficients)
                        
                        
                        if summed_data is None:
                            
                            self.saved_perturb_data.append([(species,species_and_wavelengths[species][index],cc),
                                                            changed_data])
                        else:  
                            #print(changed_data,summed_data,orig_cc[1]*del_param)       
                            #test = self.ln_abs(changed_data,summed_data,dk=orig_cc[1]*del_param)
                            #print(test)
                            self.saved_perturb_data.append([(species,species_and_wavelengths[species][index],cc),
                                                            self.ln_abs(changed_data,summed_data,dk=orig_cc[1]*del_param)])
                    else:
                        #cc = (orig_cc[0],orig_cc[1] + .01)
                        cc = (orig_cc[0],orig_cc[1]+orig_cc[1]*.01)
                        changed_data = self.get_abs_data(simulation,
                                                         absorb,
                                                         pathlength,
                                                         kinetic_sens = 0,
                                                         pert_spec_coef = species_and_coupled_coefficients)
                        if summed_data is None:
                            self.saved_perturb_data.append([(species,species_and_wavelengths[species][index],cc),
                                                            changed_data])
                        else:                                
                            self.saved_perturb_data.append([(species,species_and_wavelengths[species][index],cc),
                                                            self.ln_abs(changed_data,summed_data,dk=.01)])

                species_and_coupled_coefficients[species][i] = orig_cc
                index += 1
                
        return self.saved_perturb_data
    
    def superimpose_batch_reactor(self,simulation:sim.instruments.batch_reactor.batchReactor,
                               absorb:dict,pathlength:float,
                               kinetic_sens=0):
        '''Input:
            time_history: time history from some run used to get temperature matrix
            absorb: dict of yaml parsed absorbance data ie yaml_parser.parse_batch_reactor_obj was run
            pathLenth: diameter of shock tube, integer
            absorbanceCsvWavelengths: list of wavelengths absorbances were measured at, in nm
        '''
        
        
        abs_data = self.get_abs_data(simulation,
                                absorb,
                                pathlength,
                                kinetic_sens)
        self.saved_abs_data.append(abs_data)
        return abs_data 
    
    def absorb_phys_sensitivities(self,simulation:sim.instruments.batch_reactor.batchReactor,
                                  summed_data:dict,
                                  absorb:dict,
                                  pathlength:float,
                                  dk:float = .01):
        if len(simulation.timeHistories) < 2:
            print("Error: must have perturbed time histories to interpolate against")
            return -1

        interpolated_times = simulation.interpolate_range(0,len(simulation.timeHistories))
        summed_interp_abs = []
        for interp_time in interpolated_times:
            summed_interp_abs.append(self.get_abs_data(simulation,
                                     absorb,
                                     pathlength,
                                     time_history=interp_time)) 
        
        ln_abs = []
        for i,int_abs_data in enumerate(summed_interp_abs[1:]):
            ln_abs.append(self.ln_abs(int_abs_data,summed_data,index=i+1,sim=simulation)) 
        return ln_abs
    
    def ln_abs(self,changed_data,orig_data,index=None,sim=None,dk=.01):
        temp = []
        ln_dict = {}
        for key in orig_data.keys():
            temp.clear()
            for i in range(0,len(orig_data[key])):
                if sim is not None and index is not None: 
                    temp.append((np.log(changed_data[key][i]) - np.log(orig_data[key][i]))/.01) 
                    #sim.dk[index]
                else:
                    temp.append((np.log(changed_data[key][i]) - np.log(orig_data[key][i]))/.01) 
                    #dk
            ln_dict[key] = temp
        #print(ln_dict)
        return ln_dict
    
    def map_ksens(self,sheet,time_history=None):
        A = sheet
        N = np.zeros(A.shape)
        Ea = np.zeros(A.shape)
        for x,column in enumerate(A.T):
            N[:,x]= np.multiply(column,np.log(self.timeHistories[0]['temperature'])) if time_history is None else np.multiply(column,np.log(time_history['temperature']))
            #to_mult_ea = np.divide(-1,np.multiply(1/ct.gas_constant,self.timeHistories[0]['temperature'])) if time_history is None else np.divide(-1,np.multiply(ct.gas_constant,time_history['temperature']))
            to_mult_ea = np.divide(-1,np.multiply(1,self.timeHistories[0]['temperature'])) if time_history is None else np.divide(-1,np.multiply(1,time_history['temperature']))
            Ea[:,x]= np.multiply(column,to_mult_ea)

        return [A,N,Ea]
    
    def interpolate_experimental(self, simulation, experimental_data, 
                                 original_summed_absorption=None,
                                 abs_kinetic_sens=None,
                                 map_kinetic_sens=1,
                                 abs_phys_sens=None,
                                 abs_coef_sens=None,
                                 time_history=None):
        if time_history is not None:
            dic = {}
            p_and_t = ['pressure','temperature']
            for df in experimental_data:
                temp = []
                for variable in p_and_t:
                    interpolated_data = np.interp(df['time'],simulation.timeHistories[0]['time'],simulation.timeHistories[0][variable])
                    interpolated_data = interpolated_data.reshape((interpolated_data.shape[0],1))
                    temp.append(interpolated_data)
                temp = np.hstack(temp)    
                temp = pd.DataFrame(temp)
                temp.columns = p_and_t
                dic[int(df.columns.tolist()[1][-3:])] = temp
                self.timeHistoryInterpToAbsorbaceExperiment = dic
            return dic
                
                
        if original_summed_absorption is None and abs_kinetic_sens is None and abs_phys_sens is None and abs_coef_sens is None:
            print("Error: must give something to interpolate")
            return -1
        #each absorbance already against the original time history
        interp_original = {}
        interp_abs_kinetic_sens = {}
        interp_abs_phys_sens = []
        interp_abs_coef_sens = []
        
        #then interpolate that against each experimental file
	#loop over each experimental wavelength
        #and set up the abs_phys interp dict if needed
        if abs_phys_sens is not None:
            for i in range(0,len(abs_phys_sens)):
                interp_abs_phys_sens.append({})
        #fill the abs_coef_sens interp with tuples beforehand
        #to avoid indexes not existing in the list in the later loop
        if abs_coef_sens is not None:
            for i in range(0,len(abs_coef_sens)):
                interp_abs_coef_sens.append([abs_coef_sens[i][0],{}])
        for time_absorb in experimental_data:
            if original_summed_absorption is not None:
                #get the wavelength of the csv file and match to the dict entry
                wavelength = int(time_absorb.columns.values[1].split("_")[1])
                #print(original_summed_absorption)
                data_to_interpolate = original_summed_absorption[wavelength]
                #construct dataframe for the interpolation using the OG time history
                interpolated_data = np.interp(time_absorb['time'],simulation.timeHistories[0]['time'],data_to_interpolate)
                interp_original[wavelength] = interpolated_data
            if abs_kinetic_sens is not None:
                #get the wavelength
                wavelength = int(time_absorb.columns.values[1].split("_")[1])
                if map_kinetic_sens==0:
                    for i,reaction_abs in enumerate(abs_kinetic_sens[wavelength].T): #loop over the rows which is looping time steps
                        #now have a single column ie the reaction at the time steps and k. sens at that time step
                        #now interpolate this
                        data_to_interpolate = reaction_abs
                        interpolated_data = np.interp(time_absorb['time'],simulation.timeHistories[0]['time'],data_to_interpolate)
                        if i == 0:
                            interp_abs_kinetic_sens[wavelength]=np.ndarray(shape=(len(interpolated_data),abs_kinetic_sens[wavelength].shape[1]))
                        #now we have an interpolated column
                        interp_abs_kinetic_sens[wavelength][:,i] = interpolated_data
                else:
                    #get list of sheets
                    list_of_sheets_to_interp = self.map_ksens(abs_kinetic_sens[wavelength],simulation.timeHistories[0])
                    interpolated_sheets = []
                    for sheet in list_of_sheets_to_interp:
                        sheet_cpy = None
                        for i,reaction_abs in enumerate(sheet.T):
                            #now have a single column ie the reaction at the time steps and k. sens at that time step
                            #now interpolate this
                            interpolated_data = np.interp(time_absorb['time'],simulation.timeHistories[0]['time'],reaction_abs)
                            #now we have an interpolated column
                            if i == 0:
                                sheet_cpy = np.ndarray(shape=(len(interpolated_data),sheet.shape[1]))
                            sheet_cpy[:,i] = interpolated_data
                        sheet = sheet_cpy
                        interpolated_sheets.append(sheet)
                    interp_abs_kinetic_sens[wavelength]=interpolated_sheets

            if abs_phys_sens is not None:
                #loop over all the adjusted abs, that have been interpolated already and ln'd
                for i,abs_dict in enumerate(abs_phys_sens): #loops over each adjusted dict
                    #get the wavelength of the csv file and match to the dict entry
                    wavelength = int(time_absorb.columns.values[1].split("_")[1])
                    data_to_interpolate = abs_dict[wavelength]
                    #construct dataframe for the interpolation using the OG time history
                    interpolated_data = np.interp(time_absorb['time'],simulation.timeHistories[0]['time'],data_to_interpolate)
                    interp_abs_phys_sens[i][wavelength] = interpolated_data
            #rember abs_coef_sens is a list of tuples where the last tuple is a abs dict, should always be the ln'd data
            if abs_coef_sens is not None:
                wavelength = int(time_absorb.columns.values[1].split("_")[1])
                #construct dataframe for the interpolation using the OG time history
                for i,abs_entry in enumerate(abs_coef_sens):
                    data_to_interpolate = abs_entry[1][wavelength]
                    interpolated_data = np.interp(time_absorb['time'],simulation.timeHistories[0]['time'],data_to_interpolate)
                    interp_abs_coef_sens[i][1][wavelength] = interpolated_data

        return [interp_original,interp_abs_kinetic_sens,interp_abs_phys_sens,interp_abs_coef_sens]
    
    
    
    def import_experimental_data(self, absorbance_csv_files:list=[]): 
        if len(absorbance_csv_files) == 0:
            print("Error: please give at least 1 experimental absorbance file")
            return -1
        print('Importing shock tube absorbance data the following csv files...') 
        print(absorbance_csv_files)
        
        experimental_data = [pd.read_csv(csv) for csv in absorbance_csv_files]
        experimental_data = [experimental_data[x].dropna(how='any') for x in range(len(experimental_data))]
        experimental_data = [experimental_data[x].apply(pd.to_numeric, errors = 'coerce').dropna() for x in range(len(experimental_data))]
        self.experimental_data = experimental_data
        return experimental_data   
    
    def get_abs_data(self, simulation, absorb,pathlength, kinetic_sens = 0,time_history=None, pert_spec_coef=None):
        
        coupled_coefficients = self.couple_parameters(absorb)
       # print(coupled_coefficients,'inside get abs data function')
        if coupled_coefficients == -1:
            print("Error: could not construct coupled coefficients")
            return -1
        
        species = [species['species'] for species in absorb['Absorption-coefficients']]
        wavelengths = self.get_wavelengths(absorb)
        #functional form takes A B C D, changes which equation used
        functional_form = self.get_functional(absorb) 
        #group data by species for easier manipulation
        species_and_wavelengths = dict(list(zip(species, wavelengths)))
        species_and_coupled_coefficients = dict(list(zip(species,coupled_coefficients))) if pert_spec_coef is None else pert_spec_coef
        #print(species_and_coupled_coefficients,'inside absorption')
        species_and_functional_form = dict(list(zip(species,functional_form))) 
        
        flat_list = [item for sublist in wavelengths for item in sublist]
        flat_list = list(set(flat_list))
        absorbance_species_list = []
        for i, wl in enumerate(flat_list):
            absorbance_species_list.append([])
            for j,specie in enumerate(species):
                if wl in species_and_wavelengths[specie]:
                    absorbance_species_list[i].append(specie)
     
        absorbance_species_wavelengths= []
        for i in range(len(absorbance_species_list)):
            wavelength = flat_list[i]
            for j in range(len(absorbance_species_list[i])):            
                value = absorbance_species_list[i][j]
                index = species_and_wavelengths[value].index(wavelength)
                absorbance_species_wavelengths.append((self.calc_absorb(value,
                                                                  species_and_functional_form[value][index],
                                                                  species_and_coupled_coefficients[value][index],
                                                                  wavelength,
                                                                  pathlength,
                                                                  simulation.timeHistories[0] if time_history is None else time_history),
                                                       value,
                                                       wavelength))

        
        summed_data = {} 
        for x in absorbance_species_wavelengths:
            if x[2] not in summed_data.keys():
                summed_data[x[2]] = x[0]
            else:
                summed_data[x[2]] += x[0]
            
        if kinetic_sens == 0:
            return summed_data
        else:
            return summed_data, self.calc_abs_sens(simulation, 
                                              species_and_wavelengths,
                                              species_and_functional_form,
                                              species_and_coupled_coefficients,
                                              absorbance_species_wavelengths,
                                              pathlength,
                                              summed_data)

    def calc_abs_sens(self,simulation,
                      species_and_wavelengths,
                      species_and_functional_form,
                      species_and_coupled_coefficients,
                      absorbance_species_wavelengths,
                      pathlength,
                      summed_absorption):
        species_and_sensitivities = {}
        for x,i in enumerate(simulation.observables):
            #print(x,i,'these are the observables')
            slice_2d = simulation.kineticSensitivities[:,:,x]
            species_and_sensitivities[i]=slice_2d
        temperature_matrix = simulation.timeHistories[0]['temperature'].values
        pressure_matrix = simulation.timeHistories[0]['pressure'].values

        ind_wl_derivs = {} 
        for species in species_and_sensitivities.keys():
            if species not in species_and_wavelengths.keys():
                continue
            #do epsilon and con calc, then mult
            wavelengths = species_and_wavelengths[species]
            #print(wavelengths)
            for j in range(0,len(wavelengths)):
                wavelength = species_and_wavelengths[species][j] 
                if wavelength not in ind_wl_derivs.keys():

                    net_sum = np.zeros(shape=(simulation.kineticSensitivities.shape[0:2])) #only need 2d info, since sum over observables
                else:
                    net_sum = ind_wl_derivs[wavelength]

                index = species_and_wavelengths[species].index(wavelength)
                #print(index)
                #print(species_and_wavelengths[species])
                cc = species_and_coupled_coefficients[species][index]
               # print(cc)
                ff = species_and_functional_form[species][index]
                #print(ff)
                if ff == 'A':
                    epsilon = ((cc[1]*temperature_matrix) + cc[0])
                if ff == 'B':
                    epsilon = (cc[0]*(1-(np.exp(np.true_divide(-cc[1],temperature_matrix)))))
                if ff == 'C':
                    epsilon = cc[0]                     
                    if type(epsilon)==float or type(epsilon)==int:
                        epsilon = np.ones(shape=temperature_matrix.shape)
                        epsilon *= cc[0]
                #if wavelength == 215: #does this really need to be here, takes care of specific paper case?
                   #epsilon *= 1

                concentration = np.true_divide(1,temperature_matrix.flatten())*pressure_matrix.flatten()
                concentration *= (1/(8.314e6))*simulation.timeHistories[0][species].values.flatten()
                #print(species)
                #mole_fraction = simulation.timeHistories[0][species].values.flatten()
                temp = np.multiply(species_and_sensitivities[species],concentration.reshape((np.shape(concentration)[0],1)))
                #temp = np.multiply(species_and_sensitivities[species],1)
                #print(epsilon)
                net_sum += np.multiply(temp,epsilon.reshape((np.shape(epsilon)[0],1)))
                
                ind_wl_derivs[wavelength]=net_sum
        
        for single_wl in ind_wl_derivs.keys():
            flat_list = np.array(list(summed_absorption[single_wl]))
            for i in range(0,len(flat_list)):
                flat_list[i] = (1/flat_list[i])*pathlength
            for column in ind_wl_derivs[single_wl].T:
                column*=flat_list.flatten()
        
        return ind_wl_derivs


    def calc_absorb(self,species,
                    ff,
                    cc,
                    wavelength,
                    pathlength,
                    time_history):
        
        temperature_matrix = time_history['temperature'].values
        pressure_matrix = time_history['pressure'].values
        if ff == 'A':

            epsilon = ((cc[1]*temperature_matrix) + cc[0])
        if ff == 'B':
            epsilon = (cc[0]*(1-(np.exp(np.true_divide(-cc[1],temperature_matrix)))))
        if ff == 'C':
            epsilon = cc[0] 

            if type(epsilon)==int:
                epsilon = np.ones(shape=temperature_matrix.shape)
                epsilon *= cc[0]

        if wavelength == 215: #does this really need to be here?
           epsilon *= 1
           #multiplying by 1000 to convert from L to cm^3 from the epsilon given in paper 
           #this applies if the units on epsilon are given as they are in kappl paper 
           #must calcuate and pass in reactor volume 
        concentration = np.true_divide(1,temperature_matrix.flatten())*pressure_matrix.flatten()
        concentration *= (1/(8.314e6))*time_history[species].values.flatten()
        
        
        absorb = pathlength*(epsilon*concentration)
        return absorb
     
    def get_wavelengths(self,absorb:dict):
        wavelengths = []  #get the wavelengths
        for sp in range(len(absorb['Absorption-coefficients'])):
            temp = [wl['value'] for wl in absorb['Absorption-coefficients'][sp]['wave-lengths']]
            wavelengths.append(temp)
        self.wavelengths = wavelengths
        return wavelengths

    def couple_parameters(self,absorb:dict()):
        parameter_ones = [] #for the epsilon calculations, there is always a parameter
        for p1 in range(len(absorb['Absorption-coefficients'])):
            temp = [wl['parameter-one']['value'] for wl in absorb['Absorption-coefficients'][p1]['wave-lengths']]
            parameter_ones.append(temp)
        
        #parameter two does not always really exist, but we will always define it in the yaml file to be 0 in that case
        #do this for easy index matching
        parameter_twos = [] 
        for p2 in range(len(absorb['Absorption-coefficients'])):
            temp = [wl['parameter-two']['value'] for wl in absorb['Absorption-coefficients'][p2]['wave-lengths']]
            parameter_twos.append(temp)
        if len(parameter_ones) != len(parameter_twos):
            print("Error: number of parameters do not match, change the yaml file")
            return -1

        coupled_coefficients = [list(zip(parameter_ones[x],parameter_twos[x])) for x in range(len(parameter_ones))]
        
        return coupled_coefficients 


    def get_functional(self,absorb:dict):
        functional_form = []
        for form in range(len(absorb['Absorption-coefficients'])):
            temp = [wl['functional-form'] for wl in absorb['Absorption-coefficients'][form]['wave-lengths']]
            functional_form.append(temp)
        return functional_form
    
    def calculate_time_shift_sensitivity_abs(self,absorabnce_from_model,absorbance_experimental_data,simulation,dk):
        flat_list = [item for sublist in self.wavelengths for item in sublist]
        flat_list = list(set(flat_list))
        wavelengths = flat_list

        one_percent_of_average = 1e-8
        original_time = simulation.timeHistories[0]['time']
        new_time = original_time + one_percent_of_average

        interpolated_against_original_time = []
        for i,wl in enumerate(wavelengths):
            interpolated_original_wl_against_original_time = np.interp(original_time,new_time,absorabnce_from_model[wl])
    
            s1 = pd.Series(interpolated_original_wl_against_original_time,name=wl)
            interpolated_against_original_time.append(s1)
        
        observables_interpolated_against_original_time_df = pd.concat(interpolated_against_original_time,axis=1)


        calculated_sensitivity = []
        for i,wl in enumerate(wavelengths):
                      
           sens = (observables_interpolated_against_original_time_df[wl].apply(np.log).values - np.log(absorabnce_from_model[wl]))/one_percent_of_average
           s1 = pd.Series(sens,name=wl)
           calculated_sensitivity.append(s1)    
        calculated_sensitivity_df = pd.concat(calculated_sensitivity,axis=1)

                
        
        time_shift_sensitivity_dict = {}
        for i,wl in enumerate(wavelengths):
            
            interpolated_sensitivty_per_original_observable = np.interp(absorbance_experimental_data[i]['time'],simulation.timeHistories[0]['time'],calculated_sensitivity_df[wl])
            interpolated_sensitivty_per_original_observable = interpolated_sensitivty_per_original_observable.reshape((interpolated_sensitivty_per_original_observable.shape[0],1))
            
            time_shift_sensitivity_dict[wl] = interpolated_sensitivty_per_original_observable
        
        self.time_shift_sensitivity_dict = time_shift_sensitivity_dict
        return time_shift_sensitivity_dict
    
   

