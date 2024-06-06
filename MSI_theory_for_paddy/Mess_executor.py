# -*- coding: utf-8 -*-
"""
Write Perturbed PAPR-MESS input files.

@author: Lei Lei
"""

import preprocessor, io, os, sys, copy, shutil

class Mess_Executor:
    """Parent class for PAPR-MESS simulations."""

    def __init__(self, input_path, nominal_file, perturb_dict, nominal_dict, abstraction=False):
        self.input_path = input_path
        self.input_name = nominal_file
        self.nominal_model = preprocessor.Preprocessor(input_path + nominal_file)
        self.nominal_model.clean_input()
        self.nominal_model.generate_species_classes(abstraction=abstraction)
        self.perturb_dict = perturb_dict
        self.nominal_dict = nominal_dict
        self.abstraction = abstraction

    def make_directory(self):
        """Declare the related working directories related to a calculation."""
        self.mwd = os.getcwd()    # main directory the contains all the source codes
        self.cwd = self.mwd + '/PAPR-MESS_calculation'  # calculation working directory

    def write_class(self, output_name, model, tar_class):
        """Write a Mess_input class to file based on the order attribute."""
        # define the model and input class to be written
        curr_model = self.__dict__[model]
        curr_class = curr_model.species_classes[tar_class]
        curr_class.Hindered_rotor_correction() # to make hindered rotor axis and symmetry have to be integers, otherwise MESS raises errors
        order = curr_class.__dict__["order"]
        # append a sepcific part into the input file
        fhand = io.open(output_name, 'a')
        spec_list = ['Frequencies', 'ElectronicLevels', 'WellDepth', 'FourierExpansion']

        for k in order:
            if type(curr_class.__dict__[k]) is list:
                if k.split(' ')[0] in spec_list:
                    unit = curr_class.__dict__[k][-1]
                    value = curr_class.__dict__[k][0].split(',')
                    for x in value:
                        line = "%s%s \t\t\t\t %s\n" %(k.split(' ')[0], unit, x.strip('[]'))
                        fhand.write(line)
                elif k.split(' ')[0] == 'HotEnergies':
                    unit = curr_class.__dict__[k][-1]
                    e_levels = curr_class.__dict__[k][-2]
                    species = curr_class.__dict__[k][-3]
                    line = "%s%s \t\t\t\t %s\n" %(k.split(' ')[0], unit, len(e_levels))
                    fhand.write(line)
                    for _ in e_levels:
                        fhand.write("%s \t\t %s\n" %(species,_))
                elif curr_class.hasunit(k):
                    unit = curr_class.__dict__[k][-1]
                    value = curr_class.__dict__[k][0].replace(',', '  ')
                    line = "%s%s \t\t\t\t %s\n" %(k.split(' ')[0], unit, value.strip('[]'))
                    fhand.write(line)
                else:
                    value = str(curr_class.__dict__[k]).replace(',', '  ')
                    value = value.replace('\'', '  ')
                    temp_key = k.split(' ')[0]
                    if temp_key in ['Group', 'Axis', 'Symmetry', 'MassExpansionSize', 'PotentialExpansionSize', 'HamiltonSizeMin', 'HamiltonSizeMax', 'GridSize']:
                        temp_value = curr_class.__dict__[k]
                        line = "%s \t\t\t\t %s\n" %(temp_key, '  '.join([str(int(_)) for _ in temp_value]).replace('\'', '  '))
                    else:
                        line = "%s \t\t\t\t %s\n" %(temp_key, value.strip('[]'))
                    fhand.write(line)

            elif type(curr_class.__dict__[k]) is dict:
                if k.split(' ')[0] not in spec_list:
                    temp = copy.deepcopy(curr_class.__dict__[k])
                    unit = temp.pop('unit')
                    order = temp.pop('order')
                    value = temp
                    line = "%s%s \t\t\t\t %s\n" %(k.split(' ')[0], unit, len(value))
                    fhand.write(line)
                    for atom in order:
                        geo = str(value[atom]).replace(',', '    ')
                        line = "%s \t\t\t\t %s\n" %(atom.split(' ')[1], geo.strip('[]'))
                        fhand.write(line)
                else:
                    temp = copy.deepcopy(curr_class.__dict__[k])
                    unit = temp.pop('unit')
                    value = temp['value']
                    line = "%s%s \t\t\t\t %s\n" %(k.split(' ')[0], unit, len(value))
                    fhand.write(line)
                    for x in value:
                        tup = type(x) is tuple
                        value_x = str(x).replace(',', '   ')
                        fhand.write(value_x.strip('[]()') + '    ' + '\n' * tup)
                    fhand.write('\n' * (k.split(' ')[0] == 'Frequencies'))
        fhand.write('\n')
        fhand.close()

    def write_file(self, output_name, model, class_order):
        """Write the entire PAPR-MESS input file."""
        # initialize the file
        fhand = io.open(output_name, 'w')
        fhand.close()
        # write classes in the given order
        for tar_class in class_order:
            self.write_class(output_name, model, tar_class)

    def generate_perturbed_files(self, pert_nom, fit_rate=True):
        """Generate perturbed files based on the perturbation dictionary specified."""
        # create a working directory for each species and active parameter
        if pert_nom == 'perturbed':
            per_directory = self.twd + '/perturbation'
            if not os.path.exists(per_directory):
                os.makedirs(per_directory)
            os.chdir(per_directory)
            self.pwd = os.getcwd()
            pertb_dict = self.perturb_dict

        elif pert_nom == 'nominal':
            nom_directory = self.twd + '/nominal'
            if not os.path.exists(nom_directory):
                os.makedirs(nom_directory)
            os.chdir(nom_directory)
            self.nwd = os.getcwd()
            pertb_dict = self.nominal_dict

        # generate top layer bash file
        if os.path.exists('top_bash.sh'):
            top_bash = io.open('top_bash.sh', 'a')
        else:
            top_bash = io.open('top_bash.sh', 'w')

        key_ls = list(pertb_dict.keys())
        X_ls = []
        for key in key_ls:
            species, X = key.split("_")[:2]       # species and parameter to perturb
            Temp_ls = pertb_dict[key][0]          # should be a list
            Pres_ls = pertb_dict[key][1]          # should be a list
            perturb_diff = pertb_dict[key][2]     # should be a number

            if species =='colrel':
                species ='col_rel'



            if X in list(self.pertb.keys()):
                self.pertb[X].append(perturb_diff)
            else:
                self.pertb[X] = [perturb_diff]

            if pert_nom == 'perturbed':
                self.pert_P = Pres_ls
                dwd = self.pwd
            elif pert_nom == 'nominal':
                self.nom_P = Pres_ls
                dwd = self.nwd

            active_dir = dwd + '/' + X
            # generate sub-directory for each perturbation
            if not os.path.exists(active_dir):
                os.makedirs(active_dir)
            if X not in X_ls:
                top_bash.write('cd %s \n' %active_dir)
                top_bash.write('sh perturb_bash.sh \n')
                top_bash.write('rm perturb_bash.sh \n')
                if fit_rate:
                    top_bash.write('sh rate_bash.sh \n')
                    top_bash.write('rm MESS_rate_extractor.py \n')
                    top_bash.write('rm rate_bash.sh \n')
            os.chdir(active_dir)

            # copy files
            if len(self.nominal_model.files_to_copy) > 0:
                for f in self.nominal_model.files_to_copy:
                    try:
                        shutil.copy('%s/%s' %(self.input_path, f), './')
                    except IOError:
                        sys.exit('Error: No such file in the directory: %s' %f)

            # write the bash file for running perturbed input files and extract rate constants
            if os.path.isfile('perturb_bash.sh'):
                Mess_bash = io.open('perturb_bash.sh', 'a')
            else:
                Mess_bash = io.open('perturb_bash.sh', 'w')

            # get species to be perturbated and change the simulation conditionsS



            curr_species = copy.deepcopy(self.nominal_model.species_classes[species])
            condition_class = self.nominal_model.species_classes['condition']
            #print(dir(condition_class),'THIS IS CONDITIONS CLASS')
            # change calculation conditions
            condition_class.change_Temperature(Temp_ls)
            if not self.abstraction:
                self.Punit = condition_class.change_Pressure(Pres_ls)
            else:
                self.Punit = '--'

            # get the attributes to be perturbated
            #print(X)
            #print(species)
            active_para = curr_species.partial_match_key(X)
            if active_para == False:
                sys.exit("No such attribute defined in the PAPR-MESS input.")
            # modified for Variational Barrier model
            elif "Variational" in active_para:
                target = active_para.split('_')[-1]
                curr_species.change_Variational(target, perturb_diff)
            # for RRHO model
            elif "Energy" in active_para:
                curr_species.change_Energy(perturb_diff)
            elif "Frequencies" in active_para:
                curr_species.change_Vib_Frequency(perturb_diff)
            elif "SymmetryFactor" in active_para:
                curr_species.change_Symmetry(perturb_diff)
            elif "ImaginaryFrequency" in active_para:
                curr_species.change_Img_Frequency(perturb_diff)
            elif "NejScale" in active_para:
                curr_species.change_Nej_file(perturb_diff)
            elif "FourierExpansion" in active_para:
                curr_species.change_Hind_rotor(perturb_diff)
            elif 'PowerOne' in active_para:
                curr_species.change_power(perturb_diff,power_one_or_power_two=0)
            elif 'PowerTwo' in active_para:
                curr_species.change_power(perturb_diff,power_one_or_power_two=1)    
            elif 'FactorOne' in active_para:
                curr_species.change_exponential_factor(perturb_diff,factor_one_or_factor_two=0)
            elif 'FactorTwo' in active_para:
                curr_species.change_exponential_factor(perturb_diff,factor_one_or_factor_two=1)
            elif 'Fraction' in active_para:
                curr_species.change_fraction(perturb_diff)
            elif 'Sigmas' in active_para:
                curr_species.change_sigmas(perturb_diff)
            elif 'Epsilons' in active_para:
                curr_species.change_epsilons(perturb_diff)


            self.perturb_model = copy.deepcopy(self.nominal_model)
            self.perturb_model.species_classes[species] = curr_species
            # generate the perturbed input files
            output_file = self.input_name.split('.')[0] + '_' + key + '_' + str(perturb_diff) + '.inp'
            self.write_file(output_file, "perturb_model", self.nominal_model.section_order)
            # add the perturbed files into bash file
            Mess_bash.write("echo '--running %s' \n" %key)
            # different runs for regular MESS and abstraction calculations
            if not self.abstraction:
                Mess_bash.write('mess %s \n' %output_file)
            else:
                Mess_bash.write('abstraction %s \n' %output_file)
            Mess_bash.close()
            if X not in X_ls:
                top_bash.write('cd %s \n' %self.twd)
            os.chdir(self.twd)
            X_ls.append(X)

        top_bash.close()
        if pert_nom == 'perturbed':
            print(("Generating perturbed PAPR-MESS input files for system %s ..." %self.input_name.split(".")[0]))
        elif pert_nom == 'nominal':
            print(("Generating nominal PAPR-MESS input files for system %s ..." %self.input_name.split(".")[0]))

    def execute_MESS(self):
        """Execute the nominal and generated perturbed PAPR-MESS files."""
        os.chdir(self.nwd)
        try:
            print(("Running nominal PAPR-MESS for system %s ..." %self.input_name.split(".")[0]))
            self.get_rate_constants(self.R_P_list, 'nominal')
            os.system('sh top_bash.sh')
            os.system('rm top_bash.sh')
        except:
            sys.exit('Wrong input file or directory.')
        os.chdir(self.pwd)
        try:
            print(("Running perturbed PAPR-MESS for system %s ..." %self.input_name.split(".")[0]))
            self.get_rate_constants(self.R_P_list, 'perturbed')
            os.system('sh top_bash.sh')
            os.system('rm top_bash.sh')
            print(("Extracting channel-specific rate constants for system %s ..." %self.input_name.split(".")[0]))
        except:
            sys.exit('Wrong input file or directory.')

    def new_trial_directory(self):
        """Create directory for a new calculation."""
        self.make_directory()

        # change path to the PAPR-MESS calculation directory
        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)
        os.system('cp ' + self.input_path + self.input_name + ' ' + self.cwd)
        os.chdir(self.cwd)

        # create the system working directory
        system_directory = os.getcwd() + '/%s' %self.input_name.split('.')[0]
        if not os.path.exists(system_directory):
            os.makedirs(system_directory)
        os.chdir(system_directory)
        self.swd = os.getcwd()

        # create a working directory for each calculation
        n = 1
        while True:
            trial_directory = self.swd + '/calculation_%s' %n
            if not os.path.exists(trial_directory):
                os.makedirs(trial_directory)
                break
            n += 1
        self.twd = trial_directory  # trial working directory
        return self.twd

    def new_calculation(self, R_P_list, twd, run_MESS=True):
        """Create directories for each systems in the same calculation."""
        self.R_P_list = R_P_list
        self.twd = twd
        self.pertb = {}                       # a dictionary containing the perturbation
        # change path to the trial directory
        os.system('mv ' + self.cwd + '/' + self.input_name + ' ' + self.twd)
        os.chdir(self.twd)
        self.generate_perturbed_files('nominal')
        self.generate_perturbed_files('perturbed')
        # check if both nominal and perturbed files are generated
        if os.path.exists(self.twd + '/nominal') and os.path.exists(self.twd + '/perturbation'):
            os.system('rm ./' + self.input_name)
            if run_MESS:
                # execute PAPR-MESS calculation
                self.execute_MESS()
        else:
            sys.exit("Error: Nominal or perturbed files are not generated.")

        os.chdir(self.mwd)

    def get_channel_rate_constants(self, reactant, product, pert_nom):
        """Get temperature-dependent rate constants at specific perssure for given reactant and product combination."""
        if pert_nom == 'perturbed':
            pertb_dict = self.perturb_dict
            dwd = self.pwd
        elif pert_nom == 'nominal':
            pertb_dict = self.nominal_dict
            dwd = self.nwd
        key_ls = list(pertb_dict.keys())
        cwd = os.getcwd()
        os.chdir(self.mwd)
        for key in key_ls:
            species, X = key.split("_")[:2]
            Pres_ls = pertb_dict[key][1]
            perturb_diff = pertb_dict[key][2]
            active_dir = dwd + '/' + X
            os.system('cp %s/MESS_rate_extractor.py %s' %(self.mwd, active_dir))
            os.chdir(active_dir)
            if os.path.exists('rate_bash.sh'):
                Rate_bash = io.open('rate_bash.sh', 'a')
            else:
                Rate_bash = io.open('rate_bash.sh', 'w')
            # write bash command to get channel specific data for each pressure
            # for abstraction
            if self.abstraction:
                P = '--'
                output_file = self.input_name.split('.')[0] + '_' + key + '_' + str(perturb_diff) + '.out'
                Rate_bash.write('python MESS_rate_extractor.py %s %s %s %s %s %s \n' %(output_file, P, reactant, product, dwd, self.abstraction))
            else:
                for P in Pres_ls[0]:
                    #print(P,key,'WE ARE HERE')
                    P = str(P) + '\ ' + self.Punit
                    output_file = self.input_name.split('.')[0] + '_' + key + '_' + str(perturb_diff) + '.out'
                    Rate_bash.write('python MESS_rate_extractor.py %s %s %s %s %s %s \n' %(output_file, P, reactant, product, dwd, self.abstraction))
            Rate_bash.close()
        os.chdir(cwd)

    def get_rate_constants(self, R_P_list, pert_nom):
        """Output the temperature-dependent rate constants for the specified reactant-product pairs."""
        for x in R_P_list:
            reactant, product = x.split('->')
            self.get_channel_rate_constants(reactant, product, pert_nom)



