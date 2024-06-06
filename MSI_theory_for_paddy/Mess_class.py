# -*- coding: utf-8 -*-
"""

PAPR-MESS Species Classes.

@author: Lei Lei
"""

import sys

class Mess_Input:
    """Define classes of species involved in the PAPR-MESS inputs."""
    # assign name to the class
    def name(self):
        if hasattr(self, 'Well'):
            self.name = self.Well
        elif hasattr(self, 'Bimolecular'):
            self.name = self.Bimolecular
        elif hasattr(self, 'Barrier'):
            self.name = self.Barrier
        elif hasattr(self, 'Model'):
            self.name = self.Model


    # determine if units are attched to the key words
    def hasunit(self, attr):
        """True if unit is associated to the command."""
        value = self.__dict__[attr]
        unit_pool = ['[1/cm]', '[kcal/mol]', '[K]', '[torr]', '[angstrom]', '[amu]', '[au]', '[atm]', '[km/mol]']
        if type(value) is list:
            if value[-1] in unit_pool:
                return True
            else: return False
        elif type(value) is dict:
            if value['unit'] in unit_pool:
                return True
            else: return False
        else: return False

    def partial_match_key(self, keyword):
        """Partial match the key and get the key."""
        key_ls = list(self.__dict__.keys())
        #print(key_ls)
        for key in key_ls:
            if keyword in key:
                return key
            elif 'Nej' in keyword:
                return 'NejScale'
            elif 'HinderRotor' in keyword:
                return 'FourierExpansion'
            elif 'Variational' in keyword:
                target = keyword.split('Variational')[-1]
                return 'Variational_%s' %target
            elif 'PowerOne' in keyword:
                return 'PowerOne'
            elif 'PowerTwo' in keyword:
                return 'PowerTwo'                
            elif 'FactorOne' in keyword:
                return 'FactorOne'
            elif 'FactorTwo' in keyword:
                return 'FactorTwo'
            elif 'Epsilons' in keyword:
                return 'Epsilons'
            elif 'Sigmas' in keyword:
                return 'Sigmas'
            elif 'Fraction' in keyword:
                return 'Fraction'

        return 0

    def perturb_Energy(self, original_eng, percentage_diff):
        """Inputs are original Energy dictionary and percentage change."""

        org_eng, unit = float(original_eng[0]), original_eng[1]
        #print(org_eng,'THIS IS ORIGINAL ENERGY')

        if unit == '[kcal/mol]':
            new_eng = org_eng + percentage_diff / 349.759
            print((percentage_diff / 349.759))
            print((new_eng, 'THIS IS THE NEW ENERGY'))
        elif unit == '[1/cm]':
            new_eng = org_eng + percentage_diff
        else:
            sys.exit("Error: Unrecognizable unit: %s." %unit)
        return new_eng

    def perturb_Frequencies(self, original_fre, percentage_diff):
        """Inputs are a list of original vibrational frequencies and percentage change."""
        temp = []
        for f in original_fre:
            nf = f * (1. + percentage_diff)
            temp.append(nf)
        return temp



    def change_Energy(self, percentage_diff):
        """Change the energy of specific species by defined percentage."""
        key = self.partial_match_key('Energy')
        new_eng = self.perturb_Energy(self.__dict__[key], percentage_diff)
        self.__dict__[key][0] = str(new_eng)



        #print('this is self dict')
        #print(self.__dict)
        # org_eng = float(self.__dict__[key][0])
        # unit = self.__dict__[key][1]
        # if unit == '[kcal/mol]':
        #     new_eng = org_eng + percentage_diff / 349.759
        # elif unit == '[1/cm]':
        #     new_eng = org_eng + percentage_diff
        # else:
        #     sys.exit("Error: Unrecognizable unit: %s." %unit)
        # self.__dict__[key][0] = str(new_eng)

    def change_Vib_Frequency(self, percentage_diff):
        """Scale the vibrational frequencies of specific species by defined percentage."""
        key = 'Frequencies'
        temp = self.perturb_Frequencies(self.__dict__[key]['value'], percentage_diff)
        self.__dict__[key]['value'] = temp

        # org_fre = self.__dict__[key]['value']
        # temp = []
        # for f in org_fre:
        #     nf = f * (1. + percentage_diff)
        #     temp.append(nf)
        # self.__dict__[key]['value'] = temp

    def change_Symmetry(self, percentage_diff):
        """Scale the symmetry factor of specific species by defined percentage."""
        key = 'SymmetryFactor'
        org_sym = self.__dict__[key][0]
        new_sym = org_sym * (1. + percentage_diff)
        self.__dict__[key][0] = new_sym

    def change_Hind_rotor(self, percentage_diff):
        """Scale the Hindered rotor frequencies by the provided percentage."""
        key = 'FourierExpansion'
        #print(self.__dict__)
        org_exp = self.__dict__[key]['value']
        temp = []
        for exp in org_exp:
            n, e = exp
            ne = e * (1. + percentage_diff)
            temp.append((n, ne))
        self.__dict__[key]['value'] = temp

    def Hindered_rotor_correction(self):
        """In PAPR-MESS code, for hindered rotor axis and symmetry have to be integers,
           otherwise it causes error."""
        target_list = ['Axis', 'Symmetry']
        for target in target_list:
            if not hasattr(self, target):
                break
            else:
                temp = []
                for x in self.__dict__[target]:
                    temp.append(int(x))
                self.__dict__[target] = temp

      



class Computation_Cond(Mess_Input):
    """Computational conditions of MESS."""
    def __init__(self):
        self.order = []

    def Pressure_unit(self):
        key = self.partial_match_key('Pressure')
        unit = self.__dict__[key][-1]
        return unit.strip('[]')

    def change_Temperature(self, Temp_list):
        """Change the simulation temperature list."""
        key = 'TemperatureList'
        self.__dict__[key][0] = str(Temp_list)


    def change_Pressure(self, Pres_list):
        """Change the simulation temperature list."""
        key = 'PressureList'
        unit = Pres_list[-1][0]
        self.__dict__[key][1] = unit
        self.__dict__[key][0] = str(Pres_list[0])
        if not unit in ['[atm]', '[torr]']:
            sys.exit("Error: Unrecognizable unit: %s." %unit)
        return unit[1:-1]

    def change_energy_grid(self, energy_grid=100):
        """Cahnge the energy grid (in [1/cm]) for calculation."""
        key = 'EnergyStep'
        if hasattr(self, key):
            self.__dict__[key][0] = str(energy_grid)
        else:
            del(self.__dict__['EnergyStepOverTemperature'])
            self.__dict__[key] = [str(energy_grid), '[1/cm]']
            self.order[self.order.index('EnergyStepOverTemperature')] = 'EnergyStep'

    def hot_reaction(self, species, E_levels):
        """Calculate the hot energy branching fractions."""
        self.__dict__['HotEnergies'] = [str(species), E_levels, '[kcal/mol]']
        self.order.append('HotEnergies')

    def ped(self, output_name):
        """Calculate the PED output."""
        if not hasattr(self, 'PEDOutput'):
            self.__dict__['PEDOutput'] = [output_name]
            self.order.append('PEDOutput')

    def drop_log_output_command(self):
        cmd_ls = ['RateOutput', 'LogOutput', 'PEDSpecies']
        for x in cmd_ls:
            if hasattr(self, x):
                del(self.__dict__[x])
                self.order.pop(self.order.index(x))

# Define collision and relaxation energy transfer
class Collision_Relaxation(Mess_Input):
    """Collisional energy transfer and relaxation energy transfer."""
    def __init__(self):
        self.order = []


    def change_power(self,percentage_diff,power_one_or_power_two=0):
        key = 'Power'
        #print(self.__dict__[key][0])
        temp = self.perturb_power(self.__dict__[key], percentage_diff, power_one_or_power_two=0)
        self.__dict__[key] = temp  
        #print(temp)

    def change_exponential_factor(self,percentage_diff,factor_one_or_factor_two=0):
        key = 'Factor'
        temp = self.perturb_exponentail_factor(self.__dict__[key], percentage_diff,factor_one_or_factor_two=factor_one_or_factor_two)
        #actually don't think i need to speceify the difference here 
        self.__dict__[key][0] = str(temp) 

    def change_epsilons(self,percentage_diff):
        key = 'Epsilons'
        temp = self.perturb_epsilons(self.__dict__[key], percentage_diff)
        self.__dict__[key][0] = str(temp)

    def change_sigmas(self,percentage_diff):
        key = 'Sigmas'
        temp = self.perturb_sigmas(self.__dict__[key], percentage_diff)
        self.__dict__[key][0] = str(temp)

    def change_fraction(self,percentage_diff):
        key = 'Fraction'
        temp = self.perturb_fraction(self.__dict__[key], percentage_diff)
        self.__dict__[key] = temp 

    def perturb_power(self,org_power,percentage_diff,power_one_or_power_two=0):
        if len(org_power) >1 :
            orig_power_list = org_power
            perturbed_power = orig_power_list[power_one_or_power_two] + percentage_diff
            orig_power_list[power_one_or_power_two] = perturbed_power
            return orig_power_list

        else:
            perturbed_power = org_power[0] + percentage_diff
            perturbed_power = [perturbed_power]
            return perturbed_power

    def perturb_exponentail_factor(self,org_exponentail_factor,percentage_diff,factor_one_or_factor_two=0):
        #check if this is a list or not 

        org_exponentail_factor_list, unit = org_exponentail_factor[0], org_exponentail_factor[1]
        if '[' and ']' in org_exponentail_factor_list:
            
            org_exponentail_factor_list, unit = org_exponentail_factor[0], org_exponentail_factor[1]
            org_exponentail_factor_list = org_exponentail_factor_list.strip('[').strip(']').split(',')
            org_exponentail_factor_list = [float(item) for item in org_exponentail_factor_list]
            
            exponentail_factor_being_perturbed = org_exponentail_factor_list[factor_one_or_factor_two]
            if unit == '[kcal/mol]':
                new_exponentail_factor = exponentail_factor_being_perturbed * (1+percentage_diff)

            elif unit == '[1/cm]':
                new_exponentail_factor = exponentail_factor_being_perturbed * (1+percentage_diff)

            else:
                sys.exit("Error: Unrecognizable unit: %s." %unit)

            org_exponentail_factor_list[factor_one_or_factor_two] = new_exponentail_factor

            return org_exponentail_factor_list

        else:
            org_exponentail_factor, unit = float(org_exponentail_factor[0]), org_exponentail_factor[1]
            if unit == '[kcal/mol]':
                #new_exponentail_factor = org_exponentail_factor + percentage_diff / 349.759
                new_exponentail_factor = org_exponentail_factor * (1+percentage_diff)

            elif unit == '[1/cm]':
                #new_exponentail_factor = org_exponentail_factor + percentage_diff
                new_exponentail_factor = org_exponentail_factor * (1+percentage_diff)

            else:
                sys.exit("Error: Unrecognizable unit: %s." %unit)
            
            return new_exponentail_factor

    def perturb_epsilons(self,org_epsilon,percentage_diff):
        #this function is only going to perturb the second epsiolon
        epsilon_list, unit = org_epsilon[0], org_epsilon[1]
        epsilon_list = epsilon_list.strip('[').strip(']').split(',')
        epsilon_list = [float(item) for item in epsilon_list]

        epsilon_being_perturbed = epsilon_list[1]

        if unit == '[kcal/mol]':
            new_epsilon_being_perturbed = epsilon_being_perturbed *(1+percentage_diff)

        elif unit == '[1/cm]':
            new_epsilon_being_perturbed = epsilon_being_perturbed *(1+percentage_diff)



        else:
            sys.exit("Error: Unrecognizable unit: %s." %unit)

        epsilon_list[1] = new_epsilon_being_perturbed
        return epsilon_list

    def perturb_sigmas(self,org_sigma,percentage_diff):
        #this function is only going to perturb the second sigma
        sigma_list, unit = org_sigma[0], org_sigma[1]
        sigma_list = sigma_list.strip('[').strip(']').split(',')
        sigma_list = [float(item) for item in sigma_list]

        sigma_being_perturbed = sigma_list[1]

        if unit == '[angstrom]':
            #new_epsilon_being_perturbed = epsilon_being_perturbed + percentage_diff
            new_sigma_being_perturbed = sigma_being_perturbed *(1+percentage_diff) 

        else:
            sys.exit("Error: Unrecognizable unit: %s." %unit)

        sigma_list[1] = new_sigma_being_perturbed
        return sigma_list

    def perturb_fraction(self,org_fraction,percentage_diff):
        fraction_list  = org_fraction
        first_fraction = fraction_list[0]
        first_fraction_perturbed = first_fraction * (1+percentage_diff)
        second_fraction_calculated = 1 - first_fraction_perturbed

        fraction_list_perturbed = [first_fraction_perturbed,second_fraction_calculated]

        return fraction_list_perturbed




class Relaxtion_Exponential(Collision_Relaxation):
    def __init__(self, Factor, Power, ExponentCutoff):
        self.Factor = [str(Factor), '[1/cm]']
        self.Power = Power
        self.ExponentCutoff = ExponentCutoff

class Lennard_Jones(Collision_Relaxation):
    def __init__(self, Epsilons, Sigmas, Masses):
        self.Epsilons = [str(Epsilons), '[1/cm]']
        self.Sigmas = [str(Sigmas), '[angstrom]']
        self.Masses = [str(Masses), '[amu]']

class Bimolecular(Mess_Input):
    """Bimolecular species initilizor."""

    def __init__(self):
        self.order = []

    def get_bimolecular_pair(self):
        """Get the PAPR-MESS bimolecular species paris."""
        key_ls = list(self.__dict__.keys())
        fra = []
        for key in key_ls:
            if 'Fragment' in key:
                fra.append(self.__dict__[key][0])
        return "+".join(fra)


class Well(Mess_Input):
    """Well initilizor."""

    def __init__(self):
        self.order = []

class Barrier(Mess_Input):
    """Barrier initilizor."""

    def __init__(self):
        self.order = []


    def change_Img_Frequency(self, percentage_diff):
        """Scale the tunneling imaginary frequencies by specified percentage."""
        org_fre = float(self.ImaginaryFrequency[0])
        new_fre = org_fre * (1. + percentage_diff)
        self.ImaginaryFrequency[0] = str(new_fre)

    def change_Nej_file(self, percentage_diff):
        """Scale the Nej file by specified percentage."""
        nej_file = self.__dict__['File'][0]
        with open(nej_file, 'r') as fhand:
            base = fhand.readlines()
        for n, x in enumerate(base):
            if x.strip():
                line = x.strip().split()
                line[1] = str(float(line[1]) * (1. + percentage_diff))
                base[n] = '    '.join(line) + '\n'
        with open(nej_file, 'w') as fhand:
            fhand.writelines(base)

    def change_Variational(self, target, percentage_diff):
        """Change the target variable by specified percentage."""
        for k in self.order:
            if target in k:
                if 'Energy' in k:
                    new_eng = self.perturb_Energy(self.__dict__[k], percentage_diff)
                    self.__dict__[k][0] = str(new_eng)
                elif 'Frequencies' in k:
                    temp = self.perturb_Frequencies(self.__dict__[k]['value'], percentage_diff)
                    self.__dict__[k]['value'] = temp

#Added these two fucntion
    # def change_Hind_rotor(self, percentage_diff):
    #     """Scale the Hindered rotor frequencies by the provided percentage."""
    #     key = 'FourierExpansion'
    #     org_exp = self.__dict__[key]['value']
    #     temp = []
    #     for exp in org_exp:
    #         n, e = exp
    #         ne = e * (1. + percentage_diff)
    #         temp.append((n, ne))
    #     self.__dict__[key]['value'] = temp

    # def Hindered_rotor_correction(self):
    #     """In PAPR-MESS code, for hindered rotor axis and symmetry have to be integers,
    #        otherwise it causes error."""
    #     target_list = ['Axis', 'Symmetry']
    #     for target in target_list:
    #         if not hasattr(self, target):
    #             break
    #         else:
    #             temp = []
    #             for x in self.__dict__[target]:
    #                 temp.append(int(x))
    #             self.__dict__[target] = temp
