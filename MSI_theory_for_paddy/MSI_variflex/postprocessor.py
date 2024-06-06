# -*- coding: utf-8 -*-
"""

Perturb the input and run the calculations.

@author: Lei Lei
"""

import input_cleaner as IC
import class_generator as CG
import preprocessor as PP
import os, copy, sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

###########################################################################################################
class Variflex():

    def __init__(self, input_name, nom_dict, pert_dict):
        '''
           Initialize the class.
              - input_name: str, the name of nominal variflex input file
              - nom_dict, pert_dict: dict, the keys in the dict specify where the perturbations happen, and  the associated values contain a temperature range, a pressure range, and percentage of perturbation from the corresponding values in the nominal variflex input file given in "input_name"
        '''
        self.nominal_file = input_name
        self.pert_dict = pert_dict
        self.nom_dict = nom_dict
        self.T_range = nom_dict[nom_dict.keys()[0]][0]   # Temperature range of calculations
        self.P_range = nom_dict[nom_dict.keys()[0]][1]   # Pressure range of calculations

    def run_Variflex(self):
        '''This function serves four purposes:
             1. Clean the nominal Variflex input file;
             2. Generate Variflex input files for nominal perturbations;
             3. Generate Variflex input files for perturbations;
             4. Execute Variflex.exe for the generated inputs to get the phenomenological rate constants.'''

        self.pwd = os.getcwd()                  # pwd, parent directory, i.e. where this python code lives
        IC.clean_input(self.nominal_file)
        # genreate class for nominal file
        ori_variflex = CG.generate_class()
        os.system('rm cleaned_input.txt')
        # generate perturbed files and run Variflex simulation
        cwd = PP.generate_inputs(ori_variflex, self.nom_dict, self.pert_dict)           # cwd, the calculaion directory
        # the next line is for debugging
        #cwd = "/home/lab-lei/Documents/MSI/Variflex/Variflex_Calculations/OH+OH/Calculation_6/"
        os.chdir(cwd)
        # run Variflex calculations
        os.system('sh calculation.sh')
        self.cwd = cwd
        self.calculation_method = ori_variflex.CalculationType.order

    def extract_rate_constants_low_P(self):
        '''Extract the low-pressure collision-less rate constants from variflex.out file.
           Return:
              - nominal_rate: dict, rate constants for each production channel for the nominal perturbation
              - perturbed_rate: dict, rate constants for each production channel for the perturbations'''
        dir_list = [f for f in os.listdir(self.cwd) if os.path.isdir(f)]
        nominal_rate = {}
        perturbed_rate = {}

        header = 'Multiple well bimolecular rates '

        for f in dir_list:
            os.chdir(self.cwd + "/%s" %f)
            with open('variflex.out', 'r') as fhand:
                lines = fhand.readlines()
            header_section = False
            temp = []
            k = []
            for n, line in enumerate(lines):
                if header in line:
                    header_section = True
                    continue
                if header_section and 'For bimolecular products' in line:
                    temp.append(float(lines[n-1].strip().split('T =')[-1].strip().split('K')[0]))
                    k.append([float(x) for x in lines[n+1].strip().split()])
                    header_section = False
                    continue
            k = np.array(k)
            for c in range(k.shape[1]):
                key = f + '_Channel%d' %(c+1)
                rate_ls = [','.join([str(x[0]), str(x[1])]) for x in zip(temp, k[:,c])]
                rate_ls.insert(0,'T(K),k_bi_LPL')
                if "nominal" in f:
                    nominal_rate[key] = rate_ls
                elif "perturb" in f:
                    perturbed_rate[key] = rate_ls
        self.write_rates_to_file(nominal_rate, perturbed_rate, 'nom_LPL.csv', 'pert_LPL.csv')
        return nominal_rate, perturbed_rate

    def extract_rate_constants(self):
        '''Extract rate constants according to specific calculation methods from the trate.out files.'''
        # get all the folders in the calculation directory
        dir_list = [f for f in os.listdir(self.cwd) if os.path.isdir(f)]
        nominal_rate = {}
        perturbed_rate = {}

        # extract rate constants for the high-pressure limit calculations
        if "HighPresP" in self.calculation_method or "AssociationP" in self.calculation_method:
            header = "High pressure limit"
            HP = True
        elif "1DMasterP PReacPQ EigenvaluePQ" in self.calculation_method:
            header = "T and P dependent rate constants"
            HP = False
            # for HighPressP and EigenvaluePQ methods, use trate.out for the high-pressure-limit rate constants
        for f in dir_list:
            os.chdir(self.cwd + "/%s" %f)
            with open('trate.out', 'r') as fhand:
                lines = fhand.readlines()
            section_header = False
            rate_header = False
            temp_rate = []
            for line in lines:
                if header in line:
                    section_header = True
                    continue
                if section_header and "Temp" in line:
                    rate_header = True
                    # determine columns to slice
                    split_header = line.strip().split()
                    if HP:
                        try:
                            ind = max([split_header.index(x) for x in ['k_bi-TST', 'k_uni-TST']]) + 1
                        except ValueError:
                            ind = split_header.index('k_bi-TST') + 1
                    else:
                        ind = 3
                    temp_rate.append(','.join(split_header[:ind]))
                    temp_rate[-1] = temp_rate[-1].replace("Temp", "T(K)")
                    if "Pressure" in temp_rate[-1]:
                        temp_rate[-1] = temp_rate[-1].replace("Pressure", "P(torr)")
                    continue
                if rate_header and not line.strip():
                    rate_header = False
                    break
                if rate_header:
                    temp_rate.append(','.join(line.strip().split()[:ind]))
            # note that the first line in temp_rate is unit, ignore...
            temp_rate.pop(1)
            # reorganize the rate constant results by pressure, temperature
            if not HP:
                r_ls = [temp_rate[0]]
                for P in range(len(self.P_range)):
                    r_ls.extend(temp_rate[P+1::len(self.P_range)])
            else:
                r_ls = temp_rate
            if "nominal" in f:
                nominal_rate[f] = r_ls
            elif "perturb" in f:
                perturbed_rate[f] = r_ls
        # write rate constants to files
        self.write_rates_to_file(nominal_rate, perturbed_rate)
        return nominal_rate, perturbed_rate

    def write_rates_to_file(self, nominal_rate, perturbed_rate, nom_file="nonimal_T_rate.csv", pert_file="perturbed_T_rate.csv"):
        # write rate constants into csv files
        os.chdir(self.cwd)
        for rate_ls in [nominal_rate, perturbed_rate]:
            if rate_ls == nominal_rate:
                file_name = nom_file
            else:
                file_name = pert_file
            with open(file_name, 'w+') as fhand:
                for f in rate_ls.keys():
                    fhand.write('='*30 + '\n')
                    fhand.write('%s\n' %f)
                    if "HighPresP" in self.calculation_method:
                        fhand.write("High pressure limit\n")
                    for i in rate_ls[f]:
                        fhand.write("%s\n" %i)

    def fit_Arrhenius_rates(self, nominal_rate, perturbed_rate):
        """Fit rate constants into Arrhenius formula."""
        print("Fitting channel-specific rate conctants into Arrhenius forms...")
        os.chdir(self.cwd)
        fitted_coef = {}
        for f in [nominal_rate, perturbed_rate]:
            for r in f.keys():
                rate_ls = []
                temp_ls = []
                for x in f[r][1:]:
                    T, R = x.strip().split(',')
                    rate_ls.append(float(R))
                    temp_ls.append(float(T))
                if any(np.array(rate_ls) < 0):
                    sys.exit("Error: Negative rate constants detected...")
                fitted_coef[r] = self.log_three_para_Arr_fit(temp_ls, rate_ls)
                if os.path.exists('Arrhenius_fit.txt'):
                    fhand = open('Arrhenius_fit.txt', 'ab')
                else:
                    fhand = open('Arrhenius_fit.txt', 'wb')
                fhand.write('=' * 30 + '\n')
                fhand.write('%s\n' %r)
                fhand.write('T_min:%dK    T_max:%dK    Channel:%s\n'%(min(temp_ls), max(temp_ls), f[r][0].split(',')[-1]))
                fhand.write(str(fitted_coef[r].tolist()) + '\n')
                fhand.close()
        self.Arr_coef = fitted_coef

    def log_three_para_Arr_fit(self, Temp, rate, ini_guess = (1,1,1), max_initeration = 1000000):
        '''Fit three-parameter Arrhenius rate coefficient expression.'''
        rate = np.log(rate)
        func = lambda T, A, n, Ea: np.log(A) + n * np.log(T) - Ea / T
        fit = curve_fit(func, Temp, rate, p0 = ini_guess, maxfev = max_initeration, ftol = 1E-11, xtol = 1E-11, gtol = 1E-11)
        return fit[0]

    def first_cheby_poly(self, x, n):
        '''Generate n-th order Chebyshev ploynominals of first kind.'''
        if n == 0: return 1
        elif n == 1: return x
        result = 2. * x * self.first_cheby_poly(x, 1) - self.first_cheby_poly(x, 0)
        m = 0
        while n - m > 2:
            result = 2. * x * result - self.first_cheby_poly(x, m+1)
            m += 1
        return result

    def reduced_T(self, T, T_min, T_max):
        '''Calculate the reduced temperature.'''
        T_tilde = 2. * float(T) ** (-1) - T_min ** (-1) - T_max ** (-1)
        T_tilde /= (T_max ** (-1) - T_min ** (-1))
        return T_tilde

    def reduced_P(self, P, P_min, P_max):
        '''Calculate the reduced pressure.'''
        P_tilde = 2. * np.log10(P) - np.log10(P_min) - np.log10(P_max)
        P_tilde = P_tilde / (np.log10(P_max) - np.log10(P_min))
        return P_tilde

    def cheby_poly(self, n_T, n_P, k, T_ls, P_ls, P_min=0.01, P_max=100, T_min=200, T_max=3000):
        '''Fit the Chebyshev polynominals to rate constants.
           Input rate constants vector k should be arranged based on pressure.'''
        cheb_mat = np.zeros((len(k), n_T * n_P))
        for n, P in enumerate(P_ls):       # !! assume that at each presssure, we have the same temperateure range
            for m, T in enumerate(T_ls):
                for i in range(n_T):
                    T_tilde = self.reduced_T(T, T_min, T_max)
                    T_cheb = self.first_cheby_poly(T_tilde, i)
                    for j in range(n_P):
                        P_tilde = self.reduced_P(P, P_min, P_max)
                        P_cheb = self.first_cheby_poly(P_tilde, j)
                        cheb_mat[n*len(T_ls)+m, i*n_P+j] = P_cheb * T_cheb
        coef = np.linalg.lstsq(cheb_mat, np.log10(np.array(k)))[0]
        return coef

    def fit_Chebyshev_rates(self, n_T, n_P, nominal_rate, perturbed_rate, P_min=7.6, P_max=76000, T_min=200, T_max=3000, output_file='Chebyshev_fit.txt'):
        """Fit rate constants into Chebyshev formula."""
        print("Fitting channel-specific rate conctants into Chebyshev forms...")
        os.chdir(self.cwd)
        # cache Chebyshev parameters
        self.n_T = n_T
        self.n_P = n_P
        self.T_min = T_min
        self.T_max = T_max
        self.P_min = P_min
        self.P_max = P_max
        # fitting starts here
        fitted_coef = {}
        for f in [nominal_rate, perturbed_rate]:
            for r in f.keys():
                rate_ls = []
                for x in f[r][1:]:
                    R = x.strip().split(',')[-1]
                    if float(R) < 0:
                        sys.exit("Error: Negative rate constants detected...")
                    rate_ls.append(float(R))
                fitted_coef[r] = self.cheby_poly(n_T, n_P, rate_ls, self.T_range, self.P_range, P_min, P_max, T_min, T_max).reshape((n_P, n_T))
                if os.path.exists(output_file):
                    fhand = open(output_file, 'ab')
                else:
                    fhand = open(output_file, 'wb')
                fhand.write('=' * 30 + '\n')
                fhand.write('%s\n' %r)
                fhand.write('T_min:%s K    T_max:%s K    P_min:%s Torr    P_max:%s Torr\n'%(T_min, T_max, P_min, P_max))
                for P in range(len(self.P_range)):
                    fhand.write(str(fitted_coef[r][P,:].tolist()) + '\n')
                fhand.close()
        self.Cheb_coef = fitted_coef

    def sensitivity_coef(self, fitting="Chebyshev", output_file='Chebyshev_sens.txt', aggregated_sens=True):
        """Calculate the sensitivity coefficients for fitted coefficients."""
        if aggregated_sens:
            self.aggregated_sens = pd.DataFrame()
        if fitting == "Chebyshev":
            print("Calculating sensitivity coefficients for Chebyshev fittings...")
            nominal_fits = {}
            for key in self.Cheb_coef.keys():
                # find all the nominal channel
                if 'nominal' in key:
                    nominal_fits[key] = self.Cheb_coef[key]
            # calculate sensitivity coefficients
            with open(output_file, 'w+') as fhand:
                for key in self.Cheb_coef.keys():
                    channel = None
                    nominal_key = None
                    nominal_pert = None
                    if key in nominal_fits.keys(): continue
                    temp_key = key.split('_')
                    if 'Channel' in temp_key[-1]:
                        channel = temp_key[-1]
                        pert = float(temp_key[-2])
                    else:
                        pert = float(temp_key[-1])
                    # if there is a single channel in the output
                    if channel == None:
                        nominal_key = nominal_fits.keys()[0]
                        nominal_pert = float(nominal_key.split('_')[-1])
                    else:
                        for k in nominal_fits.keys():
                            if channel in k:
                                nominal_key = k
                                break
                        nominal_pert = float(nominal_key.split('_')[-2])

                    if 'Energy' in key:
                        sens = (self.Cheb_coef[key] - self.Cheb_coef[nominal_key]) / pert
                        # convert wavenumber into kcal/mol
                        if aggregated_sens:
                            self.aggregated_sens[key] = self.calculate_sensitivity(sens).reshape((1,-1))[0] * 349.758 * np.log(10)
                    else:
                        sens = (self.Cheb_coef[key] - self.Cheb_coef[nominal_key]) / np.log((1. + pert) / (1. + nominal_pert))
                        # convert log10 based sensitivity to ln based sensitivity
                        if aggregated_sens:
                            self.aggregated_sens[key] = self.calculate_sensitivity(sens).reshape((1,-1))[0] * np.log(10)

                    fhand.write('=' * 30 + '\n')
                    fhand.write('%s\n' %key)
                    for P in range(len(self.P_range)):
                        fhand.write(str(sens[P,:].tolist()) + '\n')
            # write aggregated sensitivity into file
            self.aggregated_sens.to_csv("Aggregated_sens.csv", index=False)

    def calculate_sensitivity(self, sens_coef):
        '''Calculate the aggregated sensiticity as a function of temperature and pressure.'''
        cheb_mat = np.zeros((len(self.P_range) * len(self.T_range), self.n_T * self.n_P))
        for n, P in enumerate(self.P_range):       # !! assume that at each presssure, we have the same temperateure range
            for m, T in enumerate(self.T_range):
                for i in range(self.n_T):
                    T_tilde = self.reduced_T(T, self.T_min, self.T_max)
                    T_cheb = self.first_cheby_poly(T_tilde, i)
                    for j in range(self.n_P):
                        P_tilde = self.reduced_P(P, self.P_min, self.P_max)
                        P_cheb = self.first_cheby_poly(P_tilde, j)
                        cheb_mat[n*len(self.T_range)+m, i*self.n_P+j] = P_cheb * T_cheb
        sens = np.dot(cheb_mat, sens_coef.reshape((-1,1)))
        return sens

###########################################################################################################

if __name__ == '__main__':

    input_file = 'variflex.dat'
    temperature_range = np.arange(200,700,10).tolist() + np.arange(700, 3100, 100).tolist()  # K
    pressure_range = [760.] # Torr
    nominal_cond = {'B1_Energy_1': [temperature_range, pressure_range, 0.0]}
    pert_cond = {#'W1_Energy_1': [temperature_range, pressure_range, 0.35],
                 #'W1_Frequencies_1': [temperature_range, pressure_range, 0.001],
                 #'W1_Symmetry_1': [temperature_range, pressure_range, 0.001],
                 #'B1_NejScale_1': [temperature_range, pressure_range, 0.05],
                 #'B2_HindRotor_1': [temperature_range, pressure_range, 0.05],
                 #'B2_Frequencies_1': [temperature_range, pressure_range, 0.05],
                 #'B2_Energy_1': [temperature_range, pressure_range, 35.],
                 #'B2_Symmetry_1': [temperature_range, pressure_range, 0.05],
                 #'B2_Imaginary_1': [temperature_range, pressure_range, 0.05],
                 #'B1_HindRotor_1': [temperature_range, pressure_range, 0.4],
                 #'B1_Frequencies_1': [temperature_range, pressure_range, 0.4],
                 #'B1_Energy_1': [temperature_range, pressure_range, 140.],}
                 #'B1_Symmetry_1': [temperature_range, pressure_range, 0.4],
                 #'B1_Imaginary_1': [temperature_range, pressure_range, 0.4],
                 'B1_ElecStates_1': [temperature_range, pressure_range, 35.],}

    # create Variflex classes and input files
    Model = Variflex(input_file, nominal_cond, pert_cond)
    # run Variflex Calculations
    Model.run_Variflex()
    # extract rate constants and write into files
    nom_rate, pert_rate = Model.extract_rate_constants()
    
    # fit rate constants into the Chebeyshev polynominals
    # for high-pressure-limit, apply the Arrhenius fitting
    if "HighPresP" in Model.calculation_method:
        # Model.fit_Arrhenius_rates(nom_rate, pert_rate)
        # For high-pressure limit rate constants, no pressure dependence, use 1 for the number of pressure polynominal
        Model.fit_Chebyshev_rates(15, 1, nom_rate, pert_rate)
        Model.sensitivity_coef(fitting="Chebyshev")
        
    # for T-, P-dependent rate constants, apply the Chebyshev fitting
    elif "1DMasterP PReacPQ EigenvaluePQ" in Model.calculation_method:
        Model.fit_Chebyshev_rates(7, 3, nom_rate, pert_rate)

    # for T-dependent rate constants in the collision-less limit
    elif "AssociationP" in Model.calculation_method:
        Model.fit_Chebyshev_rates(15, 1, nom_rate, pert_rate, output_file='Chebyshev_fit_HPL.txt')
        Model.sensitivity_coef(fitting="Chebyshev")
        nom_rate, pert_rate = Model.extract_rate_constants_low_P()
        Model.fit_Chebyshev_rates(15, 1, nom_rate, pert_rate, output_file='Chebyshev_fit_LPL.txt')
        Model.sensitivity_coef(fitting="Chebyshev", output_file='Chebyshev_sens_LPL.txt')


