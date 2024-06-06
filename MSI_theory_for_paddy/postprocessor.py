# -*- coding: utf-8 -*-
"""

Execute the nominal and perturbed PAPR-MESS calculations,
extract rate constants and fit them into Chebyshev expressions.

@author: Lei Lei
"""

import Mess_executor as ME
import os, io, sys
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

class PAPR_MESS:

    def __init__(self, input_path, input_name, nom_dict, pert_dict, channels, abstraction=False):
        self.input_path = input_path
        self.nominal_file = input_name
        self.pert_dict = pert_dict
        self.nom_dict = nom_dict
        self.channels = channels
        self.abstraction = abstraction

    def Run(self):
        """Generate the perturbed PAPR-MESS inputs, execute them, obtain the temperature-dependent rate constants
           for the specified channels."""
        model = ME.Mess_Executor(self.input_path, self.nominal_file, self.pert_dict, self.nom_dict, self.abstraction)

        # Note: file structure for debugging
        #     -- nwd: nominal calculation directory, twd/nonimal/
        #     -- pwd: purturbed calculations directory, twd/perturbation/
        #     -- mwd: main parent directory, ./
        #     -- twd: trial directory that contains all new calculations, mwd/PAPR-MESS_calculation/system_name/calculation_#/

        # create new directory for new calculation
        self.twd = model.new_trial_directory()
        # generate PAPR-MESS files and execute them
        model.new_calculation(self.channels, self.twd, run_MESS=True)

        # logistics
        if model.pert_P != model.nom_P:
            sys.exit('Error: perturbed and nominal systems have different pressure range.')

        self.pertb_ls = model.pertb
        self.nwd = model.nwd
        self.pwd = model.pwd
        self.mwd = model.mwd
        self.input_name = model.input_name
        self.Punit = model.Punit # Pressure unit

    def fit_Cheb_rates(self, n_P, n_T, P_min=0.01, P_max=100, T_min=200, T_max=3000, target_dir=None, same_line_result=False):
        """Fit the rate constants into Chebyshev polynomials."""
        # Execute the nominal and perturbed PAPR-MESS files
        # self.Run()
        self.n_P = n_P      # number of perssure degree
        self.n_T = n_T      # number of temperature degree
        self.Cheb_coef = {}
        self.pert_ls = {}
        self.P_min = P_min
        self.P_max = P_max
        self.T_min = T_min
        self.T_max = T_max

        # for debugging: fit rate constants in a give trail directory
        if target_dir != None:

            self.twd = target_dir
            self.nwd = target_dir + "nominal/"
            self.pwd = target_dir + "perturbation/"

        # read in the temperature-dependent rate constants for each channel of perturbed files
        for wd in [self.nwd, self.pwd]:
            os.chdir(wd)

            if wd == self.nwd:
                wpert = self.nom_dict
            else:
                wpert = self.pert_dict

            fhand = io.open('T_rate.csv', 'r')
            lines = fhand.readlines()
            fhand.close()

            file_flag = False
            pres_flag = False
            channel_flag = False
            final_line = False

            rate = {}
            pert = {}
            coef_dict = {}
            # generate a rate constants dictionary for each perturbation
            for key in list(wpert.keys()):
                rate[key] = {}
                pert[key] = {}
                coef_dict[key] = {}

            T_ls = []
            chan_rate = []
            self.P_ls = []

            for num, line in enumerate(lines):
                # for the last line in the file
                if num == len(lines) - 1:
                    final_line = True
                    T_ls.append(float(line.split(',')[0]))
                    chan_rate.append(float(line.split(',')[1]))
                # for the other lines
                line = line.strip()
                if line.startswith('=') or final_line:
                    file_flag = True
                    # fit Chebyshev formula and write them in files
                    if len(T_ls) > 2:
                        T_ls = np.array(T_ls)
                        chan_rate = np.array(chan_rate)
                        
                        print(rate)
                        print((self.key))


                        if self.channel in list(rate[self.key].keys()):
                            rate[self.key][self.channel] = np.concatenate((rate[self.key][self.channel], chan_rate))
                        else:
                            rate[self.key][self.channel] = np.array(chan_rate)
                        if any(chan_rate < 0):
                            sys.exit("Error: Negative rate constants detected in run %s-%s..." %(self.key, self.channel))
                        self.T_ls = T_ls
                        T_ls = []
                        chan_rate = []
                    continue
                if file_flag:
                    file_flag = False
                    pres_flag = True
                    line = line[:-4]
                    self.system = line.split('_')[0]
                    self.key = '_'.join(line.split('_')[1:-1])
                    self.pertb = float(line.split('_')[-1])
                    pert[self.key] = self.pertb
                    continue
                if pres_flag:
                    pres_flag = False
                    channel_flag = True
                    # modified for abstraction reactions
                    try:
                        self.pressure = float(line.split(' ')[0])
                        Punit = line.split(' ')[1]
                    except ValueError:
                        self.pressure = line.split(' ')[0]
                        Punit = 'atm'

                    if self.pressure in self.P_ls:
                        continue
                    else:
                        self.P_ls.append(self.pressure)
                    continue
                if channel_flag:
                    channel_flag = False
                    self.channel = line.split(',')[1]
                    continue
                if not file_flag:
                    T_ls.append(float(line.split(',')[0]))
                    chan_rate.append(float(line.split(',')[1]))

            
            if os.path.exists('Chebyshev_fit.txt'):
                fhand = io.open('Chebyshev_fit.txt', 'a')
            else:
                fhand = io.open('Chebyshev_fit.txt', 'w')
            for spc in list(rate.keys()):
                fhand.write('=' * 30 + '\n')
                fhand.write(spc + '\n')
                for key in list(rate[spc].keys()):
                    k = rate[spc][key]
                    fhand.write(key + '\n')
                    coef = self.cheby_poly(n_T, n_P, k, self.T_ls, self.P_ls, P_min, P_max, T_min, T_max).reshape((n_P, n_T))
                    coef_dict[spc][key] = coef
                    #print(key,coef)
                    fhand.write('T_min:%s K    T_max:%s K    P_min:%s %s    P_max:%s %s\n'%(T_min, T_max, P_min, Punit, P_max, Punit))
                    if same_line_result:
                        fhand.write(str(coef.reshape((1,-1)).tolist()) + '\n')
                    else:
                        for P in range(n_P):
                           fhand.write(str(coef[P,:].tolist()) + '\n')
            
            fhand.close()
            # store the fitted coefficients for Chebyshev polynomials
            if wd == self.nwd:
                self.Cheb_coef['nominal'] = coef_dict
                self.pert_ls['nominal'] = pert
            else:
                self.Cheb_coef['perturbed'] = coef_dict
                self.pert_ls['perturbed'] = pert
        if target_dir == None:
            print(("Fitting channel-specific rate conctants into Chebyshev form for system %s ..." %self.input_name.split(".")[0]))

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
        T_tilde = 2. * T ** (-1) - T_min ** (-1) - T_max ** (-1)
        T_tilde /= (T_max ** (-1) - T_min ** (-1))
        return T_tilde

    def reduced_P(self, P, P_min, P_max):
        '''Calculate the reduced pressure.'''
        P_tilde = 2. * np.log(P) - np.log(P_min) - np.log(P_max)
        P_tilde /= (np.log(P_max) - np.log(P_min))
        return P_tilde

    def cheby_poly(self, n_T, n_P, k, T_ls, P_ls, P_min=0.01, P_max=100, T_min=200, T_max=3000):
        '''Fit the Chebyshev polynominals to rate constants.
           Input rate constants vector k should be arranged based on pressure.'''
        # modified for abstraction reactions
        if P_ls == ['--']:
            P_ls = [1.0]

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

    def Cheb_sens_coeff(self, same_line_result=False, aggregated_sens=True, debug=False):
        """Calculate the sensitivity coefficients for the Chebyshev rate constants."""
        # initialization
        if aggregated_sens:
            self.aggregated_sens = pd.DataFrame()
            if not debug:
                self.aggregated_sens['Pressure (%s)' %self.Punit] = np.repeat(self.P_ls, len(self.T_ls))
            self.aggregated_sens['Temperature (K)'] = np.tile(self.T_ls, len(self.P_ls))
        self.Cheb_sens = {}
        # decide sensitivity coefficients for each perturbation
        for key in list(self.pert_dict.keys()):
            # write the sensitivity coefficients into file
            os.chdir(self.twd)
            if os.path.exists('Chebyshev_sens.txt'):
                fhand = io.open('Chebyshev_sens.txt', 'a')
            else:
                fhand = io.open('Chebyshev_sens.txt', 'w')
            fhand.write('='*30 + '\n')
            fhand.write(key + '\n')

            Cheb_sens = {}
            # determine nominal calculation
            nom_key = key[:-1] + '1'
            if len(list(self.Cheb_coef['nominal'].keys())) == 1:
                nom_key = list(self.Cheb_coef['nominal'].keys())[0]
            # calculate the channel-specific sensitivity coefficients
            for chan in list(self.Cheb_coef['perturbed'][key].keys()):
                rate_diff = self.Cheb_coef['perturbed'][key][chan] - self.Cheb_coef['nominal'][nom_key][chan]
                if 'Energy' in key:
                    sens = rate_diff / (self.pert_ls['perturbed'][key] - self.pert_ls['nominal'][nom_key])
                    print(((self.pert_ls['perturbed'][key] - self.pert_ls['nominal'][nom_key]),'DIFFERNENCE'))
                    print((self.pert_ls['perturbed'][key],self.pert_ls['nominal'][nom_key]))
                    if aggregated_sens:
                        self.aggregated_sens['%s_%s'%(key,chan)] = self.calculate_sensitivity(sens).reshape((1,-1))[0] * 349.758 * np.log(10)
                elif 'Power' in key:
                    #print('LOOKING HERE')
                    #print(chan,'perturbebd')
                    #print(self.Cheb_coef['perturbed'][key][chan])
                    #print(chan,'nominal')
                    #print(self.Cheb_coef['nominal'][nom_key][chan])
                    #print(chan,'This is power rate diff')
                    #print(self.Cheb_coef['perturbed'][key][chan] - self.Cheb_coef['nominal'][nom_key][chan])
                    
                    #print('This is power denominator')
                    #print(self.pert_ls['perturbed'][key], self.pert_ls['nominal'][nom_key], self.pert_ls['perturbed'][key] - self.pert_ls['nominal'][nom_key])
                    sens = rate_diff / (self.pert_ls['perturbed'][key] - self.pert_ls['nominal'][nom_key])
                    if aggregated_sens:
                        self.aggregated_sens['%s_%s'%(key,chan)] = self.calculate_sensitivity(sens).reshape((1,-1))[0] * np.log(10)                    
                else:
                    sens = rate_diff / np.log((1. + self.pert_ls['perturbed'][key]) / (1. + self.pert_ls['nominal'][nom_key]))
                    if aggregated_sens:
                        self.aggregated_sens['%s_%s'%(key,chan)] = self.calculate_sensitivity(sens).reshape((1,-1))[0] * np.log(10)
                Cheb_sens[chan] = sens
                # write into output file
                fhand.write(str(chan) + '\n')
                if same_line_result:
                    fhand.write(str(sens.reshape((1,-1)).tolist()) + '\n')
                else:
                    sens = sens.reshape(self.n_P, self.n_T)
                    for P in range(self.n_P):
                        fhand.write(str(sens[P,:].tolist()) + '\n')
            self.Cheb_sens[key] = Cheb_sens
        fhand.close()
        if aggregated_sens:
            # write aggregated sensitivity into file
            self.aggregated_sens.to_csv("Aggregated_sens.csv", index=False)
        if not debug:
            os.chdir(self.mwd)
            print(("Calculating channel-specific sensitivity coefficients for Chebyshev polynomials for system %s ..." %self.input_name.split('.')[0]))

    def fit_Arr_perturbed_rates(self, target_dir=None):
        """Fit rate constants into Arrhenius formula."""
        self.Arr_coef = {}
        # read in the temperature-dependent rate constants for each channel of perturbed files
        for wd in [self.nwd, self.pwd]:
            os.chdir(wd)

            fhand = io.open('T_rate.csv', 'r')
            lines = fhand.readlines()
            fhand.close()

            file_flag = False
            pres_flag = False
            channel_flag = False
            final_line = False

            temp = []
            rate = []

            for line in lines:
                if line == lines[-1]:
                    final_line = True
                    temp.append(float(line.split(',')[0]))
                    rate.append(float(line.split(',')[1]))
                line = line.strip()
                if line.startswith('=') or final_line:
                    file_flag = True
                    # fit the three-parameter Arrhenius formula and write them in files
                    if len(temp) > 2:
                        temp = np.array(temp)
                        rate = np.array(rate)
                        if any(rate < 0):
                            sys.exit("Error: Negative rate constants detected in run %s..." %(self.channel))
                        fit = self.log_three_para_Arr_fit(temp, rate)
                        # determine nominal or perturbation
                        if wd == self.nwd:
                            self.Arr_coef[self.channel] = {}
                            self.Arr_coef[self.channel][self.pressure] = {}
                            self.Arr_coef[self.channel][self.pressure]['nominal_%s' %(self.pertb)] = fit
                        else:
                            self.Arr_coef[self.channel][self.pressure]['pert_%s' %(self.pertb)] = fit
                        # reset
                        temp = []
                        rate = []
                        if os.path.exists('Arrhenius_fit.txt'):
                            fhand = io.open('Arrhenius_fit.txt', 'a')
                        else:
                            fhand = io.open('Arrhenius_fit.txt', 'w')
                        fhand.write('=' * 30 + '\n')
                        fhand.write(self.system + '\n')
                        fhand.write(self.pertb + '\n')
                        fhand.write(self.pressure + '\n')
                        fhand.write(self.channel + ' (' + ', '.join([str(_) for _ in fit]) + ') \n')
                        fhand.close()
                    continue
                if file_flag:
                    file_flag = False
                    pres_flag = True
                    line = line[:-4]
                    self.system = line.split('_')[0]
                    self.pertb = '_'.join(line.split('_')[1:])
                    continue
                if pres_flag:
                    pres_flag = False
                    channel_flag = True
                    self.pressure = line
                    continue
                if channel_flag:
                    channel_flag = False
                    self.channel = line.split(',')[1]
                    continue
                if not file_flag:
                    temp.append(float(line.split(',')[0]))
                    rate.append(float(line.split(',')[1]))
        if target_dir == None:
            print(("Fitting channel-specific rate conctants into Arrhenius form for system %s ..." %self.input_name.split(".")[0]))

    def log_three_para_Arr_fit(self, Temp, rate, ini_guess = (1,1,1), max_initeration = 1000000):
        '''Fit three-parameter Arrhenius rate coefficient expression.'''
        rate = np.log(rate)
        func = lambda T, A, n, Ea: np.log(A) + n * np.log(T) - Ea / T
        fit = curve_fit(func, Temp, rate, p0 = ini_guess, maxfev = max_initeration, ftol = 1E-11, xtol = 1E-11, gtol = 1E-11)
        return fit[0]

    def Arr_sens_coeff(self):
        """Calculate sensitivity coefficients for Arrhenius fittings."""
        # print(self.Arr_coef)
        # decide sensitivity coefficients for each perturbation
        for key in list(self.Arr_coef.keys()):
            # write the sensitivity coefficients into file
            os.chdir(self.twd)
            if os.path.exists('Arrhenius_sens.txt'):
                fhand = io.open('Arrhenius_sens.txt', 'a')
            else:
                fhand = io.open('Arrhenius_sens.txt', 'w')
            fhand.write('='*30 + '\n')
            fhand.write(key + '\n')
            Arr_sens = {}
            # determine nominal calculation
            for channel in list(self.Arr_coef.keys()):
                for pressure in list(self.Arr_coef[channel].keys()):
                    # get nominla fittings
                    for key in list(self.Arr_coef[channel][pressure].keys()):
                        if 'nominal' in key:
                            nom_pert = float(key.split('_')[-1])
                            nom_coef = self.Arr_coef[channel][pressure][key]
                            break
                    for key in list(self.Arr_coef[channel][pressure].keys()):
                        if 'nominal' in key:
                            continue
                        else:
                            pert = float(key.split('_')[-1])
                            rate_diff = self.Arr_coef[channel][pressure][key] - nom_coef
                            if 'Energy' in key:
                                sens = rate_diff / (pert - nom_pert)
                            else:
                                sens = rate_diff / np.log((1. + pert) / (1. + nom_pert))
                        # write into output file
                        fhand.write('Pressure: ' + str(pressure) + ', ' + str(key) + '\n')
                        fhand.write(str(sens) + '\n')
        fhand.close()
        os.chdir(self.mwd)
        print(("Calculating channel-specific sensitivity coefficients for Arrhenius fittings for system %s ..." %self.input_name.split('.')[0]))

    def calculate_sensitivity(self, sens_coef):
        '''Calculate the aggregated sensiticity as a function of temperature and pressure.'''
        cheb_mat = np.zeros((len(self.P_ls) * len(self.T_ls), self.n_T * self.n_P))
        for n, P in enumerate(self.P_ls):       # !! assume that at each presssure, we have the same temperateure range
            for m, T in enumerate(self.T_ls):
                for i in range(self.n_T):
                    T_tilde = self.reduced_T(T, self.T_min, self.T_max)
                    T_cheb = self.first_cheby_poly(T_tilde, i)
                    for j in range(self.n_P):
                        P_tilde = self.reduced_P(P, self.P_min, self.P_max)
                        P_cheb = self.first_cheby_poly(P_tilde, j)
                        cheb_mat[n*len(self.T_ls)+m, i*self.n_P+j] = P_cheb * T_cheb
        sens = np.dot(cheb_mat, sens_coef.reshape((-1,1)))
        return sens

