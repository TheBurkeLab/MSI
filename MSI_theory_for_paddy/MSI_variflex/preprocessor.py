# -*- coding: utf-8 -*-
"""

Perturb the input and run the calculations.

@author: Lei Lei
"""

import input_cleaner as IC
import class_generator as CG
import os, copy, shutil, sys

# keep the same structure as PAPR-MESS code:
# 1. need to specify the nominal conditions to allow the
#    flexiblity of starting from nonzero perturbation as
#    nominal conditions
# 2. all the sensitivity analysis for perturbed consitions
#    is based on the pre-defined nominal conditions
# 3. less flexibilities in the channel-specific rate constants
#    calculations compared to PAPR-MESS, Variflex only
#    outputs unimolecular (well-to-bimolecular-species) and
#    bimolecular (bimolecular-species-to-well) rate constants
#    but not bimolcular-to-bimolecular rate constants

def generate_perturb_class(ori_class, perturb_key, perturb_values):

    # create new Variflex class based on the original class and perturbation
    perturb_class = copy.deepcopy(ori_class)
    component, parameter = perturb_key.split('_')[0], perturb_key.split('_')[1]
    TempL, PresL = perturb_values[0], perturb_values[1]
    pert_percent = perturb_values[2]

    perturb_class.CalculationRanges.set_T_P_ranges(TempL, PresL)

    if "Energy" in parameter:
        if 'W' in component:
            perturb_class.CalculationRanges.change_complex_Energy(pert_percent)
        if 'B' in component:
            num = int(component.split('B')[-1])
            species = 'TransState%s' %num
            if perturb_class.Channels.transitions[species].hasNej:
                sys.exit("Error: Cannot perturb barrier energy when using Nej...")
            perturb_class.Channels.transitions[species].change_TS_Energy(pert_percent)

    if 'Imaginary' in parameter:
        num = int(component.split('B')[-1])
        species = 'TransState%s' %num
        perturb_class.Channels.transitions[species].change_Img_Freq(pert_percent)

    if 'Symmetry' in parameter:
        if 'W' in component:
            perturb_class.Complex.change_Sym_Factor(pert_percent)
        if 'B' in component:
            num = int(component.split('B')[-1])
            species = 'TransState%s' %num
            if perturb_class.Channels.transitions[species].hasNej:
                sys.exit("Error: Cannot perturb barrier energy when using Nej...")
            perturb_class.Channels.transitions[species].change_Sym_Factor(pert_percent)

    if 'Frequencies' in parameter:
        if 'W' in component:
            perturb_class.Complex.change_Vib_Freq(pert_percent)
        if 'B' in component:
            num = int(component.split('B')[-1])
            species = 'TransState%s' %num
            if perturb_class.Channels.transitions[species].hasNej:
                sys.exit("Error: Cannot perturb barrier vibrational frequencies when using Nej...")
            perturb_class.Channels.transitions[species].change_Vib_Freq(pert_percent)

    if 'HindRotor' in parameter:
        if 'W' in component:
            perturb_class.Complex.change_Hind_Rot(pert_percent)
        if 'B' in component:
            num = int(component.split('B')[-1])
            species = 'TransState%s' %num
            if perturb_class.Channels.transitions[species].hasNej:
                sys.exit("Error: Cannot perturb barrier hinder rotor frequencies when using Nej...")
            perturb_class.Channels.transitions[species].change_Hind_Rot(pert_percent)

    if 'NejScale' in parameter:
        num = int(component.split('B')[-1])
        species = 'TransState%s' %num
        perturb_class.Channels.transitions[species].Nej_scale(pert_percent, num)
		
    if 'ElecStates' in parameter:
        num = int(component.split('B')[-1])
        species = 'TransState%s' %num
        perturb_class.Channels.transitions[species].change_elec_states(pert_percent)

    return perturb_class

def generate_inputs(ori_class, nominal_cond, perturb_cond):
    '''Generate input files for both nomial condition and perturbed conditions.'''
    # parent working directory
    pwd = os.getcwd()

    # MSI working directory
    mwd = pwd + '/Variflex_Calculations'
    if not os.path.exists(mwd):
        os.mkdir(mwd)
    os.chdir(mwd)

    system_name = ori_class.Title.order[1]
    system_name = system_name.replace('&','')
    system_name = ''.join(system_name.split()[:3])

    # system working directory
    swd = mwd + '/' + system_name
    if not os.path.exists(swd):
        os.mkdir(swd)
    os.chdir(swd)

    # calculation working directory
    n = 0
    while True:
        n += 1
        cwd = swd + '/Calculation_%s'%n
        if not os.path.exists(cwd):
            os.mkdir(cwd)
            os.chdir(cwd)
            break

    # generate the bash file to run all the inputs
    fhand = open('calculation.sh', 'w+')

    # nominal working directory and perturbation working directory
    nom_key = nominal_cond.keys()[0]
    nom_values = nominal_cond[nom_key]
    nwd = cwd + '/nominal_' + nom_key + '_' + str(nom_values[-1])
    os.mkdir(nwd)
    os.chdir(nwd)
    if ori_class.hasNej:
        try:
            shutil.copy2('%s/nej.dat'%pwd, nwd)
        except IOError:
            sys.exit("Nej file not found...Abort")
    nom_class = generate_perturb_class(ori_class, nom_key, nom_values)
    nom_class.write_file('variflex.dat')
    os.system('cp %s/variflex.exe ./' %pwd)

    fhand.write('cd %s \n' %nwd)
    fhand.write('echo Running nominal Variflex...\n')
    fhand.write('./variflex.exe \n')
    fhand.write('echo Running perturbed Variflex...\n')


    for pert_key in perturb_cond.keys():
        os.chdir(cwd)
        pert_class = None
        qwd = None
        pert_values = perturb_cond[pert_key]
        qwd = cwd + '/perturb_' + pert_key + '_' + str(pert_values[-1])
        os.mkdir(qwd)
        os.chdir(qwd)
        if ori_class.hasNej:
            try:
                shutil.copy2('%s/nej.dat'%pwd, qwd)
            except IOError:
                sys.exit("Nej file not found...Abort")
        pert_class = generate_perturb_class(ori_class, pert_key, pert_values)
        pert_class.write_file('variflex.dat')
        os.system('cp %s/variflex.exe ./' %pwd)

        fhand.write('cd %s \n' %qwd)
        fhand.write('./variflex.exe \n')
    fhand.close()
    print("Generating Variflex Inputs...")
    return cwd

