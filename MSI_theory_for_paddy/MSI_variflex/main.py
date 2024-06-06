import postprocessor as PST
import numpy as np
import os

base_pert = [0.01, 3.5] # for multiplicative and additive perturbations respectively
multipliers = [1., 2., 5., 10., 20., 30., 40.]
multipliers = [50., 60., 70., 80., 90.]
cwd = os.getcwd()

input_file = 'variflex.dat'
temperature_range = np.arange(200,700,10).tolist() + np.arange(700, 3100, 100).tolist()  # K
pressure_range = [760.] # Torr

for x in multipliers:
    print('Running at perturbation %s...' %(base_pert[0] * x))
    nominal_cond = {'B1_Energy_1': [temperature_range, pressure_range, 0.0]}
    pert_cond = {#'W1_Energy_1': [temperature_range, pressure_range, base_pert[1] * x],
                 #'W1_Frequencies_1': [temperature_range, pressure_range, base_pert[0] * x],
                 #'W1_Symmetry_1': [temperature_range, pressure_range, base_pert[0] * x],
                 #'B1_NejScale_1': [temperature_range, pressure_range, base_pert[0] * x],
                 #'B2_HindRotor_1': [temperature_range, pressure_range, base_pert[0] * x],
                 #'B2_Frequencies_1': [temperature_range, pressure_range, base_pert[0] * x],
                 #'B2_Energy_1': [temperature_range, pressure_range, base_pert[1] * x],
                 #'B2_Symmetry_1': [temperature_range, pressure_range, base_pert[0] * x],
                 #'B2_Imaginary_1': [temperature_range, pressure_range, base_pert[0] * x],
                 #'B1_HindRotor_1': [temperature_range, pressure_range, base_pert[0] * x],
                 #'B1_Frequencies_1': [temperature_range, pressure_range, base_pert[0] * x],
                 'B1_Energy_1': [temperature_range, pressure_range, base_pert[1] * x],
                 'B1_Symmetry_1': [temperature_range, pressure_range, base_pert[0] * x],
                 #'B1_Imaginary_1': [temperature_range, pressure_range, base_pert[0] * x],
                 }

    # create Variflex classes and input files
    Model = PST.Variflex(input_file, nominal_cond, pert_cond)
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
    
    os.chdir(cwd)