import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.simulations.absorbance.curve_superimpose as csp 
import MSI.simulations.yaml_parser as yp
import cantera as ct


test_p = pr.Processor('MSI/data/test_data/optimized_burke.cti')
test_tube = st.shockTube(pressure=3.44187,
                         temperature=1079,
                         observables=['H2O2','HO2'],
                         kineticSens=1,
                         physicalSens=0,
                         conditions={'H2O2':0.00195373,'Ar':0.99804627},
                         initialTime=0,
                         finalTime=0.0014,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1)

test_tube.run()
abs_instance = csp.Absorb()
parser = yp.Parser()
exp_loaded = parser.load_to_obj('MSI/data/test_data/Troe_6.yaml')
abs_loaded = parser.load_to_obj('MSI/data/test_data/Troe_6_abs.yaml')
abs_data = abs_instance.superimpose_shock_tube(test_tube,abs_loaded,30,kinetic_sens=1)
perturbed_data = abs_instance.perturb_abs_coef(.01,
                                          test_tube,
                                          abs_loaded,30,
                                          summed_data = abs_data[0])


test_tube.sensitivity_adjustment(temp_del=.01)
test_tube.sensitivity_adjustment(pres_del=.01)
test_tube.species_adjustment(.01)

abs_phys_sens= abs_instance.absorb_phys_sensitivities(test_tube,abs_data[0],abs_loaded,30,dk=.01)
#print("PERTURBED DATA FIRST:", perturbed_data[0])
#print("FIRST SUMMED DATA:", abs_data[0])

loaded_experimental_data = abs_instance.import_experimental_data(['MSI/data/test_data/tro_6_abs_1.csv',
                                                                  ])
interp_exp_data = abs_instance.interpolate_experimental(test_tube,loaded_experimental_data,
                                                        original_summed_absorption=abs_data[0],
                                                        abs_kinetic_sens = abs_data[1],
                                                        abs_phys_sens = abs_phys_sens,
                                                        abs_coef_sens = perturbed_data)

