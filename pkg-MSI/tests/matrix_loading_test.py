import sys
sys.path.append('.') #get rid of this at some point with central test script or when package is built

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.optimization.matrix_loader as ml
import MSI.optimization.opt_runner as opt
import MSI.simulations.absorbance.curve_superimpose as csp
import MSI.simulations.yaml_parser as yp
import cantera as ct
#################################################################################
# This first test includes only one observable and no absorbance 
################################################################################
test_p = pr.Processor('MSI/data/test_data/FFCM1.cti')
test_tube = st.shockTube(pressure=1.74,
                         temperature=1880,
                         observables=['OH'],
                         kineticSens=1,
                         physicalSens=1,
                         conditions={'H2O': 0.013,'O2':.0099 ,'H':0.0000007,'Ar':0.9770993},
                         initialTime=0,
                         finalTime=0.001,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1)

csv_paths = ['MSI/data/test_data/hong_oh_1.csv']
exp_data = test_tube.importExperimentalData(csv_paths)

test_tube.run() #set up original time history

parser = yp.Parser()
#exp1_loaded = parser.load_to_obj('MSI/data/test_data/Troe_6.yaml')
#put in once opt_runner has load functionality
int_ksens_exp_mapped= test_tube.map_and_interp_ksens()#ksens is wiped on rerun so int it before
test_tube.sensitivity_adjustment(temp_del = .01)
test_tube.sensitivity_adjustment(pres_del = .01)
test_tube.species_adjustment(.01) #do some sensitivity adjustments

int_tp_psen_against_experimental = test_tube.interpolate_experimental([test_tube.interpolate_physical_sensitivities(index=1),
                                                                       test_tube.interpolate_physical_sensitivities(index=2)])

int_spec_psen_against_experimental = test_tube.interpolate_experimental(pre_interpolated=test_tube.interpolate_species_sensitivities())

#################################################################################
# This Second test includes two observables and an absorbance file 
################################################################################
test_p2 = pr.Processor('MSI/data/test_data/FFCM1.cti')
test_tube2 = st.shockTube(pressure=1.672,
                         temperature=1182,
                         observables=['H2O','OH','HO2','H2O2'],
                         kineticSens=1,
                         physicalSens=1,
                         conditions={'H2O2':0.002046 ,'H2O': 0.001113,'O2':0.000556,'Ar':0.996285},
                         initialTime=0,
                         finalTime=0.001,
                         thermalBoundary='Adiabatic',
                         mechanicalBoundary='constant pressure',
                         processor=test_p,
                         save_timeHistories=1,
                         save_physSensHistories=1)

csv_paths2 = ['MSI/data/test_data/hong_h2o_4.csv','MSI/data/test_data/hong_oh_4.csv']
exp_data2 = test_tube2.importExperimentalData(csv_paths2)

test_tube2.run() #set up original time history
abs2_instance = csp.Absorb()
abs2_loaded = parser.load_to_obj('MSI/data/test_data/Hong_4_abs.yaml')
abs2_data = abs2_instance.superimpose_shock_tube(test_tube2,abs2_loaded,15.2,kinetic_sens=1)

perturbed_coef2 = abs2_instance.perturb_abs_coef(.01,
                                          test_tube2,
                                          abs2_loaded,30,
                                          summed_data = abs2_data[0]) 

int_ksens_exp_mapped2= test_tube2.map_and_interp_ksens()#ksens is wiped on rerun so int it before
test_tube2.sensitivity_adjustment(temp_del = .01)
test_tube2.sensitivity_adjustment(pres_del = .01)
test_tube2.species_adjustment(.01) #do some sensitivity adjustments

abs2_phys_sens = abs2_instance.absorb_phys_sensitivities(test_tube2,abs2_data[0],abs2_loaded,15.2,dk=.01)

loaded_experimental_data2 = abs2_instance.import_experimental_data(['MSI/data/test_data/hong_abs_4.csv'])

interp_abs2_exp= abs2_instance.interpolate_experimental(test_tube2,loaded_experimental_data2,
                                                        original_summed_absorption=abs2_data[0],
                                                        abs_kinetic_sens = abs2_data[1],
                                                        abs_phys_sens = abs2_phys_sens,
                                                        abs_coef_sens = perturbed_coef2)


int_tp_psen_against_experimental2 = test_tube2.interpolate_experimental([test_tube2.interpolate_physical_sensitivities(index=1),
                                                                        test_tube2.interpolate_physical_sensitivities(index=2)])
int_spec_psen_against_experimental2 = test_tube2.interpolate_experimental(pre_interpolated=test_tube2.interpolate_species_sensitivities())


 ####################################
# Stick the two experiments together #
 ####################################
list_of_interpolated_kinetic_sens = [int_ksens_exp_mapped,int_ksens_exp_mapped2]
list_of_interpolated_tp_sens = [int_tp_psen_against_experimental,int_tp_psen_against_experimental2]
list_of_interpolated_species_sens = [int_spec_psen_against_experimental,int_spec_psen_against_experimental2]
#def build_single_exp_dict(self,exp_index:int,
#                          simulation:sim.instruments.shock_tube.shockTube,
#                          interpolated_kinetic_sens:dict,
#                          interpolated_tp_sens:list,
#                          interpolated_species_sens:list,
#                          interpolated_absorbance:list=[]):
optimization_instance = opt.Optimization_Utility() 
exp_1 = optimization_instance.build_single_exp_dict(1,test_tube,
                                  int_ksens_exp_mapped,
                                  int_tp_psen_against_experimental,
                                  int_spec_psen_against_experimental) #no absorbance in experiment 1
 
exp_2 = optimization_instance.build_single_exp_dict(2,test_tube2,
                                  int_ksens_exp_mapped2,
                                  int_tp_psen_against_experimental2,
                                  int_spec_psen_against_experimental2,
                                  interpolated_absorbance=interp_abs2_exp) #absorbance in experiment 2
print("Experiments built successfully")
#print(exp_2['ksens']['A'][0].shape)
#print(exp_2['species'][0].shape)
#print(exp_2['species'][1].shape)
#print(exp_2['species'][2].shape)



 ####################
# Build the S matrix #
 ####################
mloader = ml.OptMatrix()
S = mloader.load_S([exp_1,exp_2])

print(S)
