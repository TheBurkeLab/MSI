import sys
import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')
import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.optimization.matrix_loader as ml
import MSI.optimization.opt_runner as opt
import MSI.simulations.absorbance.curve_superimpose as csp
import MSI.simulations.yaml_parser as yp
import cantera as ct
#

test_p = pr.Processor('/Users/carlylagrotta/Desktop/MSI/data/test_data/FFCM1_custom.cti')
yaml_file_list = [('MSI/data/test_data/Hong_4.yaml','MSI/data/test_data/Hong_4_abs.yaml'),('MSI/data/test_data/Hong_1.yaml',),
                  ('MSI/data/test_data/Troe_6.yaml','MSI/data/test_data/Troe_6_abs.yaml'),
                  ('MSI/data/test_data/Hong_4.yaml','MSI/data/test_data/Hong_4_abs.yaml')]
yaml_instance = yp.Parser()
list_of_yaml_objects = yaml_instance.load_yaml_list(yaml_list=yaml_file_list)
list_of_experiment_dicts = yaml_instance.parsing_multiple_dictonaries(list_of_yaml_objects = list_of_yaml_objects)

optimization_instance = opt.Optimization_Utility()

test = optimization_instance.looping_over_parsed_yaml_files(list_of_experiment_dicts,
                                          yaml_file_list,
                                          processor=test_p, 
                                          kineticSens=1,
                                          physicalSens=1,
                                          dk=.01)
matix_instance = ml.OptMatrix()


Y_matrix,Y1 = matix_instance.load_Y(test,list_of_experiment_dicts,loop_counter=0)


z_matrix,z1,sigma = matix_instance.build_Z(test,list_of_experiment_dicts,loop_counter=0,reaction_uncertainty='MSI/data/test_data/uncertainty_test.csv')



x1,x2,x3,x4 = matix_instance.breakup_delta_x(z_matrix[257:],test,loop_counter=0)

S_matrix = matix_instance.load_S(test,list_of_experiment_dicts,dk=.01)

#adding target values
target_value_instance = ml.Adding_Target_Values(S_matrix,Y_matrix,z_matrix,sigma)
k_targets_for_y = target_value_instance.target_values_Y('MSI/data/test_data/FFCM1_target_values.csv',test)
k_targets_for_z,sigma = target_value_instance.target_values_for_Z('MSI/data/test_data/FFCM1_target_values.csv')
s_target_values = target_value_instance.target_values_for_S('MSI/data/test_data/FFCM1_target_values.csv',test)


