import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.simulations.absorbance.curve_superimpose as csp 
import MSI.simulations.yaml_parser as yp
import cantera as ct

#yaml_file_list = [('MSI/data/test_data/Hong_4.yaml','MSI/data/test_data/Hong_4_abs.yaml'),
                 # ('MSI/data/test_data/Hong_4.yaml',)]
                  
                  
yaml_file_list = [('MSI/data/low_press_high_temp_fake_simulations/Troe_4_fake_data.yaml','MSI/data/low_press_high_temp_fake_simulations/Troe_4_abs.yaml'),
                  ('MSI/data/low_press_high_temp_fake_simulations/Troe_5_fake_data.yaml','MSI/data/low_press_high_temp_fake_simulations/Troe_5_abs.yaml'),
                  ('MSI/data/low_press_high_temp_fake_simulations/Troe_6_fake_data.yaml','MSI/data/low_press_high_temp_fake_simulations/Troe_6_abs.yaml'),
                  ('MSI/data/low_press_high_temp_fake_simulations/Troe_7_fake_data.yaml','MSI/data/low_press_high_temp_fake_simulations/Troe_7_abs.yaml'),
                  ('MSI/data/low_press_high_temp_fake_simulations/Troe_8_fake_data.yaml','MSI/data/low_press_high_temp_fake_simulations/Troe_8_abs.yaml'),
                  ('MSI/data/low_press_high_temp_fake_simulations/Hong_6_fake_data.yaml','MSI/data/low_press_high_temp_fake_simulations/Hong_6_high_temp_abs.yaml')]                  
yaml_instance = yp.Parser()
list_of_yaml_objects = yaml_instance.load_yaml_list(yaml_list=yaml_file_list)
list_of_parsed_yaml = yaml_instance.parsing_multiple_dictonaries(list_of_yaml_objects = list_of_yaml_objects)

