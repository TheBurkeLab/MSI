import sys, os
sys.path.append('../../') #get rid of this at some point with central test script or when package is built
os.chdir('../../')

import MSI.simulations.instruments.shock_tube as st
import MSI.cti_core.cti_processor as pr
import MSI.simulations.yaml_parser as yp
import cantera as ct

parser = yp.Parser()
                   
exp = parser.load_to_obj('MSI/data/test_data/Hong_5.yaml')
absp = parser.load_to_obj('MSI/data/test_data/Troe_8_abs_updated.yaml')
loaded_tube = parser.parse_shock_tube_obj(loaded_exp=exp, loaded_absorption=absp)
print(loaded_tube)
