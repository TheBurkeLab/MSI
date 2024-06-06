# -*- coding: utf-8 -*-
"""
Preprocessor for PAPR-MESS input files.

Input: nominal PAPR-MESS input files
Output: species classes for stationary points on the PES

@author: Lei Lei
"""

from input_cleaner import file_cleaner
from class_generator import *

class Preprocessor:
    """Parent class for preprocessing, used to obtain the species classes."""

    def __init__(self, nominal_file):
        self.nominal_file = nominal_file

    def clean_input(self):
        print(("Cleaning input file for %s..." %self.nominal_file.split(".")[0]))
        self.cleaned_file = file_cleaner(self.nominal_file)
        return 1

    def generate_species_classes(self, abstraction=False):
        print(("Generating PAPR-MESS classes for %s..." %self.nominal_file.split(".")[0]))
        if not abstraction:
            self.species_classes, self.section_order, self.files_to_copy = class_generator(self.cleaned_file)
        else:
            self.species_classes, self.section_order, self.files_to_copy = class_generator_abstraction(self.cleaned_file)
        return 1
    
#nominal_species = Preprocessor('CH2O-1.inp')
#nominal_species.clean_input()
#nominal_species.generate_species_class()
