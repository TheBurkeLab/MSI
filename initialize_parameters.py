"""

Script to initialize parameters given in master YAML file.

@author: Patrick Singal
"""

import ext.pyyaml.yaml as yaml
import numpy as np
import sys, os

class yaml_parser():
    def __init__(self,fname): #initialize mandatory information from yaml
        with open (fname) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        # Input data directory
        workspace_path=data["path_to_workspace"]
        experiments_path=workspace_path+"\\input_data\\targets\\experiments"
        reactions_to_plot_path=workspace_path+"\\input_data\\targets\\rate_constants\\reactions_to_plot"
        reactions_targets_path=workspace_path+"\\input_data\\targets\\rate_constants\\reaction_targets"
        mechanism_files_path=workspace_path+"\\input_data\\mechanism\\files"
        mechanism_uncertainties_model_path=workspace_path+"\\input_data\\mechanism\\uncertainties\\model"
        mechanism_uncertainties_real_path=workspace_path+"\\input_data\\mechanism\\uncertainties\\real"
        sys.path.append(workspace_path) 
        os.makedirs(experiments_path,exist_ok=True) #makes directory with this name if it doesn't already exist
        os.makedirs(reactions_to_plot_path,exist_ok=True)
        os.makedirs(reactions_targets_path,exist_ok=True)
        os.makedirs(mechanism_files_path,exist_ok=True)
        os.makedirs(mechanism_uncertainties_model_path,exist_ok=True)
        os.makedirs(mechanism_uncertainties_real_path,exist_ok=True)

        # Optimization results directoruy
        workspace_path=data["path_to_workspace"]
        experiments_path=workspace_path+"\\input_data\\targets\\experiments"
        reactions_to_plot_path=workspace_path+"\\input_data\\targets\\rate_constants\\reactions_to_plot"
        reactions_targets_path=workspace_path+"\\input_data\\targets\\rate_constants\\reaction_targets"
        mechanism_files_path=workspace_path+"\\input_data\\mechanism\\files"
        mechanism_uncertainties_model_path=workspace_path+"\\input_data\\mechanism\\uncertainties\\model"
        mechanism_uncertainties_real_path=workspace_path+"\\input_data\\mechanism\\uncertainties\\real"


        self.plot_only = data["MSI-options"]["plot_only"]
        if self.plot_only==False:
            #start structuring directory
            main_directory = os.path.dirname(os.path.abspath(__file__)) 
            test_directory = os.path.join(main_directory,'test')
            test_directory_list = [f for f in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory,f))]
            test_number_list = [eval(f.split('test')[1]) for f in test_directory_list if f.split('test')[1].isnumeric()]
            current_test_directory =  'test' + str(max(test_number_list)+1)
            working_directory =  os.path.join(test_directory, current_test_directory) # /home/jl/main/test/test#
            if not os.path.exists(working_directory):
                os.makedirs(working_directory)
            os.chdir(working_directory)
            matrix_path = os.path.join(working_directory,'matrix')
            if not os.path.exists(matrix_path):
                os.makedirs(matrix_path)
            out_path = os.path.join(working_directory,'out')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            data_path = os.path.join(working_directory,'data')
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            yaml_path = os.path.join(working_directory,'yaml')
            if not os.path.exists(yaml_path):
                os.makedirs(yaml_path)




        self.number_of_iterations = data["MSI-options"]["number_of_iterations"]
        self.step_size = data["MSI-options"]["step_size"]
        self.cti_file = data["mechanism"]["files"]
        self.model_uncertainty_csv = data["mechanism"]["uncertainties"]["model"]
        self.real_uncertainty_csv = data["mechanism"]["uncertainties"]["real"]
        
        if "X-prior" in data["MSI-options"]:
            self.X_prior = data["MSI-options"]["X-prior"]
        
        if "reactions_to_plot" in data["targets"]["rate-constants"]:
            self.rate_constant_plots_csv = data["targets"]["rate-constants"]["reactions_to_plot"]     

        if "reaction_targets" in data["targets"]["rate-constants"]:
            self.rate_constant_target_csv = data["targets"]["rate-constants"]["reaction_targets"]     

        if "master-equation" in data:
            self.master_reaction_equation_cti_name = data["master-equation"]["files"]
            self.master_equation_uncertainty_csv = data["master-equation"]["uncertainties"]
            self.master_equation_reactions=[]
            self.master_index=[]
            self.T_P_min_max_dict={}
            self.cheb_sensitivity_dict={}
            for i in range(len(data["master-equation"]["reactions"])):
                self.master_equation_reactions.append(data["master-equation"]["reactions"][i]["equation"])
                self.master_index.append(data["master-equation"]["reactions"][i]["idx"])
                name = data["master-equation"]["reactions"][i]["equation"]
                T_min=data["master-equation"]["reactions"][i]["temperature-range"][0]
                T_max=data["master-equation"]["reactions"][i]["temperature-range"][1]
                P_min=data["master-equation"]["reactions"][i]["pressure-range"][0]
                P_max=data["master-equation"]["reactions"][i]["pressure-range"][1]
                self.T_P_min_max_dict[name]={'T_min':T_min, 'T_max':T_max, 'P_min':P_min, 'P_max':P_max}

                cheb_data=[]
                for j in range(len(data["master-equation"]["reactions"][i]["data"])):
                    cheb_data.append(np.array(data["master-equation"]["reactions"][i]["data"][j]))   
                self.cheb_sensitivity_dict[name]=cheb_data


        self.cti_file = data["mechanism"]["files"]
        self.experiments = data["targets"]["experiments"]

        #Check if valid inputs have been given
        if type(self.plot_only)!=bool:
            raise TypeError("The 'plot_only' key must be of type bool")
        if type(self.experiments)!=list:
            raise TypeError("The 'experiments' key must be of type list")

print(yaml_parser("newYaml.yaml").cheb_sensitivity_dict)