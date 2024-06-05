"""

Script to initialize parameters given in master YAML file.

@author: Patrick Singal
"""

import ext.pyyaml.yaml as yaml
import numpy as np

class yaml_parser():
    def __init__(self,fname): #initialize mandatory information from yaml
        with open (fname) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.plot_only = data["MSI-options"]["plot_only"]
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