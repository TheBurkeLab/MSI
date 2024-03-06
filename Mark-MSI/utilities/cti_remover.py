import cantera as ct
import re
import sys
import soln2cti
if len(sys.argv) != 3:
    print("error: must provide single argument for element and argument for cti file path")
else:
    element = sys.argv[1]
    cti_file_path = sys.argv[2]
    cti_path_clean =sys.argv[2].split(".cti")[0]+"_clean.cti"
    mechanism_to_clean = ct.Solution(cti_file_path)
    print(mechanism_to_clean.n_total_species, "total species, ",mechanism_to_clean.n_reactions," reactions.")
    print("Cleaning all species and reactions with "+element)
    clean_species=()
    clean_reactions=()
    #loop once over the species
    for i in mechanism_to_clean.species():
        if not element in i.name:
            clean_species+=(i,)
    #loop once over the reactions
    for i in range(0,mechanism_to_clean.n_reactions):
        if not element in mechanism_to_clean.reaction_equation(i):
            reaction = mechanism_to_clean.reaction(i)
            if hasattr(reaction,'efficiencies'):
                copy = {}
                for key in reaction.efficiencies.keys():
                    if not re.search(element,key):
                        copy.update({key:reaction.efficiencies[key]})
                reaction.efficiencies=copy
            clean_reactions+=(mechanism_to_clean.reaction(i),)
            
    #print(mechanism_to_clean.element_names)
    clean_mechanism=ct.Solution(thermo='IdealGas',
                                kinetics='GasKinetics',
                                species=clean_species,
                                reactions=clean_reactions)
    
    #print(clean_species)
    for x in clean_mechanism.reactions():
       print(x,'\n')
    print("writing cleaned file at ", cti_path_clean)
    soln2cti.write(clean_mechanism,cti_path_clean)
