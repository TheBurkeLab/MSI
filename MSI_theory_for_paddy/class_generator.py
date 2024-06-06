# -*- coding: utf-8 -*-
"""

PAPR-MESS species generator.

@author: Lei Lei
"""

import Mess_class, os, copy, io


# read in cleaned file and generate output classes
# class_generator() is for regular MESS-input and class_generator_abstraction() is for abstraction reactions
def class_generator(cleaned_file, keep_file = True):
    # define output classes
    results = {}
    save_flag = False
    cond_flag = False
    prev_species = None
    repeated_att = 0
    curr_species = Mess_class.Computation_Cond()        # the first calss has to be computation conditions
    section_order = []
    files_to_copy = []

    # commands that need sepcical treatment (has unit and accross multiple lines)
    #spec_list = ['Geometry', 'Frequencies', 'ElectronicLevels', 'FragmentGeometry', 'FourierExpansion','Potential']
    spec_list = ['Geometry', 'Frequencies', 'ElectronicLevels', 'FragmentGeometry', 'FourierExpansion']


    # read in the cleaned file
    fhand = io.open(cleaned_file, 'r')
    lines = fhand.readlines()
    fhand.close()

    # delete temporary cleaned file generated if asked
    if not keep_file:
        os.remove(cleaned_file)

    for line in lines:
        # get rid of space in the line
        line = line.replace('\t', ' ')
        #print(line)
        key, value = line.split(' ')[0], line.split(' ')[1:]

        value = [i.replace('\n', '').replace('\r', '').strip('[]\'\,') for i in value if i.strip()]
        #print(key,value)

        # determine the files to copy over
        for x in value:
            if '.dat' in x or 'File' in key:
                files_to_copy.append(x)

        # detremine if a unit is included in the line
        # com is for defining the molecular species class
        has_unit = False
        try:
            com, unit = key.split('[')[0], key.split('[')[1]
            unit = '[' + unit
            has_unit = True
        except IndexError:
            com = key

        # initialize new class and save old finished class
        #print(key)
        if "Model" == key:
            #print(key)
            save_flag = True
            cond_flag = True
            prev_species = copy.deepcopy(curr_species)
            #print(dir(prev_species))
            curr_species = Mess_class.Collision_Relaxation()
        elif "Well" in key:
            if not 'WellDepth' in key and not 'WellCutoff' in key and not 'WellExtension' in key and not 'WellReductionThreshold' in key:
            #stub made this change
            #if not 'WellDepth' in key and not 'WellCutoff' in key and not 'WellExtension' in key and not 'WellReductionThreshold':
                #print(key)
                save_flag = True
                prev_species = copy.deepcopy(curr_species)
                curr_species = Mess_class.Well()
        elif "Bimolecular" in key:
            save_flag = True
            prev_species = copy.deepcopy(curr_species)
            curr_species = Mess_class.Bimolecular()
        elif "Barrier" in key:
            save_flag = True
            prev_species = copy.deepcopy(curr_species)
            curr_species = Mess_class.Barrier()



        # avoid repeated attributes in the same class
        if com == "End":
            curr_species.__dict__[com] = value

        elif com in list(curr_species.__dict__.keys()):
            repeated_att += 1
            com = str(com) + ' ' + str(repeated_att)

        # parse the data from the input file
        # for energy, mass, and rotational constants (has unit but in the same line)
        if has_unit and not com.split(' ')[0] in spec_list:
            if len(value) > 1:
                value = [float(i) for i in value]
                value = [str(value), unit]
            else:
                value.append(unit)

            #print(key,com,value,curr_species)
            curr_species.__dict__[com] = value
            curr_species.__dict__['order'].append(com)            

            #code is going under here for values but i think we want it to go under the multi line situation


        # for geometry, frequencies, electroniclevels and hindered rotors (has unit and in multiple lines)
        elif has_unit and com.split(' ')[0] in spec_list:
            temp = {'unit' : unit}
            if 'Geometry' in com:
                mole_pool = []
                step = len(value) + 1
                # the number 4 comes from 1 atom symbol + 3 Cartesian Coordinates
                while step - 4 > 0:
                    mole = value.pop(0)
                    if len(mole_pool) == 0:
                        mole = '1 ' + mole
                    else:
                        mole = '%d ' %(int(mole_pool[-1].split(' ')[0]) + 1) + mole
                    mole_pool.append(mole)
                    temp_list = []
                    for y in range(3):
                        temp_list.append(float(value.pop(0)))
                    temp[mole] = temp_list
                    step -= 4
                temp['order'] = mole_pool

            elif 'Frequencies' in com:
                
               #print(value)
                value = [float(i) for i in value]
                temp['value'] = value
#added this 
            #elif 'Potential' in com:
            #    #print(com)
                #print(value)
            #    value = [float(i) for i in value]
            #    #print(value)
            #    temp['value']=value

            elif 'ElectronicLevels' in com or 'FourierExpansion' in com:

                temp_list = []
                for y in range(len(value) // int(2)):
                    y1 = float(value.pop(0))
                    y2 = float(value.pop(0))
                    temp_list.append(tuple([y1,y2]))
                temp['value'] = temp_list



            curr_species.__dict__[com] = temp
            curr_species.__dict__['order'].append(com)


        else:
            try:
                value = [float(i) for i in value]
            except ValueError:
                value = value

            curr_species.__dict__[com] = value
            curr_species.__dict__['order'].append(com)


        # save the classes into a dictionary

        if save_flag and cond_flag:
            results['condition'] = prev_species
            section_order.append('condition')
            save_flag = False
            cond_flag = False
        elif save_flag and hasattr(prev_species, 'Model'):

            results['col_rel'] = prev_species
            section_order.append('col_rel')
            save_flag = False


        elif save_flag and hasattr(prev_species, 'Well'):
            results[str(prev_species.Well[0])] = prev_species
            section_order.append(str(prev_species.Well[0]))
            save_flag = False
        elif save_flag and hasattr(prev_species, 'Bimolecular'):
            results[str(prev_species.Bimolecular[0])] = prev_species
            section_order.append(str(prev_species.Bimolecular[0]))
            save_flag = False
        if save_flag and hasattr(prev_species, 'Barrier'):
            results[str(prev_species.Barrier[0])] = prev_species
            section_order.append(str(prev_species.Barrier[0]))
            save_flag = False

    results[str(curr_species.Barrier[0])] = curr_species
    section_order.append(str(curr_species.Barrier[0]))

    
    #print(results)
    return results, section_order, files_to_copy

def class_generator_abstraction(cleaned_file, keep_file = False):
    # define output classes
    results = {}
    save_flag = False
    cond_flag = False
    prev_species = ''
    repeated_att = 0
    curr_species = Mess_class.Computation_Cond()        # the first calss has to be computation conditions
    section_order = []
    files_to_copy = []

    # commands that need sepcical treatment (has unit and accross multiple lines)
    spec_list = ['Geometry', 'Frequencies', 'ElectronicLevels', 'FragmentGeometry', 'FourierExpansion']

    # read in the cleaned file
    fhand = io.open(cleaned_file, 'r')
    lines = fhand.readlines()
    fhand.close()

    # delete temporary cleaned file generated if asked
    if not keep_file:
        os.remove(cleaned_file)

    for line in lines:
        # get rid of space in the line
        line = line.replace('\t', ' ')
        key, value = line.split(' ')[0], line.split(' ')[1:]
        value = [i.replace('\n', '').replace('\r', '').strip('[]\'\,') for i in value if i.strip()]

        # determine the files to copy over
        for x in value:
            if '.dat' in x or 'File' in key:
                files_to_copy.append(x)

        # detremine if a unit is included in the line
        # com is for defining the molecular species class
        has_unit = False
        try:
            com, unit = key.split('[')[0], key.split('[')[1]
            unit = '[' + unit
            has_unit = True
        except IndexError:
            com = key

        # initialize new class and save old finished class
        if "Bimolecular" in key:
            save_flag = True
            if len(section_order) == 0:
                cond_flag = True
            prev_species = copy.deepcopy(curr_species)
            curr_species = Mess_class.Bimolecular()
        elif "Barrier" in key:
            save_flag = True
            prev_species = copy.deepcopy(curr_species)
            curr_species = Mess_class.Barrier()

        # avoid repeated attributes in the same class
        if com == "End":
            curr_species.__dict__[com] = value

        elif com in list(curr_species.__dict__.keys()):
            repeated_att += 1
            com = str(com) + ' ' + str(repeated_att)

        # parse the data from the input file
        # for energy, mass, and rotational constants (has unit but in the same line)
        if has_unit and not com.split(' ')[0] in spec_list:
            if len(value) > 1:
                value = [float(i) for i in value]
                value = [str(value), unit]
            else:
                value.append(unit)

            curr_species.__dict__[com] = value
            curr_species.__dict__['order'].append(com)

        # for geometry, frequencies, electroniclevels and hindered rotors (has unit and in multiple lines)
        elif has_unit and com.split(' ')[0] in spec_list:
            temp = {'unit' : unit}
            if 'Geometry' in com:
                mole_pool = []
                step = len(value) + 1
                # the number 4 comes from 1 atom symbol + 3 Cartesian Coordinates
                while step - 4 > 0:
                    mole = value.pop(0)
                    if len(mole_pool) == 0:
                        mole = '1 ' + mole
                    else:
                        mole = '%d ' %(int(mole_pool[-1].split(' ')[0]) + 1) + mole
                    mole_pool.append(mole)
                    temp_list = []
                    for y in range(3):
                        temp_list.append(float(value.pop(0)))
                    temp[mole] = temp_list
                    step -= 4
                temp['order'] = mole_pool
            elif 'Frequencies' in com:
                value = [float(i) for i in value]
                temp['value'] = value
            elif 'ElectronicLevels' in com or 'FourierExpansion' in com:
                temp_list = []
                for y in range(len(value) / 2):
                    y1 = float(value.pop(0))
                    y2 = float(value.pop(0))
                    temp_list.append(tuple([y1,y2]))
                temp['value'] = temp_list
            curr_species.__dict__[com] = temp
            curr_species.__dict__['order'].append(com)

        else:
            try:
                value = [float(i) for i in value]
            except ValueError:
                value = value

            curr_species.__dict__[com] = value
            curr_species.__dict__['order'].append(com)

        # save the classes into a dictionary
        if save_flag and cond_flag:
            results['condition'] = prev_species
            section_order.append('condition')
            save_flag = False
            cond_flag = False
        elif save_flag and hasattr(prev_species, 'Bimolecular'):
            results[str(prev_species.Bimolecular[0])] = prev_species
            section_order.append(str(prev_species.Bimolecular[0]))
            save_flag = False
        if save_flag and hasattr(prev_species, 'Barrier'):
            results[str(prev_species.Barrier[0])] = prev_species
            section_order.append(str(prev_species.Barrier[0]))
            save_flag = False

    results[str(curr_species.Barrier[0])] = curr_species
    section_order.append(str(curr_species.Barrier[0]))
    return results, section_order, files_to_copy

