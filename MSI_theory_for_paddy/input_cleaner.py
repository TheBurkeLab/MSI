# -*- coding: utf-8 -*-
"""

PAPR-MESS input file cleaner.

@author: Lei Lei
"""

import io

# read input file and get rid of the comments and unnecessary commands
def file_cleaner(file_name, simplifid = False):
    fhand = io.open(file_name, "r")
    lines = fhand.readlines()
    fhand.close()
    cleaned_file = 'cleaned_input.txt'
    cleaned_input = io.open(cleaned_file, 'w')
    spec_list = ["Geometry[angstrom]", "ElectronicLevels[1/cm]", "FourierExpansion[1/cm]", "FragmentGeometry[angstrom]", "FourierExpansion[kcal/mol]"]
    #, "Potential[kcal/mol]"
    skip_start = -1
    skip_end = -1
    comment_symbol = ['!', '#']
    
    for num, line in enumerate(lines):
        if num >= skip_start and num < skip_end:
            continue
        else:
            # replace every special spacing character to spcae
            print(line)
            line = line.replace('\t', ' ').replace('\n', '').replace('\r', '')
            line = line.strip(' ')
            # Skip empty lines and comment lines
            if not line.strip() or line.lstrip(' ').startswith(('!','#')):
                continue
            # Get info. from lines
            else:
                # Get rid of the comments
                for c in comment_symbol:
                    line = line.split(c)[0]
                
                key,value = line.split(" ")[0], line.split(" ")[1:]
                value = [i for i in value if i.strip()]
                # Special treat for Geometry, Frequencies and ElectronicLevels
                if key in spec_list:
                    skip_start = num + 1
                    skip_end = num + int(value[0]) + 1
                    temp_list = []
                    for x in range(skip_start, skip_end):
                        temp = lines[x].replace('\t', ' ').replace('\n', '').replace('\r', '')
                        temp = temp.strip(' ')
                        temp_list.append(temp)
                    value = temp_list
                elif 'Frequencies' in key:
                    skip_start = num + 1
                    temp_list = []
                    while True:
                        temp = lines[skip_start].replace('\t', ' ').replace('\n', '').replace('\r', '')
                        temp = temp.strip(' ').split(' ')
                        temp = [i for i in temp if i.strip()]
                        try: 
                            float(temp[0])
                            temp_list.append(temp[::1])
                            skip_start += 1
                        except ValueError:
                            skip_end = skip_start
                            skip_start = num + 1
                            break
                    value = temp_list
                elif 'WellDepth' in key:
                    skip_start = num
                    skip_end = num + 2
                    temp_list = []
                    for x in range(skip_start, skip_end):
                        temp = lines[x].split(']')[-1]
                        temp = temp.strip()
                        temp_list.append(temp)
                    value = temp_list

                # write the cleaned file
                # if simplified == True, skip all lines without input parameters
                if simplifid and len(value) > 0:
                    cleaned_input.write(key)
                    cleaned_input.write("\t\t")
                    cleaned_input.write(str(value))
                    cleaned_input.write('\n')
                # for a complete input file, using simplified = False
                elif simplifid == False:
                    cleaned_input.write(key)
                    cleaned_input.write("\t\t")
                    cleaned_input.write(str(value))
                    cleaned_input.write('\n')
    return cleaned_file


