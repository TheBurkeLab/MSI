# -*- coding: utf-8 -*-
"""

Read in cleaned input and generate the sepcies classes

@author: Lei Lei
"""

import Variflex_class as vc

def generate_class(cleaned_file = 'cleaned_input.txt'):

    result = vc.Variflex()

    with open(cleaned_file, 'r+') as fhand:
        lines = fhand.readlines()

    curr_class = None

    spec_commands = ['PositionL', 'HindParL', 'ElecStatesL']

    for line in lines:
        line = line.strip()

        if 'ReadNP' in line:
            curr_class.hasNej = True
            result.hasNej = True

        # create classes for each component
        if '*' in line:
            result.order.append(line[1:])
            if 'Title' in line:
                curr_class = vc.Title()
                continue
            if 'CalculationType' in line:
                result.Title = curr_class
                curr_class = vc.CalculationType()
                continue
            if 'CalculationRanges' in line:
                result.CalculationType = curr_class
                curr_class = vc.CalculationRanges()
                continue
            if 'Convolution' in line:
                result.CalculationRanges = curr_class
                curr_class = vc.ConvolutionRanges()
                continue
            # input file either has both of Collision and Complex together
            # or neither of them
            if 'Collision' in line:
                result.ConvolutionRanges = curr_class
                curr_class = vc.Collision()
                continue
            if 'Complex' in line:
                #### Fix for single well system
                # result.CalculationType.order.insert(-1, 'WellWellConnectD')
                # result.CalculationType.__dict__['WellWellConnectD'] = ' '
                ###
                if "Collision" in result.order:
                    result.Collision = curr_class
                else:
                    result.ConvolutionRanges = curr_class
                curr_class = vc.Complex()
                continue
            # input file can circumvent potential well
            if 'Channel' in line:
                num = int(line.split('l')[-1])
                if not 'Complex' in result.order:
                    result.ConvolutionRanges = curr_class
                elif num == 1:
                    result.Complex = curr_class
                if num == 1:
                    curr_class = result.Channels.fragments['Channel1']
                    continue
                if num > 1:
                    result.Channels.transitions['TransState%s' %(num-1)] = curr_class
                    result.Channels.add_new_channel()
                    curr_class = result.Channels.fragments['Channel%s' %num]
                    continue
            if 'Fragments' in line:
                result.order.remove('Fragments')
            if 'TransState' in line:
                result.order.remove('TransState')
                result.Channels.fragments['Channel%s' %num] = curr_class
                if num == 1:
                    curr_class = result.Channels.transitions['TransState1']
                    continue
                if num > 1:
                    result.Channels.add_new_transition()
                    curr_class = result.Channels.transitions['TransState%s' %num]
                    continue
            if 'End' in line:
                result.order.remove('End')
                result.Channels.transitions['TransState%s' %num] = curr_class


        # after clean-up, there are only three types of commands in the file,
        # 1. pure commands without values (no ':' or ';')
        # 2. commands with values in the following line (with ':' but no ';')
        # 3. commands with values in multiple lines (with ':' and ';', where ';' is
        #    used to separate different lines)

        if not ':' in line:
            curr_class.order.append(line)
        else:
            key, values = line.split(':')[0], line.split(':')[1]
            if hasattr(curr_class, key):
                n = 2
                old_key = key
                while True:
                    key =  old_key + '_%s'%n
                    if not hasattr(curr_class, key):
                        break
                    n += 1
            curr_class.order.append(key)
            if key.split('_')[0] in spec_commands:
                curr_class.__dict__[key] = values
            else:
                values = values.replace(';', ' ')
                curr_class.__dict__[key] = values
    print('Generating python classes...')
    return result

if __name__ == "__main__":
    result = generate_class()
    result.write_file('test.dat')
