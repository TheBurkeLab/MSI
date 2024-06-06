# -*- coding: utf-8 -*-
"""

Extracte temperature-dependent rate coefficients from PAPR-MESS output for a specific pressure.

@author: Lei Lei
"""

import io
import sys
import pandas as pd
import os


print('HELLLLLLLLLLLLLLLLLLLLLLLLLLO')

def T_rate_extractor(file_name, P, reactant, product):
    #print(file_name ,'THIS IS THE FILE NAME')
    #print(P)


    print('GOOOOOOOOOOOOOOOOOOODBYE1')

    channel = reactant + '->' + product
    rate = []
    temp = []
    fhand = io.open(file_name, 'rb')

    

    lines = fhand.readlines()

    # print(lines)

    fhand.close()

    # print(lines)

    startline = 1e10
    endline = 1e10
    T_section = False
    section = False
    channel_flag = False
    channel_section=False
    T_header = 'Temperature-Species Rate Tables:'
    P_header = 'Pressure = '
    P, unit = float(P.split()[0]), P.split()[1]
    # locate the rate coefficients lines

    # print(lines)

    for num, line in enumerate(lines):

        # print(line)

        line = line.strip('\n')

        # print(line)


        if T_header in line:
            T_section = True
            continue
        if T_section and P_header in line:
            temp_line = line.strip().split()
            if float(temp_line[-2]) == P and temp_line[-1] == unit:
                
                section = True
                #T_section = False
                if channel in lines[num+2]:
                    channel_section=True
                else:
                    channel_section=False
                continue

        #if section and channel in line:
        if channel_section==True and section==True and channel in line:    
            #print(P,channel,num)
            channel_line = [i for i in line.split(' ') if len(i) != 0]
            rate_loc = channel_line.index(channel)

            print(rate_loc)

            startline = num + 1
            #print(startline,'THIS IS THE START LINE')
            channel_flag = True
            section = False
            continue

        if channel_flag and not line.strip():
            endline = num - 1
            channel_flag = False
            continue
    # extract corresponding rate constants
    for num, line in enumerate(lines):
        if num >= startline and num <= endline:
            line = line.strip('\n')
            line = [i for i in line.split(' ') if len(i) != 0]

            # print(rate_loc)
            # print(line)
            # print(line[rate_loc])


            ###################################### WTF
            rate_loc = 3

            rate.append(float(line[rate_loc]))
            temp.append(float(line[0]))
    


    return temp, rate

# for abstraction reactions
def T_rate_abstraction(file_name, reactant, product):



    print('GOOOOOOOOOOOOOOOOOOODBYE2')

    channel = reactant + '->' + product
    rate = []
    temp = []
    fhand = io.open(file_name, 'rb')
    lines = fhand.readlines()
    fhand.close()
    startline = 1e10
    endline = 1e10
    channel_flag = False
    T_header = 'Temperature-Species Rate Tables:'
    # locate the rate coefficients lines
    for num, line in enumerate(lines):
        line = line.strip('\n')
        if channel in line:
            channel_line = [i for i in line.split(' ') if len(i) != 0]
            rate_loc = channel_line.index(channel) - 1 # to compensate for T, K
            startline = num + 1
            channel_flag = True
            continue
        if channel_flag and not line.strip():
            endline = num - 1
            channel_flag = False
            continue
    # extract corresponding rate constants
    for num, line in enumerate(lines):
        if num >= startline and num <= endline:
            line = line.strip('\n')
            line = [i for i in line.split(' ') if len(i) != 0]
            rate.append(float(line[rate_loc]))
            temp.append(float(line[0]))
    return temp, rate

# extract rate constants

print('GOOOOOOOOOOOOOOOOOOODBYE3')

abstraction = (sys.argv[6].lower() == 'true')

print('GOOOOOOOOOOOOOOOOOOODBYE4')

if abstraction:
    temp, rate = T_rate_abstraction(sys.argv[1], sys.argv[3], sys.argv[4])


    print('GOOOOOOOOOOOOOOOOOOODBYE5')  

else:

    print('GOOOOOOOOOOOOOOOOOOODBYE6')

    temp, rate = T_rate_extractor(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    print('GOOOOOOOOOOOOOOOOOOODBYE7')


print('GOOOOOOOOOOOOOOOOOOODBYE8')


output = pd.DataFrame([])

print(output)

channel = sys.argv[3] + '->' + sys.argv[4]
output['T'] = temp
output[channel] = rate

cwd = os.getcwd()
os.chdir(sys.argv[5])    # change directory to the perturbed working directory
if os.path.exists('T_rate.csv'):
    fhand = io.open('T_rate.csv', 'ab')
else:
    fhand = io.open('T_rate.csv', 'wb')
fhand.write('=' * 40 + '\n')
fhand.write(sys.argv[1] + '\n')
fhand.write(sys.argv[2] + '\n')
fhand.close()

output.to_csv('T_rate.csv', index=False, mode='a')

os.chdir(cwd)
