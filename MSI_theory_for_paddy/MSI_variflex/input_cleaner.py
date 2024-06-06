# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:11:25 2018

Variflex input file cleaner.

@author: Lei Lei
"""


def clean_input(input_file):
    with open(input_file, 'r+') as fhand:
        lines = fhand.readlines()

    skip_line = False

    output = []
    cleaned_input = []

    for line in lines:
        line = line.strip().replace('\t', ' ')
        line = line.replace(',', ' ')
        # skip the empty lines
        if not line.strip() or line.startswith('!'):
            continue
        # get rid of all the comments
        line = line.strip().split('!')[0]
        line = line.strip()
        cleaned_input.append(line)

    for n, line in enumerate(cleaned_input):
        # generate cleaned input files
        # for commands with values in the following lines
        # use the formatt of "command: values" to facilitate
        # building up species classes in next step
        # inset a ':' if the command and values are not in the same line
        # inset a ';' if the values expands to multiple lines
        if line[-1] in ['L', 'D'] or line[-2:] == 'DQ':
            if skip_line:
                output.append(cleaned_input[n-1])
            skip_line = True
            temp = [line]
            temp.append(': ')
            continue

        if skip_line:
            temp.append(line)
            nxt_line = cleaned_input[n+1].strip()[:2]
            try:
                float(nxt_line)
            except ValueError:
                skip_line = False
                output.append(temp)
                temp = None
                continue
            temp.append('; ')
            continue

        output.append(line)

    with open('cleaned_input.txt', 'w+') as fhand:
        for line in output:
            fhand.writelines(line)
            fhand.writelines('\n')
    print("Cleaning input files...")
    return 1

#input_file = 'variflex.dat'
#clean_input(input_file)
