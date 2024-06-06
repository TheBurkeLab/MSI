# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:47:10 2018

Variflex species classes.

@author: Lei Lei
"""

class Variflex:
    # parent class of input file
    def __init__(self):
        self.Channels = Channels()
        self.order = []
        self.hasNej = False

    def change_Sym_Factor(self, percent):
        '''Change the symmetry factor of the given components.'''
        temp = self.SigRotD
        temp = float(temp.strip())
        temp = temp * (1. + percent)
        self.SigRotD = str(temp)

    def change_Vib_Freq(self, percent):
        '''Change the vibrational frequencies of the RRHO model.'''
        temp = self.ModesL
        temp = temp.strip().split(' ')
        temp = [float(x) * (1.+percent) for x in temp if len(x)>1]
        temp = ['%s' %x for x in temp]
        self.ModesL = '  '.join(temp)

    def change_Hind_Rot(self, percent):
        '''Apply a multiplicative factor to the hinder rotor polynominals.'''
        temp = self.HindParL
        temp = temp.strip().split()
        for n, x in enumerate(temp[::2]):
            temp[int(2*n)] = float(x) * (1. + percent)
        temp = ['%s' %x for x in temp]
        self.HindParL = '  '.join(temp)

    def write_component(self, component, output):
        # write a component of the Variflex input
        fhand = open(output, 'a+')
        for com in component.order:
            fhand.write(com.split('_')[0])
            fhand.write('\n')
            if hasattr(component, com):
                values = component.__dict__[com].split(';')
                for x in values:
                    fhand.write(x)
                    fhand.write('\n')
        fhand.close()

    def write_file(self, output):
        # write the entire Variflex input
        fhand = open(output, 'w+')
        fhand.close()
        for component in self.order:
            if not 'Channel' in component:
                self.write_component(self.__dict__[component], output)
            else:
                num = int(component.split('l')[-1])
                self.write_component(self.Channels.fragments['Channel%s'%num], output)
                self.write_component(self.Channels.transitions['TransState%s'%num], output)

class Title(Variflex):

    def __init__(self):
        self.order = ['*Title']

class CalculationType(Variflex):

    def __init__(self):
        self.order = ['*CalculationType']

class CalculationRanges(Variflex):

    def __init__(self):
        self.order = ['*CalculationRanges']

    def set_T_P_ranges(self, TempL, PresL):
        '''Change the calculation temperature ranges.'''
        num_temp = len(TempL)
        num_pres = len(PresL)

        # if pressure list is not specified in the original input
        # add a command into the file, note that temperature list
        # has to be in the file (otherwise cannot even run)
        if not hasattr(self, 'PRangeP ListDQ'):
            self.__dict__['PRangeP ListDQ'] = 0
            self.order.insert(1, 'PRangeP ListDQ')
            p = 2
            temp = None
            while hasattr(self,'ValuesL_%s'%p):
                try:
                    temp = self.__dict__['ValuesL_%s'%(p+1)]
                    self.__dict__['ValuesL_%s'%(p+1)] = self.__dict__['ValuesL_%s'%p]
                    self.order[self.order.index('ValuesL_%s'%p)] = 'ValuesL_%s'%(p+1)
                    p += 1
                except KeyError:
                    temp = self.__dict__['ValuesL_%s'%(p)]
                    self.order[self.order.index('ValuesL_%s'%p)] = 'ValuesL_%s'%(p+1)
                    p += 1
                    break
            self.__dict__['ValuesL_%s'%p] = temp
            self.order[self.order.index('ValuesL')] = 'ValuesL_2'
            self.order.insert(2, 'ValuesL')

        self.__dict__['TRangeP ListDQ'] = str(num_temp)
        self.__dict__['PRangeP ListDQ'] = str(num_pres)
        temp_loc = self.order.index('TRangeP ListDQ')
        pres_loc = self.order.index('PRangeP ListDQ')
        TempL = ['%s' %n for n in TempL]
        PresL = ['%s' %n for n in PresL]

        if temp_loc < pres_loc:
            self.__dict__['ValuesL'] = '  '.join(TempL)
            self.__dict__['ValuesL_2'] = '  '.join(PresL)
        else:
            self.__dict__['ValuesL_2'] = '  '.join(TempL)
            self.__dict__['ValuesL'] = '  '.join(PresL)

    def change_complex_Energy(self, amount):
        '''Change the well depth for the complex.'''
        if 'DZeroRangeP ListDQ' in self.order:
            loc = self.order.index('DZeroRangeP ListDQ')+1
            temp = float(self.__dict__[self.order[loc]])
            temp = temp + amount
            self.__dict__[self.order[loc]] = str(temp)
        if 'DZeroRangeP StepDQ' in self.order:
            loc = self.order.index('DZeroRangeP StepDQ')+1
            temp = self.__dict__[self.order[loc]].split()
            temp[0] = str(float(temp[0]) + amount)
            self.__dict__[self.order[loc]] = ', '.join(temp)


class ConvolutionRanges(Variflex):

    def __init__(self):
        self.order = ['*ConvolutionRanges']

class Collision(Variflex):

    def __init__(self):
        self.order = ['*Collision']

class Complex(Variflex):

    def __init__(self):
        self.order = ['*Complex']

class Fragments(Variflex):

    def __init__(self, num):
        self.order = ['*Channel%s' %num]

class TransState(Variflex):

    def __init__(self):
        self.order = ['*TransState']
        self.hasNej = False

    def change_Img_Freq(self, percent):
        '''Change the imaginary frequency for transition state.'''
        temp = self.EckTunD
        temp = temp.strip().split(' ')
        temp = [float(x) for x in temp if len(x)>1]
        temp[0] = temp[0] * (1. + percent)
        temp = ['%s' %x for x in temp]
        self.EckTunD = '  '.join(temp)

    def change_TS_Energy(self, amount):
        '''Change the transition state potential energy.'''
        temp = self.TSEnergyD
        temp = float(temp.strip())
        temp = temp + amount
        self.TSEnergyD = str(temp)

    def Nej_scale(self, pert_percent, num):
        '''Scale the Nej file.'''
        with open('nej.dat', 'r') as fhand:
            lines = fhand.readlines()
        barrier_start_line = 0
        while num > 1:
            line = lines[barrier_start_line].strip().split()
            barrier_start_line += 1 + int(line[0]) * int(line[1])
            num -= 1

        line = lines[barrier_start_line].strip().split()
        line[-1] = str(float(line[-1]) * (1. + pert_percent)) + '\n'
        lines[barrier_start_line] = '    '.join(line)

        with open('nej.dat', 'w') as fhand:
            fhand.writelines(lines)
			
    def change_elec_states(self, amount):
        '''Change the electronic excited state energy.'''
        temp = self.ElecStatesL
        temp = temp.strip().split(';')
        for n,x in enumerate(temp):
            x = x.strip()
            temp[n] = ' '.join([x.split(' ')[0], str(float(x.split(' ')[1]) + amount)])
        self.ElecStatesL = ';'.join(temp)


class Channels(Variflex):

    def __init__(self):
        self.fragments = {'Channel1': Fragments(1)}
        self.transitions = {'TransState1': TransState()}

    def add_new_channel(self):
        curr_channels = self.fragments.keys()
        curr_num = 1
        if len(curr_channels) == 1:
            self.fragments['Channel2'] = Fragments(2)
        elif len(curr_channels) >= 2:
            for chan in curr_channels:
                num = int(chan.split('l')[-1])
                if num > curr_num:
                    curr_num = num
                else: continue
            self.fragments['Channel%s' %(curr_num+1)] = Fragments(curr_num+1)

    def add_new_transition(self):
        curr_transitions = self.transitions.keys()
        curr_num = 1
        if len(curr_transitions) == 1:
            self.transitions['TransState2'] = TransState()
        elif len(curr_transitions) >= 2:
            for trans in curr_transitions:
                num = int(trans.split('e')[-1])
                if num > curr_num:
                    curr_num = num
                else: continue
            self.transitions['TransState%s' %(curr_num+1)] = TransState()
