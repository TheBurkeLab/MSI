
import os
import textwrap
from string import Template
import numpy as np
import cantera as ct
from decimal import Decimal
 
def write(solution,cwd='',file_name='',original_cti=''):
    """Function to write cantera solution object to cti file.
 
    :param solution:
        Cantera solution object
 
    :return output_file_name:
        Name of trimmed mechanism file (.cti)
 
    >>> soln2cti.write(gas)
    """
    
    trimmed_solution = solution
     
    input_file_name_stripped = trimmed_solution.name
     
    if cwd !='':
        cwd =cwd
    else:
        cwd = os.getcwd()
    #pass in curret working directory , and change name to updated version?
    if file_name == '':
        output_file_name = os.path.join(
                                        cwd,
                                        'pym_' +
                                        input_file_name_stripped +
                                        '.cti')
    else:
        output_file_name = os.path.join(
                                        cwd,
                                        file_name +
                                        '.cti')
    #output_file_name = filearg
    with open(output_file_name, 'w+') as f:
 
        #Get solution temperature and pressure
        solution_temperature = trimmed_solution.T
        solution_pressure = trimmed_solution.P
 
        #Work Functions
 
        # number of calories in 1000 Joules of energy
        calories_constant = 4184.0
 
        def eliminate(input_string, char_to_replace, spaces='single'):
            """
            Eliminate characters from a string
 
            :param input_string
                string to be modified
            :param char_to_replace
                array of character strings to be removed
            """
            for char in char_to_replace:
                input_string = input_string.replace(char, "")
            if spaces == 'double':
                input_string = input_string.replace(" ", "  ")
            return input_string
 
        def wrap_nasa(input_string):
            """
            Wrap string to cti NASA format width
            """
            output_string = textwrap.fill(
                                        input_string,
                                        width=50,
                                        subsequent_indent=16 * ' '
                                            )
            return output_string
 
        def section_break(title):
            """
            Insert break and new section title into cti file
 
            :param title:
                title string for next section_break
            """
            f.write('#' + '-' * 75 + '\n')
            f.write('#  ' + title + '\n')
            f.write('#' + '-' * 75 + '\n\n')
 
        def replace_multiple(input_string, replace_list):
            """
            Replace multiple characters in a string
 
            :param input_string
                string to be modified
            :param replace list
                list containing items to be replaced (value replaces key)
            """
            for original_character, new_character in list(replace_list.items()):
                input_string = input_string.replace(original_character,
                                                    new_character)
            return input_string
 
        def build_arrhenius(equation_object, equation_type):
            """
            Builds Arrhenius coefficient string
 
            :param equation_objects
                cantera equation object1
            :param equation_type:
                string of equation type
            """
            coeff_sum = sum(equation_object.reactants.values())
            pre_exponential_factor = equation_object.rate.pre_exponential_factor
            temperature_exponent = equation_object.rate.temperature_exponent
            activation_energy = (equation_object.rate.activation_energy /
                                calories_constant)
 
            if equation_type == 'ElementaryReaction':
                if coeff_sum == 1:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor))
                if coeff_sum == 2:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**3))
                if coeff_sum == 3:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**6))
            if equation_type == 'ThreeBodyReaction':
                if coeff_sum == 1:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**3))
                if coeff_sum == 2:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**6))
 
            if (equation_type != 'ElementaryReaction'
                and equation_type != 'ThreeBodyReaction'):
                pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor))
 
            arrhenius = [pre_exponential_factor,
                        temperature_exponent,
                        activation_energy
                        ]
            return str(arrhenius).replace("\'", "")
 
        def build_modified_arrhenius(equation_object, t_range):
            """
            Builds Arrhenius coefficient strings for high and low temperature ranges
 
            :param equation_objects
                cantera equation object
            :param t_range:
                simple string ('high' or 'low') to designate temperature range
            """
 
            if t_range == 'high':
                pre_exponential_factor = equation_object.high_rate.pre_exponential_factor
                temperature_exponent = equation_object.high_rate.temperature_exponent
                activation_energy = (equation_object.high_rate.activation_energy /
                                    calories_constant)
                dictlength=0
                try:
                    dictlength=list(equation_object.products.values())[0]
                    #reaclength=equation_object.reactants.values()[0]
                except:
                    pass          
                 
                prodcount=0
                reaccount=0
                for num in np.arange(len(list(equation_object.products.values()))):
                    prodcount=prodcount+list(equation_object.products.values())[num]
                for num in np.arange(len(list(equation_object.reactants.values()))):
                    reaccount=reaccount+list(equation_object.reactants.values())[num]
                     
                     
                if prodcount==1.0 and reaccount==1.0:
                    not_single_and_equal=False
                else:
                    not_single_and_equal=True
                if len(equation_object.products) == 1 and dictlength==1.0 and not_single_and_equal:
                     
                     
                        pre_exponential_factor = str(
                                        '{:.5E}'.format(pre_exponential_factor*10**3))
                     
                else:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor))
                arrhenius_high = [pre_exponential_factor,
                                    temperature_exponent,
                                    activation_energy
                                    ]
                return str(arrhenius_high).replace("\'", "")
 
            if t_range == 'low':
                pre_exponential_factor = equation_object.low_rate.pre_exponential_factor
                temperature_exponent = equation_object.low_rate.temperature_exponent
                activation_energy = (equation_object.low_rate.activation_energy /
                                    calories_constant)
                dictlength=0
                try:
                    dictlength=list(equation_object.products.values())[0]
                except:
                    pass
                 
                prodcount=0
                reaccount=0
                for num in np.arange(len(list(equation_object.products.values()))):
                    prodcount=prodcount+list(equation_object.products.values())[num]
                for num in np.arange(len(list(equation_object.reactants.values()))):
                    reaccount=reaccount+list(equation_object.reactants.values())[num]
                     
                     
                if prodcount==1.0 and reaccount==1.0:
                    not_single_and_equal=False
                else:
                    not_single_and_equal=True
                 
                 
                if len(equation_object.products) == 1 and dictlength==1.0 and not_single_and_equal:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**6))
                else:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**3))
                arrhenius_low = [pre_exponential_factor,
                                temperature_exponent,
                                activation_energy
                                ]
                return str(arrhenius_low).replace("\'", "")
 
        def build_falloff(eq,j):
            #print(eq,j)
            if eq.falloff.type=='Troe':
                """
                Creates falloff reaction Troe parameter string
     
                param j:
                    Cantera falloff parameters object
                """
                falloff_string = str(
                            ',\n        falloff = Troe(' +
                            'A = ' + str(j[0]) +
                            ', T3 = ' + str(j[1]) +
                            ', T1 = ' + str(j[2]) + ')       )\n\n'
                            )
                if j[3]!=0.0:
                    falloff_string=falloff_string.rstrip(')       )\n\n')+', T2 = ' + str(j[3]) + ')       )\n\n'
                return falloff_string
            elif eq.falloff.type=='SRI':
                falloff_string = str(
                            ',\n        falloff = SRI(' +
                            'A = ' + str(j[0]) +
                            ', B = ' + str(j[1]) +
                            ', C = ' + str(j[2]) + ')       )\n\n'
                            )
                if j[3]!=1.0:
                    falloff_string=falloff_string.rstrip(')       )\n\n')+', D = ' + str(j[3]) + ')       )\n\n'
                 
                if j[4]!=0.0:
                    falloff_string=falloff_string.rstrip(')       )\n\n')+', E = ' + str(j[3]) + ')       )\n\n'
                 
                return falloff_string
                     
                     
        def build_species_string():
            """
            Formats species list at top of mechanism file
            """
            species_list_string = ''
            line = 1
            for sp_str in trimmed_solution.species_names:
                #get length of string next species is added
                length_new = len(sp_str)
                length_string = len(species_list_string)
                total = length_new +length_string +3
                #if string will go over width, wrap to new line
                if line == 1:
                    if total >= 55:
                        species_list_string += '\n'
                        species_list_string += ' ' * 17
                        line += 1
                if line > 1:
                    if total >= 70 * line:
                        species_list_string += '\n'
                        species_list_string += ' ' * 17
                        line += 1
                species_list_string += sp_str + ' '
            return species_list_string
 
        #Write title block to file
 
        section_break('CTI File converted from Solution Object')
        unit_string = "units(length = \"cm\", time = \"s\"," +\
                            " quantity = \"mol\", act_energy = \"cal/mol\")"
        f.write(unit_string + '\n\n')
 
        #Write Phase definition to file
 
        element_names = eliminate(str(trimmed_solution.element_names),
                                  ['[',
                                  ']',
                                  '\'',
                                  ','])
        element_names = element_names.replace('AR', 'Ar')
        species_names = build_species_string()
        phase_string = Template(
                        'ideal_gas(name = \"$input_file_name_stripped\", \n' +
                        '     elements = \"$elements\", \n' +
                        '     species =""" $species""", \n' +
                        '     reactions = \"all\", \n' +
                        '     initial_state = state(temperature = $solution_temperature, '
                        'pressure= $solution_pressure)   )       \n\n'
                       )
 
        f.write(phase_string.substitute(
                                elements=element_names,
                                species=species_names,
                                input_file_name_stripped=input_file_name_stripped,
                                solution_temperature=solution_temperature,
                                solution_pressure=solution_pressure
                                ))
 
        #Write species data to file
 
        section_break('Species data')
        for sp_index, name in enumerate(trimmed_solution.species_names):
            #joules/kelvin, boltzmann constant
            boltzmann = ct.boltzmann
            #1 debye = d coulomb-meters
            debeye_conversion_constant = 3.33564e-30
            species = trimmed_solution.species(sp_index)
            name = str(trimmed_solution.species(sp_index).name)
            nasa_coeffs = trimmed_solution.species(sp_index).thermo.coeffs
            replace_list_1 = {'{':'\"',
                              '}':'\"',
                              '\'':'',
                              ':  ':':',
                              '.0':"",
                              ',':'',
                              ' ': ' '
                              }
            #build 7-coeff NASA polynomial array
            nasa_coeffs_1 = []
            for j, k in enumerate(nasa_coeffs):
                coeff = "{:.9e}".format(nasa_coeffs[j+8])
                nasa_coeffs_1.append(coeff)
                if j == 6:
                    nasa_coeffs_1 = wrap_nasa(eliminate(str(nasa_coeffs_1),
                                                        {'\'':""}))
                    break
            nasa_coeffs_2 = []
            for j, k in enumerate(nasa_coeffs):
                coeff = "{:.9e}".format(nasa_coeffs[j+1])
                nasa_coeffs_2.append(coeff)
                if j == 6:
                    nasa_coeffs_2 = wrap_nasa(eliminate(
                                                str(nasa_coeffs_2),
                                                {'\'':""}))
                    break
            #Species attributes from trimmed solution object
            composition = replace_multiple(
                                            str(species.composition),
                                                replace_list_1)
            composition='\"'
            for el in np.arange(len(list(species.composition.keys()))):
                composition=composition+list(species.composition.keys())[el]+':'+str(int(list(species.composition.values())[el]))+' '
            composition=composition.rstrip(' ')+'\"'
            nasa_range_1 = str([species.thermo.min_temp, nasa_coeffs[0]])
            nasa_range_2 = str([nasa_coeffs[0], species.thermo.max_temp])
            #check if species has defined transport data
            if bool(species.transport) is True:
                transport_geometry = species.transport.geometry
                diameter = str(species.transport.diameter*(10**10))
                well_depth = str(species.transport.well_depth/boltzmann)
                polar = str(species.transport.polarizability*10**30)
                rot_relax = str(species.transport.rotational_relaxation)
                dipole = str(species.transport.dipole/debeye_conversion_constant)
                #create and fill string templates for each species
                if species.transport.dipole != 0:
                    species_string = Template(
                            'species(name = "$name",\n' +
                            '    atoms = $composition, \n' +
                            '    thermo = (\n' +
                            '       NASA(   $nasa_range_1, $nasa_coeffs_1  ),\n' +
                            '       NASA(   $nasa_range_2, $nasa_coeffs_2  )\n' +
                            '               ),\n'
                            '    transport = gas_transport(\n' +
                            '                   geom = \"$transport_geometry\",\n' +
                            '                   diam = $diameter, \n' +
                            '                   well_depth = $well_depth, \n' +
                            '                   polar = $polar, \n' +
                            '                   rot_relax = $rot_relax, \n' +
                            '                   dipole= $dipole) \n' +
                            '        )\n\n'
                            )
                    f.write(species_string.substitute(
                                name=name,
                                composition=composition,
                                nasa_range_1=nasa_range_1,
                                nasa_coeffs_1=nasa_coeffs_1,
                                nasa_range_2=nasa_range_2,
                                nasa_coeffs_2=nasa_coeffs_2,
                                transport_geometry=transport_geometry,
                                diameter=diameter,
                                well_depth=well_depth,
                                polar=polar,
                                rot_relax=rot_relax,
                                dipole=dipole
                                ))
                if species.transport.dipole == 0:
                    species_string = Template(
                            'species(name = "$name",\n'
                            '    atoms = $composition, \n'
                            '    thermo = (\n'
                            '       NASA(   $nasa_range_1, $nasa_coeffs_1  ),\n'
                            '       NASA(   $nasa_range_2, $nasa_coeffs_2  )\n'
                            '               ),\n'
                            '    transport = gas_transport(\n'
                            '                   geom = \"$transport_geometry\",\n'
                            '                   diam = $diameter, \n'
                            '                   well_depth = $well_depth, \n'
                            '                   polar = $polar, \n'
                            '                   rot_relax = $rot_relax) \n'
                            '        )\n\n'
                            )
                    f.write(species_string.substitute(
                                name=name,
                                composition=composition,
                                nasa_range_1=nasa_range_1,
                                nasa_coeffs_1=nasa_coeffs_1,
                                nasa_range_2=nasa_range_2,
                                nasa_coeffs_2=nasa_coeffs_2,
                                transport_geometry=transport_geometry,
                                diameter=diameter,
                                well_depth=well_depth,
                                polar=polar,
                                rot_relax=rot_relax,
                                ))
            if bool(species.transport) is False:
                species_string = Template(
                            'species(name = "$name",\n'
                            '    atoms = $composition, \n'
                            '    thermo = (\n'
                            '       NASA(   $nasa_range_1, $nasa_coeffs_1  ),\n'
                            '       NASA(   $nasa_range_2, $nasa_coeffs_2  )\n'
                            '               ),\n'
                            '        )\n\n'
                            )
                f.write(species_string.substitute(
                                name=name,
                                composition=composition,
                                nasa_range_1=nasa_range_1,
                                nasa_coeffs_1=nasa_coeffs_1,
                                nasa_range_2=nasa_range_2,
                                nasa_coeffs_2=nasa_coeffs_2,
                                ))
 
        #Write reactions to file
 
        section_break('Reaction Data')
 
        #write data for each reaction in the Solution Object
        for eq_index in range(len(trimmed_solution.reaction_equations())):
            equation_string = str(trimmed_solution.reaction_equation(eq_index))
            equation_object = trimmed_solution.reaction(eq_index)
            equation_type = type(equation_object).__name__
            m = str(eq_index+1)
            if equation_type == 'ThreeBodyReaction':
                #trimms efficiencies list
                efficiencies = equation_object.efficiencies
                trimmed_efficiencies = equation_object.efficiencies
                for s in efficiencies:
                    if s not in trimmed_solution.species_names:
                        del trimmed_efficiencies[s]
                arrhenius = build_arrhenius(equation_object, equation_type)
                replace_list_2 = {"{": "\"",
                                  "\'": "",
                                  ": ": ":",
                                  ",": " ",
                                  "}": "\""
                                  }
                efficiencies_string = replace_multiple(
                                                    str(trimmed_efficiencies),
                                                        replace_list_2)
                
                reaction_string = Template(
                            '#  Reaction $m\n'
                            'three_body_reaction( \"$equation_string\",  $Arr,\n'
                            '       efficiencies = $Efficiencies'
                            )
                

                
                #reaction_string = reaction_string
                f.write(reaction_string.substitute(
                        m=m,
                        equation_string=equation_string,
                        Arr=arrhenius,
                        Efficiencies=efficiencies_string
                        ))
                
                    
                if equation_object.duplicate is True:
                    f.write(', options = \'duplicate\')\n\n')

                else:
                    f.write(')\n\n')
                    
                    
            if equation_type == 'ElementaryReaction':
                arrhenius = build_arrhenius(equation_object, equation_type)
                if equation_object.duplicate is True:
                    reaction_string = Template(
                            '#  Reaction $m\n'
                            'reaction( \"$equation_string\", $Arr,\n'
                            '        options = \'duplicate\')\n\n'
                            )
                    if equation_object.allow_negative_pre_exponential_factor is True:
                        reaction_string = Template(
                            '#  Reaction $m\n'
                            'reaction( \"$equation_string\", $Arr,\n'
                            '        options = [\'negative_A\',\'duplicate\'])\n\n'
                            )
                else:
                    reaction_string = Template(
                            '#  Reaction $m\n'
                            'reaction( \"$equation_string\", $Arr)\n\n'
                            )
                f.write(reaction_string.substitute(
                        m=m,
                        equation_string=equation_string,
                        Arr=arrhenius
                        ))
            if equation_type == 'FalloffReaction':
                #trimms efficiencies list
                efficiencies = equation_object.efficiencies
                trimmed_efficiencies = equation_object.efficiencies
                for s in efficiencies:
                    if s not in trimmed_solution.species_names:
                        del trimmed_efficiencies[s]
                    if trimmed_efficiencies[s]==1:
                        del trimmed_efficiencies[s]
 
                kf = build_modified_arrhenius(equation_object, 'high')
                kf0 = build_modified_arrhenius(equation_object, 'low')
                replace_list_2 = {
                                "{":"\"",
                                "\'":"",
                                ": ":":",
                                ",":" ",
                                "}":"\""
                                }
                if trimmed_efficiencies!={}:
                    efficiencies_string = replace_multiple(
                                                str(trimmed_efficiencies),
                                                    replace_list_2)
                    reaction_string = Template(
                            '#  Reaction $m\n' +
                            'falloff_reaction( \"$equation_string\",\n' +
                            '        kf = $kf,\n' +
                            '        kf0   = $kf0,\n' +
                            '        efficiencies = $Efficiencies'
                            )
                    f.write(reaction_string.substitute(
                        m=m,
                        equation_string=equation_string,
                        kf=kf,
                        kf0=kf0,
                        Efficiencies=efficiencies_string
                        ))
                elif trimmed_efficiencies=={}:
                     
                    reaction_string = Template(
                            '#  Reaction $m\n' +
                            'falloff_reaction( \"$equation_string\",\n' +
                            '        kf = $kf,\n' +
                            '        kf0   = $kf0'
                            )
                    f.write(reaction_string.substitute(
                        m=m,
                        equation_string=equation_string,
                        kf=kf,
                        kf0=kf0
                        ))
                j = equation_object.falloff.parameters
                #If optional Arrhenius data included:
                 

                if equation_object.falloff.type!='Lindemann' and equation_object.falloff.type!='Simple':
                    try:
                        falloff_str = build_falloff(equation_object,j)
                        #print(equation_object.falloff.type)
                        f.write(falloff_str)
                    except IndexError:
                        f.write('\n           )\n\n')

                elif equation_object.falloff.type=='Lindemann' or equation_object.falloff.type=='Simple':
                    f.write('\n           )\n\n')
            if equation_type=='PlogReaction':
                f.write('#  Reaction '+str(m)+'\n')
                f.write('pdep_arrhenius(\''+equation_object.equation+'\',\n')
                n=len(equation_object.rates)
                count=1
                for plog in equation_object.rates:
                    #print(plog)
                    #pre_exponential_factor=plog[1].pre_exponential_factor
                    #temperature_exponent=plog[1].temperature_exponent
                    activation_energy=plog[1].activation_energy/calories_constant
                    convertedPressure=float(plog[0])*9.86923e-6
                    if sum(equation_object.reactants.values())==1.0:
                        f.write('               [('+'%s' % float('%.5g' % convertedPressure)+', \'atm\'), ')
                        f.write('%.7e' % Decimal(str(plog[1].pre_exponential_factor)))
                    elif sum(equation_object.reactants.values())==2.0:
                        f.write('               [('+'%s' % float('%.5g' % convertedPressure)+', \'atm\'), ')
                        f.write('%.7e' % Decimal(str(plog[1].pre_exponential_factor*1000.0)))
                    f.write(', '+'%s' % float('%.3g' % plog[1].temperature_exponent))
                    f.write(', '+'%s' % float('%.5g' % activation_energy))
                    f.write(']')
                    if count==n:
                        if equation_object.duplicate is True:
                            f.write(', options = \'duplicate\')\n\n')
                        else:
                            f.write(')\n\n')
                    elif count<n:
                        f.write(',\n')
                    count=count+1
            if equation_type=='ChebyshevReaction':
                f.write('#  Reaction '+str(m)+'\n')
                f.write('chebyshev_reaction(\''+equation_object.equation+'\',\n')
                f.write('                   Tmin='+'%s' % float('%.5g' % equation_object.Tmin)+', Tmax='+'%s' % float('%.5g' % equation_object.Tmax)+',\n')
                convertedPressure1=float(equation_object.Pmin)*9.86923e-6
                convertedPressure2=float(equation_object.Pmax)*9.86923e-6
                f.write('                   Pmin=('+'%s' % float('%.5g' % convertedPressure1)+', \'atm\'),')
                f.write(' Pmax=('+'%s' % float('%.5g' % convertedPressure2)+', \'atm\'),\n')
                f.write('                   coeffs=[')
                tempvar=np.arange(len(equation_object.coeffs[:,0]))
                #print(equation_object.coeffs,equation_object.equation)
                for i in np.arange(len(equation_object.coeffs[:,0])):
                    if i==0:
                        converted_first_row = equation_object.coeffs[i,:]
                        if converted_first_row.ndim ==1:
                           converted_first_row[0] = converted_first_row[0]+3
                        else:
                            converted_first_row[0,0] = converted_first_row[0,0] +3
                       # f.write(str(list(equation_object.coeffs[i,:])))
                        f.write(str(list(converted_first_row)))
                    elif i!=0 and i!=tempvar[-1]:
                        f.write('                           '+str(list(equation_object.coeffs[i,:])))
                    if i!=tempvar[-1]:
                        f.write(',\n')
                    elif i==tempvar[-1]:
                        f.write('                           '+str(list(equation_object.coeffs[i,:])))
                        f.write('])\n\n')
    return output_file_name