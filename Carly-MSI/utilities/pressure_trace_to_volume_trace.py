import cantera as ct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pressure_trace = pd.read_csv('/Users/carlylagrotta/Dropbox/Columbia/MSI/data/variable_pressure_shock_tube/pressure_trace_2.csv',
                             float_precision='round_trip')
Pressure_array=pressure_trace['Pressure'].to_numpy()
plt.figure()
plt.plot(pressure_trace['Time'],pressure_trace['Pressure'].to_numpy(),label='Data Given On Github Taken From Experiment')
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]')
plt.legend()
class VolumeFromPressure(object):
    r"""Create a volume trace given a pressure trace.
    Using Cantera to evaluate the thermodynamic properties, compute a
    volume trace from a pressure trace.
    Parameters
    ----------
    pressure : `numpy.ndarray`
        1-D array containing the reactor pressure
    v_initial : `float`
        Initial volume of the experiment, in m**3
    T_initial : `float`, optional
        Initial temperature of the experiment, in Kelvin. Optional for
        Cantera versions greater than 2.2.0.
    chem_file : `str`, optional
        Filename of the chemistry file to be used
    Attributes
    ----------
    volume : `numpy.ndarray`
        The volume trace
    Notes
    -----
    The volume is computed according to the formula
    .. math:: v_i = v_{initial}*\rho_{initial}/\rho_i
    where the index :math:`i` indicates the current point. The state
    is set at each point by setting the pressure from the input array
    and the entropy to be constant. The volume is computed by the
    isentropic relationship described above.
    """
    def __init__(self, pressure, v_initial, T_initial, chem_file='species.cti', cti_source=None,time=None):
        initial_density = []
        gas_density = []
        
        one_bar_in_pa = 100000
        if cti_source is None:
            gas = ct.Solution(chem_file)
        else:
            gas = ct.Solution(source=cti_source)
        gas.TP = T_initial, pressure[0]*one_bar_in_pa
        initial_entropy = gas.entropy_mass
        initial_density = gas.density
        
        print(initial_entropy,initial_density)

        self.volume = np.zeros((len(pressure)))
        for i, p in enumerate(pressure):
            gas.SP = initial_entropy, p*one_bar_in_pa
            self.volume[i] = v_initial*initial_density/gas.density
            gas_density.append(gas.density)
        
        plt.figure()    
        plt.plot(time,gas_density)
        plt.ylabel('density')
    def __repr__(self):
        return 'VolumeFromPressure(volume={self.volume!r})'.format(self=self)


instance = VolumeFromPressure(Pressure_array,
                              1, 1400,
                              chem_file ='/Users/carlylagrotta/Dropbox/Columbia/MSI/data/RCM/mech.cti',
                              time=pressure_trace['Time'])
print(instance.volume)

df = pd.DataFrame(columns=['Time','Volume'])
df['Volume']=instance.volume
df['Time']=pressure_trace['Time'].values
df.to_csv('/Users/carlylagrotta/Dropbox/Columbia/MSI/data/variable_pressure_shock_tube/volume_trace_file_2.csv',index=False)

cantera_calculated_volume_trace = pd.read_csv('/Users/carlylagrotta/Dropbox/Columbia/MSI/data/variable_pressure_shock_tube/volume_trace_file_2.csv',float_precision='round_trip')
plt.figure()
plt.plot(cantera_calculated_volume_trace['Time'],cantera_calculated_volume_trace['Volume'],label='Data Given On Github')
plt.plot(pressure_trace['Time'],instance.volume,label='Calculated Here')
plt.xlabel('Time [s]')
plt.ylabel('Volume [m**3]')
plt.legend()