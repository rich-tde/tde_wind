# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:51:56 2022

@author: Konstantinos


"""
#%% Imports
import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()

import numpy as np
import h5py

## File structure is
# box, cycle, time, mpi, rank0 ... rank99.
# This iterates over all the ranks

def days_since_distruption(filename, m, mstar, rstar, choose = 'day'):
    '''
    Loads the file, extracts specific kinetic and potential energies 
    
    Parameters
    ----------
    f : str, 
        hdf5 file name. Contains the data
    
    Returns
    -------
    days: float, days since the distruption begun.
    
    '''
    # Read File
    f = h5py.File(filename, "r")
    G = 6.6743e-11 # SI
    Msol = 1.98847e30 # kg
    Rsol = 6.957e8 # m
    t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G=1
    Mbh = 10**m # * Msol
    time = np.array(f['Time'])
    time = time.sum()
    days = time*t / (24*60*60)
    t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(rstar, 3/2)
    # print(f'days after disruption: {days} // t_fall: {t_fall} // sim_time: {time}')
    if choose == 'tfb':
        days /= t_fall
    return days

#%%
if __name__ == '__main__':
    choose = 'tfb'
    snap = 216
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    check = ''
    if alice:
            prepath = f'/data1/martirep/shocks/shock_capturing/'
    else: 
        prepath = f'/Users/paolamartire/shocks/TDE'

    path = f'{prepath}/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{check}/{snap}'
    days = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose)
    if choose == 'tfb':
        print(f'In fallback time: {days}')
        
    
    
    
    
    
    
    
    
    
