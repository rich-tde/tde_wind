# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:51:56 2022

@author: Konstantinos


"""
#%% Imports
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from datetime import datetime
import h5py
import prelude

#%% Extract Energy
# snapshot233= "/Users/paolamartire/data_sim/.h5"
# snapshot254= "/Users/paolamartire/data_sim/.h5"
# snapshot263 = "/Users/paolamartire/data_sim/.h5"
# snapshot277 = "/Users/paolamartire/data_sim/.h5"
#%% Get Energies

## File structure is
# box, cycle, time, mpi, rank0 ... rank99.
# This iterates over all the ranks

def days_since_distruption(filename):
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
    # Timing start
    start_time = datetime.now()
    # Read File
    f = h5py.File(filename, "r")
    G = 6.6743e-11 # SI
    Msol = 1.98847e30 # kg
    Rsol = 6.957e8 # m
    t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G=1
    Mbh = 1e6 # * Msol
    time = np.array(f['Time'])
    days = time.sum()*t / (24*60*60)
    return days


#%% 
# def linear_fit_days(x):
#     '''
#     Converts from snapshot number to the more 
#     intuitive days since distruption metric. 
    
#     Uses a linear fit from snapshots 243, 881, and,
#     254 and thus could prove to not be 100% accurate.
    
#     Parameters
#     ----------
#     x : int,
#         Snapshot number to convert from.

#     Returns
#     -------
#     y : int,
#         Days since distruption.

#     '''
#     days233= days_since_distruption(snapshot233)
#     days254 = days_since_distruption(snapshot254)
#     days263 = days_since_distruption(snapshot263)
#     days277 = days_since_distruption(snapshot277)
#     snaps = [233, 254, 263, 277]
#     days = [days233, days254, days263, days277]
#     time_fit = np.polyfit(snaps, days, deg=1)
#     y = time_fit[0]*x + time_fit[1]
#     return y
#%%
if __name__ == '__main__':
    # days322 = linear_fit_days(322)
    time = days_since_distruption('data_sim/sedov_100.h5')
    print(time)
        
    
    
    
    
    
    
    
    
    
