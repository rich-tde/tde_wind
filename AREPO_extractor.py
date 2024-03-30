"""
Created on Fri Feb 24 17:06:56 2023

@author: konstantinos, paola 
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import h5py
from datetime import datetime
import os


## File structure is
# box, cycle, time, mpi, rank0 ... rank99.
# This iterates over all the ranks


def extractor(filename):
    '''
    Loads the file, extracts quantites from it. 
    '''
    # Timing start
    start_time = datetime.now()
    # Read File
    f = h5py.File(filename, "r")
    # HDF5 are dicts, get the keys.
    keys = f.keys() 
    print('In the file: ', f)
    print('Keys of f: ', keys)
    # List with keys that don't hold relevant data
    ranks = ['Config', 'Header', 'Parameters', 'PartType0']
    sub_keys = f['PartType0'].keys()
    print('sub keys: ', sub_keys)
    
    for skey in sub_keys:
        print(f['PartType0'][skey])

    # Close the file
    f.close()
    return 1


extractor(f'AREPOsedov100.hdf5')


