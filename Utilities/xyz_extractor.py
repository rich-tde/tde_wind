"""
Created on Fri Feb 24 17:06:56 2023

@author: konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import h5py
from datetime import datetime
import os

#%% Get Densities

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
    # List to store the length of each rank
    lengths = []
    # List with keys that don't hold relevant data
    not_ranks = ['Box', 'Cycle', 'Time', 'mpi']
    
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            continue
        else:
            # Store the length of the dataset
            lengths.append(len(f[key]['X']))
    
    # Use lists for clarity
    X = []
    Y = []
    Z = []
    Den = []
    Vx = []
    Vy = []
    Vz = []
    Vol = []
    Mass = []
    IE = []
    T = []
    P = []
    
    # Iterate over ranks
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            continue
        else:
            # Sanity Check
            # Timing
            end_time = datetime.now()
            print('Duration: {}'.format(end_time - start_time))
            # For some reason, having the collumns into variables is way faster.
            x_data = f[key]['CMx']
            y_data = f[key]['CMy']
            z_data = f[key]['CMz']
            den_data = f[key]['Density']
            
            vx_data = f[key]['Vx']
            vy_data = f[key]['Vy']
            vz_data = f[key]['Vz']
            vol_data = f[key]['Volume']
            
            ie_data = f[key]['InternalEnergy']
            # rad_data = f[key]
            T_data = f[key]['Temperature']
            P_data = f[key]['Pressure']

            for i in range(len(P_data)):
                X.append(x_data[i])
                Y.append(y_data[i])
                Z.append(z_data[i])
                Den.append(den_data[i])
                Vx.append(vx_data[i])
                Vy.append(vy_data[i])
                Vz.append(vz_data[i])
                Vol.append(vol_data[i])
                IE.append(ie_data[i])
                Mass.append(vol_data[i] * den_data[i])
                T.append(T_data[i])
                P.append(P_data[i])


    # Close the file
    f.close()
    return X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, T, P

if __name__ == '__main__':
    name = '683'
    path = f'TDE/{name}/snap_{name}'
    X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, T, P = extractor(f'{path}.h5')

    # Save to another file.
    # np.save(f'{path}CMx', X)   
    # np.save(f'{path}CMy', Y) 
    # np.save(f'{path}CMz', Z) 
    # np.save(f'{path}Den', Den)
    # np.save(f'{path}Vx', Vx)   
    # np.save(f'{path}Vy', Vy) 
    # np.save(f'{path}Vz', Vz)
    # np.save(f'{path}Vol', Vol)
    # np.save(f'{path}Mass', Mass)   
    # np.save(f'{path}IE', IE) 
    # np.save(f'{path}T', T)
    # np.save(f'{path}P', P) 
            
