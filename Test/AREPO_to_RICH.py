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


def extractor(path, filename):
    '''
    Loads the file, extracts quantites from it. 
    '''
    # Timing start
    start_time = datetime.now()
    # Read File
    f = h5py.File(f'{path}{filename}.h5', "r")
    # HDF5 are dicts, get the keys.
    first_keys = f.keys() 
    print('keys: ', first_keys)
    for key in first_keys:
        print('Inside keys', f[key])

    # List with keys that don't hold relevant data
    not_ranks = ['Box', 'Cycle', 'Time', 'mpi']

    hf = h5py.File(f'Test/AREPO{filename}.hdf5', 'w')
    hf.create_group('Config')
    hf.create_group('Header')
    hf.create_group('Parameters')
    group = hf.create_group('PartType0')
    
    CenterOfMass = []
    # CMX = []
    # CMY = []
    # CMZ = []
    Den = []
    ParticleIDs = []
    IE = []
    P = []
    T = []
    Vol = []
    Velocities = []
    Mass = []
    # Vx = []
    # Vy = []
    # Vz = []
    Coordinates = []
    # X = []
    # Y = []
    # Z = []
    stickers = []
    tracers = []
    
    # Iterate over ranks
    for key in first_keys:
        if key in not_ranks:
            continue
        else:
            cmx_data = f[key]['CMx']
            cmy_data = f[key]['CMy']
            cmz_data = f[key]['CMz']
            den_data = f[key]['Density']
            id_data = f[key]['ID']
            ie_data = f[key]['InternalEnergy']
            P_data = f[key]['Pressure']
            T_data = f[key]['Temperature']
            vol_data = f[key]['Volume']
            vx_data = f[key]['Vx']
            vy_data = f[key]['Vy']
            vz_data = f[key]['Vz']
            x_data = f[key]['X']
            y_data = f[key]['Y']
            z_data = f[key]['Z']
            # stickers_data = f[key]['stickers']
            # tracers_data = f[key]['tracers']

            for i in range(len(den_data)):
                CenterOfMass.append([cmx_data[i], cmy_data[i], cmz_data[i]])
                # CMX.append(cmx_data[i])
                # CMY.append(cmy_data[i])
                # CMZ.append(cmz_data[i])
                Den.append(den_data[i])
                ParticleIDs.append(id_data[i])
                IE.append(ie_data[i])
                P.append(P_data[i])
                T.append(T_data[i])
                Vol.append(vol_data[i])
                Velocities.append([vx_data[i],vy_data[i],vz_data[i]])
                Mass.append(vol_data[i] * den_data[i])
                # Vx.append(vx_data[i])
                # Vy.append(vy_data[i])
                # Vz.append(vz_data[i])
                Coordinates.append([x_data[i], y_data[i], z_data[i]])
                # X.append(x_data[i])
                # Y.append(y_data[i])
                # Z.append(z_data[i])
                # stickers.append(stickers_data[i])
                # tracers.append(tracers_data[i]) 

    group.create_dataset(f'CenterOfMass', data=CenterOfMass)
    group.create_dataset(f'Coordinates', data=Coordinates)
    group.create_dataset(f'Density', data=Den)
    group.create_dataset(f'InternalEnergy', data=IE)
    group.create_dataset(f'Masses', data=Mass)
    group.create_dataset(f'ID', data=ParticleIDs)
    group.create_dataset(f'Pressure', data=P)
    group.create_dataset(f'Velocities', data=Velocities)
    # group.create_dataset(f'Temperature', data=T)
    # group.create_dataset(f'Volume', data=Vol)
    # group.create_dataset(f'stickers', data=stickers)
    # group.create_dataset(f'tracers', data=tracers)

    # Close the file
    hf.close()
    f.close()
    return 0

if __name__ == '__main__':
    name = '100'
    path = f'sedov/{name}/'
    extractor(path, f'snap_{name}')

