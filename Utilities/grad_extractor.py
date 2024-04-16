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


def extractor(folder, filename):
    '''
    Loads the file, extracts quantites from it. 
    '''
    # Timing start
    start_time = datetime.now()
    # Read File
    f = h5py.File(filename, "r")
    # HDF5 are dicts, get the keys.
    keys = f.keys() 
    # List with keys that don't hold relevant data
    not_ranks = ['Box', 'Cycle', 'Time', 'mpi']
    
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
    Diss = []
    
    DrhoDx = []
    DrhoDy = []
    DrhoDz = []
    DrhoDxLimited = []
    DrhoDyLimited =[]
    DrhoDzLimited = []

    DpDx = []
    DpDy = []
    DpDz = []
    DpDxLimited = []
    DpDyLimited =[]
    DpDzLimited = []

    divV = []
    divVLimited = []

    Star = []
    Entropy = []
    
    # Iterate over ranks
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            continue
        else:
            x_data = f[key]['CMx']
            y_data = f[key]['CMy']
            z_data = f[key]['CMz']
            den_data = f[key]['Density']
            vx_data = f[key]['Vx']
            vy_data = f[key]['Vy']
            vz_data = f[key]['Vz']
            vol_data = f[key]['Volume']
            ie_data = f[key]['InternalEnergy']
            T_data = f[key]['Temperature']
            P_data = f[key]['Pressure']
            Diss_data = f[key]['Dissipation']

            DrhoDx_data = f[key]['DrhoDx']
            DrhoDy_data = f[key]['DrhoDy']
            DrhoDz_data = f[key]['DrhoDz']
            DrhoDxLimited_data = f[key]['DrhoDxLimited']
            DrhoDyLimited_data = f[key]['DrhoDyLimited']
            DrhoDzLimited_data = f[key]['DrhoDzLimited']

            DpDx_data = f[key]['DpDx']
            DpDy_data = f[key]['DpDy']
            DpDz_data = f[key]['DpDz']
            DpDxLimited_data = f[key]['DpDxLimited']
            DpDyLimited_data = f[key]['DpDyLimited']
            DpDzLimited_data = f[key]['DpDzLimited']
            divV_data = f[key]['divV']
            divVLimited_data = f[key]['divVLimited']
            
            if folder == 'TDE':
                star_data = f[key]['tracers']['Star']
                entropy_data = f[key]['tracers']['Entropy']

            for i in range(len(entropy_data)):
                if i%20_000 == 0:
                    print(i)
                    
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
                Diss.append(Diss_data[i])

                DrhoDx.append(DrhoDx_data[i])
                DrhoDy.append(DrhoDy_data[i])
                DrhoDz.append(DrhoDz_data[i])
                DrhoDxLimited.append(DrhoDxLimited_data[i])
                DrhoDyLimited.append(DrhoDyLimited_data[i])
                DrhoDzLimited.append(DrhoDzLimited_data[i])

                DpDx.append(DpDx_data[i])
                DpDy.append(DpDy_data[i])
                DpDz.append(DpDz_data[i])
                DpDxLimited.append(DpDxLimited_data[i])
                DpDyLimited.append(DpDyLimited_data[i])
                DpDzLimited.append(DpDzLimited_data[i])

                divV.append(divV_data[i])
                divVLimited.append(divVLimited_data[i])

                if folder == 'TDE':
                    Star.append(star_data[i]) #mass of the disrupted star for TDE
                    Entropy.append(entropy_data[i]) 


    # Close the file
    f.close()
    if folder == 'TDE':
        return X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, T, P, DrhoDx, DrhoDxLimited, DrhoDy, DrhoDyLimited, DrhoDz, DrhoDzLimited, DpDx, DpDxLimited, DpDy, DpDyLimited, DpDz, DpDzLimited, divV, divVLimited, Diss, Star, Entropy
    else:
        return X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, T, P, DrhoDx, DrhoDxLimited, DrhoDy, DrhoDyLimited, DrhoDz, DrhoDzLimited, DpDx, DpDxLimited, DpDy, DpDyLimited, DpDz, DpDzLimited, divV, divVLimited, Diss

if __name__ == '__main__':
    name = '196'
    folder = 'TDE'
    path = f'{folder}/{name}/'
    
    if folder == 'TDE':
        X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, T, P, DrhoDx, DrhoDxLimited, DrhoDy, DrhoDyLimited, DrhoDz, DrhoDzLimited, DpDx, DpDxLimited, DpDy, DpDyLimited, DpDz, DpDzLimited, divV, divVLimited, Diss, Star, Entropy = extractor(folder, f'{path}/snap_{name}_grad.h5')
    else:
        X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, T, P, DrhoDx, DrhoDxLimited, DrhoDy, DrhoDyLimited, DrhoDz, DrhoDzLimited, DpDx, DpDxLimited, DpDy, DpDyLimited, DpDz, DpDzLimited, divV, divVLimited, Diss = extractor(folder, f'{path}/snap_{name}_grad.h5')

    # Save to another file.
    np.save(f'{path}CMx_{name}', X)   
    np.save(f'{path}CMy_{name}', Y) 
    np.save(f'{path}CMz_{name}', Z) 
    np.save(f'{path}Den_{name}', Den)
    np.save(f'{path}Vx_{name}', Vx)   
    np.save(f'{path}Vy_{name}', Vy) 
    np.save(f'{path}Vz_{name}', Vz)
    np.save(f'{path}Vol_{name}', Vol)
    np.save(f'{path}Mass_{name}', Mass)   
    np.save(f'{path}IE_{name}', IE) 
    np.save(f'{path}T_{name}', T)
    np.save(f'{path}P_{name}', P) 
    np.save(f'{path}Diss_{name}', Diss) 

    np.save(f'{path}DrhoDx_{name}', DrhoDx)   
    np.save(f'{path}DrhoDy_{name}', DrhoDy) 
    np.save(f'{path}DrhoDz_{name}', DrhoDz) 
    np.save(f'{path}DrhoDxLimited_{name}', DrhoDxLimited) 
    np.save(f'{path}DrhoDyLimited_{name}', DrhoDyLimited) 
    np.save(f'{path}DrhoDzLimited_{name}', DrhoDzLimited) 

    np.save(f'{path}DpDx_{name}', DpDx)   
    np.save(f'{path}DpDy_{name}', DpDy) 
    np.save(f'{path}DpDz_{name}', DpDz) 
    np.save(f'{path}DpDxLimited_{name}', DpDxLimited) 
    np.save(f'{path}DpDyLimited_{name}', DpDyLimited) 
    np.save(f'{path}DpDzLimited_{name}', DpDzLimited) 

    np.save(f'{path}DivV_{name}', divV) 
    np.save(f'{path}divVLimited_{name}', divVLimited) 

    if folder == 'TDE':
        np.save(f'{path}Star_{name}', Star) 
        np.save(f'{path}Entropy_{name}', Entropy) 

