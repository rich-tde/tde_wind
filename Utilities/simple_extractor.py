"""
Created on Fri Feb 24 17:06:56 2023

@author: konstantinos, paola 
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import h5py
import os
from Utilities.selectors_for_snap import select_snap, select_prefix


## File structure is
# box, cycle, time, mpi, rank0 ... rank99.
# This iterates over all the ranks


def extractor(filename):
    '''
    Loads the file, extracts quantites from it. 
    '''
    # Timing start
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
    Erad = []
    T = []
    P = []
    #cs = []
    Star = []
    Entropy = []
    box = np.zeros(6)
    
    print('tot ranks: ', len(keys))
    # Iterate over ranks
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            if key == 'Box':
                print(key)
                for i in range(len(box)):
                    box[i] = f[key][i]
            else:
                continue
        else:
            print(key)

            x_data = f[key]['CMx']
            y_data = f[key]['CMy']
            z_data = f[key]['CMz']
            den_data = f[key]['Density']
            
            vx_data = f[key]['Vx']
            vy_data = f[key]['Vy']
            vz_data = f[key]['Vz']
            vol_data = f[key]['Volume']
            
            ie_data = f[key]['InternalEnergy']
            rad_data = f[key]['Erad']
            T_data = f[key]['Temperature']
            P_data = f[key]['Pressure']
            #cs_data = f[key]['SoundSpeed']
            star_data = f[key]['tracers']['Star']
            entropy_data = f[key]['tracers']['Entropy']
            
            for i in range(len(rad_data)):
                X.append(x_data[i])
                Y.append(y_data[i])
                Z.append(z_data[i])
                Den.append(den_data[i])
                Vx.append(vx_data[i])
                Vy.append(vy_data[i])
                Vz.append(vz_data[i])
                Vol.append(vol_data[i])
                IE.append(ie_data[i])
                Erad.append(rad_data[i])
                Mass.append(vol_data[i] * den_data[i])
                T.append(T_data[i])
                P.append(P_data[i])
                Star.append(star_data[i]) #mass of the disrupted star for TDE
                Entropy.append(entropy_data[i])

    # Close the file
    f.close()
    return box, X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Erad, T, P, Star, Entropy

##
# MAIN
##

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'LowRes'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True)

for snap in snaps:
    prepath = select_prefix(m, check, mstar, Rstar, beta, n, compton)
    if alice:
        prepath = f'{prepath}/snap_{snap}'
    else: 
        prepath = f'{prepath}/{snap}'
    file = f'{prepath}/snap_{snap}.h5'
    print(file)
    box, X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Erad, T, P, Star, Entropy = extractor(file)

    #%%
    # Save to another file.
    np.save(f'{prepath}/box_{snap}', box) 
    np.save(f'{prepath}/CMx_{snap}', X)   
    np.save(f'{prepath}/CMy_{snap}', Y) 
    np.save(f'{prepath}/CMz_{snap}', Z) 
    np.save(f'{prepath}/Den_{snap}', Den)
    np.save(f'{prepath}/Vx_{snap}', Vx)   
    np.save(f'{prepath}/Vy_{snap}', Vy) 
    np.save(f'{prepath}/Vz_{snap}', Vz)
    np.save(f'{prepath}/Vol_{snap}', Vol)
    np.save(f'{prepath}/Mass_{snap}', Mass)   
    np.save(f'{prepath}/IE_{snap}', IE) 
    np.save(f'{prepath}/T_{snap}', T)
    np.save(f'{prepath}/P_{snap}', P) 
    np.save(f'{prepath}/Star_{snap}', Star) 
    np.save(f'{prepath}/Entropy_{snap}', Entropy) 
    np.save(f'{prepath}/Rad_{snap}', Erad) 
    print('Done')
                
