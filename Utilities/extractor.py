"""
Created on Fri Feb 24 17:06:56 2023

@author: konstantinos, paola 

File structure is: box, cycle, time, mpi, rank0 ... rank99. 
extractor iterates over all the ranks
FLD scheme: 
    - keys for each rank in: ['CMx', 'CMy', 'CMz', 'Density', 'Dissipation', 'DpDx', 'DpDy', 'DpDz', 'DrhoDx', 'DrhoDy', 'DrhoDz', 'DsieDx', 'DsieDy', 'DsieDz', 'Eg_0', 'Erad', 'ID', 'InternalEnergy', 'Pressure', 'Temperature', 'Volume', 'Vx', 'Vy', 'Vz', 'X', 'Y', 'Z', 'divV', 'stickers', 'tracers']
    - keys in tracers: ['Entropy', 'Star', 'WasRemoved']
Multiband scheme: [
    - keys: ['CMx', 'CMy', 'CMz', 'Density', 'Dissipation', 'DpDx', 'DpDy', 'DpDz', 'DrhoDx', 'DrhoDy', 'DrhoDz', 'DsieDx', 'DsieDy', 'DsieDz', 'Eg_0', 'Eg_1', 'Eg_2', 'Eg_3', 'Eg_4', 'Eg_5', 'Eg_6', 'Eg_7', 'Eg_8', 'Eg_9', 'Erad', 'ID', 'InternalEnergy', 'Pressure', 'Temperature', 'Volume', 'Vx', 'Vy', 'Vz', 'X', 'Y', 'Z', 'divV', 'stickers', 'tracers']
    - keys in tracers: ['Entropy', 'Star']
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import h5py
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix

def days_since_distruption(time, m, mstar, rstar, choose = 'day'):
    """ Loads the file, extracts time """
    # Read File
    # f = h5py.File(filename, "r")
    # time = np.array(file['Time'])
    t = np.sqrt(prel.Rsol_SI**3 / (prel.Msol_SI* prel.G_SI )) # Follows from G=1
    Mbh = 10**m # * Msol
    time = np.array(time)
    time = time.sum()
    days = time * t / (24 * 60 * 60)
    t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(rstar, 3/2)
    # print(f'days after disruption: {days} // t_fall: {t_fall} // sim_time: {time}')
    if choose == 'tfb':
        # print('Time in tfb')
        days /= t_fall
    return days

def extractor(filename, extended = False, MG = False):
    '''
    Loads the file, extracts quantites from it. 
    '''
    # Timing start
    # Read File
    f = h5py.File(filename, "r")
    # HDF5 are dicts, get the keys.
    keys = f.keys() 
    # List with keys that don't hold relevant data
    not_ranks = ['Box', 'Cycle', 'Time', 'mpi'] # mpi doesn't exist anymore in the new data
    
    box = np.zeros(6)
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
    Star = []
    Entropy = []
    Diss = []
    if extended:
        DpDx = []
        DpDy = []
        DpDz = []
        DivV = []
    if MG:
        Eg_0 = []
        Eg_1 = []
        Eg_2 = []
        Eg_3 = []           
        Eg_4 = []
        Eg_5 = []
        Eg_6 = []
        Eg_7 = []
        Eg_8 = []
        Eg_9 = []
    
    # Iterate over ranks
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            if key == 'Box':
                for i in range(len(box)):
                    box[i] = f[key][i]
            elif key == 'Time':
                tfb = days_since_distruption(f[key], m, mstar, Rstar, choose = 'tfb')
            else:
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
            rad_data = f[key]['Erad']
            T_data = f[key]['Temperature']
            P_data = f[key]['Pressure']
            Diss_data = f[key]['Dissipation']
            star_data = f[key]['tracers']['Star']
            entropy_data = f[key]['tracers']['Entropy']
            if extended:
                DpDx_data = f[key]['DpDx']
                DpDy_data = f[key]['DpDy']
                DpDz_data = f[key]['DpDz']
                DivV_data = f[key]['divV']

            for i in range(len(entropy_data)):
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
                Diss.append(Diss_data[i])
                Entropy.append(entropy_data[i])
                if extended:
                    DpDx.append(DpDx_data[i])
                    DpDy.append(DpDy_data[i])
                    DpDz.append(DpDz_data[i])
                    DivV.append(DivV_data[i])
                if MG:
                    Eg_0.append(f[key]['Eg_0'][i])
                    Eg_1.append(f[key]['Eg_1'][i])
                    Eg_2.append(f[key]['Eg_2'][i])
                    Eg_3.append(f[key]['Eg_3'][i])           
                    Eg_4.append(f[key]['Eg_4'][i])
                    Eg_5.append(f[key]['Eg_5'][i])
                    Eg_6.append(f[key]['Eg_6'][i])
                    Eg_7.append(f[key]['Eg_7'][i])
                    Eg_8.append(f[key]['Eg_8'][i])
                    Eg_9.append(f[key]['Eg_9'][i])
 
    f.close()
    if MG:
        if extended:
            return tfb, box, X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Eg_0, Eg_1, Eg_2, Eg_3, Eg_4, Eg_5, Eg_6, Eg_7, Eg_8, Eg_9, Erad, T, P, Star, Diss, Entropy, DpDx, DpDy, DpDz, DivV
        else: 
            return tfb, box, X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Eg_0, Eg_1, Eg_2, Eg_3, Eg_4, Eg_5, Eg_6, Eg_7, Eg_8, Eg_9, Erad, T, P, Star, Diss, Entropy
    else:
        if extended:
            return tfb, box, X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Erad, T, P, Star, Diss, Entropy, DpDx, DpDy, DpDz, DivV
        else:
            return tfb, box, X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Erad, T, P, Star, Diss, Entropy 


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
check = 'MG'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
print(f'We are in folder: {folder}', flush=True)
prepath_all = select_prefix(m, check, mstar, Rstar, beta, n, compton)

snaps = select_snap(m, check, mstar, Rstar, beta, n, time = False)

for i, snap in enumerate(snaps):
    # if snap != 21:
    #     continue
    if alice:
        prepath = f'{prepath_all}/snap_{snap}'
    else: 
        prepath = f'{prepath_all}/{snap}'
    file = f'{prepath}/snap_{snap}.h5'

    if check == 'MG':
        tfb, box, X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Eg_0, Eg_1, Eg_2, Eg_3, Eg_4, Eg_5, Eg_6, Eg_7, Eg_8, Eg_9, Erad, T, P, Star, Diss, Entropy = extractor(file, extended = False, MG = True)
    else:
        tfb, box, X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Erad, T, P, Star, Diss, Entropy = extractor(file, extended = False, MG = False)
   
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
    np.save(f'{prepath}/Rad_{snap}', Erad) 
    np.save(f'{prepath}/T_{snap}', T)
    np.save(f'{prepath}/P_{snap}', P) 
    np.save(f'{prepath}/Star_{snap}', Star) 
    np.save(f'{prepath}/Diss_{snap}', Diss)
    np.save(f'{prepath}/Entropy_{snap}', Entropy) 
    np.savetxt(f'{prepath}/tfb_{snap}.txt', [tfb])
    if check == 'MG':
        np.save(f'{prepath}/Eg_0_{snap}', Eg_0) 
        np.save(f'{prepath}/Eg_1_{snap}', Eg_1) 
        np.save(f'{prepath}/Eg_2_{snap}', Eg_2) 
        np.save(f'{prepath}/Eg_3_{snap}', Eg_3) 
        np.save(f'{prepath}/Eg_4_{snap}', Eg_4) 
        np.save(f'{prepath}/Eg_5_{snap}', Eg_5) 
        np.save(f'{prepath}/Eg_6_{snap}', Eg_6) 
        np.save(f'{prepath}/Eg_7_{snap}', Eg_7) 
        np.save(f'{prepath}/Eg_8_{snap}', Eg_8) 
        np.save(f'{prepath}/Eg_9_{snap}', Eg_9)
    # np.save(f'{prepath}/DpDx_{snap}', DpDx)
    # np.save(f'{prepath}/DpDy_{snap}', DpDy)
    # np.save(f'{prepath}/DpDz_{snap}', DpDz)
    # np.save(f'{prepath}/DivV_{snap}', DivV)

    del box, X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Erad, T, P, Star, Diss, Entropy #, DpDx, DpDy, DpDz, DivV
    if check == 'MG':
        del Eg_0, Eg_1, Eg_2, Eg_3, Eg_4, Eg_5, Eg_6, Eg_7, Eg_8, Eg_9
    print(f'Done {snap}', flush = True)
                
