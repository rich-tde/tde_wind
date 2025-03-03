""" Find photosphere. 
If when == 'fromfld', it will use the data (using the idx of the cells) from fld_curve.py to calculate the photo. 
If when == 'justph', it will compute here the photosphere (just X,Y,Z) as it does in fld_curve.py. 
"""
import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'

import sys
sys.path.append(abspath)

import numpy as np
from Utilities.operators import make_tree
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices

#%% Choose parameters -----------------------------------------------------------------
save = True

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'LowRes' 

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
pre_saving = f'{abspath}/data/{folder}'

# Load the data
# alldata_ph = np.loadtxt(f'{pre_saving}/{check}{extr}_phidx.txt')
alldata_ph = np.loadtxt(f'{pre_saving}/{check}_phidx_fluxes.txt') 
snaps_ph, alltimes_ph, allindices_ph = alldata_ph[:, 0], alldata_ph[:, 1], alldata_ph[:, 2:]
snaps_ph = np.array(snaps_ph)

snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
Lphoto_all = np.zeros(len(snaps)) 

pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)

for idx_s, snap in enumerate(snaps):
    print('\n Snapshot: ', snap, '\n')
    # Find the line corresponding to the snap and take the indices of the photosphere radii 
    # selected_idx = np.argmin(np.abs(snaps_ph - snap))
    selected_lines = np.concatenate(np.where(snaps_ph == snap))
    selected_idx, selected_fluxes = selected_lines[0], selected_lines[1]
    single_indices_ph = allindices_ph[selected_idx]
    single_indices_ph = single_indices_ph.astype(int)
    # Load the data
    path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    datasnap = make_tree(path, snap, energy = True)
    x, y, z, vol, den, Temp, Rad_den, Vx, Vy, Vz = datasnap.X, datasnap.Y, datasnap.Z, datasnap.Vol, datasnap.Den, datasnap.Temp, datasnap.Rad, datasnap.VX, datasnap.VY, datasnap.VZ
    cut = den > 1e-19
    x, y, z, vol, den, Temp, Rad_den, Vx, Vy, Vz = make_slices([x, y, z, vol, den, Temp, Rad_den, Vx, Vy, Vz], cut)
    xph, yph, zph, volph, denph, Tempph, Radph, Vxph, Vyph, Vzph = make_slices([x, y, z, vol, den, Temp, Rad_den, Vx, Vy, Vz], single_indices_ph)
    # save the photosphere
    with open(f'{pre_saving}/photo/{check}_photo{snap}.txt', 'a') as f:
        f.write('# Data for the photospere. Lines are: xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph \n')
        f.write(' '.join(map(str, xph)) + '\n')
        f.write(' '.join(map(str, yph)) + '\n')
        f.write(' '.join(map(str, zph)) + '\n')
        f.write(' '.join(map(str, volph)) + '\n')
        f.write(' '.join(map(str, denph)) + '\n')
        f.write(' '.join(map(str, Tempph)) + '\n')
        f.write(' '.join(map(str, Radph)) + '\n')
        f.write(' '.join(map(str, Vxph)) + '\n')
        f.write(' '.join(map(str, Vyph)) + '\n')
        f.write(' '.join(map(str, Vzph)) + '\n')
        f.close()
