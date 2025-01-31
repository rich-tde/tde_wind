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

import gc
import time
import warnings
warnings.filterwarnings('ignore')
import matlab.engine

import numpy as np
# import h5py
import healpy as hp
from sklearn.neighbors import KDTree

import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
import src.orbits as orb


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
extr = '' #'' if you use Konst new extr
how = 'fromfld' #'justph' or 'fromfld'
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
r_arr = np.logspace(10, np.log10(10*apo) , 100)

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
print(folder)
pre_saving = f'{abspath}/data/{folder}'

# Load the data
alldata_ph = np.loadtxt(f'{pre_saving}/{check}{extr}_phidx_fluxes.txt')
snaps_ph, alltimes_ph, allindices_ph = alldata_ph[:, 0], alldata_ph[:, 1], alldata_ph[:, 2:]
snaps_ph = np.array(snaps_ph)

snap = 267
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)

print('\n Snapshot: ', snap, '\n')
eng = matlab.engine.start_matlab()
# Find the line corresponding to the snap and take the indices of the photosphere radii 
selected_lines = np.concatenate(np.where(snaps_ph == snap))
selected_idx = selected_lines[0] # the second line is the fluxes
single_indices_ph = allindices_ph[selected_idx]
single_indices_ph = single_indices_ph.astype(int)
box = np.zeros(6)
# Load data -----------------------------------------------------------------
X = np.load(f'{pre}/snap_{snap}/CMx_{snap}.npy')
Y = np.load(f'{pre}/snap_{snap}/CMy_{snap}.npy')
Z = np.load(f'{pre}/snap_{snap}/CMz_{snap}.npy')
Den = np.load(f'{pre}/snap_{snap}/Den_{snap}.npy')
Rad = np.load(f'{pre}/snap_{snap}/Rad_{snap}.npy')
Vol = np.load(f'{pre}/snap_{snap}/Vol_{snap}.npy')
box = np.load(f'{pre}/snap_{snap}/box_{snap}.npy') 
denmask = Den > 1e-19
X, Y, Z, Den, Rad, Vol = make_slices([X, Y, Z, Den, Rad, Vol], denmask)
Rad_den = np.multiply(Rad, Den) # now you have enrgy density
del Rad   
R = np.sqrt(X**2 + Y**2 + Z**2)
xph, yph, zph = make_slices([X, Y, Z,], single_indices_ph)
rph = np.sqrt(xph**2 + yph**2 + zph**2)

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T

# Tree ----------------------------------------------------------------------
#from scipy.spatial import KDTree
xyz = np.array([X, Y, Z]).T
N_ray = 5_000
Rad_den_tot = np.zeros(len(r_arr))
# Dynamic Box ----------------------------------------------------------------
for j, r_plot in enumerate(r_arr):
    print(f'R: {j}', flush=False) 
    sys.stdout.flush()
    for i in range(prel.NPIX):
        if rph[i] > r_plot:
            continue
        else:
            # Progress 
            mu_x = observers_xyz[i][0]
            mu_y = observers_xyz[i][1]
            mu_z = observers_xyz[i][2]

            # Box is for dynamic ray making
            if mu_x < 0:
                rmax = box[0] / mu_x
                # print('x-', rmax)
            else:
                rmax = box[3] / mu_x
                # print('x+', rmax)
            if mu_y < 0:
                rmax = min(rmax, box[1] / mu_y)
                # print('y-', rmax)
            else:
                rmax = min(rmax, box[4] / mu_y)
                # print('y+', rmax)

            if mu_z < 0:
                rmax = min(rmax, box[2] / mu_z)
                # print('z-', rmax)
            else:
                rmax = min(rmax, box[5] / mu_z)
                # print('z+', rmax)

            r = np.logspace( -0.25, np.log10(rmax), N_ray)
            alpha = (r[1] - r[0]) / (0.5 * ( r[0] + r[1]))
            dr = alpha * r

            x = r*mu_x
            y = r*mu_y
            z = r*mu_z
            xyz2 = np.array([x, y, z]).T
            del x, y, z
            tree = KDTree(xyz, leaf_size=50)
            _, idx = tree.query(xyz2, k=1)
            idx = [ int(idx[i][0]) for i in range(len(idx))] # no -1 because we start from 0
            xobs, yobs, zobs, Rad_denobs = X[idx], Y[idx], Z[idx], Rad_den[idx]
            robs = np.sqrt(xobs**2 + yobs**2 + zobs**2)
            idx_r = np.abs(np.argmin(np.abs(robs - r_plot)))
            Rad_den_tot[j] += Rad_denobs[idx_r]
            del xobs, yobs, zobs, Rad_denobs, robs
            gc.collect()
    # in case it doesn't finish, you have the values printed
    print(Rad_den_tot[j])

eng.exit()

if save:
    np.save(f'{pre_saving}/{check}{extr}{snap}_Radden.npy', Rad_den_tot)
    np.save(f'{pre_saving}/{check}{extr}{snap}_R_Radden.npy', r_arr)
