""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    abspath = '/Users/paolamartire/shocks'
    save = False

import gc
import warnings
warnings.filterwarnings('ignore')
import csv

import numpy as np
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
from src.Opacity.linextrapolator import nouveau_rich
from scipy.ndimage import uniform_filter1d

import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices

#%% Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' # '' or 'HiRes'

## Snapshots stuff
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
print('we are in: ', pre)

#%% Opacities: load and interpolate ----------------------------------------------------------------
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
min_T = np.exp(T_cool[0]) 
max_T = np.exp(T_cool[-1])
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
min_rho = np.exp(Rho_cool[0]) /prel.den_converter #from log-cgs to code units
rossland = np.loadtxt(f'{opac_path}/ross.txt')
T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5, treat_den = 'linear')
R_bin = np.logspace(-2, 3.5, 5_000) 

for idx_s, snap in enumerate(snaps):
    print('\n Snapshot: ', snap, '\n')
    # Load data -----------------------------------------------------------------
    if alice:
        X = np.load(f'{pre}/snap_{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}/snap_{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}/snap_{snap}/CMz_{snap}.npy')
        T = np.load(f'{pre}/snap_{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}/snap_{snap}/Den_{snap}.npy')
    else:
        X = np.load(f'{pre}/{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}/{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}/{snap}/CMz_{snap}.npy')
        T = np.load(f'{pre}/{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}/{snap}/Den_{snap}.npy')
    
    R = np.sqrt(X**2 + Y**2 + Z**2)
    denmask = Den > 1e-19
    R, T, Den = make_slices([R, T, Den], denmask) 
    f_Rall, bins_edges = np.histogram(R, bins = R_bin)

    lowden = Den < min_rho
    R, T = make_slices([R, T], lowden)
    f_R, bins_edges = np.histogram(R, bins = R_bin)
    f_R = f_R/f_Rall

    inside_tab = np.logical_and(T > min_T, T < max_T)
    R = R[inside_tab]
    f_RnoT, bins_edges = np.histogram(R, bins = R_bin)
    f_RnoT = f_RnoT/f_Rall

    if save:
        # Save red of the single snap
        snap_Rbin = np.concatenate([snap, R_bin])
        time_fR = np.concatenate([tfb[idx_s], f_R])
        time_fRnoT = np.concatenate([tfb[idx_s], f_RnoT])

        with open(f'{abspath}/data/{folder}/{check}_TestOpac{snap}.txt', 'a') as file:
            file.write('# first number: snap, then: Rbin' + f' '.join(map(str, snap_Rbin)) + '\n')      
            file.write('# first number: tfb, then: f_R (i.e. fraction of cells in each Rbin with Density lower than Table lower limit)\n')
            file.write(f' '.join(map(str, time_fR)) + '\n')  
            file.write('# first number: tfb, then: f_RnoT (i.e. fraction of cells in each Rbin with Density lower than Table lower limit but T in table)\n')
            file.write(f' '.join(map(str, time_fRnoT)) + '\n')
            file.close()
        



