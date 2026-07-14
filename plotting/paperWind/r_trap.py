
import sys

sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import choose_observers, sort_list
import src.orbits as orb
from src.Wind.Rtrapp_tdiff import load_and_adjust_rtrap

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snaps = [76, 109, 131, 151]
times = [1, 1.5, 2, 2.2]
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
t_fb_days = things['t_fb_days']
Rt = things['Rt']
which_obs = 'left_right_z'

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
indices_obs, label_obs, colors_obs, _ = choose_observers(observers_xyz, which_obs)
markers_obs = ['o', 's', 'X', 'X']
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
longitude_moll = theta              
latitude_moll = np.pi / 2 - phi 

data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps_Lum, tfb_Lum, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps_Lum, Lum, tfb_Lum = sort_list([snaps_Lum, Lum, tfb_Lum], tfb_Lum, unique=True) 
snaps_Lum = snaps_Lum.astype(int)

for snap in snaps:
    time_idx = np.where(snaps_Lum == snap)[0][0]
    time = tfb_Lum[time_idx]
    pathtrap = f"{abspath}/data/{folder}/trap"
    dataRtr = load_and_adjust_rtrap(pathtrap, check, snap)
    kappa_tr, d_tr, Vr_tr = \
        dataRtr['kappa_tr'], dataRtr['den_tr'], dataRtr['Vr_tr']
    kappa_tr = kappa_tr * prel.Rsol_cgs**2 / prel.Msol_cgs

    plt.figure(figsize=(10, 5))
    plt.subplot(111, projection='mollweide')
    for i, idx in enumerate(indices_obs):
        # img = plt.scatter(longitude_moll[idx], latitude_moll[idx],  c = kappa_tr[idx], edgecolor='k', marker=markers_obs[i], s = 120, cmap='rainbow', norm=colors.LogNorm(vmin=3e-1, vmax=40), label=label_obs[i])  
        img = plt.scatter(longitude_moll[idx], latitude_moll[idx],  c = d_tr[idx]*prel.den_converter, edgecolor='k', marker=markers_obs[i], s = 120, cmap='rainbow', norm=colors.LogNorm(vmin=2e-13, vmax=1e-9), label=label_obs[i])  
    cbar = plt.colorbar(img)
    cbar.set_label(r'$\kappa$ [cm$^2$/g]')
    # plt.legend(fontsize=15)
    plt.axhline(np.pi/6, color='k', ls = '--')
    plt.axhline(-np.pi/6, color='k', ls = '--')
    plt.axvline(np.pi/2, color='k', ls = '--')
    plt.axvline(-np.pi/2, color='k', ls = '--')
    plt.suptitle(f'time = {time:.1f} ' + r't$_{\rm fb}$', fontsize=20)
    plt.xticks(np.radians(np.arange(-180, 181, 90))) 
    ytick_labels = ['180°', '135°', '90°', '45°', '0°']
    plt.yticks(np.radians(np.arange(-90, 91, 45)), labels=ytick_labels)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()