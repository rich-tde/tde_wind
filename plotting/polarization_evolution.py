abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import sort_list
from src.Wind.polarization import compute_polarization
from scipy.ndimage import uniform_filter1d
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
avg_in_los = True
avg_in_time = True

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfb, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps, Lum, tfb = sort_list([snaps, Lum, tfb], tfb, unique=True) 
snaps = snaps.astype(int)

observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) # shape: (3, 192)
observers_xyz = np.array(observers_xyz)
x_obs, y_obs, z_obs = observers_xyz

if avg_in_los:
    from Utilities.operators import choose_observers
    indices_sorted, label_obs, colors_obs, lines_obs = choose_observers(observers_xyz, choice = 'tenths')
    # n_obs_all = [observers_xyz[idx] for idx in indices_sorted]
    len_obs = len(indices_sorted)
    
else:
    n_obs_all_params = [[[1, 0, 0], 'solid', '+x', 'navy'],
                [[-1, 0, 0], 'dashed', '-x', 'dodgerblue'],
                [[0, 1, 0], 'solid', '+y', 'darkorange'],
                [[0, -1, 0], 'dashed', '-y', 'r'],
                [[0, 0, 1], 'solid', '+z', 'forestgreen'],
                [[0, 0, -1], 'dashed', '-z', 'yellowgreen']]

    n_obs_all = [params[0] for params in n_obs_all_params]
    len_obs = len(n_obs_all)
    lines_obs = [params[1] for params in n_obs_all_params]
    label_obs = [params[2] for params in n_obs_all_params]
    colors_obs = [params[3] for params in n_obs_all_params]

P_all = np.zeros((len(snaps), len_obs))

P_all = np.zeros((len(snaps), len_obs))
for s, snap in enumerate(snaps): 
    photo = np.loadtxt(f'{abspath}/data/{folder}/photoPOL/{check}_photo{snap}POL.txt')
    x, y, z, Lum, Fx, Fy, Fz = photo[0], photo[1], photo[2], photo[14], photo[16], photo[17], photo[18]
    for i_o in range(len_obs):
        if avg_in_los:
            indices_ray = np.array(indices_sorted[i_o], dtype=int)
            P_ray = np.zeros(len(indices_ray))
            n_obs_ray = [[x_obs[idx], y_obs[idx], z_obs[idx]] for idx in indices_ray]
            for i_r in range(len(indices_ray)):
                n_obs = n_obs_ray[i_r]
                P, I, Q, U = compute_polarization(Fx, Fy, Fz, n_obs, flux=True)
                P_ray[i_r] = P
            P_all[s, i_o] = np.median(P_ray)
        else:
            n_obs = n_obs_all[i_o]
            P, I, Q, U = compute_polarization(Fx, Fy, Fz, n_obs, flux=True)
            P_all[s, i_o] = P
    # print(P)

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))
for n_idx in range(len_obs):
    if n_idx > 9:
        ax = ax2
    else:
        ax = ax1
    P_idx = P_all[:, n_idx]
    if avg_in_time:
        smoothed_P_idx = uniform_filter1d(P_idx, 3) 
    else:
        smoothed_P_idx = P_idx
    
    ax.plot(tfb, smoothed_P_idx, label=f"{label_obs[n_idx]}", color=colors_obs[n_idx])
ax1.set_ylabel('Polarization fraction')
for ax in [ax1, ax2]:
    ax.set_xlabel(r't$/t_{\rm fb}$')
    ax.legend()
    ax.set_ylim(0, 1)
# %%
