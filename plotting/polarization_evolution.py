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
from Utilities.basic_units import radians

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
time_evolution = True
angle_evolution = True
if time_evolution:
    avg_in_time = True
    avg_in_los = True
    which_obs = 'left_right_in_out_z' 

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps_lum, tfb_lum, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps_lum, Lum, tfb_lum = sort_list([snaps_lum, Lum, tfb_lum], tfb_lum, unique=True) 
snaps_lum = snaps_lum.astype(int)

observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) # shape: (3, 192)
observers_xyz = np.array(observers_xyz)
x_obs, y_obs, z_obs = observers_xyz
healp_obs = np.vstack((x_obs, y_obs, z_obs)).T # shape: (192, 3)

if time_evolution:
    if avg_in_los:
        from Utilities.operators import choose_observers
        indices_sorted, label_obs, colors_obs, lines_obs = choose_observers(observers_xyz, choice = which_obs)
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

    P_all_median = np.zeros((len(snaps_lum), len(healp_obs)))

    P_all = np.zeros((len(snaps_lum), len_obs))
    for s, snap in enumerate(snaps_lum): 
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
                P_all[s, i_o] = np.median(P_ray, axis=0)
            else:
                n_obs = n_obs_all[i_o]
                P, I, Q, U = compute_polarization(Fx, Fy, Fz, n_obs, flux=True)
                P_all[s, i_o] = P
        for i_o_all in range(len(healp_obs)):
            n_healp = healp_obs[i_o_all]
            P, I, Q, U = compute_polarization(Fx, Fy, Fz, n_healp, flux=True)
            P_all_median[s, i_o_all] = P

    P_all_median = np.median(P_all_median, axis=1)
    
    if len_obs > 8:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10,6))
        ax2 = ax1
    for n_idx in range(len_obs):
        # if n_idx > 2:
        #     continue
        if which_obs == 'tenths':
            if n_idx > 8:
                ax = ax1
                col = prel.wanted_colors[n_idx]
            else:
                ax = ax2
                col = prel.reverse_colors[-n_idx+8]         
        else:
            ax = ax1
            col = colors_obs[n_idx]
        P_idx = P_all[:, n_idx]
        if avg_in_time:
            smoothed_P_idx = uniform_filter1d(P_idx, 3) 
        else:
            smoothed_P_idx = P_idx
        
        ax.plot(tfb_lum, smoothed_P_idx, label=f"{label_obs[n_idx]}", color=col)
        
    if avg_in_time:
        P_all_median = uniform_filter1d(P_all_median, 3)
    ax.plot(tfb_lum, P_all_median, color='k', linestyle='dashed', label='median over all obs')
    
    ax1.set_ylabel('Polarization fraction P')
    for ax in [ax1, ax2]:
        ax.set_xlabel(r't$/t_{\rm fb}$')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid()
    ax1.grid()
    plt.savefig(f'{abspath}/Figs/wind_paper/P_time_evolution_{which_obs}.pdf', dpi=300, bbox_inches='tight')

if angle_evolution:
    snaps = [76, 109, 151]
    n_obs_all = [[1, 0, 0], 
                [np.sqrt(3)/2, 0, 1/2],
                [1/np.sqrt(2), 0, 1/np.sqrt(2)], 
                [1/2, 0, np.sqrt(3)/2],
                [0, 0, 1], 
                [-1/2, 0, np.sqrt(3)/2],
                [-1/np.sqrt(2), 0, 1/np.sqrt(2)], 
                [-np.sqrt(3)/2, 0, 1/2],
                [-1, 0, 0]]

    fig, ax = plt.subplots(figsize=(8,6))
    for snap in snaps:
        time = tfb_lum[np.argmin(np.abs(snaps_lum - snap))]
        photo = np.loadtxt(f'{abspath}/data/{folder}/photoPOL/{check}_photo{snap}POL.txt')
        x, y, z, Lum, Fx, Fy, Fz = photo[0], photo[1], photo[2], photo[14], photo[16], photo[17], photo[18]

        r_obs = np.linalg.norm(n_obs_all, axis=1)
        n_obs_all = n_obs_all / r_obs[:, np.newaxis]
        len_obs = len(n_obs_all)
        angle_x = np.arccos(n_obs_all)[:, 0]
        
        P_all = np.zeros(len_obs)
        for i_o in range(len_obs):
            n_obs = n_obs_all[i_o]
            P, I, Q, U = compute_polarization(Fx, Fy, Fz, n_obs, flux=True)
            P_all[i_o] = P
        ax.plot(angle_x * radians, P_all, marker = 'o', linestyle = '-', label = f't = {time:.2f}' + r' $t_{\rm fb}$')
        ax.set_ylabel('Polarization fraction P')
        ax.set_xlabel('Observer angle (rad) in xz plane')
        ax.set_ylim(0.01, 1)
        ax.grid()
    ax.set_title(f'y=0', fontsize = 18) #t = {time:.2f}' + r'$t_{\rm fb}$', fontsize=18)
    plt.legend(fontsize = 16)
    plt.savefig(f'{abspath}/Figs/wind_paper/P_angle_evolution.pdf', dpi=300, bbox_inches='tight')



# %%
