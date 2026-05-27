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
angle_time_evolution = True
angle_evolution = True
albedo_evolution = True
if time_evolution:
    avg_in_time = True
    avg_in_los = True
    which_obs = 'chunky_axes' # you use it if avg_in_los is True

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps_lum, tfb_lum, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps_lum, Lum, tfb_lum = sort_list([snaps_lum, Lum, tfb_lum], tfb_lum, unique=True) 
snaps_lum = snaps_lum.astype(int)

observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) # shape: (3, 192)
observers_xyz = np.array(observers_xyz)
x_obs, y_obs, z_obs = observers_xyz
healp_obs = np.vstack((x_obs, y_obs, z_obs)).T # shape: (192, 3)
Pmin = 0
Pmax = 0.6

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
    A_all_median = np.zeros((len(snaps_lum), len(healp_obs)))

    P_all = np.zeros((len(snaps_lum), len_obs))
    A_all = np.zeros((len(snaps_lum), len_obs))
    for s, snap in enumerate(snaps_lum): 
        try:
            photo = np.load(f'{abspath}/data/{folder}/photoPOL/{check}_photo{snap}.npz')
        except FileNotFoundError:
            continue
        Fx, Fy, Fz, alpha_rossland, alpha_scatter = photo['Fx'], photo['Fy'], photo['Fz'], photo['alpha_rossland'], photo['alpha_scatter']
        weight = alpha_scatter/alpha_rossland

        # photo = np.loadtxt(f'{abspath}/data/{folder}/photoPOL/{check}_photo{snap}POL.txt')
        # x, y, z, den, alpha, Lum, Fx, Fy, Fz = photo[0], photo[1], photo[2], photo[4], photo[12], photo[14], photo[16], photo[17], photo[18]
        # kappa = alpha/den
        for i_o in range(len_obs):
            if avg_in_los:
                indices_ray = np.array(indices_sorted[i_o], dtype=int)
                P_ray = np.zeros(len(indices_ray))
                A_ray = np.zeros(len(indices_ray))
                n_obs_ray = [[x_obs[idx], y_obs[idx], z_obs[idx]] for idx in indices_ray]
                for i_r in range(len(indices_ray)):
                    n_obs = n_obs_ray[i_r]
                    P, I, Q, U = compute_polarization(Fx, Fy, Fz, weight, n_obs, flux=True)
                    P_ray[i_r] = P
                    A_ray[i_r] = np.arctan2(U, Q) / 2
                P_all[s, i_o] = np.median(P_ray)
                A_all[s, i_o] = np.median(A_ray)
            else:
                n_obs = n_obs_all[i_o]
                P, I, Q, U = compute_polarization(Fx, Fy, Fz, weight, n_obs, flux=True)
                P_all[s, i_o] = P
                A_all[s, i_o] = np.arctan2(U, Q) / 2
        for i_o_all in range(len(healp_obs)):
            n_healp = healp_obs[i_o_all]
            P, I, Q, U = compute_polarization(Fx, Fy, Fz, weight, n_healp, flux=True)
            P_all_median[s, i_o_all] = P
            A_all_median[s, i_o_all] = np.arctan2(U, Q) / 2

    P_all_median = np.median(P_all_median, axis=1)
    A_all_median = np.median(A_all_median, axis=1)
    
    if angle_time_evolution:
        figP, axP = plt.subplots(1, 1, figsize=(8,6))
        figA, axA = plt.subplots(1, 1, figsize=(8,6))
        axis = [axP, axA]
    else:
        figP, axP = plt.subplots(figsize=(8,6))
        axis = [axP]
    for n_idx in range(len_obs-1):
        P_idx = P_all[:, n_idx]
        A_idx = A_all[:, n_idx]
        if avg_in_time:
            smoothed_P_idx = uniform_filter1d(P_idx, 5) 
            smoothed_A_idx = uniform_filter1d(A_idx, 5)
        else:
            smoothed_P_idx = P_idx
            smoothed_A_idx = A_idx
        
        axP.plot(tfb_lum, smoothed_P_idx, label=f"{label_obs[n_idx]}", color=colors_obs[n_idx])
        axA.plot(tfb_lum, smoothed_A_idx, label=f"{label_obs[n_idx]}", color=colors_obs[n_idx])
        
    if avg_in_time:
        P_all_median = uniform_filter1d(P_all_median, 5)
    axP.plot(tfb_lum, P_all_median, color='k', linestyle='dashed', label='median')
    if angle_time_evolution:
        axA.plot(tfb_lum, A_all_median * radians, color='k', linestyle='dashed', label='median')
    
    for ax in axis:
        ax.set_xlabel(r't$/t_{\rm fb}$')
        ax.tick_params(axis = 'both', which = 'major', length = 8, width = 1.5)
        ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1)
        ax.grid()
    axP.set_ylabel('Polarization fraction P')
    axA.set_ylabel(r'Polarization angle $\chi$ (rad)')
    axP.legend(fontsize = 16)
    axP.set_ylim(1e-2, 1)
    axP.set_xlim(0.02, 2.25)
    axP.set_yscale('log')
    figP.savefig(f'{abspath}/Figs/3.paperPolarization/P_time_evolution_{which_obs}.pdf', dpi=300, bbox_inches='tight')

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
    # angle_x = np.radians(np.arange(0, 180, 1))
    # n_obs_all = np.array([np.cos(angle_x), np.zeros_like(angle_x), np.sin(angle_x)]).T

    fig, ax = plt.subplots(figsize=(8,6))
    for snap in snaps:
        time = tfb_lum[np.argmin(np.abs(snaps_lum - snap))]
        photo = np.load(f'{abspath}/data/{folder}/photoPOL/{check}_photo{snap}.npz')
        Fx, Fy, Fz, alpha_rossland, alpha_scatter = photo['Fx'], photo['Fy'], photo['Fz'], photo['alpha_rossland'], photo['alpha_scatter']
        weight = alpha_scatter/alpha_rossland
        
        r_obs = np.linalg.norm(n_obs_all, axis=1)
        n_obs_all = n_obs_all / r_obs[:, np.newaxis]
        len_obs = len(n_obs_all)
        angle_x = np.arccos(n_obs_all)[:, 0]
        
        P_all = np.zeros(len_obs)
        for i_o in range(len_obs):
            n_obs = n_obs_all[i_o]
            P, I, Q, U = compute_polarization(Fx, Fy, Fz, weight, n_obs, flux=True)
            P_all[i_o] = P
        ax.plot(angle_x * radians, P_all, marker='o', linestyle = '-', label = f't = {time:.2f}' + r' $t_{\rm fb}$')
    ax.set_ylabel('Polarization fraction P')
    ax.set_xlabel(r'$\delta_{\rm obs}$ (rad)') # in y=0 plane')
    ax.set_ylim(Pmin, Pmax)
    ax.grid()
    plt.legend(fontsize = 16)

    axcos = ax.twiny()
    x_ticks = ax.get_xticks()
    axcos.set_xticks(x_ticks)
    axcos.set_xticklabels([f"{np.cos(t):.2f}" for t in x_ticks])
    axcos.set_xlim(ax.get_xlim()[::-1])
    axcos.set_xlabel(r'$\cos\delta_{\rm obs}$')
    plt.savefig(f'{abspath}/Figs/3.paperPolarization/P_angle_evolution.pdf', dpi=300, bbox_inches='tight')

if albedo_evolution:
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
        photo = np.load(f'{abspath}/data/{folder}/photoPOL/{check}_photo{snap}.npz')
        Fx, Fy, Fz, alpha_rossland, alpha_scatter = photo['Fx'], photo['Fy'], photo['Fz'], photo['alpha_rossland'], photo['alpha_scatter']
        albedo = alpha_scatter/alpha_rossland
        flux = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        flux_norm = flux / np.max(flux)
        
        P_all = np.zeros(len(healp_obs))
        for i_o, n_obs in enumerate(healp_obs):
            P, I, Q, U = compute_polarization(Fx, Fy, Fz, albedo, n_obs, flux=True)
            P_all[i_o] = P
        ax.scatter(albedo, P_all, label = f't = {time:.2f}' + r' $t_{\rm fb}$')
    ax.set_ylabel('Polarization fraction P')
    ax.set_xlabel(r'$\sigma_{\rm s}/\alpha_{\rm Ross}$')
    ax.set_ylim(Pmin, Pmax)
    ax.grid()
    plt.legend(fontsize = 16)

    # plt.savefig(f'{abspath}/Figs/3.paperPolarization/P_angle_evolution.pdf', dpi=300, bbox_inches='tight')
