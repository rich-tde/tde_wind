#!/usr/bin/env python3
abspath = '/Users/paolamartire/shocks'
import sys

sys.path.append(abspath)
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from src import orbits as orb
from Utilities.operators import split_cells
from plotting.paperPol.P_moll import extract_poli


m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'MG' 
generat = 2500
snr_thresh = 5
groups = [2, 4, 7, 8]
snaps = [320, 360, 429]
lines_group = ['-', '--', ':']
choice = 'split_stream'
snr_thresh = 3

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600

tfbs = np.zeros(len(snaps))
P_evol = {}
P_evol_arr = []
snr_P_evol_arr = []
for s, snap in enumerate(snaps):  
    print(f'Processing snap {snap}...')
    time = np.loadtxt(f'{abspath}/data/{folder}/snap_{snap}/Time_snap_{snap}.txt')
    tfbs[s] = time*prel.tsol_cgs/tfallback_cgs
    path_file =  f'{abspath}/TDE/{folder}/Polarization/{snap}/tde_{generat}polarization{snap}.vtk'

    P_group = []
    snr_P_group = []
    for group in groups:
        data = extract_poli(
            path_file,
            fields=[f"photosphere_g{group}_radius_thermalization", 
                    f"photosphere_g{group}_radius_tau_total",
                    f"forward_polarization_degree_group_{group:03d}",
                    f"forward_polarization_snr_group_{group:03d}"])
        
        x = data["x"]
        y = data["y"]
        z = data["z"]
        theta = data["theta"]
        phi = data["phi"]
        r_th = data[f"photosphere_g{group}_radius_thermalization"]
        r_tau = data[f"photosphere_g{group}_radius_tau_total"]
        # check if any ratio r_th/r_tau > 1
        if np.any(r_th/r_tau > 1):
            idx_wrong = np.where(r_th/r_tau > 1)[0]
            print(f"Warning: r_th/r_tau > 1 for group {group} at indices {idx_wrong}")
            print(f"r_th/r_tau values: {r_th[idx_wrong] / r_tau[idx_wrong]}")
        P = data[f"forward_polarization_degree_group_{group:03d}"]
        snr_P = data[f"forward_polarization_snr_group_{group:03d}"]
        indices_sec, labels_sec, color_obs = split_cells(x, y, z, choice=choice)
        P_med_idx = np.zeros(len(indices_sec))
        snr_P_med_idx = np.zeros(len(indices_sec))
        
        for i, indices in enumerate(indices_sec):
            P_sector = P[indices]
            snr_P_sector = snr_P[indices]
            P_tocare = P_sector[snr_P_sector > snr_thresh]
            snr_P_tocare = snr_P_sector[snr_P_sector > snr_thresh]
            P_med_idx[i] = np.median(P_tocare) if len(P_tocare) > 0 else np.nan
            snr_P_med_idx[i] = np.median(snr_P_tocare) if len(snr_P_tocare) > 0 else np.nan

        P_group.append(P_med_idx) # at the end:shape: (len(groups), len(indices_sec))
        snr_P_group.append(snr_P_med_idx)
    P_evol_arr.append(P_group) # at the end:shape: (len(snaps), len(groups), len(indices_sec))
    snr_P_evol_arr.append(snr_P_group)

P_evol_arr = np.transpose(P_evol_arr) # (len(indices_sec), len(groups), len(snaps))
snr_P_evol_arr = np.transpose(snr_P_evol_arr)
P_evol['time']= tfbs
for m in range(len(labels_sec)): 
    P_evol[labels_sec[m]] = {}
    for g, group in enumerate(groups):
        P_evol[labels_sec[m]][f'group {group}'] = P_evol_arr[m][g]
        P_evol[labels_sec[m]][f'snr group {group}'] = snr_P_evol_arr[m][g]

figP, ax = plt.subplots(2, len(groups), figsize=(8*len(groups), 12)) 
time_plot = P_evol['time']
for l, lab in enumerate(labels_sec):  
    for g, group in enumerate(groups): 
        P_line = P_evol[lab][f'group {group}']
        snr_line = P_evol[lab][f'snr group {group}']
        ax[0][g].plot(time_plot, P_line, 'o-', color=color_obs[l], label=lab)
        ax[1][g].plot(time_plot, snr_line, 'o-', color=color_obs[l], label=lab)
        ax[0][g].set_title(f'Group {group}', fontsize = 20)
ax[0][0].set_ylabel(r'median $P$')
ax[1][0].set_ylabel(r'median $S/N$')
for j in range(len(groups)):
    ax[0][j].set_ylim(5e-3, 1)
    ax[1][j].set_ylim(1, 20)
    ax[1][j].set_xlabel(r't/t$_{\rm fb}$')
    for i in range(2):
        ax[i][j].legend()
        ax[i][j].tick_params(axis='both', which='major', length=8)
        ax[i][j].tick_params(axis='both', which='minor', length=6)
        ax[i][j].set_yscale('log')
        ax[i][j].grid()
figP.suptitle(f'Polarization evolution for {generat} GEN', fontsize = 20)
figP.tight_layout()

