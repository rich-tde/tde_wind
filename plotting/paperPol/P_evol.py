#!/usr/bin/env python3
abspath = '/Users/paolamartire/shocks'
import sys

sys.path.append(abspath)
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from src import orbits as orb
from src.Wind.Mdot_Rfixed_sec import split_cells
from src.Polarization.RICH.extract_vtk import extract_poli
from plotting.paperWind.spectra_lc import add_spectral_regions


m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'MG' 
generat = 2500
choice = 'split_stream'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600
x_axis = 'wavelength' # wavelength 

if x_axis == 'wavelength':
    snap = 429

    groups = np.arange(10)
    f_edges_ev = np.loadtxt(f'{abspath}/src/Opacity/fromRICH/MG/frequency_edges.txt') # [Hz] frequency range for each group. shape: (10, 2)
    f_edges = f_edges_ev * prel.ev_toHz
    f_mid_points = (f_edges[:-1] + f_edges[1:]) / 2
    time = np.loadtxt(f'{abspath}/data/{folder}/snap_{snap}/Time_snap_{snap}.txt')
    time *= prel.tsol_cgs/tfallback_cgs
    path_file =  f'{abspath}/TDE/{folder}/Polarization/{snap}/tde_{generat}polarization{snap}.vtk'
    data = extract_poli(
                path_file,
                fields=[f"forward_polarization_degree",
                        f"forward_polarization_snr"])   
    x = data["x"]
    y = data["y"]
    z = data["z"]
    indices_sec, labels_sec, color_obs = split_cells(x, y, z, choice=choice)
    P_gray = data["forward_polarization_degree"]
    snr_P_gray = data["forward_polarization_snr"] 
    P_gray_med_idx = np.zeros(len(indices_sec))
    snr_P_gray_med_idx = np.zeros(len(indices_sec))
    for i, indices in enumerate(indices_sec):
        P_gray_med_idx[i] = np.median(P_gray[indices])
        snr_P_gray_med_idx[i] = np.median(snr_P_gray[indices])

    P_wave = []
    snr_P_wave = []
    Lum_wave = []
    ratioR_wave = []
    for group in groups:
        data = extract_poli(
            path_file,
            fields=[f"photosphere_g{group}_radius_thermalization", 
                    f"photosphere_g{group}_radius_tau_total",
                    f"forward_polarization_degree_group_{group:03d}",
                    f"forward_polarization_snr_group_{group:03d}",
                    f'forward_group_{group}_luminosity'])         
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
        Lum = data[f'forward_group_{group}_luminosity']
        r_th = data[f"photosphere_g{group}_radius_thermalization"]
        r_ph = data[f"photosphere_g{group}_radius_tau_total"]
        ratioR = r_th/r_ph
        
        P_med_idx = np.zeros(len(indices_sec))
        snr_P_med_idx = np.zeros(len(indices_sec))
        Lum_med_idx = np.zeros(len(indices_sec))
        ratioR_med_idx = np.zeros(len(indices_sec))
        
        for i, indices in enumerate(indices_sec):
            P_med_idx[i] = np.median(P[indices])
            snr_P_med_idx[i] = np.median(snr_P[indices])
            Lum_med_idx[i] = np.median(Lum[indices])
            ratioR_med_idx[i] = np.median(ratioR[indices])
        P_wave.append(P_med_idx)
        snr_P_wave.append(snr_P_med_idx)
        Lum_wave.append(Lum_med_idx)
        ratioR_wave.append(ratioR_med_idx)

    P_wave = np.transpose(P_wave)# shape: (len(groups), len(indices_sec))
    snr_P_wave = np.transpose(snr_P_wave)
    Lum_wave = np.transpose(Lum_wave)
    ratioR_wave = np.transpose(ratioR_wave)
    figP, ((axP, axLum), (axSNR, axR)) = plt.subplots(2, 2, figsize=(20, 15)) 
    for i, indices in enumerate(indices_sec):
        axP.plot(f_mid_points, P_wave[i], '-o', label=f'{labels_sec[i]}', color=color_obs[i])
        axSNR.plot(f_mid_points, snr_P_wave[i],  '-o', color=color_obs[i])
        axLum.plot(f_mid_points, Lum_wave[i], '-o', label=f'{labels_sec[i]}', color=color_obs[i])
        axR.plot(f_mid_points, ratioR_wave[i], '-o', label=f'{labels_sec[i]}', color=color_obs[i])
        axP.axhline(P_gray_med_idx[i], ls = '--', color=color_obs[i])
        axSNR.axhline(snr_P_gray_med_idx[i], ls = '--', color=color_obs[i])
    axP.legend(fontsize=15)
    axP.set_ylabel(r'median $P$')
    axSNR.set_ylabel(r'median $S/N$ for $P$')
    axLum.set_ylabel(r'median $L$ [erg/s]')
    axR.set_ylabel(r'median $r_{\rm th}/r_{\rm ph}$')
    axP.set_ylim(5e-3, 0.5)
    axSNR.set_ylim(9e-1, 40)
    for ax in [axP, axSNR, axLum, axR]:
        add_spectral_regions(ax, x_axis, text_band=True if ax == axP else False)
        ax.tick_params(axis='both', which='major', length=8)
        ax.tick_params(axis='both', which='minor', length=6)
        ax.loglog()
        ax.grid() 
        ax.set_xlim(f_edges[0], f_edges[-1])
        top = ax.twiny() 
        frequencies = np.logspace(np.log10(f_edges[0]), np.log10(f_edges[-1]), 5)
        wavelengths = prel.c_cgs * 1e8 / frequencies
        top.set_xticks(wavelengths, [f"{x:.2f}" for x in wavelengths])
        top.set_xlim(wavelengths.max(), wavelengths.min())
        top.set_xscale("log")
        if ax in [axP, axLum]:
            ax.set_xlabel(r'', fontsize=30)
            top.set_xlabel(r"$\lambda\;(\AA)$", fontsize=30)
    figP.suptitle(f't={time:.2f}' + r't$_{\rm fb}$,' + f' {generat} GEN', fontsize = 20)
    figP.tight_layout()



elif x_axis == 'time':
    groups = [3, 4, 7, 8]
    lines_group = ['-', '--', ':']
    snaps = [320, 429]
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
                P_med_idx[i] = np.median(P[indices])
                snr_P_med_idx[i] = np.median(snr_P[indices])

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
            ax[0][g].plot(time_plot, P_line, color_obs[l])
            ax[1][g].plot(time_plot, snr_line, color_obs[l])
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