""" Compute the time evolution of Mdot fallback and Mdot wind across a spherical surface"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    compute = True
else:
    abspath = '/Users/paolamartire/shocks'
    compute = False

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv
import os
import healpy as hp
from sklearn.neighbors import KDTree
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree, to_spherical_components, choose_sections, choose_observers, to_cylindric
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices

##
# PARAMETERS
#%%
m = 4
Mbh = 10**4
Mbh_cgs = Mbh * prel.Msol_cgs
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR'
choice = 'left_right_z' # 'left_right_in_out_z', 'left_right_z', 'all' or 'in_out_z', 'thirties'
how = '' # '' for the normalized sum or 'mean' for mean of Mw of each cells

NPIX = hp.nside2npix(prel.NSIDE)
observers_xyz = hp.pix2vec(prel.NSIDE, range(NPIX))
observers_xyz = np.array(observers_xyz)
_, label_obs, _, _ = choose_observers(observers_xyz, choice)

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 # converted to seconds
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb'] # semimajor axis of the bound orbit

Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs 
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
#%%
# MAIN
def split_observers(X, Y, Z, dim_cell):
    global x_obs, y_obs, z_obs, indices_obs
    xyz = np.transpose([X/r_chosen, Y/r_chosen, Z/r_chosen]) # normalize to r_chosen
    tree = KDTree(xyz) 
    sections_tocheck = choose_sections(X, Y, Z, choice)
    indices_all = np.arange(len(X))
    indices_sec_tocheck = []
    for key in sections_tocheck.keys():
        cond_sec_tocheck = sections_tocheck[key]['cond']
        indices_sec_tocheck.append(indices_all[cond_sec_tocheck])

    indices_sec = []
    for j, indices in enumerate(indices_obs):
        x_obs_sec = x_obs[indices]
        y_obs_sec = y_obs[indices]
        z_obs_sec = z_obs[indices]
        dist, idx = tree.query(np.transpose([x_obs_sec, y_obs_sec, z_obs_sec]), k = 70)
        dist = dist.flatten()
        idx = idx.flatten()
        correct_idx = np.intersect1d(idx, indices_sec_tocheck[j])
        indices_sec.append(correct_idx)
    return indices_sec

def split_cells(X, Y, Z, choice):
    indices = np.arange(len(X))
    indices_sec = []
    sections = choose_sections(X, Y, Z, choice)
    cond_sec = []
    label_obs = []
    for key in sections.keys():
        cond_sec.append(sections[key]['cond'])
        label_obs.append(sections[key]['label'])
        # color_obs.append(sections[key]['color'])

    for j, cond in enumerate(cond_sec):
        # select the particles in the chosen section and at the chosen radius
        condR = cond #np.logical_and(np.abs(Rsph-r_chosen) < dim_cell, cond)
        indices_sec.append(indices[condR])
    
    return indices_sec, label_obs
    
def Mdot_sec(path, snap, r_chosen, choice, how = ''):
    # Load data and pick the ones unbound and with positive velocity
    data = make_tree(path, snap)
    X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Press, data.VX, data.VY, data.VZ, data.IE, data.Rad
    cut = Den > 1e-19
    X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
        make_slices([X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den], cut)
    dim_cell = Vol**(1/3)
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)
    V = np.sqrt(VX**2 + VY**2 + VZ**2)
    
    cut, _, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')
    
    
    X_wind, Y_wind, Z_wind, Den_wind, v_rad_wind, dim_cell_wind, Rad_den_wind = \
        make_slices([X, Y, Z, Den, V_r, dim_cell, Rad_den], cut)
    if Den_wind.size == 0:
        print(f'no positive', flush=True)
        if r_chosen > apo:
            data = [*np.zeros(12)]
        else:
            data = [*np.zeros(4)] # wathc out: you put 4 beacuse you're looking at 4 sections

    else:
        Mdot = np.pi * dim_cell_wind**2 * Den_wind * v_rad_wind 
        indices_sec, _ = split_cells(X_wind, Y_wind, Z_wind, choice)

        mwind = np.zeros(len(indices_sec))
        Lum_fs = np.zeros(len(indices_sec))
        Lkin = np.zeros(len(indices_sec))

        if plot: # see what I'm selecting
            figd, axd = plt.subplots(3, 3, figsize=(21, 21))
            figV, axV = plt.subplots(3, 3, figsize=(21, 21))
            figB, axB = plt.subplots(3, 3, figsize=(21, 21))
            figOE, axOE = plt.subplots(3, 3, figsize=(21, 21))
            # figOEB, axOEB = plt.subplots(1,1, figsize=(8,6))
            if r_chosen > amin:
                normaliz = apo
                for ax in [axd, axV, axB, axOE]:
                    for j in range(3):
                        for i in range(3):
                            ax[i, j].set_xlim(-3,3)
                            ax[i, j].set_ylim(-3,3)
                        ax[0, j].set_xlabel(r'$X [r_{\rm a}]$')
                        ax[1, j].set_xlabel(r'$Y [r_{\rm a}]$')
                        ax[2, j].set_xlabel(r'$X [r_{\rm a}]$')
                    ax[0, 0].set_ylabel(r'$Y [r_{\rm a}]$')
                    ax[1, 0].set_ylabel(r'$Z [r_{\rm a}]$')
                    ax[2, 0].set_ylabel(r'$Z [r_{\rm a}]$')
            else:
                normaliz = Rt
                for ax in [axd, axV, axB, axOE]:
                    for j in range(3):
                        for i in range(3):
                            ax[i, j].set_xlim(-10,10)
                            ax[i, j].set_ylim(-10,10)
                        ax[0, j].set_xlabel(r'$X [r_{\rm t}]$')
                        ax[1, j].set_xlabel(r'$Y [r_{\rm t}]$')
                        ax[2, j].set_xlabel(r'$X [r_{\rm t}]$')
                    ax[0, 0].set_ylabel(r'$Y [r_{\rm t}]$')
                    ax[1, 0].set_ylabel(r'$Z [r_{\rm t}]$')
                    ax[2, 0].set_ylabel(r'$Z [r_{\rm t}]$')

            # if with_who == '':
            #     plt.suptitle(f'Selected with spherical sections at snap {snap}', fontsize = 18)
            # elif with_who == 'Obs':
            #     plt.suptitle(f'Selected Healpix observers at snap {snap}', fontsize = 18)
            plt.tight_layout()

        C_mult = 4/len(indices_sec) # to have the right normalization in all cases
        for j, indices in enumerate(indices_sec):
            # select the particles in the chosen section and at the chosen radius
            if how == '':   
                mwind[j] = C_mult * r_chosen**2 * np.sum(Mdot[indices]) / np.sum(dim_cell_wind[indices]**2)
                Lum_fs[j] = C_mult * r_chosen**2 * np.pi * np.sum(Rad_den_wind[indices] * dim_cell_wind[indices]**2) * prel.csol_cgs / np.sum(dim_cell_wind[indices]**2)
                Lkin[j] = 0.5 * C_mult * r_chosen**2 * np.sum(Mdot[indices] * v_rad_wind[indices]**2) / np.sum(dim_cell_wind[indices]**2)
            elif how == 'mean': 
                mwind[j] = C_mult * np.pi * r_chosen**2 * np.mean(Den_wind[indices] * v_rad_wind[indices])
                Lum_fs[j] = C_mult * np.pi * r_chosen**2 * np.mean(Rad_den_wind[indices]) * prel.csol_cgs
                # Lkin[j] = 0.5 * np.mean(Mdot[indices] * v_rad_wind[indices]**2)
                Lkin[j] = 0.5 * C_mult * np.pi * r_chosen**2 * np.mean(Den_wind[indices] * v_rad_wind[indices]**3) 
            
            data = np.concatenate([mwind, Lum_fs, Lkin])

    return data

if __name__ == '__main__':
    if compute: 
        r_chosen = 0.5*amin
        which_r_title = '05amin' 
        snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

        for i, snap in enumerate(snaps):
            if alice:
                path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
            else: 
                if snap not in [109, 151]:
                    continue
                path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
            print(snap, flush=True)
            
            data_wind = Mdot_sec(path, snap, r_chosen, choice, how)
            data_tosave = np.concatenate(([snap], [tfb[i]], data_wind))  
            csv_path = f'{abspath}/data/{folder}/wind/{choice}/MdotSec{how}_{check}{which_r_title}{choice}.csv'
            if alice:
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                        writer.writerow(['snap', 'tfb'] + [f'Mw {lab}' for lab in label_obs] + [f'Lum_fs {lab}' for lab in label_obs] + [f'Lkin {lab}' for lab in label_obs])
                    writer.writerow(data_tosave)
                file.close()

    if plot:
        which_r_title = 'apo'
        
        snap_for_scatter = 109
        path_scat = f'/Users/paolamartire/shocks/TDE/{folder}/{snap_for_scatter}'
        data = make_tree(path_scat, snap_for_scatter)
        X, Y, Z, VX, VY, VZ, Vol, Den, Mass, Press, IE_den, Rad_den = \
            data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Vol, data.Den, data.Mass, data.Press, data.IE, data.Rad
        cut = np.logical_and(Den > 1e-19, np.abs(Y) < Vol**(1/3)) 
        X, Y, Z, VX, VY, VZ, Vol, Den, Mass, Press, IE_den, Rad_den = make_slices([X, Y, Z, VX, VY, VZ, Vol, Den, Mass, Press, IE_den, Rad_den], cut)
        cut, bern, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params)
        # X, Y, Z, V_r, Den = make_slices([X, Y, Z, V_r, Den], cut)
        Mdot_approx = 4 * np.pi * (X**2 + Y**2 + Z**2) * Den * V_r 
        sec, lab_scat = split_cells(X, Y, Z, choice)

        figM, axM =plt.subplots(1,1, figsize = (10,10))
        axM.scatter(X/Rt, Z/Rt, c = Mdot_approx/Medd_sol, s = 2, norm = colors.LogNorm(vmin = 1e3, vmax = 1e6), cmap = 'rainbow',)
        
        fig, ((pos_scatt, axEdd_pos), (neg_scatt, axEdd_neg)) = plt.subplots(2, 2, figsize = (15, 15))
        # fig, (axall, axfb) = plt.subplots(1, 2, figsize = (15, 7))

        fallback = \
                np.loadtxt(f'{abspath}/data/{folder}/paper1/wind/Mdot_{check}05aminmean.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True)
        tfbfb, mfb, mwind_dimCellOld = fallback[1], fallback[2], fallback[3]
        
        wind = \
                np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/MdotSec_{check}{which_r_title}{choice}.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True) 
        tfbH = wind[1]
        rest = wind[2:]
        rest = rest/len(rest) # to have the right normalization in all cases

        with open(f'{abspath}/data/{folder}/wind/{choice}/MdotSec_{check}{which_r_title}{choice}.csv', newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            label_obs = next(reader)
        label_obs = label_obs[2:]
        

        # wind_NOnorm = \
        #         np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/MdotNOnormSec_{check}{which_r_title}{choice}.csv', 
        #                 delimiter = ',', 
        #                 skiprows=1, 
        #                 unpack=True) 
        # tfbH_NOnorm = wind_NOnorm[1]
        # rest_NOnorm = wind_NOnorm[2:]

        # wind_mean = \
        #         np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/MdotSecmean_{check}{which_r_title}{choice}.csv', 
        #                 delimiter = ',', 
        #                 skiprows=1, 
        #                 unpack=True) 
        # tfbH_mean = wind_mean[1]
        # rest_mean = wind_mean[2:]

        # wind_OE = \
        #         np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/MdotOESec_{check}{which_r_title}{choice}.csv', 
        #                 delimiter = ',', 
        #                 skiprows=1, 
        #                 unpack=True) 
        # tfbH_OE = wind_OE[1]
        # rest_OE = wind_OE[2:]

        # wind_obs = \
        #         np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/MdotObsSec_{check}{which_r_title}{choice}.csv', 
        #                 delimiter = ',', 
        #                 skiprows=1, 
        #                 unpack=True) 
        # tfbH_obs = wind_obs[1]
        # rest_obs = wind_obs[2:]

        # wind_Bound = \
        #         np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/MdotBoundSec_{check}{which_r_title}{choice}.csv', 
        #                 delimiter = ',', 
        #                 skiprows=1, 
        #                 unpack=True) 
        # tfbH_Bound = wind_Bound[1]
        # rest_Bound = wind_Bound[2:] 

        # wind_obs8 = \
        #         np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/MdotObs_npix8Sec_{check}{which_r_title}{choice}.csv', 
        #                 delimiter = ',', 
        #                 skiprows=1, 
        #                 unpack=True) 
        # tfbH_obs8 = wind_obs8[1]
        # rest_obs8 = wind_obs8[2:]

        # Plot
        # cm = plt.get_cmap('tab20')[0:10]        # 20 discrete colors
        # ncolors = cm.N 
        for i in range(len(rest)):
            if label_obs[i] in ['0-10',  '10-20',  '20-30',  '30-40',  '40-50',  '50-60',  '60-70',  '70-80',  '80-90']:
                pos_scatt.scatter(X[sec[i]]/Rt, Z[sec[i]]/Rt, s = 10, label = lab_scat[i])
                axEdd_pos.plot(tfbH, rest[i]/Medd_sol,  label = label_obs[i])
            else:
                neg_scatt.scatter(X[sec[i]]/Rt, Z[sec[i]]/Rt, s = 10, label = lab_scat[i], c = prel.reverse_colors[i-9])
                if np.sum(np.isnan(rest[i])) > 0.35 * len(rest[i]):
                    continue
                axEdd_neg.plot(tfbH, rest[i]/Medd_sol, label = label_obs[i],  c = prel.reverse_colors[i-9])
            
            # Mw_sum += rest[i]
            # axEdd.plot(tfbH_Bound, rest_Bound[i]/Medd_sol, c = colors_obs[0], ls = '--', label = r'$\dot{M}_{\rm out, b}$')
            # axEdd.plot(tfbH_mean, rest_mean[i]/Medd_sol, c = colors_obs[i], ls = '--', label = f'Mean' if i==0 else None)
            # axEdd.plot(tfbH_NOnorm, rest_NOnorm[i]/Medd_sol, c = colors_obs[i], ls = ':', label = f'No norm' if i==0 else None)
            # axEdd.plot(tfbH_obs, rest_obs[i]/Medd_sol, c = colors_obs[i], ls = ':', label = f'Obs' if i==0 else None)
            # axEdd.plot(tfbH_obs8, rest_obs8[i]/Medd_sol, c = colors_obs[i], ls = '-.', label = f'Obs8' if i==0 else None)
            # axEdd.plot(tfbH_OE, rest_OE[i]/Medd_sol, c = colors_obs[i], ls = '--', label = f'OE cut' if i==0 else None)
        # axEdd.plot(tfbH, Mw_sum/Medd_sol, c = 'black', ls = '-', label = 'Total')
        # axEdd.plot(tfbH_full, Mw_full/Medd_sol, c = 'orchid', label = 'all')
        # axEdd.plot(tfbfb, mwind_dimCellOld/Medd_sol, c = 'gold', ls = '--', label = r'paper1')

        # for i in range(len(rest)):
        #     axall.plot(tfbH, rest[i]/Mw_sum, c = colors_obs[i], label = label_obs[i])
        #     axfb.plot(tfbH[6:], np.abs(rest[i]/mfb)[6:], c = colors_obs[i], label = label_obs[i])
        
        # integrate mwind_dimCell in tfb 
        # mwind_dimCell_int = cumulative_trapezoid(np.abs(mwind_dimCell), tfb, initial = 0)
        # mfall_int = cumulative_trapezoid(np.abs(mfall), tfb, initial = 0)
        # print(f'integral of Mw at the last time: {mwind_dimCell_int[-1]/mstar} Mstar')
        # print(f'integral of Mfb at the last time: {mfall_int[-1]/mstar} Mstar')
        # print(f'End of simualation, Mw/Mfb in {check}:', np.abs(mwind_dimCell[-1]/mfall[-1]))
        
        original_ticks = axEdd_pos.get_xticks()
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        labels = [str(np.round(tick,2)) if tick in original_ticks else '' for tick in new_ticks]    
        for ax in [axEdd_pos, axEdd_neg]:
            ax.set_yscale('log')
            ax.set_xlabel(r'$t [t_{\rm fb}]$')
            ax.set_xticks(new_ticks)
            ax.set_xticklabels(labels)  
            ax.set_xlim(0, np.max(tfbH))
            ax.tick_params(axis='both', which='major', width=1.2, length=9)
            ax.tick_params(axis='both', which='minor', width=1, length=5)
            ax.grid()
            ax.set_ylabel(r'$\dot{M}_{{\rm w}} [\dot{M}_{\rm Edd}]$')  

        axEdd_neg.plot(tfbfb, np.abs(mfb)/Medd_sol, c = 'grey', ls = '--', label = r'$|\dot{M}_{\rm fb}|$')
        axEdd_pos.set_ylim(1e1, 7e6)
        axEdd_neg.set_ylim(1e1, 7e6)
        # axall.set_ylim(5e-2, 1.1)
        # axall.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm w}]$')
        # axfb.set_ylim(1e-3, 2)
        # axfb.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm fb}]$')
        for ax in [pos_scatt, neg_scatt, axM]:
            ax.legend(fontsize = 18)
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_xlabel(r'$X (r_{\rm t})$')
            ax.set_ylabel(r'$Z (r_{\rm t})$')
        fig.suptitle(rf'$\dot{{M}}_{{\rm w}}$ at {which_r_title}', fontsize = 20)
        fig.tight_layout()
        # fig.savefig(f'{abspath}/Figs/{folder}/Wind/MdotSec_{which_r_title}{choice}.png', dpi = 150)

        
        # fig, ax = plt.subplots(1,1, figsize = (8,6))
        # ax.plot(tfbH, np.abs(mwind_dimCellH/mfallH), c = 'k')
        # ax.set_yscale('log')
        # ax.set_xlabel(r'$t [t_{\rm fb}]$')
        # ax.set_ylabel(r'$|\dot{M}_{\rm w}/\dot{M}_{\rm fb}|$')
        # original_ticks = ax.get_xticks()
        # midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        # new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        # ax.set_xticks(new_ticks)
        # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]    
        # ax.set_xticklabels(labels)
        # ax.tick_params(axis='both', which='major', width=1.2, length=9)
        # ax.tick_params(axis='both', which='minor', width=1, length=5)
        # ax.set_ylim(1e-2, 1)
        # ax.set_xlim(np.min(tfbH), np.max(tfbH))
        # ax.grid()
        # fig.tight_layout()

        # if which_r_title not in ['05amin', 'amin', 'apo']: # plot energies
        #     Lum_fsI, Lum_fsO, Lum_fsN, Lum_fsS, LkinI, LkinO, LkinN, LkinS = \
        #         wind[6], wind[7], wind[8], wind[9], wind[10], wind[11], wind[12], wind[13]
        #     fig, ax = plt.subplots(1, 1, figsize = (10, 7))
        #     ax.plot(tfbH, np.abs(Lum_fsO)/(4*Ledd_sol), c = colors_obs[0], label = r'$L_{\rm fs}$')
        #     ax.plot(tfbH, np.abs(Lum_fsO)/(4*Ledd_sol), c = colors_obs[0], label = label_obs[0])
        #     ax.plot(tfbH, np.abs(Lum_fsI)/(4*Ledd_sol), c = colors_obs[1], label = label_obs[1])
        #     ax.plot(tfbH, np.abs(Lum_fsN)/(4*Ledd_sol), c = colors_obs[2], label = label_obs[2])
        #     ax.plot(tfbH, np.abs(LkinO)/(4*Ledd_sol), c = colors_obs[0], ls = '--', label = r'$L_{\rm kin}$')
        #     ax.plot(tfbH, np.abs(LkinI)/(4*Ledd_sol), c = colors_obs[1], ls = '--')
        #     ax.plot(tfbH, np.abs(LkinN)/(4*Ledd_sol), c = colors_obs[2], ls = '--')
        #     original_ticks = ax.get_xticks()
        #     midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        #     new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        #     labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]    
        #     ax.set_yscale('log')
        #     ax.set_xlabel(r'$t [t_{\rm fb}]$')
        #     ax.set_xticks(new_ticks)
        #     ax.set_xticklabels(labels)  
        #     ax.tick_params(axis='both', which='major', width=1.2, length=9)
        #     ax.tick_params(axis='both', which='minor', width=1, length=5)
        #     ax.set_ylabel(r'$L [L_{\rm Edd}]$')   
        #     ax.set_xlim(0, np.max(tfbH))
        #     ax.set_ylim(1e-4, 1e3)
        #     ax.legend(fontsize = 18)
        #     ax.grid()
        #     plt.suptitle(rf'r = {which_r_title}', fontsize = 20)
        #     fig.tight_layout()



    # %%
