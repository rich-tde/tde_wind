""" Compute Mdot fallback and wind across a symmetrical (eventually fixed) surface"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    compute = True
else:
    abspath = '/Users/paolamartire/shocks'
    compute = True

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import healpy as hp
import csv
import os
from sklearn.neighbors import KDTree
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree, to_spherical_components, choose_observers
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

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
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

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz)
indices_axis, label_axis, colors_axis, lines_axis = choose_observers(observers_xyz, 'dark_bright_z')
observers_xyz = observers_xyz.T
x_obs, y_obs, z_obs = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]

#%%
# MAIN
if compute: 
    r_chosen = 0.5*amin # 'Rtr' for radius of the trap, value that you want for a fixed value
    which_r_title = '05amin'

    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

    for i, snap in enumerate(snaps):
        if alice:
            path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        else:
            if snap != 109:
                continue
            path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'

        print(snap, flush=True)
        # Load data and pick the ones unbound and with positive velocity
        data = make_tree(path, snap, energy = True)
        X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Press, data.VX, data.VY, data.VZ, data.IE, data.Rad
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        dim_cell = Vol**(1/3)
        cut = np.logical_and(Den > 1e-19, np.abs(Rsph-r_chosen) < dim_cell) # alredy select only cells around r_chosen
        X, Y, Z, Rsph, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
            make_slices([X, Y, Z, Rsph, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den], cut)
        V = np.sqrt(VX**2 + VY**2 + VZ**2)
        bern = orb.bern_coeff(Rsph, V, Den, Mass, Press, IE_den, Rad_den, params)
        v_rad, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)

        cond_wind = np.logical_and(v_rad >= 0, bern > 0)
        X_pos, Y_pos, Z_pos, Den_pos, Rsph_pos, v_rad_pos, dim_cell_pos = \
            make_slices([X, Y, Z, Den, Rsph, v_rad, dim_cell], cond_wind)
        if Den_pos.size == 0:
            print(f'no positive', flush=True)
            data = [snap, tfb[i], np.zeros(len(indices_axis))]

        else: 
            Mdot_pos = np.pi * dim_cell_pos**2 * Den_pos * v_rad_pos  
            xyz = np.transpose([X_pos/r_chosen, Y_pos/r_chosen, Z_pos/r_chosen]) # normalize to r_chosen
            tree = KDTree(xyz) 
            mwind = np.zeros(len(indices_axis)) #[]
            for j, indices in enumerate(indices_axis):
                x_obs_axis = x_obs[indices]
                y_obs_axis = y_obs[indices]
                z_obs_axis = z_obs[indices]
                dist, idx = tree.query( np.transpose([x_obs_axis, y_obs_axis, z_obs_axis]), k=1)
                Mdot_tokeep, dim_cell_tokeep = Mdot_pos[idx], dim_cell_pos[idx]
                far = dist > dim_cell_tokeep
                # print(Rsph_pos[idx]/r_chosen)
                # print('shape Mdot', np.shape(Mdot_tokeep))
                # print('shape far', np.shape(far))
                # print('shape NON zero Mdot', np.shape(Mdot_tokeep[Mdot_tokeep!=0]))
                Mdot_tokeep[far] = 0 
                mwind[j] = 4 * r_chosen**2 * np.sum(Mdot_tokeep) / np.sum(dim_cell_tokeep**2)
            data = np.concatenate(([snap], [tfb[i]], mwind))  
    
        csv_path = f'{abspath}/data/{folder}/wind/MdotObsSec_{check}{which_r_title}.csv'
        if alice:
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    writer.writerow(['snap', 'tfb'] + [f'Mw {lab}' for lab in label_axis])
                writer.writerow(data)
            file.close()
        # print(data/Medd_sol)

if plot:
    # from scipy.integrate import cumulative_trapezoid
    from plotting.paper.IHopeIsTheLast import ratio_BigOverSmall
    from Utilities.operators import sort_list
    import matplotlib.colors as mcolors
    which_r_title = '05amin'

    _, tfbfb, mfb, _, _, _, _, _, _, _ = \
            np.loadtxt(f'{abspath}/data/{folder}/wind/Mdot_{check}{which_r_title}mean.csv', 
                    delimiter = ',', 
                    skiprows=1, 
                    unpack=True)
    
    _, tfbH, MwR, MwL ,MwN, MwS = \
            np.loadtxt(f'{abspath}/data/{folder}/wind/MdotObsSec_{check}{which_r_title}_old.csv', 
                    delimiter = ',', 
                    skiprows=1, 
                    unpack=True) 
    # integrate mwind_dimCell in tfb 
    # mwind_dimCell_int = cumulative_trapezoid(np.abs(mwind_dimCell), tfb, initial = 0)
    # mfall_int = cumulative_trapezoid(np.abs(mfall), tfb, initial = 0)
    # print(f'integral of Mw at the last time: {mwind_dimCell_int[-1]/mstar} Mstar')
    # print(f'integral of Mfb at the last time: {mfall_int[-1]/mstar} Mstar')
    # print(f'End of simualation, Mw/Mfb in {check}:', np.abs(mwind_dimCell[-1]/mfall[-1]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18, 7))
    ax1.plot(tfbfb, np.abs(mfb)/Medd_sol, c = 'grey', ls = '--')
    ax1.plot(tfbH, np.abs(MwL)/Medd_sol, c = 'forestgreen', label = r'left')
    ax1.plot(tfbH, np.abs(MwR)/Medd_sol, c = 'deepskyblue', label = r'right')
    ax1.plot(tfbH, np.abs(MwN)/Medd_sol, c = 'orange', label = r'N pole')
    ax1.plot(tfbH, np.abs(MwS)/Medd_sol, c = 'sienna', label = r'S pole')

    ax2.plot(tfbH, np.abs(MwL/mfb), c = 'forestgreen')
    ax2.plot(tfbH, np.abs(MwR/mfb), c = 'deepskyblue')
    ax2.plot(tfbH, np.abs(MwN/mfb), c = 'orange')
    ax2.plot(tfbH, np.abs(MwS/mfb), c = 'sienna')
    
    original_ticks = ax1.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]    
    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(labels)  
        ax.set_xlim(0, np.max(tfbH))
        ax.tick_params(axis='both', which='major', width=1.2, length=9)
        ax.tick_params(axis='both', which='minor', width=1, length=5)
        ax.legend(fontsize = 18)
        ax.grid()
    ax1.set_ylim(5e2, 5e6)
    ax1.set_ylabel(r'$|\dot{M}_{{\rm w}}| [\dot{M}_{\rm Edd}]$')   
    ax2.set_ylim(1e-3, 2)
    ax2.set_ylabel(r'$|\dot{M}_{\rm w}| [\dot{M}_{\rm fb}]$')
    plt.suptitle(rf'$\dot{{M}}_{{\rm w}}$ at {which_r_title}', fontsize = 20)
    fig.tight_layout()

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


    
