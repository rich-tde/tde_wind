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
import csv
import os
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree, to_spherical_components, choose_sections
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

#%%
# MAIN
if compute: # compute dM/dt = dM/dE * dE/dt
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
        cut = Den > 1e-19
        X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
            make_slices([X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den], cut)
        dim_cell = Vol**(1/3)
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        V = np.sqrt(VX**2 + VY**2 + VZ**2)
        bern = orb.bern_coeff(Rsph, V, Den, Mass, Press, IE_den, Rad_den, params)
        v_rad, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)

        # select just outflowing and unbound material
        cond_wind = np.logical_and(v_rad >= 0, bern > 0)
        X_wind, Y_wind, Z_wind, Den_wind, Rsph_wind, v_rad_wind, dim_cell_wind = \
            make_slices([X, Y, Z, Den, Rsph, v_rad, dim_cell], cond_wind)
        if Den_wind.size == 0:
            print(f'no positive', flush=True)
            data = [snap, tfb[i], 0, 0, 0, 0]

        else: 
            # split the wind material in sections
            sections = choose_sections(X_wind, Y_wind, Z_wind, choice = 'dark_bright_z')
            cond_sec = []
            label_obs = []
            color_obs = []
            for key in sections.keys():
                cond_sec.append(sections[key]['cond'])
                label_obs.append(sections[key]['label'])
                color_obs.append(sections[key]['color'])
            

            mwind = np.zeros(len(cond_sec))
            Mdot = np.pi * dim_cell_wind**2 * Den_wind * v_rad_wind 
            if plot: # see what I'm selecting
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
                for ax in [ax1, ax2]:    
                        ax.set_xlabel(r'$X [r_{\rm t}]$')
                ax1.set_xlim(-30,30)
                ax1.set_ylim(-30,30)
                ax2.set_xlim(-30,30)
                ax2.set_ylim(-30,30)
                ax1.set_ylabel(r'$Y [r_{\rm t}]$')
                ax1.set_title('Wind cells')
                ax2.set_title('Selected')
                plt.tight_layout()
            for j, cond in enumerate(cond_sec):
                # select the particles in the chosen section and at the chosen radius
                condR = np.logical_and(np.abs(Rsph_wind-r_chosen) < dim_cell_wind, cond)
                mwind[j] = 4 * r_chosen**2 * np.sum(Mdot[condR]) / np.sum(dim_cell_wind[condR]**2)
                
                if plot:
                    # see what I'm selecting
                    ax1.scatter(X_wind[cond]/Rt, Y_wind[cond]/Rt, s=1, c = color_obs[j], label=label_obs[j])
                    ax2.scatter(X_wind[condR]/Rt, Y_wind[condR]/Rt, s=1, c = color_obs[j])
                    
            data = np.concatenate(([snap], [tfb[i]], mwind))  
    
        csv_path = f'{abspath}/data/{folder}/wind/MdotSec_{check}{which_r_title}.csv'
        if alice:
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    writer.writerow(['snap', 'tfb'] + [f'Mw {lab}' for lab in label_obs])
                writer.writerow(data)
            file.close()

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
            np.loadtxt(f'{abspath}/data/{folder}/wind/MdotSec_{check}{which_r_title}.csv', 
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
    # ax1.plot(tfbH, np.abs(MwS)/Medd_sol, c = 'sienna', label = r'S pole')

    ax2.plot(tfbH, np.abs(MwL/mfb), c = 'forestgreen')
    ax2.plot(tfbH, np.abs(MwR/mfb), c = 'deepskyblue')
    ax2.plot(tfbH, np.abs(MwN/mfb), c = 'orange')
    # ax2.plot(tfbH, np.abs(MwS/mfb), c = 'sienna')
    
    original_ticks = ax1.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]    
    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(labels)  
        ax.set_xlim(0, np.max(tfbfb))
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


    
