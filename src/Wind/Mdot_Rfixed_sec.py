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
what = 'outflow' # '' for wind or 'outflow'

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
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)
    dim_cell = Vol**(1/3)
    # find the spherical shell with r = r_chosen
    cut = np.logical_and(Den > 1e-19, np.abs(Rsph - r_chosen) < dim_cell)
    X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
        make_slices([X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den], cut)
    if X.size == 0:
        return np.array([0]*len(label_obs)*3) # to have the right shape in all cases
    
    cut, _, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')

    if what == 'outflow':
        cut =  V_r > 0

    X_wind, Y_wind, Z_wind, Den_wind, v_rad_wind, dim_cell_wind, Rad_den_wind = \
        make_slices([X, Y, Z, Den, V_r, dim_cell, Rad_den], cut)
    if Den_wind.size == 0:
        print(f'no positive', flush=True)
        return np.array([0]*len(label_obs)*3)

    Mdot = np.pi * dim_cell_wind**2 * Den_wind * v_rad_wind 
    indices_sec, _ = split_cells(X_wind, Y_wind, Z_wind, choice)

    mwind = np.zeros(len(indices_sec))
    Lum_fs = np.zeros(len(indices_sec))
    Lkin = np.zeros(len(indices_sec))

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
    NPIX = hp.nside2npix(prel.NSIDE)
    observers_xyz = hp.pix2vec(prel.NSIDE, range(NPIX))
    observers_xyz = np.array(observers_xyz)
    _, label_obs, color_obs, _ = choose_observers(observers_xyz, choice)

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
            csv_path = f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}{which_r_title}{choice}_{what}.csv'
            if alice:
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                        writer.writerow(['snap', 'tfb'] + [f'Mw {lab}' for lab in label_obs] + [f'Lum_fs {lab}' for lab in label_obs] + [f'Lkin {lab}' for lab in label_obs])
                    writer.writerow(data_tosave)
                file.close()

    if plot:
        which_r_title = '05amin'
        
        # snap_for_scatter = 109
        # path_scat = f'/Users/paolamartire/shocks/TDE/{folder}/{snap_for_scatter}'
        # data = make_tree(path_scat, snap_for_scatter)
        # X, Y, Z, VX, VY, VZ, Vol, Den, Mass, Press, IE_den, Rad_den = \
        #     data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Vol, data.Den, data.Mass, data.Press, data.IE, data.Rad
        # cut = np.logical_and(Den > 1e-19, np.abs(Y) < Vol**(1/3)) 
        # X, Y, Z, VX, VY, VZ, Vol, Den, Mass, Press, IE_den, Rad_den = make_slices([X, Y, Z, VX, VY, VZ, Vol, Den, Mass, Press, IE_den, Rad_den], cut)
        # cut, bern, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params)
        # X, Y, Z, V_r, Den = make_slices([X, Y, Z, V_r, Den], cut)
        # Mdot_approx = 4 * np.pi * (X**2 + Y**2 + Z**2) * Den * V_r 
        # sec, lab_scat = split_cells(X, Y, Z, choice)

        figM, axM =plt.subplots(1,1, figsize = (10,10))
        # axM.scatter(X/Rt, Z/Rt, c = Mdot_approx/Medd_sol, s = 2, norm = colors.LogNorm(vmin = 1e3, vmax = 1e6), cmap = 'rainbow',)
        
        fallback = \
                np.loadtxt(f'{abspath}/data/{folder}/paper1/wind/Mdot_{check}05aminmean.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True)
        tfbfb, mfb, mwind_dimCellOld = fallback[1], fallback[2], fallback[3]
        
        wind = \
                np.loadtxt(f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}{which_r_title}{choice}_{what}.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True) 
        tfbH = wind[1]
        rest = wind[2:2+len(label_obs)]
        # print(len(rest), len(label_obs))
        
        for i in range(len(rest)):
            axM.plot(tfbH, rest[i]/Medd_sol,  label = label_obs[i], c = color_obs[i])

        original_ticks = axM.get_xticks()
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        labels = [str(np.round(tick,2)) if tick in original_ticks else '' for tick in new_ticks]    
        axM.set_yscale('log')
        axM.set_xlabel(r'$t [t_{\rm fb}]$')
        axM.set_xticks(new_ticks)
        axM.set_xticklabels(labels)  
        axM.set_xlim(0, np.max(tfbH))
        axM.tick_params(axis='both', which='major', width=1.2, length=9)
        axM.tick_params(axis='both', which='minor', width=1, length=5)
        axM.grid()
        axM.set_ylabel(r'$\dot{M}_{{\rm w}} [\dot{M}_{\rm Edd}]$')  
        axM.legend(fontsize = 12)

        axM.plot(tfbfb, np.abs(mfb)/Medd_sol, c = 'grey', ls = '--', label = r'$|\dot{M}_{\rm fb}|$')
        axM.set_ylim(1e1, 7e6)
        figM.suptitle(rf'$\dot{{M}}_{{\rm w}}$ at {which_r_title}', fontsize = 20)
        figM.tight_layout()
        # figM.savefig(f'{abspath}/Figs/{folder}/Wind/MdotSec_{which_r_title}{choice}.png', dpi = 150)

       