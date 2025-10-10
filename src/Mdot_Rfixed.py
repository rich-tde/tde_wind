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
    compute = False
    plot = True

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import healpy as hp
import csv
import os
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree, to_spherical_components
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices
from sklearn.neighbors import KDTree

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
statist = 'mean' # '' for mean, 'median' for median

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb'] # semimajor axis of the bound orbit
norm_dMdE = things['E_mb']
t_fb_days_cgs = things['t_fb_days'] * 24 * 3600 # in seconds
max_Mdot = mstar*prel.Msol_cgs/(3*t_fb_days_cgs) # in code units

Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.49/(prel.Rsol_cgs**2/prel.Msol_cgs), 0.014, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs 
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
convers_kms = prel.Rsol_cgs * 1e-5/prel.tsol_cgs # it's aorund 400

# MAIN
if compute: # compute dM/dt = dM/dE * dE/dt
    which_r = 0.5*amin # 'Rtr' for radius of the trap, value that you want for a fixed value
    which_r_title = '05amin'

    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    tfb_cgs = tfb * tfallback_cgs #converted to seconds
    bins = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}_bins.txt')
    max_bin_negative = np.abs(np.min(bins))
    mid_points = (bins[:-1]+bins[1:]) * norm_dMdE/2  # get rid of the normalization
    dMdE_distr = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}.txt')[0] # distribution just after the disruption
    bins_tokeep, dMdE_distr_tokeep = mid_points[mid_points<0], dMdE_distr[mid_points<0] # keep only the bound energies
   
    # define the observers and find their corresponding angles (they're radius is 1)
    # observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) #shape: (3, 192)
    # observers_xyz = np.array(observers_xyz).T # shape: (192, 3)
    
    for i, snap in enumerate(snaps):
        print(snap, flush=True)
        if alice:
            path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        else:
            path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
        t = tfb_cgs[i] 
        # convert to code units
        tsol = t / prel.tsol_cgs
        # Find the energy of the element at time t
        energy = orb.keplerian_energy(Mbh, prel.G, tsol) # it'll give it positive
        i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
        if energy-max_bin_negative*norm_dMdE > 0:
            print(f'You overcome the maximum negative bin ({max_bin_negative*norm_dMdE}). You required {energy}')
            continue
            
        dMdE_t = dMdE_distr_tokeep[i_bin]
        mfall = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)
 
        # Load data 
        if which_r == 'Rtr':
            data_tr = np.load(f'{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz')
            x_tr, y_tr, z_tr = data_tr['x_tr'], data_tr['y_tr'], data_tr['z_tr']
            r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
            r_tr = np.median(r_tr_all[r_tr_all!=0])
        else:
            r_tr = which_r

        data = make_tree(path, snap, energy = True)
        X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Press, data.VX, data.VY, data.VZ, data.IE, data.Rad
        dim_cell = Vol**(1/3)
        cut = Den > 1e-19
        X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
            make_slices([X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den], cut)
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        V = np.sqrt(VX**2 + VY**2 + VZ**2)
        bern = orb.bern_coeff(Rsph, V, Den, Mass, Press, IE_den, Rad_den, params)
        v_rad, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
        # Positive velocity (and unbound)
        cond = np.logical_and(v_rad >= 0, bern > 0)
        X_pos, Y_pos, Z_pos, Den_pos, Rsph_pos, v_rad_pos, dim_cell_pos = \
            make_slices([X, Y, Z, Den, Rsph, v_rad, dim_cell], cond)
        if Den_pos.size == 0:
            print(f'no positive', flush=True)
            data = [snap, tfb[i], mfall, 0, 0, 0, 0, 0]

        else: 
            Mdot_dimCell = np.pi * dim_cell_pos**2 * Den_pos * v_rad_pos  
            Mdot_R = 4 * np.pi * r_tr**2 * Den_pos * v_rad_pos 
            # xyz = np.array([X_pos, Y_pos, Z_pos]).T # shape: (N_pos, 3)
            # tree = KDTree(xyz, leaf_size = 50) 
            # xyz_obs = r_tr * observers_xyz
            # dist, idx = tree.query(xyz_obs, k=1) 
            # dist = np.concatenate(dist)
            # idx = np.array([ int(idx[i][0]) for i in range(len(idx))])
            condRtr = np.abs(Rsph_pos-r_tr) < dim_cell_pos
            
            # Quantity corresponding to the ray
            Mdot_dimCell_casted = Mdot_dimCell[condRtr] 
            Mdot_R_casted = Mdot_R[condRtr]
            v_rad_pos_casted = v_rad_pos[condRtr]
            r_sim = np.sqrt(X_pos[condRtr]**2 + Y_pos[condRtr]**2 + Z_pos[condRtr]**2)
            check_dist = np.logical_and(dist <= dim_cell_pos[condRtr], r_sim >= Rt)
            Mdot_dimCell_casted[~check_dist] = 0 
            Mdot_R_casted[~check_dist] = 0
            v_rad_pos_casted[~check_dist] = 0

            mwind_dimCell = np.sum(Mdot_dimCell_casted)
            if statist == 'mean':
                mwind_R = np.mean(Mdot_R_casted)
                mwind_R_nonzero = np.mean(Mdot_R_casted[Mdot_R_casted!=0])
                Vwind = np.mean(v_rad_pos_casted)
                Vwind_nonzero = np.mean(v_rad_pos_casted[v_rad_pos_casted!=0])
            elif statist == 'median':
                mwind_R = np.median(Mdot_R_casted)
                mwind_R_nonzero = np.median(Mdot_R_casted[Mdot_R_casted!=0])
                Vwind = np.median(v_rad_pos_casted)
                Vwind_nonzero = np.median(v_rad_pos_casted[v_rad_pos_casted!=0])
        
            data = [snap, tfb[i], mfall, mwind_dimCell, mwind_R, mwind_R_nonzero, Vwind, Vwind_nonzero]

        csv_path = f'{abspath}/data/{folder}/wind/Mdot_{check}Rtr{statist}.csv'
        if alice:
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    writer.writerow(['snap', ' tfb', ' Mdot_fb', ' Mw with dimCell', 'Mw with Rtr', 'Mw with Rtr (nonzero)', 'Vwind', 'Vwind (nonzero)'])
                writer.writerow(data)
            file.close()

if plot:
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    snap, tfb, mfall, mwind_dimCell, mwind_R, mwind_R_nonzero, Vwind, Vwind_nonzero = \
        np.loadtxt(f'{abspath}/data/{folder}{check}/wind/Mdot_{check}05amin{statist}.csv', 
                   delimiter = ',', 
                   skiprows=1, 
                   unpack=True)
    
    fig, ax1 = plt.subplots(1, 1, figsize = (8, 6))
    fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6))
    ax1.plot(tfb, np.abs(mfall)/Medd_sol, c = 'k', label = r'$\dot{M}_{\rm fb}$')
    # ax1.plot(tfb, np.abs(mwind_dimCell)/Medd_sol, c = 'dodgerblue', label = r'$\dot{M}_{\rm w}$ dim cell')
    ax1.plot(tfb, np.abs(mwind_R)/Medd_sol, c = 'orange', label = r'$\dot{M}_{\rm w}$ with  $r={\rm 0.5a_{\rm min}}$')
    # ax1.plot(tfb, np.abs(mwind_R_nonzero)/Medd_sol, c = 'green', label = r'$\dot{M}_{\rm w}$ at  $r={\rm 0.5a_{\rm min}}$ (nonzero)')
    # ax1.plot(tfb, np.abs(mfall)/Medd_sol, label = r'$\dot{M}_{\rm fb}$', c = 'k')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-1, 5e5)
    ax1.set_ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')   
    ax1.legend(fontsize = 18)

    ax2.plot(tfb, Vwind/v_esc, c = 'dodgerblue')
    ax2.set_ylabel(r'$v_{\rm w} [v_{\rm esc}(R_{\rm p})]$')
    original_ticks = ax1.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    for ax in (ax1, ax2):
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        ax.set_xticks(new_ticks)
        # ax.set_xticklabels(labels)
        ax.tick_params(axis='both', which='major', width=1, length=7)
        ax.tick_params(axis='both', which='minor', width=.8, length=4)
        ax.set_xlim(0, 1.8)
        ax.grid()
        ax.set_title(f'Using {statist}', fontsize = 18)
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(f'{abspath}/Figs/next_meeting/Mdot_{check}{statist}05amin.pdf', bbox_inches = 'tight')


    
# %%
