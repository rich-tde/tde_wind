""" Compute Mdot fallback and wind"""
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

Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
Medd = Ledd/(0.1*prel.c_cgs**2)
Medd_code = Medd * prel.tsol_cgs / prel.Msol_cgs  # [g/s]
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
convers_kms = prel.Rsol_cgs * 1e-5/prel.tsol_cgs # it's aorund 400

# MAIN
if compute: # compute dM/dt = dM/dE * dE/dt
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    tfb_cgs = tfb * tfallback_cgs #converted to seconds
    bins = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}_bins.txt')
    max_bin_negative = np.abs(np.min(bins))
    mid_points = (bins[:-1]+bins[1:]) * norm_dMdE/2  # get rid of the normalization
    dMdE_distr = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}.txt')[0] # distribution just after the disruption
    bins_tokeep, dMdE_distr_tokeep = mid_points[mid_points<0], dMdE_distr[mid_points<0] # keep only the bound energies
   
    # define the observers and find their corresponding angles (they're radius is 1)
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz).T # shape: (192, 3)
    
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
        data_tr = np.load(f'{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz')
        x_tr, y_tr, z_tr = data_tr['x_tr'], data_tr['y_tr'], data_tr['z_tr']
        r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
        r_tr = np.median(r_tr_all[r_tr_all!=0])

        data = make_tree(path, snap, energy = True)
        X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Press, data.VX, data.VY, data.VZ, data.IE
        dim_cell = Vol**(1/3)
        cut = Den > 1e-19
        X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den= \
            make_slices([X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den], cut)
        IE_spec = IE_den/Den
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        V = np.sqrt(VX**2 + VY**2 + VZ**2)
        orb_en = orb.orbital_energy(Rsph, V, Mass, params, prel.G) 
        bern = orb_en/Mass + IE_spec + Press/Den
        v_rad, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
        # Positive velocity (and unbound)
        cond = np.logical_and(v_rad >= 0, bern > 0)
        X_pos, Y_pos, Z_pos, Den_pos, Rsph_pos, v_rad_pos, dim_cell_pos = \
            make_slices([X, Y, Z, Den, Rsph, v_rad, dim_cell], cond)
        if Den_pos.size == 0:
            print(f'no positive', flush=True)
            mwind_pos = 0
            Vwind_pos = 0
            continue

        Mdot_pos = dim_cell_pos**2 * Den_pos * v_rad_pos  # there should be a pi factor here, but you put it later
        xyz = np.array([X_pos, Y_pos, Z_pos]).T # shape: (N_pos, 3)
        tree = KDTree(xyz, leaf_size = 50) 
        N_ray = 5_000

        xyz_obs = r_tr * observers_xyz
        dist, idx = tree.query(xyz_obs, k=1) 
        dist = np.concatenate(dist)
        idx = np.array([ int(idx[i][0]) for i in range(len(idx))])
        
        # Quantity corresponding to the ray
        Mdot_pos_casted = Mdot_pos[idx] 
        v_rad_pos_casted = v_rad_pos[idx]
        r_sim = np.sqrt(X_pos[idx]**2 + Y_pos[idx]**2 + Z_pos[idx]**2)
        check_dist = np.logical_and(dist <= dim_cell_pos[idx], r_sim >= Rt)
        Mdot_pos_casted[~check_dist] = 0 
        v_rad_pos_casted[~check_dist] = 0

        mwind_pos = np.sum(Mdot_pos_casted) * np.pi
        Vwind_pos = np.mean(v_rad_pos_casted)
    
        csv_path = f'{abspath}/data/{folder}/wind/Mdot_{check}{where_to_measure}.csv'
        data = [snap, tfb[i], mfall, mwind_pos, Vwind_pos]
        if alice:
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    writer.writerow(['snap', ' tfb', ' Mdot_fb', ' Mdot_wind_pos', ' Vwind_pos'])
                writer.writerow(data)
            file.close()

if plot:
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    snap, tfb, mfall, mwind_pos, Vwind_pos, mwind_neg, Vwind_neg = \
        np.loadtxt(f'{abspath}/data/{folder}{check}/wind/Mdot_{check}Rtr.csv', 
                   delimiter = ',', 
                   skiprows=1, 
                   unpack=True)
    
    fig, ax1 = plt.subplots(1, 1, figsize = (8,7))
    fig2, ax2 = plt.subplots(1, 1, figsize = (8,7))
    ax1.plot(tfb, np.abs(mwind_pos)/Medd_code, c = 'dodgerblue', label = r'$\dot{M}_{\rm w}$')
    # ax1.plot(tfb, np.abs(mwind_neg)/Medd_code,  c = 'forestgreen', label = r'$\dot{M}_{\rm in}$')
    ax1.plot(tfb, np.abs(mfall)/Medd_code, label = r'$\dot{M}_{\rm fb}$', c = 'k')
    # ax1.axhline(max_Mdot/Medd, ls = '--', c = 'k', label = r'theoretical $\dot{M}_{\rm max}$')
    # ax1.plot(tfb[-35:], 4e5*np.array(tfb[-35:])**(-5/9), ls = 'dotted', c = 'k', label = r'$t^{-5/9}$')
    # ax1.axvline(tfb[np.argmax(np.abs(mfall)/Medd_code)], c = 'k', linestyle = 'dotted')
    # ax1.text(tfb[np.argmax(np.abs(mfall)/Medd_code)]+0.01, 0.1, r'$t_{\dot{M}_{\rm peak}}$', fontsize = 20, rotation = 90)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-1, 8e5)
    ax1.set_ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')    
    ax2.plot(tfb, Vwind_pos/v_esc, c = 'dodgerblue', label = r'$v_{\rm w} [B>0]$')
    ax2.plot(tfb, Vwind_neg/v_esc, c = 'forestgreen', label = r'$v_{\rm in} [B<0]$')
    ax2.set_ylabel(r'$v_{\rm w} [v_{\rm esc}(R_{\rm p})]$')
    original_ticks = ax1.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    for ax in (ax1, ax2):
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        ax.legend(fontsize = 18)
        ax.set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both', which='major', width=1, length=7)
        ax.tick_params(axis='both', which='minor', width=.8, length=4)
        ax.set_xlim(0, 1.7)
        ax.grid()
        ax.set_title(r'Wind: $v_r>0, B>0, X>-a_{\rm min}$', fontsize = 20)
    plt.tight_layout()
    # fig.savefig(f'{abspath}/Figs/outflow/Mdot_{check}.pdf', bbox_inches = 'tight')


    