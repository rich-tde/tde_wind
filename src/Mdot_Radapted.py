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
check = 'NewAMR'
where_to_measure = 'Rtr' # 'Rtr' or 'Rph' or pick an arbitrary values

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
    n_obs = len(observers_xyz)
    r_obs = np.sqrt(observers_xyz[:, 0]**2 + observers_xyz[:, 1]**2 + observers_xyz[:, 2]**2)
    long_obs = np.arctan2(observers_xyz[:, 1], observers_xyz[:, 0])          
    lat_obs = np.arccos(observers_xyz[:, 2] / r_obs)
    
    if not alice:
        fig, ax = plt.subplots(1,1,figsize = (8,6))
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        ax.set_ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')
        ax.set_yscale('log')
        if where_to_measure == 'Rtr':
            ax.set_ylim(1e-3, 8e5)
        else:
            ax.set_ylim(1e-1, 8e5)
        ax.grid()
        fig.suptitle(f'Computed at {where_to_measure}', fontsize = 20)

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
        if where_to_measure == 'Rtr':
            data_tr = np.load(f'{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz')
            x_tr, y_tr, z_tr = data_tr['x_tr'], data_tr['y_tr'], data_tr['z_tr']
            r_all_tocompare = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
            label_r = r'$r_{\rm tr}$'
        elif where_to_measure == 'Rph':
            data_ph = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
            x_ph, y_ph, z_ph = data_ph[0], data_ph[1], data_ph[2]
            r_all_tocompare = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
            label_r = r'$r_{\rm ph}$'      
        elif where_to_measure == 'amin':
            r_compare = amin
            label_r = r'$a_{\rm min}$'      
        elif where_to_measure == '50Rt':
            r_compare = 50*Rt
            label_r = r'$50 R_{\rm t}$'      
        
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
        long = np.arctan2(Y, X)          # Azimuthal angle in radians
        lat = np.arccos(Z / Rsph)
        v_rad, _, _ = to_spherical_components(VX, VY, VZ, lat, long)
        # Positive velocity (and unbound)
        cond = np.logical_and(v_rad >= 0, np.logical_and(bern > 0, X > -amin))
        # cond = np.logical_and(bern > 0, X > amin)
        X_pos, Y_pos, Z_pos, Den_pos, Rsph_pos, v_rad_pos, dim_cell_pos = \
            make_slices([X, Y, Z, Den, Rsph, v_rad, dim_cell], cond)
        if Den_pos.size == 0:
            print(f'no positive', flush=True)
            mwind_pos = 0
            Vwind_pos = 0
        else:
            Mdot_pos = dim_cell_pos**2 * Den_pos * v_rad_pos  # there should be a pi factor here, but you put it later

            if where_to_measure == 'Rtr' or where_to_measure == 'Rph':
                # assign each (positive, unbound) cell to the nearest HEALPix observer direction
                ux_pos, uy_pos, uz_pos = X_pos / Rsph_pos, Y_pos / Rsph_pos, Z_pos / Rsph_pos
                ipix_pos = hp.vec2pix(prel.NSIDE, ux_pos, uy_pos, uz_pos)

                Mdot_pos_casted = np.zeros(n_obs)
                v_rad_pos_casted = np.zeros(n_obs)

                for p in range(n_obs):
                    r_compare = r_all_tocompare[p]
                    if not np.isfinite(r_compare) or r_compare <= 0:
                        continue
                    mask_p = ipix_pos == p
                    if not np.any(mask_p):
                        continue
                    # cells close to the trapping radius along this observer
                    near = np.abs(Rsph_pos[mask_p] - r_compare) < dim_cell_pos[mask_p]
                    if not np.any(near):
                        continue

                    Mdot_pos_casted[p] = np.mean(Mdot_pos[mask_p][near])
                    v_rad_pos_casted[p] = np.mean(v_rad_pos[mask_p][near])

                    if np.logical_and(where_to_measure == 'Rtr', np.logical_and(snap == 318, not alice)):
                        fig1, ax1 = plt.subplots(1, 1, figsize = (8,6))
                        ax1.set_xlabel(r'cell')
                        ax1.set_ylabel(r'$r [r_{\rm a}]$')
                        ax1.set_yscale('log')
                        ax1.scatter(np.arange(len(Rsph_pos[mask_p])), Rsph_pos[mask_p]/apo, s = 2, c = 'k', label = r'r [$B>0, v_r>0, X>a_{\rm min}$]')
                        # place error bars equal to the cell size
                        ax1.errorbar(np.arange(len(Rsph_pos[mask_p])), Rsph_pos[mask_p]/apo, yerr = dim_cell_pos[mask_p]/apo, fmt = 'o', color = 'k')
                        ax1.axhline(r_compare/apo, color = 'r', ls = '--', label = label_r)
                        ax1.legend(fontsize = 14)
                        fig1.suptitle(f'Snap {snap}, observer {p}')

                mwind_pos = np.sum(Mdot_pos_casted) * np.pi
                Vwind_pos = np.mean(v_rad_pos_casted)
            
            else:
                selected_pos = np.abs(Rsph_pos - r_compare) < dim_cell_pos
                if Mdot_pos[selected_pos].size == 0:
                    mwind_pos = 0
                    Vwind_pos = 0
                else:
                    mwind_pos = np.sum(Mdot_pos[selected_pos]) * np.pi 
                    v_rad_pos = np.mean(v_rad_pos[selected_pos])
            
        # Negative velocity (and bound)
        cond = np.logical_and(v_rad < 0, bern <= 0)
        # cond = bern <= 0
        X_neg, Y_neg, Z_neg, Den_neg, Rsph_neg, v_rad_neg, dim_cell_neg = \
            make_slices([X, Y, Z, Den, Rsph, v_rad, dim_cell], cond)
        if Den_neg.size == 0:
            print(f'no bern negative: {bern}', flush=True)
            mwind_neg = 0
            Vwind_neg = 0
        else:
            Mdot_neg = dim_cell_neg**2 * Den_neg * v_rad_neg  # there should be a pi factor here, but you put it later
            
            if where_to_measure == 'Rtr' or where_to_measure == 'Rph':
                ux_neg, uy_neg, uz_neg = X_neg / Rsph_neg, Y_neg / Rsph_neg, Z_neg / Rsph_neg
                ipix_neg = hp.vec2pix(prel.NSIDE, ux_neg, uy_neg, uz_neg)

                Mdot_neg_casted = np.zeros(n_obs)
                v_rad_neg_casted = np.zeros(n_obs)

                for p in range(n_obs):
                    r_tr = r_all_tocompare[p]
                    if not np.isfinite(r_tr) or r_tr <= 0:
                        continue
                    mask_p = ipix_neg == p
                    if not np.any(mask_p):
                        continue
                    near = np.abs(Rsph_neg[mask_p] - r_tr) < dim_cell_neg[mask_p]
                    if not np.any(near):
                        continue
                    Mdot_neg_casted[p] = np.mean(Mdot_neg[mask_p][near])
                    v_rad_neg_casted[p] = np.mean(v_rad_neg[mask_p][near])

                mwind_neg = np.sum(Mdot_neg_casted) * np.pi
                Vwind_neg = np.mean(v_rad_neg_casted)
            else:
                selected_neg = np.abs(Rsph_neg - r_compare) < dim_cell_neg
                if Mdot_neg[selected_neg].size == 0:
                    mwind_neg = 0
                    Vwind_neg = 0
                else:
                    mwind_neg = np.sum(Mdot_neg[selected_neg]) * np.pi #4 *  * radii**2
                    v_rad_neg = np.mean(v_rad_neg[selected_neg])
    
        csv_path = f'{abspath}/data/{folder}/wind/Mdot_{check}{where_to_measure}.csv'
        data = [snap, tfb[i], mfall, mwind_pos, Vwind_pos, mwind_neg, Vwind_neg]
        if alice:
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    writer.writerow(['snap', 'tfb', 'Mdot_fb', 'Mdot_wind_pos', 'Vwind_pos', 'Mdot_wind_neg', 'Vwind_neg'])
                writer.writerow(data)
            file.close()
        else:
            ax.scatter(tfb[i], np.abs(mwind_pos)/Medd_code, c = 'dodgerblue', label = r'$\dot{M}_{\rm w}$')
            ax.scatter(tfb[i], np.abs(mwind_neg)/Medd_code,  c = 'forestgreen', label = r'$\dot{M}_{\rm in}$')
            ax.scatter(tfb[i], np.abs(mfall)/Medd_code, label = r'$\dot{M}_{\rm fb}$', c = 'k')
            if i == 0:
                ax.legend(fontsize = 16)    
                fig.tight_layout()

if plot:
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    snap, tfb, mfall, mwind_pos, Vwind_pos, mwind_neg, Vwind_neg = \
        np.loadtxt(f'{abspath}/data/{folder}{check}/wind/Mdot_{check}{where_to_measure}.csv', 
                   delimiter = ',', 
                   skiprows=1, 
                   unpack=True)
    
    fig, ax1 = plt.subplots(1, 1, figsize = (8,7))
    fig2, ax2 = plt.subplots(1, 1, figsize = (8,7))
    ax1.plot(tfb, np.abs(mwind_pos)/Medd_code, c = 'dodgerblue', label = r'$\dot{M}_{\rm w}$')
    ax1.plot(tfb, np.abs(mwind_neg)/Medd_code,  c = 'forestgreen', label = r'$\dot{M}_{\rm in}$')
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
        ax.set_title(r'Wind: $v_r>0, B>0, X>a_{\rm min}$', fontsize = 20)
    plt.tight_layout()
    # fig.savefig(f'{abspath}/Figs/outflow/Mdot_{check}.pdf', bbox_inches = 'tight')


    