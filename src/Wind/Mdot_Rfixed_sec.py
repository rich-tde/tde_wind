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
with_who = ''  # '' or 'Obs'
n_obs = '' #'_npix8' or ''
choice = 'thirties' # 'left_right_in_out_z', 'left_right_z', 'all' or 'in_out_z', 'thirties'
wind_cond = '' # '' for bernouilli coeff or 'OE' for orbital energy
how = '' # '' for the normalized sum or 'mean' for mean of Mw of each cells

if with_who == '':
    n_obs = ''  # to avoid confusion
    NSIDE = prel.NSIDE
else:
    if n_obs == '_npix8':
        NSIDE = 8  
    else:
        NSIDE = prel.NSIDE
NPIX = hp.nside2npix(NSIDE)
observers_xyz = hp.pix2vec(NSIDE, range(NPIX))
observers_xyz = np.array(observers_xyz)
indices_obs, label_obs, colors_obs, lines_obs = choose_observers(observers_xyz, choice)
observers_xyz = observers_xyz.T
x_obs, y_obs, z_obs = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
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
    color_obs = []
    for key in sections.keys():
        cond_sec.append(sections[key]['cond'])
        label_obs.append(sections[key]['label'])
        color_obs.append(sections[key]['color'])

    for j, cond in enumerate(cond_sec):
        # select the particles in the chosen section and at the chosen radius
        condR = cond #np.logical_and(np.abs(Rsph-r_chosen) < dim_cell, cond)
        indices_sec.append(indices[condR])
    
    return indices_sec, label_obs, color_obs
    
def Mdot_sec(path, snap, r_chosen, with_who, choice, wind_cond = '', how = ''):
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
    OE = orb.orbital_energy(Rsph, V, Mass, params, prel.G)
    OE_spec = OE / Mass
    if (bern < OE_spec).any():
        print('Warning: some particles have Bernoulli < OE_spec', flush=True)

    # select just outflowing and unbound material
    if wind_cond == '':
        cond_wind = np.logical_and(v_rad >= 0, bern > 0)
    if wind_cond == 'OE':
        cond_wind = np.logical_and(v_rad >= 0, OE_spec > 0)
    cond_wind = np.logical_and(cond_wind, np.abs(Rsph - r_chosen) < dim_cell)
    X_wind, Y_wind, Z_wind, Den_wind, v_rad_wind, dim_cell_wind, Rad_den_wind, bern_wind, OE_spec_wind = \
        make_slices([X, Y, Z, Den, v_rad, dim_cell, Rad_den, bern, OE_spec], cond_wind)
    if Den_wind.size == 0:
        print(f'no positive', flush=True)
        if r_chosen > apo:
            data = [*np.zeros(12)]
        else:
            data = [*np.zeros(4)] # wathc out: you put 4 beacuse you're looking at 4 sections

    else:
        Mdot = np.pi * dim_cell_wind**2 * Den_wind * v_rad_wind 
        if with_who == '':
            indices_sec, _, _ = split_cells(X_wind, Y_wind, Z_wind, choice)

        elif with_who == 'Obs':
            indices_sec = split_observers(X_wind, Y_wind, Z_wind, dim_cell_wind)

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
        print('C:', C_mult, flush=True)
        for j, indices in enumerate(indices_sec):
            # select the particles in the chosen section and at the chosen radius
            if how == '':  
                mwind[j] = C_mult * r_chosen**2 * np.sum(Mdot[indices]) / np.sum(dim_cell_wind[indices]**2)
            elif how == 'mean':
                mwind[j] = C_mult * np.pi * r_chosen**2 * np.mean(Den_wind[indices] * v_rad_wind[indices])
            if r_chosen > apo: 
                if how == '':  
                    Lum_fs[j] = C_mult * r_chosen**2 * np.pi * np.sum(Rad_den_wind[indices] * dim_cell_wind[indices]**2) * prel.csol_cgs / np.sum(dim_cell_wind[indices]**2)
                    Lkin[j] = 0.5 * C_mult * r_chosen**2 * np.sum(Mdot[indices] * v_rad_wind[indices]**2) / np.sum(dim_cell_wind[indices]**2)
            
                elif how == 'mean':
                    Lum_fs[j] = C_mult * np.pi * r_chosen**2 * np.mean(Rad_den_wind[indices]) * prel.csol_cgs
                    # Lkin[j] = 0.5 * np.mean(Mdot[indices] * v_rad_wind[indices]**2)
                    Lkin[j] = 0.5 * C_mult * np.pi * r_chosen**2 * np.mean(Den_wind[indices] * v_rad_wind[indices]**3) 
            
            if np.logical_and(plot, j < 3):
                theta_ourConv, _ = to_cylindric(X_wind[indices], Y_wind[indices])
                # see what I'm selecting
                img_xy = axd[0][j].scatter(X_wind[indices]/normaliz, Y_wind[indices]/normaliz, s=10, c = Den_wind[indices]*prel.den_converter, norm = colors.LogNorm(vmin=1e-14, vmax=2e-9))
                img_yz = axd[1][j].scatter(Y_wind[indices]/normaliz, Z_wind[indices]/normaliz, s=10, c = Den_wind[indices]*prel.den_converter, norm = colors.LogNorm(vmin=1e-14, vmax=2e-9))
                img_xz = axd[2][j].scatter(X_wind[indices]/normaliz, Z_wind[indices]/normaliz, s=10, c = Den_wind[indices]*prel.den_converter, norm = colors.LogNorm(vmin=1e-14, vmax=2e-9))
                axd[0][j].set_title(f'Section: {label_obs[j]}', fontsize = 16)

                imgV_xy = axV[0][j].scatter(X_wind[indices]/normaliz, Y_wind[indices]/normaliz, s=10, c = v_rad_wind[indices]*conversion_sol_kms, norm = colors.LogNorm(vmin=1e3, vmax=2e4))
                imgV_yz = axV[1][j].scatter(Y_wind[indices]/normaliz, Z_wind[indices]/normaliz, s=10, c = v_rad_wind[indices]*conversion_sol_kms, norm = colors.LogNorm(vmin=1e3, vmax=2e4))
                imgV_xz = axV[2][j].scatter(X_wind[indices]/normaliz, Z_wind[indices]/normaliz, s=10, c = v_rad_wind[indices]*conversion_sol_kms, norm = colors.LogNorm(vmin=1e3, vmax=2e4))
                axV[0][j].set_title(f'Section: {label_obs[j]}', fontsize = 16)

                imgB_xy = axB[0][j].scatter(X_wind[indices]/normaliz, Y_wind[indices]/normaliz, s=10, c = bern_wind[indices], cmap = 'coolwarm', vmin = - 50, vmax= 50)
                imgB_yz = axB[1][j].scatter(Y_wind[indices]/normaliz, Z_wind[indices]/normaliz, s=10, c = bern_wind[indices], cmap = 'coolwarm', vmin = - 50, vmax= 50)
                imgB_xz = axB[2][j].scatter(X_wind[indices]/normaliz, Z_wind[indices]/normaliz, s=10, c = bern_wind[indices], cmap = 'coolwarm', vmin = - 50, vmax= 50)
                axB[0][j].set_title(f'Section: {label_obs[j]}', fontsize = 16)

                imgOE_xy = axOE[0][j].scatter(X_wind[indices]/normaliz, Y_wind[indices]/normaliz, s=10, c = OE_spec_wind[indices], cmap = 'coolwarm', vmin = - 50, vmax= 50)
                imgOE_yz = axOE[1][j].scatter(Y_wind[indices]/normaliz, Z_wind[indices]/normaliz, s=10, c = OE_spec_wind[indices], cmap = 'coolwarm', vmin = - 50, vmax= 50)
                imgOE_xz = axOE[2][j].scatter(X_wind[indices]/normaliz, Z_wind[indices]/normaliz, s=10, c = OE_spec_wind[indices], cmap = 'coolwarm', vmin = - 50, vmax= 50)
                axOE[0][j].set_title(f'Section: {label_obs[j]}', fontsize = 16)
                
                # axOEB.scatter(theta_ourConv, np.abs(bern_wind[indices]/OE_spec_wind[indices]), s=10, c = colors_obs[j], label = label_obs[j])
        
        if plot:
            cbar = plt.colorbar(img_xy)
            cbar.set_label(r'Density [g cm$^{-3}$]', fontsize = 18)
            cbar = plt.colorbar(img_yz)
            cbar.set_label(r'Density [g cm$^{-3}$]', fontsize = 18)
            cbar = plt.colorbar(img_xz)
            cbar.set_label(r'Density [g cm$^{-3}$]', fontsize = 18)
            figd.savefig(f'{abspath}/Figs/{folder}/Wind/{choice}/{wind_cond}scatter/Den_scatter{snap}_first3.png', dpi = 150)

            cbar = plt.colorbar(imgV_xy)
            cbar.set_label(r'$v_{\rm r}$ [km/s]', fontsize = 18)
            cbar = plt.colorbar(imgV_yz)
            cbar.set_label(r'$v_{\rm r}$ [km/s]', fontsize = 18)
            cbar = plt.colorbar(imgV_xz)
            cbar.set_label(r'$v_{\rm r}$ [km/s]', fontsize = 18)
            figV.savefig(f'{abspath}/Figs/{folder}/Wind/{choice}/{wind_cond}scatter/rad_scatter{snap}_first3.png', dpi = 150)

            cbar = plt.colorbar(imgB_xy)
            cbar.set_label(r'$\mathcal{B}$', fontsize = 18)
            cbar = plt.colorbar(imgB_yz)
            cbar.set_ticks([-50, 0, 50])
            cbar.set_ticklabels([r'$<0$', '0', r'$>0$'])
            cbar.set_label(r'$\mathcal{B}$', fontsize = 18)
            cbar.set_ticks([-50, 0, 50])
            cbar.set_ticklabels([r'$<0$', '0', r'$>0$'])
            cbar = plt.colorbar(imgB_xz)
            cbar.set_ticks([-50, 0, 50])
            cbar.set_ticklabels([r'$<0$', '0', r'$>0$'])
            cbar.set_label(r'$\mathcal{B}$', fontsize = 18)
            figB.savefig(f'{abspath}/Figs/{folder}/Wind/{choice}/{wind_cond}scatter/Bernoulli_scatter{snap}_first3.png', dpi = 150)

            cbar = plt.colorbar(imgOE_xy)
            cbar.set_label(r'spec OE', fontsize = 18)
            cbar = plt.colorbar(imgOE_yz)
            cbar.set_ticks([-50, 0, 50])
            cbar.set_ticklabels([r'$<0$', '0', r'$>0$'])
            cbar.set_label(r'spec OE', fontsize = 18)
            cbar.set_ticks([-50, 0, 50])
            cbar.set_ticklabels([r'$<0$', '0', r'$>0$'])
            cbar = plt.colorbar(imgOE_xz)
            cbar.set_ticks([-50, 0, 50])
            cbar.set_ticklabels([r'$<0$', '0', r'$>0$'])
            cbar.set_label(r'spec OE', fontsize = 18)
            figOE.savefig(f'{abspath}/Figs/{folder}/Wind/{choice}/{wind_cond}scatter/OEspec_scatter{snap}_first3.png', dpi = 150)

            # axOEB.axvline(-np.pi/2, c='grey', ls='--')
            # axOEB.axvline(np.pi/2, c='grey', ls='--')
            # axOEB.set_yscale('log')
            # axOEB.set_xlabel(r'$\phi$ [rad]', fontsize = 18)
            # axOEB.set_ylabel(r'$|\mathcal{B}/OE|$', fontsize = 18)
            # axOEB.legend(fontsize = 18)

        if r_chosen > apo:
            data = np.concatenate([mwind, Lum_fs, Lkin])
        else:        
            data = mwind

    return data

if __name__ == '__main__':
    if compute: # compute dM/dt = dM/dE * dE/dt
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
            
            data_wind = Mdot_sec(path, snap, r_chosen, with_who, choice, wind_cond = wind_cond, how = how)
            data_tosave = np.concatenate(([snap], [tfb[i]], data_wind))  
            csv_path = f'{abspath}/data/{folder}/wind/{choice}/Mdot{wind_cond}{with_who}{n_obs}Sec{how}_{check}{which_r_title}{choice}.csv'
            if alice:
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                        if r_chosen > apo:
                            writer.writerow(['snap', 'tfb'] + [f'Mw {lab}' for lab in label_obs] + [f'Lum_fs {lab}' for lab in label_obs] + [f'Lkin {lab}' for lab in label_obs])
                        else:
                            writer.writerow(['snap', 'tfb'] + [f'Mw {lab}' for lab in label_obs])
                    writer.writerow(data_tosave)
                file.close()

    if plot:
        which_r_title = '05amin'
        fig, (axEdd, axall, axfb) = plt.subplots(1, 3, figsize = (24, 7))

        fallback = \
                np.loadtxt(f'{abspath}/data/{folder}/paper1/wind/Mdot_{check}05aminmean.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True)
        tfbfb, mfb, mwind_dimCellOld = fallback[1], fallback[2], fallback[3]

        # wind_all = \
        #         np.loadtxt(f'{abspath}/data/{folder}/wind/MdotSec_{check}{which_r_title}all.csv', 
        #                 delimiter = ',', 
        #                 skiprows=1, 
        #                 unpack=True) 
        # tfbH_full, Mw_full = wind_all[1], wind_all[2] # MW_full is the same as mwind_dimCellOld
        
        wind = \
                np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/MdotSec_{check}{which_r_title}{choice}.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True) 
        tfbH = wind[1]
        rest = wind[2:]

        wind_NOnorm = \
                np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/MdotNOnormSec_{check}{which_r_title}{choice}.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True) 
        tfbH_NOnorm = wind_NOnorm[1]
        rest_NOnorm = wind_NOnorm[2:]

        wind_mean = \
                np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/MdotSecmean_{check}{which_r_title}{choice}.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True) 
        tfbH_mean = wind_mean[1]
        rest_mean = wind_mean[2:]

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
        axEdd.plot(tfbfb, np.abs(mfb)/Medd_sol, c = 'grey', ls = '--', label = r'$|\dot{M}_{\rm fb}|$')
        Mw_sum = np.zeros(len(tfbH))
        for i in range(len(rest)):
            axEdd.plot(tfbH, rest[i]/Medd_sol, c = colors_obs[i], label = label_obs[i])
            Mw_sum += rest[i]
            # axEdd.plot(tfbH_Bound, rest_Bound[i]/Medd_sol, c = colors_obs[0], ls = '--', label = r'$\dot{M}_{\rm out, b}$')
            # axEdd.plot(tfbH_mean, rest_mean[i]/Medd_sol, c = colors_obs[i], ls = '--', label = f'Mean' if i==0 else None)
            axEdd.plot(tfbH_NOnorm, rest_NOnorm[i]/Medd_sol, c = colors_obs[i], ls = ':', label = f'No norm' if i==0 else None)
            # axEdd.plot(tfbH_obs, rest_obs[i]/Medd_sol, c = colors_obs[i], ls = ':', label = f'Obs' if i==0 else None)
            # axEdd.plot(tfbH_obs8, rest_obs8[i]/Medd_sol, c = colors_obs[i], ls = '-.', label = f'Obs8' if i==0 else None)
            # axEdd.plot(tfbH_OE, rest_OE[i]/Medd_sol, c = colors_obs[i], ls = '--', label = f'OE cut' if i==0 else None)
        # axEdd.plot(tfbH, Mw_sum/Medd_sol, c = 'black', ls = '-', label = 'Total')
        # axEdd.plot(tfbH_full, Mw_full/Medd_sol, c = 'orchid', label = 'all')
        # axEdd.plot(tfbfb, mwind_dimCellOld/Medd_sol, c = 'gold', ls = '--', label = r'paper1')

        for i in range(len(rest)):
            axall.plot(tfbH, rest[i]/Mw_sum, c = colors_obs[i], label = label_obs[i])
            axfb.plot(tfbH[6:], np.abs(rest[i]/mfb)[6:], c = colors_obs[i], label = label_obs[i])
        
        # integrate mwind_dimCell in tfb 
        # mwind_dimCell_int = cumulative_trapezoid(np.abs(mwind_dimCell), tfb, initial = 0)
        # mfall_int = cumulative_trapezoid(np.abs(mfall), tfb, initial = 0)
        # print(f'integral of Mw at the last time: {mwind_dimCell_int[-1]/mstar} Mstar')
        # print(f'integral of Mfb at the last time: {mfall_int[-1]/mstar} Mstar')
        # print(f'End of simualation, Mw/Mfb in {check}:', np.abs(mwind_dimCell[-1]/mfall[-1]))
        
        original_ticks = axEdd.get_xticks()
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]    
        for ax in [axEdd, axall, axfb]:
            ax.set_yscale('log')
            ax.set_xlabel(r'$t [t_{\rm fb}]$')
            ax.set_xticks(new_ticks)
            ax.set_xticklabels(labels)  
            ax.set_xlim(0, np.max(tfbH))
            ax.tick_params(axis='both', which='major', width=1.2, length=9)
            ax.tick_params(axis='both', which='minor', width=1, length=5)
            ax.legend(fontsize = 18)
            ax.grid()
        axEdd.set_ylim(1e1, 7e6)
        axEdd.set_ylabel(r'$\dot{M}_{{\rm w}} [\dot{M}_{\rm Edd}]$')   
        axall.set_ylim(5e-2, 1.1)
        axall.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm w}]$')
        axfb.set_ylim(1e-3, 2)
        axfb.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm fb}]$')
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

        if which_r_title not in ['05amin', 'amin']: # plot energies
            Lum_fsI, Lum_fsO, Lum_fsN, Lum_fsS, LkinI, LkinO, LkinN, LkinS = \
                wind[6], wind[7], wind[8], wind[9], wind[10], wind[11], wind[12], wind[13]
            fig, ax = plt.subplots(1, 1, figsize = (10, 7))
            ax.plot(tfbH, np.abs(Lum_fsO)/(4*Ledd_sol), c = colors_obs[0], label = r'$L_{\rm fs}$')
            ax.plot(tfbH, np.abs(Lum_fsO)/(4*Ledd_sol), c = colors_obs[0], label = label_obs[0])
            ax.plot(tfbH, np.abs(Lum_fsI)/(4*Ledd_sol), c = colors_obs[1], label = label_obs[1])
            ax.plot(tfbH, np.abs(Lum_fsN)/(4*Ledd_sol), c = colors_obs[2], label = label_obs[2])
            ax.plot(tfbH, np.abs(LkinO)/(4*Ledd_sol), c = colors_obs[0], ls = '--', label = r'$L_{\rm kin}$')
            ax.plot(tfbH, np.abs(LkinI)/(4*Ledd_sol), c = colors_obs[1], ls = '--')
            ax.plot(tfbH, np.abs(LkinN)/(4*Ledd_sol), c = colors_obs[2], ls = '--')
            original_ticks = ax.get_xticks()
            midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
            new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
            labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]    
            ax.set_yscale('log')
            ax.set_xlabel(r'$t [t_{\rm fb}]$')
            ax.set_xticks(new_ticks)
            ax.set_xticklabels(labels)  
            ax.tick_params(axis='both', which='major', width=1.2, length=9)
            ax.tick_params(axis='both', which='minor', width=1, length=5)
            ax.set_ylabel(r'$L [L_{\rm Edd}]$')   
            ax.set_xlim(0, np.max(tfbH))
            ax.set_ylim(1e-4, 1e3)
            ax.legend(fontsize = 18)
            ax.grid()
            plt.suptitle(rf'r = {which_r_title}', fontsize = 20)
            fig.tight_layout()



    # %%
