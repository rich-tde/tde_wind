""" Compute the time evolution of Mdot wind across a spherical surface"""
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
    import scipy.integrate as sci
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib import lines as MdotMines

import numpy as np
import csv
import os
import healpy as hp
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree, choose_sections, choose_observers
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
choice = 'split_stream' # 'left_right_in_out_z', 'left_right_z', 'all' or 'in_out_z', 'thirties'
how = 'isot' # '' for the normalized sum or 'mean' for mean of Mw of each cells
what = 'wind' # '' for wind or 'boundOut'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 # converted to seconds
Rt = things['Rt']
apo = things['apo']
amin = things['a_mb'] # semimajor axis of the bound orbit

Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs 
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
#%%
# MAIN

def split_cells(X, Y, Z, choice):
    indices = np.arange(len(X))
    indices_sec = []
    sections = choose_sections(X, Y, Z, choice)
    cond_sec = []
    label_obs = []
    for key in sections.keys():
        cond_sec.append(sections[key]['cond'])
        label_obs.append(sections[key]['label'])

    for j, cond in enumerate(cond_sec):
        # select the particles in the chosen section and at the chosen radius
        condR = cond #np.logical_and(np.abs(Rsph-r_chosen) < dim_cell, cond)
        indices_sec.append(indices[condR])
    
    return indices_sec, label_obs
    
def Mdot_sec(path, snap, r_chosen, choice, what, how):
    global label_obs
    # Load data and pick the ones unbound and with positive velocity
    data_snap = make_tree(path, snap)
    X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
        data_snap.X, data_snap.Y, data_snap.Z, data_snap.Vol, data_snap.Den, data_snap.Mass, data_snap.Press, data_snap.VX, data_snap.VY, data_snap.VZ, data_snap.IE, data_snap.Rad
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)
    dim_cell = Vol**(1/3)
    # find the spherical shell with r = r_chosen
    cut = np.logical_and(Den > 1e-19, np.abs(Rsph - r_chosen) < dim_cell)
    X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
        make_slices([X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den], cut)
    
    if X.size == 0:
        return {
            lab: {
                'mwind': np.nan,
                'Lum_fs': np.nan,
                'Lkin': np.nan,
                'Ekin': np.nan,
                'area': np.nan,
                'mass': np.nan,
                'R': np.nan
            } for lab in label_obs} 

    cut_wind, bern, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')

    if what == 'wind':
        cutM = cut_wind

    if what == 'boundOut':
        cutM = np.logical_and(V_r > 0, bern < 0)

    X_wind, Y_wind, Z_wind, R_wind, Den_wind, Mass_wind, v_rad_wind, dim_cell_wind, Rad_den_wind = \
        make_slices([X, Y, Z, Rsph, Den, Mass, V_r, dim_cell, Rad_den], cutM)
    if Den_wind.size == 0:
        return {
            lab: {
                'mwind': np.nan,
                'Lum_fs': np.nan,
                'Lkin': np.nan,
                'Ekin': np.nan,
                'area': np.nan,
                'mass': np.nan,
                'R': np.nan 
            } for lab in label_obs} 

    Mdot = np.pi * dim_cell_wind**2 * Den_wind * v_rad_wind 
    indices_sec, _ = split_cells(X_wind, Y_wind, Z_wind, choice)

    data = {}
    C_mult = 4/len(indices_sec) # to have the right normalization in all cases
    # if not alice:
    #     fig, ((axd, axR, axdim), (axX, axY, axZ)) = plt.subplots(2,3, figsize = (18, 12))
    #     R_wind = np.sqrt(X_wind**2 + Y_wind**2 + Z_wind**2)
    #     for ax in [axd, axR, axdim, axX, axY, axZ]:
    #         ax.set_yscale('log')
    #         ax.grid()
    #     axd.set_xscale('log')
    #     axd.set_xlabel(r'$\rho$ [g/cm$^3$]')
    #     axd.set_ylabel(r'$N_{\rm cell}$')
    #     axR.set_xlabel(r'$r/r_{\rm t}$')
    #     axdim.set_xlabel(r'$r_{\rm cell}/r_{\rm t}$')
    #     axX.set_ylabel(r'$N_{\rm cell}$')
    #     axX.set_xlabel(r'$X/r_{\rm t}$')
    #     axY.set_xlabel(r'$Y/r_{\rm t}$')
    #     axZ.set_xlabel(r'$Z/r_{\rm t}$')
    #     fig.suptitle(f't = {tfb[i]:.2f} ' + r't$_{\rm fb}$', fontsize = 20)
    #     axR.axvline(r_chosen/Rt, c = 'k', ls = '--')
    for j, indices in enumerate(indices_sec):
        # if not alice: 
        #     if j not in [0, 1]: 
        #         continue
        #     # ratio = dim_cell_wind[indices]/np.abs(R_wind[indices]-r_chosen)
        #     # print(ratio[ratio<=1])
        #     counts_d, bin_d = np.histogram(Den_wind[indices], bins = 80)
        #     counts_R, bin_R = np.histogram(R_wind[indices], bins = 80)
        #     counts_dim, bin_dim = np.histogram(dim_cell_wind[indices], bins = 80)
        #     counts_X, bin_X = np.histogram(X_wind[indices], bins = 80)
        #     counts_Y, bin_Y = np.histogram(Y_wind[indices], bins = 80)
        #     counts_Z, bin_Z = np.histogram(Z_wind[indices], bins = 80)
        #     axd.plot(bin_d[:-1]*prel.den_converter, counts_d, label = label_obs[j], color = color_obs[j])
        #     axR.plot(bin_R[:-1]/Rt, counts_R, color = color_obs[j])
        #     axdim.plot(bin_dim[:-1]/Rt, counts_dim, color = color_obs[j])
        #     axX.plot(bin_X[:-1]/Rt, counts_X, color = color_obs[j])
        #     axY.plot(bin_Y[:-1]/Rt, counts_Y, color = color_obs[j])
        #     axZ.plot(bin_Z[:-1]/Rt, counts_Z, color = color_obs[j])
            
        # select the particles in the chosen section and at the chosen radius
        if how == '':   
            mwind = C_mult * r_chosen**2 * np.sum(Mdot[indices]) / np.sum(dim_cell_wind[indices]**2)
            Lum_fs = C_mult * r_chosen**2 * np.pi * np.sum(Rad_den_wind[indices] * dim_cell_wind[indices]**2) * prel.csol_cgs / np.sum(dim_cell_wind[indices]**2)
            Lkin = 0.5 * C_mult * r_chosen**2 * np.sum(Mdot[indices] * v_rad_wind[indices]**2) / np.sum(dim_cell_wind[indices]**2)
            Ekin = 0.5 * C_mult * r_chosen**2 * np.sum(Mass_wind[indices] * v_rad_wind[indices]**2) / np.sum(dim_cell_wind[indices]**2)
        elif how == 'isot':   
            mwind = np.sum(Mdot[indices])
            Lum_fs = np.pi * np.sum(Rad_den_wind[indices] * dim_cell_wind[indices]**2) * prel.csol_cgs
            Lkin = 0.5 * np.sum(Mdot[indices] * v_rad_wind[indices]**2)
            Ekin = 0.5 * np.sum(Mass_wind[indices] * v_rad_wind[indices]**2)
        elif how == 'mean': 
            mwind = C_mult * np.pi * r_chosen**2 * np.mean(Den_wind[indices] * v_rad_wind[indices])
            Lum_fs = C_mult * np.pi * r_chosen**2 * np.mean(Rad_den_wind[indices]) * prel.csol_cgs
            # Lkin = 0.5 * np.mean(Mdot[indices] * v_rad_wind[indices]**2)
            Lkin = 0.5 * C_mult * np.pi * r_chosen**2 * np.mean(Den_wind[indices] * v_rad_wind[indices]**3) 
            Ekin = 0.5 * C_mult * np.pi * r_chosen**2 * np.mean(Mass_wind[indices] * v_rad_wind[indices]**3) 
        mass = np.sum(Mass_wind[indices])
        Ravg = np.sum(R_wind[indices] * Mass_wind[indices]) / np.sum(Mass_wind[indices])
        area = np.pi * np.sum(dim_cell_wind[indices]**2)
        data[label_obs[j]] = {'mwind': mwind, 'Lum_fs': Lum_fs, 'Lkin': Lkin, 'Ekin': Ekin, 'area': area, 'mass': mass, 'R': Ravg}

    return data

if __name__ == '__main__':
    NPIX = hp.nside2npix(prel.NSIDE)
    observers_xyz = hp.pix2vec(prel.NSIDE, range(NPIX))
    observers_xyz = np.array(observers_xyz)
    _, label_obs, color_obs, _, _, _ = choose_observers(observers_xyz, choice)
        
    if compute: 
        r_chosen = 0.5 * amin
        which_r_title = '05amin' 
        snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

        save_path = f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}{which_r_title}{choice}_{what}.npy'
        if os.path.exists(save_path):
            all_data = np.load(save_path, allow_pickle=True).item()
        else:
            all_data = {}
        for i, snap in enumerate(snaps):
            if alice:
                path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
            else: 
                if snap not in [109, 151]:
                    continue
                path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
            print(snap, flush=True)

            if snap in all_data:
                print(f'Snapshot {snap} already computed, skipping', flush=True)
                continue

            data_wind = Mdot_sec(path, snap, r_chosen, choice, what, how)
            all_data[snap] = {'tfb': tfb[i], **data_wind}

            np.save(save_path, all_data, allow_pickle=True)

    else:
        r_chosen = 2 * apo
        which_r_title = '2apo'
        
        dataMass = np.loadtxt(f'{abspath}/data/{folder}/wind/Mass_unbound{choice}.csv', 
                                delimiter=',', skiprows=1, unpack=True)
        tfbMass = dataMass[1]
        M_tot = dataMass[2:2+len(label_obs)] 
        M_wind = dataMass[2+2*(len(label_obs)):2+3*(len(label_obs))] 
    
        for i in range(len(label_obs)):
            M_wind[i, :] -= M_wind[i, 0]

        figM, (axM, axEkin) = plt.subplots(1,2, figsize = (16,8))
        figMass_tot, axMass_tot = plt.subplots(1,1, figsize = (9,8))
        
        fallback = \
                np.loadtxt(f'{abspath}/data/{folder}/1.paperEdd/wind/Mdot_{check}05aminmean.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True)
        tfbfb, mfb, mwind_dimCellOld = fallback[1], fallback[2], fallback[3]
        tfb_to_int = tfbfb * 24 * 3600 / prel.tsol_cgs
        where_nan = np.isnan(mfb)
        mfb[where_nan] = 0
        mass_fb = sci.cumulative_trapezoid(np.abs(mfb), tfb_to_int, initial = 0)

        # wind = \
        #         np.loadtxt(f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}{which_r_title}{choice}_{what}.csv', 
        #                 delimiter = ',', 
        #                 skiprows=1, 
        #                 unpack=True) 
        # tfbH = wind[1]
        # rest = wind[2:2+len(label_obs)]
        wind = np.load(f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}{which_r_title}{choice}_{what}.npy', allow_pickle=True).item()
        snaps = list(wind.keys())
        tfbH = np.array([wind[snap]['tfb'] for snap in snaps])
        mwind = np.array([[wind[snap][label]['mwind'] for label in label_obs] for snap in snaps]).T
        Ekin = np.array([[wind[snap][label]['Ekin'] for label in label_obs] for snap in snaps]).T
        # masswind = np.array([[wind[snap][label]['mass'] for label in label_obs] for snap in snaps]).T
        if how == 'isot':
            # area = wind[2+3*len(label_obs):2+4*len(label_obs)]
            area = np.array([[wind[snap][label]['area'] for label in label_obs] for snap in snaps]).T
            mwind_isot =  4 * np.pi * r_chosen**2 * mwind/ area 

        # extra_time_tfb = np.linspace(tfbH[-1], 40, 100)
        # all_time = np.concatenate([tfbH, extra_time_tfb])
        # extend_Mdot = []
        # for i in range(len(label_obs)): 
        #     extend_Mdot.append(np.concatenate([mwind[i], mwind[i][-1]*np.ones(len(extra_time_tfb))]))
        # extend_Mdot = np.array(extend_Mdot)

        handles_colorMass = []
        labels_handlesMass = []

        handles_colorMdot = []
        labels_handlesMdot = []
        for i in range(len(mwind)):
            if label_obs[i] in ['Eccentric flow side','South pole']:
                continue
            axM.plot(tfbH, mwind[i]/Medd_sol, c = color_obs[i], ls = '--')
            line = axM.plot(tfbH, mwind_isot[i]/Medd_sol,  label = label_obs[i], c = color_obs[i])[0]
            handles_colorMdot.append(line)
            labels_handlesMdot.append(label_obs[i])

            # plot Ekin
            axEkin.plot(tfbH, Ekin[i]*prel.en_converter, label = label_obs[i], c = color_obs[i])

            # plot Mass
            # axMass.plot(tfbH, masswind[i]/(mstar/2), label = label_obs[i], c = color_obs[i])

            # total mass
            # axM.plot(all_time, extend_Mdot[i]/Medd_sol,  label = label_obs[i], c = color_obs[i])
            # from Mdot
            time_to_int = tfbH * 24 * 3600 / prel.tsol_cgs # convert to code units
            where_nan = np.isnan(mwind[i])
            mwind[i][where_nan] = 0
            mass = sci.cumulative_trapezoid(mwind[i], time_to_int, initial = 0)
            axMass_tot.plot(tfbH, mass/(mstar/2), label = label_obs[i], c = color_obs[i], ls = '--')
            # extending 
            # time_to_int = all_time * 24 * 3600 / prel.tsol_cgs # convert to code units
            # where_nan = np.isnan(extend_Mdot[i])
            # extend_Mdot[i][where_nan] = 0
            # mass = sci.cumulative_trapezoid(extend_Mdot[i], time_to_int, initial = 0)
            # axMass_tot.plot(all_time, mass/(mstar/2), label = label_obs[i], c = color_obs[i], ls = '--')
            line = axMass_tot.plot(tfbMass, M_wind[i]/(mstar/2), label = label_obs[i], c = color_obs[i])[0]
            handles_colorMass.append(line)
            labels_handlesMass.append(label_obs[i])

        line = axM.plot(tfbfb, np.abs(mfb)/Medd_sol, c = 'k', ls = ':', label = r'$|\dot{M}_{\rm fb}|$')[0]
        handles_colorMdot.append(line)
        labels_handlesMdot.append(r'$|\dot{M}_{\rm fb}|$')

        line = axMass_tot.plot(tfbfb, mass_fb/(mstar/2), c = 'k', ls = ':', label = r'$|\dot{M}_{\rm fb}|$')[0]
        handles_colorMass.append(line)
        labels_handlesMass.append(r'$|\dot{M}_{\rm fb}|$')

        original_ticks = axM.get_xticks()
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        labels = [str(np.round(tick,2)) if tick in original_ticks else '' for tick in new_ticks]    
        for ax in [axMass_tot, axM, axEkin]: 
            ax.set_yscale('log')
            ax.set_xlabel(r'$t [t_{\rm fb}]$')
            ax.set_xticks(new_ticks)
            ax.set_xticklabels(labels)  
            ax.set_xlim(0, np.max(tfbH))
            ax.tick_params(axis='both', which='major', width=1.2, length=9)
            ax.tick_params(axis='both', which='minor', width=1, length=5)
            ax.grid()
        axM.set_ylabel(r'$\dot{M} (\dot{M}_{\rm Edd})$' + f' at {which_r_title}')
        # axMass.set_ylabel(r'$M_{\rm w} (m_\star/2)$')
        axMass_tot.set_ylabel(r'$M_{\rm tot} (m_\star/2)$')
        axEkin.set_ylabel(r'$E_{\rm kin}$ [erg]' + f' at {which_r_title}')
        axEkin.set_ylim(1e42, 5e45)
        axMass_tot.set_ylim(1e-5, 1) 

        legend1 = axM.legend(handles=handles_colorMdot,
                            labels=labels_handlesMdot, loc='upper left',
                            fontsize=18)
        axM.add_artist(legend1)

        legend1 = axMass_tot.legend(handles=handles_colorMass,
                    labels=labels_handlesMass, loc='upper left',
                    fontsize=18)

        axMass_tot.add_artist(legend1)
        line_styles_parts = ['-', '--']
        labels_partsM = [r'isotropic $\dot{M}_{\rm w}$', r'sectoral $\dot{M}_{\rm w}$'] 
        labels_partsMass = [r'direct count $M_{\rm w}$', r'$\int \dot{M}_{\rm w} dt$']
        proxy_linesM = []
        proxy_linesMass = []
        for l, line in enumerate(line_styles_parts):
            proxy_linesM.append(
                            MdotMines.Line2D([0], [0], color='k', ls=line, linewidth=2,
                                        label=labels_partsM[l]))
            
            proxy_linesMass.append(
                MdotMines.Line2D([0], [0], color='k', ls=line, linewidth=2,
                            label=labels_partsMass[l]))

        axM.legend(handles=proxy_linesM, fontsize=20, loc='upper right')
        axMass_tot.legend(handles=proxy_linesMass, fontsize=20, loc='upper right')

        # axM.plot(tfbfb, np.abs(mfb)/Medd_sol, c = 'grey', ls = '--', label = r'$|\dot{M}_{\rm fb}|$')
        axM.set_ylim(1e2, 7e7)
        figM.suptitle(rf'$\dot{{M}}_{{\rm w}}$ at {which_r_title}', fontsize = 20)
        figM.tight_layout()
        # figM.savefig(f'{abspath}/Figs/{folder}/Wind/MdotSec_{which_r_title}{choice}.png', dpi = 150)

       