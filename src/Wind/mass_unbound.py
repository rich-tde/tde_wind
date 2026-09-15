""" If alice: Compute and save the unbound mass.
If local: plots"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    path = '/home/martirep/data_pi-rossiem/TDE_data'
else:
    abspath = '/Users/paolamartire/shocks'
    path = f'{abspath}/TDE'
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from Utilities.operators import make_tree
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices
import src.orbits as orb
from src.Wind.Mdot_Rfixed_sec import split_cells

#
# PARAMETERS
## 
m = 4
Mbh = 10**m
Mbh_cgs = Mbh * prel.Msol_cgs
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR'
choice = 'split_stream'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

#%%
# MAIN
##
params = [Mbh, Rstar, mstar, beta]

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    prepath = f'{path}/{folder}/snap_'

    csv_path = f'{abspath}/data/{folder}/wind/Mass_unbound{choice}.csv'
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        existing = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        existing_snaps = set(existing[:, 0].astype(int))
    else:
        existing_snaps = set()

    for snap_idx, snap in enumerate(snaps):
        if snap in existing_snaps:
            print(f'Snapshot {snap} already computed, skipping.', flush=True)
            continue
        time = tfb[snap_idx]
        print(snap, flush = True)
        pathfold = f'{prepath}{snap}'
        data = make_tree(pathfold, snap)
        X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Press, data.VX, data.VY, data.VZ, data.IE, data.Rad
        dim_cell = Vol**(1/3)
        cut = Den > 1e-19
        X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
            make_slices([X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den], cut)
        vel = np.sqrt(VX**2 + VY**2 + VZ**2)
        Ekin = 0.5 * Mass * vel**2
        indices_allsec, label_obs = split_cells(X, Y, Z, choice)

        cut_wind, _, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')
        cut_out = V_r > 0 
        X_out, Y_out, Z_out, Mass_out, Ekin_out = make_slices([X, Y, Z, Mass, Ekin], cut_out)
        X_wind, Y_wind, Z_wind, Mass_wind, Ekin_wind = make_slices([X, Y, Z, Mass, Ekin], cut_wind)
        indices_allsec_wind, label_obs = split_cells(X_wind, Y_wind, Z_wind, choice)
        indices_sec_out, _ = split_cells(X_out, Y_out, Z_out, choice)

        nsec = len(label_obs)
        tot_M_snap = np.zeros(nsec)
        M_out_snap = np.zeros(nsec)
        E_out_snap = np.zeros(nsec)
        M_wind_snap = np.zeros(nsec)
        E_wind_snap = np.zeros(nsec)

        for i in range(len(indices_allsec_wind)):
            i_singlesec = indices_allsec[i]
            tot_M_snap[i] = np.sum(Mass[i_singlesec])
            i_singlesec_out = indices_sec_out[i]
            mass_out_single = Mass_out[i_singlesec_out] #if Mass_out.size > 0 else np.array([0])
            M_out_snap[i] = np.sum(mass_out_single) 
            Ekin_out_single = Ekin_out[i_singlesec_out] 
            E_out_snap[i] = np.sum(Ekin_out_single)
            i_singlesec_wind = indices_allsec_wind[i] 
            mass_w = Mass_wind[i_singlesec_wind] 
            M_wind_snap[i] = np.sum(mass_w) 
            Ekin_wind_single = Ekin_wind[i_singlesec_wind] 
            E_wind_snap[i] = np.sum(Ekin_wind_single)

        data = np.concatenate([[snap, time], tot_M_snap, M_out_snap, M_wind_snap, E_out_snap, E_wind_snap])

        with open(csv_path,'a', newline='') as file:
            writer = csv.writer(file)
            if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                writer.writerow(['snap', 'tfb'] + 
                                [f'M_tot {lab}' for lab in label_obs] + 
                                [f'M_out {lab}' for lab in label_obs] + 
                                [f'M_w {lab}' for lab in label_obs] + 
                                [f'Ekin_out {lab}' for lab in label_obs] + 
                                [f'Ekin_w {lab}' for lab in label_obs])
            writer.writerow(data)
        del data, X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den


if plot:
    import healpy as hp
    from src.Wind.Mdot_Rfixed_sec import choose_observers
    observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz)
    indices_sorted, label_obs, colors_obs, _, _, _ = choose_observers(observers_xyz, choice = choice)

    csv_path = f'{abspath}/data/{folder}/wind/Mass_unbound{choice}.csv'
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1, unpack=True)
    tfb = data[1]
    M_tot = data[2:2+len(label_obs)] 
    M_out = data[2+(len(label_obs)):2+2*(len(label_obs))] 
    M_wind = data[2+2*(len(label_obs)):2+3*(len(label_obs))] 
    E_out = data[2+3*(len(label_obs)):2+4*(len(label_obs))] 
    E_wind = data[2+4*(len(label_obs)):2+5*(len(label_obs))] 

    for i in range(len(label_obs)):
        M_out[i, :] -= M_wind[i, 0]
        M_wind[i, :] -= M_wind[i, 0]
    plt.figure(figsize=(8,6))
    for i, lab in enumerate(label_obs):
        if lab == 'South pole':
            continue
        plt.plot(tfb, M_out[i]/mstar, c = colors_obs[i],  ls = '--' )
        plt.plot(tfb, M_wind[i]/mstar, c = colors_obs[i], label = lab)
        # print(lab, 'outflow/half star mass: ', np.median(M_out[i, -3:])/(0.5*mstar), ', wind/out: ', np.median(M_wind[i, -3:])/np.median(M_out[i, -3:]))
        # print(lab, 'wind/half star mass: ', M_wind[i, -1]/0.5)
    # print('sum stream: ', (np.median(M_out[0, -3:])+np.median(M_out[1, -3:])+np.median(M_out[2, -3:]))/(0.5*mstar), ', sum wind/out: ', np.sum(np.median(M_wind[0, -3:])+np.median(M_wind[1, -3:])+np.median(M_wind[2, -3:]))/np.sum(np.median(M_out[0, -3:])+np.median(M_out[1, -3:])+np.median(M_out[2, -3:])))
    # print('sum mid+high stream: ', (np.median(M_out[1, -3:])+np.median(M_out[2, -3:]))/(0.5*mstar), ', sum wind/out: ', np.sum(np.median(M_wind[1, -3:])+np.median(M_wind[2, -3:]))/np.sum(np.median(M_out[1, -3:])+np.median(M_out[2, -3:])))
    # print('sum wind: ', np.sum(M_wind[0, -3:]+M_wind[2, -3:]+M_wind[1, -3:])/0.5)
    plt.xlabel(r'$t/t_{\rm fb}$')
    plt.ylabel(r'$M/M_\star$')    
    plt.yscale('log')
    plt.ylim(1e-4, 1.2)
    plt.legend(fontsize = 16)
    plt.show()
# %%
