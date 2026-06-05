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
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from Utilities.operators import make_tree
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices
import src.orbits as orb
import csv
import os
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
choice = 'left_right_z'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

#%%
# MAIN
##
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
norm = things['E_mb']
amin = things['a_mb'] # semimajor axis of the bound orbit

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

    # compute the outflow/wind mass for all the snapshots
    for i, snap in enumerate(snaps):
        time = tfb[i]
        print(snap, flush = True)
        pathfold = f'{path}/{folder}/snap_{snap}'
        data = make_tree(pathfold, snap)
        X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Press, data.VX, data.VY, data.VZ, data.IE, data.Rad
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        dim_cell = Vol**(1/3)
        # find the spherical shell with r = r_chosen
        cut = Den > 1e-19
        X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
            make_slices([X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den], cut)
        indices_allsec, label_obs = split_cells(X, Y, Z, choice)

        cut_wind, _, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')
        cut_out = V_r > 0 
        X_out, Y_out, Z_out, Mass_out = make_slices([X, Y, Z, Mass], cut_out)
        X_wind, Y_wind, Z_wind, Mass_wind = make_slices([X, Y, Z, Mass], cut_wind)
        indices_allsec_wind, label_obs = split_cells(X_wind, Y_wind, Z_wind, choice)
        indices_sec_out, _ = split_cells(X_out, Y_out, Z_out, choice)
        
        tot_M = np.zeros(len(indices_allsec_wind))
        M_out = np.zeros(len(indices_sec_out))
        M_wind = np.zeros(len(indices_allsec_wind))

        for i in range(len(indices_allsec_wind)):
            i_singlesec = indices_allsec[i]
            tot_M[i] = np.sum(Mass[i_singlesec])
            i_singlesec_out = indices_sec_out[i]
            mass_out = Mass_out[i_singlesec_out] if Mass_out.size > 0 else np.array([0])
            M_out[i] = np.sum(mass_out) 
            i_singlesec_wind = indices_allsec_wind[i] 
            mass_w = Mass_wind[i_singlesec_wind] if Mass_wind.size > 0 else np.array([0])
            M_wind[i] = np.sum(mass_w) 

        data = np.concatenate([[snap, time], tot_M, M_out, M_wind])

        csv_path = f'{abspath}/data/{folder}/wind/Mass_unbound{choice}.csv'
        with open(csv_path,'a', newline='') as file:
            writer = csv.writer(file)
            if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                writer.writerow(['snap', 'tfb'] + [f'M_tot {lab}' for lab in label_obs] + [f'M_out {lab}' for lab in label_obs] + [f'M_w {lab}' for lab in label_obs])
            writer.writerow(data)
            file.close()
        
if plot:
    csv_path = f'{abspath}/data/{folder}/wind/Mass_unbound{choice}.csv'
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    snap = data['snap']
    tfb = data['tfb']
    M_out = np.array([data[f'M_out {lab}'] for lab in label_obs])
    M_wind = np.array([data[f'M_w {lab}'] for lab in label_obs])
    plt.figure(figsize=(8,6))
    for i, lab in enumerate(label_obs):
        plt.plot(tfb, M_out[i], label = f'M_out {lab}')
        plt.plot(tfb, M_wind[i], label = f'M_wind {lab}')
    plt.xlabel('t/t_fb')
    plt.ylabel('Mass [g]')
    plt.legend()
    plt.title(f'Mass unbound, {choice}')
    plt.savefig(f'{abspath}/plots/{folder}/wind/Mass_unbound{choice}.png')
    plt.show()