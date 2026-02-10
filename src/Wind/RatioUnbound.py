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
choice = 'left_right_in_out_z' # 'left_right_in_out_z', 'left_right_z', 'all' or 'in_out_z'
wind_cond = 'OE' # '' for bernouilli coeff or 'OE' for orbital energy

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
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

# MAIN
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
    
def Mdot_sec(path, snap, r_chosen, choice, wind_cond = ''):
    # Load data and pick the ones unbound and with positive velocity
    data = make_tree(path, snap, energy = True)
    X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Press, data.VX, data.VY, data.VZ, data.IE, data.Rad
    dim_cell = Vol**(1/3)
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)
    cut = np.logical_and(Den > 1e-19, np.abs(Rsph - r_chosen) < dim_cell)
    X, Y, Z, Rsph, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
        make_slices([X, Y, Z, Rsph, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den], cut)
    V = np.sqrt(VX**2 + VY**2 + VZ**2)
    bern = orb.bern_coeff(Rsph, V, Den, Mass, Press, IE_den, Rad_den, params)
    v_rad, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
    OE = orb.orbital_energy(Rsph, V, Mass, params, prel.G)

    # select just outflowing and unbound material
    if wind_cond == '':
        cond_wind = np.logical_and(v_rad >= 0, bern > 0)
    if wind_cond == 'OE': 
        cond_wind = np.logical_and(v_rad >= 0, OE > 0)
    X_wind, Y_wind, Z_wind = make_slices([X, Y, Z], cond_wind)
    
    indices_sec, _, _ = split_cells(X_wind, Y_wind, Z_wind, choice)
    ratio_u = np.zeros(len(indices_sec))

    for j, indices in enumerate(indices_sec):
    # select the particles in the chosen section and at the chosen radius
        ratio_u[j] = len(X_wind[indices]) / len(X)
        
    return ratio_u

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
            
            data_wind = Mdot_sec(path, snap, r_chosen, choice, wind_cond = wind_cond)
            data_tosave = np.concatenate(([snap], [tfb[i]], data_wind))  
            csv_path = f'{abspath}/data/{folder}/wind/{choice}/RatioUnb{wind_cond}Sec_{check}{which_r_title}{choice}.csv'
            if alice:
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                        writer.writerow(['snap', 'tfb'] + [f'{lab}' for lab in label_obs])
                    writer.writerow(data_tosave)
                file.close()

    if plot:
        which_r_title = '05amin'
        fig, (axB, axOE) = plt.subplots(1, 2, figsize = (21, 7))

        wind = \
                np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/RatioUnbSec_{check}{which_r_title}{choice}.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True) 
        tfbH = wind[1]
        rest = wind[2:]

        wind_OE = \
                np.loadtxt(f'{abspath}/data/{folder}/wind/{choice}/RatioUnbOESec_{check}{which_r_title}{choice}.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True) 
        tfbH_OE = wind_OE[1]
        rest_OE = wind_OE[2:]

        # Plot
        for i in range(len(rest)):
            if label_obs[i] in ['south pole']:
                continue
            axB.plot(tfbH, rest[i], c = colors_obs[i], label = label_obs[i])
            axOE.plot(tfbH_OE, rest_OE[i], c = colors_obs[i], label = label_obs[i])
        
        for ax in [axB, axOE]:  
            ax.set_xlabel(r'$t [t_{\rm fb}]$')
            ax.tick_params(axis='both', which='major', width=1.2, length=9)
            ax.tick_params(axis='both', which='minor', width=1, length=5)
            ax.legend(fontsize = 18)
            ax.set_yscale('log')
            ax.grid()

            

