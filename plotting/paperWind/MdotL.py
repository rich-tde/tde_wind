""" Compute the time evolution of Mdot fallback and Mdot wind across a spherical surface"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv
import os
import healpy as hp
import Utilities.prelude as prel
from Utilities.operators import choose_observers, sort_list
from src import orbits as orb

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
choice = 'left_right_z' 
how = '' # '' for sum or 'mean'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz)
indices_obs, label_obs, color_obs, _ = choose_observers(observers_xyz, choice)
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs 

data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfbs, Lums = data[:, 0], data[:, 1], data[:, 2]
tfbs, snaps, Lums = sort_list([tfbs, snaps, Lums], snaps, unique=True)
Lum_sec = []
snaps = np.array(snaps, dtype=int)
for s, snap in enumerate(snaps): 
    photo = np.load(f'{abspath}/data/{folder}/photoPOL/{check}_photo{snap}.npz')
    Lum_ph = photo['Lum']
    Lum_sec.append(np.mean(Lum_ph[indices_obs], axis = 1))
Lum_sec = np.transpose(np.array(Lum_sec))
    
# fallback = \
#         np.loadtxt(f'{abspath}/data/{folder}/paper1/wind/Mdot_{check}05aminmean.csv', 
#                 delimiter = ',', 
#                 skiprows=1, 
#                 unpack=True)
# tfbfb, mfb, mwind_dimCellOld = fallback[1], fallback[2], fallback[3]

wind = \
        np.loadtxt(f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}05amin{choice}_.csv', 
                delimiter = ',', 
                skiprows=1, 
                unpack=True) 
tfbH = wind[1]
rest = wind[2:2+len(label_obs)]

outflow = \
        np.loadtxt(f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}05amin{choice}_outflow.csv', 
                delimiter = ',', 
                skiprows=1, 
                unpack=True) 
tfbO = outflow[1]
restO = outflow[2:2+len(label_obs)]

fig, (axM, axL) =plt.subplots(1,2, figsize = (18,7))    
for i in range(len(rest)):
    if i == 3:
        continue
    axM.plot(tfbH, rest[i]/Medd_sol,  label = label_obs[i], c = color_obs[i])
    axM.plot(tfbO[4:], restO[i][4:]/Medd_sol, c = color_obs[i], ls = '--')
    axL.plot(tfbH, Lum_sec[i],  label = label_obs[i], c = color_obs[i])

original_ticks = axM.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
labels = [str(np.round(tick,2)) if tick in original_ticks else '' for tick in new_ticks]    
for ax in [axM, axL]:
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(labels)
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(labels)  
    ax.set_yscale('log')
    ax.set_xlabel(r'$t (t_{\rm fb})$')
    ax.set_xlim(0, np.max(tfbH))
    ax.tick_params(axis='both', which='major', width=1.2, length=9)
    ax.tick_params(axis='both', which='minor', width=1, length=5)
    ax.grid()
axM.set_ylim(1e1, 7e6)
axM.set_ylabel(r'$\dot{M}_{{\rm w}} (\dot{M}_{\rm Edd})$')  
axL.set_ylim(1e38, 2e42)
axL.set_ylabel(r'$L_{{\rm ph}}$ (erg/s)')  
axM.legend(fontsize = 24)

fig.tight_layout()
fig.savefig(f'{abspath}/Figs/2.paperWind/ML_intime.pdf', dpi = 300)
