""" Compute the time evolution of Mdot fallback and Mdot wind across a spherical surface"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv
import os
from matplotlib import lines as mlines
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
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
t_fb_days = things['t_fb_days']
Rt = things['Rt']

data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfb_Lum, Lums = data[:, 0], data[:, 1], data[:, 2]
tfb_Lum, snaps, Lums = sort_list([tfb_Lum, snaps, Lums], snaps, unique=True)
Lum_sec = []
rph_sec = []
rtr_sec = []
snaps = np.array(snaps, dtype=int)
for s, snap in enumerate(snaps): 
    photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
    x_ph, y_ph, z_ph, Lum_ph = photo['x'], photo['y'], photo['z'], photo['Lum']
    r_ph = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
    Lum_sec.append(np.mean(Lum_ph[indices_obs], axis = 1))
    trap = np.load(f'{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz')
    x_tr, y_tr, z_tr = trap['x_tr'], trap['y_tr'], trap['z_tr']
    r_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
    mask = r_tr[indices_obs] > 0
    indices_sec = [row[m] for row, m in zip(indices_obs, mask)]
    rtr_sec.append([np.median(r_tr[row]) for row in indices_sec])
    rph_sec.append([np.median(r_ph[row]) for row in indices_sec])
    # rtr_sec.append(np.mean(r_tr[indices_obs], axis = 1))
    # rph_sec.append(np.mean(r_ph[indices_obs], axis = 1))
Lum_sec = np.transpose(np.array(Lum_sec))
rph_sec = np.transpose(np.array(rph_sec))
rtr_sec = np.transpose(np.array(rtr_sec))
    
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
tfb = wind[1]
rest = wind[2:2+len(label_obs)]

outflow = \
        np.loadtxt(f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}05amin{choice}_outflow.csv', 
                delimiter = ',', 
                skiprows=1, 
                unpack=True) 
tfbO = outflow[1]
restO = outflow[2:2+len(label_obs)]

fig, (axM, axL) =plt.subplots(1,2, figsize = (16,7))  
figr, axr  = plt.subplots(1,1, figsize = (9,7))
handles_color = []
labels_color = []
line_styles_parts = ['-', '--']
labels_parts = [r'$\dot{M}_{\rm w}$', r'$\dot{M}_{\rm out}$']
for i in range(len(rest)):
    if i == 3:
        continue
    line = axM.plot(tfb, rest[i]/Medd_sol,  label = label_obs[i], linewidth = 2, c = color_obs[i], ls = line_styles_parts[0])[0]
    axM.plot(tfbO[4:], restO[i][4:]/Medd_sol, linewidth = 2, c = color_obs[i], ls = line_styles_parts[1])
    axr.plot(tfb, rph_sec[i]/Rt, linewidth = 2, c = color_obs[i], label = r'r$_{\rm ph}$' if i == 2 else "")
    axr.plot(tfb, rtr_sec[i]/Rt, linewidth = 2, c = color_obs[i], ls = ':', label = r'r$_{\rm trap}$' if i == 2 else "")
    axL.plot(tfb, Lum_sec[i],  label = label_obs[i], linewidth = 2, c = color_obs[i])
    if i ==0:
        print('ratio wind/outflow at last time step:', rest[i][-1]/restO[i][-1])
    
    handles_color.append(line)
    labels_color.append(label_obs[i])

original_ticks = axM.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
labels = [str(np.round(tick,2)) if tick in original_ticks else '' for tick in new_ticks]   
days_ticks = new_ticks*t_fb_days
days_labels = [str(np.round(days_ticks[k],2)) if new_ticks[k] in original_ticks else "" for k in range(len(days_ticks))] 
for ax in [axM, axr, axL]:
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(labels)
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(labels)  
    ax.set_yscale('log')
    ax.set_xlabel(r'$t / t_{\rm fb}$')
    ax.set_xlim(0, np.max(tfb))
    ax.tick_params(axis='both', which='major', width=1.2, length=9)
    ax.tick_params(axis='both', which='minor', width=1, length=5)
    ax.grid()
    ax2 = ax.twiny()
    ax2.set_xticks(days_ticks)
    ax2.set_xlim(-0.05*t_fb_days, np.max(tfb)*t_fb_days)
    ax2.set_xticklabels(days_labels)
    ax2.set_xlabel(r't (days)', fontsize = 30)
axM.set_ylim(1e1, 7e6)
axM.set_ylabel(r'$\dot{M} (\dot{M}_{\rm Edd})$')  
axr.set_ylabel(r'$r (r_{\rm t})$')
axr.set_ylim(1, 1.2e2)
axL.set_ylim(1e38, 2e42)
axL.set_ylabel(r'$L_{{\rm FLD}}$ (erg/s)')  
# axM.legend(fontsize = 24)

# Legend 1: colored observer lines (three colors)
legend1 = axM.legend(handles=handles_color,
                    labels=labels_color,
                    fontsize=21,
                    loc='upper left')
axM.add_artist(legend1)

# Legend 2: line-style explanation (solid vs dashed)
proxy_lines = []
proxy_lines = []
for l, line in enumerate(line_styles_parts):
    proxy_lines.append(
        mlines.Line2D([0], [0], color='cornflowerblue', ls=line, linewidth=2,
                    label=labels_parts[l])
    )
axM.legend(handles=proxy_lines, fontsize=20, 
                                loc='lower right')
axr.legend(fontsize = 20, loc = 'lower right')
fig.tight_layout()
fig.savefig(f'{abspath}/Figs/2.paperWind/ML_intime.pdf', dpi = 300)

