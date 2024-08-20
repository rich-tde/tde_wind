#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:50:40 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import colorcet
import matplotlib.colors as colors
import Utilities.prelude
from Utilities.gaussian_smoothing import time_average

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5 
compton = 'Compton'
save = False

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'/Users/paolamartire/shocks/data/{folder}'

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

# Low data
tfb_dataLow = np.loadtxt(f'{path}/coloredE_Low_days.txt')
tfb_Low = tfb_dataLow[1]
dataLow = np.load(f'{path}/coloredE_Low.npy') #shape (3, len(tfb), len(radii))
radiiLow = np.load(f'{path}/coloredE_Low_radii.npy')
radiiLow /=apo
col_ie, col_orb_en, col_Rad = dataLow[0], dataLow[1], dataLow[2]
col_ie, col_orb_en, col_Rad = col_ie.T, col_orb_en.T, col_Rad.T
for i in range(len(col_ie)):
    col_ie[i] = time_average(tfb_Low, col_ie[i])
    col_orb_en[i] = time_average(tfb_Low, col_orb_en[i])
    col_Rad[i] = time_average(tfb_Low, col_Rad[i])
col_ie, col_orb_en, col_Rad = col_ie.T, col_orb_en.T, col_Rad.T
abs_col_orb_en = np.abs(col_orb_en)

# Middle data
tfb_dataMiddle = np.loadtxt(f'{path}/coloredE_HiRes_days.txt')
tfb_Middle = tfb_dataMiddle[1]
dataMiddle = np.load(f'{path}/coloredE_HiRes.npy')
radiiMiddle = np.load(f'{path}/coloredE_HiRes_radii.npy')
radiiMiddle /=apo
col_ieMiddle, col_orb_enMiddle, col_RadMiddle = dataMiddle[0], dataMiddle[1], dataMiddle[2]
col_ieMiddle, col_orb_enMiddle, col_RadMiddle = col_ieMiddle.T, col_orb_enMiddle.T, col_RadMiddle.T
for i in range(len(col_ieMiddle)):  
    col_ieMiddle[i] = time_average(tfb_Middle, col_ieMiddle[i])
    col_orb_enMiddle[i] = time_average(tfb_Middle, col_orb_enMiddle[i])
    col_RadMiddle[i] = time_average(tfb_Middle, col_RadMiddle[i])
col_ieMiddle, col_orb_enMiddle, col_RadMiddle = col_ieMiddle.T, col_orb_enMiddle.T, col_RadMiddle.T
abs_col_orb_enMiddle = np.abs(col_orb_enMiddle)

# Consider Low data only up to the time of the Middle data
n_Middle = len(col_ieMiddle)
tfb_Low = tfb_Low[:n_Middle]
col_ie = col_ie[:n_Middle]
col_orb_en = col_orb_en[:n_Middle]
abs_col_orb_en = abs_col_orb_en[:n_Middle]
col_Rad = col_Rad[:n_Middle]
#%%
fig, ax = plt.subplots(2,3, figsize = (12,6))
# Low
img = ax[0][0].pcolormesh(radiiLow, tfb_Low, col_ie,  norm=colors.LogNorm(vmin=col_ie.min(), vmax=col_ie.max()),
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}$IE/Mass', fontsize = 14, labelpad = 5)
ax[0][0].set_xscale('log')

img = ax[0][1].pcolormesh(radiiLow, tfb_Low, abs_col_orb_en, norm=colors.LogNorm(vmin=abs_col_orb_en.min(), vmax=abs_col_orb_en.max()),
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}|E_{orb}|$/Mass', fontsize = 14, labelpad = 5)
ax[0][1].set_xscale('log')
ax[0][1].set_title('Low resolution', fontsize = 14)

img = ax[0][2].pcolormesh(radiiLow, tfb_Low, col_Rad, norm=colors.LogNorm(vmin=col_Rad.min(), vmax=col_Rad.max()),
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}E_{rad}$/Vol', fontsize = 14, labelpad = 5)
ax[0][2].set_xscale('log')

# Middle
img = ax[1][0].pcolormesh(radiiMiddle, tfb_Middle, col_ieMiddle, norm=colors.LogNorm(vmin=col_ieMiddle.min(), vmax=col_ieMiddle.max()),
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label('IE/Mass', fontsize = 14, labelpad = 5)
ax[1][0].set_xscale('log')

img = ax[1][1].pcolormesh(radiiMiddle, tfb_Middle, abs_col_orb_enMiddle,  norm=colors.LogNorm(vmin=abs_col_orb_enMiddle.min(), vmax=abs_col_orb_enMiddle.max()),
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$|E_{orb}|$/Mass', fontsize = 14, labelpad = 5)
ax[1][1].set_xscale('log')
ax[1][1].set_title('Middle resolution', fontsize = 14)

img = ax[1][2].pcolormesh(radiiMiddle, tfb_Middle, col_RadMiddle,  norm=colors.LogNorm(vmin=col_RadMiddle.min(), vmax=col_RadMiddle.max()),
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'E$_{rad}$/Vol', fontsize = 14, labelpad = 5)
ax[1][2].set_xscale('log')
for i in range(2):
    for j in range(3):
        ax[i][j].axvline(Rt/apo, linestyle ='--', c = 'white')

# Layout
ax[0][0].set_ylabel(r't/t$_{fb}$', fontsize = 14)
ax[1][0].set_ylabel(r't/t$_{fb}$', fontsize = 14)
ax[1][0].set_xlabel(r'$R/R_a$', fontsize = 14)
ax[1][1].set_xlabel(r'$R/R_a$', fontsize = 14)
ax[1][2].set_xlabel(r'$R/R_a$', fontsize = 14)
plt.tick_params(axis = 'both', which = 'both', direction='in')
plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/coloredE_timeavg.png')
plt.show()


# %% Plot (absolute) differences. They start from the same point
diff_ie = np.abs(col_ie - col_ieMiddle)
diff_orb_en = np.abs(col_orb_en - col_orb_enMiddle)
diff_Rad = np.abs(col_Rad - col_RadMiddle)

fig, ax = plt.subplots(1,3, figsize = (18,5))
img = ax[0].pcolormesh(radiiMiddle, tfb_Middle, diff_ie,  norm=colors.LogNorm(vmin=diff_ie.min(), vmax=diff_ie.max()),
                     cmap = 'bwr')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}\Delta|$IE/Mass$|$', fontsize = 14, labelpad = 5)
ax[0].set_xscale('log')

img = ax[1].pcolormesh(radiiMiddle, tfb_Middle, diff_orb_en, norm=colors.LogNorm(vmin=diff_orb_en.min(), vmax=diff_orb_en.max()),
                     cmap = 'bwr')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}\Delta|E_{orb}/$Mass$|$', fontsize = 14, labelpad = 5)
ax[1].set_xscale('log')

img = ax[2].pcolormesh(radiiMiddle, tfb_Middle, diff_Rad, norm=colors.LogNorm(vmin=diff_Rad.min(), vmax=diff_Rad.max()),
                     cmap = 'bwr')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}\Delta|E_{rad}/$Vol$|$', fontsize = 14, labelpad = 5)
ax[2].set_xscale('log')

for i in range(3):
    ax[i].axvline(Rt/apo, linestyle ='--', c = 'white')

# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 18)
ax[0].set_xlabel(r'$R/R_a$', fontsize = 18)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 18)
ax[2].set_xlabel(r'$R/R_a$', fontsize = 18)
plt.tick_params(axis = 'both', which = 'both', direction='in')
plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/coloredE_diff_timeavg.png')
plt.show()

######################
# %% Relative differences
rel_ie = np.abs(diff_ie / col_ieMiddle)
rel_orb_en = np.abs(diff_orb_en / col_orb_enMiddle)
rel_Rad = np.abs(diff_Rad / col_RadMiddle)

fig, ax = plt.subplots(1,3, figsize = (18,5))
img = ax[0].pcolormesh(radiiMiddle, tfb_Middle, rel_ie,  
                     cmap = 'bwr', vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'Relative difference $|$IE$|$', fontsize = 14, labelpad = 5)
ax[0].set_xscale('log')

img = ax[1].pcolormesh(radiiMiddle, tfb_Middle, rel_orb_en,
                     cmap = 'bwr', vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'Relative difference $E_{orb}$', fontsize = 14, labelpad = 5)
ax[1].set_xscale('log')

img = ax[2].pcolormesh(radiiMiddle, tfb_Middle, rel_Rad,
                     cmap = 'bwr', vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'Relative difference $|$E_{rad}$|$', fontsize = 14, labelpad = 5)
ax[2].set_xscale('log')

for i in range(3):
    ax[i].axvline(Rt/apo, linestyle ='--', c = 'white')

# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 18)
ax[0].set_xlabel(r'$R/R_a$', fontsize = 18)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 18)
ax[2].set_xlabel(r'$R/R_a$', fontsize = 18)
plt.tick_params(axis = 'both', which = 'both', direction='in')
plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/coloredE_relative_diff_timeavg.png')
plt.show()
