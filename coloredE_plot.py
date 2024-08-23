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

##
# PARAMETERS
##
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5 
compton = 'Compton'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'/Users/paolamartire/shocks/data/{folder}/colormapE_Alice'

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

#
## DECISIONS
##
save = False
cutoff = 'cut' # or 'bound or ''

#
## DATA
#

# Low data
dataLow = np.load(f'{path}/{cutoff}TESTboundcoloredE_Low.npy') #shape (3, len(tfb), len(radii))
tfb_dataLow = np.loadtxt(f'{path}/{cutoff}TESTboundcoloredE_Low_days.txt')
tfb_Low = tfb_dataLow[1]
radiiLow = np.load(f'{path}/{cutoff}TESTboundcoloredE_Low_radii.npy')
radiiLowplot = radiiLow
radiiLow /=apo
col_ie, col_orb_en, col_Rad = dataLow[0], dataLow[1], dataLow[2]
# Average over time
# col_ie, col_orb_en, col_Rad = col_ie.T, col_orb_en.T, col_Rad.T
# for i in range(len(col_ie)):
#     col_ie[i] = time_average(tfb_Low, col_ie[i])
#     col_orb_en[i] = time_average(tfb_Low, col_orb_en[i])
#     col_Rad[i] = time_average(tfb_Low, col_Rad[i])
# col_ie, col_orb_en, col_Rad = col_ie.T, col_orb_en.T, col_Rad.T
abs_col_orb_en = np.abs(col_orb_en)

# Middle data
dataMiddle = np.load(f'{path}/{cutoff}TESTboundcoloredE_HiRes.npy')
tfb_dataMiddle = np.loadtxt(f'{path}/{cutoff}TESTboundcoloredE_HiRes_days.txt')
tfb_Middle = tfb_dataMiddle[1]
radiiMiddle = np.load(f'{path}/{cutoff}TESTboundcoloredE_HiRes_radii.npy')
radiiMiddleplot = radiiMiddle
radiiMiddle /=apo
col_ieMiddle, col_orb_enMiddle, col_RadMiddle = dataMiddle[0], dataMiddle[1], dataMiddle[2]
# Average over time
# col_ieMiddle, col_orb_enMiddle, col_RadMiddle = col_ieMiddle.T, col_orb_enMiddle.T, col_RadMiddle.T
# for i in range(len(col_ieMiddle)):  
#     col_ieMiddle[i] = time_average(tfb_Middle, col_ieMiddle[i])
#     col_orb_enMiddle[i] = time_average(tfb_Middle, col_orb_enMiddle[i])
#     col_RadMiddle[i] = time_average(tfb_Middle, col_RadMiddle[i])
# col_ieMiddle, col_orb_enMiddle, col_RadMiddle = col_ieMiddle.T, col_orb_enMiddle.T, col_RadMiddle.T
abs_col_orb_enMiddle = np.abs(col_orb_enMiddle)

# Consider Low data only up to the time of the Middle data
n_Middle = len(col_ieMiddle)
tfb_Low = tfb_Low[:n_Middle]
col_ie = col_ie[:n_Middle]
col_orb_en = col_orb_en[:n_Middle]
abs_col_orb_en = abs_col_orb_en[:n_Middle]
col_Rad = col_Rad[:n_Middle]
#%%
# PLOT
##

fig, ax = plt.subplots(2,3, figsize = (12,6))
# Low
img = ax[0][0].pcolormesh(radiiLow, tfb_Low, col_ie,  norm=colors.LogNorm(vmin=1e-2, vmax=1),
                     cmap = 'cet_rainbow4')
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}$IE/Mass', fontsize = 14, labelpad = 5)
ax[0][0].set_title('Low resolution', fontsize = 14)

img = ax[0][1].pcolormesh(radiiLow, tfb_Low, abs_col_orb_en, norm=colors.LogNorm(vmin=10, vmax=110),
                     cmap = 'cet_rainbow4')
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}|E_{orb}|$/Mass', fontsize = 14, labelpad = 5)
ax[0][1].set_xscale('log')

img = ax[0][2].pcolormesh(radiiLow, tfb_Low, col_Rad*radiiLowplot**2, norm=colors.LogNorm(vmin=1e-13, vmax=3e-11),
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}E_{rad}$/Vol', fontsize = 14, labelpad = 5)
ax[0][2].set_xscale('log')

# Middle
img = ax[1][0].pcolormesh(radiiMiddle, tfb_Middle, col_ieMiddle, norm=colors.LogNorm(vmin=1e-2, vmax=1),
                     cmap = 'cet_rainbow4')
cb = fig.colorbar(img)
cb.set_label('IE/Mass', fontsize = 14, labelpad = 5)
ax[1][0].set_xscale('log')
ax[1][0].set_title('Middle resolution', fontsize = 14)

img = ax[1][1].pcolormesh(radiiMiddle, tfb_Middle, abs_col_orb_enMiddle,  norm=colors.LogNorm(vmin=10, vmax=120),
                     cmap = 'cet_rainbow4')
cb = fig.colorbar(img)
cb.set_label(r'$|E_{orb}|$/Mass', fontsize = 14, labelpad = 5)
ax[1][1].set_xscale('log')

img = ax[1][2].pcolormesh(radiiMiddle, tfb_Middle, col_RadMiddle*radiiMiddleplot**2,  norm=colors.LogNorm(vmin=1e-13, vmax=3e-11),
                     cmap = 'cet_rainbow4')
cb = fig.colorbar(img)
cb.set_label(r'E$_{rad}$/Vol', fontsize = 14, labelpad = 5)
ax[1][2].set_xscale('log')
for i in range(2):
    for j in range(3):
        # ax[i][j].set_ylim(0.3, np.max(tfb_Middle))
        ax[i][j].axvline(Rt/apo, linestyle ='--', c = 'white', linewidth = 0.8)
        ax[i][j].axhline(0.5, c = 'white', linewidth = 0.4)
        ax[i][j].axhline(0.7, c = 'white', linewidth = 0.4)

        # Grid for radii, to be matched with the cfr in slices.py
        ax[i][j].axvline(0.1, c = 'white', linewidth = 0.4)
        ax[i][j].axvline(0.3, c = 'white', linewidth = 0.4)
        ax[i][j].axvline(0.5, c = 'white', linewidth = 0.4)
        ax[i][j].set_xscale('log')

# Layout
ax[0][0].set_ylabel(r't/t$_{fb}$', fontsize = 14)
ax[1][0].set_ylabel(r't/t$_{fb}$', fontsize = 14)
ax[1][0].set_xlabel(r'$R/R_a$', fontsize = 14)
ax[1][1].set_xlabel(r'$R/R_a$', fontsize = 14)
ax[1][2].set_xlabel(r'$R/R_a$', fontsize = 14)
plt.tick_params(axis = 'both', which = 'both', direction='in')
if cutoff == 'bound':
    plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$, bound elements only', fontsize = 18)
else:
    plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
# for i in range(2):
#     for j in range(3):
#         ax[i][j].grid()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{cutoff}coloredE.png')
plt.show()


# %% Plot (absolute) differences. They start from the same point
diff_ie = np.abs(col_ie - col_ieMiddle)
diff_orb_en = np.abs(col_orb_en - col_orb_enMiddle)
diff_Rad = np.abs(col_Rad - col_RadMiddle)

fig, ax = plt.subplots(1,3, figsize = (18,5))
img = ax[0].pcolormesh(radiiMiddle, tfb_Middle, diff_ie,  norm=colors.LogNorm(vmin=1e-3, vmax=1),
                     cmap = 'bwr')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}\Delta|$IE/Mass$|$', fontsize = 14, labelpad = 5)
ax[0].set_xscale('log')

img = ax[1].pcolormesh(radiiMiddle, tfb_Middle, diff_orb_en, norm=colors.LogNorm(vmin=0.1, vmax=100),
                     cmap = 'bwr')#, vmin = 7e-7, vmax = 4e2)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}\Delta|E_{orb}/$Mass$|$', fontsize = 14, labelpad = 5)
ax[1].set_xscale('log')

img = ax[2].pcolormesh(radiiMiddle, tfb_Middle, diff_Rad, norm=colors.LogNorm(vmin=1e-10, vmax=1e-7),
                     cmap = 'bwr')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}\Delta|E_{rad}/$Vol$|$', fontsize = 14, labelpad = 5)
ax[2].set_xscale('log')

for i in range(3):
    ax[i].axvline(Rt/apo, linestyle ='--', c = 'white')
    # ax[i].set_ylim(0.3, np.max(tfb_Middle))

# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 18)
ax[0].set_xlabel(r'$R/R_a$', fontsize = 18)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 18)
ax[2].set_xlabel(r'$R/R_a$', fontsize = 18)
plt.tick_params(axis = 'both', which = 'both', direction='in')
if cutoff == 'bound':
    plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$, bound elements only', fontsize = 18)
else:
    plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{cutoff}coloredE_diff.png')
plt.show()

######################
# %% Relative differences
rel_ie = np.abs(diff_ie / col_ie)
rel_orb_en = np.abs(diff_orb_en / col_orb_en)
rel_Rad = np.abs(diff_Rad / col_Rad)

fig, ax = plt.subplots(1,3, figsize = (18,5))
img = ax[0].pcolormesh(radiiMiddle, tfb_Middle, rel_ie,  
                     cmap = 'bwr', vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'Relative difference $|$IE$|$/Mass', fontsize = 14, labelpad = 5)
ax[0].set_xscale('log')

img = ax[1].pcolormesh(radiiMiddle, tfb_Middle, rel_orb_en,
                     cmap = 'bwr', vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'Relative difference $E_{orb}$/Mass', fontsize = 14, labelpad = 5)
ax[1].set_xscale('log')

img = ax[2].pcolormesh(radiiMiddle, tfb_Middle, rel_Rad,
                     cmap = 'bwr', vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'Relative difference $|E_{rad}$/Vol$|$', fontsize = 14, labelpad = 5)
ax[2].set_xscale('log')

for i in range(3):
    ax[i].axvline(Rt/apo, linestyle ='--', c = 'white')
    # ax[i].set_ylim(0.3, np.max(tfb_Middle))

# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 18)
ax[0].set_xlabel(r'$R/R_a$', fontsize = 18)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 18)
ax[2].set_xlabel(r'$R/R_a$', fontsize = 18)
plt.tick_params(axis = 'both', which = 'both', direction='in')
if cutoff == 'bound':
    plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$, bound elements only', fontsize = 18)
else:
    plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{cutoff}coloredE_relative_diff.png')
plt.show()

# %%
plt.plot(radiiLow, col_Rad[10]*radiiLow**2, c = 'k', label = f't/tfb = {np.round(tfb_Low[10],1)}')
plt.plot(radiiMiddle, col_RadMiddle[10]*radiiLow**2, '--', c = 'k', label = f'Middle t/tfb = {np.round(tfb_Middle[10],1)}')
plt.plot(radiiLow, col_Rad[20]*radiiLow**2, c = 'r', label = f't/tfb = {np.round(tfb_Low[20],1)}')
plt.plot(radiiMiddle, col_RadMiddle[20]*radiiMiddle**2, '--', c = 'r', label = f'Middle t/tfb = {np.round(tfb_Middle[20],1)}')
plt.plot(radiiLow, col_Rad[27]*radiiMiddle**2, c = 'b', label = f't/tfb = {np.round(tfb_Low[27],1)}')
plt.plot(radiiMiddle, col_RadMiddle[27]*radiiMiddle**2, '--', c = 'b', label = f'Middle t/tfb = {np.round(tfb_Middle[27],1)}')
plt.loglog()
plt.legend()
plt.ylabel('Rad/Vol', fontsize = 18)
plt.show()