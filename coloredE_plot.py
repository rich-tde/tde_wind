#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:50:40 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import colorcet
import Utilities.prelude

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5 

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
path = f'/Users/paolamartire/shocks/data/{folder}'
tfb = np.loadtxt(f'{path}/coloredE_days.txt')
dataLow = np.load(f'{path}/coloredE_HiRes.npy')
dataMiddle = np.load(f'{path}/coloredE_.npy')
col_ie, col_orb_en, col_Rad = dataLow[0], dataLow[1], dataLow[2]
col_ie2, col_orb_en2, col_Rad2 = dataMiddle[0], dataMiddle[1], dataMiddle[2]

Rt = Rstar * (Mbh/mstar)**(1/3)
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

radii_start = np.log10(0.6*Rt)
radii_stop = np.log10(apo) # apocenter
radii = np.logspace(radii_start, radii_stop, 100) / apo

diff = np.abs(dataLow - dataMiddle)
#%%
fig, ax = plt.subplots(2,3, figsize = (12,6))
# Low
img = ax[0][0].pcolormesh(radii, tfb, col_ie,
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label('IE/Mass', fontsize = 14, labelpad = 5)
ax[0][0].set_xscale('log')

img = ax[0][1].pcolormesh(radii, tfb, col_orb_en,
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'E$_{orb}$/Mass', fontsize = 14, labelpad = 5)
ax[0][1].set_xscale('log')

img = ax[0][2].pcolormesh(radii, tfb, col_Rad,
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'E$_{rad}$/Vol', fontsize = 14, labelpad = 5)
ax[0][2].set_xscale('log')

# Middle
img = ax[1][0].pcolormesh(radii, tfb, col_ie2,
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label('IE/Mass', fontsize = 14, labelpad = 5)
ax[1][0].set_xscale('log')

img = ax[1][1].pcolormesh(radii, tfb, col_orb_en2,
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'E$_{orb}$/Mass', fontsize = 14, labelpad = 5)
ax[1][1].set_xscale('log')

img = ax[1][2].pcolormesh(radii, tfb, col_Rad2,
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'E$_{rad}$/Vol', fontsize = 14, labelpad = 5)
ax[1][2].set_xscale('log')
for i in range(2):
    for j in range(3):
        ax[i][j].axvline(Rt/apo, c = 'white')

# Layout
ax[0][0].set_ylabel(r't/t$_{fb}$', fontsize = 14)
ax[1][0].set_ylabel(r't/t$_{fb}$', fontsize = 14)
ax.tick_params(axis = 'both', which = 'both', direction='in')
ax.set_title(r'$10^4$ M$_\odot$')


