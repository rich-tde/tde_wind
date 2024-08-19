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

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5 

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
path = f'/Users/paolamartire/shocks/data/{folder}'
tfb_Low = np.loadtxt(f'{path}/coloredE_days.txt')[1]
dataLow = np.load(f'{path}/coloredE_.npy')
col_ie, col_orb_en, col_Rad = dataLow[0], dataLow[1], dataLow[2]
abs_col_orb_en = np.abs(col_orb_en)

# tfb_Middle = np.loadtxt(f'{path}/coloredE_HiRes.txt')[1]
# dataMiddle = np.load(f'{path}/coloredE_HiRes.npy')
# col_ieMiddle, col_orb_enMiddle, col_RadMiddle = dataMiddle[0], dataMiddle[1], dataMiddle[2]
# abs_col_orb_en_Middle = np.abs(col_orb_en_Middle)

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

radii = np.logspace(np.log10(R0), np.log10(apo),
                        num=100)
radii /=apo

# diff = np.abs(dataLow - dataMiddle)
#%%
fig, ax = plt.subplots(2,3, figsize = (12,6))
# Low
img = ax[0][0].pcolormesh(radii, tfb_Low, col_ie,  norm=colors.LogNorm(vmin=col_ie.min(), vmax=col_ie.max()),
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}$IE/Mass', fontsize = 14, labelpad = 5)
ax[0][0].set_xscale('log')

img = ax[0][1].pcolormesh(radii, tfb_Low, abs_col_orb_en, norm=colors.LogNorm(vmin=abs_col_orb_en.min(), vmax=abs_col_orb_en.max()),
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}|E_{orb}|$/Mass', fontsize = 14, labelpad = 5)
ax[0][1].set_xscale('log')

img = ax[0][2].pcolormesh(radii, tfb_Low, col_Rad, norm=colors.LogNorm(vmin=col_Rad.min(), vmax=col_Rad.max()),
                     cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
cb = fig.colorbar(img)
cb.set_label(r'$\log_{10}E_{rad}$/Vol', fontsize = 14, labelpad = 5)
ax[0][2].set_xscale('log')

# Middle
# img = ax[1][0].pcolormesh(radii, tfb_Middle, col_ieMiddle,
#                      cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
# cb = fig.colorbar(img)
# cb.set_label('IE/Mass', fontsize = 14, labelpad = 5)
# ax[1][0].set_xscale('log')

# img = ax[1][1].pcolormesh(radii, tfb_Middle, col_orb_enMiddle,
#                      cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
# cb = fig.colorbar(img)
# cb.set_label(r'E$_{orb}$/Mass', fontsize = 14, labelpad = 5)
# ax[1][1].set_xscale('log')

# img = ax[1][2].pcolormesh(radii, tfb_Middle, col_RadMiddle,
#                      cmap = 'cet_rainbow4')#, vmin = 0, vmax = 1)
# cb = fig.colorbar(img)
# cb.set_label(r'E$_{rad}$/Vol', fontsize = 14, labelpad = 5)
# ax[1][2].set_xscale('log')
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
plt.suptitle(r'$M_{BH}=10^4$ M$_\odot$', fontsize = 18)
plt.tight_layout()



# %%
