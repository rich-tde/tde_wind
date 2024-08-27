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
import Utilities.prelude as prel
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

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

#
## DECISIONS
##
save = True
cutoff = 'cutden' # or 'bound or ''

#
## DATA
#

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'/Users/paolamartire/shocks/data/{folder}/colormapE_Alice'
# Low data
dataLow = np.load(f'{path}/{cutoff}coloredE_Low.npy') #shape (3, len(tfb), len(radii))
tfb_dataLow = np.loadtxt(f'{path}/{cutoff}coloredE_Low_days.txt')
tfb_Low = tfb_dataLow[1]
radiiLow = np.load(f'{path}/{cutoff}coloredE_Low_radii.npy')
col_ie, col_orb_en, col_Rad = dataLow[0], dataLow[1], dataLow[2]
# convert to cgs
col_ie *= prel.en_converter/prel.Msol_to_g
col_orb_en *= prel.en_converter/prel.Msol_to_g
col_Rad *= prel.en_den_converter
# Average over time
# col_ie, col_orb_en, col_Rad = col_ie.T, col_orb_en.T, col_Rad.T
# for i in range(len(col_ie)):
#     col_ie[i] = time_average(tfb_Low, col_ie[i])
#     col_orb_en[i] = time_average(tfb_Low, col_orb_en[i])
#     col_Rad[i] = time_average(tfb_Low, col_Rad[i])
# col_ie, col_orb_en, col_Rad = col_ie.T, col_orb_en.T, col_Rad.T
abs_col_orb_en = np.abs(col_orb_en)

# Middle data
dataMiddle = np.load(f'{path}/{cutoff}coloredE_HiRes.npy')
tfb_dataMiddle = np.loadtxt(f'{path}/{cutoff}coloredE_HiRes_days.txt')
tfb_Middle = tfb_dataMiddle[1]
radiiMiddle = np.load(f'{path}/{cutoff}coloredE_HiRes_radii.npy')
col_ieMiddle, col_orb_enMiddle, col_RadMiddle = dataMiddle[0], dataMiddle[1], dataMiddle[2]
# convert to cgs
col_ieMiddle *= prel.en_converter/prel.Msol_to_g
col_orb_enMiddle *= prel.en_converter/prel.Msol_to_g
col_RadMiddle *= prel.en_den_converter
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
# col_ie[col_ie<=0] = 1 #nothing, so in logscale they will be zero
# abs_col_orb_en[abs_col_orb_en<=0] = 1
# col_Rad[col_Rad<=0] = 1

p40_iesix = np.percentile(col_ie, 40)
p99_iesix = np.percentile(col_ie, 99)
p40_orb_ensix = np.percentile(abs_col_orb_en, 40)
p99_orb_ensix = np.percentile(abs_col_orb_en, 99)
p40_Radsix = np.percentile(col_Rad, 40)
p99_Radsix = np.percentile(col_Rad, 99)

cmap = plt.cm.viridis
norm_orb_ensix = colors.LogNorm(vmin=p40_orb_ensix, vmax=p99_orb_ensix)
norm_iesix = colors.LogNorm(vmin=p40_iesix, vmax=p99_iesix)
norm_Radsix = colors.LogNorm(vmin=p40_Radsix, vmax=p99_Radsix)

fig, ax = plt.subplots(2,3, figsize = (12,6))
# Low
img = ax[0][0].pcolormesh(radiiLow/apo, tfb_Low, abs_col_orb_en, norm=norm_orb_ensix, cmap = cmap)
cb = fig.colorbar(img)
ax[0][0].set_title('Specific (absolute) orbital energy', fontsize = 14)
# cb.set_label(r'$|E_{orb}|$/Mass [erg/g]', fontsize = 14, labelpad = 5)
ax[0][0].set_xscale('log')
ax[0][0].text(0.05, 0.15,'Low res', fontsize = 14)

img = ax[0][1].pcolormesh(radiiLow/apo, tfb_Low, col_ie,  norm=norm_iesix, cmap = cmap)
cb = fig.colorbar(img)
# cb.set_label(r'IE/Mass [erg/g]', fontsize = 14, labelpad = 5)
ax[0][1].set_title('Specific internal energy', fontsize = 14)
ax[0][1].set_xscale('log')

img = ax[0][2].pcolormesh(radiiLow/apo, tfb_Low, col_Rad, norm=norm_Radsix, cmap = cmap)
cb = fig.colorbar(img)
ax[0][2].set_title('Radiation energy density', fontsize = 14)
# cb.set_label(r'$\log_{10}E_{rad}$/Vol [erg/cm$^3$]', fontsize = 14, labelpad = 5)
ax[0][2].set_xscale('log')

# Middle
img = ax[1][0].pcolormesh(radiiMiddle/apo, tfb_Middle, abs_col_orb_enMiddle, norm=norm_orb_ensix, cmap = cmap)
cb = fig.colorbar(img)
# cb.set_label(r'$|E_{orb}|$/Mass [erg/g]', fontsize = 14, labelpad = 5)
ax[1][0].set_xscale('log')
ax[1][0].text(0.05, 0.15,'High res', fontsize = 14)

img = ax[1][1].pcolormesh(radiiMiddle/apo, tfb_Middle, col_ieMiddle, norm=norm_iesix, cmap = cmap)
cb = fig.colorbar(img)
# cb.set_label('IE/Mass [erg/g]', fontsize = 14, labelpad = 5)
ax[1][1].set_xscale('log')

img = ax[1][2].pcolormesh(radiiMiddle/apo, tfb_Middle, col_RadMiddle, norm=norm_Radsix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'E$_{rad}$/Vol [erg/cm$^3$]', fontsize = 14, labelpad = 5)
ax[1][2].set_xscale('log')
for i in range(2):
    for j in range(3):
        ax[i][j].axvline(Rt/apo, linestyle ='--', c = 'white', linewidth = 0.8)
        ax[i][j].text(Rt/apo+0.07, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
        ax[i][j].axhline(0.205, c = 'white', linewidth = 0.4)
        ax[i][j].axhline(0.52, c = 'white', linewidth = 0.4)
        # ax[i][j].axhline(0.7, c = 'white', linewidth = 0.4)

        # Grid for radii, to be matched with the cfr in slices.py
        ax[i][j].axvline(0.1, c = 'white', linewidth = 0.4)
        ax[i][j].axvline(0.3, c = 'white', linewidth = 0.4)
        ax[i][j].axvline(0.5, c = 'white', linewidth = 0.4)
        ax[i][j].set_xscale('log')

# Layout
ax[0][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
ax[1][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
ax[1][0].set_xlabel(r'$R/R_a$', fontsize = 20)
ax[1][1].set_xlabel(r'$R/R_a$', fontsize = 20)
ax[1][2].set_xlabel(r'$R/R_a$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', direction='in')
if cutoff == 'bound':
    plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$, bound elements only', fontsize = 18)
else:
    plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{cutoff}coloredE.png')
plt.show()

# %% Plot (absolute) differences. They start from the same point
diff_ie = np.abs(col_ie - col_ieMiddle)
diff_orb_en = np.abs(col_orb_en - col_orb_enMiddle)
diff_Rad = np.abs(col_Rad - col_RadMiddle)
# set to zero where the value of one of them is zero
diff_ie[col_ie==0 ] = 0
diff_ie[col_ieMiddle==0] = 0
diff_orb_en[col_orb_en==0] = 0
diff_orb_en[col_orb_enMiddle==0] = 0
diff_Rad[col_Rad==0] = 0
diff_Rad[col_RadMiddle==0] = 0
# pass to log
# difflog_ie = np.log10(diff_ie)
# difflog_orb_en = np.log10(diff_orb_en)
# difflog_Rad = np.log10(diff_Rad)
# difflog_ie[np.isnan(difflog_ie)] = 0
# difflog_orb_en[np.isnan(difflog_orb_en)] = 0
# difflog_Rad[np.isnan(difflog_Rad)] = 0

fig, ax = plt.subplots(1,3, figsize = (18,5))

img = ax[0].pcolormesh(radiiMiddle/apo, tfb_Middle, diff_orb_en, norm=colors.LogNorm(vmin=1e15, vmax=1e17),
                     cmap = 'bwr')#, vmin = 15, vmax = 17)
cb = fig.colorbar(img)
cb.set_label(r'$\Delta$', fontsize = 20)#'$\Delta|E_{orb}/$Mass$|$', fontsize = 14, labelpad = 5)
ax[0].set_title('Specific (absolute) orbital energy', fontsize = 14)
ax[0].set_xscale('log')

img = ax[1].pcolormesh(radiiMiddle/apo, tfb_Middle, diff_ie, norm=colors.LogNorm(vmin=4e11, vmax=4e13),
                     cmap = 'bwr')#, vmin = 11.5, vmax = 14)
cb = fig.colorbar(img)
cb.set_label(r'$\Delta$', fontsize = 20)#'$\Delta|$IE/Mass$|$', fontsize = 14, labelpad = 5)
ax[1].set_title('Specific internal energy', fontsize = 14)
ax[1].set_xscale('log')

img = ax[2].pcolormesh(radiiMiddle/apo, tfb_Middle, diff_Rad, norm=colors.LogNorm(vmin=1e4, vmax=4e7),
                     cmap = 'bwr')#, vmin = 3, vmax = 8.5)
cb = fig.colorbar(img)
ax[2].set_title('Radiation energy density', fontsize = 14)
cb.set_label(r'$\Delta$', fontsize = 20)#$\Delta|E_{rad}/$Vol$|$', fontsize = 14, labelpad = 5)
ax[2].set_xscale('log')

for i in range(3):
    ax[i].axvline(Rt/apo, linestyle ='--', c = 'white')
    ax[i].text(Rt/apo+0.07, 0.65, r'R$_t$', fontsize = 16, rotation = 90, transform = ax[i].transAxes, color = 'k')

# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 25)
ax[0].set_xlabel(r'$R/R_a$', fontsize = 20)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 20)
ax[2].set_xlabel(r'$R/R_a$', fontsize = 20)
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

rel_orb_en[np.isnan(rel_orb_en)] = 0 #nothing, so in logscale they will be zero
rel_ie[np.isnan(rel_ie)] = 0
rel_Rad[np.isnan(rel_Rad)] = 0

rel_orb_en[rel_orb_en<=0] = 1 #you need it for the Log
rel_ie[rel_ie<=0] =1
rel_Rad[rel_Rad<=0] = 1

median_ie = np.median(rel_ie)
median_orb_en = np.median(rel_orb_en)
median_Rad = np.median(rel_Rad)
print(f'Median relative difference in Orbital energy: {median_orb_en}')
print(f'Median relative difference in IE: {median_ie}')
print(f'Median relative difference in Radiation energy density: {median_Rad}')
p5_ie = np.percentile(rel_ie, 5)
p5_orb_en = np.percentile(rel_orb_en, 5)
p5_Rad = np.percentile(rel_Rad, 5)
p95_ie = np.percentile(rel_ie, 95)
p95_orb_en = np.percentile(rel_orb_en, 95)
p95_Rad = np.percentile(rel_Rad, 95)


cmap = plt.cm.inferno
norm_orb_en = colors.LogNorm(vmin=p5_orb_en, vmax=p95_orb_en)
norm_ie = colors.LogNorm(vmin=p5_ie, vmax=p95_ie)
norm_Rad = colors.LogNorm(vmin=p5_Rad, vmax=p95_Rad)

fig, ax = plt.subplots(1,3, figsize = (18,5))
img = ax[0].pcolormesh(radiiMiddle/apo, tfb_Middle, rel_orb_en, cmap=cmap, norm=norm_orb_en)#, vmin = np.min(rel_orb_en), vmax = 0.2)
cb = fig.colorbar(img)
ax[0].set_title('Specific (absolute) orbital energy', fontsize = 14)
cb.set_label(r'$\Delta_{rel}$', fontsize = 20)#Relative difference $|E_{orb}|$/Mass', fontsize = 14, labelpad = 5)
ax[0].set_xscale('log')

img = ax[1].pcolormesh(radiiMiddle/apo, tfb_Middle, rel_ie, cmap=cmap, norm=norm_ie)#, vmin = np.min(rel_ie), vmax = np.max(rel_ie))
cb = fig.colorbar(img)
ax[1].set_title('Specific internal energy', fontsize = 14)
cb.set_label(r'$\Delta_{rel}$', fontsize = 20)#Relative difference $|$IE$|$/Mass', fontsize = 14, labelpad = 5)
ax[1].set_xscale('log')

img = ax[2].pcolormesh(radiiMiddle/apo, tfb_Middle, rel_Rad, cmap=cmap, norm=norm_Rad)#, vmin = np.min(rel_Rad), vmax = np.max(rel_Rad))
cb = fig.colorbar(img)
ax[2].set_title('Radiation energy density', fontsize = 14)
cb.set_label(r'$\Delta_{rel}$', fontsize = 20)#'Relative difference $|E_{rad}|$/Vol', fontsize = 14, labelpad = 5)
ax[2].set_xscale('log')

for i in range(3):
    ax[i].axvline(Rt/apo, linestyle ='--', c = 'k')
    ax[i].text(Rt/apo+0.07, 0.65, r'R$_t$', fontsize = 16, rotation = 90, transform = ax[i].transAxes, color = 'k')

# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 25)
ax[0].set_xlabel(r'$R/R_a$', fontsize = 20)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 20)
ax[2].set_xlabel(r'$R/R_a$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', direction='in')
if cutoff == 'bound':
    plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$, bound elements only', fontsize = 18)
else:
    plt.suptitle(r'$M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{cutoff}coloredE_relative_diff.png')
plt.show()

# %% Lines
indices = [70, 100, 136]
colors_indices = ['navy', 'royalblue', 'deepskyblue']
lines_difference = (col_Rad-col_RadMiddle)/col_RadMiddle

img, ax = plt.subplots(1,2, figsize = (16,6))
for i,idx in enumerate(indices):
    ax[0].plot(radiiLow, col_Rad[idx], c = colors_indices[i], label = f'Low t/tfb = {np.round(tfb_Low[idx],2)}')
    ax[0].plot(radiiMiddle, col_RadMiddle[idx], '--', c = colors_indices[i], label = f'Middle t/tfb = {np.round(tfb_Middle[idx],2)}')
    ax[1].plot(radiiLow, lines_difference[idx], c = colors_indices[i], label = f't/tfb = {np.round(tfb_Low[idx],2)}')
ax[0].set_xlabel(r'R/R$_a$', fontsize = 20)
ax[0].set_ylabel(r'(Rad/Vol) [erg/cm$^3$]', fontsize = 20)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 20)
ax[1].set_ylabel(r'Low/Middle - 1', fontsize = 20)
ax[0].loglog()
ax[1].loglog()
ax[0].legend(fontsize = 16)
ax[1].legend(fontsize = 18)
ax[0].grid()
ax[1].grid()
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/Rad_lines.png')
plt.show()
# %%
Lum_cgs = col_Rad  * prel.c * 4 * np.pi * (radiiLow*prel.Rsol_to_cm)**2 
LumMiddle_cgs = col_RadMiddle * prel.c * 4 * np.pi * (radiiMiddle*prel.Rsol_to_cm)**2 
Lum_difference = (Lum_cgs-LumMiddle_cgs)/Lum_cgs
img, ax = plt.subplots(1,2, figsize = (16,6))
for i,idx in enumerate(indices):
    if i == 0:
        ax[0].plot(radiiLow, Lum_cgs[idx], c = colors_indices[i], label = f'Low res')#t/tfb = {np.round(tfb_Low[idx],2)}')
        ax[0].plot(radiiMiddle, LumMiddle_cgs[idx], '--', c = colors_indices[i], label = f'Middle res')#t/tfb = {np.round(tfb_Middle[idx],2)}')
        ax[1].plot(radiiLow, Lum_difference[idx], c = colors_indices[i])#, label = f't/tfb = {np.round(tfb_Low[idx],2)}')
    else:   
        ax[0].plot(radiiLow, Lum_cgs[idx], c = colors_indices[i])#, label = f'Low t/tfb = {np.round(tfb_Low[idx],2)}')
        ax[0].plot(radiiMiddle, LumMiddle_cgs[idx], '--', c = colors_indices[i])#, label = f'Middle t/tfb = {np.round(tfb_Middle[idx],2)}')
        ax[1].plot(radiiLow, Lum_difference[idx], c = colors_indices[i])#, label = f't/tfb = {np.round(tfb_Low[idx],2)}')
ax[0].set_xlabel(r'R/R$_a$', fontsize = 20)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 20)
ax[0].set_ylabel(r'Luminosity [erg/s]', fontsize = 20)
ax[1].set_ylabel(r'1-Middle/Low', fontsize = 20)
ax[0].loglog()
ax[0].legend(fontsize = 16)
ax[0].grid()
ax[1].grid()
ax[1].loglog()
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/Luminosity.png')
plt.show()
# %%
