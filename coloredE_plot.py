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
save = False
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

fig, ax = plt.subplots(2,3, figsize = (14,8))
# Low
img = ax[0][0].pcolormesh(radiiLow/apo, tfb_Low, abs_col_orb_en, norm=norm_orb_ensix, cmap = cmap)
cb = fig.colorbar(img)
ax[0][0].set_title('Specific (absolute) orbital energy', fontsize = 20)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[0][0].set_xscale('log')
ax[0][0].text(0.04, 0.15,'Low res', fontsize = 25)

img = ax[0][1].pcolormesh(radiiLow/apo, tfb_Low, col_ie,  norm=norm_iesix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[0][1].set_title('Specific internal energy', fontsize = 20)
ax[0][1].set_xscale('log')
ax[0][1].text(0.04, 0.15,'Low res', fontsize = 25)


img = ax[0][2].pcolormesh(radiiLow/apo, tfb_Low, col_Rad, norm=norm_Radsix, cmap = cmap)
cb = fig.colorbar(img)
ax[0][2].set_title('Radiation energy density', fontsize = 20)
cb.set_label(r'Eenergy density [erg/cm$^3$]', fontsize = 20, labelpad = 2)
ax[0][2].set_xscale('log')
ax[0][2].text(0.1, 0.1,'Low res', fontsize = 25, color = 'white')


# Middle
img = ax[1][0].pcolormesh(radiiMiddle/apo, tfb_Middle, abs_col_orb_enMiddle, norm=norm_orb_ensix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[1][0].set_xscale('log')
ax[1][0].text(0.04, 0.15,'High res', fontsize = 25)

img = ax[1][1].pcolormesh(radiiMiddle/apo, tfb_Middle, col_ieMiddle, norm=norm_iesix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[1][1].set_xscale('log')
ax[1][1].text(0.04, 0.15,'High res', fontsize = 25)

img = ax[1][2].pcolormesh(radiiMiddle/apo, tfb_Middle, col_RadMiddle, norm=norm_Radsix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Eenergy density [erg/cm$^3$]', fontsize = 20, labelpad = 2)
ax[1][2].set_xscale('log')
ax[1][2].text(0.1, 0.1,'High res', fontsize = 25, color = 'white')
for i in range(2):
    for j in range(3):
        ax[i][j].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        ax[i][j].text(Rt/apo+0.07, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
        ax[i][j].axhline(0.205, c = 'white', linewidth = 0.4)
        ax[i][j].axhline(0.52, c = 'white', linewidth = 0.4)

        # Grid for radii, to be matched with the cfr in slices.py
        ax[i][j].axvline(0.1, c = 'white', linewidth = 0.4)
        ax[i][j].axvline(0.3, c = 'white', linewidth = 0.4)
        ax[i][j].axvline(0.5, c = 'white', linewidth = 0.4)
        ax[i][j].axvline(1, c = 'white', linewidth = 0.4)
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

#%% Just one for Crete
fig, ax = plt.subplots(1,1, figsize = (14,12))
img = ax.pcolormesh(radiiMiddle/apo, tfb_Middle, abs_col_orb_enMiddle, norm=norm_orb_ensix, cmap = cmap)
cb = fig.colorbar(img)
cb.ax.tick_params(labelsize=30)
cb.ax.tick_params(which='major', size=7) 
cb.ax.tick_params(which='minor', size=5)  
cb.set_label(r'Specific energy [erg/g]', fontsize = 35, labelpad = 5)
cb.ax.tick_params(which = 'minor', size=10)
ax.set_xscale('log')
# ax.text(0.05, 0.15,'High res', fontsize = 14)
ax.set_ylabel(r't/t$_{fb}$', fontsize = 35)
ax.set_xlabel(r'$R/R_a$', fontsize = 35)
plt.tick_params(axis = 'both', which = 'both', direction='in', size = 10, labelsize=35)
plt.tick_params(axis = 'both', which = 'major',  size = 10)
plt.tick_params(axis = 'x', which = 'minor',  size = 7)
ax.text(Rt/apo+0.07, 0.65, r'R$_t$', fontsize = 35, rotation = 90, transform = ax.transAxes, color = 'k')
ax.axhline(0.205, c = 'k', linewidth = 0.5)
ax.axhline(0.52, c = 'k', linewidth = 0.5)

# Grid for radii, to be matched with the cfr in slices.py
ax.axvline(Rt/apo, linestyle ='dashed', c = 'k', linewidth = 1)
ax.axvline(0.1, c = 'k', linewidth = 0.5)
ax.axvline(0.3, c = 'k', linewidth = 0.5)
ax.axvline(0.5, c = 'k', linewidth = 0.5)
ax.axvline(1, c = 'k', linewidth = 0.5)
ax.set_xscale('log')
plt.title('Specific (absolute) orbital energy', fontsize = 30)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{cutoff}coloredEorbMiddle.png')
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
denominator_orb_en = (col_orb_en + col_orb_enMiddle)/2
denominator_ie = (col_ie + col_ieMiddle)/2
denominator_Rad = (col_Rad + col_RadMiddle)/2
rel_orb_en = np.abs(diff_orb_en / denominator_orb_en)
rel_ie = np.abs(diff_ie / denominator_ie)
rel_Rad = np.abs(diff_Rad / denominator_Rad)

rel_orb_en[np.isnan(rel_orb_en)] = 0 #nothing
rel_ie[np.isnan(rel_ie)] = 0
rel_Rad[np.isnan(rel_Rad)] = 0

#%%
# binsorb = np.linspace(np.min(rel_orb_en), np.max(rel_orb_en), 2000)
median_relorb = np.median(rel_orb_en)
p5_relorb = np.percentile(rel_orb_en, 5)
p95_relorb = np.percentile(rel_orb_en, 95)

# binsie = np.linspace(np.min(rel_ie), np.max(rel_ie), 2000)
median_relie = np.median(rel_ie)
p5_relie = np.percentile(rel_ie, 5)
p95_relie = np.percentile(rel_ie, 95)

# binsRad = np.linspace(np.min(rel_Rad), np.max(rel_Rad), 2000)
median_relRad = np.median(rel_Rad)
p5_relRad = np.percentile(rel_Rad, 5)
p95_relRad = np.percentile(rel_Rad, 95)

fix, ax = plt.subplots(1,3, figsize = (14,5))
ax[0].hist(rel_orb_en.flatten(), bins = 8000, cumulative = True, density = True,  color = 'mediumorchid', alpha = 0.5)
ax[0].set_title('Specific (absolute) orbital energy relative', fontsize = 15)
ax[0].text(0.25, 0.1, r'$\Delta_{rel}$ = ' + f'{np.round(median_relorb,4)}\n' + r'$p_5$ = ' + f'{np.round(p5_relorb,4)}\n'+ r'$p_{95}$ = ' + f'{np.round(p95_relorb,4)}', fontsize = 20)
ax[0].set_xlim(0, 0.5)

ax[1].hist(rel_ie.flatten(), bins = 8000, cumulative = True, density = True,  color = 'crimson', alpha = 0.5)
ax[1].set_title('Specific internal energy relative', fontsize = 15)
ax[1].text(0.5, 0.1, r'$\Delta_{rel}$ = ' + f'{np.round(median_relie,4)}\n' + r'$p_5$ = ' + f'{np.round(p5_relie,4)}\n'+ r'$p_{95}$ = ' + f'{np.round(p95_relie,4)}', fontsize = 20)
ax[1].set_xlim(0, 1)

ax[2].hist(rel_Rad.flatten(), bins = 8000, cumulative = True, density = True,  color = 'orange', alpha = 0.5)
ax[2].set_title('Radiation energy density relative', fontsize = 15)
ax[2].text(1, 0.1, r'$\Delta_{rel}$ = ' + f'{np.round(median_relRad,4)}\n' + r'$p_5$ = ' + f'{np.round(p5_relRad,4)}\n'+ r'$p_{95}$ = ' + f'{np.round(p95_relRad,4)}', fontsize = 20)
ax[2].set_xlim(0, 2)

for i in range(3):
    ax[i].set_ylabel('CDF', fontsize = 18)
    ax[i].set_xlabel('Relative error', fontsize = 20)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{cutoff}coloredE_relative_diff_hist.png')
plt.show()

#%%
median_orb_en = np.median(rel_orb_en)
median_ie = np.median(rel_ie)
median_Rad = np.median(rel_Rad)

print(f'Median relative difference in Orbital energy: {median_orb_en}')
print(f'Median relative difference in IE: {median_ie}')
print(f'Median relative difference in Radiation energy density: {median_Rad}')

rel_orb_en_forlog = np.copy(rel_orb_en) #you need it for the Log
rel_ie_forlog = np.copy(rel_ie)
rel_Rad_forlog = np.copy(rel_Rad)
rel_orb_en_forlog[rel_orb_en_forlog<=0] = 1 #you need it for the Log
rel_ie_forlog[rel_ie_forlog<=0] =1
rel_Rad_forlog[rel_Rad_forlog<=0] = 1

cmap = plt.cm.inferno
norm_orb_en = colors.LogNorm(vmin=np.percentile(rel_orb_en_forlog[rel_orb_en_forlog!=1], 5), vmax=np.percentile(rel_orb_en_forlog[rel_orb_en_forlog!=1], 95))
norm_ie = colors.LogNorm(vmin=np.percentile(rel_ie_forlog[rel_ie_forlog!=1], 5), vmax=np.percentile(rel_ie_forlog[rel_ie_forlog!=1], 95))
norm_Rad = colors.LogNorm(vmin=np.percentile(rel_Rad_forlog[rel_Rad_forlog!=1], 5), vmax=np.percentile(rel_Rad_forlog[rel_Rad_forlog!=1], 95))

fig, ax = plt.subplots(1,3, figsize = (20,6))
img = ax[0].pcolormesh(radiiMiddle/apo, tfb_Middle, rel_orb_en_forlog, cmap=cmap, norm=norm_orb_en)#, vmin = np.min(rel_orel_orb_en_fologrb_en), vmax = 0.2)
cb = fig.colorbar(img)
# cb.ax.tick_params(labelsize=20)
ax[0].set_title('Specific (absolute) orbital energy', fontsize = 20)
cb.set_label('Relative difference', fontsize = 25)#Relative difference $|E_{orb}|$/Mass', fontsize = 14, labelpad = 5)
ax[0].set_xscale('log')

img = ax[1].pcolormesh(radiiMiddle/apo, tfb_Middle, rel_ie_forlog, cmap=cmap, norm=norm_ie)#, vmin = np.min(rel_ie_forlog), vmax = np.max(rel_ie_forlog))
cb = fig.colorbar(img)
cb.ax.tick_params(labelsize=20)
ax[1].set_title('Specific internal energy', fontsize = 20)
cb.set_label('Relative difference', fontsize = 25)#Relative difference $|$IE$|$/Mass', fontsize = 14, labelpad = 5)
ax[1].set_xscale('log')

img = ax[2].pcolormesh(radiiMiddle/apo, tfb_Middle, rel_Rad_forlog, cmap=cmap, norm=norm_Rad)#, vmin = np.min(rel_Rad_forlog), vmax = np.max(rel_Rad_forlog))
cb = fig.colorbar(img)
# cb.ax.tick_params(labelsize=2)
ax[2].set_title('Radiation energy density', fontsize = 20)
cb.set_label('Relative difference', fontsize = 25)#'Relative difference $|E_{rad}|$/Vol', fontsize = 14, labelpad = 5)
ax[2].set_xscale('log')

for i in range(3):
    ax[i].axvline(Rt/apo, linestyle ='--', c = 'k')
    ax[i].text(Rt/apo+0.08, 0.25, r'R$_t$', fontsize = 20, rotation = 90, transform = ax[i].transAxes, color = 'k')
    ax[i].tick_params(axis = 'both', which = 'both', direction='in', labelsize=20)

# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 25)
ax[0].set_xlabel(r'$R/R_a$', fontsize = 25)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 25)
ax[2].set_xlabel(r'$R/R_a$', fontsize = 25)
plt.tick_params(axis = 'both', which = 'both', direction='in')
if cutoff == 'bound':
    plt.suptitle(r'Relative differences: $|$Low-High$|$/mean $M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$, bound elements only', fontsize = 18)
else:
    plt.suptitle(r'Relative differences: $|$Low-High$|$/mean $M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{cutoff}coloredE_relative_diff.png')
plt.show()


# %% Lines
# indices = [70, 100, 136]
indices = [np.argmin(np.abs(tfb_Low-0.46)), np.argmin(np.abs(tfb_Low-0.66)), np.argmin(np.abs(tfb_Low-0.86))]
colors_indices = ['navy', 'royalblue', 'deepskyblue']
lines_difference = (col_Rad[indices]-col_RadMiddle[indices])/col_RadMiddle[indices]

img, ax = plt.subplots(1,2, figsize = (16,4))
for i,idx in enumerate(indices):
    ax[0].plot(radiiLow, col_Rad[idx], c = colors_indices[i], label = f'Low t/tfb = {np.round(tfb_Low[idx],2)}')
    ax[0].plot(radiiMiddle, col_RadMiddle[idx], '--', c = colors_indices[i], label = f'Middle t/tfb = {np.round(tfb_Middle[idx],2)}')
    ax[1].plot(radiiLow, lines_difference[i], c = colors_indices[i], label = f't/tfb = {np.round(tfb_Low[idx],2)}')
ax[0].set_xlabel(r'R/R$_a$', fontsize = 20)
ax[0].set_ylabel(r'(Rad/Vol) [erg/cm$^3$]', fontsize = 20)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 20)
ax[1].set_ylabel(r'$|$Low-High$|$/mean', fontsize = 20)
ax[0].loglog()
ax[1].loglog()
ax[0].legend(fontsize = 16)
ax[1].legend(fontsize = 18)
ax[0].grid()
ax[1].grid()
plt.suptitle('Relative differences: 1-High/Low')
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/Rad_lines.png')
plt.show()

#%%
Lum_cgs = col_Rad  * prel.c * 4 * np.pi * (radiiLow*prel.Rsol_to_cm)**2 
LumMiddle_cgs = col_RadMiddle * prel.c * 4 * np.pi * (radiiMiddle*prel.Rsol_to_cm)**2 
denom = (Lum_cgs + LumMiddle_cgs)/2
Lum_difference = (Lum_cgs[indices]-LumMiddle_cgs[indices])/denom[indices]

img, ax = plt.subplots(1,2, figsize = (20,7))
for i,idx in enumerate(indices):
    if i==2 :
        continue
    if i == 0:
        ax[0].plot(radiiLow, Lum_cgs[idx], c = colors_indices[i], label = f'Low res')#t/tfb = {np.round(tfb_Low[idx],2)}')
        ax[0].plot(radiiMiddle, LumMiddle_cgs[idx], '--', c = colors_indices[i], label = f'High res')#t/tfb = {np.round(tfb_Middle[idx],2)}')
        ax[1].plot(radiiLow, Lum_difference[i], c = colors_indices[i])#, label = f't/tfb = {np.round(tfb_Low[idx],2)}')
    else:   
        ax[0].plot(radiiLow, Lum_cgs[idx], c = colors_indices[i])#, label = f'Low t/tfb = {np.round(tfb_Low[idx],2)}')
        ax[0].plot(radiiMiddle, LumMiddle_cgs[idx], '--', c = colors_indices[i])#, label = f'Middle t/tfb = {np.round(tfb_Middle[idx],2)}')
        ax[1].plot(radiiLow, Lum_difference[i], c = colors_indices[i])#, label = f't/tfb = {np.round(tfb_Low[idx],2)}')
ax[0].set_ylim(1e40, 1e44)
ax[1].set_ylim(0.3, 1.8)
ax[0].text(15, 1.5e41, r'$t/t_{fb}$ = '+ f'{np.round(tfb_Low[indices[0]],2)}', fontsize = 20)
ax[0].text(20, 2e42, r'$t/t_{fb}$ = '+ f'{np.round(tfb_Low[indices[1]],2)}', fontsize = 20)
# ax[0].text(50, 4e43, r'$t/t_{fb}$ = '+ f'{np.round(tfb_Low[indices[2]],2)}', fontsize = 20)
# ax[1].text(620, 0.7, r'$\Delta_{rel}\approx$ '+ f'{np.round(np.mean(Lum_difference[0][-10:-1]),2)}', fontsize = 20)
# ax[1].text(620, 0.5, r'$\Delta_{rel}\approx$ '+ f'{np.round(np.mean(Lum_difference[1][-10:-1]),2)}', fontsize = 19)
# ax[1].text(620, 0.4, r'$\Delta_{rel}\approx$ '+ f'{np.round(np.mean(Lum_difference[2][-10:-1]),2)}', fontsize = 20)

ax[0].tick_params(axis='both', which='major', labelsize=25)
ax[1].tick_params(axis='both', which='major', labelsize=25)
ax[0].tick_params(axis='both', which='minor', size=4)
ax[1].tick_params(axis='both', which='minor', size=4)
ax[0].tick_params(axis='both', which='major', size=6)
ax[1].tick_params(axis='both', which='major', size=6)
ax[0].set_xlabel(r'R/R$_a$', fontsize = 28)
ax[1].set_xlabel(r'$R/R_a$', fontsize = 28)
ax[0].set_ylabel(r'Luminosity [erg/s]', fontsize = 25)
ax[1].set_ylabel(r'Relative difference', fontsize = 25, labelpad = 1)
ax[0].loglog()
ax[0].legend(fontsize = 25)
ax[0].grid()
ax[1].grid()
ax[1].loglog()
plt.subplots_adjust(wspace=1)
plt.suptitle('Relative differences: $|$Low-High$|$/mean')
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/Luminosity.png')
plt.show()


# %%
for i in range(3):
    mean_error =  np.round(np.mean(Lum_difference[i][-10:-1]),2)
    print(f'Mean relative error for t/tfb = {tfb_Low[indices[i]]} is {mean_error}')
# print the difference of the last point of Hires lum line
for i,idx in enumerate(indices):
    if i ==2:
        continue
    high = LumMiddle_cgs[indices[i+1]]
    low = LumMiddle_cgs[indices[i]]
    mean_lasthigh = np.mean(high[-10:-1])
    mean_lastlow = np.mean(low[-10:-1])
    print(tfb_Low[indices[i+1]], tfb_Low[indices[i]])
    print(f'The relative difference of the last point of the high res line is {np.round((mean_lasthigh-mean_lastlow),2)}')

# %%
