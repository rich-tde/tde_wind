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
check = 'Low' 

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
path = f'/Users/paolamartire/shocks/data/{folder}'
data = np.loadtxt(f'{path}/coloredE_{check}.txt')
tfb = data[3]

Rt = rstar * (Mbh/mstar)**(1/3) # Msol = 1, Rsol = 1
apocenter = Rt * (Mbh/mstar)**(1/3)

radii_start = np.log10(0.4*Rt)
radii_stop = np.log10(apocenter) # apocenter
radii = np.logspace(radii_start, radii_stop, 100) / apocenter
# radii4 = np.linspace(0.2*2*Rt4, apocenter, 100) 

#diff = np.abs(ecc[:len(ecc2)] - ecc2)
#%%
fig, ax = plt.subplots(1,1, figsize = (4,4))

img1 = ax.pcolormesh(radii, days, ecc,
                     cmap = 'cet_rainbow4', vmin = 0, vmax = 1)


cb = fig.colorbar(img1)
cb.set_label('Eccentricity', fontsize = 14, labelpad = 5)
plt.xscale('log')
plt.ylim(0.3)

plt.axvline(Rt/apocenter, c = 'white')
plt.text(Rt/apocenter + 0.005, 0.5, '$R_\mathrm{T}$', 
         c = 'white', fontsize = 14)
# Axis labels
fig.text(0.5, -0.01, r'r/R$_a$', ha='center', fontsize = 14)
fig.text(-0.02, 0.5, r' Time / Fallback time $\left[ t/t_{FB} \right]$', va='center', rotation='vertical', fontsize = 14)
ax.tick_params(axis = 'both', which = 'both', direction='in')
ax.set_title(r'$10^5$ M$_\odot$')

#%%
import src.Utilities.prelude as c
fig, ax = plt.subplots(1,1, figsize = (4,4))

plt.plot(radii, np.abs(ecc2[136] - ecc[136]), 
         c = 'k', lw = 2, label = f'{days[136]:.2f} t/$t_{{FB}}$')
plt.plot(radii, np.abs(ecc2[80] - ecc[80]), ls = ':', 
         c = c.AEK, lw = 2, label = f'{days[80]:.2f} t/$t_{{FB}}$')
plt.plot(radii, np.abs(ecc2[50] - ecc[50]), ls = ':',
         c = 'maroon', lw = 2, label = f'{days[50]:.2f} t/$t_{{FB}}$')

plt.xscale('log')
plt.yscale('log')

# Rt
plt.axvline(Rt/apocenter, c = 'r', ls = '--')
plt.text(Rt/apocenter + 0.001, 0.0001, '$R_\mathrm{T}$', 
         c = 'r', fontsize = 17)

# Labels
plt.xlabel('Radius [r/$R_\mathrm{a}$]', fontsize = 14)
plt.ylabel('Eccentricity Difference', fontsize = 14)
plt.legend()
