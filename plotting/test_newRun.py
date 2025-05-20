""" Compare different resolutions"""
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'

import numpy as np
from Utilities.operators import make_tree
from Utilities.time_extractor import days_since_distruption
import Utilities.sections as sec
import src.orbits as orb
import Utilities.prelude as prel
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#%%
## PARAMETERS STAR AND BH
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
Rt = orb.tidal_radius(Rstar, mstar, Mbh) 
Rp = orb.pericentre(Rstar, mstar, Mbh, beta)
R0 = 0.6*Rp
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
compton = 'Compton'
snap = 214
checks = ['LowRes', '', 'HiRes', 'QuadraticOpacity']
check_name = ['Low', 'Fid','High', 'NewFid']

fig, ax = plt.subplots(2, 2, figsize = (20, 15))
for i,check in enumerate(checks):
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}' 
    path = f'{abspath}/TDE/{folder}/{snap}'
    tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    data = make_tree(path, snap, energy = True)
    cut = data.Den > 1e-19
    X, Y, Z, mass, den, Vol, Press, Temp, Diss_den = \
       sec. make_slices([data.X, data.Y, data.Z, data.Mass, data.Den, data.Vol, data.Press, data.Temp, data.Diss], cut)
    Ncell = len(data.X)*1e-6
    dim_cell = Vol**(1/3) 

    midplane = np.abs(Z) < dim_cell
    X_midplane, Y_midplane, Z_midplane, dim_midplane, Mass_midplane, Den_midplane, Press_midplane, Temp_midplane, Diss_den_midplane = \
    sec.make_slices([X, Y, Z, dim_cell, mass, den, Press, Temp, Diss_den], midplane)

    idx1_img = i%2
    idx2_img = i//2
    img = ax[idx1_img][idx2_img].scatter(X_midplane/apo, Y_midplane/apo, c = Mass_midplane, s = 1, cmap = 'turbo', norm = colors.LogNorm(vmin = 1e-11, vmax = 1e-8))
    ax[idx1_img][idx2_img].set_title(f'{check_name[i]}, ' + f't = {str(np.round(tfb,2))} ' + r't$_{\rm fb}$, total $N_{\rm cell} \approx$ ' + str(int(Ncell)) + r'$\cdot 10^6$', fontsize = 20)
    ax[idx1_img][idx2_img].axhline(2*Rt/apo, c = 'k', ls = '--', alpha = 0.5)
    ax[idx1_img][idx2_img].axhline(-2*Rt/apo, c = 'k', ls = '--', alpha = 0.5)
    ax[idx1_img][idx2_img].axvline(2*Rt/apo, c = 'k', ls = '--', alpha = 0.5)
    ax[idx1_img][idx2_img].axvline(-2*Rt/apo, c = 'k', ls = '--', alpha = 0.5)
    ax[idx1_img][idx2_img].set_xlim(-.15,.15)
    ax[idx1_img][idx2_img].set_ylim(-.15, .15)
    cbar = plt.colorbar(img)
    cbar.set_label(r'Mass $[M_\odot]$', fontsize = 16)
    
for i in range(2):
    ax[i][0].set_ylabel(r'Y [$R_{\rm a}$]', fontsize = 18)
for j in range(2):
    ax[1][j].set_xlabel(r'X [$R_{\rm a}$]', fontsize = 18)

plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/TestOpacMass{snap}.png', dpi = 100, bbox_inches = 'tight')
