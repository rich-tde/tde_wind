"""
Density cross sections.

@author: paola

"""
import sys
sys.path.append('/Users/paolamartire/shocks')


import numpy as np
import matplotlib.pyplot as plt
import numba

from Utilities.operators import make_tree
from Utilities.time_extractor import days_since_distruption
import Utilities.prelude as prel

cross_section = True

folder = 'TDE'
is_tde = True
snap = '683'
m = 6
Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
apocenter = 2 * Rt * Mbh**(1/3)
path = f'{folder}/{snap}'
gamma = 5/3
time, tfb = days_since_distruption(f'{path}/snap_{snap}.h5', choose = 'tfb')

sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, Press, Temp = make_tree(path, snap, is_tde, int_energy=False)
entropy = Press / Den**gamma

condition = np.abs(Z)<Vol**(1/3)
cross_x = X[condition]
cross_y = Y[condition]
cross_den = Den[condition]
cross_entropy = entropy[condition]
#cross_entropy = entropy[condition]

# Plot density
choose = 'Entropy'
fig, ax = plt.subplots(1,1, figsize = (12,7))
ax.set_xlabel(r'X [x/R$_\odot$]', fontsize = 14)
ax.set_ylabel(r'Y [y/R$_\odot$]', fontsize = 14)

if choose == 'Density':
    toplot = cross_den
    vmin = -12
    vmax = -7
if choose == 'Entropy':
    toplot = cross_entropy
    vmin = 1
    vmax = 5

img = ax.scatter(cross_x/apocenter, cross_y/apocenter, c = np.log10(toplot), s = 4, cmap = 'jet', vmin = vmin, vmax = vmax)
cb = plt.colorbar(img)
if choose == 'Density':
    cb.set_label(r'$log_{10}$ Density [$M_\odot/R_\odot^2$]', fontsize = 14)
if choose == 'Entropy':
    cb.set_label(r'$log_{10}$ Entropy', fontsize = 14)

ax.set_xlim(-0.8,0.1)
ax.set_ylim(-0.3,0.2)
# ax.set_xlim(3,46)
# ax.set_ylim(-16,21)
plt.grid()
plt.title(r'Cross section, $|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,3)))
plt.savefig(f'Figs/{snap}/{choose}_crosssect.png')

plt.show()
