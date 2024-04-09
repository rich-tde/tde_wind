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
import Utilities.prelude as prel

cross_section = True

folder = 'sedov'
is_tde = False
snap = '100'
path = f'{folder}/{snap}'
gamma = 5/3

sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, Press, Temp = make_tree(path, snap, is_tde, int_energy=False)
entropy = Press / Den**gamma
condition = np.abs(Z)<Vol**(1/3)
cross_x = X[condition]
cross_y = Y[condition]
cross_den = Den[condition]
cross_entropy = entropy[condition]

# Plot 
choose = 'Density'
fig, ax = plt.subplots(1,1, figsize = (8,7))

if choose == 'Density':
    toplot = cross_den
if choose == 'Entropy':
    toplot = cross_entropy

img = ax.scatter(cross_x, cross_y, c = np.log10(toplot), cmap = 'jet')#, vmin = -12, vmax = -7)
cb = plt.colorbar(img)
cb.set_label(r'$log_{10}$' + f'{choose}', fontsize = 14)
ax.set_xlim(-1.,1.)
ax.set_ylim(-1.,1.)
ax.set_xlabel(r'X', fontsize = 14)
ax.set_ylabel(r'Y', fontsize = 14)
plt.grid()
plt.title(r'Cross section, $|z|<V^{1/3}$')
plt.savefig(f'Figs/{snap}/{choose}.png')

plt.show()
