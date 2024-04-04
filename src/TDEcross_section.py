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
is_tde =True
snap = '196'
path = f'{folder}/{snap}'
time, tfb = days_since_distruption(f'{path}/snap_{snap}.h5', choose = 'tfb')

sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, Press, Temp = make_tree(path, snap, is_tde, int_energy=False)

z_chosen = 0
delta = 0.3
cross_x = X[np.abs(Z-z_chosen)<delta]
cross_y = Y[np.abs(Z-z_chosen)<delta]
cross_den = Den[np.abs(Z-z_chosen)<delta]
logcross_den = np.log10(cross_den)

# Plot density
fig, ax = plt.subplots(1,1, figsize = (10,7))
ax.set_xlabel(r'X [x/R$_\odot$]', fontsize = 14)
ax.set_ylabel(r'Y [y/R$_\odot$]', fontsize = 14)
img = ax.scatter(cross_x, cross_y, c = logcross_den, s = 4, cmap = 'jet', vmin = -12, vmax = -7)
cb = plt.colorbar(img)
cb.set_label(r'$log_{10}$ Density [$M_\odot/R_\odot$]', fontsize = 14)
ax.set_xlim(3,46)
ax.set_ylim(-16,21)
plt.grid()
plt.title(f'Cross section, z = {z_chosen}, ' + r't/t$_{fb}$ = ' + str(np.round(tfb,3)))
plt.savefig(f'Figs/{snap}/den_crossect.png')

plt.show()
