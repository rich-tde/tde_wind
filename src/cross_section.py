"""
Project quantities or do cross sections.

@author: paola

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')


import numpy as np
import matplotlib.pyplot as plt
import numba

from Utilities.operators import make_tree
import Utilities.prelude as prel

z_chosen = 0
delta = 0.039

# shockzone = np.loadtxt(f'shockzone.txt')
# x_zone = shockzone[0]
# y_zone = shockzone[1]
# z_zone = shockzone[2]
# slice_zone = np.logical_and(z_zone<z_chosen+delta, z_zone>z_chosen-delta)
# zone_cross_x = x_zone[slice_zone]
# zone_cross_y = y_zone[slice_zone]

shocksurface = np.loadtxt(f'shocksurface.txt')
x_surf = shocksurface[0]
y_surf = shocksurface[1]
z_surf = shocksurface[2]
slice_surf = np.logical_and(z_surf<z_chosen+delta, z_surf>z_chosen-delta)
surf_cross_x = x_surf[slice_surf]
surf_cross_y = y_surf[slice_surf]

sim_tree, X, Y, Z, VX, VY, VZ, Den, Press, Temp = make_tree()

slice = np.logical_and(Z<z_chosen+delta, Z>z_chosen-delta)
cross_x = X[slice]
cross_y = Y[slice]
cross_den = Den[slice]
logcross_den = np.log10(cross_den)

# Plot density
fig, ax = plt.subplots(1,1)
ax.set_xlabel('X', fontsize = 14)
ax.set_ylabel('Y', fontsize = 14)
img = ax.scatter(cross_x, cross_y, c = logcross_den, cmap = 'jet')
# ax.plot(zone_cross_x, zone_cross_y, 'ks', markerfacecolor='none', ms = 5, markeredgecolor='b')
ax.plot(surf_cross_x, surf_cross_y, 'ks', markerfacecolor='none', ms = 5, markeredgecolor='k')
cb = plt.colorbar(img)
cb.set_label(r'$log_{10}$ Density', fontsize = 14)
plt.title(f'Sedov blastwave, z = {z_chosen}')
plt.savefig('dencrossect.png')
plt.show()
