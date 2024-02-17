"""
Project quantities or do cross sections.

@author: paola

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')


import numpy as np
import matplotlib.pyplot as plt
import numba

import Utilities.prelude as prel
from grid_maker import make_grid


@numba.njit
def projector(gridded_den, x_radii, y_radii, z_radii):
    """ Project density on XY plane. NB: to plot you have to transpose the saved data"""
    # Make the 3D grid 
    flat_den =  np.zeros(( len(x_radii), len(y_radii) ))
    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
            mass_zsum = 0
            for k in range(len(z_radii) - 1): # NOTE SKIPPING LAST Z PLANE
                dz = (z_radii[k+1] - z_radii[k]) 
                # mass_zsum += gridded_mass[i,j,k]
                flat_den[i,j] += gridded_den[i,j,k] * dz #* gridded_mass[i,j,k]
            #flat_den[i,j] = np.divide(flat_den[i,j], mass_zsum)
    return flat_den
 
if __name__ == '__main__':
    num = 45

    _, gridded_den, gridded_T, gridded_P, gridded_Vx, gridded_Vy, gridded_Vz, gridded_V, _, x_radii, y_radii, z_radii = make_grid(num)

    flat_den = projector(gridded_den, x_radii, y_radii, z_radii)
    den_plot = np.log10(flat_den)
    den_plot = np.nan_to_num(den_plot, neginf = 0)
    
    # Plot density
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('X', fontsize = 14)
    ax.set_ylabel('Y', fontsize = 14)
    img = ax.pcolormesh(x_radii, y_radii, den_plot.T, cmap = 'jet', vmin = 0, vmax = 0.4)
    cb = plt.colorbar(img)
    cb.set_label(r'$log_{10}$ Density', fontsize = 14)
    plt.title('Density projection')
    plt.savefig('denproj.png')
    plt.show()
