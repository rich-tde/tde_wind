
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import numba
from Utilities.operators import make_tree
from Utilities.selectors_for_snap import select_snap
from Utilities.isalice import isalice
alice, plot = isalice()

#
## CONSTANTS
#

G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

#
## FUNCTIONS
#

def grid_maker(path, snap, m, mstar, Rstar, x_num, y_num, z_num = 100):
    """ ALL outputs are in in solar units """
    Mbh = 10**m
    Rt = Rstar * (Mbh/mstar)**(1/3)
    R0 = 0.6 * Rt
    apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

    x_start = -1.2*apo
    x_stop = R0
    xs = np.linspace(x_start, x_stop, num = x_num )
    y_start = -0.2 * apo 
    y_stop = 0.2 * apo
    ys = np.linspace(y_start, y_stop, num = y_num)
    z_start = -2 * Rt
    z_stop = 2 * Rt
    zs = np.linspace(z_start, z_stop, z_num) #simulator units

    data = make_tree(path, snap, energy = True)
    sim_tree = data.sim_tree

    gridded_indexes =  np.zeros(( len(xs), len(ys), len(zs) ))
    gridded_den =  np.zeros(( len(xs), len(ys), len(zs) ))
    for i in range(len(xs)):
        for j in range(len(ys)):
            for k in range(len(zs)):
                queried_value = [xs[i], ys[j], zs[k]]
                _, idx = sim_tree.query(queried_value)
                                    
                # Store
                gridded_indexes[i, j, k] = idx
                gridded_den[i, j, k] = data.Den[idx]

    return gridded_indexes, gridded_den, xs, ys, zs

@numba.njit
def projector(gridded_den, x_radii, y_radii, z_radii):
    """ Project density on XY plane. NB: to plot you have to transpose the saved data"""
    # Make the 3D grid 
    flat_den =  np.zeros(( len(x_radii), len(y_radii) ))
    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
            for k in range(len(z_radii) - 1): # NOTE SKIPPING LAST Z PLANE
                dz = (z_radii[k+1] - z_radii[k]) 
                flat_den[i,j] += gridded_den[i,j,k] * dz

    return flat_den
 
if __name__ == '__main__':
    save = True
    plot = True
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    check = 'Low'
    compton = 'Compton'

    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) #[100,115,164,199,216]

    for snap in snaps:
        if alice:
            if check == 'Low':
                check = ''
            path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}/snap_{snap}'
        else:
            path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'

            _, grid_den, x_radii, y_radii, z_radii = grid_maker(path, snap, m, mstar, Rstar, x_num=51, y_num=50, z_num = 10)
            flat_den = projector(grid_den, x_radii, y_radii, z_radii)

        if save:
            if alice:
                if check == '':
                    check = 'Low'
                prepath = f'/data1/martirep/shocks/shock_capturing'
            else: 
                prepath = f'/Users/paolamartire/shocks'
            np.savetxt(f'{prepath}/data/{folder}/{check}/denproj{snap}.txt', flat_den) 
            np.savetxt(f'{prepath}/data/{folder}/{check}/xarray.txt', x_radii)
            np.savetxt(f'{prepath}/data/{folder}/{check}/yarray.txt', y_radii)

#%% Plot
        if plot:
            import colorcet
            fig, ax = plt.subplots(1,1)
            plt.rcParams['text.usetex'] = True
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['figure.figsize'] = [6, 4]
            plt.rcParams['axes.facecolor']=     'whitesmoke'
            
            # Clean
            den_plot = np.nan_to_num(flat_den, nan = -1, neginf = -1)
            den_plot = np.log10(den_plot)
            den_plot = np.nan_to_num(den_plot, neginf= 0)
            
            # Specify
            
            cb_text = r'Density [g/cm$^2$]'
            vmin = 0
            vmax = 6
            
            # ax.set_xlim(-1, 10/20_000)
            # ax.set_ylim(-0.2, 0.2)
            ax.set_xlabel(r' X [$R_\odot$]', fontsize = 14)
            ax.set_ylabel(r' Y [$R_\odot$]', fontsize = 14)
            img = ax.pcolormesh(x_radii, y_radii, den_plot.T, cmap = 'cet_fire')
                                #vmin = vmin, vmax = vmax)
            cb = plt.colorbar(img)
            cb.set_label(cb_text, fontsize = 14)

            ax.set_title('XY Projection', fontsize = 16)
            plt.show()
