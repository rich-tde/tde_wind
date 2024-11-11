
""" 
Project density on XY plane and save data.
Plot the projection.
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.spatial import KDTree
from Utilities.selectors_for_snap import select_snap
from Utilities.isalice import isalice
alice, plot = isalice()
from src.orbits import make_cfr
from Utilities.sections import make_slices

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
    # R0 = 0.6 * Rt
    apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

    x_start = -1.2*apo
    x_stop = 40
    xs = np.linspace(x_start, x_stop, num = x_num )
    y_start = -0.2 * apo 
    y_stop = 0.2 * apo
    ys = np.linspace(y_start, y_stop, num = y_num)
    z_start = -2 * Rt
    z_stop = 2 * Rt
    zs = np.linspace(z_start, z_stop, z_num) #simulator units

    # data = make_tree(path, snap, energy = True)
    # sim_tree = data.sim_tree
    X = np.load(f'{path}/CMx_{snap}.npy')
    Y = np.load(f'{path}/CMy_{snap}.npy')
    Z = np.load(f'{path}/CMz_{snap}.npy')
    Den = np.load(f'{path}/Den_{snap}.npy')
    # make cut in density
    cutden = Den > 1e-9
    x_cut, y_cut, z_ccut, den_cut = \
        make_slices([X, Y, Z, Den], cutden)
    points_tree = np.array([x_cut, y_cut, z_ccut]).T
    sim_tree = KDTree(points_tree)

    gridded_indexes =  np.zeros(( len(xs), len(ys), len(zs) ))
    gridded_den =  np.zeros(( len(xs), len(ys), len(zs) ))
    for i in range(len(xs)):
        for j in range(len(ys)):
            for k in range(len(zs)):
                queried_value = [xs[i], ys[j], zs[k]]
                _, idx = sim_tree.query(queried_value)
                                    
                # Store
                gridded_indexes[i, j, k] = idx
                gridded_den[i, j, k] = den_cut[idx]

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
    if alice:
        cast = True
    else:
        cast = False
    
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    Rt = Rstar * (Mbh/mstar)**(1/3)
    xcrt, ycrt, crt = make_cfr(Rt)
    apocenter = Rt**2 / Rstar
    check = 'HiRes'
    compton = 'Compton'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

    if cast:
        snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 
        for snap in snaps:
            if snap < 216:
                continue
            if alice:
                if check == 'Low':
                    check = ''
                path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}/snap_{snap}'
            else:
                path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'

            _, grid_den, x_radii, y_radii, z_radii = grid_maker(path, snap, m, mstar, Rstar, x_num=500, y_num=500, z_num = 100)
            flat_den = projector(grid_den, x_radii, y_radii, z_radii)

            if save:
                if alice:
                    if check == '':
                        check = 'Low'
                    prepath = f'/data1/martirep/shocks/shock_capturing'
                else: 
                    prepath = f'/Users/paolamartire/shocks'
                np.savetxt(f'{prepath}/data/{folder}/{check}/projection/denproj{snap}.txt', flat_den) 
                np.savetxt(f'{prepath}/data/{folder}/{check}/projection/xarray.txt', x_radii)
                np.savetxt(f'{prepath}/data/{folder}/{check}/projection/yarray.txt', y_radii)
        if save:
            with open(f'{prepath}/data/{folder}/{check}/projection/time_proj.txt', 'a') as f:
                f.write(f'# snaps \n' + ' '.join(map(str, snaps)) + '\n')
                f.write(f'# t/t_fb \n' + ' '.join(map(str, tfb)) + '\n')
                f.close()

#%% Plot
    if plot:
        import Utilities.prelude as prel

        first_s = 100
        last_s = 217
        snaps = np.arange(first_s,last_s,1)
        tfb = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{check}/projection/time_proj.txt')[1]
        cut = 'NOCUT' #or 'NOCUT' or ''

        # Load and clean
        for i, snap in enumerate(snaps):
            flat_den = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{check}/projection/denproj{snap}.txt')
            x_radii = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{check}/projection/xarray.txt')
            y_radii = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{check}/projection/yarray.txt')
            tfb_single = tfb[i]

            den_plot = np.log10(flat_den)
            den_plot = np.nan_to_num(den_plot, neginf= 0, posinf = 0)

            if cut != 'NOCUT':
                cut_den_plot = den_plot.copy()
                cut_x_radii = x_radii.copy()
                cut_y_radii = y_radii.copy()
                for i in range(len(x_radii)):
                    for j in range(len(y_radii)):
                        if den_plot[i,j] < -9:
                            cut_den_plot[i,j] = -200
                
            p5 = np.percentile(den_plot, 5)
            p95 = np.percentile(den_plot, 95)
            # print('put min and max as ', p5, p95)
            
            fig, ax = plt.subplots(1,1, figsize = (14,4))
            ax.set_xlabel(r'$X [R_\odot]$', fontsize = 20)
            ax.set_ylabel(r'$Y [R_\odot]$', fontsize = 20)
            if cut == 'NOCUT':
                img = ax.pcolormesh(x_radii, y_radii, den_plot.T, cmap = 'inferno',
                                vmin = -10, vmax = -5.6)
            else:
                img = ax.pcolormesh(x_radii, y_radii, cut_den_plot.T, cmap = 'inferno',
                                vmin = -9, vmax = -5.6)
            # add an  point with just white contour at 0,0 
            # ax.plot(0,0, 'o', color = 'k', markersize = 8, markeredgecolor = 'darkorange')
            # ax.contour(xcrt, ycrt, crt, [0], linestyles = 'dashed', colors = 'white', alpha = 0.8)
            cb = plt.colorbar(img)
            cb.set_label(r'$\log_{10}$ (Column density [$M_\odot/R_\odot^2$])', fontsize = 18)

            # ax.set_title(f'XY Projection, snap {snap}', fontsize = 16)
            # ax.text(-400, 50, r't/t$_{fb}$ = ' + f'{np.round(tfb_single,2)}', color = 'white', fontsize = 18)
            plt.tight_layout()
            if save:
                plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{check}/projection{cut}/denproj{snap}.png')
            # plt.show()
            plt.close()

# %%
