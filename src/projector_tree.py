
""" 
Column density.
If alice: project density on XY plane and save data.
Else: plot the projection.
"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    prepath = '/data1/martirep/shocks/shock_capturing'
else:
    prepath = '/Users/paolamartire/shocks/'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numba
from scipy.spatial import KDTree
from Utilities.selectors_for_snap import select_snap
from src.orbits import make_cfr
from Utilities.sections import make_slices

#
## FUNCTIONS
#

def grid_maker(path, snap, m, mstar, Rstar, x_num, y_num, z_num = 100):
    """ ALL outputs are in in solar units """
    Mbh = 10**m
    Rt = Rstar * (Mbh/mstar)**(1/3)
    # R0 = 0.6 * Rt
    apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

    x_start = -8*apo#-1.2*apo
    x_stop = 5*apo#40
    xs = np.linspace(x_start, x_stop, num = x_num )
    y_start = -4*apo #-0.5 * apo 
    y_stop = 4*apo #0.5 * apo
    ys = np.linspace(y_start, y_stop, num = y_num)
    z_start = -apo #-2 * Rt
    z_stop = apo #2 * Rt
    zs = np.linspace(z_start, z_stop, z_num) #simulator units

    # data = make_tree(path, snap, energy = True)
    # sim_tree = data.sim_tree
    X = np.load(f'{path}/CMx_{snap}.npy')
    Y = np.load(f'{path}/CMy_{snap}.npy')
    Z = np.load(f'{path}/CMz_{snap}.npy')
    Den = np.load(f'{path}/Den_{snap}.npy')
    # make cut in density
    cutden = Den > 1e-19
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
            for k in range(len(z_radii) - 1): 
                dz = (z_radii[k+1] - z_radii[k]) 
                flat_den[i,j] += gridded_den[i,j,k] * dz

    return flat_den
 
if __name__ == '__main__':
    save = True
    
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    Rt = Rstar * (Mbh/mstar)**(1/3)
    xcrt, ycrt, crt = make_cfr(Rt)
    apocenter = Rt**2 / Rstar
    check = ''
    compton = 'Compton'
    if m== 6:
        folder = f'R{Rstar}M{mstar}BH1e+0{m}beta{beta}S60n{n}{compton}{check}'
    else:
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

    if alice:
        snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 
        t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)

        if save:
            with open(f'{prepath}/data/{folder}/projection/time_proj.txt', 'a') as f:
                f.write(f'# snaps \n' + ' '.join(map(str, snaps)) + '\n')
                f.write(f'# t/t_fb (t_fb = {t_fall})\n' + ' '.join(map(str, tfb)) + '\n')
                f.close()
        for snap in snaps:
            if alice:
                path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
            
            _, grid_den, x_radii, y_radii, z_radii = grid_maker(path, snap, m, mstar, Rstar, x_num=500, y_num=500, z_num = 100)
            flat_den = projector(grid_den, x_radii, y_radii, z_radii)

            if save:
                np.savetxt(f'{prepath}/data/{folder}/projection/bigdenproj{snap}.txt', flat_den) 
                np.savetxt(f'{prepath}/data/{folder}/projection/bigxarray.txt', x_radii)
                np.savetxt(f'{prepath}/data/{folder}/projection/bigyarray.txt', y_radii)

    else:
        import src.orbits as orb
        import Utilities.prelude as prel
        t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)
        print(t_fall)
        time = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/projection/time_proj.txt')
        snaps = time[0]#[int(i) for i in time[0]]
        tfb = time[1]
        xcrt, ycrt, crt = orb.make_cfr(Rt)

        # Load and clean
        snaps = [348]
        for i, snap in enumerate(snaps):
            flat_den = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/projection/bigdenproj{snap}.txt')
            flat_den *= prel.Msol_cgs/prel.Rsol_cgs**2
            x_radii = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/projection/bigxarray.txt')
            y_radii = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/projection/bigyarray.txt')
                
            # p5 = np.percentile(flat_den, 5)
            # p95 = np.percentile(flat_den, 95)
            
            fig, ax = plt.subplots(1,1, figsize = (14,8))
            ax.set_xlabel(r'$X [AU]$', fontsize = 20)
            ax.set_ylabel(r'$Y [AU]$', fontsize = 20)
            img = ax.pcolormesh(x_radii*prel.Rsol_AU, y_radii*prel.Rsol_AU, flat_den.T, cmap = 'inferno',
                                norm = colors.LogNorm(vmin = 5e-2, vmax = 1e2))
            cb = plt.colorbar(img)
            cb.set_label(r'Column density [$g/cm^2$])', fontsize = 18)
            # ax.contour(xcrt, ycrt, crt,  linestyles = 'dashed', colors = 'w', alpha = 1)
            ax.scatter(0, 0, color = 'k', edgecolors = 'orange', s = 40)
            tfb_single = tfb#tfb[i]
            # ax.text(-7*apocenter**prel.Rsol_AU, 3.5*apocenter**prel.Rsol_AU, r't = ' + f'{np.round(tfb_single*t_fall,2)} days', color = 'white', fontsize = 18)
            # ax.set_xlim(-210, 140)
            # ax.set_ylim(-90, 90) 
            plt.tight_layout()
            if save:
                plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/projection/denproj{snap}.png')
            plt.show()
            plt.close()

