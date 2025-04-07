
""" 
Column density.
If alice: project density/dissipation on XY plane and save data.
Else: plot the projection.
"""
import sys
sys.path.append('/Users/paolamartire/shocks/')
abspath = '/Users/paolamartire/shocks'

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    prepath = '/data1/martirep/shocks/shock_capturing'
    compute = True
else:
    prepath = '/Users/paolamartire/shocks/'
    compute = False

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numba
from scipy.spatial import KDTree
from Utilities.selectors_for_snap import select_snap
from src.orbits import make_cfr
from Utilities.sections import make_slices
import Utilities.prelude as prel

#
## FUNCTIONS
#

def grid_maker(path, snap, m, mstar, Rstar, what_to_grid, x_num, y_num, z_num = 100):
    """ ALL outputs are in in solar units """
    Mbh = 10**m
    Rt = Rstar * (Mbh/mstar)**(1/3)
    # R0 = 0.6 * Rt
    apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

    x_start = -1.2*apo #-7*apo
    x_stop = apo #2.5*apo
    xs = np.linspace(x_start, x_stop, num = x_num )
    y_start = - apo #-4*apo 
    y_stop = apo  #3*apo
    ys = np.linspace(y_start, y_stop, num = y_num)
    z_start = -2*Rt #-apo 
    z_stop = 2*Rt #apo 
    zs = np.linspace(z_start, z_stop, z_num) #simulator units

    # data = make_tree(path, snap, energy = True)
    # sim_tree = data.sim_tree
    X = np.load(f'{path}/CMx_{snap}.npy')
    Y = np.load(f'{path}/CMy_{snap}.npy')
    Z = np.load(f'{path}/CMz_{snap}.npy')
    Den = np.load(f'{path}/Den_{snap}.npy')
    if what_to_grid == 'Diss':
        to_grid = np.load(f'{path}/Diss_{snap}.npy')
    if what_to_grid == 'Den':
        to_grid = Den
    if what_to_grid == 'both':
        to_grid = np.load(f'{path}/Diss_{snap}.npy') 
    if what_to_grid == 'tau_scatt':
        kappa_scatt = 0.34 / (prel.Rsol_cgs**2/prel.Msol_cgs) # it's 0.34cm^2/g, you want it in code units
        to_grid = kappa_scatt * Den
    if what_to_grid == 'tau_ross':
        Temp = np.load(f'{path}/T_{snap}.npy')
        sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(Temp), np.log(Den), 'linear', 0)
        sigma_rossland = np.array(sigma_rossland)[0]
        sigma_rossland_eval = np.exp(sigma_rossland) # [1/cm]
        sigma_rossland_eval[sigma_rossland == 0.0] = 1e-20
        del sigma_rossland
        to_grid = sigma_rossland_eval * prel.Rsol_cgs # should be /(sigma/Rsol_cgs) since sigma is in cgs
    # make cut in density
    cutden = Den > 1e-19
    x_cut, y_cut, z_cut, to_grid_cut, Den_cut = \
        make_slices([X, Y, Z, to_grid, Den], cutden)
    points_tree = np.array([x_cut, y_cut, z_cut]).T
    sim_tree = KDTree(points_tree)

    gridded_indexes =  np.zeros(( len(xs), len(ys), len(zs) ))
    gridded =  np.zeros(( len(xs), len(ys), len(zs) ))
    if what_to_grid == 'both':
        gridded_den =  np.zeros(( len(xs), len(ys), len(zs) ))
    for i in range(len(xs)):
        for j in range(len(ys)):
            for k in range(len(zs)):
                queried_value = [xs[i], ys[j], zs[k]]
                _, idx = sim_tree.query(queried_value)
                                    
                # Store
                gridded_indexes[i, j, k] = idx
                gridded_value = to_grid_cut[idx]
                if gridded_value <= 0: #anyway it just happens for Diss
                    gridded_value = 1e-20
                gridded[i, j, k] = gridded_value
                if what_to_grid == 'both':
                    gridded_den[i, j, k] = Den_cut[idx]
    if what_to_grid == 'both':
        return gridded_indexes, gridded_den, gridded, xs, ys, zs
    else:
        return gridded_indexes, gridded, xs, ys, zs

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
    from src import orbits as orb
    save = True
    what_to_grid = 'tau_ross' # Den or Diss or both
    if what_to_grid == 'tau_ross':
        import matlab.engine
        from src.Opacity.linextrapolator import nouveau_rich
        eng = matlab.engine.start_matlab()
        opac_path = f'{prepath}/src/Opacity'
        T_cool = np.loadtxt(f'{opac_path}/T.txt')
        Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
        rossland = np.loadtxt(f'{opac_path}/ross.txt')
        T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5)
    
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    Rt = Rstar * (Mbh/mstar)**(1/3)
    xcrt, ycrt, crt = make_cfr(Rt)
    apocenter = orb.apocentre(Rstar, mstar, Mbh, beta)
    check = ''
    compton = 'Compton'
    if m== 6:
        folder = f'R{Rstar}M{mstar}BH1e+0{m}beta{beta}S60n{n}{compton}{check}'
    else:
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

    if compute:
        snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 
        t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)

        snaps = np.array(snaps)
        idx_chosen = np.array([np.argmin(np.abs(snaps-100)),
                               np.argmin(np.abs(snaps-237)),
                               np.argmin(np.abs(snaps-348))])
        snaps, tfb = snaps[idx_chosen], tfb[idx_chosen]
        
        with open(f'{prepath}/data/{folder}/projection/{what_to_grid}time_proj.txt', 'a') as f:
            f.write(f'# snaps \n' + ' '.join(map(str, snaps)) + '\n')
            f.write(f'# t/t_fb (t_fb = {t_fall})\n' + ' '.join(map(str, tfb)) + '\n')
            f.close()
        for snap in snaps:
            print(snap)
            if alice:
                path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
            else:
                path = f'{prepath}/TDE/{folder}/{snap}'
            
            if what_to_grid == 'both':
                _, grid_den, grid_diss, x_radii, y_radii, z_radii = grid_maker(path, snap, m, mstar, Rstar, what_to_grid, x_num=800, y_num=800, z_num = 100)
                flat_diss = projector(grid_diss, x_radii, y_radii, z_radii)
                flat_den = projector(grid_den, x_radii, y_radii, z_radii)
                np.save(f'{prepath}/data/{folder}/projection/Denproj{snap}.npy', flat_den)
                np.save(f'{prepath}/data/{folder}/projection/Dissproj{snap}.npy', flat_diss)    
            else:
                _, grid_q, x_radii, y_radii, z_radii = grid_maker(path, snap, m, mstar, Rstar, what_to_grid, x_num=800, y_num=800, z_num = 100)
                flat_q = projector(grid_q, x_radii, y_radii, z_radii)
                np.save(f'{prepath}/data/{folder}/projection/{what_to_grid}proj{snap}.npy', flat_q)
                
        np.save(f'{prepath}/data/{folder}/projection/{what_to_grid}xarray.npy', x_radii)
        np.save(f'{prepath}/data/{folder}/projection/{what_to_grid}yarray.npy', y_radii)

    else:
        import src.orbits as orb
        faraway = False

        # t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)
        time = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/projection/time_proj.txt')
        snaps = [int(i) for i in time[0]]
        tfb = time[1]
        xcrt, ycrt, crt = orb.make_cfr(Rt)

        # Load and clean
        for i, snap in enumerate(snaps):
            # if int(snap)!= 348:
            #     continue
            
            if faraway:
                vmin = 5e-2
                vmax = 2e7
            else:
                flat_den = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/projection/denproj{snap}.txt')
                x_radii = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/projection/xarray.txt')
                y_radii = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/projection/yarray.txt')
                vmin = 1e3
                vmax = 7e6
            flat_den *= prel.Msol_cgs/prel.Rsol_cgs**2
            
            fig, ax = plt.subplots(1,1, figsize = (15,7))
            ax.set_xlabel(r'$X [R_\odot]$', fontsize = 20)
            ax.set_ylabel(r'$Y [R_\odot]$', fontsize = 20)
            img = ax.pcolormesh(x_radii, y_radii, flat_den.T, cmap = 'inferno',
                                norm = colors.LogNorm(vmin = vmin, vmax = vmax))
            cb = plt.colorbar(img)
            cb.set_label(r'Column density [g/cm$^2$])', fontsize = 18)
            ax.contour(xcrt, ycrt, crt, [0], linestyles = 'dashed', colors = 'w', alpha = 1)
            ax.scatter(0, 0, color = 'k', edgecolors = 'orange', s = 40)
            tfb_single = tfb[i]
            ax.text(-400, 110, f't = {np.round(tfb_single,2)}' + r't$_{\rm fb}$', color = 'white', fontsize = 18)
            # ax.set_xlim(-400, 140)
            ax.set_ylim(-142, 142) 
            plt.tight_layout()
            # if save:
            #     plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/projection/denproj{snap}.png')
            plt.show()
            plt.close()


