
""" 
Column density/dissipation density rate.
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
    prepath = '/Users/paolamartire/shocks'
    compute = False
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from src.orbits import make_cfr

import numpy as np
import numba
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
import Utilities.prelude as prel
from src import orbits as orb

#
## FUNCTIONS
#

# def piecewise_log_grid(xmin, x_fine_min, x_fine_max, xmax,
#                        n_fine, n_log):
#     """
#     Piecewise grid with fine inner region and logarithmic outer regions.

#     Parameters
#     ----------
#     xmin, xmax : float
#         Total grid bounds
#     x_fine_min, x_fine_max : float
#         Fine region bounds
#     n_fine : int
#         Number of points in fine region
#     n_log : int
#         Number of points in EACH log tail

#     Returns
#     -------
#     grid : ndarray
#     """
#     assert xmin < x_fine_min < x_fine_max < xmax

#     # fine inner region
#     fine = np.linspace(x_fine_min, x_fine_max, n_fine)

#     # left log tail
#     left_log = np.logspace(np.log10(xmin),
#                             np.log10(x_fine_min),
#                             n_log)
#     left = -left_log

#     # right log tail
#     right = np.logspace(np.log10(x_fine_max),
#                              np.log10(xmax),
#                              n_log)

#     return np.concatenate([left, fine, right])

def piecewise_log_grid(xmin, xmax, n_log):
    """
    Piecewise grid with fine inner region and logarithmic outer regions.

    Parameters
    ----------
    xmin, xmax : float
        Total grid bounds
    x_fine_min, x_fine_max : float
        Fine region bounds
    n_fine : int
        Number of points in fine region
    n_log : int
        Number of points in EACH log tail

    Returns
    -------
    grid : ndarray
    """
    assert xmin < 0 < xmax

    left_log = np.logspace(np.log10(np.abs(xmin)),
                            np.log10(1e-2),
                            n_log)
    left = -left_log

    right = np.logspace(np.log10(1e-2),
                             np.log10(xmax),
                             n_log)

    return np.concatenate([left, right])

def grid_maker(path, snap, m, mstar, Rstar, what_to_grid, x_num, y_num, z_num = 100, how_far = ''):
    """ ALL outputs are in in solar units """
    Mbh = 10**m
    Rt = Rstar * (Mbh/mstar)**(1/3)
    # R0 = 0.6 * Rt
    apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

    if how_far == 'star_frame':
        x_start = -20
        x_stop = 20
        y_start = -20
        y_stop = 20
        z_start = -100 
        z_stop = 100 

    elif how_far == 'nozzle':
        x_start = -apo
        x_stop = 0.5*apo
        y_start = -.5*apo
        y_stop = .5*apo
        z_start = -.5*apo 
        z_stop = .5*apo 

    elif how_far == 'big':
        x_start = -6*apo
        x_stop = 2.5*apo
        y_start = -4*apo 
        y_stop = 3*apo
        z_start = -2*apo 
        z_stop = 2*apo 

    else:
        x_start = -3*apo #-7*apo
        x_stop = 2*apo #2.5*apo
        y_start = -2*apo #-4*apo 
        y_stop = 2*apo  #3*apo
        z_start = -0.8*Rt #-2*apo 
        z_stop = 0.8*Rt #2*apo  
    
    xs = piecewise_log_grid(x_start, x_stop, n_log = x_num)
    ys = piecewise_log_grid(y_start, y_stop, n_log = y_num)
    zs = piecewise_log_grid(z_start, z_stop, n_log = z_num)
    # print(zs)
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
    if what_to_grid == 'tau_scatt':
        kappa_scatt = 0.34 / (prel.Rsol_cgs**2/prel.Msol_cgs) # it's 0.34cm^2/g, you want it in code units
        to_grid = kappa_scatt * Den
    if what_to_grid == 'tau_ross':
        import matlab.engine
        eng = matlab.engine.start_matlab()
        opac_path = f'{prepath}/src/Opacity'
        T_cool = np.loadtxt(f'{opac_path}/T.txt')
        Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
        rossland = np.loadtxt(f'{opac_path}/ross.txt')
        if check in ['LowRes', '', 'HiRes']:
            from src.Opacity.linextrapolator import first_rich_extrap
            T_cool2, Rho_cool2, rossland2 = first_rich_extrap(T_cool, Rho_cool, rossland, what = 'scattering_limit', slope_length = 5, highT_slope=-3.5)
        if check in ['QuadraticOpacity', 'QuadraticOpacityNewAMR']:
            from src.Opacity.linextrapolator import linear_rich
            T_cool2, Rho_cool2, rossland2 = linear_rich(T_cool, Rho_cool, rossland, what = 'scattering_limit', highT_slope = 0)
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
    del X, Y, Z, to_grid, Den
    # make the tree
    points_tree = np.stack([x_cut, y_cut, z_cut], axis=-1)   # join xyz grid to a (xnum, ynum, znum, 3) array
    sim_tree = KDTree(points_tree)
    
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')  # shapes (Nx, Ny, Nz)
    query_points = np.stack([Xg, Yg, Zg], axis=-1).reshape(-1, 3)  # shape (Nx*Ny*Nz, 3)

    _, gridded_indexes = sim_tree.query(query_points)
    gridded_indexes = gridded_indexes.ravel()
    gridded = to_grid_cut[gridded_indexes]
    gridded = np.maximum(gridded, 1e-20)
    gridded_indexes = gridded_indexes.reshape(len(xs), len(ys), len(zs))
    gridded = gridded.reshape(len(xs), len(ys), len(zs))

    del to_grid_cut, Den_cut, x_cut, y_cut, z_cut, points_tree 

    return gridded_indexes, gridded, xs, ys, zs

@numba.njit
def projector(gridded_den, z_radii):
    """ Project density on XY plane. NB: to plot you have to transpose the saved data.
    z_radii has to be linspaced. """
    dz = np.diff(z_radii)
    # if np.round(dz, 4) == np.round(z_radii[-1] - z_radii[-2], 4):
    flat_den = np.sum(gridded_den[:,:,:-1], axis = -1) * dz
    # else:
    #     print(dz, z_radii[-1] - z_radii[-2])
    #     raise ValueError("z_radii has to be linspaced.")
    return flat_den

#%%
if __name__ == '__main__':    
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'

    params = [Mbh, Rstar, mstar, beta]
    things = orb.get_things_about(params)
    Rs = things['Rs']
    Rt = things['Rt']
    Rp = things['Rp']
    R0 = things['R0']
    apo = things['apo']
    a_mb = things['a_mb']
    e_mb = things['ecc_mb']

    if compute:
        check = 'HiResNewAMR'
        print(f'We are in {check}', flush=True)
        how_far = 'nozzle' # 'big' for big grid, '' for usual grid, 'nozzle' for nearby nozzle 
        what_to_grid = 'Diss'

        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 
        t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)

        snaps = np.array(snaps)
        if how_far == 'big' or how_far == 'nozzle':
            idx_chosen = np.array([0,
                                np.argmin(np.abs(tfb-1)),
                                np.argmax(tfb)])
            # idx_chosen = np.array([np.argmin(np.abs(tfb-0.1)),
            #                     np.argmin(np.abs(tfb-0.2)),
            #                     np.argmin(np.abs(tfb-0.32))])
            snaps, tfb = snaps[idx_chosen], tfb[idx_chosen]
        
        with open(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}time_projlog.txt', 'w') as f:
            f.write(f'# snaps \n' + ' '.join(map(str, snaps)) + '\n')
            f.write(f'# t/t_fb (t_fb = {t_fall})\n' + ' '.join(map(str, tfb)) + '\n')
            f.close()
            
        for snap in snaps:
            # if snap not in [35, 40, 48]:
            #     continue
            print(snap, flush=True)
            path = select_prefix(m, check, mstar, Rstar, beta, n, compton)
            if alice:
                path = f'{path}/snap_{snap}'
            else:
                path = f'{path}/{snap}'
            
            _, grid_q, x_radii, y_radii, z_radii = grid_maker(path, snap, m, mstar, Rstar, what_to_grid, x_num=400, y_num=300, z_num = 400, how_far = how_far)
            flat_q = projector(grid_q, z_radii)
            np.save(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}projlog{snap}.npy', flat_q)
                
        np.save(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}xarraylog.npy', x_radii)
        np.save(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}yarraylog.npy', y_radii)

    else:
        import healpy as hp
        import src.orbits as orb
        snap = 28
        # what_to_grid = 'Diss' #['tau_scatt', 'tau_ross', 'Den']
        sign = '' # '' for positive, '_neg' for negative
        how_far = ''
        check = 'HiResNewAMR'
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

        snaps, tfb = np.loadtxt(f'{prepath}/data/{folder}/projection/{how_far}Dentime_proj.txt')
        tfb_single = tfb[np.argmin(np.abs(snap-snaps))] 
        
        vmin_den = 1e-4
        vmax_den = 1e3
        vmin_diss = 1e10 #1e14
        vmax_diss = 1e18
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18,7))
        Den_flat = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}Denproj{snap}{sign}.npy')
        x_radii_den = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}Denxarray.npy')
        y_radii_den = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}Denyarray.npy')
        Den_flat *= prel.Msol_cgs/prel.Rsol_cgs**2
        Diss_flat = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}Dissproj{snap}{sign}.npy')
        x_radii_diss = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}Dissxarray.npy')
        y_radii_diss = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}Dissyarray.npy')
        Diss_flat *= prel.en_converter/(prel.tsol_cgs * prel.Rsol_cgs**2)
        img = ax1.pcolormesh(x_radii_den/Rt, y_radii_den/Rt, np.abs(Den_flat).T, cmap = 'plasma',
                            norm = colors.LogNorm(vmin = vmin_den, vmax = vmax_den))
        cb = plt.colorbar(img)
        cb.set_label(r'Column density [g/cm$^2$]')
        img = ax2.pcolormesh(x_radii_diss/Rt, y_radii_diss/Rt, np.abs(Diss_flat).T, cmap = 'viridis',
                            norm = colors.LogNorm(vmin = vmin_diss, vmax = vmax_diss))
        cb = plt.colorbar(img)
        cb.set_label(r'Dissipation energy column density [erg s$^{-1}$cm$^{-2}]$')

        ax1.set_ylabel(r'$Y [R_{\rm t}]$', fontsize = 20)
        for ax in [ax1, ax2]:
            ax.set_xlim(-40, 20)
            ax.set_ylim(-12, 12)
            ax.set_xlabel(r'$X [R_{\rm t}]$', fontsize = 20)
       
        plt.suptitle(f't = {np.round(tfb_single,2)}' + r't$_{\rm fb}$, res: ' + f'{check}', color = 'k', fontsize = 25)
        plt.tight_layout()
        plt.savefig(f'{prepath}/Figs/{folder}/projection/DenDiss{how_far}proj{snap}.png', dpi = 300)
        plt.show()
# %%
