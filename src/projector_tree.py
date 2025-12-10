
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
    prepath = '/Users/paolamartire/shocks/'
    compute = False
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from src.orbits import make_cfr

import numpy as np
import numba
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices
import Utilities.prelude as prel
from src import orbits as orb

#
## FUNCTIONS
#

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

    if how_far == 'nozzle':
        x_start = -2*apo
        x_stop = apo
        y_start = -.5*apo
        y_stop = .5*apo
        z_start = -2*apo 
        z_stop = 2*apo 

    if how_far == 'big':
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
    
    xs = np.linspace(x_start, x_stop, num = x_num)
    ys = np.linspace(y_start, y_stop, num = y_num)
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
def projector(gridded_den, x_radii, y_radii, z_radii):
    """ Project density on XY plane. NB: to plot you have to transpose the saved data.
    z_radii has to be linspaced. """
    # Make the 3D grid 
    dz = z_radii[1] - z_radii[0]  # constant spacing
    if dz == z_radii[-1] - z_radii[-2]:
        flat_den = np.sum(gridded_den, axis = -1) * dz
    else:
        raise ValueError("z_radii has to be linspaced.")
    return flat_den

#%%
if __name__ == '__main__':    
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    check = 'HiResNewAMR'
    compton = 'Compton'
    what_to_grid = 'Diss'
    how_far = 'star_frame' # 'big' for big grid, '' for usual grid, 'nozzle' for nearby nozzle 
    save_fig = False

    params = [Mbh, Rstar, mstar, beta]
    things = orb.get_things_about(params)
    Rs = things['Rs']
    Rt = things['Rt']
    Rp = things['Rp']
    R0 = things['R0']
    apo = things['apo']
    a_mb = things['a_mb']
    e_mb = things['ecc_mb']

    if check == '':
        folder = f'opacity_tests/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    else:
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

    if compute:
        snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 
        t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)

        snaps = np.array(snaps)
        if how_far == 'big' or how_far == 'nozzle':
            idx_chosen = np.array([0,
                                np.argmin(np.abs(tfb-1)),
                                np.argmax(tfb)])
            snaps, tfb = snaps[idx_chosen], tfb[idx_chosen]
        
        with open(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}time_proj.txt', 'w') as f:
            f.write(f'# snaps \n' + ' '.join(map(str, snaps)) + '\n')
            f.write(f'# t/t_fb (t_fb = {t_fall})\n' + ' '.join(map(str, tfb)) + '\n')
            f.close()
            
        for snap in snaps:
            # if snap != 150:
            #     continue
            print(snap, flush=True)
            if alice:
                path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
            else: 
                path = f'{prepath}/TDE/{folder}/{snap}'
            
            _, grid_q, x_radii, y_radii, z_radii = grid_maker(path, snap, m, mstar, Rstar, what_to_grid, x_num=500, y_num=500, z_num = 100, how_far = how_far)
            flat_q = projector(grid_q, x_radii, y_radii, z_radii)
            np.save(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}proj{snap}.npy', flat_q)
                
        np.save(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}xarray.npy', x_radii)
        np.save(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}yarray.npy', y_radii)

    else:
        import healpy as hp
        import src.orbits as orb
        from plotting.paper.IHopeIsTheLast import split_data_red
        from Utilities.operators import from_cylindric
        snap = 10
        what_to_grid = 'Diss' #['tau_scatt', 'tau_ross', 'Den']
        sign = '' # '' for positive, '_neg' for negative
        how_far = 'star_frame'

        snaps, Lum, tfb = split_data_red(check)
        tfb_single = tfb[np.argmin(np.abs(snap-snaps))] 
        
        fig, ax = plt.subplots(1, 1, figsize = (10,7))
        flat_q = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}{what_to_grid}proj{snap}{sign}.npy')
        x_radii = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}{what_to_grid}xarray.npy')
        y_radii = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}{what_to_grid}yarray.npy')
        print(x_radii)
        # dataph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}_photo{snap}.txt')
        # xph, yph, zph, volph= dataph[0], dataph[1], dataph[2], dataph[3]
        # midph= np.abs(zph) < volph**(1/3)
        # xph_mid, yph_mid, zph_mid = make_slices([xph, yph, zph], midph)
        
        if how_far == 'big':
            ax.set_xlim(-6, 2.5)
            ax.set_ylim(-3, 2)
        # elif how_far == '':
        #     ax.set_xlim(-1.2, 0.1)
        #     ax.set_ylim(-0.4, 0.4)
        if what_to_grid == 'Den':
            flat_q *= prel.Msol_cgs/prel.Rsol_cgs**2
            vmin = 1
            vmax = 9e7
            cbar_label = r'Column density [g/cm$^2$]'
            cmap = 'plasma'
        elif what_to_grid == 'Diss':
            flat_q *= prel.en_converter/(prel.tsol_cgs * prel.Rsol_cgs**2)
            vmin = 1e14
            vmax = 1e19
            cbar_label = r'Dissipation energy column density [erg s$^{-1}$cm$^{-2}]$'
            cmap = 'viridis'
            
        img = ax.pcolormesh(x_radii, y_radii, np.abs(flat_q).T, cmap = cmap,
                            norm = colors.LogNorm(vmin = vmin, vmax = vmax))
        cb = plt.colorbar(img)
        # ax.plot(xph[indecesorbital]/apo, yph[indecesorbital]/apo, c = 'white', markersize = 5, marker = 'H', label = r'$R_{\rm ph}$')
        # just to connect the first and last 
        # ax.plot([xph[first_idx]/apo, xph[last_idx]/apo], [yph[first_idx]/apo, yph[last_idx]/apo], c = 'white', markersize = 1, marker = 'H')
        cb.set_label(cbar_label)
        ax.set_xlabel(r'$X [R_\odot]$', fontsize = 20)
        ax.set_ylabel(r'$Y [R_\odot]$', fontsize = 20)
        # ax.set_xlabel(r'$X [R_{\rm a}]$', fontsize = 20)
        # ax.set_ylabel(r'$Y [R_{\rm a}]$', fontsize = 20)
        ax.scatter(0, 0, color = 'k', edgecolors = 'orange', s = 40)
        # ax.plot(x_arr_ell/apo, y_arr_ell/apo, c= 'white', linestyle = 'dashed', alpha = 0.7)
        # for j in range(len(radii_grid)):
        #     ax.contour(xcfr_grid[j]/apo, ycfr_grid[j]/apo, cfr_grid[j]/apo, levels=[0], colors='white')
            
        plt.tight_layout()
        # ax.set_title(f't = {np.round(tfb_single,2)}' + r't$_{\rm fb}$, res: ' + f'{check}', color = 'k', fontsize = 25)
        # plt.savefig(f'{prepath}/Figs/Test/{folder}/projection/{what_to_grid}{faraway}proj{snap}{sign}.png', dpi = 300)
        plt.show()
# %%
