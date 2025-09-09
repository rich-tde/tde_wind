
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

    if how_far == 'nozzle':
        x_start = -3*Rt
        x_stop = 3*Rt
        y_start = -3*Rt 
        y_stop = 3*Rt
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
        x_start = -1.2*apo #-7*apo
        x_stop = 0.2*apo #2.5*apo
        y_start = - 0.5*apo #-4*apo 
        y_stop = 0.5*apo  #3*apo
        z_start = -2*Rt #-2*apo 
        z_stop = 2*Rt #2*apo 

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
    points_tree = np.array([x_cut, y_cut, z_cut]).T
    sim_tree = KDTree(points_tree)
    gridded_indexes =  np.zeros(( len(xs), len(ys), len(zs) ))
    gridded =  np.zeros(( len(xs), len(ys), len(zs) ))
    for i in range(len(xs)):
        for j in range(len(ys)):
            for k in range(len(zs)):
                # queried_value = [xs[i], ys[j], zs[k]] # if u use scipy.spatial
                queried_value = np.array([xs[i], ys[j], zs[k]]).reshape(1, -1) # if u use sklearn, so it has shape (1,3)
                _, idx = sim_tree.query(queried_value)
                idx = int(idx[0][0]) # only ifyou use sklearn
                                    
                # Store
                gridded_indexes[i, j, k] = idx
                gridded_value = max(1e-20, to_grid_cut[idx]) # so you don't have negative values in Diss
                gridded[i, j, k] = gridded_value
    del to_grid_cut, Den_cut, x_cut, y_cut, z_cut, points_tree 

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
    what_to_grid = 'Den'
    how_far = 'nozzle' # 'big' for big grid, '' for usual grid, 'nozzle' for nearby nozzle 
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
        if how_far == 'big':
            idx_chosen = np.array([np.argmin(np.abs(snaps-97)),
                                np.argmin(np.abs(snaps-238)),
                                np.argmin(np.abs(snaps-318))])
            snaps, tfb = snaps[idx_chosen], tfb[idx_chosen]
        
        with open(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}time_proj.txt', 'w') as f:
            f.write(f'# snaps \n' + ' '.join(map(str, snaps)) + '\n')
            f.write(f'# t/t_fb (t_fb = {t_fall})\n' + ' '.join(map(str, tfb)) + '\n')
            f.close()
            
        for snap in snaps:
            if snap != 60:
                continue
            print(snap, flush=True)
            if alice:
                path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
            else:
                path = f'{prepath}/TDE/{folder}/{snap}'
            
            _, grid_q, x_radii, y_radii, z_radii = grid_maker(path, snap, m, mstar, Rstar, what_to_grid, x_num=800, y_num=800, z_num = 100, how_far = how_far)
            flat_q = projector(grid_q, x_radii, y_radii, z_radii)
            np.save(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}proj{snap}.npy', flat_q)
                
        np.save(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}xarray.npy', x_radii)
        np.save(f'{prepath}/data/{folder}/projection/{how_far}{what_to_grid}yarray.npy', y_radii)

    else:
        import healpy as hp
        import src.orbits as orb
        from plotting.paper.IHopeIsTheLast import split_data_red
        from Utilities.operators import from_cylindric
        snap = 318
        what_to_grid = 'Den' #['tau_scatt', 'tau_ross', 'Den']
        sign = '' # '' for positive, '_neg' for negative

        snaps, Lum, tfb = split_data_red('NewAMR')
        tfb_single = tfb[np.argmin(np.abs(snap-snaps))]

        # geometric thigns
        observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
        observers_xyz = np.array(observers_xyz).T
        x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
        theta = np.arctan2(y, x)          # Azimuthal angle in radians
        phi = np.arccos(z / r)            # Elevation angle in radians
        longitude_moll = theta              
        latitude_moll = np.pi / 2 - phi 
        indecesorbital = np.concatenate(np.where(latitude_moll==0))
        first_idx, last_idx = np.min(indecesorbital), np.max(indecesorbital)
        radii_grid = [Rt/apo, a_mb/apo, 1] #*apo 
        xcfr_grid, ycfr_grid, cfr_grid = [], [], []
        for i, radius_grid in enumerate(radii_grid):
            xcr, ycr, cr = orb.make_cfr(radius_grid)
            xcfr_grid.append(xcr)
            ycfr_grid.append(ycr)
            cfr_grid.append(cr)

        theta_arr = np.arange(0, 2*np.pi, 0.01)
        r_arr_ell = orb.keplerian_orbit(theta_arr, a_mb, Rp, ecc=e_mb)
        x_arr_ell, y_arr_ell = from_cylindric(theta_arr, r_arr_ell)
        
        fig, ax = plt.subplots(1, 1, figsize = (14,7))
        flat_q = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}{what_to_grid}proj{snap}{sign}.npy')
        x_radii = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}{what_to_grid}xarray.npy')
        y_radii = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/{how_far}{what_to_grid}yarray.npy')
        dataph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}_photo{snap}.txt')
        xph, yph, zph, volph= dataph[0], dataph[1], dataph[2], dataph[3]
        # midph= np.abs(zph) < volph**(1/3)
        # xph_mid, yph_mid, zph_mid = make_slices([xph, yph, zph], midph)
       
        if how_far == 'big':
            ax.set_xlim(-6, 2.5)
            ax.set_ylim(-3, 2)
        else:
            ax.set_xlim(-1.2, 0.1)
            ax.set_ylim(-0.4, 0.4)
        if what_to_grid == 'Den':
            flat_q *= prel.Msol_cgs/prel.Rsol_cgs**2
            vmin = 1e2
            vmax = 9e7
            cbar_label = r'Column density [g/cm$^2$]'
            cmap = 'plasma'
        elif what_to_grid == 'Diss':
            flat_q *= prel.en_converter/(prel.tsol_cgs * prel.Rsol_cgs**2)
            vmin = 1e14
            vmax = 1e19
            cbar_label = r'Dissipation energy column density [erg s$^{-1}$cm$^{-2}]$'
            cmap = 'viridis'
            
        img = ax.pcolormesh(x_radii/apo, y_radii/apo, np.abs(flat_q).T, cmap = cmap,
                            norm = colors.LogNorm(vmin = vmin, vmax = vmax))
        cb = plt.colorbar(img)
        # ax.plot(xph[indecesorbital]/apo, yph[indecesorbital]/apo, c = 'white', markersize = 5, marker = 'H', label = r'$R_{\rm ph}$')
        # just to connect the first and last 
        # ax.plot([xph[first_idx]/apo, xph[last_idx]/apo], [yph[first_idx]/apo, yph[last_idx]/apo], c = 'white', markersize = 1, marker = 'H')
        cb.set_label(cbar_label)
        ax.set_xlabel(r'$X [R_{\rm a}]$', fontsize = 20)
        ax.set_ylabel(r'$Y [R_{\rm a}]$', fontsize = 20)
        ax.scatter(0, 0, color = 'k', edgecolors = 'orange', s = 40)
        # ax.plot(x_arr_ell/apo, y_arr_ell/apo, c= 'white', linestyle = 'dashed', alpha = 0.7)
        # for j in range(len(radii_grid)):
        #     ax.contour(xcfr_grid[j]/apo, ycfr_grid[j]/apo, cfr_grid[j]/apo, levels=[0], colors='white')
            
        plt.tight_layout()
        ax.set_title(f't = {np.round(tfb_single,2)}' + r't$_{\rm fb}$, res: ' + f'{check}', color = 'k', fontsize = 25)
        # plt.savefig(f'{prepath}/Figs/Test/{folder}/projection/{what_to_grid}{faraway}proj{snap}{sign}.png', dpi = 300)
        plt.show()
# %%
