""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
import sys
sys.path.append('/Users/paolamartire/shocks')
# import resource
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    abspath = '/Users/paolamartire/shocks'
    save = True
    mpl.rcParams['savefig.transparent'] = False 
    mpl.rcParams['figure.facecolor'] = 'white'    # Figure when displayed
    mpl.rcParams['axes.facecolor'] = 'white'  

import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
from sklearn.neighbors import KDTree
from src.Opacity.interpolator_vectorized import calc_planck_opacity_vectorized, calc_ross_opacity_vectorized, calc_scattering_opacity_vectorized

import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
from Utilities.operators import make_tree, sort_list

def fld_lightcurve(params, compton, check, N_ray):
    m, Rstar, mstar, beta, n, compton = params
    Mbh = 10**m
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
    pre_saving = f'{abspath}/data/{folder}'

    # observers 
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) # shape: (3, 192)
    observers_xyz = np.array(observers_xyz).T # shape: (192, 3)

    for idx_s, snap in enumerate(snaps):
        if snap not in [76]: 
            continue
        print(f'Snap: {snap}', flush=True)
        if alice:
            loadpath = f'{pre}/snap_{snap}'
        else:
            loadpath = f'{pre}/{snap}'
        colorsphere_groups, colorsphere, freqs, L_col = single_fld(loadpath, snap, observers_xyz, N_ray)

        if save:
            # Save spectrum
            # np.savez(f"{pre_saving}/spectra/mixMG/{check}_Rcolgroup_mixMG{snap}.npz", **colorsphere_groups)
            np.savez(f"{pre_saving}/spectra/mixMG/{check}_Rcol_mixMG{snap}.npz", **colorsphere)
            np.savetxt(f'{pre_saving}/spectra/mixMG/freqs_mixMG.txt', freqs)
            np.savetxt(f'{pre_saving}/spectra/mixMG/{check}_spectra_mixMG{snap}.txt', L_col)
        del colorsphere, L_col
        gc.collect()

def single_fld(loadpath, snap, observers_xyz, N_ray):
    num_obs = len(observers_xyz)
    data = make_tree(loadpath, snap)
    box = np.load(f'{loadpath}/box_{snap}.npy')
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Rad, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE
    denmask = Den > 1e-19
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        make_slices([X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den], denmask)
    xyz = np.array([X, Y, Z]).T
    R = np.sqrt(X**2 + Y**2 + Z**2)

    colorsphere = {'idx': [], 'x': [], 'y': [], 'z': [], 'vol': [], 'den': [], 'temp': [], 'radden': [], 'vx': [], 'vy': [], 'vz': [], 'P': []} # each list have length num_obs
    colorsphere_groups = {g: {
                            'idx': [], 'x': [], 'y': [], 'z': [],
                            'vol': [], 'den': [], 'temp': [], 'radden': [],
                            'vx': [], 'vy': [], 'vz': [], 'P': [],
                            'alpha_rossland': [], 'alpha_scatter': [], 'alpha_abs': [], 'alpha_eff': []}
                        for g in range(Ng)}
    freqs = prel.freqs
    L_col = np.zeros((num_obs, len(prel.freqs)))

    for i in range(num_obs):
        # if i not in [0, 10, 20, 30, 40, 70, 80, 90, 100, 110, 140, 150, 180]:
        #     continue
        print(f'Obs: {i}', flush=True)

        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        # be sure is normalized, but it should be because it's a healpy vector
        norm_mu = np.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
        if norm_mu != 1.:
            print('normalizing observers direction. norm:', norm_mu, flush=True)
            mu_x /= norm_mu
            mu_y /= norm_mu
            mu_z /= norm_mu

        if mu_x < 0:
            rmax = box[0] / mu_x
        else:
            rmax = box[3] / mu_x
        if mu_y < 0:
            rmax = min(rmax, box[1] / mu_y)
        else:
            rmax = min(rmax, box[4] / mu_y)
        if mu_z < 0:
            rmax = min(rmax, box[2] / mu_z)
        else:
            rmax = min(rmax, box[5] / mu_z)

        r = np.logspace(-0.25, np.log10(rmax), N_ray)

        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        xyz2 = np.array([x, y, z]).T
        del x, y, z
        # find the simulation cell corresponding to cells in the wanted ray
        tree = KDTree(xyz, leaf_size=50) 
        _, idx = tree.query(xyz2, k=1)
        idx = idx.ravel()
        idx = np.array(idx)
        # idx = np.unique(idx).astype(np.int64)
        # Quantity corresponding to the ray
        d = Den[idx] * prel.den_converter
        t = T[idx]
        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        ray_radDen = Rad_den[idx]
        volume = Vol[idx]
        ray_vx = VX[idx]
        ray_vy = VY[idx]
        ray_vz = VZ[idx]
        ray_press = Press[idx]
        ray_ie_den = IE_den[idx]

        # Optical depth
        r_fuT = np.flipud(r) #.T
        alpha_planck = calc_planck_opacity_vectorized(T_cool, Rho_cool, planck, np.log(t), np.log(d))
        alpha_planck = np.array(alpha_planck)
        alpha_planck_groups = np.zeros((Ng, len(r)))
        alpha_rossland_groups = np.zeros((Ng, len(r)))
        los_effective_groups = np.zeros((Ng, len(r)))
        # print(np.log(t))
        for g in range(Ng):
            rossland_kappa_g_table = np.loadtxt(f'{opac_path_mg}/sigma_rossland_{g+1}.txt').T # they are NOT in log, are in cm2/g
            planck_kappa_g_table = np.loadtxt(f'{opac_path_mg}/sigma_absorption_rossland_{g+1}.txt').T
            scattering_kappa_g_table = np.loadtxt(f'{opac_path_mg}/sigma_scattering_planck_{g+1}.txt').T
            rossland_g_table = rossland_kappa_g_table * Rho_cool_mg[:, None]
            planck_g_table = planck_kappa_g_table * Rho_cool_mg[:, None]
            scattering_g_table = scattering_kappa_g_table * Rho_cool_mg[:, None]
            Rho_cool_mg_log = np.log(Rho_cool_mg)
            T_cool_mg_log = np.log(T_cool_mg)
            rossland_g_table = np.log(rossland_g_table)
            planck_g_table = np.log(planck_g_table)
            scattering_g_table = np.log(scattering_g_table)

            alpha_rossland_g = calc_ross_opacity_vectorized(
                                T_cool_mg_log,
                                Rho_cool_mg_log,
                                rossland_g_table,
                                scattering_g_table,
                                np.log(t),
                                np.log(d))

            alpha_planck_g = calc_planck_opacity_vectorized(
                                T_cool_mg_log,
                                Rho_cool_mg_log,
                                planck_g_table,
                                np.log(t),
                                np.log(d))

            alpha_rossland_g = np.asarray(alpha_rossland_g)
            alpha_planck_g = np.asarray(alpha_planck_g)


            # underflow_mask = np.logical_and(np.log(alpha_rossland_g) != 0.0, np.log(alpha_planck_g) != 0.0)
            # alpha_rossland_g[~underflow_mask] = 0.0
            # alpha_planck_g[~underflow_mask] = 0.0

            alpha_rossland_groups[g] = alpha_rossland_g
            alpha_planck_groups[g] = alpha_planck_g

            alpha_effective_g = np.sqrt(3.0 * np.minimum(alpha_planck_g, alpha_rossland_g) * alpha_rossland_g)
            # print(f'alpha_effective_g for group {g}:', alpha_effective_g, flush=True)
            
            los_effective_groups[g] = -np.flipud(sci.cumulative_trapezoid(np.flipud(alpha_effective_g), r_fuT, initial=0)) * prel.Rsol_cgs

        los_effective_groups[los_effective_groups > 30] = 30
        

        color_idx_groups = np.full(Ng, -1, dtype=int)
        for g in range(Ng):
            idx_g = np.where(los_effective_groups[g] < 5)[0]
            if len(idx_g) == 0:
                print(f'No color index found for observer {i}, group {g}', flush=True)
                colorsphere_groups[g]['idx'].append(np.nan)
                colorsphere_groups[g]['x'].append(np.nan)
                colorsphere_groups[g]['y'].append(np.nan)
                colorsphere_groups[g]['z'].append(np.nan)
                colorsphere_groups[g]['vol'].append(np.nan)
                colorsphere_groups[g]['den'].append(np.nan)
                colorsphere_groups[g]['temp'].append(np.nan)
                colorsphere_groups[g]['radden'].append(np.nan)
                colorsphere_groups[g]['vx'].append(np.nan)
                colorsphere_groups[g]['vy'].append(np.nan)
                colorsphere_groups[g]['vz'].append(np.nan)
                colorsphere_groups[g]['P'].append(np.nan)
                colorsphere_groups[g]['alpha_rossland'].append(np.nan)
                # colorsphere_groups[g]['alpha_scatter'].append(np.nan)
                colorsphere_groups[g]['alpha_abs'].append(np.nan)
                colorsphere_groups[g]['alpha_eff'].append(np.nan)
                continue
            color_idx_groups[g] = idx_g[0]
            colorsphere_groups[g]['idx'].append(idx[idx_g])
            colorsphere_groups[g]['x'].append(ray_x[idx_g])
            colorsphere_groups[g]['y'].append(ray_y[idx_g])
            colorsphere_groups[g]['z'].append(ray_z[idx_g])
            colorsphere_groups[g]['vol'].append(volume[idx_g])
            colorsphere_groups[g]['den'].append(d[idx_g])
            colorsphere_groups[g]['temp'].append(t[idx_g])
            colorsphere_groups[g]['radden'].append(ray_radDen[idx_g])
            colorsphere_groups[g]['vx'].append(ray_vx[idx_g])
            colorsphere_groups[g]['vy'].append(ray_vy[idx_g])
            colorsphere_groups[g]['vz'].append(ray_vz[idx_g])
            colorsphere_groups[g]['P'].append(ray_press[idx_g])
            colorsphere_groups[g]['alpha_rossland'].append(alpha_rossland_groups[g][idx_g])
            # colorsphere_groups[g]['alpha_scatter'].append(scattering_g_table[idx_g])
            colorsphere_groups[g]['alpha_abs'].append(alpha_planck_groups[g][idx_g])
            colorsphere_groups[g]['alpha_eff'].append(alpha_effective_g[idx_g])

        valid = color_idx_groups >= 0
        if not np.any(valid):
            print(f'No color surface found for observer {i}', flush=True)
            continue
        r_th_groups = r[color_idx_groups]
        r_th_min = np.min(r_th_groups)
        color_idx = np.argmin(np.abs(r - r_th_min))

        colorsphere['idx'].append(idx[color_idx])
        colorsphere['x'].append(ray_x[color_idx])
        colorsphere['y'].append(ray_y[color_idx])
        colorsphere['z'].append(ray_z[color_idx])
        colorsphere['vol'].append(volume[color_idx])
        colorsphere['den'].append(d[color_idx])
        colorsphere['temp'].append(t[color_idx])
        colorsphere['radden'].append(ray_radDen[color_idx])
        colorsphere['vx'].append(ray_vx[color_idx])
        colorsphere['vy'].append(ray_vy[color_idx])
        colorsphere['vz'].append(ray_vz[color_idx])
        colorsphere['P'].append(ray_press[color_idx])

        # Spectra ---
        for g in range(Ng):
            # Frequencies belonging to this MG group
            freq_mask = ((freqs >= f_edges[g]) & (freqs < f_edges[g+1]))
            if not np.any(freq_mask):
                continue
            nu = freqs[freq_mask]

            for k in range(max(color_idx, 1), len(r)):
                dr = r[k] - r[k-1]
                Vcell = r[k]**2 * dr
                wien = np.exp(prel.h_cgs * nu / (prel.Kb_cgs * t[k])) - 1
                black_body = nu**3 / wien
                L_col[i, freq_mask] += (
                    alpha_planck_groups[g, k]
                    * Vcell
                    * np.exp(-los_effective_groups[g, k])
                    * black_body)

        del ray_radDen, tree, volume, ray_x, ray_y, ray_z, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den 
        gc.collect()

    return colorsphere_groups, colorsphere, freqs, L_col


def load_fld_data(folder, check):
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter = ",")
    snaps, tfb, luminosity = data[:, 0], data[:, 1], data[:, 2]
    snaps, luminosity, tfb = sort_list([snaps, luminosity, tfb], tfb, unique=True)
    return (np.asarray(snaps, dtype=int),
        np.asarray(tfb, dtype=float),
        np.asarray(luminosity, dtype=float))

if __name__ == "__main__":
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'HiResNewAMR' 
    N_ray = 1_000
    params = [m, Rstar, mstar, beta, n, compton]
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    Ng = 10
    
    # Load opacity tables
    opac_path = f'{abspath}/src/Opacity/fromRICH/'
    T_cool = np.loadtxt(f'{opac_path}/T.txt')
    Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
    rossland = np.loadtxt(f'{opac_path}/ross.txt')
    planck = np.loadtxt(f'{opac_path}/planck.txt')
    T_cool_plot = np.exp(T_cool)
    Rho_cool_plot = np.exp(Rho_cool)
    rossland_plot = np.exp(rossland)
    # print(np.shape(rossland), np.shape(T_cool), np.shape(Rho_cool), flush=True)

    opac_path_mg = f'{abspath}/src/Opacity/fromRICH/MG'
    T_cool_mg = np.loadtxt(f'{opac_path_mg}/T.txt')
    T_cool_mg *= prel.ev_toK
    Rho_cool_mg = np.loadtxt(f'{opac_path_mg}/rho.txt')
    rossland_opt_table = np.loadtxt(f'{opac_path_mg}/sigma_rossland_4.txt').T # they are NOT in log
    # print(np.shape(T_cool_mg), np.shape(Rho_cool_mg))
    f_edges_ev = np.loadtxt(f'{opac_path_mg}/frequency_edges.txt') # [eV] 
    f_edges = f_edges_ev * prel.ev_toHz # [Hz] 

    # kappa_ross_plot = rossland_plot
    kappa_ross_plot = []
    for i in range(len(T_cool_plot)): 
        kappa_ross_plot.append(rossland_plot[i, :]/Rho_cool_plot)
    kappa_ross_plot = np.array(kappa_ross_plot)

    # fig, (axg, axm) = plt.subplots(1, 2, figsize=(14, 6))
    # axg.pcolormesh(np.log10(T_cool_plot), np.log10(Rho_cool_plot), kappa_ross_plot.T, cmap='jet', norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1e7))
    # img = axm.pcolormesh(np.log10(T_cool_mg_plot), np.log10(Rho_cool_mg_plot), rossland_opt_table.T, cmap='jet', norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1e7))
    # cbar = plt.colorbar(img)
    # cbar.set_label(r'$\kappa_R$ (cm$^2$/g)', fontsize=18)
    # for ax in [axg, axm]:
    #     ax.set_xlabel(r'$\log_{10}$ Temperature (K)', fontsize=18)
    #     ax.set_xlim(3.5, 9.5)
    #     ax.set_ylim(-14, 2)
    # axg.set_ylabel(r'$\log_{10}$ Density (g/cm$^3$)', fontsize=18)

    fld_lightcurve(params, compton, check, N_ray)

    # colorsphere_mix = np.load(f"{abspath}/data/{folder}/spectra/{check}_Rcol_mixMG76.npz")
    # x_mix, y_mix, z_mix = colorsphere_mix['x'], colorsphere_mix['y'], colorsphere_mix['z']
    # r_mix = np.sqrt(x_mix**2 + y_mix**2 + z_mix**2)

    # colorsphere = np.load(f"{abspath}/data/{folder}/spectra/{check}_Rcol76.npz")
    # x, y, z = colorsphere['x'], colorsphere['y'], colorsphere['z']
    # r = np.sqrt(x**2 + y**2 + z**2)

    # plt.figure(figsize=(6, 6))
    # plt.scatter(np.arange(len(r))/13, r, color='orange', label='grey')
    # plt.scatter(np.arange(len(r_mix))/13, r_mix, color='blue', label='mixMG')
    # plt.xlabel('Observer index', fontsize=18)
    # plt.ylabel(r'$r_{\rm col} (r_t)$', fontsize=18)
    # plt.yscale('log')
    # plt.legend()
