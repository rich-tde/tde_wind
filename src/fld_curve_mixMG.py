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
import csv

import numpy as np
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
from sklearn.neighbors import KDTree
from src.Opacity.interpolator_vectorized import calc_planck_opacity_vectorized, calc_ross_opacity_vectorized, calc_scattering_opacity_vectorized

from scipy.ndimage import uniform_filter1d
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
        if snap not in [109]: 
            continue
        print(f'Snap: {snap}', flush=True)
        if alice:
            loadpath = f'{pre}/snap_{snap}'
        else:
            loadpath = f'{pre}/{snap}'
        colorsphere, freqs, L_col = single_fld(loadpath, snap, observers_xyz, N_ray)

        if save:
            # Save spectrum
            np.savez(f"{pre_saving}/spectra/{check}_Rcol_mixMG{snap}.npz", **colorsphere)
            np.savetxt(f'{pre_saving}/spectra/freqs_mixMG.txt', freqs)
            np.savetxt(f'{pre_saving}/spectra/{check}_spectra_migMG{snap}.txt', L_col)
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

    colorsphere = {'idx': [], 'x': [], 'y': [], 'z': [], 'vol': [], 'den': [], 'temp': [], 'radden': [], 'vx': [], 'vy': [], 'vz': [], 'P': []}
    freqs = prel.freqs
    L_col = np.zeros((num_obs, len(prel.freqs)))

    for i in range(num_obs):
        if i not in [0]:
            continue
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
        # idx = np.unique(idx).astype(np.int64) #####
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
        alpha_planck_groups = np.zeros((Ng, len(r)))
        alpha_rossland_groups = np.zeros((Ng, len(r)))
        los_effective_groups = np.zeros((Ng, len(r)))
        for g in range(Ng):
            rossland_g_table = np.loadtxt(f'{opac_path_mg}/sigma_rossland_{g+1}.txt').T # they are NOT in log
            planck_g_table = np.loadtxt(f'{opac_path_mg}/sigma_absorption_rossland_{g+1}.txt').T
            rossland_g_table = np.log(rossland_g_table)
            planck_g_table = np.log(planck_g_table)

            alpha_rossland_g = calc_ross_opacity_vectorized(
                                T_cool_mg,
                                Rho_cool_mg,
                                rossland_g_table,
                                scattering,
                                np.log(t),
                                np.log(d))

            alpha_planck_g = calc_planck_opacity_vectorized(
                                T_cool_mg,
                                Rho_cool_mg,
                                planck_g_table,
                                np.log(t),
                                np.log(d))

            alpha_rossland_g = np.asarray(alpha_rossland_g)
            alpha_planck_g = np.asarray(alpha_planck_g)

            alpha_rossland_groups[g] = alpha_rossland_g
            alpha_planck_groups[g] = alpha_planck_g

            alpha_effective_g = np.sqrt(
                3.0
                * np.minimum(alpha_planck_g, alpha_rossland_g)
                * alpha_rossland_g)
            
            los_effective_groups[g] = -np.flipud(sci.cumulative_trapezoid( np.flipud(alpha_effective_g), r_fuT, initial=0)) * prel.Rsol_cgs

        los_effective_groups[los_effective_groups > 30] = 30
        

        color_idx_groups = np.full(Ng, -1, dtype=int)
        for g in range(Ng):
            idx_g = np.where(los_effective_groups[g] < 5)[0]
            if len(idx_g) == 0:
                print(f'No color index found for observer {i}, group {g}', flush=True)
                continue
            color_idx_groups[g] = idx_g[0]

        valid = color_idx_groups >= 0
        if not np.any(valid):
            print(f'No color surface found for observer {i}', flush=True)
            continue
        r_th_groups = r[color_idx_groups[valid]]
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
            freq_mask = ((freqs >= f_edges[g, 0]) & (freqs < f_edges[g, 1]))
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


    return colorsphere, freqs, L_col


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
    scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm

    opac_path_mg = f'{abspath}/src/Opacity/fromRICH/MG'
    T_cool_mg = np.loadtxt(f'{opac_path_mg}/T.txt')
    Rho_cool_mg = np.loadtxt(f'{opac_path_mg}/rho.txt')
    T_cool_mg = np.log(T_cool_mg)
    Rho_cool_mg = np.log(Rho_cool_mg)
    f_edges_ev = np.loadtxt(f'{opac_path_mg}/frequency_edges.txt') # [Hz] frequency range for each group. shape: (10, 2)
    f_edges = f_edges_ev * prel.ev_toHz # [Hz] frequency range for each group. shape: (10, 2)

    fld_lightcurve(params, compton, check, N_ray)