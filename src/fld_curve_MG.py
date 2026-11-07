""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
import sys

sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    abspath = '/Users/paolamartire/shocks'
    save = False
    mpl.rcParams['savefig.transparent'] = False 
    mpl.rcParams['figure.facecolor'] = 'white'    # Figure when displayed
    mpl.rcParams['axes.facecolor'] = 'white'  
import numpy as np
import csv
import healpy as hp
import scipy.integrate as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
mpl.rcParams['savefig.transparent'] = False 
mpl.rcParams['figure.facecolor'] = 'white'    # Figure when displayed
mpl.rcParams['axes.facecolor'] = 'white'  
from src.Opacity.interpolator_vectorized import calc_planck_opacity_vectorized, calc_ross_opacity_vectorized, calc_scattering_opacity_vectorized
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
from Utilities.operators import make_tree, sort_list
import gc
import warnings
warnings.filterwarnings('ignore')

compute = True
save = True

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
        if snap not in [420]:#, 109, 151]:
            continue
        print(f'Snap: {snap}', flush=True)
        if alice:
            loadpath = f'{pre}/snap_{snap}'
        else:
            loadpath = f'{pre}/{snap}'
        photo_groups, colorsphere_groups = single_fld(loadpath, snap, observers_xyz, N_ray)
        if save:
            np.save(f"{pre_saving}/photo/{check}_photo{snap}", photo_groups, allow_pickle=True)
            np.save(f"{pre_saving}/spectra/{check}_Rcol{snap}", colorsphere_groups, allow_pickle=True)
        del photo_groups, colorsphere_groups
        gc.collect()

def single_fld(loadpath, snap, observers_xyz, N_ray):
    num_obs = len(observers_xyz)
    data = make_tree(loadpath, snap, MG=True)
    box = np.load(f'{loadpath}/box_{snap}.npy')
    X, Y, Z, T, Den, Vol, VX, VY, VZ, Press, IE_den, Eg0, Eg1, Eg2, Eg3, Eg4, Eg5, Eg6, Eg7, Eg8, Eg9 = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE,\
        data.Eg0, data.Eg1, data.Eg2, data.Eg3, data.Eg4, data.Eg5, data.Eg6, data.Eg7, data.Eg8, data.Eg9
    denmask = Den > 1e-19
    X, Y, Z, T, Den, Vol, VX, VY, VZ, Press, IE_den, Eg0, Eg1, Eg2, Eg3, Eg4, Eg5, Eg6, Eg7, Eg8, Eg9 = \
        make_slices([X, Y, Z, T, Den, Vol, VX, VY, VZ, Press, IE_den, Eg0, Eg1, Eg2, Eg3, Eg4, Eg5, Eg6, Eg7, Eg8, Eg9], denmask)
    xyz = np.array([X, Y, Z]).T

    photo_groups = {g: {
                        'idx': [], 'x': [], 'y': [], 'z': [],
                        'vol': [], 'den': [], 'temp': [], f'Eg{g}': [],
                        'vx': [], 'vy': [], 'vz': [], 'P': [],
                        'alpha_rossland': [], 'alpha_scatter': [], 'alpha_abs': [],
                        'tau_rossland': [], 'tau_scatter': []
                        }
                    for g in range(Ng)}
    colorsphere_groups = {g: {
                            'idx': [], 'x': [], 'y': [], 'z': [],
                            'vol': [], 'den': [], 'temp': [], f'Eg{g}': [],
                            'vx': [], 'vy': [], 'vz': [], 'P': [],
                            'alpha_rossland': [], 'alpha_scatter': [], 'alpha_abs': [], 'alpha_eff': [],
                            'tau_rossland': [], 'tau_scatter': [], 'tau_eff': []
                            }
                        for g in range(Ng)}
 
    for i in range(num_obs):
        # if i not in [0]:
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
        volume = Vol[idx]
        ray_vx = VX[idx]
        ray_vy = VY[idx]
        ray_vz = VZ[idx]
        ray_press = Press[idx]
        ray_ie_den = IE_den[idx]
        Egs = [Eg0[idx], Eg1[idx], Eg2[idx], Eg3[idx], Eg4[idx], Eg5[idx], Eg6[idx], Eg7[idx], Eg8[idx], Eg9[idx]] # L = 4πcRphoto²Eg

        # Optical depth
        r_fuT = np.flipud(r) #.T
        alpha_rossland_groups = np.zeros((Ng, len(r)))
        alpha_scatter_groups = np.zeros((Ng, len(r)))
        alpha_planck_groups = np.zeros((Ng, len(r)))
        los_groups = np.zeros((Ng, len(r)))
        los_effective_groups = np.zeros((Ng, len(r)))
        los_scatter_g = np.zeros((Ng, len(r)))
        # print(np.log(t))
        for g in range(Ng):
            rossland_kappa_g_table = np.loadtxt(f'{opac_path_mg}/sigma_rossland_{g+1}.txt') # they are NOT in log, are in cm2/g
            planck_kappa_g_table = np.loadtxt(f'{opac_path_mg}/sigma_absorption_rossland_{g+1}.txt')
            scattering_kappa_g_table = np.loadtxt(f'{opac_path_mg}/sigma_scattering_planck_{g+1}.txt')
            # print(np.shape(rossland_kappa_g_table), np.shape(planck_kappa_g_table), np.shape(scattering_kappa_g_table), np.shape(Rho_cool_mg))
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

            alpha_scatter_g = calc_scattering_opacity_vectorized(
                                T_cool_mg_log,
                                Rho_cool_mg_log,
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
            alpha_scatter_g = np.asarray(alpha_scatter_g)
            alpha_planck_g = np.asarray(alpha_planck_g)

            # underflow_mask = np.logical_and(np.log(alpha_rossland_g) != 0.0, np.log(alpha_planck_g) != 0.0)
            # alpha_rossland_g[~underflow_mask] = 0.0
            # alpha_planck_g[~underflow_mask] = 0.0

            alpha_rossland_groups[g] = alpha_rossland_g
            alpha_scatter_groups[g] = alpha_scatter_g 
            alpha_planck_groups[g] = alpha_planck_g

            alpha_effective_g = np.sqrt(3.0 * np.minimum(alpha_planck_g, alpha_rossland_g) * alpha_rossland_g)
            # print(f'alpha_effective_g for group {g}:', alpha_effective_g, flush=True)
            
            los_groups[g] = - np.flipud(sci.cumulative_trapezoid(np.flipud(alpha_rossland_g), r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
            los_scatter_g[g] = - np.flipud(sci.cumulative_trapezoid(np.flipud(alpha_scatter_g), r_fuT, initial = 0)) * prel.Rsol_cgs
            los_effective_groups[g] = -np.flipud(sci.cumulative_trapezoid(np.flipud(alpha_effective_g), r_fuT, initial=0)) * prel.Rsol_cgs
        
        los_effective_groups[los_effective_groups > 30] = 30  
        photo_idx_groups = np.full(Ng, -1, dtype=int)
        color_idx_groups = np.full(Ng, -1, dtype=int)
        for g in range(Ng):
            Egg = Egs[g]    
            idx_p = np.where(los_groups[g] < 2/3)[0]
            idx_g = np.where(los_effective_groups[g] < 5)[0]
            if len(idx_p) == 0:
                print(f'No photo index found for observer {i}, group {g}', flush=True)
                photo_groups[g]['idx'].append(np.nan)
                photo_groups[g]['x'].append(np.nan)
                photo_groups[g]['y'].append(np.nan)
                photo_groups[g]['z'].append(np.nan)
                photo_groups[g]['vol'].append(np.nan)
                photo_groups[g]['den'].append(np.nan)
                photo_groups[g]['temp'].append(np.nan)
                photo_groups[g][f'Eg{g}'].append(np.nan)
                photo_groups[g]['vx'].append(np.nan)
                photo_groups[g]['vy'].append(np.nan)
                photo_groups[g]['vz'].append(np.nan)
                photo_groups[g]['P'].append(np.nan)
                photo_groups[g]['tau_rossland'].append(np.nan)
                photo_groups[g]['tau_scatter'].append(np.nan)
                photo_groups[g]['tau_abs'].append(np.nan)
                photo_groups[g]['alpha_rossland'].append(np.nan)
                photo_groups[g]['alpha_scatter'].append(np.nan)
                photo_groups[g]['alpha_abs'].append(np.nan)
            elif len(idx_p) != 0:
                idx_p = idx_p[0]
                photo_idx_groups[g] = idx_p
                photo_groups[g]['idx'].append(int(idx[idx_p]))
                photo_groups[g]['x'].append(float(ray_x[idx_p]))
                photo_groups[g]['y'].append(float(ray_y[idx_p]))
                photo_groups[g]['z'].append(float(ray_z[idx_p]))
                photo_groups[g]['vol'].append(float(volume[idx_p]))
                photo_groups[g]['den'].append(float(d[idx_p]))
                photo_groups[g]['temp'].append(float(t[idx_p]))
                photo_groups[g][f'Eg{g}'].append(float(Egg[idx_p]))
                photo_groups[g]['vx'].append(float(ray_vx[idx_p]))
                photo_groups[g]['vy'].append(float(ray_vy[idx_p]))
                photo_groups[g]['vz'].append(float(ray_vz[idx_p]))
                photo_groups[g]['P'].append(float(ray_press[idx_p]))
                photo_groups[g]['alpha_rossland'].append(float(alpha_rossland_groups[g][idx_p]))
                photo_groups[g]['alpha_scatter'].append(float(alpha_scatter_groups[g][idx_p]))
                photo_groups[g]['alpha_abs'].append(float(alpha_planck_groups[g][idx_p]))
                photo_groups[g]['tau_rossland'].append(float(los_groups[g][idx_p]))
                photo_groups[g]['tau_scatter'].append(float(los_scatter_g[g][idx_p]))

            if len(idx_g) == 0:
                print(f'No color index found for observer {i}, group {g}', flush=True)
                colorsphere_groups[g]['idx'].append(np.nan)
                colorsphere_groups[g]['x'].append(np.nan)
                colorsphere_groups[g]['y'].append(np.nan)
                colorsphere_groups[g]['z'].append(np.nan)
                colorsphere_groups[g]['vol'].append(np.nan)
                colorsphere_groups[g]['den'].append(np.nan)
                colorsphere_groups[g]['temp'].append(np.nan)
                colorsphere_groups[g][f'Eg{g}'].append(np.nan)
                colorsphere_groups[g]['vx'].append(np.nan)
                colorsphere_groups[g]['vy'].append(np.nan)
                colorsphere_groups[g]['vz'].append(np.nan)
                colorsphere_groups[g]['P'].append(np.nan)
                colorsphere_groups[g]['alpha_rossland'].append(np.nan)
                colorsphere_groups[g]['alpha_scatter'].append(np.nan)
                colorsphere_groups[g]['alpha_abs'].append(np.nan)
                colorsphere_groups[g]['alpha_eff'].append(np.nan)
                colorsphere_groups[g]['tau_rossland'].append(np.nan)
                colorsphere_groups[g]['tau_scatter'].append(np.nan)
                colorsphere_groups[g]['tau_abs'].append(np.nan)
            elif len(idx_g) != 0:
                idx_g = idx_g[0]
                color_idx_groups[g] = idx_g
                colorsphere_groups[g]['idx'].append(int(idx[idx_g]))
                colorsphere_groups[g]['x'].append(float(ray_x[idx_g]))
                colorsphere_groups[g]['y'].append(float(ray_y[idx_g]))
                colorsphere_groups[g]['z'].append(float(ray_z[idx_g]))
                colorsphere_groups[g]['vol'].append(float(volume[idx_g]))
                colorsphere_groups[g]['den'].append(float(d[idx_g]))
                colorsphere_groups[g]['temp'].append(float(t[idx_g]))
                colorsphere_groups[g][f'Eg{g}'].append(float(Egg[idx_g]))
                colorsphere_groups[g]['vx'].append(float(ray_vx[idx_g]))
                colorsphere_groups[g]['vy'].append(float(ray_vy[idx_g]))
                colorsphere_groups[g]['vz'].append(float(ray_vz[idx_g]))
                colorsphere_groups[g]['P'].append(float(ray_press[idx_g]))
                colorsphere_groups[g]['alpha_rossland'].append(float(alpha_rossland_groups[g][idx_g]))
                colorsphere_groups[g]['alpha_scatter'].append(float(alpha_scatter_groups[g][idx_g]))
                colorsphere_groups[g]['alpha_abs'].append(float(alpha_planck_groups[g][idx_g]))
                colorsphere_groups[g]['alpha_eff'].append(float(alpha_effective_g[idx_g]))
                colorsphere_groups[g]['tau_rossland'].append(float(los_groups[g][idx_g]))
                colorsphere_groups[g]['tau_scatter'].append(float(los_scatter_g[g][idx_g]))
                colorsphere_groups[g]['tau_eff'].append(float(los_effective_groups[g][idx_g]))
        

        del Egg, tree, volume, ray_x, ray_y, ray_z, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den 
        gc.collect()
    return photo_groups, colorsphere_groups


def load_fld_data(folder, check):
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter = ",")
    snaps, tfb, luminosity = data[:, 0], data[:, 1], data[:, 2]
    snaps, luminosity, tfb = sort_list([snaps, luminosity, tfb], tfb, unique=True)
    return (np.asarray(snaps, dtype=int),
        np.asarray(tfb, dtype=float),
        np.asarray(luminosity, dtype=float))

if __name__ == "__main__":
    import plotting.paperWind.spectra_lc as sp
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'MG' 
    N_ray = 1_000
    params = [m, Rstar, mstar, beta, n, compton]
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    Ng = 10
    
    # Load opacity tables
    opac_path_mg = f'{abspath}/src/Opacity/fromRICH/MG'
    T_cool_mg = np.loadtxt(f'{opac_path_mg}/T.txt')
    T_cool_mg *= prel.ev_toK
    Rho_cool_mg = np.loadtxt(f'{opac_path_mg}/rho.txt')
    # print(np.shape(T_cool_mg), np.shape(Rho_cool_mg))
    f_edges_ev = np.loadtxt(f'{opac_path_mg}/frequency_edges.txt') # [eV] 
    f_edges = f_edges_ev * prel.ev_toHz # [Hz] 

    if compute:
        fld_lightcurve(params, compton, check, N_ray)

    else:
        snap = 320
        g = 8
        photo_data = np.load(f"{abspath}/data/{folder}/photo/{check}_photo{snap}.npy", allow_pickle=True).item()
        colorsphere_data = np.load(f"{abspath}/data/{folder}/spectra/{check}_Rcol{snap}.npy", allow_pickle=True).item()
        r_ph_x = photo_data[g]['x'][0]
        r_ph_y = photo_data[g]['y'][0]
        r_ph_z = photo_data[g]['z'][0]
        r_ph = np.sqrt(r_ph_x**2 + r_ph_y**2 + r_ph_z**2)
        Egg = colorsphere_data[g][f'Eg{g}'][0]
        Lum = 4 * np.pi * (r_ph*prel.Rsol_cgs)**2 * prel.c_cgs * Egg * prel.en_den_converter
        # photo_from_RICH = np.loadtxt(f"{abspath}/data/{folder}/snap_{snap}/Rphoto_snap_{snap}.txt")

    