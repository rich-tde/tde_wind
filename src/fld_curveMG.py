""" FLD curve for the multigroup.
For opacities (which have the shape (T_index, rho_index)), from Elad:
- sigma_rossland_  is the rossland mean opacity (absorption+scattering).
Use this for photosphere.
- sigma_absorption_rossland_ is the absorption opacity that was calculated 
by collapsing the spectrum into a gray opacity using rossland weighting """
import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    abspath = '/Users/paolamartire/shocks'
    save = True
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

import gc
import warnings
warnings.filterwarnings('ignore')
import csv

import numpy as np
import healpy as hp
import scipy.integrate as sci
from sklearn.neighbors import KDTree
from src.Opacity.interpolator_vectorized import calc_planck_opacity_vectorized, calc_ross_opacity_vectorized, calc_scattering_opacity_vectorized
from scipy.ndimage import uniform_filter1d

import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
import src.orbits as orb
from Utilities.operators import make_tree

#%% Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'MG' 
N_ray = 5_000

## Snapshots stuff
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
print('we are in: ', pre, flush=True)

#%% Opacities: load and interpolate ----------------------------------------------------------------
opac_path = f'{abspath}/src/Opacity/fromRICH/MG'
T_cool_lin = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool_lin = np.loadtxt(f'{opac_path}/rho.txt')
T_cool = np.log(T_cool_lin)
Rho_cool = np.log(Rho_cool_lin)
f_edges_ev = np.loadtxt(f'{opac_path}/frequency_edges.txt') # [Hz] frequency range for each group. shape: (10, 2)
scattering = np.loadtxt(f'{abspath}/src/Opacity/fromRICH/scatter.txt') 
f_edges = f_edges_ev * prel.ev_toHz # [Hz] frequency range for each group. shape: (10, 2)

# observers 
num_obs = prel.NPIX
observers_xyz = hp.pix2vec(prel.NSIDE, range(num_obs)) # shape: (3, 192)
observers_xyz = np.array(observers_xyz).T # shape: (192, 3)
for idx_s, snap in enumerate(snaps):
    if snap not in [250, 320, 420]: 
        continue
    print('\n Snapshot: ', snap, '\n', flush=True)
    # Load data and avoid fluff -----------------------------------------------------------------
    if alice:
        loadpath = f'{pre}/snap_{snap}'
    else:
        loadpath = f'{pre}/{snap}'
    data = make_tree(loadpath, snap, MG = True)
    box = np.load(f'{loadpath}/box_{snap}.npy')
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Rad, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE
    Eg0, Eg1, Eg2, Eg3, Eg4, Eg5, Eg6, Eg7, Eg8, Eg9 = data.Eg0, data.Eg1, data.Eg2, data.Eg3, data.Eg4, data.Eg5, data.Eg6, data.Eg7, data.Eg8, data.Eg9
    denmask = Den > 1e-19
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den, Eg0, Eg1, Eg2, Eg3, Eg4, Eg5, Eg6, Eg7, Eg8, Eg9 = \
        make_slices([X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den, Eg0, Eg1, Eg2, Eg3, Eg4, Eg5, Eg6, Eg7, Eg8, Eg9], denmask)
    xyz = np.array([X, Y, Z]).T
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Egs = [Eg0, Eg1, Eg2, Eg3, Eg4, Eg5, Eg6, Eg7, Eg8, Eg9]

    L_col = np.zeros((num_obs, len(prel.freqs)))
    r_initial = np.zeros(num_obs) # initial starting point for Rph

    photosphere = {e_idx: 
                   {'idx': [],
                    'r': [],
                    'x': [], 
                    'y': [], 
                    'z': [],
                    'vol': [],
                    'den': [],
                    'alpha_rossland': [],
                    'Lum': []}
                for e_idx in range(len(Egs))}

    colorsphere = {e_idx: 
                   {'idx': [],
                    'r': [],
                    'x': [], 
                    'y': [], 
                    'z': [],
                    'temp': [],
                    'los_eff': []}
                for e_idx in range(len(Egs))}
    
    freqs = prel.freqs
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
        r_initial[i] = rmax # this is true if the observers are nomalized to have |R|=1

        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        xyz2 = np.array([x, y, z]).T
        del x, y, z
        # find the simulation cell corresponding to cells in the wanted ray
        tree = KDTree(xyz, leaf_size=50) 
        _, idx = tree.query(xyz2, k=1)
        idx = idx.ravel()
        # Quantity corresponding to the ray
        d = Den[idx] * prel.den_converter
        t = T[idx]
        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        ray_Egs = [Eg[idx] for Eg in Egs]
        volume = Vol[idx]

        alpha_planck_groups = np.zeros((len(Egs), N_ray))
        los_effective_groups = np.zeros((len(Egs), N_ray))
        for e_idx, Eg in enumerate(Egs):
            ray_Eg = ray_Egs[e_idx]
            rossland_lin = np.loadtxt(f'{opac_path}/sigma_rossland_{e_idx+1}.txt').T
            planck_lin = np.loadtxt(f'{opac_path}/sigma_absorption_rossland_{e_idx+1}.txt').T
            rosseland = np.log(rossland_lin)
            planck = np.log(planck_lin)
            ######
            alpha_scatter = calc_scattering_opacity_vectorized(T_cool, Rho_cool, scattering, np.log(t), np.log(d))
            alpha_scatter = np.array(alpha_scatter)
            alpha_rossland = calc_ross_opacity_vectorized(T_cool, Rho_cool, rosseland, scattering, np.log(t), np.log(d))
            alpha_rossland = np.array(alpha_rossland)
            alpha_planck = calc_planck_opacity_vectorized(T_cool, Rho_cool, planck, np.log(t), np.log(d))
            alpha_planck = np.array(alpha_planck)
            alpha_planck_groups[e_idx, :] = alpha_planck
            ######
            # underflow_mask = np.logical_and(np.log(alpha_rossland) != 0.0, np.log(alpha_planck) != 0.0)
            # d_g = d.copy()
            # t_g = t.copy()
            # r_g = r.copy()
            # ray_x_g = ray_x.copy()
            # ray_y_g = ray_y.copy()
            # ray_z_g = ray_z.copy()
            # idx_g = idx.copy()
            # volume_g = volume.copy()
            # ray_Eg = ray_Egs[e_idx].copy()
            # d_g, t_g, r_g, ray_x_g, ray_y_g, ray_z_g, alpha_rossland_g, alpha_planck_g, ray_Eg, volume_g, idx_g = \
            #     make_slices([d_g, t_g, r_g, ray_x_g, ray_y_g, ray_z_g, alpha_rossland_g, alpha_planck_g, ray_Eg, volume_g, idx_g], underflow_mask)
            idx = np.array(idx)
            # Optical depth
            r_fuT = np.flipud(r) #.T
            alpha_rossland_fuT = np.flipud(alpha_rossland) 
            # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
            los = - np.flipud(sci.cumulative_trapezoid(alpha_rossland_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
            smoothed_flux_r2 = uniform_filter1d(r**2 * ray_Eg, 7) #r^2 is here (but it's for the flux) otherwise you get annoying errors in the if. 

            alpha_effective_g = np.sqrt(3 * np.minimum(alpha_planck, alpha_rossland) * alpha_rossland)
            alpha_effective_fuT = np.flipud(alpha_effective_g)
            los_effective = - np.flipud(sci.cumulative_trapezoid(alpha_effective_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs
            los_effective[los_effective > 30] = 30
            los_effective_groups[e_idx, :] = los_effective

            try: 
                photo_idx = np.where( ((smoothed_flux_r2>0) & (los<2/3) ))[0][0] 
            except IndexError: # if you don't find the photosphere, skip the observer
                print(f'No photosphere found for observer {i}', flush=True)
                continue
            Lg = 4*np.pi * smoothed_flux_r2[photo_idx] * prel.en_converter/prel.Rsol_cgs * prel.c_cgs # you have to convert smoothed_flux_r2 (en_den*r^2)
            photosphere[e_idx]['idx'].append(idx[photo_idx])
            photosphere[e_idx]['Lum'].append(Lg)
            photosphere[e_idx]['x'].append(ray_x[photo_idx])
            photosphere[e_idx]['y'].append(ray_y[photo_idx])
            photosphere[e_idx]['z'].append(ray_z[photo_idx])
            photosphere[e_idx]['r'].append(r[photo_idx])
            photosphere[e_idx]['vol'].append(volume[photo_idx])
            photosphere[e_idx]['den'].append(d[photo_idx]) # CGS
            photosphere[e_idx]['alpha_rossland'].append(alpha_rossland[photo_idx])

            # Spectra
            try: 
                color_idx = np.where(los_effective<5)[0][0]
            except IndexError:
                print(f'No color index found for observer {i}', flush=True)
                continue
            colorsphere[e_idx]['idx'].append(idx[color_idx])
            colorsphere[e_idx]['x'].append(ray_x[color_idx])
            colorsphere[e_idx]['y'].append(ray_y[color_idx])
            colorsphere[e_idx]['z'].append(ray_z[color_idx])
            colorsphere[e_idx]['r'].append(r[color_idx])
            colorsphere[e_idx]['temp'].append(t[color_idx])
            colorsphere[e_idx]['los_eff'].append(los_effective[color_idx])

        r_th_groups = []

        for g in range(len(Egs)):
            if len(colorsphere[g]['r']) == 0:
                raise RuntimeError(
                    f'No thermalization radius for observer {i}, group {g}'
                )

            r_th_groups.append(colorsphere[g]['r'][-1])

        r_th_groups = np.asarray(r_th_groups)
        r_th_min = np.min(r_th_groups)
        r_th_min_idx = np.argmin(np.abs(r - r_th_min))
        # Build spectrum --------------------------------------------------
        for g in range(len(Egs)):

            # Frequencies belonging to MG group g
            freq_mask = (
                (freqs >= f_edges[g])
                & (freqs < f_edges[g+1])
            )

            # Nothing to calculate if your spectral grid has no
            # frequencies inside this MG group
            if not np.any(freq_mask):
                continue

            nu = freqs[freq_mask]

            for k in range(max(r_th_min_idx, 1), len(r)):
                dr = r[k] - r[k-1]
                Vcell = r[k]**2 * dr

                wien = np.exp(
                    prel.h_cgs * nu /
                    (prel.Kb_cgs * t[k])
                ) - 1

                black_body = nu**3 / wien

                L_col[i, freq_mask] += (
                    alpha_planck_groups[g, k]
                    * Vcell
                    * np.exp(-los_effective_groups[g, k])
                    * black_body
                )

    Lg_snap = np.mean([photosphere[e_idx]['Lum'] for e_idx in range(len(Egs))], axis=0)
    Lphoto_snap = np.mean(Lg_snap)

    if save:
        # Save red of the single snap
        pre_saving = f'{abspath}/data/{folder}'
        data = np.concatenate([[snap], [tfb[idx_s]], [Lphoto_snap], Lg_snap])
        with open(f'{pre_saving}/{check}_red.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()

        np.savez(f"{pre_saving}/photo/{check}_photo{snap}.npz", photosphere=photosphere)
        np.savez(f"{pre_saving}/spectra/{check}_Rcol{snap}.npz", colorsphere=colorsphere)
        np.savetxt(f'{pre_saving}/spectra/freqs.txt', freqs)
        np.savetxt(f'{pre_saving}/spectra/{check}_spectra{snap}.txt', L_col)
             
        
# usage = resource.getrusage(resource.RUSAGE_SELF)
# print(f"Peak RAM usage: {usage.ru_maxrss / 1024**2:.2f} MB")
