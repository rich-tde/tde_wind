""" FLD curve for the multigroup. """
import sys
sys.path.append('/Users/paolamartire/shocks')
# import resource
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
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
from src.Opacity.linextrapolator import opacity_extrap, opacity_linear
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
opac_path = f'{abspath}/src/Opacity/MG'
T_cool2 = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool2 = np.loadtxt(f'{opac_path}/rho.txt')

# observers 
num_obs = prel.NPIX
observers_xyz = hp.pix2vec(prel.NSIDE, range(num_obs)) # shape: (3, 192)
observers_xyz = np.array(observers_xyz).T # shape: (192, 3)
#%% MATLAB, thanks Cindy.
eng = matlab.engine.start_matlab()
for idx_s, snap in enumerate(snaps):
    if snap not in [320]: 
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

    ph_idx = np.zeros((len(Egs), num_obs))
    xph = np.zeros((len(Egs), num_obs)) 
    yph = np.zeros((len(Egs), num_obs))
    zph = np.zeros((len(Egs), num_obs))
    volph = np.zeros((len(Egs), num_obs))
    denph = np.zeros((len(Egs), num_obs)) 
    rph = np.zeros((len(Egs), num_obs)) 
    alphaph = np.zeros((len(Egs), num_obs)) 
    Lg = np.zeros((len(Egs), num_obs))
    r_initial = np.zeros(num_obs) # initial starting point for Rph
    for i in range(num_obs):
        if i not in [0, 191]:
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
        
        for e_idx, Eg in enumerate(Egs):
            ray_Eg = ray_Egs[e_idx]
            rossland2 = np.loadtxt(f'{opac_path}/sigma_rossland_{e_idx}.txt') 
            ######
            ln_alpha_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
            ln_alpha_rossland = np.array(ln_alpha_rossland)[0]
            underflow_mask = ln_alpha_rossland != 0.0
            ######
            d, t, r, ray_x, ray_y, ray_z, ln_alpha_rossland, ray_Eg, volume, idx = \
                make_slices([d, t, r, ray_x, ray_y, ray_z, ln_alpha_rossland, ray_Eg, volume, idx], underflow_mask)
            idx = np.array(idx)
            alpha_rossland = np.exp(ln_alpha_rossland) # [1/cm]
            # Optical depth
            r_fuT = np.flipud(r) #.T
            alpha_rossland_fuT = np.flipud(alpha_rossland) 
            # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
            los = - np.flipud(sci.cumulative_trapezoid(alpha_rossland_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
            smoothed_flux_r2 = uniform_filter1d(r**2 * Eg, 7) #r^2 is here (but it's for the flux) otherwise you get annoying errors in the if. 

            try: 
                photosphere = np.where( ((smoothed_flux_r2>0) & (los<2/3) ))[0][0] 
            except IndexError: # if you don't find the photosphere, skip the observer
                print(f'No photosphere found for observer {i}', flush=True)
                continue
            Lg[e_idx][i] = 4*np.pi * smoothed_flux_r2[photosphere] * prel.Rsol_cgs**2 * prel.c_cgs# you have to convert the r^2 in smoothed_flux_r2
            ph_idx[e_idx][i] = idx[photosphere]
            xph[e_idx][i] = ray_x[photosphere]
            yph[e_idx][i] = ray_y[photosphere]
            zph[e_idx][i] = ray_z[photosphere]
            volph[e_idx][i] = volume[photosphere]
            denph[e_idx][i] = d[photosphere]
            rph[e_idx][i] = r[photosphere]  
            alphaph[e_idx][i] = alpha_rossland[photosphere]

    Lphoto_snap = np.mean(Lg, axis = 1) # take the mean
    print(np.shape(Lphoto_snap))

    if save:
        # Save red of the single snap
        pre_saving = f'{abspath}/data/{folder}'
        data = [snap, tfb[idx_s], Lphoto_snap]
        with open(f'{pre_saving}/{check}_red.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()
        
        for e_idx, Eg in enumerate(Egs):
            with open(f'{pre_saving}/photo/{check}_photo{snap}Eg{e_idx}.txt', 'w') as f:
                f.write('# Data for the photospere.\n')
                f.write('# xph\n' + ' '.join(map(str, xph[e_idx])) + '\n')
                f.write('# yph\n' + ' '.join(map(str, yph[e_idx])) + '\n')
                f.write('# zph\n' + ' '.join(map(str, zph[e_idx])) + '\n')
                f.write('# volph\n' + ' '.join(map(str, volph[e_idx])) + '\n')
                f.write('# denph CGS\n' + ' '.join(map(str, denph[e_idx])) + '\n')
                f.write('# alpha CGS\n' + ' '.join(map(str, alphaph[e_idx])) + '\n')
                f.write('# rph\n' + ' '.join(map(str, rph[e_idx])) + '\n')
                f.write('# Lg CGS\n' + ' '.join(map(str, Lg[e_idx])) + '\n')
                f.write('# indices\n' + ' '.join(map(str, ph_idx[e_idx])) + '\n')
                f.close()
             
    del smoothed_flux_r2, ln_alpha_rossland, xph, yph, zph, volph, denph, alphaph, rph, Lg, ph_idx
    gc.collect()
        
eng.exit()
# usage = resource.getrusage(resource.RUSAGE_SELF)
# print(f"Peak RAM usage: {usage.ru_maxrss / 1024**2:.2f} MB")
