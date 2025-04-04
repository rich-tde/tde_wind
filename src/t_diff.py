"""Compute diffusion time"""
import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    abspath = '/Users/paolamartire/shocks'
    save = False

import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.integrate as sci
import healpy as hp
import matlab.engine
from sklearn.neighbors import KDTree
from src.Opacity.linextrapolator import nouveau_rich
import Utilities.prelude as prel
from Utilities.operators import make_tree, sort_list, to_spherical_components
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
import src.orbits as orb

#%% matlab
eng = matlab.engine.start_matlab()
#%% Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' 
snap = 348
direction = 'r' # 'r' or 'z
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/_photo{snap}.txt')
xph, yph, zph = photo[0], photo[1], photo[2]
rph = np.sqrt(xph**2 + yph**2 + zph**2)
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
Rt = orb.tidal_radius(Rstar, mstar, Mbh)
a_min = orb.semimajor_axis(Rstar, mstar, Mbh, prel.G)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #days
t_fall_cgs = t_fall * 24 * 3600

# Load data
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5)

data = make_tree(f'{pre}/{snap}', snap, energy = True)
if alice:
    box = np.load(f'{pre}/snap_{snap}/box_{snap}.npy')
else:
    box = np.load(f'{pre}/{snap}/box_{snap}.npy')
X, Y, Z, T, Den, Mass, Rad, Vol, VX, VY, VZ = \
    data.X, data.Y, data.Z, data.Temp, data.Den, data.Mass, data.Rad, data.Vol, data.VX, data.VY, data.VZ
dim_cell = Vol**(1/3)
# consider only cells in a slice
denmask = Den > 1e-19
X, Y, Z, dim_cell, T, Den, Vol, VX, VY, VZ = \
    make_slices([X, Y, Z, dim_cell, T, Den, Vol, VX, VY, VZ], denmask)
xyz = np.array([X, Y, Z]).T
N_ray = 500
#%
if direction == 'z':
    # observers
    Rt_obs = [Rt, 0, 0]
    amin_obs = [a_min, 0, 0]
    apo_obs = [apo, 0, 0]
    observers_xyz = np.array([Rt_obs, amin_obs, apo_obs])
    i = 0
    mu_x = observers_xyz[i][0]
    mu_y = observers_xyz[i][1]
    mu_z = observers_xyz[i][2]
    # x_tr = np.zeros(len(mu_z))
    # y_tr = np.zeros(len(mu_z))
    # z_tr = np.zeros(len(mu_z))
    # dim_tr = np.zeros(len(mu_z))
    # den_tr = np.zeros(len(mu_z))
    # Temp_tr = np.zeros(len(mu_z))
    # Vx_tr = np.zeros(len(mu_z))
    # Vy_tr = np.zeros(len(mu_z))
    # Vz_tr = np.zeros(len(mu_z))

    # Box is for dynamic ray making
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

    r_height = np.logspace(-0.25, np.log10(rmax), N_ray) # create the z array
    #go vertically 
    x = np.repeat(mu_x, len(r_height))
    y = np.repeat(mu_y, len(r_height))
    z = r_height
    xyz2 = np.array([x, y, z]).T
    del x, y, z

    tree = KDTree(xyz, leaf_size=50)
    _, idx = tree.query(xyz2, k=1)
    idx = [ int(idx[i][0]) for i in range(len(idx))] # no -1 because we start from 0
    idx = np.unique(idx)
    d = Den[idx] * prel.den_converter
    t = T[idx]
    ray_x = X[idx]
    ray_y = Y[idx]
    ray_z = Z[idx]
    ray_dim = dim_cell[idx]
    ray_vz = VZ[idx]
    d, t, ray_x, ray_y, ray_vz, ray_dim, idx, ray_z = \
        sort_list([d, t, ray_x, ray_y, ray_vz, ray_dim, idx, ray_z], ray_z)

    # Interpolate ----------------------------------------------------------
    sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
    sigma_rossland = np.array(sigma_rossland)[0]
    underflow_mask = sigma_rossland != 0.0
    idx = np.array(idx)
    d, t, ray_x, ray_y, ray_z, ray_dim, ray_vz, idx = \
        make_slices([d, t, ray_x, ray_y, ray_z, ray_dim, ray_vz, idx], underflow_mask)
    sigma_rossland_eval = np.exp(sigma_rossland) # [1/cm]
    del sigma_rossland
    gc.collect()

    # Optical Depth
    ray_fuT = np.flipud(ray_z) 
    kappa_rossland = np.flipud(sigma_rossland_eval) #np.flipud(d)*0.34cm2/g
    # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
    los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, ray_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for ray_z. YOu integrate in the z direction

    # los ofr each cell
    # los = sigma_rossland_eval * 2 * ray_dim # this is the conversion for ray_z. YOu integrate in the z direction

    #
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,10))
    cutplot = np.logical_and(Den > 1e-19, np.abs(X-mu_x)<dim_cell)
    img = ax1.scatter(Y[cutplot], Z[cutplot]/apo, c = X[cutplot]/Rt,  cmap = 'jet', alpha = 0.7, vmin = 0.5, vmax = 1.5)
    img = ax1.scatter(xyz2[:,1], xyz2[:,2]/apo, c = xyz2[:,0]/Rt, cmap = 'jet', marker = 's', edgecolors= 'k', label = 'What we want', vmin = 0.5, vmax = 1.5)
    img = ax1.scatter(ray_y, ray_z/apo, c = ray_x/Rt, cmap = 'jet', edgecolors= 'k', label = 'From simulation', vmin = 0.5, vmax = 1.5)
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'X [$R_{\rm t}$]')
    ax1.legend(fontsize = 18)
    img = ax2.scatter(ray_y, ray_z/apo, c = los, cmap = 'jet', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e2))
    ax2.plot(ray_y, ray_z/apo, c = 'k', alpha = 0.5)
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'$\tau$')
    ax1.set_ylabel(r'Z [$R_{\rm a}$]')
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'Y [$R_{\odot}$]')
        ax.set_xlim(-50,50)
        ax.set_ylim(-1/apo, 3.5)
    ax1.set_title('Points wanted and selected', fontsize = 18)
    ax2.set_title('Optical depth for each cell', fontsize = 18)
    plt.tight_layout()
    #
    ctau = prel.csol_cgs/los #c/tau [code units]
    try: 
        ctauV = ctau/np.abs(ray_vz)
        Rtr_idx_all = np.where(ctauV<1)[0]
        x_tr_all = ray_x[Rtr_idx_all]
        y_tr_all = ray_y[Rtr_idx_all]
        z_tr_all = ray_z[Rtr_idx_all]
        # Rtr_idx_all = Rtr_idx_all[ctauV[~np.isnan(ctauV)]]
        Rtr_idx_all_inside = np.where(np.abs(z_tr_all)<np.mean(rph))[0] # you should find the nearest observer from healpix and compare with Rph in this direction
        Rtr_idx_all = Rtr_idx_all[Rtr_idx_all_inside]
        Rtr_idx = Rtr_idx_all[-1]
        x_tr= ray_x[Rtr_idx]
        y_tr= ray_y[Rtr_idx]
        z_tr= ray_z[Rtr_idx]
        dim_tr = ray_dim[Rtr_idx]
        den_tr = d[Rtr_idx]
        Temp_tr = t[Rtr_idx]
        Vz_tr = ray_vz[Rtr_idx]
        print(f'Rtr found', flush=False)
    except IndexError: # if you don't find the photosphere, exlude the observer
        print(f'No Rtr found', flush=False)
        sys.stdout.flush()
        # continue

    tdiff = 2*ray_dim * los * prel.Rsol_cgs/ prel.c_cgs # [cgs] H=2*dim_cell
    los_fuT = np.flipud(los)
    tdiff_cumulative = - np.flipud(sci.cumulative_trapezoid(los_fuT, ray_fuT, initial = 0))*prel.Rsol_cgs/ prel.c_cgs # this is the conversion for ray_z. YOu integrate in the z direction
    
    #
    plt.figure(figsize = (10,5))
    img = plt.scatter(ray_z/apo, sigma_rossland_eval/d, c = t, cmap = 'jet', norm = colors.LogNorm(vmin = 3e3, vmax = 5e5))
    cbar = plt.colorbar(img)
    cbar.set_label(r'$\tau$')
    plt.xlabel(r'$R [R_{\rm a}]$')
    plt.ylabel(r'$\kappa$ [cm$^2$/g]')
    plt.xlim(0,7)
    plt.axhline(0.34, c = 'k', linestyle = '--', label = r'$\kappa_{\rm Th}$')
    # plt.ylim(1, 1e4)
    plt.yscale('log')
    #
    idx_trap = np.argmin(np.abs(ray_z-z_tr)) # index of the trapping point   
    fig, (ax1, ax2) = plt.subplots(1, 2 , figsize = (12,5))
    ax1.plot(ray_z/apo, tdiff_cumulative/t_fall_cgs, c = 'k')
    ax1.set_ylabel(r'$t_{\rm diff} [t_{\rm fb}]$')
    ax1.set_ylim(0, 5) # a bit further than the Rtrapp
    # ax1.axvline(ray_z[np.argmin(np.abs(ctau[1:]/ray_vz[1:]-1))]/Rt)
    img = ax2.scatter(ray_z/apo, los, c = ctau/np.abs(ray_vz), cmap = 'rainbow', vmin = 0, vmax = 2)
    cbar = plt.colorbar(img)
    cbar.set_label(r'c$\tau^{-1}/V_z$')
    ax2.set_ylabel(r'$\tau$')
    ax2.set_yscale('log')
    # ax2.set_ylim(0.1, 1e2)
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$Z [R_{\rm a}]$')
        ax.set_xlim(-0.1, 7)
        ax.axvline(ray_z[idx_trap]/apo, c = 'k', linestyle = '--', label =  r'$R_{\rm tr} (c/\tau=V_r)$')
        ax.axvline(np.mean(rph)/apo, c = 'k', linestyle = 'dotted', label =  r'$<R_{\rm ph}>$')
        ax.legend(fontsize = 14)
    plt.tight_layout()

#%%
eng.exit()