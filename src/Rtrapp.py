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
check = 'HiRes' 
Rt = orb.tidal_radius(Rstar, mstar, Mbh)
a_min = orb.semimajor_axis(Rstar, mstar, Mbh, prel.G)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #days
t_fall_cgs = t_fall * 24 * 3600

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) #shape: (3, 192)
observers_xyz = np.array(observers_xyz).T # shape: (192, 3)
num_obs = prel.NPIX # you'll use it for the mean of the observers. It's 192, unless you don't find the photosphere for someone and so decrease of 1
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
pre_saving = f'{abspath}/data/{folder}'

# Load data
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5)

for snap in snaps:
    photo = np.loadtxt(f'{pre_saving}/photo/_photo{snap}.txt')
    xph, yph, zph = photo[0], photo[1], photo[2]
    rph = np.sqrt(xph**2 + yph**2 + zph**2)
    if alice:
        data = make_tree(f'{pre}/snap_{snap}', snap, energy = True)        
        box = np.load(f'{pre}/snap_{snap}/box_{snap}.npy')
    else:
        data = make_tree(f'{pre}/{snap}', snap, energy = True)
        box = np.load(f'{pre}/{snap}/box_{snap}.npy')
    X, Y, Z, T, Den, Mass, Rad, Vol, VX, VY, VZ = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Mass, data.Rad, data.Vol, data.VX, data.VY, data.VZ
    # consider only cells in a slice
    denmask = Den > 1e-19
    X, Y, Z, Vol, T, Den, Vol, VX, VY, VZ = \
        make_slices([X, Y, Z, Vol, T, Den, Vol, VX, VY, VZ], denmask)
    xyz = np.array([X, Y, Z]).T
    N_ray = 500

    x_tr = np.zeros(len(observers_xyz))
    y_tr = np.zeros(len(observers_xyz))
    z_tr = np.zeros(len(observers_xyz))
    den_tr = np.zeros(len(observers_xyz))
    dim_tr = np.zeros(len(observers_xyz))
    vol_tr = np.zeros(len(observers_xyz))
    Temp_tr = np.zeros(len(observers_xyz))
    Vx_tr = np.zeros(len(observers_xyz))
    Vy_tr = np.zeros(len(observers_xyz))
    Vz_tr = np.zeros(len(observers_xyz))
    Vr_tr = np.zeros(len(observers_xyz))

    for i in range(len(observers_xyz)):
        print(f'{i}', flush=True)
        sys.stdout.flush()
        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]
        
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
        r = np.logspace(-0.25, np.log10(rmax), N_ray)
        x = r*mu_x
        y = r*mu_y
        z = r*mu_z

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
        ray_vol = Vol[idx]
        ray_vx = VX[idx]
        ray_vy = VY[idx]
        ray_vz = VZ[idx]
        ray_r = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
        d, t, ray_x, ray_y, ray_z, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_r = \
            sort_list([d, t, ray_x, ray_y, ray_z, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_r], ray_r)
        long_ph_s = np.arctan2(ray_y, ray_x)          # Azimuthal angle in radians
        lat_ph_s = np.arccos(ray_z/ ray_r) 
        v_rad, v_theta, v_phi= to_spherical_components(ray_vx, ray_vy, ray_vz, lat_ph_s, long_ph_s)

        # Interpolate ----------------------------------------------------------
        sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
        sigma_rossland = np.array(sigma_rossland)[0]
        underflow_mask = sigma_rossland != 0.0
        idx = np.array(idx)
        d, t, ray_x, ray_y, ray_z, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_r = \
            make_slices([d, t, ray_x, ray_y, ray_z, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_r], underflow_mask)
        sigma_rossland_eval = np.exp(sigma_rossland) # [1/cm]
        del sigma_rossland
        gc.collect()

        # Optical Depth
        ray_fuT = np.flipud(ray_r)#ray_z) 
        kappa_rossland = np.flipud(sigma_rossland_eval) #np.flipud(d)*0.34cm2/g
        # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
        los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, ray_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for ray_z. YOu integrate in the z direction
        los_zero = los != 0
        d, t, ray_x, ray_y, ray_z, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_r, v_rad, los = \
            make_slices([d, t, ray_x, ray_y, ray_z, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_r, v_rad, los], los_zero)
        ctau = prel.csol_cgs/los #c/tau [code units]

        try: 
            Rtr_idx_all = np.where(ctau/np.abs(v_rad)<1)[0]
            x_tr_all = ray_x[Rtr_idx_all]
            y_tr_all = ray_y[Rtr_idx_all]
            z_tr_all = ray_z[Rtr_idx_all]
            R_tr_all = np.sqrt(x_tr_all**2 + y_tr_all**2 + z_tr_all**2)
            Rtr_idx_all_inside = np.where(R_tr_all<rph[i])[0]
            Rtr_idx_all = Rtr_idx_all[Rtr_idx_all_inside]
            Rtr_idx = Rtr_idx_all[-1]
            x_tr[i] = ray_x[Rtr_idx]
            y_tr[i] = ray_y[Rtr_idx]
            z_tr[i] = ray_z[Rtr_idx]
            vol_tr[i] = ray_vol[Rtr_idx]
            den_tr[i] = d[Rtr_idx]/prel.den_converter
            Temp_tr[i] = t[Rtr_idx]
            Vx_tr[i] = ray_vx[Rtr_idx]
            Vy_tr[i] = ray_vy[Rtr_idx]
            Vz_tr[i] = ray_vz[Rtr_idx]
            Vr_tr[i] = v_rad[Rtr_idx]
        except IndexError: # if you don't find the photosphere, exlude the observer
            print(f'No Rtr found in {i}', flush=False)
            sys.stdout.flush()
            continue

    if alice:
        with open(f'{pre_saving}/trap/{check}_Rtr{snap}.txt', 'a') as f:
                f.write('# Data for the Rtr (all in CGS). Lines are: x_tr, y_tr, z_tr, vol_tr, den_tr, Temp_tr, Vx_tr, Vy_tr, Vz_tr, Vr_tr \n')
                f.write(' '.join(map(str, x_tr)) + '\n')
                f.write(' '.join(map(str, y_tr)) + '\n')
                f.write(' '.join(map(str, z_tr)) + '\n')
                f.write(' '.join(map(str, vol_tr)) + '\n')
                f.write(' '.join(map(str, den_tr)) + '\n')
                f.write(' '.join(map(str, Temp_tr)) + '\n')
                f.write(' '.join(map(str, Vx_tr)) + '\n')
                f.write(' '.join(map(str, Vy_tr)) + '\n')
                f.write(' '.join(map(str, Vz_tr)) + '\n')
                f.write(' '.join(map(str, Vr_tr)) + '\n')
                f.close()

if plot:
    trap = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/trap/{check}_Rtr{snap}.txt')
    x_tr_i, y_tr_i, z_tr_i, Vx_tr, Vy_tr, Vz_tr, Vr_tr = trap[0], trap[1], trap[2], trap[-4], trap[-3], trap[-2], trap[-1]
    v_tr_i = np.sqrt(Vx_tr**2 + Vy_tr**2 + Vz_tr**2) * prel.Rsol_cgs * 1e-5/ prel.tsol_cgs
    R_tr_i = np.sqrt(x_tr_i**2 + y_tr_i**2 + z_tr_i**2)

    fig, ax2 = plt.subplots(1, 1 , figsize = (6,5))
    img = ax2.scatter(R_tr_i/apo, v_tr_i*1e-4)
    ax2.set_ylabel(r'$|V| [10^4$ km/s]')
    ax2.set_xlabel(r'$R [R_{\rm a}]$')
    ax2.set_ylim(.1, 2)
    ax2.axvline(np.mean(rph)/apo, c = 'k', linestyle = 'dotted', label =  r'$<R_{\rm ph}>$')
    ax2.legend(fontsize = 14)
    plt.tight_layout()

#%%
eng.exit()