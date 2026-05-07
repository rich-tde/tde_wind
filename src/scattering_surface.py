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
check = 'HiResNewAMR' 
N_ray = 5_000

## Snapshots stuff
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
print('we are in: ', pre, flush=True)

#%% Opacities: load and interpolate ----------------------------------------------------------------
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
planck = np.loadtxt(f'{opac_path}/planck.txt')
scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm
_, _, scatter2 = opacity_linear(T_cool, Rho_cool, scattering)
T_cool2, Rho_cool2, rossland2 = opacity_extrap(T_cool, Rho_cool, rossland, which_opacity = 'rossland', scatter = scatter2)
_, _, planck2 = opacity_extrap(T_cool, Rho_cool, planck, which_opacity = 'planck', slope_length = 10, scatter = None)

# observers 
num_obs = prel.NPIX
observers_xyz = hp.pix2vec(prel.NSIDE, range(num_obs)) # shape: (3, 192)
observers_xyz = np.array(observers_xyz).T # shape: (192, 3)
#%% MATLAB, thanks Cindy.
eng = matlab.engine.start_matlab()
for idx_s, snap in enumerate(snaps):
    if snap not in [76]:
        continue
    print('\n Snapshot: ', snap, '\n', flush=True)
    # Load data and avoid fluff -----------------------------------------------------------------
    if alice:
        loadpath = f'{pre}/snap_{snap}'
    else:
        loadpath = f'{pre}/{snap}'
    data = make_tree(loadpath, snap)
    box = np.load(f'{loadpath}/box_{snap}.npy')
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Rad, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE
    denmask = Den > 1e-19
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        make_slices([X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den], denmask)
    xyz = np.array([X, Y, Z]).T
    R = np.sqrt(X**2 + Y**2 + Z**2)

    xscatt = np.zeros(num_obs) 
    yscatt = np.zeros(num_obs)
    zscatt = np.zeros(num_obs)
    denscatt = np.zeros(num_obs) 
    Tempscatt = np.zeros(num_obs)
    Fxscatt = np.zeros(num_obs)
    Fyscatt = np.zeros(num_obs)
    Fzscatt = np.zeros(num_obs)
    alphaS_scatt = np.zeros(num_obs)
    tauS_scatt = np.zeros(num_obs)
    alphaR_scatt = np.zeros(num_obs)
    tauR_scatt = np.zeros(num_obs)

    xph = np.zeros(num_obs) 
    yph = np.zeros(num_obs)
    zph = np.zeros(num_obs)
    denph = np.zeros(num_obs) 
    Tempph = np.zeros(num_obs)
    Fxph = np.zeros(num_obs)
    Fyph = np.zeros(num_obs)
    Fzph = np.zeros(num_obs)
    alphaS_ph = np.zeros(num_obs)
    alphaR_ph = np.zeros(num_obs)
    alphaS_ph= np.zeros(num_obs)
    tauS_ph = np.zeros(num_obs)
    alphaR_ph= np.zeros(num_obs)
    tauR_ph = np.zeros(num_obs)
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
        
        # Interpolate opacity 
        ln_alpha_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
        ln_alpha_rossland = np.array(ln_alpha_rossland)[0]
        ln_alpha_scatter = eng.interp2(T_cool2, Rho_cool2, scatter2.T, np.log(t), np.log(d), 'linear', 0)
        ln_alpha_scatter = np.array(ln_alpha_scatter)[0]
        underflow_mask = np.logical_and(ln_alpha_rossland != 0.0, ln_alpha_scatter != 0.0)
        d, t, r, ray_x, ray_y, ray_z, ln_alpha_rossland, ln_alpha_scatter, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx = \
            make_slices([d, t, r, ray_x, ray_y, ray_z, ln_alpha_rossland, ln_alpha_scatter, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx], underflow_mask)
        idx = np.array(idx)
        alpha_rossland = np.exp(ln_alpha_rossland) # [1/cm]
        alpha_scatter = np.exp(ln_alpha_scatter)
        del ln_alpha_rossland
        gc.collect()

        # Optical depth
        r_fuT = np.flipud(r) #.T
        alpha_rossland_fuT = np.flipud(alpha_rossland) 
        # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
        tau = - np.flipud(sci.cumulative_trapezoid(alpha_rossland_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r

        alpha_scatter_fuT = np.flipud(alpha_scatter) 
        # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
        tau_scat = - np.flipud(sci.cumulative_trapezoid(alpha_scatter_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r

        # FLD curve 
        # Get 20 unique nearest neighbors to each cell in the wanted ray and use them to compute the gradient along the ray
        xyz3 = np.array([ray_x, ray_y, ray_z]).T
        _, idxnew = tree.query(xyz3, k=20)
        idxnew = np.unique(idxnew) #.T
        dx = 0.5 * volume**(1/3) # Cell radius 
        f_inter_input = np.array([X[idxnew], Y[idxnew], Z[idxnew]]).T

        gradx_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x+dx, ray_y, ray_z]).T )
        gradx_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x-dx, ray_y, ray_z]).T )
        gradx = (gradx_p - gradx_m)/ (2*dx)
        gradx = np.nan_to_num(gradx, nan =  0)
        del gradx_p, gradx_m

        grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y+dx, ray_z]).T )
        grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y-dx, ray_z]).T )
        grady = (grady_p - grady_m)/ (2*dx)
        grady = np.nan_to_num(grady, nan =  0)
        del grady_p, grady_m

        gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y, ray_z+dx]).T )
        gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y, ray_z-dx]).T )
        gradz = (gradz_p - gradz_m)/ (2*dx)
        gradz = np.nan_to_num(gradz, nan =  0)
        del gradz_p, gradz_m

        grad = np.sqrt(gradx**2 + grady**2 + gradz**2) # grad = |grad|
        # gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz) # projection of the gradient along the radial direction
        gc.collect()

        # Eq.(28) from Krumholz07.
        R_lamda = grad / ( prel.Rsol_cgs * alpha_rossland* ray_radDen) # this is the conversion for /r from the gradient. It's dimensionless
        R_lamda[R_lamda < 1e-10] = 1e-10
        # Eq.(27) from Krumholz07.
        fld_factor = (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
        # Eq.(26) from Krumholz07. 
        Fx = -prel.c_cgs * fld_factor * gradx * (prel.en_den_converter/prel.Rsol_cgs)/ alpha_rossland # CGS
        Fy = -prel.c_cgs * fld_factor * grady * (prel.en_den_converter/prel.Rsol_cgs)/ alpha_rossland
        Fz = -prel.c_cgs * fld_factor * gradz * (prel.en_den_converter/prel.Rsol_cgs)/ alpha_rossland
        Fr = (mu_x * Fx) + (mu_y*Fy) + (mu_z*Fz)
        # smoothed_flux_r2 = -prel.c_cgs * uniform_filter1d(r**2 * fld_factor * gradr / alpha_rossland, 7) #r^2 is here (but it's for the flux) otherwise you get annoying errors in the if. 
        smoothed_flux_r2 = uniform_filter1d(r**2 * Fr, 7) #r^2 is here (but it's for the flux) otherwise you get annoying errors in the if. 
        del gradx, grady, gradz

        try: 
            photosphere = np.where( ((smoothed_flux_r2>0) & (tau<2/3) ))[0][0] 
        except IndexError: # if you don't find the photosphere, skip the observer
            print(f'No photosphere found for observer {i}', flush=True)
            continue
        
        try: 
            scatt_surf = np.where( ((smoothed_flux_r2>0) & (tau_scat<2/3) ))[0][0]  
        except IndexError: # if you don't find the scatt_surf, skip the observer
            print(f'No scatt_surf found for observer {i}', flush=True)
            continue
    
        # fig, ax1 = plt.subplots(1, 1, figsize = (8, 6))
        # Rt = 13
        # ax1.plot(r_int/Rt, tau_scat, label = f'obs {i}, scat', c = 'r')
        # ax1.plot(r/Rt, tau, label = f'obs {i}, photo', ls = '--', c = 'b')
        # ax1.axvline(r[scatt_surf]/Rt, c = 'r')
        # ax1.axvline(r[photosphere]/Rt, ls = '--', c = 'b')
        # ax1.set_ylabel(r'$\kappa$ [cm$^2$/g]')
        # ax1.set_ylim(1e-1, 1e1)
        # ax1.set_xlim(2, 1.5 * r[scatt_surf]/Rt)
        # ax1.loglog()
        # ax1.set_xlabel(r'$r/r_{\rm t}$')
        # ax1.grid()
        # ax1.legend(fontsize = 16)
        # ax1.tick_params(axis='both', which='major',length=10, width=1.5)
        # ax1.tick_params(axis='both', which='minor',length=5, width=1)
        # ax1.legend(fontsize = 16)

        xscatt[i] = ray_x[scatt_surf]
        yscatt[i] = ray_y[scatt_surf]
        zscatt[i] = ray_z[scatt_surf]
        denscatt[i] = d[scatt_surf]
        alphaS_scatt[i] = alpha_scatter[scatt_surf]
        tauS_scatt[i] = tau_scat[scatt_surf]
        alphaR_scatt[i] = alpha_rossland[scatt_surf]
        tauR_scatt[i] = tau[scatt_surf]
        Fxscatt[i] = Fx[scatt_surf]
        Fyscatt[i] = Fy[scatt_surf] 
        Fzscatt[i] = Fz[scatt_surf]

        xph[i] = ray_x[photosphere]
        yph[i] = ray_y[photosphere]
        zph[i] = ray_z[photosphere]
        denph[i] = d[photosphere]
        alphaS_ph[i] = alpha_scatter[photosphere]
        tauS_ph[i] = tau_scat[photosphere]
        alphaR_ph[i] = alpha_rossland[photosphere]
        tauR_ph[i] = tau[photosphere]
        Fxph[i] = Fx[photosphere]
        Fyph[i] = Fy[photosphere] 
        Fzph[i] = Fz[photosphere]

        del smoothed_flux_r2, R_lamda, fld_factor, ray_radDen
        gc.collect()


    if save:
        # Save red of the single snap
        pre_saving = f'{abspath}/data/{folder}'

        with open(f'{pre_saving}/scatt_surf/{check}_scatt{snap}taus.txt', 'w') as f:
            f.write('# Data for the scattering surface.\n')
            f.write('# xscatt\n' + ' '.join(map(str, xscatt)) + '\n')
            f.write('# yscatt\n' + ' '.join(map(str, yscatt)) + '\n')
            f.write('# zscatt\n' + ' '.join(map(str, zscatt)) + '\n')
            f.write('# denscatt CGS\n' + ' '.join(map(str, denscatt)) + '\n')
            f.write('# alphaS_scatt\n' + ' '.join(map(str, alphaS_scatt)) + '\n')
            f.write('# tauSscatt\n' + ' '.join(map(str, tauS_scatt)) + '\n')
            f.write('# alphaR_scatt\n' + ' '.join(map(str, alphaR_scatt)) + '\n')
            f.write('# tauRscatt\n' + ' '.join(map(str, tauR_scatt)) + '\n')
            f.write('# Fxscatt CGS\n' + ' '.join(map(str, Fxscatt)) + '\n')
            f.write('# Fyscatt CGS\n' + ' '.join(map(str, Fyscatt)) + '\n')
            f.write('# Fzscatt CGS\n' + ' '.join(map(str, Fzscatt)) + '\n')
            f.close()
        
        with open(f'{pre_saving}/scatt_surf/{check}_photo{snap}taus.txt', 'w') as f:
            f.write('# Data for the photospere.\n')
            f.write('# xph\n' + ' '.join(map(str, xph)) + '\n')
            f.write('# yph\n' + ' '.join(map(str, yph)) + '\n')
            f.write('# zph\n' + ' '.join(map(str, zph)) + '\n')
            f.write('# denph CGS\n' + ' '.join(map(str, denph)) + '\n')
            f.write('# alphaS_ph\n' + ' '.join(map(str, alphaS_ph)) + '\n')
            f.write('# tauS_ph\n' + ' '.join(map(str, tauS_ph)) + '\n')
            f.write('# alphaR_ph\n' + ' '.join(map(str, alphaR_ph)) + '\n')
            f.write('# tauR_ph\n' + ' '.join(map(str, tauR_ph)) + '\n')
            f.write('# Fxph CGS\n' + ' '.join(map(str, Fxph)) + '\n')
            f.write('# Fyph CGS\n' + ' '.join(map(str, Fyph)) + '\n')
            f.write('# Fzph CGS\n' + ' '.join(map(str, Fzph)) + '\n')
            f.close()
             
    del xscatt, yscatt, zscatt
    gc.collect()
        
eng.exit()
# usage = resource.getrusage(resource.RUSAGE_SELF)
# print(f"Peak RAM usage: {usage.ru_maxrss / 1024**2:.2f} MB")

# %%
if plot:
    from Utilities.operators import sort_list
    snaps = [76, 109, 151]
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps_lum, tfb_lum, Lum = data[:, 0], data[:, 1], data[:, 2]
    snaps_lum, Lum, tfb_lum = sort_list([snaps_lum, Lum, tfb_lum], tfb_lum, unique=True) 
    snaps_lum = snaps_lum.astype(int)

    fig, ax = plt.subplots(1,1,figsize=(10, 8))
    for snap in snaps:
        time = tfb_lum[np.argmin(np.abs(snaps_lum - snap))]
        photo_taus = np.loadtxt(f'{abspath}/data/{folder}/scatt_surf/{check}_photo{snap}taus.txt')
        den, alphaS, alphaR = photo_taus[3], photo_taus[4], photo_taus[6]
        kappaS = alphaS/den
        kappaR = alphaR/den

        kappa_ratio = list(np.sort(kappaS/kappaR))
        bin_kappa = list(np.arange(len(kappa_ratio))/len(kappa_ratio))
        ax.plot(kappa_ratio, bin_kappa, linewidth = 2, label = f't = {time:.2f}' + r't$_{\rm fb}$')
        # ax.hist(kappaS/kappaR, bins=30, color='navy', alpha=0.7)
    ax.set_xlabel(r'$\kappa_{\rm S}/\kappa_{\rm R}$')
    ax.legend(fontsize=20)
    ax.grid()
# %%
