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
    save = False
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
check = 'HiResNewAMR' # 

## Snapshots stuff
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
print('we are in: ', pre, flush=True)

# Opacities: load and interpolate ----------------------------------------------------------------
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
planck = np.loadtxt(f'{opac_path}/planck.txt')
scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm
_, _, scatter2 = opacity_linear(T_cool, Rho_cool, scattering)
T_cool2, Rho_cool2, rossland2 = opacity_extrap(T_cool, Rho_cool, rossland, which_opacity = 'rossland', scatter = scatter2)
_, _, planck2 = opacity_extrap(T_cool, Rho_cool, planck, which_opacity = 'planck', scatter = None)

N_ray = 5_000
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

#%% MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()
Lphoto_all = np.zeros(len(snaps))
for idx_s, snap in enumerate(snaps):
    # if snap != 109:
    #     continue
    print('\n Snapshot: ', snap, '\n', flush=True)
    box = np.zeros(6)
    # Load data -----------------------------------------------------------------
    if alice:
        loadpath = f'{pre}/snap_{snap}'
    else:
        loadpath = f'{pre}/{snap}'
     
    data = make_tree(loadpath, snap, energy = True)
    box = np.load(f'{loadpath}/box_{snap}.npy')
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Rad, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE

    denmask = Den > 1e-19
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        make_slices([X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den], denmask)

    xyz = np.array([X, Y, Z]).T
    R = np.sqrt(X**2 + Y**2 + Z**2)
    # Cross dot -----------------------------------------------------------------
    num_obs = prel.NPIX # you'll use it for the mean of the observers. It's 192, unless you don't find the photosphere for someone and so decrease of 1
    observers_xyz = hp.pix2vec(prel.NSIDE, range(num_obs)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz).T#[:,:,0] # shape: (192, 3)

    cross_dot = np.matmul(observers_xyz,  observers_xyz.T)
    cross_dot[cross_dot<0] = 0
    cross_dot *= 4/192

    # Dynamic Box 
    F_photo = np.zeros((prel.NPIX, len(prel.freqs)))
    F_photo_temp = np.zeros((prel.NPIX, len(prel.freqs)))
    ph_idx = np.zeros(num_obs)
    xph = np.zeros(num_obs) 
    yph = np.zeros(num_obs)
    zph = np.zeros(num_obs)
    volph = np.zeros(num_obs)
    denph = np.zeros(num_obs) 
    Tempph = np.zeros(num_obs)
    Rad_denph = np.zeros(num_obs)
    Vxph = np.zeros(num_obs) 
    Vyph = np.zeros(num_obs)
    Vzph = np.zeros(num_obs)
    Pressph = np.zeros(num_obs)
    IE_denph = np.zeros(num_obs)
    fluxes = np.zeros(num_obs)
    rph = np.zeros(num_obs) 
    alphaph = np.zeros(num_obs) 
    Lph = np.zeros(num_obs) 
    Fxph = np.zeros(num_obs)
    Fyph = np.zeros(num_obs)
    Fzph = np.zeros(num_obs)
    r_initial = np.zeros(num_obs) # initial starting point for Rph
    for i in range(num_obs):
        # if i not in [0, 90]:
        #     continue
        # Progress 
        print(f'Snap: {snap}, Obs: {i}', flush=True)
        # sys.stdout.flush()

        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        # Box is for dynamic ray making
        # box gives -x, -y, -z, +x, +y, +z. 
        if mu_x < 0:
            rmax = box[0] / mu_x
            # print('x-', rmax)
        else:
            rmax = box[3] / mu_x
            # print('x+', rmax)
        if mu_y < 0:
            rmax = min(rmax, box[1] / mu_y)
            # print('y-', rmax)
        else:
            rmax = min(rmax, box[4] / mu_y)
            # print('y+', rmax)

        if mu_z < 0:
            rmax = min(rmax, box[2] / mu_z)
            # print('z-', rmax)
        else:
            rmax = min(rmax, box[5] / mu_z)
            # print('z+', rmax)

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
        ln_alpha_planck = eng.interp2(T_cool2, Rho_cool2, planck2.T, np.log(t), np.log(d), 'linear', 0)
        ln_alpha_planck = np.array(ln_alpha_planck)[0]
        underflow_mask = np.logical_and(ln_alpha_rossland != 0.0, ln_alpha_planck != 0.0)
        d, t, r, ray_x, ray_y, ray_z, ln_alpha_rossland, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx = \
            make_slices([d, t, r, ray_x, ray_y, ray_z, ln_alpha_rossland, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx], underflow_mask)
        idx = np.array(idx)
        alpha_rossland = np.exp(ln_alpha_rossland) # [1/cm]
        alpha_planck = np.exp(ln_alpha_planck) # [1/cm]
        del ln_alpha_rossland, ln_alpha_planck
        gc.collect()

        # Optical Depth
        r_fuT = np.flipud(r) #.T
        alpha_rossland_fuT = np.flipud(alpha_rossland) 
        # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
        los = - np.flipud(sci.cumulative_trapezoid(alpha_rossland_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
        
        alpha_effective = np.sqrt(3 * alpha_planck * alpha_rossland) 
        alpha_effective_fuT = np.flipud(alpha_effective)
        los_effective = - np.flipud(sci.cumulative_trapezoid(alpha_effective_fuT, 
                                                         r_fuT, initial = 0)) * prel.Rsol_cgs
        los_effective[los_effective>30] = 30

        # Red 
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
        gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz) # projection of the gradient along the radial direction
        gc.collect()

        # Eq.(28) from Krumholz07.
        R_lamda = grad / ( prel.Rsol_cgs * alpha_rossland* ray_radDen) # this is the conversion for /r from the gradient. It's dimensionless
        R_lamda[R_lamda < 1e-10] = 1e-10
        # Eq.(27) from Krumholz07.
        fld_factor = (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
        # Eq.(26) from Krumholz07. You miss a c, but it's in Lphoto2 for computational reasons.
        # Before it was r.T
        smoothed_flux = -uniform_filter1d(r**2 * fld_factor * gradr / alpha_rossland, 7) #r^2 is here (but it's for the flux) otherwise you get annoying errors in the if. 
        Fx = - fld_factor * gradx / alpha_rossland
        Fy = - fld_factor * grady / alpha_rossland
        Fz = - fld_factor * gradz / alpha_rossland
        del gradx, grady, gradz

        # You can have numerical errors at early times
        try: 
            photosphere = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0] 
        except IndexError: # if you don't find the photosphere, skip the observer
            # num_obs -= 1 # you don't have light from there, but the observers are still 192
            print(f'No photosphere found for observer {i}', flush=True)
            # sys.stdout.flush()
            continue
        Lphoto2 = 4*np.pi * prel.c_cgs*smoothed_flux[photosphere] * prel.Msol_cgs / (prel.tsol_cgs**2) # you have to convert ray_radDen*r^2/lenght = energy/lenght^2 = mass/time^2
        if Lphoto2 < 0:
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives
        # free streaming emission
        max_length = 4*np.pi*(r[photosphere]**2) * prel.c_cgs * ray_radDen[photosphere] * prel.Msol_cgs * prel.Rsol_cgs / (prel.tsol_cgs**2) #the conversion is for ray_radDen*r^2 = mass*len/time^2
        Lphoto = np.min( [Lphoto2, max_length]) #that's usually Lphoto2
        ph_idx[i] = idx[photosphere]
        xph[i] = ray_x[photosphere]
        yph[i] = ray_y[photosphere]
        zph[i] = ray_z[photosphere]
        volph[i] = volume[photosphere]
        denph[i] = d[photosphere]
        Tempph[i] = t[photosphere]
        Rad_denph[i] = ray_radDen[photosphere]
        Vxph[i] = ray_vx[photosphere]
        Vyph[i] = ray_vy[photosphere]
        Vzph[i] = ray_vz[photosphere]
        Pressph[i] = ray_press[photosphere]
        IE_denph[i] = ray_ie_den[photosphere]
        rph[i] = r[photosphere] 
        alphaph[i] = alpha_rossland[photosphere]
        fluxes[i] = Lphoto / (4*np.pi*(r[photosphere]*prel.Rsol_cgs)**2)
        Lph[i] = Lphoto 
        Fxph[i] = Fx[photosphere]
        Fyph[i] = Fy[photosphere]
        Fzph[i] = Fz[photosphere]

        del smoothed_flux, R_lamda, fld_factor, ray_radDen
        gc.collect()

    Lphoto_snap = np.mean(Lph) # take the mean
    print(Lphoto_snap, flush=True)

    if save:
        # Save red of the single snap
        pre_saving = f'{abspath}/data/{folder}'
        
        with open(f'{pre_saving}/photo/{check}_photo{snap}.txt', 'w') as f:
            f.write('# Data for the photospere.\n')
            f.write('# xph\n' + ' '.join(map(str, xph)) + '\n')
            f.write('# yph\n' + ' '.join(map(str, yph)) + '\n')
            f.write('# zph\n' + ' '.join(map(str, zph)) + '\n')
            f.write('# volph\n' + ' '.join(map(str, volph)) + '\n')
            f.write('# denph CGS\n' + ' '.join(map(str, denph)) + '\n')
            f.write('# Tempph\n' + ' '.join(map(str, Tempph)) + '\n')
            f.write('# Rad_denph\n' + ' '.join(map(str, Rad_denph)) + '\n')
            f.write('# Vxph\n' + ' '.join(map(str, Vxph)) + '\n')
            f.write('# Vyph\n' + ' '.join(map(str, Vyph)) + '\n')
            f.write('# Vzph\n' + ' '.join(map(str, Vzph)) + '\n')
            f.write('# Pressph\n' + ' '.join(map(str, Pressph)) + '\n')
            f.write('# IE_denph\n' + ' '.join(map(str, IE_denph)) + '\n')
            f.write('# alpha CGS\n' + ' '.join(map(str, alphaph)) + '\n')
            f.write('# rph\n' + ' '.join(map(str, rph)) + '\n')
            f.write('# Lph CGS\n' + ' '.join(map(str, Lph)) + '\n')
            f.write('# indices\n' + ' '.join(map(str, ph_idx)) + '\n')
            f.write('# Fxph CGS\n' + ' '.join(map(str, Fxph)) + '\n')
            f.write('# Fyph CGS\n' + ' '.join(map(str, Fyph)) + '\n')
            f.write('# Fzph CGS\n' + ' '.join(map(str, Fzph)) + '\n')
            f.close()

    del xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph, Pressph, IE_denph, rph, alphaph, Lph, ph_idx, Fxph, Fyph, Fzph
    gc.collect()
        
eng.exit()
# usage = resource.getrusage(resource.RUSAGE_SELF)
# print(f"Peak RAM usage: {usage.ru_maxrss / 1024**2:.2f} MB")
#%%

# def compute_polarization(xph, yph, zph, dimph,
#                         Fxph, Fyph, Fzph,
#                         k_obs):
#     """
#     Compute polarization for a single observer direction k_obs.

#     Inputs:
#         xph, yph, zph : photosphere positions
#         Fxph, Fyph, Fzph : flux vector components at photosphere
#         k_obs : observer direction (3-vector, not necessarily normalized)

#     Returns:
#         P : polarization fraction
#         Q, U : Stokes parameters (normalized to total intensity)
#     """

#     # Number of surface elements
#     N = len(xph)

#     # Normalize observer direction
#     k = np.array(k_obs)
#     k = k / np.linalg.norm(k)

#     # Positions
#     r_vec = np.vstack((xph, yph, zph)).T # shape: (192,3)
#     r_mag = np.linalg.norm(r_vec, axis=1)
#     n_hat = r_vec / r_mag[:, None]   # surface normal (radial approx)

#     # Flux vectors
#     F_vec = np.vstack((Fxph, Fyph, Fzph)).T
#     F_mag = np.linalg.norm(F_vec, axis=1)

#     # Avoid division by zero
#     valid = F_mag > 0
#     F_hat = np.zeros_like(F_vec)
#     F_hat[valid] = F_vec[valid] / F_mag[valid][:, None]

#     # Visibility condition
#     mu_surface = np.dot(n_hat, k)
#     visible = mu_surface > 0

#     # Keep only visible patches
#     F_hat, F_mag, mu_surface, r_mag, dimph = make_slices([F_hat, F_mag, mu_surface, r_mag, dimph], visible)

#     # Scattering angle
#     cos_theta = np.dot(F_hat, k)

#     # Thomson polarization fraction
#     P_local = (1 - cos_theta**2) / (1 + cos_theta**2)

#     # --- Define sky plane basis vectors: k, e1, e2 ---
#     # Choose arbitrary reference axis not parallel to k to find e1
#     tmp = np.array([1.0, 0.0, 0.0])
#     if np.allclose(np.abs(np.dot(tmp, k)), 1.0):
#         tmp = np.array([0.0, 1.0, 0.0])
#     e1 = np.cross(k, tmp)
#     e1 /= np.linalg.norm(e1)
#     e2 = np.cross(k, e1)

#     # Polarization direction vector
#     # e_p ∝ k × (F × k)
#     cross1 = np.cross(F_hat, k)
#     e_pol = np.cross(k, cross1)

#     # Project polarization direction onto sky plane
#     e_pol_mag = np.linalg.norm(e_pol, axis=1)
#     nonzero = e_pol_mag > 0
#     e_pol[nonzero] /= e_pol_mag[nonzero][:, None]

#     # Compute cos(2phi) and sin(2phi)
#     cos_phi = np.dot(e_pol, e1)
#     sin_phi = np.dot(e_pol, e2)

#     cos2phi = cos_phi**2 - sin_phi**2
#     sin2phi = 2 * cos_phi * sin_phi

#     # Surface area weight (uniform sphere sampling assumption)
#     # dOmega = 4*np.pi / N
#     # dA = r_mag**2 * dOmega
#     dA = np.pi * dimph**2  

#     # Intensity weight
#     I_local = F_mag * mu_surface * dA

#     # Stokes parameters
#     Q = np.sum(I_local * P_local * cos2phi)
#     U = np.sum(I_local * P_local * sin2phi)
#     I = np.sum(I_local)

#     P = np.sqrt(Q**2 + U**2) / (I + 1e-20)

#     return P, Q/I, U/I

# photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo109POL.txt')
# xph, yph, zph, volph, Fxph, Fyph, Fzph = photo[0], photo[1], photo[2], photo[3], photo[-3], photo[-2], photo[-1]
# dimph = (volph)**(1/3)
# xph, yph, zph, dimph, Fxph, Fyph, Fzph = xph[xph!=0], yph[xph!=0], zph[xph!=0], dimph[xph!=0], Fxph[xph!=0], Fyph[xph!=0], Fzph[xph!=0]

# k_obs = [0, 0, 1]  # face-on observer
# P, Qnorm, Unorm = compute_polarization(
#     xph, yph, zph, dimph,
#     Fxph, Fyph, Fzph,
#     k_obs
# )

# print("Polarization:", P)




# # %%
