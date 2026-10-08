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
    save = False
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

from scipy.ndimage import uniform_filter1d
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
from Utilities.operators import make_tree

# observers 
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) # shape: (3, 192)
observers_xyz = np.array(observers_xyz).T # shape: (192, 3)


def compute_flux_for_Healp(loadpath, snap, observers_xyz, N_ray):
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

    all_radii = {}
    for i in range(num_obs):
        if i not in [0, 83, 90, 191]:
            continue
        print(f'Obs: {i}', flush=True)
        if plot:
            fig, (axFtheta, axFphi) = plt.subplots(1, 2, figsize = (16, 8))

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
        # r = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2) 

        alpha_scatter = calc_scattering_opacity_vectorized(T_cool, Rho_cool, scattering, np.log(t), np.log(d))
        alpha_scatter = np.array(alpha_scatter)
        alpha_rossland = calc_ross_opacity_vectorized(T_cool, Rho_cool, rossland, scattering, np.log(t), np.log(d))
        alpha_rossland = np.array(alpha_rossland)
        alpha_planck = calc_planck_opacity_vectorized(T_cool, Rho_cool, planck, np.log(t), np.log(d))
        alpha_planck = np.array(alpha_planck)
        
        underflow_mask = np.logical_and(np.logical_and(np.log(alpha_rossland) != 0.0, np.log(alpha_planck) != 0.0), np.log(alpha_scatter) != 0.0)
        d, t, r, ray_x, ray_y, ray_z, alpha_rossland, alpha_planck, alpha_scatter, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx = \
            make_slices([d, t, r, ray_x, ray_y, ray_z, alpha_rossland, alpha_planck, alpha_scatter, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx], underflow_mask)
        idx = np.array(idx)

        # Optical depth
        r_fuT = np.flipud(r) #.T
        alpha_rossland_fuT = np.flipud(alpha_rossland) 
        # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
        los = - np.flipud(sci.cumulative_trapezoid(alpha_rossland_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r

        alpha_scatter_fuT = np.flipud(alpha_scatter) 
        # los_scatt = - np.flipud(sci.cumulative_trapezoid(alpha_scatter_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
        
        alpha_effective = np.sqrt(3 * np.minimum(alpha_planck, alpha_rossland) * alpha_rossland)
        alpha_effective_fuT = np.flipud(alpha_effective)
        los_effective = - np.flipud(sci.cumulative_trapezoid(alpha_effective_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs
        los_effective[los_effective > 30] = 30
        del r_fuT, alpha_rossland_fuT, alpha_scatter_fuT, alpha_effective_fuT

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
        del gradx, grady, gradz, R_lamda, fld_factor

        try: 
            photo_idx = np.where( ((smoothed_flux_r2>0) & (los<2/3) ))[0][0] 
        except IndexError: # if you don't find the photosphere, skip the observer
            print(f'No photosphere found for observer {i}', flush=True)
            continue

        radii = {'x': ray_x, 
                 'y': ray_y, 
                 'z': ray_z, 
                 'r': r,
                 'den': d,
                 'temp': t, 
                 'radden': ray_radDen, 
                 'tau': los,
                 'alpha_rossland': alpha_rossland, 
                 'alpha_scatter': alpha_scatter, 
                 'alpha_abs': alpha_planck, 
                 'Fx': Fx, 
                 'Fy': Fy, 
                 'Fz': Fz}

        if plot:
            Fmag = np.sqrt(Fx**2 + Fy**2 + Fz**2)
            theta_F = np.arccos(Fz/Fmag) 
            phi_F = np.arctan2(Fy, Fx)
            axFtheta.scatter(r/13, theta_F, c = np.log10(Fmag), s = 10, cmap = 'rainbow', vmin = 11, vmax = 17)
            img = axFphi.scatter(r/13, phi_F, c = np.log10(Fmag), s = 10, cmap = 'rainbow', vmin = 11, vmax = 17)
            cbar = plt.colorbar(img, ax = axFtheta)
            axFtheta.set_xlabel(r'r/r$_{\rm t}$')
            axFtheta.set_ylabel(r'$\theta_F$')
            axFphi.set_xlabel(r'r/r$_{\rm t}$')
            axFphi.set_ylabel(r'$\phi_F$')
            for ax in [axFtheta, axFphi]:
                ax.set_xscale('log')
                ax.tick_params(axis='both', which='major', length = 10, width = 1)
                ax.tick_params(axis='both', which='minor', length = 6, width = .8)
                ax.grid()
                ax.axvline(r[photo_idx]/13, ls = '-.')
                ax.set_xlim(1, 8e2)
            plt.suptitle(f'Obs: {i}, Snap: {snap}', fontsize = 22)
            plt.tight_layout()

        all_radii[f'{i}'] = radii

    return all_radii

def check_constant_flux_along_patch(loadpath, snap, n_samples=50):
    """
    Check whether FLD flux is approximately constant along the
    local flux direction around one photospheric patch.

    s is given in the same length units as X,Y,Z.

    Negative s = inward relative to Fhat_ph
    Positive s = outward relative to Fhat_ph
    """
    data = make_tree(loadpath, snap)
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Rad, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE
    denmask = Den > 1e-19
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        make_slices([X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den], denmask)
    xyz = np.array([X, Y, Z]).T
    tree = KDTree(xyz, leaf_size=50) 

    # Photospheric position and flux
    # --------------------------------------------------
    photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
    x_ph_all, y_ph_all, z_ph_all, Fx_ph_all, Fy_ph_all, Fz_ph_all = photo['x'], photo['y'], photo['z'], photo['Fx'], photo['Fy'], photo['Fz']

    all_radii = {}
    for i in range(len(x_ph_all)):
        # if i not in [0]:
        #     continue
        x_ph, y_ph, z_ph, Fx_ph, Fy_ph, Fz_ph = x_ph_all[i], y_ph_all[i], z_ph_all[i], Fx_ph_all[i], Fy_ph_all[i], Fz_ph_all[i]
        r_ph = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
        r0 = np.array([x_ph, y_ph, z_ph], dtype=float)
        s_min = 1e-2 * r_ph
        s_max = 3 * r_ph
        F0 = np.array([Fx_ph, Fy_ph, Fz_ph], dtype=float)
        F0_mag = np.linalg.norm(F0)
        # Local plane-parallel direction
        nhat = F0 / F0_mag
 
        # Sampling positions: you have to go backwords as well
        s = np.logspace(np.log10(s_min), np.log10(s_max), int(n_samples/2))
        s_backfor = np.concatenate((-s[::-1], s))
        # Positions along the ray
        positions = r0[None, :] + s_backfor[:, None] * nhat[None, :]
        ray_r = np.linalg.norm(positions, axis=1)
        # Monotonic, non-negative path coordinate along the ray
        # This is the coordinate to use for optical-depth integration.
        # ray_s[0] = 0 and then increases along the sampled ray.

        ds = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        ray_s = np.concatenate(([0.0], np.cumsum(ds)))

        # Recompute FLD flux
        _, idx = tree.query(positions, k=1)
        idx = idx.ravel()
        # idx = np.unique(idx).astype(np.int64) #####
        # Quantity corresponding to the ray
        t = T[idx]
        d = Den[idx] * prel.den_converter
        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        volume = Vol[idx]
        ray_radDen = Rad_den[idx]

        alpha_scatter = calc_scattering_opacity_vectorized(T_cool, Rho_cool, scattering, np.log(t), np.log(d))
        alpha_scatter = np.array(alpha_scatter)
        alpha_rossland = calc_ross_opacity_vectorized(T_cool, Rho_cool, rossland, scattering, np.log(t), np.log(d))
        alpha_rossland = np.array(alpha_rossland)
        alpha_planck = calc_planck_opacity_vectorized(T_cool, Rho_cool, planck, np.log(t), np.log(d))
        alpha_planck = np.array(alpha_planck)
        
        underflow_mask = np.logical_and(np.logical_and(np.log(alpha_rossland) != 0.0, np.log(alpha_planck) != 0.0), np.log(alpha_scatter) != 0.0)
        t, d, ray_x, ray_y, ray_z, ray_r, alpha_rossland, alpha_scatter, ray_radDen, volume, idx = \
            make_slices([t, d, ray_x, ray_y, ray_z, ray_r, alpha_rossland, alpha_scatter, ray_radDen, volume, idx], underflow_mask)
        idx = np.array(idx)

        # Optical depth
        s_fuT = np.flipud(ray_s) #.T
        alpha_rossland_fuT = np.flipud(alpha_rossland) 
        # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
        los = - np.flipud(sci.cumulative_trapezoid(alpha_rossland_fuT, s_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
        

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

        Fvec = np.column_stack((Fx, Fy, Fz))
        Fmag = np.linalg.norm(Fvec, axis=1)
        F_parallel = Fvec @ nhat

        # Directional alignment with the original photospheric F
        cos_F = (F_parallel/ np.maximum(Fmag, 1e-30))
        cos_F = np.clip(cos_F, -1.0, 1.0)
        angle_F = np.arccos(cos_F)

        Fparallel_rel = (F_parallel / F0_mag)
        Fmag_rel = (Fmag / F0_mag)

        # --------------------------------------------------
        # Simple diagnostic numbers

        valid = Fmag > 0
        if np.any(valid):
            variation_parallel = (np.nanmax(F_parallel[valid])
                            - np.nanmin(F_parallel[valid])) / np.maximum(
                            np.nanmedian(np.abs(F_parallel[valid])), 1e-30)

            max_angle_deg = np.degrees(np.nanmax(angle_F[valid]))

        else:
            variation_parallel = np.nan
            max_angle_deg = np.nan

        # Plot
        # if plot:
        #     plt.figure(figsize=(8,6))
        #     plt.plot(s_backfor/r_ph, Fparallel_rel, label=r'$F_\parallel/F_{\rm ph}$')
        #     plt.plot(s_backfor/r_ph, Fmag_rel, ls='dashed', label=r'$|F|/F_{\rm ph}$')
        #     plt.xlabel(r'$s$ along $\hat{ F}_{\rm ph} (r_{\rm ph})$')
        #     plt.ylabel(r'$F/F_{\rm ph}$')
        #     # plt.loglog()
        #     plt.legend(fontsize = 16)
        #     plt.title(f'Obs: {i}, relative variation in F_parallel = {variation_parallel:.2f}', fontsize = 16)
        #     plt.tight_layout()


        #     plt.figure(figsize=(8,6))
        #     plt.plot(s_backfor/r_ph, np.degrees(angle_F))
        #     plt.xlabel(r'$s$ along $\hat{F}_{\rm ph} (r_{\rm ph})$')
        #     # plt.xscale('log')
        #     plt.ylabel(r'angle$(F(s),' r'F_{\rm ph})$ [deg]')
        #     plt.title(f'Obs: {i}, max angle = {max_angle_deg:.2f} deg', fontsize = 16)
        #     plt.tight_layout()
        #     plt.show()

        normal_radii =  {
                        "s": s_backfor,
                        "temp": t,
                        "den": d,
                        "ray_r": ray_r,
                        "positions": positions,
                        "radden": ray_radDen,
                        "Fx": Fx,
                        "Fy": Fy,
                        "Fz": Fz,
                        "tau": los, 
                        "alpha_rossland": alpha_rossland,
                        "alpha_scatter": alpha_scatter,
                        "alpha_abs": alpha_planck,
                        }
        all_radii[f'{i}'] = normal_radii

    return all_radii
                        
if __name__ == "__main__":
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'HiResNewAMR' 
    params = [m, Rstar, mstar, beta, n, compton]
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    snap = 109
    
    #%% Load opacity tables
    opac_path = f'{abspath}/src/Opacity'
    T_cool = np.loadtxt(f'{opac_path}/T.txt')
    Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
    rossland = np.loadtxt(f'{opac_path}/ross.txt')
    planck = np.loadtxt(f'{opac_path}/planck.txt')
    scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm
    pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
    pre_saving = f'{abspath}/data/{folder}'
    if alice:
        loadpath = f'{pre}/snap_{snap}'
    else:
        loadpath = f'{pre}/{snap}'

    photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
    #%%
    radii_flux = check_constant_flux_along_patch(
                        loadpath, snap,
                        n_samples = 100)
    np.savez(f'{abspath}/data/{folder}/flux_along_patch{snap}.npz', **radii_flux)
    #%%
    all_radii = np.load(f'{abspath}/data/{folder}/flux_along_patch{snap}.npz', allow_pickle=True)    
    keys = list(all_radii.keys())
    #%%
    for key in keys:
        normal_radii = all_radii[key].item()
        key = int(key)
        if key not in [0, 20, 50, 90, 120, 190]:
            continue
        x_ph, y_ph, z_ph, Fx_ph, Fy_ph, Fz_ph = photo['x'][key], photo['y'][key], photo['z'][key], photo['Fx'][key], photo['Fy'][key], photo['Fz'][key]
        r_ph = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
        r_ph_vec = np.array([x_ph, y_ph, z_ph])
        F_ph_vec = np.array([Fx_ph, Fy_ph, Fz_ph])
        F_ph_mag = np.linalg.norm(F_ph_vec)
        # print(np.dot(F_ph_vec, r_ph_vec)/(F_ph_mag*r_ph))
        s, ray_r, temp, Fx, Fy, Fz, radden = normal_radii['s'], normal_radii['ray_r'], normal_radii['temp'], normal_radii['Fx'], normal_radii['Fy'], normal_radii['Fz'], normal_radii['radden']
        Fvec = np.column_stack((Fx, Fy, Fz))
        Fmag = np.linalg.norm(Fvec, axis=1)
        F_ph_hat = F_ph_vec / F_ph_mag
        F_parallel = Fvec @ F_ph_hat

        # Directional alignment with the original photospheric F
        cos_F = (F_parallel/ np.maximum(Fmag, 1e-30))
        cos_F = np.clip(cos_F, -1.0, 1.0)
        angle_F = np.arccos(cos_F)

        Fparallel_rel = (F_parallel / F_ph_mag)
        Fmag_rel = (Fmag / F_ph_mag)  

        variation_parallel = (np.nanmax(F_parallel)
                        - np.nanmin(F_parallel)) / np.maximum(
                        np.nanmedian(np.abs(F_parallel)), 1e-30)

        max_angle_deg = np.degrees(np.nanmax(angle_F))

        plt.figure(figsize=(8,6))
        plt.plot(s/r_ph, Fparallel_rel, label=r'$F_\parallel/F_{\rm ph}$')
        plt.plot(s/r_ph, radden*prel.en_den_converter*prel.c_cgs/F_ph_mag, label=r'$cu_{\rm rad}/F_{\rm ph}$')
        plt.plot(s/r_ph, Fmag_rel, ls='dashed', label=r'$|F|/F_{\rm ph}$')
        plt.xlabel(r'$s$ along $\hat{ F}_{\rm ph} (r_{\rm ph})$')
        plt.ylabel(r'$F/F_{\rm ph}$')
        plt.yscale('log')
        # plt.xscale('symlog', linthresh=1e-1)
        plt.ylim(1e-2, 10)
        plt.legend(fontsize = 16)
        plt.tick_params(axis='both', which='major', length = 10, width = 1)
        plt.tick_params(axis='both', which='minor', length = 6, width = .8)
        plt.title(f'Obs: {key}, relative variation in F_parallel = {variation_parallel:.2f}', fontsize = 16)
        plt.tight_layout()

        # plt.figure(figsize=(8,6))
        # plt.plot(s/r_ph, np.degrees(angle_F))
        # plt.xlabel(r'$s$ along $\hat{F}_{\rm ph} (r_{\rm ph})$')
        # plt.xscale('symlog', linthresh=1e-1)
        # plt.ylabel(r'angle$(F(s),' r'F_{\rm ph})$ [deg]')
        # plt.title(f'Obs: {key}, max angle = {max_angle_deg:.2f} deg', fontsize = 16)
        # plt.tight_layout()
        # plt.show()

# %%
