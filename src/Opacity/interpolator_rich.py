abspath = '/Users/paolamartire/shocks'
import sys

from matplotlib.colors import LogNorm
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
import warnings
warnings.filterwarnings('ignore')
import csv

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
from scipy.ndimage import uniform_filter1d

import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
import src.orbits as orb
from Utilities.operators import make_tree

def bilinear_interpolation(x_vec, y_vec, data, x, y):
    """
    Bilinear interpolation inside a 2D table.
    data[i][j] corresponds to x_vec[i], y_vec[j].
    Clamps x and y to grid bounds.
    """
    x_vec = np.asarray(x_vec)
    y_vec = np.asarray(y_vec)
    data  = np.asarray(data)

    # Clamp to grid: force x and y to be within the bounds of x_vec and y_vec
    x = np.clip(x, x_vec[0], x_vec[-1])
    y = np.clip(y, y_vec[0], y_vec[-1])

    i = np.searchsorted(x_vec, x, side='right') - 1 # searchsorted returns the index of the first element in x_vec that is greater than x
    j = np.searchsorted(y_vec, y, side='right') - 1

    i = int(np.clip(i, 0, len(x_vec) - 2))
    j = int(np.clip(j, 0, len(y_vec) - 2))

    x0, x1 = x_vec[i], x_vec[i + 1]
    y0, y1 = y_vec[j], y_vec[j + 1]

    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)

    return (data[i,     j    ] * (1 - tx) * (1 - ty) +
            data[i + 1, j    ] * tx       * (1 - ty) +
            data[i,     j + 1] * (1 - tx) * ty       +
            data[i + 1, j + 1] * tx       * ty)

def interpolate_2d_table(x_vec, y_vec, data, x, y, x_vec_high_slope = 0, slope_length = 7 ):
    """
    2D extrapolation-aware table interpolation (mirrors C++ Interpolate2DTable).

    Parameters
    ----------
    x_vec, y_vec     : 1D grid arrays (must be longer than slope_length)
    data             : 2D array, data[i][j] at (x_vec[i], y_vec[j])
    x, y             : query point (scalars)
    x_vec_high_slope : slope used when x > x_vec[-1]
    slope_length     : position of point used to estimate extrapolation slopes

    Returns
    -------
    float
    """
    x_vec = np.asarray(x_vec)
    y_vec = np.asarray(y_vec)
    data  = np.asarray(data)

    if len(x_vec) <= slope_length or len(y_vec) <= slope_length:
        raise ValueError(
            "interpolate_2d_table: x_vec or y_vec is not longer than slope_length"
        )

    # --- T below grid ---
    if x < x_vec[0]:
        if y < y_vec[0]:
            x_slope = (data[slope_length - 1, 0] - data[0, 0]) / (x_vec[slope_length - 1] - x_vec[0])
            y_slope     = (data[0, slope_length - 1] - data[0, 0]) / (y_vec[slope_length - 1] - y_vec[0])
            interp_val = np.exp(data[0, 0] + y_slope * (y - y_vec[0]) + x_slope * (x - x_vec[0]))
            return interp_val, x_slope, y_slope
        else:
            data_x0     = bilinear_interpolation(x_vec, y_vec, data, x_vec[0] * 1.00001, y)
            x_slope = (bilinear_interpolation(x_vec, y_vec, data, x_vec[slope_length - 1], y) - data_x0) / \
                          (x_vec[slope_length - 1] - x_vec[0])
            interp_val =  np.exp(data_x0 + x_slope * (x - x_vec[0]))
            return interp_val, x_slope, 0

    # --- x above grid ---
    if x > x_vec[-1]:
        if y < y_vec[0]:
            y_slope = (data[-1, slope_length - 1] - data[-1, 0]) / (y_vec[slope_length - 1] - y_vec[0])
            interp_val =  np.exp(data[-1, 0] + y_slope * (y - y_vec[0]) + x_vec_high_slope * (x - x_vec[-1]))
            return interp_val, x_vec_high_slope, y_slope
        else:
            interp_val = np.exp(
                bilinear_interpolation(x_vec, y_vec, data, x_vec[-1] * 0.99999, y)
                + x_vec_high_slope * (x - x_vec[-1]))
            return interp_val, x_vec_high_slope, 0

    # --- y below grid (x is within bounds) ---
    if y < y_vec[0]:
        data_y0 = bilinear_interpolation(x_vec, y_vec, data, x, y_vec[0] * 0.9999)
        y_slope = (bilinear_interpolation(x_vec, y_vec, data, x, y_vec[slope_length - 1]) - data_y0) / \
                  (y_vec[slope_length - 1] - y_vec[0])
        interp_val =  np.exp(data_y0 + y_slope * (y - y_vec[0]))
        return interp_val, 0, y_slope

    # --- fully within grid ---
    interp_val = np.exp(bilinear_interpolation(x_vec, y_vec, data, x, y))
    return interp_val, 0, 0

def calc_scattering_opacity(T_, rho_, scatter_, Tcell, rhocell, return_coeff = False):
    """
    Parameters
    ----------
    T_, rho_    : 1D grid arrays
    scatter_    : 2D opacity table
    Tcell, rhocell  : ln(temperature) and ln(density) of query point
    """
    T_      = np.asarray(T_)
    rho_    = np.asarray(rho_)
    scatter_= np.asarray(scatter_)

    d       = rhocell
    d_ratio = 1.0

    if d < rho_[0]:
        d_ratio = np.exp(rhocell) / np.exp(rho_[0])
        d       = rho_[0]
        interp_val, T_slope, _ = interpolate_2d_table(T_, rho_, scatter_, Tcell, d) 
        scatter = interp_val * d_ratio
        if return_coeff:
            return scatter, T_slope, 1
        return scatter

    if d > rho_[-1]:
        d_ratio = np.exp(rhocell) / np.exp(rho_[-1])
        d       = rho_[-1]
        interp_val, T_slope, _ = interpolate_2d_table(T_, rho_, scatter_, Tcell, d)
        scatter = interp_val * d_ratio
        if return_coeff:
            return scatter, T_slope, 1
        return scatter
    
    interp_val, T_slope, d_slope = interpolate_2d_table(T_, rho_, scatter_, Tcell, d) 
    scatter = interp_val * d_ratio

    if return_coeff:
        return scatter, T_slope, d_slope
    return scatter

def calc_ross_opacity(T_, rho_, rossland_, scatter_, Tcell, rhocell, return_coeff = False):
    """
    Parameters
    ----------
    T_, rho_              : 1D grid arrays
    rossland_             : 2D opacity table
    Tcell, rhocell        : ln(temperature) and ln(density) of query point 
    """
    T_   = np.asarray(T_)
    rho_ = np.asarray(rho_)

    d = rhocell
    d_ratio = 1.0

    if d < rho_[0]:
        scattering, Tscatt_slope, dscatt_slope = calc_scattering_opacity(T_, rho_, scatter_, Tcell, d, return_coeff=True)
        sigma_rossland, T_slope, d_slope = interpolate_2d_table(T_, rho_, rossland_, Tcell, d)
        if sigma_rossland > scattering:
            if return_coeff:
                return sigma_rossland, T_slope, d_slope
            return sigma_rossland
        else:
            if return_coeff:
                return scattering, Tscatt_slope, dscatt_slope
            return scattering

    if d > rho_[-1]:
        d_ratio = np.exp(rhocell) / np.exp(rho_[-1])
        d       = rho_[-1]
        interp_val, T_slope, _ = interpolate_2d_table(T_, rho_, rossland_, Tcell, d) 
        rossland = interp_val * d_ratio
        if return_coeff:
            return rossland, T_slope, 1
        return rossland

    interp_val, T_slope, d_slope = interpolate_2d_table(T_, rho_, rossland_, Tcell, d) 
    rossland = interp_val * d_ratio

    if return_coeff:
        return rossland, T_slope, d_slope
    return rossland

def calc_planck_opacity(T_, rho_, planck_, Tcell, rhocell, return_coeff = False):
    """
    Calculate Planck opacity for a computational cell.

    Parameters
    ----------
    T_           : 1D array of log-temperature grid points
    rho_         : 1D array of log-density grid points
    planck_      : 2D array of shape (len(T_), len(rho_))
    Tcell      : float, ln-temperature of the cell
    rhocell    : float, ln-density of the cell
    """
    T_      = np.asarray(T_)
    rho_    = np.asarray(rho_)
    planck_ = np.asarray(planck_)

    d_ratio = 1.0
    d_slope = 2.0
    d_log = rhocell

    if rhocell < rho_[0]:
        if T_[0] < Tcell < T_[-1]:
            idx     = np.searchsorted(T_, Tcell)   # lower_bound equivalent
            d_slope = (planck_[idx, 10] - planck_[idx, 0]) / (rho_[10] - rho_[0])
            if not (0.35 <= d_slope <= 2.75):
                raise ValueError(
                    f"Planck opacity interpolation failed "
                    f"(slope={d_slope:.2f}, T={Tcell:.2f}, idx={idx})"
                )
        d_ratio = np.exp(rhocell) / np.exp(rho_[0])
        d_log   = rho_[0]

    elif rhocell > rho_[-1]:
        d_ratio = np.exp(rhocell) / np.exp(rho_[-1])
        d_log   = rho_[-1]

    else:
        d_slope = 0 # doesn't really matter beacuse d_ratio=1 in this case, but it's just for when you plot d_slope: since you're inside the table you don't extrapolate so d_sloep=0

    interp_val, T_slope, _ = interpolate_2d_table(
        T_, rho_, planck_, Tcell, d_log, 
        x_vec_high_slope=-3.5) #d_slope would be 0 beacuse d_log in the table
    
    planck = interp_val * (d_ratio ** d_slope)

    if return_coeff:
        return planck, T_slope, d_slope
    return planck

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
        if snap not in [151]: # for testing
            continue
        print(f'Snap: {snap}', flush=True)
        if alice:
            loadpath = f'{pre}/snap_{snap}'
        else:
            loadpath = f'{pre}/{snap}'
        Lphoto_snap, photosphere, colorsphere, freqs, L_col = single_fld(loadpath, snap, observers_xyz, N_ray, tfb[idx_s])

        del Lphoto_snap, photosphere, colorsphere, L_col
        gc.collect()

def single_fld(loadpath, snap, observers_xyz, N_ray, time):
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

    photosphere = {'idx': [], 'x': [], 'y': [], 'z': [], 'vol': [], 'den': [], 'temp': [], 'radden': [], 'vx': [], 'vy': [], 'vz': [], 'P': [], 'ieden': [], 'alpha_rossland': [], 'alpha_scatter': [], 'alpha_abs': [], 'r': [], 'Lum': [], 'Fx': [], 'Fy': [], 'Fz': []}
    colorsphere = {'idx': [], 'x': [], 'y': [], 'z': [], 'vol': [], 'den': [], 'temp': [], 'radden': [], 'vx': [], 'vy': [], 'vz': [], 'P': [], 'ieden': [], 'alpha_rossland': [], 'alpha_scatter': [], 'alpha_abs': [], 'alpha_eff': []}
    freqs = prel.freqs
    L_col = np.zeros((num_obs, len(prel.freqs)))
    for i in range(num_obs):
        if i not in [10, 100]:
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
        # r = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2) ####
        
        alpha_scatter = np.zeros_like(t)
        alpha_rossland = np.zeros_like(t)
        A_ross = np.zeros_like(t)
        B_ross = np.zeros_like(t)
        alpha_planck = np.zeros_like(t)
        A_planck = np.zeros_like(t)
        B_planck = np.zeros_like(t)
        for k in range(len(t)):
            alpha_scatter[k], _, _ = calc_scattering_opacity(T_cool, Rho_cool, scattering, np.log(t[k]), np.log(d[k]))
            alpha_rossland[k], A_ross[k], B_ross[k] = calc_ross_opacity(T_cool, Rho_cool, rossland, scattering, np.log(t[k]), np.log(d[k]), return_coeff=True)
            alpha_planck[k], A_planck[k], B_planck[k] = calc_planck_opacity(T_cool, Rho_cool, planck, np.log(t[k]), np.log(d[k]), return_coeff=True)
            # if alpha_planck[k] > 100.0 / (prel.c_cgs * prel.tsol_cgs * time):
            #     print('Change Planck') 
            #     alpha_planck[k] = 100.0 / (prel.c_cgs * prel.tsol_cgs * time)
        # alpha_rossland = alpha_planck + alpha_scatter
        ln_alpha_planck = np.log(alpha_planck)
        ln_alpha_rossland = np.log(alpha_rossland)
        ln_alpha_scatter = np.log(alpha_scatter)
        underflow_mask = np.logical_and(np.logical_and(ln_alpha_rossland != 0.0, ln_alpha_planck != 0.0), ln_alpha_scatter != 0.0)
        d, t, r, ray_x, ray_y, ray_z, alpha_rossland, alpha_planck, alpha_scatter, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx = \
            make_slices([d, t, r, ray_x, ray_y, ray_z, alpha_rossland, alpha_planck, alpha_scatter, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx], underflow_mask)
        idx = np.array(idx)
        # alpha_planck = np.exp(ln_alpha_planck) # [1/cm]
        # for al in alpha_planck:
        # del ln_alpha_rossland, ln_alpha_planck, ln_alpha_scatter
        # gc.collect()

        # Optical depth
        r_fuT = np.flipud(r) #.T
        alpha_rossland_fuT = np.flipud(alpha_rossland) 
        # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
        los = - np.flipud(sci.cumulative_trapezoid(alpha_rossland_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
        
        # alpha_effective = np.sqrt(3 * alpha_planck * (alpha_planck + alpha_scatter)) 
        alpha_effective = np.sqrt(3 * np.minimum(alpha_planck, alpha_rossland) * alpha_rossland)
        print(np.minimum(alpha_planck, alpha_rossland))
        alpha_effective_fuT = np.flipud(alpha_effective)
        los_effective = - np.flipud(sci.cumulative_trapezoid(alpha_effective_fuT, 
                                                         r_fuT, initial = 0)) * prel.Rsol_cgs
        los_effective[los_effective > 30] = 30

        # FLD curve 
        # Get 20 unique nearest neighbors to each cell in the wanted ray and use them to compute the gradient along the ray
        xyz3 = np.array([ray_x, ray_y, ray_z]).T
        _, idxnew = tree.query(xyz3, k = 20)
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
            photo_idx = np.where( ((smoothed_flux_r2>0) & (los<2/3) ))[0][0] 
        except IndexError: # if you don't find the photosphere, skip the observer
            print(f'No photosphere found for observer {i}', flush=True)
            continue
        # Lphoto2 = 4*np.pi * smoothed_flux_r2[photosphere] * prel.Msol_cgs / (prel.tsol_cgs**2) # you have to convert ray_radDen*r^2/lenght = energy/lenght^2 = mass/time^2
        Lphoto2 = 4*np.pi * smoothed_flux_r2[photo_idx] * prel.Rsol_cgs**2 # you have to convert the r^2 in smoothed_flux_r2
        if Lphoto2 < 0: 
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives
        # free streaming emission
        max_length = 4*np.pi*(r[photo_idx]**2) * prel.c_cgs * ray_radDen[photo_idx] * prel.Msol_cgs * prel.Rsol_cgs / (prel.tsol_cgs**2) #the conversion is for ray_radDen*r^2 = mass*len/time^2
        Lphoto = np.min( [Lphoto2, max_length]) #that's usually Lphoto2
        photosphere['idx'].append(idx[photo_idx])
        photosphere['x'].append(ray_x[photo_idx])
        photosphere['y'].append(ray_y[photo_idx])
        photosphere['z'].append(ray_z[photo_idx])
        photosphere['vol'].append(volume[photo_idx])
        photosphere['den'].append(d[photo_idx])
        photosphere['temp'].append(t[photo_idx])
        photosphere['radden'].append(ray_radDen[photo_idx])
        photosphere['vx'].append(ray_vx[photo_idx])
        photosphere['vy'].append(ray_vy[photo_idx])
        photosphere['vz'].append(ray_vz[photo_idx])
        photosphere['P'].append(ray_press[photo_idx])
        photosphere['ieden'].append(ray_ie_den[photo_idx])
        photosphere['alpha_rossland'].append(alpha_rossland[photo_idx])
        photosphere['alpha_scatter'].append(alpha_scatter[photo_idx])
        photosphere['alpha_abs'].append(alpha_planck[photo_idx])
        photosphere['r'].append(r[photo_idx])
        photosphere['Fx'].append(Fx[photo_idx])
        photosphere['Fy'].append(Fy[photo_idx])
        photosphere['Fz'].append(Fz[photo_idx])
        photosphere['Lum'].append(Lphoto) # fluxes was from here as L/4pi r[photo)idx]**2

        # Spectra
        # color_idx = np.argmin(np.abs(los_effective-5))
        try:
            color_idx = np.where(los_effective<5)[0][0]
        except IndexError: # if you don't find the photosphere, skip the observer
            print(f'No thermalization radius found for observer {i}', flush=True)
            continue
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
        colorsphere['ieden'].append(ray_ie_den[color_idx])
        colorsphere['alpha_rossland'].append(alpha_rossland[color_idx])
        colorsphere['alpha_scatter'].append(alpha_scatter[color_idx])
        colorsphere['alpha_abs'].append(alpha_planck[color_idx])
        colorsphere['alpha_eff'].append(alpha_effective[color_idx])

        # Spectra ---
        for k in range(color_idx, len(r)):
            # if k == 0:
            #     continue
            dr = r[k]-r[k-1]
            Vcell =  r[k]**2 * dr # there should be a (4 * np.pi / 192)*, but doesn't matter because we normalize
            wien = np.exp(prel.h_cgs * freqs / (prel.Kb_cgs * t[k])) - 1
            black_body = freqs**3 / wien # There should be a 2 * prel.h_cgs/c^2, but it doesn't matter because we normalize. BB udm: erg/s/cm^2/Hz/ster. 
            L_col[i,:] += alpha_planck[k] * Vcell * np.exp(-los_effective[k]) * black_body # erg/s/Hz.
        
        norm = Lphoto / np.trapezoid(L_col[i,:], freqs)
        L_col[i,:] *= norm

        if plot:
            apo = 330
            trad = (ray_radDen * prel.en_den_converter/prel.alpha_cgs)**(1/4)
            kappa_ross = alpha_rossland/d
            kappa_planck = alpha_planck/d
            kappa_scatter = alpha_scatter/d
            
            figk, axk = plt.subplots(figsize = (10, 6))
            axk.plot(r/apo, kappa_ross, c = 'dodgerblue', label = r'$\kappa_{\rm R}$')
            axk.plot(r/apo, kappa_planck, c = 'r', label = r'$\kappa_{\rm a}$')
            axk.plot(r/apo, kappa_scatter, c = 'forestgreen', label = r'$\sigma_{\rm s}$')
            axk.set_ylabel(r'$\kappa$ [cm$^2$/g]')
            axk.set_ylim(1e-2, 1e3)
            axk.legend(fontsize = 14)

            fig, ((axal, axtau), (axA, axB), (axT, axd)) = plt.subplots(3, 2, figsize = (15, 21))
            axal.plot(r/apo, alpha_rossland, c = 'dodgerblue', label = r'$\alpha_{\rm R}$')
            axal.plot(r/apo, alpha_planck, c = 'r', label = r'$\alpha_{\rm a}$')
            axal.plot(r/apo, alpha_scatter, c = 'forestgreen', label = r'$\sigma_{\rm s}$')
            axal.set_ylabel(r'$\alpha$ [1/cm]')
            axal.set_ylim(1e-15, 1e-4)
            axal.legend(fontsize = 18)

            axtau.plot(r/apo, los, c = 'dodgerblue', label = r'$\tau_{\rm R}$')
            axtau.axhline(2/3, c = 'gray', ls = '--')
            axtau.plot(r/apo, los_effective, c = 'r', label = r'$\tau_{\rm eff}$')
            axtau.axhline(5, c = 'gray', ls = '--')
            axtau.set_ylabel(r'$\tau$')
            axtau.set_ylim(1e-4, 1e2)

            axT.plot(r/apo, t, c = 'k', label = r'T')
            axT.plot(r/apo, trad, ls = '--', c = 'gray', label = r'T$_{\rm rad}$')
            axT.set_ylabel(r'T [K]')
            axT.set_ylim(1e3, 1e8)
            axT.axhspan(min_T, max_T, color = 'gold', alpha = 0.2, label = 'table range')

            axd.plot(r/apo, d, c = 'k')
            axd.set_ylabel(r'Den [g/cm$^3$]')
            axd.set_ylim(1e-17, 1e9)
            axd.axhspan(min_Rho, max_Rho, color = 'gold', alpha = 0.2)

            axA.scatter(r/apo, A_ross, c = 'dodgerblue', s = 50, label = r'$\Delta \ln\alpha_{\rm R}/\Delta \ln T$')
            axA.scatter(r/apo, A_planck, c = 'r', s = 25, label = r'$\Delta \ln\alpha_{\rm a}/\Delta \ln T$')
            axA.set_ylabel(r'coeff extrap T')
            # axA.set_ylim(-10, 10)

            axB.scatter(r/apo, B_ross, c = 'dodgerblue', s = 50, label = r'$\Delta \ln\alpha_{\rm R}/\Delta \ln \rho$')
            axB.scatter(r/apo, B_planck, c = 'r', s = 25, label = r'$\Delta \ln\alpha_{\rm a}/\Delta \ln \rho$')
            axB.set_ylabel(r'coeff extrap $\rho$')
            # axB.set_ylim(-10, 10)

            for ax in [axal, axtau, axd, axT, axA, axB, axk]:
                ax.set_xlim(5e-2, 15)
                if ax in [axA, axB]:
                    ax.set_xscale('log')
                else:
                    ax.loglog()
                ax.axvline(r[photo_idx]/apo, ls = '--', c = 'dodgerblue', label = f'obs {i}, ' +  r'$r_{\rm ph}$')
                ax.axvline(r[color_idx]/apo, ls = '--', c = 'r', label = f'obs {i}, ' +  r'$r_{\rm col}$')
                ax.grid()
                ax.legend(fontsize = 18)
                ax.tick_params(axis='both', which='major',length=10, width=1.5)
                ax.tick_params(axis='both', which='minor',length=5, width=1)
                if ax in [axT, axd, axA, axB, axk]:
                    ax.set_xlabel(r'$r/r_{\rm a}$')
                    ax.set_xlabel(r'$r/r_{\rm a}$')
            plt.tight_layout() 
            # plt.savefig(f'{abspath}/Figs/{folder}/Test/{snap}/alphai_{snap}_{i}.png')
            # plt.close()

        del smoothed_flux_r2, R_lamda, fld_factor, ray_radDen, alpha_rossland, alpha_planck, alpha_scatter, los, los_effective, tree, idxnew, f_inter_input, volume, ray_x, ray_y, ray_z, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den 
        gc.collect()

    Lphoto_snap = np.mean(photosphere['Lum'])
    print('L :', Lphoto_snap, flush=True)

    return Lphoto_snap, photosphere, colorsphere, freqs, L_col

if __name__ == "__main__":
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'HiResNewAMR' 
    N_ray = 5_000
    params = [m, Rstar, mstar, beta, n, compton]
    things = orb.get_things_about([Mbh, Rstar, mstar, beta])
    t_fall = things['t_fb_days']
    t_fall_cgs = t_fall * 24 * 3600
    # Load opacity tables
    opac_path = f'{abspath}/src/Opacity'

    # Load data (they are the ln of the values)
    ln_T_tab = np.loadtxt(f'{opac_path}/T.txt') 
    ln_Rho_tab = np.loadtxt(f'{opac_path}/rho.txt') 
    ln_rossland_tab = np.loadtxt(f'{opac_path}/ross.txt') # Each row is a fixed T, column a fixed rho
    ln_planck_tab = np.loadtxt(f'{opac_path}/planck.txt') # Each row is a fixed T, column a fixed rho
    ln_scatt_tab = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm
    T_tab = np.exp(ln_T_tab)
    Rho_tab = np.exp(ln_Rho_tab)
    ross_tab = np.exp(ln_rossland_tab)
    pl_tab = np.exp(ln_planck_tab)
    min_T, max_T = np.min(T_tab), np.max(T_tab)
    min_Rho, max_Rho = np.min(Rho_tab), np.max(Rho_tab)
    print(f'min T in table: {min_T}, \nmax T in table: {max_T}, \nmin Rho in table: {min_Rho}, \nmax Rho in table: {max_Rho}')
    kappa_ross_tab = []
    for i in range(len(T_tab)):
        kappa_ross_tab.append(ross_tab[i, :]/Rho_tab)
    kappa_ross_tab = np.array(kappa_ross_tab)
    kappa_theory_scatt = 0.2*(1+prel.X_nf) 
    alpha_theory_scatt = kappa_theory_scatt * Rho_tab #1/cm

    kappa_planck = []
    for i in range(len(T_tab)):
        kappa_planck.append(pl_tab[i, :]/Rho_tab)
    kappa_planck = np.array(kappa_planck)
    plt.figure(figsize = (9, 14))
    img = plt.pcolormesh(np.log10(T_tab), np.log10(Rho_tab), kappa_planck.T,  norm = LogNorm(vmin = 1e-5, vmax=1e7), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'$\kappa_{\rm a}$ (cm$^2$/g)', fontsize=40)

    #%%
    # T_extr = np.logspace(2, 11, 200)
    # Rho_extr = np.logspace(-18, 10, 201)
    deltaxn_low = ln_T_tab[1] - ln_T_tab[0]
    deltayn_low = ln_Rho_tab[1] - ln_Rho_tab[0] 
    T_extra_low = [ln_T_tab[0] - deltaxn_low * (i + 1) for i in range(100)]
    Rho_extra_low = [ln_Rho_tab[0] - deltayn_low * (i + 1) for i in range(101)]
    # High extrapolation
    deltaxn_high = ln_T_tab[-1] - ln_T_tab[-2]
    deltayn_high = ln_Rho_tab[-1] - ln_Rho_tab[-2]
    T_extra_high = [ln_T_tab[-1] + deltaxn_high * (i + 1) for i in range(100)]
    Rho_extra_high = [ln_Rho_tab[-1] + deltayn_high * (i + 1) for i in range(101)]
    ln_new_T = np.concatenate([T_extra_low[::-1], ln_T_tab, T_extra_high])
    ln_new_Rho = np.concatenate([Rho_extra_low[::-1], ln_Rho_tab, Rho_extra_high])
    T_extr = np.exp(ln_new_T)
    Rho_extr = np.exp(ln_new_Rho)

    scatter_extr = np.zeros((len(T_extr), len(Rho_extr)))
    T_slope_scatt = np.zeros((len(T_extr), len(Rho_extr)))
    d_slope_scatt = np.zeros((len(T_extr), len(Rho_extr)))
    ross_extrap = np.zeros((len(T_extr), len(Rho_extr)))
    T_slope_ross = np.zeros((len(T_extr), len(Rho_extr)))
    d_slope_ross = np.zeros((len(T_extr), len(Rho_extr)))
    planck_extrap = np.zeros((len(T_extr), len(Rho_extr)))
    T_slope_planck = np.zeros((len(T_extr), len(Rho_extr)))
    d_slope_planck = np.zeros((len(T_extr), len(Rho_extr)))
    for i, T_val in enumerate(ln_new_T):
        for j, Rho_val in enumerate(ln_new_Rho):
            scatter_extr[i][j], T_slope_scatt[i][j], d_slope_scatt[i][j] = calc_scattering_opacity(ln_T_tab, ln_Rho_tab, ln_scatt_tab, T_val, Rho_val)
            ross_extrap[i][j], T_slope_ross[i][j], d_slope_ross[i][j] = calc_ross_opacity(ln_T_tab, ln_Rho_tab, ln_rossland_tab, ln_scatt_tab, T_val, Rho_val)
            planck_extrap[i][j], T_slope_planck[i][j], d_slope_planck[i][j] = calc_planck_opacity(ln_T_tab, ln_Rho_tab, ln_planck_tab, T_val, Rho_val)
    
    # find kappa 
    kappa_scatter_extr = []
    kappa_ross_extr = []
    kappa_planck_extr = []
    for i in range(len(T_extr)):
        kappa_scatter_extr.append(scatter_extr[i, :]/Rho_extr)
        kappa_ross_extr.append(ross_extrap[i, :]/Rho_extr)
        kappa_planck_extr.append(planck_extrap[i, :]/Rho_extr)
    kappa_scatter_extr = np.array(kappa_scatter_extr)
    kappa_ross_extr = np.array(kappa_ross_extr)
    kappa_planck_extr = np.array(kappa_planck_extr)

    #%%
    fig, (ax0, axR, axP) = plt.subplots(1,3, figsize = (30,15))
    figR, (axR_t, axR_d) = plt.subplots(1,2, figsize = (20,15))
    figP, (axP_t, axP_d) = plt.subplots(1,2, figsize = (20,15))
    figS, (axS_t, axS_d) = plt.subplots(1,2, figsize = (20,15))
    img = ax0.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), kappa_scatter_extr.T,  norm = LogNorm(vmin = 1e-4, vmax=.5), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'$\kappa_{\rm s}$ (cm$^2$/g)', fontsize=40)

    img = axR.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), kappa_ross_extr.T,  norm = LogNorm(vmin = 1e-5, vmax=1e7), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'$\kappa_{\rm R}$ (cm$^2$/g)', fontsize=40)

    img = axP.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), kappa_planck_extr.T,  norm = LogNorm(vmin = 1e-5, vmax=1e7), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'$\kappa_{\rm a}$ (cm$^2$/g)', fontsize=40)

    figratio, (axratio1, axratio2) = plt.subplots(1,2, figsize = (25,15))
    img = axratio1.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), (kappa_ross_extr/kappa_planck_extr).T,  norm = LogNorm(vmin = 1e-2, vmax=1e2), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'$\kappa_{\rm R}/\kappa_{\rm a}$', fontsize=40)
    
    img = axratio2.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), (kappa_scatter_extr/kappa_planck_extr).T,  norm = LogNorm(vmin = 1e-2, vmax=1e2), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'$\kappa_{\rm s}/\kappa_{\rm a}$', fontsize=40)

    ## coeff
    img = axS_t.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), T_slope_scatt.T,  vmin = -4, vmax = 4, cmap = 'jet', alpha = 0.7) #exp_scatt.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'A', fontsize=40)
    axS_t.set_title('Temperature extrapolation coefficient', fontsize = 40)

    img = axS_d.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), d_slope_scatt.T,  vmin = -4, vmax = 4, cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'A', fontsize=40)
    axS_d.set_title('Density extrapolation coefficient', fontsize = 40)

    img = axR_t.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), T_slope_ross.T,  vmin = -4, vmax = 4, cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'A', fontsize=40)
    axR_t.set_title('Temperature extrapolation coefficient', fontsize = 40)

    img = axR_d.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), d_slope_ross.T,  vmin = -4, vmax = 4, cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'A', fontsize=40)
    axR_d.set_title('Density extrapolation coefficient', fontsize = 40)

    img = axP_t.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), T_slope_planck.T,  vmin = -4, vmax = 4, cmap = 'jet', alpha = 0.7) #exp_planck.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'A', fontsize=40)
    axP_t.set_title('Temperature extrapolation coefficient', fontsize = 40)

    img = axP_d.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), d_slope_planck.T,  vmin = -4, vmax = 4, cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'A', fontsize=40)
    axP_d.set_title('Density extrapolation coefficient', fontsize = 40)
   
    for ax in [ax0, axR, axP, axratio1, axratio2, axP_t, axP_d, axR_t, axR_d, axS_t, axS_d]:
        ax.axvline(np.log10(min_T), color = 'grey', linestyle = '--', label = 'lim table')
        ax.axvline(np.log10(max_T), color = 'grey', linestyle = '--')
        ax.axhline(np.log10(min_Rho), color = 'grey', linestyle = '--')
        ax.axhline(np.log10(max_Rho), color = 'grey', linestyle = '--')
        # ax.scatter(np.log10(T_col), np.log10(den_col), c = 'white', s = 100)
        # ax.axhline(np.log10(1e-19*prel.Msol_cgs/prel.Rsol_cgs**3), color = 'grey', linestyle = ':', label = 'simulation cut')
        # Get the existing ticks on the x-axis
        big_ticks = [-10, -5, 0, 5, 10, 15] #ax.get_xticks()
        # Calculate midpoints between each pair of ticks
        midpointsx = np.arange(big_ticks[0], big_ticks[-1])
        # Combine the original ticks and midpointsx
        new_ticksx = np.sort(np.concatenate((big_ticks, midpointsx)))
        labelsx = [str(np.round(tick,2)) if tick in big_ticks else "" for tick in new_ticksx]   
        ax.set_xticks(new_ticksx)
        ax.set_xticklabels(labelsx, fontsize = 40)

        big_ticks = [-20, -15, -10, -5, 0, 5, 10, 15] #ax.get_yticks()
        # Calculate midpoints between each pair of ticks
        midpoints = np.arange(big_ticks[0], big_ticks[-1])
        # Combine the original ticks and midpoints
        new_ticks = np.sort(np.concatenate((big_ticks, midpoints)))
        labels = [str(np.round(tick,2)) if tick in big_ticks else "" for tick in new_ticks]   
        ax.set_yticks(new_ticks)
        ax.set_yticklabels(labels, fontsize = 40)

        ax.tick_params(axis='x', which='major', width=1.2, length=7, color = 'k')
        ax.tick_params(axis='y', which='major', width=1.2, length=7, color = 'k')
        ax.set_xlabel(r'$\log_{10} T$ (K)', fontsize=40)
        ax.set_xlim(0.8,11)
        ax.set_ylim(-19.5,11)
        ax.set_ylabel(r'$\log_{10} \rho$ (g/cm$^3$)', fontsize=40)
    # ax0.legend(fontsize=12, loc='center right')
    figS.suptitle(r'Scattering extrapolation coefficients', fontsize = 40)
    figR.suptitle(r'Rossland extrapolation coefficients', fontsize = 40)
    figP.suptitle(r'Planck extrapolation coefficients', fontsize = 40)
    plt.tight_layout()
    # %%
    T_cool = np.loadtxt(f'{opac_path}/T.txt')
    Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
    rossland = np.loadtxt(f'{opac_path}/ross.txt')
    planck = np.loadtxt(f'{opac_path}/planck.txt')
    scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm

    #%%MATLAB, thanks Cindy.
    eng = matlab.engine.start_matlab()

    #%%
    fld_lightcurve(params, compton, check, N_ray)
# %%
