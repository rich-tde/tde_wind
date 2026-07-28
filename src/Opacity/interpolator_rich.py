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

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy.integrate as sci

import Utilities.prelude as prel
import src.orbits as orb
from Utilities import operators as op

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
                return scattering, np.nan, np.nan
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


# def create_bigger_table(x, y, extrarowsx = 100, extrarowsy = 101):
#     deltaxn_low = x[1] - x[0]
#     deltayn_low = y[1] - y[0] 
#     x_extra_low = [x[0] - deltaxn_low * (i + 1) for i in range(extrarowsx)]
#     y_extra_low = [y[0] - deltayn_low * (i + 1) for i in range(extrarowsy)]
    
#     # High extrapolation
#     deltaxn_high = x[-1] - x[-2]
#     deltayn_high = y[-1] - y[-2]
#     x_extra_high = [x[-1] + deltaxn_high * (i + 1) for i in range(extrarowsx)]
#     y_extra_high = [y[-1] + deltayn_high * (i + 1) for i in range(extrarowsy)]
    
#     # Stack, reverse low to stack properly
#     xn = np.concatenate([x_extra_low[::-1], x, x_extra_high])
#     yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
#     return xn, yn


def create_bigger_table(x, y, extrarowsx=100, extrarowsy=101, density_factor=2):
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    dx_ext = dx / density_factor
    dy_ext = dy / density_factor

    # same extension distance as original function
    x_extra_low = x[0] - dx_ext * np.arange(extrarowsx * density_factor, 0, -1)
    y_extra_low = y[0] - dy_ext * np.arange(extrarowsy * density_factor, 0, -1)

    x_extra_high = x[-1] + dx_ext * np.arange(1, extrarowsx * density_factor + 1)
    y_extra_high = y[-1] + dy_ext * np.arange(1, extrarowsy * density_factor + 1)

    xn = np.concatenate([x_extra_low, x, x_extra_high])
    yn = np.concatenate([y_extra_low, y, y_extra_high])

    return xn, yn

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
            scatter_extr[i][j], T_slope_scatt[i][j], d_slope_scatt[i][j] = calc_scattering_opacity(ln_T_tab, ln_Rho_tab, ln_scatt_tab, T_val, Rho_val, return_coeff=True)
            ross_extrap[i][j], T_slope_ross[i][j], d_slope_ross[i][j] = calc_ross_opacity(ln_T_tab, ln_Rho_tab, ln_rossland_tab, ln_scatt_tab, T_val, Rho_val, return_coeff=True)
            planck_extrap[i][j], T_slope_planck[i][j], d_slope_planck[i][j] = calc_planck_opacity(ln_T_tab, ln_Rho_tab, ln_planck_tab, T_val, Rho_val, return_coeff=True)
    
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
    img = ax0.pcolormesh(np.log10(T_extr), np.log10(Rho_extr), kappa_scatter_extr.T,  norm = LogNorm(vmin = 1e-2, vmax=.5), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
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

    snap = 151
    i_obs = 0
    sim_path = f"{abspath}/data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR"
    dataph = np.load(f"{sim_path}/photo/{check}_photo{snap}.npz") 
    den_ph, T_ph = dataph['den'], dataph['temp'] 
    dataRtr = np.load(f"{sim_path}/trap/{check}_Rtr{snap}.npz") 
    x_tr, y_tr, z_tr, den_tr, T_tr = dataph['x'], dataph['y'], dataph['z'], dataRtr['den_tr'], dataRtr['Temp_tr'] 
    r_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
    kappa_all = np.load(f'{sim_path}/wind/kappa_fromFLD{snap}.npy', allow_pickle=True).item()
    r_prof = kappa_all[f'obs_{i_obs}']['r']
    d_prof = kappa_all[f'obs_{i_obs}']['d']
    t_prof = kappa_all[f'obs_{i_obs}']['t']
    d_prof = d_prof[r_prof>r_tr[i_obs]]
    t_prof = t_prof[r_prof>r_tr[i_obs]]
    r_prof = r_prof[r_prof>r_tr[i_obs]]
        
    for ax in [ax0, axR, axP, axratio1, axratio2, axP_t, axP_d, axR_t, axR_d, axS_t, axS_d]:
        ax.axvline(np.log10(min_T), color = 'grey', linestyle = '--', label = 'lim table')
        ax.axvline(np.log10(max_T), color = 'grey', linestyle = '--')
        ax.axhline(np.log10(min_Rho), color = 'grey', linestyle = '--')
        ax.axhline(np.log10(max_Rho), color = 'grey', linestyle = '--')
        # ax.scatter(np.log10(T_ph), np.log10(den_ph), c = 'r', s = 100)
        # ax.scatter(np.log10(T_tr), np.log10(den_tr), c = 'black', s = 100)
        ax.scatter(np.log10(t_prof), np.log10(d_prof*prel.den_converter), c = r_prof/330, s = 100, cmap = 'rainbow')
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
    #%%
    x_test = np.linspace(1e4, 1e8, 100)
    y_test = op.draw_line(x_test, [1e18, -3.5], 'powerlaw')
    chosenRhos = [1e-12, 1e-9] # you want 1e-6, 1e-11 kg/m^3 (too far from Elad's table, u want plot it)
    colors_plot = ['forestgreen', 'r']
    lines = ['solid', 'dashed']
    fig, ax = plt.subplots(1,2,figsize = (15,6))
    for i,chosenRho in enumerate(chosenRhos):
        if np.logical_and(chosenRho < max_Rho, chosenRho > min_Rho):
            irho = np.argmin(np.abs(Rho_tab - chosenRho))
            ax[i].plot(T_tab, kappa_ross_tab[:, irho], c = 'k', linewidth = 2.5, label = 'original')
        i_Rho = np.argmin(np.abs(Rho_extr - chosenRho))
        ax[i].plot(T_extr, kappa_ross_extr[:, i_Rho], c = 'yellowgreen', ls = '--', label = r'extrap extrapolation')
        ax[i].set_xlabel(r'T [K]')
        ax[i].set_xlim(1e1,5e8)
        ax[i].set_ylim(1e-1, 2e2) #the axis from 7e-4 to 2e1 m2/g
        ax[i].axvline(min_T, color = 'grey', linestyle = '--', label = 'lim table')
        ax[i].axvline(max_T, color = 'grey', linestyle = '--')
        ax[i].plot(T_extr, kappa_scatter_extr[:, i_Rho], c = 'dodgerblue', ls = '--', label = 'scattering')
        ax[i].axhline(0.2 * (1 + prel.X_nf), color = 'firebrick', linestyle = '--', label = 'Thomson scattering')
        ax[i].loglog()
        ax[i].grid()
        ax[i].plot(x_test, y_test, c = 'k', ls = ':')
        ax[i].set_title(f'Density: {chosenRho:.0e} g/cm$^3$', fontsize = 16)
    ax[1].set_ylabel(r'$\kappa$ [cm$^2$/g]')
    ax[0].legend(fontsize=15, loc='upper right')
    plt.tight_layout()
# %%
