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
from scipy.interpolate import griddata
from sklearn.neighbors import KDTree
from scipy.ndimage import uniform_filter1d

import Utilities.prelude as prel
from Utilities.sections import make_slices
import src.orbits as orb
from Utilities.operators import make_tree


def bilinear_interpolation_vectorized(x_vec, y_vec, data, x_arr, y_arr):
    """
    Vectorized bilinear interpolation.
    x_arr, y_arr: arrays of same length.
    data[i, j] corresponds to (x_vec[i], y_vec[j]).
    Returns an array of interpolated values.
    """
    x_vec = np.asarray(x_vec)
    y_vec = np.asarray(y_vec)
    data = np.asarray(data)

    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)

    N = len(x_arr)
    result = np.empty(N, dtype=np.float64)

    # Clamp to grid
    x_arr = np.clip(x_arr, x_vec[0], x_vec[-1])
    y_arr = np.clip(y_arr, y_vec[0], y_vec[-1])

    # Find indices for all points
    i = np.searchsorted(x_vec, x_arr, side='right') - 1
    j = np.searchsorted(y_vec, y_arr, side='right') - 1

    i = np.clip(i, 0, len(x_vec) - 2).astype(np.intp)
    j = np.clip(j, 0, len(y_vec) - 2).astype(np.intp)

    x0 = x_vec[i]
    x1 = x_vec[i + 1]
    y0 = y_vec[j]
    y1 = y_vec[j + 1]

    tx = (x_arr - x0) / (x1 - x0)
    ty = (y_arr - y0) / (y1 - y0)

    # Use advanced indexing to get the four corners
    d00 = data[i,     j    ]
    d10 = data[i + 1, j    ]
    d01 = data[i,     j + 1]
    d11 = data[i + 1, j + 1]

    result = (d00 * (1 - tx) * (1 - ty) +
              d10 * tx       * (1 - ty) +
              d01 * (1 - tx) * ty       +
              d11 * tx       * ty)

    return result

def interpolate_2d_table_vectorized(x_vec, y_vec, data, x_arr, y_arr, x_vec_high_slope=0.0, slope_length=7):
    """
    Vectorized 2D extrapolation-aware interpolation.
    x_arr, y_arr: arrays of same length.
    Returns: interp_val (array), x_slope (array), y_slope (array).
    """
    x_vec = np.asarray(x_vec, dtype=np.float64)
    y_vec = np.asarray(y_vec, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)

    N = len(x_arr)
    interp_val = np.empty(N, dtype=np.float64)
    x_slope = np.empty(N, dtype=np.float64)
    y_slope = np.empty(N, dtype=np.float64)

    # Masks
    mask_x_low = x_arr < x_vec[0]
    mask_x_high = x_arr > x_vec[-1]
    mask_x_mid = ~mask_x_low & ~mask_x_high

    mask_y_low = y_arr < y_vec[0]
    # mask_y_mid = ~mask_y_low

    # --- Region: x < x_vec[0] and y < y_vec[0] ---
    mask = mask_x_low & mask_y_low
    if np.any(mask):
        x_slope[mask] = (data[slope_length - 1, 0] - data[0, 0]) / (x_vec[slope_length - 1] - x_vec[0])
        y_slope[mask] = (data[0, slope_length - 1] - data[0, 0]) / (y_vec[slope_length - 1] - y_vec[0])
        base = data[0, 0] + y_slope[mask] * (y_arr[mask] - y_vec[0]) + x_slope[mask] * (x_arr[mask] - x_vec[0])
        interp_val[mask] = np.exp(base)

    # --- Region: x < x_vec[0] and y >= y_vec[0] ---
    mask = mask_x_low & ~mask_y_low
    if np.any(mask):
        x0 = x_vec[0] * 1.00001
        data_x0 = bilinear_interpolation_vectorized(x_vec, y_vec, data, np.full_like(y_arr[mask], x0), y_arr[mask])
        x_high = x_vec[slope_length - 1]
        data_xhigh = bilinear_interpolation_vectorized(
            x_vec, y_vec, data, np.full_like(y_arr[mask], x_high), y_arr[mask]
        )
        x_slope[mask] = (data_xhigh - data_x0) / (x_vec[slope_length - 1] - x_vec[0])
        interp_val[mask] = np.exp(data_x0 + x_slope[mask] * (x_arr[mask] - x_vec[0]))
        y_slope[mask] = 0.0

    # --- Region: x > x_vec[-1] and y < y_vec[0] ---
    mask = mask_x_high & mask_y_low
    if np.any(mask):
        y_slope[mask] = (data[-1, slope_length - 1] - data[-1, 0]) / (y_vec[slope_length - 1] - y_vec[0])
        base = data[-1, 0] + y_slope[mask] * (y_arr[mask] - y_vec[0]) + x_vec_high_slope * (x_arr[mask] - x_vec[-1])
        interp_val[mask] = np.exp(base)
        x_slope[mask] = x_vec_high_slope

    # --- Region: x > x_vec[-1] and y >= y_vec[0] ---
    mask = mask_x_high & ~mask_y_low
    if np.any(mask):
        x_near = x_vec[-1] * 0.99999
        base = bilinear_interpolation_vectorized(x_vec, y_vec, data, np.full_like(y_arr[mask], x_near), y_arr[mask])
        interp_val[mask] = np.exp(base + x_vec_high_slope * (x_arr[mask] - x_vec[-1]))
        x_slope[mask] = x_vec_high_slope
        y_slope[mask] = 0.0

    # --- Region: x mid, y < y_vec[0] ---
    mask = mask_x_mid & mask_y_low
    if np.any(mask):
        y0 = y_vec[0] * 0.9999
        data_y0 = bilinear_interpolation_vectorized(x_vec, y_vec, data, x_arr[mask], np.full_like(x_arr[mask], y0))
        y_high = y_vec[slope_length - 1]
        data_yhigh = bilinear_interpolation_vectorized(x_vec, y_vec, data, x_arr[mask], np.full_like(x_arr[mask], y_high))
        y_slope[mask] = (data_yhigh - data_y0) / (y_vec[slope_length - 1] - y_vec[0])
        interp_val[mask] = np.exp(data_y0 + y_slope[mask] * (y_arr[mask] - y_vec[0]))
        x_slope[mask] = 0.0

    # --- Region: fully inside grid ---
    mask = mask_x_mid & ~mask_y_low
    if np.any(mask):
        interp_val[mask] = np.exp(
            bilinear_interpolation_vectorized(x_vec, y_vec, data, x_arr[mask], y_arr[mask])
        )
        x_slope[mask] = 0.0
        y_slope[mask] = 0.0

    return interp_val, x_slope, y_slope

def calc_scattering_opacity_vectorized( T_, rho_, scatter_, Tcell_arr, rhocell_arr, return_coeff=False):
    T_ = np.asarray(T_, dtype=np.float64)
    rho_ = np.asarray(rho_, dtype=np.float64)
    scatter_ = np.asarray(scatter_, dtype=np.float64)

    Tcell_arr = np.asarray(Tcell_arr, dtype=np.float64)
    rhocell_arr = np.asarray(rhocell_arr, dtype=np.float64)

    d_log = rhocell_arr.copy()
    d_ratio = np.ones_like(rhocell_arr)

    mask_low = rhocell_arr < rho_[0]
    mask_high = rhocell_arr > rho_[-1]
    
    rho_min = rho_[0]
    rho_max = rho_[-1]
    d_ratio[mask_low] = np.exp(rhocell_arr[mask_low])/np.exp(rho_min)
    d_log[mask_low] = rho_min
    d_ratio[mask_high] = np.exp(rhocell_arr[mask_high])/np.exp(rho_max)
    d_log[mask_high] = rho_max

    interp_val, T_slope, d_slope = \
        interpolate_2d_table_vectorized(T_, rho_, scatter_, Tcell_arr, d_log)

    scatter = interp_val * d_ratio

    if return_coeff:
        # you don't use the slope from interpolate_2d_table_vectorized beacause you shifted d inside the table
        d_slope[mask_low] = 1
        d_slope[mask_high] = 1
        return scatter, T_slope, d_slope

    return scatter

def calc_ross_opacity_vectorized( T_, rho_, rossland_, scatter_, Tcell_arr, rhocell_arr, return_coeff=False):
    T_ = np.asarray(T_, dtype=np.float64)
    rho_ = np.asarray(rho_, dtype=np.float64)

    Tcell_arr = np.asarray(Tcell_arr, dtype=np.float64)
    rhocell_arr = np.asarray(rhocell_arr, dtype=np.float64)

    d_log = rhocell_arr.copy()
    d_ratio = np.ones_like(rhocell_arr)
    d_slope = np.ones_like(rhocell_arr)

    mask_low = rhocell_arr < rho_[0]
    mask_high = rhocell_arr > rho_[-1]

    rho_max = rho_[-1]
    d_log[mask_high] = rho_max
    d_ratio[mask_high] = np.exp(rhocell_arr[mask_high])/np.exp(rho_max)

    # Interpolate Rosseland table everywhere
    interp_val, T_slope, d_slope = interpolate_2d_table_vectorized(
        T_, rho_, rossland_,
        Tcell_arr, d_log)

    # rossland = interp_val.copy()
    # rossland[mask_high] *= d_ratio[mask_high]
    # d_slope_out = d_slope.copy()

    rossland = interp_val * d_ratio

    # Handle rho < rho_min using scattering opacity
    if np.any(mask_low):
        scattering, Tscatt_slope, dscatt_slope = (
            calc_scattering_opacity_vectorized(
                T_, rho_, scatter_,
                Tcell_arr[mask_low],
                rhocell_arr[mask_low],
                return_coeff=True))

        use_ross = rossland[mask_low] > scattering
        use_scatt = ~use_ross

        rossland[mask_low] = np.where(
        use_scatt,
        scattering,
        rossland[mask_low])

        T_slope[mask_low] = np.where(
            use_scatt,
            Tscatt_slope,
            T_slope[mask_low])

        d_slope[mask_low] = np.where(
            use_scatt,
            dscatt_slope,
            d_slope[mask_low])

        if return_coeff:
            return rossland, T_slope, d_slope

    return rossland

def calc_planck_opacity_vectorized(T_, rho_, planck_, Tcell_arr, rhocell_arr, return_coeff=False):
    T_ = np.asarray(T_, dtype=np.float64)
    rho_ = np.asarray(rho_, dtype=np.float64)
    planck_ = np.asarray(planck_, dtype=np.float64)

    Tcell_arr = np.asarray(Tcell_arr, dtype=np.float64)
    rhocell_arr = np.asarray(rhocell_arr, dtype=np.float64)

    d_ratio = np.ones_like(rhocell_arr)
    d_slope = np.full_like(rhocell_arr, 2.0)
    d_log = rhocell_arr.copy()

    mask_low = rhocell_arr < rho_[0]
    mask_high = rhocell_arr > rho_[-1]
    mask_mid = ~mask_low & ~mask_high

    # Inside: d_slope = 0
    d_slope[mask_mid] = 0.0

    # Below grid
    if np.any(mask_low):
        # For T in range, compute slope from table
        mask_in_T = mask_low & (T_[0] < Tcell_arr) & (Tcell_arr < T_[-1])
        if np.any(mask_in_T):
            # idx = np.searchsorted(T_, Tcell_arr[mask_in_T]) - 1
            # idx = np.clip(idx, 0, len(T_) - 2)
            idx = np.searchsorted(T_, Tcell_arr[mask_in_T]) 
            d_slope[mask_in_T] = (planck_[idx, 10] - planck_[idx, 0]) / (rho_[10] - rho_[0])

        d_ratio[mask_low] = np.exp(rhocell_arr[mask_low]) / np.exp(rho_[0])
        d_log[mask_low] = rho_[0]

    # Above grid
    if np.any(mask_high):
        d_ratio[mask_high] = np.exp(rhocell_arr[mask_high]) / np.exp(rho_[-1])
        d_log[mask_high] = rho_[-1]

    interp_val, T_slope, d_slope_out = interpolate_2d_table_vectorized(
        T_, rho_, planck_, Tcell_arr, d_log,
        x_vec_high_slope=-3.5
    )

    planck = interp_val * (d_ratio ** d_slope)

    if return_coeff:
        return planck, T_slope, d_slope
    return planck




if __name__ == "__main__":
    # test with the non vectorized version for consistency
    from src.Opacity.interpolator_rich import calc_scattering_opacity, calc_ross_opacity, calc_planck_opacity, create_bigger_table
    opac_path = f'{abspath}/src/Opacity'
    ln_T_tab = np.loadtxt(f'{opac_path}/T.txt') 
    ln_Rho_tab = np.loadtxt(f'{opac_path}/rho.txt') 
    ln_rossland_tab = np.loadtxt(f'{opac_path}/ross.txt') # Each row is a fixed T, column a fixed rho
    ln_planck_tab = np.loadtxt(f'{opac_path}/planck.txt') # Each row is a fixed T, column a fixed rho
    ln_scatt_tab = np.loadtxt(f'{opac_path}/scatter.txt')

    Tcell_arr = np.array([-25, -20, -10.2, -15, -10, 2, -5])
    rhocell_arr = np.array([1, 5, 10, -10, 10, 4,  11])
    planck_vec = calc_planck_opacity_vectorized(ln_T_tab, ln_Rho_tab, ln_planck_tab, Tcell_arr, rhocell_arr)
    ross_vec = calc_ross_opacity_vectorized(ln_T_tab, ln_Rho_tab, ln_rossland_tab, ln_scatt_tab, Tcell_arr, rhocell_arr)
    scatt_vec = calc_scattering_opacity_vectorized(ln_T_tab, ln_Rho_tab, ln_scatt_tab, Tcell_arr, rhocell_arr)
    planck_nonvec = np.zeros_like(planck_vec)
    ross_nonvec = np.zeros_like(ross_vec)
    scatt_nonvec = np.zeros_like(scatt_vec)
    for k in range(len(Tcell_arr)):
        planck_nonvec[k] = calc_planck_opacity(ln_T_tab, ln_Rho_tab, ln_planck_tab, Tcell_arr[k], rhocell_arr[k])
        ross_nonvec[k] = calc_ross_opacity(ln_T_tab, ln_Rho_tab, ln_rossland_tab, ln_scatt_tab, Tcell_arr[k], rhocell_arr[k])
        scatt_nonvec[k] = calc_scattering_opacity(ln_T_tab, ln_Rho_tab, ln_scatt_tab, Tcell_arr[k], rhocell_arr[k])
    print(planck_vec/planck_nonvec)
    print(ross_vec/ross_nonvec)
    print(scatt_vec/scatt_nonvec)
