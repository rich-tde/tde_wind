""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
import sys
sys.path.append('/Users/paolamartire/shocks')
# import resource
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
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
from Utilities.operators import make_tree, sort_list

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

def compute_polarization(xph, yph, zph, dimph,
                        Fxph, Fyph, Fzph,
                        n_obs):
    """
    Compute polarization for a single observer direction n_obs.

    Inputs:
        xph, yph, zph : photosphere positions
        Fxph, Fyph, Fzph : flux vector components at photosphere
        n_obs : observer direction (3-vector, not necessarily normalized)

    Returns:
        P : polarization fraction
        Q, U : Stokes parameters (normalized to total intensity)
    """
    # Normalize observer direction
    n = np.array(n_obs)
    n = n / np.linalg.norm(n)

    # Positions
    r_vec = np.vstack((xph, yph, zph)).T # shape: (192,3)
    r_mag = np.linalg.norm(r_vec, axis=1)
    r_hat = r_vec / r_mag[:, None]   # surface normal (radial approx)

    # Flux vectors
    F_vec = np.vstack((Fxph, Fyph, Fzph)).T
    F_mag = np.linalg.norm(F_vec, axis=1)

    # Flux vectors avoiding division by zero. F_hat and n will define the scattering plane
    valid = F_mag > 0
    F_hat = np.zeros_like(F_vec)
    F_hat[valid] = F_vec[valid] / F_mag[valid][:, None]

    # Visibility condition
    mu_surface = np.dot(r_hat, n)
    visible = mu_surface > 0
    # Keep only visible patches
    F_hat, F_mag, mu_surface, r_mag, dimph = make_slices([F_hat, F_mag, mu_surface, r_mag, dimph], visible)

    # Scattering angle
    cos_theta = np.dot(F_hat, n)

    # Thomson polarization fraction for the  cell
    P_local = (1 - cos_theta**2) / (1 + cos_theta**2)

    # --- Define a (fixed, arbitrary) sky basis with a plane perpendicular to the line-of-sight direction n
    # vectors: n, e1, e2
    tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, tmp)
    if np.linalg.norm(e1) < 1e-6:   # avoid degeneracy if n || tmp
        tmp = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    # Polarization direction vector, in the scattering plane, orthogonal to n
    cross1 = np.cross(F_hat, n)
    e_pol = np.cross(n, cross1)

    # Project polarization direction onto sky plane
    e_pol_mag = np.linalg.norm(e_pol, axis=1)
    nonzero = e_pol_mag > 0
    e_pol[nonzero] /= e_pol_mag[nonzero][:, None]

    # Compute cos(2phi) and sin(2phi)
    cos_phi = np.dot(e_pol, e1)
    sin_phi = np.dot(e_pol, e2)

    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2 * cos_phi * sin_phi

    # Surface area weight 
    # uniform sphere sampling assumption: dOmega = 4*np.pi / N,  dA = r_mag**2 * dOmega
    # dA = np.pi * dimph**2  
    # dA_proj = dA * mu_surface  # projected area toward observer

    # Intensity weighted by the area projected toward the observer
    # I_local = F_mag * dA_proj
    I_local = F_mag # otherwise I has the wrong units

    # Stokes parameters. 
    # I_local * P_local make you consider only the light that is locally scattered
    Q = np.sum(I_local * P_local * cos2phi)
    U = np.sum(I_local * P_local * sin2phi)
    I = np.sum(I_local)

    P = np.sqrt(Q**2 + U**2) / (I + 1e-20) #if you do sum(P_local) you have a number exceeding 1

    return P, Q, U, I

# TEST1: single cell at the photosphere, with flux along x and observer along z.
print("TEST one wave with incident/observer direction parallel to scattered.")
ph_obs = np.array([0, 0, 1]) # observer along z
F_obs = ph_obs # flux along x
k_obs = ph_obs
dim_obs = np.array([1])
P, Q, U, I = compute_polarization(
    ph_obs[0], ph_obs[1], ph_obs[2], dim_obs,
    F_obs[0], F_obs[1], F_obs[2],
    k_obs)
print("P =", P)
##
print("TEST one wave. incident/observer direction perpendicular to scattered.")
ph_obs = np.array([0, 0, 1]) # observer along z
F_obs = np.array([0, 1, 0]) # flux along y
k_obs = ph_obs
dim_obs = np.array([1])
P, Q, U, I = compute_polarization(
    ph_obs[0], ph_obs[1], ph_obs[2], dim_obs,
    F_obs[0], F_obs[1], F_obs[2],
    k_obs)
print("P =", P)
##
print("TEST of symmetry (all radial fluxes)")
num_obs = prel.NPIX
observers_xyz = hp.pix2vec(prel.NSIDE, range(num_obs)) # shape: (3, 192)
x_obs, y_obs, z_obs = observers_xyz
dim_obs = np.ones_like(x_obs) * 192
Fr_obs = np.ones_like(x_obs) * 2
Fx_obs = Fr_obs * x_obs
Fy_obs = Fr_obs * y_obs
Fz_obs = Fr_obs * z_obs
k_obs = [1, 2, 0.5]  
P, Q, U, I = compute_polarization(
    x_obs, y_obs, z_obs, dim_obs,
    Fx_obs, Fy_obs, Fz_obs,
    k_obs)
print("P = ", P)
#%%
photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo109POL.txt')
xph, yph, zph, volph, Fxph, Fyph, Fzph = photo[0], photo[1], photo[2], photo[3], photo[-3], photo[-2], photo[-1]
dimph = (volph)**(1/3)
xph, yph, zph, dimph, Fxph, Fyph, Fzph = xph[xph!=0], yph[xph!=0], zph[xph!=0], dimph[xph!=0], Fxph[xph!=0], Fyph[xph!=0], Fzph[xph!=0]
k_obs = [0, 0, 1] 
P, Q, U, I = compute_polarization(
    xph, yph, zph, dimph,
    Fxph, Fyph, Fzph,
    k_obs
)
print(P)
# %%
