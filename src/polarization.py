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
                        k_obs):
    """
    Compute polarization for a single observer direction k_obs.

    Inputs:
        xph, yph, zph : photosphere positions
        Fxph, Fyph, Fzph : flux vector components at photosphere
        k_obs : observer direction (3-vector, not necessarily normalized)

    Returns:
        P : polarization fraction
        Q, U : Stokes parameters (normalized to total intensity)
    """
    # Normalize observer direction
    k = np.array(k_obs)
    k = k / np.linalg.norm(k)

    # Positions
    r_vec = np.vstack((xph, yph, zph)).T # shape: (192,3)
    r_mag = np.linalg.norm(r_vec, axis=1)
    n_hat = r_vec / r_mag[:, None]   # surface normal (radial approx)

    # Flux vectors
    F_vec = np.vstack((Fxph, Fyph, Fzph)).T
    F_mag = np.linalg.norm(F_vec, axis=1)

    # Flux vectors avoiding division by zero. F_hat and k will define the scattering plane
    valid = F_mag > 0
    F_hat = np.zeros_like(F_vec)
    F_hat[valid] = F_vec[valid] / F_mag[valid][:, None]

    # Visibility condition
    mu_surface = np.dot(n_hat, k)
    visible = mu_surface > 0

    # Keep only visible patches
    F_hat, F_mag, mu_surface, r_mag, dimph = make_slices([F_hat, F_mag, mu_surface, r_mag, dimph], visible)

    # Scattering angle
    cos_theta = np.dot(F_hat, k)

    # Thomson polarization fraction for the  cell
    P_local = (1 - cos_theta**2) / (1 + cos_theta**2)

    # --- Define sky plane basis vectors: k, e1, e2 ---
    # Choose arbitrary reference axis not parallel to k to find e1
    tmp = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(np.dot(tmp, k)), 1.0):
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(k, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(k, e1)

    # Polarization direction vector, perpendicular to the scattering plane (so to Fhat x k)
    # e_p ∝ k × (F × k)
    cross1 = np.cross(F_hat, k)
    e_pol = np.cross(k, cross1)

    # Project polarization direction onto sky plane
    e_pol_mag = np.linalg.norm(e_pol, axis=1)
    nonzero = e_pol_mag > 0
    e_pol[nonzero] /= e_pol_mag[nonzero][:, None]

    # Compute cos(2phi) and sin(2phi)
    cos_phi = np.dot(e_pol, e1)
    sin_phi = np.dot(e_pol, e2)

    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2 * cos_phi * sin_phi

    # Surface area weight (uniform sphere sampling assumption)
    # dOmega = 4*np.pi / N
    # dA = r_mag**2 * dOmega
    dA = np.pi * dimph**2  

    # Intensity weighted by the area projected toward the observer
    I_local = F_mag * mu_surface * dA

    # Stokes parameters
    Q = np.sum(I_local * P_local * cos2phi)
    U = np.sum(I_local * P_local * sin2phi)
    I = np.sum(I_local)

    P = np.sqrt(Q**2 + U**2) / (I + 1e-20)

    return P, Q, U, I

# TEST1: single cell at the photosphere, with flux along x and observer along z.
# So theta =90, cos(theta)=0, P_local = 1, and the polarization direction should be along x 
# So P=1, Q=1, U=0.
ph_obs = np.array([0, 0, 1]) # observer along z
F_obs = np.array([1, 0, 0]) # flux along x
dim_obs = np.array([1])
k_obs = [0, 0, 1]  

P, Q, U, I = compute_polarization(
    ph_obs[0], ph_obs[1], ph_obs[2], dim_obs,
    F_obs[0], F_obs[1], F_obs[2],
    k_obs
)
print("Polarization TEST1:", P)

#%%
# photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo109POL.txt')
# xph, yph, zph, volph, Fxph, Fyph, Fzph = photo[0], photo[1], photo[2], photo[3], photo[-3], photo[-2], photo[-1]
# dimph = (volph)**(1/3)
# xph, yph, zph, dimph, Fxph, Fyph, Fzph = xph[xph!=0], yph[xph!=0], zph[xph!=0], dimph[xph!=0], Fxph[xph!=0], Fyph[xph!=0], Fzph[xph!=0]


# %%
num_obs = prel.NPIX # you'll use it for the mean of the observers. It's 192, unless you don't find the photosphere for someone and so decrease of 1
observers_xyz = hp.pix2vec(prel.NSIDE, range(num_obs)) #shape: (3, 192)
x_obs = observers_xyz[0]
y_obs = observers_xyz[1]
z_obs = observers_xyz[2]
r_obs = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
phi_obs = y_obs/np.sqrt(x_obs**2 + y_obs**2)
theta_obs = z_obs/r_obs
k_obs = [0, 1, 0] 
Fx_test = np.ones(192)#np.sin(theta_obs) * np.cos(phi_obs) 
Fy_test = np.zeros(192)#np.sin(theta_obs) * np.sin(phi_obs)
Fz_test = np.zeros(192)#np.cos(theta_obs)
P, Q, U, I = compute_polarization(
    observers_xyz[0], observers_xyz[1], observers_xyz[2], np.ones(192),
    Fx_test, Fy_test, Fz_test,
    k_obs
)
print(Q, U)
# %%
