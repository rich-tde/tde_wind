""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices

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
        dimph : cell size at photosphere
        Fxph, Fyph, Fzph : flux vector components at photosphere
        n_obs : observer direction (3-vector, not necessarily normalized)

    Returns:
        P : polarization fraction
        I, Q, U : Stokes parameters (normalized to total intensity)
    """
    Nall_obs = len(xph) if isinstance(xph, np.ndarray) else 1
    # Normalize observer direction (i.e. scattered light direction)
    n_obs = np.array(n_obs)
    n = n_obs / np.linalg.norm(n_obs)

    # Positions
    r_vec = np.vstack((xph, yph, zph)).T # shape: (192,3)
    r_mag = np.linalg.norm(r_vec, axis=1)
    r_hat = r_vec / r_mag[:, None]   

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
    # Surface area weight under uniform sphere sampling assumption: 
    dOmega = 4*np.pi / Nall_obs  # solid angle per cell
    dA = r_mag**2 * dOmega
    dA_proj = dA * mu_surface  # projected area toward observer
    # dA = np.pi * dimph**2 

    F_hat, F_mag, dA_proj = make_slices([F_hat, F_mag, dA_proj], visible)

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

    # Intensity weighted by the area projected toward the observer
    I_local = F_mag * dA_proj

    # Stokes parameters. 
    # I_local * P_local make you consider only the light that is locally scattered
    Q = np.sum(I_local * P_local * cos2phi)
    U = np.sum(I_local * P_local * sin2phi)
    I = np.sum(I_local)

    P = np.sqrt(Q**2 + U**2) / (I + 1e-20) #if you do sum(P_local) you have a number exceeding 1

    return P, I, Q, U

# TEST1: single cell at the photosphere, with flux along x and observer along z.
print("TEST one wave with incident/observer direction parallel to scattered.")
ph_obs = np.array([0, 0, 1]) # observer along z
F_obs = ph_obs # flux along x
n_obs = ph_obs
dim_obs = np.array([1])
P, I, Q, U = compute_polarization(
    ph_obs[0], ph_obs[1], ph_obs[2], dim_obs,
    F_obs[0], F_obs[1], F_obs[2],
    n_obs)
print(f"n_obs: {n_obs}, F_obs: {F_obs}, P = {P}\n---------")
##
print("TEST one wave. incident/observer direction perpendicular to scattered.")
ph_obs = np.array([0, 0, 1]) # observer along z
F_obs = np.array([1, 0, 0]) # flux along x
n_obs = ph_obs
dim_obs = np.array([1])
P, I, Q, U = compute_polarization(
    ph_obs[0], ph_obs[1], ph_obs[2], dim_obs,
    F_obs[0], F_obs[1], F_obs[2],
    n_obs)
print(f"n_obs: {n_obs}, F_obs: {F_obs}, P = {P}\n---------")
##
print("TEST of symmetry (all radial fluxes)")
theta = np.linspace(0, np.pi/2, 20) # 
phi = np.linspace(0, 2*np.pi, 10, endpoint=False)
THETA, PHI = np.meshgrid(theta, phi)
THETA = THETA.flatten()
PHI = PHI.flatten()
x = np.sin(THETA) * np.cos(PHI)
y = np.sin(THETA) * np.sin(PHI)
z = np.cos(THETA)
dim = np.ones_like(x) * 2
Fx, Fy, Fz = x, y, z
n_obs = np.array([0, 0, 1])
P, I, Q, U = compute_polarization(
    x, y, z, dim,
    Fx, Fy, Fz,
    n_obs)
print(f"n_obs: {n_obs}, P = {P}\n---------")
##
print("TEST of symmetry (all radial fluxes) with healpix")
# NB Healpix doesn't necessarily give symmetric points, so we expect a small polarization signal (anyway ~0)
num_obs = prel.NPIX
observers_xyz = hp.pix2vec(prel.NSIDE, range(num_obs)) # shape: (3, 192)
x_obs, y_obs, z_obs = observers_xyz
dim_obs = np.ones_like(x_obs) * 192
Fr_obs = np.ones_like(x_obs) * 2
Fx_obs = Fr_obs * x_obs
Fy_obs = Fr_obs * y_obs
Fz_obs = Fr_obs * z_obs
n_obs = [1, 0, 0]  
P, I, Q, U = compute_polarization(
    x_obs, y_obs, z_obs, dim_obs,
    Fx_obs, Fy_obs, Fz_obs,
    n_obs)
print(f"n_obs: {n_obs}, P = {P}")
#%%
photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo109POL.txt')
xph, yph, zph, volph, Fxph, Fyph, Fzph = photo[0], photo[1], photo[2], photo[3], photo[-3], photo[-2], photo[-1]
dimph = (volph)**(1/3)
xph, yph, zph, dimph, Fxph, Fyph, Fzph = xph[xph!=0], yph[xph!=0], zph[xph!=0], dimph[xph!=0], Fxph[xph!=0], Fyph[xph!=0], Fzph[xph!=0]
n_obs = [0, 0, 1] 
P, I, Q, U = compute_polarization(
    xph, yph, zph, dimph,
    Fxph, Fyph, Fzph,
    n_obs
)
print(P)
# %%
