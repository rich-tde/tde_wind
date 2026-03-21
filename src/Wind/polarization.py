""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices

#%% Choose parameters -----------------------------------------------------------------
# m = 4
# Mbh = 10**m
# beta = 1
# mstar = .5
# Rstar = .47
# n = 1.5
# compton = 'Compton'
# check = 'HiResNewAMR' 

# ## Snapshots stuff
# folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
# snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

def uniform_disk_observers(Nobs, HR_ratio=0.3, R=1.0):
    """
    Uniform observers on oblate spheroid surface H/R
    Returns: x,y,z unit vectors from spheroid surface
    """
    H = HR_ratio * R
    
    observers = []
    while len(observers) < Nobs:
        # Start with uniform sphere points
        u, v = np.random.uniform(-1, 1, 2)
        theta = np.arccos(2*u - 1)
        phi = 2*np.pi*v
        
        # Spherical coordinates → ellipsoid coordinates
        x_sph = np.sin(theta) * np.cos(phi)
        y_sph = np.sin(theta) * np.sin(phi) 
        z_sph = np.cos(theta)
        
        # Project radially to ellipsoid surface
        # Solve: scale_sph * r_sph lies on ellipsoid
        r_sph = np.sqrt(x_sph**2 + y_sph**2 + z_sph**2)
        # Ellipsoid equation: (x/r_sph)^2/R^2 + (z/r_sph)^2/H^2 = 1
        a = (x_sph**2 + y_sph**2)/R**2 + z_sph**2/H**2
        scale = 1.0 / np.sqrt(a)
        
        # Ellipsoid surface point
        x_obs = scale * x_sph / r_sph
        y_obs = scale * y_sph / r_sph  
        z_obs = scale * z_sph / r_sph
        
        observers.append([x_obs, y_obs, z_obs])
    
    return np.array(observers).T  # (3, Nobs)


def compute_polarization(x, y, z,
                        Ix, Iy, Iz,
                        n_obs):
    """
    Compute polarization for a single observer direction n_obs.

    Inputs:
        x, y, z : positions
        Ix, Iy, Iz : intensity vector components
        n_obs : observer direction (3-vector, not necessarily normalized)

    Returns:
        P : polarization fraction
        I, Q, U : Stokes parameters (normalized to total intensity)
    """
    Nall_obs = len(x) if isinstance(x, np.ndarray) else 1
    # Normalize observer direction (i.e. scattered light direction)
    n_obs = np.array(n_obs)
    n = n_obs / np.linalg.norm(n_obs)

    # Positions
    r_vec = np.vstack((x, y, z)).T # shape: (192,3)
    r_mag = np.linalg.norm(r_vec, axis=1)
    r_hat = r_vec / r_mag[:, None]   

    # Intensity vectors
    I_vec = np.vstack((Ix, Iy, Iz)).T
    I_mag = np.linalg.norm(I_vec, axis=1)
    # Intensity vectors avoiding division by zero. I_hat and n will define the scattering plane
    valid = I_mag > 0
    I_hat = np.zeros_like(I_vec)
    I_hat[valid] = I_vec[valid] / I_mag[valid][:, None]

    # Visibility condition
    mu_surface = np.dot(r_hat, n)
    visible = mu_surface > 0
    # Surface area weight under uniform sphere sampling assumption: 
    # dOmega = 4*np.pi / Nall_obs  # solid angle per cell
    # dA = r_mag**2 * dOmega
    # dA_proj = dA * mu_surface  # projected area toward observer
    I_hat, I_mag = make_slices([I_hat, I_mag], visible)

    # Scattering angle
    cos_theta = np.dot(I_hat, n)

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
    e2 /= np.linalg.norm(e2)

    # Polarization direction vector, in the scattering plane, orthogonal to n
    cross1 = np.cross(n, I_hat)
    e_pol = np.cross(cross1, n)

    e_pol_mag = np.linalg.norm(e_pol, axis=1)
    nonzero = e_pol_mag > 0
    e_pol[nonzero] /= e_pol_mag[nonzero][:, None]

    # Project polarization direction onto sky plane
    cos_phi = np.dot(e_pol, e1)
    sin_phi = np.dot(e_pol, e2)
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2 * cos_phi * sin_phi 
    # Intensity weighted by the area projected toward the observer
    I_local = I_mag #/ dA_proj

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
I_obs = ph_obs # flux along x
n_obs = ph_obs
P, I, Q, U = compute_polarization(
    ph_obs[0], ph_obs[1], ph_obs[2],
    I_obs[0], I_obs[1], I_obs[2],
    n_obs)
print(f"n_obs: {n_obs}, I_obs: {I_obs}, P = {P}\n---------")
##
print("TEST one wave. incident/observer direction perpendicular to scattered.")
ph_obs = np.array([0, 0, 1]) # observer along z
I_obs = np.array([1, 0, 0]) # flux along x
n_obs = ph_obs
P, I, Q, U = compute_polarization(
    ph_obs[0], ph_obs[1], ph_obs[2],
    I_obs[0], I_obs[1], I_obs[2],
    n_obs)
print(f"n_obs: {n_obs}, I_obs: {I_obs}, P = {P}\n---------")
##
print("TEST of symmetry (all radial fluxes) with healpix")
# NB Healpix doesn't necessarily give symmetric points, so we expect a small polarization signal (anyway ~0)
nside = 32 
Npix = hp.nside2npix(nside)
observers_xyz = hp.pix2vec(nside, np.arange(Npix)) # shape: (3, 192)
x_obs, y_obs, z_obs = observers_xyz
Fr_obs = np.ones_like(x_obs) * 2
Fx_obs = Fr_obs * x_obs
Fy_obs = Fr_obs * y_obs
Fz_obs = Fr_obs * z_obs
n_obs = [0, 0, 1]  
P, I, Q, U = compute_polarization(
    x_obs, y_obs, z_obs,
    Fx_obs, Fy_obs, Fz_obs,
    n_obs)
print(f"n_obs: {n_obs}, P = {P}")
#%%
# photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo109POL.txt')
# xph, yph, zph, volph, Fxph, Fyph, Fzph = photo[0], photo[1], photo[2], photo[3], photo[-3], photo[-2], photo[-1]
# dimph = (volph)**(1/3)
# xph, yph, zph, dimph, Fxph, Fyph, Fzph = xph[xph!=0], yph[xph!=0], zph[xph!=0], dimph[xph!=0], Fxph[xph!=0], Fyph[xph!=0], Fzph[xph!=0]
# n_obs = [0, 0, 1] 
# P, I, Q, U = compute_polarization(
#     xph, yph, zph, dimph,
#     Fxph, Fyph, Fzph,
#     n_obs
# )
# print(P)
x, y, z = uniform_disk_observers(100, HR_ratio=1, R=1.0)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.scatter(x, y, z)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
# %%
