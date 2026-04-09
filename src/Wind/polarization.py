""" Compute and test polarization with toy models and with simultion data"""
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
from Utilities.basic_units import radians
#%% Choose parameters -----------------------------------------------------------------
# test = True
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 

def ellipsoid_surface(n_bins, a, b, c, healpix = False):
    """Sample uniform points on ellipsoid surface x²/a² + y²/b² + z²/c² = 1"""

    if healpix:
        if n_bins > 16:
            print("Warning: high nside. I don't want to die so I pick nside=16 for you")
            nside = 16
        else:
            nside = int(n_bins)
        npix = hp.nside2npix(nside)
        # Get uniform observer directions (theta, phi) from HEALPix pixels
        theta, phi = hp.pix2ang(nside, np.arange(npix))

    else:
        if n_bins % 2: n_bins -= 1  # Force even
        n_bins = int(n_bins)
        # make the sample symmetric with respect to cartesian axis
        phi_up = np.concatenate([np.linspace(0, np.pi/2, int(n_bins/4), endpoint=False),
                            np.linspace(np.pi/2, np.pi, int(n_bins/4), endpoint=False)])
        phi_down = phi_up + np.pi
        phi = np.concatenate([phi_up, phi_down])
        theta_up = np.linspace(0, np.pi/2, int(n_bins/2))
        theta_down = theta_up + np.pi/2
        theta = np.concatenate([theta_up, theta_down])
        theta = np.unique(theta)

    PHI, THETA = np.meshgrid(phi, theta)    
    # Ellipsoid coordinates
    x = a * np.sin(THETA) * np.cos(PHI)
    y = b * np.sin(THETA) * np.sin(PHI)
    z = c * np.cos(THETA)
    
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    # plt.figure(figsize=(6,6))
    # plt.scatter(THETA, PHI, s=5)
    # plt.xlabel(r'$\theta$')
    # plt.ylabel(r'$\phi$')
    # plt.axhline(np.pi/2)
    # plt.title('Spherical coordinates of points on ellipsoid surface')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    
    return x, y, z

def ellipsoid_normal(x, y, z, a, b, c):
    """
    Compute surface normal at points (x,y,z) on ellipsoid x²/a² + y²/b² + z²/c² = 1
    
    Args:
        x, y, z: coordinates (arrays or scalars)
        a, b, c: ellipsoid semi-axes
    
    Returns:
        n: unit normal vectors, shape same as input (Nx3)
    """
    # Gradient of F(x,y,z) = x²/a² + y²/b² + z²/c² - 1
    # c = HR * np.sqrt(a**2 + b**2)  # Compute c from H/R and a,b
    nx = 2*x / a**2
    ny = 2*y / b**2
    nz = 2*z / c**2
    
    # Stack into vectors
    n_vec = np.vstack((nx, ny, nz)).T  # (N,3)
    
    # Normalize
    n_mag = np.linalg.norm(n_vec, axis=1)[:, None]
    n_unit = n_vec / np.maximum(n_mag, 1e-12)  # avoid div by zero

    # r_hat = np.vstack((x, y, z)).T / np.sqrt(x**2 + y**2 + z**2)[:, None]
    # for i in range(len(x)):
    #     print(np.dot(n_unit[i], r_hat[i]))
    # n_unit = r_hat
    
    return n_unit

def create_disk(radius=1.0, height=0.1, n_radial=50, n_vertical=10):
    """
    Create full 3D disk: height H centered at z=0 (from -H to +H).
    Returns: X, Y, Z meshes for volumetric plotting or simulation.
    """
    theta = np.linspace(0, 2*np.pi, n_radial)
    r_vals = np.linspace(0, radius, n_radial)
    z_vals = np.linspace(-height, height, n_vertical)
    
    Theta, R, Z = np.meshgrid(theta, r_vals, z_vals, indexing='ij')
    
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()
    
    return X, Y, Z

def polarization_for_disk(obs, angle):
    """
    Compute polarization for a disk with normal along z, observed from obs.
    Assumes uniform intensity across the disk.
    Returns polarization fraction P.
    """
    # Scattering angle is theta_obs for all points on the disk
    if not angle:
        theta_obs = np.arccos(obs[2] / np.linalg.norm(obs))
    else:
        theta_obs = obs
    cos_theta_scat = np.cos(theta_obs)

    # Thomson polarization fraction
    P = (1 - cos_theta_scat**2) / (1 + cos_theta_scat**2)
    
    return P

def compute_polarization(nx, ny, nz,
                        Ix, Iy, Iz,
                        n_obs,
                        flux = False):
    """
    Compute polarization for a single observer direction n_obs.

    Inputs:
        nx, ny, nz : components of the local normal at each patch
        Ix, Iy, Iz : intensity/flux vector components
        n_obs : observer direction (3-vector, not necessarily normalized)
        flux : if True, (Ix, Iy, Iz) are fluxes and not intensities, 
        so you need to divide by the projected area toward the observer to get the local intensity before computing Stokes parameters.
    
    Returns:
        P : polarization fraction
        I, Q, U : Stokes parameters 
    """
    Nall_obs = len(nx) if isinstance(nx, np.ndarray) else 1
    # Normalize observer direction (i.e. scattered light direction)
    n_obs = np.array(n_obs)
    n = n_obs / np.linalg.norm(n_obs)

    # Surface normal
    norm_surf_vec = np.vstack((nx, ny, nz)).T # shape: (192,3)
    norm_mag = np.linalg.norm(norm_surf_vec, axis=1)
    norm_surf_hat = norm_surf_vec / np.maximum(norm_mag[:, None], 1e-12)

    # Intensity/flux vector
    I_vec = np.vstack((Ix, Iy, Iz)).T
    I_mag = np.linalg.norm(I_vec, axis=1)
    # Intensity vectors avoiding division by zero. I_hat and n will define the scattering plane
    I_hat = I_vec / np.maximum(I_mag[:, None], 1e-20)

    # Find intensity from flux through surface (i.e. radial projection of flux)
    if flux:
        print("Finding I from flux")
        cos_theta_geom = np.sum(norm_surf_hat * I_hat, axis=1) 
        dOmega = 4*np.pi / Nall_obs  # solid angle per cell
        dA_proj = dOmega * cos_theta_geom  # projected area toward observer
        F_obs = I_mag 
        I_local = F_obs / dA_proj
        I_local[dA_proj==0] = 0
        # Only what is radial can be seen
        I_hat = norm_surf_hat
    else:
        I_local = I_mag

    # Scattering angle
    cos_theta_scat = np.dot(I_hat, n)
    visible = cos_theta_scat >= 0
    # visible = np.logical_and(cos_theta_scat >= 0, cos_theta_geom>=0)
    I_hat, I_local, cos_theta_scat = make_slices([I_hat, I_local, cos_theta_scat], visible)
    # Thomson polarization fraction for the  cell
    P_local = (1 - cos_theta_scat**2) / (1 + cos_theta_scat**2)

    # --- Define a (fixed, arbitrary) sky basis with a plane perpendicular to the line-of-sight direction n
    # vectors: (e1, e2, n). e1 is the first, it will give you the cos(2\phi) which define Q param
    tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, tmp)
    if np.linalg.norm(e1) < 1e-6:   # avoid degeneracy if n || tmp
        tmp = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1) # not viceversa or you flip the sign. Like that, e1xe2 = n (as x,y,z in cartesian)
    e2 /= np.linalg.norm(e2)

    # tmp = np.array([0.0, 0.0, 1.0])
    # e1 = tmp - np.dot(tmp, n) * n  # proj of tmp onto sky plane
    # if np.linalg.norm(e1) < 1e-6:   # avoid degeneracy if n || tmp
    #     tmp = np.array([0.0, 1.0, 0.0])
    #     e1 = tmp - np.dot(tmp, n) * n 
    # e1 /= np.linalg.norm(e1)
    # e2 = np.cross(n, e1) # not viceversa or you flip the sign. Like that, e1xe2 = n (as x,y,z in cartesian)
    # e2 /= np.linalg.norm(e2)
    
    # Polarization direction vector, in the scattering plane, orthogonal to n
    cross1 = np.cross(I_hat, n)     # I_hat × n (incident × scattered)
    e_pol = np.cross(n, cross1)     # n × (I_hat × n) = proj of I_hat onto sky plane
    e_pol_mag = np.linalg.norm(e_pol, axis=1)
    e_pol/= np.maximum(e_pol_mag[:, None], 1e-20)

    # Project polarization direction onto sky plane
    cos_phi = np.dot(e_pol, e1)
    sin_phi = np.dot(e_pol, e2)
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2 * cos_phi * sin_phi 
   
    # # Intensity weighted by the area projected toward the observer
    # # Only what is radial can be seen
    # if dA is not None:
    #     print("Finding I from flux")
    #     norm_surf_hat, I_vec = norm_surf_hat[visible], I_vec[visible]
    #     cos_theta_geom = np.dot(norm_surf_hat, n)
    #     # dOmega = 4*np.pi / Nall_obs  # solid angle per cell
    #     dA_proj = dA[visible] * cos_theta_geom  # projected area toward observer
    #     F_obs = I_mag 
    #     I_local = F_obs / dA_proj
    #     I_local[dA_proj==0] = 0
    # else:   
    #     I_local = I_mag 

    # Stokes parameters. 
    # I_local * P_local make you consider only the light that is locally scattered
    Q = np.sum(I_local * P_local * cos2phi)
    U = np.sum(I_local * P_local * sin2phi)
    I = np.sum(I_local)
    P = np.sqrt(Q**2 + U**2) / (I + 1e-20) #if you do sum(P_local) you have a number exceeding 1

    return P, I, Q, U


