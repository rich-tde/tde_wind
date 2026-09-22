""" Functions for different geometrical surfaces to test polarization"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import healpy as hp

def create_disk(radius=1.0, height=0.1, n_radial=50, n_vertical=10):
    """
    Create full 3D disk: height H centered at z=0 (from -H to +H).
    Returns: X, Y, Z meshes for volumetric plotting or simulation.
    """
    theta = np.linspace(0, 2*np.pi, n_radial)
    r_vals = np.linspace(0.1, radius, n_radial)
    z_vals = np.linspace(-height, height, n_vertical)
    # delete points if z = 0 to avoid singularity in the normal vector
    z_vals = z_vals[z_vals != 0]
    
    Theta, R, Z = np.meshgrid(theta, r_vals, z_vals, indexing='ij')
    
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()
    
    return X, Y, Z

def ellipsoid_surface(n_bins, a, b, c, x0=0, y0=0, z0=0, healpix = False, stay_helpix = False):
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

    if stay_helpix:
        PHI, THETA = phi, theta
    else:
        PHI, THETA = np.meshgrid(phi, theta)    
    # Ellipsoid coordinates
    x = a * np.sin(THETA) * np.cos(PHI)
    y = b * np.sin(THETA) * np.sin(PHI)
    z = c * np.cos(THETA)
    
    x = x.ravel() + x0
    y = y.ravel() + y0
    z = z.ravel() + z0

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

def ellipsoid_unit_normal(x, y, z, a, b, c, x0=0, y0=0, z0=0):
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
    nx = 2*(x-x0) / a**2
    ny = 2*(y-y0) / b**2
    nz = 2*(z-z0) / c**2
    
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