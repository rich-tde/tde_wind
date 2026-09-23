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

def ellipsoid_surface(n_bins, a, b, c,
                      x0=0, y0=0, z0=0,
                      healpix=False):
    """
    Sample points on the ellipsoid surface

        x^2/a^2 + y^2/b^2 + z^2/c^2 = 1

    and compute the finite surface area dA associated with each point.

    Returns
    -------
    x, y, z : ndarray
        Coordinates of the surface points.

    dA : ndarray
        Surface area associated with each point.
    """

    if healpix:
        if n_bins > 16:
            print("Warning: high nside. I don't want to die so I pick nside=16 for you")
            nside = 16
        else:
            nside = int(n_bins)
        npix = hp.nside2npix(nside)
        # Get uniform observer directions (theta, phi) from HEALPix pixels
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        THETA = theta
        PHI = phi
        dOmega = np.full(npix, 4.0 * np.pi / npix)
        

    else:
        if n_bins % 2: 
            n_bins -= 1  # Force even
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

        # angle edges
        theta_edges = np.empty(len(theta) + 1)
        theta_edges[0] = 0.0
        theta_edges[-1] = np.pi
        theta_edges[1:-1] = (theta[:-1] + theta[1:]) / 2.0
        # ∫ sin(theta) dtheta, while phi is uniformly sampled
        dmu = (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))
        dphi = 2.0 * np.pi / len(phi)
        dOmega = dmu[:, None] * dphi
        # Repeat along phi
        dOmega = np.broadcast_to(dOmega, THETA.shape)


    # Unit radial vector of the parameter sphere
    nx = np.sin(THETA) * np.cos(PHI)
    ny = np.sin(THETA) * np.sin(PHI)
    nz = np.cos(THETA)
    # Ellipsoid coordinates
    x = a * nx + x0
    y = b * ny + y0
    z = c * nz + z0

    dA_dOmega = np.sqrt(
        (b * c * nx)**2
        + (a * c * ny)**2
        + (a * b * nz)**2)

    # Finite area represented by each point
    dA = dA_dOmega * dOmega
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    dA = dA.ravel()

    # plt.figure(figsize=(6,6))
    # plt.scatter(THETA, PHI, s=5)
    # plt.xlabel(r'$\theta$')
    # plt.ylabel(r'$\phi$')
    # plt.axhline(np.pi/2)
    # plt.title('Spherical coordinates of points on ellipsoid surface')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    
    return x, y, z, dA

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
    # if it's a disk, adjust the normal to point along z-axis
    if a == b and c < 1e-6:
        print('It is a disk')
        nz = np.sign(z) * np.ones_like(z)  if np.any(z != 0) else np.ones_like(z)
        nx = np.zeros_like(x)
        ny = np.zeros_like(y)
    
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