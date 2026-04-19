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
from matplotlib import gridspec
from scipy.interpolate import griddata
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

def compute_polarization(Ix, Iy, Iz,
                        n_obs,
                        flux = False):
    """
    Compute polarization for a single observer direction n_obs.

    Inputs:
        Ix, Iy, Iz : intensity/flux vector components
        n_obs : observer direction (3-vector, not necessarily normalized)
        flux : if True, (Ix, Iy, Iz) are fluxes and not intensities, 
        so you need to divide by the projected area toward the observer to get the local intensity before computing Stokes parameters.
        r_mag : radial distance magnitude (optional, used for flux calculations)

    Returns:
        P : polarization fraction
        I, Q, U : Stokes parameters 
    """
    Nall_obs = len(Ix) if isinstance(Ix, np.ndarray) else 1
    # Normalize observer direction (i.e. scattered light direction)
    n_obs = np.array(n_obs)
    n = n_obs / np.linalg.norm(n_obs)

    # Surface normal
    # norm_surf_vec = np.vstack((nx, ny, nz)).T # shape: (192,3)
    # norm_mag = np.linalg.norm(norm_surf_vec, axis=1)
    # norm_surf_hat = norm_surf_vec / np.maximum(norm_mag[:, None], 1e-12)

    # Intensity/flux vector
    I_vec = np.vstack((Ix, Iy, Iz)).T # if flux, this is the flux vector, otherwise it's the intensity vector. 
    I_mag = np.linalg.norm(I_vec, axis=1)
    # Intensity vectors avoiding division by zero. I_hat and n will define the scattering plane
    I_hat = I_vec / np.maximum(I_mag[:, None], 1e-20)

    # Scattering angle
    cos_theta_scat = np.dot(I_hat, n)
    visible = cos_theta_scat >= 0
    I_hat, I_mag, cos_theta_scat = make_slices([I_hat, I_mag, cos_theta_scat], visible)

    # Find intensity from flux through surface (i.e. radial projection of flux)
    if flux:
        dOmega = 4*np.pi / Nall_obs  # solid angle per cell
        # cos_theta_geom = np.dot(norm_surf_hat, n)
        # dA_proj = dOmega * cos_theta_scat  # projected area toward observer nobs or use cos_theta_geom?
        # I_local = I_mag / dA_proj # I_mag is the flux in this case
        # I_local[cos_theta_scat==0] = 0 # cosTheta=0 menas theta=90, so max polarization
        dA_proj = dOmega * cos_theta_scat
        I_local = I_mag / np.maximum(dA_proj, 1e-15)
        I_local[cos_theta_scat == 0] = 0 # only consider cells that are locally visible (i.e. cosTheta>0) for polarization. Otherwise, you consider also the light that is scattered toward the observer but then absorbed by the disk itself, which is not what you want.
    else:
        I_local = I_mag
        # visible = cos_theta_scat >= 0

    # Thomson polarization fraction for the  cell
    P_local = (1 - cos_theta_scat**2) / (1 + cos_theta_scat**2)

    # --- Define a (fixed, arbitrary) sky basis with a plane perpendicular to the line-of-sight direction n
    # vectors: (e1, e2, n). e1 is the first, it will give you the cos(2\phi) which define Q param
    tmp = np.array([0.0, 0.0, 1.0])
    e1 = tmp - np.dot(tmp, n) * n  # proj of tmp onto sky plane
    if np.linalg.norm(e1) < 1e-6:   # avoid degeneracy if n || tmp
        tmp = np.array([0.0, 1.0, 0.0])
        e1 = tmp - np.dot(tmp, n) * n 
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1) # not viceversa or you flip the sign. Like that, e1xe2 = n (as x,y,z in cartesian)
    e2 /= np.linalg.norm(e2)

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

    # Stokes parameters. 
    # I_local * P_local make you consider only the light that is locally scattered
    Q = np.sum(I_local * P_local * cos2phi)
    U = np.sum(I_local * P_local * sin2phi)
    I = np.sum(I_local)
    P = np.sqrt(Q**2 + U**2) / (I + 1e-20) #if you do sum(P_local) you have a number exceeding 1

    return P, I, Q, U


if __name__ == "__main__":
    from scipy.optimize import least_squares
    # test if fluxes work
    print("TEST of symmetry (all radial equal intensities) with healpix and fluxes")
    n_obs = [0, 0, 1]
    n_obs_hat = n_obs / np.linalg.norm(n_obs)
    x_k, y_k, z_k = 1, 1, -1
    Fx_k, Fy_k, Fz_k = 1, 0, 0
    P, _, _, _ = compute_polarization(Fx_k, Fy_k, Fz_k, n_obs, flux=True)
    print(P)

    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'HiResNewAMR' 
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    snap = 151
    few_obs = False
 
    photo = np.loadtxt(f'{abspath}/data/{folder}/photoPOL/{check}_photo{snap}POL.txt')
    x, y, z, Lum, Fx, Fy, Fz = photo[0], photo[1], photo[2], photo[14], photo[16], photo[17], photo[18]
    r_vec = np.vstack((x, y, z)).T
    r_ph = np.linalg.norm(r_vec, axis=1)
    r_hat = r_vec / np.maximum(r_ph[:, None], 1e-20)
    F_vec = np.vstack((Fx, Fy, Fz)).T
    F_mag = np.linalg.norm(F_vec, axis=1)
    F_hat = F_vec / np.maximum(F_mag[:, None], 1e-20)
    cos_Fhat_r = np.sum(F_hat * r_hat, axis=1)
    F_mag_median = np.median(F_mag)
    # dA = 4*np.pi*r_ph**2/len(x) # cell area
    # don't need to convert, because cancel out
    # Fx /= (prel.en_converter / prel.Rsol_cgs**2) # convert to code units
    # Fy /= (prel.en_converter / prel.Rsol_cgs**2)
    # Fz /= (prel.en_converter / prel.Rsol_cgs**2)
    P_all = np.zeros(len(x))
    phi_obs = np.arctan2(y, x)
    theta_obs = np.arccos(z/r_ph)
    longitude_moll = phi_obs 
    latitude_moll = np.pi/2 - theta_obs

    for idx in range(len(x)):
        n_obs = [x[idx], y[idx], z[idx]]
        P, I, Q, U = compute_polarization(Fx, Fy, Fz, n_obs, flux=True)
        P_all[idx] = P
    # print(f"n_obs: {n_obs}, P = {P}\n---------")
    if few_obs:
        cut = np.abs(longitude_moll)<4e-1
    else:
        cut = latitude_moll > -20 #i.e. all obs
    
    lon_1d = longitude_moll[cut]
    lat_1d = latitude_moll[cut]
    # Define a regular grid in (lon, lat) for visualization
    nlon = 360
    nlat = 180
    lon_grid = np.linspace(lon_1d.min(), lon_1d.max(), nlon)
    lat_grid = np.linspace(lat_1d.min(), lat_1d.max(), nlat)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    data_1d = np.abs(Fx[cut]) / F_mag_median
    data_grid = griddata(
    points=(lon_1d, lat_1d),
    values=data_1d,
    xi=(lon_mesh, lat_mesh),
    method='linear')

    fig = plt.figure(figsize=(30,10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1], height_ratios=[1,.05], hspace=0.01, wspace = 0.2)
    axx = fig.add_subplot(gs[0, 0], projection='mollweide')
    axx.pcolormesh(lon_mesh, lat_mesh, data_grid, cmap='rainbow', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1))  #color by intensity
    # axx.scatter(longitude_moll[cut], latitude_moll[cut], c=np.abs(Fx[cut]) / F_mag_median, cmap='rainbow', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1))  #color by intensity
    axx.grid(True)
    axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axx.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
    axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axx.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])
    axx.set_title('Fx', fontsize = 20)

    data_1d = np.abs(Fy[cut]) / F_mag_median
    data_grid = griddata(
    points=(lon_1d, lat_1d),
    values=data_1d,
    xi=(lon_mesh, lat_mesh),
    method='linear')

    axy = fig.add_subplot(gs[0, 1], projection='mollweide')
    axy.pcolormesh(lon_mesh, lat_mesh, data_grid, cmap='rainbow', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1))  #color by intensity
    axy.grid(True)
    axy.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axy.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
    axy.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axy.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])
    axy.set_title('Fy', fontsize = 20)

    data_1d = np.abs(Fz[cut]) / F_mag_median
    data_grid = griddata(
    points=(lon_1d, lat_1d),
    values=data_1d,
    xi=(lon_mesh, lat_mesh),
    method='linear')

    axz = fig.add_subplot(gs[0, 2], projection='mollweide')
    img = axz.pcolormesh(lon_mesh, lat_mesh, data_grid, cmap='rainbow', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1))  #color by intensity
    axz.grid(True)
    axz.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axz.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
    axz.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axz.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])
    axz.set_title('Fz', fontsize = 20)

    cbar_ax = fig.add_subplot(gs[1, 0:3]) 
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label =r'|F$_{\rm i}|$/F$_{\rm r, med}$')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
    
    fig, ax = plt.subplots(1,1,figsize=(10, 8))
    img = ax.scatter(latitude_moll[cut]*radians, P_all[cut], c = longitude_moll[cut]*radians, cmap='rainbow', edgecolors='k', s = 70, vmin = -np.pi, vmax = np.pi) #color by phi
    cbar = fig.colorbar(img, label=r'$\phi_{\rm obs}$ [rad]', orientation='horizontal')
    ax.set_xlabel(r'$\theta_{\rm obs}$ [rad]')
    ax.set_ylabel('Polarization Fraction P')
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(0, np.max(P_all)+0.02)
    plt.suptitle(f'Snap {snap}', fontsize=16)
    plt.tight_layout()
    print(np.median(P_all))

    data_1d = P_all 
    data_grid_P = griddata(
    points=(lon_1d, lat_1d),
    values=data_1d,
    xi=(lon_mesh, lat_mesh),
    method='linear')

    fig = plt.figure(figsize=(30,10))
    axx = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = axx.pcolormesh(lon_mesh, lat_mesh, data_grid_P, cmap='rainbow', vmin = 0, vmax = 1)  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', label =r'P')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
    axx.grid(True)
    axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axx.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
    axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axx.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])

    # data_1d = np.abs(r_ph/330) 
    # data_grid = griddata(
    # points=(lon_1d, lat_1d),
    # values=data_1d,
    # xi=(lon_mesh, lat_mesh),
    # method='linear')

    # fig = plt.figure(figsize=(30,10))
    # axx = fig.add_subplot(gs[0, 0], projection='mollweide')
    # img = axx.pcolormesh(lon_mesh, lat_mesh, data_grid, cmap='rainbow', vmin = 0, vmax = 6)  #color by intensity
    # cbar = plt.colorbar(img, orientation='horizontal', label =r'r$_{\rm ph}/r_{\rm a}$ ')
    # cbar.ax.tick_params(which='major',length = 5)
    # cbar.ax.tick_params(which='minor',length = 3)
    # axx.grid(True)
    # axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    # axx.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
    # axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    # axx.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])

    #%% what if it's radial
    x_heal, y_heal, z_heal = x/r_ph, y/r_ph, z/r_ph 
    F_x_rad = F_mag_median * x_heal
    F_y_rad = F_mag_median * y_heal
    F_z_rad = F_mag_median * z_heal
    F_r_vec = np.vstack((F_x_rad, F_y_rad, F_z_rad)).T
    F_r_hat = F_r_vec / np.maximum(np.linalg.norm(F_r_vec, axis=1)[:, None], 1e-20)
    cos_Fhat_r_rad = np.sum(F_r_hat * r_hat, axis=1)
    P_radial = np.zeros(len(x_heal))

    for idx in range(len(x)):
        n_obs = [x[idx], y[idx], z[idx]]
        P, I, Q, U = compute_polarization(F_x_rad, F_y_rad, F_z_rad, n_obs, flux=True)
        P_radial[idx] = P

    data_1d= np.abs(P_radial) 
    data_grid_Pr = griddata(
    points=(lon_1d, lat_1d),
    values=data_1d,
    xi=(lon_mesh, lat_mesh),
    method='linear')

    fig = plt.figure(figsize=(30,10))
    axx = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = axx.pcolormesh(lon_mesh, lat_mesh, data_grid_Pr, cmap='rainbow', vmin = 0, vmax = 1)  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', label =r'P')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
    axx.grid(True)
    axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axx.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
    axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axx.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])
    axx.set_title('If flux', fontsize=16)

# %%
