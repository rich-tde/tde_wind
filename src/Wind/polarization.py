""" Compute and test polarization with toy models and with simultion data"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import healpy as hp
from scipy.optimize import minimize
from matplotlib import gridspec
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.sections import make_slices
from Utilities.basic_units import radians
from Utilities.operators import sort_list
from scipy.linalg import inv
#%% Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
snap = 151
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
Pmin = 0
Pmax = 0.4

def mvee_fit(points, tol=1e-6, max_iter=1000):
    """
    Minimum Volume Enclosing Ellipsoid (MVEE) for N points in d dimensions.

    Based on Khachiyan's algorithm.
    Returns:
        center: (d,)
        shape: (d,d) such that   (x - c) @ shape @ (x - c) <= 1
    """
    points = np.atleast_2d(points)
    d, N = points.shape[1], len(points)

    # Khachiyan algorithm
    Q = np.column_stack([points, np.ones(N)])  # (N, d+1)
    uu = np.ones(N) / N

    for it in range(max_iter):
        X = Q.T @ np.diag(uu) @ Q
        M = np.diag(Q @ inv(X) @ Q.T)  # leverage scores
        j = np.argmax(M)
        step_size = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
        new_uu = (1 - step_size) * uu
        new_uu[j] += step_size
        if np.max(np.abs(new_uu - uu)) < tol:
            uu = new_uu
            break
        uu = new_uu

    # Center and shape matrix
    U = np.diag(uu)
    c = points.T @ uu            # center
    X = (points - c.reshape(1,-1)).T @ U @ (points - c.reshape(1,-1))
    Xinv = inv(X) / d
    return np.array(c), Xinv

def ellipsoid_fit(points):
    """
    Fit an axis‑aligned ellipsoid to 3D points.
    
    Parameters:
    -----------
    points : array of shape (N, 3)
    
    Returns:
    --------
    center : array (3,) of (x0, y0, z0)
    radii  : array (3,) of (a, b, c)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Objective: minimize sum of squared residuals
    def residuals(pars):
        x0, y0, z0, a, b, c = pars
        X = (x - x0) / a
        Y = (y - y0) / b
        Z = (z - z0) / c
        return np.sum(X**2 + Y**2 + Z**2 - 1)**2
    
    # Initial guess: sample mean as center, rough scale from stds
    x0, y0, z0 = np.mean(points, axis=0)
    a0, b0, c0 = np.std(points, axis=0) + 1e-6  # avoid zero
    x0, y0, z0, a0, b0, c0 = float(x0), float(y0), float(z0), float(a0), float(b0), float(c0)

    res = minimize(residuals, [x0, y0, z0, a0, b0, c0],
                   bounds=[(None, None), (None, None), (None, None),
                           (1e-6, None), (1e-6, None), (1e-6, None)])
    pars = res.x
    center = pars[:3]
    radii = pars[3:]
    return center, radii

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

def I_plane_parall(F_mag, cos_theta):
    I_parallel = F_mag/np.pi * (1 + 3/4 * cos_theta)
    return I_parallel

def compute_polarization(Ix, Iy, Iz,
                        weight,
                        n_obs,
                        flux = False, all_points = False):
    """
    Compute polarization for an observer of direction n_obs.

    Inputs:
        Ix, Iy, Iz : intensity/flux vector components
        weight: weighting coefficient to take into account only scattering contribution
        n_obs : observer direction (3-vector, not necessarily normalized)
        flux : if True, (Ix, Iy, Iz) are fluxes and not intensities, 
        so you need to get the local intensity before computing Stokes parameters.
        all_points: if True, return also the local polarization and Stokes parameters for each cell, otherwise return only the total polarization and Stokes parameters.
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
        # cos_theta_geom = np.dot(norm_surf_hat[visible], n)
        dOmega = 4*np.pi / Nall_obs  # solid angle per cell
        # dA_proj = dOmega * cos_theta_scat  # projected area toward observer nobs or use cos_theta_geom?
        # I_local = I_mag / dA_proj # I_mag is the flux in this case
        I_local =  I_plane_parall(I_mag, cos_theta_scat) 
        # I_local[cos_theta_scat==0] = 0 # cosTheta=0 menas theta=90, so max polarization
        # I_local =  I_mag / np.maximum(dA_proj, 1e-15)
        # I_local[cos_theta_scat == 0] = 0 # only consider cells that are locally visible (i.e. cosTheta>0) for polarization. Otherwise, you consider also the light that is scattered toward the observer but then absorbed by the disk itself, which is not what you want.
    else:
        I_local = I_mag
        # visible = cos_theta_scat >= 0

    # Thomson polarization fraction for the  cell
    P_local = (1 - cos_theta_scat**2) / (1 + cos_theta_scat**2)
    if type(weight) != float:
        # print('weight different from 0.34')
        weight = weight[visible]
    
    # P_local *= np.exp(-weight)
    P_local *= weight

    # --- Define a (fixed, arbitrary) sky basis with a plane perpendicular to the line-of-sight direction n
    # vectors: (e1, e2, n). e1 is the first, it will give you the cos(2\phi) which define Q param
    tmp = np.array([0.0, 0.0, 1.0])
    e2 = tmp - np.dot(tmp, n) * n  # proj of tmp onto sky plane
    # e2 = np.cross(tmp, n)
    if np.linalg.norm(e2) < 1e-6:   # avoid degeneracy if n || tmp
        tmp = np.array([1.0, 0.0, 0.0]) # so if n is along z, e1 will be along x
        e2 = tmp - np.dot(tmp, n) * n 
        # e2 = np.cross(tmp, n)
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, n) # not viceversa or you flip the sign. Like that, e1xe2 = n (as x,y,z in cartesian)
    e1 /= np.linalg.norm(e1)

    # Polarization direction vector, in the scattering plane, orthogonal to n
    cross1 = np.cross(n, I_hat)     # I_hat × n (incident × scattered)
    e_pol = np.cross(n, cross1)     # n × (I_hat × n) = proj of I_hat onto sky plane
    e_pol_mag = np.linalg.norm(e_pol, axis=1)
    e_pol/= np.maximum(e_pol_mag[:, None], 1e-20)

    # Project polarization direction onto sky plane
    cos_phi = np.dot(e_pol, e1)
    sin_phi = np.dot(e_pol, e2)
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2 * cos_phi * sin_phi 

    # Stokes parameters. 
    Q_local = I_local * P_local * cos2phi
    U_local = I_local * P_local * sin2phi
    Q = np.sum(Q_local)
    U = np.sum(U_local)
    I = np.sum(I_local)
    P = np.sqrt(Q**2 + U**2) / (I + 1e-20) #if you do sum(P_local) you have a number exceeding 1

    if all_points:
        fig, ax = plt.subplots(1,1,figsize=(10, 8))
        plt.scatter(Q_local/I_local, U_local/I_local, edgecolors='k', s = 70, vmin = -np.pi, vmax = np.pi) #color by phi
        ax.set_xlabel(r'Q/I')
        ax.set_ylabel(r'U/I')
    
    return P, I, Q, U

if __name__ == "__main__":
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps_lum, tfb_lum, Lum = data[:, 0], data[:, 1], data[:, 2]
    snaps_lum, Lum, tfb_lum = sort_list([snaps_lum, Lum, tfb_lum], tfb_lum, unique=True) 
    snaps_lum = snaps_lum.astype(int)
    time = tfb_lum[np.argmin(np.abs(snaps_lum - snap))]
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) # shape: (3, 192)
    x_heal, y_heal, z_heal = observers_xyz[0], observers_xyz[1], observers_xyz[2]
    
    photo = np.load(f'{abspath}/data/{folder}/photoNEW/{check}_photo{snap}.npz')
    x, y, z, den, Fx, Fy, Fz, alpha_rossland, alpha_scatter, alpha_abs = \
        photo['x'], photo['y'], photo['z'], photo['den'], photo['Fx'], photo['Fy'], photo['Fz'], photo['alpha_rossland'], photo['alpha_scatter'], photo['alpha_abs']
    
    kappaS = alpha_scatter/den
    kappaR = alpha_rossland/den
    # kappaA = alpha_abs/den
    # print(kappaA)
    albedo = alpha_scatter/alpha_rossland
    # albedo = alpha_scatter/(alpha_scatter + alpha_abs)
    # weight = tau_scatt
    # weight = alpha_abs/alpha_scatter

    r_vec = np.vstack((x, y, z)).T
    r_ph = np.linalg.norm(r_vec, axis=1)
    Npole_idx = np.arange(4)
    Spole_idx = np.arange(-4, 0)
    op_idx = np.arange(88, 104)

    r_hat = r_vec / np.maximum(r_ph[:, None], 1e-20)
    F_vec = np.vstack((Fx, Fy, Fz)).T
    F_mag = np.linalg.norm(F_vec, axis=1)
    F_hat = F_vec / np.maximum(F_mag[:, None], 1e-20)
    cos_Fhat_r = np.sum(F_hat * r_hat, axis=1)
    F_mag_median = np.median(F_mag)

    # PHOTOSPHERE
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x/330, y/330, z/330, s = 40)
    ax.quiver(x/330, y/330, z/330, Fx/F_mag, Fy/F_mag, Fz/F_mag, color='k', length=0.4)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xlim(-10, 2.5); ax.set_ylim(-6, 6); ax.set_zlim(-6, 6)
    ax.set_xlabel(r'x / r$_{\rm a}$', labelpad=15)
    ax.set_ylabel(r'y / r$_{\rm a}$', labelpad=15)
    ax.set_zlabel(r'z / r$_{\rm a}$', labelpad=15)
    ax.set_title('Our photosphere', fontsize=16)

    # dA = 4*np.pi*r_ph**2/len(x) # cell area
    # don't need to convert, because cancel out
    # Fx /= (prel.en_converter / prel.Rsol_cgs**2) # convert to code units
    # Fy /= (prel.en_converter / prel.Rsol_cgs**2)
    # Fz /= (prel.en_converter / prel.Rsol_cgs**2)
    phi_heal = np.arctan2(y_heal, x_heal)
    theta_heal = np.arccos(z_heal)
    longitude_heal_moll = phi_heal 
    latitude_heal_moll = np.pi/2 - theta_heal

    phi_obs = np.arctan2(y, x)
    theta_obs = np.arccos(z/r_ph)
    longitude_moll = phi_obs 
    latitude_moll = np.pi/2 - theta_obs

    # KAPPA DISTRIBUTION
    fig, axR = plt.subplots(1,1,figsize=(10,10))
    img = axR.scatter(theta_obs*radians, kappaS/kappaR, c = longitude_moll*radians, cmap='rainbow', edgecolors='k', s = 70, vmin = -np.pi, vmax = np.pi) #color by phi
    # img = axa.scatter(theta_obs*radians, kappaS/kappaA, c = longitude_moll*radians, cmap='rainbow', edgecolors='k', s = 70, vmin = -np.pi, vmax = np.pi) #color by phi
    cbar = fig.colorbar(img, label=r'$\phi_{\rm obs}$ [rad]', orientation='horizontal')
    # for ax in [axR, axa]:
    axR.set_xlim(0, 3.2)
    axR.set_ylabel(r'$\kappa_S/\kappa_R$')
    # axa.set_ylabel(r'$\kappa_S/\kappa_A$')
    axR.set_xlabel(r'$\theta_{\rm obs}$ [rad]')
    axR.set_ylim(0, np.max(kappaS/kappaR)+0.02)
    plt.suptitle(f'Snap {snap}', fontsize=20)
    plt.tight_layout()
    
    # fig = plt.figure(figsize=(10,8))
    # axx = fig.add_subplot(projection='mollweide')
    # axx.scatter(longitude_moll[op_idx], latitude_moll[op_idx])  
    # axx.grid(True)
    # axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    # axx.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    # axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    # axx.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])

    #%% FLUX FIELD
    fig = plt.figure(figsize=(30,20))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1,1,1], height_ratios=[1,1,.05], hspace=0.1, wspace = 0.2)
    axx_hist = fig.add_subplot(gs[0, 0])
    axy_hist = fig.add_subplot(gs[0, 1])
    axz_hist = fig.add_subplot(gs[0, 2])

    axx_hist.hist(Fx/F_mag_median, bins=30, color='navy', alpha=0.7)
    axx_hist.set_title('Fx distribution', fontsize=16)
    axy_hist.hist(Fy/F_mag_median, bins=30, color='darkorange', alpha=0.7)
    axy_hist.set_title('Fy distribution', fontsize=16)
    axz_hist.hist(Fz/F_mag_median, bins=30, color='forestgreen', alpha=0.7)
    axz_hist.set_title('Fz distribution', fontsize=16)
    axx_hist.set_xlabel(r'$F_x/|F_{\rm med}|$')
    axy_hist.set_xlabel(r'$F_y/|F_{\rm med}|$')
    axz_hist.set_xlabel(r'$F_z/|F_{\rm med}|$')
    for ax in [axx_hist, axy_hist, axz_hist]:
        # ax.set_xlim(0,8)
        ax.set_ylim(0, 30)
    plt.tight_layout()

    lon_1d = longitude_moll
    lat_1d = latitude_moll
    # Define a regular grid in (lon, lat) for visualization
    nlon = 360
    nlat = 180
    lon_grid = np.linspace(lon_1d.min(), lon_1d.max(), nlon)
    lat_grid = np.linspace(lat_1d.min(), lat_1d.max(), nlat)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    data_1d = np.abs(Fx) / F_mag_median
    data_grid = griddata(
    points=(lon_1d, lat_1d),
    values=data_1d,
    xi=(lon_mesh, lat_mesh),
    method='linear')

    axx = fig.add_subplot(gs[1, 0], projection='mollweide')
    axx.pcolormesh(lon_mesh, lat_mesh, data_grid, cmap='rainbow', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1))  #color by intensity
    axx.grid(True)
    axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axx.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axx.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])

    data_1d = np.abs(Fy) / F_mag_median
    data_grid = griddata(
    points=(lon_1d, lat_1d),
    values=data_1d,
    xi=(lon_mesh, lat_mesh),
    method='linear')

    axy = fig.add_subplot(gs[1, 1], projection='mollweide')
    axy.pcolormesh(lon_mesh, lat_mesh, data_grid, cmap='rainbow', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1))  #color by intensity
    axy.grid(True)
    axy.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axy.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    axy.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axy.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])

    data_1d = np.abs(Fz) / F_mag_median
    data_grid = griddata(
    points=(lon_1d, lat_1d),
    values=data_1d,
    xi=(lon_mesh, lat_mesh),
    method='linear')
    
    axz = fig.add_subplot(gs[1, 2], projection='mollweide')
    img = axz.pcolormesh(lon_mesh, lat_mesh, data_grid, cmap='rainbow', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1))  #color by intensity
    axz.grid(True)
    axz.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axz.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    axz.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axz.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])
    
    cbar_ax = fig.add_subplot(gs[2, 0:3]) 
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label =r'|F$_{\rm i}|$/F$_{\rm r, med}$')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)

    #%% POLARIZATION
    P_all = np.zeros(len(x))
    q_all = np.zeros(len(x))
    u_all = np.zeros(len(x))
    for idx in range(len(x)):
        # n_obs = [x[idx], y[idx], z[idx]] # observers = my photospheric cells
        n_obs = [x_heal[idx], y_heal[idx], z_heal[idx]]
        P, I, Q, U = compute_polarization(Fx, Fy, Fz, albedo, n_obs, flux=True)
        P_all[idx] = P
        q_all[idx] = Q/I
        u_all[idx] = U/I
    print(f'Mean P: {np.mean(P_all):.2f}, \nMedian P: {np.median(P_all):.2f}, \nMax P: {np.max(P_all):.2f}')
    
    #%% POLARIZATION MAPS
    lon_1d_heal = longitude_heal_moll
    lat_1d_heal = latitude_heal_moll
    # Define a regular grid in (lon, lat) for visualization
    nlon = 360
    nlat = 180
    lon_heal_grid = np.linspace(lon_1d_heal.min(), lon_1d_heal.max(), nlon)
    lat_heal_grid = np.linspace(lat_1d_heal.min(), lat_1d_heal.max(), nlat)
    lon_heal_mesh, lat_heal_mesh = np.meshgrid(lon_heal_grid, lat_heal_grid)

    data_grid_q = griddata(
    points=(lon_1d_heal, lat_1d_heal), 
    values=q_all,
    xi=(lon_heal_mesh, lat_heal_mesh),
    method='linear')

    data_grid_u = griddata(
    points=(lon_1d_heal, lat_1d_heal), 
    values=u_all,
    xi=(lon_heal_mesh, lat_heal_mesh),
    method='linear')
    
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(1, 2, hspace=0.1, wspace = 0.2)
    axq = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = axq.pcolormesh(lon_mesh, lat_mesh, data_grid_q, cmap='magma', vmin = -.3, vmax = .3)  
    cbar = plt.colorbar(img, orientation='horizontal', pad = 0.1, label =r'$Q/I$ ')
    cbar.ax.tick_params(which='major',length = 6)
    cbar.ax.tick_params(which='minor',length = 4)
    axu = fig.add_subplot(gs[0, 1], projection='mollweide')
    img = axu.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_u, cmap='magma', vmin = -.3, vmax = .3) 
    cbar = plt.colorbar(img, orientation='horizontal', pad = 0.1, label =r'$U/I$')
    cbar.ax.tick_params(which='major',length = 6)
    for ax in [axq, axu]:
        ax.grid(True)
        ax.set_xticks(np.radians(np.arange(-180, 181, 90))) 
        ax.set_xticklabels(['180°', '270°', '0°','90°', '180°']) #'-180°', '-90°', '0°','90°', '180°']
        ax.set_yticks(np.radians(np.arange(-90, 91, 45))) 
        ax.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])
    plt.suptitle(f't = {time:.1f} ' + r'$t_{\rm fb}$', fontsize=20, x = 0.5, y = .71)
    plt.savefig(f'{abspath}/Figs/3.paperPolarization/QU_map{snap}.png', dpi=300, bbox_inches='tight')
    
    #%%
    fig, ax = plt.subplots(1,1,figsize=(10, 8))
    img = ax.scatter(theta_heal*radians, P_all, c = longitude_moll*radians, cmap='rainbow', edgecolors='k', s = 70, vmin = -np.pi, vmax = np.pi) #color by phi
    cbar = fig.colorbar(img, label=r'$\phi_{\rm obs}$ [rad]', orientation='horizontal')
    ax.set_xlabel(r'$\theta_{\rm obs}$ [rad]')
    ax.set_ylabel('Polarization Fraction P')
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, np.max(P_all)+0.02)
    plt.suptitle(f'Snap {snap}', fontsize=16)
    plt.tight_layout()

    data_grid_F = griddata(
    points=(lon_1d, lat_1d),
    values=F_mag/F_mag_median,
    xi=(lon_mesh, lat_mesh),
    method='linear')

    lon_1d_heal = longitude_heal_moll
    lat_1d_heal = latitude_heal_moll
    # Define a regular grid in (lon, lat) for visualization
    nlon = 360
    nlat = 180
    lon_heal_grid = np.linspace(lon_1d_heal.min(), lon_1d_heal.max(), nlon)
    lat_heal_grid = np.linspace(lat_1d_heal.min(), lat_1d_heal.max(), nlat)
    lon_heal_mesh, lat_heal_mesh = np.meshgrid(lon_heal_grid, lat_heal_grid)
    data_grid_P = griddata(
    points=(lon_1d_heal, lat_1d_heal), 
    values=P_all,
    xi=(lon_heal_mesh, lat_heal_mesh),
    method='linear')

    data_grid_alb = griddata(
    points=(lon_1d, lat_1d),
    values=albedo, 
    xi=(lon_mesh, lat_mesh),
    method='linear')
    
    fig = plt.figure(figsize=(30,15))
    gs = gridspec.GridSpec(1, 3, hspace=0.1, wspace = 0.2)
    axf = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = axf.pcolormesh(lon_mesh, lat_mesh, data_grid_F, cmap='rainbow', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e3))  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', pad = 0.1, label =r'$|\,\vec{F}\,|/|\,\vec{F}\,|_{\rm med}$ ')
    cbar.ax.tick_params(which='major',length = 6)
    cbar.ax.tick_params(which='minor',length = 4)

    axalb = fig.add_subplot(gs[0, 1], projection='mollweide')
    img = axalb.pcolormesh(lon_mesh, lat_mesh, data_grid_alb, cmap='rainbow', vmin = 0, vmax = 1)  
    cbar = plt.colorbar(img, orientation='horizontal', pad = 0.1, label =r'$\sigma_{\rm s}/\alpha_{\rm Ross}$')
    cbar.ax.tick_params(which='major',length = 6)
    cbar.ax.tick_params(which='minor',length = 4)

    axP = fig.add_subplot(gs[0, 2], projection='mollweide')
    img = axP.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_P, cmap='magma', vmin = Pmin, vmax = Pmax)  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', pad = 0.1, label =r'P')
    cbar.ax.tick_params(which='major',length = 6)
    for ax in [axf, axalb, axP]:
        ax.grid(True)
        ax.set_xticks(np.radians(np.arange(-180, 181, 90))) 
        ax.set_xticklabels(['180°', '270°', '0°','90°', '180°']) #'-180°', '-90°', '0°','90°', '180°']
        ax.set_yticks(np.radians(np.arange(-90, 91, 45))) 
        ax.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])
    plt.suptitle(f't = {time:.1f} ' + r'$t_{\rm fb}$', fontsize=25, x = 0.5, y = .58)
    plt.savefig(f'{abspath}/Figs/3.paperPolarization/FP_map{snap}.png', dpi=300, bbox_inches='tight')

    #%% WHAT IF PHOTOSPHERE IS A SPHERE (AND SO FLUX IS RADIAL)?
    F_x_rad = F_mag * x_heal
    F_y_rad = F_mag * y_heal
    F_z_rad = F_mag * z_heal
    F_r_vec = np.vstack((F_x_rad, F_y_rad, F_z_rad)).T
    P_radial = np.zeros(len(x_heal))

    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(1, 2, hspace=0.1, wspace = 0.2)
    ax = fig.add_subplot(gs[0, 0], projection = '3d')
    ax.scatter(x_heal, y_heal, z_heal, s = 40)
    ax.quiver(x_heal, y_heal, z_heal, F_x_rad/F_mag, F_y_rad/F_mag, F_z_rad/F_mag, length=0.1, color='k')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z') 
    # ax.set_xlim(-10, 2.5); ax.set_ylim(-6, 6); ax.set_zlim(-6, 6)
    ax.set_xlabel(r'x / r$_{\rm a}$', labelpad=15)
    ax.set_ylabel(r'y / r$_{\rm a}$', labelpad=15)
    ax.set_zlabel(r'z / r$_{\rm a}$', labelpad=15)
    ax.set_title('Spherical model', fontsize=16)

    for idx in range(len(x)):
        # n_obs = [x[idx], y[idx], z[idx]]
        n_obs = [x_heal[idx], y_heal[idx], z_heal[idx]]
        P, I, Q, U = compute_polarization(F_x_rad, F_y_rad, F_z_rad, albedo, n_obs, flux=True)
        P_radial[idx] = P
    print(f'Spherical photosphere \n----------\nMean P: {np.mean(P_radial):.2f}, \nMedian P: {np.median(P_radial):.2f}, \nMax P: {np.max(P_radial):.2f}')

    data_grid_Pr = griddata(
    points=(lon_1d_heal, lat_1d_heal),
    values=P_radial,
    xi=(lon_heal_mesh, lat_heal_mesh),
    method='linear')

    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(1, 2, hspace=0.1, wspace = 0.2)
    axx = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = axx.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_Pr, cmap='magma', vmin = Pmin, vmax = Pmax)  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', label =r'P')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
    axx.grid(True)
    axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axx.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axx.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])
    axx.set_title('If photosphere is a sphere', fontsize=16, y = 1.2)

    fig = plt.figure(figsize=(30,15))
    axx = fig.add_subplot(gs[0,0], projection='mollweide')
    img = axx.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_Pr/data_grid_P, cmap='coolwarm', norm=colors.LogNorm(vmin=1e-1, vmax=10))  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', label =r'P$_{\rm s}/P$')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
    axx.grid(True)
    axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axx.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axx.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])

    #%%
    P_z, I_Z, Q_z, U_z = compute_polarization(Fx, Fy, Fz, albedo, [0, 0, 1], flux=True, all_points=True)

    #%% WHAT IF PHOTOSPHERE IS AN ELLIPSOID?
    # a = (np.max(x[op_idx]) - np.min(x[op_idx]))/2
    # b = (np.max(y[op_idx]) - np.min(y[op_idx]))/2
    # c = (np.max(z[Npole_idx]) - np.min(z[Spole_idx]))/2
    # x0 = (np.max(x[op_idx]) + np.min(x[op_idx]))/2
    # y0 = (np.max(y[op_idx]) + np.min(y[op_idx]))/2
    # z0 = (np.max(z[Npole_idx]) + np.min(z[Spole_idx]))/2

    ## centre, abc = ellipsoid_fit(np.column_stack([x, y, z]))
    centre, Xinv = mvee_fit(np.column_stack([x, y, z]))
    # To get radii and axes, diagonalize the shape matrix
    w, v = np.linalg.eigh(Xinv)  # w = eigenvalues, v = eigenvectors
    w = np.clip(w, 1e-15, None)   # avoid negative eigenvalues from rounding
    abc = 1.0 / np.sqrt(w)
    a, b, c = abc
    x0, y0, z0 = centre

    print(f'Fitted ellipsoid parameters: a={a}, b={b}, c={c}, center=({x0}, {y0}, {z0})')
    x_ell, y_ell, z_ell = ellipsoid_surface(prel.NSIDE, a, b, c, x0=x0, y0=y0, z0=z0, healpix=True, stay_helpix=True)

    F_vec = ellipsoid_unit_normal(x_ell, y_ell, z_ell, a, b, c, x0=x0, y0=y0, z0=z0)
    Fx_ell, Fy_ell, Fz_ell = F_vec[:,0], F_vec[:,1], F_vec[:,2] 

    # plot here so they are normalized
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(1, 2, hspace=0.1, wspace = 0.2)
    ax = fig.add_subplot(gs[0, 0], projection = '3d')
    ax.scatter(x_ell/330, y_ell/330, z_ell/330, s = 40)
    ax.quiver(x_ell/330, y_ell/330, z_ell/330, Fx_ell, Fy_ell, Fz_ell, length=0.4, color='k')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xlim(-10, 2.5); ax.set_ylim(-6, 6); ax.set_zlim(-6, 6)
    ax.set_xlabel(r'x / r$_{\rm a}$', labelpad=15)
    ax.set_ylabel(r'y / r$_{\rm a}$', labelpad=15)
    ax.set_zlabel(r'z / r$_{\rm a}$', labelpad=15)
    ax.set_title('Ellipsoid model', fontsize=16)
    # plt.tight_layout()

    Fx_ell *= F_mag 
    Fy_ell *= F_mag 
    Fz_ell *= F_mag
    P_ell = np.zeros(len(Fx_ell))

    for idx in range(len(x)):
        # n_obs = [x[idx], y[idx], z[idx]]
        n_obs = [x_heal[idx], y_heal[idx], z_heal[idx]]
        P, I, Q, U = compute_polarization(Fx_ell, Fy_ell, Fz_ell, albedo, n_obs, flux=True)
        P_ell[idx] = P
    print(f'Ellipsoidal photosphere \n----------\nMean P: {np.mean(P_ell):.2f}, \nMedian P: {np.median(P_ell):.2f}, \nMax P: {np.max(P_ell):.2f}')

    data_grid_Pell = griddata(
    points=(lon_1d_heal, lat_1d_heal),
    values=P_ell,
    xi=(lon_heal_mesh, lat_heal_mesh),
    method='linear')

    fig = plt.figure(figsize=(30,15))
    axx = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = axx.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_Pell, cmap='magma', vmin = Pmin, vmax = Pmax)  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', label =r'P')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
    axx.grid(True)
    axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axx.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axx.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])
    axx.set_title('If photosphere is an ellipsoid', fontsize=16, y = 1.2)

    fig = plt.figure(figsize=(30,15))
    axx = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = axx.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_Pell/data_grid_P, cmap='coolwarm', norm = colors.LogNorm(vmin=1e-1, vmax=10))  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', label =r'P$_{\rm e}/P$')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
    axx.grid(True)
    axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axx.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axx.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])


    # %% IF OUR PHOTOSPHERE BUT EQUAL MAGNITUD (1) FOR INTENSITY
    Fx_normal = Fx / F_mag 
    Fy_normal = Fy / F_mag
    Fz_normal = Fz / F_mag 
    F_vec_normal = np.vstack((Fx_normal, Fy_normal, Fz_normal)).T
    
    P_normal = np.zeros(len(x))
    for idx in range(len(x)):
        # n_obs = [x[idx], y[idx], z[idx]] # observers = my photospheric cells
        n_obs = [x_heal[idx], y_heal[idx], z_heal[idx]]
        P, I, Q, U = compute_polarization(Fx_normal, Fy_normal, Fz_normal, albedo, n_obs, flux=False) # put false so uses F as intensity and magnitude doesn't change
        P_normal[idx] = P
    print(f'Equal intensity \n----------\nMean P: {np.mean(P_normal):.2f}, \nMedian P: {np.median(P_normal):.2f}, \nMax P: {np.max(P_normal):.2f}')

    data_grid_Pnorm = griddata(
    points=(lon_1d_heal, lat_1d_heal),
    values=P_normal,
    xi=(lon_heal_mesh, lat_heal_mesh),
    method='linear')

    fig = plt.figure(figsize=(30,15))
    axx = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = axx.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_Pnorm, cmap='magma', vmin = Pmin, vmax = Pmax)  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', label =r'P')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
    axx.grid(True)
    axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axx.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axx.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])
    axx.set_title('If photosphere have same intensity everywhere', fontsize=16, y = 1.2)

    fig = plt.figure(figsize=(30,15))
    axx = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = axx.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_Pnorm/data_grid_P, cmap='coolwarm', norm = colors.LogNorm(vmin=1e-1, vmax=10))  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', label =r'P$_{\rm norm}/P$')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
    axx.grid(True)
    axx.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axx.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    axx.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axx.set_yticklabels(['180°','135°', '90°', '45°', '0°'])
    

