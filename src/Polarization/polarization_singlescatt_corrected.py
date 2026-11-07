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
from scipy.linalg import inv
from Utilities.sections import make_slices
from Utilities.basic_units import radians
from Utilities.operators import sort_list
from src.Polarization.geometries import ellipsoid_surface, ellipsoid_unit_normal
#%% Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'MG' 
snap = 420
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
Pmin = 0
Pmax = 0.2

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


def I_plane_parall(F_mag, cos_theta):
    I_parallel = F_mag/np.pi * (1 + 3/4 * cos_theta)
    return I_parallel


def compute_single_scattering_polarization(Fx, Fy, Fz, x, y, z, tau_es, n_obs,
                                           exact_one_scattering=False,
                                           all_points=False):
    """
    Approximate polarization from a photospheric source plus one Thomson scattering
    in the material above the photosphere.

    Parameters
    ----------
    Fx, Fy, Fz : arrays
        FLD radiative-flux components evaluated at the photosphere.
    x, y, z : arrays
        Photospheric positions for the HEALPix rays.
    tau_es : array
        Electron-scattering optical depth integrated from the photosphere outward
        along each ray.
    n_obs : array-like, shape (3,)
        Observer direction.
    exact_one_scattering : bool
        If False, use the optically thin Brown-McLean weight w=tau_es.
        If True, use w=tau_es*exp(-tau_es), the probability of exactly one
        interaction in a Poisson model.
    all_points : bool
        If True, also make a Q/I--U/I diagnostic plot for the local scattered
        contributions.

    Returns
    -------
    P : float
        Net polarization including dilution by direct unpolarized light.
    I_tot : float
        Direct + scattered intensity in common arbitrary units.
    Q, U : float
        Net linear Stokes parameters in the same units.
    I_direct, I_scat : float
        Direct and scattered contributions separately.

    Notes
    -----
    The HEALPix photosphere is treated as a set of radial surface elements with
    dA ~= r_ph^2 dOmega. The photospheric FLD flux sets the source luminosity.
    tau_es controls how much radiation is scattered; it does NOT multiply the
    local Thomson polarization fraction.
    """
    Fx = np.asarray(Fx, dtype=float)
    Fy = np.asarray(Fy, dtype=float)
    Fz = np.asarray(Fz, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    tau_es = np.asarray(tau_es, dtype=float)

    if not (len(Fx) == len(Fy) == len(Fz) == len(x) == len(y) == len(z) == len(tau_es)):
        raise ValueError('Fx, Fy, Fz, x, y, z, and tau_es must have the same length')

    n = np.asarray(n_obs, dtype=float)
    n /= np.linalg.norm(n)

    # Photospheric geometry and FLD flux.
    r_vec = np.column_stack((x, y, z))
    r_ph = np.linalg.norm(r_vec, axis=1)
    r_hat = r_vec / np.maximum(r_ph[:, None], 1e-30)

    F_vec = np.column_stack((Fx, Fy, Fz))
    F_mag = np.linalg.norm(F_vec, axis=1)
    F_hat = F_vec / np.maximum(F_mag[:, None], 1e-30)

    # Equal HEALPix solid angle. Approximating the local surface element as radial:
    # dA ~= r_ph^2 dOmega.
    dOmega = 4.0 * np.pi / len(F_mag)
    dA = r_ph**2 * dOmega

    # Outward photospheric luminosity carried by each ray.
    # Use the radial component of F so inward numerical flux does not contribute.
    F_out = np.maximum(np.sum(F_vec * r_hat, axis=1), 0.0)
    L_ray = F_out * dA

    # Scattering angle: incident direction = local FLD flux direction;
    # scattered direction = observer.
    mu = np.clip(F_hat @ n, -1.0, 1.0)

    # --- Direct, unpolarized photospheric radiation ---------------------------
    # In the same plane-parallel approximation already used in this script,
    # only surface elements facing the observer contribute directly.
    mu_surf = np.clip(r_hat @ n, -1.0, 1.0)
    visible = mu_surf > 0.0
    I_emit = np.zeros_like(F_mag)
    I_emit[visible] = I_plane_parall(F_out[visible], mu_surf[visible])
    # dL/dOmega = I * projected area = I * mu * dA.
    I_direct_local = np.zeros_like(F_mag)
    I_direct_local[visible] = I_emit[visible] * mu_surf[visible] * dA[visible]
    I_direct = np.sum(I_direct_local)

    # --- Single-scattered component ------------------------------------------
    if exact_one_scattering:
        scatter_weight = tau_es * np.exp(-tau_es)
    else:
        # First-order optically thin / Brown-McLean limit.
        scatter_weight = tau_es

    # Thomson differential cross-section angular factor.
    thomson_prefactor = 3.0 / (16.0 * np.pi)
    I_scat_local = (thomson_prefactor * L_ray * scatter_weight * (1.0 + mu**2))

    # Polarization direction is perpendicular to the scattering plane.
    e_pol = np.cross(F_hat, n)
    e_pol_mag = np.linalg.norm(e_pol, axis=1)
    valid_pol = e_pol_mag > 1e-20
    e_pol[valid_pol] /= e_pol_mag[valid_pol, None]
    e_pol[~valid_pol] = 0.0

    # Fixed right-handed sky basis (e1, e2, n), matching the convention used
    # elsewhere in the script as closely as possible.
    tmp = np.array([0.0, 0.0, 1.0])
    e2 = tmp - np.dot(tmp, n) * n
    if np.linalg.norm(e2) < 1e-6:
        tmp = np.array([1.0, 0.0, 0.0])
        e2 = tmp - np.dot(tmp, n) * n
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, n)
    e1 /= np.linalg.norm(e1)

    cos_phi = e_pol @ e1
    sin_phi = e_pol @ e2
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2.0 * cos_phi * sin_phi

    # Since I_scat already contains (1+mu^2), multiplying by
    # P_Th=(1-mu^2)/(1+mu^2) leaves the expected (1-mu^2) factor.
    P_thomson = np.zeros_like(mu)
    P_thomson[valid_pol] = (1.0 - mu[valid_pol]**2) / (1.0 + mu[valid_pol]**2)

    Q_local = I_scat_local * P_thomson * cos2phi
    U_local = I_scat_local * P_thomson * sin2phi

    I_scat = np.sum(I_scat_local)
    Q = np.sum(Q_local)
    U = np.sum(U_local)
    I_tot = I_direct + I_scat
    P = np.sqrt(Q**2 + U**2) / max(I_tot, 1e-30)

    if all_points:
        mask = I_scat_local > 0
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(Q_local[mask] / I_scat_local[mask],
                   U_local[mask] / I_scat_local[mask],
                   edgecolors='k', s=70)
        ax.set_xlabel(r'$Q_{\rm sc}/I_{\rm sc}$')
        ax.set_ylabel(r'$U_{\rm sc}/I_{\rm sc}$')

    return P, I_tot, Q, U, I_direct, I_scat

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
    if check != 'MG':
        data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
        snaps_lum, tfb_lum, Lum = data[:, 0], data[:, 1], data[:, 2]
        snaps_lum, Lum, tfb_lum = sort_list([snaps_lum, Lum, tfb_lum], tfb_lum, unique=True) 
        snaps_lum = snaps_lum.astype(int)
        time = tfb_lum[np.argmin(np.abs(snaps_lum - snap))]
    else:
        time = np.loadtxt(f'{abspath}/data/{folder}/snap_{snap}/Time_snap_{snap}.txt')
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) # shape: (3, 192)
    x_heal, y_heal, z_heal = observers_xyz[0], observers_xyz[1], observers_xyz[2]

    if check != 'MG':
        photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
        x, y, z, den, Fx, Fy, Fz, alpha_rossland, alpha_scatter, alpha_abs, tau_es = \
            photo['x'], photo['y'], photo['z'], photo['den'], photo['Fx'], photo['Fy'], photo['Fz'], photo['alpha_rossland'], photo['alpha_scatter'], photo['alpha_abs'], photo['los_scatt']
    else:
        g = 4
        photo = np.load(f"{abspath}/data/{folder}/photo/{check}_photo{snap}.npy", allow_pickle=True).item()
        x = np.array(photo[g]['x'])
        y = np.array(photo[g]['y'])
        z = np.array(photo[g]['z'])
        tau_rossland = np.array(photo[g][f'tau_rossland'])  
        tau_es = np.array(photo[g][f'tau_scatter']) 
        alpha_rossland = np.array(photo[g][f'alpha_rossland'])
        alpha_abs = np.array(photo[g][f'alpha_abs'])
        alpha_scatter = np.array(photo[g][f'alpha_scatter'])
        print(alpha_rossland, alpha_scatter)
        den = np.array(photo[g]['den'])
        r_ph = np.sqrt(x**2 + y**2 + z**2)
        Eg = np.array(photo[g][f'Eg{g}'])
        F = prel.c_cgs * Eg * prel.en_den_converter
        Fx = F * x/r_ph
        Fy = F * y/r_ph
        Fz = F * z/r_ph
                
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
    img = axR.scatter(theta_obs*radians, kappaS/kappaR, c = longitude_moll*radians, cmap='viridis', edgecolors='k', s = 70, vmin = -np.pi, vmax = np.pi) #color by phi
    # img = axa.scatter(theta_obs*radians, kappaS/kappaA, c = longitude_moll*radians, cmap='viridis', edgecolors='k', s = 70, vmin = -np.pi, vmax = np.pi) #color by phi
    cbar = fig.colorbar(img, label=r'$\phi_{\rm obs}$ [rad]', orientation='horizontal')
    # for ax in [axR, axa]:
    axR.set_xlim(0, 3.2)
    axR.set_ylabel(r'$\kappa_S/\kappa_R$')
    # axa.set_ylabel(r'$\kappa_S/\kappa_A$')
    axR.set_xlabel(r'$\theta_{\rm obs}$ [rad]')
    axR.set_ylim(0, np.max(kappaS/kappaR)+0.02)
    plt.suptitle(f'Snap {snap}', fontsize=20)
    plt.tight_layout()
    

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
    axx.pcolormesh(lon_mesh, lat_mesh, data_grid, cmap='viridis', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1))  #color by intensity
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
    axy.pcolormesh(lon_mesh, lat_mesh, data_grid, cmap='viridis', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1))  #color by intensity
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
    img = axz.pcolormesh(lon_mesh, lat_mesh, data_grid, cmap='viridis', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1))  #color by intensity
    axz.grid(True)
    axz.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    axz.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    axz.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    axz.set_yticklabels(['180°','135°', '90°', '45°', '0°']) #['-90°', '-45°', '0°', '45°','90°'])
    
    cbar_ax = fig.add_subplot(gs[2, 0:3]) 
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label =r'|F$_{\rm i}|$/F$_{\rm r, med}$')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)

    #%% POLARIZATION: PHOTOSPHERE + SINGLE THOMSON SCATTERING
    P_all = np.zeros(len(x_heal))
    q_all = np.zeros(len(x_heal))
    u_all = np.zeros(len(x_heal))
    Idir_all = np.zeros(len(x_heal))
    Iscat_all = np.zeros(len(x_heal))

    # False -> strict first-order optically thin weight w=tau_es.
    # True  -> finite-tau check w=tau_es*exp(-tau_es), i.e. exactly one scattering.
    exact_one_scattering = False

    for idx in range(len(x_heal)):
        n_obs = [x_heal[idx], y_heal[idx], z_heal[idx]]
        P, I, Q, U, I_direct, I_scat = compute_single_scattering_polarization(
            Fx, Fy, Fz, x, y, z, tau_es, n_obs,
            exact_one_scattering=exact_one_scattering
        )
        P_all[idx] = P
        q_all[idx] = Q / max(I, 1e-30)
        u_all[idx] = U / max(I, 1e-30)
        Idir_all[idx] = I_direct
        Iscat_all[idx] = I_scat

    print(
        f'Single-scattering model\n'
        f'-----------------------\n'
        f'Mean P: {np.mean(P_all):.4f}\n'
        f'Median P: {np.median(P_all):.4f}\n'
        f'Max P: {np.max(P_all):.4f}\n'
        f'Median I_scat/I_direct: {np.median(Iscat_all / np.maximum(Idir_all, 1e-30)):.4f}'
    )
    
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
    img = ax.scatter(theta_heal*radians, P_all, c = longitude_moll*radians, cmap='viridis', edgecolors='k', s = 70, vmin = -np.pi, vmax = np.pi) #color by phi
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
    img = axf.pcolormesh(lon_mesh, lat_mesh, data_grid_F, cmap='viridis', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e3))  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', pad = 0.1, label =r'$|\,\vec{F}\,|/|\,\vec{F}\,|_{\rm med}$ ')
    cbar.ax.tick_params(which='major',length = 6)
    cbar.ax.tick_params(which='minor',length = 4)

    axalb = fig.add_subplot(gs[0, 1], projection='mollweide')
    img = axalb.pcolormesh(lon_mesh, lat_mesh, data_grid_alb, cmap='viridis', vmin = 0, vmax = 1)  
    cbar = plt.colorbar(img, orientation='horizontal', pad = 0.1, label =r'$\sigma_{\rm s}/\alpha_{\rm Ross}$')
    cbar.ax.tick_params(which='major',length = 6)
    cbar.ax.tick_params(which='minor',length = 4)

    axP = fig.add_subplot(gs[0, 2], projection='mollweide')
    img = axP.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_P, cmap='viridis', vmin = Pmin, vmax = Pmax)  #color by intensity
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