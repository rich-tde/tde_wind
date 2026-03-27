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
from Utilities.basic_units import radians
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

def compute_polarization(nx, ny, nz,
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
        I, Q, U : Stokes parameters 
    """
    Nall_obs = len(nx) if isinstance(nx, np.ndarray) else 1
    # Normalize observer direction (i.e. scattered light direction)
    n_obs = np.array(n_obs)
    n = n_obs / np.linalg.norm(n_obs)

    # Positions
    norm_surf_vec = np.vstack((nx, ny, nz)).T # shape: (192,3)
    norm_mag = np.linalg.norm(norm_surf_vec, axis=1)
    norm_surf_hat = norm_surf_vec / norm_mag[:, None]   

    # Intensity vectors
    I_vec = np.vstack((Ix, Iy, Iz)).T
    I_mag = np.linalg.norm(I_vec, axis=1)
    # Intensity vectors avoiding division by zero. I_hat and n will define the scattering plane
    I_hat = I_vec / np.maximum(I_mag[:, None], 1e-20)

    # Visibility condition
    mu_surface = np.dot(norm_surf_hat, n)
    visible = mu_surface > 0
    # weights = np.maximum(mu_surface, 0) 
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
    cross1 = np.cross(I_hat, n)     # I_hat × n (incident × scattered)
    e_pol = np.cross(n, cross1)     # n × (I_hat × n) = proj of I_hat onto sky plane

    e_pol_mag = np.linalg.norm(e_pol, axis=1)
    e_pol/= np.maximum(e_pol_mag[:, None], 1e-20)

    # Project polarization direction onto sky plane
    cos_phi = np.dot(e_pol, e1)
    sin_phi = np.dot(e_pol, e2)
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2 * cos_phi * sin_phi 
   
    # Intensity weighted by the area projected toward the observer
    I_local = I_mag #* np.maximum(mu_surface, 0)#/ dA_proj

    # Stokes parameters. 
    # I_local * P_local make you consider only the light that is locally scattered
    Q = np.sum(I_local * P_local * cos2phi)
    U = np.sum(I_local * P_local * sin2phi)
    I = np.sum(I_local)
    P = np.sqrt(Q**2 + U**2) / (I + 1e-20) #if you do sum(P_local) you have a number exceeding 1
    
    return P, I, Q, U

## TESTS -----------------------------------------------------------------
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
n_obs = [1, 0, 0]  
P, I, Q, U = compute_polarization(
    x_obs, y_obs, z_obs,
    Fx_obs, Fy_obs, Fz_obs,
    n_obs)
print(f"n_obs: {n_obs}, P = {P}")
n_obs = [0, 1, 0]  
P, I, Q, U = compute_polarization(
    x_obs, y_obs, z_obs,
    Fx_obs, Fy_obs, Fz_obs,
    n_obs)
print(f"n_obs: {n_obs}, P = {P}")
n_obs = [0, 0, 1]  
P, I, Q, U = compute_polarization(
    x_obs, y_obs, z_obs,
    Fx_obs, Fy_obs, Fz_obs,
    n_obs)
print(f"n_obs: {n_obs}, P = {P}\n---------")
#%%
print("TEST ellipsoid surface")
a = 1.0
b = 1.0
c_all = np.arange(0.1, 1.1, 0.1)
n_obs_all_params = [[[1, 0, 0], 'solid', 'navy'],
             [[-1, 0, 0], 'dashed', 'dodgerblue'],
             [[0, 1, 0], 'solid', 'darkorange'],
             [[0, -1, 0], 'dashed', 'r'],
             [[0, 0, 1], 'solid', 'forestgreen'],
             [[0, 0, -1], 'dashed', 'yellowgreen'],
             [[1, 1, 1], 'dotted', 'k']]
n_obs_all = [params[0] for params in n_obs_all_params]
P_HR_n = np.zeros((len(c_all), len(n_obs_all)))

for h_idx, c in enumerate(c_all):
    x_obs, y_obs, z_obs = ellipsoid_surface(1e3, a, b, c)
    I_vec = ellipsoid_normal(x_obs, y_obs, z_obs, a, b, c)
    Ix_obs, Iy_obs, Iz_obs = I_vec[:,0], I_vec[:,1], I_vec[:,2]
    if not np.allclose(np.sum(x_obs), 0, atol=1e-10):
        print(f"Warning: x-coordinates not symmetric for c={c}. sum(x) = {np.sum(x_obs)}")
    if not np.allclose(np.sum(y_obs), 0, atol=1e-10):
        print(f"Warning: y-coordinates not symmetric for c={c}. sum(y) = {np.sum(y_obs)}")
    if not np.allclose(np.sum(z_obs), 0, atol=1e-10):
        print(f"Warning: z-coordinates not symmetric for c={c}. sum(z) = {np.sum(z_obs)}")

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection = '3d')
    # ax.scatter(x_obs, y_obs, z_obs, s = 40)
    # ax.quiver(x_obs, y_obs, z_obs, Ix_obs, Iy_obs, Iz_obs, length=0.1, color='k')
    # ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    # ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_zlim(-1.45, 1.45)
    # plt.tight_layout()

    for n_idx in range(len(n_obs_all)):
        n_obs = n_obs_all[n_idx]
        P, I, Q, U = compute_polarization(x_obs, y_obs, z_obs, Ix_obs, Iy_obs, Iz_obs, n_obs)
        P_HR_n[h_idx, n_idx] = P

plt.figure(figsize=(8,8))
for n_idx in range(len(n_obs_all)):   
    n_obs = n_obs_all[n_idx] 
    plt.plot(c_all, P_HR_n[:, n_idx], label=r'$n_{\rm obs}$'+ f' = ({n_obs[0]}, {n_obs[1]}, {n_obs[2]})', ls = n_obs_all_params[n_idx][1], c = n_obs_all_params[n_idx][2]) # color by observer direction
plt.xlabel('c')
plt.ylabel('Polarization Fraction P')
plt.legend(fontsize=16)

#%%
print("TEST ellipsoid surface with modified healpix observers")
a = 1.0
b = 1.0
c_all = np.arange(0.1, 1.1, 0.1)
n_obs_all_params = [[[1, 0, 0], 'solid', 'navy', 5],
             [[-1, 0, 0], 'dashed', 'dodgerblue', 4],
             [[0, 1, 0], 'solid', 'darkorange', 2],
             [[0, -1, 0], 'dashed', 'r', 1.5],
             [[0, 0, 1], 'solid', 'forestgreen', 2],
             [[0, 0, -1], 'dashed', 'yellowgreen', 2],
             [[1, 1, 1], 'dotted', 'k', 2]]
n_obs_all = [params[0] for params in n_obs_all_params]

P_HR_n = np.zeros((len(c_all), len(n_obs_all)))
for h_idx, c in enumerate(c_all):
    x_obs, y_obs, z_obs = ellipsoid_surface(4, a, b, c, healpix=True)
    I_vec = ellipsoid_normal(x_obs, y_obs, z_obs, a, b, c)
    Ix_obs, Iy_obs, Iz_obs = I_vec[:,0], I_vec[:,1], I_vec[:,2]
    if not np.allclose(np.sum(x_obs), 0, atol=1e-10):
        print(f"Warning: x-coordinates not symmetric for c={c}. sum(x) = {np.sum(x_obs)}")
    if not np.allclose(np.sum(y_obs), 0, atol=1e-10):
        print(f"Warning: y-coordinates not symmetric for c={c}. sum(y) = {np.sum(y_obs)}")
    if not np.allclose(np.sum(z_obs), 0, atol=1e-10):
        print(f"Warning: z-coordinates not symmetric for c={c}. sum(z) = {np.sum(z_obs)}")

    if c == 1:
        print(f"For c = {c}:")
        print(np.max(x_obs), np.min(x_obs))
        print(np.max(y_obs), np.min(y_obs))
        print(np.max(z_obs), np.min(z_obs))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(x_obs, y_obs, z_obs, s = 40)
        ax.quiver(x_obs, y_obs, z_obs, Ix_obs, Iy_obs, Iz_obs, length=0.1, color='k')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_zlim(-1.45, 1.45)
        plt.tight_layout()

    for n_idx in range(len(n_obs_all)):
        n_obs = n_obs_all[n_idx]
        P, I, Q, U = compute_polarization(x_obs, y_obs, z_obs, Ix_obs, Iy_obs, Iz_obs, n_obs)
        P_HR_n[h_idx, n_idx] = P

plt.figure(figsize=(8,8))
for n_idx in range(len(n_obs_all)):   
    n_obs = n_obs_all[n_idx] 
    plt.plot(c_all, P_HR_n[:, n_idx], label=r'$n_{\rm obs}$'+ f' = ({n_obs[0]}, {n_obs[1]}, {n_obs[2]})', ls = n_obs_all_params[n_idx][1], c = n_obs_all_params[n_idx][2], linewidth=n_obs_all_params[n_idx][3]) # color by observer direction
plt.xlabel('c')
plt.ylabel('Polarization Fraction P')
plt.legend(fontsize=16)

#%% Healpix observers with radial fluxes but of different magnitudes
# NB: if you want to see less observers, DON'T cut here, or you change polarization computation
nside = 16
Npix = hp.nside2npix(nside)
observers_xyz = hp.pix2vec(nside, np.arange(Npix)) # shape: (3, 192)
x_obs, y_obs, z_obs = observers_xyz
numb = np.arange(len(x_obs))
phi_obs = np.arctan2(y_obs, x_obs)
mult_fact = np.array([1, 2, 10, 50])
Iy_obs = y_obs
Iz_obs = z_obs

theta_obs = np.arccos(z_obs)
P_theta = np.zeros((len(mult_fact), len(theta_obs)))
for i_idx, mult in enumerate(mult_fact):
    Ix_obs =  mult * x_obs  # flux increases with x
    for n_idx in range(len(theta_obs)):
        n_obs = [x_obs[n_idx], y_obs[n_idx], z_obs[n_idx]]
        P, I, Q, U = compute_polarization(x_obs, y_obs, z_obs, Ix_obs, Iy_obs, Iz_obs, n_obs)
        P_theta[i_idx][n_idx] = P

longitude_moll_h = phi_obs 
latitude_moll_h = np.pi/2 - theta_obs # z = -1 would give theta = pi, but mollweide has latitude = -90° at z=-1
if len(mult_fact) > 1:
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    for m_idx, mult in enumerate(mult_fact):
        ax.plot(latitude_moll_h*radians, P_theta[m_idx], label=r'I$_{\rm x}$ ' + f'= {mult} * x', c = plt.cm.rainbow(m_idx / len(mult_fact)))
else:
    from matplotlib import gridspec
    vmin_color = -np.pi
    vmax_color = np.pi
    cut = numb > -1 #np.logical_and(numb>87, numb<92) # so no cut
    fig = plt.figure(figsize=(24,10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1, .05], hspace=0.4, wspace = 0.2)
    ax1 = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = ax1.scatter(longitude_moll_h[cut], latitude_moll_h[cut], s = 100, c= longitude_moll_h[cut], cmap='rainbow', edgecolors='k', vmin = vmin_color, vmax = vmax_color)
    ax1.grid(True)
    ax1.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    ax1.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
    ax1.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    ax1.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])
    ax = fig.add_subplot(gs[0, 1]) 
    img = ax.scatter(latitude_moll_h[cut]*radians, P_theta[:,cut], s = 100, cmap = 'rainbow', c = longitude_moll_h[cut], vmin = vmin_color, vmax = vmax_color) #plt.cm.rainbow(m_idx / len(mult_fact)))
    cbar_ax = fig.add_subplot(gs[1, 0:2]) 
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label =r'longitude [rad]')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
ax.set_xlabel(r'latitude obs [rad]')
ax.set_ylabel('Polarization Fraction P')
ax.set_ylim(-0.02, 1)
ax.set_xlim(-1.8, 1.8)
if len(mult_fact) > 1:
    ax.legend(fontsize=24)
# fig.suptitle('Increasing I with x', fontsize=30)

# %%
nside = 16
Npix = hp.nside2npix(nside)
observers_xyz = hp.pix2vec(nside, np.arange(Npix)) # shape: (3, 192)
x_obs, y_obs, z_obs = observers_xyz
mult_fact = np.array([1, 2, 10, 50])
Ix_obs = x_obs
Iy_obs = y_obs

theta_obs = np.arccos(z_obs)
latitude_moll_h = np.pi/2 - theta_obs 
P_theta = np.zeros((len(mult_fact), len(theta_obs)))
for i_idx, mult in enumerate(mult_fact):
    Iz_obs = mult * z_obs     # flux increases with z
    for n_idx in range(len(theta_obs)):
        n_obs = [x_obs[n_idx], y_obs[n_idx], z_obs[n_idx]]
        P, I, Q, U = compute_polarization(x_obs, y_obs, z_obs, Ix_obs, Iy_obs, Iz_obs, n_obs)
        P_theta[i_idx][n_idx] = P

plt.figure(figsize=(8,8))
for m_idx, mult in enumerate(mult_fact):
    plt.plot(latitude_moll_h*radians, P_theta[m_idx], label=r'I$_{\rm z}$ ' + f'= {mult} * z', c = plt.cm.rainbow(m_idx / len(mult_fact)))
plt.xlabel(r'latitude obs [rad]')
plt.ylabel('Polarization Fraction P')
plt.legend(fontsize=16)
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-0.02, 1)
# plt.title('Increasing I with z')
# %%
