""" Compute and test polarization with toy models and with simultion data"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
import wesanderson
from Utilities.basic_units import radians
from src.Wind.polarization import compute_polarization, ellipsoid_surface, ellipsoid_unit_normal
wes_palette = wesanderson.film_palette('Rushmore', 0)
cmap = colors.LinearSegmentedColormap.from_list('Rushmore0', wes_palette)

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
#%%
print("TEST one wave with incident/observer direction parallel to scattered (expect: P = 0).")
ph_obs = np.array([0, 0, 1]) # observer along z
I_obs = ph_obs # flux along x
n_obs = ph_obs
P, I, Q, U = compute_polarization(
    # ph_obs[0], ph_obs[1], ph_obs[2],
    I_obs[0], I_obs[1], I_obs[2],
    1.,
    n_obs)
print(f"n_obs: {n_obs}, I_obs: {I_obs}, P = {P}\n---------")
print("TEST one wave. incident/observer direction perpendicular to scattered (expect: P = 1).")
ph_obs = np.array([0, 0, 1]) # observer along z
I_obs = np.array([1, 0, 0]) # flux along x
n_obs = ph_obs
P, I, Q, U = compute_polarization(
    # ph_obs[0], ph_obs[1], ph_obs[2],
    I_obs[0], I_obs[1], I_obs[2],
    1.,
    n_obs)
print(f"n_obs: {n_obs}, I_obs: {I_obs}, Q = {Q}, U = {U}, P = {P}\n---------")
#%%
# NB Healpix doesn't necessarily give symmetric points, so we expect a small polarization signal (anyway ~0)
print("TEST of symmetry (all radial intensities) with healpix")
nside = 64
Npix = hp.nside2npix(nside)
observers_xyz = hp.pix2vec(nside, np.arange(Npix)) # shape: (3, 192)
x_obs, y_obs, z_obs = observers_xyz
Ir_obs = np.ones_like(x_obs) * 4
Ix_obs = Ir_obs * x_obs
Iy_obs = Ir_obs * y_obs
Iz_obs = Ir_obs * z_obs

if nside <= 8: # check points and fluxes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x_obs, y_obs, z_obs, s = 40)
    ax.quiver(x_obs, y_obs, z_obs, Ix_obs, Iy_obs, Iz_obs, length=0.1, color='k')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_zlim(-1.45, 1.45)
    plt.tight_layout()

plt.figure(figsize=(5,5))
n_obs = [1, 0, 0]  
P, I, Q, U = compute_polarization(
    Ix_obs, Iy_obs, Iz_obs,
    1.,
    n_obs, flux = True)
print(f"n_obs: {n_obs}, P = {P}")
plt.scatter(Q/I, U/I, label = f'n_obs: {n_obs}')
n_obs = [0, 1, 0]  
P, I, Q, U = compute_polarization(
    Ix_obs, Iy_obs, Iz_obs,
    1.,
    n_obs, flux = True)
print(f"n_obs: {n_obs}, P = {P}")
plt.scatter(Q/I, U/I, label = f'n_obs: {n_obs}')
n_obs = [0, 0, 1]  
P, I, Q, U = compute_polarization(
    Ix_obs, Iy_obs, Iz_obs,
    1.,
    n_obs, flux = True)
plt.scatter(Q/I, U/I, label = f'n_obs: {n_obs}')
plt.legend(fontsize=16)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('Q/I')
plt.ylabel('U/I')
print(f"n_obs: {n_obs}, P = {P}\n---------")
#%%
print("TEST disk")
x_obs, y_obs, z_obs = create_disk(radius=1.0, height=0.5, n_radial = 20, n_vertical=20)
Ix_obs = np.zeros_like(x_obs)
Iy_obs = np.zeros_like(y_obs)
Iz_obs = 2 *np.ones_like(z_obs) 
Iz_obs[z_obs<0] = -Iz_obs[z_obs<0] # points outward

if len(x_obs) < 100: # check points and fluxes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x_obs, y_obs, z_obs, s = 40)
    ax.quiver(x_obs, y_obs, z_obs, Ix_obs, Iy_obs, Iz_obs, length=0.1, color='k')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_zlim(-1.45, 1.45)
    plt.tight_layout()

n_obs_all_params = [[[1, 0, 0], 'solid', 'navy'],
            [[-1, 0, 0], 'dashed', 'dodgerblue'],
            [[0, 1, 0], 'solid', 'darkorange'],
            [[0, -1, 0], 'dashed', 'r'],
            [[0, 0, 1], 'solid', 'forestgreen'],
            [[0, 0, -1], 'dashed', 'yellowgreen'],
            [[1, 0, 1], 'dotted', 'k'],
            [[1, 0, -1], 'dotted', 'k'],
            [[np.sin(np.pi/3), 0, np.cos(np.pi/3)], 'dotted', 'k'],
            [[np.sin(np.pi/3), 0, -np.cos(np.pi/3)], 'dotted', 'k']]

n_obs_all = [params[0] for params in n_obs_all_params]
P_HR_n = np.zeros(len(n_obs_all))
# P_an = np.zeros(len(n_obs_all))
theta_obs = np.zeros(len(n_obs_all))
for n_idx in range(len(n_obs_all)):
    n_obs = n_obs_all[n_idx]
    theta_obs[n_idx] = np.arccos(n_obs[2]/np.linalg.norm(n_obs))
for n_idx in range(len(n_obs_all)):
        n_obs = n_obs_all[n_idx]
        P, I, Q, U = compute_polarization(Ix_obs, Iy_obs, Iz_obs, 1., n_obs)
        P_HR_n[n_idx] = P

angle_test = np.linspace(0, 1.1*np.pi, 100)
P_an = np.zeros_like(angle_test)
for idx, a in enumerate(angle_test):
    P_an[idx] = polarization_for_disk(a, angle = True)

plt.figure(figsize=(8,8))
plt.scatter(theta_obs*radians, P_HR_n, label='numerical', c = wes_palette[4], s = 50)
plt.plot(angle_test*radians, P_an, label='Analytic', ls = 'dashed', c = 'k')
plt.xlabel(r'$\theta$')
plt.ylabel('Polarization Fraction P')
plt.legend(fontsize=16)
plt.savefig(f'{abspath}/Figs/wind_paper/disk_test.pdf', bbox_inches='tight')
# plt.title('Disk test', fontsize=20)

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
    I_vec = ellipsoid_unit_normal(x_obs, y_obs, z_obs, a, b, c)
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
        P, I, Q, U = compute_polarization(Ix_obs, Iy_obs, Iz_obs, 1., n_obs)
        P_HR_n[h_idx, n_idx] = P

plt.figure(figsize=(8,8))
for n_idx in range(len(n_obs_all)):   
    n_obs = n_obs_all[n_idx] 
    plt.plot(c_all, P_HR_n[:, n_idx], label=r'$n_{\rm obs}$'+ f' = ({n_obs[0]}, {n_obs[1]}, {n_obs[2]})', ls = n_obs_all_params[n_idx][1], c = n_obs_all_params[n_idx][2]) # color by observer direction
plt.xlabel('c')
plt.ylabel('Polarization Fraction P')
plt.legend(fontsize=16)
plt.title('Ellipsoid test without healpix', fontsize=20)

#%%
print("TEST ellipsoid surface with modified healpix observers")
nside = 16
a = 1.0
b = a
c_all = np.concatenate([[1e-3], np.arange(0.1, 1.1, 0.1)])
n_obs_all_params = [[[1, 0, 0], 'solid', 'navy'],
            [[-1, 0, 0], 'dashed', 'dodgerblue'],
            [[0, 1, 0], 'solid', 'darkorange'],
            [[0, -1, 0], 'dashed', 'r'],
            [[0, 0, 1], 'solid', 'forestgreen'],
            [[0, 0, -1], 'dashed', 'yellowgreen'],
            [[1, 0, 1], 'dotted', 'k'],
            [[np.sin(np.pi/3), 0, np.cos(np.pi/3)], 'dotted', 'magenta']]

n_obs_all = [params[0] for params in n_obs_all_params]

P_HR_n = np.zeros((len(c_all), len(n_obs_all)))
for h_idx, c in enumerate(c_all):
    x_obs, y_obs, z_obs = ellipsoid_surface(nside, a, b, c, healpix=True)
    I_vec = ellipsoid_unit_normal(x_obs, y_obs, z_obs, a, b, c)
    Ix_obs, Iy_obs, Iz_obs = I_vec[:,0], I_vec[:,1], I_vec[:,2]
    if not np.allclose(np.sum(x_obs), 0, atol=1e-10):
        print(f"Warning: x-coordinates not symmetric for c={c}. sum(x) = {np.sum(x_obs)}")
    if not np.allclose(np.sum(y_obs), 0, atol=1e-10):
        print(f"Warning: y-coordinates not symmetric for c={c}. sum(y) = {np.sum(y_obs)}")
    if not np.allclose(np.sum(z_obs), 0, atol=1e-10):
        print(f"Warning: z-coordinates not symmetric for c={c}. sum(z) = {np.sum(z_obs)}")

    # if c == 1:
    #     print(f"For c = {c}:")
    #     print(np.max(x_obs), np.min(x_obs))
    #     print(np.max(y_obs), np.min(y_obs))
    #     print(np.max(z_obs), np.min(z_obs))
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111, projection = '3d')
    #     ax.scatter(x_obs, y_obs, z_obs, s = 40)
    #     ax.quiver(x_obs, y_obs, z_obs, Ix_obs, Iy_obs, Iz_obs, length=0.1, color='k')
    #     ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    #     ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_zlim(-1.45, 1.45)
    #     plt.tight_layout()

    for n_idx in range(len(n_obs_all)):
        n_obs = n_obs_all[n_idx]
        P, I, Q, U = compute_polarization(Ix_obs, Iy_obs, Iz_obs, 1.,n_obs)
        P_HR_n[h_idx, n_idx] = P

plt.figure(figsize=(8,8))
for n_idx in range(len(n_obs_all)):   
    n_obs = n_obs_all[n_idx] 
    plt.plot(c_all/a, P_HR_n[:, n_idx], label=r'$n_{\rm obs}$'+ f' = ({n_obs[0]:.1f}, {n_obs[1]:.1f}, {n_obs[2]:.1f})', ls = n_obs_all_params[n_idx][1], c = n_obs_all_params[n_idx][2]) # color by observer direction
plt.xlabel('c/a')
plt.ylabel('Polarization Fraction P')
plt.legend(fontsize=16)
plt.ylim(-0.1, 1.1)
plt.title('Ellipsoid test with healpix', fontsize=20)

#%% Healpix observers with magnitudes different according to the hemisphere
# NB: if you want to see less observers, DON'T cut here, or you change polarization computation
few_obs = False
change_y = True
div_y = 2
mult_fact = [4] #np.array([1, 4, 10, 100])
nside = 16
Npix = hp.nside2npix(nside)
observers_xyz = hp.pix2vec(nside, np.arange(Npix)) # shape: (3, 192)
x_obs, y_obs, z_obs = observers_xyz
I_vec = np.vstack((x_obs, y_obs, z_obs)).T
# I_vec *= 2 # so that the fluxes are not too small and you can see the color gradient in the plot
numb = np.arange(len(x_obs))
phi_obs = np.arctan2(y_obs, x_obs)
phi_obs += np.pi
theta_obs = np.arccos(z_obs)

#print(phi_obs[0], theta_obs[0], x_obs[0], y_obs[0], z_obs[0])
P_theta = np.zeros((len(mult_fact), len(theta_obs)))
for i_idx, mult in enumerate(mult_fact):
    I_vec[x_obs>0] *= mult
    if change_y:
        I_vec[y_obs<0] /= div_y
    # handle the x = points
    Ix_obs, Iy_obs, Iz_obs = I_vec[:,0], I_vec[:,1], I_vec[:,2]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x_obs, y_obs, z_obs, s = 40, c = np.linalg.norm(I_vec, axis=1), cmap=cmap, edgecolors='k', vmin = 0, vmax = np.max(np.linalg.norm(I_vec, axis=1))) #color by intensity
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_zlim(-1.45, 1.45)
    if change_y:
        plt.title(r'$I_{+x}$' + f'= {mult}I, ' + r'$I_{-y}$' + f' = I/{div_y}', fontsize=20)
    else:
        plt.title(r'$I_{+x}$' + f'= {mult}I', fontsize=20)
    plt.tight_layout()
    # plt.show()
    for n_idx in range(len(theta_obs)):
        n_obs = [x_obs[n_idx], y_obs[n_idx], z_obs[n_idx]]
        P, I, Q, U = compute_polarization(Ix_obs, Iy_obs, Iz_obs, 1., n_obs)
        P_theta[i_idx][n_idx] = P
        if np.abs(np.dot(n_obs, [0,0,1])-1) < 5e-3:
            print('Q:', Q, 'U:', U)
longitude_moll_h = phi_obs - np.pi
latitude_moll_h = np.pi/2 - theta_obs # z = -1 would give theta = pi, but mollweide has latitude = -90° at z=-1
if len(mult_fact) > 1:
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    for m_idx, mult in enumerate(mult_fact):
        ax.plot(latitude_moll_h*radians, P_theta[m_idx], label= r'$I_{+x}$' + f'= {mult}I, ' + r'$I_{-y}$' + f' = I/{div_y}' if change_y else r'$I_{+x}$' + f'= {mult}I', c = plt.cm.rainbow(m_idx / len(mult_fact)))
        ax.set_ylim(-0.02, 1)
else:
    from matplotlib import gridspec
    vmin_color = -np.pi
    vmax_color = np.pi
    if few_obs: 
        cut =  np.abs(x_obs) < 1e-1 #np.logical_or(np.logical_and(x_obs < 0, longitude_moll_h < -3*np.pi/4), np.logical_and(x_obs > 0, np.abs(y_obs) < 1e-1)) # so you see only the points along x axis, where the flux changes. You can change the cut to see other points, but you won't see the color gradient in the plot because you have too few points
        # print('long:', longitude_moll_h[cut], '\nmean lat: ', np.mean(latitude_moll_h[cut]))
    else:
        cut = numb > -1 # so no cut
    fig = plt.figure(figsize=(24,10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1, .05], hspace=0.4, wspace = 0.2)
    ax1 = fig.add_subplot(gs[0, 0], projection='mollweide')
    img = ax1.scatter(longitude_moll_h[cut], latitude_moll_h[cut], s = 100, c=  np.linalg.norm(I_vec[cut], axis=1), cmap = cmap, alpha = 0.8, vmin = 0, vmax = np.max(np.linalg.norm(I_vec, axis=1))) #color by intensity
    cbar_ax = fig.add_subplot(gs[1, 0]) 
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label =r'I')
    cbar.ax.tick_params(which='major',length = 5)
    cbar.ax.tick_params(which='minor',length = 3)
    ax1.grid(True)
    ax1.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    ax1.set_xticklabels(['180°', '270°', '0°','90°', '180°'])
    ax1.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    ax1.set_yticklabels(['180°', '135°', '90°', '45°','0°'])
    ax = fig.add_subplot(gs[0, 1]) 
    # img = ax.scatter(latitude_moll_h[cut]*radians, P_theta[0,cut], s = 100, cmap = 'rainbow', c = longitude_moll_h[cut], vmin = -np.pi, vmax = np.pi) #plt.cm.rainbow(m_idx / len(mult_fact)))
    # cbar_ax = fig.add_subplot(gs[1, 1]) 
    # cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label =r'longitude [rad]')
    # cbar.ax.tick_params(which='major',length = 5)
    # cbar.ax.tick_params(which='minor',length = 3)
    img = ax.plot(theta_obs[cut]*radians, P_theta[0,cut],c = wes_palette[1]) #plt.cm.rainbow(m_idx / len(mult_fact)))
    # if change_y:
    #     plt.title(r'$I_{+x}$' + f'= {mult}I, ' + r'$I_{-y}$' + f' = I/{div_y}', fontsize=20)
    # else:
    #     plt.title(r'$I_{+x}$' + f'= {mult}I', fontsize=20)
    # ax.set_ylim(-0.02, 0.5)
ax.set_xlabel(r'$\theta$ [rad]')
ax.set_ylabel('Net polarization P')
ax.set_xlim(0, 3.2)
if len(mult_fact) > 1:
    ax.legend(fontsize=24)
plt.savefig(f'{abspath}/Figs/wind_paper/aniso_sphere.pdf', bbox_inches='tight',transparent=True)

#%% Healpix observers with radial fluxes but of different magnitudes
# NB: if you want to see less observers, DON'T cut here, or you change polarization computation
# nside = 16
# Npix = hp.nside2npix(nside)
# observers_xyz = hp.pix2vec(nside, np.arange(Npix)) # shape: (3, 192)
# x_obs, y_obs, z_obs = observers_xyz
# numb = np.arange(len(x_obs))
# phi_obs = np.arctan2(y_obs, x_obs)
# phi_obs += np.pi
# theta_obs = np.arccos(z_obs)
# mult_fact = np.array([1, 2, 10, 50])
# Iy_obs = y_obs
# Iz_obs = z_obs

# # print(phi_obs[0], theta_obs[0], x_obs[0], y_obs[0], z_obs[0])
# P_theta = np.zeros((len(mult_fact), len(theta_obs)))
# for i_idx, mult in enumerate(mult_fact):
#     Ix_obs =  mult * x_obs  # flux increases with x
#     for n_idx in range(len(theta_obs)):
#         n_obs = [x_obs[n_idx], y_obs[n_idx], z_obs[n_idx]]
#         P, I, Q, U = compute_polarization(Ix_obs, Iy_obs, Iz_obs, n_obs)
#         P_theta[i_idx][n_idx] = P

# longitude_moll_h = phi_obs - np.pi
# latitude_moll_h = np.pi/2 - theta_obs # z = -1 would give theta = pi, but mollweide has latitude = -90° at z=-1

# if len(mult_fact) > 1:
#     fig, ax = plt.subplots(1, 1, figsize=(8,8))
#     for m_idx, mult in enumerate(mult_fact):
#         ax.plot(latitude_moll_h*radians, P_theta[m_idx], label=r'I$_{\rm x}$ ' + f'= {mult} * x', c = plt.cm.rainbow(m_idx / len(mult_fact)))
#         ax.set_ylim(-0.02, 1)
# else:
#     from matplotlib import gridspec
#     vmin_color = -np.pi
#     vmax_color = np.pi
#     if few_obs:
#         cut = np.logical_and(numb>87, numb<92) 
#     else:
#         cut = numb > -1 # so no cut
#     fig = plt.figure(figsize=(24,10))
#     gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1, .05], hspace=0.4, wspace = 0.2)
#     ax1 = fig.add_subplot(gs[0, 0], projection='mollweide')
#     img = ax1.scatter(longitude_moll_h[cut], latitude_moll_h[cut], s = 100, c= longitude_moll_h[cut], cmap='rainbow', edgecolors='k', vmin = vmin_color, vmax = vmax_color)
#     ax1.grid(True)
#     ax1.set_xticks(np.radians(np.arange(-180, 181, 90))) 
#     ax1.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
#     ax1.set_yticks(np.radians(np.arange(-90, 91, 45))) 
#     ax1.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])
#     ax = fig.add_subplot(gs[0, 1]) 
#     img = ax.scatter(latitude_moll_h[cut]*radians, P_theta[:,cut], s = 100, cmap = 'rainbow', c = longitude_moll_h[cut], vmin = vmin_color, vmax = vmax_color) #plt.cm.rainbow(m_idx / len(mult_fact)))
#     cbar_ax = fig.add_subplot(gs[1, 0:2]) 
#     cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label =r'longitude [rad]')
#     cbar.ax.tick_params(which='major',length = 5)
#     cbar.ax.tick_params(which='minor',length = 3)
#     ax.set_ylim(-0.02, 0.5)
# ax.set_xlabel(r'latitude obs [rad]')
# ax.set_ylabel('Polarization Fraction P')
# ax.set_xlim(-1.8, 1.8)
# if len(mult_fact) > 1:
#     ax.legend(fontsize=24)
# # fig.suptitle('Increasing I with x', fontsize=30)

# #%% same but along z
# nside = 16
# Npix = hp.nside2npix(nside)
# observers_xyz = hp.pix2vec(nside, np.arange(Npix)) # shape: (3, 192)
# x_obs, y_obs, z_obs = observers_xyz
# mult_fact = np.array([1, 2, 10, 50]) 
# Ix_obs = x_obs
# Iy_obs = y_obs

# theta_obs = np.arccos(z_obs)
# latitude_moll_h = np.pi/2 - theta_obs 
# P_theta = np.zeros((len(mult_fact), len(theta_obs)))
# for i_idx, mult in enumerate(mult_fact):
#     Iz_obs = mult * z_obs     # flux increases with z
#     for n_idx in range(len(theta_obs)):
#         n_obs = [x_obs[n_idx], y_obs[n_idx], z_obs[n_idx]]
#         P, I, Q, U = compute_polarization(Ix_obs, Iy_obs, Iz_obs, n_obs)
#         P_theta[i_idx][n_idx] = P

# plt.figure(figsize=(8,8))
# for m_idx, mult in enumerate(mult_fact):
#     plt.plot(latitude_moll_h*radians, P_theta[m_idx], label=r'I$_{\rm z}$ ' + f'= {mult} * z', c = plt.cm.rainbow(m_idx / len(mult_fact)))
# plt.xlabel(r'latitude obs [rad]')
# plt.ylabel('Polarization Fraction P')
# plt.legend(fontsize=16)
# plt.xlim(-1.8, 1.8)
# plt.ylim(-0.02, 1)
# # plt.title('Increasing I with z')

#%% Test with fluxes
print("TEST of symmetry (all radial equal intensities) with healpix and fluxes")
n_obs = [0, 1, 1]
n_obs_hat = n_obs / np.linalg.norm(n_obs)
nside = 16
Npix = hp.nside2npix(nside)
observers_xyz = hp.pix2vec(nside, np.arange(Npix)) # shape: (3, 192)
x_obs, y_obs, z_obs = observers_xyz
Ix_obs, Iy_obs, Iz_obs = 2*x_obs, 2*y_obs, 2*z_obs 
I_vec = np.vstack([Ix_obs, Iy_obs, Iz_obs]).T
I_obs = np.linalg.norm(I_vec, axis=1)
I_hat = I_vec / np.maximum(I_obs[:, None], 1e-12) # unit vector of the flux direction

dOmega = 4*np.pi / Npix  
cos_theta = np.sum(n_obs_hat * I_hat, axis=1) # cos of the angle between normal and observer direction
proj_area = dOmega * cos_theta # projected area toward observer
Fx_obs = Ix_obs * np.abs(proj_area) # you already take into account the sign od I in Ix_obs
Fy_obs = Iy_obs * np.abs(proj_area)
Fz_obs = Iz_obs * np.abs(proj_area)

P, _, _, _ = compute_polarization(Ix_obs, Iy_obs, Iz_obs, n_obs)
P_f, _, _, _ = compute_polarization(Fx_obs, Fy_obs, Fz_obs, n_obs, flux=True)
print('Polarization using flux/polarization using intensities:', P_f/P)
print('Polarization using flux:', P_f)

#%%
print("TEST ellipsoid surface (from modified healpix observers) and asymmetric intensities")
a = 1.0
b = a
nside = 4
c_all = np.concatenate([[1e-3], np.arange(0.1, 1.2, 0.1)])
n_obs_all_params = [[[1, 0, 0], 'solid', 'navy'],
            [[-1, 0, 0], 'dashed', 'dodgerblue'],
            [[0, 1, 0], 'solid', 'darkorange'],
            [[0, -1, 0], 'dashed', 'r'],
            [[0, 0, 1], 'solid', 'forestgreen'],
            [[0, 0, -1], 'dashed', 'yellowgreen'],
            [[1, 0, 1], 'dotted', 'k'],
            [[np.sin(np.pi/3), 0, np.cos(np.pi/3)], 'dotted', 'magenta']]

n_obs_all = [params[0] for params in n_obs_all_params]

P_HR_n = np.zeros((len(c_all), len(n_obs_all)))
for h_idx, c in enumerate(c_all):
    x_obs, y_obs, z_obs = ellipsoid_surface(nside, a, b, c, healpix=True)
    I_vec = ellipsoid_unit_normal(x_obs, y_obs, z_obs, a, b, c)
    I_vec[x_obs<-0.5] /= 5
    I_vec[np.abs(y_obs)<0.5] *= 10
    Ix_obs, Iy_obs, Iz_obs = I_vec[:,0], I_vec[:,1], I_vec[:,2]
    if not np.allclose(np.sum(x_obs), 0, atol=1e-10):
        print(f"Warning: x-coordinates not symmetric for c={c}. sum(x) = {np.sum(x_obs)}")
    if not np.allclose(np.sum(y_obs), 0, atol=1e-10):
        print(f"Warning: y-coordinates not symmetric for c={c}. sum(y) = {np.sum(y_obs)}")
    if not np.allclose(np.sum(z_obs), 0, atol=1e-10):
        print(f"Warning: z-coordinates not symmetric for c={c}. sum(z) = {np.sum(z_obs)}")

    if np.logical_and(nside <= 4, c == 1.0): # check points and fluxes for the most symmetric case
        print(f"For c = {c}:")
        I_obs = np.linalg.norm(I_vec, axis=1)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection = '3d')
        img = ax.scatter(x_obs, y_obs, z_obs, c = I_obs, cmap='rainbow', vmin = 0, vmax = 10)
        cbar = plt.colorbar(img, ax=ax, label='Intensity')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_zlim(-1.45, 1.45)
        plt.tight_layout()

    for n_idx in range(len(n_obs_all)):
        n_obs = n_obs_all[n_idx]
        P, I, Q, U = compute_polarization(Ix_obs, Iy_obs, Iz_obs, n_obs)
        P_HR_n[h_idx, n_idx] = P
#%%
plt.figure(figsize=(8,8))
for n_idx in range(len(n_obs_all)):   
    n_obs = n_obs_all[n_idx] 
    plt.plot(c_all/a, P_HR_n[:, n_idx], label=r'$n_{\rm obs}$'+ f' = ({n_obs[0]:.1f}, {n_obs[1]:.1f}, {n_obs[2]:.1f})', ls = n_obs_all_params[n_idx][1], c = n_obs_all_params[n_idx][2]) # color by observer direction
plt.xlabel('c/a')
plt.ylabel('Polarization Fraction P')
plt.legend(fontsize=16)
plt.ylim(-0.1, 1.1)
# %%
