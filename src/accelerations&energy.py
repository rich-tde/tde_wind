""" Compute the accelerations due to centrifugal force and radial pressure to understand the Eddingotn envelope.
Compare kinetik and potential energy to understand the dynamics of the system."""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy.coordinates as coord
import Utilities.prelude as prel
from astropy import units as u
import src.orbits as orb
from Utilities.operators import make_tree, to_spherical_components
from Utilities.sections import make_slices
from Utilities.time_extractor import days_since_distruption
import healpy as hp

##
# PARAMETERS
## 

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = ''
Rs = 2 * prel.G * Mbh / prel.csol_cgs**2
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snap = 348

Rt = Rstar * (Mbh/mstar)**(1/3)
apo = orb.apocentre(Rstar, mstar, Mbh, beta) 
amin = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
solarR_to_au = 215
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
longitude_moll = theta              
latitude_moll = np.pi / 2 - phi 
indecesorbital = np.concatenate(np.where(latitude_moll==0))

#%% test coordinate transposition
# vec_test = np.array([1, 1, 0])
# R_test, lat_astro_test, long_test = coord.cartesian_to_spherical(vec_test[0], vec_test[1], vec_test[2]) # r, latitude, longitude 
# lat_test = lat_astro_test + (np.pi / 2) * u.rad # so is from 0 to pi
# V_r, V_theta, V_phi = to_spherical_components(vec_test[0], vec_test[1], vec_test[2], lat_test, long_test)
# print(f'Velocity in spherical coordinates: {V_r, V_theta, V_phi}')

##
# MAIN
## 
#%% Load the data
path = f'{abspath}/TDE/{folder}/{snap}' #f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
tfb = days_since_distruption(f'{abspath}/TDE/{folder}/{snap}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
DpDx = np.load(f'{path}/DpDx_{snap}.npy')
DpDy = np.load(f'{path}/DpDy_{snap}.npy')
DpDz = np.load(f'{path}/DpDz_{snap}.npy')
# dataph = np.loadtxt(f'{abspath}/data/{folder}/photo/_photo{snap}.txt')
# xph, yph, zph, volph= dataph[0], dataph[1], dataph[2], dataph[3]

data = make_tree(path, snap, energy = False)
X, Y, Z, Den, Mass, Vx, Vy, Vz, Vol, Temp = data.X, data.Y, data.Z, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Vol, data.Temp
den_cut = Den > 1e-19
X, Y, Z, Den, Mass, Vx, Vy, Vz, Vol, Temp, DpDx, DpDy, DpDz = make_slices([X, Y, Z, Den, Mass, Vx, Vy, Vz, Vol, Temp, DpDx, DpDy, DpDz], den_cut)

mid = np.abs(Z) < Vol**(1/3)
X, Y, Z, Den, Mass, Vx, Vy, Vz, Vol, Temp, DpDx, DpDy, DpDz = make_slices([X, Y, Z, Den, Mass, Vx, Vy, Vz, Vol, Temp, DpDx, DpDy, DpDz], mid)
R, lat_astro, long = coord.cartesian_to_spherical(X, Y, Z) # r, latitude, longitude 
lat = lat_astro + (np.pi / 2) * u.rad # so is from 0 to pi
dP_radial, _, _ = to_spherical_components(DpDx, DpDy, DpDz, lat, long)
V_r, V_theta, V_phi = to_spherical_components(Vx, Vy, Vz, lat, long)
V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
# orb_en = orb.orbital_energy(R, V, data.Mass, 1, 3e8 / (7e8/t), Mbh)

# compute accelerations
a_P = np.abs(dP_radial) / Den
a_centrifugal = V_phi**2 / R
ratio = a_centrifugal / a_P
a_grav = prel.G * Mbh / (R-Rs)**2
ratio_env = (a_P + a_centrifugal)/a_grav
alpha_plot = Den/np.max(Den)

#%%
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(22,5))
img = ax1.scatter(X/apo, Y/apo, c = np.abs(V_r), cmap = 'coolwarm', s = 7, norm=colors.LogNorm(vmin=1e-1, vmax=5e1))
cbar = plt.colorbar(img)
cbar.set_label(r'$|V_r|$', fontsize = 20)
# ax1.quiver(X[::1500]/apo, Y[::1500]/apo, Vx[::1500], Vy[::1500], color = 'k', scale = 100, scale_units = 'xy', angles = 'xy', width = 0.002)
ax1.set_ylabel(r'$Y [R_{\rm a}]$', fontsize = 20)

img = ax2.scatter(X/apo, Y/apo, c = np.abs(V_phi), cmap = 'coolwarm', s = 7, norm=colors.LogNorm(vmin=7e-2, vmax=5e1))
cbar = plt.colorbar(img)
cbar.set_label(r'$|V_\phi|$', fontsize = 20)
# # ax2.quiver(X[::1500]/apo, Y[::1500]/apo, Vx[::1500], Vy[::1500], color = 'k', scale = 120, scale_units = 'xy', angles = 'xy', width = 0.002)

img = ax3.scatter(X/apo, Y/apo, c = np.abs(V_theta), cmap = 'coolwarm', s = 7, norm=colors.LogNorm(vmin=1e-2, vmax=9))
cbar = plt.colorbar(img)
cbar.set_label(r'$|V_\theta|$', fontsize = 20)
# ax3.quiver(X[::1500]/apo, Y[::1500]/apo, Vx[::1500], Vy[::1500], color = 'k', scale = 120, scale_units = 'xy', angles = 'xy', width = 0.002)

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel(r'$X [R_{\rm a}]$', fontsize = 20)
    ax.scatter(0, 0, c='k', s = 40)
    ax.set_xlim(-.5, 2)
    ax.set_ylim(-.5,.5)#1, 1)
plt.suptitle(r'Z=0')
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/EddingtonEnvelope/vel_{snap}.png', bbox_inches = 'tight')

# if you slice for Z
# fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(22,5))
# img = ax1.scatter(Y/apo, Z/apo, c = V_r, cmap = 'coolwarm', s = 7, vmin = -2, vmax = 2)#norm=colors.LogNorm(vmin=1e-10, vmax=10))
# cbar = plt.colorbar(img)
# cbar.set_label(r'$V_r$', fontsize = 20)
# ax1.quiver(Y[::1500]/apo, Z[::1500]/apo, Vy[::1500], Vz[::1500], color = 'k', scale = 120, scale_units = 'xy', angles = 'xy', width = 0.002)
# ax1.set_ylabel(r'$Z [R_{\rm a}]$', fontsize = 20)

# img = ax2.scatter(Y/apo, Z/apo, c = V_phi, cmap = 'coolwarm', s = 7, vmin = -2, vmax = 2)#norm=colors.LogNorm(vmin=1e-10, vmax=1e-4))
# cbar = plt.colorbar(img)
# cbar.set_label(r'$V_\phi$', fontsize = 20)
# ax2.quiver(Y[::1500]/apo, Z[::1500]/apo, Vy[::1500], Vz[::1500], color = 'k', scale = 120, scale_units = 'xy', angles = 'xy', width = 0.002)

# img = ax3.scatter(Y/apo, Z/apo, c = V_theta, cmap = 'coolwarm', s = 7, vmin = -2, vmax = 2)#norm=colors.LogNorm(vmin=1e-10, vmax=1e-4))
# cbar = plt.colorbar(img)
# cbar.set_label(r'$V_\theta$', fontsize = 20)
# ax3.quiver(Y[::1500]/apo, Z[::1500]/apo, Vy[::1500], Vz[::1500], color = 'k', scale = 120, scale_units = 'xy', angles = 'xy', width = 0.002)

# for ax in [ax1, ax2, ax3]:
#     ax.set_xlabel(r'$Y [R_{\rm a}]$', fontsize = 20)
#     ax.scatter(0, 0, c='k', s = 40)
#     ax.set_xlim(-.5, .5)
#     ax.set_ylim(-.5,.5)#1, 1)
# plt.tight_layout()
# plt.suptitle(r'X=$R_p$')
# plt.savefig(f'{abspath}/Figs/EddingtonEnvelope/velyz_{snap}.png', bbox_inches = 'tight')
#%%
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(20, 10))
img = ax1.scatter(X/apo, Y/apo, c=Den, cmap = 'viridis', norm= colors.LogNorm(vmin = 1e-13, vmax = 1e-6), s = 7)
cbar = plt.colorbar(img)
cbar.set_label(r'$\rho$', fontsize = 20)
img = ax2.scatter(X/apo, Y/apo, c=V_r, cmap = 'coolwarm', vmin = -10, vmax = 10, s = 7)#, alpha = alpha_plot)
cbar = plt.colorbar(img)
cbar.set_label(r'$V_r$', fontsize = 20)
img = ax3.scatter(X/apo, Y/apo, c = V_phi, cmap = 'coolwarm', s = 7, vmin = -2, vmax = 2)#, alpha = alpha_plot)
cbar = plt.colorbar(img)
cbar.set_label(r'$V_\phi$', fontsize = 20)
img = ax4.scatter(X/apo, Y/apo, c=ratio, cmap = 'viridis', s = 7, norm=colors.LogNorm(vmin=1e-2, vmax=1e2))#, alpha = alpha_plot)
cbar = plt.colorbar(img)
cbar.set_label(r'$a_{\rm \omega}/a_{\rm P}$', fontsize = 20)

for ax in [ax1, ax2, ax3, ax4]:
    if ax in [ax1, ax3]:
        ax.set_ylabel(r'$Y [R_{\rm a}]$', fontsize = 20)
    if ax in [ax3, ax4]:
        ax.set_xlabel(r'$X [R_{\rm a}]$', fontsize = 20)
    ax.scatter(0, 0, c='k', s = 10)
    ax.set_xlim(-4, 1)
    ax.set_ylim(-1,1)#1, 1)
plt.suptitle(f't = {np.round(tfb,2)}' + r' $t_{\rm fb}$', fontsize = 20)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/EddingtonEnvelope/accelerations{m}_{snap}.png', bbox_inches = 'tight')









# %%
