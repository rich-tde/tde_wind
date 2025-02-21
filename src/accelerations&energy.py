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
from Utilities.operators import make_tree
from Utilities.sections import make_slices
from Utilities.time_extractor import days_since_distruption
import healpy as hp

##
# FUNCTIONS
##

def dPdR(DpDx, DpDy, DpDz, lat, long):
    dP_radial = DpDx * np.sin(lat) * np.cos(long) + DpDy * np.sin(lat) * np.sin(long) + DpDz * np.cos(lat)
    return dP_radial

def v_phi(Vx, Vy, long):
    v_phi = - Vx * np.sin(long) + Vy * np.cos(long)
    return v_phi

def v_theta(Vx, Vy, Vz, lat, long):
    v_theta = Vx * np.cos(lat) * np.cos(long) + Vy * np.cos(lat) * np.sin(long) - Vz * np.sin(lat)
    return v_theta

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


##
# MAIN
## 
#%% Load the data
path = f'{abspath}/TDE/{folder}/{snap}' #f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
tfb = days_since_distruption(f'{abspath}/TDE/{folder}/{snap}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
DpDx = np.load(f'{path}/DpDx_{snap}.npy')
DpDy = np.load(f'{path}/DpDy_{snap}.npy')
DpDz = np.load(f'{path}/DpDz_{snap}.npy')
dataph = np.loadtxt(f'{abspath}/data/{folder}/photo/_photo{snap}.txt')
xph, yph, zph, volph, Vxph, Vyph, Vzph = dataph[0], dataph[1], dataph[2], dataph[3], dataph[-3], dataph[-2], dataph[-1]

data = make_tree(path, snap, energy = False)
X, Y, Z, Den, Mass, Vx, Vy, Vz, Vol, Temp = data.X, data.Y, data.Z, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Vol, data.Temp
den_cut = Den > 1e-19
X, Y, Z, Den, Mass, Vx, Vy, Vz, Vol, Temp, DpDx, DpDy, DpDz = make_slices([X, Y, Z, Den, Mass, Vx, Vy, Vz, Vol, Temp, DpDx, DpDy, DpDz], den_cut)

mid = np.abs(Z) < Vol**(1/3)
X, Y, Z, Den, Mass, Vx, Vy, Vz, Vol, Temp, DpDx, DpDy, DpDz = make_slices([X, Y, Z, Den, Mass, Vx, Vy, Vz, Vol, Temp, DpDx, DpDy, DpDz], mid)
V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
R, lat_astro, long = coord.cartesian_to_spherical(X, Y, Z) # r, latitude, longitude 
lat = lat_astro + (np.pi / 2) * u.rad # so is from 0 to pi
# orb_en = orb.orbital_energy(R, V, data.Mass, 1, 3e8 / (7e8/t), Mbh)
dP_radial = dPdR(DpDx, DpDy, DpDz, lat, long)

#%% compute energies
KE = 0.5 * Mass * V**2
PE = - prel.G * Mbh * Mass / R
raio_E = KE / np.abs(PE)
# compute accelerations
a_P = np.abs(dP_radial) / Den
V_phi = v_phi(Vx, Vy, long)
a_centrifugal = V_phi**2 / R
# V_theta = v_theta(Vx, Vy, Vz, lat, long)
ratio = a_centrifugal / a_P
a_grav = prel.G * Mbh / (R-Rs)**2
ratio_env = (a_P + a_centrifugal)/a_grav

#%% Plot energies
plt.figure(figsize=(15, 7))
img = plt.scatter(X/apo, Y/apo, c=raio_E, cmap = 'coolwarm', s = 15, norm = colors.LogNorm(vmin = 1e-1, vmax = 10))
cbar = plt.colorbar(img)
cbar.set_label(r'$E_{\rm kin}/|E_{\rm pot}|$', fontsize = 20)
plt.scatter(xph[indecesorbital]/apo, yph[indecesorbital]/apo, facecolors='none', edgecolors = 'k', s = 50)
# arrow vx, vy on xph, yph
for i in range(len(xph[indecesorbital])):
    plt.arrow(xph[indecesorbital][i]/apo, yph[indecesorbital][i]/apo, Vxph[indecesorbital][i]/50, Vyph[indecesorbital][i]/50, color = 'k', head_width = 0.1, head_length = 0.1)
plt.scatter(0, 0, c='k', s = 10)
plt.xlim(-6, 2)
plt.ylim(-3,2)#1, 1)
plt.xlabel(r'$X [R_{\rm a}]$', fontsize = 20)
plt.ylabel(r'$Y [R_{\rm a}]$', fontsize = 20)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/EddingtonEnvelope/ratioE{m}_{snap}.png', bbox_inches = 'tight')

#%%
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(20, 7))
img = ax1.scatter(X/apo, Y/apo, c=Vx, cmap = 'coolwarm', vmin = -1, vmax = 1, s = 7)
cbar = plt.colorbar(img)
cbar.set_label(r'$V_x$', fontsize = 20)
img = ax2.scatter(X/apo, Y/apo, c=Vy, cmap = 'coolwarm', vmin = -1, vmax = 1, s = 7)
cbar = plt.colorbar(img)
cbar.set_label(r'$V_y$', fontsize = 20)
img = ax3.scatter(X/apo, Y/apo, c = ratio_env, cmap = 'viridis', s = 7, vmin = 1e-2, vmax = 2)#norm=colors.LogNorm(vmin=1e-10, vmax=1e-4))
cbar = plt.colorbar(img)
cbar.set_label(r'$(a_{\rm \omega}+a_{\rm P})/a_{\rm g}$', fontsize = 20)
img = ax4.scatter(X/apo, Y/apo, c=ratio, cmap = 'viridis', s = 7, norm=colors.LogNorm(vmin=1e-2, vmax=1e2))

cbar = plt.colorbar(img)
cbar.set_label(r'$a_{\rm \omega}/a_{\rm P}$', fontsize = 20)
for ax in [ax1, ax2, ax3, ax4]:
    if ax in [ax1, ax3]:
        ax.set_ylabel(r'$Y [R_{\rm a}]$', fontsize = 20)
    if ax in [ax3, ax4]:
        ax.set_xlabel(r'$X [R_{\rm a}]$', fontsize = 20)
    ax.scatter(0, 0, c='k', s = 10)
    ax.set_xlim(-6, 2)
    ax.set_ylim(-3,2)#1, 1)
plt.suptitle(f't = {np.round(tfb,2)}' + r' $t_{\rm fb}$', fontsize = 20)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/EddingtonEnvelope/accelerations{m}_{snap}big.png', bbox_inches = 'tight')






