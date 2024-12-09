""" Plot the photosphere."""
import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src import orbits as orb
import matplotlib.colors as colors
import Utilities.prelude as prel
import healpy as hp
from Utilities.sections import make_slices
from Utilities.operators import make_tree
matplotlib.rcParams['figure.dpi'] = 150


abspath = '/Users/paolamartire/shocks'
G = 1
first_eq = 88 # observer eq. plane
final_eq = 104 #observer eq. plane
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
check = '' # '' or 'HiRes' or 'LowRes'
snap = '216'
compton = 'Compton'
extr = 'rich'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
path = f'{abspath}TDE/{folder}/{snap}'
saving_path = f'Figs/{folder}/{check}'
print(f'We are in: {path}, \nWe save in: {saving_path}')

Rt = Rstar * (Mbh/mstar)**(1/3)
# Rs = 2*G*Mbh / c**2
R0 = 0.6 * Rt
Rp =  Rt / beta
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T

#%% HEALPIX
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
# Convert to latitude and longitude
longitude = theta              
latitude = np.pi / 2 - phi 
# Plot in 2D using a Mollweide projection
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
img = ax.scatter(longitude, latitude, s=20, c=np.arange(192))
ax.scatter(longitude[first_eq:final_eq], latitude[first_eq:final_eq], s=10, c='r')
plt.colorbar(img, ax=ax, label='Observer Number')
ax.set_title("Observers on the Sphere (Mollweide Projection)")
ax.grid(True)
ax.set_xticks(np.radians(np.linspace(-180, 180, 9)))
ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
plt.tight_layout()
plt.show()
#%% Load data
zslice = np.load(f'/Users/paolamartire/shocks/data/{folder}/slices/z0slice_{snap}.npy')
x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, ie_den_mid, orb_en_den_mid, Rad_den_mid =\
        zslice[0], zslice[1], zslice[2], zslice[3], zslice[4], zslice[5], zslice[6], zslice[7], zslice[8]
    
# Load the data
# photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}{extr}_photo{snap}.txt')
# xph, yph, zph, volph, denph, Tempph = photo[0], photo[1], photo[2], photo[3], photo[4], photo[5]
# dim_cell_ph = (volph)**(1/3)
# mid = np.abs(zph) < dim_cell_ph
# xph_mid, yph_mid, zph_mid, denph_mid, Tempph_mid = make_slices([xph, yph, zph, denph, Tempph], mid)
# rph_mid = np.sqrt(xph_mid**2 + yph_mid**2 + zph_mid**2)
photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}{extr}_photo{snap}.txt')
xph, yph, zph = photo[0], photo[1], photo[2]
rph = np.sqrt(xph**2 + yph**2 + zph**2)
# Midplane
xph_mid, yph_mid, zph_mid = xph[first_eq:final_eq], yph[first_eq:final_eq], zph[first_eq:final_eq]

# data = make_tree(path, snap, energy = False)
# x, y, z, den = data.X, data.Y, data.Z, data.Den
# x_xz, z_xz, den_xz = x[np.abs(y<10)], z[np.abs(y<10)], den[np.abs(y<10)]
#%% Plot on the equatorial plane
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25,8))
img = ax1.scatter(x_mid, y_mid, c = den_mid, s = 1, cmap = 'winter', norm = colors.LogNorm(vmin = 1e-10, vmax = 1e-5))
cbar = plt.colorbar(img)
cbar.set_label(r'$\rho$ ', fontsize = 16)
ax1.scatter(xph_mid, yph_mid, c = 'r', s = 20)
ax1.scatter(0,0,c= 'k', marker = 'x', s=80)
ax1.set_xlim(-1200,200)
ax1.set_ylim(-800,200)
ax1.set_xlabel(r'X [$R_a$]', fontsize = 18)
ax1.set_ylabel(r'Y [$R_a$]', fontsize = 18)

ax2.scatter(np.arange(192),rph, c = 'orange')
ax2.set_ylabel(r'R [$R_a$]', fontsize = 18)
ax2.axhline(np.mean(rph), c = 'r', ls = '--')
ax2.text(0, np.mean(rph)+0.1, f'mean: {np.mean(rph):.2f}' + r'$R_\odot$', fontsize = 20)
ax2.set_xlabel('observer number', fontsize = 18)

ax3.scatter(np.arange(first_eq, final_eq),rph[first_eq:final_eq], c = 'r')
ax3.set_ylabel(r'R [$R_a$]', fontsize = 18)
ax3.set_xlabel('observer number', fontsize = 18)

# img = ax4.scatter(x_xz, z_xz, c = den_xz, s = 1, cmap = 'winter', norm = colors.LogNorm(vmin = 1e-10, vmax = 1e-5))
# cbar = plt.colorbar(img)
# cbar.set_label(r'$\rho$ ', fontsize = 16)
# ax4.scatter(xph_mid, yph_mid, c = 'r', s = 20)
# ax4.scatter(0,0,c= 'k', marker = 'x', s=80)
# ax4.set_xlim(-5,0.8)
# ax4.set_ylim(-2,2)
# ax4.set_xlabel(r'X [$R_a$]', fontsize = 18)
# ax4.set_ylabel(r'Z [$R_a$]', fontsize = 18)

# ax5.scatter(np.arange(192),rph, c = 'orange')
# ax5.set_ylabel(r'R [$R_a$]', fontsize = 18)
# ax5.axhline(np.mean(rph), c = 'r', ls = '--')
# ax5.text(0, np.mean(rph)+0.1, f'mean: {np.mean(rph):.2f}' + r'$R_\odot$', fontsize = 20)
# ax5.set_xlabel('observer number', fontsize = 18)


plt.suptitle(f'Snap {snap}', fontsize = 20)
plt.tight_layout()
plt.show()
# %%
