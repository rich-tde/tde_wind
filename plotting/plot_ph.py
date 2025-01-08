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
from scipy.stats import gmean
from Utilities.sections import make_slices
from Utilities.operators import make_tree, sort_list
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
snap = '199'
compton = 'Compton'
extr = 'rich'
if snap in ['164', '248']:
        also_xz = True
else:
      also_xz = False

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
path = f'{abspath}/TDE/{folder}/{snap}'
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
photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}{extr}_photo{snap}.txt')
xph, yph, zph, volph, denph, Tempph = photo[0], photo[1], photo[2], photo[3], photo[4], photo[5]
rph = np.sqrt(xph**2 + yph**2 + zph**2)
dim_cell_ph = (volph)**(1/3)
xph_mid, yph_mid, zph_mid, rph_mid, denph_mid, Tempph_mid = xph[first_eq:final_eq], yph[first_eq:final_eq], zph[first_eq:final_eq], rph[first_eq:final_eq], denph[first_eq:final_eq], Tempph[first_eq:final_eq]

# justph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/justph_photo{snap}.txt')
# xph_justph, yph_justph, zph_justph = justph[0], justph[1], justph[2]
# rph_justph = np.sqrt(xph_justph**2 + yph_justph**2 + zph_justph**2)
#%% Plot on the equatorial plane
if also_xz:
        data = make_tree(path, snap, energy = False)
        x, y, z, den, vol = data.X, data.Y, data.Z, data.Den, data.Vol
        cut = np.logical_and(den>1e-19, np.abs(y) < vol**(1/3))
        x_xz, z_xz, den_xz = x[cut], z[cut], den[cut]
        xz_cut = np.abs(yph) < dim_cell_ph
        xph_xz, zph_xz, rph_xz, denph_xz, Tempph_xz = make_slices([xph, zph, rph, denph, Tempph], xz_cut)
        fig, ((ax1, ax4), (ax2, ax3)) = plt.subplots(2,2, figsize = (20,15))
else:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25,8))

img = ax1.scatter(x_mid, y_mid, c = den_mid, s = 1, cmap = 'viridis', norm = colors.LogNorm(vmin = 1e-10, vmax = 1e-5))
cbar = plt.colorbar(img)
cbar.set_label(r'$\rho$ ', fontsize = 16)
ax1.scatter(xph_mid, yph_mid, c = 'r', s = 25, label = 'Midplane photo')
ax1.scatter(0,0,c= 'k', marker = 'x', s=80)
ax1.set_xlim(-1200,600)
ax1.set_ylim(-800,500)
ax1.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
ax1.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
ax1.legend(fontsize = 25)

ax2.scatter(np.arange(192), rph, c = denph, s = 25, cmap = 'viridis', norm = colors.LogNorm(vmin = 1e-14, vmax = 1e-11))
ax2.set_ylabel(r'R [$R_\odot$]', fontsize = 18)
ax2.axhline(np.mean(rph), c = 'r', ls = '--')
ax2.text(0, np.mean(rph)+0.1, f'mean: {np.mean(rph):.2f}' + r'$R_\odot$', fontsize = 20)
ax2.set_xlabel('observer number', fontsize = 18)
ax2.grid()

img = ax3.scatter(np.arange(len(rph_mid)), rph_mid, c = denph_mid, s = 25, cmap = 'viridis', norm = colors.LogNorm(vmin = 1e-14, vmax = 1e-11))
cbar = plt.colorbar(img)
cbar.set_label(r'$\rho$ ', fontsize = 20)
ax3.set_ylabel(r'R [$R_\odot$]', fontsize = 18)
# ax3.set_xlabel('observer number', fontsize = 18)
ax3.grid()

if also_xz:
        img = ax4.scatter(x_xz, z_xz, c = den_xz, s = 1, cmap = 'viridis', norm = colors.LogNorm(vmin = 1e-10, vmax = 1e-5))
        cbar = plt.colorbar(img)
        cbar.set_label(r'$\rho$ ', fontsize = 16)
        ax4.scatter(xph, zph, c = 'b', s = 25, label = 'Projected photo')
        ax4.scatter(0,0,c= 'k', marker = 'x', s=80)
        ax4.set_xlim(-50,50)
        ax4.set_ylim(-10,10)
        ax4.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
        ax4.set_ylabel(r'Z [$R_\odot$]', fontsize = 18)
        ax4.legend(fontsize = 25)

plt.suptitle(f'Snap {snap}', fontsize = 20)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/{folder}/photo_{snap}.png')
# %% NB DATA ARE NOT SORTED
ph_data = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{check}{extr}_phidx_fluxes.txt')
snaps, tfb, allindices_ph = ph_data[:, 0].astype(int), ph_data[:, 1], ph_data[:, 2:]
allindices_ph = sort_list(allindices_ph, snaps)
tfb = np.sort(tfb)
# eliminate the even rows (photosphere indices) of allindices_ph
fluxes = allindices_ph[1::2]
snaps = np.unique(np.sort(snaps))
tfb = np.unique(tfb)

mean_rph = np.zeros(len(tfb))
mean_rph_weig = np.zeros(len(tfb))
gmean_ph = np.zeros(len(tfb))

for i, snapi in enumerate(snaps):
        photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}{extr}_photo{snapi}.txt')
        xph_i, yph_i, zph_i = photo[0], photo[1], photo[2] 
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        mean_rph[i] = np.mean(rph_i)
        gmean_ph[i] = gmean(rph_i)
        mean_rph_weig[i] = np.sum(rph_i*fluxes[i])/np.sum(fluxes[i])
plt.figure()
plt.plot(tfb, mean_rph/apo, c = 'k')
plt.plot(tfb, mean_rph_weig/apo, c = 'r')
plt.plot(tfb, gmean_ph/apo, c = 'b')
plt.xlabel(r't [$t_{fb}$]')
plt.ylabel(r'$\langle R_{ph} [R_a] \rangle$')
plt.yscale('log')
plt.grid()
plt.savefig(f'{abspath}/Figs/{folder}/photo_mean.png')

#%%
with open(f'{abspath}/data/{folder}/photo_mean.txt', 'a') as f:
        f.write(f'# tfb, mean_rph,gmean,  weighted by flux\n')
        f.write(' '.join(map(str, tfb)) + '\n')
        f.write(' '.join(map(str, mean_rph)) + '\n')
        f.write(' '.join(map(str, gmean_ph)) + '\n')
        f.write(' '.join(map(str, mean_rph_weig)) + '\n')
        f.close()

