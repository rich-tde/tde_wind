""" Plot density proj / Tslice / Rad energy slice at 3 different times."""
#%%
abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import colorcet
import src.orbits as orb
import Utilities.prelude as prel
from Utilities.sections import make_slices
from Utilities.operators import from_cylindric
from Utilities.time_extractor import days_since_distruption

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

save = True
snap = 267

Rt = Rstar * (Mbh/mstar)**(1/3)
Rp = beta * Rt
R0 = 0.6 * Rt
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
e_mb = orb.eccentricity(Rstar, mstar, Mbh, beta)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
tfb = days_since_distruption(f'{abspath}TDE/{folder}/{snap}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
solarR_to_au = 215

# make cfr
radii_grid = [Rt/apo, 0.1, a_mb/apo, 1] #*apo 
styles = ['dashed', 'solid', 'solid', 'solid', 'solid']
xcfr_grid, ycfr_grid, cfr_grid = [], [], []
for i, radius_grid in enumerate(radii_grid):
    xcr, ycr, cr = orb.make_cfr(radius_grid)
    xcfr_grid.append(xcr)
    ycfr_grid.append(ycr)
    cfr_grid.append(cr)

theta_arr = np.arange(0, 2*np.pi, 0.01)
r_arr_ell = orb.keplerian_orbit(theta_arr, a_mb, Rp, ecc=e_mb)
x_arr_ell, y_arr_ell = from_cylindric(theta_arr, r_arr_ell)
r_arr_par = orb.keplerian_orbit(theta_arr, a_mb, Rp, ecc=1)
x_arr_par, y_arr_par = from_cylindric(theta_arr, r_arr_par)

#%%
x_radii = np.loadtxt(f'{abspath}data/{folder}/projection/bigxarray{snap}.txt')
y_radii = np.loadtxt(f'{abspath}data/{folder}/projection/bigyarray{snap}.txt')
flat_den = np.loadtxt(f'{abspath}data/{folder}/projection/bigdenproj{snap}.txt')
flat_den_cgs = flat_den * prel.den_converter * prel.Rsol_cgs # [g/cm2]
dataph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/rich_photo{snap}.txt')
xph, yph, zph, volph, denph, Tempph = dataph[0], dataph[1], dataph[2], dataph[3], dataph[4], dataph[5]
massph = denph * volph
midph= np.abs(zph) < volph**(1/3)
xph_mid, yph_mid, zph_mid, denph_mid, Tempph_mid, massph_mid = make_slices([xph, yph, zph, denph, Tempph, massph], midph)


plt.figure(figsize=(20, 8))
img = plt.pcolormesh(x_radii/solarR_to_au, y_radii/solarR_to_au, flat_den_cgs.T, cmap = 'plasma',
                    norm = colors.LogNorm(vmin = 5e-2, vmax = 1e2))
plt.scatter(xph_mid/solarR_to_au, yph_mid/solarR_to_au, c = 'k', s = 30, marker = 's', 
            norm = colors.LogNorm(vmin = 3e3, vmax = 5e5), label = 'Photosphere orbital plane')
plt.ylim(-6,4.8)
plt.text(-10, 4, f't = {np.round(tfb,2)}' + r' $t_{\rm fb}$', color = 'white', fontsize = 25)
plt.tick_params(axis='x', which='major', width = .7, length = 7, color = 'white')
plt.tick_params(axis='y', which='major', width = .7, length = 7, color = 'white')
# Create a colorbar that spans the first two subplots
cb = plt.colorbar(img)
cb.ax.tick_params(which='major',length = 5)
cb.ax.tick_params(which='minor',length = 3)
cb.set_label(r'Column density [g/cm$^2$]', fontsize = 20)
plt.xlabel(r'$X [AU]$', fontsize = 20)
plt.ylabel(r'$Y [AU]$', fontsize = 20)
    
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/zoomoutproj{snap}{check}.png')

#%% Compare with high res
x_radiiH = np.loadtxt(f'{abspath}data/{folder}HiRes/projection/bigxarray{snap}.txt')
y_radiiH = np.loadtxt(f'{abspath}data/{folder}HiRes/projection/bigyarray{snap}.txt')
flat_denH = np.loadtxt(f'{abspath}data/{folder}HiRes/projection/bigdenproj{snap}.txt')
flat_den_cgsH = flat_denH * prel.den_converter * prel.Rsol_cgs # [g/cm2]
rel_diff = 2*np.abs(flat_den_cgs - flat_den_cgsH/(flat_den_cgs + flat_den_cgsH))

plt.figure(figsize=(20, 8))
img = plt.pcolormesh(x_radii/solarR_to_au, y_radii/solarR_to_au, rel_diff.T, cmap = 'inferno',
                    norm = colors.LogNorm(vmin = 1e-3, vmax = 1))
plt.text(-10, 4, f't = {np.round(tfb,2)}' + r' $t_{\rm fb}$', color = 'white', fontsize = 25)
plt.tick_params(axis='x', which='major', width = .7, length = 7, color = 'white')
plt.tick_params(axis='y', which='major', width = .7, length = 7, color = 'white')
# Create a colorbar that spans the first two subplots
cb = plt.colorbar(img)
cb.ax.tick_params(which='major',length = 5)
cb.ax.tick_params(which='minor',length = 3)
cb.set_label(r'$\Delta_\Sigma$', fontsize = 20)
plt.xlabel(r'$X [AU]$', fontsize = 20)
plt.ylabel(r'$Y [AU]$', fontsize = 20)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/diffproj{snap}{check}.png')

#%% Orbital plane
zslice = np.load(f'/Users/paolamartire/shocks/data/{folder}/slices/z0slice_{snap}.npy')
x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, ie_den_mid, orb_en_den_mid, Rad_den_mid =\
    zslice[0], zslice[1], zslice[2], zslice[3], zslice[4], zslice[5], zslice[6], zslice[7], zslice[8]
mass_mid = den_mid * (dim_mid)**3
total_mass_mid = np.sum(mass_mid)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))
img = ax1.scatter(x_mid/solarR_to_au, y_mid/solarR_to_au, c = temp_mid, cmap = 'cet_fire', s = 2,
                    norm = colors.LogNorm(vmin = 3e3, vmax = 5e5))
cb = plt.colorbar(img)
cb.set_label(r'Temperature [K]', fontsize = 20)
ax1.scatter(xph_mid/solarR_to_au, yph_mid/solarR_to_au, c = Tempph_mid, cmap = 'cet_fire', s = 30, marker = 's', 
            norm = colors.LogNorm(vmin = 3e3, vmax = 5e5), label = 'Photosphere orbital plane')
ax1.legend(fontsize = 20)
img = ax2.scatter(x_mid/solarR_to_au, y_mid/solarR_to_au, c = mass_mid/total_mass_mid, cmap = 'viridis', s = 2,
                    norm = colors.LogNorm(vmin = 1e-9, vmax = 1e-2))
cb = plt.colorbar(img)
cb.set_label(r'Cell mass/M$_{\rm tot mid}$', fontsize = 20)
ax2.scatter(xph_mid/solarR_to_au, yph_mid/solarR_to_au, c = massph_mid/total_mass_mid, cmap = 'viridis', s = 30, marker = 's', 
            norm = colors.LogNorm(vmin = 1e-9, vmax = 1e-2))
plt.suptitle(f'Orbital plane, t = {np.round(tfb,2)}' + r' $t_{\rm fb}$', fontsize = 25)

for ax in [ax1, ax2]:
    ax.set_xlabel(r'$X [AU]$', fontsize = 20)
    ax.set_xlim(-11.6, 3.6)
    ax.set_ylim(-4.1, 4.1)
ax1.set_ylabel(r'$Y [AU]$', fontsize = 20)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/slicephoto{snap}.png')

# %%
