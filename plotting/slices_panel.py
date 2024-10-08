abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from Utilities.basic_units import radians
from src import orbits as orb
from Utilities import sections as sec
import Utilities.prelude as prel
from Utilities.operators import make_tree, to_cylindric
from Utilities.time_extractor import days_since_distruption


##
# CONSTANTS
##

G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)
G = 1

##
# PARAMETERS
##
G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

#
## PARAMETERS STAR AND BH
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

check = 'Low' # 'Low' or 'HiRes'
step = ''

saving_path = f'{abspath}/Figs/{folder}/{check}'
Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

#%%
# DECISIONS
##
save = False
cutoff = '' # or '' or 'cutden
snap = '115'
fromfile = True

#%%
# DATA
##
if fromfile:
    # load the data
    datacut = np.load(f'{abspath}data/{folder}/{check}/slices/midplaneIEorb_{snap}.npy')
    x_cut_mid, y_cut_mid, ie_onmass_cut_mid, orb_en_onmass_cut_mid, den_cut_mid =\
        datacut[0], datacut[1], datacut[2], datacut[3], datacut[4]
    data = np.load(f'{abspath}data/{folder}/{check}/slices/midplaneRad_{snap}.npy')
    x_mid, y_mid, Rad_den_mid = data[0], data[1], data[2]
else:
    path = f'{abspath}TDE/{folder}{check}/{snap}'
    tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    data = make_tree(path, snap, energy = True)
    x_coord, y_coord, z_coord, mass, den, temp, rad_den, ie_den = \
        data.X, data.Y, data.Z, data.Mass, data.Den, data.Temp, data.Rad, data.IE
dim_cell = data.Vol**(1/3) 
ie_mass = ie_den/den # internal energy per unit mass
ie = ie_den*data.Vol
rad = rad_den*data.Vol
Rsph = np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
Vsph = np.sqrt(data.VX**2 + data.VY**2 + data.VZ**2)
orb_en = orb.orbital_energy(Rsph, Vsph, mass, G, c, Mbh)
orb_en_mass = orb_en/mass # orbital energy per unit mass
# converter
ie_mass *= prel.en_converter/prel.Msol_to_g
orb_en_mass *= prel.en_converter/prel.Msol_to_g
rad_den *= prel.en_den_converter

if cutoff == 'cutden':
    print('Cutting off low density elements')
    cut = data.Den > 1e-9 # throw fluff
else: 
    cut = data.Den > 1e-19 # throw vert low fluff

x_coord, y_coord, z_coord, Rsph, dim_cell, mass, den, temp, rad, ie, orb_en, rad_den, ie_mass, orb_en_mass = \
    sec.make_slices([x_coord, y_coord, z_coord, Rsph, dim_cell, mass, den, temp, rad, ie, orb_en, rad_den, ie_mass, orb_en_mass], cut)
THETA, RADIUS_cyl = to_cylindric(x_coord, y_coord)

#%%
""" Slice orbital plane """
midplane = np.abs(z_coord) < dim_cell
X_midplane, Y_midplane, Z_midplane, dim_midplane, Mass_midplane, Den_midplane, Temp_midplane, Rad_midplane, IE_midplane, orb_en_midplane, Rad_den_midplane, IEmass_midplane, orb_en_mass_midplane = \
    sec.make_slices([x_coord, y_coord, z_coord, dim_cell, mass, den, temp, rad, ie, orb_en, rad_den, ie_mass, orb_en_mass], midplane)
abs_orb_en_mass_midplane = np.abs(orb_en_mass_midplane)

#%% Plots
fig, ax = plt.subplots(2,3, figsize = (18,8))
img = ax[0][0].scatter(X_midplane/apo, Y_midplane/apo, c = Den_midplane, s = .1, cmap = 'viridis', norm=colors.LogNorm(vmin=1e-8, vmax=1e-5))
cbar = plt.colorbar(img)
cbar.set_label(r'Density $[M_\odot/R_\odot^3$]', fontsize = 16)
ax[0][0].set_ylabel(r'Y [$R_\odot$]', fontsize = 18)

img = ax[0][1].scatter(X_midplane/apo, Y_midplane/apo, c = Mass_midplane, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=1e-11, vmax=1e-7))
cbar = plt.colorbar(img)
cbar.set_label(r'Mass [M$_\odot$]', fontsize = 16)

img = ax[0][2].scatter(X_midplane/apo, Y_midplane/apo, c = dim_midplane, s = .1, cmap = 'viridis', norm=colors.LogNorm(vmin=4e-2, vmax=2))
cbar = plt.colorbar(img)
cbar.set_label(r'Cell size [$R_\odot$]', fontsize = 16)

img = ax[1][0].scatter(X_midplane/apo, Y_midplane/apo, c = np.abs(orb_en_midplane), s = .1, cmap = 'viridis', norm=colors.LogNorm(vmin = 1e-8, vmax = 1e-4))
cbar = plt.colorbar(img)
cbar.set_label(r'Absolute orbital energy', fontsize = 16)

img = ax[1][1].scatter(X_midplane/apo, Y_midplane/apo, c = IE_midplane, s = .1, cmap = 'viridis', norm=colors.LogNorm(vmin = 1e-11, vmax = 1e-8))
cbar = plt.colorbar(img)
cbar.set_label(r'Internal energy', fontsize = 16)

img = ax[1][2].scatter(X_midplane/apo, Y_midplane/apo, c = Rad_midplane, s = .1, cmap = 'viridis', norm=colors.LogNorm(vmin = 1e-14, vmax = 1e-9))
cbar = plt.colorbar(img)
cbar.set_label(r'Radiation energy', fontsize = 16)

for i in range(2):
    for j in range(3):
        ax[i][j].scatter(0,0,c= 'k', marker = 'x', s=80)
        ax[i][j].set_xlim(-1.3,30/apo)
        ax[i][j].set_ylim(-0.5,0.5)
        if i == 1:
            ax[i][j].set_xlabel(r'X [$R_\odot$]', fontsize = 18)
        if j == 0:
            ax[i][j].set_ylabel(r'Y [$R_\odot$]', fontsize = 18)

if cutoff == '':
    plt.suptitle(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', cut over 1e-19')
else:
    plt.suptitle(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', cut over 1e-9')
plt.tight_layout()
if save:
    plt.savefig(f'{saving_path}/slices/single/PanelSlice{snap}{cutoff}.png')
plt.show()


#%%
fig, ax = plt.subplots(1,1, figsize = (12,6))
img = ax.scatter(X_midplane/apo, Y_midplane/apo, c = abs_orb_en_mass_midplane, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=4e16, vmax=1.2e18))
cbar = plt.colorbar(img)
cbar.set_label(r'Specific absolute orbital energy', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
ax.set_xlim(-1.3,30/apo)
ax.set_ylim(-0.5,0.5)
ax.set_xlabel(r'X/$R_a$', fontsize = 18)
ax.set_ylabel(r'Y/$R_a$', fontsize = 18)
plt.tick_params(axis = 'both', which = 'both', direction='in', labelsize=20)
ax.text(-1.2, 0.4, r't/t$_{fb}$ = ' + str(np.round(tfb,2)), fontsize = 20)
if cutoff == '':
    plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', cut over 1e-19')
else:
    plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', cut over 1e-9')
plt.tight_layout()
if save:
    plt.savefig(f'{saving_path}/slices/single/midplaneEorb_{snap}{cutoff}.png')
plt.show()

#%%
fig, ax = plt.subplots(1,1, figsize = (12,6))
img = ax.scatter(X_midplane/apo, Y_midplane/apo, c = IEmass_midplane, s = 1, cmap = 'viridis',norm=colors.LogNorm(vmin=6e12, vmax=3.5e14))
cbar = plt.colorbar(img)
cbar.set_label(r'$\log_{10}$ specific IE', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(y_arrayline)):
    ax.plot(x_arrayline[i], y_arrayline[i], c = 'k', alpha=0.5)
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i]/apo, ycfr_grid[i]/apo, cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(-1.3,30/apo)
ax.set_ylim(-0.5,0.5)
ax.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
if cutoff == '':
    plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', cut over 1e-19')
else:
    plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', cut over 1e-9')
plt.tight_layout()
if save:
    plt.savefig(f'{saving_path}/slices/single/midplaneIE_{snap}{cutoff}.png')
plt.show()

fig, ax = plt.subplots(1,1, figsize = (12,6))
img = ax.scatter(X_midplane/apo, Y_midplane/apo, c = Rad_den_midplane, s = 1, cmap = 'viridis',norm=colors.LogNorm(vmin=2e3, vmax=9e8))
cbar = plt.colorbar(img)
cbar.set_label(r'$\log_{10}$ Rad energy density', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(y_arrayline)):
    ax.plot(x_arrayline[i], y_arrayline[i], c = 'k', alpha=0.5)
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i]/apo, ycfr_grid[i]/apo, cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(-1.3,30/apo)
ax.set_ylim(-0.5,0.5)
ax.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
if cutoff == '':
    plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', cut over 1e-19')
else:
    plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', cut over 1e-9')
plt.tight_layout()
if save:
    plt.savefig(f'{saving_path}/slices/single/midplaneRad_{snap}{cutoff}.png')
plt.show()

