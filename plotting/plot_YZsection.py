abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
matplotlib.rcParams['figure.dpi'] = 150
import Utilities.prelude as prel
from Utilities.operators import make_tree
import Utilities.sections as sec
import src.orbits as orb
import colorcet
from Utilities.time_extractor import days_since_distruption

#
## CONSTANTS
#

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
step = ''
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

check = 'Low' # 'Low' or 'HiRes'
check1 = 'HiRes'
snap = 199
xchosen = 0
energy = 'internal' # 'internal' or 'radiation'
save = True

path = f'/Users/paolamartire/shocks/TDE/{folder}{check}{step}/{snap}'
path1 = f'/Users/paolamartire/shocks/TDE/{folder}{check1}{step}/{snap}'
Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

#%% Load data and cut
data = make_tree(path, snap, energy = True)
tfb = days_since_distruption(f'{path1}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
mass, vol, ie_den, rad_den = data.Mass, data.Vol, data.IE, data.Rad
orb_en = orb.orbital_energy(Rsph, vel, mass, G, c, Mbh)
rad = rad_den * vol
ie = ie_den * vol
dim_cell = (3/(4*np.pi) * vol)**(1/3)
ie_onmass = (ie_den / data.Den) * prel.en_converter/prel.Msol_to_g
orb_en_onmass = (orb_en / mass) * prel.en_converter/prel.Msol_to_g
rad_den = rad_den * prel.en_den_converter
yz_cut = np.abs(data.X-xchosen) < dim_cell
Y_yz, Z_yz, dim_yz, Temp_yz, Rad_yz, Den_yz, mass_yz, orb_en_yz, ie_yz, rad_yz, IE_onmass_yz, rad_den_yz = \
    sec.make_slices([data.Y, data.Z, dim_cell, data.Temp, data.Rad, data.Den, data.Mass, orb_en, ie, rad, ie_onmass, rad_den], yz_cut)
#%% Load data1 and cut
data1 = make_tree(path1, snap, energy = True)
Rsph1 = np.sqrt(data1.X**2 + data1.Y**2 + data1.Z**2)
vel1 = np.sqrt(data1.VX**2 + data1.VY**2 + data1.VZ**2)
mass1, vol1, ie_den1, rad_den1 = data1.Mass, data1.Vol, data1.IE, data1.Rad
orb_en1 = orb.orbital_energy(Rsph1, vel1, mass1, G, c, Mbh)
rad1 = rad_den1 * vol1
ie1 = ie_den1 * vol1
rad_den1 = rad_den1 * prel.en_den_converter
dim_cell1 = (3/(4*np.pi) * vol1)**(1/3)
ie_onmass1 = (ie_den1 / data1.Den) * prel.en_converter/prel.Msol_to_g
orb_en_onmass1 = (orb_en1 / mass1) * prel.en_converter/prel.Msol_to_g
yz_cut1 = np.abs(data1.X-xchosen) < dim_cell1
Y1_yz, Z1_yz, dim1_yz, Temp1_yz, Rad1_yz, Den1_yz, mass1_yz, orb_en1_yz, ie1_yz, rad1_yz, IE1_onmass_yz, rad_den1_yz = \
    sec.make_slices([data1.Y, data1.Z, dim_cell1, data1.Temp, data1.Rad, data1.Den, data1.Mass, orb_en1, ie1, rad1, ie_onmass1, rad_den1], yz_cut1)

#%% YZ plane plot
if energy == 'internal':
    fig, ax = plt.subplots(5,2, figsize = (25,20))
    ax[0][0].set_title('Low Res', fontsize = 18)
    img = ax[0][0].scatter(Y_yz/apo, Z_yz/apo, c = Den_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-19, vmax = 1e-9))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Density [code units]', fontsize = 18)
    img = ax[1][0].scatter(Y_yz/apo, Z_yz/apo, c = mass_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-13, vmax = 1e-9))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Mass [code units]', fontsize = 18)
    img = ax[2][0].scatter(Y_yz/apo, Z_yz/apo, c = Temp_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e4, vmax = 1e7))
    cbar = plt.colorbar(img)
    cbar.set_label(r' Temperature [K]', fontsize = 18)
    img = ax[3][0].scatter(Y_yz/apo, Z_yz/apo, c = ie_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-14, vmax = 1e-11))#np.max(Den_tra)))
    cbar = plt.colorbar(img)
    cbar.set_label(r' Internal energy [code units]', fontsize = 18)
    img = ax[4][0].scatter(Y_yz/apo, Z_yz/apo, c = IE_onmass_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 8e12, vmax = 3e14))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Specific internal energy [erg/g]', fontsize = 18)
    # HiRes
    ax[0][1].set_title('High Res', fontsize = 18)
    img = ax[0][1].scatter(Y1_yz/apo, Z1_yz/apo, c = Den1_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-19, vmax = 2e-9))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Density [code units]', fontsize = 18)
    img = ax[1][1].scatter(Y1_yz/apo, Z1_yz/apo, c = mass1_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-13, vmax = 1e-9))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Mass [code units]', fontsize = 18)
    img = ax[2][1].scatter(Y1_yz/apo, Z1_yz/apo, c = Temp1_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e4, vmax = 1e7))
    cbar = plt.colorbar(img)
    cbar.set_label(r' Temperature [K]', fontsize = 18)
    img = ax[3][1].scatter(Y1_yz/apo, Z1_yz/apo, c = ie1_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-14, vmax = 1e-11))#np.max(Den_tra)))
    cbar = plt.colorbar(img)
    cbar.set_label(r' Internal energy [code units]', fontsize = 18)
    img = ax[4][1].scatter(Y1_yz/apo, Z1_yz/apo, c = IE1_onmass_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 8e12, vmax = 3e14))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Specific internal energy [erg/g]', fontsize = 18)
    for i in range(5):
        for j in range(2):
            ax[i][j].set_xlim(-0.4,0.4)#xlim_neg, xlim)
            ax[i][j].set_ylim(-1, 1)#ylim_neg, ylim)
            if i == 2:
                ax[i][j].set_xlabel(r'Y/$R_a$', fontsize = 20)
            if j == 0:
                ax[i][j].set_ylabel(r'Z/$R_a$', fontsize = 20)
    plt.suptitle(f'YZ plane at X={xchosen}, t = {np.round(tfb, 2)}' + r'$t_{fb}$', fontsize = 25)
    plt.tight_layout()
    if save:
        plt.savefig(f'{abspath}/Figs/{folder}/multiple/YZplanex{xchosen}_{snap}.png')

elif energy == 'radiation':
    fig, ax = plt.subplots(5,2, figsize = (25,20))
    ax[0][0].set_title('Low Res', fontsize = 18)
    img = ax[0][0].scatter(Y_yz/apo, Z_yz/apo, c = Den_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-19, vmax = 1e-9))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Density [code units]', fontsize = 18)
    img = ax[1][0].scatter(Y_yz/apo, Z_yz/apo, c = dim_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-1, vmax = 20))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Cell size [code units]', fontsize = 18)
    img = ax[2][0].scatter(Y_yz/apo, Z_yz/apo, c = Temp_yz,  cmap = 'viridis', s = 20, norm = colors.LogNorm(vmin = 1e4, vmax = 1e7))
    cbar = plt.colorbar(img)
    cbar.set_label(r' Temperature [K]', fontsize = 18)
    img = ax[3][0].scatter(Y_yz/apo, Z_yz/apo, c = rad_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-12, vmax = 1e-7))
    cbar = plt.colorbar(img)
    cbar.set_label(r' Radiation energy [code units]', fontsize = 18)
    img = ax[4][0].scatter(Y_yz/apo, Z_yz/apo, c = rad_den_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 5e3, vmax = 1e8))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 18)
    # HiRes
    ax[0][1].set_title('High Res', fontsize = 18)
    img = ax[0][1].scatter(Y1_yz/apo, Z1_yz/apo, c = Den1_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-19, vmax = 2e-9))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Density [code units]', fontsize = 18)
    img = ax[1][1].scatter(Y1_yz/apo, Z1_yz/apo, c = dim1_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-1, vmax = 10))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Cell size [code units]', fontsize = 18)
    img = ax[2][1].scatter(Y1_yz/apo, Z1_yz/apo, c = Temp1_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e4, vmax = 1e7))
    cbar = plt.colorbar(img)
    cbar.set_label(r' Temperature [K]', fontsize = 18)
    img = ax[3][1].scatter(Y1_yz/apo, Z1_yz/apo, c = rad1_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 1e-12, vmax = 1e-7))
    cbar = plt.colorbar(img)
    cbar.set_label(r' Radiation energy [code units]', fontsize = 18)
    img = ax[4][1].scatter(Y1_yz/apo, Z1_yz/apo, c = rad_den1_yz,  cmap = 'viridis', s = 10, norm = colors.LogNorm(vmin = 5e3, vmax = 1e8))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 18)
    for i in range(5):
        for j in range(2):
            ax[i][j].set_xlim(-0.4,0.4)#xlim_neg, xlim)
            ax[i][j].set_ylim(-1, 1)#ylim_neg, ylim)
            if i == 2:
                ax[i][j].set_xlabel(r'Y/$R_a$', fontsize = 20)
            if j == 0:
                ax[i][j].set_ylabel(r'Z/$R_a$', fontsize = 20)
    plt.suptitle(f'YZ plane at X={xchosen}, t = {np.round(tfb, 2)}' + r'$t_{fb}$', fontsize = 25)
    plt.tight_layout()
    if save:
        plt.savefig(f'{abspath}/Figs/{folder}/multiple/YZplanex{xchosen}Rad_{snap}.png')
    # %%

