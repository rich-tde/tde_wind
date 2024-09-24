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
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
check = 'Low' # '' or 'HiRes' or 'Res20'
snap = '164'
compton = 'Compton'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'{abspath}TDE/{folder}{check}/{snap}'
saving_path = f'Figs/{folder}/{check}'
print(f'We are in: {path}, \nWe save in: {saving_path}')

Rt = Rstar * (Mbh/mstar)**(1/3)
Rs = 2*G*Mbh / c**2
R0 = 0.6 * Rt
Rp =  Rt / beta
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

# cfr tidal disruption and at smoothing lenght
xcfr, ycfr, cfr = orb.make_cfr(Rt)
xcfr0, ycfr0, cfr0 = orb.make_cfr(R0)
# cfr for grid
radii_grid = [Rt, 0.1*apo, 0.3*apo, 0.5*apo, apo]#, '1.5*Rt'] 
styles = ['dashed','solid', 'solid', 'solid', 'solid']#'dotted']
xcfr_grid, ycfr_grid, cfr_grid = [], [], []
for i,radius_grid in enumerate(radii_grid):
    xcr, ycr, cr = orb.make_cfr(radius_grid)
    xcfr_grid.append(xcr)
    ycfr_grid.append(ycr)
    cfr_grid.append(cr)

#%%
# DECISIONS
##
save = True
difference = True
cutoff = 'cutden' # or '' or 'bound' or 'cutdenbound'

#%%
# DATA
##

tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
data = make_tree(path, snap, energy = True)
x_coord, y_coord, z_coord, mass, den, temp, rad_den, ie_den = \
    data.X, data.Y, data.Z, data.Mass, data.Den, data.Temp, data.Rad, data.IE
ie_mass = ie_den/den # internal energy per unit mass
dim_cell = data.Vol**(1/3) 
Rsph = np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
Vsph = np.sqrt(data.VX**2 + data.VY**2 + data.VZ**2)
orb_en = orb.orbital_energy(Rsph, Vsph, mass, G, c, Mbh)
orb_en_mass = orb_en/mass # orbital energy per unit mass
# converter
ie_mass *= prel.en_converter/prel.Msol_to_g
orb_en_mass *= prel.en_converter/prel.Msol_to_g
rad_den *= prel.en_den_converter

##
# MAIN
##
#%%
if cutoff != '':
    if cutoff == 'bound' or cutoff == 'cutdenbound':
        print('Cutting off unbound elements')
        cutbound = orb_en < 0 
        if cutoff == 'cutdenbound':
            print('Cutting off low density elements')
            cutden = data.Den > 1e-9 # throw fluff
            cut = cutden & cutbound 
    if cutoff == 'cutden':
        print('Cutting off low density elements')
        cut = data.Den > 1e-9 # throw fluff

    x_coord, y_coord, z_coord, Rsph, dim_cell, mass, den, temp, rad_den, ie_mass, orb_en_mass = \
        sec.make_slices([x_coord, y_coord, z_coord, Rsph, dim_cell, mass, den, temp, rad_den, ie_mass, orb_en_mass], cut)

THETA, RADIUS_cyl = to_cylindric(x_coord, y_coord)
#%%
#%%
""" Slice orbital plane """
midplane = np.abs(z_coord) < dim_cell
X_midplane, Y_midplane, Z_midplane, dim_midplane, Mass_midplane, Den_midplane, Temp_midplane, Rad_den_midplane, IEmass_midplane, orb_en_mass_midplane = \
    sec.make_slices([x_coord, y_coord, z_coord, dim_cell, mass, den, temp, rad_den, ie_mass, orb_en_mass], midplane)
abs_orb_en_mass_midplane = np.abs(orb_en_mass_midplane)


#%%
fig, ax = plt.subplots(1,1, figsize = (12,4))
img = ax.scatter(X_midplane/apo, Y_midplane/apo, c = abs_orb_en_mass_midplane, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=2e16, vmax=9e17))
cbar = plt.colorbar(img)
cbar.set_label(r'Specific absolute orbital energy', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i]/apo, ycfr_grid[i]/apo, cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(-1.3,R0/apo)
ax.set_ylim(-0.8,0.5)
ax.set_xlabel(r'X/$R_a$', fontsize = 18)
ax.set_ylabel(r'Y/$R_a$', fontsize = 18)
plt.tick_params(axis = 'both', which = 'both', direction='in', labelsize=20)
ax.text(-1.2, 0.4, r't/t$_{fb}$ = ' + str(np.round(tfb,2)), fontsize = 20)
plt.tight_layout()
if save:
    plt.savefig(f'{saving_path}/slices/midplaneEorb_{snap}{cutoff}.png')
plt.show()
#%%
fig, ax = plt.subplots(1,1, figsize = (6,6))
img = ax.scatter(X_midplane, Y_midplane, c = Den_midplane, s = .1, cmap = 'viridis', norm=colors.LogNorm(vmin=1e-9, vmax=np.max(Den_midplane[np.logical_and(X_midplane>-40, np.abs(Y_midplane)<40)])))
cbar = plt.colorbar(img)
cbar.set_label(r'Density $[M_\odot/R_\odot^3$]', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i], ycfr_grid[i], cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
    # ax.text(-radii_grid[i]-0.1, 0.1, f'R = {radii_grid[i]}', fontsize = 12)
# ax.plot(x_stream, y_stream, c = 'k')
ax.hlines(y=0, xmin = 0, xmax = 30, color = 'k', linestyle = 'dashed', alpha = 0.8)
ax.set_xlim(-40,25)#340,25)
ax.set_ylim(-40,40)#70)
ax.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{saving_path}/slices/midplaneDen_{snap}{cutoff}.png')
plt.show()

#%%
fig, ax = plt.subplots(1,1, figsize = (6,6))
img = ax.scatter(X_midplane, Y_midplane, c = Mass_midplane, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=1e-11, vmax=1e-7))
cbar = plt.colorbar(img)
cbar.set_label(r'Mass [M$_\odot$]', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i], ycfr_grid[i], cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.hlines(y=0, xmin = 0, xmax = 30, color = 'k', linestyle = 'dashed', alpha = 0.8)
ax.set_xlim(-40,25)#340,25)
ax.set_ylim(-40,40)#70)
ax.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{saving_path}/slices/midplaneMass_{snap}{cutoff}.png')
plt.show()

#%%
fig, ax = plt.subplots(1,1, figsize = (6,6))
img = ax.scatter(X_midplane, Y_midplane, c = dim_midplane, s = .1, cmap = 'viridis', norm=colors.LogNorm(vmin=4e-2, vmax=9e-1))
cbar = plt.colorbar(img)
cbar.set_label(r'size$_{cell} [R_\odot$]', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i], ycfr_grid[i], cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.hlines(y=0, xmin = 0, xmax = 30, color = 'k', linestyle = 'dashed', alpha = 0.8)
ax.set_xlim(-40,25)#340,25)
ax.set_ylim(-40,40)#70)
ax.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{saving_path}/slices/midplaneSize_{snap}{cutoff}.png')
plt.show()

#%%
fig, ax = plt.subplots(1,1, figsize = (12,4))
img = ax.scatter(X_midplane, Y_midplane, c = Temp_midplane, s = 1, cmap = 'viridis',norm=colors.LogNorm(vmin=1e4, vmax=6e5))
cbar = plt.colorbar(img)
cbar.set_label(r'$\log_{10}$ Temp', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i], ycfr_grid[i], cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(-400,100)
ax.set_ylim(-200,200)
ax.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{saving_path}/slices/midplaneTemp_{snap}{cutoff}.png')
plt.show()

#%%
fig, ax = plt.subplots(1,1, figsize = (12,4))
img = ax.scatter(X_midplane, Y_midplane, c = IEmass_midplane, s = 1, cmap = 'viridis',norm=colors.LogNorm(vmin=2e-16, vmax=9e17))
cbar = plt.colorbar(img)
cbar.set_label(r'$\log_{10}$ specific IE', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i], ycfr_grid[i], cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(-400,100)
ax.set_ylim(-200,200)
ax.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{saving_path}/slices/midplaneIE_{snap}{cutoff}.png')
plt.show()

#%%
fig, ax = plt.subplots(1,1, figsize = (12,4))
img = ax.scatter(X_midplane/apo, Y_midplane/apo, c = abs_orb_en_mass_midplane, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=2e16, vmax=9e17))
cbar = plt.colorbar(img)
cbar.set_label(r'Specific absolute orbital energy', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i]/apo, ycfr_grid[i]/apo, cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(-1.3,R0/apo)
ax.set_ylim(-0.8,0.5)
ax.set_xlabel(r'X/$R_a$', fontsize = 18)
ax.set_ylabel(r'Y/$R_a$', fontsize = 18)
plt.tick_params(axis = 'both', which = 'both', direction='in', labelsize=20)
ax.text(-1.2, 0.4, r't/t$_{fb}$ = ' + str(np.round(tfb,2)), fontsize = 20)
plt.tight_layout()
if save:
    plt.savefig(f'{saving_path}/slices/midplaneEorb_{snap}{cutoff}.png')
plt.show()

#%%
fig, ax = plt.subplots(1,1, figsize = (8,4))
img = ax.scatter(X_midplane, Y_midplane, c = Rad_den_midplane, s = 1, cmap = 'viridis',norm=colors.LogNorm(vmin=1e-10, vmax=5e-8))
cbar = plt.colorbar(img)
cbar.set_label(r'$\log_{10}$ Rad energy density', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i], ycfr_grid[i], cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(-400,100)
ax.set_ylim(-200,200)
ax.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{saving_path}/slices/midplaneRad_{snap}{cutoff}.png')
plt.show()

#%%
""" Radial slices"""
thetas = [-2,-1,0,1, 2]#[-np.pi, -3/4*np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3/4*np.pi, np.pi]
# Visualize the chosen thetas for the slies
fig, ax = plt.subplots(2,1, figsize = (10,8))
img = ax[0].scatter(X_midplane, Y_midplane, c = Den_midplane, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin = 1e-9, vmax = 1e-5))
cbar = plt.colorbar(img)
cbar.set_label(r'$\log_{10}$ Density', fontsize = 16)
img1 = ax[1].scatter(X_midplane, Y_midplane, c = Mass_midplane, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin = 1e-10, vmax = 1e-7))
cbar1 = plt.colorbar(img1)
cbar1.set_label(r'$\log_{10}$ Mass', fontsize = 16)
# drawn radial lines (#put the - to follow the convention of the theta in the sim data)
for i in range(2):
    # for theta in thetas:
    #     ax[i].plot([0, -400*np.cos(theta)],[0, -400*np.sin(theta)], c = 'k', linestyle = '--')
    ax[i].scatter(0,0,c= 'k', marker = 'x', s=80)
    ax[i].set_xlim(-80,20)
    ax[i].set_ylim(-20,60)
    ax[i].set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
ax[1].set_xlabel(r'X [$R_\odot$]', fontsize = 18)
plt.suptitle(f'{check},' + r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + r', $\theta\in$[' + str(np.round(thetas[0]/np.pi,2)) + ',' + str(np.round(thetas[-1]/np.pi,2)) + r']$\pi$')
plt.tight_layout()

#%%
theta_chosen = thetas[0]
radial_slice = np.abs(THETA-theta_chosen) < 0.01
R_radial, Z_radial, dim_radial, Mass_radial, Den_radial, Temp_radial, Rad_den_radial, IEmass_radial, abs_orb_en_mass_radial = \
    sec.make_slices([Rsph, z_coord, dim_cell, mass, den, temp, rad_den, ie_mass, np.abs(orb_en_mass)], radial_slice)

#%%
fig, ax = plt.subplots(1,1, figsize = (12,4))
img = ax.scatter(R_radial, Z_radial, c = Den_radial, s = 1, cmap = 'viridis',norm=colors.LogNorm(vmin=1e-9, vmax=1e-5))
cbar = plt.colorbar(img)
cbar.set_label(r'$\log_{10}$ Density', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i], ycfr_grid[i], cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(0,1.2*apo)
ax.set_ylim(-50,50)
ax.set_xlabel(r'R [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Z [$R_\odot$]', fontsize = 18)
plt.title(r'$\theta = $' + str(np.round(theta_chosen/np.pi,1)) + r'$\pi, t/t_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{saving_path}/slices/Radial/radial{np.round(theta_chosen,1)}Den_{snap}{cutoff}.png')
plt.show()

#%%
fig, ax = plt.subplots(1,1, figsize = (12,4))
img = ax.scatter(R_radial, Z_radial, c = IEmass_radial, s = 1, cmap = 'viridis',norm=colors.LogNorm(vmin=1e-2, vmax=1))
cbar = plt.colorbar(img)
cbar.set_label(r'$\log_{10}$ specific IE', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i], ycfr_grid[i], cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(0,1.2*apo)
ax.set_ylim(-50,50)
ax.set_xlabel(r'R [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Z [$R_\odot$]', fontsize = 18)
plt.title(r'$\theta = $' + str(np.round(theta_chosen/np.pi,1)) + r'$\pi, t/t_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{saving_path}/slices/Radial/radial{np.round(theta_chosen,1)}IE_{snap}{cutoff}.png')
plt.show()

#%%
fig, ax = plt.subplots(1,1, figsize = (12,4))
img = ax.scatter(R_radial, Z_radial, c = abs_orb_en_mass_radial, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=10, vmax=110))
cbar = plt.colorbar(img)
cbar.set_label(r'$\log_{10}$ specific $|E_{orb}|$', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i], ycfr_grid[i], cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(0,1.2*apo)
ax.set_ylim(-50,50)
ax.set_xlabel(r'R [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Z [$R_\odot$]', fontsize = 18)
plt.title(r'$\theta = $' + str(np.round(theta_chosen/np.pi,1)) + r'$\pi, t/t_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{saving_path}/slices/Radial/radial{np.round(theta_chosen,1)}Eorb_{snap}{cutoff}.png')
plt.show()

#%%
fig, ax = plt.subplots(1,1, figsize = (12,4))
img = ax.scatter(R_radial, Z_radial, c = Rad_den_radial, s = 1, cmap = 'viridis',norm=colors.LogNorm(vmin=1e-10, vmax=5e-8))
cbar = plt.colorbar(img)
cbar.set_label(r'$\log_{10}$ Rad energy density', fontsize = 16)
ax.scatter(0,0,c= 'k', marker = 'x', s=80)
# plot cfr
for i in range(len(radii_grid)):
    ax.contour(xcfr_grid[i], ycfr_grid[i], cfr_grid[i], [0], linestyles = styles[i], colors = 'k', alpha = 0.8)
# ax.plot(x_stream, y_stream, c = 'k')
ax.set_xlim(0,1.2*apo)
ax.set_ylim(-50,50)
ax.set_xlabel(r'R [$R_\odot$]', fontsize = 18)
ax.set_ylabel(r'Z [$R_\odot$]', fontsize = 18)
plt.title(r'$\theta = $' + str(np.round(theta_chosen/np.pi,1)) + r'$\pi, t/t_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{saving_path}/slices/Radial/radial{np.round(theta_chosen,1)}Rad_{snap}{cutoff}.png')
plt.show()


#%%
from matplotlib.colors import ListedColormap, BoundaryNorm
colorsthree = ['orchid', 'white', 'dodgerblue']  # Example: red for negative, white for zero, green for positive
cmapthree = ListedColormap(colorsthree)
boundaries = [-1e7, -0.1, 0.1, 1e7]  # Slight offset to differentiate zero from positive
normthree = BoundaryNorm(boundaries, cmapthree.N, clip=True)

cutoff = 'cutden'
if difference:
    from scipy.spatial import KDTree
    check = 'Low' 
    check1 = 'HiRes'
    compton = 'Compton' 
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    saving_path = f'Figs/{folder}'
    snap = '199'

    path = f'{abspath}/TDE/{folder}{check}/{snap}'
    pathHiRes = f'{abspath}TDE/{folder}{check1}/{snap}'
    data = make_tree(path, snap, energy = True)
    dataHiRes = make_tree(pathHiRes, snap, energy = True)
    dim_cell = data.Vol**(1/3)
    dim_cellHiRes = dataHiRes.Vol**(1/3)
    tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    tfbHiRes = days_since_distruption(f'{pathHiRes}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    # print(f'tfb = {tfb}, tfbHiRes = {tfbHiRes}')
    x_coord, y_coord, z_coord,rad_den = \
        data.X, data.Y, data.Z, data.Rad,
    Rsph = np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
    Vsph = np.sqrt(data.VX**2 + data.VY**2 + data.VZ**2)
    orb_en = orb.orbital_energy(Rsph, Vsph, data.Mass, G, c, Mbh)
    x_coordHiRes, y_coordHiRes, z_coordHiRes, rad_den1 = \
        dataHiRes.X, dataHiRes.Y, dataHiRes.Z, dataHiRes.Rad
    RsphHiRes = np.sqrt(x_coordHiRes**2 + y_coordHiRes**2 + z_coordHiRes**2)
    VsphHiRes = np.sqrt(dataHiRes.VX**2 + dataHiRes.VY**2 + dataHiRes.VZ**2)
    orb_enHiRes = orb.orbital_energy(RsphHiRes, VsphHiRes, dataHiRes.Mass, G, c, Mbh)
    
    midplane = np.logical_and(np.abs(z_coord) < dim_cell, Rsph < 1.2*apo)
    midplaneHiRes = np.logical_and(np.abs(z_coordHiRes) < dim_cellHiRes, RsphHiRes < 1.2*apo)
    if cutoff == 'cutden':
        print('Cutting off low density elements')
        cutden = data.Den > 1e-9 # throw fluff
        cutdenHiRes = dataHiRes.Den > 1e-9 # throw fluff
        finalcut = cutden  & midplane
        finalcutHiRes = cutdenHiRes & midplaneHiRes
    else:
        finalcut = midplane
        finalcutHiRes = midplaneHiRes

    X_midplane, Y_midplane, Z_midplane, Rad_den_midplane = \
        sec.make_slices([x_coord, y_coord, z_coord, rad_den], finalcut)
    
    X_midplaneHiRes, Y_midplaneHiRes, Z_midplaneHiRes, Rad_den_midplaneHiRes = \
        sec.make_slices([x_coordHiRes, y_coordHiRes, z_coordHiRes, rad_den1], finalcutHiRes)
    
    points_toplot = np.array([X_midplane, Y_midplane, Z_midplane])
    points = points_toplot.T
    pointsHiRes = np.array([X_midplaneHiRes, Y_midplaneHiRes, Z_midplaneHiRes]).T
    treeHiRes = KDTree(pointsHiRes)
    new_Rad_den_midplaneHiRes = np.zeros_like(Rad_den_midplane)
    for i,pt in enumerate(points):
        _, idx = treeHiRes.query(pt)
        new_Rad_den_midplaneHiRes[i] = Rad_den_midplaneHiRes[idx]


    #%%
    RadDiff = (Rad_den_midplane - new_Rad_den_midplaneHiRes)*prel.en_den_converter
    RadDiff_rel = np.abs(RadDiff)/new_Rad_den_midplaneHiRes
    fig, ax = plt.subplots(1,1, figsize = (12,4))
    img = ax.scatter(points_toplot[0]/apo, points_toplot[1]/apo, c = RadDiff, s = 1, cmap = cmapthree, norm=normthree)
    cbar = plt.colorbar(img)
    cbar.set_label(r'Rad energy density', fontsize = 16)
    ax.set_xlim(-1.2,0.02)
    ax.set_ylim(-0.5,0.5)
    ax.set_xlabel(r'X/$R_a$', fontsize = 18)
    ax.set_ylabel(r'Y/$R_a$', fontsize = 18)
    if cutoff == 'bound':
        plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', bound elements only')
    elif cutoff == 'cutden':
        plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', cut on density')
    elif cutoff == 'cutdenbound':
        plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)) + ', cut on density and bound elements')
    else:
        plt.title(r'$|z|<V^{1/3}$, t/t$_{fb}$ = ' + str(np.round(tfb,2)))
    if save:
        plt.savefig(f'{abspath}{saving_path}/midplaneRadDiff_{snap}{cutoff}.png')
    # %%
