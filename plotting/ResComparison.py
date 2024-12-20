#%%
abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from src import orbits as orb
from Utilities import sections as sec
from Utilities.operators import make_tree
import matplotlib.gridspec as gridspec
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
price24 = 2.3e-7
bonlu20 = 1e-8
ryu23 = 0.096
RgSad = 1e5/c**2
r_sad16 = np.logspace(1.85*RgSad, 1000*RgSad, 256)
sad16 = np.min(np.diff(r_sad16))

##
# PARAMETERS
##
save = True

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
Rt = Rstar * (Mbh/mstar)**(1/3)
Rs = 2*G*Mbh / c**2
Rg = Rs/2
R0 = 0.6 * Rt
Rp =  Rt / beta
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
snap = '267'
compton = 'Compton'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
print(f'Rt: {Rt}, Rg: {Rs}, R0: {R0}, Rp: {Rp}, apo: {apo}')
print(f'In term of Rg: Rt: {Rt/Rg}, R0: {R0/Rg}, Rp: {Rp/Rg}, apo: {apo/Rg}')
xcr0, ycr0, cr0 = orb.make_cfr(R0)
xcrt, ycrt, crt = orb.make_cfr(Rt)
#%%
# LowRes
pathL = f'{abspath}TDE/{folder}LowRes/{snap}'
dataL = make_tree(pathL, snap)#, energy = True)
finalcutL = dataL.Den > 1e-19 # throw fluff
x_coordL, y_coordL, z_coordL, massL, denL, dim_cellL = \
    sec.make_slices([dataL.X, dataL.Y, dataL.Z, dataL.Mass, dataL.Den, dataL.Vol**(1/3)], finalcutL)
RadiusL = np.sqrt(x_coordL**2 + y_coordL**2 + z_coordL**2)
midplaneL = np.abs(z_coordL) < dim_cellL
X_midplaneL, Y_midplaneL, Z_midplaneL, dim_midplaneL, Mass_midplaneL, Den_midplaneL = \
    sec.make_slices([x_coordL, y_coordL, z_coordL, dim_cellL, massL, denL], midplaneL)

# Fiducial
path = f'{abspath}TDE/{folder}/{snap}'
data = make_tree(path, snap)#, energy = True)
finalcut = data.Den > 1e-19 # throw fluff
x_coord, y_coord, z_coord, mass, den, dim_cell = \
    sec.make_slices([data.X, data.Y, data.Z, data.Mass, data.Den, data.Vol**(1/3)], finalcut)
Radius = np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
midplane = np.abs(z_coord) < dim_cell
X_midplane, Y_midplane, Z_midplane, dim_midplane, Mass_midplane, Den_midplane = \
    sec.make_slices([x_coord, y_coord, z_coord, dim_cell, mass, den], midplane)

# HiRes
pathH = f'{abspath}TDE/{folder}HiRes/{snap}'
dataH = make_tree(pathH, snap)#, energy = True)
finalcutH = dataH.Den > 1e-19 # throw fluff
x_coordH, y_coordH, z_coordH, massH, denH, dim_cellH = \
    sec.make_slices([ dataH.X, dataH.Y, dataH.Z, dataH.Mass, dataH.Den, dataH.Vol**(1/3) ], finalcutH)
RadiusH = np.sqrt(x_coordH**2 + y_coordH**2 + z_coordH**2)
midplaneH = np.abs(z_coordH) < dim_cellH
X_midplaneH, Y_midplaneH, Z_midplaneH,  dim_midplaneH, Mass_midplaneH, Den_midplaneH = \
    sec.make_slices([x_coordH, y_coordH, z_coordH,  dim_cellH, massH, denH], midplaneH)


#%% Compare midplane resolution with a scatterplot
vminmass = np.percentile(Mass_midplaneH, 5)
vmaxmass = np.percentile(Mass_midplaneH, 95)
vmindim = np.percentile(dim_midplaneH, 5)
vmaxdim = np.percentile(dim_midplaneH, 95)
fig = plt.figure(figsize=(13, 8))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05])  # Adjust width_ratios for panels
ax0 = fig.add_subplot(gs[0, 0])
imgL = ax0.scatter(X_midplaneL, Y_midplaneL, c = Mass_midplaneL, s = 1, cmap = 'plasma', norm=colors.LogNorm(vmin=vminmass, vmax=vmaxmass))
ax1 = fig.add_subplot(gs[0, 1])
img = ax1.scatter(X_midplane, Y_midplane, c = Mass_midplane, s = 1, cmap = 'plasma', norm=colors.LogNorm(vmin=vminmass, vmax=vmaxmass))
ax2 = fig.add_subplot(gs[0, 2])
imgH = ax2.scatter(X_midplaneH, Y_midplaneH, c = Mass_midplaneH, s = 1, cmap = 'plasma', norm=colors.LogNorm(vmin=vminmass, vmax=vmaxmass))
cbar_ax = fig.add_subplot(gs[0, 3])  # Narrow subplot for colorbar
cbarH = plt.colorbar(imgH, cax=cbar_ax)
cbarH.set_label(r'Cell mass [$M_\odot$]', fontsize=18)
# cbarH.ax.tick_params(labelsize=18)
# Set the ticks to include the first and last values
# cbarH.set_ticks([vminmass, vmaxmass])
# Set the tick labels to the corresponding values
# cbarH.set_ticklabels([f'{vminmass:.1e}', f'{vmaxmass:.1e}'])
## Same for dim
ax3 = fig.add_subplot(gs[1, 0])
imgLdim = ax3.scatter(X_midplaneL, Y_midplaneL, c = dim_midplaneL, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=vmindim, vmax=vmaxdim))
ax4 = fig.add_subplot(gs[1, 1])
imgdim = ax4.scatter(X_midplane, Y_midplane, c = dim_midplane, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=vmindim, vmax=vmaxdim))
ax5 = fig.add_subplot(gs[1, 2])
imgHdim = ax5.scatter(X_midplaneH, Y_midplaneH, c = dim_midplaneH, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=vmindim, vmax=vmaxdim))
cbar_axdim = fig.add_subplot(gs[1, 3]) 
cbarHdim = plt.colorbar(imgHdim, cax=cbar_axdim)
cbarHdim.set_label(r'Cell size [$R_\odot$]', fontsize=18)
# cbarHdim.ax.tick_params(labelsize=18)
# cbarHdim.set_ticks([vmindim, vmaxdim])
# cbarHdim.set_ticklabels([f'{vmindim:.1e}', f'{vmaxdim:.1e}'])
# Layout adjustments
for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5]):
    ax.contour(xcr0, ycr0, cr0, [0], linestyles = 'dotted', colors = 'w', alpha = 1, linewidth = 3.5)
    ax.contour(xcrt, ycrt, crt, [0], linestyles = 'dashed', colors = 'w', alpha = 1, linewidth = 3.5)
    ax.hlines(y=0, xmin = 0, xmax = 30, color = 'w', linestyle = 'dashed', alpha = 0.8)
    ax.scatter(0,0, c= 'k', marker = 'x', s=80)
    ax.set_xlim(-40,25)
    ax.set_ylim(-40,40)
    if ax == ax0:
        check = 'Low'
    elif ax == ax1:
        check = 'Fid'
    elif ax == ax2:
        check = 'High'
    if ax in [ax0, ax1, ax2]:
        ax.text(-35,-35, f'{check} res', fontsize = 20, color = 'w')
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=20)
ax3.set_xlabel(r'X [$R_\odot$]', fontsize = 22)
ax4.set_xlabel(r'X [$R_\odot$]', fontsize = 22)
ax5.set_xlabel(r'X [$R_\odot$]', fontsize = 22)
ax0.set_ylabel(r'Y [$R_\odot$]', fontsize = 22)
ax3.set_ylabel(r'Y [$R_\odot$]', fontsize = 22)
plt.subplots_adjust(wspace=0.25)  # Wider spacing between the first and second plot
plt.tight_layout()
# plt.suptitle(r't/t$_{fb}$ = ' + str(np.round(tfb,2)))
if save:
    plt.savefig(f'{abspath}/Figs/multiple/compareRpMass_{snap}.png')
plt.show()

#%% CDF 
# Around the pericenter
cutzoomL = np.logical_and(x_coordL>R0, np.logical_and(x_coordL<2*Rt, np.abs(y_coordL)<Rt))
cutzoom = np.logical_and(x_coord>R0, np.logical_and(x_coord<2*Rt, np.abs(y_coord)<Rt))
cutzoomH = np.logical_and(x_coordH>R0, np.logical_and(x_coordH<2*Rt, np.abs(y_coordH)<Rt))
x_coordLzoom, y_coordLzoom, z_coordLzoom, massLzoom, denLzoom, dim_cellLzoom = \
    sec.make_slices([x_coordL, y_coordL, z_coordL, massL, denL, dim_cellL], cutzoomL)
x_coordzoom, y_coordzoom, z_coordzoom, masszoom, denzoom, dim_cellzoom = \
    sec.make_slices([x_coord, y_coord, z_coord, mass, den, dim_cell], cutzoom)
x_coordHzoom, y_coordHzoom, z_coordHzoom, massHzoom, denHzoom, dim_cellHzoom = \
    sec.make_slices([x_coordH, y_coordH, z_coordH, massH, denH, dim_cellH], cutzoomH)
# sort the arrays
massLzoom = np.sort(massLzoom)
masszoom = np.sort(masszoom)
massHzoom = np.sort(massHzoom)
dim_cellLzoom = np.sort(dim_cellLzoom)
dim_cellzoom = np.sort(dim_cellzoom)
dim_cellHzoom = np.sort(dim_cellHzoom)
# grazie Sill
cumHzoom = list(np.arange(len(massHzoom))/len(massHzoom))
cumzoom = list(np.arange(len(masszoom))/len(masszoom))
cumLzoom = list(np.arange(len(massLzoom))/len(massLzoom))
cumdimHzoom = list(np.arange(len(dim_cellHzoom))/len(dim_cellHzoom))
cumdimLzoom = list(np.arange(len(dim_cellLzoom))/len(dim_cellLzoom))
cumdimzoom = list(np.arange(len(dim_cellzoom))/len(dim_cellzoom))
massLzoom = list(massLzoom)
masszoom = list(masszoom)
massHzoom = list(massHzoom)
dim_cellLzoom = list(dim_cellLzoom)
dim_cellzoom = list(dim_cellzoom)
dim_cellHzoom = list(dim_cellHzoom)
# In all the volume
massL = np.sort(massL)
mass = np.sort(mass)
massH = np.sort(massH)
dim_cellL = np.sort(dim_cellL)
dim_cell = np.sort(dim_cell)
dim_cellH = np.sort(dim_cellH)
cumH = list(np.arange(len(massH))/len(massH))
cumL = list(np.arange(len(massL))/len(massL))
cum = list(np.arange(len(mass))/len(mass))
cumH.append(cumH[-1])
cumL.append(cumL[-1])
cum.append(cum[-1])
cumdimH = list(np.arange(len(dim_cellH))/len(dim_cellH))
cumdimL = list(np.arange(len(dim_cellL))/len(dim_cellL))
cumdim = list(np.arange(len(dim_cell))/len(dim_cell))
cumdimH.append(cumdimH[-1])
cumdimL.append(cumdimL[-1])
cumdim.append(cumdim[-1])
massL = list(massL)
mass = list(mass)
massH = list(massH)
mass.append(3e-7)
massL.append(3e-7)
massH.append(3e-7)
dim_cellL = list(dim_cellL)
dim_cell = list(dim_cell)
dim_cellH = list(dim_cellH)
dim_cell.append(2)
dim_cellL.append(2)
dim_cellH.append(2)

# Plot
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (5,8))
ax1.plot(massH, cumH, color = 'darkviolet', linestyle = 'dashed')#, label = 'High res')
ax1.plot(mass, cum, color = 'yellowgreen', linestyle = 'dashed')#, label = 'Middle res')
ax1.plot(massL, cumL, color = 'C1', linestyle = 'dashed')#, label = 'Low res')
ax1.axvline(price24, color = 'k', linestyle = 'dotted', label = 'Price24')
ax1.axvline(bonlu20, color = 'dodgerblue', linestyle = 'dashdot', label = 'BonnerotLu20')
ax1.plot(massHzoom, cumHzoom, color = 'darkviolet')#, label = 'High res')
ax1.plot(masszoom, cumzoom, color = 'yellowgreen')#, label = 'Middle res')
ax1.plot(massLzoom, cumLzoom, color = 'C1')#, label = 'Low res')
ax2.plot(dim_cellH, cumdimH, color = 'darkviolet',  linestyle = 'dashed')#, label = 'High res')
ax2.plot(dim_cell, cumdim, color = 'yellowgreen', linestyle = 'dashed')#, label = 'Middle res')
ax2.plot(dim_cellL, cumdimL, color = 'C1', linestyle = 'dashed')#, label = 'Low res')
ax2.plot(dim_cellHzoom, cumdimHzoom, color = 'darkviolet')#, label = 'High res')
ax2.plot(dim_cellzoom, cumdimzoom, color = 'yellowgreen')#, label = 'Middle res')
ax2.plot(dim_cellLzoom, cumdimLzoom, color = 'C1')#, label = 'Low res')
ax2.axvline(ryu23, color = 'k', linestyle = 'dotted', label = 'Ryu+23 initial')
# ax2.axvline(sad16, color = 'r', linestyle = 'dotted', label = 'Sadowski+16')

for ax in [ax1, ax2]:
    ax.set_xscale('log')
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=12)
    ax.legend(loc ='upper left', fontsize = 12)
    ax.set_ylabel('CDF', fontsize = 20)
    ax.set_ylim(0,1.1)
ax1.set_xlabel(r'Cell mass [$M_\odot$]', fontsize = 15)
ax2.set_xlabel(r'Cell size [$R_\odot$]', fontsize = 15)
ax1.set_xlim(5e-13, 3e-7)
ax2.set_xlim(7e-2, 2)
# plt.suptitle(r'Near pericenter: $R_0<X<25, \, |Y|<4$', fontsize = 20)
plt.tight_layout()
if save:
    plt.savefig(f'{abspath}/Figs/multiple/compareHistToget_{snap}.pdf')
plt.show()
