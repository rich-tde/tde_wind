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
from Utilities.operators import make_tree, to_cylindric
import matplotlib.gridspec as gridspec
from Utilities.time_extractor import days_since_distruption
from scipy.spatial import KDTree

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
save = False

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
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)
snap = '216'
compton = 'Compton'
checks = ['Low', 'HiRes'] 
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
print(f'Rt: {Rt}, Rg: {Rs}, R0: {R0}, Rp: {Rp}, apo: {apo}')
print(f'In term of Rg: Rt: {Rt/Rg}, R0: {R0/Rg}, Rp: {Rp/Rg}, apo: {apo/Rg}')

#%%
pathL = f'{abspath}TDE/{folder}{checks[0]}/{snap}'
tfb = days_since_distruption(f'{pathL}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
dataL = make_tree(pathL, snap, energy = True)
x_coordL, y_coordL, z_coordL, massL, denL = \
    dataL.X, dataL.Y, dataL.Z, dataL.Mass, dataL.Den
dim_cellL = dataL.Vol**(1/3) 

finalcutL = dataL.Den > 1e-9 # throw fluff
x_coordL, y_coordL, z_coordL, massL, denL, dim_cellL = \
    sec.make_slices([x_coordL, y_coordL, z_coordL, massL, denL, dim_cellL], finalcutL)
midplaneL = np.abs(z_coordL) < dim_cellL
X_midplaneL, Y_midplaneL, Z_midplaneL, dim_midplaneL, Mass_midplaneL, Den_midplaneL = \
    sec.make_slices([x_coordL, y_coordL, z_coordL, dim_cellL, massL, denL], midplaneL)

pathH = f'{abspath}TDE/{folder}{checks[1]}/{snap}'
dataH = make_tree(pathH, snap, energy = True)
x_coordH, y_coordH, z_coordH, massH, denH = \
    dataH.X, dataH.Y, dataH.Z, dataH.Mass, dataH.Den
dim_cellH = dataH.Vol**(1/3) 
finalcutH = dataH.Den > 1e-9 # throw fluff
x_coordH, y_coordH, z_coordH, massH, denH, dim_cellH = \
    sec.make_slices([x_coordH, y_coordH, z_coordH, massH, denH, dim_cellH], finalcutH)
midplaneH = np.abs(z_coordH) < dim_cellH
X_midplaneH, Y_midplaneH, Z_midplaneH,  dim_midplaneH, Mass_midplaneH, Den_midplaneH = \
    sec.make_slices([x_coordH, y_coordH, z_coordH,  dim_cellH, massH, denH], midplaneH)

RadiusL = np.sqrt(x_coordL**2 + y_coordL**2 + z_coordL**2)
RadiusH = np.sqrt(x_coordH**2 + y_coordH**2 + z_coordH**2)
#%%
xcr0, ycr0, cr0 = orb.make_cfr(R0)
xcrt, ycrt, crt = orb.make_cfr(Rt)

#%% Find min value
findminMassL = RadiusL>R0
findminMassH = RadiusH>R0
x_coordLfind, y_coordLfind, z_coordLfind, massLfind = \
    sec.make_slices([x_coordL, y_coordL, z_coordL, massL], findminMassL)
x_coordHfind, y_coordHfind, z_coordHfind, massHfind = \
    sec.make_slices([x_coordH, y_coordH, z_coordH, massH], findminMassH)
idxL = np.argmin(massLfind)
idxH = np.argmin(massHfind)
# mean near pericenter
print('Low res mean:', np.mean(massLfind[np.sqrt(x_coordLfind**2 + (y_coordLfind-Rp)**2 + z_coordLfind**2) < 5]))
print('Middle res mean:', np.mean(massHfind[np.sqrt(x_coordHfind**2 + (y_coordHfind-Rp)**2 + z_coordHfind**2) < 5]))
print('Low res min:', x_coordLfind[idxL], y_coordLfind[idxL], z_coordLfind[idxL], 'Mass:', massLfind[idxL])
print('Middle res min: ', x_coordHfind[idxH], y_coordHfind[idxH], z_coordHfind[idxH], 'Mass:', massHfind[idxH])

#%% Midplane resolution 
colorstext = ['k', 'w', 'k', 'w']
vminmass = np.percentile(Mass_midplaneH, 5)
vmaxmass = np.percentile(Mass_midplaneH, 95)
vmindim = np.percentile(dim_midplaneH, 5)
vmaxdim = np.percentile(dim_midplaneH, 95)
fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])  # Adjust width_ratios for panels
ax1 = fig.add_subplot(gs[0, 0])
imgL = ax1.scatter(X_midplaneL, Y_midplaneL, c = Mass_midplaneL, s = 1, cmap = 'inferno', norm=colors.LogNorm(vmin=vminmass, vmax=vmaxmass))
ax2 = fig.add_subplot(gs[0, 1])
imgH = ax2.scatter(X_midplaneH, Y_midplaneH, c = Mass_midplaneH, s = 1, cmap = 'inferno', norm=colors.LogNorm(vmin=vminmass, vmax=vmaxmass))
cbar_ax = fig.add_subplot(gs[0, 2])  # Narrow subplot for colorbar
cbarH = plt.colorbar(imgH, cax=cbar_ax)
cbarH.set_label(r'Cell mass [$M_\odot$]', fontsize=18)
cbarH.ax.tick_params(labelsize=18)
# Set the ticks to include the first and last values
# cbarH.set_ticks([vminmass, vmaxmass])
# Set the tick labels to the corresponding values
# cbarH.set_ticklabels([f'{vminmass:.1e}', f'{vmaxmass:.1e}'])
## Same for dim
ax3 = fig.add_subplot(gs[1, 0])
imgLdim = ax3.scatter(X_midplaneL, Y_midplaneL, c = dim_midplaneL, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=vmindim, vmax=vmaxdim))
ax4 = fig.add_subplot(gs[1, 1])
imgHdim = ax4.scatter(X_midplaneH, Y_midplaneH, c = dim_midplaneH, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=vmindim, vmax=vmaxdim))
cbar_axdim = fig.add_subplot(gs[1, 2]) 
cbarHdim = plt.colorbar(imgHdim, cax=cbar_axdim)
cbarHdim.set_label(r'Cell size [$R_\odot$]', fontsize=18)
cbarHdim.ax.tick_params(labelsize=18)
# cbarHdim.set_ticks([vmindim, vmaxdim])
# cbarHdim.set_ticklabels([f'{vmindim:.1e}', f'{vmaxdim:.1e}'])
# Layout adjustments
for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    ax.contour(xcr0, ycr0, cr0, [0], linestyles = 'dotted', colors = colorstext[i], alpha = 1, linewidth = 3.5)
    ax.contour(xcrt, ycrt, crt, [0], linestyles = 'dashed', colors = colorstext[i], alpha = 1, linewidth = 3.5)
    # ax.text(Rp+1, -4, r'$R_p$', fontsize = 20, color = colorstext[i], weight = 'bold')
    # ax.text(-7, Rp+1, r'$R_t$', fontsize = 20, color = colorstext[i], weight = 'bold')
    ax.hlines(y=0, xmin = 0, xmax = 30, color = colorstext[i], linestyle = 'dashed', alpha = 0.8)
    ax.scatter(0,0, c= 'k', marker = 'x', s=80)
    ax.set_xlim(-40,25)
    ax.set_ylim(-40,40)
    if i == 0 or i==2:
        check = 'Low'
    else:
        check = 'High'
    ax.text(9,-37, f'{check} res', fontsize = 18)
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=20)
ax1.set_ylabel(r'Y [$R_\odot$]', fontsize = 22)
ax3.set_xlabel(r'X [$R_\odot$]', fontsize = 22)
ax3.set_ylabel(r'Y [$R_\odot$]', fontsize = 22)
ax4.set_xlabel(r'X [$R_\odot$]', fontsize = 22)
plt.subplots_adjust(wspace=0.25)  # Wider spacing between the first and second plot
# plt.suptitle(r't/t$_{fb}$ = ' + str(np.round(tfb,2)))
plt.tight_layout()
if save:
    plt.savefig(f'{abspath}/Figs/{folder}/multiple/compareRpMass_{snap}.png')
plt.show()

# %%
cuthistogramsL = np.logical_and(x_coordL>R0, np.logical_and(x_coordL<25, np.abs(y_coordL)<4))
cuthistogramsH = np.logical_and(x_coordH>R0, np.logical_and(x_coordH<25, np.abs(y_coordH)<4))
x_coordLhist, y_coordLhist, z_coordLhist, massLhist, denLhist, dim_cellLhist = \
    sec.make_slices([x_coordL, y_coordL, z_coordL, massL, denL, dim_cellL], cuthistogramsL)
x_coordHhist, y_coordHhist, z_coordHhist, massHhist, denHhist, dim_cellHhist = \
    sec.make_slices([x_coordH, y_coordH, z_coordH, massH, denH, dim_cellH], cuthistogramsH)

# sort the arrays
massLhist = np.sort(massLhist)
massHhist = np.sort(massHhist)
dim_cellLhist = np.sort(dim_cellLhist)
dim_cellHhist = np.sort(dim_cellHhist)
# make histograms of mass and size
binsL = np.logspace(np.log10(np.min(massLhist)), np.log10(np.max(massLhist)), 100)
binsH = np.logspace(np.log10(np.min(massHhist)), np.log10(np.max(massHhist)), 100)
binsdimL = np.logspace(np.log10(np.min(dim_cellLhist)), np.log10(np.max(dim_cellLhist)), 100)
binsdimH = np.logspace(np.log10(np.min(dim_cellHhist)), np.log10(np.max(dim_cellHhist)), 100)
fig, ax = plt.subplots(1,2, figsize = (12,6), sharey='row')
ax[0].hist(massLhist, bins = binsL, color = 'orange', alpha = 0.5, label = 'Low res')
ax[1].hist(massHhist, bins = binsH, color = 'mediumpurple', alpha = 0.5, label = 'High res')
for axs in ax:
    axs.loglog()
    axs.set_xlim(5e-13, 1e-7)
    axs.set_xlabel(r'Cell mass [$M_\odot$]', fontsize = 20)
    axs.tick_params(axis = 'both', which = 'both', direction='in', labelsize=20)
    axs.tick_params(which = 'major', size= 7)
    axs.tick_params(which = 'minor', size= 5)
    axs.legend(fontsize = 20)
ax[0].set_ylabel('Counts', fontsize = 20)
if save:
    plt.savefig(f'{abspath}/Figs/{folder}/multiple/compareHistMass_{snap}.png')
plt.show()

#%%
cumH = np.arange(len(massHhist))/len(massHhist)
cumL = np.arange(len(massLhist))/len(massLhist)
cumdimH = np.arange(len(dim_cellHhist))/len(dim_cellHhist)
cumdimL = np.arange(len(dim_cellLhist))/len(dim_cellLhist)
deltaMass = np.max(massLhist) - np.max(massHhist)
deltaDim = np.max(dim_cellLhist) - np.max(dim_cellHhist)

massHhistshifted = massHhist + deltaMass
dim_cellHhistshifted = dim_cellHhist + deltaDim

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))
# grazie Sill
ax1.plot(massHhistshifted, cumH, color = 'mediumpurple', label = 'High res')
ax1.plot(massLhist, cumL, color = 'orange', label = 'Low res')
ax2.plot(dim_cellHhistshifted, cumdimH, color = 'mediumpurple',  label = 'High res')
ax2.plot(dim_cellLhist, cumdimL, color = 'orange', linestyle='--',label = 'Low res')

for ax in [ax1, ax2]:
    ax.set_xscale('log')
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=15)
    ax.legend(loc ='upper left', fontsize = 18)
ax1.set_xlabel(r'Cell mass [$M_\odot$]', fontsize = 20)
ax2.set_xlabel(r'Cell size [$R_\odot$]', fontsize = 20)
ax1.set_ylabel('CDF', fontsize = 25)
# ax1.set_xlim(5e-13, 3e-8)
# ax2.set_xlim(7e-2, 4e-1)
plt.suptitle(r'Near pericenter: $R_0<X<25, \, |Y|<4$', fontsize = 20)
if save:
    plt.savefig(f'{abspath}/Figs/{folder}/multiple/compareHistToget_{snap}.png')
plt.show()

#%% As before, but trying to stretch the x axis of orange to overlap with purple
coeffL = 0.1
coeffdimL = 0.55
widthbinsL = np.diff(binsL)
widthbinsdimL = np.diff(binsdimL)
binsLwider = np.copy(binsL)
binsdimLwider = np.copy(binsdimL)
for i in range(-1, -len(binsL), -1):
    binsLwider[i-1] = binsLwider[i] - widthbinsL[i]*coeffL
    binsdimLwider[i-1] = binsdimLwider[i] - widthbinsdimL[i]*coeffdimL

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))
# grazie Sill
ax1.plot(massHhist, cumH, color = 'mediumpurple', label = 'High res')
ax1.plot(coeffL*massLhist, cumL, color = 'orange', label = f'Low res stretched of {np.round(1/coeffL,1)}')
ax2.plot(dim_cellHhist, cumdimH, color = 'mediumpurple', label = 'High res')
ax2.plot(coeffdimL*dim_cellLhist, cumdimL, color = 'orange', label = f'Low res stretched of {np.round(1/coeffdimL,1)}')

for ax in [ax1, ax2]:
    ax.set_xscale('log')
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=15)
    ax.legend(loc ='lower right', fontsize = 15)
ax1.set_xlabel(r'Cell mass [$M_\odot$]', fontsize = 20)
ax2.set_xlabel(r'Cell size [$R_\odot$]', fontsize = 20)
ax1.set_ylabel('CDF', fontsize = 25)
ax1.set_xlim(5e-13, 3e-8)
ax2.set_xlim(7e-2, 4e-1)
plt.suptitle(r'Near pericenter: $R_0<X<25, \, |Y|<4$', fontsize = 20)
if save:
    plt.savefig(f'{abspath}/Figs/{folder}/multiple/compareHistStretch_{snap}.png')
plt.show()
#%%
binsLall = np.logspace(np.log10(np.min(massL)), np.log10(np.max(massL)), 100)
binsHall = np.logspace(np.log10(np.min(massH)), np.log10(np.max(massH)), 100)
fig = plt.subplots(1,1, figsize = (8,8))
plt.hist(massH, bins = binsHall, ec = 'mediumpurple', histtype='stepfilled', color = 'w', label = 'High res', cumulative = True, density= True) # put an histogram over the other. one just the contour
plt.hist(massL, bins = binsLall, color = 'orange', alpha = 0.5, label = 'Low res', cumulative=True, density= True)
plt.xlim(3e-12, 1e-6)
plt.xlabel(r'Mass per cell [$M_\odot$]', fontsize = 20)
plt.ylabel('CDF', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', direction='in', labelsize=20)
# plt.loglog()
plt.xscale('log')
plt.legend(loc ='upper left', fontsize = 18)
plt.title(r'CDF All cells with $\rho>10^{-9}$', fontsize = 20)
if save:
    plt.savefig(f'{abspath}/Figs/{folder}/multiple/compareAllHistMass_{snap}.png')
plt.show()

# %% Find resolution along the stream
fileL = np.load(f'{abspath}data/{folder}/DENstream_{checks[0]}{snap}.npy')
x_streamL, y_streamL, z_streamL = fileL[1], fileL[2], fileL[3]
pointsL = np.array([x_streamL, y_streamL, z_streamL]).T
xapproxL = np.zeros_like(x_streamL)
yapproxL = np.zeros_like(y_streamL)
zapproxL = np.zeros_like(z_streamL)
mass_streamL = np.zeros_like(x_streamL)
nearbyxL = []
nearbyyL = []
nearbyzL = []
nearbymassL = []
nearbydimcellL = []
treeL = dataL.sim_tree
for i, point in enumerate(pointsL):
    _, idx = treeL.query(point)
    mass_streamL[i] = dataL.Mass[idx]
    xapproxL[i] = dataL.X[idx]
    yapproxL[i] = dataL.Y[idx]
    zapproxL[i] = dataL.Z[idx]
    _, indices = treeL.query(point, k=20)
    nearbyxL.append(dataL.X[indices])
    nearbyyL.append(dataL.Y[indices])
    nearbyzL.append(dataL.Z[indices])
    nearbymassL.append(dataL.Mass[indices])
    nearbydimcellL.append(dataL.Vol[indices]**(1/3))

fileH = np.load(f'{abspath}data/{folder}/DENstream_{checks[1]}{snap}.npy')
x_streamH, y_streamH, z_streamH = fileH[1], fileH[2], fileH[3]
pointsH = np.array([x_streamH, y_streamH, z_streamH]).T
mass_streamH = np.zeros_like(x_streamH)
xapproxH = np.zeros_like(x_streamH)
yapproxH = np.zeros_like(y_streamH)
zapproxH = np.zeros_like(z_streamH)
nearbyxH = []
nearbyyH = []
nearbyzH = []
nearbymassH = []
nearbydimcellH = []
treeH = dataH.sim_tree
for i, point in enumerate(pointsH):
    _, idx = treeH.query(point)
    mass_streamH[i] = dataH.Mass[idx]
    xapproxH[i] = dataH.X[idx]
    yapproxH[i] = dataH.Y[idx]
    zapproxH[i] = dataH.Z[idx]
    _, indices = treeH.query(point, k=20)
    # indices = [int(i) for i in indices]
    nearbyxH.append(dataH.X[indices])
    nearbyyH.append(dataH.Y[indices])
    nearbyzH.append(dataH.Z[indices])
    nearbymassH.append(dataH.Mass[indices])
    nearbydimcellH.append(dataH.Vol[indices]**(1/3))

nearbyxL = np.concatenate(nearbyxL)
nearbyyL = np.concatenate(nearbyyL)
nearbyzL = np.concatenate(nearbyzL)
nearbymassL = np.concatenate(nearbymassL)
nearbydimcellL = np.concatenate(nearbydimcellL)
nearbyxH = np.concatenate(nearbyxH)
nearbyyH = np.concatenate(nearbyyH)
nearbyzH = np.concatenate(nearbyzH)
nearbymassH = np.concatenate(nearbymassH)
nearbydimcellH = np.concatenate(nearbydimcellH)
# plt.plot(1-xapproxL/x_streamL, label = 'Low res')
# plt.plot(1-xapproxH/x_streamH, label = 'High res')
# plt.legend()
# plt.show()
#%% Midplane
nearbymidplaneL = np.abs(nearbyzL) < nearbydimcellL
nearbyxL, nearbyyL, nearbyzL, nearbymassL, nearbydimcellL = \
    sec.make_slices([nearbyxL, nearbyyL, nearbyzL, nearbymassL, nearbydimcellL], nearbymidplaneL)
nearbymidplaneH = np.abs(nearbyzH) < nearbydimcellH
nearbyxH, nearbyyH, nearbyzH, nearbymassH, nearbydimcellH = \
    sec.make_slices([nearbyxH, nearbyyH, nearbyzH, nearbymassH, nearbydimcellH], nearbymidplaneH)
# %%
fig, ax = plt.subplots(2,1, figsize = (10,8))
img = ax[0].scatter(nearbyxL, nearbyyL, c = nearbymassL, s = 2, cmap = 'inferno', norm=colors.LogNorm(vmin=5e-9, vmax=5e-6))
cbar = plt.colorbar(img)
cbar.set_label(r'Mass per cell $M_\odot$', fontsize = 16)
cbar.ax.tick_params(labelsize=16)
ax[0].text(-320, -20, f'Mean mass stream: {np.mean(mass_streamL):.0e}', fontsize = 16)

img = ax[1].scatter(nearbyxH, nearbyyH, c = nearbymassH, s = 2, cmap = 'inferno', norm=colors.LogNorm(vmin=1e-9, vmax=1e-7))
cbar = plt.colorbar(img)
cbar.set_label(r'Mass per cell [$M_\odot$]', fontsize = 16)
cbar.ax.tick_params(labelsize=16)
ax[1].set_xlabel(r'X [$R_\odot$]', fontsize = 20)
ax[1].text(-320, -20, f'Mean mass stream: {np.mean(mass_streamH):.0e}', fontsize = 16)
for axs in ax:
    axs.set_ylabel(r'Y [$R_\odot$]', fontsize = 20)
    axs.scatter(0,0, c= 'k', marker = 'x', s=80)
    axs.set_xlim(-340,25)
    axs.set_ylim(-40,70)
    axs.tick_params(axis = 'both', which = 'both', direction='in', labelsize=16)
    # print the mean mass in exponential notation
if save:
    plt.savefig(f'{abspath}/Figs/{folder}/multiple/compareStreamMass_{snap}.png')
plt.show()
# %%
print('Mean mass along stream low res:', np.mean(mass_streamL))
print('Mean mass nearby stream low res:', np.mean(nearbymassL))
print('Mean mass along stream high res:', np.mean(mass_streamH))
print('Mean mass nearby stream high res:', np.mean(nearbymassH))

#%%
binsLstream = np.logspace(np.log10(np.min(nearbymassL)), np.log10(np.max(nearbymassL)), 100)
binsHstream = np.logspace(np.log10(np.min(nearbymassH)), np.log10(np.max(nearbymassH)), 100)
plt.hist(nearbymassH, bins = binsHstream, cumulative=True, density= True, ec = 'mediumpurple', histtype='stepfilled', color = 'w', alpha = 0.5, label = 'High res')
plt.hist(nearbymassL, bins = binsLstream, cumulative=True, density= True,color = 'orange', alpha = 0.5, label = 'Low res')
plt.xscale('log')
plt.title(r'CDF Nearby stream', fontsize = 20)
plt.xlabel(r'Mass per cell [$M_\odot$]', fontsize = 20)
plt.ylabel('Counts', fontsize = 20)
plt.legend(fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', direction='in', labelsize=20)
if save:
    plt.savefig(f'{abspath}/Figs/{folder}/multiple/compareHistStreamMass_{snap}.png')
plt.show()
