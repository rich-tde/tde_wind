"""Compare resolutions: scatter plot of pericenter region and CDF of mass and size with comparison to other papers"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from scipy.stats import gmean
from src import orbits as orb
from Utilities import sections as sec
from Utilities.operators import make_tree
import matplotlib.gridspec as gridspec
from Utilities.time_extractor import days_since_distruption

##
# CONSTANTS
#

price24 = 2.3e-7
bonlu20 = 8.4e-9 #this is at late times, but they start with Mp = 1e-8
Hu25 = 1/7e11
# Ryu
dthetaRyu = 0.008
dphiryu = 0.0079
RgRyu = prel.G*10**5 / prel.csol_cgs**2
arrR_Ryu23 = np.logspace(np.log10(40*RgRyu), np.log10(18000*RgRyu), 800)
dR_Ryu23 = np.diff(arrR_Ryu23)
dRminRyu = np.min(dR_Ryu23)
dRmeanRyu = gmean(dR_Ryu23)
dRoverR_Ryu23 = (dR_Ryu23/arrR_Ryu23[:-1])[0] #they are all the same
sizehuang24 = 0.7
# cell size = R*(alpha*dtheta*cos(theta)*dphi)**(1/3) where theta is the latitude angle ad alpha = dR/R
# and so cell size = (R**2 * dR * dtheta * dphi * cos(theta))**(1/3)
# we consider the midplane
sizeminRyu = arrR_Ryu23[0] * (dRoverR_Ryu23 * dthetaRyu * 1 * dphiryu)**(1/3) #
sizemeanRyu = gmean(arrR_Ryu23) * (dRoverR_Ryu23 * dthetaRyu * 1 * dphiryu)**(1/3)
# Sadowski 
RgSad = 1e5/prel.csol_cgs**2
r_sad16 = np.logspace(1.85*RgSad, 1000*RgSad, 256)
sad16 = np.min(np.diff(r_sad16))
##

#%%
# PARAMETERS
##
save = True
include_mid = '' # 'mid' or '

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
snapH = '116'
snapM = '326'
snapL = '340'
compton = 'Compton'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
tfb = np.loadtxt(f'{abspath}/TDE/{folder}NewAMR/{snapM}/tfb_{snapM}.txt')
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
R0 = things['R0']
xcr0, ycr0, cr0 = orb.make_cfr(R0)
xcrt, ycrt, crt = orb.make_cfr(Rt)
xcrtRes, ycrtRes, crtRes = orb.make_cfr(1.5*Rt)
xcrtRes2, ycrtRes2, crtRes2 = orb.make_cfr(1.2*Rt)
#%%
# LowRes
pathL = f'{abspath}/TDE/{folder}LowResNewAMR/{snapL}'
dataL = make_tree(pathL, snapL)#, energy = True)
finalcutL = dataL.Den > 1e-19 # throw fluff
x_coordL, y_coordL, z_coordL, massL, denL, dim_cellL = \
    sec.make_slices([dataL.X, dataL.Y, dataL.Z, dataL.Mass, dataL.Den, dataL.Vol**(1/3)], finalcutL)
RadiusL = np.sqrt(x_coordL**2 + y_coordL**2 + z_coordL**2)
midplaneL = np.abs(z_coordL) < dim_cellL
X_midplaneL, Y_midplaneL, Z_midplaneL, dim_midplaneL, mass_midplaneL, Den_midplaneL = \
    sec.make_slices([x_coordL, y_coordL, z_coordL, dim_cellL, massL, denL], midplaneL)

# Fiducial
path = f'{abspath}/TDE/{folder}NewAMR/{snapM}'
data = make_tree(path, snapM)#, energy = True)
finalcut = data.Den > 1e-19 # throw fluff
x_coord, y_coord, z_coord, mass, den, dim_cell = \
    sec.make_slices([data.X, data.Y, data.Z, data.Mass, data.Den, data.Vol**(1/3)], finalcut)
Radius = np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
midplane = np.abs(z_coord) < dim_cell
X_midplane, Y_midplane, Z_midplane, dim_midplane, mass_midplane, Den_midplane = \
    sec.make_slices([x_coord, y_coord, z_coord, dim_cell, mass, den], midplane)

# HiRes
pathH = f'{abspath}/TDE/{folder}HiResNewAMR/{snapH}'
dataH = make_tree(pathH, snapH)#, energy = True)
finalcutH = dataH.Den > 1e-19 # throw fluff
x_coordH, y_coordH, z_coordH, massH, denH, dim_cellH = \
    sec.make_slices([ dataH.X, dataH.Y, dataH.Z, dataH.Mass, dataH.Den, dataH.Vol**(1/3) ], finalcutH)
RadiusH = np.sqrt(x_coordH**2 + y_coordH**2 + z_coordH**2)
midplaneH = np.abs(z_coordH) < dim_cellH
X_midplaneH, Y_midplaneH, Z_midplaneH,  dim_midplaneH, mass_midplaneH, Den_midplaneH = \
    sec.make_slices([x_coordH, y_coordH, z_coordH,  dim_cellH, massH, denH], midplaneH)

#%%
print('min mass low:', np.min(massL), 'min dim low:', np.min(dim_cellL))
print('min mass fiducial:', np.min(mass),'min dim fiducial:', np.min(dim_cell))
print('min mass high:', np.min(massH), 'min dim high:', np.min(dim_cellH))
#
print('50 percentile low mass', np.percentile(massL, 50), '90 percentile low mass', np.percentile(massL, 90))
print('50 percentile fiducial mass', np.percentile(mass, 50), '90 percentile fiducial mass', np.percentile(mass, 90))
print('50 percentile high mass', np.percentile(massH, 50), '90 percentile high mass', np.percentile(massH, 90))
#
print('50 percentile low dim', np.percentile(dim_cellL, 50), '90 percentile low dim', np.percentile(dim_cellL, 90))
print('50 percentile fiducial dim', np.percentile(dim_cell, 50), '90 percentile fiducial dim', np.percentile(dim_cell, 90))
print('50 percentile high dim', np.percentile(dim_cellH, 50), '90 percentile high dim', np.percentile(dim_cellH, 90))
#
print('min mass at midplane low:', np.min(mass_midplaneL), 'min dim at midplane low:', np.min(dim_midplaneL))
print('min mass at midplane fiducial:', np.min(mass_midplane), 'min dim at midplane fiducial:', np.min(dim_midplane))
print('min mass at midplane high:', np.min(mass_midplaneH), 'min dim at midplane high:', np.min(dim_midplaneH))
#%% Compare midplane resolution with a scatterplot
vminmass = np.percentile(mass_midplaneH, 5)
vmaxmass = np.percentile(mass_midplaneH, 95)
vmindim = 7e-2 #np.percentile(dim_midplaneH, 5)
vmaxdim = 1.1#np.percentile(dim_midplaneH, 95)
fig = plt.figure(figsize=(13, 8))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05])  # Adjust width_ratios for panels
ax0 = fig.add_subplot(gs[0, 0])
imgL = ax0.scatter(X_midplaneL, Y_midplaneL, c = mass_midplaneL, s = 1, cmap = 'plasma', norm=colors.LogNorm(vmin=vminmass, vmax=vmaxmass), rasterized = True)
ax1 = fig.add_subplot(gs[0, 1])
img = ax1.scatter(X_midplane, Y_midplane, c = mass_midplane, s = 1, cmap = 'plasma', norm=colors.LogNorm(vmin=vminmass, vmax=vmaxmass), rasterized = True)
ax2 = fig.add_subplot(gs[0, 2])
imgH = ax2.scatter(X_midplaneH, Y_midplaneH, c = mass_midplaneH, s = 1, cmap = 'plasma', norm=colors.LogNorm(vmin=vminmass, vmax=vmaxmass), rasterized = True)
cbar_ax = fig.add_subplot(gs[0, 3])  # Narrow subplot for colorbar
cbarH = plt.colorbar(imgH, cax=cbar_ax)
cbarH.set_label(r'Cell mass [$M_\odot$]', fontsize=20)
# cbarH.ax.tick_params(labelsize=20)
# Set the ticks to include the first and last values
# cbarH.set_ticks([vminmass, vmaxmass])
# Set the tick labels to the corresponding values
# cbarH.set_ticklabels([f'{vminmass:.1e}', f'{vmaxmass:.1e}'])
## Same for dim
ax3 = fig.add_subplot(gs[1, 0])
imgLdim = ax3.scatter(X_midplaneL, Y_midplaneL, c = dim_midplaneL, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=vmindim, vmax=vmaxdim), rasterized = True)
ax4 = fig.add_subplot(gs[1, 1])
imgdim = ax4.scatter(X_midplane, Y_midplane, c = dim_midplane, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=vmindim, vmax=vmaxdim), rasterized = True)
ax5 = fig.add_subplot(gs[1, 2])
imgHdim = ax5.scatter(X_midplaneH, Y_midplaneH, c = dim_midplaneH, s = 1, cmap = 'viridis', norm=colors.LogNorm(vmin=vmindim, vmax=vmaxdim), rasterized = True)
cbar_axdim = fig.add_subplot(gs[1, 3]) 
cbarHdim = plt.colorbar(imgHdim, cax=cbar_axdim)
cbarHdim.set_label(r'Cell size [$R_\odot$]', fontsize=20)
# cbarHdim.ax.tick_params(labelsize=18)
# cbarHdim.set_ticks([vmindim, vmaxdim])
# cbarHdim.set_ticklabels([f'{vmindim:.1e}', f'{vmaxdim:.1e}'])
# Layout adjustments
for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5]):
    ax.contour(xcr0, ycr0, cr0, [0], linestyles = 'dotted', colors = 'w', alpha = 1, linewidth = 3.5)
    ax.contour(xcrt, ycrt, crt, [0], linestyles = 'dashed', colors = 'w', alpha = 1, linewidth = 3.5)
    # ax.contour(xcrtRes, ycrtRes, crtRes, [0], linestyles = 'solid', colors = 'w', alpha = 1, linewidth = 3.5)
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
        ax.text(-35,-35, f'{check} res', fontsize = 20, color = 'w', weight='bold')
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=20)
ax3.set_xlabel(r'X [$R_\odot$]')#, fontsize = 22)
ax4.set_xlabel(r'X [$R_\odot$]')#, fontsize = 22)
ax5.set_xlabel(r'X [$R_\odot$]')#, fontsize = 22)
ax0.set_ylabel(r'Y [$R_\odot$]')#, fontsize = 22)
ax3.set_ylabel(r'Y [$R_\odot$]')#, fontsize = 22)
plt.subplots_adjust(wspace=0.25)  # Wider spacing between the first and second plot
# plt.suptitle(r't/t$_{fb}$ = ' + str(np.round(tfb,2)))
plt.tight_layout()
if save:
    plt.savefig(f'{abspath}/Figs/paper/compareRpMass.pdf', bbox_inches='tight')
plt.show()

#%% CDF, grazie Sill
massL = np.sort(massL)
mass = np.sort(mass)
massH = np.sort(massH)
dim_cellL = np.sort(dim_cellL)
dim_cell = np.sort(dim_cell)
dim_cellH = np.sort(dim_cellH)
cumH = list(np.arange(len(massH))/len(massH))
cumL = list(np.arange(len(massL))/len(massL))
cum = list(np.arange(len(mass))/len(mass))
cumdimH = list(np.arange(len(dim_cellH))/len(dim_cellH))
cumdimL = list(np.arange(len(dim_cellL))/len(dim_cellL))
cumdim = list(np.arange(len(dim_cell))/len(dim_cell))
massL = list(massL)
mass = list(mass)
massH = list(massH)
dim_cellL = list(dim_cellL)
dim_cell = list(dim_cell)
dim_cellH = list(dim_cellH)

# adjust for plot
cumH.append(cumH[-1])
cumL.append(cumL[-1])
cum.append(cum[-1])
cumdimH.append(cumdimH[-1])
cumdimL.append(cumdimL[-1])
cumdim.append(cumdim[-1])
mass.append(3e-7)
massL.append(3e-7)
massH.append(3e-7)
dim_cell.append(2)
dim_cellL.append(2)
dim_cellH.append(2)

print('min midplane mass fiducial:', np.min(mass_midplane), 'min midplane mass high:', np.min(mass_midplaneH))
print('min midplane dim fiducial:', np.min(dim_midplane), 'min midplane dim high:', np.min(dim_midplaneH))

if include_mid == 'mid':
    mass_midplaneL = np.sort(mass_midplaneL)
    mass_midplane = np.sort(mass_midplane)
    mass_midplaneH = np.sort(mass_midplaneH)
    dim_midplaneL = np.sort(dim_midplaneL)
    dim_midplane = np.sort(dim_midplane)
    dim_midplaneH = np.sort(dim_midplaneH)
    cum_midplaneH = list(np.arange(len(mass_midplaneH))/len(mass_midplaneH))
    cum_midplaneL = list(np.arange(len(mass_midplaneL))/len(mass_midplaneL))
    cum_midplane = list(np.arange(len(mass_midplane))/len(mass_midplane))
    cum_midplanedimH = list(np.arange(len(dim_midplaneH))/len(dim_midplaneH))
    cum_midplanedimL = list(np.arange(len(dim_midplaneL))/len(dim_midplaneL))
    cum_midplanedim = list(np.arange(len(dim_midplane))/len(dim_midplane))
    mass_midplaneL = list(mass_midplaneL)
    mass_midplane = list(mass_midplane)
    mass_midplaneH = list(mass_midplaneH)
    dim_midplaneL = list(dim_midplaneL)
    dim_midplane = list(dim_midplane)
    dim_midplaneH = list(dim_midplaneH)
    cum_midplaneH.append(cumH[-1])
    cum_midplaneL.append(cumL[-1])
    cum_midplane.append(cum[-1])
    cum_midplanedimH.append(cumdimH[-1])
    cum_midplanedimL.append(cumdimL[-1])
    cum_midplanedim.append(cumdim[-1])
    mass_midplane.append(3e-7)
    mass_midplaneL.append(3e-7)
    mass_midplaneH.append(3e-7)
    dim_midplane.append(2)
    dim_midplaneL.append(2)
    dim_midplaneH.append(2)

#%% Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8))
ax1.plot(massL, cumL, color = 'C1', label = 'This work, Low res')
ax1.plot(mass, cum, color = 'yellowgreen', label = 'This work, Fid res')
ax1.plot(massH, cumH, color = 'darkviolet', label = 'This work, High res')
ax1.axvline(price24, color = 'forestgreen', linestyle = '--', label = 'Price24')
ax1.axvline(bonlu20, color = 'mediumorchid', linestyle = 'dashdot', label = 'BonnerotLu20')
ax1.axvline(Hu25, color = 'orangered', linestyle = 'dotted', label = 'Hu+25')
ax2.plot(dim_cellL, cumdimL, color = 'C1')# label = 'Low res')
ax2.plot(dim_cell, cumdim, color = 'yellowgreen')# label = 'Middle res')
ax2.plot(dim_cellH, cumdimH, color = 'darkviolet',)# label = 'High res')
ax2.axvline(sizeminRyu, color = 'k', linestyle = (0, (5, 10)), label = 'Ryu+23 min orb.plane')
ax2.axvline(sizemeanRyu, color = 'k', linestyle = '--', label = 'Ryu+23 gmean orb.plane')
ax2.axvline(sizehuang24, color = 'deepskyblue', linestyle = 'dashdot', label = 'Huang+24')

if include_mid == 'mid':
    ax1.plot(mass_midplaneL, cum_midplaneL, '--', color = 'C1')#, label = 'Low res')
    ax1.plot(mass_midplane, cum_midplane, '--', color = 'yellowgreen')#, label = 'Fid res')
    ax1.plot(mass_midplaneH, cum_midplaneH, '--', color = 'darkviolet')#, label = 'High res')
    ax2.plot(dim_midplaneL, cum_midplanedimL, '--', color = 'C1')# label = 'Low res')
    ax2.plot(dim_midplane, cum_midplanedim, '--', color = 'yellowgreen')# label = 'Middle res')
    ax2.plot(dim_midplaneH, cum_midplanedimH, '--', color = 'darkviolet',)# label = 'High res')

for ax in [ax1, ax2]:
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', width=1.2, length=7, labelsize=28)
    ax.tick_params(axis='both', which='minor', width=0.9, length=5)
    ax.legend(loc ='upper left', fontsize = 22)
    ax.set_ylim(0,1.1)
    ax.grid()
ax1.set_ylabel('CDF', fontsize = 30)
ax1.set_xlabel(r'Cell mass [$M_\odot$]', fontsize = 30)
ax2.set_xlabel(r'Cell size [$R_\odot$]', fontsize = 30)
ax1.set_xlim(5e-13, 3e-7)
ax2.set_xlim(4e-2, 2)
# plt.suptitle(r'Near pericenter: $R_0<X<25, \, |Y|<4$', fontsize = 20)
plt.tight_layout()
if save:
    plt.savefig(f'{abspath}/Figs/paper/compareHistToget{include_mid}.pdf', bbox_inches='tight')
plt.show()

# %%
