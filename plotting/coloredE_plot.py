abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(f'{abspath}')
import numpy as np
import matplotlib.pyplot as plt
# import colorcet
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import Utilities.prelude as prel
import src.orbits as orb

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

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
DeltaE = orb.energy_mb(Rstar, mstar, Mbh, G=1) # specific energy of the mb debris 
DeltaE_cgs = DeltaE * prel.en_converter/prel.Msol_cgs
a = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
ph_data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/photo_mean.txt')
tfbRph, Rph = ph_data[0], ph_data[3]
Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
Enden_norm_single = Ledd / (4 * np.pi * prel.c_cgs * (Rph*prel.Rsol_cgs)**2) # [erg/cm^3] 

#%%
## DECISIONS
##
save = True
res0 = 'LowRes'
res1 = '' #'', 'HiRes', 'DoubleRad', 'LowRes'
res2 = 'HiRes' 

#
## DATA
#

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'{abspath}/data/{folder}'
#Res0 data
datares0 = np.load(f'{path}{res0}/colormapE_Alice/coloredE_{res0}_radii.npy') #shape (3, len(tfb), len(radii))
tfb_datares0 = np.loadtxt(f'{path}{res0}/colormapE_Alice/coloredE_{res0}_days.txt')
snap_res0 = tfb_datares0[0]
tfb_res0 = tfb_datares0[1]
radiires0 = np.load(f'{path}{res0}/colormapE_Alice/radiiEn_{res0}.npy')
col_ieres0, col_orb_enres0, col_Radres0 = datares0[0], datares0[1], datares0[2]
# convert to cgs
col_ieres0 *= prel.en_converter/prel.Msol_cgs
col_orb_enres0 *= prel.en_converter/prel.Msol_cgs
col_Radres0 *= prel.en_den_converter
abs_col_orb_enres0 = np.abs(col_orb_enres0)

#%% Res1 data
datares1 = np.load(f'{path}{res1}/colormapE_Alice/coloredE_{res1}.npy') #shape (3, len(tfb), len(radii))
tfb_datares1 = np.loadtxt(f'{path}{res1}/colormapE_Alice/coloredE_{res1}_days.txt')
snap_res1 = tfb_datares1[0]
tfb_res1 = tfb_datares1[1]
radiires1 = np.load(f'{path}{res1}/colormapE_Alice/coloredE_{res1}_radii.npy')
col_ieres1, col_orb_enres1, col_Rad_res1, col_Rad_denres1 = datares1[0], datares1[1], datares1[2], datares1[3]
# convert to cgs
col_ieres1 *= prel.en_converter/prel.Msol_cgs
col_orb_enres1 *= prel.en_converter/prel.Msol_cgs
abs_col_orb_enres1 = np.abs(col_orb_enres1)
col_Rad_res1 *= prel.en_converter/prel.Msol_cgs
col_Rad_denres1 *= prel.en_den_converter
# Enden_norm = Enden_quasi_norm / (radiires1*prel.Rsol_cgs)**2
Enden_norm = np.transpose([Enden_norm_single]*len(radiires1))

#%% Res2 data
datares2 = np.load(f'{path}{res2}/colormapE_Alice/coloredE_{res2}_radii.npy')
tfb_datares2 = np.loadtxt(f'{path}{res2}/colormapE_Alice/coloredE_{res2}_days.txt')
snap_res2 = tfb_datares2[0]
tfb_res2 = tfb_datares2[1]
radiires2 = np.load(f'{path}{res2}/colormapE_Alice/radiiEn_{res2}.npy')
col_ieres2, col_orb_enres2, col_Radres2 = datares2[0], datares2[1], datares2[2]

# convert to cgs
col_ieres2 *= prel.en_converter/prel.Msol_cgs
col_orb_enres2 *= prel.en_converter/prel.Msol_cgs
col_Radres2 *= prel.en_den_converter
abs_col_orb_enres2 = np.abs(col_orb_enres2)

#%% Plot Res1 NORMALIZED
fig = plt.figure(figsize=(21, 7))
gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1], height_ratios=[0.2, 3], hspace=0.2, wspace = 0.3)
ax1 = fig.add_subplot(gs[1, 0])  # First plot
ax2 = fig.add_subplot(gs[1, 1])  # Second plot
ax3 = fig.add_subplot(gs[1, 2])  # Third plot

img = ax1.pcolormesh(radiires1/apo, tfb_res1, abs_col_orb_enres1/DeltaE_cgs, norm=colors.LogNorm(vmin=4e-1, vmax = 11), cmap = 'viridis')
cbar_ax = fig.add_subplot(gs[0, 0])  # Colorbar subplot below the first two
cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
#longer ticks to cb
cb.ax.tick_params(which='minor',length = 3)
cb.ax.tick_params(which='major',length = 6)
# ax1.set_title('Orbital energy', fontsize = 20)
cb.set_label(r'Specific orbital energy/$\Delta$E', fontsize = 18, labelpad = 3)
# cb.ax.xaxis.set_label_position('top')
cb.ax.xaxis.set_ticks_position('top')

img = ax2.pcolormesh(radiires1/apo, tfb_res1, col_ieres1/DeltaE_cgs,  norm=colors.LogNorm(vmin=1e-4, vmax = 1.1), cmap = 'viridis')
cbar_ax = fig.add_subplot(gs[0, 1])  # Colorbar subplot below the first two
cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
cb.ax.tick_params(which='minor',length = 3)
cb.ax.tick_params(which='major',length = 6)
# put the label on the top of the colorbar
cb.set_label(r'Specific internal energy/$\Delta$E', fontsize = 18, labelpad = 5)
# cb.ax.xaxis.set_label_position('top')
cb.ax.xaxis.set_ticks_position('top')

img = ax3.pcolormesh(radiires1/apo, tfb_res1, col_Rad_denres1/Enden_norm, norm=colors.LogNorm(vmin=4e-2, vmax= 5e3), cmap = 'viridis')
cbar_ax = fig.add_subplot(gs[0, 2])  # Colorbar subplot below the first two
cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
cb.ax.tick_params(which='minor',length = 3)
cb.ax.tick_params(which='major',length = 6)
cb.set_label(r'Radiation energy density/u$_{\rm Edd}$', fontsize = 18, labelpad = 5)
# cb.ax.xaxis.set_label_position('top')
cb.ax.xaxis.set_ticks_position('top')

for ax in [ax1, ax2, ax3]:
    ax.axvline(Rt/apo, linestyle ='dashed', c = 'k', linewidth = 1.2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$R [R_a$]', fontsize = 20)
    # Get the existing ticks on the x-axis
    original_ticks = ax.get_yticks()
    # Calculate midpoints between each pair of ticks
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    # Combine the original ticks and midpoints
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    # Set tick labels: empty labels for midpoints
    ax.set_yticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    ax.set_yticklabels(labels)
    ax.tick_params(axis='x', which='major', width = 1.5, length = 10, color = 'white')
    ax.tick_params(axis='x', which='minor', width = 1, length = 7, color = 'white')
    ax.tick_params(axis='y', which='both', width = 1.5, length = 7)
    ax.set_ylim(np.min(tfb_res1), np.max(tfb_res1))
ax1.set_ylabel(r't [t$_{fb}]$', fontsize = 20)
plt.tight_layout()
if save:
    plt.savefig(f'{abspath}/Figs/{folder}{res1}/coloredE_norm.pdf')
    plt.savefig(f'{abspath}/Figs/{folder}{res1}/coloredE_norm.png')

#%% Plot Res1
fig, ax = plt.subplots(1,3, figsize = (16,5))
img = ax[0].pcolormesh(radiires1/apo, tfb_res1, abs_col_orb_enres1, norm=colors.LogNorm(vmin=4e16, vmax = 1e18), cmap = 'viridis')
cb = fig.colorbar(img)
ax[0].set_title('Specific (absolute) orbital energy', fontsize = 20)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)

img = ax[1].pcolormesh(radiires1/apo, tfb_res1, col_ieres1,  norm=colors.LogNorm(vmin=7e12, vmax = 4e14), cmap = 'viridis')
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[1].set_title('Specific internal energy', fontsize = 20)

img = ax[2].pcolormesh(radiires1/apo, tfb_res1, col_Rad_denres1, norm=colors.LogNorm(vmin=1e4, vmax= 7e9), cmap = 'viridis')
cb = fig.colorbar(img)
ax[2].set_title('Radiation energy density', fontsize = 20)
cb.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 20, labelpad = 2)

for i in range(3):
    ax[i].axvline(Rt/apo, linestyle ='dashed', c = 'k', linewidth = 1.2)
    ax[i].set_xscale('log')
    ax[i].set_xlabel(r'$R [R_a$]', fontsize = 20)
    # Get the existing ticks on the x-axis
    original_ticks = ax[i].get_yticks()
    # Calculate midpoints between each pair of ticks
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    # Combine the original ticks and midpoints
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    # Set tick labels: empty labels for midpoints
    ax[i].set_yticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    ax[i].set_yticklabels(labels)
    ax[i].tick_params(axis='x', which='major', width=1, length=7)
    ax[i].tick_params(axis='x', which='minor', width=0.5, length=5)
    ax[i].tick_params(axis='y', which='both', width=1, length=5)
    ax[i].set_ylim(np.min(tfb_res1), np.max(tfb_res1))
ax[0].set_ylabel(r't [t$_{fb}]$', fontsize = 20)
plt.tight_layout()
if save:
    plt.savefig(f'{abspath}/Figs/{folder}{res1}/coloredE_radii.pdf')

#%% relative difference L and middle
rel_orb_en_absL = []
rel_ie_absL = []
rel_Rad_absL = []
median_rel_orbL = np.zeros(len(tfb_res0))
median_rel_ieL = np.zeros(len(tfb_res0))
median_rel_radL = np.zeros(len(tfb_res0))
for i,time in enumerate(tfb_res0):
    # find the comparable time in res1
    idx = np.argmin(np.abs(tfb_res1 - time))
    # compute the relative difference for orbital energy
    rel_orb_time = 2*np.abs((col_orb_enres1[idx] - col_orb_enres0[i]) / (col_orb_enres1[idx] + col_orb_enres0[i]))
    # exclude the points where one of the two sim han no value
    idx_zero_orbL = np.concatenate([np.concatenate(np.where(col_orb_enres1[idx]==0)), np.concatenate(np.where(col_orb_enres0[i]==0))])
    # set to 0 the error in that points
    rel_orb_time[idx_zero_orbL] = 0
    # Do the same for the internal and radation energy
    rel_ie_time = 2*np.abs((col_ieres1[idx] - col_ieres0[i]) / (col_ieres1[idx] + col_ieres0[i]))
    rel_rad_time = 2*np.abs((col_Rad_denres1[idx] - col_Radres0[i]) / (col_Rad_denres1[idx] + col_Radres0[i]))
    idx_zero_ieL = np.concatenate([np.concatenate(np.where(col_ieres1[idx]==0)), np.concatenate(np.where(col_ieres0[i]==0))])
    idx_zero_RadL = np.concatenate([np.concatenate(np.where(col_Rad_denres1[idx]==0)), np.concatenate(np.where(col_Radres0[i]==0))])
    rel_ie_time[idx_zero_ieL] = 0
    rel_rad_time[idx_zero_RadL] = 0
    # Save the values
    rel_ie_absL.append(rel_ie_time)
    rel_orb_en_absL.append(rel_orb_time)
    rel_Rad_absL.append(rel_rad_time)
    median_rel_orbL[i] = np.median(rel_orb_time)
    median_rel_ieL[i] = np.median(rel_ie_time)
    median_rel_radL[i] = np.median(rel_rad_time)

# compute the median value of all the data in the matrixes rel_ie_absL, rel_orb_en_absL, rel_Rad_absL 
median_orbL = np.median(rel_orb_en_absL, axis = 1)
median_ieL = np.median(rel_ie_absL, axis = 1)
median_radL = np.median(rel_Rad_absL, axis = 1)
idx30 = np.argmin(np.abs(tfb_res0 - 0.3))
median_orbL30 = np.median(rel_orb_en_absL[idx30:], axis = 1)
median_ieL30 = np.median(rel_ie_absL[idx30:], axis = 1)
median_radL30 = np.median(rel_Rad_absL[idx30:], axis = 1)

print(f'The median relative difference of fiducial from L are')
print(f'-orbital energy: {np.round(np.median(rel_orb_en_absL),2)}, \n-specific internal energy: {np.round(np.median(rel_ie_absL),2)}, \n-radiation energy density: {np.round(np.median(rel_Rad_absL),2)}')
print(f'\n For times after 0.30 are: \n-orbital energy: {np.round(np.median(median_orbL30),2)}, \n-specific internal energy: {np.round(np.median(median_ieL30),2)}, \n-radiation energy density: {np.round(np.median(median_radL30),2)}')

# relative difference H and middle
rel_ie_absH = []
rel_orb_en_absH = []
rel_Rad_absH = []
median_rel_orbH = np.zeros(len(tfb_res2))
median_rel_ieH = np.zeros(len(tfb_res2))
median_rel_radH = np.zeros(len(tfb_res2))
for i in range(len(tfb_res2)):
    # find the comparable time in res1
    time = tfb_res2[i]
    idx = np.argmin(np.abs(tfb_res1 - time))
    # compute the relative difference for orbital energy
    rel_orb_time = 2*np.abs((col_orb_enres1[idx] - col_orb_enres2[i]) / (col_orb_enres1[idx] + col_orb_enres2[i]))
    # exclude the points where one of the two sim han no value
    idx_zero_orbH = np.concatenate([np.concatenate(np.where(col_orb_enres1[idx]==0)), np.concatenate(np.where(col_orb_enres2[i]==0))])
    # set to 0 the error in that points
    rel_orb_time[idx_zero_orbH] = 0
    # Do the same for the internal and radation energy
    rel_ie_time = 2*np.abs((col_ieres1[idx] - col_ieres2[i]) / (col_ieres1[idx] + col_ieres2[i]))
    rel_rad_time = 2*np.abs((col_Rad_denres1[idx] - col_Radres2[i]) / (col_Rad_denres1[idx] + col_Radres2[i]))
    idx_zero_ieH = np.concatenate([np.concatenate(np.where(col_ieres1[idx]==0)), np.concatenate(np.where(col_ieres2[i]==0))])
    idx_zero_RadH = np.concatenate([np.concatenate(np.where(col_Rad_denres1[idx]==0)), np.concatenate(np.where(col_Radres2[i]==0))])
    rel_ie_time[idx_zero_ieH] = 0
    rel_rad_time[idx_zero_RadH] = 0
    # Save the values
    rel_ie_absH.append(rel_ie_time)
    rel_orb_en_absH.append(rel_orb_time)
    rel_Rad_absH.append(rel_rad_time)
    median_rel_orbH[i] = np.median(rel_orb_time)
    median_rel_ieH[i] = np.median(rel_ie_time)
    median_rel_radH[i] = np.median(rel_rad_time)

# compute the median value of all the data in the matrixes rel_ie_absH, rel_orb_en_absH, rel_Rad_absH
median_orbH = np.median(rel_orb_en_absH, axis = 1)
median_ieH = np.median(rel_ie_absH, axis = 1)
median_radH = np.median(rel_Rad_absH, axis = 1)
idx30H = np.argmin(np.abs(tfb_res2 - 0.30))
median_orbH30 = np.median(rel_orb_en_absH[idx30H:], axis = 1)
median_ieH30 = np.median(rel_ie_absH[idx30H:], axis = 1)
median_radH30 = np.median(rel_Rad_absH[idx30H:], axis = 1)
print(f'\nThe median relative difference of fiducial from H are')
print(f'-orbital energy: {np.round(np.median(median_orbH),2)}, \n-specific internal energy: {np.round(np.median(median_ieH),2)}, \n-radiation energy density: {np.round(np.median(median_radH),2)}')
print(f'\n For times after 0.30 are: \n -orbital energy: {np.round(np.median(median_orbH30),2)}, \n-specific internal energy: {np.round(np.median(median_ieH30),2)}, \n-radiation energy density: {np.round(np.median(median_radH30),2)}')

#%%
cmap = plt.cm.inferno
norm_orb_en = colors.LogNorm(vmin=2e-3, vmax=6e-1)#np.percentile(rel_orb_en_forlog[rel_orb_en_forlog!=1], 5), vmax=np.percentile(rel_orb_en_forlog[rel_orb_en_forlog!=1], 95))
norm_ie = colors.LogNorm(vmin=2e-2, vmax=6e-1)#np.percentile(rel_ie_forlog[rel_ie_forlog!=1], 5), vmax=np.percentile(rel_ie_forlog[rel_ie_forlog!=1], 95))
norm_Rad = colors.LogNorm(vmin=0.04, vmax=1.5)#np.percentile(rel_Rad_forlog[rel_Rad_forlog!=1], 5), vmax=np.percentile(rel_Rad_forlog[rel_Rad_forlog!=1], 95))

fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(4, 3, width_ratios=[1,1,1], height_ratios=[3, 0.2, 3, 0.2], hspace=0.4, wspace = 0.3)
ax1 = fig.add_subplot(gs[0, 0])  # First plot
ax2 = fig.add_subplot(gs[0, 1])  # Second plot
ax3 = fig.add_subplot(gs[0, 2])  # Third plot
ax4 = fig.add_subplot(gs[2, 0])  # First plot
ax5 = fig.add_subplot(gs[2, 1])  # Second plot
ax6 = fig.add_subplot(gs[2, 2])  # Third plot

img = ax1.pcolormesh(radiires0/apo, tfb_res0, rel_orb_en_absL, cmap=cmap, norm=norm_orb_en)
ax1.text(0.3, 0.9*np.max(tfb_res0), 'Low and Fid', fontsize = 20, color = 'white')
img = ax2.pcolormesh(radiires2/apo, tfb_res2, rel_orb_en_absH, cmap=cmap, norm=norm_orb_en)
ax2.text(0.3, 0.9*np.max(tfb_res2), 'High and Fid', fontsize = 20, color = 'white')
cbar_ax = fig.add_subplot(gs[1, 0:2])  # Colorbar subplot below the first two
cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
cb.ax.tick_params(labelsize=20)
cb.set_label(r'$\Delta_{\rm rel}$ specific (absolute) orbital energy', fontsize = 16, labelpad = 3)

ax4.pcolormesh(radiires0/apo, tfb_res0, rel_ie_absL, cmap=cmap, norm=norm_ie)
img = ax5.pcolormesh(radiires2/apo, tfb_res2, rel_ie_absH, cmap=cmap, norm=norm_ie)
cbar_ax = fig.add_subplot(gs[3, 0:2])
cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
cb.ax.tick_params(labelsize=20)
cb.set_label(r'$\Delta_{\rm rel}$ specific internal energy', fontsize = 16, labelpad = 3)

ax3.plot(tfb_res0, median_rel_orbL, '--', label = r'$\Delta_{rad}$ Low and Fid', c = 'darkorange')
ax3.plot(tfb_res2, median_rel_orbH, label = r'$\Delta_{rad}$ Fid and High', c = 'darkviolet')
# plt.ylabel('Median relative difference', fontsize = 25)
ax3.set_yscale('log')
ax3.set_ylim(1e-2, 1)
ax3.legend(fontsize = 15)
ax3.grid()

ax6.plot(tfb_res0, median_rel_ieL, '--', label = r'$\Delta_{rad}$ Low and Fid', c = 'darkorange')
ax6.plot(tfb_res2, median_rel_ieH, label = r'$\Delta_{rad}$ Fid and High', c = 'darkviolet')
# plt.ylabel('Median relative difference', fontsize = 25)
ax6.set_yscale('log')
ax6.set_ylim(3e-2, 2)
ax6.legend(fontsize = 15)
ax6.grid()

# Get the existing ticks on the x-axis
original_ticks = ax1.get_yticks()
# Calculate midpoints between each pair of ticks
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
# Combine the original ticks and midpoints
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    if ax in [ax1, ax2, ax4, ax5]:
        ax.axvline(Rt/apo, linestyle ='dashed', c = 'k', linewidth = 1.2)
        ax.set_xscale('log')
        if ax in [ax1, ax4]:
            ax.set_ylabel(r't [t$_{\rm fb}]$', fontsize = 24)
        # Set tick labels: empty labels for midpoints
        ax.set_yticks(new_ticks)
        # ax.set_yticklabels(labels)
        ax.tick_params(axis='x', which='major', width=1.2, length=7, color = 'white')
        ax.tick_params(axis='x', which='minor', width=1, length=5, color = 'white')
        ax.tick_params(axis='y', which='both', width=1.2, length=6, color = 'k')
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        if i in [ax1, ax2]:
            ax.set_ylim(np.min(tfb_res0), np.max(tfb_res0))
        else:
            ax.set_ylim(np.min(tfb_res2), np.max(tfb_res2))
        if ax in [ax4, ax5, ax6]:
            ax.set_xlabel(r'$R [R_{\rm a}]$', fontsize = 22)
        if ax == ax6:
            ax.set_xlabel(r'$t [t_{\rm fb}]$', fontsize = 22)

plt.tick_params(axis = 'both', which = 'both', direction='in', labelsize = 20)
plt.tight_layout()
if save:
    plt.savefig(f'{abspath}/Figs/multiple/orbE_IE_relative_diff.png')
    plt.savefig(f'{abspath}/Figs/multiple/orbE_IE_relative_diff.pdf')
plt.show()

#%%
# plt.figure(figsize = (7,5))
# plt.plot(tfb_res0, median_orbL, label = r'$\Delta_{OE}$ Low-Middle', c = 'navy')
# plt.plot(tfb_res2, median_orbH, '--', label = r'$\Delta_{OE}$ Middle-High', c = 'navy')
# plt.plot(tfb_res0, median_ieL, label = r'$\Delta_{IE}$ Low-Middle', c = 'coral')
# plt.plot(tfb_res2, median_ieH, '--', label = r'$\Delta_{IE}$ Middle-High', c = 'coral')
# plt.ylabel('Median relative difference', fontsize = 25)
# plt.xlabel(r't [t$_{fb}]$', fontsize = 25)
# plt.yscale('log')
# plt.legend(fontsize = 20)
# plt.tight_layout()
# if save:
#     plt.savefig(f'{abspath}/Figs/multiple/orbE_IE_diffTime.pdf')
# plt.show()

#%% Relative errors for the radiation energy density
import matplotlib.patheffects as pe
error_ph0 = np.zeros(len(tfb_res0))
for i,t in enumerate(tfb_res0):
    idx_r = np.argmin(np.abs(radiires0 - Rph[i]))
    error_ph0[i] = rel_Rad_absL[i][idx_r]
error_ph2 = np.zeros(len(tfb_res2))
for i,t in enumerate(tfb_res2):
    idx_r = np.argmin(np.abs(radiires2 - Rph[i]))
    error_ph2[i] = rel_Rad_absH[i][idx_r]
print(f'The median relative difference of the photosphere from the fiducial are')
print(f'-Low-Middle: {np.round(np.median(error_ph0),2)}, \n-Middle-High: {np.round(np.median(error_ph2),2)}')

fig, ax = plt.subplots(2,1, figsize = (7,9))
img = ax[0].pcolormesh(radiires0/apo, tfb_res0, rel_Rad_absL, cmap = 'inferno', norm=norm_Rad)
cb = fig.colorbar(img)
cb.set_label(r'$\Delta_{\rm rel}$ Low-Fid', fontsize = 24)

img = ax[1].pcolormesh(radiires2/apo, tfb_res2, rel_Rad_absH, cmap = 'inferno', norm=norm_Rad)
cb = fig.colorbar(img)
cb.set_label(r'$\Delta_{\rm rel}$ Fid-High', fontsize = 24)
ax[1].set_xlim(np.min(radiires2/apo), np.max(radiires2/apo))

# Get the existing ticks on the x-axis
original_ticks = ax[0].get_yticks()
# Calculate midpoints between each pair of ticks
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
# Combine the original ticks and midpoints
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
# labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
for i in range(2):
    ax[i].set_xlim(np.min(radiires0/apo), np.max(radiires0/apo))
    ax[i].axvline(Rt/apo, linestyle ='--', c = 'k', linewidth = 2)
    ax[i].plot(Rph[5:]/apo, tfbRph[5:], c = 'yellowgreen', path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()], linestyle = 'solid', label = 'Photosphere', linewidth = 2)
    ax[i].set_xscale('log')
    ax[i].set_ylabel(r't [t$_{\rm fb}]$', fontsize = 25)
    # Set tick labels: empty labels for midpoints
    ax[i].set_yticks(new_ticks)
    # ax[i].set_yticklabels(labels)
    ax[i].tick_params(axis='x', which='major', width=1.5, length=7, color = 'white')
    ax[i].tick_params(axis='x', which='minor', width=1.1, length=5, color = 'white')
    ax[i].tick_params(axis='y', which='both', width=1.5, length=7, color = 'white')
ax[0].set_ylim(np.min(tfb_res0), np.max(tfb_res0))
ax[1].set_ylim(np.min(tfb_res2), np.max(tfb_res2))
ax[1].set_xlabel(r'$R [R_{\rm a}]$', fontsize = 25)

plt.tick_params(axis = 'both', which = 'both', direction='in')
plt.tight_layout()
if save:
    plt.savefig(f'{abspath}/Figs/multiple/Rad_relative_diff.pdf')
    plt.savefig(f'{abspath}/Figs/multiple/Rad_relative_diff.png')
plt.show()

#%%
# Raderror = []
# tRaderror = []
# for i,tph in enumerate(tfbRph):
#     photo = Rph[i]
#     if photo < np.min(radiires0) or photo > np.max(radiires0):
#         continue
#     idxt = np.argmin(np.abs(tfb_res0 - tph))
#     idxr = np.argmin(np.abs(radiires0 - photo))
#     tRaderror.append(tph)
#     Raderror.append(rel_Rad_absL[idxt][idxr])
# Raderror = np.array(Raderror)
# tRaderror = np.array(tRaderror)

plt.figure(figsize = (7,5))
plt.plot(tfb_res0, median_rel_radL, '--', label = r'$\Delta_{rad}$ Low-Middle', c = 'darkorange')
plt.plot(tfb_res2, median_rel_radH, label = r'$\Delta_{rad}$ Middle-High', c = 'darkviolet')
# plt.plot(tRaderror, Raderror, 'o', label = 'Photosphere', c = 'k')
plt.ylabel('Median relative difference', fontsize = 25)
plt.xlabel(r't [t$_{fb}]$', fontsize = 25)
plt.yscale('log')
plt.ylim(9e-2, 1.5)
plt.legend(fontsize = 20)
plt.tight_layout()
plt.grid()
if save:
    plt.savefig(f'{abspath}/Figs/multiple/rad_diffTime.pdf')
plt.show()

#%% Free streaming luminosity. See what happpen at large radii, where L=4pi R^2 c Erad
free_streaming = False
if free_streaming:
    indices = [np.argmin(np.abs(tfb_res1-0.5)), np.argmin(np.abs(tfb_res1-0.7)), np.argmin(np.abs(tfb_res1-0.86))]
    heigth_text = np.array([3e40, 1e43, 4e44])
    colors_indices = ['navy', 'royalblue', 'deepskyblue']
    col_Lum0 = np.load(f'{path}LowRes/colormapE_alice/coloredE_LowRes_radiimixedLum.npy') 
    radiiLum0 = np.load(f'{path}LowRes/colormapE_alice/radiiEn_LowResLum.npy')
    col_Lum0 = col_Lum0 * prel.en_den_converter
    col_Lum1 = np.load(f'{path}/colormapE_alice/coloredE__radiimixedLum.npy') 
    radiiLum1 = np.load(f'{path}/colormapE_alice/radiiEn_Lum.npy')
    col_Lum1 = col_Lum1 * prel.en_den_converter
    col_Lum2 = np.load(f'{path}HiRes/colormapE_alice/coloredE_HiRes_radiimixedLum.npy') 
    radiiLum2 = np.load(f'{path}HiRes/colormapE_alice/radiiEn_HiResLum.npy')
    col_Lum2 = col_Lum2 * prel.en_den_converter

    radiiLum0 = np.repeat([radiiLum0],len(col_Lum0), axis = 0)
    radiiLum1 = np.repeat([radiiLum1],len(col_Lum1), axis = 0)
    radiiLum2 = np.repeat([radiiLum2],len(col_Lum2), axis = 0)

    Lum0_cgs = col_Lum0  * prel.c * 4 * np.pi * (radiiLum0*prel.Rsol_cgs)**2 
    Lum_cgs = col_Lum1  * prel.c * 4 * np.pi * (radiiLum1*prel.Rsol_cgs)**2 
    Lum2_cgs = col_Lum2 * prel.c * 4 * np.pi * (radiiLum2*prel.Rsol_cgs)**2 

    Lum_difference0 = []
    for i in range(len(tfb_res0)):
        # find the comparable time in res1
        time = tfb_res0[i]
        idx = np.argmin(np.abs(tfb_res1 - time))
        denom0 = (Lum_cgs[idx] + Lum0_cgs[i])/2
        Lum_difference0.append(np.abs(Lum_cgs[idx]-Lum0_cgs[i])/denom0)

    Lum_difference2 = []
    for i in range(len(tfb_res2)):
        # find the comparable time in res1
        time = tfb_res2[i]
        idx = np.argmin(np.abs(tfb_res1 - time))
        denom2 = (Lum_cgs[idx] + Lum2_cgs[i])/2
        Lum_difference2.append(np.abs(Lum_cgs[idx]-Lum2_cgs[i])/denom2)

    img, ax = plt.subplots(1,2, figsize = (20,7))
    for i,idx in enumerate(indices):
        where_zero0 = np.concatenate(np.where(Lum0_cgs[idx]<1e-18))
        where_zero = np.concatenate(np.where(Lum_cgs[idx]<1e-18))
        where_zero2 = np.concatenate(np.where(Lum2_cgs[idx]<1e-18))
        where_zero0 = np.concatenate([where_zero0, where_zero])
        where_zero2 = np.concatenate([where_zero2, where_zero])
        Lum0_cgs_toplot= np.delete(Lum0_cgs[idx], where_zero0)
        Lum_cgs_toplot= np.delete(Lum_cgs[idx], where_zero)
        Lum2_cgs_toplot = np.delete(Lum2_cgs[idx], where_zero2)

        radiiLum0_toplot = np.delete(radiiLum0[idx], where_zero0)
        radiiLum1_toplot = np.delete(radiiLum1[idx], where_zero)
        radiiLum2_toplot = np.delete(radiiLum2[idx], where_zero2)

        Lum_difference0_toplot =  np.delete(Lum_difference0[idx], where_zero0) #Lum_difference0[idx]#
        Lum_difference2_toplot = np.delete(Lum_difference2[idx], where_zero2) #Lum_difference2[idx]

        ax[0].text(np.min(radiiLum0_toplot)/apo, heigth_text[i], f't = {np.round(tfb_res1[indices[i]],2)}' +  r'$t_{fb}$', fontsize = 16)
        if i == 0:
            ax[0].plot(radiiLum0_toplot/apo, Lum0_cgs_toplot, linestyle = 'dotted', c = colors_indices[i], label = f'{res0} res')#t/tfb = {np.round(tfb_Low,2)}')
            ax[0].plot(radiiLum1_toplot/apo, Lum_cgs_toplot, c = colors_indices[i], label = f'{res1} res')#t/tfb = {np.round(tfb_Low,2)}')
            ax[0].plot(radiiLum2_toplot/apo, Lum2_cgs_toplot, '--', c = colors_indices[i], label = f'{res2} res')#t/tfb = {np.round(tfb_res2,2)}')
            ax[1].plot(radiiLum0_toplot/apo, Lum_difference0_toplot,  c = colors_indices[i], label = f'Low-middle')
            ax[1].plot(radiiLum2_toplot/apo, Lum_difference2_toplot, c = colors_indices[i], label = f'High-middle')
        else:   
            ax[0].plot(radiiLum0_toplot/apo, Lum0_cgs_toplot, linestyle = 'dotted', c = colors_indices[i])
            ax[0].plot(radiiLum1_toplot/apo, Lum_cgs_toplot, c = colors_indices[i])
            ax[0].plot(radiiLum2_toplot/apo, Lum2_cgs_toplot, '--', c = colors_indices[i])
            ax[1].plot(radiiLum0_toplot/apo, Lum_difference0_toplot, c = colors_indices[i])
            ax[1].plot(radiiLum2_toplot/apo, Lum_difference2_toplot, '--', c = colors_indices[i])

        mean_error0 =  np.round(np.mean(Lum_difference0_toplot[-15:-1]),2)
        mean_error2 =  np.round(np.mean(Lum_difference2_toplot[-15:-1]),2)
        print(f'Mean relative error for t/tfb = {tfb_res1[indices[i]]} is {mean_error0} for the low-middle and {mean_error2} for the high-middle')

    ax[0].set_ylim(1e40, 5e45)
    ax[1].set_ylim(0.1, 1.8)

    ax[0].tick_params(axis='both', which='major', labelsize=25)
    ax[1].tick_params(axis='both', which='major', labelsize=25)
    ax[0].tick_params(axis='both', which='minor', size=4)
    ax[1].tick_params(axis='both', which='minor', size=4)
    ax[0].tick_params(axis='both', which='major', size=6)
    ax[1].tick_params(axis='both', which='major', size=6)
    ax[0].set_xlabel(r'R/R$_a$', fontsize = 28)
    ax[1].set_xlabel(r'$R/R_a$', fontsize = 28)
    ax[0].set_ylabel(r'Luminosity [erg/s]', fontsize = 25)
    ax[1].set_ylabel(r'Relative difference', fontsize = 25, labelpad = 1)
    ax[0].loglog()
    ax[0].legend(fontsize = 25)
    ax[0].grid()
    ax[1].grid()
    ax[1].loglog()
    plt.subplots_adjust(wspace=4)
    plt.tight_layout()
    if save:
        plt.savefig(f'{abspath}/Figs/multiple/Luminositymixed.pdf')
    plt.show()

    # for i,idx in enumerate(indices):
    #     if i ==2:
    #         continue
    #     after = Lum2_cgs[indices[i+1]]
    #     before = Lum2_cgs[indices[i]]
    #     mean_lastafter = np.mean(after[-10:-1])
    #     mean_lastbefore = np.mean(before[-10:-1])
    #     print(tfb_res1[indices[i+1]], tfb_res1[indices[i]])
        # print(f'The relative difference of the last point of the high res line is {np.round((mean_lastafter-mean_lastbefore),2)}')


