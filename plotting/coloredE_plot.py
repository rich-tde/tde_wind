import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import matplotlib.pyplot as plt
import colorcet
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.gaussian_smoothing import time_average

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
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

#
## DECISIONS
##
save = False
xaxis = 'angles'
if xaxis == 'angles':
    apo = 1
#
## DATA
#

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'/Users/paolamartire/shocks/data/{folder}/colormapE_Alice'
# Low data
dataLow = np.load(f'{path}/coloredE_Low_{xaxis}weightE.npy') #shape (3, len(tfb), len(radii))
tfb_dataLow = np.loadtxt(f'{path}/coloredE_Low_days.txt')
snap_Low = tfb_dataLow[0]
tfb_Low = tfb_dataLow[1]
radiiLow = np.load(f'{path}/{xaxis}En_Low.npy')
col_ie, col_orb_en, col_Rad, col_Rad_nofluff = dataLow[0], dataLow[1], dataLow[2], dataLow[3]
# convert to cgs
col_ie *= prel.en_converter/prel.Msol_to_g
col_orb_en *= prel.en_converter/prel.Msol_to_g
col_Rad *= prel.en_den_converter
col_Rad_nofluff *= prel.en_den_converter
# Average over time
# col_ie, col_orb_en, col_Rad = col_ie.T, col_orb_en.T, col_Rad.T
# for i in range(len(col_ie)):
#     col_ie[i] = time_average(tfb_Low, col_ie[i])
#     col_orb_en[i] = time_average(tfb_Low, col_orb_en[i])
#     col_Rad[i] = time_average(tfb_Low, col_Rad[i])
# col_ie, col_orb_en, col_Rad = col_ie.T, col_orb_en.T, col_Rad.T
abs_col_orb_en = np.abs(col_orb_en)

# Middle data
dataMiddle = np.load(f'{path}/coloredE_HiRes_{xaxis}weightE.npy')
tfb_dataMiddle = np.loadtxt(f'{path}/coloredE_HiRes_days.txt')
snap_Middle = tfb_dataMiddle[0]
tfb_Middle = tfb_dataMiddle[1]
radiiMiddle = np.load(f'{path}/{xaxis}En_HiRes.npy')
col_ieMiddle, col_orb_enMiddle, col_RadMiddle, col_RadMiddle_nofluff = dataMiddle[0], dataMiddle[1], dataMiddle[2], dataMiddle[3]
# convert to cgs
col_ieMiddle *= prel.en_converter/prel.Msol_to_g
col_orb_enMiddle *= prel.en_converter/prel.Msol_to_g
col_RadMiddle *= prel.en_den_converter
col_RadMiddle_nofluff *= prel.en_den_converter
abs_col_orb_enMiddle = np.abs(col_orb_enMiddle)

# Consider Low data only up to the time of the Middle data
n_Middle = len(col_ieMiddle)
snap_Low = snap_Low[:n_Middle]
tfb_Low = tfb_Low[:n_Middle]
col_ie = col_ie[:n_Middle]
col_orb_en = col_orb_en[:n_Middle]
abs_col_orb_en = abs_col_orb_en[:n_Middle]
col_Rad = col_Rad[:n_Middle]
col_Rad_nofluff = col_Rad_nofluff[:n_Middle]
#%%
# PLOT
##
# check the difference in time
# plt.scatter(snap_Middle,tfb_Low-tfb_Middle, color='k', label = 'time')
# # plt.plot(snap_Low-snap_Middle, color='r', label = 'snap')
# plt.scatter(snap_Middle[37], tfb_Low[37]-tfb_Middle[37], c='dodgerblue')
# plt.scatter(snap_Middle[119], tfb_Low[119]-tfb_Middle[119], c='orange')
# plt.scatter(snap_Middle[129], tfb_Low[129]-tfb_Middle[129], c='orchid')
# plt.xlabel('Snapshot')
# plt.ylabel(r'$t_L-t_H$')
# plt.legend(fontsize = 20)

#%%
p40_orb_ensix = np.percentile(abs_col_orb_en, 40)
p95_orb_ensix = 1e18#np.percentile(abs_col_orb_en, 99)
p40_iesix = np.percentile(col_ie, 40)
p95_iesix = np.percentile(col_ie, 95)
p40_Radsix = np.percentile(col_Rad, 40)
p95_Radsix = np.percentile(col_Rad, 95)

cmap = plt.cm.viridis
norm_orb_ensix = colors.LogNorm(vmin=p40_orb_ensix, vmax=p95_orb_ensix)
norm_iesix = colors.LogNorm(vmin=p40_iesix, vmax=p95_iesix)
norm_Radsix = colors.LogNorm(vmin=p40_Radsix, vmax=p95_Radsix)

fig, ax = plt.subplots(2,3, figsize = (14,8))
# Low
img = ax[0][0].pcolormesh(radiiLow/apo, tfb_Low, abs_col_orb_en, norm=norm_orb_ensix, cmap = cmap)
cb = fig.colorbar(img)
ax[0][0].set_title('Specific (absolute) orbital energy', fontsize = 20)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[0][0].text(np.min(radiiMiddle/apo), 0.1,'Low res', fontsize = 20)

img = ax[0][1].pcolormesh(radiiLow/apo, tfb_Low, col_ie,  norm=norm_iesix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[0][1].set_title('Specific internal energy', fontsize = 20)

img = ax[0][2].pcolormesh(radiiLow/apo, tfb_Low, col_Rad, norm=norm_Radsix, cmap = cmap)
cb = fig.colorbar(img)
ax[0][2].set_title('Radiation energy density', fontsize = 20)
cb.set_label(r'Energy density [erg/cm$^3$]', fontsize = 20, labelpad = 2)

# Middle
img = ax[1][0].pcolormesh(radiiMiddle/apo, tfb_Middle, abs_col_orb_enMiddle, norm=norm_orb_ensix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[1][0].text(np.min(radiiMiddle/apo), 0.1,'High res', fontsize = 20)

img = ax[1][1].pcolormesh(radiiMiddle/apo, tfb_Middle, col_ieMiddle, norm=norm_iesix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)

img = ax[1][2].pcolormesh(radiiMiddle/apo, tfb_Middle, col_RadMiddle, norm=norm_Radsix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Energy density [erg/cm$^3$]', fontsize = 20, labelpad = 2)

for i in range(2):
    for j in range(3):
        ax[i][j].axhline(0.205, c = 'white', linewidth = 0.4)
        ax[i][j].axhline(0.52, c = 'white', linewidth = 0.4)
        # Grid for radii, to be matched with the cfr in slices.py
        if xaxis == 'radii':
            ax[i][j].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
            ax[i][j].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
            ax[i][j].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
            ax[i][j].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
            ax[i][j].axvline(0.1, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(0.3, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(0.5, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(1, c = 'white', linewidth = 0.4)
            ax[i][j].set_xscale('log')
        
        elif xaxis == 'angles':
            ax[i][j].axvline(-2.5, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(-1, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(2.5, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(1, c = 'white', linewidth = 0.4)

        if i == 1:
            if xaxis == 'radii':
                ax[i][j].set_xlabel(r'$R/R_a$', fontsize = 20)
                ax[i][j].set_xlabel(r'$R/R_a$', fontsize = 20)
                ax[i][j].set_xlabel(r'$R/R_a$', fontsize = 20)
            elif xaxis == 'angles':
                ax[i][j].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
                ax[i][j].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
                ax[i][j].set_xlabel(r'$\theta$ [rad]', fontsize = 20)

# Layout
ax[0][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
ax[1][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', direction='in')

plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredE_{xaxis}.png')
plt.show()

# %% Plot (absolute) differences. They start from the same point
diff_ie = col_ie - col_ieMiddle
diff_orb_en = col_orb_en - col_orb_enMiddle
diff_Rad = col_Rad - col_RadMiddle
diff_ie_abs = np.abs(diff_ie)
diff_orb_en_abs = np.abs(diff_orb_en)
diff_Rad_abs = np.abs(diff_Rad)
# set to zero where the value of one of them is zero
diff_ie_abs[col_ie==0 ] = 0
diff_ie_abs[col_ieMiddle==0] = 0
diff_orb_en_abs[col_orb_en==0] = 0
diff_orb_en_abs[col_orb_enMiddle==0] = 0
diff_Rad_abs[col_Rad==0] = 0
diff_Rad_abs[col_RadMiddle==0] = 0
# pass to log
# difflog_ie = np.log10(diff_ie_abs)
# difflog_orb_en = np.log10(diff_orb_en_abs)
# difflog_Rad = np.log10(diff_Rad_abs)
# difflog_ie[np.isnan(difflog_ie)] = 0
# difflog_orb_en[np.isnan(difflog_orb_en)] = 0
# difflog_Rad[np.isnan(difflog_Rad)] = 0

fig, ax = plt.subplots(1,3, figsize = (18,5))

img = ax[0].pcolormesh(radiiMiddle/apo, tfb_Middle, diff_orb_en_abs, norm=colors.LogNorm(vmin=1e15, vmax=1e17),
                     cmap = 'bwr')#, vmin = 15, vmax = 17)
cb = fig.colorbar(img)
cb.set_label(r'$|\Delta|$', fontsize = 20)
ax[0].set_title('Specific (absolute) orbital energy', fontsize = 14)

img = ax[1].pcolormesh(radiiMiddle/apo, tfb_Middle, diff_ie_abs, norm=colors.LogNorm(vmin=4e11, vmax=4e13),
                     cmap = 'bwr')#, vmin = 11.5, vmax = 14)
cb = fig.colorbar(img)
cb.set_label(r'$|\Delta|$', fontsize = 20)
ax[1].set_title('Specific internal energy', fontsize = 14)

img = ax[2].pcolormesh(radiiMiddle/apo, tfb_Middle, diff_Rad_abs, norm=colors.LogNorm(vmin=1e4, vmax=4e7),
                     cmap = 'bwr')#, vmin = 3, vmax = 8.5)
cb = fig.colorbar(img)
ax[2].set_title('Radiation energy density', fontsize = 14)
cb.set_label(r'$|\Delta|$', fontsize = 20)#$\Delta|E_{rad}/$Vol$|$', fontsize = 14, labelpad = 5)

for i in range(3):
    ax[i].axhline(0.205, c = 'white', linewidth = 0.4)
    ax[i].axhline(0.52, c = 'white', linewidth = 0.4)
    ax[i].axhline(0.75, c = 'white', linewidth = 0.4)
    if xaxis == 'radii':
        ax[i].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        ax[i].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        ax[i].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        ax[i].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        ax[i].axvline(0.1, c = 'white', linewidth = 0.4)
        ax[i].axvline(0.3, c = 'white', linewidth = 0.4)
        ax[i].axvline(0.5, c = 'white', linewidth = 0.4)
        ax[i].axvline(1, c = 'white', linewidth = 0.4)
        ax[i].set_xscale('log')
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)

    elif xaxis == 'angles':
        ax[i].axvline(-2.5, c = 'white', linewidth = 0.4)
        ax[i].axvline(-1, c = 'white', linewidth = 0.4)
        ax[i].axvline(2.5, c = 'white', linewidth = 0.4)
        ax[i].axvline(1, c = 'white', linewidth = 0.4)
        ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
        ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
        ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 25)
ax[0].set_xlabel(r'$R/R_a$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', direction='in')
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredE_diff_{xaxis}.png')
plt.show()

######################
# %% Relative differences
denominator_orb_en = (col_orb_en + col_orb_enMiddle)/2
denominator_ie = (col_ie + col_ieMiddle)/2
denominator_Rad = (col_Rad + col_RadMiddle)/2
rel_orb_en = np.abs(diff_orb_en / denominator_orb_en)
rel_ie = np.abs(diff_ie / denominator_ie)
rel_Rad = np.abs(diff_Rad / denominator_Rad)

rel_orb_en[np.isnan(rel_orb_en)] = 0 #nothing
rel_ie[np.isnan(rel_ie)] = 0
rel_Rad[np.isnan(rel_Rad)] = 0


#%%
median_orb_en = np.median(rel_orb_en)
median_ie = np.median(rel_ie)
median_Rad = np.median(rel_Rad)

print(f'Median relative difference in Orbital energy: {median_orb_en}')
print(f'Median relative difference in IE: {median_ie}')
print(f'Median relative difference in Radiation energy density: {median_Rad}')

rel_orb_en_forlog = np.copy(rel_orb_en) #you need it for the Log
rel_ie_forlog = np.copy(rel_ie)
rel_Rad_forlog = np.copy(rel_Rad)
rel_orb_en_forlog[rel_orb_en_forlog<=0] = 1 #you need it for the Log
rel_ie_forlog[rel_ie_forlog<=0] =1
rel_Rad_forlog[rel_Rad_forlog<=0] = 1

cmap = plt.cm.inferno
norm_orb_en = colors.LogNorm(vmin=np.percentile(rel_orb_en_forlog[rel_orb_en_forlog!=1], 5), vmax=np.percentile(rel_orb_en_forlog[rel_orb_en_forlog!=1], 95))
norm_ie = colors.LogNorm(vmin=np.percentile(rel_ie_forlog[rel_ie_forlog!=1], 5), vmax=np.percentile(rel_ie_forlog[rel_ie_forlog!=1], 95))
norm_Rad = colors.LogNorm(vmin=np.percentile(rel_Rad_forlog[rel_Rad_forlog!=1], 5), vmax=np.percentile(rel_Rad_forlog[rel_Rad_forlog!=1], 95))

fig, ax = plt.subplots(1,3, figsize = (20,6))
img = ax[0].pcolormesh(radiiMiddle/apo, tfb_Middle, rel_orb_en_forlog, cmap=cmap, norm=norm_orb_en)#, vmin = np.min(rel_orel_orb_en_fologrb_en), vmax = 0.2)
cb = fig.colorbar(img)
# cb.ax.tick_params(labelsize=20)
ax[0].set_title('Specific (absolute) orbital energy', fontsize = 20)
cb.set_label('Relative difference', fontsize = 25)#Relative difference $|E_{orb}|$/Mass', fontsize = 14, labelpad = 5)

img = ax[1].pcolormesh(radiiMiddle/apo, tfb_Middle, rel_ie_forlog, cmap=cmap, norm=norm_ie)#, vmin = np.min(rel_ie_forlog), vmax = np.max(rel_ie_forlog))
cb = fig.colorbar(img)
cb.ax.tick_params(labelsize=20)
ax[1].set_title('Specific internal energy', fontsize = 20)
cb.set_label('Relative difference', fontsize = 25)#Relative difference $|$IE$|$/Mass', fontsize = 14, labelpad = 5)

img = ax[2].pcolormesh(radiiMiddle/apo, tfb_Middle, rel_Rad_forlog, cmap=cmap, norm=norm_Rad)#, vmin = np.min(rel_Rad_forlog), vmax = np.max(rel_Rad_forlog))
cb = fig.colorbar(img)
# cb.ax.tick_params(labelsize=2)
ax[2].set_title('Radiation energy density', fontsize = 20)
cb.set_label('Relative difference', fontsize = 25)#'Relative difference $|E_{rad}|$/Vol', fontsize = 14, labelpad = 5)

for i in range(3):
    ax[i].axhline(0.205, c = 'white', linewidth = 0.4)
    ax[i].axhline(0.52, c = 'white', linewidth = 0.4)
    ax[i].axhline(0.75, c = 'white', linewidth = 0.4)
    if xaxis == 'radii':
        ax[i].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        ax[i].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        ax[i].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        ax[i].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        ax[i].axvline(0.1, c = 'white', linewidth = 0.4)
        ax[i].axvline(0.3, c = 'white', linewidth = 0.4)
        ax[i].axvline(0.5, c = 'white', linewidth = 0.4)
        ax[i].axvline(1, c = 'white', linewidth = 0.4)
        ax[i].set_xscale('log')
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)

    elif xaxis == 'angles':
        if xaxis == 'angles':
            ax[i].axvline(-2.5, c = 'white', linewidth = 0.4)
            ax[i].axvline(-1, c = 'white', linewidth = 0.4)
            ax[i].axvline(2.5, c = 'white', linewidth = 0.4)
            ax[i].axvline(1, c = 'white', linewidth = 0.4)
            ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
            ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
            ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)

# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 25)

plt.tick_params(axis = 'both', which = 'both', direction='in')
# plt.suptitle(r'Relative differences: $|$Low-High$|$/mean $M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredE_relative_diff_{xaxis}.png')
plt.show()

# %% Do just for radiation without fluff as  check
diff_Rad_nofluff = col_Rad_nofluff - col_RadMiddle_nofluff
denominator_Rad_nofluff = (col_Rad_nofluff + col_RadMiddle_nofluff)/2
rel_Rad_nofluff = np.abs(diff_Rad_nofluff / denominator_Rad_nofluff)
rel_Rad_nofluff[np.isnan(rel_Rad_nofluff)] = 0
normRadnofluff = colors.LogNorm(vmin=np.percentile(col_Rad_nofluff[col_Rad_nofluff>0], 5), vmax=np.percentile(col_Rad_nofluff[col_Rad_nofluff>0], 95))

fig, ax = plt.subplots(2,2, figsize = (12,10))
img = ax[0][0].pcolormesh(radiiLow/apo, tfb_Low, col_Rad_nofluff, norm=normRadnofluff, cmap = 'viridis')
cb = fig.colorbar(img)
cb.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 15, labelpad = 5)
ax[0][0].text(np.min(radiiLow/apo), 0.15,'Low res', fontsize = 25)
img = ax[0][1].pcolormesh(radiiMiddle/apo, tfb_Middle, col_RadMiddle_nofluff, norm=normRadnofluff, cmap = 'viridis')
cb = fig.colorbar(img)
cb.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 15, labelpad = 5)
ax[0][1].text(np.min(radiiMiddle/apo), 0.15,'High res', fontsize = 25)
# plot the relaive difference
img = ax[1][0].pcolormesh(radiiMiddle/apo, tfb_Middle, rel_Rad_nofluff, cmap='inferno', norm=norm_Rad)
cb = fig.colorbar(img)
cb.set_label('Relative difference', fontsize = 15, labelpad = 5)
for i in range(2):
    ax[i][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
    for j in range(2):
        ax[i][j].axhline(0.205, c = 'white', linewidth = 0.4)
        ax[i][j].axhline(0.52, c = 'white', linewidth = 0.4)
        ax[i][j].axhline(0.75, c = 'white', linewidth = 0.4)
        if xaxis == 'radii':
            ax[i][j].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
            ax[i][j].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
            ax[i][j].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
            ax[i][j].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
            ax[i][j].axvline(0.1, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(0.3, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(0.5, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(1, c = 'white', linewidth = 0.4)
            ax[i][j].set_xscale('log')

        elif xaxis == 'angles':
            ax[i][j].axvline(-2.5, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(-1, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(2.5, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(1, c = 'white', linewidth = 0.4)

        if i == 1:
            if xaxis == 'radii':
                ax[i][j].set_xlabel(r'$R/R_a$', fontsize = 20)
                ax[i][j].set_xlabel(r'$R/R_a$', fontsize = 20)
                ax[i][j].set_xlabel(r'$R/R_a$', fontsize = 20)
            elif xaxis == 'angles':
                ax[i][j].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
                ax[i][j].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
                ax[i][j].set_xlabel(r'$\theta$ [rad]', fontsize = 20)

plt.tick_params(axis = 'both', which = 'both', direction='in', size = 10, labelsize=15)
plt.tick_params(axis = 'both', which = 'major',  size = 10)
plt.tick_params(axis = 'x', which = 'minor',  size = 7)
plt.suptitle('Radiation energy density without fluff', fontsize = 30)
fig.delaxes(ax[1][1])
plt.tight_layout()

if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredRadE_nofluff_{xaxis}.png')
plt.show()

#%%
if xaxis == 'radii':
    from matplotlib.colors import ListedColormap, BoundaryNorm
    # Check wh has the higher energy between Low and High
    colorsthree = ['orchid', 'white', 'dodgerblue']  # Example: red for negative, white for zero, green for positive
    cmapthree = ListedColormap(colorsthree)
    boundaries = [-1e9, -0.1, 0.1, 1e9]  # Slight offset to differentiate zero from positive
    normthree = BoundaryNorm(boundaries, cmapthree.N, clip=True)

    fig, ax = plt.subplots(2,2, figsize = (12,12))
    img = ax[0][0].pcolormesh(radiiMiddle/apo, tfb_Middle, np.abs(col_orb_en) - np.abs(col_orb_enMiddle), cmap = cmapthree, norm = normthree)
    ax[0][0].set_title('Specific (absolute) orbital energy', fontsize = 20)
    img1 = ax[0][1].pcolormesh(radiiMiddle/apo, tfb_Middle, diff_ie, cmap = cmapthree, norm = normthree)
    ax[0][1].set_title('Specific internal energy', fontsize = 20)
    img2 = ax[1][0].pcolormesh(radiiMiddle/apo, tfb_Middle, diff_Rad, cmap = cmapthree, norm = normthree)
    ax[1][0].set_title('Radiation energy density', fontsize = 20)
    img3 = ax[1][1].pcolormesh(radiiMiddle/apo, tfb_Middle, diff_Rad_nofluff, cmap = cmapthree, norm = normthree)
    ax[1][1].set_title('Radiation energy density NO fluff', fontsize = 20)
    # Create an axis for the colorbar
    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # Add the colorbar to the figure
    cb = fig.colorbar(img, cax=cbar_ax)
    cb.set_label('Low-High', fontsize = 30)
    ax[1][0].set_xlabel(r'$R/R_a$', fontsize = 20)
    ax[1][1].set_xlabel(r'$R/R_a$', fontsize = 20)
    ax[0][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
    ax[1][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xscale('log')
    plt.tight_layout()
    if save:
        plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredE_whoIsHigher.png')

    # Lines
    # indices = [70, 100, 136]
    indices = [np.argmin(np.abs(tfb_Low-0.46)), np.argmin(np.abs(tfb_Low-0.66)), np.argmin(np.abs(tfb_Low-0.86))]
    colors_indices = ['navy', 'royalblue', 'deepskyblue']
    lines_difference = (col_Rad[indices]-col_RadMiddle[indices])/col_RadMiddle[indices]
    img, ax = plt.subplots(1,2, figsize = (16,4))
    for i,idx in enumerate(indices):
        ax[0].plot(radiiLow, col_Rad[idx], c = colors_indices[i], label = f'Low t/tfb = {np.round(tfb_Low[idx],2)}')
        ax[0].plot(radiiMiddle, col_RadMiddle[idx], '--', c = colors_indices[i], label = f'Middle t/tfb = {np.round(tfb_Middle[idx],2)}')
        ax[1].plot(radiiLow, lines_difference[i], c = colors_indices[i], label = f't/tfb = {np.round(tfb_Low[idx],2)}')
    ax[0].set_xlabel(r'R/R$_a$', fontsize = 20)
    ax[0].set_ylabel(r'(Rad/Vol) [erg/cm$^3$]', fontsize = 20)
    ax[1].set_xlabel(r'$R/R_a$', fontsize = 20)
    ax[1].set_ylabel(r'$|$Low-High$|$/mean', fontsize = 20)
    ax[0].loglog()
    ax[1].loglog()
    ax[0].legend(fontsize = 16)
    ax[1].legend(fontsize = 18)
    ax[0].grid()
    ax[1].grid()
    plt.suptitle('Relative differences: 1-High/Low')
    plt.tight_layout()
    if save:
        plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/Rad_lines.png')
    plt.show()

    #
    Lum_cgs = col_Rad  * prel.c * 4 * np.pi * (radiiLow*prel.Rsol_to_cm)**2 
    LumMiddle_cgs = col_RadMiddle * prel.c * 4 * np.pi * (radiiMiddle*prel.Rsol_to_cm)**2 
    denom = (Lum_cgs + LumMiddle_cgs)/2
    Lum_difference = (Lum_cgs[indices]-LumMiddle_cgs[indices])/denom[indices]

    img, ax = plt.subplots(1,2, figsize = (20,7))
    for i,idx in enumerate(indices):
        if i == 0:
            ax[0].plot(radiiLow, Lum_cgs[idx], c = colors_indices[i], label = f'Low res')#t/tfb = {np.round(tfb_Low[idx],2)}')
            ax[0].plot(radiiMiddle, LumMiddle_cgs[idx], '--', c = colors_indices[i], label = f'High res')#t/tfb = {np.round(tfb_Middle[idx],2)}')
            ax[1].plot(radiiLow, Lum_difference[i], c = colors_indices[i])#, label = f't/tfb = {np.round(tfb_Low[idx],2)}')
        else:   
            ax[0].plot(radiiLow, Lum_cgs[idx], c = colors_indices[i])#, label = f'Low t/tfb = {np.round(tfb_Low[idx],2)}')
            ax[0].plot(radiiMiddle, LumMiddle_cgs[idx], '--', c = colors_indices[i])#, label = f'Middle t/tfb = {np.round(tfb_Middle[idx],2)}')
            ax[1].plot(radiiLow, Lum_difference[i], c = colors_indices[i])#, label = f't/tfb = {np.round(tfb_Low[idx],2)}')
    ax[0].set_ylim(1e40, 1e44)
    ax[1].set_ylim(0.3, 1.8)
    ax[0].text(15, 1.5e41, r'$t/t_{fb}$ = '+ f'{np.round(tfb_Low[indices[0]],2)}', fontsize = 20)
    ax[0].text(20, 2e42, r'$t/t_{fb}$ = '+ f'{np.round(tfb_Low[indices[1]],2)}', fontsize = 20)
    ax[0].text(50, 4e43, r'$t/t_{fb}$ = '+ f'{np.round(tfb_Low[indices[2]],2)}', fontsize = 20)
    # ax[1].text(620, 0.7, r'$\Delta_{rel}\approx$ '+ f'{np.round(np.mean(Lum_difference[0][-10:-1]),2)}', fontsize = 20)
    # ax[1].text(620, 0.5, r'$\Delta_{rel}\approx$ '+ f'{np.round(np.mean(Lum_difference[1][-10:-1]),2)}', fontsize = 19)
    # ax[1].text(620, 0.4, r'$\Delta_{rel}\approx$ '+ f'{np.round(np.mean(Lum_difference[2][-10:-1]),2)}', fontsize = 20)

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
    plt.subplots_adjust(wspace=1)
    # plt.suptitle('Relative differences: $|$Low-High$|$/mean')
    plt.tight_layout()
    if save:
        plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/Luminosity.png')
    plt.show()

    for i in range(3):
        mean_error =  np.round(np.mean(Lum_difference[i][-10:-1]),2)
        print(f'Mean relative error for t/tfb = {tfb_Low[indices[i]]} is {mean_error}')
    # print the difference of the last point of Hires lum line
    for i,idx in enumerate(indices):
        if i ==2:
            continue
        high = LumMiddle_cgs[indices[i+1]]
        low = LumMiddle_cgs[indices[i]]
        mean_lasthigh = np.mean(high[-10:-1])
        mean_lastlow = np.mean(low[-10:-1])
        print(tfb_Low[indices[i+1]], tfb_Low[indices[i]])
        print(f'The relative difference of the last point of the high res line is {np.round((mean_lasthigh-mean_lastlow),2)}')
